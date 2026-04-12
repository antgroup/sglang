"""Kernel precision debug decorator for SGLang.

A self-contained, portable decorator that wraps kernel functions to log shapes/stats
and dump input/output tensors for offline precision comparison. All behavior is
controlled via environment variables — no code changes needed after wrapping.

Environment Variables:
    SGLANG_PRECISION_DEBUG (int, default=0):
        0 = off (zero overhead), 1 = log shapes/dtypes (no GPU sync),
        2 = + stats min/max/mean/nan/inf (GPU sync), 3 = + dump tensors to disk,
        4 = align mode: load inputs from ALIGN_DIR, execute, compare outputs inline.
    SGLANG_PRECISION_DUMP_DIR (str, default="sglang_precision_dumps"):
        Dump directory. Supports %i for PID, %r for rank.
    SGLANG_PRECISION_OP_FILTER (str, default=""):
        Comma-separated fnmatch patterns for op names. Empty = all ops.
    SGLANG_PRECISION_INDEX_FILTER (str, default="<10"):
        Which invocation indices to capture per op (0-based).
        Formats: "5-10", "5,10,15", ">=100", "<10".
    SGLANG_PRECISION_SHAPE_FILTER (str, default=""):
        Shape conditions on tensor args. E.g. "dim0>=1024,dim-1<256".
    SGLANG_PRECISION_LOG_DEST (str, default="stderr"):
        Log destination: stdout, stderr, or a filepath.
    SGLANG_PRECISION_TAG (str, default=""):
        Tag subdirectory name, e.g. "baseline" or "target".
    SGLANG_PRECISION_RANK_FILTER (str, default=""):
        Which ranks to capture. E.g. "0", "0,1", "0-3". Empty = all.
    SGLANG_PRECISION_RANGE_START (str, default=""):
        Activate capture when an op matching this pattern is encountered.
        Supports optional per-op call count suffix: "attention" or "attention#3".
        The #N uses the per-op call counter (same as INDEX_FILTER).
        Both RANGE_START and RANGE_END must be set for range filtering to activate.
    SGLANG_PRECISION_RANGE_END (str, default=""):
        Deactivate capture after processing the op matching this pattern.
        Same format as RANGE_START. The end-anchor op is included in the range.
    SGLANG_PRECISION_ALIGN_DIR (str, default=""):
        Level 4 only. Path to a baseline dump tag directory.
    SGLANG_PRECISION_ALIGN_THRESHOLD (float, default=1e-3):
        Level 4 only. rel_diff threshold for PASS/FAIL in align-mode comparison.

Usage:
    from sglang.kernel_precision_debug import precision_debug

    @precision_debug
    def rmsnorm(input, weight, out=None, eps=1e-6):
        ...

Comparison CLI:
    python -m sglang.kernel_precision_debug compare \\
        --baseline sglang_precision_dumps/baseline \\
        --target sglang_precision_dumps/target \\
        [--op-filter "rmsnorm*"] [--rank 0] [--diff-threshold 1e-3] \\
        [--baseline-slice "5:10"] [--target-slice "0:5"] \\
        [--baseline-index "0,3,7"] [--target-index "1,4,8"]
"""

from __future__ import annotations

import argparse
import fnmatch
import functools
import inspect
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, overload

import torch

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_logger = logging.getLogger("sglang.precision_debug")
_F = TypeVar("_F", bound=Callable[..., Any])


def _get_env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _get_env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _str_with_pid_rank(path: str, rank: int) -> str:
    path = path.replace("%i", str(os.getpid()))
    path = path.replace("%r", str(rank))
    return path


# ---------------------------------------------------------------------------
# Configuration (read once at import time)
# ---------------------------------------------------------------------------

_LEVEL = _get_env_int("SGLANG_PRECISION_DEBUG", 0)
_DUMP_DIR_TEMPLATE = _get_env("SGLANG_PRECISION_DUMP_DIR", "sglang_precision_dumps")
_OP_FILTER_RAW = _get_env("SGLANG_PRECISION_OP_FILTER", "")
_INDEX_FILTER_RAW = _get_env("SGLANG_PRECISION_INDEX_FILTER", "<10")
_SHAPE_FILTER_RAW = _get_env("SGLANG_PRECISION_SHAPE_FILTER", "")
_LOG_DEST = _get_env("SGLANG_PRECISION_LOG_DEST", "stderr")
_TAG = _get_env("SGLANG_PRECISION_TAG", "")
_ALIGN_DIR = _get_env("SGLANG_PRECISION_ALIGN_DIR", "")
_ALIGN_THRESHOLD = float(_get_env("SGLANG_PRECISION_ALIGN_THRESHOLD", "1e-3"))

_RANK_FILTER_RAW = _get_env("SGLANG_PRECISION_RANK_FILTER", "")

_RANGE_START_RAW = _get_env("SGLANG_PRECISION_RANGE_START", "")
_RANGE_END_RAW = _get_env("SGLANG_PRECISION_RANGE_END", "")

_OP_PATTERNS = [p.strip() for p in _OP_FILTER_RAW.split(",") if p.strip()]

# Per-op call counter (keyed by op_name) — shared by index_filter and range #N
_call_counters: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------


def _setup_logger() -> None:
    for h in list(_logger.handlers):
        _logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    if _LEVEL == 0:
        _logger.addHandler(logging.NullHandler())
        _logger.setLevel(logging.CRITICAL + 1)
        return

    _logger.setLevel(logging.DEBUG)
    if _LOG_DEST == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    elif _LOG_DEST == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.FileHandler(
            _str_with_pid_rank(_LOG_DEST, _get_current_rank()), mode="a"
        )
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.propagate = False


# ---------------------------------------------------------------------------
# Rank detection
# ---------------------------------------------------------------------------

_cached_rank: int | None = None


def _get_current_rank() -> int:
    global _cached_rank
    if _cached_rank is not None:
        return _cached_rank

    try:
        import torch.distributed as dist

        if dist.is_initialized():
            _cached_rank = dist.get_rank()
            return _cached_rank
    except Exception:
        pass

    for env_key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        v = os.environ.get(env_key)
        if v is not None:
            try:
                _cached_rank = int(v)
                return _cached_rank
            except ValueError:
                pass

    _cached_rank = 0
    return _cached_rank


# ---------------------------------------------------------------------------
# Filter: rank
# ---------------------------------------------------------------------------


def _parse_int_set(raw: str) -> set[int] | None:
    """Parse '0', '0,1,3', '0-3' into a set of ints. Empty string → None (=all)."""
    raw = raw.strip()
    if not raw:
        return None
    result: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if "-" in part and not part.startswith("-"):
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result


_RANK_SET = _parse_int_set(_RANK_FILTER_RAW)


def _rank_allowed() -> bool:
    if _RANK_SET is None:
        return True
    return _get_current_rank() in _RANK_SET


# ---------------------------------------------------------------------------
# Filter: op name
# ---------------------------------------------------------------------------


def _op_allowed(op_name: str) -> bool:
    if _OP_PATTERNS:
        if not any(fnmatch.fnmatch(op_name, p) for p in _OP_PATTERNS):
            return False
    return True


# ---------------------------------------------------------------------------
# Filter: index
# ---------------------------------------------------------------------------


def _parse_index_filter(raw: str) -> Callable[[int], bool]:
    """Parse index filter expression into a predicate function.

    Supported formats:
        "5-10"    → 5 <= idx <= 10
        "5,10,15" → idx in {5, 10, 15}
        ">=100"   → idx >= 100
        ">100"    → idx > 100
        "<=10"    → idx <= 10
        "<5"      → idx < 5
        ""        → always True
    """
    raw = raw.strip()
    if not raw:
        return lambda _: True

    # Comparison operators
    m = re.fullmatch(r"(>=|<=|>|<)\s*(\d+)", raw)
    if m:
        op_str, val = m.group(1), int(m.group(2))
        if op_str == ">=":
            return lambda idx, v=val: idx >= v
        if op_str == "<=":
            return lambda idx, v=val: idx <= v
        if op_str == ">":
            return lambda idx, v=val: idx > v
        if op_str == "<":
            return lambda idx, v=val: idx < v

    # Range: "5-10"
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", raw)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return lambda idx, lo=lo, hi=hi: lo <= idx <= hi

    # Explicit list: "5,10,15"
    if "," in raw:
        vals = {int(x.strip()) for x in raw.split(",")}
        return lambda idx, s=vals: idx in s

    # Single number
    try:
        single = int(raw)
        return lambda idx, v=single: idx == v
    except ValueError:
        pass

    return lambda _: True


_INDEX_PRED = _parse_index_filter(_INDEX_FILTER_RAW)


# ---------------------------------------------------------------------------
# Filter: shape
# ---------------------------------------------------------------------------


def _parse_shape_filter(raw: str) -> list[tuple[int, str, int]]:
    """Parse shape filter like 'dim0>=1024,dim-1<256'.

    Returns list of (dim_index, operator, value).
    """
    raw = raw.strip()
    if not raw:
        return []
    conditions = []
    for part in raw.split(","):
        part = part.strip()
        m = re.fullmatch(r"dim(-?\d+)\s*(>=|<=|>|<|==|!=)\s*(\d+)", part)
        if m:
            dim_idx = int(m.group(1))
            op_str = m.group(2)
            value = int(m.group(3))
            conditions.append((dim_idx, op_str, value))
    return conditions


_SHAPE_CONDITIONS = _parse_shape_filter(_SHAPE_FILTER_RAW)

_SHAPE_OPS: dict[str, Callable[[int, int], bool]] = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _shape_allowed(tensors: list[torch.Tensor]) -> bool:
    if not _SHAPE_CONDITIONS:
        return True
    for t in tensors:
        shape = t.shape
        all_match = True
        for dim_idx, op_str, value in _SHAPE_CONDITIONS:
            try:
                dim_size = shape[dim_idx]
            except IndexError:
                all_match = False
                break
            if not _SHAPE_OPS[op_str](dim_size, value):
                all_match = False
                break
        if all_match:
            return True
    return False


# ---------------------------------------------------------------------------
# Filter: range (between two anchor ops)
# ---------------------------------------------------------------------------


def _parse_range_spec(raw: str) -> tuple[str, int | None]:
    """Parse 'pattern' or 'pattern#N' → (pattern, count_or_None).

    'pattern' matches any call of the matching op.
    'pattern#N' matches the Nth call (using the per-op call counter).
    """
    raw = raw.strip()
    if not raw:
        return ("", None)
    m = re.fullmatch(r"(.+)#(\d+)", raw)
    if m:
        return (m.group(1), int(m.group(2)))
    return (raw, None)


_RANGE_START_PATTERN, _RANGE_START_COUNT = _parse_range_spec(_RANGE_START_RAW)
_RANGE_END_PATTERN, _RANGE_END_COUNT = _parse_range_spec(_RANGE_END_RAW)
_RANGE_ENABLED = bool(_RANGE_START_PATTERN and _RANGE_END_PATTERN)

# Range state machine
_range_active: bool = False


def _range_check(op_name: str, call_idx: int) -> bool:
    """Check range start/end. Called AFTER counter increment, BEFORE index_filter.

    Uses the per-op call_idx (same counter as index_filter) for #N matching.
    The start-anchor op and end-anchor op are both included in the range.
    """
    global _range_active

    if not _RANGE_ENABLED:
        return True

    # Check start trigger
    if not _range_active:
        if fnmatch.fnmatch(op_name, _RANGE_START_PATTERN):
            if _RANGE_START_COUNT is None or call_idx == _RANGE_START_COUNT:
                _range_active = True
                _logger.debug(
                    f"[RANGE] Activated at {op_name} #{call_idx}"
                )
        return _range_active

    # Already active — check end trigger
    if fnmatch.fnmatch(op_name, _RANGE_END_PATTERN):
        if _RANGE_END_COUNT is None or call_idx == _RANGE_END_COUNT:
            _range_active = False
            _logger.debug(
                f"[RANGE] Deactivated after {op_name} #{call_idx}"
            )
            return True  # end-anchor op is included

    return True  # still active


# ---------------------------------------------------------------------------
# Tensor stats (level 2+)
# ---------------------------------------------------------------------------


def _tensor_stats_str(t: torch.Tensor) -> str:
    """Compute min/max/mean/nan_count/inf_count. Requires GPU sync."""
    try:
        d = t.detach().float()
        nan_c = int(torch.isnan(d).sum().item())
        inf_c = int(torch.isinf(d).sum().item())
        return (
            f"min={d.min().item():.6g} max={d.max().item():.6g} "
            f"mean={d.mean().item():.6g} nan={nan_c} inf={inf_c}"
        )
    except Exception as e:
        return f"stats_error={e}"


# ---------------------------------------------------------------------------
# Op name inference
# ---------------------------------------------------------------------------


def _infer_op_name(func: Callable) -> str:
    """Infer the op name from the function (module + qualname)."""
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", "unknown"))
    qualname = qualname.replace(".<locals>.", ".").replace("<locals>.", "")
    module = getattr(func, "__module__", "")
    for prefix in ("sglang.", "sgl_kernel."):
        if module.startswith(prefix):
            module = module[len(prefix) :]
            break
    if module and module not in {"__main__", "builtins"}:
        return f"{module}.{qualname}"
    try:
        source_path = inspect.getsourcefile(func)
        if source_path is not None:
            return f"{Path(source_path).stem}.{qualname}"
    except Exception:
        pass
    return qualname


# ---------------------------------------------------------------------------
# Extract tensor args (supports nested dict/tuple/list)
# ---------------------------------------------------------------------------

# Type for user-defined input_filter callback.
# Signature: (param_name: str, value: Any) -> dict[str, Tensor] | None
# Return a dict of named tensors extracted from value, or None to skip.
InputFilter = Callable[[str, Any], Optional[Dict[str, torch.Tensor]]]

# input_names: list (whitelist) or dict (whitelist + rename {orig: dump_name})
InputNames = Union[List[str], Dict[str, str], None]

# Simple scalar types that are safe to include in metadata
_SIMPLE_TYPES = (str, int, float, bool, type(None))


def _resolve_input_names(
    input_names: InputNames,
) -> Optional[Dict[str, str]]:
    """Normalize input_names to a dict {original_param_name: dump_name}, or None."""
    if input_names is None:
        return None
    if isinstance(input_names, list):
        return {n: n for n in input_names}
    return dict(input_names)  # already a dict


def _extract_tensors_recursive(
    prefix: str, value: Any, max_depth: int = 3,
    input_filter: InputFilter | None = None,
) -> dict[str, torch.Tensor]:
    """Recursively extract tensors from nested dict/tuple/list structures.

    Keys use dot-path notation: ``prefix.key``, ``prefix.0``, etc.
    Complex objects (non-dict/tuple/list/Tensor/scalar) are passed to
    *input_filter* if provided; otherwise they are skipped.
    """
    if isinstance(value, torch.Tensor):
        return {prefix: value}

    if max_depth <= 0:
        return {}

    if isinstance(value, dict):
        result: dict[str, torch.Tensor] = {}
        for k, v in value.items():
            sub = _extract_tensors_recursive(
                f"{prefix}.{k}", v, max_depth - 1, input_filter)
            result.update(sub)
        return result

    if isinstance(value, (tuple, list)):
        result = {}
        for i, v in enumerate(value):
            sub = _extract_tensors_recursive(
                f"{prefix}.{i}", v, max_depth - 1, input_filter)
            result.update(sub)
        return result

    if isinstance(value, _SIMPLE_TYPES):
        return {}

    # Complex object — delegate to input_filter if available
    if input_filter is not None:
        custom = input_filter(prefix, value)
        if custom is not None:
            return custom
    return {}


def _collect_tensors(args: tuple, kwargs: dict,
                     input_filter: InputFilter | None = None,
                     name_map: Optional[Dict[str, str]] = None,
                     param_names: Optional[List[str]] = None,
                     ) -> list[torch.Tensor]:
    """Collect all tensors from args/kwargs (including nested structures).

    If *name_map* is provided, only include args whose original param name
    is a key in name_map (whitelist filtering).
    """
    tensors: list[torch.Tensor] = []
    _pnames = param_names or []

    def _should_include(orig_name: str) -> bool:
        return name_map is None or orig_name in name_map

    for i, a in enumerate(args):
        orig = _pnames[i] if i < len(_pnames) else f"arg{i}"
        if not _should_include(orig):
            continue
        if isinstance(a, torch.Tensor):
            tensors.append(a)
        elif isinstance(a, (dict, tuple, list)):
            found = _extract_tensors_recursive(
                orig, a, input_filter=input_filter)
            tensors.extend(found.values())
        elif input_filter is not None and not isinstance(a, _SIMPLE_TYPES):
            custom = input_filter(orig, a)
            if custom:
                tensors.extend(custom.values())
    for k, v in kwargs.items():
        if not _should_include(k):
            continue
        if isinstance(v, torch.Tensor):
            tensors.append(v)
        elif isinstance(v, (dict, tuple, list)):
            found = _extract_tensors_recursive(
                k, v, input_filter=input_filter)
            tensors.extend(found.values())
        elif input_filter is not None and not isinstance(v, _SIMPLE_TYPES):
            custom = input_filter(k, v)
            if custom:
                tensors.extend(custom.values())
    return tensors


def _build_named_tensors(
    params: tuple, args: tuple, kwargs: dict,
    input_filter: InputFilter | None = None,
    name_map: Optional[Dict[str, str]] = None,
) -> dict[str, torch.Tensor]:
    """Build a dict mapping parameter names (dot-path) to tensor values.

    If *name_map* is provided:
    - Only params whose original name is a key in name_map are included.
    - The dump key uses name_map[orig] instead of orig (renaming).
    - For nested paths like ``orig.sub.0``, the prefix is replaced:
      ``name_map[orig].sub.0``.
    """
    result: dict[str, torch.Tensor] = {}
    param_names = [p.name for p in params]

    def _add_value(orig_name: str, value: Any) -> None:
        dump_name = name_map[orig_name] if name_map and orig_name in name_map else orig_name
        if isinstance(value, torch.Tensor):
            result[dump_name] = value
        elif isinstance(value, (dict, tuple, list)):
            sub = _extract_tensors_recursive(
                dump_name, value, input_filter=input_filter)
            result.update(sub)
        elif input_filter is not None and not isinstance(value, _SIMPLE_TYPES):
            custom = input_filter(dump_name, value)
            if custom:
                result.update(custom)

    for i, a in enumerate(args):
        orig = param_names[i] if i < len(param_names) else f"arg{i}"
        if name_map is not None and orig not in name_map:
            continue
        _add_value(orig, a)
    for k, v in kwargs.items():
        if name_map is not None and k not in name_map:
            continue
        _add_value(k, v)
    return result


def _build_named_values(
    params: tuple, args: tuple, kwargs: dict
) -> dict[str, Any]:
    """Build a dict mapping parameter names to all values (for metadata)."""
    result: dict[str, Any] = {}
    param_names = [p.name for p in params]
    for i, a in enumerate(args):
        name = param_names[i] if i < len(param_names) else f"arg{i}"
        result[name] = a
    for k, v in kwargs.items():
        result[k] = v
    return result


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log_shapes(op_name: str, call_idx: int,
                named_tensors: dict[str, torch.Tensor],
                named_values: dict[str, Any], prefix: str = "IN") -> None:
    parts = [f"[{prefix}] {op_name} #{call_idx}"]
    for name, t in named_tensors.items():
        parts.append(f"{name}:shape={list(t.shape)},dtype={t.dtype}")
    # Also log non-tensor scalar args
    for name, v in named_values.items():
        if not isinstance(v, torch.Tensor):
            parts.append(f"{name}={v}")
    _logger.debug("  ".join(parts))


def _log_stats(op_name: str, call_idx: int, named_tensors: dict[str, torch.Tensor],
               prefix: str = "IN") -> None:
    for name, t in named_tensors.items():
        stats = _tensor_stats_str(t)
        _logger.debug(f"  [{prefix}] {op_name} #{call_idx} {name}: {stats}")


# ---------------------------------------------------------------------------
# Dump helpers (level 3)
# ---------------------------------------------------------------------------


def _get_dump_dir(op_name: str, call_idx: int) -> Path:
    rank = _get_current_rank()
    base = _str_with_pid_rank(_DUMP_DIR_TEMPLATE, rank)
    parts = [base]
    if _TAG:
        parts.append(_TAG)
    parts.append(f"rank{rank}")
    safe_op = op_name.replace("/", "_").replace("<", "_").replace(">", "_")
    parts.append(safe_op)
    parts.append(f"call_{call_idx:06d}")
    return Path(*parts)


def _dump_tensors(dump_dir: Path, named_tensors: dict[str, torch.Tensor],
                  filename: str) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    data = {k: v.detach().cpu() for k, v in named_tensors.items()}
    torch.save(data, dump_dir / filename)


def _dump_metadata(dump_dir: Path, op_name: str, call_idx: int,
                   named_values: dict[str, Any],
                   named_tensors_in: dict[str, torch.Tensor],
                   named_tensors_out: dict[str, torch.Tensor] | None = None,
                   exception: str | None = None) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)

    def _tensor_meta(t: torch.Tensor) -> dict:
        info: dict[str, Any] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
        }
        try:
            d = t.detach().float()
            info["min"] = d.min().item()
            info["max"] = d.max().item()
            info["mean"] = d.mean().item()
            info["nan_count"] = int(torch.isnan(d).sum().item())
            info["inf_count"] = int(torch.isinf(d).sum().item())
        except Exception:
            pass
        return info

    def _serialize_value(v: Any) -> Any:
        if isinstance(v, torch.Tensor):
            return _tensor_meta(v)
        if isinstance(v, torch.dtype):
            return str(v)
        if isinstance(v, (str, int, float, bool, type(None))):
            return v
        return repr(v)[:200]

    meta: dict[str, Any] = {
        "op_name": op_name,
        "call_index": call_idx,
        "rank": _get_current_rank(),
        "pid": os.getpid(),
        "timestamp": datetime.now().isoformat(),
        "inputs": {k: _serialize_value(v) for k, v in named_values.items()},
    }
    if named_tensors_out is not None:
        meta["outputs"] = {k: _tensor_meta(v) for k, v in named_tensors_out.items()}
    if exception is not None:
        meta["exception"] = exception
    (dump_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


# ---------------------------------------------------------------------------
# Level 4: Align mode helpers
# ---------------------------------------------------------------------------


def _get_align_dir(op_name: str, call_idx: int) -> Path | None:
    """Locate the baseline dump directory for an (op_name, call_idx) pair."""
    if not _ALIGN_DIR:
        return None
    rank = _get_current_rank()
    base = Path(_ALIGN_DIR)
    rank_dir = base / f"rank{rank}"
    if not rank_dir.exists():
        rank_dir = base  # fallback: no rank subdirectory
    safe_op = op_name.replace("/", "_").replace("<", "_").replace(">", "_")
    call_dir = rank_dir / safe_op / f"call_{call_idx:06d}"
    if call_dir.exists():
        return call_dir
    return None


def _load_align_tensors(align_dir: Path, filename: str,
                        device: torch.device) -> dict[str, torch.Tensor] | None:
    """Load tensors from a baseline dump file, moving to the target device."""
    pt_file = align_dir / filename
    if not pt_file.exists():
        return None
    try:
        data = torch.load(pt_file, weights_only=True, map_location="cpu")
        return {k: v.to(device) for k, v in data.items()}
    except Exception as e:
        _logger.debug(f"[ALIGN] Failed to load {pt_file}: {e}")
        return None


def _set_by_dotpath(container: Any, path_parts: list[str],
                    value: torch.Tensor) -> Any:
    """Set a value in a nested dict/tuple/list by dot-path parts.

    Returns the (possibly new) container since tuples are immutable.
    """
    if not path_parts:
        return value
    key = path_parts[0]
    rest = path_parts[1:]

    if isinstance(container, dict):
        child = container.get(key)
        if child is not None:
            container[key] = _set_by_dotpath(child, rest, value)
        return container
    elif isinstance(container, (list, tuple)):
        try:
            idx = int(key)
        except ValueError:
            return container
        if 0 <= idx < len(container):
            if isinstance(container, tuple):
                items = list(container)
                items[idx] = _set_by_dotpath(items[idx], rest, value)
                return tuple(items)
            else:
                container[idx] = _set_by_dotpath(container[idx], rest, value)
                return container
    return container


def _replace_tensor_args(
    params: tuple, positional: tuple, kwargs: dict,
    replacements: dict[str, torch.Tensor],
    name_map: Optional[Dict[str, str]] = None,
) -> tuple[tuple, dict]:
    """Replace tensor args in positional/kwargs with values from replacements dict.

    Supports dot-path keys (e.g. ``x.hidden_states.0``) for nested structures.

    If *name_map* is provided, replacement keys are dump names (from baseline).
    We reverse-map dump_name → orig_name to locate the correct positional/kwarg slot.
    """
    param_names = [p.name for p in params]
    new_positional = list(positional)
    new_kwargs = dict(kwargs)

    # Build reverse map: dump_name -> orig_param_name
    reverse_map: Optional[Dict[str, str]] = None
    if name_map:
        reverse_map = {v: k for k, v in name_map.items()}

    for rkey, replacement in replacements.items():
        parts = rkey.split(".")
        dump_root = parts[0]
        rest = parts[1:]

        # Resolve dump_root back to original param name for lookup
        orig_root = dump_root
        if reverse_map and dump_root in reverse_map:
            orig_root = reverse_map[dump_root]

        # Find which positional arg or kwarg this root belongs to
        found_pos = -1
        for i in range(len(new_positional)):
            name = param_names[i] if i < len(param_names) else f"arg{i}"
            if name == orig_root:
                found_pos = i
                break

        if found_pos >= 0:
            target = new_positional[found_pos]
            if not rest:
                # Direct top-level tensor replacement
                if isinstance(target, torch.Tensor):
                    if replacement.dtype != target.dtype:
                        replacement = replacement.to(target.dtype)
                    new_positional[found_pos] = replacement
            else:
                # Nested path replacement
                new_positional[found_pos] = _set_by_dotpath(
                    target, rest, replacement)
        elif orig_root in new_kwargs:
            target = new_kwargs[orig_root]
            if not rest:
                if isinstance(target, torch.Tensor):
                    if replacement.dtype != target.dtype:
                        replacement = replacement.to(target.dtype)
                    new_kwargs[orig_root] = replacement
            else:
                new_kwargs[orig_root] = _set_by_dotpath(
                    target, rest, replacement)

    return tuple(new_positional), new_kwargs


def _compare_and_log(op_name: str, call_idx: int, label: str,
                     actual: dict[str, torch.Tensor],
                     baseline: dict[str, torch.Tensor],
                     threshold: float) -> None:
    """Compare actual vs baseline tensors and log results."""
    common_keys = sorted(set(actual.keys()) & set(baseline.keys()))
    for key in common_keys:
        a = actual[key].detach().float()
        b = baseline[key].float()
        # Auto-truncate dim0 if mismatched
        if a.shape != b.shape and len(a.shape) > 0 and a.shape[0] != b.shape[0]:
            min_d0 = min(a.shape[0], b.shape[0])
            a, b = a[:min_d0], b[:min_d0]
        if a.shape != b.shape:
            _logger.debug(
                f"  [ALIGN] {op_name} #{call_idx} {label}/{key}: "
                f"shape mismatch {list(a.shape)} vs {list(b.shape)}, skipped"
            )
            continue
        rel_diff = _calc_rel_diff(a, b)
        abs_diff = (a - b).abs()
        max_abs = abs_diff.max().item()
        mean_abs = abs_diff.mean().item()
        passed = rel_diff <= threshold
        marker = "PASS" if passed else "FAIL"
        _logger.debug(
            f"  [ALIGN {marker}] {op_name} #{call_idx} {label}/{key}: "
            f"rel_diff={rel_diff:.6g} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}"
        )
        if not passed:
            flat_idx = abs_diff.argmax()
            coord = tuple(
                idx.item() for idx in torch.unravel_index(flat_idx, abs_diff.shape)
            )
            _logger.debug(
                f"           max_abs at coord={coord} "
                f"baseline={b[coord].item():.6g} actual={a[coord].item():.6g}"
            )


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

_setup_logger()


# Type alias for align_transform callback:
#   (tensors: dict[str, Tensor], category: str) -> dict[str, Tensor]
# category is "inputs", "outputs", or "inputs_after"
AlignTransform = Callable[[dict[str, torch.Tensor], str], dict[str, torch.Tensor]]


@overload
def precision_debug(func: _F) -> _F: ...


@overload
def precision_debug(
    *,
    op_name: str | None = None,
    align_transform: AlignTransform | None = None,
    input_filter: InputFilter | None = None,
    input_names: InputNames = None,
) -> Callable[[_F], _F]: ...


def precision_debug(
    func: Callable | None = None,
    *,
    op_name: str | None = None,
    align_transform: AlignTransform | None = None,
    input_filter: InputFilter | None = None,
    input_names: InputNames = None,
) -> Callable:
    """Decorator to instrument kernel functions for precision debugging.

    Args:
        align_transform: Optional callback for Level 4 align mode. Called after
            loading baseline tensors from disk, before using them for replacement
            or comparison. Signature: (tensors: dict[str, Tensor], category: str)
            -> dict[str, Tensor]. Category is "inputs", "outputs", or "inputs_after".
            Use this for dtype conversion, reshape, slice, etc.
        input_filter: Optional callback for extracting tensors from complex
            input objects. Signature: (name: str, value: Any) -> dict[str, Tensor] | None.
            Called when an argument is not a Tensor, dict, tuple, list, or simple scalar.
            Return a dict of named tensors to include, or None to skip.
        input_names: Optional whitelist / rename mapping for input parameters.
            - list[str]: only dump/log these parameter names (whitelist).
            - dict[str, str]: only dump/log keys, using values as dump names
              (for cross-framework alignment where param names differ).
            - None: process all parameters (default).

    When SGLANG_PRECISION_DEBUG=0 (default), returns the original function
    with zero overhead.
    """
    if _LEVEL == 0:
        if func is None:
            return lambda f: f
        return func

    def decorator(f: Callable) -> Callable:
        if getattr(f, "_precision_debug_wrapped", False):
            return f

        resolved_name = op_name or _infer_op_name(f)
        resolved_name_map = _resolve_input_names(input_names)

        try:
            sig_params = tuple(inspect.signature(f).parameters.values())
        except (TypeError, ValueError):
            sig_params = ()

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Skip during torch.compile tracing
            try:
                if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
                    if torch.compiler.is_compiling():
                        return f(*args, **kwargs)
            except Exception:
                pass

            # --- Rank filter ---
            if not _rank_allowed():
                return f(*args, **kwargs)

            # --- Op name filter ---
            if not _op_allowed(resolved_name):
                return f(*args, **kwargs)

            # --- Prepare args (skip self/cls) ---
            positional = args
            params = sig_params
            if args and params and params[0].name in {"self", "cls"}:
                positional = args[1:]
                params = params[1:]

            # --- Shape filter (uses name_map for whitelist) ---
            _pnames = [p.name for p in params]
            tensors = _collect_tensors(
                positional, kwargs, input_filter,
                name_map=resolved_name_map, param_names=_pnames)
            if not _shape_allowed(tensors):
                return f(*args, **kwargs)

            # --- Increment per-op counter (after rank/op/shape, before range/index) ---
            idx = _call_counters.get(resolved_name, -1) + 1
            _call_counters[resolved_name] = idx

            # --- Range filter (uses per-op counter for #N) ---
            if not _range_check(resolved_name, idx):
                return f(*args, **kwargs)

            # --- Index filter (uses same per-op counter) ---
            if not _INDEX_PRED(idx):
                return f(*args, **kwargs)

            # --- Build named maps ---
            named_tensors_in = _build_named_tensors(
                params, positional, kwargs, input_filter,
                name_map=resolved_name_map)
            named_values = _build_named_values(params, positional, kwargs)

            # --- Warn about skipped args ---
            if _LEVEL >= 2:
                # Warn about args skipped by input_names whitelist
                if resolved_name_map is not None:
                    for i, a in enumerate(positional):
                        orig = _pnames[i] if i < len(_pnames) else f"arg{i}"
                        if orig not in resolved_name_map:
                            _logger.debug(
                                f"  [SKIP] {resolved_name} #{idx} — "
                                f"arg '{orig}' not in input_names, skipped")
                    for k in kwargs:
                        if k not in resolved_name_map:
                            _logger.debug(
                                f"  [SKIP] {resolved_name} #{idx} — "
                                f"kwarg '{k}' not in input_names, skipped")
                # Warn about complex objects without input_filter
                if input_filter is None:
                    for i, a in enumerate(positional):
                        if (not isinstance(a, (torch.Tensor, dict, tuple, list))
                                and not isinstance(a, _SIMPLE_TYPES)):
                            orig = _pnames[i] if i < len(_pnames) else f"arg{i}"
                            if resolved_name_map is None or orig in resolved_name_map:
                                _logger.debug(
                                    f"  [WARN] {resolved_name} #{idx} — "
                                    f"arg '{orig}' is {type(a).__name__}, "
                                    f"skipped (provide input_filter to extract tensors)")
                    for k, v in kwargs.items():
                        if (not isinstance(v, (torch.Tensor, dict, tuple, list))
                                and not isinstance(v, _SIMPLE_TYPES)):
                            if resolved_name_map is None or k in resolved_name_map:
                                _logger.debug(
                                    f"  [WARN] {resolved_name} #{idx} — "
                                    f"kwarg '{k}' is {type(v).__name__}, "
                                    f"skipped (provide input_filter to extract tensors)")

            # --- Level 4: Align mode ---
            if _LEVEL >= 4:
                align_dir = _get_align_dir(resolved_name, idx)
                if align_dir is not None:
                    _device = torch.device("cpu")
                    for t in named_tensors_in.values():
                        _device = t.device
                        break

                    baseline_inputs = _load_align_tensors(align_dir, "inputs.pt", _device)
                    if baseline_inputs is not None:
                        if align_transform is not None:
                            baseline_inputs = align_transform(baseline_inputs, "inputs")
                        _logger.debug(f"[ALIGN] {resolved_name} #{idx} — "
                                       f"replacing inputs from {align_dir}")
                        _compare_and_log(resolved_name, idx, "input_diff",
                                         named_tensors_in, baseline_inputs,
                                         _ALIGN_THRESHOLD)

                        positional, kwargs = _replace_tensor_args(
                            params, positional, kwargs, baseline_inputs,
                            name_map=resolved_name_map)

                        # Warn about baseline keys not found in actual inputs
                        unmatched_baseline = set(baseline_inputs.keys()) - set(named_tensors_in.keys())
                        if unmatched_baseline:
                            _logger.debug(
                                f"  [WARN] {resolved_name} #{idx} — "
                                f"baseline keys not found in actual inputs "
                                f"(not replaced): {sorted(unmatched_baseline)}")
                        # Warn about actual keys not found in baseline
                        unmatched_actual = set(named_tensors_in.keys()) - set(baseline_inputs.keys())
                        if unmatched_actual:
                            _logger.debug(
                                f"  [WARN] {resolved_name} #{idx} — "
                                f"actual input keys not in baseline "
                                f"(not compared): {sorted(unmatched_actual)}")

                        if args and sig_params and sig_params[0].name in {"self", "cls"}:
                            exec_args = (args[0],) + positional
                        else:
                            exec_args = positional

                        result = f(*exec_args, **kwargs)

                        baseline_outputs = _load_align_tensors(
                            align_dir, "outputs.pt", _device)
                        if baseline_outputs is not None:
                            if align_transform is not None:
                                baseline_outputs = align_transform(baseline_outputs, "outputs")
                        if baseline_outputs is not None and result is not None:
                            if isinstance(result, torch.Tensor):
                                actual_out = {"return": result}
                            elif isinstance(result, (tuple, list)):
                                actual_out = {
                                    f"return_{i}": v for i, v in enumerate(result)
                                    if isinstance(v, torch.Tensor)
                                }
                            else:
                                actual_out = {}
                            if actual_out:
                                _compare_and_log(resolved_name, idx, "output_aligned",
                                                 actual_out, baseline_outputs,
                                                 _ALIGN_THRESHOLD)

                        baseline_after = _load_align_tensors(
                            align_dir, "inputs_after.pt", _device)
                        if baseline_after is not None:
                            if align_transform is not None:
                                baseline_after = align_transform(baseline_after, "inputs_after")
                            actual_after = _build_named_tensors(
                                params, positional, kwargs, input_filter,
                                name_map=resolved_name_map)
                            _compare_and_log(resolved_name, idx, "inplace_aligned",
                                             actual_after, baseline_after,
                                             _ALIGN_THRESHOLD)

                        if _LEVEL >= 3:
                            dump_dir = _get_dump_dir(resolved_name, idx)
                            if result is not None:
                                if isinstance(result, torch.Tensor):
                                    _dump_tensors(dump_dir, {"return": result},
                                                  "outputs.pt")
                                elif isinstance(result, (tuple, list)):
                                    ot = {f"return_{i}": v for i, v in enumerate(result)
                                          if isinstance(v, torch.Tensor)}
                                    if ot:
                                        _dump_tensors(dump_dir, ot, "outputs.pt")
                            actual_after_dump = _build_named_tensors(
                                params, positional, kwargs, input_filter,
                                name_map=resolved_name_map)
                            _dump_tensors(dump_dir, actual_after_dump,
                                          "inputs_after.pt")

                        return result
                    else:
                        _logger.debug(f"[ALIGN] {resolved_name} #{idx} — "
                                       f"no inputs.pt in {align_dir}, falling back")
                else:
                    _logger.debug(f"[ALIGN] {resolved_name} #{idx} — "
                                   f"no baseline dump found, falling back")

            # --- Level 1: shapes only (no GPU sync) ---
            _log_shapes(resolved_name, idx,
                        named_tensors_in, named_values, "IN")

            # --- Level 2: stats (GPU sync) ---
            if _LEVEL >= 2:
                _log_stats(resolved_name, idx, named_tensors_in, "IN")

            # --- Level 3: dump inputs ---
            # Warn if no tensors were extracted from inputs
            if not named_tensors_in and _LEVEL >= 2:
                _logger.debug(
                    f"  [WARN] {resolved_name} #{idx} — "
                    f"no tensors extracted from inputs "
                    f"(args: {[type(a).__name__ for a in positional]})")

            dump_dir: Path | None = None
            if _LEVEL >= 3:
                dump_dir = _get_dump_dir(resolved_name, idx)
                _dump_tensors(dump_dir, named_tensors_in, "inputs.pt")

            # --- Execute ---
            try:
                result = f(*args, **kwargs)
            except Exception as exc:
                if dump_dir is not None:
                    _dump_metadata(dump_dir, resolved_name, idx,
                                   named_values, named_tensors_in, exception=str(exc))
                raise

            # --- Log/dump outputs ---
            if result is not None:
                if isinstance(result, torch.Tensor):
                    out_tensors = {"return": result}
                elif isinstance(result, (tuple, list)):
                    out_tensors = {
                        f"return_{i}": v for i, v in enumerate(result)
                        if isinstance(v, torch.Tensor)
                    }
                else:
                    out_tensors = {}

                if out_tensors:
                    _log_shapes(resolved_name, idx, out_tensors, {}, "OUT")
                    if _LEVEL >= 2:
                        _log_stats(resolved_name, idx, out_tensors, "OUT")
                    if _LEVEL >= 3 and dump_dir is not None:
                        _dump_tensors(dump_dir, out_tensors, "outputs.pt")

            # --- Also capture inplace outputs ---
            if _LEVEL >= 3 and dump_dir is not None:
                named_tensors_after = _build_named_tensors(
                    params, positional, kwargs, input_filter,
                    name_map=resolved_name_map)
                _dump_tensors(dump_dir, named_tensors_after, "inputs_after.pt")

            if dump_dir is not None:
                out_t = {}
                if result is not None and isinstance(result, torch.Tensor):
                    out_t = {"return": result}
                _dump_metadata(dump_dir, resolved_name, idx,
                               named_values, named_tensors_in,
                               out_t if out_t else None)

            return result

        wrapper._precision_debug_wrapped = True  # type: ignore[attr-defined]
        return wrapper

    return decorator if func is None else decorator(func)


# ===========================================================================
# Comparison CLI
# ===========================================================================


def _calc_rel_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Relative difference: 1 - 2*(x·y) / (x²+y²). From DeepGEMM."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def _parse_slice(s: str) -> slice | None:
    """Parse 'start:stop' or 'start:stop:step' into a slice."""
    if not s:
        return None
    parts = s.split(":")
    ints = [int(p) if p else None for p in parts]
    if len(ints) == 1:
        return slice(ints[0], ints[0] + 1 if ints[0] is not None else None)
    if len(ints) == 2:
        return slice(ints[0], ints[1])
    if len(ints) == 3:
        return slice(ints[0], ints[1], ints[2])
    return None


def _parse_index_list(s: str) -> list[int] | None:
    """Parse '0,3,7' into a list of ints."""
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",")]


def _apply_dim0_selection(
    t: torch.Tensor, slc: slice | None, indices: list[int] | None
) -> torch.Tensor:
    """Apply slice or index selection on dim0."""
    if indices is not None:
        return t[indices]
    if slc is not None:
        return t[slc]
    return t


def _compare_main(args: argparse.Namespace) -> None:
    baseline_root = Path(args.baseline)
    target_root = Path(args.target)

    if not baseline_root.exists():
        print(f"Error: baseline path does not exist: {baseline_root}")
        return
    if not target_root.exists():
        print(f"Error: target path does not exist: {target_root}")
        return

    rank_dir = f"rank{args.rank}"
    baseline_rank = baseline_root / rank_dir
    target_rank = target_root / rank_dir

    if not baseline_rank.exists():
        baseline_rank = baseline_root
    if not target_rank.exists():
        target_rank = target_root

    op_filter = [p.strip() for p in args.op_filter.split(",") if p.strip()] if args.op_filter else []

    # Parse per-side slicing/indexing
    b_slice = _parse_slice(args.baseline_slice or args.slice or "")
    t_slice = _parse_slice(args.target_slice or args.slice or "")
    b_indices = _parse_index_list(args.baseline_index or args.index or "")
    t_indices = _parse_index_list(args.target_index or args.index or "")

    # Discover ops
    baseline_ops = sorted(
        [d for d in baseline_rank.iterdir() if d.is_dir()]
    ) if baseline_rank.is_dir() else []
    target_ops = sorted(
        [d for d in target_rank.iterdir() if d.is_dir()]
    ) if target_rank.is_dir() else []

    baseline_op_names = {d.name for d in baseline_ops}
    target_op_names = {d.name for d in target_ops}
    common_ops = sorted(baseline_op_names & target_op_names)

    if not common_ops:
        print(f"No common ops found between baseline and target under {rank_dir}/")
        print(f"  Baseline ops: {sorted(baseline_op_names)}")
        print(f"  Target ops: {sorted(target_op_names)}")
        return

    total_compared = 0
    total_passed = 0

    for op_name in common_ops:
        if op_filter and not any(fnmatch.fnmatch(op_name, p) for p in op_filter):
            continue

        b_op_dir = baseline_rank / op_name
        t_op_dir = target_rank / op_name

        b_calls = sorted([d for d in b_op_dir.iterdir() if d.is_dir()])
        t_calls = sorted([d for d in t_op_dir.iterdir() if d.is_dir()])

        n_compare = min(len(b_calls), len(t_calls))
        if n_compare == 0:
            continue

        print(f"\n{'='*80}")
        print(f"Op: {op_name} (baseline={len(b_calls)} calls, target={len(t_calls)} calls)")
        print(f"{'='*80}")

        for i in range(n_compare):
            b_dir = b_calls[i]
            t_dir = t_calls[i]

            for tensor_file in ["inputs.pt", "outputs.pt", "inputs_after.pt"]:
                b_file = b_dir / tensor_file
                t_file = t_dir / tensor_file
                if not b_file.exists() or not t_file.exists():
                    continue

                b_data = torch.load(b_file, weights_only=True, map_location="cpu")
                t_data = torch.load(t_file, weights_only=True, map_location="cpu")

                common_keys = sorted(set(b_data.keys()) & set(t_data.keys()))
                for key in common_keys:
                    b_tensor = b_data[key].float()
                    t_tensor = t_data[key].float()

                    # Apply per-side dim0 selection
                    b_tensor = _apply_dim0_selection(b_tensor, b_slice, b_indices)
                    t_tensor = _apply_dim0_selection(t_tensor, t_slice, t_indices)

                    # Auto-truncate dim0 if still mismatched
                    if b_tensor.shape[0] != t_tensor.shape[0] and len(b_tensor.shape) > 0:
                        min_dim0 = min(b_tensor.shape[0], t_tensor.shape[0])
                        b_tensor = b_tensor[:min_dim0]
                        t_tensor = t_tensor[:min_dim0]

                    if b_tensor.shape != t_tensor.shape:
                        print(
                            f"  {b_dir.name}/{tensor_file}/{key}: "
                            f"shape mismatch {list(b_tensor.shape)} vs {list(t_tensor.shape)}, skipped"
                        )
                        continue

                    total_compared += 1
                    rel_diff = _calc_rel_diff(b_tensor, t_tensor)
                    abs_diff = (b_tensor - t_tensor).abs()
                    max_abs = abs_diff.max().item()
                    mean_abs = abs_diff.mean().item()

                    passed = rel_diff <= args.diff_threshold
                    if passed:
                        total_passed += 1
                    marker = "PASS" if passed else "FAIL"

                    print(
                        f"  [{marker}] {b_dir.name}/{tensor_file}/{key}: "
                        f"shape={list(b_tensor.shape)} "
                        f"rel_diff={rel_diff:.6g} "
                        f"max_abs={max_abs:.6g} "
                        f"mean_abs={mean_abs:.6g}"
                    )

                    if not passed:
                        flat_idx = abs_diff.argmax()
                        coord = tuple(
                            idx.item()
                            for idx in torch.unravel_index(flat_idx, abs_diff.shape)
                        )
                        print(
                            f"         max_abs at coord={coord} "
                            f"baseline={b_tensor[coord].item():.6g} "
                            f"target={t_tensor[coord].item():.6g}"
                        )

    print(f"\n{'='*80}")
    print(f"Summary: {total_passed}/{total_compared} passed (threshold={args.diff_threshold})")
    print(f"{'='*80}")


def _cli_main() -> None:
    parser = argparse.ArgumentParser(
        prog="sglang.kernel_precision_debug",
        description="Kernel precision debug: compare dumped tensors",
    )
    sub = parser.add_subparsers(dest="command")

    cmp = sub.add_parser("compare", help="Compare two dump directories")
    cmp.add_argument("--baseline", required=True, help="Baseline dump directory")
    cmp.add_argument("--target", required=True, help="Target dump directory")
    cmp.add_argument("--op-filter", default="", help="Op name filter (fnmatch)")
    cmp.add_argument("--rank", type=int, default=0, help="Rank to compare")
    cmp.add_argument("--diff-threshold", type=float, default=1e-3, help="Rel diff threshold")
    cmp.add_argument("--slice", default="", help="Dim0 slice for both sides (e.g. '5:10')")
    cmp.add_argument("--baseline-slice", default="", help="Dim0 slice for baseline")
    cmp.add_argument("--target-slice", default="", help="Dim0 slice for target")
    cmp.add_argument("--index", default="", help="Dim0 indices for both sides (e.g. '0,3,7')")
    cmp.add_argument("--baseline-index", default="", help="Dim0 indices for baseline")
    cmp.add_argument("--target-index", default="", help="Dim0 indices for target")

    args = parser.parse_args()
    if args.command == "compare":
        _compare_main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli_main()
