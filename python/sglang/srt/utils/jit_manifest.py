"""Startup JIT manifest: compile a deployment's kernel specializations in parallel at launch.

An engine started with empty JIT caches (every fresh container) compiles hundreds of kernel
specializations (Triton, DeepGEMM, sglang JIT, humming, TileLang) one at a time on every
rank, inside weight loading, CUDA-graph capture and the startup warmups. Which
specializations a deployment needs is stable, and none of these compilers needs a GPU.
Both sides run under the DeepSeek-V4.1 SM90 ``startup`` subsystem only.

Record: ``python -m sglang.srt.utils.jit_manifest record`` creates
``$SGLANG_CACHE_DIR/jit_manifest/record``; while it exists every scheduler process of an
engine sharing that cache root appends what it compiles to ``<record dir>/<pid>.jsonl``.
``build`` merges those records with the DeepGEMM cache entries of the same run into a
manifest and renames the record dir, which ends the recording.

Replay: the launcher starts ``python -m sglang.srt.utils.jit_manifest replay`` on the
bundled manifest before the schedulers spawn. Its process pool compiles every entry into
the on-disk caches the ranks read, through each compiler's own cache-key and publish path
with the recorded inputs:

* Triton: the recorded ``JITFunction.preload`` specialization, compiled for its target;
* sglang JIT: ``load_jit`` with the recorded arguments;
* humming: ``Compiler.compile`` with the recorded code, header overlay and cubin patch;
* DeepGEMM (C++ JIT, no Python compile entry): the recorded generated source and nvcc
  command, published like DeepGEMM publishes (staging dir, ``.committed``, rename) under
  the recorded entry name, only when nvcc and the include tree are the recorded ones;
* TileLang: the recorded eager call (non-tensor arguments) of a lazy ``tilelang.jit``
  kernel in mhc.py, compiled through its ``get_jit`` handle.

Every cache key covers source, flags and toolchain, so an entry recorded from other code
lands under a key nothing looks up and only costs CPU; a kernel the manifest lacks is
compiled by the rank as before. Nothing waits for the pool.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1
# The manifest the DeepSeek-V4.1 SM90 startup replays: recorded on H20 / H20-3e through the
# startup warmups and real traffic of the mix, prefill and decode roles.
BUNDLED_MANIFEST = "dsv41_sm90"
# Recorded DeepGEMM nvcc commands name the compile's staging dir; replay substitutes its own.
_DG_STAGING_TOKEN = "{staging}"

# ===================================================================
# Launcher side
# ===================================================================


def resolve_manifest(value: str) -> Path:
    """A bundled manifest name (``jit_manifests/<name>.json``) or a path."""
    if os.sep in value or value.endswith(".json"):
        return Path(value).expanduser()
    return Path(__file__).with_name("jit_manifests") / f"{value}.json"


def default_num_workers() -> int:
    # A quarter of the CPUs: the ranks' own loading and compiles keep the rest.
    return max(1, min(32, (os.cpu_count() or 4) // 4))


def start_replay(name: str = BUNDLED_MANIFEST) -> Optional[subprocess.Popen]:
    """Start the replay pool on a manifest; never blocks."""
    path = resolve_manifest(name)
    if not path.is_file():
        logger.info("JIT manifest %s: none at %s; ranks compile on demand", name, path)
        return None
    workers = default_num_workers()
    cmd = [sys.executable, "-m", __name__, "replay", str(path), f"--workers={workers}"]
    logger.info("JIT manifest replay: %s, %d workers", path, workers)
    # NGC-based images set BASH_ENV to a script that runs nvidia-smi in every
    # non-interactive shell; nvcc and DeepGEMM start several shells per compile, and
    # concurrent nvidia-smi calls serialize in the driver (8 parallel nvcc: 4.6 s vs
    # 1.5 s each on H20). The pool only compiles, so it drops it.
    env = {k: v for k, v in os.environ.items() if k != "BASH_ENV"}
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, env=env)


# ===================================================================
# Record side (scheduler processes)
# ===================================================================

_record_lock: Optional[threading.Lock] = None
_record_path: Optional[Path] = None


def record_dir() -> Path:
    return Path(envs.SGLANG_CACHE_DIR.get()).expanduser() / "jit_manifest" / "record"


def is_recording() -> bool:
    return _record_path is not None


def _record(entry: Dict[str, Any]) -> None:
    global _record_path
    if _record_path is None:
        return
    line = json.dumps({"t": time.time(), **entry}, sort_keys=True, default=str)
    with _record_lock:
        if _record_path is None:
            return
        try:
            with open(_record_path, "a") as f:
                f.write(line + "\n")
        except OSError as e:
            # `build` renamed the dir under a live engine: stop, never fail a compile.
            logger.warning("JIT manifest: recording stopped (%s)", e)
            _record_path = None


def maybe_install_recorder() -> None:
    """Record this process's compiles while ``record_dir()`` exists."""
    global _record_path, _record_lock, _tilelang_seen
    directory = record_dir()
    if _record_path is not None or not directory.is_dir():
        return
    if _record_lock is None:
        _record_lock = threading.Lock()
    if _tilelang_seen is None:
        _tilelang_seen = set()
    _record_path = directory / f"{os.getpid()}.jsonl"
    logger.info("JIT manifest: recording compiles to %s", _record_path)
    _install_triton_recorder()
    _install_humming_recorder()


def record_load_jit(args: Tuple[str, ...], kwargs: Dict[str, Any]) -> None:
    _record({"kind": "sglang_jit", "args": list(args), "kwargs": kwargs})


_tilelang_seen: Optional[set] = None


def record_tilelang(wrapper: Any, args: Tuple, kwargs: Dict[str, Any]) -> None:
    """A TileLang eager-mode call: tensor arguments only bind at launch, so the
    specialization is the function plus its non-tensor arguments."""
    import torch

    def scalar(value):
        return None if isinstance(value, torch.Tensor) else value

    entry = {
        "kind": "tilelang",
        "fn": f"{wrapper.__module__}:{wrapper.__qualname__}",
        "args": [scalar(a) for a in args],
        "kwargs": {k: scalar(v) for k, v in sorted(kwargs.items())},
    }
    key = json.dumps(entry, sort_keys=True, default=str)
    if key not in _tilelang_seen:
        _tilelang_seen.add(key)
        _record(entry)


def _install_triton_recorder() -> None:
    import triton.knobs as knobs

    prev = knobs.runtime.jit_cache_hook

    def hook(*, key, repr, fn, compile, is_manual_warmup, already_compiled):
        _record(
            {
                "kind": "triton",
                "fn": f"{fn.module}:{fn.name}",
                "spec": compile["specialization_data"],
            }
        )
        if prev is None:
            return None
        return prev(
            key=key,
            repr=repr,
            fn=fn,
            compile=compile,
            is_manual_warmup=is_manual_warmup,
            already_compiled=already_compiled,
        )

    knobs.runtime.jit_cache_hook = hook


def _install_humming_recorder() -> None:
    try:
        from humming.jit import compiler as humming_compiler
        from humming.utils import cubin as humming_cubin
    except ImportError:
        return
    import shutil
    import tempfile

    from sglang.srt.layers.moe.moe_runner import humming_overlay

    # humming builds kernels from a thread pool, so the patch capture is per thread.
    captured = threading.local()
    orig_patch = humming_cubin.patch_cubin
    orig_compile = humming_compiler.Compiler.compile.__func__

    def patch_cubin(cubin_path, mode, **kwargs):
        captured.mode = mode
        return orig_patch(cubin_path=cubin_path, mode=mode, **kwargs)

    def patch_mode(path: str, postprocess_cubin) -> Optional[str]:
        """The patch postprocess_cubin applies, learned by running it on a copy,
        so a cache hit records the same mode as the build that produced it."""
        if postprocess_cubin is None:
            return None
        captured.mode = None
        with tempfile.TemporaryDirectory() as tmp:
            copy = os.path.join(tmp, "kernel.cubin")
            shutil.copyfile(path, copy)
            try:
                postprocess_cubin(copy)
            except Exception:  # Re-patching a patched cubin may fail; the mode is set.
                pass
        return captured.mode

    def compile(
        cls,
        code,
        sm_version,
        kernel_expr,
        disable_fast_math=False,
        postprocess_cubin=None,
    ):
        path = orig_compile(
            cls, code, sm_version, kernel_expr, disable_fast_math, postprocess_cubin
        )
        _record(
            {
                "kind": "humming",
                "compiler": cls.__name__,
                "code": code,
                "sm": sm_version,
                "expr": kernel_expr,
                "no_fast_math": bool(disable_fast_math),
                **_overlay_fields(humming_overlay),
                "key": os.path.basename(os.path.dirname(path)),
                "patch": patch_mode(path, postprocess_cubin),
            }
        )
        return path

    humming_cubin.patch_cubin = patch_cubin
    humming_compiler.Compiler.compile = classmethod(compile)


@functools.cache
def _digest(path: str) -> str:
    from sglang.srt.layers.moe.moe_runner import humming_overlay

    return humming_overlay.overlay_digest(path)


def _overlay_fields(humming_overlay) -> Dict[str, Optional[str]]:
    """The header overlay humming compiles against in this process (humming_overlay.py):
    its directory under OVERLAY_ROOT and content digest, both part of the cubin's cache key."""
    name = humming_overlay.installed_overlay()
    if name is None:
        return {"overlay": None, "overlay_digest": None}
    return {
        "overlay": name,
        "overlay_digest": _digest(os.path.join(humming_overlay.OVERLAY_ROOT, name)),
    }


# ===================================================================
# Build (offline): records + DeepGEMM cache -> manifest
# ===================================================================


def _read_records(directory: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: r["t"])


def _merge_humming(rows: Iterable[Dict[str, Any]]) -> List[Tuple[float, Dict]]:
    first_t: Dict[str, float] = {}
    entries: Dict[str, Dict[str, Any]] = {}
    patches: Dict[str, set] = {}
    for r in rows:
        key = r["key"]
        first_t.setdefault(key, r["t"])
        entries.setdefault(key, {k: r[k] for k in r if k not in ("t", "key", "patch")})
        patches.setdefault(key, set()).add(r["patch"])
    out = []
    for key, entry in entries.items():
        # A cubin may be byte-patched after nvrtc: replaying without the patch
        # publishes a wrong cubin under the right key, so keep only entries whose
        # patch mode is unambiguous.
        known = patches[key]
        if len(known) == 1:
            out.append((first_t[key], {**entry, "patch": known.pop()}))
    return out


def _scan_deep_gemm(cache_root: Path) -> List[Tuple[float, Dict[str, Any]]]:
    out = []
    for entry_dir in sorted((cache_root / "cache").glob("*")):
        if not (entry_dir / ".committed").exists():
            continue
        meta = json.loads((entry_dir / "meta.json").read_text())
        staging = Path(shlex.split(meta["command"])[1]).parent
        meta["command"] = meta["command"].replace(str(staging), _DG_STAGING_TOKEN)
        out.append(
            (
                (entry_dir / "kernel.cubin").stat().st_mtime,
                {
                    "kind": "deep_gemm",
                    "dir": entry_dir.name,
                    "code": (entry_dir / "kernel.cu").read_text(),
                    "meta": meta,
                },
            )
        )
    return out


def _dedupe(rows: Iterable[Dict[str, Any]], kind: str) -> List[Tuple[float, Dict]]:
    seen: Dict[str, Tuple[float, Dict]] = {}
    for r in rows:
        if r["kind"] != kind:
            continue
        entry = {k: v for k, v in r.items() if k != "t"}
        seen.setdefault(json.dumps(entry, sort_keys=True), (r["t"], entry))
    return list(seen.values())


# Triton spec fields shared by the specializations of one kernel (or by all kernels).
_TRITON_TABLE_FIELDS = ("signature", "constant_keys", "attrs_keys", "options", "target")


def _intern(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Factor repeated content into tables: Triton spec fields, DeepGEMM meta, and the
    lines of humming's generated sources (41 sources of ~11 KB share 370 lines)."""
    tables: Dict[str, List[Any]] = {f"triton_{f}": [] for f in _TRITON_TABLE_FIELDS}
    tables.update(dg_meta=[], humming_lines=[])
    index: Dict[Tuple[str, str], int] = {}

    def ref(table: str, value: Any) -> int:
        text = json.dumps(value, sort_keys=True)
        if (table, text) not in index:
            index[(table, text)] = len(tables[table])
            tables[table].append(value)
        return index[(table, text)]

    packed = []
    for e in entries:
        if e["kind"] == "triton":
            spec = json.loads(e["spec"])
            # "key" and "name" only re-check a preload against the live kernel; replay
            # resolves the kernel by e["fn"] and rebuilds the key from the rest.
            spec.pop("key", None)
            spec.pop("name", None)
            for field in _TRITON_TABLE_FIELDS:
                spec[field] = ref(f"triton_{field}", spec[field])
            e = {**e, "spec": spec}
        elif e["kind"] == "deep_gemm":
            e = {**e, "meta": ref("dg_meta", e["meta"])}
        elif e["kind"] == "humming":
            lines = e["code"].split("\n")
            e = {**e, "code": [ref("humming_lines", line) for line in lines]}
        packed.append(e)
    return {"version": _FORMAT_VERSION, **tables, "entries": packed}


def build_manifest(*, records: Path, deep_gemm_cache: Optional[Path]) -> Dict:
    rows = _read_records(records)
    timed = (
        _dedupe(rows, "triton")
        + _dedupe(rows, "sglang_jit")
        + _merge_humming(r for r in rows if r["kind"] == "humming")
        + _dedupe(rows, "tilelang")
    )
    if deep_gemm_cache is not None:
        timed += _scan_deep_gemm(deep_gemm_cache)
    timed.sort(key=lambda x: x[0])
    return _intern([e for _, e in timed])


def _expand(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The entries of a manifest with its tables resolved: the inverse of _intern."""
    out = []
    for e in manifest["entries"]:
        if e["kind"] == "triton":
            spec = {
                k: manifest[f"triton_{k}"][v] if k in _TRITON_TABLE_FIELDS else v
                for k, v in e["spec"].items()
            }
            e = {**e, "spec": json.dumps(spec, sort_keys=True)}
        elif e["kind"] == "deep_gemm":
            e = {**e, "meta": manifest["dg_meta"][e["meta"]]}
        elif e["kind"] == "humming":
            e = {
                **e,
                "code": "\n".join(manifest["humming_lines"][i] for i in e["code"]),
            }
        out.append(e)
    return out


def merge_manifests(manifests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Union in order (e.g. per GPU model or per role), first occurrence kept."""
    seen, entries = set(), []
    for m in manifests:
        for e in _expand(m):
            key = json.dumps(e, sort_keys=True)
            if key not in seen:
                seen.add(key)
                entries.append(e)
    return _intern(entries)


def write_manifest(manifest: Dict, out: Path) -> None:
    """One table row or entry per line, so manifest updates diff line by line."""
    parts = []
    for key in sorted(manifest):
        value = manifest[key]
        if isinstance(value, list):
            rows = ",\n".join(json.dumps(v, sort_keys=True) for v in value)
            parts.append(f"{json.dumps(key)}: [\n{rows}\n]")
        else:
            parts.append(f"{json.dumps(key)}: {json.dumps(value)}")
    out.write_text("{\n" + ",\n".join(parts) + "\n}\n")


# ===================================================================
# Replay (worker processes)
# ===================================================================


def _rebase_triton_library_paths(value: Any, triton_root: Path) -> Any:
    if isinstance(value, str):
        path = Path(value)
        if path.parts[-5:-1] == ("triton", "backends", "nvidia", "lib"):
            local = triton_root / "backends" / "nvidia" / "lib" / path.name
            if local.is_file():
                return str(local)
    elif isinstance(value, dict):
        return {
            k: _rebase_triton_library_paths(v, triton_root) for k, v in value.items()
        }
    elif isinstance(value, (list, tuple)):
        return type(value)(_rebase_triton_library_paths(v, triton_root) for v in value)
    return value


def _compile_triton(entry: Dict[str, Any], tables: Dict[str, List]) -> str:
    import importlib

    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.runtime import jit as triton_jit

    def resolve(name: str):
        importlib.import_module(name.rpartition(":")[0])
        return triton_jit._triton_jit_function_registry.get(name)

    fn = resolve(entry["fn"])
    if fn is None:
        return "unresolved"

    # Same decoding as JITFunction.preload, for the recorded target instead of the device's.
    def decode(value):
        if tl.dtype.is_dtype(value):
            return tl.dtype(value)
        if isinstance(value, dict) and "constexpr" in value:
            return tl.constexpr(value["constexpr"])
        if isinstance(value, dict) and "jit_function" in value:
            return resolve(value["jit_function"])
        return value

    spec = {
        k: tables[f"triton_{k}"][v] if k in _TRITON_TABLE_FIELDS else v
        for k, v in entry["spec"].items()
    }
    constexprs = {
        tuple(k): decode(v)
        for k, v in zip(spec["constant_keys"], spec["constant_vals"])
    }
    attrs = dict(zip(map(tuple, spec["attrs_keys"]), spec["attrs_vals"]))
    signature = {
        k: triton_jit.convert_to_tuple_if_list(v) for k, v in spec["signature"].items()
    }
    options = {
        k: tuple(v) if isinstance(v, list) else v
        for k, v in _rebase_triton_library_paths(
            spec["options"], Path(triton.__file__).parent
        ).items()
    }
    target = GPUTarget(**spec["target"])
    if fn.is_gluon():
        from triton.experimental.gluon._runtime import GluonASTSource as ASTSource
    src = ASTSource(fn, signature, constexprs, attrs)
    triton.compile(src, target=target, options=options)
    return "ok"


def _compile_sglang_jit(entry: Dict[str, Any], tables: Dict[str, List]) -> str:
    from sglang.kernels.jit.utils.compile.loader import load_jit

    kwargs = {
        k: [tuple(x) if isinstance(x, list) else x for x in v]
        if isinstance(v, list)
        else v
        for k, v in entry["kwargs"].items()
    }
    load_jit(*entry["args"], **kwargs)
    return "ok"


def _compile_tilelang(entry: Dict[str, Any], tables: Dict[str, List]) -> str:
    import importlib

    module, _, qualname = entry["fn"].partition(":")
    wrapper = importlib.import_module(module)
    for part in qualname.split("."):
        wrapper = getattr(wrapper, part, None)
    if wrapper is None or not hasattr(wrapper, "get_jit"):
        return "unresolved"
    wrapper.get_jit().compile(*entry["args"], **entry["kwargs"])
    return "ok"


_humming_base: Optional[Tuple[Any, Any]] = None


def _use_humming_overlay(entry: Dict[str, Any]) -> bool:
    """Point humming's Compiler at the recorded overlay (or the stock headers), as the
    recording rank had; False when this tree lacks that overlay, whose kernels would land
    under keys no rank looks up."""
    global _humming_base
    from humming.jit.compiler import Compiler

    from sglang.srt.layers.moe.moe_runner import humming_overlay

    if _humming_base is None:
        _humming_base = (
            Compiler.__dict__["include_dirs"],
            Compiler.__dict__["cuh_last_update_time"],
        )
    Compiler.include_dirs, Compiler.cuh_last_update_time = _humming_base
    overlay = entry["overlay"]
    if overlay is None:
        return True
    path = os.path.join(humming_overlay.OVERLAY_ROOT, overlay)
    if not os.path.isdir(path) or _digest(path) != entry["overlay_digest"]:
        return False
    humming_overlay.install_include_overlay(Compiler, path)
    return True


def _compile_humming(entry: Dict[str, Any], tables: Dict[str, List]) -> str:
    from humming.jit import compiler as humming_compiler

    if not _use_humming_overlay(entry):
        return "overlay"
    postprocess = None
    if entry["patch"] is not None:
        from humming.utils.cubin import patch_cubin

        mode = entry["patch"]

        def postprocess(cubin_path: str) -> None:
            patch_cubin(cubin_path=cubin_path, mode=mode)

    getattr(humming_compiler, entry["compiler"]).compile(
        "\n".join(tables["humming_lines"][i] for i in entry["code"]),
        sm_version=entry["sm"],
        kernel_expr=entry["expr"],
        disable_fast_math=entry["no_fast_math"],
        postprocess_cubin=postprocess,
    )
    return "ok"


_nvcc_versions: Dict[str, str] = {}


def _nvcc_version(path: str) -> str:
    if path not in _nvcc_versions:
        _nvcc_versions[path] = subprocess.run(
            [path, "--version"], capture_output=True, text=True, check=True
        ).stdout
    return _nvcc_versions[path]


def _compile_deep_gemm(entry: Dict[str, Any], tables: Dict[str, List]) -> str:
    import shutil

    meta = tables["dg_meta"][entry["meta"]]
    root = Path(os.environ.get("DG_JIT_CACHE_DIR") or envs.SGLANG_DG_CACHE_DIR.get())
    final = root.expanduser() / "cache" / entry["dir"]
    if (final / ".committed").exists():
        return "cached"
    # A post hook rewrites the cubin through a script this replay does not run.
    if meta.get("compiler_options", {}).get("post_hook"):
        return "unsupported"
    compiler = meta["compiler_info"]
    if not os.path.exists(compiler["path"]) or (
        _nvcc_version(compiler["path"]) != compiler["version"]
    ):
        return "toolchain"
    if not all(os.path.isdir(d) for d in meta["config"]["include_dirs"]):
        return "toolchain"
    staging = final.parent.parent / "tmp" / f"{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        (staging / "kernel.cu").write_text(entry["code"])
        command = meta["command"].replace(_DG_STAGING_TOKEN, str(staging))
        subprocess.run(shlex.split(command), check=True, capture_output=True)
        (staging / "meta.json").write_text(json.dumps({**meta, "command": command}))
        (staging / ".committed").touch()
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(staging, final)
        except OSError:
            if not final.exists():
                raise
            return "cached"  # A rank published the same entry first.
        return "ok"
    finally:
        shutil.rmtree(staging, ignore_errors=True)


_COMPILERS = {
    "triton": _compile_triton,
    "sglang_jit": _compile_sglang_jit,
    "humming": _compile_humming,
    "deep_gemm": _compile_deep_gemm,
    "tilelang": _compile_tilelang,
}
_worker_tables: Dict[str, List] = {}


def _load_tables(manifest: Dict[str, Any]) -> Dict[str, List]:
    return {k: v for k, v in manifest.items() if k not in ("version", "entries")}


def _init_worker(manifest_path: Path) -> None:
    # Read from the file rather than passed in: a spawn start blocks on the pipe while
    # the child imports, once the pickled arguments exceed the pipe buffer.
    global _worker_tables
    _worker_tables = _load_tables(json.loads(manifest_path.read_text()))
    # Diagnostics of entries that fail here (the rank compiles those) would flood the
    # server log; the parent prints one summary.
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    # The ranks' own startup work stays ahead of speculative compiles for the CPU.
    os.nice(10)


def _run_entry(entry: Dict[str, Any]) -> Tuple[str, str, float, float, int]:
    start = time.time()
    try:
        status = _COMPILERS[entry["kind"]](entry, _worker_tables)
    # Best effort: the rank compiles whatever the pool could not.
    except Exception as e:
        detail = e.stderr if isinstance(e, subprocess.CalledProcessError) else e
        status = f"error: {type(e).__name__}: {str(detail)[-300:]}"
    return entry["kind"], status, start, time.time() - start, os.getpid()


def _entry_name(entry: Dict[str, Any]) -> str:
    for key in ("fn", "dir", "expr", "args"):
        if entry.get(key):
            return str(entry[key])
    return entry["kind"]


def replay(
    manifest_path: Path, *, workers: int, log_path: Optional[Path] = None
) -> Dict[str, Dict[str, int]]:
    import multiprocessing
    from collections import Counter, defaultdict
    from concurrent.futures import ProcessPoolExecutor, as_completed

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != _FORMAT_VERSION:
        logger.warning("JIT manifest %s: unknown version; skipped", manifest_path)
        return {}
    tic = time.perf_counter()
    counts: Dict[str, Counter] = defaultdict(Counter)
    busy = Counter()
    errors: List[str] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(manifest_path,),
    ) as pool:
        futures = {pool.submit(_run_entry, e): e for e in manifest["entries"]}
        rows = []
        for f in as_completed(futures):
            try:
                kind, status, start, secs, pid = f.result()
            except Exception as e:  # A worker died (BrokenProcessPool): count, go on.
                kind, status, start, secs, pid = "pool", f"error: {e!r}", 0.0, 0.0, 0
            counts[kind][status.split(":")[0]] += 1
            busy[kind] += secs
            name = _entry_name(futures[f])
            if status.startswith("error") and len(errors) < 5:
                errors.append(f"{kind} {name} {status}")
            rows.append(
                dict(
                    kind=kind, name=name, status=status, start=start, secs=secs, pid=pid
                )
            )
    if log_path is not None:
        log_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {k: dict(v) for k, v in counts.items()}
    print(
        f"[jit_manifest] replay {manifest_path.name}: {len(manifest['entries'])} entries, "
        f"{workers} workers, {time.perf_counter() - tic:.1f} s wall; "
        + "; ".join(f"{k} {summary[k]} {busy[k]:.0f} s" for k in sorted(summary))
        + "".join(f"\n[jit_manifest]   {e}" for e in errors),
        file=sys.stderr,
        flush=True,
    )
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    # This offline tool (also spawned by start_replay) has no model/runtime
    # context. Honor the requested startup subsystem before exposing its CLI.
    if (
        not envs.SGLANG_OPT_DSV41_SM90.get()
        or "startup" in envs.SGLANG_OPT_DSV41_SM90_DISABLE.get()
    ):
        raise SystemExit(
            "JIT manifests require SGLANG_OPT_DSV41_SM90 with startup enabled"
        )
    parser = argparse.ArgumentParser(prog="python -m sglang.srt.utils.jit_manifest")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "record",
        help="create the record dir under SGLANG_CACHE_DIR; engines started while it "
        "exists record their compiles there",
    )
    b = sub.add_parser("build", help="merge a record run into a manifest")
    b.add_argument("out", type=Path)
    b.add_argument(
        "--records",
        type=Path,
        default=None,
        help="record dir (default: the one `record` created)",
    )
    b.add_argument(
        "--deep-gemm-cache",
        type=Path,
        default=None,
        help="DG_JIT_CACHE_DIR of the record run (default: this environment's)",
    )
    b.add_argument("--no-deep-gemm", action="store_true")
    m = sub.add_parser("merge", help="union of manifests (e.g. recorded per GPU model)")
    m.add_argument("inputs", type=Path, nargs="+")
    m.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("replay", help="compile a manifest into the JIT caches")
    r.add_argument("manifest", type=Path)
    r.add_argument("--workers", type=int, default=default_num_workers())
    r.add_argument("--log", type=Path, default=None, help="per-entry JSONL timings")
    args = parser.parse_args(argv)
    if args.cmd == "record":
        directory = record_dir()
        directory.mkdir(parents=True, exist_ok=True)
        print(f"recording into {directory} while it exists")
        return
    if args.cmd == "build":
        dg = None
        if not args.no_deep_gemm:
            dg = args.deep_gemm_cache or Path(
                os.environ.get("DG_JIT_CACHE_DIR") or envs.SGLANG_DG_CACHE_DIR.get()
            )
        records = args.records or record_dir()
        manifest = build_manifest(records=records, deep_gemm_cache=dg)
    elif args.cmd == "merge":
        manifest = merge_manifests([json.loads(p.read_text()) for p in args.inputs])
    if args.cmd in ("build", "merge"):
        out = args.out
        write_manifest(manifest, out)
        kinds = {}
        for e in manifest["entries"]:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        print(f"wrote {out}: {kinds}")
        if args.cmd == "build" and records == record_dir():
            done = records.with_name(f"record-{time.strftime('%Y%m%dT%H%M%S')}")
            records.rename(done)
            print(f"recording ended; records kept in {done}")
    else:
        replay(args.manifest, workers=args.workers, log_path=args.log)


if __name__ == "__main__":
    main()
