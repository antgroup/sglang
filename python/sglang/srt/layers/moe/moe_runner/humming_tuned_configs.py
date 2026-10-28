"""Measured humming MoE tile tables for the DeepSeek-V4.1 SM90 path (dsv41_sm90 "moe").

A table file maps a layer key (expert count, padded w13/w2 GEMM shapes, dtypes, gemm type) to
the w13 and w2 tuning lists humming takes: ``[[lo_rows, hi_rows, config], ...]`` applying to
``lo_rows < rows <= hi_rows`` with rows = tokens * top_k. The tables were measured per GPU with
a humming header overlay (humming_overlay.py) and live in
``humming_configs/device_name=<GPU>,humming_overlay=<overlay>[,<variant>].json``; a variant file
re-picks rows for one role (or the DSpark draft MoE) and holds only the keys it re-picks.
Each file's ``meta`` records where and how it was measured.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

TUNED_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "humming_configs")
# humming's own tables end at 1 << 30 (humming.tune.base.DeviceHeuristics.get_configs).
MAX_ROWS = 1 << 30

# Variant tables a role reads before the overlay's main tables, first match wins: draft holds
# only the DSpark draft MoE keys; dpdec (TP4 key) is DP-attention decode, tp8dec (TP8 key) is
# TP8 MoE under DP8 attention; mixdec re-picks mix-server decode rows.
ROLE_VARIANTS = {
    "mix": ("draft", "mixdec"),
    "prefill": ("draft",),
    "decode": ("draft", "dpdec", "tp8dec"),
    "encoder": (),
}

# A device whose missing files are read from another device's: the H20 and H20-3e tables
# measured on H20-3e were byte-identical, so only the H20-3e copy ships.
DEVICE_ALIASES = {"NVIDIA H20": "NVIDIA H20-3e"}

TuningTable = list[list[Any]]


def tuned_layer_key(*, w13_meta, w2_meta, gemm_type: str) -> str:
    # Shapes are the padded kernel shapes; the scale layout changes smem use and legal tiles.
    return (
        f"E={w13_meta.num_experts},"
        f"w13={w13_meta.shape_n}x{w13_meta.shape_k},"
        f"w2={w2_meta.shape_n}x{w2_meta.shape_k},"
        f"a={w13_meta.a_dtype}/g{w13_meta.input_scale_group_size},"
        f"b={w13_meta.b_dtype}/{w13_meta.bs_dtype}/g{w13_meta.weight_scale_group_size},"
        f"gemm={gemm_type}"
    )


def tuned_config_path(
    device_name: str, overlay: str, variant: Optional[str] = None
) -> str:
    device_name = device_name.replace(" ", "_").replace("/", "_")
    suffix = f",humming_overlay={overlay}" + (f",{variant}" if variant else "")
    return os.path.join(TUNED_CONFIG_DIR, f"device_name={device_name}{suffix}.json")


def _block_m_at(table: TuningTable, rows: int) -> Optional[int]:
    for lo, hi, config in table:
        if lo < rows <= hi:
            return config["block_shape"][0]
    return None


def validate_tables(w13: TuningTable, w2: TuningTable) -> Optional[str]:
    """Why the pair is unusable, or None.

    Each table must tile (0, MAX_ROWS] without gaps (the humming launcher asserts on an uncovered
    row count), and w2 must use w13's block_m on every row range: the runner aligns the routed
    rows (moe_align_block_size) once with w13's block_m and both GEMMs read that layout."""
    for name, table in (("w13", w13), ("w2", w2)):
        prev_hi = 0
        for lo, hi, _ in table:
            if lo != prev_hi or hi <= lo:
                return f"{name} ranges are not contiguous at ({lo}, {hi}]"
            prev_hi = hi
        if prev_hi != MAX_ROWS:
            return f"{name} ends at {prev_hi}, not {MAX_ROWS}"
    for hi in sorted({entry[1] for entry in w13} | {entry[1] for entry in w2}):
        if _block_m_at(w13, hi) != _block_m_at(w2, hi):
            return f"w13/w2 block_m differ at rows {hi}"
    return None


@functools.cache
def _load_tables(path: str) -> dict[str, dict[str, TuningTable]]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)["tables"]


def _lookup_file(path: str, key: str) -> Optional[tuple[TuningTable, TuningTable]]:
    entry = _load_tables(path).get(key)
    if entry is None:
        return None
    error = validate_tables(entry["w13"], entry["w2"])
    if error is not None:
        logger.warning(f"Ignoring tuned humming MoE tables for {key}: {error}")
        return None
    logger.info(f"Using tuned humming MoE tables for {key} from {path}")
    return entry["w13"], entry["w2"]


@functools.cache
def lookup_tuned_tables(
    *, key: str, device_name: str, role: str, overlays: tuple[str, ...]
) -> Optional[tuple[TuningTable, TuningTable]]:
    """The (w13, w2) tables for a layer key, or None for humming's heuristics.

    ``overlays`` names the installed overlay's tables, its own first and the one it
    aliases last (humming_overlay.installed_overlay_tables); variants are read
    under the last. A device reads its own file before its DEVICE_ALIASES one."""
    if not overlays:
        return None
    devices = [device_name] + (
        [DEVICE_ALIASES[device_name]] if device_name in DEVICE_ALIASES else []
    )
    candidates = [(overlays[-1], variant) for variant in ROLE_VARIANTS[role]]
    candidates += [(overlay, None) for overlay in overlays]
    for overlay, variant in candidates:
        for device in devices:
            tables = _lookup_file(tuned_config_path(device, overlay, variant), key)
            if tables is not None:
                return tables
    logger.info(f"No tuned humming MoE tables for {key} on {device_name}")
    return None
