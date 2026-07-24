#!/usr/bin/env python3
"""Extract comparable steady-state LingBot metrics from one or more server logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

SESSION_RE = re.compile(r"\[session_id=([^\]]+)\]")
DEPLOY_RE = re.compile(r"chunk_idx=(\d+).*?scheduler=([\d.]+)ms.*?total=([\d.]+)ms")
DENOISE_RE = re.compile(
    r"\[LingBotWorldCausalDMDDenoisingStage\] finished in ([\d.]+) seconds"
)
KV_MODE_RE = re.compile(r"block_idx=(\d+).*?mode=([a-z_]+)")
WARMUP_RE = re.compile(r"startup warmup complete:.*?elapsed=([\d.]+)s")
BACKEND_RE = re.compile(r"Ulysses A2A router selected backend=([^\s]+)")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def stats_by_mode(
    values: list[tuple[str | None, float]],
) -> dict[str, dict[str, float | int]]:
    modes = sorted({mode for mode, _ in values if mode is not None})
    return {
        mode: stats([value for item_mode, value in values if item_mode == mode])
        for mode in modes
    }


def parse_log(path: Path, drop_chunks: int) -> dict[str, object]:
    sessions: dict[str, dict[str, object]] = {}
    warmups = []
    backends = []
    genuine_marker = False

    for line in path.read_text(errors="replace").splitlines():
        if match := WARMUP_RE.search(line):
            warmups.append(float(match.group(1)))
        if match := BACKEND_RE.search(line):
            backends.append(match.group(1))
        genuine_marker |= "Initialized genuine fast_ulysses backend" in line

        session_match = SESSION_RE.search(line)
        if session_match is None:
            continue
        session_id = session_match.group(1)
        session = sessions.setdefault(
            session_id,
            {
                "deploy": [],
                "denoise": [],
                "current_block": None,
                "current_mode": None,
            },
        )
        if match := KV_MODE_RE.search(line):
            session["current_block"] = int(match.group(1))
            session["current_mode"] = match.group(2)
        if match := DEPLOY_RE.search(line):
            session["deploy"].append(
                (
                    int(match.group(1)),
                    float(match.group(2)),
                    float(match.group(3)),
                    session["current_mode"],
                )
            )
        if match := DENOISE_RE.search(line):
            session["denoise"].append(
                (session["current_mode"], float(match.group(1)) * 1000)
            )

    candidates = [
        (session_id, values)
        for session_id, values in sessions.items()
        if values["deploy"]
    ]
    if not candidates:
        raise ValueError(f"No LingBot deploy timing session found in {path}")
    session_id, selected = max(candidates, key=lambda item: len(item[1]["deploy"]))
    deploy = [item for item in selected["deploy"] if item[0] >= drop_chunks]
    # Denoising may have one terminal/cancelled iteration without a matching
    # deploy timing, so report the complete selected-session series instead of
    # pairing it to deploy chunk indexes.
    denoise = list(selected["denoise"])

    return {
        "path": str(path.resolve()),
        "session_id": session_id,
        "detected_backends": sorted(set(backends)),
        "genuine_fast_ulysses_marker": genuine_marker,
        "startup_warmup_seconds": warmups[-1] if warmups else None,
        "dropped_initial_chunks": drop_chunks,
        "denoise_ms": stats([item[1] for item in denoise]),
        "denoise_ms_by_mode": stats_by_mode(denoise),
        "scheduler_ms": stats([item[1] for item in deploy]),
        "scheduler_ms_by_mode": stats_by_mode([(item[3], item[1]) for item in deploy]),
        "deploy_total_ms": stats([item[2] for item in deploy]),
        "deploy_total_ms_by_mode": stats_by_mode(
            [(item[3], item[2]) for item in deploy]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--drop-chunks",
        type=int,
        default=1,
        help="Drop the cold first chunk by default.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "drop_chunks": args.drop_chunks,
        "logs": [parse_log(path, args.drop_chunks) for path in args.logs],
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
