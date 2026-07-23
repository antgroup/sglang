#!/usr/bin/env python3
"""Benchmark semantic A2A on shapes observed in the LingBot model probe."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a import (
    A2ASlot,
    UlyssesA2AConfig,
    UlyssesA2ARouter,
)


@dataclass(frozen=True)
class Case:
    name: str
    mode: str
    shape: tuple[int, int, int, int]
    slot: A2ASlot


CASES = (
    Case("packed_qkv_input", "input", (1, 64, 40, 384), A2ASlot.PACKED_QKV),
    Case("attention_output", "output", (1, 256, 10, 128), A2ASlot.OUT),
)


def make_tensor(case: Case, rank: int) -> torch.Tensor:
    numel = 1
    for dim in case.shape:
        numel *= dim
    values = torch.arange(numel, device="cuda", dtype=torch.int64)
    values = ((values + rank * 97) % 2048).to(torch.bfloat16)
    return values.reshape(case.shape)


def run_once(router: UlyssesA2ARouter, case: Case, x: torch.Tensor) -> torch.Tensor:
    if case.mode == "input":
        return router.input_all_to_all(
            x,
            head_dim=2,
            slot=case.slot,
        )
    return router.output_all_to_all(
        x,
        head_dim=2,
        slot=case.slot,
    )


def max_elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.synchronize()
    value = torch.tensor(
        [start.elapsed_time(end)],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def benchmark_case(
    router: UlyssesA2ARouter,
    case: Case,
    x: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> list[float]:
    for _ in range(warmup):
        run_once(router, case, x)
    torch.cuda.synchronize()

    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run_once(router, case, x)
        end.record()
        samples.append(max_elapsed_ms(start, end) / iterations)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)

    routers = {
        backend: UlyssesA2ARouter(
            dist.group.WORLD,
            device,
            UlyssesA2AConfig(backend=backend, p2p_tk_style=True),
        )
        for backend in ("nccl", "sgl_p2p")
    }
    inputs = {case.name: make_tensor(case, rank) for case in CASES}

    correctness = {}
    for case in CASES:
        expected = run_once(routers["nccl"], case, inputs[case.name])
        actual = run_once(routers["sgl_p2p"], case, inputs[case.name])
        correctness[case.name] = bool(torch.equal(expected, actual))
        if not correctness[case.name]:
            raise AssertionError(f"{case.name}: sgl_p2p differs from NCCL")

    measurements: dict[str, dict[str, list[float]]] = {case.name: {} for case in CASES}
    # Alternate order by trial group to reduce one-sided thermal/cache bias.
    for order in (("nccl", "sgl_p2p"), ("sgl_p2p", "nccl")):
        for backend in order:
            for case in CASES:
                measurements[case.name].setdefault(backend, []).extend(
                    benchmark_case(
                        routers[backend],
                        case,
                        inputs[case.name],
                        warmup=args.warmup,
                        iterations=args.iterations,
                        trials=args.trials,
                    )
                )

    report = {
        "world_size": dist.get_world_size(),
        "device": torch.cuda.get_device_name(local_rank),
        "dtype": str(torch.bfloat16),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "trials_per_order": args.trials,
        "correctness": correctness,
        "cases": [],
    }
    for case in CASES:
        by_backend = measurements[case.name]
        nccl_median = statistics.median(by_backend["nccl"])
        p2p_median = statistics.median(by_backend["sgl_p2p"])
        report["cases"].append(
            {
                **asdict(case),
                "slot": case.slot.value,
                "nccl_ms": by_backend["nccl"],
                "sgl_p2p_ms": by_backend["sgl_p2p"],
                "nccl_median_ms": nccl_median,
                "sgl_p2p_median_ms": p2p_median,
                "speedup": nccl_median / p2p_median,
            }
        )

    for router in routers.values():
        router.close()
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
