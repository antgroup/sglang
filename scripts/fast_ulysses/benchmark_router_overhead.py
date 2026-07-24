#!/usr/bin/env python3
"""Measure the Python/router overhead around the unchanged NCCL A2A kernels."""

from __future__ import annotations

import argparse
import json
import os
import time
from statistics import mean

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a import (
    A2AMode,
    A2ASlot,
    UlyssesA2AConfig,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a.nccl import (
    NCCLUlyssesA2ABackend,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a.router import (
    UlyssesA2ARouter,
)


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": mean(values),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    }


def measure(fn, x: torch.Tensor, warmup: int, iterations: int) -> dict[str, object]:
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    gpu_ms = []
    cpu_ms = []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        cpu_begin = time.perf_counter()
        begin.record()
        fn(x)
        end.record()
        end.synchronize()
        cpu_ms.append((time.perf_counter() - cpu_begin) * 1000)
        gpu_ms.append(float(begin.elapsed_time(end)))
    return {"gpu_ms": summarize(gpu_ms), "cpu_wall_ms": summarize(cpu_ms)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    x = torch.randn(1, 585, 40, 384, dtype=torch.bfloat16, device=device)

    direct = NCCLUlyssesA2ABackend(dist.group.WORLD)
    router = UlyssesA2ARouter(
        dist.group.WORLD,
        device,
        UlyssesA2AConfig(backend="nccl"),
    )

    def direct_roundtrip(value: torch.Tensor) -> torch.Tensor:
        gathered = direct.input_all_to_all(value, head_dim=2, seq_lens=None)
        return direct.output_all_to_all(gathered, head_dim=2, seq_lens=None)

    def router_roundtrip(value: torch.Tensor) -> torch.Tensor:
        transaction = router.begin_transaction()
        gathered = transaction.input_all_to_all(
            value,
            head_dim=2,
            slot=A2ASlot.PACKED_QKV,
        )
        restored = transaction.output_all_to_all(
            gathered,
            head_dim=2,
            slot=A2ASlot.OUT,
        )
        transaction.close()
        return restored

    def fixed_roundtrip(value: torch.Tensor) -> torch.Tensor:
        gathered = router.fixed_backend_all_to_all(
            value,
            mode=A2AMode.INPUT,
            head_dim=2,
            seq_lens=None,
            slot=A2ASlot.PACKED_QKV,
        )
        return router.fixed_backend_all_to_all(
            gathered,
            mode=A2AMode.OUTPUT,
            head_dim=2,
            seq_lens=None,
            slot=A2ASlot.OUT,
        )

    direct_result = measure(direct_roundtrip, x, args.warmup, args.iterations)
    router_result = measure(router_roundtrip, x, args.warmup, args.iterations)
    fixed_result = measure(fixed_roundtrip, x, args.warmup, args.iterations)
    local_result = {
        "rank": dist.get_rank(),
        "direct": direct_result,
        "router": router_result,
        "fixed": fixed_result,
    }
    gathered_results: list[object] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_results, local_result)
    if dist.get_rank() == 0:
        direct_critical = max(
            item["direct"]["cpu_wall_ms"]["mean"] for item in gathered_results
        )
        router_critical = max(
            item["router"]["cpu_wall_ms"]["mean"] for item in gathered_results
        )
        fixed_critical = max(
            item["fixed"]["cpu_wall_ms"]["mean"] for item in gathered_results
        )
        print(
            "ULYSSES_ROUTER_OVERHEAD_JSON="
            + json.dumps(
                {
                    "iterations": args.iterations,
                    "critical_rank_direct_cpu_wall_mean_ms": direct_critical,
                    "critical_rank_router_cpu_wall_mean_ms": router_critical,
                    "critical_rank_overhead_ms": router_critical - direct_critical,
                    "critical_rank_fixed_cpu_wall_mean_ms": fixed_critical,
                    "critical_rank_fixed_overhead_ms": fixed_critical - direct_critical,
                    "ranks": gathered_results,
                },
                sort_keys=True,
            )
        )

    router.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
