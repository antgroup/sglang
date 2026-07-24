#!/usr/bin/env python3
"""Correctness, rank-skew safety, and latency check for the SGLang adapters."""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a import (
    A2AMode,
    A2ASlot,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a.fast_ulysses import (
    FastUlyssesA2ABackend,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a.nccl import (
    NCCLUlyssesA2ABackend,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--local-seq", type=int, default=585)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--packed-head-dim", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--skew-iterations", type=int, default=40)
    parser.add_argument("--skew-cycles", type=int, default=10_000_000)
    parser.add_argument(
        "--transfer", choices=("auto", "sm", "tma", "ce"), default="auto"
    )
    return parser.parse_args()


def elapsed_roundtrips(
    fn,
    x: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn(x)
        end.record()
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)))
    return samples


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean_ms": mean(samples),
        "p50_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def main() -> None:
    args = parse_args()
    tag_pool_size = int(os.environ.get("SGLANG_FAST_ULYSSES_TAG_POOL_SIZE", "1"))
    if args.skew_iterations <= tag_pool_size:
        raise ValueError(
            "skew-iterations must exceed SGLANG_FAST_ULYSSES_TAG_POOL_SIZE "
            "so the rank-skew test exercises an actual tag reuse"
        )
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if args.heads % world_size:
        raise ValueError(
            f"heads={args.heads} must be divisible by world_size={world_size}"
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(20260724 + rank)
    x = torch.randn(
        args.batch,
        args.local_seq,
        args.heads,
        args.packed_head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    nccl = NCCLUlyssesA2ABackend(dist.group.WORLD)
    fast = FastUlyssesA2ABackend(
        dist.group.WORLD,
        device,
        transfer=args.transfer,
    )

    reference = nccl.input_all_to_all(x, head_dim=2, seq_lens=None)
    actual = fast.all_to_all(x, int(A2AMode.INPUT), slot=A2ASlot.PACKED_QKV)
    actual_snapshot = actual.clone()
    restored = fast.all_to_all(
        actual,
        int(A2AMode.OUTPUT),
        slot=A2ASlot.OUT,
    ).clone()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual_snapshot, reference, rtol=0, atol=0)
    torch.testing.assert_close(restored, x, rtol=0, atol=0)

    # Rank 0 deliberately delays consumption. The next collective's
    # stream-ordered pre-write barrier must prevent peers from overwriting it.
    for iteration in range(args.skew_iterations):
        skew_input = x + iteration
        skew_output = fast.all_to_all(
            skew_input,
            int(A2AMode.INPUT),
            slot=A2ASlot.PACKED_QKV,
        )
        if rank == 0 and args.skew_cycles:
            torch.cuda._sleep(args.skew_cycles)
        skew_snapshot = skew_output.clone()
        skew_restored = fast.all_to_all(
            skew_output,
            int(A2AMode.OUTPUT),
            slot=A2ASlot.OUT,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(skew_restored, skew_input, rtol=0, atol=0)
        expected = nccl.input_all_to_all(skew_input, head_dim=2, seq_lens=None)
        torch.testing.assert_close(skew_snapshot, expected, rtol=0, atol=0)

    def nccl_roundtrip(value: torch.Tensor) -> torch.Tensor:
        gathered = nccl.input_all_to_all(value, head_dim=2, seq_lens=None)
        return nccl.output_all_to_all(gathered, head_dim=2, seq_lens=None)

    def fast_roundtrip(value: torch.Tensor) -> torch.Tensor:
        gathered = fast.all_to_all(
            value,
            int(A2AMode.INPUT),
            slot=A2ASlot.PACKED_QKV,
        )
        return fast.all_to_all(
            gathered,
            int(A2AMode.OUTPUT),
            slot=A2ASlot.OUT,
        )

    nccl_samples = elapsed_roundtrips(
        nccl_roundtrip,
        x,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    fast_samples = elapsed_roundtrips(
        fast_roundtrip,
        x,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    local_result = {
        "rank": rank,
        "device": torch.cuda.get_device_name(device),
        "shape": list(x.shape),
        "bytes_per_rank": x.numel() * x.element_size(),
        "transfer": args.transfer,
        "nccl_roundtrip": summarize(nccl_samples),
        "fast_ulysses_roundtrip": summarize(fast_samples),
    }
    gathered_results: list[object] = [None] * world_size
    dist.all_gather_object(gathered_results, local_result)

    if rank == 0:
        nccl_mean = max(
            result["nccl_roundtrip"]["mean_ms"] for result in gathered_results
        )
        fast_mean = max(
            result["fast_ulysses_roundtrip"]["mean_ms"] for result in gathered_results
        )
        report = {
            "genuine_fast_ulysses": True,
            "correctness": "pass",
            "rank_skew_reuse_safety": "pass",
            "world_size": world_size,
            "iterations": args.iterations,
            "critical_rank_nccl_roundtrip_mean_ms": nccl_mean,
            "critical_rank_fast_roundtrip_mean_ms": fast_mean,
            "critical_rank_speedup": nccl_mean / fast_mean,
            "ranks": gathered_results,
        }
        print("FAST_ULYSSES_BENCHMARK_JSON=" + json.dumps(report, sort_keys=True))

    fast.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
