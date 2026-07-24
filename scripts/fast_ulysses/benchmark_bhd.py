#!/usr/bin/env python3
"""Compare genuine fast-ulysses and NCCL for one uniform Ulysses shape."""

from __future__ import annotations

import argparse
import json
import os
from statistics import median

import torch
import torch.distributed as dist
from fast_ulysses import UlyssesGroup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-seq", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def nccl_mode0(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """[B, N/ws, H, D] -> [B, N, H/ws, D]."""
    batch, local_seq, global_heads, head_dim = x.shape
    local_heads = global_heads // world_size
    send = (
        x.view(batch, local_seq, world_size, local_heads, head_dim)
        .permute(2, 0, 1, 3, 4)
        .contiguous()
    )
    output = torch.empty_like(send)
    dist.all_to_all_single(output, send)
    return (
        output.permute(1, 0, 2, 3, 4)
        .contiguous()
        .view(batch, world_size * local_seq, local_heads, head_dim)
    )


def nccl_mode1(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """[B, N, H/ws, D] -> [B, N/ws, H, D]."""
    batch, global_seq, local_heads, head_dim = x.shape
    local_seq = global_seq // world_size
    send = (
        x.view(batch, world_size, local_seq, local_heads, head_dim)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )
    output = torch.empty_like(send)
    dist.all_to_all_single(output, send)
    return (
        output.permute(1, 2, 0, 3, 4)
        .contiguous()
        .view(batch, local_seq, world_size * local_heads, head_dim)
    )


def timed_us(fn, *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        fn()
        ends[index].record()
    torch.cuda.synchronize()
    return [
        float(starts[index].elapsed_time(ends[index]) * 1000)
        for index in range(iterations)
    ]


def local_mode_result(
    *,
    mode: int,
    x: torch.Tensor,
    group: UlyssesGroup,
    world_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    nccl_fn = nccl_mode0 if mode == 0 else nccl_mode1
    tag = f"bhd_mode{mode}"

    expected = nccl_fn(x, world_size)
    actual = group.all_to_all_single_4d(
        x,
        mode=mode,
        tag=tag,
        use_tma=None,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.cuda.synchronize()
    dist.barrier()

    fast_samples = timed_us(
        lambda: group.all_to_all_single_4d(
            x,
            mode=mode,
            tag=tag,
            use_tma=None,
        ),
        warmup=warmup,
        iterations=iterations,
    )
    dist.barrier()
    nccl_samples = timed_us(
        lambda: nccl_fn(x, world_size),
        warmup=warmup,
        iterations=iterations,
    )
    dist.barrier()

    remote_bytes = x.numel() * x.element_size() * (world_size - 1) / world_size
    return {
        "input_shape": list(x.shape),
        "remote_bytes_per_rank": remote_bytes,
        "fast_median_us": median(fast_samples),
        "nccl_median_us": median(nccl_samples),
    }


def main() -> None:
    args = parse_args()
    if args.global_seq <= 0 or args.heads <= 0 or args.head_dim <= 0:
        raise ValueError("N, H, and D must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if args.global_seq % world_size:
        raise ValueError(
            f"N={args.global_seq} must be divisible by Ulysses degree={world_size}"
        )
    if args.heads % world_size:
        raise ValueError(
            f"H={args.heads} must be divisible by Ulysses degree={world_size}"
        )
    if args.head_dim * torch.bfloat16.itemsize % 16:
        raise ValueError(
            "D * sizeof(bf16) must be 16-byte aligned; D must be a multiple of 8"
        )

    local_seq = args.global_seq // world_size
    local_heads = args.heads // world_size
    generator = torch.Generator(device=device)
    generator.manual_seed(20260724 + rank)
    mode0_input = torch.randn(
        1,
        local_seq,
        args.heads,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    mode1_input = torch.randn(
        1,
        args.global_seq,
        local_heads,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    initial_pool_bytes = int(
        os.environ.get("FAST_ULYSSES_BENCH_POOL_BYTES", str(12 << 30))
    )
    group = UlyssesGroup(
        process_group=dist.group.WORLD,
        device=device,
        initial_pool_bytes=initial_pool_bytes,
    )
    try:
        local_results = {
            "rank": rank,
            "gpu": torch.cuda.get_device_name(device),
            "mode0": local_mode_result(
                mode=0,
                x=mode0_input,
                group=group,
                world_size=world_size,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            "mode1": local_mode_result(
                mode=1,
                x=mode1_input,
                group=group,
                world_size=world_size,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
        }
        gathered: list[object] = [None] * world_size
        dist.all_gather_object(gathered, local_results)

        if rank == 0:
            summary: dict[str, object] = {}
            for mode_name in ("mode0", "mode1"):
                fast_us = max(item[mode_name]["fast_median_us"] for item in gathered)
                nccl_us = max(item[mode_name]["nccl_median_us"] for item in gathered)
                remote_bytes = gathered[0][mode_name]["remote_bytes_per_rank"]
                summary[mode_name] = {
                    "input_shape_per_rank": gathered[0][mode_name]["input_shape"],
                    "fast_median_us_critical_rank": fast_us,
                    "fast_remote_gbps": remote_bytes / (fast_us * 1000),
                    "nccl_median_us_critical_rank": nccl_us,
                    "nccl_remote_gbps": remote_bytes / (nccl_us * 1000),
                    "latency_speedup": nccl_us / fast_us,
                }

            fast_pair_us = sum(
                max(item[mode]["fast_median_us"] for item in gathered)
                for mode in ("mode0", "mode1")
            )
            nccl_pair_us = sum(
                max(item[mode]["nccl_median_us"] for item in gathered)
                for mode in ("mode0", "mode1")
            )
            report = {
                "correctness": "bitwise_pass",
                "dtype": "bfloat16",
                "batch": 1,
                "ulysses_degree": world_size,
                "N": args.global_seq,
                "H": args.heads,
                "D": args.head_dim,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "fast_transfer": "auto",
                **summary,
                "mode0_plus_mode1": {
                    "fast_median_us_sum": fast_pair_us,
                    "nccl_median_us_sum": nccl_pair_us,
                    "latency_speedup": nccl_pair_us / fast_pair_us,
                },
            }
            print(
                "\n"
                "mode   Fast median   Fast GB/s   NCCL median   NCCL GB/s   speedup\n"
                f"0      {summary['mode0']['fast_median_us_critical_rank']:9.1f} us"
                f"   {summary['mode0']['fast_remote_gbps']:9.1f}"
                f"   {summary['mode0']['nccl_median_us_critical_rank']:10.1f} us"
                f"   {summary['mode0']['nccl_remote_gbps']:9.1f}"
                f"   {summary['mode0']['latency_speedup']:6.3f}x\n"
                f"1      {summary['mode1']['fast_median_us_critical_rank']:9.1f} us"
                f"   {summary['mode1']['fast_remote_gbps']:9.1f}"
                f"   {summary['mode1']['nccl_median_us_critical_rank']:10.1f} us"
                f"   {summary['mode1']['nccl_remote_gbps']:9.1f}"
                f"   {summary['mode1']['latency_speedup']:6.3f}x\n"
                f"pair   {fast_pair_us:9.1f} us"
                "             "
                f"   {nccl_pair_us:10.1f} us"
                "             "
                f"   {nccl_pair_us / fast_pair_us:6.3f}x",
                flush=True,
            )
            print(
                "FAST_ULYSSES_BHD_JSON=" + json.dumps(report, sort_keys=True),
                flush=True,
            )
    finally:
        group.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
