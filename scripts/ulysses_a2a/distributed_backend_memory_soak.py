#!/usr/bin/env python3
"""Isolate backend memory stability while keeping the process group fixed."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a import (
    A2ASlot,
    UlyssesA2AConfig,
    UlyssesA2ARouter,
)

MIB = 1024 * 1024


def make_tensor(rank: int, generation: int) -> torch.Tensor:
    shape = (1, 8, 8, 128)
    count = 1
    for size in shape:
        count *= size
    values = (
        torch.arange(count, dtype=torch.int64, device="cuda")
        + rank * 17
        + generation * 29
    ) % 1021
    return values.reshape(shape).to(torch.bfloat16)


def slope_mib(values: list[int]) -> float:
    xs = list(range(len(values)))
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return numerator / denominator / MIB


def summarize(values: list[int]) -> dict[str, float]:
    return {
        "first_mib": values[0] / MIB,
        "last_mib": values[-1] / MIB,
        "delta_mib": (values[-1] - values[0]) / MIB,
        "range_mib": (max(values) - min(values)) / MIB,
        "slope_mib_per_cycle": slope_mib(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["nccl", "sgl_p2p"], required=True)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--calls-per-cycle", type=int, default=8)
    args = parser.parse_args()
    if args.cycles < 3:
        raise ValueError("backend memory soak requires at least three cycles")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    if dist.get_world_size() != 4:
        raise ValueError("backend memory soak requires world_size=4")

    metric_tensor = torch.zeros(3, dtype=torch.int64, device=device)
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.MAX)
    torch.cuda.synchronize()

    device_used_samples: list[int] = []
    allocated_samples: list[int] = []
    reserved_samples: list[int] = []
    backend_peak_bytes = 0

    for generation in range(args.cycles):
        router = UlyssesA2ARouter(
            dist.group.WORLD,
            device,
            UlyssesA2AConfig(
                backend=args.backend,
                p2p_tk_style=True,
            ),
        )
        for call in range(args.calls_per_cycle):
            x = make_tensor(rank, generation * args.calls_per_cycle + call)
            with router.begin_transaction() as transaction:
                y = transaction.input_all_to_all(
                    x,
                    head_dim=2,
                    slot=A2ASlot.PACKED_QKV,
                )
                z = transaction.output_all_to_all(
                    y,
                    head_dim=2,
                    slot=A2ASlot.OUT,
                )
            if not torch.equal(z, x):
                raise AssertionError(
                    f"{args.backend} backend cycle {generation} mismatch"
                )
            del x, y, z
        torch.cuda.synchronize()
        if router.sgl_p2p is not None:
            backend_peak_bytes = max(
                backend_peak_bytes,
                int(router.sgl_p2p.max_bytes),
            )
        router.close()
        del router
        torch.cuda.synchronize()

        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        metric_tensor[0] = total_bytes - free_bytes
        metric_tensor[1] = torch.cuda.memory_allocated(device)
        metric_tensor[2] = torch.cuda.memory_reserved(device)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.MAX)
        torch.cuda.synchronize()
        device_used, allocated, reserved = [
            int(value) for value in metric_tensor.cpu().tolist()
        ]
        device_used_samples.append(device_used)
        allocated_samples.append(allocated)
        reserved_samples.append(reserved)

    report = {
        "backend": args.backend,
        "world_size": dist.get_world_size(),
        "process_group": "fixed_world",
        "cycles": args.cycles,
        "calls_per_cycle": args.calls_per_cycle,
        "backend_peak_buffer_mib": backend_peak_bytes / MIB,
        "device_used": summarize(device_used_samples),
        "torch_allocated": summarize(allocated_samples),
        "torch_reserved": summarize(reserved_samples),
    }

    failures = []
    for key in ("device_used", "torch_allocated", "torch_reserved"):
        summary = report[key]
        if summary["range_mib"] >= 64:
            failures.append(f"{key} range >=64 MiB: {summary}")
        if summary["slope_mib_per_cycle"] > 1:
            failures.append(f"{key} slope >1 MiB/cycle: {summary}")

    dist.destroy_process_group()
    if rank == 0:
        print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise AssertionError("; ".join(failures))
    if rank == 0:
        print(
            "DISTRIBUTED_BACKEND_MEMORY_SOAK_OK "
            f"backend={args.backend} cycles={args.cycles}"
        )


if __name__ == "__main__":
    main()
