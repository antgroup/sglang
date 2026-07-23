#!/usr/bin/env python3
"""Correctness/lifecycle checks for the coordinator-owned sgl_p2p backend."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a import (
    A2ASlot,
    UlyssesA2AConfig,
    UlyssesA2ARouter,
)


def make_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, rank: int, offset: int = 0
) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    values = (
        (torch.arange(count, device="cuda", dtype=torch.int64) + offset) % 89
    ) + rank
    return values.reshape(shape).to(dtype=dtype)


def gather(x: torch.Tensor) -> list[torch.Tensor]:
    values = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(values, x)
    return values


def expected_input(x: torch.Tensor, *, rank: int, world_size: int) -> torch.Tensor:
    h_local = x.shape[2] // world_size
    return torch.cat(
        [item[:, :, rank * h_local : (rank + 1) * h_local, :] for item in gather(x)],
        dim=1,
    )


def expected_output(x: torch.Tensor, *, rank: int, world_size: int) -> torch.Tensor:
    s_local = x.shape[1] // world_size
    return torch.cat(
        [item[:, rank * s_local : (rank + 1) * s_local, :, :] for item in gather(x)],
        dim=2,
    )


def assert_equal(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        delta = (
            (actual.float() - expected.float()).abs().max().item()
            if actual.shape == expected.shape
            else "shape"
        )
        raise AssertionError(
            f"{label}: actual={tuple(actual.shape)} expected={tuple(expected.shape)} "
            f"max_abs={delta}"
        )


def run_roundtrip(
    router: UlyssesA2ARouter,
    x: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    label: str,
) -> None:
    with router.begin_transaction() as transaction:
        y = transaction.input_all_to_all(x, head_dim=2, slot=A2ASlot.PACKED_QKV)
        assert_equal(
            y,
            expected_input(x, rank=rank, world_size=world_size),
            f"{label}-input",
        )
        z = transaction.output_all_to_all(y, head_dim=2, slot=A2ASlot.OUT)
    assert_equal(
        z,
        expected_output(y, rank=rank, world_size=world_size),
        f"{label}-output",
    )
    assert_equal(z, x, f"{label}-roundtrip")


def run_cross_stream(
    router: UlyssesA2ARouter,
    x: torch.Tensor,
    *,
    rank: int,
    world_size: int,
) -> None:
    stream_a = torch.cuda.Stream(priority=0)
    stream_b = torch.cuda.Stream(priority=0)
    transaction = router.begin_transaction()
    with torch.cuda.stream(stream_a):
        y = transaction.input_all_to_all(x, head_dim=2, slot=A2ASlot.PACKED_QKV)
    with torch.cuda.stream(stream_b):
        z = transaction.output_all_to_all(y, head_dim=2, slot=A2ASlot.OUT)
        done = torch.cuda.Event()
        done.record(stream_b)
    transaction.close()

    torch.cuda.current_stream().wait_event(done)
    assert_equal(
        z,
        expected_output(y, rank=rank, world_size=world_size),
        "cross-stream-output",
    )
    assert_equal(z, x, "cross-stream-roundtrip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-style", choices=["legacy", "tk"], required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    config = UlyssesA2AConfig(
        backend="sgl_p2p",
        p2p_tk_style=args.kernel_style == "tk",
    )
    router = UlyssesA2ARouter(
        dist.group.WORLD, torch.device("cuda", local_rank), config
    )

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for row_width in (7, 384):
            x = make_tensor((1, 3, world_size * 2, row_width), dtype, rank)
            run_roundtrip(
                router,
                x,
                rank=rank,
                world_size=world_size,
                label=f"{args.kernel_style}-{dtype}-d{row_width}",
            )

    # Force a capacity grow after prior kernels have completed.
    grown = make_tensor((1, 19, world_size * 2, 384), torch.bfloat16, rank, offset=17)
    run_roundtrip(
        router,
        grown,
        rank=rank,
        world_size=world_size,
        label=f"{args.kernel_style}-grow",
    )

    cross_stream = make_tensor(
        (1, 5, world_size * 2, 128), torch.bfloat16, rank, offset=31
    )
    run_cross_stream(
        router,
        cross_stream,
        rank=rank,
        world_size=world_size,
    )

    assert router.sgl_p2p is not None
    assert router.sgl_p2p.max_inflight == 1
    assert router.stats.sgl_p2p_calls > 0
    assert router.stats.nccl_calls == 0
    calls = router.stats.sgl_p2p_calls
    router.close()
    assert router.closed

    dist.barrier()
    if rank == 0:
        print(
            "DISTRIBUTED_P2P_CORRECTNESS_OK "
            f"world_size={world_size} kernel={args.kernel_style} calls={calls}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
