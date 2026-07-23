#!/usr/bin/env python3
"""Inject one-rank failure after shared allocation has collectively committed."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a import (
    A2ASlot,
    UlyssesA2AConfig,
    UlyssesA2ARouter,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a.base import (
    UlyssesA2ACommitError,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise ValueError("post-commit fault test requires world_size=2")

    original_create = CustomAllreduce.create_shared_buffer
    call_count = 0

    def create_then_fail(*args, **kwargs):
        nonlocal call_count
        pointers = original_create(*args, **kwargs)
        call_count += 1
        if rank == 1 and call_count == 1:
            raise RuntimeError("injected_post_commit_signal_allocation_failure")
        return pointers

    CustomAllreduce.create_shared_buffer = create_then_fail
    device = torch.device("cuda", local_rank)
    router = UlyssesA2ARouter(
        dist.group.WORLD,
        device,
        UlyssesA2AConfig(backend="auto"),
    )
    x = torch.arange(
        1 * 4 * (world_size * 2) * 32,
        dtype=torch.float16,
        device=device,
    ).reshape(1, 4, world_size * 2, 32)
    try:
        router.input_all_to_all(
            x,
            head_dim=2,
            slot=A2ASlot.PACKED_QKV,
        )
    except UlyssesA2ACommitError:
        print(
            "POST_COMMIT_RANK_FATAL "
            f"rank={rank} nccl_calls={router.stats.nccl_calls} "
            f"p2p_calls={router.stats.sgl_p2p_calls}",
            flush=True,
        )
        raise
    raise AssertionError(
        "post-commit injected failure returned or fell back instead of aborting"
    )


if __name__ == "__main__":
    main()
