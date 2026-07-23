#!/usr/bin/env python3
"""Exercise two live overlapping P2P communicators in one process."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a import (
    A2ASlot,
    UlyssesA2AConfig,
    UlyssesA2ARouter,
)


def make_tensor(
    shape: tuple[int, ...],
    global_rank: int,
    generation: int,
) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    values = (
        torch.arange(count, dtype=torch.int64, device="cuda")
        + global_rank * 19
        + generation * 23
    ) % 509
    return values.reshape(shape).to(torch.float16)


def roundtrip(
    router: UlyssesA2ARouter,
    global_rank: int,
    generation: int,
) -> None:
    x = make_tensor((1, 5, 4, 96), global_rank, generation)
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
            f"overlapping subgroup roundtrip mismatch at generation {generation}"
        )


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    if dist.get_world_size() != 4:
        raise ValueError("overlapping subgroup test requires world_size=4")

    group_a_ranks = (0, 1)
    group_b_ranks = (0, 2)
    group_a = dist.new_group(ranks=list(group_a_ranks), backend="nccl")
    group_b = dist.new_group(ranks=list(group_b_ranks), backend="nccl")
    device = torch.device("cuda", local_rank)
    config = UlyssesA2AConfig(backend="sgl_p2p", p2p_tk_style=True)

    router_a = None
    router_b = None
    if rank in group_a_ranks:
        router_a = UlyssesA2ARouter(group_a, device, config)
    dist.barrier()
    if rank in group_b_ranks:
        router_b = UlyssesA2ARouter(group_b, device, config)
    dist.barrier()

    for generation in range(4):
        if rank in group_a_ranks:
            assert router_a is not None
            roundtrip(router_a, rank, generation)
        dist.barrier()
        if rank in group_b_ranks:
            assert router_b is not None
            roundtrip(router_b, rank, generation + 100)
        dist.barrier()

    if rank == 0:
        assert router_a is not None and router_b is not None
        if router_a.sgl_p2p is router_b.sgl_p2p:
            raise AssertionError("overlapping subgroups reused one P2P manager")
        if router_a.sgl_p2p.fa == router_b.sgl_p2p.fa:
            raise AssertionError("overlapping subgroups reused one kernel handle")

    if rank in group_a_ranks:
        assert router_a is not None
        router_a.close()
    dist.barrier()

    # Group B must remain usable on rank 0 after group A has been closed.
    if rank in group_b_ranks:
        assert router_b is not None
        roundtrip(router_b, rank, 999)
    dist.barrier()

    if rank in group_b_ranks:
        assert router_b is not None
        router_b.close()
    dist.barrier()

    if rank in group_a_ranks:
        dist.destroy_process_group(group_a)
    if rank in group_b_ranks:
        dist.destroy_process_group(group_b)
    dist.barrier()
    if rank == 0:
        print(
            "DISTRIBUTED_OVERLAPPING_SUBGROUPS_OK "
            "groups=[0,1]+[0,2] rank0_instances=2 "
            "roundtrips=4+post_close"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
