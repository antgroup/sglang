#!/usr/bin/env python3
"""Exercise SP subgroup/backend destroy and reinitialization."""

from __future__ import annotations

import argparse
import os

import torch

from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a import (
    A2ASlot,
)
from sglang.multimodal_gen.runtime.distributed.parallel_groups import PROCESS_GROUP
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)


def make_tensor(shape: tuple[int, ...], rank: int, generation: int) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    values = (
        torch.arange(count, dtype=torch.int64, device="cuda") + rank + generation * 13
    ) % 101
    return values.reshape(shape).to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["nccl", "sgl_p2p"], required=True)
    parser.add_argument("--topology", choices=["tp2_sp2", "ring2"], required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise ValueError("lifecycle test requires world_size=4")
    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
        device_id=torch.device("cuda", local_rank),
        timeout=120,
    )

    if args.topology == "tp2_sp2":
        tp_degree, sp_degree, ulysses_degree, ring_degree = 2, 2, 2, 1
    else:
        tp_degree, sp_degree, ulysses_degree, ring_degree = 1, 4, 2, 2

    router_ids: list[int] = []
    for generation in range(3):
        initialize_model_parallel(
            data_parallel_size=1,
            classifier_free_guidance_degree=1,
            sequence_parallel_degree=sp_degree,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            tensor_parallel_degree=tp_degree,
            pipeline_parallel_degree=1,
            ulysses_a2a_backend=args.backend,
            ulysses_a2a_p2p_tk_style=True,
        )
        sp_group = get_sp_group()
        router = sp_group.get_ulysses_a2a_router(torch.device("cuda", local_rank))
        router_ids.append(id(router))
        x = make_tensor(
            (1, 3 + generation, ulysses_degree * 2, 128),
            rank,
            generation,
        )
        with router.begin_transaction() as transaction:
            y = transaction.input_all_to_all(x, head_dim=2, slot=A2ASlot.PACKED_QKV)
            z = transaction.output_all_to_all(y, head_dim=2, slot=A2ASlot.OUT)
        if not torch.equal(z, x):
            raise AssertionError(
                f"generation {generation} roundtrip mismatch for {args.backend}"
            )
        destroy_model_parallel()
        if PROCESS_GROUP.ULYSSES_PG is not None:
            raise AssertionError("Ulysses singleton was not reset")
        if PROCESS_GROUP.RING_PG is not None:
            raise AssertionError("Ring singleton was not reset")
        if PROCESS_GROUP.OWNED_PGS:
            raise AssertionError("Owned subgroup registry was not reset")

    if len(set(router_ids)) != len(router_ids):
        raise AssertionError(f"stale router was reused: {router_ids}")

    destroy_distributed_environment()
    if rank == 0:
        print(
            "DISTRIBUTED_LIFECYCLE_REINIT_OK "
            f"backend={args.backend} topology={args.topology} generations=3"
        )


if __name__ == "__main__":
    main()
