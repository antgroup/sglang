#!/usr/bin/env python3
"""Raw-bit Ulysses NCCL semantic checks on a torchrun process group."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a import (
    A2ASlot,
    UlyssesA2AConfig,
    UlyssesA2ARouter,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a.nccl import (
    NCCLUlyssesA2ABackend,
)


def make_tensor(shape: tuple[int, ...], dtype: torch.dtype, rank: int) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    values = (torch.arange(count, device="cuda", dtype=torch.int64) % 97) + rank
    return values.reshape(shape).to(dtype=dtype)


def all_gather_uniform(x: torch.Tensor) -> list[torch.Tensor]:
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, x)
    return gathered


def expected_uniform_input(
    x: torch.Tensor, *, head_dim: int, rank: int, world_size: int
) -> torch.Tensor:
    gathered = all_gather_uniform(x)
    if head_dim == 2:
        h_local = x.shape[2] // world_size
        return torch.cat(
            [item[:, :, rank * h_local : (rank + 1) * h_local, :] for item in gathered],
            dim=1,
        )
    h_local = x.shape[1] // world_size
    return torch.cat(
        [item[:, rank * h_local : (rank + 1) * h_local, :, :] for item in gathered],
        dim=2,
    )


def expected_uniform_output(
    x: torch.Tensor, *, head_dim: int, rank: int, world_size: int
) -> torch.Tensor:
    gathered = all_gather_uniform(x)
    if head_dim == 2:
        s_local = x.shape[1] // world_size
        return torch.cat(
            [item[:, rank * s_local : (rank + 1) * s_local, :, :] for item in gathered],
            dim=2,
        )
    s_local = x.shape[2] // world_size
    return torch.cat(
        [item[:, :, rank * s_local : (rank + 1) * s_local, :] for item in gathered],
        dim=1,
    )


def all_gather_variable(
    x: torch.Tensor, seq_lens: list[int], *, head_dim: int
) -> list[torch.Tensor]:
    seq_dim = 1 if head_dim == 2 else 2
    max_seq = max(seq_lens)
    padded_shape = list(x.shape)
    padded_shape[seq_dim] = max_seq
    padded = torch.zeros(padded_shape, dtype=x.dtype, device=x.device)
    if head_dim == 2:
        padded[:, : x.shape[1]] = x
    else:
        padded[:, :, : x.shape[2]] = x
    gathered = [torch.empty_like(padded) for _ in seq_lens]
    dist.all_gather(gathered, padded)
    if head_dim == 2:
        return [item[:, :seq_len] for item, seq_len in zip(gathered, seq_lens)]
    return [item[:, :, :seq_len] for item, seq_len in zip(gathered, seq_lens)]


def expected_variable_input(
    x: torch.Tensor,
    seq_lens: list[int],
    *,
    head_dim: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    gathered = all_gather_variable(x, seq_lens, head_dim=head_dim)
    if head_dim == 2:
        h_local = x.shape[2] // world_size
        return torch.cat(
            [item[:, :, rank * h_local : (rank + 1) * h_local, :] for item in gathered],
            dim=1,
        )
    h_local = x.shape[1] // world_size
    return torch.cat(
        [item[:, rank * h_local : (rank + 1) * h_local, :, :] for item in gathered],
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


def run_uniform(
    backend: NCCLUlyssesA2ABackend,
    *,
    dtype: torch.dtype,
    head_dim: int,
    rank: int,
    world_size: int,
) -> None:
    b, s_local, h, d = 2, 5, world_size * 2, 7
    shape = (b, s_local, h, d) if head_dim == 2 else (b, h, s_local, d)
    x = make_tensor(shape, dtype, rank)
    actual_input = backend.input_all_to_all(x, head_dim=head_dim, seq_lens=None)
    expected_input = expected_uniform_input(
        x, head_dim=head_dim, rank=rank, world_size=world_size
    )
    assert_equal(actual_input, expected_input, f"uniform-input-{dtype}-hd{head_dim}")

    actual_output = backend.output_all_to_all(
        actual_input, head_dim=head_dim, seq_lens=None
    )
    expected_output = expected_uniform_output(
        actual_input, head_dim=head_dim, rank=rank, world_size=world_size
    )
    assert_equal(actual_output, expected_output, f"uniform-output-{dtype}-hd{head_dim}")
    assert_equal(actual_output, x, f"uniform-roundtrip-{dtype}-hd{head_dim}")


def run_variable(
    backend: NCCLUlyssesA2ABackend,
    *,
    dtype: torch.dtype,
    head_dim: int,
    rank: int,
    world_size: int,
) -> None:
    seq_lens = [rank_index + 2 for rank_index in range(world_size)]
    b, h, d = 1, world_size * 2, 5
    shape = (b, seq_lens[rank], h, d) if head_dim == 2 else (b, h, seq_lens[rank], d)
    x = make_tensor(shape, dtype, rank)
    actual_input = backend.input_all_to_all(x, head_dim=head_dim, seq_lens=seq_lens)
    expected_input = expected_variable_input(
        x,
        seq_lens,
        head_dim=head_dim,
        rank=rank,
        world_size=world_size,
    )
    assert_equal(actual_input, expected_input, f"variable-input-{dtype}-hd{head_dim}")
    actual_output = backend.output_all_to_all(
        actual_input, head_dim=head_dim, seq_lens=seq_lens
    )
    assert_equal(actual_output, x, f"variable-roundtrip-{dtype}-hd{head_dim}")


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    backend = NCCLUlyssesA2ABackend(dist.group.WORLD)
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for head_dim in (1, 2):
            run_uniform(
                backend,
                dtype=dtype,
                head_dim=head_dim,
                rank=rank,
                world_size=world_size,
            )
            run_variable(
                backend,
                dtype=dtype,
                head_dim=head_dim,
                rank=rank,
                world_size=world_size,
            )

    router = UlyssesA2ARouter(
        dist.group.WORLD,
        torch.device("cuda", local_rank),
        UlyssesA2AConfig(backend="nccl"),
    )
    x = make_tensor((1, 4, world_size * 2, 9), torch.bfloat16, rank)
    with router.begin_transaction() as transaction:
        y = transaction.input_all_to_all(x, head_dim=2, slot=A2ASlot.PACKED_QKV)
        z = transaction.output_all_to_all(y, head_dim=2, slot=A2ASlot.OUT)
    assert_equal(z, x, "router-transaction-roundtrip")
    assert router.stats.nccl_calls == 2
    assert router.stats.sgl_p2p_calls == 0
    router.close()

    dist.barrier()
    if rank == 0:
        print(
            "DISTRIBUTED_NCCL_CORRECTNESS_OK "
            f"world_size={world_size} dtypes=3 head_dims=2 variable=1"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
