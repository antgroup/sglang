#!/usr/bin/env python3
"""Rank-consensus and strict/auto selector fault checks."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

import sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_p2p_a2a as p2p_module
from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a import (
    A2ASlot,
    UlyssesA2AConfig,
    UlyssesA2ARouter,
    UlyssesA2AUnsupportedError,
)


def make_tensor(shape: tuple[int, ...], rank: int) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    return (
        ((torch.arange(count, device="cuda", dtype=torch.int64) + rank) % 83)
        .reshape(shape)
        .to(torch.bfloat16)
    )


def gather_errors(error: str) -> list[str]:
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    return errors


def expect_forced_error(router: UlyssesA2ARouter, x: torch.Tensor, needle: str) -> None:
    try:
        router.input_all_to_all(x, head_dim=2, slot=A2ASlot.PACKED_QKV)
    except UlyssesA2AUnsupportedError as exc:
        error = str(exc)
    else:
        raise AssertionError("forced backend unexpectedly accepted operation")
    errors = gather_errors(error)
    if len(set(errors)) != 1:
        raise AssertionError(f"forced errors diverged across ranks: {errors}")
    if needle not in errors[0]:
        raise AssertionError(f"expected {needle!r} in {errors[0]!r}")


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise ValueError("selector fault test requires world_size=2")
    device = torch.device("cuda", local_rank)
    x = make_tensor((1, 4, world_size * 2, 16), rank)

    original_ops_available = p2p_module._ops_available
    p2p_module._ops_available = lambda **_kwargs: rank == 0
    try:
        auto = UlyssesA2ARouter(
            dist.group.WORLD, device, UlyssesA2AConfig(backend="auto")
        )
        with auto.begin_transaction() as transaction:
            y = transaction.input_all_to_all(x, head_dim=2, slot=A2ASlot.PACKED_QKV)
            z = transaction.output_all_to_all(y, head_dim=2, slot=A2ASlot.OUT)
        if not torch.equal(z, x):
            raise AssertionError("rank-local capability auto fallback mismatch")
        assert auto.stats.nccl_calls == 2
        assert auto.stats.sgl_p2p_calls == 0
        assert auto.stats.fallback_calls_by_reason == {"rank_capability_miss": 1}
        auto.close()

        forced = UlyssesA2ARouter(
            dist.group.WORLD,
            device,
            UlyssesA2AConfig(backend="sgl_p2p"),
        )
        expect_forced_error(forced, x, "rank_capability_miss")
        forced.close()
    finally:
        p2p_module._ops_available = original_ops_available

    mismatched_x = x if rank == 0 else make_tensor((1, 8, world_size * 2, 8), rank)
    for backend in ("auto", "sgl_p2p"):
        mismatch = UlyssesA2ARouter(
            dist.group.WORLD,
            device,
            UlyssesA2AConfig(backend=backend),
        )
        try:
            mismatch.input_all_to_all(
                mismatched_x,
                head_dim=2,
                slot=A2ASlot.PACKED_QKV,
            )
        except UlyssesA2AUnsupportedError as exc:
            error = str(exc)
        else:
            raise AssertionError(f"{backend} accepted rank-mismatched descriptors")
        errors = gather_errors(error)
        assert len(set(errors)) == 1
        assert "rank_descriptor_mismatch" in errors[0]
        assert mismatch.stats.nccl_calls == 0
        assert mismatch.stats.sgl_p2p_calls == 0
        mismatch.close()

    try:
        UlyssesA2ARouter(
            dist.group.WORLD,
            device,
            UlyssesA2AConfig(backend="nccl" if rank == 0 else "sgl_p2p"),
        )
    except UlyssesA2AUnsupportedError as exc:
        error = str(exc)
    else:
        raise AssertionError("rank-mismatched configuration was accepted")
    errors = gather_errors(error)
    assert len(set(errors)) == 1
    assert "config differs across ranks" in errors[0]

    seq_lens = [3, 5]
    variable_x = make_tensor((1, seq_lens[rank], world_size * 2, 11), rank)
    auto_variable = UlyssesA2ARouter(
        dist.group.WORLD, device, UlyssesA2AConfig(backend="auto")
    )
    with auto_variable.begin_transaction() as transaction:
        y = transaction.input_all_to_all(
            variable_x,
            head_dim=2,
            seq_lens=seq_lens,
            slot=A2ASlot.PACKED_QKV,
        )
        z = transaction.output_all_to_all(
            y,
            head_dim=2,
            seq_lens=seq_lens,
            slot=A2ASlot.OUT,
        )
    if not torch.equal(z, variable_x):
        raise AssertionError("variable auto fallback mismatch")
    assert auto_variable.stats.nccl_calls == 2
    assert auto_variable.stats.fallback_calls_by_reason == {"variable_split": 1}
    auto_variable.close()

    forced_variable = UlyssesA2ARouter(
        dist.group.WORLD,
        device,
        UlyssesA2AConfig(backend="sgl_p2p"),
    )
    try:
        forced_variable.input_all_to_all(
            variable_x,
            head_dim=2,
            seq_lens=seq_lens,
            slot=A2ASlot.PACKED_QKV,
        )
    except UlyssesA2AUnsupportedError as exc:
        error = str(exc)
    else:
        raise AssertionError("forced P2P accepted variable split")
    errors = gather_errors(error)
    assert len(set(errors)) == 1 and "variable_split" in errors[0]
    forced_variable.close()

    skewed_seq_lens = [3, 5] if rank == 0 else [4, 4]
    skewed_variable_x = make_tensor(
        (1, skewed_seq_lens[rank], world_size * 2, 11), rank
    )
    skewed_variable = UlyssesA2ARouter(
        dist.group.WORLD, device, UlyssesA2AConfig(backend="auto")
    )
    try:
        skewed_variable.input_all_to_all(
            skewed_variable_x,
            head_dim=2,
            seq_lens=skewed_seq_lens,
            slot=A2ASlot.PACKED_QKV,
        )
    except UlyssesA2AUnsupportedError as exc:
        error = str(exc)
    else:
        raise AssertionError("rank-skewed variable split reached the NCCL data path")
    errors = gather_errors(error)
    assert len(set(errors)) == 1
    assert "rank_descriptor_mismatch" in errors[0]
    assert skewed_variable.stats.nccl_calls == 0
    assert skewed_variable.stats.sgl_p2p_calls == 0
    skewed_variable.close()

    fast = UlyssesA2ARouter(
        dist.group.WORLD,
        device,
        UlyssesA2AConfig(backend="fast_ulysses"),
    )
    try:
        fast.input_all_to_all(x, head_dim=2, slot=A2ASlot.PACKED_QKV)
    except UlyssesA2AUnsupportedError as exc:
        error = str(exc)
    else:
        raise AssertionError("unsafe fast_ulysses adapter unexpectedly enabled")
    errors = gather_errors(error)
    assert len(set(errors)) == 1
    assert "not installed" in errors[0] or "reclaim" in errors[0]
    fast.close()

    dist.barrier()
    if rank == 0:
        print(
            "DISTRIBUTED_SELECTOR_FAULTS_OK "
            "rank_miss=auto_fallback+forced_error "
            "descriptor=config=fail_fast "
            "variable=auto+forced+rank_skew fast=closed"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
