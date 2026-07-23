#!/usr/bin/env python3
"""Validate eager/capture routing without selector work inside capture."""

from __future__ import annotations

import gc
import os

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a import (
    A2ASlot,
    UlyssesA2AConfig,
    UlyssesA2ARouter,
    UlyssesA2AUnsupportedError,
)


def make_tensor(shape: tuple[int, ...], rank: int, generation: int) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    values = (
        torch.arange(count, dtype=torch.int64, device="cuda")
        + rank * 31
        + generation * 7
    ) % 251
    return values.reshape(shape).to(torch.bfloat16)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise ValueError("CUDA Graph test requires world_size=2")

    device = torch.device("cuda", local_rank)
    shape = (1, 8, world_size * 2, 64)
    auto = UlyssesA2ARouter(
        dist.group.WORLD,
        device,
        UlyssesA2AConfig(backend="auto"),
    )
    reference = UlyssesA2ARouter(
        dist.group.WORLD,
        device,
        UlyssesA2AConfig(backend="nccl"),
    )
    forced = UlyssesA2ARouter(
        dist.group.WORLD,
        device,
        UlyssesA2AConfig(backend="sgl_p2p"),
    )

    eager_x = make_tensor(shape, rank, 0)
    eager_expected = reference.input_all_to_all(
        eager_x,
        head_dim=2,
        slot=A2ASlot.PACKED_QKV,
    )
    eager_actual = auto.input_all_to_all(
        eager_x,
        head_dim=2,
        slot=A2ASlot.PACKED_QKV,
    )
    torch.cuda.synchronize()
    if not torch.equal(eager_actual, eager_expected):
        raise AssertionError("pre-capture eager P2P result differs from NCCL")
    if auto.stats.sgl_p2p_calls != 1:
        raise AssertionError("pre-capture eager call did not select P2P")

    # Warm NCCL and allocator state before capture.
    reference.input_all_to_all(
        eager_x,
        head_dim=2,
        slot=A2ASlot.PACKED_QKV,
    )
    torch.cuda.synchronize()
    dist.barrier()

    static_x = eager_x.clone()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    consensus_before_capture = auto.stats.control_consensus_calls
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_out = auto.input_all_to_all(
            static_x,
            head_dim=2,
            slot=A2ASlot.PACKED_QKV,
        )

    if auto.stats.control_consensus_calls != consensus_before_capture:
        raise AssertionError("selector issued a control collective inside capture")
    if auto.stats.nccl_calls != 1:
        raise AssertionError("captured auto call did not select NCCL exactly once")

    for generation in range(1, 21):
        replay_x = make_tensor(shape, rank, generation)
        expected = reference.input_all_to_all(
            replay_x,
            head_dim=2,
            slot=A2ASlot.PACKED_QKV,
        )
        static_x.copy_(replay_x)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(graph_out, expected):
            raise AssertionError(f"CUDA Graph replay {generation} differs from NCCL")

    forced_error = ""
    forced_graph = torch.cuda.CUDAGraph()
    forced_stream = torch.cuda.Stream()
    with torch.cuda.graph(forced_graph, stream=forced_stream):
        try:
            forced.input_all_to_all(
                static_x,
                head_dim=2,
                slot=A2ASlot.PACKED_QKV,
            )
        except UlyssesA2AUnsupportedError as exc:
            forced_error = str(exc)
    if "cuda_graph_capture" not in forced_error:
        raise AssertionError(
            f"forced P2P capture did not fail closed: {forced_error!r}"
        )
    if forced.stats.control_consensus_calls != 0:
        raise AssertionError("forced capture attempted a control consensus")
    if forced.stats.sgl_p2p_calls != 0:
        raise AssertionError("forced capture entered the P2P data path")

    post_x = make_tensor(shape, rank, 22)
    post_expected = reference.input_all_to_all(
        post_x,
        head_dim=2,
        slot=A2ASlot.PACKED_QKV,
    )
    post_actual = auto.input_all_to_all(
        post_x,
        head_dim=2,
        slot=A2ASlot.PACKED_QKV,
    )
    torch.cuda.synchronize()
    if not torch.equal(post_actual, post_expected):
        raise AssertionError("post-capture eager P2P result differs from NCCL")
    if auto.stats.sgl_p2p_calls != 2:
        raise AssertionError("post-capture eager call did not return to P2P")

    # A captured NCCL graph must release its executable before the process
    # group. Otherwise ProcessGroupNCCL teardown can wait indefinitely on
    # graph-owned communicator state.
    torch.cuda.synchronize()
    dist.barrier()
    graph.reset()
    forced_graph.reset()
    del graph_out
    del graph
    del forced_graph
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier()

    forced.close()
    auto.close()
    reference.close()
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(
            "DISTRIBUTED_CUDA_GRAPH_OK "
            "eager=p2p capture=nccl replay=20 forced=fail_fast post=p2p "
            f"control_consensus={consensus_before_capture}"
        )


if __name__ == "__main__":
    main()
