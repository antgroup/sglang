"""``SymmHeadA2A``: the NVLink head <-> row exchange of DSV4.1 decode attention.

Both directions must be bitwise the NCCL all-to-all plus permute they replace
(zero pad rows included), eagerly and on CUDA-graph replay, and a result must
stay intact until the next exchange even when a peer races ahead into it.

Usage::

    python test/registered/kernels/ops/communication/test_head_a2a.py --num-gpu 2,4,8
"""

from __future__ import annotations

import atexit
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.srt.distributed.device_communicators.head_a2a import (
    SymmHeadA2A,
    get_symm_head_a2a,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# DeepSeek-V4.1 attention over TP8: 8 of the 64 heads per rank, 512 wide (q and o).
LOCAL_HEADS, HEAD_DIM = 8, 512


def _device() -> torch.device:
    return torch.device("cuda", int(os.environ["LOCAL_RANK"]))


@cache_once
def _tp_group() -> GroupCoordinator:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=ps._WORLD)
    atexit.register(dist.destroy_process_group)
    return GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=local_rank,
        torch_distributed_backend="nccl",
        use_pynccl=False,
        use_mscclpp=False,
        use_custom_allreduce=True,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        group_name="head_a2a_test",
    )


@cache_once
def _a2a() -> SymmHeadA2A:
    tp = _tp_group()
    a2a = get_symm_head_a2a(tp, local_heads=LOCAL_HEADS, head_bytes=HEAD_DIM * 2)
    if a2a is None:
        pytest.skip("this TP group has no custom all-reduce v2 (no NVLink clique)")
    return a2a


def _randn(*shape: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=_device()).manual_seed(seed * 131 + dist.get_rank())
    x = torch.randn(*shape, device=_device(), generator=gen).bfloat16()
    x.view(-1)[::7] = -0.0  # signed zeros and specials must cross unchanged
    x.view(-1)[1::97] = float("nan")
    return x


def _ref_rows(q_local: torch.Tensor) -> torch.Tensor:
    """[rows, H, D] -> [chunk, W*H, D] via the NCCL all-to-all."""
    world = dist.get_world_size()
    rows, heads, dim = q_local.shape
    padded = -(-rows // world) * world
    q_local = torch.cat([q_local, q_local.new_zeros(padded - rows, heads, dim)])
    recv = torch.empty_like(q_local)
    dist.all_to_all_single(recv, q_local, group=_tp_group().device_group)
    chunk = padded // world
    return recv.view(world, chunk, heads, dim).transpose(0, 1).reshape(chunk, -1, dim)


def _ref_heads(o_rows: torch.Tensor) -> torch.Tensor:
    """[chunk, W*H, D] -> [W*chunk, H, D] via the NCCL all-to-all."""
    world = dist.get_world_size()
    chunk, all_heads, dim = o_rows.shape
    send = o_rows.view(chunk, world, all_heads // world, dim).transpose(0, 1)
    out = torch.empty_like(send.contiguous())
    dist.all_to_all_single(out, send.contiguous(), group=_tp_group().device_group)
    return out.view(world * chunk, all_heads // world, dim)


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16)


def _sync() -> None:
    torch.cuda.synchronize()
    dist.barrier()


def _assert_bitwise_on_all_ranks(got: torch.Tensor, ref: torch.Tensor) -> None:
    # A rank failing alone would leave its peers blocked in the next collective.
    ok = torch.tensor([int(torch.equal(_bits(got), _bits(ref)))])
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    assert ok.item() == 1, "a rank differs from the NCCL reference bitwise"


def _fake_attention(q_rows: torch.Tensor) -> torch.Tensor:
    # Reads all of q_rows (which lives in the shared receive buffer); rank 0
    # first sleeps so its peers reach the next exchange while it still reads.
    if dist.get_rank() == 0:
        torch.cuda._sleep(2_000_000)
    return q_rows * 2 + dist.get_rank()


def test_not_built_under_graph_capture() -> None:
    """Building allocates and rendezvouses, so a first call inside a capture
    must defer (None, NCCL path) and a later eager call builds it."""
    tp = _tp_group()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = get_symm_head_a2a(
            tp, local_heads=LOCAL_HEADS, head_bytes=HEAD_DIM * 2
        )
    assert captured is None
    assert _a2a() is not None


@pytest.mark.parametrize("rows", [1, 95, 96, 97, 192, 383, 384])
def test_matches_nccl(rows: int) -> None:
    """Rows not divisible by the world are zero-padded, even over a receive
    buffer that a larger exchange has just filled."""
    a2a = _a2a()
    world = dist.get_world_size()
    chunk = -(-rows // world)
    a2a.to_row_layout(_randn(chunk * world, LOCAL_HEADS, HEAD_DIM, seed=3))
    q_local = _randn(rows, LOCAL_HEADS, HEAD_DIM, seed=rows)
    q_got = a2a.to_row_layout(q_local).clone()
    _assert_bitwise_on_all_ranks(q_got, _ref_rows(q_local))
    o_rows = _randn(chunk, world * LOCAL_HEADS, HEAD_DIM, seed=rows + 1000)
    o_got = a2a.to_head_layout(o_rows).clone()
    _assert_bitwise_on_all_ranks(o_got, _ref_heads(o_rows))


@pytest.mark.parametrize("rows", [97, 384])
def test_cuda_graph_round_trip(rows: int) -> None:
    """q exchange, an attention stand-in reading it, then the o exchange into
    the same buffer: the result must match eager NCCL on every replay."""
    a2a = _a2a()
    q_local = _randn(rows, LOCAL_HEADS, HEAD_DIM, seed=rows)

    def run() -> torch.Tensor:
        o_rows = _fake_attention(a2a.to_row_layout(q_local))
        return a2a.to_head_layout(o_rows)[:rows]

    run()
    _sync()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for it in range(3):
        q_local.copy_(_randn(rows, LOCAL_HEADS, HEAD_DIM, seed=50 + it))
        ref = _ref_heads(_fake_attention(_ref_rows(q_local)))[:rows]
        _sync()
        graph.replay()
        _sync()
        _assert_bitwise_on_all_ranks(out, ref)


def test_eager_round_trip_with_lagging_reader() -> None:
    a2a = _a2a()
    q_local = _randn(384, LOCAL_HEADS, HEAD_DIM, seed=7)
    ref = _ref_heads(_fake_attention(_ref_rows(q_local)))
    got = a2a.to_head_layout(_fake_attention(a2a.to_row_layout(q_local))).clone()
    _assert_bitwise_on_all_ranks(got, ref)


def test_unfit_inputs_take_the_fallback() -> None:
    a2a = _a2a()
    world = dist.get_world_size()
    too_many_rows = a2a.capacity_bytes // (LOCAL_HEADS * HEAD_DIM * 2) + 1
    bf16 = dict(dtype=torch.bfloat16, device=_device())
    big = torch.empty(too_many_rows, LOCAL_HEADS, HEAD_DIM, **bf16)
    strided = torch.empty(8, LOCAL_HEADS, 2 * HEAD_DIM, **bf16)
    unaligned_seg = torch.empty(8, 1, 3, **bf16)
    assert a2a.to_row_layout(big) is None
    assert a2a.to_row_layout(strided[..., :HEAD_DIM]) is None
    assert a2a.to_row_layout(unaligned_seg) is None
    uneven_heads = torch.empty(8, world * LOCAL_HEADS + 1, HEAD_DIM, **bf16)
    assert a2a.to_head_layout(uneven_heads) is None


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4, 8))
