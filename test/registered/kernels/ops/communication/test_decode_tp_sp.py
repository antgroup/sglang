"""TP-SP mHC rows (deepseek_v4_tp_sp, mix_decode): DeepSeek-V4.1 decode layers run
on TP row slices must give every row bitwise what the replicated layers give it.

The layers are the model's own SM90 mHC reads and posts (``forward_layers``: mix
stats on the side stream, collapse, RMSNorm, post + next collapse, the DSpark
capture) around rank-specific GEMM stand-ins for attention and the MoE, whose TP
partial sums the replicated rows all-reduce and the plan reduce-scatters. The
row counts span the SM90 kernel thresholds (the replicated batch takes other
mix-stats kernels than its slices). Batches the TP size does not divide are
padded, eagerly and on CUDA-graph replay, with one rank lagging into every
collective.

Usage::

    python test/registered/kernels/ops/communication/test_decode_tp_sp.py --num-gpu 4
"""

from __future__ import annotations

import atexit
import os
from contextlib import contextmanager
from types import SimpleNamespace
from typing import List, Tuple

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.aux_hidden_states import AuxHiddenStatePacker
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models import deepseek_v4_tp_sp as tp_sp
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=120, stage="extra-b", runner_config="8-gpu-h200")

HC, HIDDEN = 4, 5120  # DeepSeek-V4.1 mHC: 4 streams of 5120
MIX = (2 + HC) * HC
NUM_LAYERS = 3
CAPTURED = (1,)
# DSpark verify rows (6 per request): bs 3, 31, 32, 64. 18 and 186 leave padding
# rows on TP4 and TP8; 384 replicated rows take the wgmma mix stats, its slices
# the v2 partials.
ROWS = [18, 186, 192, 384]


def _device() -> torch.device:
    return torch.device("cuda", int(os.environ["LOCAL_RANK"]))


@cache_once
def _tp_group() -> GroupCoordinator:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = world = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=world)
    atexit.register(dist.destroy_process_group)
    tp = GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=local_rank,
        torch_distributed_backend="nccl",
        use_pynccl=False,
        use_mscclpp=False,
        use_custom_allreduce=True,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        group_name="decode_tp_sp_test",
    )
    assert isinstance(tp.ca_comm, CustomAllReduceV2) and not tp.ca_comm.disabled
    # The plan's all-reduce fallback runs on the TP group; no prefill CP here.
    get_parallel().override_permanently(tp_group=tp, attn_cp_size=1)
    return tp


def _randn(*shape: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    """The same values on every rank."""
    gen = torch.Generator(device=_device()).manual_seed(seed)
    return torch.randn(*shape, device=_device(), generator=gen) * scale


def _partial(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # A rank lagging into each reduce: no peer may load its input early.
    if dist.get_rank() == 0:
        torch.cuda._sleep(1_000_000)
    return x @ w


class _Attention:
    """A row-parallel attention output: this rank's partial sum of every row."""

    def __init__(self, w: torch.Tensor) -> None:
        self.w = w
        self.attn_tp_size = dist.get_world_size()

    @staticmethod
    def accepts_mxfp8_swizzled_input() -> bool:
        return False

    @staticmethod
    def accepts_deepgemm_mxfp8_input() -> bool:
        return False

    @contextmanager
    def maybe_use_decode_attn_tp(self, forward_batch):
        yield

    def __call__(self, *, x, positions, forward_batch, x_quant, defer_all_reduce):
        assert x_quant is None and defer_all_reduce
        return mhc.AttnOutput(_partial(x, self.w))


class _Moe:
    def __init__(self, w: torch.Tensor) -> None:
        self.w = w
        self.tp_size = dist.get_world_size()

    def __call__(self, x, forward_batch, *, input_ids, input_ids_global, **_):
        return _partial(x, self.w)


def _norm(seed: int) -> RMSNorm:
    norm = RMSNorm(HIDDEN, eps=1e-6).to(device=_device(), dtype=torch.bfloat16)
    norm.weight.data.copy_(_randn(HIDDEN, seed=seed, scale=0.1) + 1)
    return norm


def _layer(seed: int, stats_stream: torch.cuda.Stream) -> SimpleNamespace:
    """One decode layer's mHC parameters, as the model binds them, and stand-ins."""
    cfg = mhc.HcConfig(
        mult=HC,
        sinkhorn_iters=20,
        eps=1e-6,
        rms_eps=1e-6,
        hidden=HIDDEN,
        pre_from_prev=True,
        cp_prefill=False,
        sm90=mhc.sm90_hc_rows(torch.cuda.get_device_name()),
    )
    rank = dist.get_rank()
    in_norm, post_norm = _norm(seed + 6), _norm(seed + 7)

    def hc(offset):
        return mhc.HcSubLayer(
            cfg,
            _randn(MIX, HC * HIDDEN, seed=seed + offset, scale=0.02),
            _randn(3, seed=seed + offset + 2, scale=0.1) + 1,
            _randn(MIX, seed=seed + offset + 4, scale=0.1),
            in_norm if offset == 0 else post_norm,
        )

    return SimpleNamespace(
        boundary_pre_from_prev=True,
        boundary_mhc_fusions=True,
        use_fused_mhc_post_pre=False,
        hc_stats_stream=stats_stream,
        hc_cfg=cfg,
        attn_hc=hc(0),
        ffn_hc=hc(1),
        input_layernorm=in_norm,
        post_attention_layernorm=post_norm,
        engram=None,
        self_attn=_Attention(
            _randn(HIDDEN, HIDDEN, seed=seed * 100 + rank, scale=0.01).bfloat16()
        ),
        mlp=_Moe(
            _randn(HIDDEN, HIDDEN, seed=seed * 100 + 50 + rank, scale=0.01).bfloat16()
        ),
    )


class _Replicated:
    """Every row on every rank, the TP sums all-reduced: the replicated layers."""

    def __init__(self, rows: int) -> None:
        self.rows, self.chunk, self.start, self.world = rows, rows, 0, 1

    @staticmethod
    def scatter_rows(full):
        return full

    @staticmethod
    def all_gather(local):
        return local

    @staticmethod
    def reduce_scatter(partial):
        return _tp_group().all_reduce(partial)


def _model(layers: List[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        engram_hasher=None,
        layers=layers,
        start_layer=0,
        end_layer=len(layers),
        dspark_layers_to_capture=list(CAPTURED),
    )


def _forward(plan, streams: torch.Tensor, model) -> torch.Tensor:
    """The final streams, pre-mix and captures of every row, packed."""
    rows = streams.shape[0]
    aux = AuxHiddenStatePacker(len(CAPTURED))
    ids = torch.zeros(rows, dtype=torch.int64, device=streams.device)
    residual, pre = tp_sp.forward_layers(
        model,
        plan,
        positions=ids,
        hidden_states=streams,
        forward_batch=SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY),
        input_ids=ids,
        input_ids_global=ids,
        dspark_aux_hidden_states=aux,
    )
    # The fp32 pre-mix rides as its bf16-sized halves, bit for bit.
    pre = pre.float().contiguous().view(torch.bfloat16)
    packed = torch.cat([residual.flatten(1), pre, aux.finalize()], dim=1)
    return plan.all_gather(packed)


def _inputs(rows: int, seed: int) -> Tuple[torch.Tensor, SimpleNamespace]:
    _tp_group()
    stream = torch.cuda.Stream()
    streams = _randn(rows, HC, HIDDEN, seed=seed).bfloat16()
    layers = [_layer(seed + 10 * i, stream) for i in range(NUM_LAYERS)]
    return streams, _model(layers)


def _assert_bitwise(got: torch.Tensor, ref: torch.Tensor) -> None:
    torch.cuda.synchronize()
    dist.barrier()
    assert got.shape == ref.shape
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def _skip_unless_sm90() -> None:
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("the SM90 mHC decode kernels are SM90-only")


def _sliced(rows: int) -> tp_sp.DecodeTpSp:
    if rows <= dist.get_world_size():
        pytest.skip("the plan keeps at least two rows per rank")
    return tp_sp.DecodeTpSp(rows=rows, group=_tp_group())


@pytest.mark.parametrize("rows", ROWS)
@torch.inference_mode()
def test_eager_rows_match_replicated(rows: int) -> None:
    _skip_unless_sm90()
    streams, model = _inputs(rows, seed=rows)
    plan = _sliced(rows)
    ref = _forward(_Replicated(rows), streams, model)
    _assert_bitwise(_forward(plan, streams, model), ref)


@pytest.mark.parametrize("rows", [18, 384])
@torch.inference_mode()
def test_cuda_graph_rows_match_replicated(rows: int) -> None:
    """The plan's all-gathers and reduce-scatters read registered peer inputs on
    replay; every replay must match the eager replicated layers."""
    _skip_unless_sm90()
    tp = _tp_group()
    streams, model = _inputs(rows, seed=rows + 1)
    plan = _sliced(rows)
    graph_in = streams.clone()
    _forward(plan, graph_in, model)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with tp.ca_comm.capture():
        with torch.cuda.graph(graph):
            graph_out = _forward(plan, graph_in, model)
    for it in range(2):
        fresh = _randn(rows, HC, HIDDEN, seed=500 + it).bfloat16()
        ref = _forward(_Replicated(rows), fresh, model)
        graph_in.copy_(fresh)
        graph.replay()
        _assert_bitwise(graph_out, ref)


@torch.inference_mode()
def test_unfit_reduce_falls_back_to_all_reduce_rows() -> None:
    """A partial the pull plane cannot stage (eager, beyond its workspace) is
    all-reduced like the replicated path and sliced, padding rows zeroed."""
    tp = _tp_group()
    world = tp.world_size
    hidden = tp.ca_comm.max_pull_size // (2 * world * 8) + 8
    rows = 8 * world - 3
    plan = tp_sp.DecodeTpSp(rows=rows, group=tp)
    assert not tp.ca_comm.can_reduce_scatter_input(
        numel=plan.padded_rows * hidden, dtype=torch.bfloat16
    )
    partial = _randn(rows, hidden, seed=7 + dist.get_rank()).bfloat16()
    ref = tp.all_reduce(partial.clone())
    got = plan.reduce_scatter(partial)
    want = torch.zeros_like(got)
    want[: plan.local_rows] = ref[plan.start : plan.start + plan.local_rows]
    _assert_bitwise(got, want)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4, 8))
