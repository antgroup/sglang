"""Engram TP split over real NCCL at the DeepSeek-V4.1 shapes: every split mode and
the TP-SP row slice are bitwise the replicated Engram, eager and in CUDA graphs.

wkv is the checkpoint's [32, 32]-block ue8m0 FP8 linear (25600 x 6144) through
Fp8LinearMethod with the SM90 switch on for role mix, so the device's block32
routes apply. Run with ``python test_engram_tp_split_multi_gpu.py --num-gpu 2,4``.
"""

import os
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.distributed import parallel_state as ps
from sglang.srt.environ import envs
from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.dsv41_sm90 import SUBSYSTEMS
from sglang.srt.layers.engram import Engram, EngramLayout, _EngramSplit
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

register_cuda_ci(est_time=240, stage="base-c", runner_config="4-gpu-h100")

# DeepSeek-V4.1-Flash: hc 4 x 5120, 3 n-gram sizes x 8 heads of 256 -> wkv K 6144.
HC, HIDDEN, NGRAM, HEADS, HEAD_DIM = 4, 5120, 4, 8, 256
COLS = (NGRAM - 1) * HEADS
IMAGE_ID = 129264
# Unequal per-column subtables, as the primes make them.
COLUMN_SIZES = tuple(150 + 7 * c for c in range(COLS))
ROWS = (6, 18, 48, 63, 64, 65, 96, 186, 192, 384, 512)
LAYOUTS = ("device", "peer", "shared", "per_rank")


class _LinearOnlyFp8Config(Fp8Config):
    # Fp8Config.get_quant_method imports the MoE stack for its FusedMoE check.
    def get_quant_method(self, layer, prefix):
        return Fp8LinearMethod(self)


@pytest.fixture(scope="module")
def tp():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    ps.init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=int(os.environ["LOCAL_RANK"]),
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=world, world_rank=rank)
    ps.initialize_model_parallel()
    with (
        get_flags().dsv41_sm90.override(subsystems=frozenset(SUBSYSTEMS), role="mix"),
        # The serving configuration: compile the shapes met, not every M.
        patch.object(compile_utils, "_ENABLE_JIT_DEEPGEMM_PRECOMPILE", False),
    ):
        yield ps.get_tp_group()
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def _layout_env(layout: str) -> ExitStack:
    stack = ExitStack()
    stack.enter_context(
        envs.SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS.override(layout == "peer")
    )
    host = layout in ("shared", "per_rank")
    stack.enter_context(envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(host))
    if host:
        stack.enter_context(envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.override(layout))
    return stack


def _build_engram(layout: str) -> Engram:
    """An Engram with checkpoint-like weights, identical on every rank, loaded and
    post-processed like the model's."""
    table_rows = sum(COLUMN_SIZES)
    engram_layout = EngramLayout(
        max_ngram_size=NGRAM,
        layer_ids=(1,),
        num_embeddings=(table_rows,),
        primes=(
            tuple(COLUMN_SIZES[i * HEADS : (i + 1) * HEADS] for i in range(NGRAM - 1)),
        ),
        n_heads=HEADS,
        head_dim=HEAD_DIM,
    )
    config = SimpleNamespace(rms_norm_eps=1e-20, hidden_size=HIDDEN, hc_mult=HC)
    quant = _LinearOnlyFp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[32, 32],
        scale_fmt="ue8m0",
    )
    with _layout_env(layout), torch.device("cuda"):
        engram = Engram(
            config,
            layer_id=1,
            layout=engram_layout,
            quant_config=quant,
            prefix="model.layers.1.engram",
            image_token_id=IMAGE_ID,
        )
    g = torch.Generator(device="cuda").manual_seed(2)
    n, k = HIDDEN * (HC + 1), COLS * HEAD_DIM
    wkv = engram.wkv
    wkv.weight.weight_loader(
        wkv.weight,
        (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn),
    )
    scale = torch.randint(117, 122, (n // 32, k // 32), device="cuda", generator=g)
    wkv.weight_scale_inv.weight_loader(
        wkv.weight_scale_inv, scale.to(torch.uint8).view(torch.float8_e8m0fnu)
    )
    wkv.quant_method.process_weights_after_loading(wkv)
    for w in (engram.q_weight, engram.k_weight):
        w.data.copy_(torch.randn(HC, HIDDEN, device="cuda", generator=g))
    table = torch.randn(table_rows, HEAD_DIM, device="cuda", generator=g) * 3
    table_scale = torch.randint(
        110, 135, (table_rows, HEAD_DIM // 32), device="cuda", generator=g
    )
    embed = engram.embed
    embed.weight.weight_loader(embed.weight, table.to(torch.float8_e4m3fn))
    embed.scale.weight_loader(
        embed.scale, table_scale.to(torch.uint8).view(torch.float8_e8m0fnu)
    )
    embed.finish_load(label=layout)
    return engram


def _fill_inputs(x, ids, input_ids, seed: int) -> None:
    """Fresh rows, the same on every rank; every fifth row is an image token."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x.copy_(torch.randn(x.shape, device="cuda", generator=g) * 3)
    bounds = torch.tensor((0, *COLUMN_SIZES), device="cuda").cumsum(0)
    u = torch.rand(ids.shape, device="cuda", generator=g, dtype=torch.float64)
    span = (bounds[1:] - bounds[:-1]).double()
    ids.copy_(bounds[:-1] + (u * span).long())
    input_ids.copy_(torch.randint(0, 1000, input_ids.shape, generator=g, device="cuda"))
    input_ids[::5] = IMAGE_ID


def _replicated(engram, x, ids, input_ids, forward_batch):
    engram.tp_split = False
    try:
        return engram(x, ids, forward_batch, input_ids=input_ids)
    finally:
        engram.tp_split = True


def _bits(t):
    return t.view(torch.int16)


def _split_outputs(engram, x, x_slice, ids, input_ids, *, start, chunk, world):
    outs = [
        engram._forward_split(x, ids, input_ids=input_ids, mode=mode)
        for mode in _EngramSplit
    ]
    outs.append(
        engram.forward_row_slice(
            x_slice, ids, start=start, chunk=chunk, world=world, input_ids=input_ids
        )
    )
    return outs


def _check(outs, ref, x_slice, *, start, rows, where):
    for mode, out in zip(_EngramSplit, outs[:-1]):
        assert torch.equal(_bits(out), _bits(ref)), f"{where} {mode.name}"
    local = max(0, min(x_slice.shape[0], rows - start))
    row_slice = outs[-1]
    assert torch.equal(_bits(row_slice[:local]), _bits(ref[start : start + local])), (
        f"{where} row slice"
    )
    assert torch.equal(_bits(row_slice[local:]), _bits(x_slice[local:])), (
        f"{where} row slice padding"
    )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_split_matches_replicated_eager_and_graph(tp, layout):
    world, rank = tp.world_size, tp.rank_in_group
    engram = _build_engram(layout)
    assert engram.tp_split and engram._wkv_split_ok(columns=True)
    verify = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY, engram_staged_rows=None
    )
    for rows in ROWS:
        where = f"tp{world} rank {rank} {layout} rows {rows}"
        # The model passes a strided [rows, cols] slice of [rows, n_layers, cols].
        ids = torch.empty(rows, 2, COLS, dtype=torch.int64, device="cuda")[:, 1]
        x = torch.empty(rows, HC, HIDDEN, dtype=torch.bfloat16, device="cuda")
        input_ids = torch.empty(rows, dtype=torch.int64, device="cuda")
        # TP-SP pads the batch to world * chunk rows; padding keeps its rows.
        chunk = -(-rows // world)
        start = rank * chunk
        x_slice = torch.randn(chunk, HC, HIDDEN, device="cuda").bfloat16()
        _fill_inputs(x, ids, input_ids, seed=rows)
        x_slice[: max(0, min(chunk, rows - start))] = x[start : start + chunk]

        ref = _replicated(engram, x, ids, input_ids, verify)
        assert torch.equal(
            _bits(engram(x, ids, verify, input_ids=input_ids)), _bits(ref)
        )
        outs = _split_outputs(
            engram, x, x_slice, ids, input_ids, start=start, chunk=chunk, world=world
        )
        _check(outs, ref, x_slice, start=start, rows=rows, where=f"{where} eager")

        graph = torch.cuda.CUDAGraph()
        with ps.graph_capture() as context:
            for _ in range(2):
                _split_outputs(
                    engram,
                    x,
                    x_slice,
                    ids,
                    input_ids,
                    start=start,
                    chunk=chunk,
                    world=world,
                )
            context.stream.synchronize()
            with torch.cuda.graph(graph, stream=context.stream):
                captured = _split_outputs(
                    engram,
                    x,
                    x_slice,
                    ids,
                    input_ids,
                    start=start,
                    chunk=chunk,
                    world=world,
                )
        _fill_inputs(x, ids, input_ids, seed=rows + 1)
        x_slice[: max(0, min(chunk, rows - start))] = x[start : start + chunk]
        graph.replay()
        torch.cuda.synchronize()
        ref = _replicated(engram, x, ids, input_ids, verify)
        _check(captured, ref, x_slice, start=start, rows=rows, where=f"{where} graph")
        del graph
    torch.distributed.barrier()


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4))
