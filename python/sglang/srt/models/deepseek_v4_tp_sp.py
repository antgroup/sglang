"""Sequence-parallel mHC rows of DeepSeek-V4.1 TP decode / verify (mix decode).

Under TP decode every rank holds every row of the mHC residual streams and
repeats each row's mixing, hc_post, collapse, RMSNorm, Engram gate and DSpark
capture. With ``dsv41_sm90_on("mix_decode")`` (role mix, CP decode attention TP,
no DP attention, no expert all-to-all) a batch of enough rows instead keeps rows
``[r * chunk, (r + 1) * chunk)`` of the streams on TP rank r (the batch
zero-padded to ``world * chunk`` rows). Attention and the MoE still run on every
row through an all-gather of their normalized input, and their TP partial sums
come back by a reduce-scatter in place of the all-reduce (the same bytes on the
wire).

The reduce-scatter is the reduce half of the custom all-reduce v2 pull plane:
the rank-ordered fp32 sum the all-reduce kernels compute for every element, so
each row is bitwise the replicated path wherever the per-row kernels are
batch-invariant (the SM90 mHC, RMSNorm and Engram kernels are). When the
all-reduce would not run such a kernel, the plan all-reduces like the
replicated path and keeps its rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.cp.cp_decode_attn_tp import get_cp_decode_attn_tp_ctx
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.models import deepseek_v4_boundary as hc_boundary
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models.deepseek_common.moe_transport import MoeFinalize
from sglang.srt.models.deepseek_v2 import MoEOutput
from sglang.srt.runtime_context import get_forward, get_parallel

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.layers.aux_hidden_states import AuxHiddenStatePacker
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Measured on 8x H20-3e (TP8 + prefill CP8, DSpark verify): from 64 rows,
# the per-row mHC saved beats the extra all-gathers.
_MIN_ROWS = 64


class DecodeTpSp:
    """Row ownership of one decode / verify forward over the TP group."""

    __slots__ = ("rows", "chunk", "rank", "world", "_group", "_ca_comm")

    def __init__(self, *, rows: int, group: GroupCoordinator) -> None:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        self.rows = rows
        self.world = group.world_size
        self.rank = group.rank_in_group
        self.chunk = -(-rows // self.world)
        self._group = group
        ca_comm = group.ca_comm
        self._ca_comm = (
            ca_comm
            if isinstance(ca_comm, CustomAllReduceV2) and not ca_comm.disabled
            else None
        )

    @property
    def padded_rows(self) -> int:
        return self.chunk * self.world

    @property
    def start(self) -> int:
        return self.rank * self.chunk

    @property
    def local_rows(self) -> int:
        """Real (non-padding) rows of this rank's slice."""
        return max(0, min(self.chunk, self.rows - self.start))

    def scatter_rows(self, full: torch.Tensor) -> torch.Tensor:
        """This rank's rows of a ``[rows, ...]`` tensor every rank holds whole;
        padding rows are zero."""
        assert full.shape[0] == self.rows, (full.shape, self.rows)
        local = full[self.start : self.start + self.local_rows]
        if self.local_rows == self.chunk:
            return local.contiguous()
        out = full.new_zeros(self.chunk, *full.shape[1:])
        out[: self.local_rows].copy_(local)
        return out

    def all_gather(self, local: torch.Tensor) -> torch.Tensor:
        """``[chunk, ...]`` rows of every rank -> the ``[rows, ...]`` batch."""
        assert local.shape[0] == self.chunk, (local.shape, self.chunk)
        local = local.contiguous()
        out = local.new_empty(self.padded_rows, *local.shape[1:])
        ca_comm = self._ca_comm
        if ca_comm is not None and ca_comm.can_all_gather(local, out):
            ca_comm.all_gather(local, out)
        else:
            self._group.all_gather_into_tensor(out, local)
        return out[: self.rows]

    def reduce_scatter(self, partial: torch.Tensor) -> torch.Tensor:
        """This rank's rows of the TP sum of ``partial`` ``[rows, ...]``, bitwise
        those of the all-reduce the replicated path runs on it."""
        assert partial.shape[0] == self.rows, (partial.shape, self.rows)
        partial = partial.contiguous()
        out = partial.new_empty(self.chunk, *partial.shape[1:])
        row_numel = partial[0].numel()
        ca_comm = self._ca_comm
        if (
            ca_comm is not None
            and ca_comm.all_reduce_sums_in_rank_order(
                numel=self.rows * row_numel, dtype=partial.dtype
            )
            and ca_comm.can_reduce_scatter_input(
                numel=self.padded_rows * row_numel, dtype=partial.dtype
            )
        ):
            if self.padded_rows != self.rows:
                padded = partial.new_zeros(self.padded_rows, *partial.shape[1:])
                padded[: self.rows].copy_(partial)
                partial = padded
            ca_comm.reduce_scatter(partial, out)
            return out
        reduced = tensor_model_parallel_all_reduce(partial)
        out.zero_()
        out[: self.local_rows].copy_(reduced[self.start : self.start + self.local_rows])
        return out


def decode_tp_sp_supported(layer) -> bool:
    """Whether a decoder layer runs on row slices: its MoE leaves a plain TP partial
    sum to reduce (no expert all-to-all, no replicated shared expert added after
    the sum), and its Engram, if any, splits its rows over TP."""
    mlp = layer.mlp
    return (
        not mlp._enable_a2a_moe
        and not mlp._shared_expert_tp1
        and (layer.engram is None or layer.engram.tp_split)
    )


def plan_decode_tp_sp(
    *, forward_batch: ForwardBatch, rows: int
) -> Optional[DecodeTpSp]:
    """The row plan of this decode / verify forward, or None for the replicated
    path. Depends only on the config, the forward mode and ``rows``, so every TP
    rank (and every CUDA graph replay of a captured batch) agrees."""
    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_target_verify()) or rows < _MIN_ROWS:
        return None
    if torch.compiler.is_compiling():
        return None
    parallel = get_parallel()
    group = parallel.tp_group
    if (
        group.world_size == 1
        or parallel.attn_dp_size > 1
        or parallel.pp_size > 1
        or not get_moe_a2a_backend().is_none()
        # A pymscclpp all-reduce does not sum like the custom all-reduce kernels.
        or group.pymscclpp_comm is not None
    ):
        return None
    # A one-row slice would take SM90's single-row mix-stats kernel, not the
    # multi-row one the replicated batch runs.
    if rows <= group.world_size:
        return None
    # The attention sums wo_b over the TP group only under CP decode attention TP
    # with the decode TP rank order equal to the TP group's.
    ctx = get_cp_decode_attn_tp_ctx()
    ctx.set_decode_attn_tp(forward_batch)
    if not (
        ctx.use_decode_attn_tp
        and ctx.decode_tp_size == group.world_size
        and ctx.decode_tp_rank == group.rank_in_group
    ):
        return None
    return DecodeTpSp(rows=rows, group=group)


class _FfnPending:
    """The FFN post of a layer, run by the next reader with its own seam."""

    __slots__ = ("layer", "output", "read_state")

    def __init__(self, layer, output: torch.Tensor, read_state) -> None:
        self.layer, self.output, self.read_state = layer, output, read_state

    def apply(self, next_boundary) -> mhc.HcState:
        update = hc_boundary.HcUpdate(self.layer, False)
        return update.post(self.output, self.read_state, next=next_boundary)


def _attn_next_boundary(layer, forward_batch, call):
    return hc_boundary.HcRead(layer, True).bind(forward_batch, call).next_boundary


def _forward_layer(
    layer,
    state: mhc.HcState,
    *,
    plan: DecodeTpSp,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    call: hc_boundary.LayerCall,
) -> _FfnPending:
    """One decoder layer on this rank's rows: the reads and posts the replicated
    layer runs through its stage boundaries, with the attention and MoE inputs
    all-gathered and their partial sums reduce-scattered."""
    attn_read = hc_boundary.HcRead(layer, True).bind(forward_batch, call)
    x, read_state = attn_read.read(state, layer.input_layernorm)
    x = plan.all_gather(x)
    with layer.self_attn.maybe_use_decode_attn_tp(forward_batch):
        out = layer.self_attn(
            x=x,
            positions=positions,
            forward_batch=forward_batch,
            x_quant=read_state.quantized,
            defer_all_reduce=True,
        )
    assert isinstance(out, mhc.AttnOutput), "decode attention TP leaves its sum"
    out = plan.reduce_scatter(out.partial)
    ffn_read = hc_boundary.HcRead(layer, False).bind(forward_batch, call)
    state = hc_boundary.HcUpdate(layer, True).post(
        out, read_state, next=ffn_read.next_boundary
    )

    x, read_state = ffn_read.read(state, layer.post_attention_layernorm)
    x = plan.all_gather(x)
    # The scope the replicated MoE transport runs the MoE in.
    with get_forward().scoped(mlp_reduce_scatter=False, defer_moe_finalize=False):
        out = layer.mlp(
            x,
            forward_batch,
            input_ids=call.input_ids,
            input_ids_global=call.input_ids_global,
            leave_output_sum=True,
        )
    if isinstance(out, MoEOutput):
        out = MoeFinalize(out).complete(reduce=False)
    return _FfnPending(layer, plan.reduce_scatter(out), read_state)


def forward_layers(
    model,
    plan: DecodeTpSp,
    *,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    input_ids: torch.Tensor,
    input_ids_global: torch.Tensor,
    dspark_aux_hidden_states: Optional[AuxHiddenStatePacker],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DeepseekV4Model's pre-mix layer loop on this rank's rows: the returned
    streams and pre-mix, and the DSpark captures, hold this rank's rows only."""
    hash_ids = (
        model.engram_hasher(input_ids, forward_batch)
        if model.engram_hasher is not None
        else None
    )
    call = hc_boundary.LayerCall(input_ids, input_ids_global)
    state = mhc.HcState(plan.scatter_rows(hidden_states))
    del hidden_states
    pending: Optional[_FfnPending] = None
    for i in range(model.start_layer, model.end_layer):
        layer = model.layers[i]
        if pending is not None:
            # The seam the replicated read takes: none ahead of an Engram layer.
            state = pending.apply(_attn_next_boundary(layer, forward_batch, call))
            pending = None
        if layer.engram is not None:
            engram = layer.engram
            residual = engram.forward_row_slice(
                state.residual,
                hash_ids[:, engram.layer_hash_index],
                start=plan.start,
                chunk=plan.chunk,
                world=plan.world,
                input_ids=input_ids,
            )
            state = state.with_residual(residual)
        if dspark_aux_hidden_states is not None and i in model.dspark_layers_to_capture:
            from sglang.kernels.ops.layernorm.mhc import mhc_stream_mean

            aux = state.residual
            mhc_stream_mean(aux, out=dspark_aux_hidden_states.slot(aux[:, 0]))
            del aux
        with get_global_expert_distribution_recorder().with_current_layer(i):
            pending = _forward_layer(
                layer,
                state,
                plan=plan,
                positions=positions,
                forward_batch=forward_batch,
                call=call,
            )
        del state
    state = pending.apply(None).materialized(model.layers[model.end_layer - 1].hc_cfg)
    return state.residual, state.pre


def gather_outputs(
    plan: DecodeTpSp,
    *,
    hidden_states: torch.Tensor,
    pre_hc_head: torch.Tensor,
    dspark_aux_hidden_states: Optional[AuxHiddenStatePacker],
    forward_batch: ForwardBatch,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Every row of the final hidden states and of the packed DSpark captures; the
    pre-norm streams only where the logits processor reads them (no DSpark
    captures, hidden-state capture on)."""
    hidden_states = plan.all_gather(hidden_states)
    if dspark_aux_hidden_states is not None:
        aux = plan.all_gather(dspark_aux_hidden_states.finalize())
        return hidden_states, None, aux
    if forward_batch.capture_hidden_mode.need_capture():
        return hidden_states, plan.all_gather(pre_hc_head), None
    return hidden_states, None, None
