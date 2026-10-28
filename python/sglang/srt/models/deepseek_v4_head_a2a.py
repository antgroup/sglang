"""Head <-> row all-to-all of the DeepSeek-V4.1 SM90 decode attention (mix decode).

Under attention TP (CP decode attention TP on a non-disaggregated V4.1) every TP
rank holds the whole MQA KV of every request and runs FlashMLA's sparse decode on
all rows with its ``n_heads / tp`` heads, zero-padded to the kernel's 64. With
``dsv41_sm90_on("mix_decode")`` (role mix) a decode / verify batch of enough rows
instead attends rows ``[r * chunk, (r + 1) * chunk)`` with all heads on rank r
(``chunk = ceil(rows / tp)``): one all-to-all over the TP group moves q there, a
second one returns every rank its heads of every row. The exchange runs over
NVLink symmetric memory when the TP group has a custom all-reduce v2, else over
NCCL; both move the same bytes. Not bitwise the padded path: FlashMLA's split-KV
schedule depends on the rows it attends.

From ``_ROW_Q_MIN_ROWS`` rows (CP decode attention TP only, where wq_b is
replicated) a rank projects q for its own rows with all heads straight from the
full wq_b, which skips the first exchange, and the fused wqkv_a computes its q
half for those rows only. The V4.1 low-ratio indexers then score only the rows
the rank attends (the backend's ``row_slice``).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional, Tuple

import msgspec
import torch

from sglang.srt.layers.cp.cp_decode_attn_tp import get_cp_decode_attn_tp_ctx
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Measured on H20-3e: below this threshold,
# the two exchanges cost more than the head padding they remove.
_MIN_ROWS = 96
# Measured on 8x H20-3e (TP8 + prefill CP8, DSpark verify): below this threshold,
# reading the full wq_b costs more than the exchange.
_ROW_Q_MIN_ROWS = 288
# FlashMLA's SM90 sparse decode specializes h_q for these head counts.
_KERNEL_HEADS = (64, 128)


class HeadA2APlan(msgspec.Struct, frozen=True):
    """One decode / verify forward's row slice; every layer of it takes the same."""

    # The exchange group, rank order = head order.
    group: object
    # Rows [start, end) are this rank's; rows from num_rows on are zero padding.
    start: int
    end: int
    num_rows: int
    # q of the slice is projected from the full wq_b (no first exchange).
    row_q: bool

    @property
    def chunk(self) -> int:
        return self.end - self.start

    @property
    def padded_rows(self) -> int:
        return self.chunk * self.group.world_size

    @property
    def real_end(self) -> int:
        return min(self.end, self.num_rows)

    @property
    def row_slice(self) -> Tuple[int, int, int]:
        """The backend's (start, end, num_rows)."""
        return self.start, self.end, self.num_rows


def head_a2a_rows(num_rows: int, group: GroupCoordinator) -> Tuple[int, int]:
    """[start, end) of the rows this rank attends: equal chunks in rank order of
    num_rows rows padded to a multiple of the group size."""
    chunk = -(-num_rows // group.world_size)
    start = group.rank_in_group * chunk
    return start, start + chunk


def _exchange_group(*, tp_size: int, tp_rank: int) -> Optional[GroupCoordinator]:
    """The group whose rank order is the attention head order and whose pynccl the
    CUDA graphs capture. Under CP decode attention TP the attention TP group is size
    1 and the (attention) TP ranks are the CP ranks, which the TP group matches."""
    parallel = get_parallel()
    for group in (parallel.attn_tp_group, parallel.tp_group):
        if (
            group.world_size == tp_size
            and group.rank_in_group == tp_rank
            and group.pynccl_comm is not None
        ):
            return group
    return None


def plan_head_a2a(
    *, layer, num_rows: int, forward_batch: ForwardBatch
) -> Optional[HeadA2APlan]:
    """The row slice of this forward, or None for the head-padded attention. The
    answer depends on the rows, the forward mode and the config only, so every
    rank and every layer of the batch agree: the row-sliced indexer relies on it,
    as a layer without an indexer attends with the top-k of its source layer."""
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    mode = forward_batch.forward_mode
    if not (
        num_rows >= _MIN_ROWS
        and (mode.is_decode() or mode.is_target_verify())
        and layer.attn_tp_size > 1
        # Only fewer heads than the kernel's smallest h_q are padded.
        and layer.n_local_heads < min(_KERNEL_HEADS)
        and layer.n_heads in _KERNEL_HEADS
        and not is_batch_invariant_mode_enabled()
    ):
        return None
    group = _exchange_group(tp_size=layer.attn_tp_size, tp_rank=layer.attn_tp_rank)
    if group is None:
        return None
    start, end = head_a2a_rows(num_rows, group)
    return HeadA2APlan(
        group=group,
        start=start,
        end=end,
        num_rows=num_rows,
        row_q=num_rows >= _ROW_Q_MIN_ROWS and full_wq_b(layer) is not None,
    )


def full_wq_b(layer) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The replicated wq_b weight and block scale behind CP decode attention TP's
    head slice, while that slices it; None otherwise."""
    ctx = get_cp_decode_attn_tp_ctx()
    weight = ctx.unsliced(layer.wq_b, "weight")
    scale = ctx.unsliced(layer.wq_b, "weight_scale_inv")
    method = layer.wq_b.quant_method
    if (
        weight is None
        or scale is None
        or method.w8a8_block_fp8_linear is None
        or weight.dtype != torch.float8_e4m3fn
    ):
        return None
    return weight, scale


def _block_linear(method, *, x, weight, scale, out=None):
    return method.w8a8_block_fp8_linear(
        input=x,
        weight=weight,
        block_size=method.weight_block_size,
        weight_scale=scale,
        input_scale=None,
        bias=None,
        **({} if out is None else {"out": out}),
    )


def project_row_q(
    *, layer, q_lora: torch.Tensor, positions: torch.Tensor, plan: HeadA2APlan
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q [chunk, n_heads, head_dim] (before RoPE) of this rank's rows from the
    full wq_b, and their positions; padding rows are zero."""
    start, end, chunk = plan.start, plan.real_end, plan.chunk
    q_lora, positions = q_lora[start:end], positions[start:end]
    if q_lora.shape[0] < chunk:
        pad = chunk - q_lora.shape[0]
        q_lora = torch.cat([q_lora, q_lora.new_zeros(pad, q_lora.shape[1])])
        positions = torch.cat([positions, positions.new_zeros(pad)])
    weight, scale = full_wq_b(layer)
    q = _block_linear(layer.wq_b.quant_method, x=q_lora, weight=weight, scale=scale)
    return q.view(chunk, layer.n_heads, layer.head_dim), positions


def _stores_into_slices(linear) -> bool:
    # The SM90 block32 routes (fp8_gemm) write a GEMM into a column slice of a wider buffer.
    return (
        isinstance(linear, functools.partial)
        and linear.keywords.get("block32_routes") is not None
    )


def wqkv_a_row_q(
    *, layer, x: torch.Tensor, plan: HeadA2APlan
) -> Optional[torch.Tensor]:
    """The fused wqkv_a output with its kv half for every row (every rank stores the
    whole KV) and its q half for this rank's rows only, zero elsewhere (only the
    attention and the row-sliced indexer of those rows read it). None when wqkv_a is
    not a block-FP8 linear whose q / kv split falls on a scale block."""
    if not plan.row_q:
        return None
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    method = layer.wqkv_a.quant_method
    if (
        not isinstance(method, Fp8LinearMethod)
        or not method.block_quant
        or method.w8a8_block_fp8_linear is None
    ):
        return None
    weight, scale = layer.wqkv_a.weight, layer.wqkv_a.weight_scale_inv
    if weight.dtype != torch.float8_e4m3fn:
        return None
    rank = layer.q_lora_rank
    block_n = method.weight_block_size[0]
    if rank % block_n or scale.shape[0] * block_n < weight.shape[0]:
        return None
    start, end = plan.start, plan.real_end
    out = x.new_zeros(x.shape[0], weight.shape[0])
    q_weight, q_scale = weight[:rank], scale[: rank // block_n]
    kv_weight, kv_scale = weight[rank:], scale[rank // block_n :]
    if _stores_into_slices(method.w8a8_block_fp8_linear):
        _block_linear(method, x=x, weight=kv_weight, scale=kv_scale, out=out[:, rank:])
        if end > start:
            _block_linear(
                method,
                x=x[start:end],
                weight=q_weight,
                scale=q_scale,
                out=out[start:end, :rank],
            )
        return out
    out[:, rank:] = _block_linear(method, x=x, weight=kv_weight, scale=kv_scale)
    if end > start:
        out[start:end, :rank] = _block_linear(
            method, x=x[start:end], weight=q_weight, scale=q_scale
        )
    return out


def new_q_buffer(*, layer, x: torch.Tensor, plan: HeadA2APlan) -> torch.Tensor:
    """The q the forward writes: this rank's slice with all heads under row_q, else
    this rank's heads of every row padded with zero rows to whole chunks."""
    if plan.row_q:
        return x.new_empty(plan.chunk, layer.n_heads, layer.head_dim)
    q = x.new_empty(plan.padded_rows, layer.n_local_heads, layer.head_dim)
    q[plan.num_rows :].zero_()
    return q


def attend_rows(
    *, layer, q: torch.Tensor, plan: HeadA2APlan, attn_backend, forward_batch
) -> torch.Tensor:
    """q from new_q_buffer -> the attention output of this rank's heads of every
    row, [num_rows, n_local_heads, v_head_dim]."""
    group = plan.group
    symm = _symm_exchange(layer=layer, group=group, dtype=q.dtype)
    if plan.row_q:
        q_rows = q
    else:
        q_rows = symm.to_row_layout(q[: plan.num_rows]) if symm is not None else None
        if q_rows is None:
            q_rows = _rows_over_nccl(group, q)
    o_rows = attn_backend.forward(
        q=q_rows,
        k=q_rows,
        v=q_rows,
        layer=layer.attn_mqa,
        forward_batch=forward_batch,
        compress_ratio=layer.compress_ratio,
        attn_sink=layer.attn_sink,
        save_kv_cache=False,
        row_slice=plan.row_slice,
    )
    out = symm.to_head_layout(o_rows) if symm is not None else None
    if out is None:
        out = _heads_over_nccl(group, o_rows)
    return out[: plan.num_rows]


def _symm_exchange(*, layer, group, dtype: torch.dtype):
    from sglang.srt.distributed.device_communicators.head_a2a import (
        get_symm_head_a2a,
    )

    return get_symm_head_a2a(
        group,
        local_heads=layer.n_local_heads,
        head_bytes=max(layer.head_dim, layer.attn_mqa.v_head_dim) * dtype.itemsize,
    )


def _rows_over_nccl(group, q: torch.Tensor) -> torch.Tensor:
    """[W * chunk, H, D] -> [chunk, W * H, D]: rank r gets rows [r * chunk, (r + 1)
    * chunk) of every rank's heads."""
    world = group.world_size
    padded, heads, dim = q.shape
    chunk = padded // world
    recv = torch.empty_like(q)
    group.all_to_all_single(recv.view(-1), q.view(-1))
    # recv[s, t] holds heads [s * H, (s + 1) * H) of row rank * chunk + t.
    return (
        recv.view(world, chunk, heads, dim)
        .transpose(0, 1)
        .reshape(chunk, world * heads, dim)
    )


def _heads_over_nccl(group, o_rows: torch.Tensor) -> torch.Tensor:
    """[chunk, W * H, Dv] -> [W * chunk, H, Dv]: this rank's heads of every row."""
    world = group.world_size
    chunk, all_heads, dim = o_rows.shape
    heads = all_heads // world
    send = o_rows.view(chunk, world, heads, dim).transpose(0, 1).contiguous()
    out = torch.empty_like(send)
    group.all_to_all_single(out.view(-1), send.view(-1))
    return out.view(world * chunk, heads, dim)
