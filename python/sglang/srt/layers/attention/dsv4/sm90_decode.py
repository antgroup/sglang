"""DeepSeek-V4.1 decode attention on SM90 (dsv41_sm90 "decode").

The H20 tile schedule for FlashMLA's legacy sparse decode, and the DSpark draft
rows of a request packed into one FlashMLA row. Neither is bitwise to the stock
path: another split-KV partition changes the combine, by at most one bf16 ulp of
a row's largest output on H20, with the same error against an fp64 reference.
"""

from __future__ import annotations

import functools
from typing import Callable, Optional

import msgspec
import torch

# The JIT scheduler stages its schedules in 48 KB of shared memory.
_SCHED_MAX_BATCH = 1024
# Devices the scheduler's cost model was fitted on (one 78-SM die); others keep
# FlashMLA's own schedule.
_SCHED_DEVICES = ("NVIDIA H20", "NVIDIA H20-3e")
# FlashMLA's SM90 sparse decode takes these head counts only.
_DECODE_HEADS = (64, 128)


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def flashmla_sched_tuned(device: torch.device) -> bool:
    return torch.cuda.get_device_name(device) in _SCHED_DEVICES


def fill_flashmla_sched_meta(
    flashmla_metadata,
    *,
    q: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
) -> None:
    """Hand an unscheduled FlashMLASchedMeta the H20 cost-model schedule, so the
    legacy decode skips its own get_mla_metadata_kernel; other shapes keep it."""
    if (
        flashmla_metadata is None
        or flashmla_metadata.tile_scheduler_metadata is not None
        or topk_length is None
    ):
        return
    b, s_q, h_q = q.shape[0], q.shape[1], q.shape[2]
    if s_q != 1 or h_q not in _DECODE_HEADS or b > _SCHED_MAX_BATCH:
        return
    from sglang.kernels.ops.attention.dsv4.flashmla_sparse_sched import (
        flashmla_sparse_sched_meta,
    )

    # Legacy Decode_Sm90_Impl::get_meta: num_sm_parts = num_sms / s_q / (h_q / 64).
    num_sm_parts = max(_num_sms(q.device.index) // (h_q // 64), 1)
    meta, num_splits = flashmla_sparse_sched_meta(
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        topk=indices.shape[-1],
        extra_topk=0 if extra_indices is None else extra_indices.shape[-1],
        num_sm_parts=num_sm_parts,
        # Same schedule as the per-request walk; on H20 from 6 rows up 0.4-28 us
        # faster per call (b=384 SWA128+512: 29.5 -> 12.2 us), 0.3 us slower at 1.
        per_part=True,
    )
    flashmla_metadata.tile_scheduler_metadata = meta
    flashmla_metadata.num_splits = num_splits


def packed_draft_rows_per_req(*, rows_per_req: int, num_heads: int) -> int:
    """Draft rows per FlashMLA row when the rows' heads fit one decode row, else 0."""
    if num_heads >= _DECODE_HEADS[0] or rows_per_req * num_heads > _DECODE_HEADS[-1]:
        return 0
    return rows_per_req


class PackedDraftSwa(msgspec.Struct, frozen=True):
    """One SWA index set per request, shared by the draft layers of a forward."""

    # The forward's swa_page_indices these were cut from; every forward assigns
    # a new tensor, while a draft metadata object outlives warmup into capture.
    source: torch.Tensor
    indices: torch.Tensor
    topk_lengths: torch.Tensor
    flashmla_metadata: object


def forward_packed_draft(
    *,
    flash_mla_with_kvcache: Callable,
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_page_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    attn_sink: torch.Tensor,
    rows_per_req: int,
    cached: Optional[PackedDraftSwa],
    source: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float,
    tuned_sched: bool,
) -> tuple[torch.Tensor, PackedDraftSwa]:
    """q [rows, 1, H, D] -> o [rows, H, Dv]. The draft rows of a request attend
    one SWA index set, so request r runs as one FlashMLA row whose heads are
    [row 0 heads | row 1 heads | ...] zero-padded to 64 or 128."""
    from sglang.kernels.ops.attention.flash_mla_interface import get_mla_metadata

    rows, _, heads, dim = q.shape
    assert rows % rows_per_req == 0, (rows, rows_per_req)
    reqs = rows // rows_per_req
    packed_heads = rows_per_req * heads
    padded_heads = next(h for h in _DECODE_HEADS if packed_heads <= h)
    q_packed = q.new_zeros(reqs, 1, padded_heads, dim)
    q_packed[:, 0, :packed_heads].copy_(q.reshape(reqs, packed_heads, dim))
    sink = attn_sink.new_zeros(padded_heads)
    sink[:packed_heads].copy_(attn_sink[:heads].repeat(rows_per_req))
    if cached is None or cached.source is not source:
        cached = PackedDraftSwa(
            source=source,
            indices=swa_page_indices[::rows_per_req].contiguous(),
            topk_lengths=swa_topk_lengths[::rows_per_req].contiguous(),
            flashmla_metadata=get_mla_metadata()[0],
        )
    if tuned_sched:
        fill_flashmla_sched_meta(
            cached.flashmla_metadata,
            q=q_packed,
            indices=cached.indices,
            topk_length=cached.topk_lengths,
            extra_indices=None,
            extra_topk_length=None,
        )
    o = flash_mla_with_kvcache(
        q=q_packed,
        k_cache=swa_k_cache,
        head_dim_v=head_dim_v,
        block_table=None,
        cache_seqlens=None,
        tile_scheduler_metadata=cached.flashmla_metadata,
        softmax_scale=softmax_scale,
        is_fp8_kvcache=True,
        indices=cached.indices,
        topk_length=cached.topk_lengths,
        attn_sink=sink,
    )[0]
    return o[:, 0, :packed_heads].reshape(rows, heads, o.shape[-1]), cached
