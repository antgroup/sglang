"""The SM90 decode / verify indexer behind ``dsv41_sm90_on("indexer")``: rows are
scored straight from the request table with the Hopper FP4 kernels (verify
rows of a request share each decoded key tile, wgmma where it applies), chunked
at a static score budget, and selected with top-k v2. A candidate source
publishes its sorted top blocks with their fused block maxima, and a consumer
scores only those blocks instead of masking the whole context.

Logits are bitwise the upstream paged kernel's except on the wgmma kernels,
which are bitwise while every fp32 partial dot is exact (block-scale spans up
to 2^9) and may round differently past that. Ties go to the lowest position,
except where top-k v2's cluster path (long rows, small batches) truncates a
threshold bin of more than 2048 candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.sm90_candidate import (
    candidate_block_state,
    candidate_position_counts,
    finalize_candidate_topk,
)
from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer import fp4_index_logits_decode
from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_paged_v2,
)

from .types import CandidateMetadata, DecodeInputs

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

# Eight bytes per scored (row, column): the fp32 scores plus their transients.
# 1 GiB measured on H20 (b128 decode, ~570 MiB scratch peak, no extra chunk).
_SCORE_BUDGET_BYTES = 1 << 30
# A source publishes compactly from this many candidate windows of context: a
# consumer then scores topk_blocks * block_size positions instead of lmax.
_COMPACT_MIN_WINDOWS = 16
_COMPACT_BLOCK_SIZES = (1, 2, 4, 8, 16, 32, 64)
# The top-k v2 limit.
_MAX_TOPK = 2048


class Sm90DecodeCandidates(CandidateMetadata, msgspec.Struct):
    ratio: int
    block_size: int
    # [rows, topk_blocks] int32 logical block ids, ascending, the number of
    # blocks of the source's capacity past each row's valid ones
    blocks: torch.Tensor
    counts: torch.Tensor  # [rows] int32 positions the valid blocks hold
    topk_plan: torch.Tensor  # top-k v2 plan over counts, for the whole batch


def sm90_decode_lmax(inputs: DecodeInputs, req_to_token: torch.Tensor) -> int:
    """The graph-stable replay bound of compressed positions, never a host read
    of the live lengths."""
    return min(
        inputs.paged_metadata.max_compressed_seq_len,
        req_to_token.shape[1] // inputs.compress_ratio,
    )


def sm90_decode_supported(inputs: DecodeInputs) -> bool:
    return inputs.paged_metadata.use_topk_v2 and inputs.indexer.index_topk <= _MAX_TOPK


def publishes_compact(*, lmax: int, topk_blocks: int, block_size: int) -> bool:
    return (
        block_size in _COMPACT_BLOCK_SIZES
        and 0 < topk_blocks <= _MAX_TOPK
        and lmax >= _COMPACT_MIN_WINDOWS * topk_blocks * block_size
    )


def _rows_per_chunk(width: int, rows_per_request: int) -> int:
    rows = max(1, _SCORE_BUDGET_BYTES // (width * 8))
    # A request's verify rows split across chunks score exactly but decode twice.
    if rows >= rows_per_request:
        rows -= rows % rows_per_request
    return rows


def sm90_decode_topk(
    *,
    inputs: DecodeInputs,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    candidates: Optional[Sm90DecodeCandidates] = None,
    publish_blocks: int = 0,
    block_size: int = 1,
) -> Optional[Sm90DecodeCandidates]:
    """Top-k of every row into ``inputs.out_page_indices``: among ``candidates``
    when given, else over the row's whole visible context. ``publish_blocks`` > 0
    also returns the source's ``publish_blocks`` best blocks of ``block_size``."""
    inputs.reset_outputs()
    indexer, ratio = inputs.indexer, inputs.compress_ratio
    req, pos = inputs.req_rows, inputs.positions
    bs = req.shape[0]
    lmax = sm90_decode_lmax(inputs, req_to_token)
    if bs == 0 or lmax == 0:
        return None
    assert pos.shape[0] == bs, f"one request row per query, {pos.shape=} {bs=}"
    lens = ((pos + 1) // ratio).clamp(0, lmax).to(torch.int32)
    q = indexer.queries(inputs.q_lora, inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x)
    table = token_to_kv_pool.get_index_k_with_scale_buffer(inputs.layer_id)
    blocks, score_lens, shared_plan = None, lens, None
    width = lmax
    if candidates is not None:
        assert candidates.blocks.shape[0] == bs, "candidate query mapping changed"
        blocks, score_lens = candidates.blocks, candidates.counts
        block_size = candidates.block_size
        if ratio == candidates.ratio:
            shared_plan = candidates.topk_plan
        else:
            score_lens = candidate_position_counts(
                blocks, candidates.counts, lens, block_size=block_size
            )
        width = blocks.shape[1] * block_size
    rows_per_request = inputs.paged_metadata.rows_per_request if inputs.is_verify else 1
    rows_per_chunk = _rows_per_chunk(width, rows_per_request)
    publish = publish_blocks > 0
    if publish:
        published_blocks = torch.empty(
            (bs, publish_blocks), dtype=torch.int32, device=pos.device
        )
        published_counts = torch.empty(bs, dtype=torch.int32, device=pos.device)
    k = min(indexer.index_topk, width)
    for start in range(0, bs, rows_per_chunk):
        rows = slice(start, min(start + rows_per_chunk, bs))
        lengths = score_lens[rows]
        # A plan must not be the kernel right before the top-k that reads it,
        # so every plan is built ahead of the scores.
        plan = shared_plan if bs <= rows_per_chunk else None
        if plan is None:
            plan = plan_topk_v2(lengths)
        if publish:
            num_blocks = -(-lmax // block_size)
            block_lens = (-(-lengths // block_size)).clamp(0, num_blocks)
            block_plan = plan_topk_v2(block_lens)
        scores = fp4_index_logits_decode(
            q[rows],
            weights[rows],
            req_to_token,
            lengths,
            table,
            table.shape[1] // 68,
            req_indices=req[rows],
            compress_ratio=ratio,
            max_seq_len=lmax,
            # Compact candidate windows have no long invisible tail to bound.
            bounded_grid=blocks is None,
            prefix_only=True,
            candidate_blocks=blocks[rows] if blocks is not None else None,
            candidate_block_size=block_size,
            return_block_scores=publish,
            rows_per_group=rows_per_request,
            bits_decode=True,
            use_wgmma=True,
        )
        if publish:
            scores, block_scores = scores
            published_blocks[rows], published_counts[rows] = candidate_block_state(
                block_scores,
                lengths,
                topk_blocks=publish_blocks,
                block_size=block_size,
                block_lens=block_lens,
                plan=block_plan,
            )
            del block_scores
        selected = torch.empty((len(q[rows]), k), dtype=torch.int32, device=q.device)
        topk_transform_paged_v2(scores, lengths, None, selected, 1, plan, det_ties=True)
        finalize_candidate_topk(
            selected,
            scores,
            lengths,
            req_to_token,
            req[rows],
            inputs.out_page_indices[rows],
            None,
            ratio=ratio,
            candidate_blocks=blocks[rows] if blocks is not None else None,
            candidate_block_size=block_size,
        )
        # Release the large scratch before the next chunk allocates its own,
        # including when CUDA graph capture assigns private-pool storage.
        del scores, selected
    if not publish:
        return None
    return Sm90DecodeCandidates(
        ratio=ratio,
        block_size=block_size,
        blocks=published_blocks,
        counts=published_counts,
        topk_plan=plan_topk_v2(published_counts),
    )
