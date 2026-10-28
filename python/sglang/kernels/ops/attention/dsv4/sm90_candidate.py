"""Hopper decode candidate publication and top-k output mapping, adapted from
sglang PRs #38848 and #39057: a source publishes sorted logical blocks and the
count of positions they hold; consumers resolve physical slots with their own
ratio and request mapping."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2


@triton.jit
def _sort_candidate_blocks_kernel(
    BLOCKS,
    SCORES,
    LENS,
    SORTED,
    COUNTS,
    BLOCK_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    TOPK_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, TILE)
    blocks = tl.load(
        BLOCKS + row * BLOCK_STRIDE + offsets, offsets < TOPK_BLOCKS, -1
    ).to(tl.int32)
    safe_blocks = tl.minimum(tl.maximum(blocks, 0), NUM_BLOCKS - 1)
    length = tl.load(LENS + row)
    reachable = (offsets < TOPK_BLOCKS) & (blocks >= 0) & (blocks < NUM_BLOCKS)
    reachable &= blocks < tl.cdiv(tl.maximum(length, 0), BLOCK_SIZE)
    scores = tl.load(
        SCORES + row * SCORE_STRIDE + safe_blocks,
        mask=reachable,
        other=-float("inf"),
    )
    valid = (
        (offsets < TOPK_BLOCKS)
        & (blocks >= 0)
        & (blocks < NUM_BLOCKS)
        & (scores > -float("inf"))
    )
    blocks = tl.where(valid, blocks, NUM_BLOCKS)
    blocks = tl.sort(blocks)
    tl.store(SORTED + row * TOPK_BLOCKS + offsets, blocks, offsets < TOPK_BLOCKS)

    remaining = length - blocks * BLOCK_SIZE
    contribution = tl.minimum(tl.maximum(remaining, 0), BLOCK_SIZE)
    contribution = tl.where(blocks < NUM_BLOCKS, contribution, 0)
    tl.store(COUNTS + row, tl.sum(contribution, axis=0))


@triton.jit
def _finalize_candidate_topk_kernel(
    SELECTED,
    SCORES,
    LENS,
    REQ_TO_TOKEN,
    REQ,
    CANDIDATE_BLOCKS,
    PAGE_INDICES,
    RAW_INDICES,
    SELECTED_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    CANDIDATE_BLOCK_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    RAW_STRIDE: tl.constexpr,
    SELECTED_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    SOURCE_WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    USE_CANDIDATES: tl.constexpr,
    HAS_RAW: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, TOPK)
    selected = tl.load(
        SELECTED + row * SELECTED_STRIDE + offsets, offsets < SELECTED_WIDTH, -1
    ).to(tl.int64)
    length = tl.minimum(tl.load(LENS + row).to(tl.int64), SOURCE_WIDTH)
    valid = (selected >= 0) & (selected < length)
    score = tl.load(
        SCORES + row * SCORE_STRIDE + tl.maximum(selected, 0),
        mask=valid,
        other=-float("inf"),
    )
    valid &= score > -float("inf")
    selected = tl.where(valid, selected, SOURCE_WIDTH)
    selected = tl.sort(selected)
    valid = selected < length
    safe = tl.minimum(selected, SOURCE_WIDTH - 1)

    if USE_CANDIDATES:
        block_col = safe // CANDIDATE_BLOCK_SIZE
        within = safe % CANDIDATE_BLOCK_SIZE
        block = tl.load(
            CANDIDATE_BLOCKS + row * CANDIDATE_BLOCK_STRIDE + block_col,
            mask=valid,
            other=0,
        )
        logical = block * CANDIDATE_BLOCK_SIZE + within
        req = tl.load(REQ + row).to(tl.int64)
        slot = tl.load(
            REQ_TO_TOKEN + req * REQ_STRIDE + logical * RATIO,
            mask=valid,
            other=0,
        )
        slot = slot // RATIO
    else:
        logical = safe
        req = tl.load(REQ + row).to(tl.int64)
        slot = tl.load(
            REQ_TO_TOKEN + req * REQ_STRIDE + logical * RATIO,
            mask=valid,
            other=0,
        )
        slot = slot // RATIO

    tl.store(
        PAGE_INDICES + row * OUTPUT_STRIDE + offsets,
        tl.where(valid, slot, -1),
        offsets < OUTPUT_WIDTH,
    )
    if HAS_RAW:
        tl.store(
            RAW_INDICES + row * RAW_STRIDE + offsets,
            tl.where(valid, logical, -1),
            offsets < OUTPUT_WIDTH,
        )


@triton.jit
def _candidate_counts_kernel(
    BLOCKS,
    SOURCE_COUNTS,
    LENS,
    OUT,
    STRIDE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, TILE)
    blocks = tl.load(BLOCKS + row * STRIDE + col, col < NUM_BLOCKS, 0)
    length = tl.load(LENS + row)
    n = tl.minimum(tl.maximum(length - blocks * BLOCK_SIZE, 0), BLOCK_SIZE)
    n = tl.where(col < NUM_BLOCKS, n, 0)
    # A ratio-2 source can publish a partial final block. A ratio-1 consumer
    # must not add positions outside that source-visible prefix.
    tl.store(OUT + row, tl.minimum(tl.sum(n, 0), tl.load(SOURCE_COUNTS + row)))


def candidate_block_state(
    block_scores: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    topk_blocks: int,
    block_size: int,
    block_lens: torch.Tensor,
    plan: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort the selected reachable blocks, partial tails before padding: int32
    ``[rows, k]`` block ids (``width`` past the count) and the positions they hold.

    ``block_lens`` is ``ceil(seq_lens / block_size)`` clamped to the width and
    ``plan`` its top-k v2 plan, both built before the scores are: a plan must
    not be the kernel right before the top-k that reads it.
    """
    rows, width = block_scores.shape
    k = min(topk_blocks, width)
    assert 0 < k <= 2048
    selected = torch.empty((rows, k), dtype=torch.int32, device=block_scores.device)
    topk_transform_paged_v2(
        block_scores, block_lens, None, selected, 1, plan, det_ties=True
    )
    sorted_blocks = torch.empty_like(selected)
    counts = torch.empty(rows, dtype=torch.int32, device=block_scores.device)
    _sort_candidate_blocks_kernel[(rows,)](
        selected,
        block_scores,
        seq_lens,
        sorted_blocks,
        counts,
        selected.stride(0),
        block_scores.stride(0),
        k,
        block_size,
        width,
        triton.next_power_of_2(k),
        num_warps=8,
    )
    return sorted_blocks, counts


def candidate_position_counts(blocks, source_counts, seq_lens, *, block_size):
    """Intersect a source's valid prefix with the consuming ratio's visibility."""
    counts = torch.empty_like(source_counts)
    _candidate_counts_kernel[(blocks.shape[0],)](
        blocks,
        source_counts,
        seq_lens,
        counts,
        blocks.stride(0),
        blocks.shape[1],
        block_size,
        triton.next_power_of_2(blocks.shape[1]),
        num_warps=8,
    )
    return counts


def finalize_candidate_topk(
    selected,
    scores,
    score_lens,
    req_to_token,
    req,
    page_indices,
    raw_indices,
    *,
    ratio,
    candidate_blocks=None,
    candidate_block_size=1,
):
    """Write logical-order slots and -1 padding without large gather tensors."""
    assert scores.shape[1] > 0
    _finalize_candidate_topk_kernel[(selected.shape[0],)](
        selected,
        scores,
        score_lens,
        req_to_token,
        req,
        candidate_blocks if candidate_blocks is not None else req_to_token,
        page_indices,
        raw_indices if raw_indices is not None else page_indices,
        selected.stride(0),
        scores.stride(0),
        req_to_token.stride(0),
        candidate_blocks.stride(0) if candidate_blocks is not None else 0,
        page_indices.stride(0),
        raw_indices.stride(0) if raw_indices is not None else 0,
        selected.shape[1],
        page_indices.shape[1],
        triton.next_power_of_2(page_indices.shape[1]),
        scores.shape[1],
        ratio,
        candidate_block_size,
        candidate_blocks is not None,
        raw_indices is not None,
        num_warps=8,
    )
