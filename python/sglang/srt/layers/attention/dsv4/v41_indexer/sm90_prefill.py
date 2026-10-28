"""The SM90 prefill indexer behind ``dsv41_sm90_on("indexer")``: each request's
visible index K is unpacked once to E4M3 and scored with FP8 tensor-core dots,
then selected with the deterministic-tie top-k v2 and a fused finish.

Scores match the torch BF16 path only within tolerance: the head sum runs in
FP32 here, in BF16 there."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Iterator

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.sm90_fp8_prefill import (
    finalize_prefill_topk,
    fp8_index_logits_prefill,
    keys_major_min_scores,
    quantize_bf16_index_queries_fp8,
    unpack_fp4_index_keys_to_fp8,
)
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2

from .types import PrefillInputs

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

# Arbitrary cap on one fp32 [rows, width] score chunk.
_SCORE_BUDGET_BYTES = 1 << 30
# The top-k v2 limit.
_MAX_TOPK = 2048


class Sm90PrefillRequest(msgspec.Struct, frozen=True):
    lc: int  # compressed positions visible at the request's newest row
    k: int
    slots: torch.Tensor  # [lc] int64 index-K pool slot of each position


class Sm90PrefillChunk(msgspec.Struct, frozen=True):
    rows: slice  # into the inputs' query rows
    lens: torch.Tensor  # [rows] int32 compressed positions each row sees
    # [rows, lc] fp32, -inf past each row's length; the row stride is a multiple of 4.
    scores: torch.Tensor


def sm90_prefill_supported(inputs: PrefillInputs) -> bool:
    """The per-request host lengths the FP8 path walks without a device read,
    and a k within top-k v2's limit."""
    return (
        inputs.seq_lens_cpu is not None
        and inputs.rows_per_request is not None
        and inputs.indexer.index_topk <= _MAX_TOPK
    )


def sm90_prefill_requests(
    *, inputs: PrefillInputs, token_to_kv_pool: DeepSeekV4TokenToKVPool
) -> Iterator[tuple[Sm90PrefillRequest, Iterator[Sm90PrefillChunk]]]:
    """Requests with no query row or nothing visible yet are skipped; resets
    the selection buffers first."""
    ratio = inputs.compress_ratio
    indexer = inputs.indexer
    inputs.reset_outputs()
    # Padding past the last request's rows is not a query row.
    num_rows = sum(inputs.rows_per_request)
    if num_rows == 0:
        return
    pos = inputs.positions[:num_rows]
    q = indexer.queries(inputs.q_lora[:num_rows], inputs.freqs_cis[pos])
    weights = indexer.head_weights(inputs.x[:num_rows])
    # A compressed position is visible once the query has passed its last token.
    compress_lens = ((pos + 1) // ratio).to(torch.int32)
    table = token_to_kv_pool.get_index_k_with_scale_buffer(inputs.layer_id)
    if not torch.cuda.is_current_stream_capturing():
        _prewarm_keys_major(q.device.index, q.shape[1])
    assert len(inputs.rows_per_request) == len(inputs.seq_lens_cpu)
    row = column = 0
    # inputs.k_slots lays the requests out back to back, s // ratio columns each.
    for num_request_rows, seq_len in zip(inputs.rows_per_request, inputs.seq_lens_cpu):
        lc = seq_len // ratio
        rows = slice(row, row + num_request_rows)
        slots = inputs.k_slots[column : column + lc]
        row += num_request_rows
        column += lc
        if lc == 0 or num_request_rows == 0:
            continue
        request = Sm90PrefillRequest(lc=lc, k=min(indexer.index_topk, lc), slots=slots)
        yield (
            request,
            _score_chunks(
                q=q,
                weights=weights,
                compress_lens=compress_lens,
                rows=rows,
                index_k=unpack_fp4_index_keys_to_fp8(
                    slots, table, table.shape[1] // 68
                ),
            ),
        )


def _score_chunks(
    *,
    q: torch.Tensor,
    weights: torch.Tensor,
    compress_lens: torch.Tensor,
    rows: slice,
    index_k: torch.Tensor,
) -> Iterator[Sm90PrefillChunk]:
    """index_k: the request's visible keys [lc, 128] e4m3, decoded once and
    reused by every chunk of its rows."""
    lc = index_k.shape[0]
    rows_per_chunk = max(1, _SCORE_BUDGET_BYTES // (4 * ((lc + 3) // 4 * 4)))
    for start in range(rows.start, rows.stop, rows_per_chunk):
        chunk = slice(start, min(rows.stop, start + rows_per_chunk))
        lens = compress_lens[chunk]
        # Built inline so the generator holds no reference once the caller drops it.
        yield Sm90PrefillChunk(
            rows=chunk,
            lens=lens,
            scores=fp8_index_logits_prefill(
                quantize_bf16_index_queries_fp8(q[chunk]), weights[chunk], index_k, lens
            ),
        )


def write_sm90_prefill_topk(
    inputs: PrefillInputs,
    request: Sm90PrefillRequest,
    chunk: Sm90PrefillChunk,
    *,
    mask_scores: bool,
) -> None:
    """Top-k of the chunk's scores into the selection buffers, sorted, -1 padded;
    mask_scores drops picks whose score is -inf (masked candidates, underfill)."""
    rows = chunk.rows
    selected = torch.empty(
        (rows.stop - rows.start, request.k), dtype=torch.int32, device=chunk.lens.device
    )
    topk_transform_ragged_v2(
        chunk.scores,
        chunk.lens,
        out_offsets=torch.zeros_like(chunk.lens),
        out_indices=selected,
        det_ties=True,
    )
    finalize_prefill_topk(
        selected=selected,
        scores=chunk.scores,
        lens=chunk.lens,
        slots=request.slots,
        page_indices=(
            inputs.out_page_indices[rows]
            if inputs.out_page_indices is not None
            else None
        ),
        raw_indices=inputs.out_raw_indices[rows],
        mask_scores=mask_scores,
    )


# The consumer gathers its candidate columns once the context is this many
# times their width; below it, dropping the other columns in place is cheaper.
# H20, 2048 rows x 16K candidates: dropped / gathered 5.17 / 5.54 ms at 4x,
# 7.85 / 7.66 ms at 6x, 42.4 / 35.7 ms at 32x.
_COMPACT_CONSUMER_MIN_RATIO = 6


def consume_sm90_prefill_chunk(
    inputs: PrefillInputs,
    request: Sm90PrefillRequest,
    chunk: Sm90PrefillChunk,
    *,
    blocks: torch.Tensor,
    block_size: int,
) -> None:
    """Top-k among the source's candidate ``blocks`` [rows, n] (int32, unordered,
    -1 padded) only, ties to the lowest position."""
    width = blocks.shape[1] * block_size
    if width >= request.lc:
        # A source keeps every reachable block when it has at most n of them.
        write_sm90_prefill_topk(inputs, request, chunk, mask_scores=True)
    elif request.lc >= _COMPACT_CONSUMER_MIN_RATIO * width:
        _topk_among_gathered_blocks(inputs, request, chunk, blocks, block_size)
    else:
        _drop_non_candidates(chunk.scores, blocks, block_size)
        write_sm90_prefill_topk(inputs, request, chunk, mask_scores=True)


def _drop_non_candidates(
    scores: torch.Tensor, blocks: torch.Tensor, block_size: int
) -> None:
    """-inf in place outside each row's blocks, through a per-block keep table."""
    rows, width = scores.shape
    num_blocks = -(-width // block_size)
    keep = torch.zeros((rows, num_blocks + 1), dtype=torch.bool, device=scores.device)
    keep.scatter_(1, blocks.to(torch.int64).masked_fill(blocks < 0, num_blocks), True)
    drop = ~keep[:, :num_blocks, None]
    whole = width // block_size
    scores[:, : whole * block_size].unflatten(1, (whole, block_size)).masked_fill_(
        drop[:, :whole], -torch.inf
    )
    if whole < num_blocks:
        scores[:, whole * block_size :].masked_fill_(drop[:, whole], -torch.inf)


def _topk_among_gathered_blocks(
    inputs: PrefillInputs,
    request: Sm90PrefillRequest,
    chunk: Sm90PrefillChunk,
    blocks: torch.Tensor,
    block_size: int,
) -> None:
    """The candidate columns gathered in ascending position order, so the top-k
    reads n * block_size columns, not the context, and ties keep their order."""
    rows = chunk.lens.shape[0]
    num_blocks = -(-request.lc // block_size)
    ids = blocks.masked_fill(blocks < 0, num_blocks).sort(dim=1).values
    offsets = torch.arange(block_size, device=ids.device)
    positions = (ids.to(torch.int64)[:, :, None] * block_size + offsets).flatten(1)
    # Every candidate block but the newest lies wholly before the row's length,
    # so the visible candidate columns are a prefix of the gathered row.
    counts = (positions < chunk.lens[:, None]).sum(dim=1, dtype=torch.int32)
    compact = chunk.scores.gather(1, positions.clamp_max(request.lc - 1))
    selected = torch.empty((rows, request.k), dtype=torch.int32, device=compact.device)
    topk_transform_ragged_v2(
        compact,
        counts,
        out_offsets=torch.zeros_like(counts),
        out_indices=selected,
        det_ties=True,
    )
    del compact
    picked = torch.where(
        selected >= 0, positions.gather(1, selected.clamp_min(0).to(torch.int64)), -1
    ).to(torch.int32)
    finalize_prefill_topk(
        selected=picked,
        scores=chunk.scores,
        lens=chunk.lens,
        slots=request.slots,
        page_indices=(
            inputs.out_page_indices[chunk.rows]
            if inputs.out_page_indices is not None
            else None
        ),
        raw_indices=inputs.out_raw_indices[chunk.rows],
        mask_scores=True,
    )


@functools.cache
def _prewarm_keys_major(device_index: int, heads: int) -> None:
    """Load every serving variant of the keys-major scorer at the first prefill,
    so a later chunk shape or length alignment compiles nothing mid-serving."""
    device = torch.device("cuda", device_index)
    if keys_major_min_scores(device) is None:
        return
    # Lazy: the Gluon Hopper API is experimental; import it only when it runs.
    from sglang.kernels.ops.attention.dsv4.sm90_fp8_prefill_keys_major import (
        prewarm_keys_major_scores,
    )

    prewarm_keys_major_scores(device, heads)
