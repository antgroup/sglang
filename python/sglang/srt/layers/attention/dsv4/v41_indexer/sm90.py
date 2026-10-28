"""The V4.1 indexer backends on SM90 behind ``dsv41_sm90_on("indexer")``: FP8
prefill scoring (``sm90_prefill``) and the compact-candidate decode / verify
indexer (``sm90_decode``); anything they do not cover keeps the upstream path."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    select_candidate_block_ids,
)

from .dense_blocks import BlockIds, DenseBlocksBackend
from .full_topk import FullTopKIndexer
from .sm90_decode import (
    Sm90DecodeCandidates,
    publishes_compact,
    sm90_decode_lmax,
    sm90_decode_supported,
    sm90_decode_topk,
)
from .sm90_prefill import (
    consume_sm90_prefill_chunk,
    sm90_prefill_requests,
    sm90_prefill_supported,
    write_sm90_prefill_topk,
)
from .types import DecodeInputs, PrefillInputs


class Sm90FullTopKIndexer(FullTopKIndexer):
    def topk_prefill(self, inputs: PrefillInputs) -> None:
        if not sm90_prefill_supported(inputs):
            super().topk_prefill(inputs)
            return
        requests = sm90_prefill_requests(
            inputs=inputs, token_to_kv_pool=self.token_to_kv_pool
        )
        for request, chunks in requests:
            for chunk in chunks:
                write_sm90_prefill_topk(inputs, request, chunk, mask_scores=False)
                # Free this chunk's scores before the generator scores the next one.
                del chunk

    def topk_decode(self, inputs: DecodeInputs) -> None:
        if not sm90_decode_supported(inputs):
            super().topk_decode(inputs)
            return
        sm90_decode_topk(
            inputs=inputs,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
        )


class Sm90DenseBlocksBackend(DenseBlocksBackend):
    def publish_decode(self, inputs: DecodeInputs):
        compact = sm90_decode_supported(inputs) and publishes_compact(
            lmax=sm90_decode_lmax(inputs, self.req_to_token),
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
        )
        if not compact:
            return super().publish_decode(inputs)
        return sm90_decode_topk(
            inputs=inputs,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            publish_blocks=self.topk_blocks,
            block_size=self.block_size,
        )

    def consume_decode(self, inputs: DecodeInputs, published) -> None:
        # Consumers follow their source: ratios 1 and 2 can straddle the
        # compact capacity threshold in the same forward.
        if not isinstance(published, Sm90DecodeCandidates):
            super().consume_decode(inputs, published)
            return
        sm90_decode_topk(
            inputs=inputs,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            candidates=published,
        )

    def publish_prefill(self, inputs: PrefillInputs) -> BlockIds:
        if not sm90_prefill_supported(inputs):
            return super().publish_prefill(inputs)
        picked = []  # (query rows, their block ids) per scored chunk
        requests = sm90_prefill_requests(
            inputs=inputs, token_to_kv_pool=self.token_to_kv_pool
        )
        for request, chunks in requests:
            for chunk in chunks:
                ids = select_candidate_block_ids(
                    chunk.scores,
                    chunk.lens[:, None],
                    topk_blocks=self.topk_blocks,
                    block_size=self.block_size,
                )
                picked.append((chunk.rows, ids))
                write_sm90_prefill_topk(inputs, request, chunk, mask_scores=False)
                del chunk
        width = max((ids.shape[1] for _, ids in picked), default=0)
        blocks = torch.full(
            (inputs.positions.shape[0], width),
            -1,
            dtype=torch.int32,
            device=inputs.positions.device,
        )
        for rows, ids in picked:
            blocks[rows, : ids.shape[1]] = ids
        return BlockIds(blocks=blocks, rows_per_request=inputs.rows_per_request)

    def consume_prefill(
        self, inputs: PrefillInputs, published: Optional[BlockIds]
    ) -> None:
        if not sm90_prefill_supported(inputs):
            super().consume_prefill(inputs, published)
            return
        assert published is not None
        requests = sm90_prefill_requests(
            inputs=inputs, token_to_kv_pool=self.token_to_kv_pool
        )
        for request, chunks in requests:
            for chunk in chunks:
                consume_sm90_prefill_chunk(
                    inputs,
                    request,
                    chunk,
                    blocks=published.blocks[chunk.rows],
                    block_size=self.block_size,
                )
                del chunk
