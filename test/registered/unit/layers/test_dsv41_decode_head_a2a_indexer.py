"""The row-sliced V4.1 indexer on CPU (deepseek_v4_head_a2a, mix_decode): under
the SM90 head<->row all-to-all each rank selects the ratio-1/2 index top-k of
only the rows its attention reads.

The layers of one forward (a full top-k layer, a candidate source and its
consumer) run through DeepseekV4AttnBackend.forward_low_ratio_sources once for
the whole batch and once per simulated rank with its row slice. The stand-in
scores are a function of the row's own inputs, so every rank's rows of the shared
top-k buffers must equal the same rows of the whole-batch run, and no other row
may be written."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import deepseek_v4_backend
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.attention.dsv4.v41_indexer import dense_blocks, full_topk
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
    BlockIds,
    DenseBlocksBackend,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.full_topk import FullTopKIndexer
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import DecodeScores
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_v4_head_a2a import head_a2a_rows
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TOPK = 8
LMAX = 48  # compressed positions the decode scores span
PAGE = 4  # compressed page size of the indexer page table
UNWRITTEN = 12345
# (layer id, ratio, candidate source, candidate consumer, has an indexer)
LAYERS = (
    (2, 2, False, False, True),
    (3, 2, False, False, False),  # reads layer 2's top-k
    (20, 1, True, False, True),
    (21, 1, False, False, False),
    (24, 1, False, True, True),
)


def _fake_decode_scores(
    *, inputs, token_to_kv_pool, req_to_token, candidate_mask=None, **_
):
    """Non-paged scores of the rows in ``inputs``, a function of each row's own
    x, q_lora, request and position, so a mis-sliced input changes them."""
    req, pos = inputs.req_rows, inputs.positions
    bs = req.shape[0]
    inputs.reset_outputs()
    if bs == 0:
        return None
    ratio = inputs.compress_ratio
    lens = (pos + 1) // ratio
    j = torch.arange(LMAX)
    scores = torch.sin(
        j[None, :] * 0.37
        + inputs.x[:, :1].double() * 1.7
        + inputs.q_lora[:, :1].double() * 3.1
        + req[:, None] * 2.9
        + pos[:, None] * 1.3
        + inputs.layer_id * 0.11
    ).float()
    scores = scores.masked_fill(j[None, :] >= lens[:, None], -torch.inf)
    slots = req_to_token[req[:, None], (j * ratio)[None, :]] // ratio
    slots = slots.masked_fill(j[None, :] >= lens[:, None], 0)
    return DecodeScores(bs=bs, lmax=LMAX, lens=lens, slots=slots, scores=scores)


def _fake_topk_page_transform(
    scores, topk, *, page_table, page_size, end, sorted_index, output_idx
):
    """DeepSelect's contract: top-k columns of each row through its page table."""
    assert page_table.shape[0] == scores.shape[0] == output_idx.shape[0]
    output_idx.fill_(-1)
    for r in range(scores.shape[0]):
        k = min(topk, int(end[r]))
        cols = scores[r].topk(k).indices.sort().values
        output_idx[r, :k] = (
            page_table[r, cols // page_size] * page_size + cols % page_size
        )
    return output_idx


class _Case:
    def __init__(self, mode, num_requests, rows_per_request):
        torch.manual_seed(num_requests * 10 + rows_per_request)
        rows = num_requests * rows_per_request
        self.mode, self.rows = mode, rows
        request = torch.arange(rows) // rows_per_request
        self.req = request + 3  # req_to_token rows
        # Visible lengths from 1 up to under LMAX, some shorter than TOPK.
        prefix = 2 + 3 * request
        self.pos = prefix + torch.arange(rows) % rows_per_request
        self.x = torch.randn(rows, 16)
        self.q_lora = torch.randn(rows, 8)
        self.req_to_token = (
            torch.arange(32)[:, None] * 1000 + torch.arange(2 * LMAX + 2)[None, :]
        )
        # Per batch row, distinct across rows: an offset error shows.
        self.page_table = (
            torch.arange(rows)[:, None] * 100 + torch.arange(LMAX // PAGE)[None, :]
        ).int()

    def backend(self, deep_select: bool):
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.tail_source_kv_layer_id = None
        # Target verify is an extend mode; it never runs in the prefill graph.
        backend._low_ratio_in_prefill_graph = lambda: False
        full = FullTopKIndexer(
            token_to_kv_pool=None,
            req_to_token=self.req_to_token,
            use_deep_gemm_prefill=False,
            use_deep_gemm_decode=False,
        )
        full.use_deep_select_decode = deep_select
        backend.full_topk_indexer = full
        backend.decode_candidates = DenseBlocksBackend(
            token_to_kv_pool=None,
            req_to_token=self.req_to_token,
            candidate_topk_blocks=3,
            candidate_block_size=8,
            use_deep_gemm_prefill=False,
        )
        buffers = {
            ratio: torch.full((self.rows, TOPK), UNWRITTEN, dtype=torch.int32)
            for ratio in (1, 2)
        }
        indexer_metadata = SimpleNamespace(
            max_compressed_seq_len=LMAX,
            page_table=self.page_table,
            compressed_page_size=PAGE,
        )
        backend.forward_metadata = SimpleNamespace(
            low_ratio_req_indices=self.req,
            low_ratio_pos_i64=self.pos,
            candidate_metadata=None,
            c1_indexer_metadata=indexer_metadata,
            c2_indexer_metadata=indexer_metadata,
            core_metadata=SimpleNamespace(sparse_page_indices=buffers.__getitem__),
        )
        backend.compressed_rows = []
        backend._low_ratio_compress = lambda layer, x, req, pos, fb: (
            backend.compressed_rows.append((x.shape[0], req.shape[0], pos.shape[0]))
        )
        return backend, buffers

    def run(self, deep_select, row_slice=None):
        """The top-k buffers after each layer, and the backend."""
        backend, buffers = self.backend(deep_select)
        forward_batch = SimpleNamespace(
            forward_mode=self.mode, encoder_swa_replay=False
        )
        after = []
        with (
            patch.object(deepseek_v4_backend, "dsa_use_prefill_cp", lambda fb: False),
            patch.object(full_topk, "decode_scores", _fake_decode_scores),
            patch.object(dense_blocks, "decode_scores", _fake_decode_scores),
            patch.object(full_topk, "topk_page_transform", _fake_topk_page_transform),
        ):
            for layer_id, ratio, source, consumer, has_indexer in LAYERS:
                layer = SimpleNamespace(
                    layer_id=layer_id,
                    compress_ratio=ratio,
                    freqs_cis=None,
                    compressor=object(),
                    indexer=(
                        SimpleNamespace(
                            is_candidate_source=source,
                            uses_candidates=consumer,
                            index_topk=TOPK,
                        )
                        if has_indexer
                        else None
                    ),
                )
                backend.forward_low_ratio_sources(
                    layer=layer,
                    x=self.x,
                    q_lora=self.q_lora,
                    positions=self.pos,
                    forward_batch=forward_batch,
                    **({} if row_slice is None else {"row_slice": row_slice}),
                )
                after.append(buffers[ratio].clone())
        return after, backend


def _rank_slices(rows, world):
    return [
        (*head_a2a_rows(rows, SimpleNamespace(world_size=world, rank_in_group=r)), rows)
        for r in range(world)
    ]


class TestRowSlicedIndexer(CustomTestCase):
    def _check(self, mode, num_requests, rows_per_request, world):
        case = _Case(mode, num_requests, rows_per_request)
        rows = case.rows
        for deep_select in (False, True):
            ref, ref_backend = case.run(deep_select)
            # The whole-batch run fills every row of every top-k it writes.
            for out in ref:
                self.assertFalse((out == UNWRITTEN).any())
            for start, end, num_rows in _rank_slices(rows, world):
                with self.subTest(deep_select=deep_select, start=start, end=end):
                    after, backend = case.run(deep_select, (start, end, num_rows))
                    real = slice(min(start, rows), min(end, rows))
                    others = torch.ones(rows, dtype=torch.bool)
                    others[real] = False
                    for got, want in zip(after, ref):
                        torch.testing.assert_close(got[real], want[real])
                        # The other rows are no rank's here: never written.
                        self.assertTrue((got[others] == UNWRITTEN).all())
                    # Every compressor still sees the whole batch.
                    self.assertEqual(
                        backend.compressed_rows, [(rows, rows, rows)] * len(LAYERS)
                    )
                    # The source published its slice's blocks, which its
                    # consumer read: the same rows as the whole-batch ones.
                    published = backend.forward_metadata.candidate_metadata
                    if real.start == real.stop:
                        self.assertIsNone(published)
                    else:
                        self.assertIsInstance(published, BlockIds)
                        full_blocks = ref_backend.forward_metadata.candidate_metadata
                        torch.testing.assert_close(
                            published.blocks, full_blocks.blocks[real]
                        )

    def test_verify(self):
        # 13 requests x 5 draft rows = 65 rows: the last rank's slice is padded.
        self._check(ForwardMode.TARGET_VERIFY, 13, 5, world=8)
        self._check(ForwardMode.TARGET_VERIFY, 13, 5, world=4)

    def test_verify_all_pad_ranks(self):
        # 5 rows over 8 ranks: ranks 5-7 hold padding only and select nothing.
        self._check(ForwardMode.TARGET_VERIFY, 1, 5, world=8)

    def test_decode(self):
        self._check(ForwardMode.DECODE, 13, 1, world=8)

    def test_whole_batch_only_paths(self):
        case = _Case(ForwardMode.DECODE, 13, 1)
        backend, _ = case.backend(deep_select=False)
        layer = SimpleNamespace(
            layer_id=2,
            compress_ratio=2,
            indexer=SimpleNamespace(is_candidate_source=False, uses_candidates=False),
        )
        # Prefill top-k is never row-sliced.
        with self.assertRaises(AssertionError):
            backend._low_ratio_index_topk(
                layer,
                case.x,
                case.q_lora,
                case.req,
                case.pos,
                SimpleNamespace(forward_mode=ForwardMode.EXTEND),
                row_offset=0,
            )
        # DeepGEMM plans its decode for the whole batch.
        backend.full_topk_indexer.use_deep_gemm_decode = True
        layer.compressor, layer.indexer = None, object()
        with (
            patch.object(deepseek_v4_backend, "dsa_use_prefill_cp", lambda fb: False),
            self.assertRaises(AssertionError),
        ):
            backend.forward_low_ratio_sources(
                layer=layer,
                x=case.x,
                q_lora=case.q_lora,
                positions=case.pos,
                forward_batch=SimpleNamespace(
                    forward_mode=ForwardMode.DECODE, encoder_swa_replay=False
                ),
                row_slice=(0, 2, case.rows),
            )


if __name__ == "__main__":
    unittest.main()
