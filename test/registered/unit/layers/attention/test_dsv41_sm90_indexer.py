"""The SM90 V4.1 indexer backends behind dsv41_sm90_on("indexer")."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    select_candidate_block_ids,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_paged
from sglang.kernels.ops.attention.dsv4.sm90_fp8_prefill import (
    fp8_index_logits_prefill,
    quantize_bf16_index_queries_fp8,
    unpack_fp4_index_keys_to_fp8,
)
from sglang.kernels.ops.attention.dsv4.topk import topk_v2_plan_is_written
from sglang.kernels.ops.kvcache.dsv4_fp4_index_k_cache import store_fp4_index_k_cache
from sglang.kernels.ops.quantization.fp4_fake_quant import fake_quant_fp4
from sglang.srt.layers.attention.dsv4.v41_indexer import sm90_decode, sm90_prefill
from sglang.srt.layers.attention.dsv4.v41_indexer.sm90 import (
    Sm90DenseBlocksBackend,
    Sm90FullTopKIndexer,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.sm90_decode import (
    Sm90DecodeCandidates,
)
from sglang.srt.layers.attention.dsv4.v41_indexer.types import (
    DecodeInputs,
    PrefillInputs,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

PAGE = 64
TOPK = 512
BLOCK = 8


class _Indexer:
    """Projections stubbed out: q_lora carries the RoPE'd queries, x the weights."""

    def __init__(self, index_topk=TOPK):
        self.index_topk = index_topk

    def queries(self, q_lora, freqs):
        return q_lora

    def head_weights(self, x):
        return x


class _Pool:
    def __init__(self, table):
        self.table = table

    def get_index_k_with_scale_buffer(self, layer_id):
        return self.table


def _lowest_position_topk(scores, k):
    """Positions of the k largest finite scores, ties to the lowest position."""
    order = torch.argsort(scores.cpu(), descending=True, stable=True)
    picked = order[:k]
    picked = picked[scores.cpu()[picked] > -torch.inf]
    return sorted(picked.tolist())


def _selection(raw_row):
    values = raw_row.tolist()
    kept = [v for v in values if v >= 0]
    # Sorted ascending, every hole after the picks.
    assert values == kept + [-1] * (len(values) - len(kept))
    return kept


class _Batch:
    """Four requests; the third has no row on this (CP) rank. Most keys are
    zero, so the k-th score is an exact 0.0 tie that crosses the last slot."""

    def __init__(self, ratio):
        torch.manual_seed(ratio)
        self.ratio = ratio
        # Against 48 candidate blocks of 8, the contexts take every consumer
        # path: all blocks, dropped in place, gathered.
        self.seq_lens = [3 * 1024 + 5, 700, 900, 2 * 1024 + 9]
        self.rows_per_request = [37, 5, 0, 21]
        lcs = [s // ratio for s in self.seq_lens]
        capacity = (sum(lcs) + PAGE - 1) // PAGE * PAGE
        keys = fake_quant_fp4(torch.randn(capacity, 128, device="cuda").bfloat16())
        keys[torch.rand(capacity, device="cuda") < 0.85] = 0
        self.table = torch.zeros(
            capacity // PAGE, PAGE * 68, dtype=torch.uint8, device="cuda"
        )
        store_fp4_index_k_cache(
            input=keys,
            cache=self.table,
            loc=torch.arange(capacity, device="cuda", dtype=torch.int32),
            page_size=PAGE,
            rne=True,
        )
        self.k_slots = torch.randperm(capacity, device="cuda")[: sum(lcs)]
        self.starts = [sum(lcs[:i]) for i in range(len(lcs))]
        self.lcs = lcs
        # A rank's interleaved rows: any positions up to the newest token.
        positions = []
        for seq_len, rows in zip(self.seq_lens, self.rows_per_request):
            positions += sorted(torch.randperm(seq_len)[:rows].tolist())
            if rows:
                positions[-1] = seq_len - 1
        self.positions = torch.tensor(positions, device="cuda")
        rows = len(positions)
        self.q = fake_quant_fp4(torch.randn(rows, 32, 128, device="cuda").bfloat16())
        # Positive weights keep every score >= 0: zero keys tie at exactly 0.0.
        self.weights = torch.rand(rows, 32, device="cuda").bfloat16()

    def inputs(self, *, with_pages=True):
        rows = self.positions.shape[0]
        return PrefillInputs(
            indexer=_Indexer(),
            layer_id=0,
            compress_ratio=self.ratio,
            freqs_cis=torch.zeros(max(self.seq_lens), device="cuda"),
            x=self.weights,
            q_lora=self.q,
            positions=self.positions,
            req_rows=None,
            req_pool_indices=torch.arange(4, device="cuda"),
            k_slots=self.k_slots,
            index_plans=None,
            kv_page_table=torch.zeros(1, 1, device="cuda"),
            seq_lens_cpu=self.seq_lens,
            rows_per_request=self.rows_per_request,
            rows_per_request_device=None,
            out_raw_indices=torch.full(
                (rows + 2, TOPK), 5, dtype=torch.int32, device="cuda"
            ),
            out_page_indices=(
                torch.full((rows + 2, TOPK), 5, dtype=torch.int32, device="cuda")
                if with_pages
                else None
            ),
        )

    def request_scores(self):
        """(request, row, scores [lc], slots [lc]) of every query row."""
        row = 0
        for r, rows in enumerate(self.rows_per_request):
            lc, start = self.lcs[r], self.starts[r]
            slots = self.k_slots[start : start + lc]
            keys = unpack_fp4_index_keys_to_fp8(slots, self.table, PAGE)
            for i in range(row, row + rows):
                lens = ((self.positions[i : i + 1] + 1) // self.ratio).int()
                scores = fp8_index_logits_prefill(
                    quantize_bf16_index_queries_fp8(self.q[i : i + 1]),
                    self.weights[i : i + 1],
                    keys,
                    lens,
                )[0]
                yield r, i, scores, slots
            row += rows


def _backend(cls, table, req_to_token=None, topk_blocks=48):
    kwargs = dict(token_to_kv_pool=_Pool(table), req_to_token=req_to_token)
    if cls is Sm90FullTopKIndexer:
        return cls(**kwargs, use_deep_gemm_prefill=False, use_deep_gemm_decode=False)
    return cls(
        **kwargs,
        candidate_topk_blocks=topk_blocks,
        candidate_block_size=BLOCK,
        use_deep_gemm_prefill=False,
    )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "the SM90 indexer needs Hopper",
)
class TestSm90PrefillIndexer(CustomTestCase):
    def _check_rows(self, batch, inputs, expected_of):
        raw, pages = inputs.out_raw_indices, inputs.out_page_indices
        for r, i, scores, slots in batch.request_scores():
            expected = expected_of(r, i, scores)
            with self.subTest(request=r, row=i):
                self.assertEqual(_selection(raw[i]), expected)
                if pages is not None:
                    holes = [-1] * (TOPK - len(expected))
                    self.assertEqual(
                        pages[i].tolist(), slots[expected].tolist() + holes
                    )
        # Rows past the requests' are not query rows: reset, never written.
        self.assertTrue((raw[batch.positions.shape[0] :] == -1).all())

    def test_full_topk_is_exact_with_lowest_position_ties(self):
        """Any CP row layout, any chunking of a request's rows, with or without
        page indices: the selection is the exact top-k of the FP8 scores, ties
        to the lowest positions, sorted and -1 padded."""
        for ratio, budget, with_pages in ((1, None, True), (2, 4096 * 16, False)):
            batch = _Batch(ratio)
            inputs = batch.inputs(with_pages=with_pages)
            backend = _backend(Sm90FullTopKIndexer, batch.table)
            budget = budget or sm90_prefill._SCORE_BUDGET_BYTES
            with patch.object(sm90_prefill, "_SCORE_BUDGET_BYTES", budget):
                backend.topk_prefill(inputs)
            self._check_rows(
                batch, inputs, lambda r, i, s: _lowest_position_topk(s, TOPK)
            )

    def test_candidate_source_and_consumer(self):
        """The source publishes the block ids select_candidate_block_ids keeps for
        its scores and takes the full top-k; a consumer takes the exact top-k
        among the published blocks only, holes where fewer remain."""
        batch = _Batch(1)
        backend = _backend(Sm90DenseBlocksBackend, batch.table)
        source = batch.inputs()
        with patch.object(sm90_prefill, "_SCORE_BUDGET_BYTES", 4096 * 40):
            published = backend.publish_prefill(source)
        self._check_rows(batch, source, lambda r, i, s: _lowest_position_topk(s, TOPK))
        candidates = {}
        for r, i, scores, _ in batch.request_scores():
            lens = (batch.positions[i] + 1) // batch.ratio
            want = select_candidate_block_ids(
                scores[None], lens.view(1, 1), topk_blocks=48, block_size=BLOCK
            )[0]
            got = published.blocks[i]
            self.assertEqual(set(got.tolist()) - {-1}, set(want.tolist()) - {-1})
            candidates[i] = set(got.tolist()) - {-1}

        def among_blocks(r, i, scores):
            keep = torch.zeros_like(scores, dtype=torch.bool)
            for b in candidates[i]:
                keep[b * BLOCK : (b + 1) * BLOCK] = True
            return _lowest_position_topk(scores.masked_fill(~keep, -torch.inf), TOPK)

        consumer = batch.inputs()
        backend.consume_prefill(consumer, published)
        self._check_rows(batch, consumer, among_blocks)


DECODE_TOPK = 64
DECODE_BLOCKS = 8  # compact from 16 windows of 8 blocks of 8: lmax >= 1024


class _DecodeBatch:
    """Requests (five by default) at graph capacity ``lmax`` compressed positions,
    ``group`` rows each (DSpark verify, or 1 for decode). With most keys zero,
    exact 0.0 ties fill the last slots of the longer rows; block scales within
    2^5 keep every fp32 partial dot exact, which the wgmma kernels need."""

    def __init__(self, *, ratio, lmax, group, seed=0, zero_keys=0.97, context=None):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.ratio, self.lmax, self.group = ratio, lmax, group
        num_reqs = 5 if context is None else len(context)
        num_slots = num_reqs * lmax
        self.table = torch.randint(
            0,
            256,
            (num_slots // PAGE, PAGE * 68),
            dtype=torch.uint8,
            device="cuda",
            generator=g,
        )
        self.table[:, PAGE * 64 :] = torch.randint(
            124,
            129,
            (num_slots // PAGE, PAGE * 4),
            dtype=torch.uint8,
            device="cuda",
            generator=g,
        )
        payload = self.table[:, : PAGE * 64].view(num_slots // PAGE, PAGE, 64)
        zero = (
            torch.rand(num_slots // PAGE, PAGE, device="cuda", generator=g) < zero_keys
        )
        payload.masked_fill_(zero[..., None], 0)
        slots = torch.stack(
            [
                torch.randperm(num_slots, device="cuda", generator=g)[:lmax]
                for _ in range(num_reqs)
            ]
        )
        offsets = torch.arange(ratio, device="cuda")
        self.req_to_token = (
            (slots[:, :, None] * ratio + offsets).flatten(1).to(torch.int32)
        )
        self.req = torch.arange(num_reqs, device="cuda").repeat_interleave(group)
        rows = self.req.shape[0]
        if context is None:
            context = [
                0,
                9,
                lmax * ratio // 3,
                lmax * ratio * 2 // 3,
                lmax * ratio - group,
            ]
        num_reqs = len(context)
        context = torch.tensor(context)
        positions = context.cuda()[self.req] + torch.arange(
            group, device="cuda"
        ).repeat(num_reqs)
        self.positions = positions.clamp_max(lmax * ratio - 1)
        self.q = fake_quant_fp4(
            torch.randn(rows, 32, 128, device="cuda", generator=g)
        ).bfloat16()
        self.weights = torch.rand(rows, 32, device="cuda", generator=g).bfloat16()

    def inputs(self, topk=DECODE_TOPK):
        rows = self.req.shape[0]
        return DecodeInputs(
            indexer=_Indexer(topk),
            layer_id=0,
            compress_ratio=self.ratio,
            freqs_cis=torch.zeros(self.lmax * self.ratio, device="cuda"),
            x=self.weights,
            q_lora=self.q,
            positions=self.positions,
            req_rows=self.req,
            paged_metadata=SimpleNamespace(
                max_compressed_seq_len=self.lmax,
                use_topk_v2=True,
                rows_per_request=self.group,
            ),
            is_verify=self.group > 1,
            out_page_indices=torch.full(
                (rows + 3, topk), 5, dtype=torch.int32, device="cuda"
            ),
        )

    def logits(self):
        """The upstream paged scores (the switch-off path) and visible lengths."""
        lens = ((self.positions + 1) // self.ratio).to(torch.int32)
        scores = fp4_index_logits_paged(
            self.q,
            self.weights,
            self.req,
            self.req_to_token,
            lens,
            self.table,
            PAGE,
            self.lmax,
            self.ratio,
        )
        return scores, lens

    def slots(self, row, positions):
        columns = torch.tensor(positions, dtype=torch.int64, device="cuda") * self.ratio
        return (self.req_to_token[self.req[row], columns] // self.ratio).tolist()


def _expected_blocks(scores, length):
    """The source's blocks: top DECODE_BLOCKS block maxima, the newest +inf,
    ties to the lowest block, as (ascending block ids, positions they hold)."""
    width = -(-scores.shape[0] // BLOCK) * BLOCK
    padded = torch.full((width,), -torch.inf, device=scores.device)
    padded[:length] = scores[:length]
    maxima = padded.view(-1, BLOCK).amax(-1)
    if length:
        maxima[(length - 1) // BLOCK] = torch.inf
    blocks = _lowest_position_topk(maxima, DECODE_BLOCKS)
    count = sum(min(BLOCK, max(0, length - b * BLOCK)) for b in blocks)
    return blocks, count


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "the SM90 indexer needs Hopper",
)
class TestSm90DecodeIndexer(CustomTestCase):
    def _check_pages(self, batch, out, expected_positions):
        rows = batch.req.shape[0]
        for row in range(rows):
            positions = expected_positions(row)
            want = batch.slots(row, positions) + [-1] * (out.shape[1] - len(positions))
            with self.subTest(row=row):
                self.assertEqual(out[row].tolist(), want)
        self.assertTrue((out[rows:] == -1).all())

    def _full_topk(self, batch, scores, lens):
        return lambda row: _lowest_position_topk(
            scores[row, : int(lens[row])], DECODE_TOPK
        )

    def test_full_topk_is_exact_for_decode_verify_and_chunks(self):
        """Decode and grouped verify rows, whole or split by the score budget
        (on and off group boundaries), both ratios: the selection is the exact
        top-k of the upstream logits, ties to the lowest position."""
        for ratio, group, rows_per_chunk in (
            (1, 1, None),
            (2, 6, None),
            (1, 6, 12),
            (1, 6, 4),
        ):
            batch = _DecodeBatch(ratio=ratio, lmax=2048 // ratio, group=group)
            inputs = batch.inputs()
            backend = _backend(Sm90FullTopKIndexer, batch.table, batch.req_to_token)
            budget = sm90_decode._SCORE_BUDGET_BYTES
            if rows_per_chunk is not None:
                budget = rows_per_chunk * batch.lmax * 8
            with self.subTest(ratio=ratio, group=group, rows_per_chunk=rows_per_chunk):
                with patch.object(sm90_decode, "_SCORE_BUDGET_BYTES", budget):
                    backend.topk_decode(inputs)
                scores, lens = batch.logits()
                self._check_pages(
                    batch, inputs.out_page_indices, self._full_topk(batch, scores, lens)
                )

    def test_compact_source_and_consumer_through_graph_replay(self):
        """A compact source publishes its best blocks (fused block maxima, the
        newest block forced, ties to the lowest block) and takes the full top-k;
        the consumer takes the exact top-k among those blocks. Captured once and
        replayed with new positions, requests and queries."""
        batch = _DecodeBatch(ratio=1, lmax=2048, group=6, seed=1)
        backend = _backend(
            Sm90DenseBlocksBackend, batch.table, batch.req_to_token, DECODE_BLOCKS
        )
        source, consumer = batch.inputs(), batch.inputs()

        def run():
            published = backend.publish_decode(source)
            backend.consume_decode(consumer, published)
            return published

        published = run()
        self.assertIsInstance(published, Sm90DecodeCandidates)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            published = run()
        for step in range(3):
            if step:
                batch.positions.copy_(
                    (batch.positions * 7 + 131 * step) % (batch.lmax - 1)
                )
                batch.req.copy_(batch.req.roll(6))
                batch.q.copy_(
                    fake_quant_fp4(torch.randn_like(batch.q.float())).bfloat16()
                )
            for out in (source.out_page_indices, consumer.out_page_indices):
                out.fill_(7)
            graph.replay()
            scores, lens = batch.logits()
            with self.subTest(step=step):
                self._check_pages(
                    batch, source.out_page_indices, self._full_topk(batch, scores, lens)
                )
                published_blocks = {}
                for row in range(batch.req.shape[0]):
                    length = int(lens[row])
                    blocks, count = _expected_blocks(scores[row], length)
                    published_blocks[row] = blocks
                    got = published.blocks[row].tolist()
                    self.assertEqual(got[: len(blocks)], blocks)
                    self.assertEqual(int(published.counts[row]), count)

                def among_blocks(row):
                    keep = torch.zeros(batch.lmax, dtype=torch.bool, device="cuda")
                    for b in published_blocks[row]:
                        keep[b * BLOCK : (b + 1) * BLOCK] = True
                    keep[int(lens[row]) :] = False
                    return _lowest_position_topk(
                        scores[row].masked_fill(~keep, -torch.inf), DECODE_TOPK
                    )

                self._check_pages(batch, consumer.out_page_indices, among_blocks)

    def test_planned_long_rows_take_the_cluster_top_k(self):
        """Batches large enough for a written top-k v2 plan with rows long enough
        for its cluster path: plans built ahead of the scores (a plan right before
        its top-k is read stale), the source's own top-k and the consumer's
        shared plan all give the exact selection."""
        context = [70000 + 271 * r for r in range(40)]
        batch = _DecodeBatch(
            ratio=1, lmax=81920, group=6, seed=3, zero_keys=0.0, context=context
        )
        scores, lens = batch.logits()
        self.assertTrue(topk_v2_plan_is_written(lens))
        full = _backend(Sm90FullTopKIndexer, batch.table, batch.req_to_token)
        dense = _backend(
            Sm90DenseBlocksBackend, batch.table, batch.req_to_token, DECODE_BLOCKS
        )
        inputs, source, consumer = batch.inputs(), batch.inputs(), batch.inputs()
        full.topk_decode(inputs)
        published = dense.publish_decode(source)
        dense.consume_decode(consumer, published)
        expected = self._full_topk(batch, scores, lens)
        self._check_pages(batch, inputs.out_page_indices, expected)
        self._check_pages(batch, source.out_page_indices, expected)

        def among_blocks(row):
            keep = torch.zeros(batch.lmax, dtype=torch.bool, device="cuda")
            for b in published.blocks[row, :DECODE_BLOCKS].tolist():
                keep[b * BLOCK : (b + 1) * BLOCK] = True
            keep[int(lens[row]) :] = False
            return _lowest_position_topk(
                scores[row].masked_fill(~keep, -torch.inf), DECODE_TOPK
            )

        self._check_pages(batch, consumer.out_page_indices, among_blocks)

    def test_small_capacity_source_keeps_the_upstream_scheme(self):
        """Below the compact capacity the source publishes the upstream block ids
        and the consumer consumes them on the upstream path (no exact ties here:
        that path breaks them by arrival order)."""
        batch = _DecodeBatch(ratio=1, lmax=512, group=1, seed=2, zero_keys=0.0)
        backend = _backend(
            Sm90DenseBlocksBackend, batch.table, batch.req_to_token, DECODE_BLOCKS
        )
        source, consumer = batch.inputs(), batch.inputs()
        published = backend.publish_decode(source)
        self.assertNotIsInstance(published, Sm90DecodeCandidates)
        backend.consume_decode(consumer, published)
        scores, lens = batch.logits()
        self._check_pages(
            batch, source.out_page_indices, self._full_topk(batch, scores, lens)
        )

        def among_blocks(row):
            keep = torch.zeros(batch.lmax, dtype=torch.bool, device="cuda")
            for b in set(published.blocks[row].tolist()) - {-1}:
                keep[b * BLOCK : (b + 1) * BLOCK] = True
            keep[int(lens[row]) :] = False
            return _lowest_position_topk(
                scores[row].masked_fill(~keep, -torch.inf), DECODE_TOPK
            )

        self._check_pages(batch, consumer.out_page_indices, among_blocks)


if __name__ == "__main__":
    unittest.main()
