"""SM90 decode / verify FP4 index scoring and the candidate kernels."""

import unittest
from itertools import product

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_paged
from sglang.kernels.ops.attention.dsv4.sm90_candidate import (
    candidate_block_state,
    candidate_position_counts,
    finalize_candidate_topk,
)
from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer import (
    _e2m1_decode,
    _e2m1_decode_bits,
    _e8m0_decode,
    fp4_index_logits_decode,
)
from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_paged_v2,
)
from sglang.kernels.ops.quantization.fp4_fake_quant import fake_quant_fp4
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=150, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE = 64


@triton.jit
def _decode_codes(OUT, OLD):
    code = tl.arange(0, 256)
    tl.store(OUT + code, _e8m0_decode(code))
    tl.store(OLD + code, tl.exp2(code.to(tl.float32) - 127.0))
    tl.store(OUT + 256 + code, _e2m1_decode(code % 16, True))
    tl.store(OLD + 256 + code, _e2m1_decode(code % 16))
    tl.store(OUT + 512 + code, _e2m1_decode_bits(code.to(tl.uint8) % 16))


def _live(x, lens):
    """Columns past each row's live prefix are unspecified under prefix_only."""
    live = torch.arange(x.shape[1], device=x.device)[None, :] < lens[:, None]
    return x.masked_fill(~live, 0).view(torch.int32)


def _block_lens(lens, width):
    return ((lens + 7) // 8).clamp(0, triton.cdiv(width, 8)).to(torch.int32)


def _random_pool(num_pages, scale_lo, scale_hi, generator=None):
    table = torch.randint(
        0,
        256,
        (num_pages, PAGE * 68),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    table[:, PAGE * 64 :] = torch.randint(
        scale_lo,
        scale_hi,
        (num_pages, PAGE * 4),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    return table


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "the SM90 indexer kernels need Hopper",
)
class TestSm90Fp4Indexer(CustomTestCase):
    def test_all_fp4_and_scale_encodings(self):
        """The exp2-free e8m0 and select-free e2m1 decodes reproduce every code
        bit for bit, flushed scale 0 and overflowing 255 included, except that
        the bit decode yields +0.0 for FP4 negative zero: dots accumulate from
        +0.0, so no logit can tell the two apart."""
        actual = torch.empty(768, device="cuda")
        old = torch.empty(512, device="cuda")
        _decode_codes[(1,)](actual, old)
        torch.testing.assert_close(
            actual[:512].view(torch.int32), old.view(torch.int32)
        )
        torch.testing.assert_close(
            actual[512:].view(torch.int32), (old[256:] + 0.0).view(torch.int32)
        )
        grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        expected = torch.tensor(grid + [-v for v in grid], device="cuda").repeat(16)
        torch.testing.assert_close(actual[256:512], expected, rtol=0, atol=0)

    def test_prefix_logits_are_the_upstream_paged_logits(self):
        """The prefix / bit-decode / grouped-row scoring keeps the rounding
        points and dot shapes of the upstream paged kernel, so the live logits
        are bitwise its own for every request layout and both ratios."""
        torch.manual_seed(3)
        width, num_reqs = 3000, 5
        table = _random_pool(num_reqs * 96, 96, 160)
        context = torch.tensor([3, 9, 1400, 2900, 2994])
        req = torch.arange(num_reqs).repeat_interleave(6)
        pos = context[req] + torch.arange(6).repeat(num_reqs)
        pos[0] = -1
        for ratio, group in product((1, 2), (1, 6)):
            mapping = torch.stack(
                [
                    torch.randperm(num_reqs * 96 * PAGE, device="cuda")[:width]
                    for _ in range(num_reqs)
                ]
            ).int()
            mapping = mapping.repeat_interleave(ratio, 1) * ratio
            lmax = width // ratio
            lens = ((pos + 1) // ratio).clamp(0, lmax).int().cuda()
            q = fake_quant_fp4(torch.randn(30, 32, 128, device="cuda")).bfloat16()
            w = torch.randn(30, 32, device="cuda").bfloat16()
            want = fp4_index_logits_paged(
                q, w, req.cuda(), mapping, lens, table, PAGE, lmax, ratio
            )
            got = fp4_index_logits_decode(
                q,
                w,
                mapping,
                lens,
                table,
                PAGE,
                req_indices=req.cuda(),
                compress_ratio=ratio,
                max_seq_len=lmax,
                bounded_grid=True,
                prefix_only=True,
                rows_per_group=group,
                bits_decode=True,
            )
            with self.subTest(ratio=ratio, group=group):
                self.assertTrue(torch.equal(_live(got, lens), _live(want, lens)))

    def test_grouped_rows_match_per_row_layouts(self):
        """A group holds one request's verify rows, several requests, or a cut run
        (ragged verify, chunk or row-window boundaries); all match per-row."""
        torch.manual_seed(5)
        width, num_reqs = 1500, 5
        table = _random_pool(num_reqs * 48, 96, 160)
        context = torch.tensor([3, 9, 700, 1400, 1494])
        all_req = torch.arange(num_reqs).repeat_interleave(6)
        all_pos = context[all_req] + torch.arange(6).repeat(num_reqs)
        all_pos[0] = -1
        layouts = {
            "aligned": torch.arange(30),
            "shifted": torch.arange(1, 30),
            "cut": torch.arange(5, 27),
            "ragged": torch.tensor([0, 1, 6, 7, 8, 12, 18, 19, 20, 21, 24, 25]),
            "one_per_request": torch.arange(0, 30, 6),
            "interleaved": torch.arange(30).view(num_reqs, 6).t().flatten(),
        }
        for ratio in (1, 2):
            mapping = torch.stack(
                [
                    torch.randperm(num_reqs * 48 * PAGE, device="cuda")[:width]
                    for _ in range(num_reqs)
                ]
            ).int()
            mapping = mapping.repeat_interleave(ratio, 1) * ratio
            for (name, rows), group, (prefix_only, bounded_grid, blocks) in product(
                layouts.items(),
                (6, 4),
                ((True, True, True), (False, True, True), (False, False, False)),
            ):
                req = all_req[rows].cuda()
                lens = ((all_pos[rows] + 1) // ratio).clamp(0, width).cuda()
                q = torch.randn(len(rows), 32, 128, device="cuda").bfloat16()
                weights = torch.randn(len(rows), 32, device="cuda").bfloat16()
                outputs = [
                    fp4_index_logits_decode(
                        q,
                        weights,
                        mapping,
                        lens,
                        table,
                        PAGE,
                        req_indices=req,
                        compress_ratio=ratio,
                        max_seq_len=width,
                        bounded_grid=bounded_grid,
                        prefix_only=prefix_only,
                        candidate_block_size=8,
                        return_block_scores=blocks,
                        rows_per_group=rows_per_group,
                    )
                    for rows_per_group in (1, group)
                ]
                for expected, actual in zip(*(o if blocks else (o,) for o in outputs)):
                    if prefix_only:
                        n = lens if expected.shape[1] == width else (lens + 7) // 8
                        expected, actual = _live(expected, n), _live(actual, n)
                    with self.subTest(ratio=ratio, layout=name, group=group):
                        self.assertTrue(
                            torch.equal(
                                expected.view(torch.int32), actual.view(torch.int32)
                            )
                        )

    def test_source_consumer_graph_replay_with_live_lengths(self):
        """Source logits, fused block maxima, the compact consumer and the slot
        mapping are exact through live-length graph replay, poisoned tails
        included, for one- to three-request verify groups and bit decoding."""
        cases = (
            dict(width=259, bounded_grid=False),
            dict(width=32771, bounded_grid=True),
            dict(width=259, bounded_grid=False, rows_per_group=6, bits_decode=True),
            dict(width=32771, prefix_only=True, rows_per_group=4, bits_decode=True),
            dict(width=262147, prefix_only=True, topk=2048, topk_blocks=2048),
        )
        for case in cases:
            with self.subTest(**case):
                self._check_graph_replay(**case)

    def _check_graph_replay(
        self,
        *,
        width,
        bounded_grid=True,
        prefix_only=False,
        topk=9,
        topk_blocks=7,
        rows_per_group=1,
        bits_decode=False,
    ):
        torch.manual_seed(17)
        rows = 6
        q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)
        req = torch.tensor([2, 2, 0, 0, 0, 1], device="cuda", dtype=torch.int64)
        lens = torch.tensor([259, 130, 0, 1, 8, 17], device="cuda", dtype=torch.int32)
        table = torch.randint(0, 256, (32, PAGE * 68), device="cuda", dtype=torch.uint8)
        table[:, PAGE * 64 :] = 127
        topk = min(topk, width)
        for ratio in (1, 2):
            mapping = torch.stack(
                [torch.randint(0, 2048, (width,), device="cuda") for _ in range(3)]
            ).int()
            mapping = mapping.repeat_interleave(ratio, 1) * ratio
            logical = torch.arange(width, device="cuda")

            def run(mapping=mapping, ratio=ratio):
                # Plans first: a plan must not be the kernel right before its top-k.
                source_plan = plan_topk_v2(lens)
                block_lens = _block_lens(lens, width)
                block_plan = plan_topk_v2(block_lens)
                scores, block_scores = fp4_index_logits_decode(
                    q,
                    weights,
                    mapping,
                    lens,
                    table,
                    PAGE,
                    req_indices=req,
                    compress_ratio=ratio,
                    max_seq_len=width,
                    bounded_grid=bounded_grid or prefix_only,
                    prefix_only=prefix_only,
                    candidate_block_size=8,
                    return_block_scores=True,
                    rows_per_group=rows_per_group,
                    bits_decode=bits_decode,
                )
                source_selected = torch.empty(
                    (rows, topk), device="cuda", dtype=torch.int32
                )
                topk_transform_paged_v2(
                    scores, lens, None, source_selected, 1, source_plan, det_ties=True
                )
                source_pages = torch.empty_like(source_selected)
                source_raw = torch.empty_like(source_selected)
                finalize_candidate_topk(
                    source_selected,
                    scores,
                    lens,
                    mapping,
                    req,
                    source_pages,
                    source_raw,
                    ratio=ratio,
                )
                blocks, counts = candidate_block_state(
                    block_scores,
                    lens,
                    topk_blocks=topk_blocks,
                    block_size=8,
                    block_lens=block_lens,
                    plan=block_plan,
                )
                compact_plan = plan_topk_v2(counts)
                compact = fp4_index_logits_decode(
                    q,
                    weights,
                    mapping,
                    counts,
                    table,
                    PAGE,
                    req_indices=req,
                    compress_ratio=ratio,
                    max_seq_len=width,
                    bounded_grid=bounded_grid and not prefix_only,
                    prefix_only=prefix_only,
                    candidate_blocks=blocks,
                    candidate_block_size=8,
                    bits_decode=bits_decode,
                )
                selected = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
                topk_transform_paged_v2(
                    compact, counts, None, selected, 1, compact_plan, det_ties=True
                )
                pages = torch.empty((rows, 2 * topk), device="cuda", dtype=torch.int32)[
                    :, :topk
                ]
                raw = torch.empty(
                    (rows, 2 * topk + 2), device="cuda", dtype=torch.int32
                )[:, :topk]
                finalize_candidate_topk(
                    selected,
                    compact,
                    counts,
                    mapping,
                    req,
                    pages,
                    raw,
                    ratio=ratio,
                    candidate_blocks=blocks,
                    candidate_block_size=8,
                )
                # Mixed-ratio counts: a consumer seeing a longer context.
                expanded = candidate_position_counts(
                    blocks, counts, lens * 2, block_size=8
                )
                return (
                    scores,
                    block_scores,
                    blocks,
                    counts,
                    compact,
                    pages,
                    raw,
                    expanded,
                    source_pages,
                    source_raw,
                )

            for _ in range(3):
                run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run()
            for live in (
                [width, width - 1, 0, 1, 8, 17],
                [0, 1, 2, 7, 8, 9],
                [width] * rows,
                [0] * rows,
            ):
                lens.copy_(torch.tensor(live, device="cuda"))
                q.normal_()
                weights.normal_()
                req.copy_(req.roll(1))
                explicit = mapping[req[:, None], logical * ratio] // ratio
                if prefix_only:
                    # Length-aware consumers must not select stale or NaN tails.
                    for scratch in (out[0], out[1], out[4]):
                        scratch.fill_(torch.nan)
                graph.replay()
                self._check_replay(
                    out,
                    explicit=explicit,
                    q=q,
                    weights=weights,
                    table=table,
                    lens=lens,
                    req=req,
                    mapping=mapping,
                    ratio=ratio,
                    width=width,
                    topk=topk,
                    topk_blocks=topk_blocks,
                    prefix_only=prefix_only,
                )

    def _check_replay(
        self,
        out,
        *,
        explicit,
        q,
        weights,
        table,
        lens,
        req,
        mapping,
        ratio,
        width,
        topk,
        topk_blocks,
        prefix_only,
    ):
        (
            scores,
            block_scores,
            blocks,
            counts,
            compact,
            pages,
            raw,
            expanded,
            source_pages,
            source_raw,
        ) = out
        expected = fp4_index_logits_decode(
            q, weights, explicit, lens, table, PAGE, skip_invisible=False
        )
        visible = torch.arange(width, device="cuda")[None, :] < lens[:, None]
        torch.testing.assert_close(
            scores.masked_fill(~visible, -torch.inf) if prefix_only else scores,
            expected,
            rtol=0,
            atol=0,
        )
        source_valid = source_raw >= 0
        torch.testing.assert_close(source_valid.sum(-1), lens.clamp(max=topk).long())
        source_values = scores.gather(1, source_raw.long().clamp_min(0)).masked_fill(
            ~source_valid, -torch.inf
        )
        torch.testing.assert_close(
            source_values.sort(-1).values,
            expected.topk(topk).values.sort(-1).values,
            rtol=0,
            atol=0,
        )
        source_expected_pages = (
            mapping[req[:, None], source_raw.long().clamp_min(0) * ratio] // ratio
        ).masked_fill(~source_valid, -1)
        torch.testing.assert_close(source_pages, source_expected_pages.int())
        expected_blocks = (
            F.pad(expected, (0, -width % 8), value=-torch.inf)
            .unflatten(-1, (-1, 8))
            .amax(-1)
        )
        expected_blocks.masked_fill_(
            torch.arange(expected_blocks.shape[1], device="cuda")
            == (lens[:, None] - 1) // 8,
            torch.inf,
        )
        live_blocks = torch.arange(block_scores.shape[1], device="cuda") < (
            (lens[:, None] + 7) // 8
        )
        torch.testing.assert_close(
            block_scores.masked_fill(~live_blocks, -torch.inf)
            if prefix_only
            else block_scores,
            expected_blocks,
            rtol=0,
            atol=0,
        )
        # Equal block maxima may pick different blocks; compare their scores.
        valid_blocks = blocks < block_scores.shape[1]
        selected_scores = block_scores.gather(
            1, blocks.long().clamp_max(block_scores.shape[1] - 1)
        ).masked_fill(~valid_blocks, -torch.inf)
        torch.testing.assert_close(
            selected_scores.sort(-1).values,
            expected_blocks.topk(topk_blocks).values.sort(-1).values,
        )
        positions = (blocks[:, :, None] * 8 + torch.arange(8, device="cuda")).flatten(1)
        valid = torch.arange(positions.shape[1], device="cuda") < counts[:, None]
        expected_compact = expected.gather(
            1, positions.long().clamp_max(width - 1)
        ).masked_fill(~valid, -torch.inf)
        torch.testing.assert_close(
            compact.masked_fill(~valid, -torch.inf) if prefix_only else compact,
            expected_compact,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(expanded, counts)
        visible = raw >= 0
        self.assertTrue(
            torch.all(
                raw.masked_fill(~visible, width).sort(-1).values
                == raw.masked_fill(~visible, width)
            )
        )
        selected_scores = scores.gather(1, raw.long().clamp_min(0)).masked_fill(
            ~visible, -torch.inf
        )
        torch.testing.assert_close(
            selected_scores.sort(-1).values,
            expected_compact.topk(topk).values.sort(-1).values,
        )
        expected_pages = (
            mapping[req[:, None], raw.long().clamp_min(0) * ratio] // ratio
        ).masked_fill(~visible, -1)
        torch.testing.assert_close(pages, expected_pages.int())
        self.assertEqual(scores.stride(0) % 4, 0)
        self.assertEqual(block_scores.stride(0) % 4, 0)

    def test_finalize_short_rows_nan_and_padding(self):
        scores = torch.tensor(
            [[3.0, torch.nan, -torch.inf, torch.inf, 1.0], [0.0] * 5], device="cuda"
        )
        selected = torch.tensor(
            [[1, 4, 3, 0, 2], [-1] * 5], device="cuda", dtype=torch.int32
        )
        lens = torch.tensor([5, 0], device="cuda", dtype=torch.int32)
        mapping = torch.arange(32, device="cuda").reshape(2, 16)
        pages = torch.full((2, 18), 77, device="cuda", dtype=torch.int32)
        raw = torch.full((2, 20), 88, device="cuda", dtype=torch.int32)
        finalize_candidate_topk(
            selected,
            scores,
            lens,
            mapping,
            torch.arange(2, device="cuda"),
            pages[:, :9],
            raw[:, :9],
            ratio=2,
        )
        expected = torch.tensor(
            [[0, 3, 4] + [-1] * 6, [-1] * 9], device="cuda", dtype=torch.int32
        )
        torch.testing.assert_close(pages[:, :9], expected)
        torch.testing.assert_close(raw[:, :9], expected)
        self.assertTrue(torch.all(pages[:, 9:] == 77))
        self.assertTrue(torch.all(raw[:, 9:] == 88))


WGMMA_WIDTH, WGMMA_REQS, WGMMA_ROWS = 1500, 5, 6
# Rows of request r sit at positions WGMMA_CONTEXT[r] + j; row 0 sees nothing.
WGMMA_CONTEXT = torch.tensor([3, 9, 700, 1400, 1494])
WGMMA_LAYOUTS = {
    "aligned": torch.arange(30),
    "shifted": torch.arange(1, 30),
    "cut": torch.arange(5, 27),
    "ragged": torch.tensor([0, 1, 6, 7, 8, 12, 18, 19, 20, 21, 24, 25]),
    "one_per_request": torch.arange(0, 30, 6),
    "interleaved": torch.arange(30).view(WGMMA_REQS, WGMMA_ROWS).t().flatten(),
}


def _wgmma_pool(ratio, seed):
    # Key and fp4 query block scales within 2^4 of each other keep every fp32
    # partial dot exact, so accumulation order cannot change any logit.
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = WGMMA_REQS * WGMMA_WIDTH // PAGE + 1
    table = _random_pool(pages, 124, 129, generator=g)
    mapping = torch.stack(
        [
            torch.randperm(pages * PAGE, device="cuda", generator=g)[:WGMMA_WIDTH]
            for _ in range(WGMMA_REQS)
        ]
    ).int()
    return table, mapping.repeat_interleave(ratio, 1) * ratio


def _wgmma_queries(rows):
    # Head weights spanning 2^48 make every association of the head sum
    # observable; the dot itself stays exact.
    q = fake_quant_fp4(torch.randn(rows, 32, 128, device="cuda")).bfloat16()
    span = torch.randint(-24, 25, (rows, 32), device="cuda").float()
    w = (torch.randn(rows, 32, device="cuda") * torch.exp2(span)).bfloat16()
    return q, w


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "wgmma needs Hopper",
)
class TestSm90IndexerWgmma(CustomTestCase):
    """The wgmma kernels keep the per-row kernel's rounding points and head-sum
    tree; with exact fp32 partial dots their logits are bitwise its own."""

    def _call(self, mapping, table, q, w, req, lens, ratio, *, wgmma, **kw):
        return fp4_index_logits_decode(
            q,
            w,
            mapping,
            lens,
            table,
            PAGE,
            req_indices=req,
            compress_ratio=ratio,
            max_seq_len=WGMMA_WIDTH,
            bounded_grid=kw.pop("bounded_grid", True),
            prefix_only=True,
            candidate_block_size=8,
            use_wgmma=wgmma,
            **kw,
        )

    def test_grouped_rows_match_per_row_kernel(self):
        # One request per group goes to wgmma; cut, ragged and interleaved
        # groups fall back to the Triton grouped kernel.
        torch.manual_seed(0)
        for ratio, (name, rows), blocks in product(
            (1, 2), WGMMA_LAYOUTS.items(), (False, True)
        ):
            table, mapping = _wgmma_pool(ratio, seed=ratio)
            req = torch.arange(WGMMA_REQS).repeat_interleave(WGMMA_ROWS)[rows].cuda()
            pos = WGMMA_CONTEXT.repeat_interleave(WGMMA_ROWS) + torch.arange(
                WGMMA_ROWS
            ).repeat(WGMMA_REQS)
            pos[0] = -1
            lens = ((pos[rows] + 1) // ratio).clamp(0, WGMMA_WIDTH).cuda()
            q, w = _wgmma_queries(len(rows))
            expected, actual = (
                self._call(
                    mapping,
                    table,
                    q,
                    w,
                    req,
                    lens,
                    ratio,
                    wgmma=wgmma,
                    rows_per_group=1 if not wgmma else WGMMA_ROWS,
                    return_block_scores=blocks,
                )
                for wgmma in (False, True)
            )
            if not blocks:
                expected, actual = (expected,), (actual,)
            with self.subTest(ratio=ratio, layout=name, blocks=blocks):
                for j, (e, a) in enumerate(zip(expected, actual)):
                    n = lens if j == 0 else (lens + 7) // 8
                    self.assertTrue(torch.equal(_live(e, n), _live(a, n)))

    def test_candidate_consumers_match_per_row_kernel(self):
        g = torch.Generator(device="cuda").manual_seed(7)
        for ratio in (1, 2):
            table, mapping = _wgmma_pool(ratio, seed=10 + ratio)
            rows = 12
            req = torch.arange(rows, device="cuda") % WGMMA_REQS
            visible = WGMMA_WIDTH // ratio
            blocks = torch.stack(
                [
                    torch.randperm(visible // 8, device="cuda", generator=g)[:48]
                    .sort()
                    .values
                    for _ in range(rows)
                ]
            ).int()
            counts = torch.tensor(
                [384, 383, 0, 1, 8, 9, 63, 64, 65, 200, 384, 129], device="cuda"
            )
            q, w = _wgmma_queries(rows)
            expected, actual = (
                self._call(
                    mapping,
                    table,
                    q,
                    w,
                    req,
                    counts,
                    ratio,
                    wgmma=wgmma,
                    bounded_grid=False,
                    candidate_blocks=blocks,
                )
                for wgmma in (False, True)
            )
            with self.subTest(ratio=ratio):
                self.assertTrue(
                    torch.equal(_live(expected, counts), _live(actual, counts))
                )

    def test_graph_replay_with_live_lengths_and_requests(self):
        ratio = 1
        table, mapping = _wgmma_pool(ratio, seed=21)
        req = torch.arange(WGMMA_REQS, device="cuda").repeat_interleave(WGMMA_ROWS)
        lens = torch.zeros(30, device="cuda", dtype=torch.int64)
        q, w = _wgmma_queries(30)
        blocks = torch.arange(48, device="cuda").expand(30, -1).contiguous().int()
        counts = torch.zeros(30, device="cuda", dtype=torch.int64)

        def run():
            source = self._call(
                mapping,
                table,
                q,
                w,
                req,
                lens,
                ratio,
                wgmma=True,
                rows_per_group=WGMMA_ROWS,
                return_block_scores=True,
            )
            consumer = self._call(
                mapping,
                table,
                q,
                w,
                req,
                counts,
                ratio,
                wgmma=True,
                bounded_grid=False,
                candidate_blocks=blocks,
            )
            return source, consumer

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            (scores, block_scores), compact = run()
        for step, live in enumerate(([1499] * 30, [0, 1, 2, 63, 64, 65] * 5)):
            lens.copy_(torch.tensor(live, device="cuda"))
            counts.copy_((lens // 4).clamp(max=384))
            new_q, new_w = _wgmma_queries(30)
            q.copy_(new_q)
            w.copy_(new_w)
            req.copy_(req.roll(6 * (step + 1)))
            for poisoned in (scores, block_scores, compact):
                poisoned.fill_(torch.nan)
            graph.replay()
            expected_scores, expected_blocks = self._call(
                mapping,
                table,
                q,
                w,
                req,
                lens,
                ratio,
                wgmma=False,
                return_block_scores=True,
            )
            expected_compact = self._call(
                mapping,
                table,
                q,
                w,
                req,
                counts,
                ratio,
                wgmma=False,
                bounded_grid=False,
                candidate_blocks=blocks,
            )
            nb = (lens + 7) // 8
            with self.subTest(step=step):
                self.assertTrue(
                    torch.equal(_live(expected_scores, lens), _live(scores, lens))
                )
                self.assertTrue(
                    torch.equal(_live(expected_blocks, nb), _live(block_scores, nb))
                )
                self.assertTrue(
                    torch.equal(_live(expected_compact, counts), _live(compact, counts))
                )


if __name__ == "__main__":
    unittest.main()
