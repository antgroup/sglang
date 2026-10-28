"""The DeepSeek-V4.1 SM90 decode attention (switch "decode") against the legacy
FlashMLA decode it schedules: the H20 tile scheduler reproduces FlashMLA's own
schedule with the stock overhead, its other schedules move outputs only by the
split-KV combine's rounding, and the packed DSpark draft rows match the padded
per-row attention."""

import random
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
D, PAGE, EXTRA_PAGE = 512, 128, 64
# Any other schedule only reorders the split-KV combine: measured on H20 at most
# one bf16 ulp of a row's largest output; two leave headroom.
ROW_ULPS = 2.0**-6


def _cache(num_pages, page_size, seed):
    """A V4 (584-byte) paged cache of normed, roped, quantized random latents."""
    from sglang.kernels.ops.attention.dsv4.norm_rope import (
        fused_k_norm_rope_flashmla,
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    page_bytes = -(-584 * page_size // 576) * 576
    cache = torch.zeros(num_pages, page_bytes, dtype=torch.uint8, device="cuda")
    tokens = num_pages * page_size
    kv = (torch.randn(tokens, D, device="cuda", generator=g) * 2).bfloat16()
    weight = (torch.rand(D, device="cuda", generator=g) + 0.5).bfloat16()
    angles = torch.randn(8192, 32, device="cuda", generator=g)
    freqs = torch.polar(torch.ones_like(angles), angles)
    pos = torch.randint(0, 8192, (tokens,), device="cuda", generator=g)
    loc = torch.arange(tokens, device="cuda", dtype=torch.int32)
    fused_k_norm_rope_flashmla(kv, weight, 1e-6, freqs, pos, loc, cache, page_size)
    return cache[:, : page_size * 584].view(num_pages, page_size, 1, 584), tokens


def _lengths(b, topk, mode, g):
    if mode == "full":
        return torch.full((b,), topk, device="cuda", dtype=torch.int32)
    return torch.randint(
        1, topk + 1, (b,), device="cuda", dtype=torch.int32, generator=g
    )


def _ours(q, indices, lengths, extra, overheads, per_part=True):
    from sglang.kernels.ops.attention.dsv4.flashmla_sparse_sched import (
        flashmla_sparse_sched_meta,
    )

    parts = torch.cuda.get_device_properties(0).multi_processor_count
    return flashmla_sparse_sched_meta(
        topk_length=lengths,
        extra_topk_length=None if extra is None else extra[1],
        topk=indices.shape[-1],
        extra_topk=0 if extra is None else extra[0].shape[-1],
        num_sm_parts=max(parts // (q.shape[2] // 64), 1),
        overheads=overheads,
        per_part=per_part,
    )


def _within_row_ulps(test, got, ref):
    got, ref = got.float(), ref.float()
    rowmax = ref.abs().amax(dim=-1, keepdim=True)
    test.assertTrue(torch.isfinite(got).all())
    test.assertLessEqual(((got - ref).abs() - ROW_ULPS * rowmax).max().item(), 0.0)


@unittest.skipUnless(SM90, "the legacy FlashMLA sparse decode is the SM90 build")
class TestFlashMLASm90Schedule(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.swa, cls.swa_tokens = _cache(64, PAGE, 1)
        cls.extra, cls.extra_tokens = _cache(128, EXTRA_PAGE, 2)

    def _inputs(self, b, h_q, topk, extra_topk, mode, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        q = torch.randn(b, 1, h_q, D, device="cuda", generator=g).bfloat16()
        indices = torch.randint(
            0, self.swa_tokens, (b, 1, topk), device="cuda", generator=g
        ).int()
        sink = torch.randn(h_q, device="cuda", generator=g)
        extra = None
        if extra_topk:
            extra_indices = torch.randint(
                0, self.extra_tokens, (b, 1, extra_topk), device="cuda", generator=g
            ).int()
            extra = (extra_indices, _lengths(b, extra_topk, mode, g))
        return q, indices, _lengths(b, topk, mode, g), sink, extra

    def _attn(self, q, indices, lengths, sink, extra, meta=None):
        from sglang.kernels.ops.attention import flash_mla_interface

        sched = flash_mla_interface.FlashMLASchedMeta()
        if meta is not None:
            sched.tile_scheduler_metadata, sched.num_splits = meta
        kwargs = {}
        if extra is not None:
            kwargs = dict(
                extra_k_cache=self.extra,
                extra_indices_in_kvcache=extra[0],
                extra_topk_length=extra[1],
            )
        out, _ = flash_mla_interface.flash_mla_with_kvcache(
            q=q,
            k_cache=self.swa,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=D,
            tile_scheduler_metadata=sched,
            softmax_scale=D**-0.5,
            is_fp8_kvcache=True,
            indices=indices,
            topk_length=lengths,
            attn_sink=sink,
            **kwargs,
        )
        return out, sched

    def _cases(self):
        for h_q in (64, 128):
            for b in (1, 6, 48, 320):
                for topk, extra_topk in ((128, 0), (128, 512), (512, 1024)):
                    for mode in ("full", "rand"):
                        yield h_q, b, topk, extra_topk, mode

    def test_stock_overhead_is_flashmla_schedule(self):
        """Overhead 5 alone, on num_sms / (h_q / 64) partitions, is the schedule
        sgl-flash-mla's legacy decode computes for itself, and handing it over
        leaves the output bitwise; the pad word of each entry is never written."""
        for h_q, b, topk, extra_topk, mode in self._cases():
            with self.subTest(h_q=h_q, b=b, topk=topk, extra=extra_topk, mode=mode):
                q, indices, lengths, sink, extra = self._inputs(
                    b, h_q, topk, extra_topk, mode, b + h_q + topk + extra_topk
                )
                ref, stock = self._attn(q, indices, lengths, sink, extra)
                meta = _ours(q, indices, lengths, extra, (5,))
                self.assertEqual(stock.tile_scheduler_metadata.shape, meta[0].shape)
                self.assertTrue(
                    torch.equal(stock.tile_scheduler_metadata[:, :7], meta[0][:, :7])
                )
                self.assertTrue(torch.equal(stock.num_splits, meta[1]))
                out, sched = self._attn(q, indices, lengths, sink, extra, meta)
                self.assertEqual(
                    sched.tile_scheduler_metadata.data_ptr(), meta[0].data_ptr()
                )
                self.assertTrue(
                    torch.equal(out.view(torch.int16), ref.view(torch.int16))
                )

    def test_cost_model_schedule_moves_only_the_combine(self):
        """The cost model's split pieces cover every block once: the output stays
        within the combine's rounding of the stock schedule's."""
        changed = 0
        for h_q, b, topk, extra_topk, mode in self._cases():
            with self.subTest(h_q=h_q, b=b, topk=topk, extra=extra_topk, mode=mode):
                q, indices, lengths, sink, extra = self._inputs(
                    b, h_q, topk, extra_topk, mode, 7 * b + h_q + topk
                )
                ref, stock = self._attn(q, indices, lengths, sink, extra)
                for overheads in ((5, 3, 2, 1), (0,)):
                    meta = _ours(q, indices, lengths, extra, overheads)
                    changed += not torch.equal(
                        meta[0][:, :7], stock.tile_scheduler_metadata[:, :7]
                    )
                    out, _ = self._attn(q, indices, lengths, sink, extra, meta)
                    _within_row_ulps(self, out, ref)
        self.assertGreater(changed, 0)

    def test_per_part_walk_matches_per_request(self):
        """The greedy walked one partition per step (table and ballot forms) writes
        the per-request walk's metadata and num_splits for every candidate set."""
        rng = random.Random(5)
        torch.manual_seed(5)
        from sglang.kernels.ops.attention.dsv4.flashmla_sparse_sched import (
            flashmla_sparse_sched_meta,
        )

        # b = 1024 x 2048 extra topk exceeds the table budget: ballot form.
        cases = [(192, 512, 78, "full"), (160, 0, 78, "full"), (1024, 2048, 78, "full")]
        for _ in range(100):
            cases.append(
                (
                    rng.choice([1, 6, 33, 48, 192, 384, 777]),
                    rng.choice([0, 512, 1024]),
                    rng.choice([78, 39, 7, 1]),
                    rng.choice(["full", "rand", "zero"]),
                )
            )
        for b, extra_topk, parts, mode in cases:
            for overheads in ((5, 3, 2, 1), (5, 3, 1, 0), (0,)):
                with self.subTest(b=b, extra=extra_topk, parts=parts, mode=mode):
                    hi = {"full": None, "rand": 1, "zero": 0}[mode]
                    if hi is None:
                        topk_len = torch.full((b,), 128, dtype=torch.int32)
                        extra_len = torch.full((b,), extra_topk, dtype=torch.int32)
                    else:
                        topk_len = torch.randint(0, 128 * hi + 1, (b,)).int()
                        extra_len = torch.randint(0, extra_topk * hi + 1, (b,)).int()
                    kwargs = dict(
                        topk_length=topk_len.cuda(),
                        extra_topk_length=extra_len.cuda() if extra_topk else None,
                        topk=128,
                        extra_topk=extra_topk,
                        num_sm_parts=parts,
                        overheads=overheads,
                    )
                    ref_meta, ref_splits = flashmla_sparse_sched_meta(**kwargs)
                    meta, splits = flashmla_sparse_sched_meta(**kwargs, per_part=True)
                    self.assertTrue(torch.equal(meta, ref_meta))
                    self.assertTrue(torch.equal(splits, ref_splits))

    def test_packed_draft_rows_match_padded_rows(self):
        """A request's draft rows packed head-wise into one FlashMLA row attend as
        the rows padded to 64 heads each do; a new forward's indices replace the
        cached per-request tables instead of reusing them."""
        from functools import partial

        from sglang.kernels.ops.attention import flash_mla_interface
        from sglang.srt.layers.attention.dsv4.sm90_decode import (
            forward_packed_draft,
            packed_draft_rows_per_req,
        )

        rows_per_req, heads = 5, 8
        self.assertEqual(
            packed_draft_rows_per_req(rows_per_req=rows_per_req, num_heads=heads), 5
        )
        self.assertEqual(packed_draft_rows_per_req(rows_per_req=5, num_heads=64), 0)
        self.assertEqual(packed_draft_rows_per_req(rows_per_req=9, num_heads=16), 0)
        mla = partial(flash_mla_interface.flash_mla_with_kvcache, use_legacy=True)
        for reqs, tuned in ((1, False), (6, True), (64, True), (64, False)):
            with self.subTest(reqs=reqs, tuned=tuned):
                g = torch.Generator(device="cuda").manual_seed(reqs)
                rows = reqs * rows_per_req
                q = torch.randn(
                    rows, 1, heads, D, device="cuda", generator=g
                ).bfloat16()
                per_req = torch.randint(
                    0, self.swa_tokens, (reqs, 1, 192), device="cuda", generator=g
                ).int()
                indices = per_req.repeat_interleave(rows_per_req, dim=0)
                lengths = _lengths(reqs, 192, "rand", g).repeat_interleave(rows_per_req)
                sink = torch.randn(64, device="cuda", generator=g)
                padded = q.new_zeros(rows, 1, 64, D)
                padded[:, :, :heads] = q
                ref, _ = self._attn(padded, indices, lengths, sink, None)
                cached = None
                for source in (indices, indices.clone()):
                    out, packed = forward_packed_draft(
                        flash_mla_with_kvcache=mla,
                        q=q,
                        swa_k_cache=self.swa,
                        swa_page_indices=indices,
                        swa_topk_lengths=lengths,
                        attn_sink=sink,
                        rows_per_req=rows_per_req,
                        cached=cached,
                        source=source,
                        head_dim_v=D,
                        softmax_scale=D**-0.5,
                        tuned_sched=tuned,
                    )
                    self.assertIsNot(packed, cached)
                    self.assertIs(packed.source, source)
                    _within_row_ulps(self, out, ref[:, 0, :heads])
                    cached = packed
                _, again = forward_packed_draft(
                    flash_mla_with_kvcache=mla,
                    q=q,
                    swa_k_cache=self.swa,
                    swa_page_indices=indices,
                    swa_topk_lengths=lengths,
                    attn_sink=sink,
                    rows_per_req=rows_per_req,
                    cached=cached,
                    source=cached.source,
                    head_dim_v=D,
                    softmax_scale=D**-0.5,
                    tuned_sched=tuned,
                )
                self.assertIs(again, cached)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA")
class TestExpandPageTableOneLaunch(CustomTestCase):
    def test_matches_torch_expansion(self):
        """The one-launch c1/c2 index block table is bitwise the torch expansion,
        int32 truncation of large page ids and a strided row view included."""
        from sglang.srt.layers.attention.dsv4.metadata import expand_index_page_table

        for bs, n, ratio, base in (
            (1, 1, 1, 0),
            (64, 37, 1, 0),
            (5, 300, 2, 0),
            (3, 9, 1, 2**30),
        ):
            with self.subTest(bs=bs, n=n, ratio=ratio, base=base):
                table = torch.randint(0, 1 << 20, (bs, n + 3), device="cuda").int()
                table[:, 0] = base
                table = table[:, :n]
                kwargs = dict(
                    full_page_size=256, compress_ratio=ratio, index_page_size=64
                )
                ref = expand_index_page_table(table, **kwargs)
                got = expand_index_page_table(table, **kwargs, one_launch=True)
                self.assertEqual(got.dtype, ref.dtype)
                self.assertTrue(torch.equal(got, ref))


if __name__ == "__main__":
    unittest.main()
