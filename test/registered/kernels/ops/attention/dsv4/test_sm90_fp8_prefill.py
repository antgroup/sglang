"""SM90 FP8 prefill index scoring: the tiled and keys-major scorers, the FP4 key
unpack and the fused top-k finish."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import (
    sm90_fp8_prefill_keys_major as keys_major_mod,
)
from sglang.kernels.ops.attention.dsv4.sm90_fp8_prefill import (
    finalize_prefill_topk,
    fp8_index_logits_prefill,
    keys_major_min_scores,
    quantize_bf16_index_queries_fp8,
    unpack_fp4_index_keys_to_fp8,
)
from sglang.kernels.ops.attention.dsv4.sm90_fp8_prefill_keys_major import (
    keys_major_scores,
)
from sglang.kernels.ops.kvcache.dsv4_fp4_index_k_cache import store_fp4_index_k_cache
from sglang.kernels.ops.quantization.fp4_fake_quant import fake_quant_fp4
from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or torch.cuda.get_device_capability() != (9, 0),
    reason="requires SM90",
)


def _cache(keys, page_size):
    table = torch.zeros(
        keys.shape[0] // page_size, page_size * 68, dtype=torch.uint8, device="cuda"
    )
    store_fp4_index_k_cache(
        input=keys,
        cache=table,
        loc=torch.arange(keys.shape[0], device="cuda", dtype=torch.int32),
        page_size=page_size,
        rne=True,
    )
    return table


def _queries(rows):
    # Serving queries are RoPE'd and FP4 fake-quantized before scoring.
    return fake_quant_fp4(
        torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16)
    )


def _reference(q, keys, weights, lens):
    scores = DeepseekV41Indexer.scores(None, q, keys, weights)
    scores.masked_fill_(
        torch.arange(keys.shape[0], device="cuda")[None, :] >= lens[:, None], -torch.inf
    )
    return scores


def _assert_bitwise(got, want):
    # Same NaN-ness; NaN payloads are not part of the contract.
    nan = torch.isnan(want)
    assert torch.equal(torch.isnan(got), nan)
    assert torch.equal(got.view(torch.int32)[~nan], want.view(torch.int32)[~nan])


@pytest.mark.parametrize(
    "rows,width",
    ((0, 193), (7, 0), (1, 1), (7, 193), (64, 512), (7, 4097), (6, 68404), (130, 3001)),
)
@pytest.mark.parametrize("page_size", (64, 128))
def test_fp4_inputs_match_bf16_reference_and_aligned_padding(rows, width, page_size):
    """The FP8 path converts FP4-grid queries and keys exactly and stays within
    tolerance of the BF16 torch scores (the head sum is FP32 here); rows keep a
    four-float stride for ragged top-k with -inf padding."""
    torch.manual_seed(39086)
    capacity = (max(1, width) + page_size - 1) // page_size * page_size
    keys = fake_quant_fp4(
        torch.randn(capacity, 128, device="cuda", dtype=torch.bfloat16)
    )
    table = _cache(keys, page_size)
    slots = torch.randperm(capacity, device="cuda")[:width]
    q = _queries(rows)
    weights = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)
    lens = torch.arange(rows, device="cuda") * max(1, width // max(1, rows - 1))
    lens = lens.clamp_max(width)
    if rows:
        lens[-1] = width
    q8 = quantize_bf16_index_queries_fp8(q)
    k8 = unpack_fp4_index_keys_to_fp8(slots, table, page_size)
    torch.testing.assert_close(q8.bfloat16(), q, atol=0, rtol=0)
    torch.testing.assert_close(k8.bfloat16(), keys[slots], atol=0, rtol=0)
    actual = fp8_index_logits_prefill(q8, weights, k8, lens)
    assert actual.shape == (rows, width)
    assert width == 0 or actual.stride(0) % 4 == 0
    if rows and width:
        expected = _reference(q, keys[slots], weights, lens)
        torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.02)
        storage = actual.as_strided((rows, (width + 3) // 4 * 4), actual.stride())
        assert torch.isneginf(storage[:, width:]).all()


@pytest.mark.parametrize("page_size", (64, 128))
@pytest.mark.parametrize("rows", (1, 33, 1024, 1057))
def test_row_tiled_unpack_is_bitwise_the_per_row_unpack(page_size, rows):
    """The row-tiled unpack looks each (FP4 code, scale byte) E4M3 byte up in a
    table of the per-row kernel's outputs: with every (payload byte, scale byte)
    pair present, any slot order and row counts off the block size, the bytes
    must be the per-row kernel's."""
    g = torch.Generator(device="cuda").manual_seed(rows)
    pages = 2048 // page_size
    table = torch.randint(
        0, 256, (pages, page_size * 68), dtype=torch.uint8, device="cuda", generator=g
    )
    # Block j of rows 0..1023: scale byte j % 256, payload bytes 16 * (j // 256) + i.
    j = torch.arange(4096, device="cuda")
    payload = (j // 256)[:, None] * 16 + torch.arange(16, device="cuda")
    exhaustive = table[: 1024 // page_size]
    exhaustive[:, : page_size * 64] = payload.to(torch.uint8).view(-1, page_size * 64)
    exhaustive[:, page_size * 64 :] = (j % 256).to(torch.uint8).view(-1, page_size * 4)
    slots = torch.randperm(pages * page_size, device="cuda", generator=g)[:rows]
    if rows == 1024:
        slots = torch.arange(1024, device="cuda")
    expected = unpack_fp4_index_keys_to_fp8(slots, table, page_size, row_tiled=False)
    actual = unpack_fp4_index_keys_to_fp8(slots, table, page_size, row_tiled=True)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_query_conversion_saturates():
    values = torch.tensor(
        [0, 0.5, -6, 448, -448, 768, -768, 2**-10], device="cuda", dtype=torch.bfloat16
    )
    q = values.repeat(32 * 128 // values.numel()).reshape(1, 32, 128)
    got = quantize_bf16_index_queries_fp8(q).float()
    expected = q.float().clamp(-448, 448).to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_nonfinite_weights_and_tied_scores_keep_causal_mask():
    q = torch.ones(4, 32, 128, device="cuda", dtype=torch.bfloat16)
    keys = torch.ones(7, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(4, 32, device="cuda", dtype=torch.bfloat16)
    weights[0] = 0
    weights[1, 0] = torch.nan
    weights[2, 0] = torch.inf
    lens = torch.tensor([0, 5, 2, 4], device="cuda")
    actual = fp8_index_logits_prefill(
        q.to(torch.float8_e4m3fn), weights, keys.to(torch.float8_e4m3fn), lens
    )
    torch.testing.assert_close(
        actual, _reference(q, keys, weights, lens), rtol=0, atol=0, equal_nan=True
    )


def _sequential_scores(q8, weights, k8, lens):
    """The tiled kernel's arithmetic one head at a time: exact 32-wide partial
    dots, FP32 ((p0 + p1) + p2) + p3, BF16 rounding, ReLU, BF16 weighted score,
    FP32 head sum in head order, BF16 logit."""
    q, k, w = q8.double(), k8.double(), weights.float()
    acc = torch.zeros(q.shape[0], k.shape[0], device="cuda")
    for h in range(q.shape[1]):
        p = [
            (q[:, h, 32 * j : 32 * j + 32] @ k[:, 32 * j : 32 * j + 32].T).float()
            for j in range(4)
        ]
        s = (((p[0] + p[1]) + p[2]) + p[3]).bfloat16().float().clamp_min(0)
        acc += (s * w[:, h, None]).bfloat16().float()
    logit = acc.bfloat16().float()
    columns = torch.arange(k.shape[0], device="cuda")[None, :]
    return logit.masked_fill(columns >= lens[:, None], -torch.inf)


def test_tiled_scores_are_the_sequential_arithmetic():
    """Bit for bit the stated rounding points and add orders, on block scales
    spanning 2^-6..2^4 so the cross-block FP32 adds round."""
    torch.manual_seed(11)
    rows, width = 64, 3001
    q = torch.randn(rows, 32, 4, 32, device="cuda") * torch.exp2(
        torch.randint(-6, 5, (rows, 32, 4, 1), device="cuda").float()
    )
    k = torch.randn(width, 4, 32, device="cuda") * torch.exp2(
        torch.randint(-6, 5, (width, 4, 1), device="cuda").float()
    )
    q8 = quantize_bf16_index_queries_fp8(
        fake_quant_fp4(q.view(rows, 32, 128).bfloat16())
    )
    k8 = fake_quant_fp4(k.view(width, 128).bfloat16()).to(torch.float8_e4m3fn)
    weights = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)
    lens = torch.randint(0, width + 1, (rows,), device="cuda")
    got = fp8_index_logits_prefill(q8, weights, k8, lens, keys_major=False)
    _assert_bitwise(got, _sequential_scores(q8, weights, k8, lens))


def test_tiled_scores_do_not_depend_on_tiling_or_row_split():
    """Batch/chunk invariance: a row's tiled scores are bitwise the same whatever
    the tile config and however chunking or CP splits the rows across calls."""
    torch.manual_seed(4)
    rows, width = 200, 9000
    keys = fake_quant_fp4(torch.randn(width, 128, device="cuda", dtype=torch.bfloat16))
    q8 = quantize_bf16_index_queries_fp8(_queries(rows))
    k8 = keys.to(torch.float8_e4m3fn)
    weights = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)
    lens = torch.randint(0, width + 1, (rows,), device="cuda")
    full = fp8_index_logits_prefill(q8, weights, k8, lens, keys_major=False)
    for config in ((16, 128, 4, 3), (32, 64, 4, 3), (64, 64, 4, 2), (128, 128, 8, 2)):
        got = fp8_index_logits_prefill(q8, weights, k8, lens, tiled_config=config)
        assert torch.equal(got, full), config
    for split in (1, 7, 31, 64, 133):
        parts = [
            fp8_index_logits_prefill(
                q8[i : i + split],
                weights[i : i + split],
                k8,
                lens[i : i + split],
                keys_major=False,
            )
            for i in range(0, rows, split)
        ]
        assert torch.equal(torch.cat(parts), full), split


@pytest.mark.parametrize(
    "rows,width",
    ((1, 300), (37, 5000), (64, 127), (300, 128), (1000, 20000), (2048, 131073)),
)
@pytest.mark.parametrize("lens_kind", ("causal", "random", "zero"))
def test_keys_major_scores_are_bitwise_the_tiled_scores(rows, width, lens_kind):
    """The keys-major kernel only reschedules the tiled arithmetic (same exact
    32-wide partials, FP32 add order and BF16 rounding points, bf16x2 ops equal
    for every input), so scores, the -inf mask and the padding are bitwise equal
    for any lengths, block scales and CTA split."""
    torch.manual_seed(rows * 7 + width)
    # Block scales spanning 2^-6..2^4 make the cross-block FP32 adds round.
    q = torch.randn(rows, 32, 4, 32, device="cuda") * torch.exp2(
        torch.randint(-6, 5, (rows, 32, 4, 1), device="cuda").float()
    )
    k = torch.randn(width, 4, 32, device="cuda") * torch.exp2(
        torch.randint(-6, 5, (width, 4, 1), device="cuda").float()
    )
    q8 = quantize_bf16_index_queries_fp8(
        fake_quant_fp4(q.view(rows, 32, 128).bfloat16())
    )
    k8 = fake_quant_fp4(k.view(width, 128).bfloat16()).to(torch.float8_e4m3fn)
    weights = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)
    weights[0, 1] = torch.nan
    weights[-1, 2] = torch.inf
    weights[rows // 2] = 0
    if lens_kind == "causal":
        lens = width - rows + 1 + torch.arange(rows, device="cuda")
    elif lens_kind == "random":
        lens = torch.randint(0, width + 50, (rows,), device="cuda")
    else:
        lens = torch.zeros(rows, device="cuda", dtype=torch.int64)
    lens = lens.clamp_min(0)
    want = fp8_index_logits_prefill(q8, weights, k8, lens, keys_major=False)
    for grid_size in (0, 1, 7):
        got = torch.full_like(
            want.as_strided((rows, want.stride(0)), want.stride()), 1.0
        )
        keys_major_scores(q8, weights, k8, lens.long(), got, grid_size=grid_size)
        _assert_bitwise(got, want.as_strided(got.shape, got.stride()))
    threshold = keys_major_min_scores(q8.device)
    if threshold is not None and rows * width >= threshold:
        _assert_bitwise(fp8_index_logits_prefill(q8, weights, k8, lens), want)


class TestKeysMajorPrewarm(CustomTestCase):
    def test_long_context_and_ragged_chunks_reuse_startup_binaries(self):
        """After the prewarm, CP chunk tails, ragged contexts and either length
        alignment load no new binary; a warm disk cache must not hide one."""
        kernels = (
            keys_major_mod._visible_tile_prefix_kernel,
            keys_major_mod._keys_major_scores_kernel,
        )
        device = torch.cuda.current_device()
        caches = [kernel.device_caches[device][0] for kernel in kernels]
        for cache in caches:
            cache.clear()
        keys_major_mod.prewarm_keys_major_scores(torch.device("cuda", device), 32)
        counts = [len(cache) for cache in caches]
        self.assertEqual(counts, [2, 4])
        for rows, width in (
            (1024, 65536),
            (2048, 131072),
            (1023, 131073),
            (33, 65537),
            (7, 131073),
            (1, 131072),
        ):
            with self.subTest(rows=rows, width=width):
                q8 = quantize_bf16_index_queries_fp8(_queries(rows))
                k8 = fake_quant_fp4(
                    torch.randn(width, 128, device="cuda", dtype=torch.bfloat16)
                ).to(torch.float8_e4m3fn)
                weights = torch.rand(rows, 32, device="cuda", dtype=torch.bfloat16)
                for offset in (0, 1):
                    lengths = torch.arange(rows + 1, device="cuda", dtype=torch.int64)
                    lens = (lengths + width - rows)[offset : offset + rows]
                    want = fp8_index_logits_prefill(
                        q8, weights, k8, lens, keys_major=False
                    )
                    got = torch.empty(
                        (rows, want.stride(0)), device="cuda", dtype=torch.float32
                    )
                    keys_major_scores(q8, weights, k8, lens, got)
                    _assert_bitwise(got, want.as_strided(got.shape, got.stride()))
                    self.assertEqual([len(cache) for cache in caches], counts)


def _finalize_reference(*, selected, scores, lens, slots, mask_scores):
    k = selected.shape[1]
    if mask_scores:
        columns = selected.long()
        picked = scores.gather(1, columns.clamp(0, scores.shape[1] - 1))
        valid = (columns >= 0) & (columns < scores.shape[1]) & (picked > -torch.inf)
        selected = selected.masked_fill(~valid, -1)
    unselected = torch.iinfo(torch.int32).max
    idx = selected.masked_fill(selected < 0, unselected).sort(dim=-1).values
    reach = (idx != unselected) & (idx < lens[:, None])
    pages = torch.where(reach, slots[idx.clamp_max(slots.shape[0] - 1)], -1)
    return pages.to(torch.int32)[:, :k], torch.where(reach, idx, -1)[:, :k]


@pytest.mark.parametrize("lc", (37, 513, 32768))
@pytest.mark.parametrize("mask_scores", (False, True))
def test_finalize_prefill_topk_matches_sorted_slot_mapping(lc, mask_scores):
    """Holes, duplicate ids, ids past a row's visible length, -inf and NaN scores,
    empty rows and lc < k: the finish writes the sorted, -1 last selection into
    the first k columns of strided output rows and nothing else, with or without
    the page indices."""
    g = torch.Generator(device="cuda").manual_seed(lc)
    rows, k = 1000, min(512, lc)
    selected = torch.randint(
        -1, lc, (rows, k), device="cuda", generator=g, dtype=torch.int32
    )
    selected[torch.rand(rows, k, device="cuda", generator=g) < 0.2] = -1
    selected[0] = -1
    scores = torch.randn(rows, lc + 3, device="cuda", generator=g)[:, :lc]
    scores[torch.rand(rows, lc, device="cuda", generator=g) < 0.3] = -torch.inf
    scores[torch.rand(rows, lc, device="cuda", generator=g) < 0.01] = torch.nan
    lens = torch.randint(
        0, lc + 1, (rows,), device="cuda", generator=g, dtype=torch.int32
    )
    lens[1] = 0
    slots = torch.randint(0, 1 << 31, (lc,), device="cuda", generator=g)
    want_pages, want_raw = _finalize_reference(
        selected=selected,
        scores=scores,
        lens=lens,
        slots=slots,
        mask_scores=mask_scores,
    )
    for with_pages in (True, False):
        pages = torch.full((rows + 5, k + 8), 99, device="cuda", dtype=torch.int32)
        raw = torch.full_like(pages, 77)
        finalize_prefill_topk(
            selected=selected,
            scores=scores,
            lens=lens,
            slots=slots,
            page_indices=pages[3 : 3 + rows] if with_pages else None,
            raw_indices=raw[3 : 3 + rows],
            mask_scores=mask_scores,
        )
        assert torch.equal(raw[3 : 3 + rows, :k], want_raw)
        if with_pages:
            assert torch.equal(pages[3 : 3 + rows, :k], want_pages)
        for out, fill in ((pages, 99), (raw, 77)):
            if out is pages and not with_pages:
                assert (out == fill).all()
                continue
            assert (out[:3] == fill).all() and (out[3 + rows :] == fill).all()
            assert (out[:, k:] == fill).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
