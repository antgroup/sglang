"""The DeepSeek-V4.1 SM90 ratio-2 compressor kernels (switch "decode") are bitwise the
backend's unfused paths: target verify pooled in one launch, a request's rows taking
their predecessor from the batch, and the post-pooling RMSNorm, RoPE, fake FP4 and V4
FlashMLA store in one launch. A rounding difference would reach the indexer's top-k."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

SM90_UP = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
DIM = 512


def _freqs(length=8192):
    angles = torch.randn(length, 32, device="cuda")
    return torch.polar(torch.ones_like(angles), angles)


def _page_bytes(page_size):
    return (584 * page_size + 575) // 576 * 576


def _unfused_write(pooled, weight, eps, freqs, slots, group_pos, cache, page_size):
    """The backend's _low_ratio_write_group main-KV half on a V4 cache."""
    from sglang.kernels.ops.attention.dsv4 import fused_store_cache
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import RMSNorm, _rope_fq4

    norm = RMSNorm(DIM, eps).cuda()
    norm.weight.data = weight
    latent = norm(pooled.to(torch.bfloat16))
    gathered = freqs[group_pos]
    quantized = _rope_fq4(latent, gathered, 64, compressed_kv=True)
    # The unfused store also writes the incomplete pairs' scratch slot 0; compare
    # the live slots only, and require the epilogue to leave slot 0 alone.
    live = slots > 0
    if live.any().item():
        fused_store_cache(
            quantized[live].contiguous(),
            cache,
            slots[live].contiguous(),
            page_size=page_size,
            type="flashmla",
            layout=KVLayout.V4,
        )
    return latent, gathered


@unittest.skipUnless(SM90_UP, "SM90+ CUDA")
class TestC2VerifyPool(CustomTestCase):
    def test_verify_rows_match_backend_pair_partners(self):
        """Ragged verify windows, padding rows aliasing request 0, and positions
        that wrap the ring, replayed from one CUDA graph: pooled rows, group
        positions, slots and the whole ring state match the torch path."""
        from sglang.kernels.ops.attention.dsv4.c2_decode_pool import c2_decode_pool
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )
        from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool

        for start_pos in (0, 7, 8):
            for verify_lens in ((6,), (6, 6), (1, 3, 6)):
                for padding in (0, 3):
                    with self.subTest(start=start_pos, lens=verify_lens, pad=padding):
                        torch.manual_seed(47)
                        bs, ring = len(verify_lens), 8
                        live = sum(verify_lens)
                        n = live + padding
                        state = CompressStatePool(
                            size=bs * ring,
                            ring_size=ring,
                            overlap=False,
                            head_dim=DIM,
                            dtype=torch.float32,
                            device="cuda",
                            enable_memory_saver=False,
                            ratio=2,
                        )
                        initial = torch.randn_like(state.kv_score_buffer.kv_score)
                        initial[-1, :DIM] = 0
                        initial[-1, DIM:] = -torch.inf
                        fused = initial.clone()
                        backend = object.__new__(DeepseekV4AttnBackend)
                        backend.token_to_kv_pool = SimpleNamespace(
                            get_attention_compress_states=lambda layer_id: state
                        )
                        kv, score = (
                            torch.randn(n, DIM, device="cuda") for _ in range(2)
                        )
                        req = torch.tensor(
                            [r for r, k in enumerate(verify_lens) for _ in range(k)]
                            + [0] * padding,
                            device="cuda",
                        )
                        base_pos = torch.tensor(
                            [start_pos + i for k in verify_lens for i in range(k)]
                            + [start_pos + verify_lens[0] + i for i in range(padding)],
                            device="cuda",
                        )
                        pos = base_pos.clone()
                        raw_loc = torch.arange(256, 256 + n, device="cuda").int()
                        raw_loc[live:] = 0
                        out_loc = torch.where(pos % 2 == 1, raw_loc // 2, -1)

                        def run():
                            return c2_decode_pool(
                                kv,
                                score,
                                pos,
                                raw_loc,
                                out_loc,
                                req,
                                fused[:, :DIM],
                                fused[:, DIM:],
                                fused.shape[0] - 1,
                                ring_size=ring,
                                in_batch_predecessor=True,
                            )

                        run()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            got = run()
                        fused.copy_(initial)
                        state.kv_score_buffer.kv_score.copy_(initial)
                        for advance in (0, 1, 3, 6, 7, 8, 13, 16):
                            kv.normal_()
                            score.normal_()
                            pos.copy_(base_pos + advance)
                            out_loc.copy_(torch.where(pos % 2 == 1, raw_loc // 2, -1))
                            partner_kv, partner_score = (
                                backend._low_ratio_pair_partners(
                                    layer_id=0,
                                    kv=kv,
                                    score=score,
                                    req=req,
                                    pos=pos,
                                    pad=raw_loc == 0,
                                )
                            )
                            pairs = torch.stack([partner_kv, kv], dim=1)
                            weights = torch.stack([partner_score, score], dim=1)
                            expected = (
                                (pairs * weights.softmax(dim=1)).sum(dim=1),
                                torch.where(pos % 2 == 1, pos - 1, pos),
                                out_loc.clamp_min(0),
                            )
                            graph.replay()
                            for actual, wanted in zip(got, expected):
                                self.assertTrue(torch.equal(actual, wanted))
                            self.assertTrue(
                                torch.equal(fused, state.kv_score_buffer.kv_score)
                            )


@unittest.skipUnless(SM90_UP, "SM90+ CUDA")
class TestC2Epilogue(CustomTestCase):
    def test_latent_freqs_and_cache_bytes(self):
        """Non-contiguous slots across pages, padding (0) and incomplete pairs (-1):
        latent, gathered freqs and every cache byte match, slot 0 included."""
        from sglang.kernels.ops.attention.dsv4.c2_epilogue_sm90 import (
            c2_epilogue_sm90,
        )

        for rows in (0, 1, 6, 12, 64, 384):
            for page_size in (64, 128):
                for weight_dtype in (torch.bfloat16, torch.float32):
                    with self.subTest(rows=rows, page=page_size, dtype=weight_dtype):
                        torch.manual_seed(39100)
                        pooled = torch.randn(rows, DIM, device="cuda")
                        original = pooled.clone()
                        weight = torch.randn(DIM, device="cuda", dtype=weight_dtype)
                        freqs = _freqs()
                        position = torch.randint(0, 8192, (rows,), device="cuda")
                        capacity = max(rows * 2, page_size * 2)
                        slots = torch.randperm(capacity - 1, device="cuda")[:rows] + 1
                        slots = slots.int()
                        slots[1::3] = 0
                        slots[2::5] = -1
                        pages = (capacity + page_size - 1) // page_size
                        cache = torch.randint(
                            0, 256, (pages, _page_bytes(page_size)), device="cuda"
                        ).to(torch.uint8)
                        expected_cache = cache.clone()
                        expected = _unfused_write(
                            pooled,
                            weight,
                            1e-20,
                            freqs,
                            slots,
                            position,
                            expected_cache,
                            page_size,
                        )
                        actual = c2_epilogue_sm90(
                            pooled,
                            weight,
                            1e-20,
                            freqs,
                            slots,
                            position,
                            cache.view(torch.float8_e4m3fn),
                            page_size,
                        )
                        for got, wanted in zip(actual, expected):
                            self.assertTrue(torch.equal(got, wanted))
                        self.assertTrue(torch.equal(cache, expected_cache))
                        self.assertTrue(torch.equal(pooled, original))

    def test_rounding_boundaries_and_weight_scales(self):
        """BF16 ties and their FP32 neighbours in finish()'s cast, zero and tiny
        rows, and weights from 0 to 1024 hit the same rounding as the torch path."""
        from sglang.kernels.ops.attention.dsv4.c2_epilogue_sm90 import (
            c2_epilogue_sm90,
        )

        for weight_scale in (0.0, 2.0**-16, 1.0, 1024.0):
            with self.subTest(weight_scale=weight_scale):
                torch.manual_seed(39101)
                rows, page_size = 16, 128
                pooled = torch.randn(rows, DIM, device="cuda")
                pooled[0].fill_(0)
                pooled[1].fill_(1e-12)
                tie = torch.full((DIM,), 1.00390625, device="cuda")
                pooled[2] = tie
                pooled[3] = torch.nextafter(tie, torch.full_like(tie, torch.inf))
                pooled[4] = torch.nextafter(tie, torch.full_like(tie, -torch.inf))
                weight = (torch.randn(DIM, device="cuda") * weight_scale).bfloat16()
                freqs = _freqs()
                slots = torch.arange(1, rows + 1, device="cuda")
                position = torch.arange(rows, device="cuda", dtype=torch.int32)
                cache = torch.full(
                    (1, _page_bytes(page_size)), 93, device="cuda", dtype=torch.uint8
                )
                expected_cache = cache.clone()
                expected = _unfused_write(
                    pooled,
                    weight,
                    1e-20,
                    freqs,
                    slots,
                    position,
                    expected_cache,
                    page_size,
                )
                actual = c2_epilogue_sm90(
                    pooled, weight, 1e-20, freqs, slots, position, cache, page_size
                )
                for got, wanted in zip(actual, expected):
                    self.assertTrue(torch.equal(got, wanted))
                self.assertTrue(torch.equal(cache, expected_cache))


if __name__ == "__main__":
    unittest.main()
