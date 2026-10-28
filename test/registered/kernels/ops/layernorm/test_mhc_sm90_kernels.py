"""The DeepSeek-V4.1 SM90 mHC kernels (`HcConfig.sm90`, switch "mhc") must reproduce the
kernels they replace bit for bit: a row's coefficients and streams then do not depend
on the batch size or on which kernel a row threshold picks."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import triton

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

K, MIX, H = 20480, 24, 5120
SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _x(m, special):
    x = torch.randn(m, K, device="cuda") * torch.exp(torch.randn(m, 1, device="cuda"))
    if special:
        x[::3] *= torch.exp2(torch.randint(-30, 20, (1,), device="cuda").float())
        x[:, ::97] = -0.0
        x[1::4, : K // 2] = 0.0
    return x.bfloat16().contiguous()


def _streams(m, special, scale=1.0):
    r = torch.randn(m, 4, H, device="cuda") * scale
    if special:
        r = r * torch.exp2(torch.randint(-20, 12, r.shape, device="cuda").float())
        r[:, :, ::97] = -0.0
    return r.bfloat16()


def _mixes(m):
    post = (2 * torch.sigmoid(torch.randn(m, 4, device="cuda"))).contiguous()
    comb = torch.rand(m, 4, 4, device="cuda") + 1e-6
    comb = (comb / comb.sum(-1, keepdim=True)).contiguous()
    pre = torch.sigmoid(torch.randn(m, 4, device="cuda")) + 1e-6
    return post, comb, pre


def _rmsnorm(seed):
    from sglang.srt.layers.layernorm import RMSNorm

    norm = RMSNorm(H, eps=1e-6).cuda().to(torch.bfloat16)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    norm.weight.data.copy_(
        (1 + 0.1 * torch.randn(H, device="cuda", generator=gen)).bfloat16()
    )
    return norm


def _production_partial(x, fn):
    from sglang.kernels.ops.layernorm import mhc

    m = x.shape[0]
    ns = mhc._num_slices_for(K)
    pm = torch.empty(ns, m, MIX, device="cuda")
    ps = torch.empty(ns, m, device="cuda")
    bm = mhc._block_m_for(m)
    mhc._hc_mix_stats_partial_kernel[(triton.cdiv(m, bm), ns)](
        x, fn, pm, ps, m, K, x.stride(0), fn.stride(0),
        MIX=MIX, MIX_PAD=32, NUM_SLICES=ns, BLOCK_M=bm, BLOCK_K=64,
        DOT_PRECISION=mhc._HC_MIX_DOT_PRECISION, num_warps=4,
        num_stages=mhc._HC_MIX_NUM_STAGES,
    )  # fmt: skip
    return pm, ps


def _sm90_partial(x, fn, v2_min_rows, wgmma_min_rows):
    from sglang.kernels.ops.layernorm import mhc
    from sglang.kernels.ops.layernorm.mhc_mix_sm90 import hc_mix_partial_sm90

    ns = mhc._num_slices_for(K)
    pm = torch.empty(ns, x.shape[0], MIX, device="cuda")
    ps = torch.empty(ns, x.shape[0], device="cuda")
    hc_mix_partial_sm90(
        x, fn, pm, ps, v2_min_rows=v2_min_rows, wgmma_min_rows=wgmma_min_rows
    )
    return pm, ps


class _Sm90Case(CustomTestCase):
    def setUp(self):
        from sglang.kernels.ops.layernorm import mhc

        # The TileLang post wrapper asks the parallel config about CP prefill.
        patcher = mock.patch.object(mhc, "is_dsa_prefill_cp_interleave", lambda: False)
        patcher.start()
        self.addCleanup(patcher.stop)


@unittest.skipUnless(SM90, "needs SM90")
class TestMhcSm90Kernels(_Sm90Case):
    def assert_bitwise(self, got, ref):
        for a, b in zip(got, ref):
            self.assertTrue(torch.equal(_bits(a.reshape(b.shape)), _bits(b)))

    def test_partials_match_production(self):
        """The tf32x3-unrolled and wgmma partials write the production partials: one
        and several row tiles, partial tiles, and the slice-resident W from 1024 rows
        whose contiguous units cross K slices (1100, 4100)."""
        from sglang.kernels.ops.layernorm.mhc_mix_wgmma import hc_mix_partial_wgmma

        torch.manual_seed(0)
        fn = (torch.randn(MIX, K, device="cuda") * 0.02).contiguous()
        for m in (1, 2, 5, 6, 63, 64, 65, 127, 128, 192, 366, 1024, 1100, 2048, 4100):
            for special in (False, True):
                x = _x(m, special)
                ref = _production_partial(x, fn)
                with self.subTest(m=m, special=special, kernel="v2"):
                    self.assert_bitwise(_sm90_partial(x, fn, 1, None), ref)
                for slice_w in (False, True):
                    with self.subTest(m=m, special=special, slice_w=slice_w):
                        pm, ps = torch.empty_like(ref[0]), torch.empty_like(ref[1])
                        hc_mix_partial_wgmma(
                            x, fn, pm, ps, pm.shape[0], slice_w=slice_w
                        )
                        self.assert_bitwise((pm, ps), ref)

    def test_fused_reduce_sinkhorn_matches_tilelang(self):
        """Reduce + split sinkhorn in one Gluon kernel equals hc_mix_stats followed by
        the TileLang sinkhorn, through x and through extreme partials."""
        from sglang.kernels.ops.layernorm import mhc
        from sglang.kernels.ops.layernorm.mhc_mix_sm90 import (
            hc_mix_stats_split_sinkhorn,
        )
        from sglang.kernels.ops.layernorm.mhc_reduce_sinkhorn import (
            hc_mix_reduce_sinkhorn,
        )

        torch.manual_seed(3)
        fn = (torch.randn(MIX, K, device="cuda") * 0.02).contiguous()
        scale = (0.5 + torch.rand(3, device="cuda")).contiguous()
        base = (torch.randn(MIX, device="cuda") * 0.5).contiguous()
        for rows in ((None, None), (2, 128)):
            for m in (2, 6, 63, 192, 384, 1024, 2048):
                with self.subTest(rows=rows, m=m):
                    x = _x(m, special=True)
                    mixes = mhc.hc_mix_stats(x, fn, 1e-6).unsqueeze(1)
                    ref = mhc.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
                    got = hc_mix_stats_split_sinkhorn(
                        x, fn, scale, base, 4, 20, 1e-6, 1e-6,
                        v2_min_rows=rows[0], wgmma_min_rows=rows[1],
                    )  # fmt: skip
                    self.assert_bitwise(got, ref)
        for m in (5, 384, 2048):
            with self.subTest(partials=m):
                pm = torch.randn(80, m, MIX, device="cuda") * torch.exp2(
                    torch.randint(-40, 40, (80, m, 1), device="cuda").float()
                )
                ps = torch.rand(80, m, device="cuda") * 100
                pm[:, 0::7] = float("inf")
                pm[:, 1::7, :8] = float("nan")
                pm[:, 2::7] = -0.0
                ps[:, 3::7] = 0.0
                pm[:, 4::7, 8:] *= 1e30
                mixes = torch.empty(m, MIX, device="cuda")
                mhc._hc_mix_stats_reduce_kernel[(triton.cdiv(m, 16),)](
                    pm, ps, mixes, m, 1.0 / K, 1e-6,
                    MIX=MIX, MIX_PAD=32, NUM_SLICES=80, BLOCK_M=16, num_warps=4,
                )  # fmt: skip
                ref = mhc.hc_split_sinkhorn(
                    mixes.unsqueeze(1), scale, base, 4, 20, 1e-6
                )
                got_mixes = torch.empty_like(mixes)
                got = hc_mix_reduce_sinkhorn(
                    pm, ps, scale, base, 1.0 / K, 1e-6, 20, 1e-6, mixes=got_mixes
                )
                self.assert_bitwise((got_mixes,) + tuple(got), (mixes,) + tuple(ref))

    def test_post_combine_matches_post_then_combine(self):
        """One launch for the post and the next collapse repeats both the split-h and
        the TileLang post (every row range it splits differently) and mhc_combine."""
        from sglang.kernels.ops.layernorm import mhc
        from sglang.kernels.ops.layernorm.mhc_post_combine import mhc_post_combine
        from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h

        torch.manual_seed(0)
        for m in (1, 6, 8, 9, 64, 65, 192, 256, 257, 384, 2048, 4096):
            for special in (False, True):
                with self.subTest(m=m, special=special):
                    r = _streams(m, special)
                    x = (torch.randn(m, H, device="cuda") * 3).bfloat16()
                    post, comb, pre = _mixes(m)
                    ref = mhc.mhc_post(x, r, post, comb)
                    ref_y = mhc.mhc_combine(ref.flatten(1), pre, 4, x.dtype)
                    out, y = mhc_post_combine(x, r, post, comb, pre)
                    self.assert_bitwise((out, y), (ref, ref_y))
                    if m <= 384:
                        self.assert_bitwise(
                            (out,), (mhc_post_split_h(x, r, post, comb),)
                        )

    def test_norm_fold_matches_rmsnorm(self):
        """The folded RMSNorm repeats the FlashInfer RMSNorm RMSNorm.forward_cuda runs
        (one CTA row, 64 threads per row); odd rows, tiny values, strided pre."""
        from sglang.kernels.ops.layernorm import mhc
        from sglang.kernels.ops.layernorm.mhc_post_combine import mhc_post_combine
        from sglang.kernels.ops.layernorm.mhc_post_combine_norm import (
            mhc_combine_norm,
            mhc_post_combine_norm,
        )

        torch.manual_seed(1)
        for m in (1, 3, 192, 2047, 4096):
            for case, (special, scale) in enumerate(
                ((False, 1.0), (False, 0.01), (True, 1.0))
            ):
                with self.subTest(m=m, case=case):
                    norm = _rmsnorm(seed=case)
                    r = _streams(m, special, scale)
                    x = (torch.randn(m, H, device="cuda") * 3 * scale).bfloat16()
                    post, comb, pre = _mixes(m)
                    if case == 1:
                        pre = torch.randn(m, 8, device="cuda")[:, 2:6].copy_(pre)
                    ref_out, ref_y = mhc_post_combine(x, r, post, comb, pre)
                    got = mhc_post_combine_norm(
                        x, r, post, comb, pre, norm.weight, norm.variance_epsilon
                    )
                    self.assert_bitwise(got, (ref_out, norm(ref_y)))
                    ref = norm(mhc.mhc_combine(r.flatten(1), pre, 4, r.dtype))
                    got = mhc_combine_norm(r, pre, norm.weight, norm.variance_epsilon)
                    self.assert_bitwise((got,), (ref,))


def _sm90_hc(rows, *, seed):
    from sglang.srt.models import deepseek_v4_mhc as model_mhc

    gen = torch.Generator(device="cuda").manual_seed(seed)
    cfg = model_mhc.HcConfig(4, 20, 1e-6, 1e-6, H, True, False, sm90=rows)
    fn = (torch.randn(MIX, K, device="cuda", generator=gen) * 0.02).contiguous()
    scale = (0.5 + torch.rand(3, device="cuda", generator=gen)).contiguous()
    base = (torch.randn(MIX, device="cuda", generator=gen) * 0.5).contiguous()
    return model_mhc.HcSubLayer(cfg, fn, scale, base, _rmsnorm(seed))


@unittest.skipUnless(SM90, "needs SM90")
class TestMhcSm90Dispatch(_Sm90Case):
    def test_stats_are_batch_invariant(self):
        """Multi-row triplets equal hc_mix_stats + the TileLang sinkhorn, so a row's
        bits are the same alone and in a batch across the partial-kernel thresholds."""
        from sglang.kernels.ops.layernorm import mhc
        from sglang.srt.models import deepseek_v4_mhc as model_mhc

        hc = _sm90_hc(model_mhc.sm90_hc_rows("NVIDIA H20"), seed=5)
        x = _x(4100, special=False).view(4100, 4, H)
        for n, m in ((2, 6), (127, 128), (1023, 1024), (2048, 4100)):
            with self.subTest(n=n, m=m):
                alone = model_mhc.mix_stats(hc, x[:n])
                batch = model_mhc.mix_stats(hc, x[:m])
                for a, b in zip(alone, batch):
                    self.assertTrue(torch.equal(_bits(a), _bits(b[:n])))
                mixes = mhc.hc_mix_stats(x[:m].flatten(1), hc.fn, 1e-6).unsqueeze(1)
                ref = mhc.hc_split_sinkhorn(mixes, hc.scale, hc.base, 4, 20, 1e-6)
                for a, b in zip(batch, ref):
                    self.assertTrue(torch.equal(_bits(a), _bits(b.reshape(a.shape))))

    def test_fused_post_matches_post_then_combine(self):
        """The fused post leaves the next sublayer the input its own combine would
        build, below and above the norm-fold rows and after the tail's take_rows."""
        from sglang.srt.models import deepseek_v4_mhc as model_mhc

        rows = model_mhc.sm90_hc_rows("NVIDIA H20")
        hc, next_hc = _sm90_hc(rows, seed=7), _sm90_hc(rows, seed=8)
        boundary = model_mhc.make_boundary(
            next_hc.norm, accepts_mxfp8=False, hc=next_hc
        )
        torch.manual_seed(9)
        for m in (6, 384, 1024, 2048, 4096):
            with self.subTest(m=m):
                r = _streams(m, special=False)
                y = (torch.randn(m, H, device="cuda") * 3).bfloat16()
                post, comb, pre = _mixes(m)
                state = model_mhc._post_fusion(hc, y, r, (pre, post, comb), boundary)
                ref = model_mhc.post(hc.cfg, y, r, post, comb)
                ref_x = model_mhc._combine(next_hc, model_mhc.HcState(ref, pre), None)
                self.assertTrue(torch.equal(_bits(state.residual), _bits(ref)))
                x = model_mhc.combine(next_hc, state)
                self.assertTrue(torch.equal(_bits(x), _bits(ref_x)))
                keep = torch.arange(1, m, 2, device="cuda")
                narrowed = state.take_rows(lambda t: t.index_select(0, keep))
                self.assertTrue(
                    torch.equal(
                        _bits(model_mhc.combine(next_hc, narrowed)), _bits(ref_x[keep])
                    )
                )


class TestMhcSm90Rows(CustomTestCase):
    def test_untuned_devices_keep_upstream_choices(self):
        """Thresholds measured on H20 apply only there; other SM90 parts keep the
        production partial, the 64-row stats stream and unfolded norms."""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import deepseek_v4_mhc as model_mhc

        self.assertIs(
            model_mhc.sm90_hc_rows("NVIDIA H20-3e"),
            model_mhc.sm90_hc_rows("NVIDIA H20"),
        )
        rows = model_mhc.sm90_hc_rows("NVIDIA H100 80GB HBM3")
        self.assertIsNone(rows.partial_v2_min)
        self.assertIsNone(rows.partial_wgmma_min)
        self.assertIsNone(rows.combine_norm_min)
        self.assertIsNone(rows.post_combine_norm_min)
        verify = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        for name, limit in (("NVIDIA H20", 384), ("NVIDIA H100 80GB HBM3", 64)):
            cfg = model_mhc.HcConfig(
                4, 20, 1e-6, 1e-6, H, True, False, sm90=model_mhc.sm90_hc_rows(name)
            )
            for m, expected in ((1, True), (limit, True), (limit + 1, False)):
                with self.subTest(device=name, m=m):
                    x = torch.empty((m, 4, 8), device="meta")
                    self.assertEqual(
                        model_mhc.use_stats_stream(cfg, verify, x), expected
                    )


if __name__ == "__main__":
    unittest.main()
