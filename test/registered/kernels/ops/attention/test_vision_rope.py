"""Vision RoPE parity and input constraints: interleaved complex and half-split."""

import unittest

import torch

from sglang.kernels.ops.attention.vision_rope import (
    apply_fused_qk_complex_rope,
    apply_fused_qk_complex_rope_inplace,
    prepare_fused_qk_complex_rope_inplace,
)
from sglang.kernels.ops.attention.vision_rope_half import vision_qk_rope_half
from sglang.srt.models.deepseek_v41_vit import apply_rotary, get_vision_cos_sin
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestVisionRoPE(CustomTestCase):
    def test_vision_rope(self):
        torch.manual_seed(4)
        qkv = torch.randn(480, 3, 12, 128, device="cuda", dtype=torch.bfloat16)
        q, k, _ = qkv.unbind(1)
        angles = torch.randn(480, 64, device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        freqs_expanded = freqs.unsqueeze(-2)

        def reference(x):
            value = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
            return torch.view_as_real(value * freqs_expanded).flatten(-2).type_as(x)

        actual_q, actual_k = apply_fused_qk_complex_rope(q, k, freqs)
        atol = 2 * torch.finfo(torch.bfloat16).eps
        torch.testing.assert_close(actual_q, reference(q), rtol=0, atol=atol)
        torch.testing.assert_close(actual_k, reference(k), rtol=0, atol=atol)

    def test_vision_rope_inplace(self):
        # VisionAttention hands the applier contiguous q/k, which is what the
        # in-place kernel requires; mirror that rather than qkv.unbind views.
        for dtype in (torch.bfloat16, torch.float16):
            torch.manual_seed(4)
            q = torch.randn(480, 12, 128, device="cuda", dtype=dtype)
            k = torch.randn(480, 12, 128, device="cuda", dtype=dtype)
            angles = torch.randn(480, 64, device="cuda")
            freqs = torch.polar(torch.ones_like(angles), angles)
            freqs_expanded = freqs.unsqueeze(-2)

            def reference(x):
                value = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
                return torch.view_as_real(value * freqs_expanded).flatten(-2).type_as(x)

            expected_q, expected_k = reference(q), reference(k)
            prepared = prepare_fused_qk_complex_rope_inplace(freqs)
            actual_q, actual_k = apply_fused_qk_complex_rope_inplace(q, k, prepared)

            atol = 2 * torch.finfo(dtype).eps
            torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=atol)
            torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=atol)

    def test_vision_rope_inplace_rejects_non_complex_frequencies(self):
        with self.assertRaises(ValueError):
            prepare_fused_qk_complex_rope_inplace(torch.randn(8, 64, device="cuda"))


class TestVisionRoPEHalf(CustomTestCase):
    """The DeepSeek-V4.1 ViT RoPE: FP32 half-split rotation of [tokens, 16, 64] Q/K."""

    @torch.no_grad()
    def test_half_pairing_and_strided_qkv_match_eager(self):
        """Interleaved pairs, FMA contraction or ignoring QKV strides corrupt Q/K."""
        torch.manual_seed(41)
        for h, w in ((1, 1), (1, 7), (13, 17), (28, 56), (81, 109)):
            for dtype in (torch.bfloat16, torch.float32):
                for strided in (False, True):
                    with self.subTest(grid=(h, w), dtype=dtype, strided=strided):
                        packed = torch.randn(
                            h * w, 3, 16, 64, device="cuda", dtype=dtype
                        )
                        q, k = packed[:, 0], packed[:, 1]
                        if not strided:
                            q, k = q.contiguous(), k.contiguous()
                        cos, sin = (
                            v.cuda() for v in get_vision_cos_sin(h, w, 32, 10000)
                        )
                        old_q, old_k = q.clone(), k.clone()
                        actual_q, actual_k = vision_qk_rope_half(q, k, cos, sin)
                        torch.testing.assert_close(
                            actual_q, apply_rotary(q, cos, sin), rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            actual_k, apply_rotary(k, cos, sin), rtol=0, atol=0
                        )
                        torch.testing.assert_close(q, old_q, rtol=0, atol=0)
                        torch.testing.assert_close(k, old_k, rtol=0, atol=0)

    @torch.no_grad()
    def test_fa3_after_rope_accepts_strided_v(self):
        """Q/K become dense after RoPE; FA3 must read V through its token stride."""
        if torch.cuda.get_device_capability()[0] != 9:
            self.skipTest("FA3 vision path requires Hopper")
        from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func

        torch.manual_seed(42)
        for h, w in ((1, 7), (13, 17), (28, 56)):
            with self.subTest(grid=(h, w)):
                qkv = torch.randn(h * w, 3, 16, 64, device="cuda", dtype=torch.bfloat16)
                q, k, v = qkv.unbind(1)
                cos, sin = (x.cuda() for x in get_vision_cos_sin(h, w, 32, 10000))
                fused_q, fused_k = vision_qk_rope_half(q, k, cos, sin)
                cu = torch.tensor([0, h * w], device="cuda", dtype=torch.int32)
                kwargs = {
                    "cu_seqlens_q": cu,
                    "cu_seqlens_k": cu,
                    "max_seqlen_q": h * w,
                    "max_seqlen_k": h * w,
                    "causal": False,
                }
                expected = flash_attn_varlen_func(
                    apply_rotary(q.contiguous(), cos, sin),
                    apply_rotary(k.contiguous(), cos, sin),
                    v.contiguous(),
                    **kwargs,
                )
                actual = flash_attn_varlen_func(fused_q, fused_k, v, **kwargs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_wrong_frequency_layout(self):
        q = torch.zeros(7, 16, 64, device="cuda", dtype=torch.bfloat16)
        c = torch.ones(7, 1, 64, device="cuda")
        with self.assertRaises(ValueError):
            vision_qk_rope_half(q, q, c, c)


if __name__ == "__main__":
    unittest.main()
