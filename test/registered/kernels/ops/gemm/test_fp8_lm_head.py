"""FP8 LM head (DeepSeek-V4.1 ``lm_head``): a row's logits must not depend on how
the batch is split into DeepGEMM launches or on the swap-AB kernel taking the
batch instead, the load-time quantization must stay within e4m3 rounding of the
bf16 head across its staging steps and padded vocab tail, and re-quantization
must keep the buffers CUDA graphs captured."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# Large enough to take the 128 + tail split; not a multiple of 128 (padded tail block)
# and spanning several 8192-row quantization steps.
_N = 65600
_K = 512


def _head(weight, *, use_swapab=False):
    from sglang.srt.layers.quantization.fp8_lm_head import (
        Fp8LmHead,
        fp8_lm_head_supported,
    )

    if not fp8_lm_head_supported():
        raise unittest.SkipTest("needs SM90 DeepGEMM with fp32 scales")
    return Fp8LmHead(weight, use_swapab=use_swapab)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "SM90 DeepGEMM",
)
class TestFp8LmHead(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        # Row norms spread over 2^-4..2^4 so a misplaced block scale shows up.
        row_scale = torch.exp2(torch.empty(_N, 1).uniform_(-4, 4))
        self.weight = (torch.randn(_N, _K) * row_scale).to(torch.bfloat16).cuda()

    def test_split_launches_match_single_row_launches(self):
        from sglang.srt.layers.quantization.fp8_lm_head import split_rows

        head = _head(self.weight)
        self.assertEqual(split_rows(192, head.weight.shape[0]), [(0, 128), (128, 192)])
        x = torch.randn(192, _K, device="cuda").to(torch.bfloat16)
        batched = head(x)
        self.assertEqual(tuple(batched.shape), (192, _N))
        for row in (0, 77, 127, 128, 160, 191):
            with self.subTest(row=row):
                single = head(x[row : row + 1])
                torch.testing.assert_close(
                    batched[row : row + 1], single, rtol=0, atol=0
                )

    def test_swapab_launch_is_bitwise_the_deepgemm_path(self):
        from sglang.srt.layers.quantization.fp8_lm_head import swapab_tile

        deepgemm = _head(self.weight)
        swapab = _head(self.weight, use_swapab=True)
        # Every tile family (one piece, 32/64/128 + tail), padded and exact tiles.
        for m in (1, 12, 20, 36, 48, 70, 80, 96, 132, 150, 160):
            with self.subTest(m=m):
                self.assertIsNotNone(swapab_tile(m))
                x = torch.randn(m, _K, device="cuda").to(torch.bfloat16)
                torch.testing.assert_close(swapab(x), deepgemm(x), rtol=0, atol=0)

    def test_quantization_error_is_e4m3_rounding(self):
        head = _head(self.weight)
        x = torch.randn(48, _K, device="cuda").to(torch.bfloat16)
        got = head(x).float()
        ref = x.float() @ self.weight.float().T
        # e4m3 rounding of both operands gives ~3% per vocab row; a misplaced block
        # scale (8192-row step edges, padded tail block) gives O(1).
        rel = (got - ref).norm(dim=0) / ref.norm(dim=0)
        self.assertLess(rel.max().item(), 0.1)

    def test_requantize_keeps_buffers_and_empty_batch(self):
        head = _head(self.weight)
        ptrs = (head.weight.data_ptr(), head.weight_scale.data_ptr())
        x = torch.randn(8, _K, device="cuda").to(torch.bfloat16)
        before = head(x)
        head.requantize_(-self.weight)
        self.assertEqual((head.weight.data_ptr(), head.weight_scale.data_ptr()), ptrs)
        torch.testing.assert_close(head(x), -before, rtol=0, atol=0)
        # DP-attention idle ranks run the LM head on zero rows.
        empty = head(torch.empty(0, _K, device="cuda", dtype=torch.bfloat16))
        self.assertEqual(tuple(empty.shape), (0, _N))


if __name__ == "__main__":
    unittest.main()
