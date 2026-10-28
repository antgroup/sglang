"""MN-major block-32 ue8m0 activation quant keeps the production quantizer's bytes."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "SM90 DeepGEMM scale layout",
)
class TestBlock32MnMajorQuant(CustomTestCase):
    def check(self, x, result):
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        q, scales = result
        expected_q, expected_scales = sglang_per_token_group_quant_fp8(
            x, 32, scale_ue8m0=True
        )
        torch.testing.assert_close(
            q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
        self.assertEqual(scales.stride(), (1, (x.shape[0] + 3) // 4 * 4))

    def test_rounding_alignment_and_launch_shapes(self):
        from sglang.kernels.ops.quantization.fp8_block32_act_quant import (
            block32_act_quant_ue8m0_mn_major,
        )

        torch.manual_seed(42)
        for dtype in (torch.bfloat16, torch.float16):
            # Each launch-shape bucket (< 24, < 192, < 2048, >= 2048 rows).
            for m, k in (
                (1, 288),
                (3, 5120),
                (23, 1280),
                (24, 6144),
                (191, 5120),
                (192, 6144),
                (384, 15360),
                (2048, 5120),
            ):
                with self.subTest(dtype=dtype, m=m, k=k):
                    x = torch.randn(m, k, device="cuda", dtype=dtype)
                    # Zero, floor-scale, exact and neighboring scale thresholds.
                    x[:, :32] = 0
                    x[:, 32:64] = 1e-12
                    x[:, 64:96] = 448
                    x[:, 96:128] = -450
                    self.check(x, block32_act_quant_ue8m0_mn_major(x))

    def test_empty_rows_and_graph_replay(self):
        from sglang.kernels.ops.quantization.fp8_block32_act_quant import (
            block32_act_quant_ue8m0_mn_major,
        )

        empty = torch.empty(0, 1280, device="cuda", dtype=torch.bfloat16)
        q, scales = block32_act_quant_ue8m0_mn_major(empty)
        self.assertEqual(q.shape, empty.shape)
        self.assertEqual(scales.shape, (0, 40))
        x = torch.randn(6, 1280, device="cuda", dtype=torch.bfloat16)
        block32_act_quant_ue8m0_mn_major(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = block32_act_quant_ue8m0_mn_major(x)
        for scale in (0.01, 1, 100):
            x.normal_().mul_(scale)
            graph.replay()
            self.check(x, output)


def _qdq_special_blocks(x):
    """Zero and tiny blocks (the 1e-4 amax floor), exact scale boundaries and
    adjacent bf16 values catch floor, scale-rounding and FP8 tie errors."""
    flat = x.view(-1, 32)
    flat[:3] *= 1e-7
    flat[3] = 0
    up = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    for i, v in enumerate((0.875, 1.75, 3.5, 7.0)):
        flat[4 + i] = v
        flat[4 + i, 0] = torch.nextafter(flat[4 + i, 0], up)
    return x


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "SM90 DeepGEMM scale layout",
)
class TestFp8Block32WoAQuant(CustomTestCase):
    """The V4.1 wo_a input rounding (amax floor 1e-4, unlike the 1e-10 above)."""

    def test_round_trip_and_operands_match_the_reference(self):
        from sglang.kernels.ops.quantization.fp8_block32_qdq import (
            fp8_block32_quant_dequant,
            fp8_block32_quant_dequant_native,
            fp8_block32_quant_mn_major,
        )

        torch.manual_seed(1017)
        # 1024+ rows: the reciprocal form of the quotient.
        for rows, groups in ((1, 1), (5, 2), (17, 8), (1024, 1), (2051, 8)):
            for strided in (False, True):
                with self.subTest(rows=rows, groups=groups, strided=strided):
                    # Attention output rows are strided views of padded storage.
                    storage = torch.randn(
                        rows, groups * 4096 + 512, device="cuda", dtype=torch.bfloat16
                    )
                    x = storage[:, : groups * 4096].view(rows, groups, 4096)
                    if not strided:
                        x = x.contiguous()
                    x[0] = _qdq_special_blocks(x[0].contiguous())
                    expected = fp8_block32_quant_dequant_native(x)
                    self.assertFalse(torch.equal(expected, x))
                    torch.testing.assert_close(
                        fp8_block32_quant_dequant(x), expected, rtol=0, atol=0
                    )
                    q, scale = fp8_block32_quant_mn_major(x)
                    self.assertEqual(scale.stride(), (1, (rows + 3) // 4 * 4))
                    dq = q.float().view(rows, -1, 32) * scale[..., None]
                    torch.testing.assert_close(
                        dq.view_as(x).to(x.dtype), expected, rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
