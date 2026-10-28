"""DeepSeek-V4.1 ViT SwiGLU parity with the eager silu-then-multiply."""

import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.activation.vision_swiglu import vision_swiglu
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestVisionSwiGLU(CustomTestCase):
    @torch.no_grad()
    def test_preserves_intermediate_silu_rounding(self):
        """A usual fused SwiGLU misses the eager SiLU store to the input dtype."""
        torch.manual_seed(42)
        for rows in (0, 1, 7, 8, 163, 1568):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(rows=rows, dtype=dtype):
                    x = torch.randn(rows, 5632, device="cuda", dtype=dtype) * 5
                    gate, up = x.chunk(2, -1)
                    torch.testing.assert_close(
                        vision_swiglu(x), F.silu(gate) * up, rtol=0, atol=0
                    )

    def test_rejects_nonpacked_input(self):
        x = torch.empty(8, 5632, device="cuda", dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            vision_swiglu(x[:, ::2])


if __name__ == "__main__":
    unittest.main()
