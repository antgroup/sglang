"""DeepSeek-V4.1 image span packing parity with the token-type scatter."""

import unittest

import torch

from sglang.kernels.ops.embeddings.vision_span import pack_vision_span
from sglang.srt.multimodal.deepseek_v41_image_processing import image_token_types
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestVisionSpan(CustomTestCase):
    def test_reading_order_and_row_delimiters(self):
        """Pin the external image layout, including single-column/tail grids."""
        for h, w in ((1, 1), (1, 7), (7, 1), (13, 17), (29, 31)):
            for dim in (37, 5120):
                with self.subTest(grid=(h, w), dim=dim):
                    features = torch.arange(
                        h * w * dim, device="cuda", dtype=torch.float32
                    ).reshape(-1, dim)
                    start, line, end = [
                        torch.full((dim,), v, device="cuda") for v in (-1.0, -2.0, -3.0)
                    ]
                    tags = image_token_types(h, w).cuda()
                    expected = torch.empty(len(tags), dim, device="cuda")
                    expected[tags == 0] = start
                    expected[tags == 1] = features
                    expected[tags == 2] = line
                    expected[tags == 3] = end
                    out = pack_vision_span(features, start, line, end, h, w)
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    # The span must own its storage, apart from the cached features.
                    features.zero_()
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_rejects_wrong_grid(self):
        f = torch.empty(4, 32, device="cuda")
        v = f[0]
        with self.assertRaises(ValueError):
            pack_vision_span(f, v, v, v, 3, 2)


if __name__ == "__main__":
    unittest.main()
