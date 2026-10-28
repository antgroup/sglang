"""Engram gate pack + apply kernels (the Engram TP split) against the fused gate."""

import itertools
import unittest

import torch

from sglang.kernels.ops.embeddings.engram_gate import (
    engram_gate_pack_width,
    fused_engram_gate,
    fused_engram_gate_apply,
    fused_engram_gate_pack,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HC, DIM, IMAGE_ID = 4, 5120, 129264


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


class TestEngramGatePackApply(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("needs CUDA")

    def test_split_pack_then_apply_is_bitwise_fused(self):
        """The TP split packs [value | fp32 gates] rows for row slices, from kv whole
        or in column chunks [chunks, rows (+ padding), cols / chunks], and applies them
        to all rows; that must equal the fused gate bit for bit, image rows included."""
        torch.manual_seed(29)
        cases = itertools.product(
            (torch.bfloat16, torch.float32), (1, 8, 384), (1, 4, 8), (False, True)
        )
        for weight_dtype, batch, chunks, images in cases:
            with self.subTest(
                weights=weight_dtype, batch=batch, chunks=chunks, images=images
            ):
                x = torch.randn(batch, HC, DIM, device="cuda").bfloat16() * 3
                kv = torch.randn(batch, (HC + 1) * DIM, device="cuda").bfloat16()
                x[0], kv[-1] = 0, 0  # zero graph-padding rows
                x[-1, :, ::7] = -0.0
                qw = torch.randn(HC, DIM, device="cuda", dtype=weight_dtype)
                kw = torch.randn_like(qw)
                ids = None
                if images:
                    ids = torch.randint(0, 1000, (batch,), device="cuda")
                    ids[::3] = IMAGE_ID
                    ids[-1] = IMAGE_ID
                width = engram_gate_pack_width(HC, DIM, kv.dtype)
                packed = torch.empty(batch, width, device="cuda", dtype=kv.dtype)
                for start in range(0, batch, 7):
                    rows = slice(start, start + 7)
                    n = x[rows].shape[0]
                    # Column chunks of this slice, with one spare padding row as an
                    # all-to-all receive buffer holds.
                    part = torch.zeros(chunks, n + 1, kv.shape[1] // chunks).to(kv)
                    part[:, :n] = kv[rows].view(n, chunks, -1).transpose(0, 1)
                    fused_engram_gate_pack(
                        x[rows],
                        part.squeeze(0) if chunks == 1 else part,
                        qw,
                        kw,
                        1e-20,
                        1e-6,
                        out=packed[rows],
                    )
                out = fused_engram_gate_apply(x, packed, ids, IMAGE_ID)
                ref = fused_engram_gate(x, kv, qw, kw, 1e-20, 1e-6, ids, IMAGE_ID)
                self.assertTrue(torch.equal(_bits(out), _bits(ref)))
                if images:
                    masked = ids == IMAGE_ID
                    self.assertTrue(torch.equal(_bits(out[masked]), _bits(x[masked])))


if __name__ == "__main__":
    unittest.main()
