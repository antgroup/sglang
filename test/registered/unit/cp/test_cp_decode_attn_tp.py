"""CP decode attention TP: every weight-derived tensor a sliced linear's GEMM reads
must describe the same slice as its weight."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpContext
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    triton_w8a8_block_fp8_linear,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

BLOCK = [32, 32]


def _block32_linear(cls, n, k, gen):
    """A block32 ue8m0 FP8 linear carrying the Hopper BF16 weight cache, as
    Fp8LinearMethod.process_weights_after_loading_block_quant leaves it."""
    linear = cls.__new__(cls)
    torch.nn.Module.__init__(linear)
    weight = (torch.randn(n, k, device="cuda", generator=gen) * 64).to(
        torch.float8_e4m3fn
    )
    exponents = torch.randint(-10, -4, (n // 32, k // 32), device="cuda", generator=gen)
    scale = torch.pow(2.0, exponents.float())
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    linear.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
    linear.register_buffer(
        "_block_fp8_bf16_weight",
        block_quant_dequant(weight, scale, BLOCK, torch.bfloat16),
        persistent=False,
    )
    linear.input_size_per_partition = k
    linear.output_size_per_partition = n
    linear.use_decode_attn_tp = False
    return linear


@unittest.skipUnless(torch.cuda.is_available(), "Triton block32 FP8 GEMM")
class TestCpDecodeAttnTpBf16WeightCache(CustomTestCase):
    def test_large_batch_gemm_reads_the_sliced_weight(self):
        """With a [32, 32] block-FP8 weight the SM90 linear runs batches of 64+ rows
        on a BF16 copy of the weight; under CP decode attention TP that copy must
        be sliced like the FP8 weight, or the GEMM reads the full weight."""
        tp_size, rank, n, k, rows = 4, 1, 256, 512, 64
        gen = torch.Generator(device="cuda").manual_seed(0)
        ctx = CpDecodeAttnTpContext.__new__(CpDecodeAttnTpContext)
        ctx.decode_tp_rank, ctx.decode_tp_size = rank, tp_size
        ctx.use_decode_attn_tp = False
        ctx._slice_cache = {}

        def decode(_forward_batch):
            ctx.use_decode_attn_tp = True

        for cls, in_features in (
            (ColumnParallelLinear, k),
            (RowParallelLinear, k // tp_size),
        ):
            with self.subTest(cls=cls.__name__):
                linear = _block32_linear(cls, n, k, gen)
                full_cache = linear._block_fp8_bf16_weight
                x = torch.randn(rows, in_features, device="cuda", generator=gen)
                x = x.bfloat16()
                with (
                    patch.object(ctx, "set_decode_attn_tp", side_effect=decode),
                    ctx.maybe_use_decode_attn_tp(None, [linear]),
                ):
                    args = dict(
                        input=x,
                        weight=linear.weight,
                        block_size=BLOCK,
                        weight_scale=linear.weight_scale_inv,
                        act_scale_ue8m0=True,
                    )
                    fp8 = triton_w8a8_block_fp8_linear(**args)
                    cached = triton_w8a8_block_fp8_linear(
                        **args, weight_bf16=linear._block_fp8_bf16_weight
                    )
                torch.testing.assert_close(cached, fp8, rtol=2e-2, atol=1e-1)
                self.assertIs(linear._block_fp8_bf16_weight, full_cache)


if __name__ == "__main__":
    unittest.main()
