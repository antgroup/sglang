"""DeepSeek-V4.1 SM90 FP8 block-[32, 32] grouped wo_a on its rounded input operands."""

import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _operands(rows, groups):
    x = torch.randn(rows, groups, 4096, device="cuda", dtype=torch.bfloat16)
    # Per-32-group exponents spread over 8 octaves, as in real activations.
    spread = torch.randint(-4, 5, (rows, groups, 128, 1), device="cuda")
    x = x * torch.exp2(spread.float()).expand(-1, -1, -1, 32).reshape(x.shape)
    weight = (torch.randn(groups * 1024, 4096, device="cuda") * 16).to(
        torch.float8_e4m3fn
    )
    scale = torch.exp2(
        torch.randint(-11, -6, (groups * 32, 128), device="cuda").float()
    )
    return x.to(torch.bfloat16), weight, scale


def _routes(swapab):
    from sglang.srt.layers.quantization.fp8_block32_sm90 import Block32Routes

    return Block32Routes(swapab=swapab, triton_small_m={}, triton_max_m=0)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "Hopper block32 DeepGEMM",
)
class TestDsv41WoAFp8Block32(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.deep_gemm_wrapper import compile_utils

        if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            raise unittest.SkipTest("needs JIT DeepGEMM")
        # Compile only the shapes used here, not every M up to the prefill size.
        cls._no_precompile = mock.patch.object(
            compile_utils, "_ENABLE_JIT_DEEPGEMM_PRECOMPILE", False
        )
        cls._no_precompile.start()
        cls.streams = tuple(torch.cuda.Stream() for _ in range(4))

    @classmethod
    def tearDownClass(cls):
        cls._no_precompile.stop()

    def _project(self, x, weight, scale, routes):
        from sglang.kernels.ops.quantization.fp8_block32_qdq import (
            fp8_block32_quant_mn_major,
        )
        from sglang.srt.layers.attention.dsv4.wo_a_fp8_block32 import (
            wo_a_fp8_block32_gemm,
        )

        q, xs = fp8_block32_quant_mn_major(x)
        return wo_a_fp8_block32_gemm(
            o_fp8=q,
            o_scale=xs,
            weight=weight,
            weight_scale=scale,
            num_groups=x.shape[1],
            routes=routes,
            streams=self.streams,
        )

    @staticmethod
    def _reference(x, weight, scale):
        from sglang.kernels.ops.quantization.fp8_block32_qdq import (
            fp8_block32_quant_dequant_native,
        )

        groups = x.shape[1]
        w = weight.float().view(groups * 32, 32, 128, 32) * scale[:, None, :, None]
        w = w.view(groups, 1024, 4096).to(torch.bfloat16)
        return torch.einsum("tgd,grd->tgr", fp8_block32_quant_dequant_native(x), w)

    def test_every_route_projects_the_rounded_input(self):
        from sglang.srt.layers.quantization.fp8_block32_sm90 import (
            load_block32_routes,
        )

        torch.manual_seed(2026)
        device_routes = load_block32_routes(
            device_name=torch.cuda.get_device_name(), role="mix"
        )
        # Split-K Triton (<= 16 rows), the grouped / per-group swap-AB rows the
        # device lists, and per-group DeepGEMM over side streams.
        for routes in (device_routes, _routes({})):
            for rows, groups in (
                (1, 1),
                (6, 2),
                (16, 8),
                (17, 1),
                (96, 1),
                (96, 8),
                (320, 2),
                (1024, 8),
            ):
                with self.subTest(rows=rows, groups=groups, tables=bool(routes.swapab)):
                    x, w, ws = _operands(rows, groups)
                    got = self._project(x, w, ws, routes)
                    # FP32 accumulation per 32-wide block differs from the bf16 GEMM
                    # mainly around cancellation; the operands are the same.
                    torch.testing.assert_close(
                        got, self._reference(x, w, ws), rtol=1.6e-2, atol=1e-2
                    )

    def test_grouped_swapab_launch_is_bitwise_the_per_group_deepgemm(self):
        # One launch offsets each group's token K slice; any slip reads a
        # neighbouring group's activations or scales and changes the bits.
        from sglang.kernels.ops.gemm.fp8_block32_swapab import SwapABConfig
        from sglang.srt.layers import deep_gemm_wrapper

        torch.manual_seed(1022)
        configs = (
            SwapABConfig(bt1=32, bt2=16, stages=2, split=1, maxnreg=64),
            SwapABConfig(bt1=64, bt2=0, stages=2, split=1, maxnreg=80),
        )
        for rows, cfg in ((96, configs[0]), (150, configs[1]), (192, configs[1])):
            with self.subTest(rows=rows, cfg=cfg):
                x, w, ws = _operands(rows, 8)
                expected = self._project(x, w, ws, _routes({}))
                grouped = _routes({(1024, 4096, 8): ((rows, cfg),)})
                with mock.patch.object(
                    deep_gemm_wrapper,
                    "gemm_nt_f8f8bf16",
                    side_effect=AssertionError("per-group DeepGEMM ran"),
                ):
                    got = self._project(x, w, ws, grouped)
                torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_side_stream_graph_replay_reads_new_input(self):
        for rows, groups in ((6, 8), (33, 2), (1024, 8)):
            with self.subTest(rows=rows, groups=groups):
                x, w, ws = _operands(rows, groups)
                routes = _routes({})
                for _ in range(2):
                    self._project(x, w, ws, routes)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = self._project(x, w, ws, routes)
                for factor in (1e-3, 16):
                    x.normal_().mul_(factor)
                    graph.replay()
                    fresh = self._project(x, w, ws, routes)
                    torch.testing.assert_close(captured, fresh, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
