"""Routing of SM90 block-[32, 32] FP8 linears under DeepSeek-V4.1 ``fp8_gemm``."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.gemm.fp8_block32_swapab import (
    SwapABConfig,
    load_swapab_table,
    lookup_swapab,
    parse_swapab_table,
)
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import _dsv41_block32_routes
from sglang.srt.layers.quantization.fp8_block32_sm90 import (
    _operands_fit,
    _prefers_triton,
    load_block32_routes,
)
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

CFG = {"bt1": 32, "bt2": 16, "stages": 2, "split": 1, "maxnreg": 64}
SPLIT_CFG = {"bt1": 64, "bt2": 0, "stages": 3, "split": 4, "maxnreg": 80}


def _split(table, n, k, m, groups=1):
    # Output bits depend on the split count only; DeepGEMM is split 1.
    cfg = lookup_swapab(table, n, k, m, groups)
    return 1 if cfg is None else cfg.split


class TestSwapABTable(CustomTestCase):
    def test_row_count_rounds_up_and_null_keeps_deepgemm(self):
        table = parse_swapab_table(
            {"N=64,K=5120": {"24": CFG, "48": None, "96": SPLIT_CFG}}
        )
        expected = {1: CFG, 24: CFG, 25: None, 48: None, 49: SPLIT_CFG, 96: SPLIT_CFG}
        for m, cfg in expected.items():
            with self.subTest(m=m):
                got = lookup_swapab(table, 64, 5120, m)
                self.assertEqual(got, None if cfg is None else SwapABConfig(**cfg))
        for n, k, m in ((64, 5120, 97), (64, 5120, 0), (5120, 64, 24)):
            self.assertIsNone(lookup_swapab(table, n, k, m))

    def test_parse_rejects_unlaunchable_configs(self):
        bad = [
            # Gluon tensors are power-of-two shaped; wgmma N is at least 8.
            {"N=64,K=5120": {"8": dict(CFG, bt1=48)}},
            {"N=64,K=5120": {"8": dict(CFG, bt2=4)}},
            # K = 288 has three 128-wide blocks to split.
            {"N=64,K=288": {"8": dict(CFG, split=4)}},
            # Grouped: split-K reorders the per-group K sum, a 64-row weight tile
            # must not straddle two groups, and a group's K slice starts on a TMA block.
            {"N=1024,K=4096,G=8": {"8": dict(CFG, split=2)}},
            {"N=1056,K=4096,G=8": {"8": CFG}},
            {"N=1024,K=4064,G=8": {"8": CFG}},
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_swapab_table(raw)

    def test_grouped_key_routes_only_the_grouped_lookup(self):
        table = parse_swapab_table({"N=1024,K=4096,G=8": {"192": CFG}})
        cfg = SwapABConfig(**CFG)
        self.assertEqual(lookup_swapab(table, 1024, 4096, 192, groups=8), cfg)
        self.assertIsNone(lookup_swapab(table, 1024, 4096, 192))
        self.assertIsNone(lookup_swapab(table, 1024, 4096, 192, groups=4))


class TestShippedTables(CustomTestCase):
    def test_role_tables_change_speed_not_bits(self):
        # A row a role table changes keeps the split count of the row it replaces
        # (or replaces DeepGEMM with split 1), at every row count.
        for device in ("NVIDIA H20", "NVIDIA H20-3e"):
            base = load_swapab_table(device, "prefill")
            for role in ("mix", "decode"):
                table = load_swapab_table(device, role)
                for n, k, *groups in table:
                    with self.subTest(device=device, role=role, n=n, k=k):
                        self.assertEqual(
                            [_split(table, n, k, m, *groups) for m in range(1, 1025)],
                            [_split(base, n, k, m, *groups) for m in range(1, 1025)],
                        )

    def test_device_and_role_select_the_measured_rows(self):
        # wq_b under DP attention / the head all-to-all at 24 rows per rank.
        rows = {
            (device, role): lookup_swapab(
                load_swapab_table(device, role), 32768, 1280, 24
            )
            for device in ("NVIDIA H20", "NVIDIA H20-3e", "NVIDIA H200")
            for role in ("mix", "prefill", "decode")
        }
        prefill = rows["NVIDIA H20", "prefill"]
        self.assertIsNotNone(prefill)
        # Only H20 has decode rows; H20 reads the H20-3e base and mix tables.
        self.assertNotIn(rows["NVIDIA H20", "decode"], (None, prefill))
        self.assertNotIn(rows["NVIDIA H20", "mix"], (None, prefill))
        self.assertEqual(rows["NVIDIA H20-3e", "decode"], prefill)
        self.assertEqual(rows["NVIDIA H20-3e", "mix"], rows["NVIDIA H20", "mix"])
        for role in ("mix", "prefill", "decode"):
            self.assertIsNone(rows["NVIDIA H200", role])


class TestTritonPreference(CustomTestCase):
    def test_threshold_shape_class_and_device(self):
        h20 = load_block32_routes(device_name="NVIDIA H20", role="mix")
        cases = [
            # Narrow N / long K up to 16 rows, unless the swap-AB table lists the row.
            ((16, 1536, 5120), True),
            ((17, 1536, 5120), False),
            ((6, 1792, 5120), False),
            ((16, 2048, 4096), True),
            ((1, 2080, 4096), False),
            ((1, 2048, 4064), False),
            # Short K.
            ((1, 64, 512), True),
            ((1, 64, 544), False),
            ((1, 4096, 1280), False),
        ]
        for (m, n, k), expected in cases:
            with self.subTest(m=m, n=n, k=k):
                self.assertEqual(_prefers_triton(routes=h20, m=m, n=n, k=k), expected)
        # An unmeasured device keeps DeepGEMM at every row count.
        h200 = load_block32_routes(device_name="NVIDIA H200", role="mix")
        self.assertFalse(_prefers_triton(routes=h200, m=1, n=1536, k=5120))


class TestDispatch(CustomTestCase):
    def _dispatch(self, routes):
        with (
            mock.patch.object(
                fp8_utils,
                "get_platform",
                return_value=SimpleNamespace(is_sm90=True, is_sm100=False),
            ),
            mock.patch.object(fp8_utils.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
        ):
            return fp8_utils.dispatch_w8a8_block_fp8_linear(
                weight_block_size=[32, 32], act_scale_ue8m0=True, block32_routes=routes
            )

    @mock.patch("torch.cuda.get_device_name", return_value="NVIDIA H20")
    def test_routes_only_for_ue8m0_block32_with_the_subsystem_on(self, _):
        self.assertIsNone(
            _dsv41_block32_routes(weight_block_size=[32, 32], act_scale_ue8m0=True)
        )
        self.assertEqual(self._dispatch(None).keywords, {"act_scale_ue8m0": True})
        with get_flags().dsv41_sm90.override(
            subsystems=frozenset({"fp8_gemm"}), role="mix"
        ):
            routes = _dsv41_block32_routes(
                weight_block_size=[32, 32], act_scale_ue8m0=True
            )
            for block, ue8m0 in (([128, 128], True), ([32, 32], False)):
                self.assertIsNone(
                    _dsv41_block32_routes(
                        weight_block_size=block, act_scale_ue8m0=ue8m0
                    )
                )
        self.assertIs(routes, load_block32_routes(device_name="NVIDIA H20", role="mix"))
        linear = self._dispatch(routes)
        self.assertIs(
            linear.func, fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
        )
        self.assertIs(linear.keywords["block32_routes"], routes)


class TestSwapABOperandGuard(CustomTestCase):
    """The kernel reads MN-major activation scales and K-major weight scales
    through TMA; any other layout must keep DeepGEMM."""

    def _operands(self, m=48, n=64, k=256):
        q = torch.zeros(m, k, dtype=torch.float8_e4m3fn)
        xs = torch.zeros(k // 32, m + (-m % 4)).t()[:m]
        w = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
        ws = torch.zeros((n + 31) // 32, k // 32)
        return q, xs, w, ws

    def test_scale_layouts(self):
        q, xs, w, ws = self._operands()
        self.assertTrue(_operands_fit(q, xs, w, ws, out=None))
        self.assertFalse(_operands_fit(q, xs.contiguous(), w, ws, out=None))
        self.assertFalse(_operands_fit(q, xs, w, ws.t().contiguous().t(), out=None))


if __name__ == "__main__":
    unittest.main()
