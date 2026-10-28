"""CPU unit tests for the DeepSeek-V4.1 SM90 humming MoE path (dsv41_sm90 "moe")."""

import glob
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layers.moe.moe_runner import humming as humming_runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.humming import (
    HummingDeferredMulSum,
    HummingRunnerCore,
)
from sglang.srt.layers.moe.moe_runner.humming_overlay import (
    OVERLAY_DIRS,
    TABLE_ALIASES,
    install_include_overlay,
    overlay_digest,
    overlay_dir_for,
)
from sglang.srt.layers.moe.moe_runner.humming_tuned_configs import (
    ROLE_VARIANTS,
    TUNED_CONFIG_DIR,
    lookup_tuned_tables,
    tuned_config_path,
    tuned_layer_key,
    validate_tables,
)
from sglang.srt.models.deepseek_v2 import MoEOutput
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

INDEXED = SimpleNamespace(value="indexed")
MAX_ROWS = 1 << 30
BM32, NTAIL = "v0_1_12_bm32", "v0_1_12_ntail"
H20, H20_3E = "NVIDIA H20", "NVIDIA H20-3e"


def _meta(num_experts: int, shape_n: int, shape_k: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_experts=num_experts,
        shape_n=shape_n,
        shape_k=shape_k,
        a_dtype="float8e4m3",
        b_dtype="float4e2m1",
        bs_dtype="float8e8m0",
        input_scale_group_size=0,
        weight_scale_group_size=32,
    )


def _key(num_experts: int, tp: int) -> str:
    # DeepSeek-V4.1-Flash per-rank humming shapes (I = 2304 / tp) with the SM90 pad 64 / 64.
    n13, k2 = {8: (576, 320), 4: (1152, 576)}[tp]
    return tuned_layer_key(
        w13_meta=_meta(num_experts, n13, 5120),
        w2_meta=_meta(num_experts, 5120, k2),
        gemm_type="indexed",
    )


TARGET_TP8, TARGET_TP4 = _key(384, 8), _key(384, 4)
DRAFT_TP8, DRAFT_TP4 = _key(128, 8), _key(128, 4)


def _shipped(device: str, overlay: str, variant=None) -> dict:
    with open(tuned_config_path(device, overlay, variant)) as f:
        return json.load(f)


def _cfg_at(table, rows: int) -> dict:
    return next(cfg for lo, hi, cfg in table if lo < rows <= hi)


def _row_points(*tables) -> list:
    return sorted({r for table in tables for lo, hi, _ in table for r in (lo + 1, hi)})


def _cfg(block_m: int, stream_k: bool) -> dict:
    return {
        "block_shape": [block_m, 128, 128],
        "warp_shape": [block_m, 32, 128],
        "use_stream_k": stream_k,
    }


# Shaped like humming 0.1.12's H20 W4A8 heuristics: stream-K on the decode-size
# gate/up rows and on one large-row range.
W13 = [
    [0, 18, _cfg(8, True)],
    [18, 60, _cfg(8, True)],
    [60, 1578, _cfg(8, False)],
    [1578, 3432, _cfg(16, True)],
    [3432, MAX_ROWS, _cfg(64, False)],
]
W2 = [
    [0, 18, _cfg(8, False)],
    [18, 60, _cfg(8, False)],
    [60, 1578, _cfg(8, False)],
    [1578, 3432, _cfg(16, True)],
    [3432, MAX_ROWS, _cfg(64, False)],
]


def _sm90_moe(on: bool):
    return get_flags().dsv41_sm90.override(
        subsystems=frozenset({"moe"}) if on else frozenset(), role="mix" if on else ""
    )


def _gemm_configs(on: bool, tuned=None) -> dict:
    with _sm90_moe(on):
        runner = HummingRunnerCore(
            MoeRunnerConfig(num_experts=384, num_local_experts=384, activation="silu")
        )
    runner.layer = SimpleNamespace(humming_metas={"w13": None, "w2": None})
    runner._sm90_tuned_tables = lambda gemm_type: tuned
    method = SimpleNamespace(
        get_default_tuning_configs=lambda *, sublayer_name, **_: (
            W13 if sublayer_name == "w13" else W2
        )
    )
    with mock.patch.object(humming_runner, "HummingMethod", method, create=True):
        return runner.get_humming_gemm_configs(INDEXED)


class TestHummingSm90NoStreamK(CustomTestCase):
    """Regression: short prompts (<= ~10 tokens per forward) got wrong first tokens
    while the W4A8 MoE gate/up GEMM ran stream-K; no SM90-path tile may keep it."""

    def test_sm90_moe_configs_lose_stream_k(self):
        # The heuristics, and tuned tables (the shipped ones keep stream-K on some rows).
        tuned = ([row[:2] + [dict(row[2])] for row in W2], W13)
        for configs, refs in (
            (_gemm_configs(on=True), (W13, W2)),
            (_gemm_configs(on=True, tuned=tuned), tuned),
        ):
            for sub, ref in zip(("w13", "w2"), refs):
                table = configs[f"{sub}_tuning_config"]
                self.assertEqual([r[:2] for r in table], [r[:2] for r in ref])
                self.assertFalse(any(cfg["use_stream_k"] for _, _, cfg in table), sub)
                for (_, _, got), (_, _, want) in zip(table, ref):
                    self.assertEqual(dict(got, use_stream_k=want["use_stream_k"]), want)
                # The launcher parses the string, so it must carry the same tables.
                self.assertEqual(json.loads(configs[f"{sub}_tuning_config_str"]), table)
        # The heuristics' own tables are not mutated.
        self.assertTrue(W13[0][2]["use_stream_k"])

    def test_switch_off_keeps_the_heuristics(self):
        configs = _gemm_configs(on=False)
        self.assertEqual(configs["w13_tuning_config"], W13)
        self.assertEqual(configs["w2_tuning_config"], W2)


class _RecordingDeferral(HummingDeferredMulSum):
    def finalize_with_shared_add(self, shared):
        self.shared_seen = shared
        return "merged"


class TestDeferredRoutedSumMerge(CustomTestCase):
    def test_merge_folds_only_the_tp_sharded_shared_output(self):
        # A replicated (TP1) shared output is added after the TP reduction; folding
        # it into every rank's routed sum would count it tp_size times.
        shared = object()
        for replicated, expected in ((False, shared), (True, None)):
            handle = _RecordingDeferral()
            handle.mul_sum_kwargs = {}
            out = MoEOutput(
                routed=handle,
                shared=shared,
                experts=None,
                routed_scaling_factor=2.5,
                shared_is_replicated=replicated,
            )
            self.assertEqual(out.get_merged(), "merged")
            self.assertIs(handle.shared_seen, expected)


class TestHummingTunedTables(CustomTestCase):
    def test_rejects_block_m_mismatch_and_uncovered_rows(self):
        # Both GEMMs read one moe_align_block_size layout built with w13's block_m, and
        # the humming launcher asserts on an uncovered row count.
        def cfg(block_m):
            return {
                "block_shape": [block_m, 128, 128],
                "warp_shape": [block_m, 32, 128],
            }

        full = [[0, 600, cfg(8)], [600, MAX_ROWS, cfg(64)]]
        self.assertIsNone(validate_tables(full, full))
        other_bm = [[0, 500, cfg(8)], [500, MAX_ROWS, cfg(64)]]
        self.assertIn("block_m", validate_tables(full, other_bm))
        gap = [[0, 600, cfg(8)], [700, MAX_ROWS, cfg(64)]]
        self.assertIn("contiguous", validate_tables(gap, full))
        short = [[0, 600, cfg(8)], [600, 4096, cfg(64)]]
        self.assertIn("ends at", validate_tables(full, short))

    def test_shipped_tables_are_valid(self):
        paths = glob.glob(os.path.join(TUNED_CONFIG_DIR, "device_name=*.json"))
        self.assertTrue(paths)
        for path in paths:
            with open(path) as f:
                tables = json.load(f)["tables"]
            self.assertTrue(tables, path)
            for key, entry in tables.items():
                self.assertIsNone(
                    validate_tables(entry["w13"], entry["w2"]), f"{path}: {key}"
                )

    def test_roles_read_the_reference_tables(self):
        # Reference table selection per role and device; other picks change kernel tiles,
        # and thus outputs where the tiles sum K in a different order.
        main = _shipped(H20_3E, BM32)["tables"]
        draft = _shipped(H20_3E, BM32, "draft")["tables"]
        mixdec = _shipped(H20_3E, BM32, "mixdec")["tables"]
        dpdec = _shipped(H20_3E, BM32, "dpdec")["tables"]
        tp8dec = _shipped(H20, BM32, "tp8dec")["tables"]
        ntail = _shipped(H20, NTAIL)["tables"]
        bm32_tables, ntail_tables = (BM32,), (NTAIL, BM32)
        cases = [
            # (role, device, overlay tables, key, expected entry)
            ("mix", H20, bm32_tables, TARGET_TP8, mixdec[TARGET_TP8]),
            ("mix", H20_3E, bm32_tables, TARGET_TP4, mixdec[TARGET_TP4]),
            ("prefill", H20, ntail_tables, TARGET_TP8, ntail[TARGET_TP8]),
            ("prefill", H20, ntail_tables, TARGET_TP4, main[TARGET_TP4]),
            ("prefill", H20_3E, ntail_tables, TARGET_TP8, main[TARGET_TP8]),
            ("decode", H20, bm32_tables, TARGET_TP8, tp8dec[TARGET_TP8]),
            ("decode", H20_3E, bm32_tables, TARGET_TP8, main[TARGET_TP8]),
            ("decode", H20, bm32_tables, TARGET_TP4, dpdec[TARGET_TP4]),
        ]
        cases += [
            (role, device, tables, key, draft[key])
            for role, tables in (
                ("mix", bm32_tables),
                ("prefill", ntail_tables),
                ("decode", bm32_tables),
            )
            for device in (H20, H20_3E)
            for key in (DRAFT_TP8, DRAFT_TP4)
        ]
        for role, device, overlays, key, entry in cases:
            with self.subTest(role=role, device=device, key=key[:22]):
                self.assertEqual(
                    lookup_tuned_tables(
                        key=key, device_name=device, role=role, overlays=overlays
                    ),
                    (entry["w13"], entry["w2"]),
                )
        # Unknown devices and stock headers keep humming's heuristics.
        for device, overlays in (("NVIDIA H200", bm32_tables), (H20, ())):
            self.assertIsNone(
                lookup_tuned_tables(
                    key=TARGET_TP8, device_name=device, role="mix", overlays=overlays
                )
            )

    def test_variant_rows_keep_the_main_k_order(self):
        # A variant re-picks a row's tile only for one without stream-K that sums K in the
        # main tile's order: block K == warp K adds the k32 partials in k order whatever the
        # sizes, a K-split tile keeps its (block K, warp K). Bitwise the same outputs, so mix,
        # prefill and decode servers agree. Outside its re-picked rows it is its base table.
        main = _shipped(H20_3E, BM32)["tables"]
        mixdec = _shipped(H20_3E, BM32, "mixdec")
        derived = [
            (mixdec, main, (0, mixdec["meta"]["decode_rows_max"])),
            (_shipped(H20_3E, BM32, "dpdec"), mixdec["tables"], None),
            (_shipped(H20, BM32, "tp8dec"), main, None),
            (_shipped(H20, NTAIL), main, None),
        ]

        def k_order(cfg):
            block_k, warp_k = cfg["block_shape"][2], cfg["warp_shape"][2]
            return "one" if block_k == warp_k else (block_k, warp_k)

        for variant, base, repicked in derived:
            lo, hi = repicked or variant["meta"]["repicked_rows"]
            for key, entry in variant["tables"].items():
                for sub in ("w13", "w2"):
                    for rows in _row_points(base[key][sub], entry[sub]):
                        ref, cfg = (
                            _cfg_at(base[key][sub], rows),
                            _cfg_at(entry[sub], rows),
                        )
                        msg = f"{variant['meta']['roles']} {key[:22]} {sub} rows {rows}"
                        if not lo < rows <= hi:
                            self.assertEqual(
                                dict(cfg, use_stream_k=False),
                                dict(ref, use_stream_k=False),
                                msg,
                            )
                        elif cfg != ref:
                            self.assertFalse(cfg["use_stream_k"], msg)
                            want = "one" if ref["use_stream_k"] else k_order(ref)
                            self.assertEqual(k_order(cfg), want, msg)


class _StockCompiler:
    @staticmethod
    def include_dirs():
        return ["/stock/include"]

    @staticmethod
    def cuh_last_update_time():
        return "stock-stamp"


class TestHummingHeaderOverlay(CustomTestCase):
    def test_overlay_per_role_and_version(self):
        # The ntail overlay is prefill-only; it reads the bm32 tables after its own.
        for version, dirs in OVERLAY_DIRS.items():
            self.assertTrue(
                overlay_dir_for(version, "prefill").endswith(dirs["prefill"])
            )
            for role in ("mix", "decode"):
                self.assertTrue(
                    overlay_dir_for(version, role).endswith(dirs["default"])
                )
            self.assertEqual(TABLE_ALIASES[dirs["prefill"]], dirs["default"])
        self.assertIsNone(overlay_dir_for("0.0.0", "mix"))
        self.assertEqual(set(ROLE_VARIANTS), {"mix", "prefill", "decode", "encoder"})

    def test_shipped_overlays_match_their_version(self):
        # Headers include each other as <humming/...>, so the copies keep humming's layout;
        # every overlay replaces the same headers and keys its own JIT cache entry.
        def headers(path):
            pattern = os.path.join(path, "**", "*.cuh")
            return sorted(
                os.path.relpath(h, path) for h in glob.glob(pattern, recursive=True)
            )

        for version in OVERLAY_DIRS:
            ntail, bm32 = (overlay_dir_for(version, r) for r in ("prefill", "mix"))
            self.assertEqual(headers(ntail), headers(bm32))
            self.assertNotEqual(overlay_digest(ntail), overlay_digest(bm32))
            for path in (ntail, bm32):
                self.assertEqual(
                    headers(path),
                    [
                        "humming/datatype/dequant_fused.cuh",
                        "humming/kernel/humming.cuh",
                        "humming/mma/wgmma.cuh",
                    ],
                )
                for rel in headers(path):
                    with open(os.path.join(path, rel)) as f:
                        self.assertIn(f"humming-kernels {version}", f.read(200), rel)

    def test_install_prepends_and_keys_jit_cache_on_content(self):
        # humming caches cubins by a signature; overlay edits must not hit stale kernels.
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "humming"))
            header = os.path.join(tmp, "humming", "a.cuh")
            with open(header, "w") as f:
                f.write("// v1\n")
            digest = overlay_digest(tmp)
            compiler = type("Compiler", (_StockCompiler,), {})
            install_include_overlay(compiler, tmp)
            self.assertEqual(compiler.include_dirs(), [tmp, "/stock/include"])
            self.assertEqual(
                compiler.cuh_last_update_time(), f"stock-stamp$$sglang-overlay:{digest}"
            )
            with open(header, "w") as f:
                f.write("// v2\n")
            self.assertNotEqual(overlay_digest(tmp), digest)


if __name__ == "__main__":
    unittest.main()
