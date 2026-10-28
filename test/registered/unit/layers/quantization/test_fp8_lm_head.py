"""Which logits take the FP8 LM head under DeepSeek-V4.1 ``lm_head``, and the
swap-AB tiles it launches."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8_lm_head
from sglang.srt.layers.quantization.fp8_lm_head import (
    Fp8LmHeadUse,
    resolve_fp8_lm_head_use,
    swapab_launch,
    swapab_tile,
)
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_ALL = frozenset({"lm_head"})


def _resolve(
    *,
    subsystems=_ALL,
    lossy=frozenset(),
    role="decode",
    device="NVIDIA H20",
    spec="DSPARK",
    fp32_draft=False,
    enable_lora=False,
):
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                fp8_lm_head,
                "get_lora",
                return_value=SimpleNamespace(enable_lora=enable_lora),
            )
        )
        stack.enter_context(
            get_flags().dsv41_sm90.override(
                subsystems=subsystems, lossy=lossy, role=role
            )
        )
        stack.enter_context(
            mock.patch("torch.cuda.get_device_name", return_value=device)
        )
        stack.enter_context(
            mock.patch.object(fp8_lm_head, "fp8_lm_head_supported", return_value=True)
        )
        stack.enter_context(
            mock.patch.object(
                fp8_lm_head,
                "get_spec",
                return_value=SimpleNamespace(speculative_algorithm=spec),
            )
        )
        stack.enter_context(envs.SGLANG_DSPARK_FP32_LM_HEAD.override(fp32_draft))
        return resolve_fp8_lm_head_use()


class TestResolve(CustomTestCase):
    def test_lora_keeps_target_and_draft_off_the_fp8_head(self):
        """LoRA-wrapped heads have no fp8_head and must keep the LoRA logits path."""
        for lossy in (frozenset(), frozenset({"fp8_lm_head_target"})):
            with (
                self.subTest(lossy=lossy),
                self.assertLogs(fp8_lm_head.logger, "INFO") as logs,
            ):
                use = _resolve(enable_lora=True, lossy=lossy)
            self.assertEqual(use, Fp8LmHeadUse())
            self.assertIn("LoRA is enabled", "\n".join(logs.output))

    def test_draft_and_swapab_on_with_the_subsystem(self):
        for role in ("mix", "decode"):
            with self.subTest(role=role):
                self.assertEqual(
                    _resolve(role=role),
                    Fp8LmHeadUse(target=False, draft=True, swapab=True),
                )

    def test_target_only_with_the_lossy_option(self):
        lossy = frozenset({"fp8_lm_head_target"})
        self.assertEqual(
            _resolve(lossy=lossy), Fp8LmHeadUse(target=True, draft=True, swapab=True)
        )
        # Disabling the subsystem keeps the opted-in target head, on DeepGEMM only.
        self.assertEqual(
            _resolve(subsystems=frozenset(), lossy=lossy),
            Fp8LmHeadUse(target=True, draft=False, swapab=False),
        )

    def test_bf16_head_stays_everywhere_else(self):
        off = Fp8LmHeadUse()
        cases = {
            # Switch off (empty set, no role), or the subsystem disabled.
            "switch off": dict(subsystems=frozenset(), role=""),
            "disabled": dict(subsystems=frozenset()),
            # Roles without measured decode logits.
            "prefill": dict(role="prefill"),
            "encoder": dict(role="encoder"),
            # Thresholds and launch configs were measured on H20 only.
            "H200": dict(device="NVIDIA H200"),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                self.assertEqual(_resolve(**kwargs), off)
        # No DSpark draft, or an FP32 draft head asked for: no draft copy.
        self.assertFalse(_resolve(spec=None).draft)
        self.assertFalse(_resolve(spec="EAGLE").draft)
        self.assertFalse(_resolve(fp32_draft=True).draft)


class TestSwapABTiles(CustomTestCase):
    def test_tile_covers_the_batch_below_deepgemm_padding(self):
        # Gluon tensors are power-of-two shaped and wgmma N is at least 8; the tile
        # must cover the batch in one launch and pad less than DeepGEMM's 64/128/256.
        for m in range(1, 385):
            tile = swapab_tile(m)
            if tile is None:
                continue
            with self.subTest(m=m):
                bt1, bt2 = tile
                for bt in (bt1, bt2) if bt2 else (bt1,):
                    self.assertTrue(8 <= bt <= 128 and bt & (bt - 1) == 0, tile)
                self.assertGreaterEqual(bt1 + bt2, m)
                deepgemm_rows = 64 if m <= 64 else (128 if m <= 128 else 256)
                self.assertLess(bt1 + bt2, deepgemm_rows)
                # Every tile has a launch config (a missing one raises at serve time).
                stages, maxnreg = swapab_launch(tile)
                self.assertIn(stages, (3, 4))
                self.assertLessEqual(maxnreg, 128)


if __name__ == "__main__":
    unittest.main()
