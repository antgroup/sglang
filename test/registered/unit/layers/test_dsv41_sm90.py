"""CPU unit tests for the DeepSeek-V4.1 SM90 optimization switch."""

import unittest

from sglang.srt.layers.dsv41_sm90 import (
    SUBSYSTEMS,
    dsv41_sm90_lossy,
    dsv41_sm90_on,
    resolve_dsv41_sm90,
)
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _resolve(**overrides):
    kwargs = dict(
        enabled=True,
        model_type="deepseek_v41",
        sm90=True,
        disable=(),
        lossy=(),
        disaggregation_mode="null",
        encoder_only=False,
    )
    kwargs.update(overrides)
    return resolve_dsv41_sm90(**kwargs)


class TestResolveDsv41Sm90(CustomTestCase):
    def test_on_selects_every_subsystem_and_no_lossy_option(self):
        selection = _resolve()
        self.assertEqual(selection.subsystems, frozenset(SUBSYSTEMS))
        self.assertEqual(selection.lossy, frozenset())
        self.assertEqual(selection.role, "mix")

    def test_scope_outside_v41_on_sm90_selects_nothing(self):
        # The switch must never change another model or another architecture.
        for overrides in (
            dict(enabled=False),
            dict(model_type="deepseek_v4"),
            dict(model_type="glm5"),
            dict(sm90=False),
            dict(enabled=False, lossy=("fp8_lm_head_target",)),
        ):
            selection = _resolve(**overrides)
            self.assertEqual(selection.subsystems, frozenset(), overrides)
            self.assertEqual(selection.lossy, frozenset(), overrides)
            self.assertEqual(selection.role, "", overrides)

    def test_disable_and_lossy_lists(self):
        selection = _resolve(disable=("mhc", "cp"), lossy=("fp8_lm_head_target",))
        self.assertEqual(selection.subsystems, frozenset(SUBSYSTEMS) - {"mhc", "cp"})
        self.assertEqual(selection.lossy, frozenset({"fp8_lm_head_target"}))

    def test_off_ignores_unknown_names(self):
        """Unused optimization lists must not make a switch-off startup fail."""
        selection = _resolve(enabled=False, disable=("mch",), lossy=("mhc",))
        self.assertEqual(selection, _resolve(enabled=False))

    def test_unknown_names_fail_when_on(self):
        with self.assertRaisesRegex(ValueError, "SGLANG_OPT_DSV41_SM90_DISABLE"):
            _resolve(disable=("mch",))
        with self.assertRaisesRegex(ValueError, "SGLANG_OPT_DSV41_SM90_LOSSY"):
            _resolve(lossy=("mhc",))

    def test_role_from_disaggregation(self):
        for mode, encoder_only, role in (
            ("prefill", False, "prefill"),
            ("decode", False, "decode"),
            ("null", True, "encoder"),
        ):
            selection = _resolve(disaggregation_mode=mode, encoder_only=encoder_only)
            self.assertEqual(selection.role, role)


class TestDsv41Sm90Accessors(unittest.TestCase):
    def test_accessors_read_the_flags_and_reject_unknown_names(self):
        with get_flags().dsv41_sm90.override(
            subsystems=frozenset({"moe"}), lossy=frozenset({"fp8_lm_head_target"})
        ):
            self.assertTrue(dsv41_sm90_on("moe"))
            self.assertFalse(dsv41_sm90_on("mhc"))
            self.assertTrue(dsv41_sm90_lossy("fp8_lm_head_target"))
        self.assertFalse(dsv41_sm90_on("moe"))
        self.assertFalse(dsv41_sm90_lossy("fp8_lm_head_target"))
        with self.assertRaises(AssertionError):
            dsv41_sm90_on("moes")


if __name__ == "__main__":
    unittest.main()
