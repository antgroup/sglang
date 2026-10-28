"""CPU unit tests for the DeepSeek-V4.1 SM90 ``attn_proj`` plan of an attention layer."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.models import deepseek_v4_attn_proj as ap
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _fp8(block=(32, 32), scale_fmt="ue8m0"):
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=list(block),
        scale_fmt=scale_fmt,
    )


class TestNextNPostLoad(CustomTestCase):
    def test_nextn_post_load_without_fp8_lm_head_use(self):
        """NextN post-load must succeed without target-only FP8 head settings."""
        import torch

        from sglang.srt.models.deepseek_v4_nextn import DeepseekV4ForCausalLMNextN

        model = DeepseekV4ForCausalLMNextN.__new__(DeepseekV4ForCausalLMNextN)
        torch.nn.Module.__init__(model)
        model.wo_a_fp8 = False
        model.logits_processor = SimpleNamespace(use_fp8_lm_head=False)
        model.pp_group = SimpleNamespace(is_last_rank=True)
        with get_flags().dsv41_sm90.override(subsystems=frozenset()):
            self.assertIsNone(model.post_load_weights())


class TestAttnProjPlan(CustomTestCase):
    def setUp(self):
        patches = (
            mock.patch.object(ap, "load_block32_routes", return_value="routes"),
            mock.patch.object(ap, "get_stream", side_effect=lambda name: name),
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA H20"),
            mock.patch.object(ap.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _resolve(self, *, subsystems=("attn_proj",), role="mix", **kwargs):
        kwargs.setdefault("quant_config", _fp8())
        kwargs.setdefault("fp8_wo_a_allowed", True)
        with get_flags().dsv41_sm90.override(
            subsystems=frozenset(subsystems), role=role
        ):
            plan = ap.resolve_attn_proj(**kwargs)
            loader = ap.wo_a_fp8_block32_enabled(kwargs["quant_config"])
        return plan, loader

    def test_subsystem_off_keeps_the_upstream_wo_a(self):
        # Off (or attn_proj disabled), the loader still dequantizes wo_a to bf16.
        for subsystems in ((), ("fp8_gemm", "mhc")):
            with self.subTest(subsystems=subsystems):
                plan, loader = self._resolve(subsystems=subsystems)
                self.assertIsNone(plan)
                self.assertFalse(loader)

    def test_loader_and_layer_agree_on_the_fp8_wo_a(self):
        # The loader keeps FP8 wo_a exactly when the layer allocated it in FP8;
        # a mismatch fails the weight loader's dtype check.
        cases = (
            (dict(), True),
            (dict(quant_config=_fp8(block=(128, 128))), False),
            (dict(quant_config=_fp8(scale_fmt=None)), False),
            (dict(quant_config=None), False),
        )
        for kwargs, block32 in cases:
            with self.subTest(**kwargs):
                plan, loader = self._resolve(**kwargs)
                self.assertEqual(plan.wo_a_fp8_block32, block32)
                self.assertEqual(loader, block32)
                self.assertEqual(plan.routes is not None, block32)
        with mock.patch.object(ap.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False):
            plan, loader = self._resolve()
            self.assertFalse(plan.wo_a_fp8_block32 or loader)

    def test_draft_keeps_its_bf16_wo_a_but_still_rounds(self):
        # The DSpark draft pins wo_a to bf16 (its loader dequantizes); the plan
        # exists, so its input still takes the V4.1 rounding.
        plan, _ = self._resolve(fp8_wo_a_allowed=False)
        self.assertIsNotNone(plan)
        self.assertFalse(plan.wo_a_fp8_block32)
        self.assertEqual(plan.streams, ())

    def test_decode_role_fusions(self):
        for role in ("mix", "prefill", "decode"):
            with self.subTest(role=role):
                plan, _ = self._resolve(role=role)
                self.assertEqual(plan.decode_rope_rows, role == "decode")
                self.assertEqual(plan.kv_store_q_rope, role == "decode")
        # The bf16-sourced K store rejects a query to rope.
        with envs.SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE.override(True):
            plan, _ = self._resolve(role="decode")
        self.assertFalse(plan.kv_store_q_rope)


if __name__ == "__main__":
    unittest.main()
