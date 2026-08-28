"""CPU tests for applying weight-name mappings to compressed-tensors targets."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    apply_compressed_tensors_mapper_to_list,
    should_ignore_layer,
)
from sglang.srt.models.utils import WeightsMapper
from sglang.test.test_utils import CustomTestCase

_HF_TARGET = r"re:^model\.language_model\.layers\.\d+\.self_attn\.o_proj$"
_SGLANG_TARGET = r"re:^model\.layers\.\d+\.self_attn\.o_proj$"
_HF_IGNORE = r"re:^model\.language_model\.layers\.45\..*$"
_SGLANG_IGNORE = r"re:^model\.layers\.45\..*$"

_FP8_WEIGHTS = {
    "num_bits": 8,
    "type": "float",
    "strategy": "channel",
    "symmetric": True,
    "dynamic": False,
}
_FP8_DYNAMIC_ACTS = {
    "num_bits": 8,
    "type": "float",
    "strategy": "token",
    "symmetric": True,
    "dynamic": True,
}


def _config(targets, ignore=()):
    return CompressedTensorsConfig.from_config(
        {
            "format": "float-quantized",
            "quant_method": "compressed-tensors",
            "ignore": list(ignore),
            "config_groups": {
                "group_0": {
                    "targets": list(targets),
                    "weights": _FP8_WEIGHTS,
                    "input_activations": _FP8_DYNAMIC_ACTS,
                }
            },
        }
    )


class TestCompressedTensorsMapper(CustomTestCase):
    def test_maps_anchored_literal_regex_prefix(self):
        config = _config([_HF_TARGET], ignore=[_HF_IGNORE])
        config.apply_weight_name_mapper(
            WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})
        )

        self.assertEqual(list(config.target_scheme_map), [_SGLANG_TARGET])
        self.assertEqual(config.ignore, [_SGLANG_IGNORE])
        self.assertTrue(
            should_ignore_layer("model.layers.45.self_attn.o_proj", config.ignore)
        )

        with mock.patch.object(
            CompressedTensorsConfig, "_check_scheme_supported", return_value=True
        ):
            scheme = config.get_linear_scheme(
                torch.nn.Linear(1, 1),
                layer_name="model.layers.3.self_attn.o_proj",
            )
        self.assertIsInstance(scheme, CompressedTensorsW8A8Fp8)

    def test_uses_longest_literal_prefix(self):
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.": "fallback.",
                "model.language_model.": "model.",
            }
        )
        self.assertEqual(
            apply_compressed_tensors_mapper_to_list([_HF_TARGET], mapper),
            [_SGLANG_TARGET],
        )

    def test_preserves_unanchored_and_nonliteral_regexes(self):
        mapper = WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})
        targets = [
            r"re:.*model\.language_model\.layers\..*$",
            r"re:^model[.]language_model[.]layers[.].*$",
        ]
        self.assertEqual(
            apply_compressed_tensors_mapper_to_list(targets, mapper), targets
        )

    def test_preserves_module_class_targets_with_empty_prefix_mapper(self):
        mapper = WeightsMapper(orig_to_new_prefix={"": "model."})
        targets = ["Linear", "FusedMoE", r"re:^Linear$"]
        self.assertEqual(
            apply_compressed_tensors_mapper_to_list(targets, mapper), targets
        )

    def test_maps_concrete_paths_and_honors_deletion(self):
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.language_model.": "model.",
                "model.visual.": None,
            }
        )
        self.assertEqual(
            apply_compressed_tensors_mapper_to_list(
                [
                    "model.language_model.layers.3.self_attn.o_proj",
                    "model.visual.proj",
                    r"re:^model\.visual\..*$",
                ],
                mapper,
            ),
            ["model.layers.3.self_attn.o_proj"],
        )


if __name__ == "__main__":
    unittest.main()
