"""
This unittest is introduced in #22360, preventing duplicate transformer safetensors variants being loaded together
"""

import json
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

partial_json_parser = types.ModuleType("partial_json_parser")
partial_json_parser_core = types.ModuleType("partial_json_parser.core")
partial_json_parser_exceptions = types.ModuleType("partial_json_parser.core.exceptions")
partial_json_parser_options = types.ModuleType("partial_json_parser.core.options")


class _MalformedJSON(Exception):
    pass


class _Allow:
    STR = 1
    OBJ = 2
    ARR = 4
    ALL = STR | OBJ | ARR


def _loads(input_str, _flags=None):
    return json.loads(input_str)


partial_json_parser_exceptions.MalformedJSON = _MalformedJSON
partial_json_parser_options.Allow = _Allow
partial_json_parser.loads = _loads
sys.modules.setdefault("partial_json_parser", partial_json_parser)
sys.modules.setdefault("partial_json_parser.core", partial_json_parser_core)
sys.modules.setdefault(
    "partial_json_parser.core.exceptions", partial_json_parser_exceptions
)
sys.modules.setdefault("partial_json_parser.core.options", partial_json_parser_options)

from sglang.multimodal_gen.runtime.layers.quantization import (  # noqa: E402
    get_quantization_config,
)
from sglang.multimodal_gen.runtime.layers.quantization.compressed_tensors import (  # noqa: E402
    CompressedTensorsConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (  # noqa: E402
    NunchakuConfig,
)
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (  # noqa: E402
    _filter_duplicate_precision_variant_safetensors,
    _Flux2Nvfp4FallbackAdapter,
    resolve_transformer_quant_load_spec,
    resolve_transformer_safetensors_to_load,
)


class _FakeFluxTransformer:
    pass


class _FakeQuantConfig:
    @classmethod
    def get_name(cls):
        return "modelopt_fp4"


class TestTransformerQuantHelpers(unittest.TestCase):
    def _make_server_args(self, **overrides):
        defaults = dict(
            transformer_weights_path=None,
            pipeline_config=SimpleNamespace(
                dit_precision="bf16",
                dit_config=SimpleNamespace(
                    arch_config=SimpleNamespace(param_names_mapping={})
                ),
            ),
            nunchaku_config=None,
            tp_size=1,
            dit_cpu_offload=False,
            text_encoder_cpu_offload=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_resolve_transformer_safetensors_to_load_uses_single_override_file(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            server_args = self._make_server_args(transformer_weights_path=f.name)
            resolved = resolve_transformer_safetensors_to_load(
                server_args, "/unused/component/path"
            )

        self.assertEqual(resolved, [f.name])

    def test_filter_transformer_precision_variants_prefers_canonical_file(self):
        files = [
            "/tmp/transformer/diffusion_pytorch_model.fp16.safetensors",
            "/tmp/transformer/diffusion_pytorch_model.safetensors",
            "/tmp/transformer/other.safetensors",
        ]

        resolved = _filter_duplicate_precision_variant_safetensors(files)

        self.assertEqual(
            resolved,
            [
                "/tmp/transformer/diffusion_pytorch_model.safetensors",
                "/tmp/transformer/other.safetensors",
            ],
        )

    def test_filter_transformer_precision_variants_keeps_precision_only_family(self):
        files = [
            "/tmp/transformer/diffusion_pytorch_model.bf16.safetensors",
            "/tmp/transformer/diffusion_pytorch_model.fp16.safetensors",
        ]

        resolved = _filter_duplicate_precision_variant_safetensors(files)

        self.assertEqual(resolved, files)

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.build_nvfp4_config_from_safetensors_list",
        return_value=None,
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.get_quant_config_from_safetensors_metadata",
        return_value=None,
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.get_metadata_from_safetensors_file"
    )
    def test_resolve_transformer_quant_load_spec_keeps_nunchaku_hook(
        self,
        mock_metadata,
        _mock_quant_metadata,
        _mock_nvfp4,
    ):
        mock_metadata.return_value = {
            "config": json.dumps({"_class_name": _FakeFluxTransformer.__name__})
        }
        nunchaku_config = NunchakuConfig(
            transformer_weights_path="/tmp/svdq-int4_r32.safetensors"
        )
        server_args = self._make_server_args(
            transformer_weights_path=nunchaku_config.transformer_weights_path,
            nunchaku_config=nunchaku_config,
        )

        spec = resolve_transformer_quant_load_spec(
            hf_config={},
            server_args=server_args,
            safetensors_list=[nunchaku_config.transformer_weights_path],
            component_model_path="/unused/component/path",
            model_cls=_FakeFluxTransformer,
            cls_name=_FakeFluxTransformer.__name__,
        )

        self.assertIsNone(spec.quant_config)
        self.assertIs(spec.nunchaku_config, nunchaku_config)
        self.assertIsNone(spec.param_dtype)
        self.assertEqual(len(spec.post_load_hooks), 1)
        self.assertIs(nunchaku_config.model_cls, _FakeFluxTransformer)

    def test_flux2_mixed_nvfp4_fallback_disables_conflicting_offloads(self):
        server_args = self._make_server_args(
            transformer_weights_path="/tmp/flux2-dev-nvfp4-mixed.safetensors",
            tp_size=2,
            dit_cpu_offload=True,
            text_encoder_cpu_offload=True,
        )

        _Flux2Nvfp4FallbackAdapter._maybe_adjust_flux2_nvfp4_fallback_defaults(
            cls_name="Flux2Transformer2DModel",
            server_args=server_args,
            quant_config=_FakeQuantConfig(),
        )

        self.assertFalse(server_args.dit_cpu_offload)
        self.assertFalse(server_args.text_encoder_cpu_offload)


class TestDiffusionCompressedTensorsConfig(unittest.TestCase):
    def _fp8_dynamic_config(self):
        return {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "format": "float-quantized",
                    "weights": {
                        "type": "float",
                        "num_bits": 8,
                        "strategy": "channel",
                        "dynamic": False,
                        "symmetric": True,
                    },
                    "input_activations": {
                        "type": "float",
                        "num_bits": 8,
                        "strategy": "token",
                        "dynamic": True,
                        "symmetric": True,
                    },
                    "output_activations": None,
                }
            },
            "ignore": ["patch_embedding"],
            "kv_cache_scheme": None,
            "sparsity_config": {},
        }

    def test_fp8_dynamic_config_is_registered_and_parsed(self):
        config_cls = get_quantization_config("compressed-tensors")
        self.assertIs(config_cls, CompressedTensorsConfig)

        config = config_cls.from_config(self._fp8_dynamic_config())

        self.assertEqual(config.get_name(), "compressed_tensors")
        self.assertEqual(set(config.target_scheme_map), {"Linear"})
        self.assertEqual(config.ignore, ["patch_embedding"])
        scheme_parts = config.target_scheme_map["Linear"]
        scheme = config._get_scheme_from_parts(
            scheme_parts["weights"],
            scheme_parts["input_activations"],
        )
        self.assertEqual(type(scheme).__name__, "_DiffusionCompressedTensorsW8A8Fp8")

    def test_unsupported_compressed_tensors_formats_fail_closed(self):
        mutations = (
            ("format", ("format",), "int-quantized"),
            (
                "weight_bits",
                ("config_groups", "group_0", "weights", "num_bits"),
                4,
            ),
            (
                "weight_strategy",
                ("config_groups", "group_0", "weights", "strategy"),
                "tensor",
            ),
            (
                "activation_strategy",
                ("config_groups", "group_0", "input_activations", "strategy"),
                "tensor",
            ),
            (
                "static_activations",
                ("config_groups", "group_0", "input_activations", "dynamic"),
                False,
            ),
        )
        for name, path, value in mutations:
            with self.subTest(name=name):
                config = self._fp8_dynamic_config()
                target = config
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaisesRegex(ValueError, "FP8_DYNAMIC"):
                    CompressedTensorsConfig.from_config(config)


if __name__ == "__main__":
    unittest.main()
