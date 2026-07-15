import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    WEIGHT_LOADER_V2_SUPPORTED,
    ColumnParallelLinear,
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import get_quantization_config
from sglang.multimodal_gen.runtime.layers.quantization.compressed_tensors import (
    CompressedTensorsConfig,
    MultimodalCompressedTensorsLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestMultimodalCompressedTensors(CustomTestCase):
    @staticmethod
    def _fp8_dynamic_config():
        return {
            "config_groups": {
                "group_0": {
                    "input_activations": {
                        "dynamic": True,
                        "num_bits": 8,
                        "strategy": "token",
                        "symmetric": True,
                        "type": "float",
                    },
                    "output_activations": None,
                    "targets": ["Linear"],
                    "weights": {
                        "dynamic": False,
                        "num_bits": 8,
                        "strategy": "channel",
                        "symmetric": True,
                        "type": "float",
                    },
                }
            },
            "format": "float-quantized",
            "ignore": ["re:.*proj_out.*"],
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
        }

    def test_registry_and_multimodal_linear_adapter(self):
        self.assertIs(
            get_quantization_config("compressed-tensors"), CompressedTensorsConfig
        )
        self.assertTrue(issubclass(CompressedTensorsConfig, QuantizationConfig))
        self.assertFalse(inspect.isabstract(CompressedTensorsConfig))

        config = CompressedTensorsConfig(
            target_scheme_map={},
            ignore=[],
            quant_format="float-quantized",
            sparsity_scheme_map={},
            sparsity_ignore_list=[],
        )
        scheme = object()
        config.get_linear_scheme = Mock(return_value=scheme)
        layer = LinearBase.__new__(LinearBase)
        torch.nn.Module.__init__(layer)

        method = config.get_quant_method(layer, "blocks.0.attn1.to_q")

        self.assertIs(layer.scheme, scheme)
        self.assertIsInstance(method, MultimodalCompressedTensorsLinearMethod)
        self.assertNotIn(type(method).__name__, WEIGHT_LOADER_V2_SUPPORTED)

    def test_fp8_dynamic_scheme_weight_lifecycle(self):
        config = CompressedTensorsConfig.from_config(self._fp8_dynamic_config())
        config._check_scheme_supported = Mock(return_value=True)
        layer = ReplicatedLinear(
            4,
            6,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix="blocks.0.to_q",
        )

        self.assertIsInstance(
            layer.quant_method, MultimodalCompressedTensorsLinearMethod
        )
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.weight.shape, (6, 4))
        self.assertEqual(layer.weight_scale.shape, (6, 1))

        loaded_weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        loaded_weight = loaded_weight.to(torch.float8_e4m3fn)
        loaded_scale = torch.arange(1, 7, dtype=torch.float32).reshape(6, 1)
        layer.weight_loader(layer.weight, loaded_weight)
        layer.weight_loader(layer.weight_scale, loaded_scale)

        layer.quant_method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.shape, (4, 6))
        self.assertEqual(layer.weight_scale.shape, (6, 1))
        self.assertIsNone(layer.input_scale)
        torch.testing.assert_close(layer.weight_scale, loaded_scale)

    def test_fp8_dynamic_fuses_output_partitions_column_major(self):
        config = CompressedTensorsConfig.from_config(self._fp8_dynamic_config())
        config._check_scheme_supported = Mock(return_value=True)
        layers = [
            ReplicatedLinear(
                4,
                6,
                bias=True,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix=f"blocks.0.to_{name}",
            )
            for name in ("q", "k", "v")
        ]

        loaded_weights = []
        loaded_scales = []
        loaded_biases = []
        for index, layer in enumerate(layers):
            loaded_weight = (
                torch.arange(24, dtype=torch.float32).reshape(6, 4) + index * 32
            ).to(torch.float8_e4m3fn)
            loaded_scale = (
                torch.arange(1, 7, dtype=torch.float32).reshape(6, 1) + index * 8
            )
            loaded_bias = (torch.arange(6, dtype=torch.float32) + index * 8).to(
                torch.bfloat16
            )
            layer.weight_loader(layer.weight, loaded_weight)
            layer.weight_loader(layer.weight_scale, loaded_scale)
            layer.weight_loader(layer.bias, loaded_bias)
            layer.quant_method.process_weights_after_loading(layer)
            loaded_weights.append(loaded_weight)
            loaded_scales.append(loaded_scale)
            loaded_biases.append(loaded_bias)

        method = layers[0].quant_method
        self.assertTrue(method.can_fuse_output_partitions(layers))
        self.assertTrue(method.fuse_output_partitions(layers))

        fused = layers[0]
        expected_weight = torch.cat(loaded_weights, dim=0).t()
        expected_scale = torch.cat(loaded_scales, dim=0)
        expected_bias = torch.cat(loaded_biases, dim=0)
        self.assertEqual(fused.weight.shape, (4, 18))
        self.assertEqual(fused.weight.stride(), (1, 4))
        self.assertFalse(fused.weight.is_contiguous())
        torch.testing.assert_close(fused.weight.float(), expected_weight.float())
        torch.testing.assert_close(fused.weight_scale, expected_scale)
        torch.testing.assert_close(fused.bias, expected_bias)
        self.assertEqual(fused.logical_widths, [6, 6, 6])
        self.assertEqual(fused.output_size, 18)
        self.assertFalse(fused.weight.requires_grad)
        self.assertFalse(fused.weight_scale.requires_grad)
        self.assertFalse(fused.bias.requires_grad)

        sentinel = torch.arange(36, dtype=torch.bfloat16).reshape(2, 18)
        fused.scheme.apply_weights = Mock(return_value=sentinel)
        hidden_states = torch.zeros(2, 4, dtype=torch.bfloat16)
        fused_output, output_bias = fused(hidden_states)
        fused.scheme.apply_weights.assert_called_once_with(
            fused, hidden_states, bias=fused.bias
        )
        self.assertIsNone(output_bias)
        for actual, expected in zip(
            fused_output.chunk(3, dim=-1), sentinel.chunk(3, dim=-1), strict=True
        ):
            torch.testing.assert_close(actual, expected)

    def test_fp8_dynamic_output_fusion_rejects_unprocessed_weights(self):
        config = CompressedTensorsConfig.from_config(self._fp8_dynamic_config())
        config._check_scheme_supported = Mock(return_value=True)
        layers = [
            ReplicatedLinear(
                4,
                4,
                bias=True,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix=f"blocks.0.to_{name}",
            )
            for name in ("q", "k", "v")
        ]

        original_parameters = [layer.weight for layer in layers]
        method = layers[0].quant_method
        self.assertFalse(method.can_fuse_output_partitions(layers))
        self.assertFalse(method.fuse_output_partitions(layers))
        self.assertTrue(
            all(
                layer.weight is original
                for layer, original in zip(layers, original_parameters, strict=True)
            )
        )

    def test_lingbot_quantized_qkv_projection_uses_one_linear(self):
        from sglang.multimodal_gen.runtime.models.dits.lingbot_world import (
            CausalLingBotWorldTransformerBlock,
        )

        sentinel = torch.arange(18, dtype=torch.float32).reshape(1, 18)

        class FakeLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(6, 6))
                self.bias = torch.nn.Parameter(torch.zeros(6))
                self.quant_config = object()
                self.quant_method = None
                self.output = None
                self.call_count = 0

            def forward(self, hidden_states):
                self.call_count += 1
                return self.output, None

        class FakeQuantMethod:
            @staticmethod
            def can_fuse_output_partitions(layers):
                return True

            @staticmethod
            def fuse_output_partitions(layers):
                layers[0].output = sentinel
                return True

        block = CausalLingBotWorldTransformerBlock.__new__(
            CausalLingBotWorldTransformerBlock
        )
        torch.nn.Module.__init__(block)
        block._fused_qkv_weight = None
        block._fused_qkv_bias = None
        block._fused_qkv_quantized = False
        block.to_q = FakeLinear()
        block.to_k = FakeLinear()
        block.to_v = FakeLinear()
        block.to_q.quant_method = FakeQuantMethod()

        query, key, value = block._project_qkv(torch.zeros(1, 6))

        self.assertTrue(block._fused_qkv_quantized)
        self.assertEqual(block.to_q.call_count, 1)
        self.assertFalse(hasattr(block, "to_k"))
        self.assertFalse(hasattr(block, "to_v"))
        for actual, expected in zip(
            (query, key, value), sentinel.chunk(3, dim=-1), strict=True
        ):
            torch.testing.assert_close(actual, expected)

        fused_weight = block.to_q.weight
        self.assertTrue(block.fuse_qkv_projection())
        self.assertIs(block.to_q.weight, fused_weight)
        self.assertEqual(block.to_q.call_count, 1)

    def test_generic_loaders_accept_srt_weight_and_scale_parameters(self):
        column_layer = SimpleNamespace(tp_rank=1)
        column_weight = ModelWeightParameter(
            data=torch.empty(2, 3),
            input_dim=1,
            output_dim=0,
            weight_loader=lambda *_: None,
        )
        full_column_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        ColumnParallelLinear.weight_loader(
            column_layer, column_weight, full_column_weight
        )
        torch.testing.assert_close(column_weight, full_column_weight[2:])

        column_scale = ChannelQuantScaleParameter(
            data=torch.empty(2, 1),
            output_dim=0,
            weight_loader=lambda *_: None,
        )
        full_column_scale = torch.arange(4, dtype=torch.float32).reshape(4, 1)
        ColumnParallelLinear.weight_loader(
            column_layer, column_scale, full_column_scale
        )
        torch.testing.assert_close(column_scale, full_column_scale[2:])

        row_layer = SimpleNamespace(tp_rank=1)
        row_weight = ModelWeightParameter(
            data=torch.empty(2, 2),
            input_dim=1,
            output_dim=0,
            weight_loader=lambda *_: None,
        )
        full_row_weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        RowParallelLinear.weight_loader(row_layer, row_weight, full_row_weight)
        torch.testing.assert_close(row_weight, full_row_weight[:, 2:])

        row_scale = ChannelQuantScaleParameter(
            data=torch.empty(2, 1),
            output_dim=0,
            weight_loader=lambda *_: None,
        )
        full_row_scale = torch.arange(2, dtype=torch.float32).reshape(2, 1)
        RowParallelLinear.weight_loader(row_layer, row_scale, full_row_scale)
        torch.testing.assert_close(row_scale, full_row_scale)

        tensor_scale = PerTensorScaleParameter(
            data=torch.empty(1),
            weight_loader=lambda *_: None,
        )
        RowParallelLinear.weight_loader(row_layer, tensor_scale, torch.tensor(0.25))
        torch.testing.assert_close(tensor_scale, torch.tensor([0.25]))


if __name__ == "__main__":
    unittest.main(verbosity=3)
