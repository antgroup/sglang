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
