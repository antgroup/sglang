# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from typing import Literal, get_args

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.layers.quantization.modelslim import ModelSlimConfig

QuantizationMethods = Literal["fp8", "modelslim", "compressed-tensors"]

QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {
    "modelslim": ModelSlimConfig,
    "fp8": Fp8Config,
}


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.


    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            raise ValueError(
                f"The quantization method `{quantization}` is already exists."
            )
        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of " "`QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        QUANTIZATION_METHODS.append(quantization)
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    if quantization == "compressed-tensors":
        from sglang.multimodal_gen.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )

        return CompressedTensorsConfig
    return _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization]


__all__ = [
    "QuantizationMethods",
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
