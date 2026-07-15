"""Compressed-tensors adapter for multimodal-generation linear layers."""

import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig as MultimodalQuantizationConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig as SRTCompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod as SRTCompressedTensorsLinearMethod,
)


class MultimodalCompressedTensorsLinearMethod(SRTCompressedTensorsLinearMethod):
    """Use SRT schemes with the multimodal generic weight loader.

    Multimodal linear layers select their v2 weight loader by exact quant-method
    class name. A distinct adapter name keeps SRT parameter classes on the
    compatible generic loader, whose TP rank comes from the multimodal layer.
    """


class CompressedTensorsConfig(SRTCompressedTensorsConfig, MultimodalQuantizationConfig):
    """Recognize multimodal-generation linear layers in SRT's CT config."""

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if not isinstance(layer, LinearBase):
            return super().get_quant_method(layer, prefix)

        if self.linear_fp8_config is not None:
            raise NotImplementedError(
                "linear_fp8_config is not supported by the multimodal "
                "compressed-tensors adapter"
            )

        scheme = self.get_linear_scheme(layer=layer, layer_name=prefix)
        if scheme is None:
            return UnquantizedLinearMethod()

        layer.scheme = scheme
        return MultimodalCompressedTensorsLinearMethod(self)
