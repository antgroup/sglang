"""Compressed-tensors adapter for multimodal-generation linear layers."""

from collections.abc import Sequence

import torch
from compressed_tensors.quantization import QuantizationStrategy

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
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8,
)


class MultimodalCompressedTensorsLinearMethod(SRTCompressedTensorsLinearMethod):
    """Use SRT schemes with the multimodal generic weight loader.

    Multimodal linear layers select their v2 weight loader by exact quant-method
    class name. A distinct adapter name keeps SRT parameter classes on the
    compatible generic loader, whose TP rank comes from the multimodal layer.
    """

    def can_fuse_output_partitions(self, layers: Sequence[LinearBase]) -> bool:
        """Return whether dynamic channel-wise FP8 linears can share one GEMM.

        Channel-wise compressed-tensors weights are transposed after loading to
        the column-major ``[K, N]`` layout required by SGL Kernel.  Requiring
        that post-load layout also prevents this hook from running before all
        checkpoint tensors have been materialized.
        """
        if len(layers) < 2 or layers[0].quant_method is not self:
            return False

        quant_methods = [getattr(layer, "quant_method", None) for layer in layers]
        if not all(
            isinstance(method, MultimodalCompressedTensorsLinearMethod)
            for method in quant_methods
        ):
            return False

        schemes = [getattr(layer, "scheme", None) for layer in layers]
        if not all(
            isinstance(scheme, CompressedTensorsW8A8Fp8)
            and scheme.strategy == QuantizationStrategy.CHANNEL
            and not scheme.is_static_input_scheme
            and scheme.weight_block_size is None
            for scheme in schemes
        ):
            return False

        if any(
            getattr(layer, "skip_bias_add", False)
            or getattr(layer, "gather_output", False)
            for layer in layers
        ):
            return False

        # Dynamic input scales are installed as ``None`` during post-load.
        if not all(
            hasattr(layer, "input_scale") and layer.input_scale is None
            for layer in layers
        ):
            return False

        weights = [getattr(layer, "weight", None) for layer in layers]
        scales = [getattr(layer, "weight_scale", None) for layer in layers]
        if not all(
            isinstance(weight, torch.nn.Parameter)
            and isinstance(scale, torch.nn.Parameter)
            for weight, scale in zip(weights, scales, strict=True)
        ):
            return False

        # Replacing fully-sharded parameters after FSDP has wrapped the model is
        # unsafe.  The normal LingBot sequence-parallel deployment is replicated
        # at the linear layer and does not carry a device mesh.
        if any(hasattr(tensor, "device_mesh") for tensor in (*weights, *scales)):
            return False

        first_weight = weights[0]
        first_scale = scales[0]
        if first_weight.ndim != 2:
            return False
        if first_scale.device != first_weight.device:
            return False
        input_width, output_width = first_weight.shape
        expected_weight_stride = (1, input_width)
        if not all(
            weight.shape == (input_width, output_width)
            and weight.stride() == expected_weight_stride
            and weight.dtype == first_weight.dtype
            and weight.device == first_weight.device
            for weight in weights
        ):
            return False
        if not all(
            scale.shape == (output_width, 1)
            and scale.dtype == first_scale.dtype
            and scale.device == first_scale.device
            for scale in scales
        ):
            return False

        biases = [getattr(layer, "bias", None) for layer in layers]
        if not (
            all(bias is None for bias in biases)
            or all(bias is not None for bias in biases)
        ):
            return False
        if biases[0] is not None:
            first_bias = biases[0]
            if (
                not isinstance(first_bias, torch.nn.Parameter)
                or first_bias.device != first_weight.device
            ):
                return False
            if not all(
                isinstance(bias, torch.nn.Parameter)
                and bias.shape == (output_width,)
                and bias.dtype == first_bias.dtype
                and bias.device == first_bias.device
                and not hasattr(bias, "device_mesh")
                for bias in biases
            ):
                return False

        return True

    def fuse_output_partitions(self, layers: Sequence[LinearBase]) -> bool:
        """Fuse compatible output projections into the first linear layer."""
        if not self.can_fuse_output_partitions(layers):
            return False

        destination = layers[0]
        weights = [layer.weight.detach() for layer in layers]
        scales = [layer.weight_scale.detach() for layer in layers]
        biases = [layer.bias for layer in layers]
        output_widths = [weight.shape[1] for weight in weights]

        # Build in checkpoint layout [N, K], then transpose back to the
        # column-major runtime layout [K, sum(N)].  A direct cat of runtime
        # weights would produce a row-major tensor rejected by fp8_scaled_mm.
        fused_weight = (
            torch.cat([weight.t() for weight in weights], dim=0).contiguous().t()
        )
        fused_scale = torch.cat(scales, dim=0).contiguous()
        fused_bias = (
            torch.cat([bias.detach() for bias in biases], dim=0).contiguous()
            if biases[0] is not None
            else None
        )

        if fused_weight.stride(0) != 1:
            raise RuntimeError("Fused FP8 weight must retain column-major layout")

        with torch.no_grad():
            destination.weight = torch.nn.Parameter(fused_weight, requires_grad=False)
            destination.weight_scale = torch.nn.Parameter(
                fused_scale, requires_grad=False
            )
            if fused_bias is not None:
                destination.bias = torch.nn.Parameter(fused_bias, requires_grad=False)

        destination.logical_widths = output_widths
        if hasattr(destination, "output_size"):
            destination.output_size = sum(
                getattr(layer, "output_size", width)
                for layer, width in zip(layers, output_widths, strict=True)
            )
        if hasattr(destination, "output_size_per_partition"):
            destination.output_size_per_partition = sum(output_widths)
        if hasattr(destination, "output_partition_sizes"):
            destination.output_partition_sizes = output_widths
        return True


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
