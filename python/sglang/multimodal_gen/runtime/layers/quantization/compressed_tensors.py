"""Compressed-tensors support for diffusion FP8_DYNAMIC checkpoints.

The serving runtime already contains the kernels and scheme selection used by
the LLM runtime. Diffusion has its own linear and parameter classes, though,
so it needs a small adapter instead of importing the LLM linear method
directly.

This adapter intentionally accepts only the format exercised by LingBot World:

* float-quantized FP8 weights;
* static per-output-channel weight scales; and
* dynamic per-token activation scales.

Other compressed-tensors formats fail during config parsing rather than
silently selecting an incompatible kernel.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig as SrtCompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8,
)


def _require(
    condition: bool,
    *,
    group_name: str,
    field: str,
    expected: str,
    actual: object,
) -> None:
    if not condition:
        raise ValueError(
            "Diffusion compressed-tensors currently supports only FP8_DYNAMIC; "
            f"{group_name}.{field} must be {expected}, got {actual!r}"
        )


class _DiffusionCompressedTensorsW8A8Fp8(CompressedTensorsW8A8Fp8):
    """Use diffusion parameter classes with the shared FP8 compute path."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ) -> None:
        del input_size, output_size, kwargs

        if self.strategy != QuantizationStrategy.CHANNEL:
            raise ValueError(
                "Diffusion compressed-tensors FP8 requires channel-wise weights, "
                f"got {self.strategy}"
            )
        if self.is_static_input_scheme:
            raise ValueError(
                "Diffusion compressed-tensors FP8 requires dynamic activations"
            )

        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.weight_block_size = None
        layer.orig_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty(
                (output_size_per_partition, 1),
                dtype=torch.float32,
            ),
            output_dim=0,
            weight_loader=weight_loader,
        )
        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)


class CompressedTensorsLinearMethod(LinearMethodBase):
    """Diffusion linear-method wrapper around a compressed-tensors scheme."""

    def __init__(self, quantization_config: "CompressedTensorsConfig") -> None:
        self.quantization_config = quantization_config
        self.quant_config = quantization_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        layer.scheme.create_weights(
            layer=layer,
            input_size=input_size,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if layer.scheme is None:
            raise ValueError("A compressed-tensors scheme must be set on the layer")
        return layer.scheme.apply_weights(layer, x, bias=bias)


class CompressedTensorsConfig(SrtCompressedTensorsConfig, QuantizationConfig):
    """Diffusion config for compressed-tensors FP8_DYNAMIC checkpoints."""

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompressedTensorsConfig":
        cls._validate_fp8_dynamic_config(config)
        return super().from_config(config)

    @staticmethod
    def _validate_fp8_dynamic_config(config: Dict[str, Any]) -> None:
        _require(
            config.get("format") == "float-quantized",
            group_name="config",
            field="format",
            expected="'float-quantized'",
            actual=config.get("format"),
        )
        _require(
            not config.get("sparsity_config"),
            group_name="config",
            field="sparsity_config",
            expected="empty",
            actual=config.get("sparsity_config"),
        )
        _require(
            config.get("kv_cache_scheme") is None,
            group_name="config",
            field="kv_cache_scheme",
            expected="None",
            actual=config.get("kv_cache_scheme"),
        )

        config_groups = config.get("config_groups") or {}
        _require(
            bool(config_groups),
            group_name="config",
            field="config_groups",
            expected="a non-empty mapping",
            actual=config_groups,
        )

        for group_name, group in config_groups.items():
            weights = group.get("weights") or {}
            activations = group.get("input_activations") or {}
            expected_fields = (
                (
                    "weights.type",
                    weights.get("type") == "float",
                    "'float'",
                    weights.get("type"),
                ),
                (
                    "weights.num_bits",
                    weights.get("num_bits") == 8,
                    "8",
                    weights.get("num_bits"),
                ),
                (
                    "weights.strategy",
                    weights.get("strategy") == "channel",
                    "'channel'",
                    weights.get("strategy"),
                ),
                (
                    "weights.dynamic",
                    weights.get("dynamic") is False,
                    "False",
                    weights.get("dynamic"),
                ),
                (
                    "weights.symmetric",
                    weights.get("symmetric") is True,
                    "True",
                    weights.get("symmetric"),
                ),
                (
                    "input_activations.type",
                    activations.get("type") == "float",
                    "'float'",
                    activations.get("type"),
                ),
                (
                    "input_activations.num_bits",
                    activations.get("num_bits") == 8,
                    "8",
                    activations.get("num_bits"),
                ),
                (
                    "input_activations.strategy",
                    activations.get("strategy") == "token",
                    "'token'",
                    activations.get("strategy"),
                ),
                (
                    "input_activations.dynamic",
                    activations.get("dynamic") is True,
                    "True",
                    activations.get("dynamic"),
                ),
                (
                    "input_activations.symmetric",
                    activations.get("symmetric") is True,
                    "True",
                    activations.get("symmetric"),
                ),
                (
                    "output_activations",
                    group.get("output_activations") is None,
                    "None",
                    group.get("output_activations"),
                ),
            )
            for field, condition, expected, actual in expected_fields:
                _require(
                    condition,
                    group_name=group_name,
                    field=field,
                    expected=expected,
                    actual=actual,
                )

    def _get_scheme_from_parts(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
    ) -> _DiffusionCompressedTensorsW8A8Fp8:
        if not (
            weight_quant.type == QuantizationType.FLOAT
            and weight_quant.num_bits == 8
            and weight_quant.strategy == QuantizationStrategy.CHANNEL
            and not weight_quant.dynamic
            and weight_quant.symmetric
            and input_quant.type == QuantizationType.FLOAT
            and input_quant.num_bits == 8
            and input_quant.strategy == QuantizationStrategy.TOKEN
            and input_quant.dynamic
            and input_quant.symmetric
        ):
            raise NotImplementedError(
                "Diffusion compressed-tensors supports only FP8_DYNAMIC"
            )
        return _DiffusionCompressedTensorsW8A8Fp8(
            weight_quant=weight_quant,
            is_static_input_scheme=False,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if not isinstance(layer, LinearBase):
            return None
        scheme = self.get_linear_scheme(layer=layer, layer_name=prefix)
        if scheme is None:
            return UnquantizedLinearMethod()
        layer.scheme = scheme
        return CompressedTensorsLinearMethod(self)
