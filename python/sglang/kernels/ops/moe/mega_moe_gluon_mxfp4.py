# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Copyright (c) 2025 DeepSeek
# Derived from Triton-distributed MegaMoE Gluon; Hopper scheduling and
# numerical conventions derive from DeepGEMM. MIT licensed.
"""SM90 EP8 MXFP4 MegaMoE. Requires the sibling FP8 module.

Routing IDs must be negative sentinels or lie in [0, num_experts).
Nonnegative IDs outside that range are outside the operator contract.

Public API: create_context, prepare_weights, fused_moe.
No serving-stack integration or process-group initialization at import.
"""

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional, Tuple

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from .mega_moe_gluon import (
    _FUSED_RESET_BLOCK_SIZE,
    _FUSED_RESET_NUM_WARPS,
    ExpertPool,
    MathBody,
    MegaMoEConfig,
    RegisteredInputs,
    Row,
    Shape,
    SymmetricContext,
    _fc1_epilogue,
    _fc2_bf16_scatter_epilogue,
    _fc2_swap_bf16_epilogue,
    _host_cdiv,
    _host_next_power_of_2,
    _load_contiguous_bf16_fragment,
    _load_i32_acquire_gpu,
    _load_packed_expert_counts,
    _make_selector,
    _packed_expert_count_from_cache,
    _peer_barrier_arrive_and_wait,
    a_producer_partition,
    check_policy_agreement,
    create_context,
    create_dispatch_descriptors,
    partition_barrier,
    register_inputs,
    require_sm90,
    reset_control_kernel,
    resolve_tokens_bound,
    scheduler_count,
    scheduler_next,
)

# MIT License
#
# Copyright (c) 2025 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# policy.py


# Shape-specific launch policies.

CONFIGS = MappingProxyType(
    {
        "mxfp4_split128_e16": MegaMoEConfig(
            body=MathBody.SPLIT_BN128,
            stages=3,
            math_warps=4,
            ctas_per_sm=2,
            experts_per_wave=16,
            math_registers=128,
            producer_registers=24,
            fc1_promotion_k=128,
        ),
        "mxfp4_split128_e2": MegaMoEConfig(
            body=MathBody.SPLIT_BN128,
            stages=3,
            math_warps=4,
            ctas_per_sm=2,
            experts_per_wave=2,
            math_registers=128,
            producer_registers=24,
            fc1_promotion_k=128,
        ),
        "mxfp4_split128_e32": MegaMoEConfig(
            body=MathBody.SPLIT_BN128,
            stages=3,
            math_warps=4,
            ctas_per_sm=2,
            experts_per_wave=32,
            math_registers=128,
            producer_registers=24,
            fc1_promotion_k=128,
        ),
        "mxfp4_split128_e4": MegaMoEConfig(
            body=MathBody.SPLIT_BN128,
            stages=3,
            math_warps=4,
            ctas_per_sm=2,
            experts_per_wave=4,
            math_registers=128,
            producer_registers=24,
            fc1_promotion_k=128,
        ),
        "mxfp4_split128_e8": MegaMoEConfig(
            body=MathBody.SPLIT_BN128,
            stages=3,
            math_warps=4,
            ctas_per_sm=2,
            experts_per_wave=8,
            math_registers=128,
            producer_registers=24,
            fc1_promotion_k=128,
        ),
        "mxfp4_swap_e16": MegaMoEConfig(
            body=MathBody.SWAP_BN128,
            stages=3,
            math_warps=4,
            ctas_per_sm=2,
            experts_per_wave=16,
            math_registers=128,
            producer_registers=24,
            fc1_promotion_k=128,
        ),
        "mxfp4_swap_e32": MegaMoEConfig(
            body=MathBody.SWAP_BN128,
            stages=3,
            math_warps=4,
            ctas_per_sm=2,
            experts_per_wave=32,
            math_registers=128,
            producer_registers=24,
            fc1_promotion_k=128,
        ),
        "mxfp4_swap_e8": MegaMoEConfig(
            body=MathBody.SWAP_BN128,
            stages=3,
            math_warps=4,
            ctas_per_sm=2,
            experts_per_wave=8,
            math_registers=128,
            producer_registers=24,
            fc1_promotion_k=128,
        ),
        "mxfp4_swap_eNone": MegaMoEConfig(
            body=MathBody.SWAP_BN128,
            stages=3,
            math_warps=4,
            ctas_per_sm=2,
            math_registers=128,
            producer_registers=24,
            fc1_promotion_k=128,
        ),
    }
)

TABLE = MappingProxyType(
    {
        ("ep8", "mxfp4", "h4096_i2048_e256_k6", 78): (
            Row(0, 128, "mxfp4_swap_e32"),
            Row(129, 256, "mxfp4_split128_e32"),
            Row(257, 1024, "mxfp4_split128_e16"),
            Row(1025, 4096, "mxfp4_split128_e4"),
            Row(4097, None, "mxfp4_split128_e8"),
        ),
        ("ep8", "mxfp4", "h7168_i3072_e384_k6", 78): (
            Row(0, 64, "mxfp4_swap_eNone"),
            Row(65, 256, "mxfp4_swap_e16"),
            Row(257, 512, "mxfp4_swap_e8"),
            Row(513, 1024, "mxfp4_split128_e16"),
            Row(1025, 2048, "mxfp4_split128_e32"),
            Row(2049, None, "mxfp4_split128_e8"),
        ),
    }
)

select = _make_selector("ep8", "mxfp4", TABLE, CONFIGS)


# mxfp4/constants.py

MXFP4CheckpointWeights = Tuple[torch.Tensor, torch.Tensor]


MXFP4ProcessedWeights = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


_MXFP4_BLOCK_M_VALUE = 64


_MXFP4_BLOCK_N_VALUE = 128


_MXFP4_BLOCK_K_VALUE = 128


_MXFP4_PRODUCERS = 2


_MXFP4_TMA_REGS_VALUE = 24


_MXFP4_FUSED_MAXNREG = 128


_MXFP4_BLOCK_M = gl.constexpr(_MXFP4_BLOCK_M_VALUE)


_MXFP4_BLOCK_N = gl.constexpr(_MXFP4_BLOCK_N_VALUE)


_MXFP4_BLOCK_K = gl.constexpr(_MXFP4_BLOCK_K_VALUE)


_MXFP4_PRODUCERS_CONSTEXPR = gl.constexpr(_MXFP4_PRODUCERS)


_MXFP4_COALESCED_SCALE_MAX_HIDDEN = 8192


_MXFP4_COALESCED_SCALE_MAX_HIDDEN_CONSTEXPR = gl.constexpr(
    _MXFP4_COALESCED_SCALE_MAX_HIDDEN
)


_MXFP4_MATH_WARPS = 4


_MXFP4_MATH_WARPS_CONSTEXPR = gl.constexpr(_MXFP4_MATH_WARPS)


# mxfp4/types.py


@dataclass(frozen=True)
class MXFP4Workspace:
    """Reusable storage for the MXFP4 fused MegaMoE path."""

    pool: ExpertPool
    l2_acts: torch.Tensor
    l2_acts_sf_mn_major: torch.Tensor
    output: torch.Tensor
    l1_arrival: torch.Tensor
    l2_arrival: torch.Tensor
    fc2_scatter_grid_counter: torch.Tensor
    combine_cross_rank_ready: torch.Tensor
    actual_num_pool_rows: torch.Tensor
    dispatch_counter: torch.Tensor
    expert_send_state: torch.Tensor
    l1_a_desc: object
    l1_sfa_desc: object
    l1_b_desc: object
    l2_store_desc: object
    l2_a_desc: object
    l2_sfa_desc: object
    l2_b_desc: object
    dispatch_descs: tuple[object, ...]
    l1_weight_data_ptr: int
    l2_weight_data_ptr: int
    max_pool_blocks: int
    num_pool_rows: int
    num_padded_sf_pool_tokens: int
    l1_b_sf_desc: object | None = None
    l2_b_sf_desc: object | None = None
    l1_weight_sf_data_ptr: int = 0
    l2_weight_sf_data_ptr: int = 0
    l1_secondary_data_ptr: int = 0
    l2_secondary_data_ptr: int = 0


@dataclass(frozen=True)
class MXFP4Result:
    """Artifacts from one complete MXFP4 fused MegaMoE invocation."""

    pool: ExpertPool
    l2_acts: torch.Tensor
    l2_acts_sf_mn_major: torch.Tensor
    l2_arrival: torch.Tensor
    output: torch.Tensor
    combine_buffer: torch.Tensor
    l1_arrival: torch.Tensor
    fc2_scatter_grid_counter: torch.Tensor
    combine_cross_rank_ready: torch.Tensor
    actual_num_pool_rows: torch.Tensor
    dispatch_counter: torch.Tensor
    workspace: MXFP4Workspace
    pre_dispatch: RegisteredInputs
    compiled: object


# mxfp4/weights.py


def _mxfp4_require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_valid_mxfp4_hidden_size(hidden: int) -> bool:
    """Validate the vectorized combine divisibility contract."""
    return hidden % 512 == 0 and (hidden <= 8192 or hidden % 1024 == 0)


def _normalize_mxfp4_ue8m0(sf: torch.Tensor) -> torch.Tensor:
    """Return natural-layout UE8M0 exponent bytes.

    Checkpoints normally expose UE8M0 as ``uint8`` or
    ``float8_e8m0fnu``. FP32 powers of two are also accepted so callers can
    feed the output of existing quantization helpers directly.
    """
    if not isinstance(sf, torch.Tensor):
        raise TypeError("MXFP4 scale must be a torch.Tensor")
    if sf.dtype == torch.uint8:
        return sf.contiguous()

    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None and sf.dtype == e8m0_dtype:
        return sf.contiguous().view(torch.uint8)

    if sf.dtype != torch.float32:
        raise TypeError(
            "MXFP4 scale must have dtype uint8, float8_e8m0fnu, or "
            f"float32, got {sf.dtype}"
        )

    bits = sf.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & ((1 << 23) - 1)
    is_min_subnormal = bits == (1 << 22)
    is_normal_power_of_two = (
        ((bits >> 31) == 0) & (mantissa == 0) & (exponent >= 1) & (exponent <= 254)
    )
    _mxfp4_require(
        bool((is_min_subnormal | is_normal_power_of_two).all().item()),
        "FP32 MXFP4 scales must be finite positive UE8M0 powers of two",
    )
    return torch.where(
        is_min_subnormal,
        torch.zeros_like(exponent),
        exponent,
    ).to(torch.uint8)


def _normalize_mxfp4_packed_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return packed E2M1 bytes using a signed-byte tensor view."""
    if not isinstance(weight, torch.Tensor):
        raise TypeError("MXFP4 weight must be a torch.Tensor")
    if weight.dtype not in (torch.uint8, torch.int8):
        raise TypeError(
            f"packed MXFP4 weight must have dtype uint8 or int8, got {weight.dtype}"
        )
    weight = weight.contiguous()
    return weight if weight.dtype == torch.int8 else weight.view(torch.int8)


def _validate_checkpoint_mxfp4_weights(
    weights: MXFP4CheckpointWeights,
    name: str,
) -> MXFP4CheckpointWeights:
    if not isinstance(weights, tuple) or len(weights) != 2:
        raise TypeError(f"{name} must be a (packed_weight, ue8m0_scale) tuple")
    weight = _normalize_mxfp4_packed_weight(weights[0])
    sf = _normalize_mxfp4_ue8m0(weights[1])
    _mxfp4_require(weight.ndim == 3, f"{name} weight must be [E, N, K/2]")
    _mxfp4_require(sf.ndim == 3, f"{name} scale must be [E, N, K/32]")
    _mxfp4_require(
        weight.device == sf.device,
        f"{name} weight and scale must use one device",
    )
    _mxfp4_require(
        weight.shape[:2] == sf.shape[:2] and weight.shape[2] == sf.shape[2] * 16,
        f"{name} expects packed K/2 weights and K/32 scales",
    )
    _mxfp4_require(
        all(dimension > 0 for dimension in weight.shape),
        f"{name} dimensions must be nonzero",
    )
    return weight, sf


def _validate_mxfp4_layer_pair(
    l1_weights: MXFP4CheckpointWeights,
    l2_weights: MXFP4CheckpointWeights,
) -> None:
    l1_weight, _ = l1_weights
    l2_weight, _ = l2_weights
    _mxfp4_require(
        l1_weight.device == l2_weight.device,
        "L1 and L2 MXFP4 weights must use one device",
    )
    _mxfp4_require(
        l1_weight.shape[0] == l2_weight.shape[0],
        "L1 and L2 MXFP4 weights must contain the same experts",
    )
    _mxfp4_require(
        l1_weight.shape[1] % 16 == 0,
        "L1 gate/up rows must be divisible by 16",
    )
    _mxfp4_require(
        l1_weight.shape[1] == l2_weight.shape[2] * 4,
        "incompatible MXFP4 L1/L2 intermediate dimensions",
    )
    _mxfp4_require(
        l2_weight.shape[1] == l1_weight.shape[2] * 2,
        "incompatible MXFP4 L1/L2 hidden dimensions",
    )
    hidden = l2_weight.shape[1]
    intermediate_hidden = l1_weight.shape[1] // 2
    _mxfp4_require(
        _is_valid_mxfp4_hidden_size(hidden),
        "SM90 MXFP4 hidden must be divisible by 512 and, above 8192, divisible by 1024",
    )
    _mxfp4_require(
        intermediate_hidden % 256 == 0,
        "SM90 MXFP4 intermediate hidden must be divisible by 256",
    )


def _interleave_mxfp4_rows(
    tensor: torch.Tensor,
    granularity: int = 8,
) -> torch.Tensor:
    """Convert ``[gate | up]`` rows to granularity-8 interleave."""
    _mxfp4_require(tensor.ndim == 3, "MXFP4 expert tensor must be 3D")
    num_experts, num_rows, *tail = tensor.shape
    _mxfp4_require(
        num_rows % (2 * granularity) == 0,
        f"gate/up row count must be divisible by {2 * granularity}",
    )
    half = num_rows // 2
    gate = tensor[:, :half].reshape(
        num_experts,
        half // granularity,
        granularity,
        *tail,
    )
    up = tensor[:, half:].reshape(
        num_experts,
        half // granularity,
        granularity,
        *tail,
    )
    return torch.stack((gate, up), dim=2).reshape_as(tensor).contiguous()


def _transpose_mxfp4_scales(tensor: torch.Tensor) -> torch.Tensor:
    """Store one K128 scale word contiguously across N.

    The public shape remains ``[E, N, K/32]``. Its opaque payload is laid out
    as ``[E, K/128, N, 4]`` so each N128/K128 tile is a contiguous 512-byte
    TMA transfer.
    """
    _mxfp4_require(tensor.ndim == 3, "MXFP4 scale tensor must be 3D")
    num_experts, num_rows, num_k32_groups = tensor.shape
    _mxfp4_require(
        num_k32_groups % 4 == 0,
        "MXFP4 K/32 scale dimension must be divisible by 4",
    )
    return (
        tensor.reshape(num_experts, num_rows, num_k32_groups // 4, 4)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(tensor.shape)
    )


def _uses_coalesced_mxfp4_scales(hidden: int) -> bool:
    return hidden <= _MXFP4_COALESCED_SCALE_MAX_HIDDEN


def _reorder_mxfp4_sign_bits(weight: torch.Tensor) -> torch.Tensor:
    """Move packed-word signs to the order consumed by the SM90 decoder.

    Magnitudes stay in checkpoint order. Within each four-byte packed word,
    signs change from ``[s0,s1,s2,s3,s4,s5,s6,s7]`` to
    ``[s0,s4,s1,s5,s2,s6,s3,s7]``. This is the packed-weight PR contract and lets
    the device decoder avoid two extra sign-gather permutations.
    """
    if weight.dtype not in (torch.uint8, torch.int8):
        raise TypeError("MXFP4 sign reordering requires uint8 or int8 bytes")
    _mxfp4_require(
        weight.shape[-1] % 4 == 0,
        "MXFP4 sign reordering requires K/2 divisible by 4 bytes",
    )
    original_dtype = weight.dtype
    source = weight.contiguous().view(torch.uint8).reshape(*weight.shape[:-1], -1, 4)
    reordered = source & 0x77
    reordered[..., 0] |= (source[..., 0] & 0x08) | ((source[..., 2] & 0x08) << 4)
    reordered[..., 1] |= ((source[..., 0] & 0x80) >> 4) | (source[..., 2] & 0x80)
    reordered[..., 2] |= (source[..., 1] & 0x08) | ((source[..., 3] & 0x08) << 4)
    reordered[..., 3] |= ((source[..., 1] & 0x80) >> 4) | (source[..., 3] & 0x80)
    return reordered.reshape(weight.shape).view(original_dtype)


def _process_mxfp4_e8m0(
    weight: torch.Tensor,
    sf: torch.Tensor,
    *,
    interleave_rows: bool = False,
) -> MXFP4ProcessedWeights:
    """Rebase raw UE8M0 into bounded relative offsets per expert."""
    weight, raw_sf = _validate_checkpoint_mxfp4_weights(
        (weight, sf),
        "MXFP4",
    )
    _mxfp4_require(
        not bool((raw_sf == 255).any().item()),
        "processed MXFP4 does not accept UE8M0 NaN code 255",
    )

    sf_i16 = raw_sf.to(torch.int16)
    max_exp = sf_i16.flatten(1).amax(dim=1)
    min_exp = sf_i16.flatten(1).amin(dim=1)
    base_exp = max_exp - torch.minimum(
        max_exp - min_exp,
        torch.full_like(max_exp, 11),
    )
    base_view = base_exp.view(-1, 1, 1)
    clamped = torch.maximum(sf_i16, base_view)
    delta = clamped - sf_i16
    _mxfp4_require(
        not bool((delta >= 128).any().item()),
        "MXFP4 checkpoint exponent span is outside the supported range",
    )
    offsets = (clamped - base_view + 1).to(torch.uint8)
    secondary = torch.exp2(base_exp.float() - 128.0).contiguous()

    nibble_lut = torch.tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            0,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            0,
            1,
            1,
            2,
            2,
            3,
            4,
            5,
            0,
            9,
            9,
            10,
            10,
            11,
            12,
            13,
            0,
            0,
            1,
            1,
            1,
            2,
            2,
            3,
            0,
            8,
            9,
            9,
            9,
            10,
            10,
            11,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            2,
            0,
            8,
            8,
            8,
            9,
            9,
            9,
            10,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            8,
            8,
            8,
            8,
            8,
            9,
            9,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
        ],
        dtype=torch.uint8,
        device=weight.device,
    ).view(6, 16)
    packed_values = torch.arange(256, dtype=torch.long, device=weight.device)
    packed_lut = (
        nibble_lut[:, packed_values & 0x0F] | (nibble_lut[:, packed_values >> 4] << 4)
    ).reshape(-1)

    packed = weight.view(torch.uint8)
    rewritten = torch.empty_like(packed)
    num_k_groups = raw_sf.shape[2]
    for expert_idx in range(weight.shape[0]):
        for row_start in range(0, weight.shape[1], 1024):
            row_end = min(row_start + 1024, weight.shape[1])
            packed_chunk = packed[expert_idx, row_start:row_end].view(
                row_end - row_start,
                num_k_groups,
                16,
            )
            delta_chunk = (
                delta[expert_idx, row_start:row_end].clamp_max(5).unsqueeze(-1)
            )
            lut_idx = delta_chunk.to(torch.long) * 256 + packed_chunk.to(torch.long)
            rewritten_chunk = packed_lut[lut_idx].reshape(row_end - row_start, -1)
            rewritten[expert_idx, row_start:row_end].copy_(
                _reorder_mxfp4_sign_bits(rewritten_chunk)
            )

    rewritten = rewritten.view(torch.int8)
    if interleave_rows:
        rewritten = _interleave_mxfp4_rows(rewritten)
    return rewritten, offsets.contiguous(), secondary


def _apply_mxfp4_weight_scale_2(
    scale: torch.Tensor,
    secondary: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"{name}_weight_scale_2 must be a torch.Tensor")
    if scale.dtype != torch.float32:
        raise TypeError(f"{name}_weight_scale_2 must have dtype float32")
    _mxfp4_require(
        scale.shape == secondary.shape,
        f"{name}_weight_scale_2 must have shape [E]",
    )
    _mxfp4_require(
        scale.device == secondary.device,
        f"{name}_weight_scale_2 must use the weight device",
    )
    _mxfp4_require(
        bool(torch.isfinite(scale).all().item()) and bool((scale > 0).all().item()),
        f"{name}_weight_scale_2 must contain finite positive values",
    )
    return (secondary * scale).contiguous()


def prepare_weights(
    l1_weights: MXFP4CheckpointWeights,
    l2_weights: MXFP4CheckpointWeights,
    l1_weight_scale_2: Optional[torch.Tensor] = None,
    l2_weight_scale_2: Optional[torch.Tensor] = None,
) -> Tuple[MXFP4ProcessedWeights, MXFP4ProcessedWeights]:
    """Prepare packed routed weights for the SM90 Gluon MXFP4 path."""
    l1 = _validate_checkpoint_mxfp4_weights(l1_weights, "L1 MXFP4")
    l2 = _validate_checkpoint_mxfp4_weights(l2_weights, "L2 MXFP4")
    _validate_mxfp4_layer_pair(l1, l2)
    _mxfp4_require(
        (l1_weight_scale_2 is None) == (l2_weight_scale_2 is None),
        "L1 and L2 weight_scale_2 must both be provided or both omitted",
    )

    l1_processed = _process_mxfp4_e8m0(*l1, interleave_rows=True)
    l2_processed = _process_mxfp4_e8m0(*l2)
    l1_weight, l1_sf, l1_secondary = l1_processed
    l2_weight, l2_sf, l2_secondary = l2_processed
    l1_sf = _interleave_mxfp4_rows(l1_sf)
    if l1_weight_scale_2 is not None:
        l1_secondary = _apply_mxfp4_weight_scale_2(
            l1_weight_scale_2,
            l1_secondary,
            "l1",
        )
        l2_secondary = _apply_mxfp4_weight_scale_2(
            l2_weight_scale_2,
            l2_secondary,
            "l2",
        )
    hidden = l2_weight.shape[1]
    if _uses_coalesced_mxfp4_scales(hidden):
        l1_sf = _transpose_mxfp4_scales(l1_sf)
        l2_sf = _transpose_mxfp4_scales(l2_sf)
    return (l1_weight, l1_sf, l1_secondary), (
        l2_weight.contiguous(),
        l2_sf.contiguous(),
        l2_secondary,
    )


def _validate_processed_mxfp4_weights(
    l1_weights: MXFP4ProcessedWeights,
    l2_weights: MXFP4ProcessedWeights,
) -> None:
    """Validate the exact processed-weight ABI consumed by Gluon kernels."""
    if not isinstance(l1_weights, tuple) or not isinstance(l2_weights, tuple):
        raise TypeError("processed MXFP4 weights must be tuples")
    if len(l1_weights) != 3 or len(l2_weights) != 3:
        raise ValueError("processed MXFP4 L1/L2 weights must be triples")

    logical = []
    for weights, name in ((l1_weights, "L1"), (l2_weights, "L2")):
        weight, sf, secondary = weights
        if weight.dtype != torch.int8 or sf.dtype != torch.uint8:
            raise TypeError(
                f"processed {name} MXFP4 requires int8 weight and uint8 scale"
            )
        if secondary.dtype != torch.float32:
            raise TypeError(f"processed {name} secondary scale must be float32")
        _mxfp4_require(
            weight.is_contiguous() and sf.is_contiguous() and secondary.is_contiguous(),
            f"processed {name} MXFP4 tensors must be contiguous",
        )
        _mxfp4_require(
            weight.device == sf.device == secondary.device,
            f"processed {name} MXFP4 tensors must use one device",
        )
        _mxfp4_require(
            weight.ndim == 3
            and sf.ndim == 3
            and weight.shape[:2] == sf.shape[:2]
            and weight.shape[2] == sf.shape[2] * 16,
            f"processed {name} MXFP4 has an invalid packed shape",
        )
        _mxfp4_require(
            secondary.shape == (weight.shape[0],),
            f"processed {name} secondary scale must have shape [E]",
        )
        logical.append((weight, sf))
    _validate_mxfp4_layer_pair(logical[0], logical[1])


# mxfp4/dispatch.py


@gluon.jit
def mxfp4_dispatch_partition(
    task_state,
    dispatch_state,
    symmetric_state,
    descs,
    barriers,
    buffers,
    expert_send_state,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    block_m: gl.constexpr,
    num_sms: gl.constexpr,
    num_experts: gl.constexpr,
    num_global_experts: gl.constexpr,
    num_routes: gl.constexpr,
    experts_per_rank: gl.constexpr,
    max_routes: gl.constexpr,
    rank: gl.constexpr,
    world_size: gl.constexpr,
    dispatch_worker: gl.constexpr,
    num_dispatch_workers: gl.constexpr,
):
    """Dispatch with the compact register footprint of the MXFP4 pipeline.

    Keep source-rank traversal scalar here: the FP8 backend's cached warp
    scan increases register pressure in the two-CTA MXFP4 specialization and
    causes spills in its expanded-weight pipeline. Count publication and TMA
    descriptors retain the shared symmetric-context contract.
    """
    dispatch_counter = task_state[2]
    (
        pool_acts,
        pool_acts_sf,
        pool_topk_weights,
        token_src_metadata,
        num_padded_sf_pool_tokens,
        input_topk_idx,
        actual_num_pool_rows,
    ) = dispatch_state
    (
        peer_input_sf_ptrs,
        peer_input_topk_weights_ptrs,
        symmetric_source_routes,
        symmetric_recv_count,
        peer_source_routes_ptrs,
        peer_recv_count_ptrs,
        peer_expert_state_ptrs,
        dispatch_barrier,
        peer_dispatch_barrier_ptrs,
    ) = symmetric_state
    # DeepGEMM assigns one token stream to each dispatch warp.  Model the
    # two streams as separate one-warp specialized partitions.
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    peer_acts_descs, pool_acts_desc = descs
    tma_load_barriers = barriers
    pull_buffer = buffers
    sf_groups: gl.constexpr = hidden // 128
    sf_per_lane: gl.constexpr = (sf_groups + 31) // 32
    sf_capacity: gl.constexpr = sf_per_lane * 32
    sf_layout: gl.constexpr = gl.BlockedLayout(
        [sf_per_lane],
        [32],
        [1],
        [0],
    )
    sf_offsets = gl.arange(0, sf_capacity, layout=sf_layout)
    dispatch_pid = gl.program_id(0) * num_dispatch_workers + dispatch_worker
    num_global_dispatch_workers: gl.constexpr = num_sms * num_dispatch_workers

    # DeepGEMM first reserves source-local per-expert slots, then uses
    # ordinary remote stores.  Gluon's shared-memory descriptor does
    # not expose block-scope atomics, so reserve directly in this
    # rank's local L2 rather than issuing one system-scope atomic per
    # route on the destination rank.  The old value is the unique
    # source-rank slot later consumed by the remote expert.
    route_offsets = gl.arange(0, 32, layout=layout)
    route_ones = gl.full([32], 1, gl.int64, layout=layout)
    route_base = dispatch_pid * 32
    while route_base < num_routes:
        route = route_base + route_offsets
        valid_route = route < num_routes
        routed_expert = gl.load(
            input_topk_idx + route,
            mask=valid_route,
            other=-1,
        )
        valid_route = valid_route & (routed_expert >= 0)
        safe_routed_expert = gl.where(valid_route, routed_expert, 0)
        target_rank = safe_routed_expert // experts_per_rank
        local_expert = safe_routed_expert - target_rank * experts_per_rank
        remote_routes = gl.load(
            peer_source_routes_ptrs + target_rank,
            mask=valid_route,
            other=0,
        ).to(gl.pointer_type(gl.int32))
        slot = gl.atomic_add(
            expert_send_state + safe_routed_expert,
            route_ones,
            mask=valid_route,
            sem="relaxed",
            scope="gpu",
        )
        gl.store(
            remote_routes
            + (rank * experts_per_rank + local_expert) * max_routes
            + slot,
            route,
            mask=valid_route,
        )
        route_base += num_global_dispatch_workers * 32

    gl.atomic_add(
        dispatch_counter,
        1,
        sem="release",
        scope="gpu",
    )
    local_routes_ready = _load_i32_acquire_gpu(dispatch_counter)
    while local_routes_ready < num_global_dispatch_workers:
        local_routes_ready = _load_i32_acquire_gpu(dispatch_counter)

    # Once every local worker has finished its remote route stores,
    # publish this rank's send count with one ordinary remote count
    # store and one system-release packed-state atomic per expert.
    # The acquire above plus this release forms the visibility chain
    # from every route store to the remote expert-state consumer.
    global_expert = dispatch_pid
    while global_expert < num_global_experts:
        target_rank = global_expert // experts_per_rank
        local_expert = global_expert - target_rank * experts_per_rank
        source_count = gl.load(expert_send_state + global_expert).to(gl.int32)
        remote_counts = gl.load(peer_recv_count_ptrs + target_rank).to(
            gl.pointer_type(gl.int32)
        )
        gl.store(
            remote_counts + rank * experts_per_rank + local_expert,
            source_count,
        )
        remote_expert_state = gl.load(peer_expert_state_ptrs + target_rank).to(
            gl.pointer_type(gl.int64)
        )
        packed_contribution = source_count.to(gl.int64) + 4294967296
        gl.atomic_add(
            remote_expert_state + local_expert,
            packed_contribution,
            sem="release",
            scope="sys",
        )
        global_expert += num_global_dispatch_workers

    # The peer signal is the publication-complete notification.  It
    # must not race a late worker's packed state atomic.
    gl.atomic_add(
        dispatch_counter,
        1,
        sem="release",
        scope="gpu",
    )
    if dispatch_worker == 0 and gl.program_id(0) == 0:
        local_publication_ready = _load_i32_acquire_gpu(dispatch_counter)
        while local_publication_ready < 2 * num_global_dispatch_workers:
            local_publication_ready = _load_i32_acquire_gpu(dispatch_counter)
        _peer_barrier_arrive_and_wait(
            peer_dispatch_barrier_ptrs,
            dispatch_barrier,
            world_size,
            1,
            layout,
            32,
        )

    # Snapshot every expert count once per dispatch warp.  The one-warp
    # layout lets lanes fetch experts in parallel; later token traversal
    # selects counts from registers instead of issuing a sequential chain of
    # system-scope loads for every worker.
    # One complete dispatch warp owns this snapshot. Keep at least one value
    # per lane so tiny local expert sets do not lower to sub-warp tensors.
    dispatch_count_capacity: gl.constexpr = max(
        triton.next_power_of_2(num_experts),
        32,
    )
    dispatch_counts_per_lane: gl.constexpr = max(
        dispatch_count_capacity // 32,
        1,
    )
    dispatch_count_layout: gl.constexpr = gl.BlockedLayout(
        [dispatch_counts_per_lane],
        [32],
        [1],
        [0],
    )
    dispatch_count_offsets = gl.arange(
        0,
        dispatch_count_capacity,
        layout=dispatch_count_layout,
    )
    dispatch_stored_counts = _load_packed_expert_counts(
        task_state[0],
        dispatch_count_offsets,
        num_experts,
        world_size,
    )

    if dispatch_worker == 0 and gl.program_id(0) == 0:
        num_pool_blocks = gl.sum(
            gl.where(
                dispatch_count_offsets < num_experts,
                (dispatch_stored_counts + block_m - 1) // block_m,
                0,
            ),
            axis=0,
        )
        gl.store(actual_num_pool_rows, num_pool_blocks * block_m)

    sf_block_m: gl.constexpr = (block_m + 127) // 128 * 128

    # DeepGEMM stripes only valid routed tokens over the persistent grid.
    # Pool storage remains expert/block padded for TMA, but padding rows do
    # not need to be materialized: GEMM may read stale padding because all
    # observable epilogues are guarded by valid_m.  Besides removing the
    # useless input/SF traffic, publishing only valid arrivals lets the A
    # loader start a partial expert block as soon as its real tokens land.
    dispatch_token = dispatch_pid
    current_expert = 0
    expert_token_begin = 0
    expert_token_end = 0
    expert_pool_block = 0
    current_expert_count = _packed_expert_count_from_cache(
        dispatch_stored_counts,
        dispatch_count_offsets,
        0,
    )
    expert_token_end = current_expert_count
    while dispatch_token >= expert_token_end and current_expert < num_experts:
        expert_pool_block += (current_expert_count + block_m - 1) // block_m
        current_expert += 1
        expert_token_begin = expert_token_end
        if current_expert < num_experts:
            current_expert_count = _packed_expert_count_from_cache(
                dispatch_stored_counts,
                dispatch_count_offsets,
                current_expert,
            )
        else:
            current_expert_count = 0
        expert_token_end += current_expert_count

    while current_expert < num_experts:
        local_row = dispatch_token - expert_token_begin
        pool_row = expert_pool_block * block_m + local_row
        pool_block = pool_row // block_m

        selected_rank = 0
        slot = local_row
        round_offset = 0
        token_idx_in_rank = 0
        found_route = False
        while not found_route:
            num_active = 0
            round_length = 0x7FFFFFFF
            for source_rank in gl.static_range(world_size):
                rank_count = gl.load(
                    symmetric_recv_count
                    + source_rank * experts_per_rank
                    + current_expert
                )
                remaining = gl.maximum(rank_count - round_offset, 0)
                is_active = remaining > 0
                num_active += is_active.to(gl.int32)
                round_length = gl.where(
                    is_active,
                    gl.minimum(round_length, remaining),
                    round_length,
                )
            round_tokens = round_length * num_active
            if slot < round_tokens:
                desired_rank = slot - (slot // num_active) * num_active
                num_seen = 0
                for source_rank in gl.static_range(world_size):
                    rank_count = gl.load(
                        symmetric_recv_count
                        + source_rank * experts_per_rank
                        + current_expert
                    )
                    is_active = rank_count > round_offset
                    choose = is_active and num_seen == desired_rank
                    selected_rank = gl.where(
                        choose,
                        source_rank,
                        selected_rank,
                    )
                    num_seen += is_active.to(gl.int32)
                token_idx_in_rank = round_offset + slot // num_active
                found_route = True
            else:
                slot -= round_tokens
                round_offset += round_length
        route = gl.load(
            symmetric_source_routes
            + (selected_rank * experts_per_rank + current_expert) * max_routes
            + token_idx_in_rank,
        )
        remote_sf = gl.load(peer_input_sf_ptrs + selected_rank).to(
            gl.pointer_type(gl.float32)
        )
        remote_weights = gl.load(peer_input_topk_weights_ptrs + selected_rank).to(
            gl.pointer_type(gl.float32)
        )
        source_token = route // topk
        source_slot = route - source_token * topk
        tma_phase = (dispatch_token // num_global_dispatch_workers) & 1
        mbarrier.expect(
            tma_load_barriers,
            pool_acts_desc.block_type.nbytes,
        )
        for source_rank in gl.static_range(world_size):
            if selected_rank == source_rank:
                tma.async_copy_global_to_shared(
                    peer_acts_descs[source_rank],
                    [source_token, 0, 0],
                    tma_load_barriers,
                    pull_buffer,
                )
        pool_block = pool_row // block_m
        row_in_block = pool_row - pool_block * block_m
        sf_pool_row = pool_block * sf_block_m + row_in_block
        valid_sf = sf_offsets < sf_groups
        scales = gl.load(
            remote_sf + source_token * sf_groups + sf_offsets,
            mask=valid_sf,
            other=0.0,
        )
        gl.store(
            pool_acts_sf + sf_offsets * num_padded_sf_pool_tokens + sf_pool_row,
            scales,
            mask=valid_sf,
        )
        weight = gl.load(
            remote_weights + route,
        )
        gl.store(pool_topk_weights + pool_row, weight)
        mbarrier.wait(tma_load_barriers, tma_phase)
        tma.async_copy_shared_to_global(
            pool_acts_desc,
            [pool_row, 0, 0],
            pull_buffer,
        )
        tma.store_wait(0)
        gl.store(
            token_src_metadata + pool_row * 3,
            selected_rank,
        )
        gl.store(
            token_src_metadata + pool_row * 3 + 1,
            source_token,
        )
        gl.store(
            token_src_metadata + pool_row * 3 + 2,
            source_slot,
        )
        gl.atomic_add(
            task_state[1] + pool_block,
            1,
            sem="release",
            scope="gpu",
        )
        dispatch_token += num_global_dispatch_workers
        while dispatch_token >= expert_token_end and current_expert < num_experts:
            expert_pool_block += (current_expert_count + block_m - 1) // block_m
            current_expert += 1
            expert_token_begin = expert_token_end
            if current_expert < num_experts:
                current_expert_count = _packed_expert_count_from_cache(
                    dispatch_stored_counts,
                    dispatch_count_offsets,
                    current_expert,
                )
            else:
                current_expert_count = 0
            expert_token_end += current_expert_count


# mxfp4/producers.py


@gluon.jit
def _load_natural_mxfp4_scale_stage(
    scale_ptr,
    scale_buffer,
    flat_b_row_start,
    k_tile,
    scale_stride_k: gl.constexpr,
):
    """Stage one natural-layout [N128,K32x4] scale tile without TMA."""
    scale_layout: gl.constexpr = gl.BlockedLayout([16], [32], [1], [0])
    scale_offsets = gl.arange(
        0,
        _MXFP4_BLOCK_N * (_MXFP4_BLOCK_K // 32),
        layout=scale_layout,
    )
    local_n = scale_offsets // (_MXFP4_BLOCK_K // 32)
    local_k32 = scale_offsets % (_MXFP4_BLOCK_K // 32)
    global_offsets = (
        flat_b_row_start.to(gl.int64) * scale_stride_k
        + local_n.to(gl.int64) * scale_stride_k
        + k_tile * (_MXFP4_BLOCK_K // 32)
        + local_k32.to(gl.int64)
    )
    scales = gl.load(scale_ptr + global_offsets)
    scale_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    scale_view = scale_buffer._reinterpret(
        gl.uint8,
        [_MXFP4_BLOCK_N * (_MXFP4_BLOCK_K // 32)],
        scale_smem_layout,
    )
    scale_view.store(scales)
    # The elected lane publishes its mbarrier arrival after all producer-lane
    # stores, matching DeepGEMM's __syncwarp-before-arrive ordering.
    partition_barrier()


@gluon.jit
def mxfp4_b_producer_partition(
    l1_b_desc,
    l1_b_sf_desc,
    l2_b_desc,
    l2_b_sf_desc,
    l1_weight_scales,
    l2_weight_scales,
    expert_state,
    barriers,
    buffers,
    E: gl.constexpr,
    l1_n: gl.constexpr,
    l1_k: gl.constexpr,
    l2_n: gl.constexpr,
    l2_k: gl.constexpr,
    l1_n_blocks: gl.constexpr,
    l2_n_blocks: gl.constexpr,
    num_experts_per_wave: gl.constexpr,
    num_sms: gl.constexpr,
    scheduler_count_capacity: gl.constexpr,
    scheduler_counts_per_lane: gl.constexpr,
    world_size: gl.constexpr,
):
    """Stage packed MXFP4 B tiles and their relative UE8M0 scales."""
    stage_empty, stage_ready = barriers
    b_buffers, sfb_buffers = buffers
    num_stages: gl.constexpr = b_buffers.type.shape[0]
    block_n: gl.constexpr = b_buffers.type.shape[1]
    block_k: gl.constexpr = _MXFP4_BLOCK_K

    scheduler_layout: gl.constexpr = gl.BlockedLayout(
        [scheduler_counts_per_lane],
        [32],
        [1],
        [0],
    )
    count_offsets = gl.arange(
        0,
        scheduler_count_capacity,
        layout=scheduler_layout,
    )
    stored_counts = _load_packed_expert_counts(
        expert_state,
        count_offsets,
        E,
        world_size,
    )

    block_idx = gl.program_id(0)
    scheduler_expert = 0
    scheduler_phase = 1
    current_count = scheduler_count(
        stored_counts,
        count_offsets,
        scheduler_expert,
    )
    current_pool_block_offset = 0
    pipeline_tile = 0
    (
        task_phase,
        task_expert,
        task_m_block,
        task_n_block,
        task_pool_block,
        valid_count,
        block_idx,
        scheduler_expert,
        scheduler_phase,
        current_count,
        current_pool_block_offset,
    ) = scheduler_next(
        stored_counts,
        count_offsets,
        block_idx,
        scheduler_expert,
        scheduler_phase,
        current_count,
        current_pool_block_offset,
        E,
        l1_n_blocks,
        l2_n_blocks,
        num_experts_per_wave,
        num_sms,
        _MXFP4_BLOCK_M,
    )

    while task_phase != 0:
        phase_n = gl.where(task_phase == 1, l1_n, l2_n)
        phase_k = gl.where(task_phase == 1, l1_k, l2_k)
        flat_b_row_start = task_expert * phase_n + task_n_block * block_n
        num_k_tiles = phase_k // block_k
        k_tile = 0
        while k_tile < num_k_tiles:
            stage = pipeline_tile % num_stages
            pipe_phase = pipeline_tile // num_stages & 1
            mbarrier.wait(stage_empty.index(stage), pipe_phase ^ 1)
            if task_phase == 1:
                if l2_n <= _MXFP4_COALESCED_SCALE_MAX_HIDDEN_CONSTEXPR:
                    mbarrier.expect(
                        stage_ready.index(stage),
                        l1_b_desc.block_type.nbytes + l1_b_sf_desc.block_type.nbytes,
                    )
                else:
                    _load_natural_mxfp4_scale_stage(
                        l1_weight_scales,
                        sfb_buffers.index(stage),
                        flat_b_row_start,
                        k_tile,
                        l1_k // 32,
                    )
                    mbarrier.expect(
                        stage_ready.index(stage),
                        l1_b_desc.block_type.nbytes,
                    )
                tma.async_copy_global_to_shared(
                    l1_b_desc,
                    [flat_b_row_start, k_tile * (block_k // 2)],
                    stage_ready.index(stage),
                    b_buffers.index(stage),
                )
                if l2_n <= _MXFP4_COALESCED_SCALE_MAX_HIDDEN_CONSTEXPR:
                    sf_start = (
                        (task_expert * (l1_k // block_k) + k_tile) * l1_n
                        + task_n_block * block_n
                    ) * (block_k // 32)
                    tma.async_copy_global_to_shared(
                        l1_b_sf_desc,
                        [sf_start],
                        stage_ready.index(stage),
                        sfb_buffers.index(stage),
                    )
            else:
                if l2_n <= _MXFP4_COALESCED_SCALE_MAX_HIDDEN_CONSTEXPR:
                    mbarrier.expect(
                        stage_ready.index(stage),
                        l2_b_desc.block_type.nbytes + l2_b_sf_desc.block_type.nbytes,
                    )
                else:
                    _load_natural_mxfp4_scale_stage(
                        l2_weight_scales,
                        sfb_buffers.index(stage),
                        flat_b_row_start,
                        k_tile,
                        l2_k // 32,
                    )
                    mbarrier.expect(
                        stage_ready.index(stage),
                        l2_b_desc.block_type.nbytes,
                    )
                tma.async_copy_global_to_shared(
                    l2_b_desc,
                    [flat_b_row_start, k_tile * (block_k // 2)],
                    stage_ready.index(stage),
                    b_buffers.index(stage),
                )
                if l2_n <= _MXFP4_COALESCED_SCALE_MAX_HIDDEN_CONSTEXPR:
                    sf_start = (
                        (task_expert * (l2_k // block_k) + k_tile) * l2_n
                        + task_n_block * block_n
                    ) * (block_k // 32)
                    tma.async_copy_global_to_shared(
                        l2_b_sf_desc,
                        [sf_start],
                        stage_ready.index(stage),
                        sfb_buffers.index(stage),
                    )
            pipeline_tile += 1
            k_tile += 1

        (
            task_phase,
            task_expert,
            task_m_block,
            task_n_block,
            task_pool_block,
            valid_count,
            block_idx,
            scheduler_expert,
            scheduler_phase,
            current_count,
            current_pool_block_offset,
        ) = scheduler_next(
            stored_counts,
            count_offsets,
            block_idx,
            scheduler_expert,
            scheduler_phase,
            current_count,
            current_pool_block_offset,
            E,
            l1_n_blocks,
            l2_n_blocks,
            num_experts_per_wave,
            num_sms,
            _MXFP4_BLOCK_M,
        )


@gluon.jit
def _extract_mxfp4_exponent_prmt(
    scale_word,
    k32: gl.constexpr,
):
    """Extract one relative UE8M0 byte with a single PRMT."""
    if k32 == 0:
        return gl.inline_asm_elementwise(
            "prmt.b32 $0, $1, 0, 0x4440;",
            "=r,r",
            [scale_word],
            dtype=gl.uint32,
            is_pure=True,
            pack=1,
        )
    elif k32 == 1:
        return gl.inline_asm_elementwise(
            "prmt.b32 $0, $1, 0, 0x4441;",
            "=r,r",
            [scale_word],
            dtype=gl.uint32,
            is_pure=True,
            pack=1,
        )
    elif k32 == 2:
        return gl.inline_asm_elementwise(
            "prmt.b32 $0, $1, 0, 0x4442;",
            "=r,r",
            [scale_word],
            dtype=gl.uint32,
            is_pure=True,
            pack=1,
        )
    return gl.inline_asm_elementwise(
        "prmt.b32 $0, $1, 0, 0x4443;",
        "=r,r",
        [scale_word],
        dtype=gl.uint32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _decode_mxfp4_b_stage(
    packed_buffer,
    sfb_buffer,
    expanded_buffer,
    use_prmt_exponent: gl.constexpr,
    prefetch_k32: gl.constexpr = False,
):
    """Expand one packed-weight packed E2M1 tile to B128-swizzled E4M3.

    Four adjacent packed bytes are decoded by one PTX instance. Preprocessing
    has already moved the packed-word sign bits, so the two result registers
    are the first and second groups of four logical K values. The generated
    E4M3 values intentionally carry a 1/64 bias, compensated during promotion.
    """
    # Match the routed decode ownership exactly: the 128 math threads own
    # one complete N row each.  Decode one K32 group at a time so the four
    # packed words and their eight expanded words die before the next group.
    # In particular, do not materialize exponent broadcasts through
    # convert_layout: that lowers to shared stores, warp barriers, and
    # ldmatrix reloads in the Gluon SM90 backend.
    word_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1],
        [32, 1],
        [_MXFP4_MATH_WARPS_CONSTEXPR, 1],
        [1, 0],
    )
    packed_word_smem_layout: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=64,
        transposed=False,
        element_bitwidth=32,
        rank=2,
    )
    packed_words = packed_buffer._reinterpret(
        gl.uint32,
        [_MXFP4_BLOCK_N, _MXFP4_BLOCK_K // 8],
        packed_word_smem_layout,
    )
    scale_word_smem_layout: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=0,
        transposed=False,
        element_bitwidth=32,
        rank=2,
    )
    scale_words = sfb_buffer._reinterpret(
        gl.uint32,
        [_MXFP4_BLOCK_N, 1],
        scale_word_smem_layout,
    ).load(word_layout)
    expanded_word_smem_layout: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=128,
        transposed=False,
        element_bitwidth=32,
        rank=2,
    )
    expanded_words = expanded_buffer._reinterpret(
        gl.uint32,
        [_MXFP4_BLOCK_N, _MXFP4_BLOCK_K // 4],
        expanded_word_smem_layout,
    )
    if prefetch_k32:
        packed = packed_words.slice(0, 4, dim=1).load(word_layout)
    for k32 in gl.static_range(_MXFP4_BLOCK_K // 32):
        if prefetch_k32:
            if k32 + 1 < _MXFP4_BLOCK_K // 32:
                next_packed = packed_words.slice((k32 + 1) * 4, 4, dim=1).load(
                    word_layout
                )
        else:
            packed = packed_words.slice(k32 * 4, 4, dim=1).load(word_layout)
        if use_prmt_exponent:
            exponent = _extract_mxfp4_exponent_prmt(
                scale_words,
                k32,
            )
        else:
            exponent = (scale_words >> (k32 * 8)) & 0xFF
        decoded_lo, decoded_hi = gl.inline_asm_elementwise(
            """
            {
            .reg .b32 lookup_lo, lookup_hi, temp0, temp1, out_lo, out_hi;
            mad.lo.u32 lookup_lo, $3, 0x08080800, 0x0c080000;
            mad.lo.u32 lookup_hi, $3, 0x08080808, 0x1c181410;
            shl.b32 out_lo, $2, 4;
            and.b32 temp0, $2, 0x77777777;
            prmt.b32 temp1, lookup_lo, lookup_hi, temp0;
            lop3.b32 out_lo, out_lo, 0x80808080, temp1, 0xea;
            shr.u32 temp0, temp0, 16;
            prmt.b32 temp1, lookup_lo, lookup_hi, temp0;
            lop3.b32 out_hi, $2, 0x80808080, temp1, 0xea;
            mov.b32 $0, out_lo;
            mov.b32 $1, out_hi;
            }
            """,
            "=r,=r,r,r",
            [packed, exponent],
            dtype=(gl.uint32, gl.uint32),
            is_pure=True,
            pack=1,
        )
        decoded = gl.join(decoded_lo, decoded_hi).reshape([_MXFP4_BLOCK_N, 8])
        expanded_words.slice(k32 * 8, 8, dim=1).store(decoded)
        if prefetch_k32 and k32 + 1 < _MXFP4_BLOCK_K // 32:
            packed = next_packed
    fence_async_shared()
    partition_barrier()


@gluon.jit
def _mxfp4_promotion_scale(activation_scale, secondary):
    """Apply packed-weight's x64 E2M1-to-E4M3 bias without endpoint overflow."""
    compensate_secondary = secondary <= (2.0**121)
    compensated_secondary = gl.where(
        compensate_secondary,
        secondary * 64.0,
        secondary,
    )
    compensated_activation = gl.where(
        compensate_secondary,
        activation_scale,
        activation_scale * 64.0,
    )
    return compensated_activation * compensated_secondary


# mxfp4/math.py


@gluon.jit
def _mxfp4_fc1_swap_mainloop(
    barriers,
    buffers,
    l1_mxfp4_secondary,
    task_expert,
    pipeline_tile,
    l1_k: gl.constexpr,
    n_swap: gl.constexpr,
    use_prmt_exponent: gl.constexpr,
    precompute_n8_scale: gl.constexpr,
    early_release_stage: gl.constexpr,
):
    """FC1 swapAB mainloop for decoded MXFP4 weights."""
    stage_empty, stage_ready = barriers
    (
        a_buffers,
        b_buffers,
        sfa_lo_buffers,
        sfa_hi_buffers,
        sfb_buffers,
        expanded_b_buffers,
    ) = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_k: gl.constexpr = a_buffers.type.shape[2]
    num_k_tiles: gl.constexpr = l1_k // block_k

    swap_mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[4, 1],
        instr_shape=[16, n_swap, 32],
    )
    final = gl.zeros(
        (_MXFP4_BLOCK_N, n_swap),
        dtype=gl.float32,
        layout=swap_mma_layout,
    )
    # This expert's secondary scale is invariant across K tiles.
    channel_scale = gl.load(l1_mxfp4_secondary + task_expert)
    first_stage = pipeline_tile % num_stages
    first_phase = pipeline_tile // num_stages & 1
    mbarrier.wait(stage_ready.index(first_stage), first_phase)
    _decode_mxfp4_b_stage(
        b_buffers.index(first_stage),
        sfb_buffers.index(first_stage),
        expanded_b_buffers.index(0),
        use_prmt_exponent,
        n_swap <= 32,
    )
    for k_tile in range(num_k_tiles):
        tile = pipeline_tile + k_tile
        stage = tile % num_stages
        expanded_slot = k_tile & 1
        b_stage = expanded_b_buffers.index(expanded_slot)

        partial = gl.zeros(
            (_MXFP4_BLOCK_N, n_swap),
            dtype=gl.float32,
            layout=swap_mma_layout,
        )
        partial = warpgroup_mma(
            b_stage,
            a_buffers.index(stage).slice(0, n_swap, dim=0).permute((1, 0)),
            partial,
            is_async=True,
            use_acc=False,
            max_num_imprecise_acc=128,
        )
        token_scale = (
            sfa_lo_buffers.index(stage)
            .slice(
                0,
                n_swap,
                dim=0,
            )
            .load(gl.SliceLayout(0, swap_mma_layout))
        )
        if precompute_n8_scale:
            combined_scale = _mxfp4_promotion_scale(
                token_scale,
                channel_scale,
            )
        if k_tile + 1 < num_k_tiles:
            next_tile = tile + 1
            next_stage = next_tile % num_stages
            next_phase = next_tile // num_stages & 1
            mbarrier.wait(stage_ready.index(next_stage), next_phase)
            _decode_mxfp4_b_stage(
                b_buffers.index(next_stage),
                sfb_buffers.index(next_stage),
                expanded_b_buffers.index(expanded_slot ^ 1),
                use_prmt_exponent,
                n_swap <= 32,
            )
        partial = warpgroup_mma_wait(
            num_outstanding=0,
            deps=(partial,),
        )
        if not precompute_n8_scale:
            combined_scale = _mxfp4_promotion_scale(
                token_scale,
                channel_scale,
            )
        if early_release_stage:
            # The wait ends expanded-B reads and token_scale is already in
            # registers. Let the producer refill this stage during the
            # remaining register-only promotion.
            partition_barrier()
            mbarrier.arrive(stage_empty.index(stage), count=1)
        final += partial * combined_scale[None, :]
        if not early_release_stage:
            partition_barrier()
            mbarrier.arrive(stage_empty.index(stage), count=1)

    return final


@gluon.jit
def _mxfp4_fc2_swap_mainloop(
    barriers,
    buffers,
    l2_mxfp4_secondary,
    task_expert,
    pipeline_tile,
    l2_k: gl.constexpr,
    n_swap: gl.constexpr,
    use_prmt_exponent: gl.constexpr,
    precompute_n8_scale: gl.constexpr,
    early_release_stage: gl.constexpr,
):
    """FC2 swapAB mainloop with independent low/high K64 A scales."""
    stage_empty, stage_ready = barriers
    (
        a_buffers,
        b_buffers,
        sfa_lo_buffers,
        sfa_hi_buffers,
        sfb_buffers,
        expanded_b_buffers,
    ) = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_k: gl.constexpr = a_buffers.type.shape[2]
    num_k_tiles: gl.constexpr = l2_k // block_k

    swap_mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[4, 1],
        instr_shape=[16, n_swap, 32],
    )
    final = gl.zeros(
        (_MXFP4_BLOCK_N, n_swap),
        dtype=gl.float32,
        layout=swap_mma_layout,
    )
    # This expert's secondary scale is invariant across K tiles.
    l2_weight_scale = gl.load(l2_mxfp4_secondary + task_expert)
    first_stage = pipeline_tile % num_stages
    first_phase = pipeline_tile // num_stages & 1
    mbarrier.wait(stage_ready.index(first_stage), first_phase)
    _decode_mxfp4_b_stage(
        b_buffers.index(first_stage),
        sfb_buffers.index(first_stage),
        expanded_b_buffers.index(0),
        use_prmt_exponent,
        n_swap <= 32,
    )
    for k_tile in range(num_k_tiles):
        tile = pipeline_tile + k_tile
        stage = tile % num_stages
        expanded_slot = k_tile & 1
        channel_stage = expanded_b_buffers.index(expanded_slot)

        token_stage = a_buffers.index(stage).slice(0, n_swap, dim=0)

        partial_lo = gl.zeros(
            (_MXFP4_BLOCK_N, n_swap),
            dtype=gl.float32,
            layout=swap_mma_layout,
        )
        partial_lo = warpgroup_mma(
            channel_stage.slice(0, 64, dim=1),
            token_stage.slice(0, 64, dim=1).permute((1, 0)),
            partial_lo,
            is_async=True,
            use_acc=False,
            max_num_imprecise_acc=64,
        )
        if precompute_n8_scale:
            token_scale_lo = (
                sfa_lo_buffers.index(stage)
                .slice(
                    0,
                    n_swap,
                    dim=0,
                )
                .load(gl.SliceLayout(0, swap_mma_layout))
            )
            scale_lo = _mxfp4_promotion_scale(
                token_scale_lo,
                l2_weight_scale,
            )
        partial_lo = warpgroup_mma_wait(
            num_outstanding=0,
            deps=(partial_lo,),
        )
        if not precompute_n8_scale:
            token_scale_lo = (
                sfa_lo_buffers.index(stage)
                .slice(
                    0,
                    n_swap,
                    dim=0,
                )
                .load(gl.SliceLayout(0, swap_mma_layout))
            )
            scale_lo = _mxfp4_promotion_scale(
                token_scale_lo,
                l2_weight_scale,
            )
        final += partial_lo * scale_lo[None, :]

        partial_hi = gl.zeros(
            (_MXFP4_BLOCK_N, n_swap),
            dtype=gl.float32,
            layout=swap_mma_layout,
        )
        partial_hi = warpgroup_mma(
            channel_stage.slice(64, 64, dim=1),
            token_stage.slice(64, 64, dim=1).permute((1, 0)),
            partial_hi,
            is_async=True,
            use_acc=False,
            max_num_imprecise_acc=64,
        )
        token_scale_hi = (
            sfa_hi_buffers.index(stage)
            .slice(
                0,
                n_swap,
                dim=0,
            )
            .load(gl.SliceLayout(0, swap_mma_layout))
        )
        if precompute_n8_scale:
            scale_hi = _mxfp4_promotion_scale(
                token_scale_hi,
                l2_weight_scale,
            )
        if k_tile + 1 < num_k_tiles:
            next_tile = tile + 1
            next_stage = next_tile % num_stages
            next_phase = next_tile // num_stages & 1
            mbarrier.wait(stage_ready.index(next_stage), next_phase)
            _decode_mxfp4_b_stage(
                b_buffers.index(next_stage),
                sfb_buffers.index(next_stage),
                expanded_b_buffers.index(expanded_slot ^ 1),
                use_prmt_exponent,
                n_swap <= 32,
            )
        partial_hi = warpgroup_mma_wait(
            num_outstanding=0,
            deps=(partial_hi,),
        )
        if not precompute_n8_scale:
            scale_hi = _mxfp4_promotion_scale(
                token_scale_hi,
                l2_weight_scale,
            )
        if early_release_stage:
            # No shared operand from this stage remains live after the final
            # wait; overlap producer refill with high-half promotion.
            partition_barrier()
            mbarrier.arrive(stage_empty.index(stage), count=1)
        final += partial_hi * scale_hi[None, :]
        if not early_release_stage:
            partition_barrier()
            mbarrier.arrive(stage_empty.index(stage), count=1)

    return final


@gluon.jit
def _mxfp4_normal_mainloop(
    barriers,
    buffers,
    l1_mxfp4_secondary,
    l2_mxfp4_secondary,
    task_expert,
    pipeline_tile,
    l1_k: gl.constexpr,
    l2_k: gl.constexpr,
    use_prmt_exponent: gl.constexpr,
    linear1: gl.constexpr,
):
    """Specialize the complete MXFP4 K loop for one linear phase.

    Release each packed/A stage after WGMMA and shared scale reads retire,
    before register-only promotion. Preserve the low/high K64 FP32 order.
    """
    stage_empty, stage_ready = barriers
    (
        a_buffers,
        b_buffers,
        sfa_lo_buffers,
        sfa_hi_buffers,
        sfb_buffers,
        expanded_b_buffers,
    ) = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_k: gl.constexpr = a_buffers.type.shape[2]
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[4, 1],
        instr_shape=[16, 128, 32],
    )
    final = gl.zeros(
        (_MXFP4_BLOCK_M, _MXFP4_BLOCK_N),
        dtype=gl.float32,
        layout=mma_layout,
    )
    num_k_tiles: gl.constexpr = (l1_k if linear1 else l2_k) // block_k
    k_tile = 0
    stage = pipeline_tile % num_stages
    pipe_phase = pipeline_tile // num_stages & 1
    mbarrier.wait(
        stage_ready.index(stage),
        pipe_phase,
    )
    _decode_mxfp4_b_stage(
        b_buffers.index(stage),
        sfb_buffers.index(stage),
        expanded_b_buffers.index(0),
        use_prmt_exponent,
    )
    while k_tile < num_k_tiles:
        expanded_slot = k_tile & 1
        b_stage = expanded_b_buffers.index(expanded_slot)

        if linear1:
            partial = gl.zeros(
                (_MXFP4_BLOCK_M, _MXFP4_BLOCK_N),
                dtype=gl.float32,
                layout=mma_layout,
            )
            partial = warpgroup_mma(
                a_buffers.index(stage),
                b_stage.permute((1, 0)),
                partial,
                is_async=True,
                use_acc=False,
                max_num_imprecise_acc=128,
            )
            a_scale = sfa_lo_buffers.index(stage).load(gl.SliceLayout(1, mma_layout))
            column_scale = gl.load(l1_mxfp4_secondary + task_expert)
            if k_tile + 1 < num_k_tiles:
                next_stage = gl.where(stage + 1 == num_stages, 0, stage + 1)
                next_pipe_phase = pipe_phase ^ (next_stage == 0)
                mbarrier.wait(
                    stage_ready.index(next_stage),
                    next_pipe_phase,
                )
                _decode_mxfp4_b_stage(
                    b_buffers.index(next_stage),
                    sfb_buffers.index(next_stage),
                    expanded_b_buffers.index(expanded_slot ^ 1),
                    use_prmt_exponent,
                )
            partial = warpgroup_mma_wait(
                num_outstanding=0,
                deps=(partial,),
            )
            combined_scale = _mxfp4_promotion_scale(
                a_scale,
                column_scale,
            )
            partition_barrier()
            mbarrier.arrive(stage_empty.index(stage), count=1)
            final += partial * combined_scale[:, None]
        else:
            l2_weight_scale = gl.load(l2_mxfp4_secondary + task_expert)
            a_stage = a_buffers.index(stage)
            partial_lo = gl.zeros(
                (_MXFP4_BLOCK_M, _MXFP4_BLOCK_N),
                dtype=gl.float32,
                layout=mma_layout,
            )
            partial_lo = warpgroup_mma(
                a_stage.slice(0, 64, dim=1),
                b_stage.slice(0, 64, dim=1).permute((1, 0)),
                partial_lo,
                is_async=True,
                use_acc=False,
                max_num_imprecise_acc=64,
            )
            partial_lo = warpgroup_mma_wait(
                num_outstanding=0,
                deps=(partial_lo,),
            )
            a_scale_lo = sfa_lo_buffers.index(stage).load(gl.SliceLayout(1, mma_layout))
            combined_scale_lo = _mxfp4_promotion_scale(
                a_scale_lo,
                l2_weight_scale,
            )
            final += partial_lo * combined_scale_lo[:, None]

            partial_hi = gl.zeros(
                (_MXFP4_BLOCK_M, _MXFP4_BLOCK_N),
                dtype=gl.float32,
                layout=mma_layout,
            )
            partial_hi = warpgroup_mma(
                a_stage.slice(64, 64, dim=1),
                b_stage.slice(64, 64, dim=1).permute((1, 0)),
                partial_hi,
                is_async=True,
                use_acc=False,
                max_num_imprecise_acc=64,
            )
            a_scale_hi = sfa_hi_buffers.index(stage).load(gl.SliceLayout(1, mma_layout))
            if k_tile + 1 < num_k_tiles:
                next_stage = gl.where(stage + 1 == num_stages, 0, stage + 1)
                next_pipe_phase = pipe_phase ^ (next_stage == 0)
                mbarrier.wait(
                    stage_ready.index(next_stage),
                    next_pipe_phase,
                )
                _decode_mxfp4_b_stage(
                    b_buffers.index(next_stage),
                    sfb_buffers.index(next_stage),
                    expanded_b_buffers.index(expanded_slot ^ 1),
                    use_prmt_exponent,
                )
            partial_hi = warpgroup_mma_wait(
                num_outstanding=0,
                deps=(partial_hi,),
            )
            combined_scale_hi = _mxfp4_promotion_scale(
                a_scale_hi,
                l2_weight_scale,
            )
            partition_barrier()
            mbarrier.arrive(stage_empty.index(stage), count=1)
            final += partial_hi * combined_scale_hi[:, None]

        stage = gl.where(stage + 1 == num_stages, 0, stage + 1)
        pipe_phase ^= stage == 0
        k_tile += 1

    return final


@gluon.jit
def _store_mxfp4_contiguous_bf16_fragment(
    ptrs,
    values,
    mask,
    width: gl.constexpr,
):
    """Store one lane-owned row fragment as one aligned global transaction."""
    packed_values = values.to(gl.uint16, bitcast=True)
    packed_mask = mask.to(gl.int8)
    if width == 8:
        return gl.inline_asm_elementwise(
            """
            {
            .reg .pred store_pred;
            setp.ne.u32 store_pred, $14, 0;
            @store_pred st.global.v4.b32 [$2], {$10, $11, $12, $13};
            mov.u32 $0, 0;
            mov.u32 $1, 0;
            }
            """,
            "=r,=r,l,l,l,l,l,l,l,l,r,r,r,r,r,r",
            [ptrs, packed_values, packed_mask],
            dtype=gl.int8,
            is_pure=False,
            pack=8,
        )
    else:
        gl.static_assert(width == 4, "BF16 vector width must be 4 or 8")
        return gl.inline_asm_elementwise(
            """
            {
            .reg .pred store_pred;
            setp.ne.u32 store_pred, $7, 0;
            @store_pred st.global.v2.b32 [$1], {$5, $6};
            mov.u32 $0, 0;
            }
            """,
            "=r,l,l,l,l,r,r,r",
            [ptrs, packed_values, packed_mask],
            dtype=gl.int8,
            is_pure=False,
            pack=4,
        )


@gluon.jit
def _mxfp4_fc2_combine_partition(
    output,
    combine_buffer,
    topk_idx,
    fused_barrier,
    peer_fused_barrier_ptrs,
    fc2_scatter_grid_counter,
    combine_cross_rank_ready,
    l2_n: gl.constexpr,
    num_tokens: gl.constexpr,
    topk: gl.constexpr,
    world_size: gl.constexpr,
    num_sms: gl.constexpr,
    num_math_warps: gl.constexpr,
):
    """Publish peer FC2 stores, then reduce source-local top-k slots."""
    # Every route row is striped across the complete math
    # partition.  Match DeepGEMM's epilogue ``sync_scope`` before publishing
    # this CTA into the grid barrier, so the leader's release transitively
    # covers ordinary peer stores issued by every participating warp.
    partition_barrier()
    gl.atomic_add(
        fc2_scatter_grid_counter,
        1,
        sem="release",
        scope="gpu",
    )
    grid_arrived = _load_i32_acquire_gpu(fc2_scatter_grid_counter)
    while grid_arrived < num_sms:
        grid_arrived = _load_i32_acquire_gpu(fc2_scatter_grid_counter)

    # The local grid counter only orders this rank's ordinary FC2 scatter
    # stores before its system-scope peer signal.  Dispatch cleanup waits for
    # combine_cross_rank_ready below instead, matching DeepGEMM's epilogue
    # rendezvous after cross-rank completion.

    if gl.program_id(0) == 0:
        barrier_participant_capacity: gl.constexpr = num_math_warps * 32
        barrier_layout: gl.constexpr = gl.BlockedLayout(
            [1],
            [32],
            [num_math_warps],
            [0],
        )
        _peer_barrier_arrive_and_wait(
            peer_fused_barrier_ptrs,
            fused_barrier,
            world_size,
            1,
            barrier_layout,
            barrier_participant_capacity,
        )
        gl.atomic_add(
            combine_cross_rank_ready,
            1,
            sem="release",
            scope="gpu",
        )

    cross_rank_ready = _load_i32_acquire_gpu(combine_cross_rank_ready)
    while cross_rank_ready < 1:
        cross_rank_ready = _load_i32_acquire_gpu(combine_cross_rank_ready)
    # This is the epilogue side of DeepGEMM's second grid sync: the local
    # leader has acquired every peer's system-scope completion chain, then
    # makes that visibility available to every combine warp before BF16 loads.
    partition_barrier()

    # Small tiles expose enough independent work at low M. Dense batches
    # amortize slot masks and address arithmetic across two 128-bit fragments.
    combine_n: gl.constexpr = (
        128
        if num_tokens <= 32
        else (512 if num_tokens >= (1024 if l2_n == 4096 else 2048) else 256)
    )
    # Dense batches expose independent rows within each warp. Bound the tile
    # to 8192 elements for four math warps so the load/accumulator pair fits
    # the existing register budget even when the channel width is 512.
    combine_rows: gl.constexpr = num_math_warps * (
        2048 // combine_n if num_tokens > 512 else 1
    )
    combine_width: gl.constexpr = 4 if combine_n == 128 else 8
    combine_layout: gl.constexpr = gl.BlockedLayout(
        [1, combine_width],
        [1, 32],
        [num_math_warps, 1],
        [1, 0],
    )
    combine_row_vector = gl.arange(
        0,
        combine_rows,
        layout=gl.SliceLayout(1, combine_layout),
    )
    combine_col_vector = gl.arange(
        0,
        combine_n,
        layout=gl.SliceLayout(0, combine_layout),
    )
    num_combine_n_blocks: gl.constexpr = (l2_n + combine_n - 1) // combine_n
    output_tile = gl.program_id(0)
    num_output_m_blocks = (num_tokens + combine_rows - 1) // combine_rows
    while output_tile < num_output_m_blocks * num_combine_n_blocks:
        output_m_block = output_tile // num_combine_n_blocks
        output_n_block = output_tile % num_combine_n_blocks
        output_rows = output_m_block * combine_rows + combine_row_vector[:, None]
        output_row_mask = output_rows < num_tokens
        valid_slot_mask = output_rows * 0
        for slot in gl.static_range(topk):
            routed_expert = gl.load(
                topk_idx + output_rows * topk + slot,
                mask=output_row_mask,
                other=-1,
            )
            valid_slot_mask = valid_slot_mask | gl.where(
                routed_expert >= 0,
                1 << slot,
                0,
            )
        output_cols = output_n_block * combine_n + combine_col_vector[None, :]
        mask = output_row_mask & (output_cols < l2_n)
        reduced = gl.zeros(
            (combine_rows, combine_n),
            dtype=gl.float32,
            layout=combine_layout,
        )
        values = _load_contiguous_bf16_fragment(
            combine_buffer + output_rows * topk * l2_n + output_cols,
            mask & ((valid_slot_mask & 1) != 0),
            combine_width,
        ).to(gl.float32)
        for slot in gl.static_range(1, topk):
            next_values = _load_contiguous_bf16_fragment(
                combine_buffer + (output_rows * topk + slot) * l2_n + output_cols,
                mask & ((valid_slot_mask & (1 << slot)) != 0),
                combine_width,
            ).to(gl.float32)
            reduced += values
            values = next_values
        reduced += values
        output_ptrs = output + output_rows * l2_n + output_cols
        _store_mxfp4_contiguous_bf16_fragment(
            output_ptrs,
            reduced.to(gl.bfloat16),
            mask,
            combine_width,
        )
        output_tile += num_sms


@gluon.jit
def _mxfp4_fc2_swap_vector_epilogue(
    final,
    token_src_metadata,
    peer_combine_buffer_ptrs,
    task_pool_block,
    task_n_block,
    valid_m,
    l2_n: gl.constexpr,
    max_tokens: gl.constexpr,
    topk: gl.constexpr,
    world_size: gl.constexpr,
    n_swap: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
):
    """Transpose swap output into contiguous source-rank row fragments."""
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 8],
        [2, 16],
        [4, 1],
        [1, 0],
    )
    values = gl.convert_layout(final.permute((1, 0)).to(gl.bfloat16), layout)
    rows = gl.arange(0, n_swap, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, block_n, layout=gl.SliceLayout(0, layout))
    output_rows = task_pool_block * block_m + rows[:, None]
    output_cols = task_n_block * block_n + cols[None, :]
    row_mask = rows[:, None] < valid_m
    source_rank = gl.load(token_src_metadata + output_rows * 3, mask=row_mask, other=-1)
    source_token = gl.load(
        token_src_metadata + output_rows * 3 + 1, mask=row_mask, other=0
    )
    source_slot = gl.load(
        token_src_metadata + output_rows * 3 + 2, mask=row_mask, other=0
    )
    safe_rank = gl.minimum(gl.maximum(source_rank, 0), world_size - 1)
    remote = gl.load(peer_combine_buffer_ptrs + safe_rank).to(
        gl.pointer_type(gl.bfloat16)
    )
    ptrs = remote + (source_token * topk + source_slot) * l2_n + output_cols
    mask = (
        row_mask
        & (source_rank >= 0)
        & (source_rank < world_size)
        & (source_token >= 0)
        & (source_token < max_tokens)
        & (source_slot >= 0)
        & (source_slot < topk)
        & (output_cols < l2_n)
    )
    _store_mxfp4_contiguous_bf16_fragment(ptrs, values, mask, 8)
    partition_barrier()


@gluon.jit
def mxfp4_math_partition(
    barriers,
    buffers,
    l2_store_desc,
    l2_epilogue_buffer,
    l2_acts_sf,
    output,
    combine_buffer,
    topk_idx,
    token_src_metadata,
    peer_combine_buffer_ptrs,
    fused_barrier,
    peer_fused_barrier_ptrs,
    fc2_scatter_grid_counter,
    combine_cross_rank_ready,
    route_weights,
    l2_arrival,
    l1_mxfp4_secondary,
    l2_mxfp4_secondary,
    expert_state,
    l1_n: gl.constexpr,
    l1_k: gl.constexpr,
    l2_n: gl.constexpr,
    l2_k: gl.constexpr,
    E: gl.constexpr,
    l1_n_blocks: gl.constexpr,
    l2_n_blocks: gl.constexpr,
    num_experts_per_wave: gl.constexpr,
    num_sms: gl.constexpr,
    scheduler_count_capacity: gl.constexpr,
    scheduler_counts_per_lane: gl.constexpr,
    num_padded_sf_pool_tokens: gl.constexpr,
    num_tokens: gl.constexpr,
    max_tokens: gl.constexpr,
    topk: gl.constexpr,
    world_size: gl.constexpr,
    activation_clamp: gl.constexpr,
    has_activation_clamp: gl.constexpr,
    fast_math: gl.constexpr,
    use_swap_ab: gl.constexpr,
):
    """Fused MXFP4 math partition executing decode, FC1, FC2, and combine."""
    stage_empty, stage_ready = barriers
    (
        a_buffers,
        b_buffers,
        sfa_lo_buffers,
        sfa_hi_buffers,
        sfb_buffers,
        expanded_b_buffers,
    ) = buffers
    a_buffers.type.shape[0]
    block_k: gl.constexpr = a_buffers.type.shape[2]

    scheduler_layout: gl.constexpr = gl.BlockedLayout(
        [scheduler_counts_per_lane],
        [32],
        [_MXFP4_MATH_WARPS_CONSTEXPR],
        [0],
    )
    count_offsets = gl.arange(
        0,
        scheduler_count_capacity,
        layout=scheduler_layout,
    )
    stored_counts = _load_packed_expert_counts(
        expert_state,
        count_offsets,
        E,
        world_size,
    )
    # The measured Pro buckets replace shift+and exponent extraction with
    # one PRMT.  Keep the specialization compile-time and out of M>=128 code.
    optimize_pro_small_m: gl.constexpr = l1_k == 7168 and (
        num_tokens == 8 or num_tokens == 16 or num_tokens == 32 or num_tokens == 64
    )
    use_prmt_exponent: gl.constexpr = optimize_pro_small_m
    # Only Pro M8/N8 benefits from carrying the scale across the WGMMA flight;
    # wider swap buckets retain their shorter register lifetime.
    precompute_pro_m8_n8_scale: gl.constexpr = (
        fast_math and l1_k == 7168 and num_tokens == 8
    )

    store_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4],
        [2, 16],
        [4, 1],
        [1, 0],
    )
    store_rows = gl.arange(
        0,
        _MXFP4_BLOCK_M,
        layout=gl.SliceLayout(1, store_layout),
    )
    store_cols = gl.arange(
        0,
        _MXFP4_BLOCK_N,
        layout=gl.SliceLayout(0, store_layout),
    )

    block_idx = gl.program_id(0)
    scheduler_expert = 0
    scheduler_phase = 1
    current_count = scheduler_count(
        stored_counts,
        count_offsets,
        scheduler_expert,
    )
    current_pool_block_offset = 0
    pipeline_tile = 0
    (
        task_phase,
        task_expert,
        task_m_block,
        task_n_block,
        task_pool_block,
        valid_count,
        block_idx,
        scheduler_expert,
        scheduler_phase,
        current_count,
        current_pool_block_offset,
    ) = scheduler_next(
        stored_counts,
        count_offsets,
        block_idx,
        scheduler_expert,
        scheduler_phase,
        current_count,
        current_pool_block_offset,
        E,
        l1_n_blocks,
        l2_n_blocks,
        num_experts_per_wave,
        num_sms,
        _MXFP4_BLOCK_M,
    )

    while task_phase != 0:
        local_row = task_m_block * _MXFP4_BLOCK_M
        valid_m = gl.minimum(
            _MXFP4_BLOCK_M,
            valid_count - local_row,
        )

        # The scheduler emits only linear1/linear2 inside this loop.
        # Keep the orientation decision constexpr so a swapped specialization
        # cannot retain an unreachable normal FC2 mainloop.
        if use_swap_ab:
            if task_phase == 1:
                if valid_m <= 8:
                    final_swap_8 = _mxfp4_fc1_swap_mainloop(
                        barriers,
                        buffers,
                        l1_mxfp4_secondary,
                        task_expert,
                        pipeline_tile,
                        l1_k,
                        8,
                        use_prmt_exponent,
                        precompute_pro_m8_n8_scale,
                        optimize_pro_small_m,
                    )
                    _fc1_epilogue(
                        final_swap_8,
                        l2_store_desc,
                        l2_epilogue_buffer,
                        l2_acts_sf,
                        route_weights,
                        l2_arrival,
                        task_pool_block,
                        task_n_block,
                        valid_m,
                        num_padded_sf_pool_tokens,
                        activation_clamp,
                        has_activation_clamp,
                        fast_math,
                        True,
                        8,
                        _MXFP4_BLOCK_M,
                        _MXFP4_BLOCK_N,
                        False,
                    )
                elif valid_m <= 16:
                    final_swap_16 = _mxfp4_fc1_swap_mainloop(
                        barriers,
                        buffers,
                        l1_mxfp4_secondary,
                        task_expert,
                        pipeline_tile,
                        l1_k,
                        16,
                        use_prmt_exponent,
                        False,
                        optimize_pro_small_m,
                    )
                    _fc1_epilogue(
                        final_swap_16,
                        l2_store_desc,
                        l2_epilogue_buffer,
                        l2_acts_sf,
                        route_weights,
                        l2_arrival,
                        task_pool_block,
                        task_n_block,
                        valid_m,
                        num_padded_sf_pool_tokens,
                        activation_clamp,
                        has_activation_clamp,
                        fast_math,
                        True,
                        16,
                        _MXFP4_BLOCK_M,
                        _MXFP4_BLOCK_N,
                        False,
                    )
                elif valid_m <= 32:
                    final_swap_32 = _mxfp4_fc1_swap_mainloop(
                        barriers,
                        buffers,
                        l1_mxfp4_secondary,
                        task_expert,
                        pipeline_tile,
                        l1_k,
                        32,
                        use_prmt_exponent,
                        False,
                        optimize_pro_small_m,
                    )
                    _fc1_epilogue(
                        final_swap_32,
                        l2_store_desc,
                        l2_epilogue_buffer,
                        l2_acts_sf,
                        route_weights,
                        l2_arrival,
                        task_pool_block,
                        task_n_block,
                        valid_m,
                        num_padded_sf_pool_tokens,
                        activation_clamp,
                        has_activation_clamp,
                        fast_math,
                        True,
                        32,
                        _MXFP4_BLOCK_M,
                        _MXFP4_BLOCK_N,
                        False,
                    )
                else:
                    final_swap_64 = _mxfp4_fc1_swap_mainloop(
                        barriers,
                        buffers,
                        l1_mxfp4_secondary,
                        task_expert,
                        pipeline_tile,
                        l1_k,
                        64,
                        use_prmt_exponent,
                        False,
                        optimize_pro_small_m,
                    )
                    _fc1_epilogue(
                        final_swap_64,
                        l2_store_desc,
                        l2_epilogue_buffer,
                        l2_acts_sf,
                        route_weights,
                        l2_arrival,
                        task_pool_block,
                        task_n_block,
                        valid_m,
                        num_padded_sf_pool_tokens,
                        activation_clamp,
                        has_activation_clamp,
                        fast_math,
                        True,
                        64,
                        _MXFP4_BLOCK_M,
                        _MXFP4_BLOCK_N,
                        False,
                    )
                pipeline_tile += l1_k // block_k
            else:
                # Gluon register tensors require power-of-two element counts, so
                # both linear phases use the same 8/16/32/64 token buckets.
                if valid_m <= 8:
                    final_swap_8 = _mxfp4_fc2_swap_mainloop(
                        barriers,
                        buffers,
                        l2_mxfp4_secondary,
                        task_expert,
                        pipeline_tile,
                        l2_k,
                        8,
                        use_prmt_exponent,
                        precompute_pro_m8_n8_scale,
                        optimize_pro_small_m,
                    )
                    _fc2_swap_bf16_epilogue(
                        final_swap_8,
                        token_src_metadata,
                        peer_combine_buffer_ptrs,
                        task_pool_block,
                        task_n_block,
                        valid_m,
                        l2_n,
                        max_tokens,
                        topk,
                        world_size,
                        8,
                        _MXFP4_BLOCK_M,
                        _MXFP4_BLOCK_N,
                    )
                elif valid_m <= 16:
                    final_swap_16 = _mxfp4_fc2_swap_mainloop(
                        barriers,
                        buffers,
                        l2_mxfp4_secondary,
                        task_expert,
                        pipeline_tile,
                        l2_k,
                        16,
                        use_prmt_exponent,
                        False,
                        optimize_pro_small_m,
                    )
                    _fc2_swap_bf16_epilogue(
                        final_swap_16,
                        token_src_metadata,
                        peer_combine_buffer_ptrs,
                        task_pool_block,
                        task_n_block,
                        valid_m,
                        l2_n,
                        max_tokens,
                        topk,
                        world_size,
                        16,
                        _MXFP4_BLOCK_M,
                        _MXFP4_BLOCK_N,
                    )
                elif valid_m <= 32:
                    final_swap_32 = _mxfp4_fc2_swap_mainloop(
                        barriers,
                        buffers,
                        l2_mxfp4_secondary,
                        task_expert,
                        pipeline_tile,
                        l2_k,
                        32,
                        use_prmt_exponent,
                        False,
                        optimize_pro_small_m,
                    )
                    _mxfp4_fc2_swap_vector_epilogue(
                        final_swap_32,
                        token_src_metadata,
                        peer_combine_buffer_ptrs,
                        task_pool_block,
                        task_n_block,
                        valid_m,
                        l2_n,
                        max_tokens,
                        topk,
                        world_size,
                        32,
                        _MXFP4_BLOCK_M,
                        _MXFP4_BLOCK_N,
                    )
                else:
                    final_swap_64 = _mxfp4_fc2_swap_mainloop(
                        barriers,
                        buffers,
                        l2_mxfp4_secondary,
                        task_expert,
                        pipeline_tile,
                        l2_k,
                        64,
                        use_prmt_exponent,
                        False,
                        optimize_pro_small_m,
                    )
                    _mxfp4_fc2_swap_vector_epilogue(
                        final_swap_64,
                        token_src_metadata,
                        peer_combine_buffer_ptrs,
                        task_pool_block,
                        task_n_block,
                        valid_m,
                        l2_n,
                        max_tokens,
                        topk,
                        world_size,
                        64,
                        _MXFP4_BLOCK_M,
                        _MXFP4_BLOCK_N,
                    )
                pipeline_tile += l2_k // block_k
        else:
            if task_phase == 1:
                final = _mxfp4_normal_mainloop(
                    barriers,
                    buffers,
                    l1_mxfp4_secondary,
                    l2_mxfp4_secondary,
                    task_expert,
                    pipeline_tile,
                    l1_k,
                    l2_k,
                    use_prmt_exponent,
                    True,
                )
                pipeline_tile += l1_k // block_k
            else:
                final = _mxfp4_normal_mainloop(
                    barriers,
                    buffers,
                    l1_mxfp4_secondary,
                    l2_mxfp4_secondary,
                    task_expert,
                    pipeline_tile,
                    l1_k,
                    l2_k,
                    use_prmt_exponent,
                    False,
                )
                pipeline_tile += l2_k // block_k

            if task_phase == 1:
                _fc1_epilogue(
                    final,
                    l2_store_desc,
                    l2_epilogue_buffer,
                    l2_acts_sf,
                    route_weights,
                    l2_arrival,
                    task_pool_block,
                    task_n_block,
                    valid_m,
                    num_padded_sf_pool_tokens,
                    activation_clamp,
                    has_activation_clamp,
                    fast_math,
                    False,
                    _MXFP4_BLOCK_M,
                    _MXFP4_BLOCK_M,
                    _MXFP4_BLOCK_N,
                    False,
                )
            else:
                fc2_tile = gl.convert_layout(final.to(gl.bfloat16), store_layout)
                _fc2_bf16_scatter_epilogue(
                    fc2_tile,
                    store_rows,
                    store_cols,
                    token_src_metadata,
                    peer_combine_buffer_ptrs,
                    task_pool_block,
                    task_n_block,
                    valid_m,
                    l2_n,
                    max_tokens,
                    topk,
                    world_size,
                    _MXFP4_BLOCK_M,
                    _MXFP4_BLOCK_N,
                )

        # The normal FC2 scatter is distributed across the complete four-warp
        # math partition.  Re-converge those warps before advancing the shared
        # scheduler/pipeline state; without this rendezvous a long expert wave
        # can let one warp reuse the next stage while peers still own the
        # previous scatter task.  SwapAB has its own bucketed epilogue and does
        # not require this extra synchronization.
        if task_phase == 2 and not use_swap_ab:
            partition_barrier()

        (
            task_phase,
            task_expert,
            task_m_block,
            task_n_block,
            task_pool_block,
            valid_count,
            block_idx,
            scheduler_expert,
            scheduler_phase,
            current_count,
            current_pool_block_offset,
        ) = scheduler_next(
            stored_counts,
            count_offsets,
            block_idx,
            scheduler_expert,
            scheduler_phase,
            current_count,
            current_pool_block_offset,
            E,
            l1_n_blocks,
            l2_n_blocks,
            num_experts_per_wave,
            num_sms,
            _MXFP4_BLOCK_M,
        )

    _mxfp4_fc2_combine_partition(
        output,
        combine_buffer,
        topk_idx,
        fused_barrier,
        peer_fused_barrier_ptrs,
        fc2_scatter_grid_counter,
        combine_cross_rank_ready,
        l2_n,
        num_tokens,
        topk,
        world_size,
        num_sms,
        _MXFP4_MATH_WARPS_CONSTEXPR,
    )


# mxfp4/kernel.py


@gluon.jit
def mxfp4_fused_kernel(
    pool_acts,
    l1_a_desc,
    l1_sfa_desc,
    l1_b_desc,
    l1_b_sf_desc,
    l2_store_desc,
    l2_a_desc,
    l2_sfa_desc,
    l2_b_desc,
    l2_b_sf_desc,
    dispatch_acts_desc_0,
    dispatch_acts_desc_1,
    dispatch_acts_desc_2,
    dispatch_acts_desc_3,
    dispatch_acts_desc_4,
    dispatch_acts_desc_5,
    dispatch_acts_desc_6,
    dispatch_acts_desc_7,
    dispatch_pool_desc,
    pool_acts_sf,
    l2_acts_sf,
    output,
    combine_buffer,
    peer_combine_buffer_ptrs,
    fused_barrier,
    peer_fused_barrier_ptrs,
    fc2_scatter_grid_counter,
    combine_cross_rank_ready,
    l1_weight_scales,
    l2_weight_scales,
    l1_mxfp4_secondary,
    l2_mxfp4_secondary,
    expert_state,
    expert_send_state,
    l1_arrival,
    l2_arrival,
    actual_num_pool_rows,
    dispatch_counter,
    input_topk_idx,
    pool_topk_weights,
    token_src_metadata,
    peer_input_sf_ptrs,
    peer_input_topk_weights_ptrs,
    symmetric_source_routes,
    symmetric_recv_count,
    peer_source_routes_ptrs,
    peer_recv_count_ptrs,
    peer_expert_state_ptrs,
    dispatch_barrier,
    peer_dispatch_barrier_ptrs,
    l1_n: gl.constexpr,
    l1_k: gl.constexpr,
    l2_n: gl.constexpr,
    l2_k: gl.constexpr,
    E: gl.constexpr,
    l1_ws_stride_e: gl.constexpr,
    l1_ws_stride_n: gl.constexpr,
    l1_ws_stride_k: gl.constexpr,
    l2_ws_stride_e: gl.constexpr,
    l2_ws_stride_n: gl.constexpr,
    l2_ws_stride_k: gl.constexpr,
    num_stages: gl.constexpr,
    l1_n_blocks: gl.constexpr,
    l2_n_blocks: gl.constexpr,
    num_experts_per_wave: gl.constexpr,
    num_sms: gl.constexpr,
    scheduler_count_capacity: gl.constexpr,
    scheduler_counts_per_lane: gl.constexpr,
    num_padded_sf_pool_tokens: gl.constexpr,
    num_tokens: gl.constexpr,
    max_tokens: gl.constexpr,
    topk: gl.constexpr,
    num_global_experts: gl.constexpr,
    num_routes: gl.constexpr,
    experts_per_rank: gl.constexpr,
    max_routes: gl.constexpr,
    rank: gl.constexpr,
    world_size: gl.constexpr,
    activation_clamp: gl.constexpr,
    has_activation_clamp: gl.constexpr,
    fast_math: gl.constexpr,
    use_swap_ab: gl.constexpr,
    num_dispatch_workers: gl.constexpr,
    a_tma_regs: gl.constexpr,
    b_tma_regs: gl.constexpr,
    dispatch_regs: gl.constexpr,
):
    """One worker CTA; the host may place one or two workers on each SM."""
    a_buffers = gl.allocate_shared_memory(
        l1_a_desc.dtype,
        [num_stages] + l1_a_desc.block_type.shape,
        l1_a_desc.layout,
    )
    b_buffers = gl.allocate_shared_memory(
        l1_b_desc.dtype,
        [num_stages] + l1_b_desc.block_type.shape,
        l1_b_desc.layout,
    )
    sfa_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [_MXFP4_BLOCK_M],
        gl.float32,
    )
    sfa_lo_buffers = gl.allocate_shared_memory(
        gl.float32,
        [num_stages, _MXFP4_BLOCK_M],
        sfa_layout,
    )
    sfa_hi_buffers = gl.allocate_shared_memory(
        gl.float32,
        [num_stages, _MXFP4_BLOCK_M],
        sfa_layout,
    )
    sfb_buffers = gl.allocate_shared_memory(
        l1_b_sf_desc.dtype,
        [num_stages] + l1_b_sf_desc.block_type.shape,
        l1_b_sf_desc.layout,
    )
    expanded_b_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [_MXFP4_BLOCK_N, _MXFP4_BLOCK_K],
        gl.float8e4nv,
    )
    expanded_b_buffers = gl.allocate_shared_memory(
        gl.float8e4nv,
        [2, _MXFP4_BLOCK_N, _MXFP4_BLOCK_K],
        expanded_b_layout,
    )
    # The second decoded-B slot is live only inside a GEMM mainloop, whereas
    # the C/D scratch starts after that mainloop has retired its last WGMMA.
    # Alias those disjoint lifetimes, matching the Pro shared-memory
    # contract and saving one 16-KiB tile.  This reduction is required for a
    # three-stage CTA to fit twice on one Hopper SM.
    l2_epilogue_buffer = expanded_b_buffers.index(1)._reinterpret(
        l2_store_desc.dtype,
        l2_store_desc.block_type.shape,
        l2_store_desc.layout,
    )
    barrier_layout: gl.constexpr = mbarrier.MBarrierLayout()
    stage_empty = gl.allocate_shared_memory(
        gl.int64,
        [num_stages, 1],
        barrier_layout,
    )
    stage_ready = gl.allocate_shared_memory(
        gl.int64,
        [num_stages, 1],
        barrier_layout,
    )
    for stage in gl.static_range(num_stages):
        mbarrier.init(stage_empty.index(stage), count=1)
        mbarrier.init(
            stage_ready.index(stage),
            count=_MXFP4_PRODUCERS_CONSTEXPR,
        )

    barriers = (stage_empty, stage_ready)
    buffers = (
        a_buffers,
        b_buffers,
        sfa_lo_buffers,
        sfa_hi_buffers,
        sfb_buffers,
        expanded_b_buffers,
    )
    task_state = (
        expert_state,
        l1_arrival,
        dispatch_counter,
    )
    dispatch_state = (
        pool_acts,
        pool_acts_sf,
        pool_topk_weights,
        token_src_metadata,
        num_padded_sf_pool_tokens,
        input_topk_idx,
        actual_num_pool_rows,
    )
    symmetric_state = (
        peer_input_sf_ptrs,
        peer_input_topk_weights_ptrs,
        symmetric_source_routes,
        symmetric_recv_count,
        peer_source_routes_ptrs,
        peer_recv_count_ptrs,
        peer_expert_state_ptrs,
        dispatch_barrier,
        peer_dispatch_barrier_ptrs,
    )
    dispatch_peer_descs = (
        dispatch_acts_desc_0,
        dispatch_acts_desc_1,
        dispatch_acts_desc_2,
        dispatch_acts_desc_3,
        dispatch_acts_desc_4,
        dispatch_acts_desc_5,
        dispatch_acts_desc_6,
        dispatch_acts_desc_7,
    )
    dispatch_descs = (dispatch_peer_descs, dispatch_pool_desc)
    dispatch_buffers_0 = gl.allocate_shared_memory(
        dispatch_acts_desc_0.dtype,
        dispatch_acts_desc_0.block_type.shape,
        dispatch_acts_desc_0.layout,
    )
    dispatch_barriers_0 = gl.allocate_shared_memory(
        gl.int64,
        [1],
        barrier_layout,
    )
    mbarrier.init(dispatch_barriers_0, count=1)
    if num_dispatch_workers == 2:
        dispatch_buffers_1 = gl.allocate_shared_memory(
            dispatch_acts_desc_0.dtype,
            dispatch_acts_desc_0.block_type.shape,
            dispatch_acts_desc_0.layout,
        )
        dispatch_barriers_1 = gl.allocate_shared_memory(
            gl.int64,
            [1],
            barrier_layout,
        )
        mbarrier.init(dispatch_barriers_1, count=1)

    if num_dispatch_workers == 2:
        gl.warp_specialize(
            [
                (
                    mxfp4_math_partition,
                    (
                        barriers,
                        buffers,
                        l2_store_desc,
                        l2_epilogue_buffer,
                        l2_acts_sf,
                        output,
                        combine_buffer,
                        input_topk_idx,
                        token_src_metadata,
                        peer_combine_buffer_ptrs,
                        fused_barrier,
                        peer_fused_barrier_ptrs,
                        fc2_scatter_grid_counter,
                        combine_cross_rank_ready,
                        pool_topk_weights,
                        l2_arrival,
                        l1_mxfp4_secondary,
                        l2_mxfp4_secondary,
                        expert_state,
                        l1_n,
                        l1_k,
                        l2_n,
                        l2_k,
                        E,
                        l1_n_blocks,
                        l2_n_blocks,
                        num_experts_per_wave,
                        num_sms,
                        scheduler_count_capacity,
                        scheduler_counts_per_lane,
                        num_padded_sf_pool_tokens,
                        num_tokens,
                        max_tokens,
                        topk,
                        world_size,
                        activation_clamp,
                        has_activation_clamp,
                        fast_math,
                        use_swap_ab,
                    ),
                ),
                (
                    a_producer_partition,
                    (
                        l1_a_desc,
                        l1_sfa_desc,
                        l2_a_desc,
                        l2_sfa_desc,
                        expert_state,
                        task_state[2],
                        l1_arrival,
                        l2_arrival,
                        barriers,
                        (a_buffers, sfa_lo_buffers, sfa_hi_buffers),
                        E,
                        l1_k,
                        l2_k,
                        l1_n_blocks,
                        l2_n_blocks,
                        num_experts_per_wave,
                        num_sms,
                        scheduler_count_capacity,
                        scheduler_counts_per_lane,
                        num_padded_sf_pool_tokens,
                        world_size,
                        False,
                        False,
                    ),
                ),
                (
                    mxfp4_b_producer_partition,
                    (
                        l1_b_desc,
                        l1_b_sf_desc,
                        l2_b_desc,
                        l2_b_sf_desc,
                        l1_weight_scales,
                        l2_weight_scales,
                        expert_state,
                        barriers,
                        (b_buffers, sfb_buffers),
                        E,
                        l1_n,
                        l1_k,
                        l2_n,
                        l2_k,
                        l1_n_blocks,
                        l2_n_blocks,
                        num_experts_per_wave,
                        num_sms,
                        scheduler_count_capacity,
                        scheduler_counts_per_lane,
                        world_size,
                    ),
                ),
                (
                    mxfp4_dispatch_partition,
                    (
                        task_state,
                        dispatch_state,
                        symmetric_state,
                        dispatch_descs,
                        dispatch_barriers_0,
                        dispatch_buffers_0,
                        expert_send_state,
                        l1_k,
                        topk,
                        _MXFP4_BLOCK_M,
                        num_sms,
                        E,
                        num_global_experts,
                        num_routes,
                        experts_per_rank,
                        max_routes,
                        rank,
                        world_size,
                        0,
                        2,
                    ),
                ),
                (
                    mxfp4_dispatch_partition,
                    (
                        task_state,
                        dispatch_state,
                        symmetric_state,
                        dispatch_descs,
                        dispatch_barriers_1,
                        dispatch_buffers_1,
                        expert_send_state,
                        l1_k,
                        topk,
                        _MXFP4_BLOCK_M,
                        num_sms,
                        E,
                        num_global_experts,
                        num_routes,
                        experts_per_rank,
                        max_routes,
                        rank,
                        world_size,
                        1,
                        2,
                    ),
                ),
            ],
            [1, 1, 1, 1],
            [a_tma_regs, b_tma_regs, dispatch_regs, dispatch_regs],
        )
    else:
        gl.warp_specialize(
            [
                (
                    mxfp4_math_partition,
                    (
                        barriers,
                        buffers,
                        l2_store_desc,
                        l2_epilogue_buffer,
                        l2_acts_sf,
                        output,
                        combine_buffer,
                        input_topk_idx,
                        token_src_metadata,
                        peer_combine_buffer_ptrs,
                        fused_barrier,
                        peer_fused_barrier_ptrs,
                        fc2_scatter_grid_counter,
                        combine_cross_rank_ready,
                        pool_topk_weights,
                        l2_arrival,
                        l1_mxfp4_secondary,
                        l2_mxfp4_secondary,
                        expert_state,
                        l1_n,
                        l1_k,
                        l2_n,
                        l2_k,
                        E,
                        l1_n_blocks,
                        l2_n_blocks,
                        num_experts_per_wave,
                        num_sms,
                        scheduler_count_capacity,
                        scheduler_counts_per_lane,
                        num_padded_sf_pool_tokens,
                        num_tokens,
                        max_tokens,
                        topk,
                        world_size,
                        activation_clamp,
                        has_activation_clamp,
                        fast_math,
                        use_swap_ab,
                    ),
                ),
                (
                    a_producer_partition,
                    (
                        l1_a_desc,
                        l1_sfa_desc,
                        l2_a_desc,
                        l2_sfa_desc,
                        expert_state,
                        task_state[2],
                        l1_arrival,
                        l2_arrival,
                        barriers,
                        (a_buffers, sfa_lo_buffers, sfa_hi_buffers),
                        E,
                        l1_k,
                        l2_k,
                        l1_n_blocks,
                        l2_n_blocks,
                        num_experts_per_wave,
                        num_sms,
                        scheduler_count_capacity,
                        scheduler_counts_per_lane,
                        num_padded_sf_pool_tokens,
                        world_size,
                        False,
                        False,
                    ),
                ),
                (
                    mxfp4_b_producer_partition,
                    (
                        l1_b_desc,
                        l1_b_sf_desc,
                        l2_b_desc,
                        l2_b_sf_desc,
                        l1_weight_scales,
                        l2_weight_scales,
                        expert_state,
                        barriers,
                        (b_buffers, sfb_buffers),
                        E,
                        l1_n,
                        l1_k,
                        l2_n,
                        l2_k,
                        l1_n_blocks,
                        l2_n_blocks,
                        num_experts_per_wave,
                        num_sms,
                        scheduler_count_capacity,
                        scheduler_counts_per_lane,
                        world_size,
                    ),
                ),
                (
                    mxfp4_dispatch_partition,
                    (
                        task_state,
                        dispatch_state,
                        symmetric_state,
                        dispatch_descs,
                        dispatch_barriers_0,
                        dispatch_buffers_0,
                        expert_send_state,
                        l1_k,
                        topk,
                        _MXFP4_BLOCK_M,
                        num_sms,
                        E,
                        num_global_experts,
                        num_routes,
                        experts_per_rank,
                        max_routes,
                        rank,
                        world_size,
                        0,
                        1,
                    ),
                ),
            ],
            [1, 1, 1],
            [a_tma_regs, b_tma_regs, dispatch_regs],
        )
    dispatch_buffers_0._keep_alive()
    dispatch_barriers_0._keep_alive()
    if num_dispatch_workers == 2:
        dispatch_buffers_1._keep_alive()
        dispatch_barriers_1._keep_alive()


# mxfp4/api.py


def fused_moe(
    ctx: SymmetricContext,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    local_l1_weights: MXFP4ProcessedWeights,
    local_l2_weights: MXFP4ProcessedWeights,
    *,
    x_sf: torch.Tensor | None = None,
    tokens_bound: int | None = None,
    config: MegaMoEConfig | None = None,
    num_sms: int | None = None,
    activation_clamp: float | None = None,
    fast_math: bool | None = None,
    workspace: MXFP4Workspace | None = None,
    routed_scaling_factor: float = 1.0,
) -> MXFP4Result:
    """Compute EP8 MXFP4 MoE using weights returned by prepare_weights.

    BF16 inputs are quantized during registration; FP8 inputs need explicit
    per-token FP32 x_sf. ``config`` owns launch geometry. Without a config,
    omitted activation options retain unclamped, non-fast-math behavior.
    Reuse result.workspace for subsequent calls with the same capacity.
    """
    _validate_processed_mxfp4_weights(local_l1_weights, local_l2_weights)
    w1, w1_scale, _l1_mxfp4_secondary = local_l1_weights
    w2, w2_scale, _l2_mxfp4_secondary = local_l2_weights
    group_k = 128
    output_dtype = torch.bfloat16
    require_sm90()
    if ctx.world_size <= 1:
        raise ValueError("fused symmetric dispatch requires world_size > 1")
    if x.ndim != 2 or x.dtype not in (
        torch.bfloat16,
        torch.float8_e4m3fn,
    ):
        raise ValueError("x must be a two-dimensional BF16 or FP8 E4M3 tensor")
    num_tokens, K = x.shape
    tokens_bound = resolve_tokens_bound(ctx, num_tokens, tokens_bound)
    if num_tokens > ctx.max_tokens or K != ctx.hidden:
        raise ValueError("input exceeds the symmetric context capacity")
    if x.dtype == torch.bfloat16:
        if x_sf is not None:
            raise ValueError("x_sf must be None when fused input is BF16")
    else:
        if x_sf is None or x_sf.shape != (num_tokens, K // group_k):
            raise ValueError("FP8 x_sf must be [num_tokens, K // group_k]")
        if x_sf.dtype != torch.float32:
            raise ValueError("x_sf must have dtype float32")
    if topk_idx.shape != (num_tokens, ctx.topk):
        raise ValueError("topk_idx does not match the symmetric context")
    if topk_weights.shape != topk_idx.shape:
        raise ValueError("topk_weights must match topk_idx")
    if topk_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_idx must have dtype int32 or int64")
    if topk_weights.dtype != torch.float32:
        raise ValueError("topk_weights must have dtype float32")
    if w1.ndim != 3:
        raise ValueError("w1 must be a three-dimensional tensor")
    E, N, stored_weight_k = w1.shape
    weight_k = stored_weight_k * 2
    if E != ctx.experts_per_rank:
        raise ValueError("w1 must contain exactly this rank's experts")
    if weight_k != K:
        raise ValueError("dispatch input and w1 K dimensions must match")
    if w1.dtype != torch.int8:
        raise ValueError("packed MXFP4 w1 must have dtype int8")
    if K % group_k or N % 256:
        raise ValueError("H must be divisible by 128 and FC1 2I by 256")
    intermediate_hidden = N // 2
    if w1_scale.shape != (E, N, K // 32):
        raise ValueError("MXFP4 L1 scale must be [E, 2I, H/32]")
    if w1_scale.dtype != torch.uint8:
        raise ValueError("MXFP4 L1 scale must have dtype uint8")
    if w2.shape != (E, K, intermediate_hidden // 2):
        raise ValueError("packed MXFP4 L2 weight must be [E, H, I/2]")
    if w2.dtype != torch.int8:
        raise ValueError("packed MXFP4 L2 weight must have dtype int8")
    if w2_scale.shape != (E, K, intermediate_hidden // 32):
        raise ValueError("MXFP4 L2 scale must be [E, H, I/32]")
    if w2_scale.dtype != torch.uint8:
        raise ValueError("MXFP4 L2 scale must have dtype uint8")
    if (
        _l1_mxfp4_secondary.dtype != torch.float32
        or _l2_mxfp4_secondary.dtype != torch.float32
        or _l1_mxfp4_secondary.shape != (E,)
        or _l2_mxfp4_secondary.shape != (E,)
    ):
        raise ValueError("MXFP4 secondary scales must be float32 [E]")
    tensors = (
        topk_idx,
        topk_weights,
        w1,
        w1_scale,
        w2,
        w2_scale,
    )
    tensors = (*tensors, _l1_mxfp4_secondary, _l2_mxfp4_secondary)
    if x_sf is not None:
        tensors = (x_sf, *tensors)
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all fused FC1 tensors must be on the same device")
    if x.device != ctx.device:
        raise ValueError("fused FC1 inputs must use the symmetric context device")
    if any(not tensor.is_contiguous() for tensor in (x, *tensors)):
        raise ValueError("all fused FC1 inputs and weights must be contiguous")
    if w1_scale.stride(-1) != 1:
        raise ValueError("w1_scale must be contiguous along K groups")
    if w2_scale.stride(-1) != 1:
        raise ValueError("w2_scale must be contiguous along K groups")
    device_sms = torch.cuda.get_device_properties(ctx.device).multi_processor_count
    selected = select(
        topology="ep8",
        fmt="mxfp4",
        shape=Shape(K, intermediate_hidden, ctx.num_experts, ctx.topk),
        tokens_bound=tokens_bound,
        num_sms=device_sms,
        override=config,
    )
    launch = selected.launch
    block_m = launch.block_m
    num_ctas_per_sm = selected.config.ctas_per_sm
    num_stages = launch.num_stages
    maxnreg = launch.launch_maxnreg
    dispatch_regs = selected.config.dispatch_registers
    num_experts_per_wave = launch.num_experts_per_wave
    use_swap_ab = launch.use_swap_ab
    if num_sms is None:
        num_sms = device_sms
    if type(num_sms) is not int or not 0 < num_sms <= device_sms:
        raise ValueError(f"num_sms must be in [1, {device_sms}], got {num_sms}")
    if num_ctas_per_sm == 2 and num_stages > 3:
        raise ValueError("2CTA MXFP4 requires num_stages <= 3")
    if activation_clamp is None:
        activation_clamp = config.activation_clamp if config is not None else math.inf
    if fast_math is None:
        fast_math = config.fast_math if config is not None else False
    if not isinstance(fast_math, bool):
        raise ValueError("fast_math must be a bool")
    activation_clamp = float(activation_clamp)
    if math.isnan(activation_clamp) or activation_clamp <= 0:
        raise ValueError("activation_clamp must be positive or infinity")

    routed_scaling_factor = float(routed_scaling_factor)
    if not math.isfinite(routed_scaling_factor):
        raise ValueError("routed_scaling_factor must be finite")
    check_policy_agreement(
        ctx,
        backend="mxfp4",
        tokens_bound=tokens_bound,
        shape=(K, intermediate_hidden, ctx.num_experts, ctx.topk),
        capacity=ctx.max_tokens,
        block_m=block_m,
        stages=num_stages,
        ctas_per_sm=num_ctas_per_sm,
        grid=num_sms * num_ctas_per_sm,
        maxnreg=maxnreg,
        dispatch_regs=dispatch_regs,
        experts_per_wave=num_experts_per_wave,
        use_swap_ab=use_swap_ab,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
        routed_scaling_factor=routed_scaling_factor,
    )
    registered_inputs = register_inputs(
        ctx,
        x,
        topk_idx,
        topk_weights,
        x_sf=x_sf,
        routed_scaling_factor=routed_scaling_factor,
    )

    max_global_routes = ctx.world_size * ctx.max_routes
    max_pool_blocks = triton.cdiv(
        max_global_routes + E * (block_m - 1),
        block_m,
    )
    num_pool_rows = max_pool_blocks * block_m
    sf_block_m = triton.cdiv(block_m, 128) * 128
    num_padded_sf_pool_tokens = max_pool_blocks * sf_block_m
    device = x.device
    if workspace is None:
        pool = ExpertPool(
            acts=torch.empty(
                (num_pool_rows, K),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            acts_sf_mn_major=torch.empty(
                (K // group_k, num_padded_sf_pool_tokens),
                dtype=torch.float32,
                device=device,
            ),
            topk_weights=torch.empty(
                num_pool_rows,
                dtype=torch.float32,
                device=device,
            ),
            token_src_metadata=torch.empty(
                (num_pool_rows, 3),
                dtype=torch.int64,
                device=device,
            ),
            expert_state=ctx.expert_state,
            source_routes=ctx.source_routes,
        )
        l2_acts = torch.empty(
            (num_pool_rows, intermediate_hidden),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        l2_acts_sf_mn_major = torch.empty(
            (intermediate_hidden // 64, num_padded_sf_pool_tokens),
            dtype=torch.float32,
            device=device,
        )
        output = torch.empty(
            (ctx.max_tokens, K),
            dtype=output_dtype,
            device=device,
        )
        # Arrival arrays are compact control state.  Recycle them in the
        # existing prologue reset rather than coupling dispatch and combine.
        l1_arrival = torch.zeros(
            max_pool_blocks,
            dtype=torch.int32,
            device=device,
        )
        l2_arrival = torch.zeros(
            max_pool_blocks,
            dtype=torch.int32,
            device=device,
        )
        fc2_scatter_grid_counter = torch.empty((), dtype=torch.int32, device=device)
        combine_cross_rank_ready = torch.empty((), dtype=torch.int32, device=device)
        actual_num_pool_rows = torch.empty((), dtype=torch.int32, device=device)
        dispatch_counter = torch.empty((), dtype=torch.int32, device=device)
        expert_send_state = torch.empty(
            ctx.num_experts,
            dtype=torch.int64,
            device=device,
        )

        a_layout = gl.NVMMASharedLayout.get_default_for(
            [_MXFP4_BLOCK_M_VALUE, _MXFP4_BLOCK_K_VALUE],
            gl.float8e4nv,
        )
        l1_a_desc = TensorDescriptor.from_tensor(
            pool.acts,
            [_MXFP4_BLOCK_M_VALUE, _MXFP4_BLOCK_K_VALUE],
            a_layout,
        )
        sfa_layout = gl.NVMMASharedLayout.get_default_for(
            [_MXFP4_BLOCK_M_VALUE],
            gl.float32,
        )
        l1_sfa_desc = TensorDescriptor.from_tensor(
            pool.acts_sf_mn_major.view(-1),
            [_MXFP4_BLOCK_M_VALUE],
            sfa_layout,
        )
        b_layout = gl.NVMMASharedLayout(
            swizzle_byte_width=64,
            transposed=False,
            element_bitwidth=8,
            rank=2,
        )
        packed_b_block = [
            _MXFP4_BLOCK_N_VALUE,
            _MXFP4_BLOCK_K_VALUE // 2,
        ]
        l1_b_desc = TensorDescriptor.from_tensor(
            w1.view(E * N, K // 2),
            packed_b_block,
            b_layout,
        )
        # Above hidden=8192 the producer uses ordinary strided global loads;
        # the descriptor still provides a legal 512-byte shared allocation.
        sfb_layout = gl.NVMMASharedLayout(
            swizzle_byte_width=0,
            transposed=False,
            element_bitwidth=8,
            rank=1,
        )
        l1_b_sf_desc = TensorDescriptor.from_tensor(
            w1_scale.view(-1),
            [_MXFP4_BLOCK_N_VALUE * (_MXFP4_BLOCK_K_VALUE // 32)],
            sfb_layout,
        )
        l2_store_layout = gl.NVMMASharedLayout.get_default_for(
            [_MXFP4_BLOCK_M_VALUE, _MXFP4_BLOCK_N_VALUE // 2],
            gl.float8e4nv,
        )
        l2_store_desc = TensorDescriptor.from_tensor(
            l2_acts,
            [_MXFP4_BLOCK_M_VALUE, _MXFP4_BLOCK_N_VALUE // 2],
            l2_store_layout,
        )
        l2_a_desc = TensorDescriptor.from_tensor(
            l2_acts,
            [_MXFP4_BLOCK_M_VALUE, _MXFP4_BLOCK_K_VALUE],
            a_layout,
        )
        l2_sfa_desc = TensorDescriptor.from_tensor(
            l2_acts_sf_mn_major.view(-1),
            [_MXFP4_BLOCK_M_VALUE],
            sfa_layout,
        )
        l2_b_desc = TensorDescriptor.from_tensor(
            w2.view(E * K, intermediate_hidden // 2),
            packed_b_block,
            b_layout,
        )
        l2_b_sf_desc = TensorDescriptor.from_tensor(
            w2_scale.view(-1),
            [_MXFP4_BLOCK_N_VALUE * (_MXFP4_BLOCK_K_VALUE // 32)],
            sfb_layout,
        )
        dispatch_descs = create_dispatch_descriptors(
            ctx,
            pool.acts,
        )
        workspace = MXFP4Workspace(
            pool=pool,
            l2_acts=l2_acts,
            l2_acts_sf_mn_major=l2_acts_sf_mn_major,
            output=output,
            l1_arrival=l1_arrival,
            l2_arrival=l2_arrival,
            fc2_scatter_grid_counter=fc2_scatter_grid_counter,
            combine_cross_rank_ready=combine_cross_rank_ready,
            actual_num_pool_rows=actual_num_pool_rows,
            dispatch_counter=dispatch_counter,
            expert_send_state=expert_send_state,
            l1_a_desc=l1_a_desc,
            l1_sfa_desc=l1_sfa_desc,
            l1_b_desc=l1_b_desc,
            l1_b_sf_desc=l1_b_sf_desc,
            l2_store_desc=l2_store_desc,
            l2_a_desc=l2_a_desc,
            l2_sfa_desc=l2_sfa_desc,
            l2_b_desc=l2_b_desc,
            l2_b_sf_desc=l2_b_sf_desc,
            dispatch_descs=dispatch_descs,
            l1_weight_data_ptr=w1.data_ptr(),
            l2_weight_data_ptr=w2.data_ptr(),
            max_pool_blocks=max_pool_blocks,
            num_pool_rows=num_pool_rows,
            num_padded_sf_pool_tokens=num_padded_sf_pool_tokens,
            l1_weight_sf_data_ptr=w1_scale.data_ptr(),
            l2_weight_sf_data_ptr=w2_scale.data_ptr(),
            l1_secondary_data_ptr=_l1_mxfp4_secondary.data_ptr(),
            l2_secondary_data_ptr=_l2_mxfp4_secondary.data_ptr(),
        )
    else:
        expected_workspace_shapes = (
            workspace.pool.acts.shape == (num_pool_rows, K),
            workspace.pool.acts_sf_mn_major.shape
            == (K // group_k, num_padded_sf_pool_tokens),
            workspace.l2_acts.shape == (num_pool_rows, intermediate_hidden),
            workspace.l2_acts_sf_mn_major.shape
            == (intermediate_hidden // 64, num_padded_sf_pool_tokens),
            workspace.output.shape == (ctx.max_tokens, K),
            workspace.l1_arrival.shape == (max_pool_blocks,),
            workspace.l2_arrival.shape == (max_pool_blocks,),
            workspace.fc2_scatter_grid_counter.shape == (),
            workspace.combine_cross_rank_ready.shape == (),
            workspace.expert_send_state.shape == (ctx.num_experts,),
            workspace.num_pool_rows == num_pool_rows,
            workspace.max_pool_blocks == max_pool_blocks,
            workspace.num_padded_sf_pool_tokens == num_padded_sf_pool_tokens,
        )
        if not all(expected_workspace_shapes):
            raise ValueError(
                "fused FC1/FC2 workspace does not match this problem shape"
            )
        if workspace.pool.acts.device != device:
            raise ValueError("fused FC1 workspace must use the input device")
        if workspace.pool.source_routes.data_ptr() != ctx.source_routes.data_ptr():
            raise ValueError("fused FC1 workspace belongs to a different context")
        if workspace.pool.expert_state.data_ptr() != ctx.expert_state.data_ptr():
            raise ValueError("fused FC1 workspace belongs to a different context")
        if workspace.l1_weight_data_ptr != w1.data_ptr():
            raise ValueError("fused workspace belongs to a different FC1 weight")
        if workspace.l2_weight_data_ptr != w2.data_ptr():
            raise ValueError("fused workspace belongs to a different FC2 weight")
        if (
            workspace.l1_weight_sf_data_ptr != w1_scale.data_ptr()
            or workspace.l2_weight_sf_data_ptr != w2_scale.data_ptr()
            or workspace.l1_secondary_data_ptr != _l1_mxfp4_secondary.data_ptr()
            or workspace.l2_secondary_data_ptr != _l2_mxfp4_secondary.data_ptr()
        ):
            raise ValueError("fused workspace belongs to different MXFP4 scales")

    pool = workspace.pool
    expert_state = pool.expert_state
    l2_acts = workspace.l2_acts
    l2_acts_sf_mn_major = workspace.l2_acts_sf_mn_major
    output = workspace.output
    l1_arrival = workspace.l1_arrival
    l2_arrival = workspace.l2_arrival
    fc2_scatter_grid_counter = workspace.fc2_scatter_grid_counter
    combine_cross_rank_ready = workspace.combine_cross_rank_ready
    actual_num_pool_rows = workspace.actual_num_pool_rows
    dispatch_counter = workspace.dispatch_counter
    expert_send_state = workspace.expert_send_state
    l1_a_desc = workspace.l1_a_desc
    l1_sfa_desc = workspace.l1_sfa_desc
    l1_b_desc = workspace.l1_b_desc
    l1_b_sf_desc = workspace.l1_b_sf_desc
    l2_store_desc = workspace.l2_store_desc
    l2_a_desc = workspace.l2_a_desc
    l2_sfa_desc = workspace.l2_sfa_desc
    l2_b_desc = workspace.l2_b_desc
    l2_b_sf_desc = workspace.l2_b_sf_desc
    dispatch_descs = workspace.dispatch_descs

    l1_n_blocks = _host_cdiv(N, _MXFP4_BLOCK_N_VALUE)
    l2_n_blocks = _host_cdiv(K, _MXFP4_BLOCK_N_VALUE)
    scheduler_count_capacity = _host_next_power_of_2(E)
    scheduler_counts_per_lane = max(
        _host_cdiv(scheduler_count_capacity, 32),
        1,
    )

    # Reset only compact control state.  Payload, SF padding, and metadata are
    # overwritten on valid rows and never capacity-cleared.  Keeping arrival
    # reset in this launch avoids the fragile dispatch/combine tail rendezvous
    # without restoring the old multi-MiB fixed cost.
    reset_layout = gl.BlockedLayout(
        [1],
        [32],
        [_FUSED_RESET_NUM_WARPS],
        [0],
    )
    reset_elements = max(
        max_pool_blocks,
        ctx.world_size,
        E,
        ctx.num_experts,
    )
    reset_control_kernel[(triton.cdiv(reset_elements, _FUSED_RESET_BLOCK_SIZE),)](
        l1_arrival,
        l2_arrival,
        actual_num_pool_rows,
        dispatch_counter,
        ctx.dispatch_barrier,
        ctx.fused_barrier,
        fc2_scatter_grid_counter,
        combine_cross_rank_ready,
        ctx.expert_state,
        expert_send_state,
        ctx.world_size,
        E,
        ctx.num_experts,
        max_pool_blocks,
        _FUSED_RESET_BLOCK_SIZE,
        reset_layout,
        num_warps=_FUSED_RESET_NUM_WARPS,
    )
    ctx.barrier()
    l1_mxfp4_secondary = _l1_mxfp4_secondary
    l2_mxfp4_secondary = _l2_mxfp4_secondary
    # Keep the single-transport kernel under a 3D-OOB-specific JIT symbol so
    # cached artifacts and assembly reports cannot be confused with the
    # former multi-backend kernel.
    # ``num_sms`` remains the physical-SM choice in the public API.  Inside
    # the persistent kernel, every independent resident CTA is a scheduler
    # worker and therefore participates in the dispatch/scatter grid counts.
    num_worker_ctas = num_sms * num_ctas_per_sm
    # The compact tiny-token math path and Flash normal path leave enough
    # shared memory for two dispatch buffers while preserving two CTAs/SM.
    num_dispatch_workers = (
        2
        if (
            num_ctas_per_sm == 1
            or (use_swap_ab and tokens_bound <= 32)
            or (not use_swap_ab and K == 4096)
        )
        else 1
    )
    compiled = mxfp4_fused_kernel[(num_worker_ctas,)](
        pool.acts,
        l1_a_desc,
        l1_sfa_desc,
        l1_b_desc,
        l1_b_sf_desc,
        l2_store_desc,
        l2_a_desc,
        l2_sfa_desc,
        l2_b_desc,
        l2_b_sf_desc,
        *dispatch_descs,
        pool.acts_sf_mn_major,
        l2_acts_sf_mn_major,
        output,
        ctx.combine_buffer,
        ctx.peer_combine_buffer_ptrs,
        ctx.fused_barrier,
        ctx.peer_fused_barrier_ptrs,
        fc2_scatter_grid_counter,
        combine_cross_rank_ready,
        w1_scale,
        w2_scale,
        l1_mxfp4_secondary,
        l2_mxfp4_secondary,
        expert_state,
        expert_send_state,
        l1_arrival,
        l2_arrival,
        actual_num_pool_rows,
        dispatch_counter,
        registered_inputs.input_topk_idx,
        pool.topk_weights,
        pool.token_src_metadata,
        ctx.peer_input_sf_ptrs,
        ctx.peer_input_topk_weights_ptrs,
        ctx.source_routes,
        ctx.recv_count,
        ctx.peer_source_routes_ptrs,
        ctx.peer_recv_count_ptrs,
        ctx.peer_expert_state_ptrs,
        ctx.dispatch_barrier,
        ctx.peer_dispatch_barrier_ptrs,
        N,
        K,
        K,
        intermediate_hidden,
        E,
        w1_scale.stride(0),
        w1_scale.stride(1),
        w1_scale.stride(2),
        w2_scale.stride(0),
        w2_scale.stride(1),
        w2_scale.stride(2),
        num_stages,
        l1_n_blocks,
        l2_n_blocks,
        num_experts_per_wave,
        num_worker_ctas,
        scheduler_count_capacity,
        scheduler_counts_per_lane,
        num_padded_sf_pool_tokens,
        num_tokens,
        ctx.max_tokens,
        ctx.topk,
        ctx.num_experts,
        num_tokens * ctx.topk,
        ctx.experts_per_rank,
        ctx.max_routes,
        ctx.rank,
        ctx.world_size,
        activation_clamp,
        math.isfinite(activation_clamp),
        fast_math,
        use_swap_ab,
        num_dispatch_workers,
        _MXFP4_TMA_REGS_VALUE,
        _MXFP4_TMA_REGS_VALUE,
        dispatch_regs,
        num_warps=_MXFP4_MATH_WARPS,
        maxnreg=maxnreg,
    )
    return MXFP4Result(
        pool=pool,
        l2_acts=l2_acts,
        l2_acts_sf_mn_major=l2_acts_sf_mn_major,
        l2_arrival=l2_arrival,
        output=output[:num_tokens],
        combine_buffer=ctx.combine_buffer[:num_tokens],
        l1_arrival=l1_arrival,
        fc2_scatter_grid_counter=fc2_scatter_grid_counter,
        combine_cross_rank_ready=combine_cross_rank_ready,
        actual_num_pool_rows=actual_num_pool_rows,
        dispatch_counter=dispatch_counter,
        workspace=workspace,
        pre_dispatch=registered_inputs,
        compiled=compiled,
    )


__all__ = [
    "create_context",
    "prepare_weights",
    "fused_moe",
    "MegaMoEConfig",
    "MathBody",
    "Shape",
    "select",
    "MXFP4CheckpointWeights",
    "MXFP4ProcessedWeights",
    "MXFP4Result",
    "MXFP4Workspace",
]
