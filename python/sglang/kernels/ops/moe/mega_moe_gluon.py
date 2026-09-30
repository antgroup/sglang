# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Copyright (c) 2025 DeepSeek
# Derived from Triton-distributed MegaMoE Gluon; Hopper scheduling and
# numerical conventions derive from DeepGEMM. MIT licensed.
"""SM90 EP8 FP8 MegaMoE and shared context/configuration primitives.

Routing IDs must be negative sentinels or lie in [0, num_experts).
Nonnegative IDs outside that range are outside the operator contract.

Public API: create_context, prepare_weights, fused_moe.
No serving-stack integration or process-group initialization at import.
"""

import math
import os
from bisect import bisect_right
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from types import MappingProxyType

import torch
import triton
from packaging.version import InvalidVersion, Version
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

# _compat.py

try:
    from triton.experimental.gluon.language import barrier as partition_barrier
except ImportError:
    # Gluon 3.6 exported the same CTA synchronization builtin under this name.
    from triton.experimental.gluon.language import thread_barrier as partition_barrier


_MINIMUM_TRITON_VERSION = Version("3.6.0")


def require_sm90() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the Gluon MegaMoE capability gate requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (9, 0):
        raise RuntimeError(f"the capability gate requires SM90, got sm{major}{minor}")

    try:
        installed_version = Version(triton.__version__.split("+", 1)[0])
    except InvalidVersion as exc:
        raise RuntimeError(
            f"cannot parse the installed Triton version {triton.__version__!r}"
        ) from exc
    if installed_version < _MINIMUM_TRITON_VERSION:
        raise RuntimeError(
            "the Gluon MegaMoE path requires Triton >= "
            f"{_MINIMUM_TRITON_VERSION}, got {triton.__version__}"
        )


# _constants.py

_BLOCK_M_VALUE = 64


_SPLIT_BLOCK_M_VALUE = 128


_MAX_CANDIDATE_BLOCK_M_VALUE = 192


_POOL_ALIGNMENT_VALUE = 384


_BLOCK_N_VALUE = 128


_NORMAL_BLOCK_N_VALUE = 256


_BLOCK_K_VALUE = 128


_SF_BLOCK_M_VALUE = 128


_GROUP_M_VALUE = 8


_NUM_STAGES = 7


_PRODUCERS = 2


_BLOCK_M = gl.constexpr(_BLOCK_M_VALUE)


_SPLIT_BLOCK_M = gl.constexpr(_SPLIT_BLOCK_M_VALUE)


_BLOCK_N = gl.constexpr(_BLOCK_N_VALUE)


_NORMAL_BLOCK_N = gl.constexpr(_NORMAL_BLOCK_N_VALUE)


_LARGE_NORMAL_BLOCK_N = gl.constexpr(_NORMAL_BLOCK_N_VALUE)


_BLOCK_K = gl.constexpr(_BLOCK_K_VALUE)


_SF_BLOCK_M = gl.constexpr(_SF_BLOCK_M_VALUE)


_GROUP_M = gl.constexpr(_GROUP_M_VALUE)


_PRODUCERS_CONSTEXPR = gl.constexpr(_PRODUCERS)


_LARGE_NORMAL_MATH_WARPS = gl.constexpr(8)


_LARGE_NORMAL_TMA_REGS_VALUE = 24


# config.py


class MathBody(str, Enum):
    SWAP_BN128 = "swap"
    SWAP_BN256 = "swap256"
    SPLIT_BN128 = "split128"
    SPLIT_BN256 = "split256"
    SPLIT_BM128_BN256 = "split_bm128"


class DispatchHandoff(str, Enum):
    D4_WAIT = "d4_wait"
    D1 = "d1"


class Rendezvous(str, Enum):
    D4 = "d4"
    NONE = "none"


class CombineMode(str, Enum):
    DIRECT = "direct"
    CHUNKED = "chunked"


@dataclass(frozen=True)
class MegaMoEConfig:
    """Discrete policy choices, independent of Torch and kernel launch state.

    ``stages=None`` uses the shared-memory calculation. Expert windows are
    derived at the actual common token bound unless a policy fixes one.
    EP16 dispatch and registration/reset fusion are explicit choices.
    Chunked combine shares one wire format across all ranks at a common bound.
    """

    body: MathBody
    stages: int | None = None
    math_warps: int = 8
    ctas_per_sm: int = 1
    experts_per_wave: int | None = None
    block_m: int = 64
    math_registers: int = 168
    producer_registers: int = 40
    dispatch_registers: int = 48
    fc1_promotion_k: int = 64
    fc2_promotion_k: int = 64
    fast_math: bool = True
    activation_clamp: float = 10.0
    dispatch_handoff: DispatchHandoff = DispatchHandoff.D4_WAIT
    rendezvous: Rendezvous = Rendezvous.D4
    fuse_reset: bool = False
    combine: CombineMode = CombineMode.DIRECT

    @property
    def block_n(self) -> int:
        return (
            256
            if self.body
            in (MathBody.SWAP_BN256, MathBody.SPLIT_BN256, MathBody.SPLIT_BM128_BN256)
            else 128
        )

    @property
    def block_k(self) -> int:
        return 128


_PRE_DISPATCH_GROUP_SIZE = 128


_PRE_DISPATCH_GROUPS_PER_CTA = 64


_PRE_DISPATCH_NUM_WARPS = 32


_PRE_DISPATCH_THREADS = _PRE_DISPATCH_NUM_WARPS * 32


_FUSED_RESET_NUM_WARPS = 8


_FUSED_RESET_BLOCK_SIZE = _FUSED_RESET_NUM_WARPS * 32


_FLASH_BM64_BN256_MAX_TOKENS_PER_RANK = 8192


_SMEM_CAPACITY = 232448


_SMEM_ALIGNMENT = 1024


_SPLIT_MN_MAX_STAGES = 2


_WARP_SPECIALIZATION_GROUP_WARPS = 4


_WARP_SPECIALIZATION_PADDING_REGS = 16


@dataclass(frozen=True)
class LaunchConfig:
    """Resolved launch parameters for a common input bound."""

    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_math_warps: int
    num_experts_per_wave: int
    use_swap_ab: bool
    use_split_bn256: bool
    expected_tokens_per_expert: float
    reuse_accum_as_final: bool
    fc2_arrival_counter: bool
    fc2_epilogue_requires_full_sync: bool
    scalarize_hot_rescale: bool
    math_register_budget: int
    dispatch_register_budget: int
    non_epilogue_register_budget: int
    launch_maxnreg: int
    combine_launch_maxnreg: int
    fc1_promotion_k: int = 64
    fc2_promotion_k: int = 64

    @property
    def mode(self) -> str:
        if self.use_split_bn256:
            return "split_bm64_bn256"
        orientation = "swap" if self.use_swap_ab else "split"
        return f"{orientation}_bm{self.block_m}_bn{self.block_n}"


def _host_align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def derive_launch_maxnreg(
    *,
    num_math_warps: int,
    math_register_budget: int,
    specialized_register_budgets: tuple[int, ...],
) -> int:
    """Match Triton's full-warpgroup padding in the launch register cap."""
    num_specialized_warps = len(specialized_register_budgets)
    num_padded_specialized_warps = _host_align(
        num_specialized_warps,
        _WARP_SPECIALIZATION_GROUP_WARPS,
    )
    num_padding_warps = num_padded_specialized_warps - num_specialized_warps
    total_warps = num_math_warps + num_padded_specialized_warps
    total_register_budget = (
        num_math_warps * math_register_budget
        + sum(specialized_register_budgets)
        + num_padding_warps * _WARP_SPECIALIZATION_PADDING_REGS
    )
    return _host_align(
        (total_register_budget + total_warps - 1) // total_warps,
        8,
    )


def _host_cdiv(a, b):
    return (a + b - 1) // b


def _host_next_power_of_2(value):
    return 1 << max(value - 1, 0).bit_length()


# policy.py


def select_stages(
    *,
    hidden: int,
    num_experts: int,
    block_m: int,
    block_n: int,
    num_math_warps: int,
    use_swap_ab: bool,
) -> int:
    """Mirror DeepGEMM's SM90 shared-memory pipeline calculation."""
    block_k = 128
    num_dispatch_warps = 2
    num_epilogue_warps = num_math_warps
    smem_expert_count = _host_align(
        num_experts * 4,
        _SMEM_ALIGNMENT,
    )
    smem_send_buffers = _host_align(
        hidden * num_dispatch_warps,
        _SMEM_ALIGNMENT,
    )
    smem_cd_l1 = block_m * (block_n // 2)
    smem_cd_l2 = block_m * block_n * 2
    # Unlike DeepGEMM's CUDA swap path, Gluon keeps the FP32 accumulator in
    # registers and only stages the quantized FC1 tile in shared memory.
    smem_cd_swap_l1 = 0
    smem_cd = _host_align(
        max(smem_cd_l1, smem_cd_l2, smem_cd_swap_l1),
        _SMEM_ALIGNMENT,
    )
    smem_sfa_per_stage = _host_align(2 * block_m * 4, 128)
    smem_per_stage = block_m * block_k + block_n * block_k + smem_sfa_per_stage
    smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8
    smem_fixed = smem_expert_count + smem_send_buffers + smem_cd + smem_barriers_fixed
    num_stages = (_SMEM_CAPACITY - smem_fixed) // (smem_per_stage + 16)
    if num_stages < 2:
        raise ValueError("selected SM90 config has fewer than two stages")
    return num_stages


def derive_experts_per_wave(
    *,
    expected_tokens_per_expert: float,
    num_experts_per_rank: int,
    intermediate_hidden: int,
    block_m: int,
    block_n: int,
    num_sms: int,
) -> int:
    """Mirror DeepGEMM's SM90 expert-wave search."""
    if expected_tokens_per_expert < 1.0 or expected_tokens_per_expert > 4.0:
        return num_experts_per_rank
    if block_m == 64 and intermediate_hidden >= 3072:
        single_wave_blocks = num_experts_per_rank * (
            (2 * intermediate_hidden) // block_n
        )
        if single_wave_blocks >= 4 * num_sms:
            return num_experts_per_rank

    expected_m_blocks = max(
        (math.ceil(expected_tokens_per_expert) + block_m - 1) // block_m,
        1,
    )
    l1_n_blocks = (2 * intermediate_hidden) // block_n
    expected_blocks_per_expert = expected_m_blocks * l1_n_blocks
    min_experts_per_wave = (
        2 * num_sms + expected_blocks_per_expert - 1
    ) // expected_blocks_per_expert
    max_experts_per_wave = num_experts_per_rank
    if min_experts_per_wave >= max_experts_per_wave:
        return max_experts_per_wave
    if expected_blocks_per_expert >= num_sms:
        return min_experts_per_wave

    best_experts_per_wave = min_experts_per_wave
    best_tail_ratio = -1.0
    sweep_end = min(max_experts_per_wave, min_experts_per_wave * 2)
    for experts_per_wave in range(min_experts_per_wave, sweep_end + 1):
        remainder = num_experts_per_rank % experts_per_wave
        tail_ratio = 1.0 if remainder == 0 else remainder / experts_per_wave
        if tail_ratio > best_tail_ratio:
            best_tail_ratio = tail_ratio
            best_experts_per_wave = experts_per_wave
    return best_experts_per_wave


def _select_generic_config(
    *,
    num_tokens_per_rank,
    hidden,
    intermediate_hidden,
    num_experts,
    num_experts_per_rank,
    topk,
    num_sms,
    num_stages=None,
    num_experts_per_wave=None,
    policy_tokens=None,
):
    """Derive generic geometry and use table overrides for known shapes."""
    policy_tokens = num_tokens_per_rank if policy_tokens is None else policy_tokens
    if policy_tokens < 0:
        raise ValueError("policy_tokens must be nonnegative")
    expected = float(num_tokens_per_rank) * topk / num_experts_per_rank
    preset = next(
        (
            key
            for key, shape in SHAPES.items()
            if (shape.hidden, shape.intermediate_hidden, shape.num_experts)
            == (hidden, intermediate_hidden, num_experts)
        ),
        None,
    )
    profile = None
    if preset is not None and policy_tokens > 0:
        rows = TABLE[("ep8", "fp8", preset, 78)]
        row = rows[bisect_right(tuple(r.m_lo for r in rows), policy_tokens) - 1]
        profile = CONFIGS[row.config]
    split_mn = profile is not None and profile.body is MathBody.SPLIT_BM128_BN256
    swap = profile.body is MathBody.SWAP_BN128 if profile else 0 < policy_tokens <= 255
    large_requested = (
        profile.body is MathBody.SPLIT_BN256 if profile else policy_tokens > 255
    )
    flash = preset == ALIASES["flash"]
    normal256 = (
        not split_mn
        and not swap
        and (
            (flash and large_requested)
            or (not flash and intermediate_hidden >= 3072 and expected >= 0.25)
        )
        and (2 * intermediate_hidden) % 256 == 0
        and hidden % 256 == 0
    )
    large = large_requested and normal256
    wide = swap and expected >= 256.0
    bm, bn = (128 if split_mn else 64), (256 if split_mn or normal256 or wide else 128)
    warps = 16 if split_mn or wide else 8
    stages = select_stages(
        hidden=hidden,
        num_experts=num_experts,
        block_m=bm,
        block_n=bn,
        num_math_warps=warps,
        use_swap_ab=swap,
    )
    if profile is not None and profile.stages is not None:
        stages = profile.stages
    if split_mn:
        stages = min(stages, _SPLIT_MN_MAX_STAGES)
    if wide:
        stages = 3
    if num_stages is not None:
        if not 2 <= num_stages <= 8:
            raise ValueError("num_stages must be in [2, 8]")
        if split_mn and num_stages > _SPLIT_MN_MAX_STAGES:
            raise ValueError("Gluon BM128/BN256 requires num_stages=2")
        stages = num_stages
    wave = derive_experts_per_wave(
        expected_tokens_per_expert=expected,
        num_experts_per_rank=num_experts_per_rank,
        intermediate_hidden=intermediate_hidden,
        block_m=bm,
        block_n=bn,
        num_sms=num_sms,
    )
    if profile is not None and profile.experts_per_wave is not None:
        wave = profile.experts_per_wave
    if num_experts_per_wave is not None:
        if not 0 < num_experts_per_wave <= num_experts_per_rank:
            raise ValueError("num_experts_per_wave must be in [1, E]")
        wave = num_experts_per_wave
    math_regs, dispatch_regs, producer_regs = (
        (112, 32, 24) if warps == 16 else (168, 48, 40)
    )
    combine_maxnreg = derive_launch_maxnreg(
        num_math_warps=warps,
        math_register_budget=math_regs,
        specialized_register_budgets=(producer_regs, producer_regs),
    )
    maxnreg = derive_launch_maxnreg(
        num_math_warps=warps,
        math_register_budget=math_regs,
        specialized_register_budgets=(
            producer_regs,
            producer_regs,
            dispatch_regs,
            dispatch_regs,
        ),
    )
    counter = split_mn or (bn == 256 and 4 <= policy_tokens <= 128)
    return LaunchConfig(
        block_m=bm,
        block_n=bn,
        block_k=128,
        num_stages=stages,
        num_math_warps=warps,
        num_experts_per_wave=wave,
        use_swap_ab=swap,
        use_split_bn256=large,
        expected_tokens_per_expert=expected,
        reuse_accum_as_final=not swap and not large,
        fc2_arrival_counter=counter,
        fc2_epilogue_requires_full_sync=not counter and not large,
        scalarize_hot_rescale=not swap and not large,
        math_register_budget=math_regs,
        dispatch_register_budget=dispatch_regs,
        non_epilogue_register_budget=producer_regs,
        launch_maxnreg=maxnreg,
        combine_launch_maxnreg=combine_maxnreg,
        fc1_promotion_k=64 if swap else 128,
        fc2_promotion_k=64,
    )


def validate_policy_agreement(contracts):
    """Reject the same differing fields on every rank of a debug collective."""
    if not contracts:
        raise ValueError("policy agreement requires at least one rank")
    keys = set().union(*(contract.keys() for contract in contracts))
    differing = sorted(
        key
        for key in keys
        if any(
            key not in contract
            or key not in contracts[0]
            or contract[key] != contracts[0][key]
            for contract in contracts
        )
    )
    if differing:
        raise ValueError("rank-common policy differs: " + ", ".join(differing))


def validate_tokens_bound(local_tokens, tokens_bound, capacity):
    bound = local_tokens if tokens_bound is None else tokens_bound
    if (
        type(local_tokens) is not int
        or type(bound) is not int
        or not 0 <= local_tokens <= bound <= capacity
    ):
        raise ValueError(
            "require 0 <= local tokens <= common tokens_bound <= context capacity"
        )
    return bound


@dataclass(frozen=True)
class Shape:
    hidden: int
    intermediate_hidden: int
    num_experts: int
    topk: int

    @property
    def key(self):
        return f"h{self.hidden}_i{self.intermediate_hidden}_e{self.num_experts}_k{self.topk}"

    def validate(self, world_size):
        values = (self.hidden, self.intermediate_hidden, self.num_experts, self.topk)
        if any(type(v) is not int or v <= 0 for v in values):
            raise ValueError("shape dimensions must be positive integers")
        if self.num_experts % world_size or self.topk > self.num_experts:
            raise ValueError(
                "experts must divide world size and topk must not exceed experts"
            )
        if self.hidden % 128 or self.intermediate_hidden % 128:
            raise ValueError(
                "hidden and intermediate dimensions must be multiples of 128"
            )


@dataclass(frozen=True)
class Row:
    m_lo: int
    m_hi: int | None
    config: str


@dataclass(frozen=True)
class Selection:
    config: MegaMoEConfig
    launch: LaunchConfig
    row: Row | None
    grid: int
    runtime_tokens_limit: int | None
    adjustments: tuple[str, ...] = ()


def _shape(shape):
    if isinstance(shape, Shape):
        return shape
    if isinstance(shape, str):
        try:
            return SHAPES[ALIASES.get(shape, shape)]
        except KeyError:
            raise ValueError(f"unknown shape alias: {shape}") from None
    raise TypeError("shape must be a Shape or a registered alias")


def validate_override(config, *, fmt, topology, experts_per_rank):
    if not isinstance(config, MegaMoEConfig) or not isinstance(config.body, MathBody):
        raise ValueError("override must be a MegaMoEConfig with a MathBody")
    if config.stages is not None and (
        type(config.stages) is not int or not 2 <= config.stages <= 8
    ):
        raise ValueError("stages must be in [2, 8]")
    if config.math_warps not in (4, 8, 16) or type(config.math_warps) is not int:
        raise ValueError("math_warps must be 4, 8 or 16")
    if config.ctas_per_sm not in (1, 2, 3) or type(config.ctas_per_sm) is not int:
        raise ValueError("current runtime supports one, two or three CTAs per SM")
    if config.experts_per_wave is not None and (
        type(config.experts_per_wave) is not int
        or not 0 < config.experts_per_wave <= experts_per_rank
    ):
        raise ValueError("experts_per_wave must be in [1, E]")
    bm16 = (
        fmt == "fp8"
        and topology == "ep16"
        and config.block_m == 16
        and config.body is MathBody.SWAP_BN128
        and config.math_warps == 4
        and config.stages in (3, 5)
    )
    if config.ctas_per_sm == 3 and not (bm16 and config.stages == 3):
        raise ValueError("three CTAs require BM16 with three stages")
    if not bm16 and config.block_m != (
        128 if config.body is MathBody.SPLIT_BM128_BN256 else 64
    ):
        raise ValueError("selected body does not yet support this block_m")
    if config.body is MathBody.SPLIT_BM128_BN256 and config.stages not in (None, 2):
        raise ValueError("Gluon BM128/BN256 requires num_stages=2")
    for value in (
        config.math_registers,
        config.producer_registers,
        config.dispatch_registers,
    ):
        if type(value) is not int or not 8 <= value <= 248 or value % 8:
            raise ValueError("register budgets must be multiples of eight in [8, 248]")
    if fmt == "fp8":
        required_warps = {
            MathBody.SPLIT_BN128: 8,
            MathBody.SPLIT_BN256: 8,
            MathBody.SPLIT_BM128_BN256: 16,
            MathBody.SWAP_BN256: 16,
        }
        if (
            config.body in required_warps
            and config.math_warps != required_warps[config.body]
        ):
            raise ValueError("math warp count does not match the selected FP8 body")
        if config.body is MathBody.SPLIT_BN256 and config.producer_registers != 40:
            raise ValueError("split256 currently retains its fixed producer budget")
    if (
        topology == "ep16"
        and config.ctas_per_sm == 2
        and not (
            config.body is MathBody.SWAP_BN128
            and config.math_warps == 4
            and (config.stages in (2, 3) or bm16)
        )
    ):
        raise ValueError(
            "two-CTA tuning requires four-warp swap with at most three stages"
        )
    if fmt == "fp8" and topology == "ep8" and config.ctas_per_sm != 1:
        raise ValueError("FP8 EP8 supports one CTA per SM")
    if fmt == "mxfp4" and (
        config.body not in (MathBody.SWAP_BN128, MathBody.SPLIT_BN128)
        or config.math_warps != 4
        or config.stages not in (2, 3)
        and config.ctas_per_sm == 2
    ):
        raise ValueError("unsupported MXFP4 body, warp count or two-CTA stage count")
    if fmt == "mxfp4" and (
        config.math_registers != 128
        or config.producer_registers != 24
        or (config.fc1_promotion_k, config.fc2_promotion_k) != (128, 64)
    ):
        raise ValueError(
            "MXFP4 requires its fixed 128/24 register and 128/64 promotion contract"
        )
    if not isinstance(config.dispatch_handoff, DispatchHandoff) or not isinstance(
        config.rendezvous, Rendezvous
    ):
        raise ValueError("dispatch protocol fields must use their enum types")
    if not isinstance(config.combine, CombineMode):
        raise ValueError("combine must use CombineMode")
    if topology != "ep16" and (
        config.combine is not CombineMode.DIRECT
        or config.dispatch_handoff is not DispatchHandoff.D4_WAIT
        or config.rendezvous is not Rendezvous.D4
        or config.fuse_reset
    ):
        raise ValueError("candidate protocol is not yet enabled in the unified runtime")
    if config.combine is CombineMode.CHUNKED and (
        config.dispatch_handoff is not DispatchHandoff.D1
        or config.dispatch_registers < 80
        or config.ctas_per_sm == 3
    ):
        raise ValueError(
            "chunked return requires tail quiet, at least 80 dispatch registers and at most two CTAs"
        )
    if topology == "ep16" and config.combine is CombineMode.DIRECT:
        dispatch_registers = 40 if config.math_warps == 16 else 48
        if config.dispatch_registers != dispatch_registers:
            raise ValueError(
                f"DIRECT return requires {dispatch_registers} dispatch registers for this math warp count"
            )
    if type(config.fuse_reset) is not bool:
        raise ValueError("fuse_reset must be boolean")
    if config.fuse_reset and not (
        config.body is MathBody.SWAP_BN128
        and config.math_warps == 4
        and config.dispatch_handoff is DispatchHandoff.D1
    ):
        raise ValueError("fused registration requires four-warp swap with tail quiet")
    if type(config.fc1_promotion_k) is not int or config.fc1_promotion_k not in (
        32,
        64,
        128,
    ):
        raise ValueError("fc1_promotion_k must be 32, 64 or 128")
    if type(config.fc2_promotion_k) is not int or config.fc2_promotion_k not in (
        32,
        64,
    ):
        raise ValueError("fc2_promotion_k must be 32 or 64")
    if topology == "ep16":
        expected_promotion = (
            (32, 32)
            if config.body in (MathBody.SWAP_BN128, MathBody.SWAP_BN256)
            else (128, 64)
        )
        if (config.fc1_promotion_k, config.fc2_promotion_k) != expected_promotion:
            raise ValueError(
                "EP16 currently supports only the body default promotion values"
            )
    if (
        type(config.fast_math) is not bool
        or math.isnan(config.activation_clamp)
        or config.activation_clamp <= 0
    ):
        raise ValueError("invalid fast_math or activation_clamp")


def derive(
    config,
    *,
    shape,
    tokens_bound,
    num_sms,
    topology,
    fmt="fp8",
    grid=None,
):
    """Resolve scalar launch arguments using the common bound, never local M."""
    world = 16 if topology == "ep16" else 8
    experts = shape.num_experts // world
    expected = float(tokens_bound) * shape.topk / experts
    grid = config.ctas_per_sm * num_sms if grid is None else grid
    swap = config.body in (MathBody.SWAP_BN128, MathBody.SWAP_BN256)
    large = config.body is MathBody.SPLIT_BN256
    split_mn = config.body is MathBody.SPLIT_BM128_BN256
    stages = config.stages
    if stages is None:
        stages = select_stages(
            hidden=shape.hidden,
            num_experts=shape.num_experts,
            block_m=config.block_m,
            block_n=config.block_n,
            num_math_warps=config.math_warps,
            use_swap_ab=swap,
        )
        if split_mn:
            stages = min(stages, 2)
    wave = config.experts_per_wave
    if wave is None:
        wave = (
            experts
            if fmt == "mxfp4"
            else derive_experts_per_wave(
                expected_tokens_per_expert=expected,
                num_experts_per_rank=experts,
                intermediate_hidden=shape.intermediate_hidden,
                block_m=config.block_m,
                block_n=config.block_n,
                num_sms=grid,
            )
        )
    wave = min(wave, experts)
    counter = split_mn or (config.block_n == 256 and 4 <= tokens_bound <= 128)
    if topology == "ep8" and fmt == "fp8" and not swap:
        counter = True
    full_sync = not counter and not large
    producer_regs = config.producer_registers
    combine_maxnreg = derive_launch_maxnreg(
        num_math_warps=config.math_warps,
        math_register_budget=config.math_registers,
        specialized_register_budgets=(producer_regs, producer_regs),
    )
    budgets = (
        producer_regs,
        producer_regs,
        config.dispatch_registers,
        config.dispatch_registers,
    )
    if topology == "ep16":
        counter = not swap
        full_sync = not counter
        tma = 24 if large else producer_regs
        budgets = (tma, tma, config.dispatch_registers, config.dispatch_registers, 40)
    maxnreg = derive_launch_maxnreg(
        num_math_warps=config.math_warps,
        math_register_budget=config.math_registers,
        specialized_register_budgets=budgets,
    )
    if fmt == "mxfp4":
        maxnreg = 128
    return LaunchConfig(
        block_m=config.block_m,
        block_n=config.block_n,
        block_k=config.block_k,
        num_stages=stages,
        num_math_warps=config.math_warps,
        num_experts_per_wave=wave,
        use_swap_ab=swap,
        use_split_bn256=large,
        expected_tokens_per_expert=expected,
        reuse_accum_as_final=not swap and not large,
        fc2_arrival_counter=counter,
        fc2_epilogue_requires_full_sync=full_sync,
        scalarize_hot_rescale=not swap and not large,
        math_register_budget=config.math_registers,
        dispatch_register_budget=config.dispatch_registers,
        non_epilogue_register_budget=producer_regs,
        launch_maxnreg=maxnreg,
        combine_launch_maxnreg=combine_maxnreg,
        fc1_promotion_k=config.fc1_promotion_k,
        fc2_promotion_k=config.fc2_promotion_k,
    )


def _make_selector(expected_topology, expected_fmt, table, configs):
    """Give each backend a private policy table and bounded selection cache."""

    @lru_cache(maxsize=4096)
    def _select_cached(
        *,
        topology,
        fmt,
        shape,
        tokens_bound,
        num_sms,
        override=None,
        strict=False,
        grid=None,
    ):
        world = 16 if topology == "ep16" else 8
        key = (topology, fmt, shape.key, num_sms)
        rows = table.get(key)
        if rows is None:
            rows = table.get((topology, fmt, shape.key, 78))
        row = None
        adjustments = ()
        if override is not None:
            config = override
        elif rows is not None:
            row = rows[bisect_right(tuple(r.m_lo for r in rows), tokens_bound) - 1]
            config = configs[row.config]
        elif strict:
            raise ValueError("no policy table for the exact shape")
        elif fmt == "mxfp4":
            config = replace(
                configs["mxfp4_split128_e2"],
                experts_per_wave=min(2, shape.num_experts // world),
            )
        else:
            # The generic fallback retains the format-neutral shape calculations.
            launch = _select_generic_config(
                num_tokens_per_rank=max(tokens_bound, 1),
                hidden=shape.hidden,
                intermediate_hidden=shape.intermediate_hidden,
                num_experts=shape.num_experts,
                num_experts_per_rank=shape.num_experts // world,
                topk=shape.topk,
                num_sms=num_sms,
            )
            body = (
                (MathBody.SWAP_BN256 if launch.block_n == 256 else MathBody.SWAP_BN128)
                if launch.use_swap_ab
                else MathBody.SPLIT_BM128_BN256
                if launch.block_m == 128
                else MathBody.SPLIT_BN256
                if launch.block_n == 256
                else MathBody.SPLIT_BN128
            )
            config = MegaMoEConfig(
                body,
                stages=launch.num_stages,
                math_warps=launch.num_math_warps,
                experts_per_wave=launch.num_experts_per_wave,
                block_m=launch.block_m,
                math_registers=launch.math_register_budget,
                producer_registers=launch.non_epilogue_register_budget,
                dispatch_registers=(40 if launch.num_math_warps == 16 else 48)
                if world == 16
                else launch.dispatch_register_budget,
                fc1_promotion_k=(32 if world == 16 else 64)
                if launch.use_swap_ab
                else 128,
                fc2_promotion_k=32 if world == 16 and launch.use_swap_ab else 64,
            )
        validate_override(
            config,
            fmt=fmt,
            topology=topology,
            experts_per_rank=shape.num_experts // world,
        )
        max_ctas = (
            3
            if topology == "ep16" and config.block_m == 16 and config.stages == 3
            else 2
        )
        if grid is not None and grid > max_ctas * num_sms:
            raise ValueError(
                f"persistent grid exceeds {max_ctas} CTAs per SM for the selected body"
            )
        launch = derive(
            config,
            shape=shape,
            tokens_bound=tokens_bound,
            num_sms=num_sms,
            topology=topology,
            fmt=fmt,
            grid=grid,
        )
        return Selection(
            config,
            launch,
            row,
            config.ctas_per_sm * num_sms if grid is None else grid,
            128 if topology == "ep16" else None,
            adjustments,
        )

    def select(
        *,
        topology,
        fmt,
        shape,
        tokens_bound,
        num_sms,
        override=None,
        strict=False,
        grid=None,
    ):
        """Select a rank-common launch policy without initializing a GPU runtime."""
        if (topology, fmt) != (expected_topology, expected_fmt):
            raise ValueError("unsupported topology or format")
        if type(tokens_bound) is not int or tokens_bound < 0:
            raise ValueError("tokens_bound must be a nonnegative integer")
        if type(num_sms) is not int or num_sms <= 0:
            raise ValueError("num_sms must be a positive integer")
        if grid is not None and (type(grid) is not int or grid <= 0):
            raise ValueError("persistent grid must be a positive integer")
        if topology == "ep16" and tokens_bound > 128:
            raise ValueError(
                "EP16 runtime capacity supports at most 128 tokens per rank"
            )
        shape = _shape(shape)
        world = 16 if topology == "ep16" else 8
        shape.validate(world)
        if override is not None:
            validate_override(
                override,
                fmt=fmt,
                topology=topology,
                experts_per_rank=shape.num_experts // world,
            )
        return _select_cached(
            topology=topology,
            fmt=fmt,
            shape=shape,
            tokens_bound=tokens_bound,
            num_sms=num_sms,
            override=override,
            strict=strict,
            grid=grid,
        )

    return select


# Shape-specific launch policies.


SHAPES = MappingProxyType(
    {
        "h4096_i2048_e256_k6": Shape(
            hidden=4096, intermediate_hidden=2048, num_experts=256, topk=6
        ),
        "h7168_i3072_e384_k6": Shape(
            hidden=7168, intermediate_hidden=3072, num_experts=384, topk=6
        ),
    }
)

ALIASES = MappingProxyType(
    {"flash": "h4096_i2048_e256_k6", "pro": "h7168_i3072_e384_k6"}
)

CONFIGS = MappingProxyType(
    {
        "split256_auto": MegaMoEConfig(
            body=MathBody.SPLIT_BN256,
            fc1_promotion_k=128,
        ),
        "split256_s3": MegaMoEConfig(
            body=MathBody.SPLIT_BN256,
            stages=3,
            fc1_promotion_k=128,
        ),
        "split_bm128_s2": MegaMoEConfig(
            body=MathBody.SPLIT_BM128_BN256,
            stages=2,
            math_warps=16,
            block_m=128,
            math_registers=112,
            producer_registers=24,
            dispatch_registers=32,
            fc1_promotion_k=128,
        ),
        "swap_auto_fc1_128": MegaMoEConfig(
            body=MathBody.SWAP_BN128,
            fc1_promotion_k=128,
        ),
        "swap_wave9_fc1_128": MegaMoEConfig(
            body=MathBody.SWAP_BN128,
            experts_per_wave=9,
            fc1_promotion_k=128,
        ),
    }
)

TABLE = MappingProxyType(
    {
        ("ep8", "fp8", "h4096_i2048_e256_k6", 78): (
            Row(0, 7, "swap_auto_fc1_128"),
            Row(8, 8, "swap_wave9_fc1_128"),
            Row(9, 128, "swap_auto_fc1_128"),
            Row(129, 8191, "split256_auto"),
            Row(8192, 8192, "split256_s3"),
            Row(8193, None, "split_bm128_s2"),
        ),
        ("ep8", "fp8", "h7168_i3072_e384_k6", 78): (
            Row(0, 241, "swap_auto_fc1_128"),
            Row(242, 1023, "split256_auto"),
            Row(1024, 1024, "split256_s3"),
            Row(1025, None, "split256_auto"),
        ),
    }
)

select = _make_selector("ep8", "fp8", TABLE, CONFIGS)


# resources.py


class ResourcePreparationError(RuntimeError):
    """All candidates failed; every rank retains the same attempt reports."""

    def __init__(self, attempts):
        self.attempts = tuple(attempts)
        super().__init__(f"no rank-common resident configuration: {self.attempts}")


@dataclass(frozen=True)
class ResourcePreparation:
    selection: object
    attempts: tuple


def prepare_collectively(candidates, *, resolve, probe, gather, contract):
    """Compile candidates, shrink a grid to the common capacity, then fall back.

    ``probe(selection)`` must not register inputs or launch the operation. It
    returns actual driver occupancy, including ``blocks_per_sm`` and
    ``physical_sms``. Every exception is gathered before any rank proceeds.
    ``candidates`` is the same ordered sequence of ``(config, initial_grid)``
    pairs on every rank. An initial grid of None delegates to the selector.
    """
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("resource preparation needs at least one candidate")
    identities = gather(dict(contract=contract, candidates=candidates))
    validate_policy_agreement(identities)
    attempts = []
    for index, (config, initial_grid) in enumerate(candidates):
        grid = initial_grid
        while True:
            selected = None
            try:
                selected = resolve(config, grid)
                # The selector can change waves and partition composition at a
                # smaller grid. Probe again after each adjustment.
                proof = dict(probe(selected))
                blocks, sms = proof["blocks_per_sm"], proof["physical_sms"]
                if (
                    type(blocks) is not int
                    or blocks < 0
                    or type(sms) is not int
                    or sms <= 0
                ):
                    raise ValueError("invalid driver occupancy proof")
                local = dict(
                    error=None,
                    grid=selected.grid,
                    config=selected.config,
                    launch=selected.launch,
                    proof=proof,
                    capacity=blocks * sms,
                )
            except Exception as exc:
                local = dict(error=f"{type(exc).__name__}: {exc}", grid=grid)
            reports = tuple(gather(local))
            attempts.append(dict(candidate=index, requested_grid=grid, ranks=reports))
            if any(report.get("error") for report in reports):
                break
            validate_policy_agreement(
                [
                    {key: report[key] for key in ("grid", "config", "launch")}
                    for report in reports
                ]
            )
            requested = reports[0]["grid"]
            capacity = min(report["capacity"] for report in reports)
            if requested <= capacity:
                adjustments = list(selected.adjustments)
                if index:
                    adjustments.append(f"resource:fallback:{index}")
                if grid != initial_grid:
                    adjustments.append(f"resource:grid:{grid}")
                if adjustments != list(selected.adjustments):
                    selected = replace(selected, adjustments=tuple(adjustments))
                return ResourcePreparation(selected, tuple(attempts))
            # A zero-residency CTA cannot be repaired by reducing the grid.
            minimum = 2 if selected.config.combine.value == "chunked" else 1
            if capacity < minimum:
                break
            grid = capacity
    raise ResourcePreparationError(attempts)


# context.py


@dataclass(frozen=True)
class ExpertPool:
    """Local form of DeepGEMM's padded expert-pool contract.

    The multi-rank implementation uses the same fields; only the source route
    table and the final scatter destinations become peer-addressed.
    """

    acts: torch.Tensor
    acts_sf_mn_major: torch.Tensor
    topk_weights: torch.Tensor
    token_src_metadata: torch.Tensor
    expert_state: torch.Tensor
    source_routes: torch.Tensor

    @property
    def expert_recv_count(self) -> torch.Tensor:
        """Return the low-32 count view of the packed expert state.

        This is a derived compatibility/debug view, not a dispatch workspace
        buffer.  Kernels consume ``expert_state`` directly.
        """
        return (self.expert_state & 0xFFFFFFFF).to(torch.int32)

    @property
    def expert_pool_block_offsets(self) -> torch.Tensor:
        """Derive BM64 offsets retained for compatibility/debug callers."""
        return self.expert_pool_block_offsets_for(_BLOCK_M_VALUE)

    def expert_pool_block_offsets_for(self, block_m: int) -> torch.Tensor:
        """Derive the padded expert-pool prefix for one selected BLOCK_M."""
        if block_m not in (_BLOCK_M_VALUE, _SPLIT_BLOCK_M_VALUE):
            raise ValueError("block_m must be 64 or 128")
        counts = self.expert_recv_count.to(torch.int64)
        blocks = torch.div(
            counts + block_m - 1,
            block_m,
            rounding_mode="floor",
        )
        return torch.cat((blocks.new_zeros(1), torch.cumsum(blocks, dim=0)))


@dataclass(frozen=True)
class GemmDescriptors:
    """TMA descriptors whose block boxes depend on GEMM BLOCK_M/BLOCK_N."""

    block_m: int
    block_n: int
    l1_a_desc: object
    l1_sfa_desc: object
    l1_b_desc: object
    l2_store_desc: object
    l2_a_desc: object
    l2_sfa_desc: object
    l2_b_desc: object


@dataclass(frozen=True)
class RegisteredInputs:
    """Inputs registered in symmetric memory for one dispatch invocation.

    Only the first ``num_tokens`` activation and scale rows are valid.  Top-k
    metadata is valid through the context's full ``max_tokens`` capacity;
    padded rows contain ``-1`` indices and zero weights, matching DeepGEMM's
    SM90 pre-dispatch contract.
    """

    input_acts_fp8: torch.Tensor
    input_acts_sf: torch.Tensor
    input_topk_idx: torch.Tensor
    input_topk_weights: torch.Tensor
    num_tokens: int
    hidden: int
    source_dtype: torch.dtype
    routed_scaling_factor: float
    compiled: object


class SymmetricArena:
    """Format-neutral allocation boundary, including unmapped remote peers.

    Keep tensors and rendezvous handles alive together. Device RMA callers pass
    the original local tensor and integer offsets, never a tensor value-add as
    an address. This arena does not change the existing EP8 context or kernels.
    """

    def __init__(self, *, backend: str, group=None):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        require_sm90()
        if not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before creating an arena")
        self.group = dist.group.WORLD if group is None else group
        self.rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)
        self.device = torch.device("cuda", torch.cuda.current_device())
        # The CUDA allocator is installed lazily by the first empty() call in
        # stock PyTorch. set_backend("CUDA") before that registration can fail
        # even though the default CUDA allocation path is available (as in EP8).
        if backend == "CUDA":
            current = symm_mem.get_backend(self.device)
            if current not in (None, "CUDA"):
                raise RuntimeError(
                    f"expected default CUDA symmetric allocator, found {current}"
                )
        else:
            symm_mem.set_backend(backend)
        self.backend = backend
        self.allocations = {}

    def allocate(self, name, shape, dtype):
        import torch.distributed._symmetric_memory as symm_mem

        if name in self.allocations:
            raise ValueError(f"duplicate symmetric allocation {name}")
        tensor = symm_mem.empty(*shape, dtype=dtype, device=self.device)
        handle = symm_mem.rendezvous(tensor, group=self.group)
        self.allocations[name] = (tensor, handle)
        return tensor, handle


def create_arena(*, backend: str, group=None):
    """Collectively create a standalone symmetric allocation context."""
    return SymmetricArena(backend=backend, group=group)


class SymmetricContext:
    """Single-node symmetric buffers used by the SM90 pull/scatter path.

    Only PyTorch's CUDA symmetric-memory backend is used.  No runtime API from
    ``triton_dist`` participates in allocation, rendezvous, or peer mapping.
    """

    def __init__(
        self,
        *,
        max_tokens: int,
        hidden: int,
        num_experts: int,
        topk: int,
        world_size: int,
        rank: int,
        device: torch.device,
        group_name: str,
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 < topk <= 32:
            raise ValueError("topk must be in [1, 32]")
        if num_experts % world_size:
            raise ValueError("num_experts must be divisible by world_size")
        if hidden % 128:
            raise ValueError("hidden must be divisible by 128")
        self.max_tokens = max_tokens
        self.hidden = hidden
        self.num_experts = num_experts
        self.topk = topk
        self.world_size = world_size
        self.rank = rank
        self.experts_per_rank = num_experts // world_size
        self.max_routes = max_tokens * topk
        self.device = device
        self.group_name = group_name

        def create(shape, dtype):
            storage_dtype = torch.int8 if dtype == torch.float8_e4m3fn else dtype
            tensor = symm_mem.empty(*shape, dtype=storage_dtype, device=device)
            if dtype == torch.float8_e4m3fn:
                tensor = tensor.view(dtype)
            handle = symm_mem.rendezvous(tensor, group=group_name)
            return tensor, handle, storage_dtype

        self.input_acts, acts_handle, acts_storage_dtype = create(
            (max_tokens, hidden),
            torch.float8_e4m3fn,
        )
        self.input_sf, sf_handle, sf_storage_dtype = create(
            (max_tokens, hidden // 128),
            torch.float32,
        )
        self.input_topk_weights, weight_handle, weight_storage_dtype = create(
            (max_tokens, topk),
            torch.float32,
        )
        self.input_topk_idx, index_handle, _ = create(
            (max_tokens, topk),
            torch.int64,
        )
        self.source_routes, route_handle, route_storage_dtype = create(
            (world_size, self.experts_per_rank, self.max_routes),
            torch.int32,
        )
        self.recv_count, count_handle, count_storage_dtype = create(
            (world_size, self.experts_per_rank),
            torch.int32,
        )
        self.expert_state, expert_state_handle, expert_state_storage_dtype = create(
            (self.experts_per_rank,),
            torch.int64,
        )
        # Gluon exposes a real contiguous PyTorch [token, topk, hidden] tensor.
        # DeepGEMM's workspace ``Buffer(..., num_ranks=topk, ...)`` is instead
        # physically slot-major; its address formula must not be reused here.
        self.combine_buffer, combine_handle, combine_storage_dtype = create(
            (max_tokens, topk, hidden),
            torch.bfloat16,
        )
        (
            self.dispatch_barrier,
            dispatch_barrier_handle,
            dispatch_barrier_storage_dtype,
        ) = create(
            (world_size,),
            torch.int32,
        )
        self.fused_barrier, fused_barrier_handle, fused_barrier_storage_dtype = create(
            (world_size,),
            torch.int32,
        )

        def peer_ptrs(handle, shape, storage_dtype):
            return torch.tensor(
                [
                    handle.get_buffer(peer, shape, storage_dtype).data_ptr()
                    for peer in range(world_size)
                ],
                dtype=torch.int64,
                device=device,
            )

        self.peer_input_acts = tuple(
            acts_handle.get_buffer(
                peer,
                (max_tokens, hidden),
                acts_storage_dtype,
            ).view(torch.float8_e4m3fn)
            for peer in range(world_size)
        )
        self.peer_input_sf_ptrs = peer_ptrs(
            sf_handle,
            (max_tokens, hidden // 128),
            sf_storage_dtype,
        )
        self.peer_input_topk_weights_ptrs = peer_ptrs(
            weight_handle,
            (max_tokens, topk),
            weight_storage_dtype,
        )
        self.peer_source_routes_ptrs = peer_ptrs(
            route_handle,
            (world_size, self.experts_per_rank, self.max_routes),
            route_storage_dtype,
        )
        self.peer_recv_count_ptrs = peer_ptrs(
            count_handle,
            (world_size, self.experts_per_rank),
            count_storage_dtype,
        )
        self.peer_expert_state_ptrs = peer_ptrs(
            expert_state_handle,
            (self.experts_per_rank,),
            expert_state_storage_dtype,
        )
        self.peer_combine_buffer_ptrs = peer_ptrs(
            combine_handle,
            (max_tokens, topk, hidden),
            combine_storage_dtype,
        )
        self.peer_dispatch_barrier_ptrs = peer_ptrs(
            dispatch_barrier_handle,
            (world_size,),
            dispatch_barrier_storage_dtype,
        )
        self.peer_fused_barrier_ptrs = peer_ptrs(
            fused_barrier_handle,
            (world_size,),
            fused_barrier_storage_dtype,
        )
        self._handles = (
            acts_handle,
            sf_handle,
            weight_handle,
            index_handle,
            route_handle,
            count_handle,
            expert_state_handle,
            combine_handle,
            dispatch_barrier_handle,
            fused_barrier_handle,
        )
        self._barrier_handle = acts_handle

    def barrier(self) -> None:
        self._barrier_handle.barrier()


@gluon.jit
def reset_control_kernel(
    l1_arrival,
    l2_arrival,
    actual_num_pool_rows,
    dispatch_counter,
    dispatch_barrier,
    fused_barrier,
    fc2_scatter_grid_counter,
    combine_cross_rank_ready,
    expert_state,
    expert_send_state,
    world_size: gl.constexpr,
    num_local_experts: gl.constexpr,
    num_global_experts: gl.constexpr,
    num_pool_blocks: gl.constexpr,
    reset_block_size: gl.constexpr,
    reset_layout: gl.constexpr,
):
    """Reset only fixed-size per-launch control state.

    Pool payload, scale-factor padding, and metadata are deliberately left
    untouched.  Every observable epilogue is guarded by ``valid_m`` and every
    valid row is overwritten before its arrival is published.  The compact
    arrival arrays are reset here instead of coupling dispatch completion to
    the combine partition through an in-kernel tail-cleanup rendezvous.
    """
    offsets = gl.program_id(0) * reset_block_size + gl.arange(
        0, reset_block_size, layout=reset_layout
    )
    gl.store(
        l1_arrival + offsets,
        0,
        mask=offsets < num_pool_blocks,
    )
    gl.store(
        l2_arrival + offsets,
        0,
        mask=offsets < num_pool_blocks,
    )
    gl.store(
        actual_num_pool_rows + offsets,
        0,
        mask=offsets < 1,
    )
    gl.store(
        dispatch_counter + offsets,
        0,
        mask=offsets < 1,
    )
    gl.store(
        dispatch_barrier + offsets,
        0,
        mask=offsets < world_size,
    )
    gl.store(
        fused_barrier + offsets,
        0,
        mask=offsets < world_size,
    )
    gl.store(
        fc2_scatter_grid_counter + offsets,
        0,
        mask=offsets < 1,
    )
    gl.store(
        combine_cross_rank_ready + offsets,
        0,
        mask=offsets < 1,
    )
    gl.store(
        expert_state + offsets,
        0,
        mask=offsets < num_local_experts,
    )
    gl.store(
        expert_send_state + offsets,
        0,
        mask=offsets < num_global_experts,
    )


@gluon.jit
def register_inputs_kernel(
    x,
    x_sf,
    topk_idx,
    topk_weights,
    registered_x,
    registered_x_sf,
    registered_topk_idx,
    registered_topk_weights,
    num_tokens,
    padded_max,
    hidden: gl.constexpr,
    num_groups: gl.constexpr,
    topk: gl.constexpr,
    routed_scaling_factor,
    input_is_bf16: gl.constexpr,
    quant_layout: gl.constexpr,
    route_layout: gl.constexpr,
    block_id=None,
):
    """Register BF16 or block-scaled FP8 inputs in symmetric memory.

    The 64x128 register tile maps one half warp to each per-128 group.  With
    32 warps this is the same 1024-thread, one-token-per-CTA decomposition as
    DeepGEMM's SM90 pre-dispatch kernel for hidden sizes up to 8192.
    """
    bid = gl.program_id(0) if block_id is None else block_id
    route_offsets = gl.arange(
        0,
        1024,
        layout=route_layout,
    )
    if bid < num_tokens:
        group_offsets = gl.arange(
            0,
            64,
            layout=gl.SliceLayout(1, quant_layout),
        )
        columns = gl.arange(
            0,
            128,
            layout=gl.SliceLayout(0, quant_layout),
        )
        element_offsets = group_offsets[:, None] * 128 + columns[None, :]
        valid_groups = group_offsets < num_groups
        valid_elements = valid_groups[:, None]
        if input_is_bf16:
            values = gl.load(
                x + bid * hidden + element_offsets,
                mask=valid_elements,
                other=0.0,
            ).to(gl.float32)
            absmax = gl.max(gl.abs(values), axis=1)
            scale = gl.maximum(absmax, 1.0e-10) * (1.0 / 448.0)
            quantized = (values * gl.fdiv(1.0, scale[:, None])).to(gl.float8e4nv)
        else:
            quantized = gl.load(
                x + bid * hidden + element_offsets,
                mask=valid_elements,
                other=0.0,
            )
            scale = gl.load(
                x_sf + bid * num_groups + group_offsets,
                mask=valid_groups,
                other=0.0,
            )
        gl.store(
            registered_x + bid * hidden + element_offsets,
            quantized,
            mask=valid_elements,
        )
        gl.store(
            registered_x_sf + bid * num_groups + group_offsets,
            scale,
            mask=valid_groups,
        )

        valid_slots = route_offsets < topk
        route = bid * topk + route_offsets
        indices = gl.load(
            topk_idx + route,
            mask=valid_slots,
            other=-1,
        ).to(gl.int64)
        weights = gl.load(
            topk_weights + route,
            mask=valid_slots,
            other=0.0,
        )
        gl.store(
            registered_topk_idx + route,
            indices,
            mask=valid_slots,
        )
        gl.store(
            registered_topk_weights + route,
            weights * routed_scaling_factor,
            mask=valid_slots,
        )
    else:
        pad_block = bid - num_tokens
        padded_route = num_tokens * topk + pad_block * 1024 + route_offsets
        valid_padding = padded_route < padded_max * topk
        gl.store(
            registered_topk_idx + padded_route,
            -1,
            mask=valid_padding,
        )
        gl.store(
            registered_topk_weights + padded_route,
            0.0,
            mask=valid_padding,
        )


def dispatch_scale_groups(hidden: int) -> int:
    """Return the K/128 extent required by the dispatch TMA shared layout."""
    sf_groups = hidden // 128
    # Hopper collapses the leading dimensions of this rank-3 layout and
    # requires at least an [8, 128] tile. This format-neutral dispatch
    # constraint also covers correctness shapes with small hidden sizes.
    return max(triton.next_power_of_2(sf_groups), 8)


def create_dispatch_descriptors(
    ctx: SymmetricContext,
    pool_acts: torch.Tensor,
):
    """Create one 3D TMA tile per token, padding K with hardware OOB.

    The logical tensor is ``[token, K // 128, 128]`` while the TMA box is
    ``[1, max(8, next_power_of_2(K // 128)), 128]``.  For Pro K=7168 this maps a
    56-group token into a 64-group (8 KiB) transaction.  TMA zero-fills the
    final eight groups on load and suppresses them on store, so the padded box
    never aliases the following token or pool row.
    """
    if ctx.world_size not in (8, 16) or len(ctx.peer_input_acts) != ctx.world_size:
        raise ValueError(
            "3D OOB TMA dispatch requires one mapped tensor per 8 or 16 logical sources"
        )
    if ctx.hidden <= 0 or ctx.hidden % 128:
        raise ValueError("3D OOB TMA dispatch requires K divisible by 128")
    sf_groups = ctx.hidden // 128
    padded_sf_groups = dispatch_scale_groups(ctx.hidden)
    if padded_sf_groups > 256:
        raise ValueError("3D OOB TMA dispatch supports at most 256 K groups")
    block_shape = [1, padded_sf_groups, 128]
    layout = gl.NVMMASharedLayout(
        swizzle_byte_width=128,
        element_bitwidth=8,
        rank=3,
    )
    peer_descs = tuple(
        TensorDescriptor.from_tensor(
            peer.view(ctx.max_tokens, sf_groups, 128),
            block_shape,
            layout,
        )
        for peer in ctx.peer_input_acts
    )
    pool_desc = TensorDescriptor.from_tensor(
        pool_acts.view(pool_acts.shape[0], sf_groups, 128),
        block_shape,
        layout,
    )
    return (*peer_descs, pool_desc)


def _gather_policy_debug(ctx, value):
    import torch.distributed as dist

    group = getattr(ctx, "group", None)
    if group is None:
        group = dist.distributed_c10d._resolve_process_group(ctx.group_name)
    world = dist.get_world_size(group)
    if world != ctx.world_size:
        raise ValueError(
            "policy debug validation requires a group containing every logical rank"
        )
    values = [None] * world
    dist.all_gather_object(values, value, group=group)
    return values


def check_policy_agreement(ctx, **contract):
    """Check resolved scalar choices before entering the peer protocol.

    Enable MEGA_MOE_GLUON_CHECK_POLICY on every rank outside graph capture.
    This is a diagnostic for valid but inconsistent choices; ordinary local
    input validation still applies. The production path has no collective.
    """
    if os.environ.get("MEGA_MOE_GLUON_CHECK_POLICY") == "1":
        validate_policy_agreement(_gather_policy_debug(ctx, contract))


def resolve_tokens_bound(ctx, local_tokens, tokens_bound):
    """Validate the common bound; optional debug collective catches disagreement.

    Debug mode must be enabled on every rank and is unsuitable for graph
    capture. The ordinary path adds no communication or device synchronization.
    """

    bound = local_tokens if tokens_bound is None else tokens_bound
    if os.environ.get("MEGA_MOE_GLUON_CHECK_POLICY") == "1":
        values = _gather_policy_debug(ctx, (local_tokens, bound, ctx.max_tokens))
        for local, common, capacity in values:
            validate_tokens_bound(local, common, capacity)
        if len({(common, capacity) for _, common, capacity in values}) != 1:
            raise ValueError(
                "tokens_bound and context capacity must agree on every rank"
            )
    return validate_tokens_bound(local_tokens, bound, ctx.max_tokens)


def create_context(
    *,
    max_tokens: int,
    hidden: int,
    num_experts: int,
    topk: int,
    group_name: str | None = None,
) -> SymmetricContext:
    """Collectively create the single-node peer buffers for MegaMoE."""
    import torch.distributed as dist

    require_sm90()
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before rendezvous")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if group_name is None:
        group_name = dist.group.WORLD.group_name
    return SymmetricContext(
        max_tokens=max_tokens,
        hidden=hidden,
        num_experts=num_experts,
        topk=topk,
        world_size=world_size,
        rank=rank,
        device=torch.device("cuda", torch.cuda.current_device()),
        group_name=group_name,
    )


def register_inputs(
    ctx: SymmetricContext,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    x_sf: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    launch: bool = True,
) -> RegisteredInputs:
    """Register dispatch inputs using the narrow DeepGEMM-style boundary.

    BF16 input is quantized to FP8 E4M3 with one FP32 scale per token/per-128
    group.  Already-quantized FP8 E4M3 input is copied together with its
    required FP32 ``x_sf`` tensor.  Both specializations copy top-k indices,
    multiply top-k weights by ``routed_scaling_factor``, and initialize padded
    top-k rows to ``-1``/zero.  Route counting, peer publication, prefix sums,
    pool materialization, and arrival signaling deliberately remain in the
    dispatch partition.
    """
    require_sm90()
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional BF16 or FP8 E4M3 tensor")
    if x.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("x must have dtype bfloat16 or float8_e4m3fn")
    num_tokens, hidden = x.shape
    if ctx.max_tokens <= 0:
        raise ValueError("the symmetric context must reserve at least one token")
    if not 0 < ctx.topk <= _PRE_DISPATCH_THREADS:
        raise ValueError("SM90 pre-dispatch requires topk in [1, 1024]")
    if num_tokens > ctx.max_tokens or hidden != ctx.hidden:
        raise ValueError("input exceeds the symmetric context capacity")
    if hidden <= 0:
        raise ValueError("SM90 pre-dispatch requires a positive hidden dimension")
    if hidden > (_PRE_DISPATCH_GROUPS_PER_CTA * _PRE_DISPATCH_GROUP_SIZE):
        raise ValueError("SM90 pre-dispatch supports hidden dimensions up to 8192")
    if hidden % _PRE_DISPATCH_GROUP_SIZE:
        raise ValueError("SM90 pre-dispatch requires hidden divisible by 128")
    if x.dtype == torch.bfloat16:
        if x_sf is not None:
            raise ValueError("x_sf must be None when pre-dispatch input is BF16")
    else:
        if x_sf is None:
            raise ValueError("x_sf is required when pre-dispatch input is FP8")
        if x_sf.shape != (num_tokens, hidden // 128):
            raise ValueError("x_sf must be [num_tokens, hidden // 128]")
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
    tensors = (topk_idx, topk_weights)
    if x_sf is not None:
        tensors = (*tensors, x_sf)
    if x.device != ctx.device or any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all pre-dispatch tensors must use the context device")
    if not x.is_contiguous() or any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all pre-dispatch tensors must be contiguous")
    routed_scaling_factor = float(routed_scaling_factor)
    if not math.isfinite(routed_scaling_factor):
        raise ValueError("routed_scaling_factor must be finite")

    quant_layout = gl.BlockedLayout(
        [1, 8],
        [2, 16],
        [_PRE_DISPATCH_NUM_WARPS, 1],
        [1, 0],
    )
    route_layout = gl.BlockedLayout(
        [1],
        [32],
        [_PRE_DISPATCH_NUM_WARPS],
        [0],
    )
    num_padding_routes = (ctx.max_tokens - num_tokens) * ctx.topk
    grid = (
        num_tokens
        + triton.cdiv(
            num_padding_routes,
            _PRE_DISPATCH_THREADS,
        ),
    )
    source_sf = ctx.input_sf if x_sf is None else x_sf
    compiled = None
    if launch:
        compiled = register_inputs_kernel[grid](
            x,
            source_sf,
            topk_idx,
            topk_weights,
            ctx.input_acts,
            ctx.input_sf,
            ctx.input_topk_idx,
            ctx.input_topk_weights,
            num_tokens,
            ctx.max_tokens,
            hidden,
            hidden // _PRE_DISPATCH_GROUP_SIZE,
            ctx.topk,
            routed_scaling_factor,
            x.dtype == torch.bfloat16,
            quant_layout,
            route_layout,
            num_warps=_PRE_DISPATCH_NUM_WARPS,
        )
    return RegisteredInputs(
        input_acts_fp8=ctx.input_acts,
        input_acts_sf=ctx.input_sf,
        input_topk_idx=ctx.input_topk_idx,
        input_topk_weights=ctx.input_topk_weights,
        num_tokens=num_tokens,
        hidden=hidden,
        source_dtype=x.dtype,
        routed_scaling_factor=routed_scaling_factor,
        compiled=compiled,
    )


# primitives.py


@gluon.jit
def _store_contiguous_bf16_fragment(
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
def _load_contiguous_bf16_fragment(ptrs, mask, width: gl.constexpr):
    """Load one lane-owned combine fragment as one aligned transaction."""
    packed_mask = mask.to(gl.int8)
    if width == 8:
        payload = gl.inline_asm_elementwise(
            """
            {
            .reg .pred load_pred;
            setp.ne.u32 load_pred, $12, 0;
            mov.u32 $0, 0;
            mov.u32 $1, 0;
            mov.u32 $2, 0;
            mov.u32 $3, 0;
            @load_pred ld.global.v4.b32 {$0, $1, $2, $3}, [$4];
            }
            """,
            "=r,=r,=r,=r,l,l,l,l,l,l,l,l,r,r",
            [ptrs, packed_mask],
            dtype=gl.uint16,
            is_pure=False,
            pack=8,
        )
    else:
        gl.static_assert(width == 4, "BF16 vector width must be 4 or 8")
        payload = gl.inline_asm_elementwise(
            """
            {
            .reg .pred load_pred;
            setp.ne.u32 load_pred, $6, 0;
            mov.u32 $0, 0;
            mov.u32 $1, 0;
            @load_pred ld.global.v2.b32 {$0, $1}, [$2];
            }
            """,
            "=r,=r,l,l,l,l,r",
            [ptrs, packed_mask],
            dtype=gl.uint16,
            is_pure=False,
            pack=4,
        )
    return payload.to(gl.bfloat16, bitcast=True)


@gluon.jit
def _store_contiguous_bf16_fragment_16b(ptrs, values, mask):
    """Store eight lane-owned BF16 values without a constexpr branch."""
    packed_values = values.to(gl.uint16, bitcast=True)
    packed_mask = mask.to(gl.int8)
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


@gluon.jit
def _load_contiguous_bf16_fragment_16b(ptrs, mask):
    """Load eight lane-owned BF16 values without a constexpr branch."""
    packed_mask = mask.to(gl.int8)
    payload = gl.inline_asm_elementwise(
        """
        {
        .reg .pred load_pred;
        setp.ne.u32 load_pred, $12, 0;
        mov.u32 $0, 0;
        mov.u32 $1, 0;
        mov.u32 $2, 0;
        mov.u32 $3, 0;
        @load_pred ld.global.v4.b32 {$0, $1, $2, $3}, [$4];
        }
        """,
        "=r,=r,=r,=r,l,l,l,l,l,l,l,l,r,r",
        [ptrs, packed_mask],
        dtype=gl.uint16,
        is_pure=False,
        pack=8,
    )
    return payload.to(gl.bfloat16, bitcast=True)


@gluon.jit
def _load_i32_acquire_gpu(ptr):
    """Issue a non-RMW GPU-scope acquire load for a local counter."""
    return gl.inline_asm_elementwise(
        "ld.global.gpu.acquire.b32 $0, [$1];",
        "=r,l",
        [ptr],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _load_i32_acquire_sys_if(ptr, load_mask):
    """Issue a predicated system-acquire load for peer-visible counters."""
    return gl.inline_asm_elementwise(
        """
        {
            .reg .pred load_pred;
            setp.ne.u32 load_pred, $2, 0;
            mov.b32 $0, 0;
            @load_pred ld.global.sys.acquire.b32 $0, [$1];
        }
        """,
        "=r,l,r",
        [ptr, load_mask.to(gl.int32)],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _red_i32_release_sys_if(ptr, value, store_mask):
    """Issue DeepGEMM's predicated system-release reduction add."""
    return gl.inline_asm_elementwise(
        """
        {
            .reg .pred store_pred;
            setp.ne.u32 store_pred, $3, 0;
            @store_pred red.release.sys.global.add.s32 [$1], $2;
            mov.b32 $0, 0;
        }
        """,
        "=r,l,r,r",
        [ptr, value, store_mask.to(gl.int32)],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _load_packed_expert_state_acquire_if(expert_state_ptr, load_mask):
    """Issue a predicated system-acquire load without touching masked state."""
    return gl.inline_asm_elementwise(
        """
        {
            .reg .pred load_pred;
            setp.ne.u32 load_pred, $2, 0;
            mov.b64 $0, 0;
            @load_pred ld.global.sys.acquire.b64 $0, [$1];
        }
        """,
        "=l,l,r",
        [expert_state_ptr, load_mask.to(gl.int32)],
        dtype=gl.int64,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _peer_barrier_arrive_and_wait(
    peer_barrier_ptrs,
    barrier,
    world_size: gl.constexpr,
    expected,
    layout: gl.constexpr,
    participant_capacity: gl.constexpr,
):
    """Run DeepGEMM's single-counter NVLink barrier protocol.

    One lane release-reduces into each peer's counter.  Lane zero then waits
    until this rank's scalar has received ``world_size`` contributions for the
    requested phase.  The remaining vector slots stay unused so existing
    symmetric allocations remain ABI-compatible.
    """
    peer_offsets = gl.arange(0, participant_capacity, layout=layout)
    valid_peer = peer_offsets < world_size
    safe_peer_offsets = gl.where(valid_peer, peer_offsets, 0)
    remote_barriers = gl.load(
        peer_barrier_ptrs + safe_peer_offsets,
        mask=valid_peer,
        other=0,
    ).to(gl.pointer_type(gl.int32))
    _red_i32_release_sys_if(
        remote_barriers,
        gl.full([participant_capacity], 1, gl.int32, layout=layout),
        valid_peer,
    )

    leader = peer_offsets == 0
    local_counter_ptrs = barrier + peer_offsets * 0
    ready = _load_i32_acquire_sys_if(
        local_counter_ptrs,
        leader,
    )
    target = expected * world_size
    pending_mask = leader & (ready < target)
    pending = gl.sum(gl.where(pending_mask, 1, 0), axis=0)
    while pending != 0:
        fresh = _load_i32_acquire_sys_if(
            local_counter_ptrs,
            pending_mask,
        )
        ready = gl.where(pending_mask, fresh, ready)
        pending_mask = leader & (ready < target)
        pending = gl.sum(gl.where(pending_mask, 1, 0), axis=0)


@gluon.jit
def _packed_expert_count_from_cache(stored_counts, count_offsets, expert):
    """Select one expert count from a lane-distributed register snapshot."""
    return gl.max(
        gl.where(count_offsets == expert, stored_counts, 0),
        axis=0,
    )


@gluon.jit
def _scan_add_i32(lhs, rhs):
    """Associative int32 add used by one-warp prefix scans."""
    return lhs + rhs


@gluon.jit
def _wait_for_dispatch_handoff(
    dispatch_counter,
    ready_target: gl.constexpr,
):
    """Wait for the GPU-local release following the cross-rank barrier."""
    ready = _load_i32_acquire_gpu(dispatch_counter)
    while ready < ready_target:
        ready = _load_i32_acquire_gpu(dispatch_counter)


@gluon.jit
def _load_packed_expert_counts(
    expert_state,
    count_offsets,
    num_experts: gl.constexpr,
    world_size: gl.constexpr,
):
    """Acquire packed states once, reloading only entries that remain pending."""
    valid = count_offsets < num_experts
    safe_offsets = gl.where(valid, count_offsets, 0)
    packed = _load_packed_expert_state_acquire_if(
        expert_state + safe_offsets,
        valid,
    )
    contributors = packed // 4294967296
    pending_mask = valid & (contributors < world_size)
    pending = gl.sum(gl.where(pending_mask, 1, 0), axis=0)
    while pending != 0:
        fresh = _load_packed_expert_state_acquire_if(
            expert_state + safe_offsets,
            pending_mask,
        )
        packed = gl.where(pending_mask, fresh, packed)
        contributors = packed // 4294967296
        pending_mask = valid & (contributors < world_size)
        pending = gl.sum(gl.where(pending_mask, 1, 0), axis=0)
    return gl.where(
        valid,
        packed - contributors * 4294967296,
        0,
    ).to(gl.int32)


@gluon.jit
def scheduler_count(stored_counts, count_offsets, expert):
    """Select one cached expert count from a distributed register tensor."""
    return _packed_expert_count_from_cache(
        stored_counts,
        count_offsets,
        expert,
    )


@gluon.jit
def load_pool_block_offset(
    stored_counts,
    count_offsets,
    expert,
    block_m: gl.constexpr,
):
    """Return the BLOCK_M-padded pool prefix for ``expert``.

    This is the Gluon form of DeepGEMM's scheduler-local prefix reduction.
    It operates only on the cached count tensor and never reloads a host-built
    expert-offset table.
    """
    blocks = (stored_counts + block_m - 1) // block_m
    return gl.sum(
        gl.where(count_offsets < expert, blocks, 0),
        axis=0,
    )


@gluon.jit
def scheduler_next(
    stored_counts,
    count_offsets,
    block_idx,
    expert,
    phase,
    current_count,
    pool_block_offset,
    num_experts: gl.constexpr,
    l1_n_blocks: gl.constexpr,
    l2_n_blocks: gl.constexpr,
    num_experts_per_wave: gl.constexpr,
    num_sms: gl.constexpr,
    block_m: gl.constexpr,
):
    """Assign one FC1/FC2 tile using DeepGEMM's expert-wave state machine.

    Phase 1 walks every FC1 tile in the current expert wave.  The same
    persistent CTA cursor is then rewound to the wave beginning for phase 2,
    so FC2 can consume any pool block whose FC1 publication counter is ready.
    All A, B, and math partitions call this helper with identical state.
    """
    found = False
    task_phase = 0
    task_expert = 0
    task_m_block = 0
    task_n_block = 0
    task_pool_block = 0
    task_valid_count = 0

    while expert < num_experts and not found:
        wave_end = gl.minimum(
            ((expert + 1 + num_experts_per_wave - 1) // num_experts_per_wave)
            * num_experts_per_wave,
            num_experts,
        )
        if phase == 1:
            while expert < wave_end and not found:
                num_m_blocks = (current_count + block_m - 1) // block_m
                m_block = block_idx // l1_n_blocks
                if m_block < num_m_blocks:
                    task_phase = 1
                    task_expert = expert
                    task_m_block = m_block
                    task_n_block = block_idx - m_block * l1_n_blocks
                    task_pool_block = pool_block_offset + m_block
                    task_valid_count = current_count
                    block_idx += num_sms
                    found = True
                else:
                    block_idx -= num_m_blocks * l1_n_blocks
                    pool_block_offset += num_m_blocks
                    expert += 1
                    current_count = scheduler_count(
                        stored_counts,
                        count_offsets,
                        expert,
                    )
            if not found:
                phase = 2
                expert = ((expert - 1) // num_experts_per_wave) * num_experts_per_wave
                current_count = scheduler_count(
                    stored_counts,
                    count_offsets,
                    expert,
                )
                pool_block_offset = load_pool_block_offset(
                    stored_counts,
                    count_offsets,
                    expert,
                    block_m,
                )
        else:
            while expert < wave_end and not found:
                num_m_blocks = (current_count + block_m - 1) // block_m
                expert_tasks = num_m_blocks * l2_n_blocks
                if block_idx < expert_tasks:
                    task_phase = 2
                    task_expert = expert
                    task_m_block = block_idx // l2_n_blocks
                    task_n_block = block_idx - task_m_block * l2_n_blocks
                    task_pool_block = pool_block_offset + task_m_block
                    task_valid_count = current_count
                    block_idx += num_sms
                    found = True
                else:
                    block_idx -= expert_tasks
                    pool_block_offset += num_m_blocks
                    expert += 1
                    current_count = scheduler_count(
                        stored_counts,
                        count_offsets,
                        expert,
                    )
            if not found:
                phase = 1

    return (
        task_phase,
        task_expert,
        task_m_block,
        task_n_block,
        task_pool_block,
        task_valid_count,
        block_idx,
        expert,
        phase,
        current_count,
        pool_block_offset,
    )


# fp8/types.py


@dataclass(frozen=True)
class FusedWorkspace:
    """Reusable storage for the complete fused MegaMoE path."""

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
    gemm_descriptor_sets: tuple[GemmDescriptors, ...]
    dispatch_descs: tuple[object, ...]
    l1_weight_data_ptr: int
    l2_weight_data_ptr: int
    max_pool_blocks: int
    num_pool_rows: int
    num_padded_sf_pool_tokens: int


@dataclass(frozen=True)
class FusedResult:
    """Artifacts from the complete registered MegaMoE chain.

    ``output`` is the observable token-major BF16 result after peer FC2 scatter
    and source-local top-k reduction.
    In ``combine_buffer``, only slots whose registered top-k index is valid
    are defined; invalid slots are deliberately not cleared or reduced.
    """

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
    workspace: FusedWorkspace
    pre_dispatch: RegisteredInputs
    config: LaunchConfig
    compiled: object


# reference.py


def _interleave_fc1_weights(weight: torch.Tensor, granularity: int = 8) -> torch.Tensor:
    """Interleave gate and up rows for the fused FC1 weight descriptor."""
    if weight.ndim < 2 or weight.shape[1] % (2 * granularity):
        raise ValueError("L1 N dimension must be divisible by 2 * granularity")
    groups, n, *rest = weight.shape
    half = n // 2
    gate = weight[:, :half].reshape(groups, half // granularity, granularity, *rest)
    up = weight[:, half:].reshape(groups, half // granularity, granularity, *rest)
    return torch.stack((gate, up), dim=2).reshape_as(weight).contiguous()


# fp8/weights.py


def prepare_weights(weight, scale):
    """Interleave quantized FC1 gate/up weights; preserve their block scales.

    ``weight`` has shape ``[local_experts, 2 * intermediate, hidden]``.
    FC2 weights retain their ordinary expert-major layout.
    """

    return _interleave_fc1_weights(weight), scale


# fp8/math_common.py


@gluon.jit
def _silu(value, fast_math: gl.constexpr):
    """SM90 SwiGLU sigmoid factor used by the Gluon FC1 epilogue."""
    exponent = gl.exp(-value)
    if fast_math:
        reciprocal = gl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            [1.0 + exponent],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        reciprocal = gl.fdiv(1.0, 1.0 + exponent)
    return value * reciprocal


@gluon.jit
def _reciprocal(value, fast_math: gl.constexpr):
    if fast_math:
        return gl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            [value],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
    return gl.fdiv(1.0, value)


@gluon.jit
def _release_stage(stage_empty, stage):
    # WGMMA completion protects its async A/B reads, not the generic shared
    # loads of SFA. Order those loads before the producers' next TMA overwrite,
    # then join every math warp before the single stage-empty arrival.
    fence_async_shared()
    partition_barrier()
    mbarrier.arrive(stage_empty.index(stage), count=1)


@gluon.jit
def _tile_complete(
    state, expert, pool_block, m_blocks, fragments_per_block: gl.constexpr
):
    """Default completion policy: the caller owns the final scatter barrier."""
    pass


# fp8/pipeline.py


@gluon.jit
def allocate_math_pipeline(
    l1_a_desc,
    l1_b_desc,
    l2_store_desc,
    num_stages: gl.constexpr,
    block_m: gl.constexpr,
    partitions: gl.constexpr,
    scale_exchange: gl.constexpr,
):
    a = gl.allocate_shared_memory(
        l1_a_desc.dtype, [num_stages] + l1_a_desc.block_type.shape, l1_a_desc.layout
    )
    b = gl.allocate_shared_memory(
        l1_b_desc.dtype, [num_stages] + l1_b_desc.block_type.shape, l1_b_desc.layout
    )
    sf_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [block_m], gl.float32
    )
    sf_lo = gl.allocate_shared_memory(gl.float32, [num_stages, block_m], sf_layout)
    sf_hi = gl.allocate_shared_memory(gl.float32, [num_stages, block_m], sf_layout)
    epilogue0 = gl.allocate_shared_memory(
        l2_store_desc.dtype, l2_store_desc.block_type.shape, l2_store_desc.layout
    )
    # Unused slots alias live descriptors: JIT returns cannot contain None.
    # The constexpr role selection never consumes these aliases and they
    # allocate no additional storage or barriers.
    epilogue1 = epilogue0
    epilogue2 = epilogue0
    epilogue3 = epilogue0
    if partitions >= 2:
        epilogue1 = gl.allocate_shared_memory(
            l2_store_desc.dtype, l2_store_desc.block_type.shape, l2_store_desc.layout
        )
    if partitions == 4:
        epilogue2 = gl.allocate_shared_memory(
            l2_store_desc.dtype, l2_store_desc.block_type.shape, l2_store_desc.layout
        )
        epilogue3 = gl.allocate_shared_memory(
            l2_store_desc.dtype, l2_store_desc.block_type.shape, l2_store_desc.layout
        )
    barrier_layout: gl.constexpr = mbarrier.MBarrierLayout()
    empty = gl.allocate_shared_memory(gl.int64, [num_stages, 1], barrier_layout)
    ready = gl.allocate_shared_memory(gl.int64, [num_stages, 1], barrier_layout)
    amax0 = sf_lo.index(0)
    amax1 = sf_lo.index(0)
    scale_ready = empty.index(0)
    scale_done = empty.index(0)
    if scale_exchange:
        amax_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
            [64], gl.float32
        )
        amax0 = gl.allocate_shared_memory(gl.float32, [64], amax_layout)
        amax1 = gl.allocate_shared_memory(gl.float32, [64], amax_layout)
        scale_ready = gl.allocate_shared_memory(gl.int64, [1], barrier_layout)
        scale_done = gl.allocate_shared_memory(gl.int64, [1], barrier_layout)
        mbarrier.init(scale_ready, count=2)
        mbarrier.init(scale_done, count=2)
    for stage in gl.static_range(num_stages):
        mbarrier.init(empty.index(stage), count=partitions)
        mbarrier.init(ready.index(stage), count=_PRODUCERS_CONSTEXPR)
    math_done = empty.index(0)
    if partitions >= 2:
        math_done = gl.allocate_shared_memory(gl.int64, [1], barrier_layout)
        mbarrier.init(math_done, count=partitions)
    return (
        a,
        b,
        sf_lo,
        sf_hi,
        epilogue0,
        epilogue1,
        epilogue2,
        epilogue3,
        empty,
        ready,
        amax0,
        amax1,
        scale_ready,
        scale_done,
        math_done,
    )


@gluon.jit
def allocate_dispatch_pipeline(descriptor):
    first = gl.allocate_shared_memory(
        descriptor.dtype, descriptor.block_type.shape, descriptor.layout
    )
    second = gl.allocate_shared_memory(
        descriptor.dtype, descriptor.block_type.shape, descriptor.layout
    )
    layout: gl.constexpr = mbarrier.MBarrierLayout()
    first_ready = gl.allocate_shared_memory(gl.int64, [1], layout)
    second_ready = gl.allocate_shared_memory(gl.int64, [1], layout)
    mbarrier.init(first_ready, count=1)
    mbarrier.init(second_ready, count=1)
    return first, second, first_ready, second_ready


# fp8/dispatch.py


@gluon.jit
def _reserve_routes(
    input_topk_idx,
    expert_send_state,
    peer_source_routes_ptrs,
    num_routes,
    dispatch_pid,
    num_global_dispatch_workers: gl.constexpr,
    rank: gl.constexpr,
    experts_per_rank: gl.constexpr,
    max_routes: gl.constexpr,
    first_target_rank: gl.constexpr,
    num_target_ranks: gl.constexpr,
):
    """Reserve source-local slots and publish routes to a contiguous peer range."""
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
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
        valid_route = (
            valid_route
            & (target_rank >= first_target_rank)
            & (target_rank < first_target_rank + num_target_ranks)
        )
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


@gluon.jit
def _select_source_route(
    symmetric_recv_count,
    symmetric_source_routes,
    current_expert,
    local_row,
    experts_per_rank: gl.constexpr,
    max_routes: gl.constexpr,
    world_size: gl.constexpr,
):
    """Round-robin active sources using the source-local reservation order."""
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    source_rank_offsets = gl.arange(0, 32, layout=layout)
    valid_source_rank = source_rank_offsets < world_size
    source_rank_counts = gl.load(
        symmetric_recv_count + source_rank_offsets * experts_per_rank + current_expert,
        mask=valid_source_rank,
        other=0,
    )
    selected_rank = 0
    slot = local_row
    round_offset = 0
    token_idx_in_rank = 0
    found_route = False
    while not found_route:
        remaining = gl.maximum(source_rank_counts - round_offset, 0)
        is_active = valid_source_rank & (remaining > 0)
        active_i32 = is_active.to(gl.int32)
        num_active = gl.sum(active_i32, axis=0)
        round_length = gl.min(
            gl.where(is_active, remaining, 0x7FFFFFFF),
            axis=0,
        )
        round_tokens = round_length * num_active
        if slot < round_tokens:
            desired_rank = slot - (slot // num_active) * num_active
            active_prefix = gl.associative_scan(
                active_i32,
                axis=0,
                combine_fn=_scan_add_i32,
            )
            choose = is_active & (active_prefix == desired_rank + 1)
            selected_rank = gl.sum(
                gl.where(choose, source_rank_offsets, 0),
                axis=0,
            )
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
    return selected_rank, route


@gluon.jit
def _pull_dispatch_row(
    selected_rank,
    route,
    pool_row,
    tma_phase,
    peer_input_sf_ptrs,
    peer_input_topk_weights_ptrs,
    peer_acts_descs,
    pool_acts_desc,
    tma_load_barriers,
    pull_buffer,
    pool_acts_sf,
    pool_topk_weights,
    token_src_metadata,
    l1_arrival,
    num_padded_sf_pool_tokens: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    block_m: gl.constexpr,
    world_size: gl.constexpr,
):
    """Copy one registered source row and release its pool arrival."""
    sf_groups: gl.constexpr = hidden // 128
    sf_per_lane: gl.constexpr = (sf_groups + 31) // 32
    sf_capacity: gl.constexpr = sf_per_lane * 32
    sf_layout: gl.constexpr = gl.BlockedLayout([sf_per_lane], [32], [1], [0])
    sf_offsets = gl.arange(0, sf_capacity, layout=sf_layout)
    sf_block_m: gl.constexpr = (block_m + 127) // 128 * 128
    remote_sf = gl.load(peer_input_sf_ptrs + selected_rank).to(
        gl.pointer_type(gl.float32)
    )
    remote_weights = gl.load(peer_input_topk_weights_ptrs + selected_rank).to(
        gl.pointer_type(gl.float32)
    )
    source_token = route // topk
    source_slot = route - source_token * topk
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
        l1_arrival + pool_block,
        1,
        sem="release",
        scope="gpu",
    )


@gluon.jit
def dispatch_partition(
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
    staged_dispatch_handoff: gl.constexpr,
):
    """Run one DeepGEMM-style independent dispatch-warp token stream."""
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
    dispatch_pid = gl.program_id(0) * num_dispatch_workers + dispatch_worker
    num_global_dispatch_workers: gl.constexpr = num_sms * num_dispatch_workers

    # DeepGEMM first reserves source-local per-expert slots, then uses
    # ordinary remote stores.  Gluon's shared-memory descriptor does
    # not expose block-scope atomics, so reserve directly in this
    # rank's local L2 rather than issuing one system-scope atomic per
    # route on the destination rank.  The old value is the unique
    # source-rank slot later consumed by the remote expert.
    _reserve_routes(
        input_topk_idx,
        expert_send_state,
        peer_source_routes_ptrs,
        num_routes,
        dispatch_pid,
        num_global_dispatch_workers,
        rank,
        experts_per_rank,
        max_routes,
        0,
        world_size,
    )
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
        if staged_dispatch_handoff:
            # Match DeepGEMM's second local handoff: the leader's system
            # acquire at the peer barrier is made visible to every resident
            # partition through one GPU-scope release.
            gl.atomic_add(
                dispatch_counter,
                1,
                sem="release",
                scope="gpu",
            )

    if staged_dispatch_handoff:
        _wait_for_dispatch_handoff(
            dispatch_counter,
            2 * num_global_dispatch_workers + 1,
        )

    # Snapshot every expert count once per dispatch warp.  The one-warp
    # layout lets lanes fetch experts in parallel; later token traversal
    # selects counts from registers instead of issuing a sequential chain of
    # system-scope loads for every worker.
    dispatch_count_capacity: gl.constexpr = max(triton.next_power_of_2(num_experts), 32)
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
        pool_row // block_m

        selected_rank, route = _select_source_route(
            symmetric_recv_count,
            symmetric_source_routes,
            current_expert,
            local_row,
            experts_per_rank,
            max_routes,
            world_size,
        )
        _pull_dispatch_row(
            selected_rank,
            route,
            pool_row,
            (dispatch_token // num_global_dispatch_workers) & 1,
            peer_input_sf_ptrs,
            peer_input_topk_weights_ptrs,
            peer_acts_descs,
            pool_acts_desc,
            tma_load_barriers,
            pull_buffer,
            pool_acts_sf,
            pool_topk_weights,
            token_src_metadata,
            task_state[1],
            num_padded_sf_pool_tokens,
            hidden,
            topk,
            block_m,
            world_size,
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


# fp8/producers.py


@gluon.jit
def a_producer_partition(
    l1_a_desc,
    l1_sfa_desc,
    l2_a_desc,
    l2_sfa_desc,
    expert_state,
    dispatch_counter,
    l1_arrival,
    l2_arrival,
    barriers,
    buffers,
    E: gl.constexpr,
    l1_k: gl.constexpr,
    l2_k: gl.constexpr,
    l1_n_blocks: gl.constexpr,
    l2_n_blocks: gl.constexpr,
    num_experts_per_wave: gl.constexpr,
    num_sms: gl.constexpr,
    scheduler_count_capacity: gl.constexpr,
    scheduler_counts_per_lane: gl.constexpr,
    num_padded_sf_pool_tokens: gl.constexpr,
    world_size: gl.constexpr,
    fc2_arrival_counter: gl.constexpr,
    staged_dispatch_handoff: gl.constexpr,
):
    """Phase-aware A producer for FC1 per-128 and FC2 per-64 SFA."""
    stage_empty, stage_ready = barriers
    a_buffers, sfa_lo_buffers, sfa_hi_buffers = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_m: gl.constexpr = a_buffers.type.shape[1]
    block_k: gl.constexpr = a_buffers.type.shape[2]

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
    if staged_dispatch_handoff:
        _wait_for_dispatch_handoff(dispatch_counter, 4 * num_sms + 1)
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
        block_m,
    )

    while task_phase != 0:
        local_row = task_m_block * block_m
        flat_row_start = task_pool_block * block_m
        scale_row_start = task_pool_block * _SF_BLOCK_M
        if task_phase == 1:
            expected = gl.minimum(block_m, valid_count - local_row)
            ready = _load_i32_acquire_gpu(l1_arrival + task_pool_block)
            while ready < expected:
                ready = _load_i32_acquire_gpu(l1_arrival + task_pool_block)
        if task_phase == 2:
            ready = _load_i32_acquire_gpu(l2_arrival + task_pool_block)
            expected_l2_arrivals = l1_n_blocks
            if fc2_arrival_counter:
                active_m_wgs = (gl.minimum(block_m, valid_count - local_row) + 63) // 64
                expected_l2_arrivals *= active_m_wgs * 2
            while ready < expected_l2_arrivals:
                ready = _load_i32_acquire_gpu(l2_arrival + task_pool_block)

        num_k_tiles = gl.where(
            task_phase == 1,
            l1_k // block_k,
            l2_k // block_k,
        )
        k_tile = 0
        while k_tile < num_k_tiles:
            stage = pipeline_tile % num_stages
            pipe_phase = pipeline_tile // num_stages & 1
            mbarrier.wait(stage_empty.index(stage), pipe_phase ^ 1)

            if task_phase == 1:
                mbarrier.expect(
                    stage_ready.index(stage),
                    l1_a_desc.block_type.nbytes + l1_sfa_desc.block_type.nbytes,
                )
                tma.async_copy_global_to_shared(
                    l1_a_desc,
                    [flat_row_start, k_tile * block_k],
                    stage_ready.index(stage),
                    a_buffers.index(stage),
                )
                tma.async_copy_global_to_shared(
                    l1_sfa_desc,
                    [k_tile * num_padded_sf_pool_tokens + scale_row_start],
                    stage_ready.index(stage),
                    sfa_lo_buffers.index(stage),
                )
            else:
                mbarrier.expect(
                    stage_ready.index(stage),
                    l2_a_desc.block_type.nbytes + 2 * l2_sfa_desc.block_type.nbytes,
                )
                tma.async_copy_global_to_shared(
                    l2_a_desc,
                    [flat_row_start, k_tile * block_k],
                    stage_ready.index(stage),
                    a_buffers.index(stage),
                )
                tma.async_copy_global_to_shared(
                    l2_sfa_desc,
                    [(k_tile * 2) * num_padded_sf_pool_tokens + scale_row_start],
                    stage_ready.index(stage),
                    sfa_lo_buffers.index(stage),
                )
                tma.async_copy_global_to_shared(
                    l2_sfa_desc,
                    [(k_tile * 2 + 1) * num_padded_sf_pool_tokens + scale_row_start],
                    stage_ready.index(stage),
                    sfa_hi_buffers.index(stage),
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
            block_m,
        )


@gluon.jit
def b_producer_partition(
    l1_b_desc,
    l2_b_desc,
    expert_state,
    dispatch_counter,
    barriers,
    b_buffers,
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
    block_m: gl.constexpr,
    staged_dispatch_handoff: gl.constexpr,
):
    """Phase-aware B producer selecting the FC1 or FC2 descriptor."""
    stage_empty, stage_ready = barriers
    num_stages: gl.constexpr = b_buffers.type.shape[0]
    block_n: gl.constexpr = b_buffers.type.shape[1]
    block_k: gl.constexpr = b_buffers.type.shape[2]

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
    if staged_dispatch_handoff:
        _wait_for_dispatch_handoff(dispatch_counter, 4 * num_sms + 1)
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
        block_m,
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
                mbarrier.expect(
                    stage_ready.index(stage),
                    l1_b_desc.block_type.nbytes,
                )
                tma.async_copy_global_to_shared(
                    l1_b_desc,
                    [flat_b_row_start, k_tile * block_k],
                    stage_ready.index(stage),
                    b_buffers.index(stage),
                )
            else:
                mbarrier.expect(
                    stage_ready.index(stage),
                    l2_b_desc.block_type.nbytes,
                )
                tma.async_copy_global_to_shared(
                    l2_b_desc,
                    [flat_b_row_start, k_tile * block_k],
                    stage_ready.index(stage),
                    b_buffers.index(stage),
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
            block_m,
        )


# fp8/epilogues.py


@gluon.jit
def _fc2_swap_bf16_epilogue(
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
    """Scatter fused swapAB output directly into source-rank slots."""
    swap_mma_layout: gl.constexpr = final.type.layout
    channel_offsets = gl.arange(
        0,
        block_n,
        layout=gl.SliceLayout(1, swap_mma_layout),
    )
    token_offsets = gl.arange(
        0,
        n_swap,
        layout=gl.SliceLayout(0, swap_mma_layout),
    )
    output_cols = task_n_block * block_n + channel_offsets
    output_rows = task_pool_block * block_m + token_offsets
    converted = final.to(gl.bfloat16)
    valid_tokens = token_offsets < valid_m
    source_rank = gl.load(
        token_src_metadata + output_rows * 3,
        mask=valid_tokens,
        other=-1,
    )
    source_token = gl.load(
        token_src_metadata + output_rows * 3 + 1,
        mask=valid_tokens,
        other=0,
    )
    source_slot = gl.load(
        token_src_metadata + output_rows * 3 + 2,
        mask=valid_tokens,
        other=0,
    )
    safe_source_rank = gl.maximum(
        gl.minimum(source_rank, world_size - 1),
        0,
    )
    remote_combine = gl.load(peer_combine_buffer_ptrs + safe_source_rank).to(
        gl.pointer_type(gl.bfloat16)
    )
    scatter_ptrs = (
        remote_combine[None, :]
        + (source_token[None, :] * topk + source_slot[None, :]) * l2_n
        + output_cols[:, None]
    )
    gl.store(
        scatter_ptrs,
        converted,
        mask=(
            valid_tokens[None, :]
            & (source_rank[None, :] >= 0)
            & (source_rank[None, :] < world_size)
            & (source_token[None, :] >= 0)
            & (source_token[None, :] < max_tokens)
            & (source_slot[None, :] >= 0)
            & (source_slot[None, :] < topk)
            & (output_cols[:, None] < l2_n)
        ),
    )


@gluon.jit
def _fc1_bm64_bn256_split_epilogue(
    final,
    l2_store_desc,
    l2_epilogue_buffer,
    l2_acts_sf,
    route_weights,
    l2_arrival,
    task_pool_block,
    task_n_block,
    valid_m,
    num_padded_sf_pool_tokens: gl.constexpr,
    activation_clamp: gl.constexpr,
    has_activation_clamp: gl.constexpr,
    fast_math: gl.constexpr,
    math_partition_idx: gl.constexpr,
):
    """Publish one independent 64x64 FC1 slice of BM64/BN256."""
    fragment_m: gl.constexpr = 64
    fragment_n: gl.constexpr = 128

    paired = final.reshape((fragment_m, fragment_n // 16, 2, 8)).permute((0, 1, 3, 2))
    gate, up = gl.split(paired)
    gate = gate.reshape((fragment_m, fragment_n // 2))
    up = up.reshape((fragment_m, fragment_n // 2))
    row_layout: gl.constexpr = gl.SliceLayout(1, gate.type.layout)
    row_offsets = gl.arange(0, fragment_m, layout=row_layout)

    if has_activation_clamp:
        gate = gl.minimum(gate, activation_clamp)
        up = gl.minimum(gl.maximum(up, -activation_clamp), activation_clamp)
    swiglu = _silu(gate, fast_math) * up
    valid_rows = row_offsets < valid_m
    weight = gl.load(
        route_weights + task_pool_block * fragment_m + row_offsets,
        mask=valid_rows,
        other=0.0,
    )
    activation = swiglu * weight[:, None]
    amax = gl.max(gl.abs(activation), axis=1)
    scale = gl.maximum(amax, 1.0e-10) * (1.0 / 448.0)
    quantized = (activation * _reciprocal(scale[:, None], fast_math)).to(gl.float8e4nv)

    sf_pool_rows = task_pool_block * _SF_BLOCK_M + row_offsets
    sf_group = task_n_block * 2 + math_partition_idx
    gl.store(
        l2_acts_sf + sf_group * num_padded_sf_pool_tokens + sf_pool_rows,
        scale,
        mask=valid_rows,
    )

    l2_epilogue_buffer.store(quantized)
    partition_barrier()
    fence_async_shared()
    tma.async_copy_shared_to_global(
        l2_store_desc,
        [
            task_pool_block * fragment_m,
            task_n_block * 128 + math_partition_idx * 64,
        ],
        l2_epilogue_buffer,
    )
    tma.store_wait(0)
    gl.atomic_add(
        l2_arrival + task_pool_block,
        1,
        sem="release",
        scope="gpu",
    )
    partition_barrier()


@gluon.jit
def _fc1_bm64_bn128_split_epilogue(
    final,
    l2_store_desc,
    l2_epilogue_buffer,
    fc1_amax_scratch_0,
    fc1_amax_scratch_1,
    fc1_scale_ready,
    fc1_scale_done,
    l2_acts_sf,
    route_weights,
    l2_arrival,
    task_pool_block,
    task_n_block,
    valid_m,
    num_padded_sf_pool_tokens: gl.constexpr,
    activation_clamp: gl.constexpr,
    has_activation_clamp: gl.constexpr,
    fast_math: gl.constexpr,
    math_partition_idx: gl.constexpr,
    sync_phase,
):
    """Publish one 64x32 slice with a shared per-64 activation scale."""
    fragment_m: gl.constexpr = 64
    fragment_n: gl.constexpr = 64

    paired = final.reshape((fragment_m, fragment_n // 16, 2, 8)).permute((0, 1, 3, 2))
    gate, up = gl.split(paired)
    gate = gate.reshape((fragment_m, fragment_n // 2))
    up = up.reshape((fragment_m, fragment_n // 2))
    row_layout: gl.constexpr = gl.SliceLayout(1, gate.type.layout)
    row_offsets = gl.arange(0, fragment_m, layout=row_layout)

    if has_activation_clamp:
        gate = gl.minimum(gate, activation_clamp)
        up = gl.minimum(gl.maximum(up, -activation_clamp), activation_clamp)
    swiglu = _silu(gate, fast_math) * up
    valid_rows = row_offsets < valid_m
    weight = gl.load(
        route_weights + task_pool_block * fragment_m + row_offsets,
        mask=valid_rows,
        other=0.0,
    )
    activation = swiglu * weight[:, None]
    local_amax = gl.max(gl.abs(activation), axis=1)
    if math_partition_idx == 0:
        fc1_amax_scratch_0.store(local_amax)
    else:
        fc1_amax_scratch_1.store(local_amax)
    partition_barrier()
    mbarrier.arrive(fc1_scale_ready, count=1)
    mbarrier.wait(fc1_scale_ready, sync_phase)

    amax = gl.maximum(
        fc1_amax_scratch_0.load(row_layout),
        fc1_amax_scratch_1.load(row_layout),
    )
    scale = gl.maximum(amax, 1.0e-10) * (1.0 / 448.0)
    quantized = (activation * _reciprocal(scale[:, None], fast_math)).to(gl.float8e4nv)

    if math_partition_idx == 0:
        sf_pool_rows = task_pool_block * _SF_BLOCK_M + row_offsets
        gl.store(
            l2_acts_sf + task_n_block * num_padded_sf_pool_tokens + sf_pool_rows,
            scale,
            mask=valid_rows,
        )

    l2_epilogue_buffer.store(quantized)
    partition_barrier()
    fence_async_shared()
    tma.async_copy_shared_to_global(
        l2_store_desc,
        [
            task_pool_block * fragment_m,
            task_n_block * 64 + math_partition_idx * 32,
        ],
        l2_epilogue_buffer,
    )
    tma.store_wait(0)
    gl.atomic_add(
        l2_arrival + task_pool_block,
        1,
        sem="release",
        scope="gpu",
    )
    partition_barrier()
    mbarrier.arrive(fc1_scale_done, count=1)
    mbarrier.wait(fc1_scale_done, sync_phase)


@gluon.jit
def _fc2_bf16_scatter_epilogue(
    output,
    row_offsets,
    col_offsets,
    token_src_metadata,
    peer_combine_buffer_ptrs,
    task_pool_block,
    task_n_block,
    valid_m,
    l2_n: gl.constexpr,
    max_tokens: gl.constexpr,
    topk: gl.constexpr,
    world_size: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
):
    """Vector-scatter a normal-layout FC2 BF16 tile to source ranks."""
    output_rows = task_pool_block * block_m + row_offsets[:, None]
    output_cols = task_n_block * block_n + col_offsets[None, :]
    row_mask = row_offsets[:, None] < valid_m
    source_rank = gl.load(
        token_src_metadata + output_rows * 3,
        mask=row_mask,
        other=-1,
    )
    source_token = gl.load(
        token_src_metadata + output_rows * 3 + 1,
        mask=row_mask,
        other=0,
    )
    source_slot = gl.load(
        token_src_metadata + output_rows * 3 + 2,
        mask=row_mask,
        other=0,
    )
    safe_source_rank = gl.maximum(
        gl.minimum(source_rank, world_size - 1),
        0,
    )
    remote_combine = gl.load(peer_combine_buffer_ptrs + safe_source_rank).to(
        gl.pointer_type(gl.bfloat16)
    )
    scatter_ptrs = (
        remote_combine + (source_token * topk + source_slot) * l2_n + output_cols
    )
    _store_contiguous_bf16_fragment(
        scatter_ptrs,
        output,
        (
            row_mask
            & (source_rank >= 0)
            & (source_rank < world_size)
            & (source_token >= 0)
            & (source_token < max_tokens)
            & (source_slot >= 0)
            & (source_slot < topk)
            & (output_cols < l2_n)
        ),
        4,
    )


@gluon.jit
def _fc1_epilogue(
    final,
    l2_store_desc,
    l2_epilogue_buffer,
    l2_acts_sf,
    route_weights,
    l2_arrival,
    task_pool_block,
    task_n_block,
    valid_m,
    num_padded_sf_pool_tokens: gl.constexpr,
    activation_clamp: gl.constexpr,
    has_activation_clamp: gl.constexpr,
    fast_math: gl.constexpr,
    use_swap_ab: gl.constexpr,
    n_swap: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    fc2_arrival_counter: gl.constexpr,
):
    """Publish one FC1 tile with independent per-row/per-64 FP32 SF.

    The complete math partition owns one output-64 scale domain.  Reducing
    the reshaped tensor along that domain therefore includes both N-split
    warpgroups and produces exactly one publisher for this FC1 N block.
    """
    if use_swap_ab:
        num_rows: gl.constexpr = n_swap
        paired = final.reshape((block_n // 16, 2, 8, n_swap)).permute((3, 0, 2, 1))
        gate, up = gl.split(paired)
        gate = gate.reshape((n_swap, block_n // 2))
        up = up.reshape((n_swap, block_n // 2))
        # ``final`` is [N, M] in swapAB, but ``gate`` has already been
        # permuted back to logical [M, N/2].  Derive the row vector from the
        # post-permute layout so ``weight[:, None]`` can restore axis 1.
        row_layout: gl.constexpr = gl.SliceLayout(1, gate.type.layout)
        row_offsets = gl.arange(0, n_swap, layout=row_layout)
    else:
        num_rows: gl.constexpr = block_m
        paired = final.reshape((block_m, block_n // 16, 2, 8)).permute((0, 1, 3, 2))
        gate, up = gl.split(paired)
        gate = gate.reshape((block_m, block_n // 2))
        up = up.reshape((block_m, block_n // 2))
        row_layout: gl.constexpr = gl.SliceLayout(1, gate.type.layout)
        row_offsets = gl.arange(
            0,
            block_m,
            layout=row_layout,
        )

    if has_activation_clamp:
        gate = gl.minimum(gate, activation_clamp)
        up = gl.minimum(gl.maximum(up, -activation_clamp), activation_clamp)
    swiglu = _silu(gate, fast_math) * up
    valid_rows = row_offsets < valid_m
    weight = gl.load(
        route_weights + task_pool_block * block_m + row_offsets,
        mask=valid_rows,
        other=0.0,
    )
    activation = swiglu * weight[:, None]
    if block_n == 128:
        # Keep the original one-dimensional scale domain for BN128.  Gluon
        # cannot broadcast the SliceLayout produced by a degenerate
        # [rows, 1, 64] reduction back to the row layout.
        amax = gl.max(gl.abs(activation), axis=1)
        scale = gl.maximum(amax, 1.0e-10) * (1.0 / 448.0)
        quantized = (activation * _reciprocal(scale[:, None], fast_math)).to(
            gl.float8e4nv
        )
        sf_pool_rows = task_pool_block * _SF_BLOCK_M + row_offsets
        gl.store(
            l2_acts_sf + task_n_block * num_padded_sf_pool_tokens + sf_pool_rows,
            scale,
            mask=valid_rows,
        )
    else:
        # Each N-split warpgroup owns one 64-column activation-scale domain.
        # Preserve that domain as an explicit axis so the reduction and store
        # stay local to the owning warpgroup without a split/join redistribution.
        num_scale_groups: gl.constexpr = block_n // 128
        activation_groups = activation.reshape((num_rows, num_scale_groups, 64))
        if n_swap == 128:
            activation_group_layout: gl.constexpr = gl.BlockedLayout(
                [1, 1, 1],
                [1, 1, 32],
                [8, 2, 1],
                [2, 1, 0],
            )
            activation_groups = gl.convert_layout(
                activation_groups,
                activation_group_layout,
            )
        amax_groups = gl.max(gl.abs(activation_groups), axis=2)
        scale_groups = gl.maximum(amax_groups, 1.0e-10) * (1.0 / 448.0)
        quantized = (
            (activation_groups * _reciprocal(scale_groups[:, :, None], fast_math))
            .to(gl.float8e4nv)
            .reshape((num_rows, block_n // 2))
        )

        scale_row_layout: gl.constexpr = gl.SliceLayout(
            1,
            scale_groups.type.layout,
        )
        scale_group_layout: gl.constexpr = gl.SliceLayout(
            0,
            scale_groups.type.layout,
        )
        scale_rows = gl.arange(0, num_rows, layout=scale_row_layout)
        scale_group_offsets = gl.arange(
            0,
            num_scale_groups,
            layout=scale_group_layout,
        )
        scale_pool_rows = task_pool_block * _SF_BLOCK_M + scale_rows[:, None]
        scale_pool_groups = task_n_block * 2 + scale_group_offsets[None, :]
        gl.store(
            l2_acts_sf
            + scale_pool_groups * num_padded_sf_pool_tokens
            + scale_pool_rows,
            scale_groups,
            mask=scale_rows[:, None] < valid_m,
        )

    if use_swap_ab:
        l2_epilogue_buffer.slice(0, n_swap, dim=0).store(quantized)
    else:
        l2_epilogue_buffer.store(quantized)
    partition_barrier()
    fence_async_shared()
    tma.async_copy_shared_to_global(
        l2_store_desc,
        [
            task_pool_block * block_m,
            task_n_block * (block_n // 2),
        ],
        l2_epilogue_buffer,
    )
    tma.store_wait(0)
    # No math warp may reuse the joint shared tile until its single combined
    # TMA store has drained.  The release counter orders both payload and SF.
    partition_barrier()
    arrival_count = 1
    if fc2_arrival_counter:
        active_m_wgs = (valid_m + 63) // 64
        arrival_count = active_m_wgs * (block_n // 128)
    gl.atomic_add(
        l2_arrival + task_pool_block,
        arrival_count,
        sem="release",
        scope="gpu",
    )
    partition_barrier()


@gluon.jit
def _fc1_bm128_bn256_split_epilogue(
    final,
    l2_store_desc,
    l2_epilogue_buffer,
    l2_acts_sf,
    route_weights,
    l2_arrival,
    task_pool_block,
    task_n_block,
    valid_m,
    num_padded_sf_pool_tokens: gl.constexpr,
    activation_clamp: gl.constexpr,
    has_activation_clamp: gl.constexpr,
    fast_math: gl.constexpr,
    math_partition_idx: gl.constexpr,
):
    """Publish one 64x64 FC1 slice from a BM128/BN256 split partition."""
    fragment_m: gl.constexpr = 64
    fragment_n: gl.constexpr = 128
    wg_m: gl.constexpr = math_partition_idx // 2
    wg_n: gl.constexpr = math_partition_idx % 2

    paired = final.reshape((fragment_m, fragment_n // 16, 2, 8)).permute((0, 1, 3, 2))
    gate, up = gl.split(paired)
    gate = gate.reshape((fragment_m, fragment_n // 2))
    up = up.reshape((fragment_m, fragment_n // 2))
    row_layout: gl.constexpr = gl.SliceLayout(1, gate.type.layout)
    local_rows = gl.arange(0, fragment_m, layout=row_layout)
    logical_rows = wg_m * fragment_m + local_rows

    if has_activation_clamp:
        gate = gl.minimum(gate, activation_clamp)
        up = gl.minimum(gl.maximum(up, -activation_clamp), activation_clamp)
    swiglu = _silu(gate, fast_math) * up
    valid_rows = logical_rows < valid_m
    weight = gl.load(
        route_weights + task_pool_block * _SPLIT_BLOCK_M + logical_rows,
        mask=valid_rows,
        other=0.0,
    )
    activation = swiglu * weight[:, None]
    amax = gl.max(gl.abs(activation), axis=1)
    scale = gl.maximum(amax, 1.0e-10) * (1.0 / 448.0)
    quantized = (activation * _reciprocal(scale[:, None], fast_math)).to(gl.float8e4nv)

    sf_pool_rows = task_pool_block * _SF_BLOCK_M + logical_rows
    sf_group = task_n_block * 2 + wg_n
    gl.store(
        l2_acts_sf + sf_group * num_padded_sf_pool_tokens + sf_pool_rows,
        scale,
        mask=valid_rows,
    )

    if valid_m > wg_m * fragment_m:
        l2_epilogue_buffer.store(quantized)
        partition_barrier()
        fence_async_shared()
        tma.async_copy_shared_to_global(
            l2_store_desc,
            [
                task_pool_block * _SPLIT_BLOCK_M + wg_m * fragment_m,
                task_n_block * (_NORMAL_BLOCK_N // 2) + wg_n * (fragment_n // 2),
            ],
            l2_epilogue_buffer,
        )
        tma.store_wait(0)
        partition_barrier()
        gl.atomic_add(
            l2_arrival + task_pool_block,
            1,
            sem="release",
            scope="gpu",
        )
    partition_barrier()


# fp8/combine.py


@gluon.jit
def combine_partition(
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

    # One math warp owns one token.  M<=2 can afford a wider live reduction;
    # from M=4 onward N1024 shows register-pressure tails, so retain N512.
    # Both paths use repeated 16-byte lane-vector transactions, moving toward
    # DeepGEMM's wide chunked combine without raw-pointer 1D bulk TMA.
    combine_block_n: gl.constexpr = 1024 if num_tokens <= 2 else 512
    combine_rows: gl.constexpr = num_math_warps
    combine_width: gl.constexpr = combine_block_n // 32
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
        combine_block_n,
        layout=gl.SliceLayout(0, combine_layout),
    )
    num_combine_n_blocks: gl.constexpr = (l2_n + combine_block_n - 1) // combine_block_n
    output_m_block = gl.program_id(0)
    num_output_m_blocks = (num_tokens + combine_rows - 1) // combine_rows
    while output_m_block < num_output_m_blocks:
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
        output_n_block = 0
        while output_n_block < num_combine_n_blocks:
            output_cols = output_n_block * combine_block_n + combine_col_vector[None, :]
            mask = output_row_mask & (output_cols < l2_n)
            reduced = gl.zeros(
                (combine_rows, combine_block_n),
                dtype=gl.float32,
                layout=combine_layout,
            )
            for slot in gl.static_range(topk):
                combine_ptrs = (
                    combine_buffer + (output_rows * topk + slot) * l2_n + output_cols
                )
                values = _load_contiguous_bf16_fragment_16b(
                    combine_ptrs,
                    mask & ((valid_slot_mask & (1 << slot)) != 0),
                ).to(gl.float32)
                reduced += values
            output_ptrs = output + output_rows * l2_n + output_cols
            _store_contiguous_bf16_fragment_16b(
                output_ptrs,
                reduced.to(gl.bfloat16),
                mask,
            )
            output_n_block += 1
        output_m_block += num_sms


@gluon.jit
def combine_split_partition(
    output,
    combine_buffer,
    topk_idx,
    fused_barrier,
    peer_fused_barrier_ptrs,
    fc2_scatter_grid_counter,
    combine_cross_rank_ready,
    math_done,
    l2_n: gl.constexpr,
    num_tokens: gl.constexpr,
    topk: gl.constexpr,
    world_size: gl.constexpr,
    num_sms: gl.constexpr,
    num_math_partitions: gl.constexpr,
    math_partition_idx: gl.constexpr,
):
    """Combine with independent 4-warp math partitions.

    The shared mbarrier is the only cross-partition rendezvous: it orders all
    FC2 scatter slices before partition zero publishes this CTA into the
    grid/cross-rank barrier.  Afterwards every local warp is an independent
    combine worker with the same global numbering as DeepGEMM.
    """
    partition_barrier()
    mbarrier.arrive(math_done, count=1)
    mbarrier.wait(math_done, 0)

    if math_partition_idx == 0:
        gl.atomic_add(
            fc2_scatter_grid_counter,
            1,
            sem="release",
            scope="gpu",
        )
        grid_arrived = _load_i32_acquire_gpu(fc2_scatter_grid_counter)
        while grid_arrived < num_sms:
            grid_arrived = _load_i32_acquire_gpu(fc2_scatter_grid_counter)

        if gl.program_id(0) == 0:
            barrier_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
            _peer_barrier_arrive_and_wait(
                peer_fused_barrier_ptrs,
                fused_barrier,
                world_size,
                1,
                barrier_layout,
                4 * 32,
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
    partition_barrier()

    combine_block_n: gl.constexpr = 1024 if num_tokens <= 2 else 512
    local_combine_rows: gl.constexpr = 4
    global_combine_rows: gl.constexpr = local_combine_rows * num_math_partitions
    combine_width: gl.constexpr = combine_block_n // 32
    combine_layout: gl.constexpr = gl.BlockedLayout(
        [1, combine_width],
        [1, 32],
        [local_combine_rows, 1],
        [1, 0],
    )
    local_row_vector = gl.arange(
        0,
        local_combine_rows,
        layout=gl.SliceLayout(1, combine_layout),
    )
    combine_col_vector = gl.arange(
        0,
        combine_block_n,
        layout=gl.SliceLayout(0, combine_layout),
    )
    num_combine_n_blocks: gl.constexpr = (l2_n + combine_block_n - 1) // combine_block_n
    output_m_block = gl.program_id(0)
    num_output_m_blocks = (num_tokens + global_combine_rows - 1) // global_combine_rows
    while output_m_block < num_output_m_blocks:
        output_rows = (
            output_m_block * global_combine_rows
            + math_partition_idx * local_combine_rows
            + local_row_vector[:, None]
        )
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
        output_n_block = 0
        while output_n_block < num_combine_n_blocks:
            output_cols = output_n_block * combine_block_n + combine_col_vector[None, :]
            mask = output_row_mask & (output_cols < l2_n)
            reduced = gl.zeros(
                (local_combine_rows, combine_block_n),
                dtype=gl.float32,
                layout=combine_layout,
            )
            for slot in gl.static_range(topk):
                combine_ptrs = (
                    combine_buffer + (output_rows * topk + slot) * l2_n + output_cols
                )
                values = _load_contiguous_bf16_fragment_16b(
                    combine_ptrs,
                    mask & ((valid_slot_mask & (1 << slot)) != 0),
                ).to(gl.float32)
                reduced += values
            output_ptrs = output + output_rows * l2_n + output_cols
            _store_contiguous_bf16_fragment_16b(
                output_ptrs,
                reduced.to(gl.bfloat16),
                mask,
            )
            output_n_block += 1
        output_m_block += num_sms


# fp8/math_split_bn128.py


@gluon.jit
def math_split_bn128_body(
    barriers,
    buffers,
    l2_store_desc,
    l2_epilogue_buffer,
    fc1_amax_scratch_0,
    fc1_amax_scratch_1,
    fc1_scale_ready,
    fc1_scale_done,
    l2_acts_sf,
    token_src_metadata,
    peer_combine_buffer_ptrs,
    route_weights,
    l2_arrival,
    l1_weight_scales,
    l2_weight_scales,
    expert_state,
    dispatch_counter,
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
    l1_n_blocks: gl.constexpr,
    l2_n_blocks: gl.constexpr,
    num_experts_per_wave: gl.constexpr,
    num_sms: gl.constexpr,
    scheduler_count_capacity: gl.constexpr,
    scheduler_counts_per_lane: gl.constexpr,
    num_padded_sf_pool_tokens: gl.constexpr,
    max_tokens: gl.constexpr,
    topk: gl.constexpr,
    world_size: gl.constexpr,
    activation_clamp: gl.constexpr,
    has_activation_clamp: gl.constexpr,
    fast_math: gl.constexpr,
    math_partition_idx: gl.constexpr,
    staged_dispatch_handoff: gl.constexpr,
    fc2_tile_complete: gl.constexpr = _tile_complete,
    fc2_completion_state=(),
    fc1_promotion_k: gl.constexpr = 128,
    fc2_promotion_k: gl.constexpr = 64,
):
    """FC1, SwiGLU quantization and FC2 scatter; publication belongs to the caller."""
    logical_block_m: gl.constexpr = 64
    logical_block_n: gl.constexpr = 128
    fragment_m: gl.constexpr = 64
    fragment_n: gl.constexpr = 64

    stage_empty, stage_ready = barriers
    a_buffers, b_buffers, sfa_lo_buffers, sfa_hi_buffers = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_k: gl.constexpr = a_buffers.type.shape[2]
    scheduler_layout: gl.constexpr = gl.BlockedLayout(
        [scheduler_counts_per_lane], [32], [4], [0]
    )
    count_offsets = gl.arange(0, scheduler_count_capacity, layout=scheduler_layout)
    if staged_dispatch_handoff:
        _wait_for_dispatch_handoff(dispatch_counter, 4 * num_sms + 1)
    stored_counts = _load_packed_expert_counts(
        expert_state, count_offsets, E, world_size
    )

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[4, 1],
        instr_shape=[16, 64, 32],
    )
    mma_cols = gl.arange(0, fragment_n, layout=gl.SliceLayout(0, mma_layout))
    row_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)
    l1_use_up_scale = ((mma_cols // 8) % 2) == 1
    store_layout: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
    store_rows = gl.arange(0, fragment_m, layout=gl.SliceLayout(1, store_layout))
    store_cols = math_partition_idx * fragment_n + gl.arange(
        0, fragment_n, layout=gl.SliceLayout(0, store_layout)
    )

    block_idx = gl.program_id(0)
    scheduler_expert = 0
    scheduler_phase = 1
    current_count = scheduler_count(stored_counts, count_offsets, scheduler_expert)
    current_pool_block_offset = 0
    pipeline_tile = 0
    fc1_sync_phase = 0
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
        logical_block_m,
    )

    while task_phase != 0:
        local_row = task_m_block * logical_block_m
        valid_m = gl.minimum(logical_block_m, valid_count - local_row)
        final = gl.zeros(
            (fragment_m, fragment_n),
            dtype=gl.float32,
            layout=mma_layout,
        )
        # Keep dequantized accumulation in CUDA FP32 registers. Rescaling
        # it back into each WGMMA accumulator loses low mantissa bits on long
        # K reductions; this matters for M=0 owners with nonempty EP16 pools.
        num_k_tiles = gl.where(task_phase == 1, l1_k // block_k, l2_k // block_k)
        k_tile = 0
        while k_tile < num_k_tiles:
            stage = pipeline_tile % num_stages
            pipe_phase = pipeline_tile // num_stages & 1
            mbarrier.wait(stage_ready.index(stage), pipe_phase)
            a_stage = a_buffers.index(stage)
            b_stage = b_buffers.index(stage).slice(
                math_partition_idx * fragment_n, fragment_n, dim=0
            )

            if task_phase == 1:
                partial = gl.zeros(
                    (fragment_m, fragment_n),
                    dtype=gl.float32,
                    layout=mma_layout,
                )
                partial = warpgroup_mma(
                    a_stage,
                    b_stage.permute((1, 0)),
                    partial,
                    is_async=True,
                    use_acc=False,
                    max_num_imprecise_acc=fc1_promotion_k,
                )
                row_scale = sfa_lo_buffers.index(stage).load(row_layout)
                gate_scale_block = task_n_block * logical_block_n // 256
                up_scale_block = l2_k // 128 + gate_scale_block
                gate_scale = gl.load(
                    l1_weight_scales
                    + task_expert.to(gl.int64) * l1_ws_stride_e
                    + gate_scale_block * l1_ws_stride_n
                    + k_tile * l1_ws_stride_k
                )
                up_scale = gl.load(
                    l1_weight_scales
                    + task_expert.to(gl.int64) * l1_ws_stride_e
                    + up_scale_block * l1_ws_stride_n
                    + k_tile * l1_ws_stride_k
                )
                column_scale = gl.where(l1_use_up_scale, up_scale, gate_scale)
                partial = warpgroup_mma_wait(num_outstanding=0, deps=(partial,))
                final += partial * (row_scale[:, None] * column_scale[None, :])
            else:
                weight_scale_block = task_n_block
                weight_scale = gl.load(
                    l2_weight_scales
                    + task_expert.to(gl.int64) * l2_ws_stride_e
                    + weight_scale_block * l2_ws_stride_n
                    + k_tile * l2_ws_stride_k
                )
                row_scale_lo = sfa_lo_buffers.index(stage).load(row_layout)
                row_scale_hi = sfa_hi_buffers.index(stage).load(row_layout)
                partial_lo = gl.zeros(
                    (fragment_m, fragment_n),
                    dtype=gl.float32,
                    layout=mma_layout,
                )
                partial_lo = warpgroup_mma(
                    a_stage.slice(0, 64, dim=1),
                    b_stage.slice(0, 64, dim=1).permute((1, 0)),
                    partial_lo,
                    is_async=True,
                    use_acc=False,
                    max_num_imprecise_acc=fc2_promotion_k,
                )
                partial_lo = warpgroup_mma_wait(num_outstanding=0, deps=(partial_lo,))
                final += partial_lo * (row_scale_lo[:, None] * weight_scale)

                partial_hi = gl.zeros(
                    (fragment_m, fragment_n),
                    dtype=gl.float32,
                    layout=mma_layout,
                )
                partial_hi = warpgroup_mma(
                    a_stage.slice(64, 64, dim=1),
                    b_stage.slice(64, 64, dim=1).permute((1, 0)),
                    partial_hi,
                    is_async=True,
                    use_acc=False,
                    max_num_imprecise_acc=fc2_promotion_k,
                )
                partial_hi = warpgroup_mma_wait(num_outstanding=0, deps=(partial_hi,))
                final += partial_hi * (row_scale_hi[:, None] * weight_scale)

            _release_stage(stage_empty, stage)
            pipeline_tile += 1
            k_tile += 1

        if task_phase == 1:
            _fc1_bm64_bn128_split_epilogue(
                final,
                l2_store_desc,
                l2_epilogue_buffer,
                fc1_amax_scratch_0,
                fc1_amax_scratch_1,
                fc1_scale_ready,
                fc1_scale_done,
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
                math_partition_idx,
                fc1_sync_phase,
            )
            fc1_sync_phase ^= 1
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
                logical_block_m,
                logical_block_n,
            )
            fc2_tile_complete(
                fc2_completion_state,
                task_expert,
                task_pool_block,
                gl.cdiv(valid_count, logical_block_m),
                2 * l2_n_blocks,
            )

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
            logical_block_m,
        )


@gluon.jit
def math_split_bn128_partition(
    barriers,
    buffers,
    l2_store_desc,
    l2_epilogue_buffer,
    fc1_amax_scratch_0,
    fc1_amax_scratch_1,
    fc1_scale_ready,
    fc1_scale_done,
    math_done,
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
    l1_weight_scales,
    l2_weight_scales,
    expert_state,
    dispatch_counter,
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
    math_partition_idx: gl.constexpr,
    staged_dispatch_handoff: gl.constexpr,
    fc1_promotion_k: gl.constexpr = 128,
    fc2_promotion_k: gl.constexpr = 64,
):
    """Compose the shared math body with the EP8 combine tail."""
    math_split_bn128_body(
        barriers,
        buffers,
        l2_store_desc,
        l2_epilogue_buffer,
        fc1_amax_scratch_0,
        fc1_amax_scratch_1,
        fc1_scale_ready,
        fc1_scale_done,
        l2_acts_sf,
        token_src_metadata,
        peer_combine_buffer_ptrs,
        route_weights,
        l2_arrival,
        l1_weight_scales,
        l2_weight_scales,
        expert_state,
        dispatch_counter,
        l1_k,
        l2_n,
        l2_k,
        E,
        l1_ws_stride_e,
        l1_ws_stride_n,
        l1_ws_stride_k,
        l2_ws_stride_e,
        l2_ws_stride_n,
        l2_ws_stride_k,
        l1_n_blocks,
        l2_n_blocks,
        num_experts_per_wave,
        num_sms,
        scheduler_count_capacity,
        scheduler_counts_per_lane,
        num_padded_sf_pool_tokens,
        max_tokens,
        topk,
        world_size,
        activation_clamp,
        has_activation_clamp,
        fast_math,
        math_partition_idx,
        staged_dispatch_handoff,
        fc1_promotion_k=fc1_promotion_k,
        fc2_promotion_k=fc2_promotion_k,
    )
    combine_split_partition(
        output,
        combine_buffer,
        topk_idx,
        fused_barrier,
        peer_fused_barrier_ptrs,
        fc2_scatter_grid_counter,
        combine_cross_rank_ready,
        math_done,
        l2_n,
        num_tokens,
        topk,
        world_size,
        num_sms,
        2,
        math_partition_idx,
    )


# fp8/math_split_bn256.py


@gluon.jit
def math_split_bn256_body(
    barriers,
    buffers,
    l2_store_desc,
    l2_epilogue_buffer,
    l2_acts_sf,
    token_src_metadata,
    peer_combine_buffer_ptrs,
    route_weights,
    l2_arrival,
    l1_weight_scales,
    l2_weight_scales,
    expert_state,
    dispatch_counter,
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
    l1_n_blocks: gl.constexpr,
    l2_n_blocks: gl.constexpr,
    num_experts_per_wave: gl.constexpr,
    num_sms: gl.constexpr,
    scheduler_count_capacity: gl.constexpr,
    scheduler_counts_per_lane: gl.constexpr,
    num_padded_sf_pool_tokens: gl.constexpr,
    max_tokens: gl.constexpr,
    topk: gl.constexpr,
    world_size: gl.constexpr,
    activation_clamp: gl.constexpr,
    has_activation_clamp: gl.constexpr,
    fast_math: gl.constexpr,
    math_partition_idx: gl.constexpr,
    staged_dispatch_handoff: gl.constexpr,
    fc2_tile_complete: gl.constexpr = _tile_complete,
    fc2_completion_state=(),
    fc1_promotion_k: gl.constexpr = 128,
    fc2_promotion_k: gl.constexpr = 64,
):
    """FC1, SwiGLU quantization and FC2 scatter; publication belongs to the caller."""
    logical_block_m: gl.constexpr = 64
    logical_block_n: gl.constexpr = 256
    fragment_m: gl.constexpr = 64
    fragment_n: gl.constexpr = 128

    stage_empty, stage_ready = barriers
    a_buffers, b_buffers, sfa_lo_buffers, sfa_hi_buffers = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_k: gl.constexpr = a_buffers.type.shape[2]
    scheduler_layout: gl.constexpr = gl.BlockedLayout(
        [scheduler_counts_per_lane], [32], [4], [0]
    )
    count_offsets = gl.arange(0, scheduler_count_capacity, layout=scheduler_layout)
    if staged_dispatch_handoff:
        _wait_for_dispatch_handoff(dispatch_counter, 4 * num_sms + 1)
    stored_counts = _load_packed_expert_counts(
        expert_state, count_offsets, E, world_size
    )

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[4, 1],
        instr_shape=[16, 128, 32],
    )
    mma_cols = gl.arange(0, fragment_n, layout=gl.SliceLayout(0, mma_layout))
    row_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)
    l1_use_up_scale = ((mma_cols // 8) % 2) == 1
    store_layout: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
    store_rows = gl.arange(0, fragment_m, layout=gl.SliceLayout(1, store_layout))
    store_cols = math_partition_idx * fragment_n + gl.arange(
        0, fragment_n, layout=gl.SliceLayout(0, store_layout)
    )

    block_idx = gl.program_id(0)
    scheduler_expert = 0
    scheduler_phase = 1
    current_count = scheduler_count(stored_counts, count_offsets, scheduler_expert)
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
        logical_block_m,
    )

    while task_phase != 0:
        local_row = task_m_block * logical_block_m
        valid_m = gl.minimum(logical_block_m, valid_count - local_row)
        final = gl.zeros(
            (fragment_m, fragment_n),
            dtype=gl.float32,
            layout=mma_layout,
        )
        num_k_tiles = gl.where(task_phase == 1, l1_k // block_k, l2_k // block_k)
        k_tile = 0
        while k_tile < num_k_tiles:
            stage = pipeline_tile % num_stages
            pipe_phase = pipeline_tile // num_stages & 1
            mbarrier.wait(stage_ready.index(stage), pipe_phase)
            a_stage = a_buffers.index(stage)
            b_stage = b_buffers.index(stage).slice(
                math_partition_idx * fragment_n, fragment_n, dim=0
            )

            if task_phase == 1:
                partial = gl.zeros(
                    (fragment_m, fragment_n),
                    dtype=gl.float32,
                    layout=mma_layout,
                )
                partial = warpgroup_mma(
                    a_stage,
                    b_stage.permute((1, 0)),
                    partial,
                    is_async=True,
                    use_acc=False,
                    max_num_imprecise_acc=fc1_promotion_k,
                )
                row_scale = sfa_lo_buffers.index(stage).load(row_layout)
                gate_scale_block = task_n_block
                up_scale_block = l2_k // 128 + gate_scale_block
                gate_scale = gl.load(
                    l1_weight_scales
                    + task_expert.to(gl.int64) * l1_ws_stride_e
                    + gate_scale_block * l1_ws_stride_n
                    + k_tile * l1_ws_stride_k
                )
                up_scale = gl.load(
                    l1_weight_scales
                    + task_expert.to(gl.int64) * l1_ws_stride_e
                    + up_scale_block * l1_ws_stride_n
                    + k_tile * l1_ws_stride_k
                )
                column_scale = gl.where(l1_use_up_scale, up_scale, gate_scale)
                partial = warpgroup_mma_wait(num_outstanding=0, deps=(partial,))
                final += partial * (row_scale[:, None] * column_scale[None, :])
            else:
                weight_scale_block = task_n_block * 2 + math_partition_idx
                weight_scale = gl.load(
                    l2_weight_scales
                    + task_expert.to(gl.int64) * l2_ws_stride_e
                    + weight_scale_block * l2_ws_stride_n
                    + k_tile * l2_ws_stride_k
                )
                row_scale_lo = sfa_lo_buffers.index(stage).load(row_layout)
                row_scale_hi = sfa_hi_buffers.index(stage).load(row_layout)
                partial_lo = gl.zeros(
                    (fragment_m, fragment_n),
                    dtype=gl.float32,
                    layout=mma_layout,
                )
                partial_lo = warpgroup_mma(
                    a_stage.slice(0, 64, dim=1),
                    b_stage.slice(0, 64, dim=1).permute((1, 0)),
                    partial_lo,
                    is_async=True,
                    use_acc=False,
                    max_num_imprecise_acc=fc2_promotion_k,
                )
                partial_lo = warpgroup_mma_wait(num_outstanding=0, deps=(partial_lo,))
                final += partial_lo * (row_scale_lo[:, None] * weight_scale)

                partial_hi = gl.zeros(
                    (fragment_m, fragment_n),
                    dtype=gl.float32,
                    layout=mma_layout,
                )
                partial_hi = warpgroup_mma(
                    a_stage.slice(64, 64, dim=1),
                    b_stage.slice(64, 64, dim=1).permute((1, 0)),
                    partial_hi,
                    is_async=True,
                    use_acc=False,
                    max_num_imprecise_acc=fc2_promotion_k,
                )
                partial_hi = warpgroup_mma_wait(num_outstanding=0, deps=(partial_hi,))
                final += partial_hi * (row_scale_hi[:, None] * weight_scale)

            _release_stage(stage_empty, stage)
            pipeline_tile += 1
            k_tile += 1

        if task_phase == 1:
            _fc1_bm64_bn256_split_epilogue(
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
                math_partition_idx,
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
                logical_block_m,
                logical_block_n,
            )
            fc2_tile_complete(
                fc2_completion_state,
                task_expert,
                task_pool_block,
                gl.cdiv(valid_count, logical_block_m),
                2 * l2_n_blocks,
            )

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
            logical_block_m,
        )


@gluon.jit
def math_split_bn256_partition(
    barriers,
    buffers,
    l2_store_desc,
    l2_epilogue_buffer,
    math_done,
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
    l1_weight_scales,
    l2_weight_scales,
    expert_state,
    dispatch_counter,
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
    math_partition_idx: gl.constexpr,
    staged_dispatch_handoff: gl.constexpr,
    fc1_promotion_k: gl.constexpr = 128,
    fc2_promotion_k: gl.constexpr = 64,
):
    """Compose the shared math body with the EP8 combine tail."""
    math_split_bn256_body(
        barriers,
        buffers,
        l2_store_desc,
        l2_epilogue_buffer,
        l2_acts_sf,
        token_src_metadata,
        peer_combine_buffer_ptrs,
        route_weights,
        l2_arrival,
        l1_weight_scales,
        l2_weight_scales,
        expert_state,
        dispatch_counter,
        l1_k,
        l2_n,
        l2_k,
        E,
        l1_ws_stride_e,
        l1_ws_stride_n,
        l1_ws_stride_k,
        l2_ws_stride_e,
        l2_ws_stride_n,
        l2_ws_stride_k,
        l1_n_blocks,
        l2_n_blocks,
        num_experts_per_wave,
        num_sms,
        scheduler_count_capacity,
        scheduler_counts_per_lane,
        num_padded_sf_pool_tokens,
        max_tokens,
        topk,
        world_size,
        activation_clamp,
        has_activation_clamp,
        fast_math,
        math_partition_idx,
        staged_dispatch_handoff,
        fc1_promotion_k=fc1_promotion_k,
        fc2_promotion_k=fc2_promotion_k,
    )
    combine_split_partition(
        output,
        combine_buffer,
        topk_idx,
        fused_barrier,
        peer_fused_barrier_ptrs,
        fc2_scatter_grid_counter,
        combine_cross_rank_ready,
        math_done,
        l2_n,
        num_tokens,
        topk,
        world_size,
        num_sms,
        2,
        math_partition_idx,
    )


# fp8/math_split_bm128.py


@gluon.jit
def math_split_bm128_partition(
    barriers,
    buffers,
    l2_store_desc,
    l2_epilogue_buffer,
    math_done,
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
    l1_weight_scales,
    l2_weight_scales,
    expert_state,
    dispatch_counter,
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
    math_partition_idx: gl.constexpr,
    staged_dispatch_handoff: gl.constexpr,
    fc1_promotion_k: gl.constexpr = 128,
    fc2_promotion_k: gl.constexpr = 64,
):
    """One 4-warp 64x128 fragment of a logical BM128/BN256 tile."""
    logical_block_m: gl.constexpr = 128
    logical_block_n: gl.constexpr = 256
    fragment_m: gl.constexpr = 64
    fragment_n: gl.constexpr = 128
    wg_m: gl.constexpr = math_partition_idx // 2
    wg_n: gl.constexpr = math_partition_idx % 2

    stage_empty, stage_ready = barriers
    a_buffers, b_buffers, sfa_lo_buffers, sfa_hi_buffers = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_k: gl.constexpr = a_buffers.type.shape[2]

    scheduler_layout: gl.constexpr = gl.BlockedLayout(
        [scheduler_counts_per_lane], [32], [4], [0]
    )
    count_offsets = gl.arange(0, scheduler_count_capacity, layout=scheduler_layout)
    if staged_dispatch_handoff:
        _wait_for_dispatch_handoff(dispatch_counter, 4 * num_sms + 1)
    stored_counts = _load_packed_expert_counts(
        expert_state, count_offsets, E, world_size
    )

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[4, 1],
        instr_shape=[16, 128, 32],
    )
    mma_cols = gl.arange(0, fragment_n, layout=gl.SliceLayout(0, mma_layout))
    row_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)
    l1_use_up_scale = ((mma_cols // 8) % 2) == 1
    store_layout: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
    local_store_rows = gl.arange(0, fragment_m, layout=gl.SliceLayout(1, store_layout))
    local_store_cols = gl.arange(0, fragment_n, layout=gl.SliceLayout(0, store_layout))
    logical_store_rows = wg_m * fragment_m + local_store_rows
    logical_store_cols = wg_n * fragment_n + local_store_cols

    block_idx = gl.program_id(0)
    scheduler_expert = 0
    scheduler_phase = 1
    current_count = scheduler_count(stored_counts, count_offsets, scheduler_expert)
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
        logical_block_m,
    )

    while task_phase != 0:
        local_row = task_m_block * logical_block_m
        valid_m = gl.minimum(logical_block_m, valid_count - local_row)
        final = gl.zeros(
            (fragment_m, fragment_n),
            dtype=gl.float32,
            layout=mma_layout,
        )
        num_k_tiles = gl.where(
            task_phase == 1,
            l1_k // block_k,
            l2_k // block_k,
        )
        k_tile = 0
        while k_tile < num_k_tiles:
            stage = pipeline_tile % num_stages
            pipe_phase = pipeline_tile // num_stages & 1
            mbarrier.wait(stage_ready.index(stage), pipe_phase)
            a_stage = a_buffers.index(stage).slice(wg_m * fragment_m, fragment_m, dim=0)
            b_stage = b_buffers.index(stage).slice(wg_n * fragment_n, fragment_n, dim=0)

            if task_phase == 1:
                partial = gl.zeros(
                    (fragment_m, fragment_n),
                    dtype=gl.float32,
                    layout=mma_layout,
                )
                partial = warpgroup_mma(
                    a_stage,
                    b_stage.permute((1, 0)),
                    partial,
                    is_async=True,
                    use_acc=False,
                    max_num_imprecise_acc=fc1_promotion_k,
                )
                row_scale = (
                    sfa_lo_buffers.index(stage)
                    .slice(wg_m * fragment_m, fragment_m, dim=0)
                    .load(row_layout)
                )
                gate_scale_block = task_n_block
                up_scale_block = l2_k // 128 + gate_scale_block
                gate_scale = gl.load(
                    l1_weight_scales
                    + task_expert.to(gl.int64) * l1_ws_stride_e
                    + gate_scale_block * l1_ws_stride_n
                    + k_tile * l1_ws_stride_k
                )
                up_scale = gl.load(
                    l1_weight_scales
                    + task_expert.to(gl.int64) * l1_ws_stride_e
                    + up_scale_block * l1_ws_stride_n
                    + k_tile * l1_ws_stride_k
                )
                column_scale = gl.where(l1_use_up_scale, up_scale, gate_scale)
                partial = warpgroup_mma_wait(num_outstanding=0, deps=(partial,))
                final += partial * (row_scale[:, None] * column_scale[None, :])
            else:
                l2_weight_scale_block = task_n_block * 2 + wg_n
                l2_weight_scale = gl.load(
                    l2_weight_scales
                    + task_expert.to(gl.int64) * l2_ws_stride_e
                    + l2_weight_scale_block * l2_ws_stride_n
                    + k_tile * l2_ws_stride_k
                )
                row_scale_lo = (
                    sfa_lo_buffers.index(stage)
                    .slice(wg_m * fragment_m, fragment_m, dim=0)
                    .load(row_layout)
                )
                row_scale_hi = (
                    sfa_hi_buffers.index(stage)
                    .slice(wg_m * fragment_m, fragment_m, dim=0)
                    .load(row_layout)
                )
                partial_lo = gl.zeros(
                    (fragment_m, fragment_n),
                    dtype=gl.float32,
                    layout=mma_layout,
                )
                partial_lo = warpgroup_mma(
                    a_stage.slice(0, 64, dim=1),
                    b_stage.slice(0, 64, dim=1).permute((1, 0)),
                    partial_lo,
                    is_async=True,
                    use_acc=False,
                    max_num_imprecise_acc=fc2_promotion_k,
                )
                partial_lo = warpgroup_mma_wait(num_outstanding=0, deps=(partial_lo,))
                final += partial_lo * (row_scale_lo[:, None] * l2_weight_scale)

                partial_hi = gl.zeros(
                    (fragment_m, fragment_n),
                    dtype=gl.float32,
                    layout=mma_layout,
                )
                partial_hi = warpgroup_mma(
                    a_stage.slice(64, 64, dim=1),
                    b_stage.slice(64, 64, dim=1).permute((1, 0)),
                    partial_hi,
                    is_async=True,
                    use_acc=False,
                    max_num_imprecise_acc=fc2_promotion_k,
                )
                partial_hi = warpgroup_mma_wait(num_outstanding=0, deps=(partial_hi,))
                final += partial_hi * (row_scale_hi[:, None] * l2_weight_scale)

            _release_stage(stage_empty, stage)
            pipeline_tile += 1
            k_tile += 1

        if task_phase == 1:
            _fc1_bm128_bn256_split_epilogue(
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
                math_partition_idx,
            )
        else:
            fc2_tile = gl.convert_layout(final.to(gl.bfloat16), store_layout)
            _fc2_bf16_scatter_epilogue(
                fc2_tile,
                logical_store_rows,
                logical_store_cols,
                token_src_metadata,
                peer_combine_buffer_ptrs,
                task_pool_block,
                task_n_block,
                valid_m,
                l2_n,
                max_tokens,
                topk,
                world_size,
                logical_block_m,
                logical_block_n,
            )

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
            logical_block_m,
        )

    combine_split_partition(
        output,
        combine_buffer,
        topk_idx,
        fused_barrier,
        peer_fused_barrier_ptrs,
        fc2_scatter_grid_counter,
        combine_cross_rank_ready,
        math_done,
        l2_n,
        num_tokens,
        topk,
        world_size,
        num_sms,
        4,
        math_partition_idx,
    )


# fp8/math_swap.py


@gluon.jit
def _fc1_swap_mainloop(
    barriers,
    buffers,
    l1_weight_scales,
    task_expert,
    task_n_block,
    pipeline_tile,
    l1_k: gl.constexpr,
    intermediate_hidden: gl.constexpr,
    l1_ws_stride_e: gl.constexpr,
    l1_ws_stride_n: gl.constexpr,
    l1_ws_stride_k: gl.constexpr,
    n_swap: gl.constexpr,
    block_n: gl.constexpr,
    max_num_imprecise_acc: gl.constexpr = 128,
):
    """FC1 swapAB mainloop with independent gate/up scale domains."""
    stage_empty, stage_ready = barriers
    a_buffers, b_buffers, sfa_lo_buffers, sfa_hi_buffers = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_k: gl.constexpr = a_buffers.type.shape[2]
    num_k_tiles: gl.constexpr = l1_k // block_k

    swap_mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[gl.num_warps(), 1],
        instr_shape=[16, n_swap, 32],
    )
    channel_offsets = gl.arange(
        0,
        block_n,
        layout=gl.SliceLayout(1, swap_mma_layout),
    )
    use_up_scale = ((channel_offsets // 8) % 2) == 1
    final = gl.zeros(
        (block_n, n_swap),
        dtype=gl.float32,
        layout=swap_mma_layout,
    )
    if block_n == 128:
        gate_scale_block = task_n_block // 2
    else:
        gate_scale_block = task_n_block * (block_n // 256) + channel_offsets // 256
    up_scale_block = intermediate_hidden // 128 + gate_scale_block
    for k_tile in range(num_k_tiles):
        tile = pipeline_tile + k_tile
        stage = tile % num_stages
        phase = tile // num_stages & 1
        mbarrier.wait(stage_ready.index(stage), phase)

        partial = gl.zeros(
            (block_n, n_swap),
            dtype=gl.float32,
            layout=swap_mma_layout,
        )
        partial = warpgroup_mma(
            b_buffers.index(stage),
            a_buffers.index(stage).slice(0, n_swap, dim=0).permute((1, 0)),
            partial,
            is_async=True,
            use_acc=False,
            max_num_imprecise_acc=min(128, max_num_imprecise_acc),
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
        gate_scale = gl.load(
            l1_weight_scales
            + task_expert.to(gl.int64) * l1_ws_stride_e
            + gate_scale_block * l1_ws_stride_n
            + k_tile * l1_ws_stride_k
        )
        up_scale = gl.load(
            l1_weight_scales
            + task_expert.to(gl.int64) * l1_ws_stride_e
            + up_scale_block * l1_ws_stride_n
            + k_tile * l1_ws_stride_k
        )
        channel_scale = gl.where(use_up_scale, up_scale, gate_scale)
        partial = warpgroup_mma_wait(
            num_outstanding=0,
            deps=(partial,),
        )
        # Refill may overlap register-only promotion once A/B/SFA reads finish.
        _release_stage(stage_empty, stage)
        final += partial * (channel_scale[:, None] * token_scale[None, :])

    return final


@gluon.jit
def _fc2_swap_mainloop(
    barriers,
    buffers,
    l2_weight_scales,
    task_expert,
    task_n_block,
    pipeline_tile,
    l2_k: gl.constexpr,
    l2_ws_stride_e: gl.constexpr,
    l2_ws_stride_n: gl.constexpr,
    l2_ws_stride_k: gl.constexpr,
    n_swap: gl.constexpr,
    block_n: gl.constexpr,
    max_num_imprecise_acc: gl.constexpr = 64,
):
    """FC2 swapAB mainloop with independent low/high K64 A scales."""
    stage_empty, stage_ready = barriers
    a_buffers, b_buffers, sfa_lo_buffers, sfa_hi_buffers = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_k: gl.constexpr = a_buffers.type.shape[2]
    num_k_tiles: gl.constexpr = l2_k // block_k

    swap_mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[gl.num_warps(), 1],
        instr_shape=[16, n_swap, 32],
    )
    channel_offsets = gl.arange(
        0,
        block_n,
        layout=gl.SliceLayout(1, swap_mma_layout),
    )
    final = gl.zeros(
        (block_n, n_swap),
        dtype=gl.float32,
        layout=swap_mma_layout,
    )
    for k_tile in range(num_k_tiles):
        tile = pipeline_tile + k_tile
        stage = tile % num_stages
        phase = tile // num_stages & 1
        mbarrier.wait(stage_ready.index(stage), phase)

        channel_stage = b_buffers.index(stage)
        token_stage = a_buffers.index(stage).slice(0, n_swap, dim=0)
        l2_weight_scale = gl.load(
            l2_weight_scales
            + task_expert.to(gl.int64) * l2_ws_stride_e
            + (task_n_block * (block_n // 128) + channel_offsets // 128)
            * l2_ws_stride_n
            + k_tile * l2_ws_stride_k
        )

        partial_lo = gl.zeros(
            (block_n, n_swap),
            dtype=gl.float32,
            layout=swap_mma_layout,
        )
        partial_lo = warpgroup_mma(
            channel_stage.slice(0, 64, dim=1),
            token_stage.slice(0, 64, dim=1).permute((1, 0)),
            partial_lo,
            is_async=True,
            use_acc=False,
            max_num_imprecise_acc=min(64, max_num_imprecise_acc),
        )
        partial_lo = warpgroup_mma_wait(
            num_outstanding=0,
            deps=(partial_lo,),
        )
        token_scale_lo = (
            sfa_lo_buffers.index(stage)
            .slice(
                0,
                n_swap,
                dim=0,
            )
            .load(gl.SliceLayout(0, swap_mma_layout))
        )
        final += partial_lo * (l2_weight_scale[:, None] * token_scale_lo[None, :])

        partial_hi = gl.zeros(
            (block_n, n_swap),
            dtype=gl.float32,
            layout=swap_mma_layout,
        )
        partial_hi = warpgroup_mma(
            channel_stage.slice(64, 64, dim=1),
            token_stage.slice(64, 64, dim=1).permute((1, 0)),
            partial_hi,
            is_async=True,
            use_acc=False,
            max_num_imprecise_acc=min(64, max_num_imprecise_acc),
        )
        partial_hi = warpgroup_mma_wait(
            num_outstanding=0,
            deps=(partial_hi,),
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
        # The high SFA load must finish before this stage can be reused.
        _release_stage(stage_empty, stage)
        final += partial_hi * (l2_weight_scale[:, None] * token_scale_hi[None, :])

    return final


@gluon.jit
def math_swap_body(
    barriers,
    buffers,
    l2_store_desc,
    l2_epilogue_buffer,
    l2_acts_sf,
    token_src_metadata,
    peer_combine_buffer_ptrs,
    route_weights,
    l2_arrival,
    l1_weight_scales,
    l2_weight_scales,
    expert_state,
    dispatch_counter,
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
    l1_n_blocks: gl.constexpr,
    l2_n_blocks: gl.constexpr,
    num_experts_per_wave: gl.constexpr,
    num_sms: gl.constexpr,
    scheduler_count_capacity: gl.constexpr,
    scheduler_counts_per_lane: gl.constexpr,
    num_padded_sf_pool_tokens: gl.constexpr,
    max_tokens: gl.constexpr,
    topk: gl.constexpr,
    world_size: gl.constexpr,
    activation_clamp: gl.constexpr,
    has_activation_clamp: gl.constexpr,
    fast_math: gl.constexpr,
    fc2_arrival_counter: gl.constexpr,
    fc2_epilogue_requires_full_sync: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    num_math_warps: gl.constexpr,
    staged_dispatch_handoff: gl.constexpr,
    max_num_imprecise_acc: gl.constexpr = 128,
    fc2_tile_complete: gl.constexpr = _tile_complete,
    fc2_completion_state=(),
    fc2_promotion_k: gl.constexpr = None,
):
    """FC1, SwiGLU quantization and FC2 scatter; publication belongs to the caller."""
    a_buffers, _, _, _ = buffers
    block_k: gl.constexpr = a_buffers.type.shape[2]

    scheduler_layout: gl.constexpr = gl.BlockedLayout(
        [scheduler_counts_per_lane],
        [32],
        [num_math_warps],
        [0],
    )
    count_offsets = gl.arange(
        0,
        scheduler_count_capacity,
        layout=scheduler_layout,
    )
    if staged_dispatch_handoff:
        _wait_for_dispatch_handoff(dispatch_counter, 4 * num_sms + 1)
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
        block_m,
    )

    while task_phase != 0:
        local_row = task_m_block * block_m
        valid_m = gl.minimum(
            block_m,
            valid_count - local_row,
        )

        if task_phase == 1:
            if valid_m <= 8:
                final_swap_8 = _fc1_swap_mainloop(
                    barriers,
                    buffers,
                    l1_weight_scales,
                    task_expert,
                    task_n_block,
                    pipeline_tile,
                    l1_k,
                    l2_k,
                    l1_ws_stride_e,
                    l1_ws_stride_n,
                    l1_ws_stride_k,
                    8,
                    block_n,
                    max_num_imprecise_acc=max_num_imprecise_acc,
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
                    block_m,
                    block_n,
                    fc2_arrival_counter,
                )
            elif block_m == 16 or valid_m <= 16:
                final_swap_16 = _fc1_swap_mainloop(
                    barriers,
                    buffers,
                    l1_weight_scales,
                    task_expert,
                    task_n_block,
                    pipeline_tile,
                    l1_k,
                    l2_k,
                    l1_ws_stride_e,
                    l1_ws_stride_n,
                    l1_ws_stride_k,
                    16,
                    block_n,
                    max_num_imprecise_acc=max_num_imprecise_acc,
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
                    block_m,
                    block_n,
                    fc2_arrival_counter,
                )
            elif valid_m <= 32:
                final_swap_32 = _fc1_swap_mainloop(
                    barriers,
                    buffers,
                    l1_weight_scales,
                    task_expert,
                    task_n_block,
                    pipeline_tile,
                    l1_k,
                    l2_k,
                    l1_ws_stride_e,
                    l1_ws_stride_n,
                    l1_ws_stride_k,
                    32,
                    block_n,
                    max_num_imprecise_acc=max_num_imprecise_acc,
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
                    block_m,
                    block_n,
                    fc2_arrival_counter,
                )
            elif block_m == 64 or valid_m <= 64:
                final_swap_64 = _fc1_swap_mainloop(
                    barriers,
                    buffers,
                    l1_weight_scales,
                    task_expert,
                    task_n_block,
                    pipeline_tile,
                    l1_k,
                    l2_k,
                    l1_ws_stride_e,
                    l1_ws_stride_n,
                    l1_ws_stride_k,
                    64,
                    block_n,
                    max_num_imprecise_acc=max_num_imprecise_acc,
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
                    block_m,
                    block_n,
                    fc2_arrival_counter,
                )
            else:
                final_swap_128 = _fc1_swap_mainloop(
                    barriers,
                    buffers,
                    l1_weight_scales,
                    task_expert,
                    task_n_block,
                    pipeline_tile,
                    l1_k,
                    l2_k,
                    l1_ws_stride_e,
                    l1_ws_stride_n,
                    l1_ws_stride_k,
                    128,
                    block_n,
                    max_num_imprecise_acc=max_num_imprecise_acc,
                )
                _fc1_epilogue(
                    final_swap_128,
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
                    128,
                    block_m,
                    block_n,
                    fc2_arrival_counter,
                )
            pipeline_tile += l1_k // block_k
        else:
            # Gluon register tensors require power-of-two element counts, so
            # both linear phases use the same 8/16/32/64 token buckets.
            if valid_m <= 8:
                final_swap_8 = _fc2_swap_mainloop(
                    barriers,
                    buffers,
                    l2_weight_scales,
                    task_expert,
                    task_n_block,
                    pipeline_tile,
                    l2_k,
                    l2_ws_stride_e,
                    l2_ws_stride_n,
                    l2_ws_stride_k,
                    8,
                    block_n,
                    max_num_imprecise_acc=max_num_imprecise_acc
                    if fc2_promotion_k is None
                    else fc2_promotion_k,
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
                    block_m,
                    block_n,
                )
            elif block_m == 16 or valid_m <= 16:
                final_swap_16 = _fc2_swap_mainloop(
                    barriers,
                    buffers,
                    l2_weight_scales,
                    task_expert,
                    task_n_block,
                    pipeline_tile,
                    l2_k,
                    l2_ws_stride_e,
                    l2_ws_stride_n,
                    l2_ws_stride_k,
                    16,
                    block_n,
                    max_num_imprecise_acc=max_num_imprecise_acc
                    if fc2_promotion_k is None
                    else fc2_promotion_k,
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
                    block_m,
                    block_n,
                )
            elif valid_m <= 32:
                final_swap_32 = _fc2_swap_mainloop(
                    barriers,
                    buffers,
                    l2_weight_scales,
                    task_expert,
                    task_n_block,
                    pipeline_tile,
                    l2_k,
                    l2_ws_stride_e,
                    l2_ws_stride_n,
                    l2_ws_stride_k,
                    32,
                    block_n,
                    max_num_imprecise_acc=max_num_imprecise_acc
                    if fc2_promotion_k is None
                    else fc2_promotion_k,
                )
                _fc2_swap_bf16_epilogue(
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
                    block_m,
                    block_n,
                )
            elif block_m == 64 or valid_m <= 64:
                final_swap_64 = _fc2_swap_mainloop(
                    barriers,
                    buffers,
                    l2_weight_scales,
                    task_expert,
                    task_n_block,
                    pipeline_tile,
                    l2_k,
                    l2_ws_stride_e,
                    l2_ws_stride_n,
                    l2_ws_stride_k,
                    64,
                    block_n,
                    max_num_imprecise_acc=max_num_imprecise_acc
                    if fc2_promotion_k is None
                    else fc2_promotion_k,
                )
                _fc2_swap_bf16_epilogue(
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
                    block_m,
                    block_n,
                )
            else:
                final_swap_128 = _fc2_swap_mainloop(
                    barriers,
                    buffers,
                    l2_weight_scales,
                    task_expert,
                    task_n_block,
                    pipeline_tile,
                    l2_k,
                    l2_ws_stride_e,
                    l2_ws_stride_n,
                    l2_ws_stride_k,
                    128,
                    block_n,
                    max_num_imprecise_acc=max_num_imprecise_acc
                    if fc2_promotion_k is None
                    else fc2_promotion_k,
                )
                _fc2_swap_bf16_epilogue(
                    final_swap_128,
                    token_src_metadata,
                    peer_combine_buffer_ptrs,
                    task_pool_block,
                    task_n_block,
                    valid_m,
                    l2_n,
                    max_tokens,
                    topk,
                    world_size,
                    128,
                    block_m,
                    block_n,
                )
            pipeline_tile += l2_k // block_k
            fc2_tile_complete(
                fc2_completion_state,
                task_expert,
                task_pool_block,
                gl.cdiv(valid_count, block_m),
                1 * l2_n_blocks,
            )

        if task_phase == 2 and fc2_epilogue_requires_full_sync:
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
            block_m,
        )


@gluon.jit
def math_swap_partition(
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
    l1_weight_scales,
    l2_weight_scales,
    expert_state,
    dispatch_counter,
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
    fc2_arrival_counter: gl.constexpr,
    fc2_epilogue_requires_full_sync: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    num_math_warps: gl.constexpr,
    staged_dispatch_handoff: gl.constexpr,
    fc1_promotion_k: gl.constexpr = 64,
    fc2_promotion_k: gl.constexpr = 64,
):
    """Compose the shared math body with the EP8 combine tail."""
    math_swap_body(
        barriers,
        buffers,
        l2_store_desc,
        l2_epilogue_buffer,
        l2_acts_sf,
        token_src_metadata,
        peer_combine_buffer_ptrs,
        route_weights,
        l2_arrival,
        l1_weight_scales,
        l2_weight_scales,
        expert_state,
        dispatch_counter,
        l1_k,
        l2_n,
        l2_k,
        E,
        l1_ws_stride_e,
        l1_ws_stride_n,
        l1_ws_stride_k,
        l2_ws_stride_e,
        l2_ws_stride_n,
        l2_ws_stride_k,
        l1_n_blocks,
        l2_n_blocks,
        num_experts_per_wave,
        num_sms,
        scheduler_count_capacity,
        scheduler_counts_per_lane,
        num_padded_sf_pool_tokens,
        max_tokens,
        topk,
        world_size,
        activation_clamp,
        has_activation_clamp,
        fast_math,
        fc2_arrival_counter,
        fc2_epilogue_requires_full_sync,
        block_m,
        block_n,
        num_math_warps,
        staged_dispatch_handoff,
        # Promotion is selected independently for FC1 and FC2. Candidate
        # values are accepted against the complete-output BF16 contract.
        max_num_imprecise_acc=fc1_promotion_k,
        fc2_promotion_k=fc2_promotion_k,
    )
    combine_partition(
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
        num_math_warps,
    )


# fp8/kernel.py


@gluon.jit
def fp8_fused_kernel(
    pool_acts,
    l1_a_desc,
    l1_sfa_desc,
    l1_b_desc,
    l2_store_desc,
    l2_a_desc,
    l2_sfa_desc,
    l2_b_desc,
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
    fc2_arrival_counter: gl.constexpr,
    fc2_epilogue_requires_full_sync: gl.constexpr,
    split_math_partitions: gl.constexpr,
    split_bm64_bn128_partitions: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    num_math_warps: gl.constexpr,
    math_regs: gl.constexpr,
    a_tma_regs: gl.constexpr,
    b_tma_regs: gl.constexpr,
    dispatch_regs: gl.constexpr,
    tokens_bound: gl.constexpr,
    fc1_promotion_k: gl.constexpr = 64,
    fc2_promotion_k: gl.constexpr = 64,
):
    """One resident CTA per SM for dispatch, FC1 publication, and FC2."""
    (
        a_buffers,
        b_buffers,
        sfa_lo_buffers,
        sfa_hi_buffers,
        l2_epilogue_buffer,
        l2_epilogue_buffer_1,
        l2_epilogue_buffer_2,
        l2_epilogue_buffer_3,
        stage_empty,
        stage_ready,
        fc1_amax_scratch_0,
        fc1_amax_scratch_1,
        fc1_scale_ready,
        fc1_scale_done,
        math_done,
    ) = allocate_math_pipeline(
        l1_a_desc,
        l1_b_desc,
        l2_store_desc,
        num_stages,
        block_m,
        4 if split_math_partitions else (2 if split_bm64_bn128_partitions else 1),
        split_bm64_bn128_partitions,
    )

    barriers = (stage_empty, stage_ready)
    buffers = (
        a_buffers,
        b_buffers,
        sfa_lo_buffers,
        sfa_hi_buffers,
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
    (
        dispatch_buffers_0,
        dispatch_buffers_1,
        dispatch_barriers_0,
        dispatch_barriers_1,
    ) = allocate_dispatch_pipeline(dispatch_acts_desc_0)
    staged_dispatch_handoff: gl.constexpr = tokens_bound <= 64

    if split_bm64_bn128_partitions:
        gl.warp_specialize(
            [
                (
                    math_split_bn128_partition,
                    (
                        barriers,
                        buffers,
                        l2_store_desc,
                        l2_epilogue_buffer,
                        fc1_amax_scratch_0,
                        fc1_amax_scratch_1,
                        fc1_scale_ready,
                        fc1_scale_done,
                        math_done,
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
                        l1_weight_scales,
                        l2_weight_scales,
                        expert_state,
                        dispatch_counter,
                        l1_n,
                        l1_k,
                        l2_n,
                        l2_k,
                        E,
                        l1_ws_stride_e,
                        l1_ws_stride_n,
                        l1_ws_stride_k,
                        l2_ws_stride_e,
                        l2_ws_stride_n,
                        l2_ws_stride_k,
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
                        0,
                        staged_dispatch_handoff,
                        fc1_promotion_k,
                        fc2_promotion_k,
                    ),
                ),
                (
                    math_split_bn128_partition,
                    (
                        barriers,
                        buffers,
                        l2_store_desc,
                        l2_epilogue_buffer_1,
                        fc1_amax_scratch_0,
                        fc1_amax_scratch_1,
                        fc1_scale_ready,
                        fc1_scale_done,
                        math_done,
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
                        l1_weight_scales,
                        l2_weight_scales,
                        expert_state,
                        dispatch_counter,
                        l1_n,
                        l1_k,
                        l2_n,
                        l2_k,
                        E,
                        l1_ws_stride_e,
                        l1_ws_stride_n,
                        l1_ws_stride_k,
                        l2_ws_stride_e,
                        l2_ws_stride_n,
                        l2_ws_stride_k,
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
                        1,
                        staged_dispatch_handoff,
                        fc1_promotion_k,
                        fc2_promotion_k,
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
                        dispatch_counter,
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
                        fc2_arrival_counter,
                        staged_dispatch_handoff,
                    ),
                ),
                (
                    b_producer_partition,
                    (
                        l1_b_desc,
                        l2_b_desc,
                        expert_state,
                        dispatch_counter,
                        barriers,
                        b_buffers,
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
                        block_m,
                        staged_dispatch_handoff,
                    ),
                ),
                (
                    dispatch_partition,
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
                        block_m,
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
                        staged_dispatch_handoff,
                    ),
                ),
                (
                    dispatch_partition,
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
                        block_m,
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
                        staged_dispatch_handoff,
                    ),
                ),
            ],
            [4, 1, 1, 1, 1],
            [
                math_regs,
                a_tma_regs,
                b_tma_regs,
                dispatch_regs,
                dispatch_regs,
            ],
        )
    elif split_math_partitions:
        gl.warp_specialize(
            [
                (
                    math_split_bm128_partition,
                    (
                        barriers,
                        buffers,
                        l2_store_desc,
                        l2_epilogue_buffer,
                        math_done,
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
                        l1_weight_scales,
                        l2_weight_scales,
                        expert_state,
                        dispatch_counter,
                        l1_n,
                        l1_k,
                        l2_n,
                        l2_k,
                        E,
                        l1_ws_stride_e,
                        l1_ws_stride_n,
                        l1_ws_stride_k,
                        l2_ws_stride_e,
                        l2_ws_stride_n,
                        l2_ws_stride_k,
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
                        0,
                        staged_dispatch_handoff,
                        fc1_promotion_k,
                        fc2_promotion_k,
                    ),
                ),
                (
                    math_split_bm128_partition,
                    (
                        barriers,
                        buffers,
                        l2_store_desc,
                        l2_epilogue_buffer_1,
                        math_done,
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
                        l1_weight_scales,
                        l2_weight_scales,
                        expert_state,
                        dispatch_counter,
                        l1_n,
                        l1_k,
                        l2_n,
                        l2_k,
                        E,
                        l1_ws_stride_e,
                        l1_ws_stride_n,
                        l1_ws_stride_k,
                        l2_ws_stride_e,
                        l2_ws_stride_n,
                        l2_ws_stride_k,
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
                        1,
                        staged_dispatch_handoff,
                        fc1_promotion_k,
                        fc2_promotion_k,
                    ),
                ),
                (
                    math_split_bm128_partition,
                    (
                        barriers,
                        buffers,
                        l2_store_desc,
                        l2_epilogue_buffer_2,
                        math_done,
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
                        l1_weight_scales,
                        l2_weight_scales,
                        expert_state,
                        dispatch_counter,
                        l1_n,
                        l1_k,
                        l2_n,
                        l2_k,
                        E,
                        l1_ws_stride_e,
                        l1_ws_stride_n,
                        l1_ws_stride_k,
                        l2_ws_stride_e,
                        l2_ws_stride_n,
                        l2_ws_stride_k,
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
                        2,
                        staged_dispatch_handoff,
                        fc1_promotion_k,
                        fc2_promotion_k,
                    ),
                ),
                (
                    math_split_bm128_partition,
                    (
                        barriers,
                        buffers,
                        l2_store_desc,
                        l2_epilogue_buffer_3,
                        math_done,
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
                        l1_weight_scales,
                        l2_weight_scales,
                        expert_state,
                        dispatch_counter,
                        l1_n,
                        l1_k,
                        l2_n,
                        l2_k,
                        E,
                        l1_ws_stride_e,
                        l1_ws_stride_n,
                        l1_ws_stride_k,
                        l2_ws_stride_e,
                        l2_ws_stride_n,
                        l2_ws_stride_k,
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
                        3,
                        staged_dispatch_handoff,
                        fc1_promotion_k,
                        fc2_promotion_k,
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
                        dispatch_counter,
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
                        fc2_arrival_counter,
                        staged_dispatch_handoff,
                    ),
                ),
                (
                    b_producer_partition,
                    (
                        l1_b_desc,
                        l2_b_desc,
                        expert_state,
                        dispatch_counter,
                        barriers,
                        b_buffers,
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
                        block_m,
                        staged_dispatch_handoff,
                    ),
                ),
                (
                    dispatch_partition,
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
                        block_m,
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
                        staged_dispatch_handoff,
                    ),
                ),
                (
                    dispatch_partition,
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
                        block_m,
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
                        staged_dispatch_handoff,
                    ),
                ),
            ],
            [4, 4, 4, 1, 1, 1, 1],
            [
                math_regs,
                math_regs,
                math_regs,
                a_tma_regs,
                b_tma_regs,
                dispatch_regs,
                dispatch_regs,
            ],
        )
    else:
        gl.warp_specialize(
            [
                (
                    math_swap_partition,
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
                        l1_weight_scales,
                        l2_weight_scales,
                        expert_state,
                        dispatch_counter,
                        l1_n,
                        l1_k,
                        l2_n,
                        l2_k,
                        E,
                        l1_ws_stride_e,
                        l1_ws_stride_n,
                        l1_ws_stride_k,
                        l2_ws_stride_e,
                        l2_ws_stride_n,
                        l2_ws_stride_k,
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
                        fc2_arrival_counter,
                        fc2_epilogue_requires_full_sync,
                        block_m,
                        block_n,
                        num_math_warps,
                        staged_dispatch_handoff,
                        fc1_promotion_k,
                        fc2_promotion_k,
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
                        dispatch_counter,
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
                        fc2_arrival_counter,
                        staged_dispatch_handoff,
                    ),
                ),
                (
                    b_producer_partition,
                    (
                        l1_b_desc,
                        l2_b_desc,
                        expert_state,
                        dispatch_counter,
                        barriers,
                        b_buffers,
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
                        block_m,
                        staged_dispatch_handoff,
                    ),
                ),
                (
                    dispatch_partition,
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
                        block_m,
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
                        staged_dispatch_handoff,
                    ),
                ),
                (
                    dispatch_partition,
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
                        block_m,
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
                        staged_dispatch_handoff,
                    ),
                ),
            ],
            [1, 1, 1, 1],
            [a_tma_regs, b_tma_regs, dispatch_regs, dispatch_regs],
        )
    dispatch_buffers_0._keep_alive()
    dispatch_buffers_1._keep_alive()
    dispatch_barriers_0._keep_alive()
    dispatch_barriers_1._keep_alive()


@gluon.jit
def fp8_large_tokens_kernel(
    pool_acts,
    l1_a_desc,
    l1_sfa_desc,
    l1_b_desc,
    l2_store_desc,
    l2_a_desc,
    l2_sfa_desc,
    l2_b_desc,
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
    math_regs: gl.constexpr,
    a_tma_regs: gl.constexpr,
    b_tma_regs: gl.constexpr,
    dispatch_regs: gl.constexpr,
    fc1_promotion_k: gl.constexpr = 128,
    fc2_promotion_k: gl.constexpr = 64,
):
    """One resident BM64/BN256 two-partition CTA per SM for M >= 256."""
    (
        a_buffers,
        b_buffers,
        sfa_lo_buffers,
        sfa_hi_buffers,
        l2_epilogue_buffer,
        l2_epilogue_buffer_1,
        l2_epilogue_buffer_2,
        l2_epilogue_buffer_3,
        stage_empty,
        stage_ready,
        fc1_amax_scratch_0,
        fc1_amax_scratch_1,
        fc1_scale_ready,
        fc1_scale_done,
        math_done,
    ) = allocate_math_pipeline(
        l1_a_desc, l1_b_desc, l2_store_desc, num_stages, _BLOCK_M, 2, False
    )

    barriers = (stage_empty, stage_ready)
    buffers = (
        a_buffers,
        b_buffers,
        sfa_lo_buffers,
        sfa_hi_buffers,
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
    (
        dispatch_buffers_0,
        dispatch_buffers_1,
        dispatch_barriers_0,
        dispatch_barriers_1,
    ) = allocate_dispatch_pipeline(dispatch_acts_desc_0)

    gl.warp_specialize(
        [
            (
                math_split_bn256_partition,
                (
                    barriers,
                    buffers,
                    l2_store_desc,
                    l2_epilogue_buffer,
                    math_done,
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
                    l1_weight_scales,
                    l2_weight_scales,
                    expert_state,
                    dispatch_counter,
                    l1_n,
                    l1_k,
                    l2_n,
                    l2_k,
                    E,
                    l1_ws_stride_e,
                    l1_ws_stride_n,
                    l1_ws_stride_k,
                    l2_ws_stride_e,
                    l2_ws_stride_n,
                    l2_ws_stride_k,
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
                    0,
                    False,
                    fc1_promotion_k,
                    fc2_promotion_k,
                ),
            ),
            (
                math_split_bn256_partition,
                (
                    barriers,
                    buffers,
                    l2_store_desc,
                    l2_epilogue_buffer_1,
                    math_done,
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
                    l1_weight_scales,
                    l2_weight_scales,
                    expert_state,
                    dispatch_counter,
                    l1_n,
                    l1_k,
                    l2_n,
                    l2_k,
                    E,
                    l1_ws_stride_e,
                    l1_ws_stride_n,
                    l1_ws_stride_k,
                    l2_ws_stride_e,
                    l2_ws_stride_n,
                    l2_ws_stride_k,
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
                    1,
                    False,
                    fc1_promotion_k,
                    fc2_promotion_k,
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
                    dispatch_counter,
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
                    True,
                    False,
                ),
            ),
            (
                b_producer_partition,
                (
                    l1_b_desc,
                    l2_b_desc,
                    expert_state,
                    dispatch_counter,
                    barriers,
                    b_buffers,
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
                    _BLOCK_M,
                    False,
                ),
            ),
            (
                dispatch_partition,
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
                    _BLOCK_M,
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
                    False,
                ),
            ),
            (
                dispatch_partition,
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
                    _BLOCK_M,
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
                    False,
                ),
            ),
        ],
        [4, 1, 1, 1, 1],
        [math_regs, a_tma_regs, b_tma_regs, dispatch_regs, dispatch_regs],
    )
    dispatch_buffers_0._keep_alive()
    dispatch_buffers_1._keep_alive()
    dispatch_barriers_0._keep_alive()
    dispatch_barriers_1._keep_alive()


# fp8/api.py


def fused_moe(
    ctx: SymmetricContext,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    x_sf: torch.Tensor | None = None,
    tokens_bound: int | None = None,
    config: MegaMoEConfig | None = None,
    num_sms: int | None = None,
    activation_clamp: float | None = None,
    fast_math: bool | None = None,
    workspace: FusedWorkspace | None = None,
    routed_scaling_factor: float = 1.0,
) -> FusedResult:
    """Compute EP8 FP8 MoE from BF16 or explicitly scaled FP8 inputs.

    All ranks use the same token bound and policy. ``config`` overrides the
    selected launch geometry; explicit activation options override its numerical
    settings. Reuse ``result.workspace`` after the first call. FC1 weights must
    come from prepare_weights; FC2 weights and block scales remain canonical.
    Registration, control reset, dispatch, FC1/activation/FP8, FC2, peer scatter
    and final top-k sum all belong to this API.
    """
    group_n = group_k = 128
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
        raise ValueError("w1 must have shape [experts_per_rank, N, K]")
    E, N, weight_k = w1.shape
    if E != ctx.experts_per_rank:
        raise ValueError("w1 must contain exactly this rank's experts")
    if weight_k != K:
        raise ValueError("dispatch input and w1 K dimensions must match")
    if w1.dtype != torch.float8_e4m3fn:
        raise ValueError("w1 must have dtype float8_e4m3fn")
    if K % group_k or N % 256:
        raise ValueError("H must be divisible by 128 and FC1 2I by 256")
    intermediate_hidden = N // 2
    if w1_scale.shape != (E, N // group_n, K // group_k):
        raise ValueError("w1_scale must be [E, N // 128, K // 128]")
    if w1_scale.dtype != torch.float32:
        raise ValueError("w1_scale must have dtype float32")
    if w2.shape != (E, K, intermediate_hidden):
        raise ValueError("w2 must be [E, H, I]")
    if w2.dtype != torch.float8_e4m3fn:
        raise ValueError("w2 must have dtype float8_e4m3fn")
    if w2_scale.shape != (
        E,
        K // 128,
        intermediate_hidden // 128,
    ):
        raise ValueError("w2_scale must be [E, H/128, I/128]")
    if w2_scale.dtype != torch.float32:
        raise ValueError("w2_scale must have dtype float32")
    tensors = (
        topk_idx,
        topk_weights,
        w1,
        w1_scale,
        w2,
        w2_scale,
    )
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
    device_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    if num_sms is None:
        num_sms = device_sms
    if not 0 < num_sms <= device_sms:
        raise ValueError(f"num_sms must be in [1, {device_sms}], got {num_sms}")
    if config is not None and not isinstance(config, MegaMoEConfig):
        raise ValueError("config must be a MegaMoEConfig")
    if ctx.world_size != 8:
        raise ValueError("fused FP8 requires eight ranks")
    selected = select(
        topology="ep8",
        fmt="fp8",
        shape=Shape(K, intermediate_hidden, ctx.num_experts, ctx.topk),
        tokens_bound=tokens_bound,
        num_sms=num_sms,
        override=config,
    )
    changes = {
        key: value
        for key, value in (
            ("fast_math", fast_math),
            ("activation_clamp", activation_clamp),
        )
        if value is not None
    }
    if changes:
        selected = select(
            topology="ep8",
            fmt="fp8",
            shape=Shape(K, intermediate_hidden, ctx.num_experts, ctx.topk),
            tokens_bound=tokens_bound,
            num_sms=num_sms,
            override=replace(selected.config, **changes),
        )
    fast_math, activation_clamp = (
        selected.config.fast_math,
        selected.config.activation_clamp,
    )
    launch_config = selected.launch
    # The fused implementation keeps the four evaluated math strategies:
    # swapAB BM64/BN128, normal BM64/BN128 (2 partitions), normal
    # BM64/BN256 (2 partitions), and normal BM128/BN256 (4 partitions).
    # In particular, every fused normal configuration is partitioned.
    split_math_partitions = (
        launch_config.block_m == _SPLIT_BLOCK_M_VALUE
        and launch_config.block_n == _NORMAL_BLOCK_N_VALUE
        and launch_config.num_math_warps == 16
        and not launch_config.use_swap_ab
        and not launch_config.use_split_bn256
    )
    split_bm64_bn128_partitions = (
        launch_config.block_m == _BLOCK_M_VALUE
        and launch_config.block_n == _BLOCK_N_VALUE
        and launch_config.num_math_warps == 8
        and not launch_config.use_swap_ab
        and not launch_config.use_split_bn256
    )
    block_m = launch_config.block_m
    block_n = launch_config.block_n
    num_stages = launch_config.num_stages
    num_experts_per_wave = launch_config.num_experts_per_wave
    maxnreg = launch_config.launch_maxnreg
    dispatch_regs = launch_config.dispatch_register_budget
    if not 32 <= maxnreg <= 255:
        raise ValueError(f"maxnreg must be in [32, 255], got {maxnreg}")
    if not 24 <= dispatch_regs <= 255:
        raise ValueError("dispatch_regs must be in [24, 255]")
    activation_clamp = float(activation_clamp)
    if math.isnan(activation_clamp) or activation_clamp <= 0:
        raise ValueError("activation_clamp must be positive or infinity")

    routed_scaling_factor = float(routed_scaling_factor)
    if not math.isfinite(routed_scaling_factor):
        raise ValueError("routed_scaling_factor must be finite")
    check_policy_agreement(
        ctx,
        backend="fp8",
        tokens_bound=tokens_bound,
        shape=(K, intermediate_hidden, ctx.num_experts, ctx.topk),
        capacity=ctx.max_tokens,
        launch=launch_config,
        grid=num_sms,
        maxnreg=maxnreg,
        dispatch_regs=dispatch_regs,
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
    # Keep the config-independent pool reservation so diagnostic BM128
    # specializations can reuse a production BM64 workspace without moving
    # payload buffers or retaining descriptors for a stale block shape.
    num_pool_rows = _host_align(
        max_global_routes + E * (_MAX_CANDIDATE_BLOCK_M_VALUE - 1),
        _POOL_ALIGNMENT_VALUE,
    )
    max_pool_blocks = num_pool_rows // _BLOCK_M_VALUE
    num_padded_sf_pool_tokens = max_pool_blocks * _SF_BLOCK_M_VALUE
    device = x.device
    gemm_descriptor_shapes = (
        (_BLOCK_M_VALUE, _BLOCK_N_VALUE),
        (_BLOCK_M_VALUE, _NORMAL_BLOCK_N_VALUE),
        (_SPLIT_BLOCK_M_VALUE, _BLOCK_N_VALUE),
        (_SPLIT_BLOCK_M_VALUE, _NORMAL_BLOCK_N_VALUE),
    )
    if split_bm64_bn128_partitions:
        gemm_descriptor_shapes = (
            (_BLOCK_M_VALUE, _BLOCK_N_VALUE // 2),
            *gemm_descriptor_shapes,
        )
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

        gemm_descriptor_sets = []
        for descriptor_block_m, descriptor_block_n in gemm_descriptor_shapes:
            a_layout = gl.NVMMASharedLayout.get_default_for(
                [descriptor_block_m, _BLOCK_K_VALUE],
                gl.float8e4nv,
            )
            sfa_layout = gl.NVMMASharedLayout.get_default_for(
                [descriptor_block_m],
                gl.float32,
            )
            b_layout = gl.NVMMASharedLayout.get_default_for(
                [descriptor_block_n, _BLOCK_K_VALUE],
                gl.float8e4nv,
            )
            l2_store_layout = gl.NVMMASharedLayout.get_default_for(
                [descriptor_block_m, descriptor_block_n // 2],
                gl.float8e4nv,
            )
            gemm_descriptor_sets.append(
                GemmDescriptors(
                    block_m=descriptor_block_m,
                    block_n=descriptor_block_n,
                    l1_a_desc=TensorDescriptor.from_tensor(
                        pool.acts,
                        [descriptor_block_m, _BLOCK_K_VALUE],
                        a_layout,
                    ),
                    l1_sfa_desc=TensorDescriptor.from_tensor(
                        pool.acts_sf_mn_major.view(-1),
                        [descriptor_block_m],
                        sfa_layout,
                    ),
                    l1_b_desc=TensorDescriptor.from_tensor(
                        w1.view(E * N, K),
                        [descriptor_block_n, _BLOCK_K_VALUE],
                        b_layout,
                    ),
                    l2_store_desc=TensorDescriptor.from_tensor(
                        l2_acts,
                        [descriptor_block_m, descriptor_block_n // 2],
                        l2_store_layout,
                    ),
                    l2_a_desc=TensorDescriptor.from_tensor(
                        l2_acts,
                        [descriptor_block_m, _BLOCK_K_VALUE],
                        a_layout,
                    ),
                    l2_sfa_desc=TensorDescriptor.from_tensor(
                        l2_acts_sf_mn_major.view(-1),
                        [descriptor_block_m],
                        sfa_layout,
                    ),
                    l2_b_desc=TensorDescriptor.from_tensor(
                        w2.view(E * K, intermediate_hidden),
                        [descriptor_block_n, _BLOCK_K_VALUE],
                        b_layout,
                    ),
                )
            )
        dispatch_descs = create_dispatch_descriptors(
            ctx,
            pool.acts,
        )
        workspace = FusedWorkspace(
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
            gemm_descriptor_sets=tuple(gemm_descriptor_sets),
            dispatch_descs=dispatch_descs,
            l1_weight_data_ptr=w1.data_ptr(),
            l2_weight_data_ptr=w2.data_ptr(),
            max_pool_blocks=max_pool_blocks,
            num_pool_rows=num_pool_rows,
            num_padded_sf_pool_tokens=num_padded_sf_pool_tokens,
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
            tuple(
                (item.block_m, item.block_n) for item in workspace.gemm_descriptor_sets
            )
            == gemm_descriptor_shapes,
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
    descriptor_set = next(
        (
            item
            for item in workspace.gemm_descriptor_sets
            if item.block_m == block_m and item.block_n == block_n
        ),
        None,
    )
    if descriptor_set is None:
        raise ValueError(f"fused workspace has no BM{block_m}/BN{block_n} descriptors")
    l1_a_desc = descriptor_set.l1_a_desc
    l1_sfa_desc = descriptor_set.l1_sfa_desc
    l1_b_desc = descriptor_set.l1_b_desc
    l2_store_desc = descriptor_set.l2_store_desc
    if split_bm64_bn128_partitions:
        fragment_descriptor_set = next(
            (
                item
                for item in workspace.gemm_descriptor_sets
                if item.block_m == _BLOCK_M_VALUE
                and item.block_n == _BLOCK_N_VALUE // 2
            ),
            None,
        )
        if fragment_descriptor_set is None:
            raise ValueError("fused workspace has no 64x32 FC1 store descriptor")
        l2_store_desc = fragment_descriptor_set.l2_store_desc
    elif split_math_partitions or launch_config.use_split_bn256:
        fragment_descriptor_set = next(
            (
                item
                for item in workspace.gemm_descriptor_sets
                if item.block_m == _BLOCK_M_VALUE and item.block_n == _BLOCK_N_VALUE
            ),
            None,
        )
        if fragment_descriptor_set is None:
            raise ValueError("fused workspace has no 64x64 FC1 store descriptor")
        l2_store_desc = fragment_descriptor_set.l2_store_desc
    l2_a_desc = descriptor_set.l2_a_desc
    l2_sfa_desc = descriptor_set.l2_sfa_desc
    l2_b_desc = descriptor_set.l2_b_desc
    dispatch_descs = workspace.dispatch_descs

    l1_n_blocks = _host_cdiv(N, block_n)
    l2_n_blocks = _host_cdiv(K, block_n)
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
    kernel_tensor_args = (
        pool.acts,
        l1_a_desc,
        l1_sfa_desc,
        l1_b_desc,
        l2_store_desc,
        l2_a_desc,
        l2_sfa_desc,
        l2_b_desc,
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
    )
    kernel_shape_args = (
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
        num_sms,
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
    )
    if launch_config.use_split_bn256:
        compiled = fp8_large_tokens_kernel[(num_sms,)](
            *kernel_tensor_args,
            *kernel_shape_args,
            launch_config.math_register_budget,
            _LARGE_NORMAL_TMA_REGS_VALUE,
            _LARGE_NORMAL_TMA_REGS_VALUE,
            dispatch_regs,
            fc1_promotion_k=launch_config.fc1_promotion_k,
            fc2_promotion_k=launch_config.fc2_promotion_k,
            num_warps=4,
            maxnreg=maxnreg,
        )
    else:
        # Compile the selected math partitions with their matching producer budgets.
        compiled = fp8_fused_kernel[(num_sms,)](
            *kernel_tensor_args,
            *kernel_shape_args,
            launch_config.fc2_arrival_counter,
            launch_config.fc2_epilogue_requires_full_sync,
            split_math_partitions,
            split_bm64_bn128_partitions,
            block_m,
            block_n,
            launch_config.num_math_warps,
            launch_config.math_register_budget,
            launch_config.non_epilogue_register_budget,
            launch_config.non_epilogue_register_budget,
            dispatch_regs,
            tokens_bound=tokens_bound,
            fc1_promotion_k=launch_config.fc1_promotion_k,
            fc2_promotion_k=launch_config.fc2_promotion_k,
            num_warps=(
                4
                if split_math_partitions or split_bm64_bn128_partitions
                else launch_config.num_math_warps
            ),
            maxnreg=maxnreg,
        )
    return FusedResult(
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
        config=launch_config,
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
    "SymmetricContext",
    "RegisteredInputs",
    "FusedResult",
    "FusedWorkspace",
]
