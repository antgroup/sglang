# SPDX-License-Identifier: Apache-2.0

"""Canonical SM90 Gluon MegaMoE implementation.

This file owns every Gluon partition used by the fused path, including
dispatch, context management, grouped GEMM, scatter, and combine.

The only GEMM ABI owned by this module is the persistent compact 1D-by-2D
path.  Exactly one CTA is launched per SM and each CTA walks a deterministic
sequence of logical M/N tiles over a BLOCK_M-padded activation pool and its
MN-major scale pool.

All A, SFA, and B tiles use Hopper TMA.  Every scheduler acquires packed
per-expert states and derives compact offsets locally.  The A producer owns
both readiness edges: dispatch-to-FC1 and FC1-publication-to-FC2.  FC1 applies
the granularity-8 gate/up SwiGLU epilogue, route weight, per-row/per-64 FP8
quantization, and release publication before FC2 consumes the compact L2
pool.  FC2 scatters every valid route directly back to its source rank and
reduces the source-local top-k slots into the final token-major BF16 output.
"""

import math
from dataclasses import dataclass, replace

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

try:
    from triton.experimental.gluon.language import barrier as _gluon_partition_barrier
except ImportError:
    # Gluon 3.6 exported the same CTA synchronization builtin under this name.
    from triton.experimental.gluon.language import (
        thread_barrier as _gluon_partition_barrier,
    )


_MINIMUM_TRITON_VERSION = Version("3.6.0")
_SM90_MEGA_MOE_PRE_DISPATCH_GROUP_SIZE = 128
_SM90_MEGA_MOE_PRE_DISPATCH_GROUPS_PER_CTA = 64
_SM90_MEGA_MOE_PRE_DISPATCH_NUM_WARPS = 32
_SM90_MEGA_MOE_PRE_DISPATCH_THREADS = _SM90_MEGA_MOE_PRE_DISPATCH_NUM_WARPS * 32
_SM90_MEGA_MOE_FUSED_RESET_NUM_WARPS = 8
_SM90_MEGA_MOE_FUSED_RESET_BLOCK_SIZE = _SM90_MEGA_MOE_FUSED_RESET_NUM_WARPS * 32
_SM90_MEGA_MOE_SWAP_AB_MAX_TOKENS_PER_RANK = 255
_DSV4_FLASH_HIDDEN = 4096
_DSV4_FLASH_INTERMEDIATE_HIDDEN = 2048
_DSV4_FLASH_NUM_EXPERTS = 256
_DSV4_FLASH_SWAP_AB_MAX_TOKENS_PER_RANK = 128
_DSV4_FLASH_BM64_BN256_MAX_TOKENS_PER_RANK = 8192
_DSV4_PRO_HIDDEN = 7168
_DSV4_PRO_INTERMEDIATE_HIDDEN = 3072
_DSV4_PRO_NUM_EXPERTS = 384
_DSV4_PRO_THREE_STAGE_TOKENS_PER_RANK = 1024
_SM90_MEGA_MOE_SMEM_CAPACITY = 232448
_SM90_MEGA_MOE_SMEM_ALIGNMENT = 1024
_SM90_MEGA_MOE_GLUON_SPLIT_MN_MAX_STAGES = 2
_GLUON_WARP_SPECIALIZATION_GROUP_WARPS = 4
_GLUON_WARP_SPECIALIZATION_PADDING_REGS = 16


def _should_use_swap_ab_for_tokens_per_rank(num_tokens_per_rank: int) -> bool:
    """Select swapAB for the DeepGEMM-compatible small-token runtime path."""
    return 0 < num_tokens_per_rank <= _SM90_MEGA_MOE_SWAP_AB_MAX_TOKENS_PER_RANK


def _should_use_large_normal_for_tokens_per_rank(
    num_tokens_per_rank: int,
) -> bool:
    """Select the isolated b505975 normal partition above the swap boundary."""
    return num_tokens_per_rank > _SM90_MEGA_MOE_SWAP_AB_MAX_TOKENS_PER_RANK


def _is_dsv4_flash_shape(
    *,
    hidden: int,
    intermediate_hidden: int,
    num_experts: int,
) -> bool:
    """Return whether the live shape is the DSV4 Flash specialization."""
    return (
        hidden == _DSV4_FLASH_HIDDEN
        and intermediate_hidden == _DSV4_FLASH_INTERMEDIATE_HIDDEN
        and num_experts == _DSV4_FLASH_NUM_EXPERTS
    )


def _is_dsv4_pro_shape(
    *,
    hidden: int,
    intermediate_hidden: int,
    num_experts: int,
) -> bool:
    """Return whether the live shape is the DSV4 Pro specialization."""
    return (
        hidden == _DSV4_PRO_HIDDEN
        and intermediate_hidden == _DSV4_PRO_INTERMEDIATE_HIDDEN
        and num_experts == _DSV4_PRO_NUM_EXPERTS
    )


@dataclass(frozen=True)
class _SM90MegaMoEConfig:
    """Measured Gluon host specialization selected from the live input."""

    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_math_warps: int
    num_experts_per_wave: int
    use_swap_ab: bool
    use_large_normal_partition: bool
    expected_tokens_per_expert: float
    l2_arrival_counter: bool
    l2_epilogue_requires_full_sync: bool
    math_register_budget: int
    dispatch_register_budget: int
    non_epilogue_register_budget: int
    launch_maxnreg: int

    @property
    def mode(self) -> str:
        if self.use_large_normal_partition:
            return "normal_bm64_bn256_large"
        orientation = "swap" if self.use_swap_ab else "normal"
        return f"{orientation}_bm{self.block_m}_bn{self.block_n}"


def _host_align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _get_gluon_launch_maxnreg(
    *,
    num_math_warps: int,
    math_register_budget: int,
    specialized_register_budgets: tuple[int, ...],
) -> int:
    """Match Triton's full-warpgroup padding in the launch register cap."""
    num_specialized_warps = len(specialized_register_budgets)
    num_padded_specialized_warps = _host_align(
        num_specialized_warps,
        _GLUON_WARP_SPECIALIZATION_GROUP_WARPS,
    )
    num_padding_warps = num_padded_specialized_warps - num_specialized_warps
    total_warps = num_math_warps + num_padded_specialized_warps
    total_register_budget = (
        num_math_warps * math_register_budget
        + sum(specialized_register_budgets)
        + num_padding_warps * _GLUON_WARP_SPECIALIZATION_PADDING_REGS
    )
    return _host_align(
        (total_register_budget + total_warps - 1) // total_warps,
        8,
    )


def _get_sm90_mega_moe_num_stages(
    *,
    hidden: int,
    num_experts: int,
    block_m: int,
    block_n: int,
    num_math_warps: int,
) -> int:
    """Mirror DeepGEMM's SM90 shared-memory pipeline calculation."""
    block_k = 128
    num_dispatch_warps = 2
    num_epilogue_warps = num_math_warps
    smem_expert_count = _host_align(
        num_experts * 4,
        _SM90_MEGA_MOE_SMEM_ALIGNMENT,
    )
    smem_send_buffers = _host_align(
        hidden * num_dispatch_warps,
        _SM90_MEGA_MOE_SMEM_ALIGNMENT,
    )
    smem_cd_l1 = block_m * (block_n // 2)
    smem_cd_l2 = block_m * block_n * 2
    # Unlike DeepGEMM's CUDA swap path, Gluon keeps the FP32 accumulator in
    # registers and only stages the quantized FC1 tile in shared memory.
    smem_cd = _host_align(
        max(smem_cd_l1, smem_cd_l2),
        _SM90_MEGA_MOE_SMEM_ALIGNMENT,
    )
    smem_sfa_per_stage = _host_align(2 * block_m * 4, 128)
    smem_per_stage = block_m * block_k + block_n * block_k + smem_sfa_per_stage
    smem_barriers_fixed = (num_dispatch_warps + 2 * num_epilogue_warps) * 8
    smem_fixed = smem_expert_count + smem_send_buffers + smem_cd + smem_barriers_fixed
    num_stages = (_SM90_MEGA_MOE_SMEM_CAPACITY - smem_fixed) // (smem_per_stage + 16)
    if num_stages < 2:
        raise ValueError("selected SM90 config has fewer than two stages")
    return num_stages


def _get_sm90_mega_moe_experts_per_wave(
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


def _get_sm90_mega_moe_config(
    *,
    num_tokens_per_rank: int,
    hidden: int,
    intermediate_hidden: int,
    num_experts: int,
    num_experts_per_rank: int,
    topk: int,
    num_sms: int,
    num_stages: int | None = None,
    num_experts_per_wave: int | None = None,
) -> _SM90MegaMoEConfig:
    """Select the fastest verified Gluon family for the live SM90 input."""
    expected = float(num_tokens_per_rank) * topk / num_experts_per_rank
    is_dsv4_flash = _is_dsv4_flash_shape(
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
    )
    is_dsv4_pro = _is_dsv4_pro_shape(
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
    )
    split_mn = (
        is_dsv4_flash
        and num_tokens_per_rank > _DSV4_FLASH_BM64_BN256_MAX_TOKENS_PER_RANK
    )
    decode_split_n = not split_mn
    if is_dsv4_flash:
        large_normal_requested = (
            _DSV4_FLASH_SWAP_AB_MAX_TOKENS_PER_RANK
            < num_tokens_per_rank
            <= _DSV4_FLASH_BM64_BN256_MAX_TOKENS_PER_RANK
        )
        use_swap_ab = 0 < num_tokens_per_rank <= _DSV4_FLASH_SWAP_AB_MAX_TOKENS_PER_RANK
    else:
        large_normal_requested = (
            decode_split_n
            and _should_use_large_normal_for_tokens_per_rank(num_tokens_per_rank)
        )
        use_swap_ab = (
            decode_split_n
            and not large_normal_requested
            and _should_use_swap_ab_for_tokens_per_rank(num_tokens_per_rank)
        )
    decode_bn256 = (
        decode_split_n
        and not use_swap_ab
        and (
            (is_dsv4_flash and large_normal_requested)
            or (not is_dsv4_flash and intermediate_hidden >= 3072 and expected >= 0.25)
        )
        and (2 * intermediate_hidden) % 256 == 0
        and hidden % 256 == 0
    )
    use_large_normal_partition = large_normal_requested and decode_bn256
    wide_swap = use_swap_ab and expected >= 256.0
    block_m = 128 if split_mn else 64
    block_n = 256 if split_mn or decode_bn256 or wide_swap else 128
    num_math_warps = 16 if split_mn or wide_swap else 8
    selected_num_stages = _get_sm90_mega_moe_num_stages(
        hidden=hidden,
        num_experts=num_experts,
        block_m=block_m,
        block_n=block_n,
        num_math_warps=num_math_warps,
    )
    if is_dsv4_flash and large_normal_requested and num_tokens_per_rank >= 8192:
        selected_num_stages = 3
    if is_dsv4_pro and num_tokens_per_rank == _DSV4_PRO_THREE_STAGE_TOKENS_PER_RANK:
        selected_num_stages = 3
    # DeepGEMM aliases its FC1 FP8 and FC2 BF16 CD staging storage.  Gluon's
    # FC1 descriptor buffer and layout-conversion scratch are distinct, so a
    # three-stage BM128/BN256 kernel exceeds Hopper's SMEM limit by 60 bytes.
    # The production DSV4 specialization already selects two stages; cap the
    # smaller correctness shapes at the same backend-safe specialization.
    if split_mn:
        selected_num_stages = min(
            selected_num_stages,
            _SM90_MEGA_MOE_GLUON_SPLIT_MN_MAX_STAGES,
        )
    if wide_swap:
        selected_num_stages = 3
    if num_stages is not None:
        if not 2 <= num_stages <= 8:
            raise ValueError("num_stages must be in [2, 8]")
        if split_mn and num_stages > _SM90_MEGA_MOE_GLUON_SPLIT_MN_MAX_STAGES:
            raise ValueError("Gluon BM128/BN256 requires num_stages=2")
        selected_num_stages = num_stages
    selected_experts_per_wave = _get_sm90_mega_moe_experts_per_wave(
        expected_tokens_per_expert=expected,
        num_experts_per_rank=num_experts_per_rank,
        intermediate_hidden=intermediate_hidden,
        block_m=block_m,
        block_n=block_n,
        num_sms=num_sms,
    )
    # The DeepGEMM tail-ratio search chooses eight for DSV4-Flash M=8.
    # Gluon's fused scheduler is measurably faster with a 9/9/9/5 wave split
    # than four exact waves of eight, while adjacent shapes retain the generic
    # policy.  Keep this as a narrowly measured implementation-specific fix.
    if is_dsv4_flash and num_tokens_per_rank == 8:
        selected_experts_per_wave = 9
    if num_experts_per_wave is not None:
        if not 0 < num_experts_per_wave <= num_experts_per_rank:
            raise ValueError("num_experts_per_wave must be in [1, E]")
        selected_experts_per_wave = num_experts_per_wave
    many_math_warps = num_math_warps == 16
    math_register_budget = 112 if many_math_warps else 168
    dispatch_register_budget = 32 if many_math_warps else 48
    non_epilogue_register_budget = 24 if many_math_warps else 40
    launch_maxnreg = _get_gluon_launch_maxnreg(
        num_math_warps=num_math_warps,
        math_register_budget=math_register_budget,
        specialized_register_budgets=(
            non_epilogue_register_budget,
            non_epilogue_register_budget,
            dispatch_register_budget,
            dispatch_register_budget,
        ),
    )
    l2_arrival_counter = split_mn or (
        decode_split_n and block_n == 256 and 4 <= num_tokens_per_rank <= 128
    )
    return _SM90MegaMoEConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=128,
        num_stages=selected_num_stages,
        num_math_warps=num_math_warps,
        num_experts_per_wave=selected_experts_per_wave,
        use_swap_ab=use_swap_ab,
        use_large_normal_partition=use_large_normal_partition,
        expected_tokens_per_expert=expected,
        l2_arrival_counter=l2_arrival_counter,
        l2_epilogue_requires_full_sync=(
            not l2_arrival_counter and not use_large_normal_partition
        ),
        math_register_budget=math_register_budget,
        dispatch_register_budget=dispatch_register_budget,
        non_epilogue_register_budget=non_epilogue_register_budget,
        launch_maxnreg=launch_maxnreg,
    )


@dataclass(frozen=True)
class SM90MegaMoESingleRankPool:
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
        return self.expert_pool_block_offsets_for(_GLUON_BLOCK_M_VALUE)

    def expert_pool_block_offsets_for(self, block_m: int) -> torch.Tensor:
        """Derive the padded expert-pool prefix for one selected BLOCK_M."""
        if block_m not in (_GLUON_BLOCK_M_VALUE, _GLUON_SPLIT_BLOCK_M_VALUE):
            raise ValueError("block_m must be 64 or 128")
        counts = self.expert_recv_count.to(torch.int64)
        blocks = torch.div(
            counts + block_m - 1,
            block_m,
            rounding_mode="floor",
        )
        return torch.cat((blocks.new_zeros(1), torch.cumsum(blocks, dim=0)))


@dataclass(frozen=True)
class SM90MegaMoEGemmDescriptorSet:
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
class SM90MegaMoEFusedFC1Workspace:
    """Reusable storage for the complete fused MegaMoE path."""

    pool: SM90MegaMoESingleRankPool
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
    gemm_descriptor_sets: tuple[SM90MegaMoEGemmDescriptorSet, ...]
    dispatch_descs: tuple[object, ...]
    l1_weight_data_ptr: int
    l2_weight_data_ptr: int
    max_pool_blocks: int
    num_pool_rows: int
    num_padded_sf_pool_tokens: int


@dataclass(frozen=True)
class SM90MegaMoEPreDispatchResult:
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


@dataclass(frozen=True)
class SM90MegaMoEFusedFC1Result:
    """Artifacts from the complete registered MegaMoE chain.

    ``output`` is the observable token-major BF16 result after peer FC2 scatter
    and source-local top-k reduction.
    In ``combine_buffer``, only slots whose registered top-k index is valid
    are defined; invalid slots are deliberately not cleared or reduced.
    """

    pool: SM90MegaMoESingleRankPool
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
    workspace: SM90MegaMoEFusedFC1Workspace
    pre_dispatch: SM90MegaMoEPreDispatchResult
    config: _SM90MegaMoEConfig
    compiled: object


class SM90MegaMoESymmetricContext:
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
def _sm90_mega_moe_fused_control_reset_kernel(
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
def _sm90_mega_moe_pre_dispatch_kernel(
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
):
    """Register BF16 or block-scaled FP8 inputs in symmetric memory.

    The 64x128 register tile maps one half warp to each per-128 group.  With
    32 warps this is the same 1024-thread, one-token-per-CTA decomposition as
    DeepGEMM's SM90 pre-dispatch kernel for hidden sizes up to 8192.
    """
    bid = gl.program_id(0)
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


def create_sm90_symmetric_dispatch_tma_oob_3d_descriptors(
    ctx: SM90MegaMoESymmetricContext,
    pool_acts: torch.Tensor,
):
    """Create one 3D TMA tile per token, padding K with hardware OOB.

    The logical tensor is ``[token, K // 128, 128]`` while the TMA box is
    ``[1, next_power_of_2(K // 128), 128]``.  For DSV4Pro K=7168 this maps a
    56-group token into a 64-group (8 KiB) transaction.  TMA zero-fills the
    final eight groups on load and suppresses them on store, so the padded box
    never aliases the following token or pool row.
    """
    if ctx.world_size != 8:
        raise ValueError("the Gluon 3D OOB TMA dispatch specialization requires EP8")
    if ctx.hidden <= 0 or ctx.hidden % 128:
        raise ValueError("3D OOB TMA dispatch requires K divisible by 128")
    sf_groups = ctx.hidden // 128
    padded_sf_groups = triton.next_power_of_2(sf_groups)
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
def _load_packed_expert_state_acquire(expert_state_ptr):
    """Issue a non-RMW system-scope acquire load for packed expert state."""
    return gl.inline_asm_elementwise(
        "ld.global.sys.acquire.b64 $0, [$1];",
        "=l,l",
        [expert_state_ptr],
        dtype=gl.int64,
        is_pure=False,
        pack=1,
    )


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
    expected: gl.constexpr,
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
    target: gl.constexpr = expected * world_size
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
def _fused_pool_dispatch_partition(
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
    dispatch_count_capacity: gl.constexpr = triton.next_power_of_2(num_experts)
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
    source_rank_offsets = gl.arange(0, 32, layout=layout)
    valid_source_rank = source_rank_offsets < world_size

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

        source_rank_counts = gl.load(
            symmetric_recv_count
            + source_rank_offsets * experts_per_rank
            + current_expert,
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


def create_sm90_mega_moe_symmetric_context(
    *,
    max_tokens: int,
    hidden: int,
    num_experts: int,
    topk: int,
    group_name: str | None = None,
) -> SM90MegaMoESymmetricContext:
    """Collectively create the single-node peer buffers for MegaMoE."""
    import torch.distributed as dist

    _require_sm90_and_gluon()
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before rendezvous")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if group_name is None:
        group_name = dist.group.WORLD.group_name
    return SM90MegaMoESymmetricContext(
        max_tokens=max_tokens,
        hidden=hidden,
        num_experts=num_experts,
        topk=topk,
        world_size=world_size,
        rank=rank,
        device=torch.device("cuda", torch.cuda.current_device()),
        group_name=group_name,
    )


def _require_sm90_and_gluon() -> None:
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


_GLUON_BLOCK_M_VALUE = 64
_GLUON_SPLIT_BLOCK_M_VALUE = 128
_GLUON_MAX_CANDIDATE_BLOCK_M_VALUE = 192
_GLUON_POOL_ALIGNMENT_VALUE = 384
# N128 remains the swapAB and combine tile.  Normal decode/split-MN also
# compile N256 specializations selected through the descriptor block box.
_GLUON_BLOCK_N_VALUE = 128
_GLUON_NORMAL_BLOCK_N_VALUE = 256
_GLUON_BLOCK_K_VALUE = 128
_GLUON_SF_BLOCK_M_VALUE = 128
_GLUON_GROUP_M_VALUE = 8
# Match DeepGEMM's scale-domain WGMMA accumulation boundaries.  FC1 chains the
# four native K32 instructions inside one K128 scale block, while FC2 chains
# two native K32 instructions inside each independently scaled K64 half.  The
# scaled K128/K64 partials are still accumulated into ``final`` in FP32 across
# scale blocks.
# Seven 24-KiB A/B stages consume about 168 KiB per persistent CTA.  On SM90
# this both deepens the pipeline and prevents two such CTAs from being resident
# on one SM, which is required before adding grid-coupled dispatch warps.
_GLUON_1D2D_NUM_STAGES = 7
_GLUON_1D2D_PRODUCERS = 2

_GLUON_BLOCK_M = gl.constexpr(_GLUON_BLOCK_M_VALUE)
_GLUON_SPLIT_BLOCK_M = gl.constexpr(_GLUON_SPLIT_BLOCK_M_VALUE)
_GLUON_BLOCK_N = gl.constexpr(_GLUON_BLOCK_N_VALUE)
_GLUON_NORMAL_BLOCK_N = gl.constexpr(_GLUON_NORMAL_BLOCK_N_VALUE)
_GLUON_BLOCK_K = gl.constexpr(_GLUON_BLOCK_K_VALUE)
_GLUON_SF_BLOCK_M = gl.constexpr(_GLUON_SF_BLOCK_M_VALUE)
_GLUON_GROUP_M = gl.constexpr(_GLUON_GROUP_M_VALUE)
_GLUON_1D2D_PRODUCERS_CONSTEXPR = gl.constexpr(_GLUON_1D2D_PRODUCERS)
_GLUON_LARGE_NORMAL_MATH_WARPS = gl.constexpr(8)
_GLUON_LARGE_NORMAL_TMA_REGS_VALUE = 24


def run_sm90_mega_moe_pre_dispatch(
    ctx: SM90MegaMoESymmetricContext,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    x_sf: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> SM90MegaMoEPreDispatchResult:
    """Register dispatch inputs using the narrow DeepGEMM-style boundary.

    BF16 input is quantized to FP8 E4M3 with one FP32 scale per token/per-128
    group.  Already-quantized FP8 E4M3 input is copied together with its
    required FP32 ``x_sf`` tensor.  Both specializations copy top-k indices,
    multiply top-k weights by ``routed_scaling_factor``, and initialize padded
    top-k rows to ``-1``/zero.  Route counting, peer publication, prefix sums,
    pool materialization, and arrival signaling deliberately remain in the
    dispatch partition.
    """
    _require_sm90_and_gluon()
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional BF16 or FP8 E4M3 tensor")
    if x.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("x must have dtype bfloat16 or float8_e4m3fn")
    num_tokens, hidden = x.shape
    if ctx.max_tokens <= 0:
        raise ValueError("the symmetric context must reserve at least one token")
    if not 0 < ctx.topk <= _SM90_MEGA_MOE_PRE_DISPATCH_THREADS:
        raise ValueError("SM90 pre-dispatch requires topk in [1, 1024]")
    if num_tokens > ctx.max_tokens or hidden != ctx.hidden:
        raise ValueError("input exceeds the symmetric context capacity")
    if hidden <= 0:
        raise ValueError("SM90 pre-dispatch requires a positive hidden dimension")
    if hidden > (
        _SM90_MEGA_MOE_PRE_DISPATCH_GROUPS_PER_CTA
        * _SM90_MEGA_MOE_PRE_DISPATCH_GROUP_SIZE
    ):
        raise ValueError("SM90 pre-dispatch supports hidden dimensions up to 8192")
    if hidden % _SM90_MEGA_MOE_PRE_DISPATCH_GROUP_SIZE:
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
        [_SM90_MEGA_MOE_PRE_DISPATCH_NUM_WARPS, 1],
        [1, 0],
    )
    route_layout = gl.BlockedLayout(
        [1],
        [32],
        [_SM90_MEGA_MOE_PRE_DISPATCH_NUM_WARPS],
        [0],
    )
    num_padding_routes = (ctx.max_tokens - num_tokens) * ctx.topk
    grid = (
        num_tokens
        + triton.cdiv(
            num_padding_routes,
            _SM90_MEGA_MOE_PRE_DISPATCH_THREADS,
        ),
    )
    source_sf = ctx.input_sf if x_sf is None else x_sf
    compiled = _sm90_mega_moe_pre_dispatch_kernel[grid](
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
        hidden // _SM90_MEGA_MOE_PRE_DISPATCH_GROUP_SIZE,
        ctx.topk,
        routed_scaling_factor,
        x.dtype == torch.bfloat16,
        quant_layout,
        route_layout,
        num_warps=_SM90_MEGA_MOE_PRE_DISPATCH_NUM_WARPS,
    )
    return SM90MegaMoEPreDispatchResult(
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


def _validate_sm90_mega_moe_pre_dispatch_result(
    ctx: SM90MegaMoESymmetricContext,
    result: SM90MegaMoEPreDispatchResult,
    *,
    num_tokens: int,
    hidden: int,
    source_dtype: torch.dtype,
    routed_scaling_factor: float,
) -> None:
    if not isinstance(result, SM90MegaMoEPreDispatchResult):
        raise TypeError("pre_dispatch_result must be a Gluon pre-dispatch result")
    if result.num_tokens != num_tokens or result.hidden != hidden:
        raise ValueError("pre_dispatch_result does not match this input shape")
    if result.source_dtype != source_dtype:
        raise ValueError("pre_dispatch_result does not match this input dtype")
    if result.routed_scaling_factor != routed_scaling_factor:
        raise ValueError("pre_dispatch_result used a different routed_scaling_factor")
    expected = (
        ctx.input_acts,
        ctx.input_sf,
        ctx.input_topk_idx,
        ctx.input_topk_weights,
    )
    actual = (
        result.input_acts_fp8,
        result.input_acts_sf,
        result.input_topk_idx,
        result.input_topk_weights,
    )
    if any(lhs.data_ptr() != rhs.data_ptr() for lhs, rhs in zip(actual, expected)):
        raise ValueError("pre_dispatch_result belongs to a different context")


def _host_cdiv(a, b):
    return (a + b - 1) // b


def _host_next_power_of_2(value):
    return 1 << max(value - 1, 0).bit_length()


@gluon.jit
def _groupgemm_scheduler_get_count(stored_counts, count_offsets, expert):
    """Select one cached expert count from a distributed register tensor."""
    return _packed_expert_count_from_cache(
        stored_counts,
        count_offsets,
        expert,
    )


@gluon.jit
def _groupgemm_phase_pool_block_offset(
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


# The two-linear scheduler, producers, math loop, and FC1 epilogue below are
# maintained from DeepGEMM's SM90 contracts in
# deep_gemm/impls/sm90_fp8_mega_moe.cuh and scheduler/mega_moe.cuh.  They are
# not adapters around another Python MegaMoE implementation.
@gluon.jit
def _groupgemm_phase_scheduler_next(
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
                    current_count = _groupgemm_scheduler_get_count(
                        stored_counts,
                        count_offsets,
                        expert,
                    )
            if not found:
                phase = 2
                expert = ((expert - 1) // num_experts_per_wave) * num_experts_per_wave
                current_count = _groupgemm_scheduler_get_count(
                    stored_counts,
                    count_offsets,
                    expert,
                )
                pool_block_offset = _groupgemm_phase_pool_block_offset(
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
                    current_count = _groupgemm_scheduler_get_count(
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


@gluon.jit
def _groupgemm_phase_a_sfa_tma_partition(
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
    l2_arrival_counter: gl.constexpr,
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
    current_count = _groupgemm_scheduler_get_count(
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
    ) = _groupgemm_phase_scheduler_next(
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
        scale_row_start = task_pool_block * _GLUON_SF_BLOCK_M
        if task_phase == 1:
            expected = gl.minimum(block_m, valid_count - local_row)
            ready = _load_i32_acquire_gpu(l1_arrival + task_pool_block)
            while ready < expected:
                ready = _load_i32_acquire_gpu(l1_arrival + task_pool_block)
        if task_phase == 2:
            ready = _load_i32_acquire_gpu(l2_arrival + task_pool_block)
            expected_l2_arrivals = l1_n_blocks
            if l2_arrival_counter:
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
        ) = _groupgemm_phase_scheduler_next(
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
def _groupgemm_phase_b_tma_partition(
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
    current_count = _groupgemm_scheduler_get_count(
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
    ) = _groupgemm_phase_scheduler_next(
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
        ) = _groupgemm_phase_scheduler_next(
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
def _sm90_silu(value, fast_math: gl.constexpr):
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
def _sm90_reciprocal(value, fast_math: gl.constexpr):
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
def _groupgemm_fc1_swap_mainloop(
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
):
    """FC1 swapAB mainloop with independent gate/up scale domains."""
    stage_empty, stage_ready = barriers
    a_buffers, b_buffers, sfa_lo_buffers, sfa_hi_buffers = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_k: gl.constexpr = a_buffers.type.shape[2]
    num_k_tiles: gl.constexpr = l1_k // block_k

    swap_mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[16 if block_n >= 256 else block_n // 16, 1],
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
        # The shared A/B payload is dead once every warpgroup has completed
        # WGMMA.  Release the stage before the register-only scale promotion
        # so the producers can overlap the next TMA fill with that arithmetic.
        _gluon_partition_barrier()
        mbarrier.arrive(stage_empty.index(stage), count=1)
        final += partial * (channel_scale[:, None] * token_scale[None, :])

    return final


@gluon.jit
def _groupgemm_fc2_swap_mainloop(
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
):
    """FC2 swapAB mainloop with independent low/high K64 A scales."""
    stage_empty, stage_ready = barriers
    a_buffers, b_buffers, sfa_lo_buffers, sfa_hi_buffers = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_k: gl.constexpr = a_buffers.type.shape[2]
    num_k_tiles: gl.constexpr = l2_k // block_k

    swap_mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[16 if block_n >= 256 else block_n // 16, 1],
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
            max_num_imprecise_acc=64,
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
            max_num_imprecise_acc=64,
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
        # All shared operands and scales have been consumed.  Let the TMA
        # producers refill this stage while the final FP32 promotion runs.
        _gluon_partition_barrier()
        mbarrier.arrive(stage_empty.index(stage), count=1)
        final += partial_hi * (l2_weight_scale[:, None] * token_scale_hi[None, :])

    return final


@gluon.jit
def _groupgemm_fc2_swap_bf16_epilogue(
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
def _groupgemm_fc1_bm64_bn256_split_epilogue(
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
    swiglu = _sm90_silu(gate, fast_math) * up
    valid_rows = row_offsets < valid_m
    weight = gl.load(
        route_weights + task_pool_block * fragment_m + row_offsets,
        mask=valid_rows,
        other=0.0,
    )
    activation = swiglu * weight[:, None]
    amax = gl.max(gl.abs(activation), axis=1)
    scale = gl.maximum(amax, 1.0e-10) * (1.0 / 448.0)
    quantized = (activation * _sm90_reciprocal(scale[:, None], fast_math)).to(
        gl.float8e4nv
    )

    sf_pool_rows = task_pool_block * _GLUON_SF_BLOCK_M + row_offsets
    sf_group = task_n_block * 2 + math_partition_idx
    gl.store(
        l2_acts_sf + sf_group * num_padded_sf_pool_tokens + sf_pool_rows,
        scale,
        mask=valid_rows,
    )

    l2_epilogue_buffer.store(quantized)
    _gluon_partition_barrier()
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
    _gluon_partition_barrier()


@gluon.jit
def _groupgemm_fc1_bm64_bn128_split_epilogue(
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
    swiglu = _sm90_silu(gate, fast_math) * up
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
    _gluon_partition_barrier()
    mbarrier.arrive(fc1_scale_ready, count=1)
    mbarrier.wait(fc1_scale_ready, sync_phase)

    amax = gl.maximum(
        fc1_amax_scratch_0.load(row_layout),
        fc1_amax_scratch_1.load(row_layout),
    )
    scale = gl.maximum(amax, 1.0e-10) * (1.0 / 448.0)
    quantized = (activation * _sm90_reciprocal(scale[:, None], fast_math)).to(
        gl.float8e4nv
    )

    if math_partition_idx == 0:
        sf_pool_rows = task_pool_block * _GLUON_SF_BLOCK_M + row_offsets
        gl.store(
            l2_acts_sf + task_n_block * num_padded_sf_pool_tokens + sf_pool_rows,
            scale,
            mask=valid_rows,
        )

    l2_epilogue_buffer.store(quantized)
    _gluon_partition_barrier()
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
    _gluon_partition_barrier()
    mbarrier.arrive(fc1_scale_done, count=1)
    mbarrier.wait(fc1_scale_done, sync_phase)


@gluon.jit
def _groupgemm_fc2_bf16_scatter_epilogue(
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
def _groupgemm_fc2_combine_partition(
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
    _gluon_partition_barrier()
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
    _gluon_partition_barrier()

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
def _groupgemm_fc2_combine_split_partition(
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
    _gluon_partition_barrier()
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
    _gluon_partition_barrier()

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


@gluon.jit
def _groupgemm_fc1_epilogue(
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
    l2_arrival_counter: gl.constexpr,
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
    swiglu = _sm90_silu(gate, fast_math) * up
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
        quantized = (activation * _sm90_reciprocal(scale[:, None], fast_math)).to(
            gl.float8e4nv
        )
        sf_pool_rows = task_pool_block * _GLUON_SF_BLOCK_M + row_offsets
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
            (activation_groups * _sm90_reciprocal(scale_groups[:, :, None], fast_math))
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
        scale_pool_rows = task_pool_block * _GLUON_SF_BLOCK_M + scale_rows[:, None]
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
    _gluon_partition_barrier()
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
    _gluon_partition_barrier()
    arrival_count = 1
    if l2_arrival_counter:
        active_m_wgs = (valid_m + 63) // 64
        arrival_count = active_m_wgs * (block_n // 128)
    gl.atomic_add(
        l2_arrival + task_pool_block,
        arrival_count,
        sem="release",
        scope="gpu",
    )
    _gluon_partition_barrier()


@gluon.jit
def _groupgemm_fc1_bm128_bn256_split_epilogue(
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
    swiglu = _sm90_silu(gate, fast_math) * up
    valid_rows = logical_rows < valid_m
    weight = gl.load(
        route_weights + task_pool_block * _GLUON_SPLIT_BLOCK_M + logical_rows,
        mask=valid_rows,
        other=0.0,
    )
    activation = swiglu * weight[:, None]
    amax = gl.max(gl.abs(activation), axis=1)
    scale = gl.maximum(amax, 1.0e-10) * (1.0 / 448.0)
    quantized = (activation * _sm90_reciprocal(scale[:, None], fast_math)).to(
        gl.float8e4nv
    )

    sf_pool_rows = task_pool_block * _GLUON_SF_BLOCK_M + logical_rows
    sf_group = task_n_block * 2 + wg_n
    gl.store(
        l2_acts_sf + sf_group * num_padded_sf_pool_tokens + sf_pool_rows,
        scale,
        mask=valid_rows,
    )

    if valid_m > wg_m * fragment_m:
        l2_epilogue_buffer.store(quantized)
        _gluon_partition_barrier()
        fence_async_shared()
        tma.async_copy_shared_to_global(
            l2_store_desc,
            [
                task_pool_block * _GLUON_SPLIT_BLOCK_M + wg_m * fragment_m,
                task_n_block * (_GLUON_NORMAL_BLOCK_N // 2) + wg_n * (fragment_n // 2),
            ],
            l2_epilogue_buffer,
        )
        tma.store_wait(0)
        _gluon_partition_barrier()
        gl.atomic_add(
            l2_arrival + task_pool_block,
            1,
            sem="release",
            scope="gpu",
        )
    _gluon_partition_barrier()


@gluon.jit
def _groupgemm_phase_math_partition(
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
    l2_arrival_counter: gl.constexpr,
    l2_epilogue_requires_full_sync: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    num_math_warps: gl.constexpr,
    staged_dispatch_handoff: gl.constexpr,
):
    """Fused swapAB FC1, FC2, and combine partition."""
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
    current_count = _groupgemm_scheduler_get_count(
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
    ) = _groupgemm_phase_scheduler_next(
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
                final_swap_8 = _groupgemm_fc1_swap_mainloop(
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
                )
                _groupgemm_fc1_epilogue(
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
                    l2_arrival_counter,
                )
            elif valid_m <= 16:
                final_swap_16 = _groupgemm_fc1_swap_mainloop(
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
                )
                _groupgemm_fc1_epilogue(
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
                    l2_arrival_counter,
                )
            elif valid_m <= 32:
                final_swap_32 = _groupgemm_fc1_swap_mainloop(
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
                )
                _groupgemm_fc1_epilogue(
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
                    l2_arrival_counter,
                )
            elif block_m == 64 or valid_m <= 64:
                final_swap_64 = _groupgemm_fc1_swap_mainloop(
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
                )
                _groupgemm_fc1_epilogue(
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
                    l2_arrival_counter,
                )
            else:
                final_swap_128 = _groupgemm_fc1_swap_mainloop(
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
                )
                _groupgemm_fc1_epilogue(
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
                    l2_arrival_counter,
                )
            pipeline_tile += l1_k // block_k
        else:
            # Gluon register tensors require power-of-two element counts, so
            # both linear phases use the same 8/16/32/64 token buckets.
            if valid_m <= 8:
                final_swap_8 = _groupgemm_fc2_swap_mainloop(
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
                )
                _groupgemm_fc2_swap_bf16_epilogue(
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
            elif valid_m <= 16:
                final_swap_16 = _groupgemm_fc2_swap_mainloop(
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
                )
                _groupgemm_fc2_swap_bf16_epilogue(
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
                final_swap_32 = _groupgemm_fc2_swap_mainloop(
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
                )
                _groupgemm_fc2_swap_bf16_epilogue(
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
                final_swap_64 = _groupgemm_fc2_swap_mainloop(
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
                )
                _groupgemm_fc2_swap_bf16_epilogue(
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
                final_swap_128 = _groupgemm_fc2_swap_mainloop(
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
                )
                _groupgemm_fc2_swap_bf16_epilogue(
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

        if task_phase == 2 and l2_epilogue_requires_full_sync:
            _gluon_partition_barrier()

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
        ) = _groupgemm_phase_scheduler_next(
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

    _groupgemm_fc2_combine_partition(
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


@gluon.jit
def _groupgemm_bm128_bn256_split_math_partition(
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
    current_count = _groupgemm_scheduler_get_count(
        stored_counts, count_offsets, scheduler_expert
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
    ) = _groupgemm_phase_scheduler_next(
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
                    max_num_imprecise_acc=128,
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
                    max_num_imprecise_acc=64,
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
                    max_num_imprecise_acc=64,
                )
                partial_hi = warpgroup_mma_wait(num_outstanding=0, deps=(partial_hi,))
                final += partial_hi * (row_scale_hi[:, None] * l2_weight_scale)

            _gluon_partition_barrier()
            mbarrier.arrive(stage_empty.index(stage), count=1)
            pipeline_tile += 1
            k_tile += 1

        if task_phase == 1:
            _groupgemm_fc1_bm128_bn256_split_epilogue(
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
            _groupgemm_fc2_bf16_scatter_epilogue(
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
        ) = _groupgemm_phase_scheduler_next(
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

    _groupgemm_fc2_combine_split_partition(
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


@gluon.jit
def _groupgemm_bm64_bn128_split_math_partition(
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
):
    """One 4-warp 64x64 fragment of logical BM64/BN128 normal."""
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
    current_count = _groupgemm_scheduler_get_count(
        stored_counts, count_offsets, scheduler_expert
    )
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
    ) = _groupgemm_phase_scheduler_next(
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
        prev_row_scale = gl.full((fragment_m,), 1.0, gl.float32, row_layout)
        prev_gate_scale = 1.0
        prev_up_scale = 1.0
        prev_weight_scale = 1.0
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
                if k_tile != 0:
                    row_ratio = prev_row_scale * _sm90_reciprocal(row_scale, fast_math)
                    gate_ratio = prev_gate_scale * _sm90_reciprocal(
                        gate_scale, fast_math
                    )
                    up_ratio = prev_up_scale * _sm90_reciprocal(up_scale, fast_math)
                    column_ratio = gl.where(l1_use_up_scale, up_ratio, gate_ratio)
                    final *= row_ratio[:, None] * column_ratio[None, :]

                final = warpgroup_mma(
                    a_stage,
                    b_stage.permute((1, 0)),
                    final,
                    is_async=True,
                    use_acc=True,
                )
                final = warpgroup_mma_wait(num_outstanding=0, deps=(final,))
                prev_row_scale = row_scale
                prev_gate_scale = gate_scale
                prev_up_scale = up_scale
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
                if k_tile != 0:
                    row_ratio = prev_row_scale * _sm90_reciprocal(
                        row_scale_lo, fast_math
                    )
                    weight_ratio = prev_weight_scale * _sm90_reciprocal(
                        weight_scale, fast_math
                    )
                    final *= row_ratio[:, None] * weight_ratio

                final = warpgroup_mma(
                    a_stage.slice(0, 64, dim=1),
                    b_stage.slice(0, 64, dim=1).permute((1, 0)),
                    final,
                    is_async=True,
                    use_acc=True,
                )
                final = warpgroup_mma_wait(num_outstanding=0, deps=(final,))
                row_half_ratio = row_scale_lo * _sm90_reciprocal(
                    row_scale_hi, fast_math
                )
                final *= row_half_ratio[:, None]
                final = warpgroup_mma(
                    a_stage.slice(64, 64, dim=1),
                    b_stage.slice(64, 64, dim=1).permute((1, 0)),
                    final,
                    is_async=True,
                    use_acc=True,
                )
                final = warpgroup_mma_wait(num_outstanding=0, deps=(final,))
                prev_row_scale = row_scale_hi
                prev_weight_scale = weight_scale

            _gluon_partition_barrier()
            mbarrier.arrive(stage_empty.index(stage), count=1)
            pipeline_tile += 1
            k_tile += 1

        if task_phase == 1:
            final_column_scale = gl.where(
                l1_use_up_scale, prev_up_scale, prev_gate_scale
            )
            final *= prev_row_scale[:, None] * final_column_scale[None, :]
            _groupgemm_fc1_bm64_bn128_split_epilogue(
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
            final *= prev_row_scale[:, None] * prev_weight_scale
            fc2_tile = gl.convert_layout(final.to(gl.bfloat16), store_layout)
            _groupgemm_fc2_bf16_scatter_epilogue(
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
        ) = _groupgemm_phase_scheduler_next(
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

    _groupgemm_fc2_combine_split_partition(
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


@gluon.jit
def _groupgemm_bm64_bn256_split_math_partition(
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
):
    """One 4-warp 64x128 fragment of logical BM64/BN256 normal."""
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
    current_count = _groupgemm_scheduler_get_count(
        stored_counts, count_offsets, scheduler_expert
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
    ) = _groupgemm_phase_scheduler_next(
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
                    max_num_imprecise_acc=128,
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
                    max_num_imprecise_acc=64,
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
                    max_num_imprecise_acc=64,
                )
                partial_hi = warpgroup_mma_wait(num_outstanding=0, deps=(partial_hi,))
                final += partial_hi * (row_scale_hi[:, None] * weight_scale)

            _gluon_partition_barrier()
            mbarrier.arrive(stage_empty.index(stage), count=1)
            pipeline_tile += 1
            k_tile += 1

        if task_phase == 1:
            _groupgemm_fc1_bm64_bn256_split_epilogue(
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
            _groupgemm_fc2_bf16_scatter_epilogue(
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
        ) = _groupgemm_phase_scheduler_next(
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

    _groupgemm_fc2_combine_split_partition(
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


@gluon.jit
def _sm90_fused_dispatch_1d2d_compact_3d_oob_tma_kernel(
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
    l2_arrival_counter: gl.constexpr,
    l2_epilogue_requires_full_sync: gl.constexpr,
    split_math_partitions: gl.constexpr,
    split_bm64_bn128_partitions: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    num_math_warps: gl.constexpr,
    math_regs: gl.constexpr,
    a_tma_regs: gl.constexpr,
    b_tma_regs: gl.constexpr,
    dispatch_regs: gl.constexpr,
):
    """One resident CTA per SM for dispatch, FC1 publication, and FC2."""
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
        [block_m],
        gl.float32,
    )
    sfa_lo_buffers = gl.allocate_shared_memory(
        gl.float32,
        [num_stages, block_m],
        sfa_layout,
    )
    sfa_hi_buffers = gl.allocate_shared_memory(
        gl.float32,
        [num_stages, block_m],
        sfa_layout,
    )
    l2_epilogue_buffer = gl.allocate_shared_memory(
        l2_store_desc.dtype,
        l2_store_desc.block_type.shape,
        l2_store_desc.layout,
    )
    if split_math_partitions or split_bm64_bn128_partitions:
        l2_epilogue_buffer_1 = gl.allocate_shared_memory(
            l2_store_desc.dtype,
            l2_store_desc.block_type.shape,
            l2_store_desc.layout,
        )
    if split_math_partitions:
        l2_epilogue_buffer_2 = gl.allocate_shared_memory(
            l2_store_desc.dtype,
            l2_store_desc.block_type.shape,
            l2_store_desc.layout,
        )
        l2_epilogue_buffer_3 = gl.allocate_shared_memory(
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
    if split_bm64_bn128_partitions:
        fc1_amax_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
            [64], gl.float32
        )
        fc1_amax_scratch_0 = gl.allocate_shared_memory(
            gl.float32, [64], fc1_amax_layout
        )
        fc1_amax_scratch_1 = gl.allocate_shared_memory(
            gl.float32, [64], fc1_amax_layout
        )
        fc1_scale_ready = gl.allocate_shared_memory(gl.int64, [1], barrier_layout)
        fc1_scale_done = gl.allocate_shared_memory(gl.int64, [1], barrier_layout)
        mbarrier.init(fc1_scale_ready, count=2)
        mbarrier.init(fc1_scale_done, count=2)
    for stage in gl.static_range(num_stages):
        if split_math_partitions:
            mbarrier.init(stage_empty.index(stage), count=4)
        elif split_bm64_bn128_partitions:
            mbarrier.init(stage_empty.index(stage), count=2)
        else:
            mbarrier.init(stage_empty.index(stage), count=1)
        mbarrier.init(
            stage_ready.index(stage),
            count=_GLUON_1D2D_PRODUCERS_CONSTEXPR,
        )
    if split_math_partitions or split_bm64_bn128_partitions:
        math_done = gl.allocate_shared_memory(
            gl.int64,
            [1],
            barrier_layout,
        )
        if split_math_partitions:
            mbarrier.init(math_done, count=4)
        else:
            mbarrier.init(math_done, count=2)

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
    dispatch_buffers_0 = gl.allocate_shared_memory(
        dispatch_acts_desc_0.dtype,
        dispatch_acts_desc_0.block_type.shape,
        dispatch_acts_desc_0.layout,
    )
    dispatch_buffers_1 = gl.allocate_shared_memory(
        dispatch_acts_desc_0.dtype,
        dispatch_acts_desc_0.block_type.shape,
        dispatch_acts_desc_0.layout,
    )
    dispatch_barriers_0 = gl.allocate_shared_memory(
        gl.int64,
        [1],
        barrier_layout,
    )
    dispatch_barriers_1 = gl.allocate_shared_memory(
        gl.int64,
        [1],
        barrier_layout,
    )
    mbarrier.init(dispatch_barriers_0, count=1)
    mbarrier.init(dispatch_barriers_1, count=1)
    staged_dispatch_handoff: gl.constexpr = num_tokens <= 64

    if split_bm64_bn128_partitions:
        gl.warp_specialize(
            [
                (
                    _groupgemm_bm64_bn128_split_math_partition,
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
                    ),
                ),
                (
                    _groupgemm_bm64_bn128_split_math_partition,
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
                    ),
                ),
                (
                    _groupgemm_phase_a_sfa_tma_partition,
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
                        l2_arrival_counter,
                        staged_dispatch_handoff,
                    ),
                ),
                (
                    _groupgemm_phase_b_tma_partition,
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
                    _fused_pool_dispatch_partition,
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
                    _fused_pool_dispatch_partition,
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
                    _groupgemm_bm128_bn256_split_math_partition,
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
                    ),
                ),
                (
                    _groupgemm_bm128_bn256_split_math_partition,
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
                    ),
                ),
                (
                    _groupgemm_bm128_bn256_split_math_partition,
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
                    ),
                ),
                (
                    _groupgemm_bm128_bn256_split_math_partition,
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
                    ),
                ),
                (
                    _groupgemm_phase_a_sfa_tma_partition,
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
                        l2_arrival_counter,
                        staged_dispatch_handoff,
                    ),
                ),
                (
                    _groupgemm_phase_b_tma_partition,
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
                    _fused_pool_dispatch_partition,
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
                    _fused_pool_dispatch_partition,
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
                    _groupgemm_phase_math_partition,
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
                        l2_arrival_counter,
                        l2_epilogue_requires_full_sync,
                        block_m,
                        block_n,
                        num_math_warps,
                        staged_dispatch_handoff,
                    ),
                ),
                (
                    _groupgemm_phase_a_sfa_tma_partition,
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
                        l2_arrival_counter,
                        staged_dispatch_handoff,
                    ),
                ),
                (
                    _groupgemm_phase_b_tma_partition,
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
                    _fused_pool_dispatch_partition,
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
                    _fused_pool_dispatch_partition,
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
def _sm90_fused_dispatch_bm64_bn256_large_kernel(
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
):
    """One resident BM64/BN256 two-partition CTA per SM for M >= 256."""
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
        [_GLUON_BLOCK_M],
        gl.float32,
    )
    sfa_lo_buffers = gl.allocate_shared_memory(
        gl.float32,
        [num_stages, _GLUON_BLOCK_M],
        sfa_layout,
    )
    sfa_hi_buffers = gl.allocate_shared_memory(
        gl.float32,
        [num_stages, _GLUON_BLOCK_M],
        sfa_layout,
    )
    l2_epilogue_buffer = gl.allocate_shared_memory(
        l2_store_desc.dtype,
        l2_store_desc.block_type.shape,
        l2_store_desc.layout,
    )
    l2_epilogue_buffer_1 = gl.allocate_shared_memory(
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
        mbarrier.init(stage_empty.index(stage), count=2)
        mbarrier.init(
            stage_ready.index(stage),
            count=_GLUON_1D2D_PRODUCERS_CONSTEXPR,
        )
    math_done = gl.allocate_shared_memory(gl.int64, [1], barrier_layout)
    mbarrier.init(math_done, count=2)

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
    dispatch_buffers_0 = gl.allocate_shared_memory(
        dispatch_acts_desc_0.dtype,
        dispatch_acts_desc_0.block_type.shape,
        dispatch_acts_desc_0.layout,
    )
    dispatch_buffers_1 = gl.allocate_shared_memory(
        dispatch_acts_desc_0.dtype,
        dispatch_acts_desc_0.block_type.shape,
        dispatch_acts_desc_0.layout,
    )
    dispatch_barriers_0 = gl.allocate_shared_memory(
        gl.int64,
        [1],
        barrier_layout,
    )
    dispatch_barriers_1 = gl.allocate_shared_memory(
        gl.int64,
        [1],
        barrier_layout,
    )
    mbarrier.init(dispatch_barriers_0, count=1)
    mbarrier.init(dispatch_barriers_1, count=1)

    gl.warp_specialize(
        [
            (
                _groupgemm_bm64_bn256_split_math_partition,
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
                ),
            ),
            (
                _groupgemm_bm64_bn256_split_math_partition,
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
                ),
            ),
            (
                _groupgemm_phase_a_sfa_tma_partition,
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
                _groupgemm_phase_b_tma_partition,
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
                    _GLUON_BLOCK_M,
                    False,
                ),
            ),
            (
                _fused_pool_dispatch_partition,
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
                    _GLUON_BLOCK_M,
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
                _fused_pool_dispatch_partition,
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
                    _GLUON_BLOCK_M,
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


def run_sm90_fused_dispatch_1d2d_compact_symmetric(
    ctx: SM90MegaMoESymmetricContext,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor | None,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    local_weight: torch.Tensor,
    local_b_scales: torch.Tensor,
    local_l2_weight: torch.Tensor,
    local_l2_b_scales: torch.Tensor,
    *,
    block_m: int | None = None,
    group_n: int = 128,
    group_k: int = 128,
    output_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    num_stages: int | None = None,
    maxnreg: int | None = None,
    dispatch_regs: int | None = None,
    num_experts_per_wave: int | None = None,
    activation_clamp: float = math.inf,
    fast_math: bool = False,
    workspace: SM90MegaMoEFusedFC1Workspace | None = None,
    routed_scaling_factor: float = 1.0,
    pre_dispatch_result: SM90MegaMoEPreDispatchResult | None = None,
) -> SM90MegaMoEFusedFC1Result:
    """Register inputs, then run fused dispatch, FC1 epilogue, and FC2.

    ``x_fp8`` retains its historical name for call-site compatibility, but it
    may now be either BF16 or FP8 E4M3.  BF16 requires ``x_sf=None`` and is
    quantized by :func:`run_sm90_mega_moe_pre_dispatch`; FP8 requires its
    FP32 per-token/per-128 scales.  A separately measured pre-dispatch result
    may be supplied to skip that registration launch.  The persistent kernel
    always consumes the registered context buffers and never republishes raw
    input tensors from its two dispatch warps.

    Each source rank release-adds its count and contributor bit into the
    destination's packed expert states.  A, B, and math acquire those states,
    cache the low-32 counts, and derive compact offsets locally.  Only the A
    producer then acquires each block's
    ``l1_arrival`` counter and publishes the matching A+SFA tiles through the
    shared stage barrier.  Math consumes that shared publication without a
    second per-block dispatch poll.  The two resident dispatch partitions use
    one 3D-OOB TMA load/store pair per activation token.

    A caller measuring repeated launches may pass ``result.workspace`` from a
    completed warmup.  Device buffers and their TMA descriptors are then
    reused.  The next call resets only compact control and arrival arrays in a
    single CTA; stale padded payload, scale, and metadata rows remain
    unobservable behind the valid-row masks.  This avoids both the former
    multi-MiB reset and a dispatch/combine tail-cleanup dependency.

    ``local_weight`` is the granularity-8 gate/up-interleaved FC1 tensor and
    ``local_b_scales`` keeps canonical gate-block then up-block ordering.
    ``local_l2_weight`` and its per-128 scale tensor extend the existing model
    ABI without changing the dispatch input ABI.  ``output`` is the complete
    token-major result after FC2 peer scatter and source-local top-k sum.

    DSV4 Pro retains its measured H20 policy.  DSV4 Flash uses swapAB through
    128 tokens/rank and normal BM64/BN256 through the 8192-token benchmark
    range; the 8192-token specialization uses three pipeline stages.  FC2
    splits every K128 tile into two K64 WGMMA groups so each half consumes its
    independent per-64 activation scale.
    """
    _require_sm90_and_gluon()
    if ctx.world_size <= 1:
        raise ValueError("fused symmetric dispatch requires world_size > 1")
    if x_fp8.ndim != 2 or x_fp8.dtype not in (
        torch.bfloat16,
        torch.float8_e4m3fn,
    ):
        raise ValueError("x_fp8 must be a two-dimensional BF16 or FP8 E4M3 tensor")
    num_tokens, K = x_fp8.shape
    if num_tokens > ctx.max_tokens or K != ctx.hidden:
        raise ValueError("input exceeds the symmetric context capacity")
    if x_fp8.dtype == torch.bfloat16:
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
    if local_weight.ndim != 3:
        raise ValueError("local_weight must have shape [experts_per_rank, N, K]")
    E, N, weight_k = local_weight.shape
    if E != ctx.experts_per_rank:
        raise ValueError("local_weight must contain exactly this rank's experts")
    if weight_k != K:
        raise ValueError("dispatch input and local_weight K dimensions must match")
    if local_weight.dtype != torch.float8_e4m3fn:
        raise ValueError("local_weight must have dtype float8_e4m3fn")
    if group_n != _GLUON_BLOCK_N_VALUE or group_k != _GLUON_BLOCK_K_VALUE:
        raise ValueError("fused compact FC1 requires group_n=group_k=128")
    if K % group_k or N % 256:
        raise ValueError("H must be divisible by 128 and FC1 2I by 256")
    intermediate_hidden = N // 2
    if local_b_scales.shape != (E, N // group_n, K // group_k):
        raise ValueError("local_b_scales must be [E, N // 128, K // 128]")
    if local_b_scales.dtype != torch.float32:
        raise ValueError("local_b_scales must have dtype float32")
    if local_l2_weight.shape != (E, K, intermediate_hidden):
        raise ValueError("local_l2_weight must be [E, H, I]")
    if local_l2_weight.dtype != torch.float8_e4m3fn:
        raise ValueError("local_l2_weight must have dtype float8_e4m3fn")
    if local_l2_b_scales.shape != (
        E,
        K // 128,
        intermediate_hidden // 128,
    ):
        raise ValueError("local_l2_b_scales must be [E, H/128, I/128]")
    if local_l2_b_scales.dtype != torch.float32:
        raise ValueError("local_l2_b_scales must have dtype float32")
    if output_dtype != torch.bfloat16:
        raise ValueError("fused compact FC1 currently supports bfloat16 output only")
    tensors = (
        topk_idx,
        topk_weights,
        local_weight,
        local_b_scales,
        local_l2_weight,
        local_l2_b_scales,
    )
    if x_sf is not None:
        tensors = (x_sf, *tensors)
    if any(tensor.device != x_fp8.device for tensor in tensors):
        raise ValueError("all fused FC1 tensors must be on the same device")
    if x_fp8.device != ctx.device:
        raise ValueError("fused FC1 inputs must use the symmetric context device")
    if any(not tensor.is_contiguous() for tensor in (x_fp8, *tensors)):
        raise ValueError("all fused FC1 inputs and weights must be contiguous")
    if local_b_scales.stride(-1) != 1:
        raise ValueError("local_b_scales must be contiguous along K groups")
    if local_l2_b_scales.stride(-1) != 1:
        raise ValueError("local_l2_b_scales must be contiguous along K groups")
    if topk_idx.numel() == 0:
        raise ValueError("the dispatch contract requires at least one route slot")

    device_sms = torch.cuda.get_device_properties(x_fp8.device).multi_processor_count
    if num_sms is None:
        num_sms = device_sms
    if not 0 < num_sms <= device_sms:
        raise ValueError(f"num_sms must be in [1, {device_sms}], got {num_sms}")
    if not isinstance(fast_math, bool):
        raise ValueError("fast_math must be a bool")
    config = _get_sm90_mega_moe_config(
        num_tokens_per_rank=num_tokens,
        hidden=K,
        intermediate_hidden=intermediate_hidden,
        num_experts=ctx.num_experts,
        num_experts_per_rank=E,
        topk=ctx.topk,
        num_sms=num_sms,
        num_stages=num_stages,
        num_experts_per_wave=num_experts_per_wave,
    )
    # The DSV4 fused implementation keeps the four evaluated math strategies:
    # swapAB BM64/BN128, normal BM64/BN128 (2 partitions), normal
    # BM64/BN256 (2 partitions), and normal BM128/BN256 (4 partitions).
    # In particular, every fused normal configuration is partitioned.
    split_math_partitions = (
        config.block_m == _GLUON_SPLIT_BLOCK_M_VALUE
        and config.block_n == _GLUON_NORMAL_BLOCK_N_VALUE
        and config.num_math_warps == 16
        and not config.use_swap_ab
        and not config.use_large_normal_partition
    )
    split_bm64_bn128_partitions = (
        config.block_m == _GLUON_BLOCK_M_VALUE
        and config.block_n == _GLUON_BLOCK_N_VALUE
        and config.num_math_warps == 8
        and not config.use_swap_ab
        and not config.use_large_normal_partition
    )
    if config.use_large_normal_partition or split_bm64_bn128_partitions:
        # The shared selector also serves the standalone serial backend,
        # whose unsplit publisher still emits one arrival per logical tile.
        # Mark only the fused result/config contract as split publication.
        config = replace(
            config,
            l2_arrival_counter=True,
            l2_epilogue_requires_full_sync=False,
        )
    if block_m is not None and block_m != config.block_m:
        raise ValueError(
            f"selected {config.mode} requires block_m={config.block_m}, got {block_m}"
        )
    block_m = config.block_m
    block_n = config.block_n
    num_stages = config.num_stages
    num_experts_per_wave = config.num_experts_per_wave
    if maxnreg is None:
        maxnreg = config.launch_maxnreg
    if dispatch_regs is None:
        dispatch_regs = config.dispatch_register_budget
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
    if pre_dispatch_result is None:
        registered_inputs = run_sm90_mega_moe_pre_dispatch(
            ctx,
            x_fp8,
            topk_idx,
            topk_weights,
            x_sf=x_sf,
            routed_scaling_factor=routed_scaling_factor,
        )
    else:
        _validate_sm90_mega_moe_pre_dispatch_result(
            ctx,
            pre_dispatch_result,
            num_tokens=num_tokens,
            hidden=K,
            source_dtype=x_fp8.dtype,
            routed_scaling_factor=routed_scaling_factor,
        )
        registered_inputs = pre_dispatch_result

    max_global_routes = ctx.world_size * ctx.max_routes
    # Keep the config-independent pool reservation so diagnostic BM128
    # specializations can reuse a production BM64 workspace without moving
    # payload buffers or retaining descriptors for a stale block shape.
    num_pool_rows = _host_align(
        max_global_routes + E * (_GLUON_MAX_CANDIDATE_BLOCK_M_VALUE - 1),
        _GLUON_POOL_ALIGNMENT_VALUE,
    )
    max_pool_blocks = num_pool_rows // _GLUON_BLOCK_M_VALUE
    num_padded_sf_pool_tokens = max_pool_blocks * _GLUON_SF_BLOCK_M_VALUE
    device = x_fp8.device
    gemm_descriptor_shapes = (
        (_GLUON_BLOCK_M_VALUE, _GLUON_BLOCK_N_VALUE),
        (_GLUON_BLOCK_M_VALUE, _GLUON_NORMAL_BLOCK_N_VALUE),
        (_GLUON_SPLIT_BLOCK_M_VALUE, _GLUON_BLOCK_N_VALUE),
        (_GLUON_SPLIT_BLOCK_M_VALUE, _GLUON_NORMAL_BLOCK_N_VALUE),
    )
    if split_bm64_bn128_partitions:
        gemm_descriptor_shapes = (
            (_GLUON_BLOCK_M_VALUE, _GLUON_BLOCK_N_VALUE // 2),
            *gemm_descriptor_shapes,
        )
    if workspace is None:
        pool = SM90MegaMoESingleRankPool(
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
        l1_arrival = torch.empty(
            max_pool_blocks,
            dtype=torch.int32,
            device=device,
        )
        l2_arrival = torch.empty(
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
                [descriptor_block_m, _GLUON_BLOCK_K_VALUE],
                gl.float8e4nv,
            )
            sfa_layout = gl.NVMMASharedLayout.get_default_for(
                [descriptor_block_m],
                gl.float32,
            )
            b_layout = gl.NVMMASharedLayout.get_default_for(
                [descriptor_block_n, _GLUON_BLOCK_K_VALUE],
                gl.float8e4nv,
            )
            l2_store_layout = gl.NVMMASharedLayout.get_default_for(
                [descriptor_block_m, descriptor_block_n // 2],
                gl.float8e4nv,
            )
            gemm_descriptor_sets.append(
                SM90MegaMoEGemmDescriptorSet(
                    block_m=descriptor_block_m,
                    block_n=descriptor_block_n,
                    l1_a_desc=TensorDescriptor.from_tensor(
                        pool.acts,
                        [descriptor_block_m, _GLUON_BLOCK_K_VALUE],
                        a_layout,
                    ),
                    l1_sfa_desc=TensorDescriptor.from_tensor(
                        pool.acts_sf_mn_major.view(-1),
                        [descriptor_block_m],
                        sfa_layout,
                    ),
                    l1_b_desc=TensorDescriptor.from_tensor(
                        local_weight.view(E * N, K),
                        [descriptor_block_n, _GLUON_BLOCK_K_VALUE],
                        b_layout,
                    ),
                    l2_store_desc=TensorDescriptor.from_tensor(
                        l2_acts,
                        [descriptor_block_m, descriptor_block_n // 2],
                        l2_store_layout,
                    ),
                    l2_a_desc=TensorDescriptor.from_tensor(
                        l2_acts,
                        [descriptor_block_m, _GLUON_BLOCK_K_VALUE],
                        a_layout,
                    ),
                    l2_sfa_desc=TensorDescriptor.from_tensor(
                        l2_acts_sf_mn_major.view(-1),
                        [descriptor_block_m],
                        sfa_layout,
                    ),
                    l2_b_desc=TensorDescriptor.from_tensor(
                        local_l2_weight.view(E * K, intermediate_hidden),
                        [descriptor_block_n, _GLUON_BLOCK_K_VALUE],
                        b_layout,
                    ),
                )
            )
        dispatch_descs = create_sm90_symmetric_dispatch_tma_oob_3d_descriptors(
            ctx,
            pool.acts,
        )
        workspace = SM90MegaMoEFusedFC1Workspace(
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
            l1_weight_data_ptr=local_weight.data_ptr(),
            l2_weight_data_ptr=local_l2_weight.data_ptr(),
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
        if workspace.l1_weight_data_ptr != local_weight.data_ptr():
            raise ValueError("fused workspace belongs to a different FC1 weight")
        if workspace.l2_weight_data_ptr != local_l2_weight.data_ptr():
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
                if item.block_m == _GLUON_BLOCK_M_VALUE
                and item.block_n == _GLUON_BLOCK_N_VALUE // 2
            ),
            None,
        )
        if fragment_descriptor_set is None:
            raise ValueError("fused workspace has no 64x32 FC1 store descriptor")
        l2_store_desc = fragment_descriptor_set.l2_store_desc
    elif split_math_partitions or config.use_large_normal_partition:
        fragment_descriptor_set = next(
            (
                item
                for item in workspace.gemm_descriptor_sets
                if item.block_m == _GLUON_BLOCK_M_VALUE
                and item.block_n == _GLUON_BLOCK_N_VALUE
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
        [_SM90_MEGA_MOE_FUSED_RESET_NUM_WARPS],
        [0],
    )
    reset_elements = max(
        max_pool_blocks,
        ctx.world_size,
        E,
        ctx.num_experts,
    )
    _sm90_mega_moe_fused_control_reset_kernel[
        (triton.cdiv(reset_elements, _SM90_MEGA_MOE_FUSED_RESET_BLOCK_SIZE),)
    ](
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
        _SM90_MEGA_MOE_FUSED_RESET_BLOCK_SIZE,
        reset_layout,
        num_warps=_SM90_MEGA_MOE_FUSED_RESET_NUM_WARPS,
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
        local_b_scales,
        local_l2_b_scales,
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
        local_b_scales.stride(0),
        local_b_scales.stride(1),
        local_b_scales.stride(2),
        local_l2_b_scales.stride(0),
        local_l2_b_scales.stride(1),
        local_l2_b_scales.stride(2),
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
    if config.use_large_normal_partition:
        compiled = _sm90_fused_dispatch_bm64_bn256_large_kernel[(num_sms,)](
            *kernel_tensor_args,
            *kernel_shape_args,
            config.math_register_budget,
            _GLUON_LARGE_NORMAL_TMA_REGS_VALUE,
            _GLUON_LARGE_NORMAL_TMA_REGS_VALUE,
            dispatch_regs,
            num_warps=4,
            maxnreg=maxnreg,
        )
    else:
        # Keep the small-token single-transport kernel under its original
        # 3D-OOB-specific JIT symbol so cached artifacts remain attributable.
        compiled = _sm90_fused_dispatch_1d2d_compact_3d_oob_tma_kernel[(num_sms,)](
            *kernel_tensor_args,
            *kernel_shape_args,
            config.l2_arrival_counter,
            config.l2_epilogue_requires_full_sync,
            split_math_partitions,
            split_bm64_bn128_partitions,
            block_m,
            block_n,
            config.num_math_warps,
            config.math_register_budget,
            config.non_epilogue_register_budget,
            config.non_epilogue_register_budget,
            dispatch_regs,
            num_warps=(
                4
                if split_math_partitions or split_bm64_bn128_partitions
                else config.num_math_warps
            ),
            maxnreg=maxnreg,
        )
    return SM90MegaMoEFusedFC1Result(
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
        config=config,
        compiled=compiled,
    )
