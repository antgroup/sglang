# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Copyright (c) 2025 DeepSeek
# Derived from Triton-distributed MegaMoE Gluon; Hopper scheduling and
# numerical conventions derive from DeepGEMM. MIT licensed.
"""SM90 two-node EP16 FP8 MegaMoE, at most 128 tokens per rank.
Call collective prepare() before the first fused_moe() execution.

Routing IDs must be negative sentinels or lie in [0, num_experts).
Nonnegative IDs outside that range are outside the operator contract.

Public API: create_context, prepare_weights, prepare, fused_moe.
No serving-stack integration or process-group initialization at import.
"""

import ctypes
import math
import os
import sysconfig
from copy import copy
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import NamedTuple

import torch
import torch.distributed as dist
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language import core

from .mega_moe_gluon import (
    _SF_BLOCK_M,
    CombineMode,
    DispatchHandoff,
    ExpertPool,
    MathBody,
    MegaMoEConfig,
    Rendezvous,
    ResourcePreparationError,
    Row,
    Shape,
    _fc1_swap_mainloop,
    _fc2_swap_bf16_epilogue,
    _fc2_swap_mainloop,
    _load_i32_acquire_gpu,
    _load_i32_acquire_sys_if,
    _load_packed_expert_counts,
    _make_selector,
    _packed_expert_count_from_cache,
    _peer_barrier_arrive_and_wait,
    _pull_dispatch_row,
    _reciprocal,
    _reserve_routes,
    _select_source_route,
    _shape,
    _silu,
    _tile_complete,
    _wait_for_dispatch_handoff,
    a_producer_partition,
    b_producer_partition,
    check_policy_agreement,
    create_arena,
    create_dispatch_descriptors,
    derive_launch_maxnreg,
    math_split_bn128_body,
    math_split_bn256_body,
    math_swap_body,
    partition_barrier,
    prepare_collectively,
    prepare_weights,
    register_inputs,
    register_inputs_kernel,
    resolve_tokens_bound,
    scheduler_count,
    validate_tokens_bound,
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


def _select_internode_config(
    *, shape, tokens_bound, num_sms, grid=None, override=None, return_selection=False
):
    """Host adapter including the current EP16 decode/residency constraints."""
    shape = _shape(shape)
    selected = select(
        topology="ep16",
        fmt="fp8",
        shape=shape,
        tokens_bound=tokens_bound,
        num_sms=num_sms,
        grid=grid,
        override=override,
    )
    grid = selected.grid
    config = selected.launch
    if config.block_m not in (16, 64) or config.num_math_warps not in (4, 8, 16):
        raise ValueError("decode math requires BM16 or BM64")
    if grid > num_sms and not (
        config.use_swap_ab
        and config.num_math_warps == 4
        and (
            config.num_stages <= 3 or (config.block_m == 16 and config.num_stages == 5)
        )
    ):
        raise ValueError(
            "two-CTA tuning requires four-warp swap with at most three stages"
        )
    return selected if return_selection else (config, grid)


# Shape-specific launch policies.

CONFIGS = MappingProxyType(
    {
        "internode_swap_auto_d1_chunked": MegaMoEConfig(
            body=MathBody.SWAP_BN128,
            dispatch_registers=80,
            fc1_promotion_k=32,
            fc2_promotion_k=32,
            dispatch_handoff=DispatchHandoff.D1,
            combine=CombineMode.CHUNKED,
        ),
        "split256_auto_w8_c1_d1_chunked": MegaMoEConfig(
            body=MathBody.SPLIT_BN256,
            dispatch_registers=80,
            fc1_promotion_k=128,
            dispatch_handoff=DispatchHandoff.D1,
            combine=CombineMode.CHUNKED,
        ),
        "split256_auto_w8_c1_d1d2_chunked": MegaMoEConfig(
            body=MathBody.SPLIT_BN256,
            dispatch_registers=80,
            fc1_promotion_k=128,
            dispatch_handoff=DispatchHandoff.D1,
            rendezvous=Rendezvous.NONE,
            combine=CombineMode.CHUNKED,
        ),
        "swap_bm16_s3_w4_c3_d1d2_a1": MegaMoEConfig(
            body=MathBody.SWAP_BN128,
            stages=3,
            math_warps=4,
            ctas_per_sm=3,
            block_m=16,
            math_registers=104,
            fc1_promotion_k=32,
            fc2_promotion_k=32,
            dispatch_handoff=DispatchHandoff.D1,
            rendezvous=Rendezvous.NONE,
            fuse_reset=True,
        ),
        "swap_bm16_s3_w4_c3_d1d2_e9_a1": MegaMoEConfig(
            body=MathBody.SWAP_BN128,
            stages=3,
            math_warps=4,
            ctas_per_sm=3,
            experts_per_wave=9,
            block_m=16,
            math_registers=104,
            fc1_promotion_k=32,
            fc2_promotion_k=32,
            dispatch_handoff=DispatchHandoff.D1,
            rendezvous=Rendezvous.NONE,
            fuse_reset=True,
        ),
        "swap_bm16_s5_w4_c2_d1": MegaMoEConfig(
            body=MathBody.SWAP_BN128,
            stages=5,
            math_warps=4,
            ctas_per_sm=2,
            block_m=16,
            math_registers=104,
            fc1_promotion_k=32,
            fc2_promotion_k=32,
            dispatch_handoff=DispatchHandoff.D1,
        ),
        "swap_s3_w4_c2_d1": MegaMoEConfig(
            body=MathBody.SWAP_BN128,
            stages=3,
            math_warps=4,
            ctas_per_sm=2,
            fc1_promotion_k=32,
            fc2_promotion_k=32,
            dispatch_handoff=DispatchHandoff.D1,
        ),
        "swap_s3_w4_c2_d1_chunked": MegaMoEConfig(
            body=MathBody.SWAP_BN128,
            stages=3,
            math_warps=4,
            ctas_per_sm=2,
            dispatch_registers=80,
            fc1_promotion_k=32,
            fc2_promotion_k=32,
            dispatch_handoff=DispatchHandoff.D1,
            combine=CombineMode.CHUNKED,
        ),
    }
)

TABLE = MappingProxyType(
    {
        ("ep16", "fp8", "h4096_i2048_e256_k6", 78): (
            Row(0, 3, "swap_bm16_s3_w4_c3_d1d2_a1"),
            Row(4, 4, "swap_bm16_s3_w4_c3_d1d2_e9_a1"),
            Row(5, 64, "swap_bm16_s3_w4_c3_d1d2_a1"),
            Row(65, 128, "split256_auto_w8_c1_d1d2_chunked"),
        ),
        ("ep16", "fp8", "h7168_i3072_e384_k6", 78): (
            Row(0, 1, "swap_bm16_s5_w4_c2_d1"),
            Row(2, 32, "swap_s3_w4_c2_d1"),
            Row(33, 64, "swap_s3_w4_c2_d1_chunked"),
            Row(65, 127, "internode_swap_auto_d1_chunked"),
            Row(128, 128, "split256_auto_w8_c1_d1_chunked"),
        ),
    }
)

select = _make_selector("ep16", "fp8", TABLE, CONFIGS)


# launch.py


@dataclass(frozen=True)
class PreparedLaunch:
    compiled: object
    kernel: object
    grid: tuple
    arguments: tuple
    options: dict

    def __call__(self):
        return self.kernel[self.grid](*self.arguments, **self.options)


def prepare_launch(kernel, grid, *args, **options):
    """Compile through warmup; execution uses the public JIT launch interface.

    The caller initializes the CUDA module and validates residency before
    executing. Specialization, argument binding and hooks belong to Triton;
    this helper never reads or constructs a private cache key.
    """
    compiled = kernel.warmup(*args, grid=grid, **options)
    if compiled is None:
        raise RuntimeError("Triton compilation hook did not produce a kernel")
    return PreparedLaunch(compiled, kernel, tuple(grid), args, options)


# nvshmem.py

NVSHMEM_SIGNAL_SET = 9


NVSHMEM_SIGNAL_ADD = 10


NVSHMEM_CMP_GE = 5


def _nvshmem_find_nvshmem_device_bitcode():
    return _nvshmem_device_bitcode(os.environ.get("NVSHMEM_LIB_DIR"))


@lru_cache(maxsize=8)
def _nvshmem_device_bitcode(base):
    path = (
        Path(base)
        if base
        else Path(sysconfig.get_path("purelib")) / "nvidia/nvshmem/lib"
    ) / "libnvshmem_device.bc"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


@core.extern
def _nvshmem_quiet(_semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [],
        {(): ("nvshmem_quiet", core.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


@gluon.jit
def _nvshmem_quiet_warp():
    """One scalar quiet per issuing warp, then reconverge all 32 lanes.

    All lanes must enter this helper. Calling the scalar API on all 32 lanes
    separately issues redundant system-scope atomics to the proxy channel.
    """
    lane = gl.inline_asm_elementwise(
        "mov.u32 $0, %laneid;", "=r", [], dtype=gl.int32, is_pure=False, pack=1
    )
    if lane == 0:
        _nvshmem_quiet()
    gl.inline_asm_elementwise(
        "bar.warp.sync 0xffffffff; mov.u32 $0, 0;",
        "=r",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@core.extern
def _nvshmem_put_signal_nbi(dst, src, size, sig, value, op, pe, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [dst, src, size, sig, value, op, pe],
        {
            (
                core.int64,
                core.int64,
                core.int64,
                core.int64,
                core.uint64,
                core.int32,
                core.int32,
            ): ("nvshmemx_putmem_signal_nbi_warp", core.int32)
        },
        is_pure=False,
        _semantic=_semantic,
    )


@gluon.jit
def _nvshmem_putmem_signal_nbi_warp(dst, src, size, sig, value, pe):
    """Enqueue data plus SET signal; source reuse still requires quiet."""
    return _nvshmem_put_signal_nbi(
        dst.to(gl.int64),
        src.to(gl.int64),
        gl.cast(size, gl.int64),
        sig.to(gl.int64),
        gl.cast(value, gl.uint64),
        gl.full((), 9, gl.int32),
        gl.cast(pe, gl.int32),
    )


@gluon.jit
def _nvshmem_wait_peer_signal_ge(sig, value):
    """Acquire a directly mapped local peer's signal before consuming its data."""
    ready = gl.inline_asm_elementwise(
        "ld.acquire.sys.global.u64 $0, [$1];",
        "=l,l",
        [sig],
        dtype=gl.uint64,
        is_pure=False,
        pack=1,
    )
    while ready < gl.cast(value, gl.uint64):
        ready = gl.inline_asm_elementwise(
            "ld.acquire.sys.global.u64 $0, [$1];",
            "=l,l",
            [sig],
            dtype=gl.uint64,
            is_pure=False,
            pack=1,
        )


@core.extern
def _nvshmem_put_nbi(dst, src, size, pe, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [dst, src, size, pe],
        {
            (core.int64, core.int64, core.int64, core.int32): (
                "nvshmemx_putmem_nbi_warp",
                core.int32,
            )
        },
        is_pure=False,
        _semantic=_semantic,
    )


@gluon.jit
def _nvshmem_putmem_nbi_warp(dst, src, size, pe):
    return _nvshmem_put_nbi(
        dst.to(gl.int64),
        src.to(gl.int64),
        gl.cast(size, gl.int64),
        gl.cast(pe, gl.int32),
    )


_initialized_modules = {}


def _nvshmem_initialize_module(compiled):
    """Register every live CUDA module with NVSHMEM before its first launch."""
    from torch._C._distributed_c10d import _nvshmemx_cumodule_init

    compiled._init_handles()
    key = compiled.module
    if key not in _initialized_modules:
        result = _nvshmemx_cumodule_init(key)
        if result not in (None, 0):
            raise RuntimeError(f"nvshmemx_cumodule_init returned {result}")
        # Retain the kernel: a recycled CUmodule handle must not hit a stale key.
        _initialized_modules[key] = compiled
    return compiled


# internode/contracts.py


class CombineTail(NamedTuple):
    state: object
    epoch: object
    tokens: object
    node: int
    operation: object


class MathPublication(NamedTuple):
    barrier: object
    peer_barriers: object
    control: object
    epoch: object
    completion: object
    tail: CombineTail | None = None


# internode/context.py


@dataclass(frozen=True)
class InternodeRegisteredInputs:
    context: object
    registered: object
    epoch: int
    num_tokens: int
    tokens_bound: int | None = None
    deferred: tuple | None = None


@dataclass
class InternodeWorkspace:
    pool: object
    l1_arrival: torch.Tensor
    actual_num_pool_rows: torch.Tensor
    control: torch.Tensor
    num_pool_rows: int
    num_padded_sf_pool_tokens: int
    dispatch_descs: tuple
    l2_acts: torch.Tensor | None = None
    l2_acts_sf_mn_major: torch.Tensor | None = None
    l2_arrival: torch.Tensor | None = None
    descriptor_set: object = None
    weight_identity: tuple | None = None
    stage_occupancy: dict | None = None
    fc2_arrival: torch.Tensor | None = None
    fc2_expert_blocks: torch.Tensor | None = None
    combine_row_claims: torch.Tensor | None = None
    combine_slot_ready: torch.Tensor | None = None

    @property
    def math_occupancy(self):
        return (self.stage_occupancy or {}).get("math")


class InternodeContext:
    """Symmetric storage and peer mappings for two eight-GPU NVSHMEM nodes."""

    def __init__(
        self,
        *,
        hidden,
        num_experts,
        topk=6,
        max_tokens=128,
        partial_dtype=torch.float16,
        group=None,
    ):
        if not dist.is_initialized():
            raise RuntimeError("initialize the physical process group first")
        if hidden % 128 or not 128 <= hidden <= 8192 or num_experts % 16:
            raise ValueError("EP16 requires H in [128,8192]/128 and E divisible by 16")
        if not 0 < max_tokens <= 128 or not 0 < topk <= min(32, num_experts):
            raise ValueError("invalid FP8 decode capacity")
        if partial_dtype not in (torch.bfloat16, torch.float32, torch.float16):
            raise ValueError("partial_dtype must be BF16, FP32, or block-scaled FP16")
        self.group = dist.group.WORLD if group is None else group
        self.physical_size = dist.get_world_size(self.group)
        physical_rank = dist.get_rank(self.group)
        if self.physical_size != 16:
            raise ValueError("EP16 requires sixteen physical ranks")
        self.node_id = physical_rank // 8
        self.local_rank = physical_rank % 8
        self.rank = self.node_id * 8 + self.local_rank
        self.world_size = 16
        self.local_participants = 8
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.physical_sms = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        self.hidden, self.num_experts, self.topk = hidden, num_experts, topk
        self.max_tokens, self.max_routes = max_tokens, max_tokens * topk
        self.experts_per_rank = num_experts // 16
        self.partial_dtype = partial_dtype
        self.epoch = 0
        self._stage_epochs = dict(dispatch=0, math=0, combine=0)
        self._barrier_epochs = dict(dispatch=0, fused=0, combine=0)
        self.arena = create_arena(backend="NVSHMEM", group=self.group)
        e, m, h, k = self.experts_per_rank, max_tokens, hidden, topk
        # FP16 wire rows carry one FP32 scale per 512 columns. Round the
        # trailer to 16 bytes so every row and scale pointer remain aligned;
        # data and scales travel in the same put+signal payload.
        wire_h = (
            h + ((h + 511) // 512 + 3) // 4 * 8 if partial_dtype == torch.float16 else h
        )
        self.combine_record_bytes = (
            16 + wire_h * torch.empty((), dtype=partial_dtype).element_size()
        )
        specs = [
            ("input_acts", (2, m, h), torch.float8_e4m3fn),
            ("input_sf", (2, m, h // 128), torch.float32),
            ("input_metadata", (2, m * k + 1), torch.int64),
            ("input_topk_weights", (2, m, k), torch.float32),
            ("landing_acts", (m, h), torch.float8_e4m3fn),
            ("landing_sf", (m, h // 128), torch.float32),
            ("landing_metadata", (m * k + 1,), torch.int64),
            ("landing_topk_weights", (m, k), torch.float32),
            ("source_routes", (16, e, m * k), torch.int32),
            ("recv_count", (16, e), torch.int32),
            ("expert_state", (2, e), torch.int64),
            ("fc2_expert_done", (e,), torch.int32),
            ("combine_buffer", (m, k, h), torch.bfloat16),
            ("combine_stage", (m, k, h), torch.bfloat16),
            ("combine_reduced", (2, m, wire_h), partial_dtype),
            ("combine_recv", (m, wire_h), partial_dtype),
            ("combine_send_records", (2, m, self.combine_record_bytes), torch.uint8),
            ("combine_recv_records", (m, self.combine_record_bytes), torch.uint8),
            ("combine_chunk_signals", (8,), torch.uint64),
            ("signals", (8,), torch.uint64),
            ("credit_payload", (2, 8), torch.uint64),
            ("node_dispatch_barrier", (1,), torch.int32),
            ("node_dispatch_consumed", (1,), torch.int32),
            ("node_fused_barrier", (1,), torch.int32),
            ("node_combine_barrier", (1,), torch.int32),
            ("node_combine_consumed", (1,), torch.int32),
        ]
        self._layout = {}
        total = 0
        for name, shape, dtype in specs:
            total = (total + 255) // 256 * 256
            size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            self._layout[name] = (total, size, shape, dtype)
            total += size
        self.storage, self.handle = self.arena.allocate(
            "internode", (total,), torch.uint8
        )
        self.storage.zero_()
        for name in self._layout:
            setattr(self, name, self._view(self.storage, name))
        self.input_topk_idx = self.input_metadata[:, : m * k].view(2, m, k)
        self.landing_topk_idx = self.landing_metadata[: m * k].view(m, k)
        self.landing_num_tokens = self.landing_metadata[m * k :]
        self.input_topk_idx.fill_(-1)
        self.landing_topk_idx.fill_(-1)
        # Peer handles stay alive with the arena; only same-node addresses are mapped.
        self._peer_storages = {}
        for local in range(self.local_participants):
            peer = self.node_id * 8 + local
            address = self.handle.buffer_ptrs[peer]
            if not address:
                raise RuntimeError(f"node-local peer {peer} is not directly mapped")
            self._peer_storages[local] = self.handle.get_buffer(
                peer, (total,), torch.uint8
            )
        for peer, address in enumerate(self.handle.buffer_ptrs):
            if bool(address) != (peer // 8 == self.node_id):
                raise RuntimeError("unexpected NVSHMEM peer mapping")
        self.peer_fc2_expert_done_ptrs = self._tensor_ptrs(
            [
                self._peer_view(local, "fc2_expert_done")
                for local in range(self.local_participants)
            ]
        )
        self.peer_source_routes_ptrs = self._logical_owner_ptrs("source_routes")
        self.peer_recv_count_ptrs = self._logical_owner_ptrs("recv_count")
        self.peer_expert_state_ptrs = tuple(
            self._logical_owner_ptrs("expert_state", p) for p in range(2)
        )
        for name in (
            "node_dispatch_barrier",
            "node_dispatch_consumed",
            "node_fused_barrier",
            "node_combine_barrier",
            "node_combine_consumed",
        ):
            setattr(
                self,
                "peer_" + name + "_ptrs",
                self._tensor_ptrs(
                    [
                        self._peer_view(local, name)
                        for local in range(self.local_participants)
                    ]
                ),
            )
        self.peer_input_sf_ptrs = tuple(
            self._source_ptrs("input_sf", "landing_sf", p) for p in range(2)
        )
        self.peer_input_topk_weights_ptrs = tuple(
            self._source_ptrs("input_topk_weights", "landing_topk_weights", p)
            for p in range(2)
        )
        self.peer_data_signal_ptrs = self._tensor_ptrs(
            [
                self._peer_view(local, "signals")[1:]
                for local in range(self.local_participants)
            ]
        )
        self.peer_input_acts = tuple(
            tuple(
                self._source_view(s, "input_acts", "landing_acts", p) for s in range(16)
            )
            for p in range(2)
        )
        self.peer_combine_target_ptrs = self._tensor_ptrs(
            [
                self._peer_view(
                    s % 8,
                    "combine_buffer" if s // 8 == self.node_id else "combine_stage",
                )
                for s in range(16)
            ]
        )
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

    def _view(self, storage, name):
        offset, size, shape, dtype = self._layout[name]
        return storage[offset : offset + size].view(dtype).view(shape)

    def _peer_view(self, local, name):
        return self._view(self._peer_storages[local % self.local_participants], name)

    def _tensor_ptrs(self, tensors):
        return torch.tensor(
            [x.data_ptr() for x in tensors], dtype=torch.int64, device=self.device
        )

    def _logical_owner_ptrs(self, name, parity=None):
        pointers = []
        for rank in range(16):
            if rank // 8 != self.node_id:
                pointers.append(0)
            else:
                view = self._peer_view(rank % 8, name)
                pointers.append((view if parity is None else view[parity]).data_ptr())
        return torch.tensor(pointers, dtype=torch.int64, device=self.device)

    def _source_view(self, source, own, landing, parity):
        view = self._peer_view(
            source % 8, own if source // 8 == self.node_id else landing
        )
        return view[parity] if source // 8 == self.node_id else view

    def _source_ptrs(self, own, landing, parity):
        return self._tensor_ptrs(
            [self._source_view(s, own, landing, parity) for s in range(16)]
        )

    def mirror(self, rank=None):
        return ((self.rank if rank is None else rank) + 8) % 16

    def slot(self, node):
        if node != 1 - self.node_id:
            raise ValueError("only the other node has a landing slot")
        return 0

    def registration_view(self, epoch):
        p = epoch % 2
        return SimpleNamespace(
            max_tokens=self.max_tokens,
            hidden=self.hidden,
            topk=self.topk,
            device=self.device,
            input_acts=self.input_acts[p],
            input_sf=self.input_sf[p],
            input_topk_idx=self.input_topk_idx[p],
            input_topk_weights=self.input_topk_weights[p],
        )

    def register_inputs(
        self, x, topk_idx, topk_weights, *, x_sf=None, tokens_bound=None, defer=False
    ):

        tokens_bound = resolve_tokens_bound(self, x.shape[0], tokens_bound)
        self.epoch += 1
        if self.epoch >= (1 << 27):
            raise RuntimeError("recreate context before int32 node counters wrap")
        registered = register_inputs(
            self.registration_view(self.epoch),
            x,
            topk_idx,
            topk_weights,
            x_sf=x_sf,
            launch=not defer,
        )
        if not defer:
            self.input_metadata[self.epoch % 2, -1:].fill_(x.shape[0])
        deferred = (x, x_sf, topk_idx, topk_weights) if defer else None
        return InternodeRegisteredInputs(
            self, registered, self.epoch, x.shape[0], tokens_bound, deferred
        )

    def stage_epoch(self, stage):
        if self._stage_epochs[stage] + 1 >= (1 << 27):
            raise RuntimeError("recreate context before int32 node counters wrap")
        self._stage_epochs[stage] += 1
        return self._stage_epochs[stage]

    def barrier_epoch(self, barrier):
        """Count arrivals to this barrier, independently of stage executions."""
        value = self._barrier_epochs[barrier] + 1
        if value >= (1 << 27):
            raise RuntimeError("recreate context before int32 node counters wrap")
        self._barrier_epochs[barrier] = value
        return value

    def create_workspace(self, epoch):
        rows = triton.cdiv(16 * self.max_routes + self.experts_per_rank * 63, 64) * 64
        sf_rows = rows // 64 * 128
        pool = ExpertPool(
            torch.empty(
                (rows, self.hidden), dtype=torch.float8_e4m3fn, device=self.device
            ),
            torch.empty(
                (self.hidden // 128, sf_rows), dtype=torch.float32, device=self.device
            ),
            torch.empty(rows, dtype=torch.float32, device=self.device),
            torch.empty((rows, 3), dtype=torch.int64, device=self.device),
            self.expert_state[epoch % 2],
            self.source_routes,
        )
        descriptors = tuple(
            create_dispatch_descriptors(
                SimpleNamespace(
                    hidden=self.hidden,
                    max_tokens=self.max_tokens,
                    world_size=16,
                    peer_input_acts=self.peer_input_acts[p],
                ),
                pool.acts,
            )
            for p in range(2)
        )
        workspace = InternodeWorkspace(
            pool,
            torch.zeros(rows // 64, dtype=torch.int32, device=self.device),
            torch.zeros((), dtype=torch.int32, device=self.device),
            torch.zeros(32, dtype=torch.int32, device=self.device),
            rows,
            sf_rows,
            descriptors,
        )
        # The existing reset grid clears the pool-sized counter prefixes.
        # Tiny capacities may need more expert counters than pool blocks;
        # _prepare_math clears that extra tail. Done epochs are never reset.
        workspace.fc2_arrival = torch.empty_like(workspace.l1_arrival)
        workspace.fc2_expert_blocks = torch.empty(
            max(rows // 64, self.experts_per_rank),
            dtype=torch.int32,
            device=self.device,
        )
        return workspace


def create_context(**kwargs):
    return InternodeContext(**kwargs)


@dataclass(frozen=True)
class InternodeMoEResult:
    context: object
    workspace: InternodeWorkspace
    inputs: InternodeRegisteredInputs
    output: torch.Tensor
    config: object
    compiled: object
    num_sms: int


# internode/transport.py


@gluon.jit
def net_begin(signals, credit_channel: gl.constexpr, epoch):
    # Only program 0's issuing warp enters any NVSHMEM operation.
    _nvshmem_quiet_warp()
    if epoch > 1:
        _nvshmem_wait_peer_signal_ge(signals + credit_channel, epoch - 1)


@gluon.jit
def credit_return(payload, signals, channel: gl.constexpr, epoch, mirror: gl.constexpr):
    gl.store(payload, gl.cast(epoch, gl.uint64))
    # payload+4 is an unused symmetric destination. The signal is the credit.
    _nvshmem_putmem_signal_nbi_warp(
        payload + 4, payload, 8, signals + channel, epoch, mirror
    )


@gluon.jit
def net_poll_pause():
    # The issuer shares CTA0's SM with math. Yield between readiness loads so
    # a long wait does not turn that math CTA into the node's straggler.
    gl.inline_asm_elementwise(
        "nanosleep.u32 4000; mov.u32 $0, 0;",
        constraints="=r",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def net_combine_send(
    combine,
    payload,
    epoch,
    partitions: gl.constexpr,
    num_sms: gl.constexpr,
    hidden: gl.constexpr,
    partial_bytes: gl.constexpr,
    mirror: gl.constexpr,
):
    if gl.program_id(0) == 0:
        (reduced, landing_m, received, signals, control) = (
            combine[1],
            combine[3],
            combine[5],
            combine[8],
            combine[9],
        )
        ready = _load_i32_acquire_gpu(control + 2)
        while ready < num_sms * partitions:
            net_poll_pause()
            ready = _load_i32_acquire_gpu(control + 2)
        m = gl.load(landing_m).to(gl.int32)
        wire_h: gl.constexpr = (
            hidden + ((hidden + 511) // 512 + 3) // 4 * 8
            if reduced.dtype.element_ty == gl.float16
            else hidden
        )
        _nvshmem_putmem_signal_nbi_warp(
            received,
            reduced,
            gl.maximum(m, 1) * wire_h * partial_bytes,
            signals + 3,
            epoch,
            mirror,
        )
        # Complete empty/all-local generations too, then return receiver credit.
        _nvshmem_wait_peer_signal_ge(signals + 3, epoch)
        ready = _load_i32_acquire_gpu(control + 4)
        while ready < 1:
            ready = _load_i32_acquire_gpu(control + 4)
        credit_return(payload, signals, 4, epoch, mirror)


@gluon.jit
def fused_net_send(network, hidden: gl.constexpr, mirror: gl.constexpr):
    (combine, payload, epoch, partitions, num_sms, partial_bytes) = network[12]
    # Dispatch N0 already quiets all prior puts and acquires combine credit.
    # Distinct credit-payload words keep both NBI sources alive until next N0.
    if gl.program_id(0) == 0:
        net_combine_send(
            combine, payload, epoch, partitions, num_sms, hidden, partial_bytes, mirror
        )


# internode/dispatch.py


@gluon.jit
def net_dispatch_send(
    network,
    epoch,
    num_tokens,
    max_routes: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    mirror: gl.constexpr,
):
    (
        src_meta,
        src_acts,
        src_sf,
        src_weights,
        dst_meta,
        dst_acts,
        dst_sf,
        dst_weights,
        signals,
        payload,
        control,
        previous_combine_epoch,
    ) = network
    _nvshmem_putmem_signal_nbi_warp(
        dst_meta, src_meta, (max_routes + 1) * 8, signals, epoch, mirror
    )
    rows = gl.maximum(num_tokens, 1)
    _nvshmem_putmem_nbi_warp(dst_acts, src_acts, rows * hidden, mirror)
    _nvshmem_putmem_nbi_warp(dst_sf, src_sf, rows * (hidden // 128) * 4, mirror)
    # Stock 3.4.5 IBRC put_signal internally enqueues proxy_fence before the
    # signal AMO, ordering the preceding same-PE puts on this issuing channel.
    _nvshmem_putmem_signal_nbi_warp(
        dst_weights, src_weights, rows * topk * 4, signals + 1, epoch, mirror
    )


@gluon.jit
def dispatch_net_begin(
    network,
    epoch,
    num_tokens,
    max_routes: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    mirror: gl.constexpr,
    dispatch_tail_quiet: gl.constexpr = False,
):
    if gl.program_id(0) == 0:
        (signals, _payload, _control) = (network[8], network[9], network[10])
        if dispatch_tail_quiet:
            # Receiver credit still precedes landing-region reuse. Move the
            # source-completion quiet to the dispatch tail (D1).
            if epoch > 1:
                _nvshmem_wait_peer_signal_ge(signals + 2, epoch - 1)
        else:
            net_begin(signals, 2, epoch)
        if network[11] > 0:
            _nvshmem_wait_peer_signal_ge(signals + 4, network[11])
        net_dispatch_send(network, epoch, num_tokens, max_routes, hidden, topk, mirror)


@gluon.jit
def dispatch_credit_tail(
    network, epoch, mirror: gl.constexpr, dispatch_tail_quiet: gl.constexpr = False
):
    if gl.program_id(0) == 0:
        (signals, payload, control) = (network[8], network[9], network[10])
        ready = _load_i32_acquire_gpu(control + 4)
        while ready < 1:
            ready = _load_i32_acquire_gpu(control + 4)
        # Even an empty/all-local dispatch completes its incoming generation.
        _nvshmem_wait_peer_signal_ge(signals + 1, epoch)
        credit_return(payload, signals, 2, epoch, mirror)
        if dispatch_tail_quiet:
            _nvshmem_quiet_warp()


@gluon.jit
def dispatch_net_partition(
    network,
    epoch,
    num_tokens,
    max_routes: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    mirror: gl.constexpr,
    dispatch_tail_quiet: gl.constexpr = False,
):
    dispatch_net_begin(
        network,
        epoch,
        num_tokens,
        max_routes,
        hidden,
        topk,
        mirror,
        dispatch_tail_quiet=dispatch_tail_quiet,
    )
    dispatch_credit_tail(
        network, epoch, mirror, dispatch_tail_quiet=dispatch_tail_quiet
    )


@gluon.jit
def publish_route_counts(
    send_state,
    peer_counts,
    peer_states,
    dispatch_pid,
    workers: gl.constexpr,
    source: gl.constexpr,
    first_expert: gl.constexpr,
    node_experts: gl.constexpr,
    experts_per_rank: gl.constexpr,
):
    expert = first_expert + dispatch_pid
    while expert < first_expert + node_experts:
        owner = expert // experts_per_rank
        local = expert % experts_per_rank
        count = gl.load(send_state + expert).to(gl.int32)
        counts = gl.load(peer_counts + owner).to(gl.pointer_type(gl.int32))
        states = gl.load(peer_states + owner).to(gl.pointer_type(gl.int64))
        gl.store(counts + source * experts_per_rank + local, count)
        gl.atomic_add(
            states + local, count.to(gl.int64) + 4294967296, sem="release", scope="sys"
        )
        expert += workers


@gluon.jit
def dispatch_publish_handoff(
    control,
    peer_barriers,
    node_barrier,
    pid,
    epoch,
    workers: gl.constexpr,
    dispatch_rendezvous: gl.constexpr = True,
):
    """Publish route completion, optionally rendezvousing before pool consumption."""
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    gl.atomic_add(control + 1, 1, sem="release", scope="gpu")
    if pid == 0:
        ready = _load_i32_acquire_gpu(control + 1)
        while ready < workers:
            ready = _load_i32_acquire_gpu(control + 1)
        if dispatch_rendezvous:
            _peer_barrier_arrive_and_wait(
                peer_barriers, node_barrier, 8, epoch, layout, 32
            )
        # Without this rendezvous (D2), each owner still acquires all sixteen
        # packed-count publications. The consumed-parity barrier remains.
        gl.atomic_add(control + 2, 1, sem="release", scope="gpu")
    ready = _load_i32_acquire_gpu(control + 2)
    while ready < 1:
        ready = _load_i32_acquire_gpu(control + 2)


@gluon.jit
def pool_dispatch_partition(
    dispatch,
    peers,
    descs,
    barrier,
    buffer,
    epoch,
    num_tokens,
    worker: gl.constexpr,
    num_sms: gl.constexpr,
    rank: gl.constexpr,
    node: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    num_experts: gl.constexpr,
    max_routes: gl.constexpr,
    sf_rows: gl.constexpr,
    dispatch_rendezvous: gl.constexpr = True,
):
    (
        own_idx,
        landing_idx,
        landing_m,
        signals,
        send_state,
        expert_state,
        inactive_state,
        source_routes,
        recv_count,
        l1_arrival,
        actual_rows,
        pool_sf,
        pool_weights,
        metadata,
        control,
    ) = dispatch
    (
        peer_routes,
        peer_counts,
        peer_states,
        peer_sf,
        peer_weights,
        peer_data_signals,
        node_barrier,
        peer_barriers,
        consumed,
        peer_consumed,
        rendezvous_epoch,
    ) = peers
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    er: gl.constexpr = num_experts // 16
    workers: gl.constexpr = num_sms * 2
    mirror: gl.constexpr = (rank + 8) % 16
    pid = gl.program_id(0) * 2 + worker
    _reserve_routes(
        own_idx,
        send_state,
        peer_routes,
        num_tokens * topk,
        pid,
        workers,
        rank,
        er,
        max_routes,
        node * 8,
        8,
    )
    _nvshmem_wait_peer_signal_ge(signals, epoch)
    remote_m = gl.load(landing_m).to(gl.int32)
    _reserve_routes(
        landing_idx,
        send_state + num_experts,
        peer_routes,
        remote_m * topk,
        pid,
        workers,
        mirror,
        er,
        max_routes,
        node * 8,
        8,
    )
    gl.atomic_add(control, 1, sem="release", scope="gpu")
    ready = _load_i32_acquire_gpu(control)
    while ready < workers:
        ready = _load_i32_acquire_gpu(control)
    publish_route_counts(
        send_state,
        peer_counts,
        peer_states,
        pid,
        workers,
        rank,
        node * num_experts // 2,
        num_experts // 2,
        er,
    )
    publish_route_counts(
        send_state + num_experts,
        peer_counts,
        peer_states,
        pid,
        workers,
        mirror,
        node * num_experts // 2,
        num_experts // 2,
        er,
    )
    dispatch_publish_handoff(
        control,
        peer_barriers,
        node_barrier,
        pid,
        rendezvous_epoch,
        workers,
        dispatch_rendezvous=dispatch_rendezvous,
    )
    capacity: gl.constexpr = max(triton.next_power_of_2(er), 32)
    count_layout: gl.constexpr = gl.BlockedLayout(
        [max(capacity // 32, 1)], [32], [1], [0]
    )
    offsets = gl.arange(0, capacity, layout=count_layout)
    counts = _load_packed_expert_counts(expert_state, offsets, er, 16)
    if pid == 0:
        gl.store(
            inactive_state + offsets,
            gl.full((capacity,), 0, gl.int64, count_layout),
            offsets < er,
        )
        gl.store(
            actual_rows, gl.sum(gl.where(offsets < er, (counts + 63) // 64, 0), 0) * 64
        )
    token = pid
    expert = 0
    begin = 0
    end = 0
    pool_block = 0
    count = _packed_expert_count_from_cache(counts, offsets, 0)
    end = count
    while token >= end and expert < er:
        pool_block += (count + 63) // 64
        expert += 1
        begin = end
        count = (
            _packed_expert_count_from_cache(counts, offsets, expert)
            if expert < er
            else 0
        )
        end += count
    data_ready_mask = 0
    while expert < er:
        row = token - begin
        pool_row = pool_block * 64 + row
        source, route = _select_source_route(
            recv_count, source_routes, expert, row, er, max_routes, 16
        )
        if source // 8 != node:
            bit = 1 << (source % 8)
            if (data_ready_mask & bit) == 0:
                peer_signal = gl.load(peer_data_signals + source % 8).to(
                    gl.pointer_type(gl.uint64)
                )
                _nvshmem_wait_peer_signal_ge(peer_signal, epoch)
                data_ready_mask = data_ready_mask | bit
        _pull_dispatch_row(
            source,
            route,
            pool_row,
            (token // workers) & 1,
            peer_sf,
            peer_weights,
            descs[0],
            descs[1],
            barrier,
            buffer,
            pool_sf,
            pool_weights,
            metadata,
            l1_arrival,
            sf_rows,
            hidden,
            topk,
            64,
            16,
        )
        token += workers
        while token >= end and expert < er:
            pool_block += (count + 63) // 64
            expert += 1
            begin = end
            count = (
                _packed_expert_count_from_cache(counts, offsets, expert)
                if expert < er
                else 0
            )
            end += count
    gl.atomic_add(control + 3, 1, sem="release", scope="gpu")
    if pid == 0:
        ready = _load_i32_acquire_gpu(control + 3)
        while ready < workers:
            ready = _load_i32_acquire_gpu(control + 3)
        _peer_barrier_arrive_and_wait(peer_consumed, consumed, 8, epoch, layout, 32)
        gl.atomic_add(control + 4, 1, sem="release", scope="gpu")


@gluon.jit
def dispatch0_net_partition(
    dispatch,
    peers,
    descs,
    barrier,
    buffer,
    epoch,
    num_tokens,
    worker: gl.constexpr,
    num_sms: gl.constexpr,
    rank: gl.constexpr,
    node: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    num_experts: gl.constexpr,
    max_routes: gl.constexpr,
    sf_rows: gl.constexpr,
    network,
    dispatch_tail_quiet: gl.constexpr = False,
    dispatch_rendezvous: gl.constexpr = True,
):
    dispatch_net_begin(
        network,
        epoch,
        num_tokens,
        max_routes,
        hidden,
        topk,
        (rank + 8) % 16,
        dispatch_tail_quiet=dispatch_tail_quiet,
    )
    pool_dispatch_partition(
        dispatch,
        peers,
        descs,
        barrier,
        buffer,
        epoch,
        num_tokens,
        worker,
        num_sms,
        rank,
        node,
        hidden,
        topk,
        num_experts,
        max_routes,
        sf_rows,
        dispatch_rendezvous=dispatch_rendezvous,
    )
    # This must follow our own pool contribution, otherwise D5 waits on itself.
    dispatch_credit_tail(
        network, epoch, (rank + 8) % 16, dispatch_tail_quiet=dispatch_tail_quiet
    )


@gluon.jit
def fused_dispatch0_partition(
    dispatch,
    peers,
    descs,
    barrier,
    buffer,
    epoch,
    num_tokens,
    worker: gl.constexpr,
    num_sms: gl.constexpr,
    rank: gl.constexpr,
    node: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    num_experts: gl.constexpr,
    max_routes: gl.constexpr,
    sf_rows: gl.constexpr,
    network,
    dispatch_tail_quiet: gl.constexpr = False,
    dispatch_rendezvous: gl.constexpr = True,
):
    dispatch0_net_partition(
        dispatch,
        peers,
        descs,
        barrier,
        buffer,
        epoch,
        num_tokens,
        worker,
        num_sms,
        rank,
        node,
        hidden,
        topk,
        num_experts,
        max_routes,
        sf_rows,
        network[:12],
        dispatch_tail_quiet=dispatch_tail_quiet,
        dispatch_rendezvous=dispatch_rendezvous,
    )
    fused_net_send(network, hidden, (rank + 8) % 16)


# internode/epilogues.py


@gluon.jit
def fc2_tile_complete(
    state, expert, pool_block, m_blocks, fragments_per_block: gl.constexpr
):
    arrivals, expert_blocks, done, epoch = state
    # All scatter-store lanes must reach this boundary before the elected
    # scalar atomic can publish them. Each split math partition arrives once.
    partition_barrier()
    old = gl.atomic_add(arrivals + pool_block, 1, sem="acq_rel", scope="gpu")
    if old == fragments_per_block - 1:
        blocks = gl.atomic_add(expert_blocks + expert, 1, sem="acq_rel", scope="gpu")
        if blocks == m_blocks - 1:
            # The two acquire/release chains collect stores from all CTAs.
            # A peer acquires this system-scope epoch before reading slots.
            gl.atomic_xchg(done + expert, epoch, sem="release", scope="sys")


@gluon.jit
def fc2_empty_experts(
    state,
    expert_state,
    E: gl.constexpr,
    num_warps: gl.constexpr,
    partition: gl.constexpr,
):
    if partition == 0:
        if gl.program_id(0) == 0:
            offsets = gl.arange(
                0,
                triton.next_power_of_2(E),
                layout=gl.BlockedLayout([1], [32], [num_warps], [0]),
            )
            # Reacquire counts in this layout: a different lane may have
            # acquired the same expert in the body's scheduler layout. Empty
            # experts have no scatter task and therefore no tile callback.
            counts = _load_packed_expert_counts(expert_state, offsets, E, 16)
            gl.atomic_xchg(
                state[2] + offsets,
                gl.full(
                    (triton.next_power_of_2(E),),
                    0,
                    gl.int32,
                    gl.BlockedLayout([1], [32], [num_warps], [0]),
                )
                + state[3],
                (offsets < E) & (counts == 0),
                sem="release",
                scope="sys",
            )


@gluon.jit
def fc2_publish(
    publish_state,
    math_done,
    partition_idx: gl.constexpr,
    num_partitions: gl.constexpr,
    num_math_warps: gl.constexpr,
    num_sms: gl.constexpr,
    local_participants: gl.constexpr,
):
    (barrier, peer_barriers, control, phase_epoch) = publish_state[:4]
    partition_barrier()
    if num_partitions > 1:
        mbarrier.arrive(math_done, count=1)
        mbarrier.wait(math_done, 0)
    if partition_idx == 0:
        gl.atomic_add(control + 0, 1, sem="release", scope="gpu")
        arrived = _load_i32_acquire_gpu(control + 0)
        while arrived < num_sms:
            arrived = _load_i32_acquire_gpu(control + 0)
        if gl.program_id(0) == 0:
            layout: gl.constexpr = gl.BlockedLayout([1], [32], [num_math_warps], [0])
            _peer_barrier_arrive_and_wait(
                peer_barriers,
                barrier,
                local_participants,
                phase_epoch,
                layout,
                num_math_warps * 32,
            )
            gl.atomic_add(control + 1, 1, sem="release", scope="gpu")
    ready = _load_i32_acquire_gpu(control + 1)
    while ready < 1:
        ready = _load_i32_acquire_gpu(control + 1)
    partition_barrier()


@gluon.jit
def fc2_publish_tail(
    publish_state,
    math_done,
    partition_idx: gl.constexpr,
    num_partitions: gl.constexpr,
    num_math_warps: gl.constexpr,
    num_sms: gl.constexpr,
    local_participants: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    experts: gl.constexpr,
):
    fc2_publish(
        publish_state,
        math_done,
        partition_idx,
        num_partitions,
        num_math_warps,
        num_sms,
        local_participants,
    )
    if publish_state.tail is not None:
        operation: gl.constexpr = publish_state.tail.operation
        operation(
            publish_state.tail.state,
            publish_state.tail.epoch,
            publish_state.tail.tokens,
            partition_idx,
            num_math_warps,
            num_partitions,
            num_sms,
            hidden,
            topk,
            experts,
            publish_state.tail.node,
        )


# internode/combine.py


@gluon.jit
def prereduce_rows(
    stage,
    reduced,
    landing_idx,
    landing_m,
    control,
    partition: gl.constexpr,
    warps: gl.constexpr,
    partitions: gl.constexpr,
    num_sms: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    num_experts: gl.constexpr,
    node: gl.constexpr,
):
    """Each warp owns one (row, 512-column block). Slot reduction order, scaled wire encoding, and per-partition completion remain unchanged."""
    scaled: gl.constexpr = reduced.dtype.element_ty == gl.float16
    wire_h: gl.constexpr = (
        hidden + ((hidden + 511) // 512 + 3) // 4 * 8 if scaled else hidden
    )
    width: gl.constexpr = 512
    layout: gl.constexpr = gl.BlockedLayout([1, 16], [1, 32], [warps, 1], [1, 0])
    row_offsets = gl.arange(0, warps, layout=gl.SliceLayout(1, layout))
    col_offsets = gl.arange(0, width, layout=gl.SliceLayout(0, layout))
    m = gl.load(landing_m).to(gl.int32)
    begin = gl.program_id(0) * (warps * partitions) + partition * warps
    num_h_blocks: gl.constexpr = triton.cdiv(hidden, width)
    while begin < m * num_h_blocks:
        tile_ids = begin + row_offsets
        rows = tile_ids // num_h_blocks
        block = tile_ids % num_h_blocks
        local_mask = gl.full((warps,), 0, gl.int32, gl.SliceLayout(1, layout))
        for slot in gl.static_range(topk):
            expert = gl.load(landing_idx + rows * topk + slot, rows < m, other=-1)
            valid = (rows < m) & (expert >= 0) & (expert // (num_experts // 2) == node)
            local_mask = local_mask | gl.where(valid, 1 << slot, 0)
        cols = block[:, None] * width + col_offsets[None, :]
        value = gl.full((warps, width), 0, gl.float32, layout)
        for slot in gl.static_range(topk):
            value += gl.load(
                stage + (rows[:, None] * topk + slot) * hidden + cols,
                (local_mask[:, None] & 1 << slot != 0) & (cols < hidden),
                other=0,
            ).to(gl.float32)
        if scaled:
            scale = gl.maximum(gl.max(gl.abs(value), 1), 1e-10) * (1.0 / 32752.0)
            inverse = gl.div_rn(1.0, scale)
            scale_ptr = (reduced + rows * wire_h + hidden + 2 * block).to(
                gl.pointer_type(gl.float32)
            )
            gl.store(scale_ptr, scale, rows < m)
            value = value * inverse[:, None]
        gl.store(
            reduced + rows[:, None] * wire_h + cols,
            value,
            (rows[:, None] < m) & (cols < hidden),
        )
        begin += num_sms * warps * partitions
    partition_barrier()
    gl.atomic_add(control + 2, 1, sem="release", scope="gpu")


@gluon.jit
def final_reduce_rows(
    buffer,
    received,
    output,
    own_idx,
    signals,
    epoch,
    num_tokens,
    partition: gl.constexpr,
    warps: gl.constexpr,
    partitions: gl.constexpr,
    num_sms: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    num_experts: gl.constexpr,
    node: gl.constexpr,
):
    """Each warp owns one (row, 512-column block). Slot reduction order, scaled wire encoding, and per-partition completion remain unchanged."""
    scaled: gl.constexpr = received.dtype.element_ty == gl.float16
    wire_h: gl.constexpr = (
        hidden + ((hidden + 511) // 512 + 3) // 4 * 8 if scaled else hidden
    )
    width: gl.constexpr = 512
    layout: gl.constexpr = gl.BlockedLayout([1, 16], [1, 32], [warps, 1], [1, 0])
    row_offsets = gl.arange(0, warps, layout=gl.SliceLayout(1, layout))
    col_offsets = gl.arange(0, width, layout=gl.SliceLayout(0, layout))
    begin = gl.program_id(0) * (warps * partitions) + partition * warps
    arrived = False
    num_h_blocks: gl.constexpr = triton.cdiv(hidden, width)
    while begin < num_tokens * num_h_blocks:
        tile_ids = begin + row_offsets
        rows = tile_ids // num_h_blocks
        block = tile_ids % num_h_blocks
        local_mask = gl.full((warps,), 0, gl.int32, gl.SliceLayout(1, layout))
        remote = gl.full((warps,), False, gl.int1, gl.SliceLayout(1, layout))
        for slot in gl.static_range(topk):
            expert = gl.load(own_idx + rows * topk + slot, rows < num_tokens, other=-1)
            valid = (rows < num_tokens) & (expert >= 0)
            local = expert // (num_experts // 2) == node
            local_mask = local_mask | gl.where(valid & local, 1 << slot, 0)
            remote = remote | valid & ~local
        cols = block[:, None] * width + col_offsets[None, :]
        value = gl.full((warps, width), 0, gl.float32, layout)
        for slot in gl.static_range(topk):
            value += gl.load(
                buffer + (rows[:, None] * topk + slot) * hidden + cols,
                (local_mask[:, None] & 1 << slot != 0) & (cols < hidden),
                other=0,
            ).to(gl.float32)
        if gl.sum(remote.to(gl.int32), 0) > 0:
            if not arrived:
                _nvshmem_wait_peer_signal_ge(signals + 3, epoch)
                arrived = True
            partial = gl.load(
                received + rows[:, None] * wire_h + cols,
                remote[:, None] & (cols < hidden),
                other=0,
            ).to(gl.float32)
            if scaled:
                scale_ptr = (received + rows * wire_h + hidden + 2 * block).to(
                    gl.pointer_type(gl.float32)
                )
                scale = gl.load(scale_ptr, remote, other=0)
                partial = gl.inline_asm_elementwise(
                    "mul.rn.f32 $0, $1, $2;",
                    constraints="=f,f,f",
                    args=[partial, scale[:, None]],
                    dtype=gl.float32,
                    is_pure=True,
                    pack=1,
                )
            value += partial
        gl.store(
            output + rows[:, None] * hidden + cols,
            value,
            (rows[:, None] < num_tokens) & (cols < hidden),
        )
        begin += num_sms * warps * partitions


@gluon.jit
def combine_tail(
    combine,
    epoch,
    num_tokens,
    partition: gl.constexpr,
    warps: gl.constexpr,
    partitions: gl.constexpr,
    num_sms: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    num_experts: gl.constexpr,
    node: gl.constexpr,
):
    (
        stage,
        reduced,
        landing_idx,
        landing_m,
        buffer,
        received,
        output,
        own_idx,
        signals,
        control,
        consumed,
        peer_consumed,
    ) = combine
    prereduce_rows(
        stage,
        reduced,
        landing_idx,
        landing_m,
        control,
        partition,
        warps,
        partitions,
        num_sms,
        hidden,
        topk,
        num_experts,
        node,
    )
    final_reduce_rows(
        buffer,
        received,
        output,
        own_idx,
        signals,
        epoch,
        num_tokens,
        partition,
        warps,
        partitions,
        num_sms,
        hidden,
        topk,
        num_experts,
        node,
    )
    partition_barrier()
    gl.atomic_add(control + 3, 1, sem="release", scope="gpu")
    if partition == 0 and gl.program_id(0) == 0:
        ready = _load_i32_acquire_gpu(control + 3)
        while ready < num_sms * partitions:
            ready = _load_i32_acquire_gpu(control + 3)
        layout: gl.constexpr = gl.BlockedLayout([1], [32], [warps], [0])
        _peer_barrier_arrive_and_wait(
            peer_consumed, consumed, 8, epoch, layout, warps * 32
        )
        gl.atomic_add(control + 4, 1, sem="release", scope="gpu")


@gluon.jit
def prereduce_ready_row(
    stage,
    reduced,
    landing_idx,
    row,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    num_experts: gl.constexpr,
    node: gl.constexpr,
    output_row=None,
):
    if output_row is None:
        output_row = row
    scaled: gl.constexpr = reduced.dtype.element_ty == gl.float16
    wire_h: gl.constexpr = (
        hidden + ((hidden + 511) // 512 + 3) // 4 * 8 if scaled else hidden
    )
    layout: gl.constexpr = gl.BlockedLayout([16], [32], [1], [0])
    offsets = gl.arange(0, 512, layout=layout)
    for block in range(triton.cdiv(hidden, 512)):
        cols = block * 512 + offsets
        value = gl.full((512,), 0, gl.float32, layout)
        for slot in gl.static_range(topk):
            expert = gl.load(landing_idx + row * topk + slot)
            local = (expert >= 0) & (expert // (num_experts // 2) == node)
            value += gl.load(
                stage + (row * topk + slot) * hidden + cols,
                local & (cols < hidden),
                other=0,
            ).to(gl.float32)
        if scaled:
            scale = gl.maximum(gl.max(gl.abs(value), 0), 1e-10) * (1.0 / 32752.0)
            inverse = gl.div_rn(1.0, scale)
            scale_ptr = (reduced + output_row * wire_h + hidden + 2 * block).to(
                gl.pointer_type(gl.float32)
            )
            gl.store(scale_ptr, scale)
            value = value * inverse
        gl.store(reduced + output_row * wire_h + cols, value, cols < hidden)


@gluon.jit
def combine_chunk_geometry(m, record_bytes: gl.constexpr):
    # Bound message count by eight, with a 256 KiB target per message.
    count = gl.minimum(
        gl.maximum(gl.cdiv(gl.maximum(m, 1) * record_bytes, 262144), 1), 8
    )
    rows = gl.cdiv(gl.maximum(m, 1), count)
    return gl.cdiv(gl.maximum(m, 1), rows), rows


@gluon.jit
def prereduce_chunk_rows(
    combine,
    num_sms: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    num_experts: gl.constexpr,
    node: gl.constexpr,
    max_tokens: gl.constexpr,
):
    stage, landing_idx, landing_m = combine[0], combine[2], combine[3]
    control = combine[9]
    (peer_done, math_epoch, claims) = (combine[12], combine[13], combine[14])
    (send, recv, chunk_signals, slot_ready) = combine[15]
    partial_type: gl.constexpr = combine[1].dtype.element_ty
    scaled: gl.constexpr = partial_type == gl.float16
    wire_h: gl.constexpr = (
        hidden + ((hidden + 511) // 512 + 3) // 4 * 8 if scaled else hidden
    )
    record_bytes: gl.constexpr = 16 + wire_h * (partial_type.primitive_bitwidth // 8)
    m = gl.load(landing_m).to(gl.int32)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    workers: gl.constexpr = num_sms - 1
    rows_per_cta: gl.constexpr = triton.next_power_of_2(
        triton.cdiv(max_tokens, workers)
    )
    rows = gl.program_id(0) - 1 + gl.arange(0, rows_per_cta, layout=layout) * workers
    pending = rows < m
    while gl.sum(pending.to(gl.int32), 0) > 0:
        ready = pending
        for slot in gl.static_range(topk):
            expert = gl.load(landing_idx + rows * topk + slot, ready, other=-1).to(
                gl.int32
            )
            local = ready & (expert >= 0) & (expert // (num_experts // 2) == node)
            safe = gl.where(local, expert, 0)
            owner = (safe // (num_experts // 16)) % 8
            address = gl.load(peer_done + owner)
            pointer = address.to(gl.pointer_type(gl.int32)) + safe % (num_experts // 16)
            done = _load_i32_acquire_sys_if(pointer, local)
            ready = ready & (~local | (done >= math_epoch))
        row = gl.min(gl.where(ready, rows, m), 0)
        if row < m:
            # Reservation is deliberately separate from payload publication.
            send_slot = gl.atomic_add(control + 5, 1, sem="relaxed", scope="gpu")
            record = send + send_slot * record_bytes
            gl.store(claims + row, 1)
            prereduce_ready_row(
                stage,
                (record + 16).to(gl.pointer_type(partial_type)),
                landing_idx,
                row,
                hidden,
                topk,
                num_experts,
                node,
                0,
            )
            gl.store(record.to(gl.pointer_type(gl.int64)), row.to(gl.int64))
            partition_barrier()
            gl.atomic_xchg(slot_ready + send_slot, 1, sem="release", scope="gpu")
            pending = pending & (rows != row)
        else:
            gl.inline_asm_elementwise(
                "nanosleep.u32 1000; mov.u32 $0, 0;",
                constraints="=r",
                args=[],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
    partition_barrier()
    gl.atomic_add(control + 2, 1, sem="release", scope="gpu")


@gluon.jit
def net_combine_send_chunks(
    combine,
    payload,
    epoch,
    num_sms: gl.constexpr,
    hidden: gl.constexpr,
    partial_bytes: gl.constexpr,
    mirror: gl.constexpr,
):
    (send, recv, chunk_signals, slot_ready) = combine[15]
    control = combine[9]
    scaled: gl.constexpr = combine[1].dtype.element_ty == gl.float16
    wire_h: gl.constexpr = (
        hidden + ((hidden + 511) // 512 + 3) // 4 * 8 if scaled else hidden
    )
    record_bytes: gl.constexpr = 16 + wire_h * partial_bytes
    m = gl.load(combine[3]).to(gl.int32)
    chunks, rows_per_chunk = combine_chunk_geometry(m, record_bytes)
    for chunk in range(chunks):
        start = chunk * rows_per_chunk
        stop = gl.minimum(start + rows_per_chunk, m)
        for slot in range(start, stop):
            ready = _load_i32_acquire_gpu(slot_ready + slot)
            while ready < 1:
                net_poll_pause()
                ready = _load_i32_acquire_gpu(slot_ready + slot)
        if m == 0:
            gl.store(send.to(gl.pointer_type(gl.int64)), 0)
        partition_barrier()
        _nvshmem_putmem_signal_nbi_warp(
            recv + start * record_bytes,
            send + start * record_bytes,
            gl.maximum((stop - start) * record_bytes, 8),
            chunk_signals + chunk,
            epoch,
            mirror,
        )
    # Incoming geometry uses own M, which can differ from the mirror's M.
    own_m = combine[16]
    incoming_chunks, _ = combine_chunk_geometry(own_m, record_bytes)
    for chunk in range(incoming_chunks):
        _nvshmem_wait_peer_signal_ge(chunk_signals + chunk, epoch)
    ready = _load_i32_acquire_gpu(control + 4)
    while ready < 1:
        ready = _load_i32_acquire_gpu(control + 4)
    credit_return(payload, combine[8], 4, epoch, mirror)


@gluon.jit
def final_reduce_chunks(
    combine,
    epoch,
    num_tokens,
    partition: gl.constexpr,
    warps: gl.constexpr,
    partitions: gl.constexpr,
    num_sms: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    num_experts: gl.constexpr,
    node: gl.constexpr,
):
    buffer, output, own_idx = combine[4], combine[6], combine[7]
    (send, recv, chunk_signals, slot_ready) = combine[15]
    partial_type: gl.constexpr = combine[1].dtype.element_ty
    scaled: gl.constexpr = partial_type == gl.float16
    wire_h: gl.constexpr = (
        hidden + ((hidden + 511) // 512 + 3) // 4 * 8 if scaled else hidden
    )
    record_bytes: gl.constexpr = 16 + wire_h * (partial_type.primitive_bitwidth // 8)
    layout: gl.constexpr = gl.BlockedLayout([1, 16], [1, 32], [warps, 1], [1, 0])
    row_offsets = gl.arange(0, warps, layout=gl.SliceLayout(1, layout))
    col_offsets = gl.arange(0, 512, layout=gl.SliceLayout(0, layout))
    h_blocks: gl.constexpr = triton.cdiv(hidden, 512)
    chunks, rows_per_chunk = combine_chunk_geometry(num_tokens, record_bytes)
    for chunk in range(chunks):
        _nvshmem_wait_peer_signal_ge(chunk_signals + chunk, epoch)
        count = gl.minimum(rows_per_chunk, num_tokens - chunk * rows_per_chunk)
        begin = gl.program_id(0) * (warps * partitions) + partition * warps
        while begin < count * h_blocks:
            tiles = begin + row_offsets
            slots = chunk * rows_per_chunk + tiles // h_blocks
            valid = tiles < count * h_blocks
            block = tiles % h_blocks
            record = recv + slots * record_bytes
            raw_rows = gl.load(record.to(gl.pointer_type(gl.int64)), valid, other=0)
            row_in_bounds = (raw_rows >= 0) & (raw_rows < num_tokens)
            # Validate the wire's int64 ID before narrowing or forming any
            # metadata/output address. Count each rejected record once, even
            # when its hidden dimension spans multiple CTA-owned tiles.
            rejected = gl.sum((valid & ~row_in_bounds & (block == 0)).to(gl.int32), 0)
            if rejected > 0:
                gl.atomic_add(combine[9] + 6, rejected, sem="relaxed", scope="gpu")
            valid = valid & row_in_bounds
            rows = gl.where(valid, raw_rows, 0).to(gl.int32)
            cols = block[:, None] * 512 + col_offsets[None, :]
            local_mask = gl.full((warps,), 0, gl.int32, gl.SliceLayout(1, layout))
            remote = gl.full((warps,), False, gl.int1, gl.SliceLayout(1, layout))
            for slot in gl.static_range(topk):
                expert = gl.load(own_idx + rows * topk + slot, valid, other=-1)
                active = valid & (expert >= 0)
                local = expert // (num_experts // 2) == node
                local_mask = local_mask | gl.where(active & local, 1 << slot, 0)
                remote = remote | active & ~local
            value = gl.full((warps, 512), 0, gl.float32, layout)
            for slot in gl.static_range(topk):
                value += gl.load(
                    buffer + (rows[:, None] * topk + slot) * hidden + cols,
                    (local_mask[:, None] & 1 << slot != 0) & (cols < hidden),
                    other=0,
                ).to(gl.float32)
            data = (record + 16).to(gl.pointer_type(partial_type))
            partial = gl.load(
                data[:, None] + cols, remote[:, None] & (cols < hidden), other=0
            ).to(gl.float32)
            if scaled:
                scale_ptr = (data + hidden + 2 * block).to(gl.pointer_type(gl.float32))
                scale = gl.load(scale_ptr, remote, other=0)
                partial = gl.inline_asm_elementwise(
                    "mul.rn.f32 $0, $1, $2;",
                    constraints="=f,f,f",
                    args=[partial, scale[:, None]],
                    dtype=gl.float32,
                    is_pure=True,
                    pack=1,
                )
            value += partial
            gl.store(
                output + rows[:, None] * hidden + cols,
                value,
                valid[:, None] & (cols < hidden),
            )
            begin += num_sms * warps * partitions


@gluon.jit
def combine_ready_tail(
    combine,
    epoch,
    num_tokens,
    partition: gl.constexpr,
    warps: gl.constexpr,
    partitions: gl.constexpr,
    num_sms: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    num_experts: gl.constexpr,
    node: gl.constexpr,
):
    (
        stage,
        reduced,
        landing_idx,
        landing_m,
        buffer,
        received,
        output,
        own_idx,
        signals,
        control,
        consumed,
        peer_consumed,
    ) = combine[:12]
    if len(combine) > 16:
        final_reduce_chunks(
            combine,
            epoch,
            num_tokens,
            partition,
            warps,
            partitions,
            num_sms,
            hidden,
            topk,
            num_experts,
            node,
        )
    else:
        final_reduce_rows(
            buffer,
            received,
            output,
            own_idx,
            signals,
            epoch,
            num_tokens,
            partition,
            warps,
            partitions,
            num_sms,
            hidden,
            topk,
            num_experts,
            node,
        )
    partition_barrier()
    gl.atomic_add(control + 3, 1, sem="release", scope="gpu")
    if partition == 0 and gl.program_id(0) == 0:
        ready = _load_i32_acquire_gpu(control + 3)
        while ready < num_sms * partitions:
            ready = _load_i32_acquire_gpu(control + 3)
        # Consumed must cover the independent readers of combine_stage too.
        reduced_done = _load_i32_acquire_gpu(control + 2)
        while reduced_done < num_sms:
            reduced_done = _load_i32_acquire_gpu(control + 2)
        layout: gl.constexpr = gl.BlockedLayout([1], [32], [warps], [0])
        _peer_barrier_arrive_and_wait(
            peer_consumed, consumed, 8, epoch, layout, warps * 32
        )
        gl.atomic_add(control + 4, 1, sem="release", scope="gpu")


@gluon.jit
def reset_composed(
    control,
    arrivals,
    num_blocks: gl.constexpr,
    BLOCK: gl.constexpr,
):
    i = gl.program_id(0) * BLOCK + gl.arange(
        0, BLOCK, layout=gl.BlockedLayout([1], [32], [4], [0])
    )
    gl.store(control + i, 0, i < 32)
    for j in gl.static_range(len(arrivals)):
        gl.store(arrivals[j] + i, 0, i < num_blocks)


@gluon.jit
def fused_net_partition(
    network,
    epoch,
    num_tokens,
    max_routes: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    mirror: gl.constexpr,
    dispatch_tail_quiet: gl.constexpr = False,
):
    dispatch_net_partition(
        network[:12],
        epoch,
        num_tokens,
        max_routes,
        hidden,
        topk,
        mirror,
        dispatch_tail_quiet=dispatch_tail_quiet,
    )
    fused_net_send(network, hidden, mirror)


# internode/bm16.py


@gluon.jit
def count_pool_blocks_before(
    stored_counts, count_offsets, expert, block_m: gl.constexpr
):
    """Return the BLOCK_M-padded pool prefix for ``expert``.

    This is the Gluon form of DeepGEMM's scheduler-local prefix reduction.
    It operates only on the cached count tensor and never reloads a host-built
    expert-offset table.
    """
    pool_m: gl.constexpr = max(64, block_m)
    blocks = (stored_counts + pool_m - 1) // pool_m * (pool_m // block_m)
    return gl.sum(gl.where(count_offsets < expert, blocks, 0), axis=0)


@gluon.jit
def _bm16_scheduler_next(
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
    while expert < num_experts and (not found):
        wave_end = gl.minimum(
            (expert + 1 + num_experts_per_wave - 1)
            // num_experts_per_wave
            * num_experts_per_wave,
            num_experts,
        )
        if phase == 1:
            while expert < wave_end and (not found):
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
                    pool_block_offset += gl.cdiv(current_count, max(64, block_m)) * (
                        max(64, block_m) // block_m
                    )
                    expert += 1
                    current_count = scheduler_count(
                        stored_counts, count_offsets, expert
                    )
            if not found:
                phase = 2
                expert = (expert - 1) // num_experts_per_wave * num_experts_per_wave
                current_count = scheduler_count(stored_counts, count_offsets, expert)
                pool_block_offset = count_pool_blocks_before(
                    stored_counts, count_offsets, expert, block_m
                )
        else:
            while expert < wave_end and (not found):
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
                    pool_block_offset += gl.cdiv(current_count, max(64, block_m)) * (
                        max(64, block_m) // block_m
                    )
                    expert += 1
                    current_count = scheduler_count(
                        stored_counts, count_offsets, expert
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
def bm16_a_partition(
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
    (stage_empty, stage_ready) = barriers
    (a_buffers, sfa_lo_buffers, sfa_hi_buffers) = buffers
    num_stages: gl.constexpr = a_buffers.type.shape[0]
    block_m: gl.constexpr = a_buffers.type.shape[1]
    block_k: gl.constexpr = a_buffers.type.shape[2]
    scheduler_layout: gl.constexpr = gl.BlockedLayout(
        [scheduler_counts_per_lane], [32], [1], [0]
    )
    count_offsets = gl.arange(0, scheduler_count_capacity, layout=scheduler_layout)
    if staged_dispatch_handoff:
        _wait_for_dispatch_handoff(dispatch_counter, 4 * num_sms + 1)
    stored_counts = _load_packed_expert_counts(
        expert_state, count_offsets, E, world_size
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
    ) = _bm16_scheduler_next(
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
        pool_m: gl.constexpr = max(64, block_m)
        subtiles: gl.constexpr = pool_m // block_m
        physical_pool_block = task_pool_block // subtiles
        scale_row_start = (
            physical_pool_block * _SF_BLOCK_M + task_pool_block % subtiles * block_m
        )
        physical_valid_rows = gl.minimum(
            pool_m, valid_count - local_row // pool_m * pool_m
        )
        if task_phase == 1:
            expected = physical_valid_rows
            ready = _load_i32_acquire_gpu(l1_arrival + physical_pool_block)
            while ready < expected:
                ready = _load_i32_acquire_gpu(l1_arrival + physical_pool_block)
        if task_phase == 2:
            ready = _load_i32_acquire_gpu(l2_arrival + physical_pool_block)
            expected_l2_arrivals = l1_n_blocks * gl.cdiv(physical_valid_rows, block_m)
            if fc2_arrival_counter:
                active_m_wgs = (gl.minimum(block_m, valid_count - local_row) + 63) // 64
                expected_l2_arrivals *= active_m_wgs * 2
            while ready < expected_l2_arrivals:
                ready = _load_i32_acquire_gpu(l2_arrival + physical_pool_block)
        num_k_tiles = gl.where(task_phase == 1, l1_k // block_k, l2_k // block_k)
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
                    sfa_lo_buffers.index(stage).slice(0, block_m, dim=0),
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
                    [k_tile * 2 * num_padded_sf_pool_tokens + scale_row_start],
                    stage_ready.index(stage),
                    sfa_lo_buffers.index(stage).slice(0, block_m, dim=0),
                )
                tma.async_copy_global_to_shared(
                    l2_sfa_desc,
                    [(k_tile * 2 + 1) * num_padded_sf_pool_tokens + scale_row_start],
                    stage_ready.index(stage),
                    sfa_hi_buffers.index(stage).slice(0, block_m, dim=0),
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
        ) = _bm16_scheduler_next(
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
def bm16_b_partition(
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
    (stage_empty, stage_ready) = barriers
    num_stages: gl.constexpr = b_buffers.type.shape[0]
    block_n: gl.constexpr = b_buffers.type.shape[1]
    block_k: gl.constexpr = b_buffers.type.shape[2]
    scheduler_layout: gl.constexpr = gl.BlockedLayout(
        [scheduler_counts_per_lane], [32], [1], [0]
    )
    count_offsets = gl.arange(0, scheduler_count_capacity, layout=scheduler_layout)
    if staged_dispatch_handoff:
        _wait_for_dispatch_handoff(dispatch_counter, 4 * num_sms + 1)
    stored_counts = _load_packed_expert_counts(
        expert_state, count_offsets, E, world_size
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
    ) = _bm16_scheduler_next(
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
                mbarrier.expect(stage_ready.index(stage), l1_b_desc.block_type.nbytes)
                tma.async_copy_global_to_shared(
                    l1_b_desc,
                    [flat_b_row_start, k_tile * block_k],
                    stage_ready.index(stage),
                    b_buffers.index(stage),
                )
            else:
                mbarrier.expect(stage_ready.index(stage), l2_b_desc.block_type.nbytes)
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
        ) = _bm16_scheduler_next(
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
def fc1_epilogue(
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
        (gate, up) = gl.split(paired)
        gate = gate.reshape((n_swap, block_n // 2))
        up = up.reshape((n_swap, block_n // 2))
        row_layout: gl.constexpr = gl.SliceLayout(1, gate.type.layout)
        row_offsets = gl.arange(0, n_swap, layout=row_layout)
    else:
        num_rows: gl.constexpr = block_m
        paired = final.reshape((block_m, block_n // 16, 2, 8)).permute((0, 1, 3, 2))
        (gate, up) = gl.split(paired)
        gate = gate.reshape((block_m, block_n // 2))
        up = up.reshape((block_m, block_n // 2))
        row_layout: gl.constexpr = gl.SliceLayout(1, gate.type.layout)
        row_offsets = gl.arange(0, block_m, layout=row_layout)
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
        amax = gl.max(gl.abs(activation), axis=1)
        scale = gl.maximum(amax, 1e-10) * (1.0 / 448.0)
        quantized = (activation * _reciprocal(scale[:, None], fast_math)).to(
            gl.float8e4nv
        )
        sf_pool_rows = (
            task_pool_block // (max(64, block_m) // block_m) * _SF_BLOCK_M
            + task_pool_block % (max(64, block_m) // block_m) * block_m
            + row_offsets
        )
        gl.store(
            l2_acts_sf + task_n_block * num_padded_sf_pool_tokens + sf_pool_rows,
            scale,
            mask=valid_rows,
        )
    else:
        num_scale_groups: gl.constexpr = block_n // 128
        activation_groups = activation.reshape((num_rows, num_scale_groups, 64))
        if n_swap == 128:
            activation_group_layout: gl.constexpr = gl.BlockedLayout(
                [1, 1, 1], [1, 1, 32], [8, 2, 1], [2, 1, 0]
            )
            activation_groups = gl.convert_layout(
                activation_groups, activation_group_layout
            )
        amax_groups = gl.max(gl.abs(activation_groups), axis=2)
        scale_groups = gl.maximum(amax_groups, 1e-10) * (1.0 / 448.0)
        quantized = (
            (activation_groups * _reciprocal(scale_groups[:, :, None], fast_math))
            .to(gl.float8e4nv)
            .reshape((num_rows, block_n // 2))
        )
        scale_row_layout: gl.constexpr = gl.SliceLayout(1, scale_groups.type.layout)
        scale_group_layout: gl.constexpr = gl.SliceLayout(0, scale_groups.type.layout)
        scale_rows = gl.arange(0, num_rows, layout=scale_row_layout)
        scale_group_offsets = gl.arange(0, num_scale_groups, layout=scale_group_layout)
        scale_pool_rows = (
            task_pool_block // (max(64, block_m) // block_m) * _SF_BLOCK_M
            + task_pool_block % (max(64, block_m) // block_m) * block_m
            + scale_rows[:, None]
        )
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
        [task_pool_block * block_m, task_n_block * (block_n // 2)],
        l2_epilogue_buffer,
    )
    tma.store_wait(0)
    partition_barrier()
    arrival_count = 1
    if fc2_arrival_counter:
        active_m_wgs = (valid_m + 63) // 64
        arrival_count = active_m_wgs * (block_n // 128)
    gl.atomic_add(
        l2_arrival + task_pool_block // (max(64, block_m) // block_m),
        arrival_count,
        sem="release",
        scope="gpu",
    )
    partition_barrier()


@gluon.jit
def bm16_math_body(
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
):
    """FC1, SwiGLU quantization and FC2 scatter; publication belongs to the caller."""
    (a_buffers, _, _, _) = buffers
    block_k: gl.constexpr = a_buffers.type.shape[2]
    scheduler_layout: gl.constexpr = gl.BlockedLayout(
        [scheduler_counts_per_lane], [32], [num_math_warps], [0]
    )
    count_offsets = gl.arange(0, scheduler_count_capacity, layout=scheduler_layout)
    if staged_dispatch_handoff:
        _wait_for_dispatch_handoff(dispatch_counter, 4 * num_sms + 1)
    stored_counts = _load_packed_expert_counts(
        expert_state, count_offsets, E, world_size
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
    ) = _bm16_scheduler_next(
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
        valid_m = gl.minimum(block_m, valid_count - local_row)
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
                fc1_epilogue(
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
                fc1_epilogue(
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
                fc1_epilogue(
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
                fc1_epilogue(
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
                fc1_epilogue(
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
                    max_num_imprecise_acc=max_num_imprecise_acc,
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
                    max_num_imprecise_acc=max_num_imprecise_acc,
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
                    max_num_imprecise_acc=max_num_imprecise_acc,
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
                    max_num_imprecise_acc=max_num_imprecise_acc,
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
                    max_num_imprecise_acc=max_num_imprecise_acc,
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
            physical_valid_rows = gl.minimum(
                max(64, block_m),
                valid_count - local_row // max(64, block_m) * max(64, block_m),
            )
            fc2_tile_complete(
                fc2_completion_state,
                task_expert,
                task_pool_block // (max(64, block_m) // block_m),
                gl.cdiv(valid_count, max(64, block_m)),
                l2_n_blocks * gl.cdiv(physical_valid_rows, block_m),
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
        ) = _bm16_scheduler_next(
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


# internode/math_swap.py


@gluon.jit
def swap_body_partition(
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
    math_done,
    publish_state,
    local_participants: gl.constexpr,
):
    body: gl.constexpr = bm16_math_body if block_m == 16 else math_swap_body
    body(
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
        max_num_imprecise_acc=32,
        fc2_tile_complete=fc2_tile_complete,
        fc2_completion_state=publish_state.completion,
    )
    fc2_empty_experts(publish_state.completion, expert_state, E, num_math_warps, 0)


@gluon.jit
def swap_math_partition(
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
    math_done,
    publish_state,
    local_participants: gl.constexpr,
):
    swap_body_partition(
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
        math_done,
        publish_state,
        local_participants,
    )
    fc2_publish_tail(
        publish_state,
        math_done,
        0,
        1,
        num_math_warps,
        num_sms,
        local_participants,
        l2_n,
        topk,
        E * world_size,
    )


# internode/math_split_bn128.py


@gluon.jit
def bm64_bn128_split_body_partition(
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
    math_done,
    publish_state,
    local_participants: gl.constexpr,
):
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
        fc2_tile_complete=fc2_tile_complete,
        fc2_completion_state=publish_state.completion,
    )
    fc2_empty_experts(publish_state.completion, expert_state, E, 4, math_partition_idx)


@gluon.jit
def bm64_bn128_split_math_partition(
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
    math_done,
    publish_state,
    local_participants: gl.constexpr,
):
    bm64_bn128_split_body_partition(
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
        math_done,
        publish_state,
        local_participants,
    )
    fc2_publish_tail(
        publish_state,
        math_done,
        math_partition_idx,
        2,
        4,
        num_sms,
        local_participants,
        l2_n,
        topk,
        E * world_size,
    )


# internode/math_split_bn256.py


@gluon.jit
def bm64_bn256_split_body_partition(
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
    math_done,
    publish_state,
    local_participants: gl.constexpr,
):
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
        fc2_tile_complete=fc2_tile_complete,
        fc2_completion_state=publish_state.completion,
    )
    fc2_empty_experts(publish_state.completion, expert_state, E, 4, math_partition_idx)


@gluon.jit
def bm64_bn256_split_math_partition(
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
    math_done,
    publish_state,
    local_participants: gl.constexpr,
):
    bm64_bn256_split_body_partition(
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
        math_done,
        publish_state,
        local_participants,
    )
    fc2_publish_tail(
        publish_state,
        math_done,
        math_partition_idx,
        2,
        4,
        num_sms,
        local_participants,
        l2_n,
        topk,
        E * world_size,
    )


# internode/chunked.py


@gluon.jit
def chunked_dispatch_partition(
    dispatch,
    peers,
    descs,
    barrier,
    buffer,
    epoch,
    num_tokens,
    worker: gl.constexpr,
    num_sms: gl.constexpr,
    rank: gl.constexpr,
    node: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    num_experts: gl.constexpr,
    max_routes: gl.constexpr,
    sf_rows: gl.constexpr,
    network,
    dispatch_tail_quiet: gl.constexpr = False,
    dispatch_rendezvous: gl.constexpr = True,
):
    dispatch0_net_partition(
        dispatch,
        peers,
        descs,
        barrier,
        buffer,
        epoch,
        num_tokens,
        worker,
        num_sms,
        rank,
        node,
        hidden,
        topk,
        num_experts,
        max_routes,
        sf_rows,
        network[:12],
        dispatch_tail_quiet=dispatch_tail_quiet,
        dispatch_rendezvous=dispatch_rendezvous,
    )
    (combine, payload, combine_epoch) = network[12][:3]
    partial_bytes: gl.constexpr = combine[1].dtype.element_ty.primitive_bitwidth // 8
    # Acquire the mirror's input metadata even when this rank has zero tokens.
    _nvshmem_wait_peer_signal_ge(network[8], epoch)
    if gl.program_id(0) == 0:
        gl.atomic_add(combine[9] + 2, 1, sem="release", scope="gpu")
        net_combine_send_chunks(
            combine,
            payload,
            combine_epoch,
            num_sms,
            hidden,
            partial_bytes,
            (rank + 8) % 16,
        )
    else:
        prereduce_chunk_rows(
            combine, num_sms, hidden, topk, num_experts, node, max_routes // topk
        )


# internode/prologue.py


@gluon.jit
def registration_reset_prologue(
    state,
    num_sms: gl.constexpr,
    warps: gl.constexpr,
    hidden: gl.constexpr,
    topk: gl.constexpr,
    max_tokens: gl.constexpr,
    experts: gl.constexpr,
):
    registration, reset, counter, target, metadata_m, send_state, send_size = state
    (
        x,
        x_sf,
        idx,
        weights,
        registered_x,
        registered_sf,
        registered_idx,
        registered_weights,
        m,
    ) = registration
    (control, arrivals) = reset
    blocks: gl.constexpr = triton.cdiv(16 * max_tokens * topk + experts * 63, 64)
    bf16: gl.constexpr = x.dtype.element_ty == gl.bfloat16
    quant_layout: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [warps, 1], [1, 0])
    route_layout: gl.constexpr = gl.BlockedLayout([1], [32], [warps], [0])
    count = m + gl.cdiv((max_tokens - m) * topk, 1024)
    bid = gl.program_id(0)
    while bid < count:
        register_inputs_kernel(
            x,
            x_sf,
            idx,
            weights,
            registered_x,
            registered_sf,
            registered_idx,
            registered_weights,
            m,
            max_tokens,
            hidden,
            hidden // 128,
            topk,
            1.0,
            bf16,
            quant_layout,
            route_layout,
            block_id=bid,
        )
        bid += num_sms
    reset_composed(control, arrivals, blocks, 256)
    offsets = gl.program_id(0) * 256 + gl.arange(0, 256, layout=route_layout)
    gl.store(send_state + offsets, 0, offsets < send_size)
    if gl.program_id(0) == 0:
        gl.store(metadata_m, m)
    partition_barrier()
    gl.atomic_add(counter, 1, sem="release", scope="gpu")
    ready = gl.inline_asm_elementwise(
        "ld.acquire.gpu.global.u64 $0, [$1];",
        "=l,l",
        [counter],
        dtype=gl.uint64,
        is_pure=False,
        pack=1,
    )
    while ready < target:
        ready = gl.inline_asm_elementwise(
            "ld.acquire.gpu.global.u64 $0, [$1];",
            "=l,l",
            [counter],
            dtype=gl.uint64,
            is_pure=False,
            pack=1,
        )
    partition_barrier()


# internode/kernel.py


# internode/kernel_fused.py


@gluon.jit
def dispatch_math_kernel(
    l1_a_desc,
    l1_sfa_desc,
    l1_b_desc,
    l2_store_desc,
    l2_a_desc,
    l2_sfa_desc,
    l2_b_desc,
    l2_acts_sf,
    token_src_metadata,
    peer_combine_buffer_ptrs,
    route_weights,
    l2_arrival,
    l1_arrival,
    l1_weight_scales,
    l2_weight_scales,
    expert_state,
    dispatch_counter,
    publish_state,
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
    max_tokens: gl.constexpr,
    topk: gl.constexpr,
    world_size: gl.constexpr,
    local_participants: gl.constexpr,
    activation_clamp: gl.constexpr,
    has_activation_clamp: gl.constexpr,
    fast_math: gl.constexpr,
    fc2_arrival_counter: gl.constexpr,
    fc2_epilogue_requires_full_sync: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    num_math_warps: gl.constexpr,
    num_stages: gl.constexpr,
    math_regs: gl.constexpr,
    tma_regs: gl.constexpr,
    mode: gl.constexpr,
    partitions: gl.constexpr,
    partition_source_keys: gl.constexpr,
    dispatch,
    peers,
    network,
    dispatch_desc_0,
    dispatch_desc_1,
    dispatch_desc_2,
    dispatch_desc_3,
    dispatch_desc_4,
    dispatch_desc_5,
    dispatch_desc_6,
    dispatch_desc_7,
    dispatch_desc_8,
    dispatch_desc_9,
    dispatch_desc_10,
    dispatch_desc_11,
    dispatch_desc_12,
    dispatch_desc_13,
    dispatch_desc_14,
    dispatch_desc_15,
    pool_desc,
    dispatch_epoch,
    num_tokens,
    rank: gl.constexpr,
    node: gl.constexpr,
    merge_net: gl.constexpr,
    dispatch_regs: gl.constexpr,
    dispatch_tail_quiet: gl.constexpr = False,
    dispatch_rendezvous: gl.constexpr = True,
):
    descs = (
        (
            dispatch_desc_0,
            dispatch_desc_1,
            dispatch_desc_2,
            dispatch_desc_3,
            dispatch_desc_4,
            dispatch_desc_5,
            dispatch_desc_6,
            dispatch_desc_7,
            dispatch_desc_8,
            dispatch_desc_9,
            dispatch_desc_10,
            dispatch_desc_11,
            dispatch_desc_12,
            dispatch_desc_13,
            dispatch_desc_14,
            dispatch_desc_15,
        ),
        pool_desc,
    )
    dispatch_buffer0 = gl.allocate_shared_memory(
        pool_desc.dtype, pool_desc.block_type.shape, pool_desc.layout
    )
    dispatch_buffer1 = gl.allocate_shared_memory(
        pool_desc.dtype, pool_desc.block_type.shape, pool_desc.layout
    )
    dispatch_barrier0 = gl.allocate_shared_memory(
        gl.int64, [1], mbarrier.MBarrierLayout()
    )
    dispatch_barrier1 = gl.allocate_shared_memory(
        gl.int64, [1], mbarrier.MBarrierLayout()
    )
    mbarrier.init(dispatch_barrier0, count=1)
    mbarrier.init(dispatch_barrier1, count=1)
    # Triton 3.7.1 hashes aggregate callable constants by repr in its disk
    # cache. Explicit recursive source keys invalidate that cache without
    # changing the function composition or adding a device execution switch.
    a_buffers = gl.allocate_shared_memory(
        l1_a_desc.dtype, [num_stages] + l1_a_desc.block_type.shape, l1_a_desc.layout
    )
    b_buffers = gl.allocate_shared_memory(
        l1_b_desc.dtype, [num_stages] + l1_b_desc.block_type.shape, l1_b_desc.layout
    )
    sf_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [block_m], gl.float32
    )
    sfa_lo_buffers = gl.allocate_shared_memory(
        gl.float32, [num_stages, max(32, block_m)], sf_layout
    )
    sfa_hi_buffers = gl.allocate_shared_memory(
        gl.float32, [num_stages, max(32, block_m)], sf_layout
    )
    l2_epilogue_buffer = gl.allocate_shared_memory(
        l2_store_desc.dtype, l2_store_desc.block_type.shape, l2_store_desc.layout
    )
    if mode != 0:
        l2_epilogue_buffer_1 = gl.allocate_shared_memory(
            l2_store_desc.dtype, l2_store_desc.block_type.shape, l2_store_desc.layout
        )
    barrier_layout: gl.constexpr = mbarrier.MBarrierLayout()
    stage_empty = gl.allocate_shared_memory(gl.int64, [num_stages, 1], barrier_layout)
    stage_ready = gl.allocate_shared_memory(gl.int64, [num_stages, 1], barrier_layout)
    math_done = gl.allocate_shared_memory(gl.int64, [1], barrier_layout)
    mbarrier.init(math_done, count=1 if mode == 0 else 2)
    for stage in gl.static_range(num_stages):
        mbarrier.init(stage_empty.index(stage), count=1 if mode == 0 else 2)
        mbarrier.init(stage_ready.index(stage), count=2)
    if mode == 1:
        amax_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
            [64], gl.float32
        )
        fc1_amax_scratch_0 = gl.allocate_shared_memory(gl.float32, [64], amax_layout)
        fc1_amax_scratch_1 = gl.allocate_shared_memory(gl.float32, [64], amax_layout)
        fc1_scale_ready = gl.allocate_shared_memory(gl.int64, [1], barrier_layout)
        fc1_scale_done = gl.allocate_shared_memory(gl.int64, [1], barrier_layout)
        mbarrier.init(fc1_scale_ready, count=2)
        mbarrier.init(fc1_scale_done, count=2)
    barriers = (stage_empty, stage_ready)
    buffers = (a_buffers, b_buffers, sfa_lo_buffers, sfa_hi_buffers)
    staged_dispatch_handoff: gl.constexpr = False
    if mode == 0:
        gl.warp_specialize(
            [
                (
                    partitions[0],
                    (
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
                        math_done,
                        publish_state,
                        local_participants,
                    ),
                ),
                (
                    bm16_a_partition if block_m == 16 else a_producer_partition,
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
                    bm16_b_partition if block_m == 16 else b_producer_partition,
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
                    partitions[3] if merge_net else pool_dispatch_partition,
                    (
                        (
                            dispatch,
                            peers,
                            descs,
                            dispatch_barrier0,
                            dispatch_buffer0,
                            dispatch_epoch,
                            num_tokens,
                            0,
                            num_sms,
                            rank,
                            node,
                            l2_n,
                            topk,
                            E * world_size,
                            max_tokens * topk,
                            num_padded_sf_pool_tokens,
                            network,
                            dispatch_tail_quiet,
                            dispatch_rendezvous,
                        )
                        if merge_net
                        else (
                            dispatch,
                            peers,
                            descs,
                            dispatch_barrier0,
                            dispatch_buffer0,
                            dispatch_epoch,
                            num_tokens,
                            0,
                            num_sms,
                            rank,
                            node,
                            l2_n,
                            topk,
                            E * world_size,
                            max_tokens * topk,
                            num_padded_sf_pool_tokens,
                            dispatch_rendezvous,
                        )
                    ),
                ),
                (
                    pool_dispatch_partition,
                    (
                        dispatch,
                        peers,
                        descs,
                        dispatch_barrier1,
                        dispatch_buffer1,
                        dispatch_epoch,
                        num_tokens,
                        1,
                        num_sms,
                        rank,
                        node,
                        l2_n,
                        topk,
                        E * world_size,
                        max_tokens * topk,
                        num_padded_sf_pool_tokens,
                        dispatch_rendezvous,
                    ),
                ),
                (
                    partitions[4],
                    (
                        network,
                        dispatch_epoch,
                        num_tokens,
                        max_tokens * topk,
                        l2_n,
                        topk,
                        (rank + 8) % 16,
                        dispatch_tail_quiet,
                    ),
                ),
            ][: 5 + (0 if merge_net else 1)],
            [1, 1, 1, 1, 1][: 4 + (0 if merge_net else 1)],
            [tma_regs, tma_regs, dispatch_regs, dispatch_regs, 40][
                : 4 + (0 if merge_net else 1)
            ],
        )
    elif mode == 1:
        gl.warp_specialize(
            [
                (
                    partitions[1],
                    (
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
                        0,
                        staged_dispatch_handoff,
                        math_done,
                        publish_state,
                        local_participants,
                    ),
                ),
                (
                    partitions[1],
                    (
                        barriers,
                        buffers,
                        l2_store_desc,
                        l2_epilogue_buffer_1,
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
                        1,
                        staged_dispatch_handoff,
                        math_done,
                        publish_state,
                        local_participants,
                    ),
                ),
                (
                    bm16_a_partition if block_m == 16 else a_producer_partition,
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
                    bm16_b_partition if block_m == 16 else b_producer_partition,
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
                    partitions[3] if merge_net else pool_dispatch_partition,
                    (
                        (
                            dispatch,
                            peers,
                            descs,
                            dispatch_barrier0,
                            dispatch_buffer0,
                            dispatch_epoch,
                            num_tokens,
                            0,
                            num_sms,
                            rank,
                            node,
                            l2_n,
                            topk,
                            E * world_size,
                            max_tokens * topk,
                            num_padded_sf_pool_tokens,
                            network,
                            dispatch_tail_quiet,
                            dispatch_rendezvous,
                        )
                        if merge_net
                        else (
                            dispatch,
                            peers,
                            descs,
                            dispatch_barrier0,
                            dispatch_buffer0,
                            dispatch_epoch,
                            num_tokens,
                            0,
                            num_sms,
                            rank,
                            node,
                            l2_n,
                            topk,
                            E * world_size,
                            max_tokens * topk,
                            num_padded_sf_pool_tokens,
                            dispatch_rendezvous,
                        )
                    ),
                ),
                (
                    pool_dispatch_partition,
                    (
                        dispatch,
                        peers,
                        descs,
                        dispatch_barrier1,
                        dispatch_buffer1,
                        dispatch_epoch,
                        num_tokens,
                        1,
                        num_sms,
                        rank,
                        node,
                        l2_n,
                        topk,
                        E * world_size,
                        max_tokens * topk,
                        num_padded_sf_pool_tokens,
                        dispatch_rendezvous,
                    ),
                ),
                (
                    partitions[4],
                    (
                        network,
                        dispatch_epoch,
                        num_tokens,
                        max_tokens * topk,
                        l2_n,
                        topk,
                        (rank + 8) % 16,
                        dispatch_tail_quiet,
                    ),
                ),
            ][: 6 + (0 if merge_net else 1)],
            [4, 1, 1, 1, 1, 1][: 5 + (0 if merge_net else 1)],
            [math_regs, tma_regs, tma_regs, dispatch_regs, dispatch_regs, 40][
                : 5 + (0 if merge_net else 1)
            ],
        )
    elif mode == 2:
        gl.warp_specialize(
            [
                (
                    partitions[2],
                    (
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
                        0,
                        staged_dispatch_handoff,
                        math_done,
                        publish_state,
                        local_participants,
                    ),
                ),
                (
                    partitions[2],
                    (
                        barriers,
                        buffers,
                        l2_store_desc,
                        l2_epilogue_buffer_1,
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
                        1,
                        staged_dispatch_handoff,
                        math_done,
                        publish_state,
                        local_participants,
                    ),
                ),
                (
                    bm16_a_partition if block_m == 16 else a_producer_partition,
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
                    bm16_b_partition if block_m == 16 else b_producer_partition,
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
                    partitions[3] if merge_net else pool_dispatch_partition,
                    (
                        (
                            dispatch,
                            peers,
                            descs,
                            dispatch_barrier0,
                            dispatch_buffer0,
                            dispatch_epoch,
                            num_tokens,
                            0,
                            num_sms,
                            rank,
                            node,
                            l2_n,
                            topk,
                            E * world_size,
                            max_tokens * topk,
                            num_padded_sf_pool_tokens,
                            network,
                            dispatch_tail_quiet,
                            dispatch_rendezvous,
                        )
                        if merge_net
                        else (
                            dispatch,
                            peers,
                            descs,
                            dispatch_barrier0,
                            dispatch_buffer0,
                            dispatch_epoch,
                            num_tokens,
                            0,
                            num_sms,
                            rank,
                            node,
                            l2_n,
                            topk,
                            E * world_size,
                            max_tokens * topk,
                            num_padded_sf_pool_tokens,
                            dispatch_rendezvous,
                        )
                    ),
                ),
                (
                    pool_dispatch_partition,
                    (
                        dispatch,
                        peers,
                        descs,
                        dispatch_barrier1,
                        dispatch_buffer1,
                        dispatch_epoch,
                        num_tokens,
                        1,
                        num_sms,
                        rank,
                        node,
                        l2_n,
                        topk,
                        E * world_size,
                        max_tokens * topk,
                        num_padded_sf_pool_tokens,
                        dispatch_rendezvous,
                    ),
                ),
                (
                    partitions[4],
                    (
                        network,
                        dispatch_epoch,
                        num_tokens,
                        max_tokens * topk,
                        l2_n,
                        topk,
                        (rank + 8) % 16,
                        dispatch_tail_quiet,
                    ),
                ),
            ][: 6 + (0 if merge_net else 1)],
            [4, 1, 1, 1, 1, 1][: 5 + (0 if merge_net else 1)],
            [math_regs, tma_regs, tma_regs, dispatch_regs, dispatch_regs, 40][
                : 5 + (0 if merge_net else 1)
            ],
        )


@gluon.jit
def internode_fused_kernel(
    l1_a_desc,
    l1_sfa_desc,
    l1_b_desc,
    l2_store_desc,
    l2_a_desc,
    l2_sfa_desc,
    l2_b_desc,
    l2_acts_sf,
    token_src_metadata,
    peer_combine_buffer_ptrs,
    route_weights,
    l2_arrival,
    l1_arrival,
    l1_weight_scales,
    l2_weight_scales,
    expert_state,
    dispatch_counter,
    publish_state,
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
    max_tokens: gl.constexpr,
    topk: gl.constexpr,
    world_size: gl.constexpr,
    local_participants: gl.constexpr,
    activation_clamp: gl.constexpr,
    has_activation_clamp: gl.constexpr,
    fast_math: gl.constexpr,
    fc2_arrival_counter: gl.constexpr,
    fc2_epilogue_requires_full_sync: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    num_math_warps: gl.constexpr,
    num_stages: gl.constexpr,
    math_regs: gl.constexpr,
    tma_regs: gl.constexpr,
    mode: gl.constexpr,
    partitions: gl.constexpr,
    partition_source_keys: gl.constexpr,
    dispatch,
    peers,
    network,
    dispatch_desc_0,
    dispatch_desc_1,
    dispatch_desc_2,
    dispatch_desc_3,
    dispatch_desc_4,
    dispatch_desc_5,
    dispatch_desc_6,
    dispatch_desc_7,
    dispatch_desc_8,
    dispatch_desc_9,
    dispatch_desc_10,
    dispatch_desc_11,
    dispatch_desc_12,
    dispatch_desc_13,
    dispatch_desc_14,
    dispatch_desc_15,
    pool_desc,
    dispatch_epoch,
    num_tokens,
    rank: gl.constexpr,
    node: gl.constexpr,
    merge_net: gl.constexpr,
    dispatch_regs: gl.constexpr,
    dispatch_tail_quiet: gl.constexpr = False,
    dispatch_rendezvous: gl.constexpr = True,
    api_prologue=None,
):
    if api_prologue is not None:
        gl.static_assert(num_math_warps == 4 and mode == 0)
        registration_reset_prologue(
            api_prologue, num_sms, num_math_warps, l2_n, topk, max_tokens, E
        )
    dispatch_math_kernel(
        l1_a_desc,
        l1_sfa_desc,
        l1_b_desc,
        l2_store_desc,
        l2_a_desc,
        l2_sfa_desc,
        l2_b_desc,
        l2_acts_sf,
        token_src_metadata,
        peer_combine_buffer_ptrs,
        route_weights,
        l2_arrival,
        l1_arrival,
        l1_weight_scales,
        l2_weight_scales,
        expert_state,
        dispatch_counter,
        publish_state,
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
        max_tokens,
        topk,
        world_size,
        local_participants,
        activation_clamp,
        has_activation_clamp,
        fast_math,
        fc2_arrival_counter,
        fc2_epilogue_requires_full_sync,
        block_m,
        block_n,
        num_math_warps,
        num_stages,
        math_regs,
        tma_regs,
        mode,
        partitions,
        partition_source_keys,
        dispatch,
        peers,
        network,
        dispatch_desc_0,
        dispatch_desc_1,
        dispatch_desc_2,
        dispatch_desc_3,
        dispatch_desc_4,
        dispatch_desc_5,
        dispatch_desc_6,
        dispatch_desc_7,
        dispatch_desc_8,
        dispatch_desc_9,
        dispatch_desc_10,
        dispatch_desc_11,
        dispatch_desc_12,
        dispatch_desc_13,
        dispatch_desc_14,
        dispatch_desc_15,
        pool_desc,
        dispatch_epoch,
        num_tokens,
        rank,
        node,
        merge_net,
        dispatch_regs,
        dispatch_tail_quiet=dispatch_tail_quiet,
        dispatch_rendezvous=dispatch_rendezvous,
    )


# internode/api.py


def _select_math_config(
    ctx,
    inputs,
    intermediate_hidden,
    num_sms=None,
    *,
    config=None,
    return_selection=False,
):
    """Select the common EP16 policy; num_sms remains an explicit grid override."""

    device_sms = ctx.physical_sms
    return _select_internode_config(
        shape=Shape(ctx.hidden, intermediate_hidden, ctx.num_experts, ctx.topk),
        tokens_bound=inputs.num_tokens
        if inputs.tokens_bound is None
        else inputs.tokens_bound,
        num_sms=device_sms,
        grid=num_sms,
        override=config,
        return_selection=return_selection,
    )


def _reset_grid(workspace, num_ctas):
    """Cover pool counters and the control prefix."""
    return (triton.cdiv(max(workspace.num_pool_rows // 64, 32), 256),)


def _resident_grid_proof(ctx, workspace, stage, compiled):
    device_sms = ctx.physical_sms
    if workspace.stage_occupancy is None:
        workspace.stage_occupancy = {}
    proof = workspace.stage_occupancy.get(stage)
    identity = dict(
        kernel_hash=compiled.hash,
        device=str(ctx.device),
        block_threads=compiled.metadata.num_warps * 32,
        shared_bytes=compiled.metadata.shared,
        physical_sms=device_sms,
    )
    if proof is None or any(proof.get(key) != value for key, value in identity.items()):
        compiled.run  # Initialize the CUDA handles without launching the kernel.
        cuda = ctypes.CDLL("libcuda.so.1")
        query = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor
        query.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        query.restype = ctypes.c_int
        blocks_per_sm = ctypes.c_int()
        block_threads = compiled.metadata.num_warps * 32
        rc = query(
            ctypes.byref(blocks_per_sm),
            ctypes.c_void_p(compiled.function),
            block_threads,
            compiled.metadata.shared,
        )
        if rc != 0:
            raise RuntimeError(f"CUDA occupancy query failed with code {rc}")
        proof = dict(
            identity, blocks_per_sm=blocks_per_sm.value, num_regs=compiled.n_regs
        )
    workspace.stage_occupancy[stage] = proof
    return proof


def _ensure_resident_grid(ctx, workspace, stage, compiled, num_ctas):
    proof = _resident_grid_proof(ctx, workspace, stage, compiled)
    proof = dict(proof, requested_ctas=num_ctas)
    workspace.stage_occupancy[stage] = proof
    if proof["blocks_per_sm"] * proof["physical_sms"] < num_ctas:
        raise ValueError(f"persistent {stage} grid is not fully resident: {proof}")


def _prepare_math(
    ctx,
    inputs,
    workspace,
    w1,
    s1,
    w2,
    s2,
    num_sms,
    *,
    config=None,
    return_selection=False,
    _initialize=True,
    _selection=None,
):
    if inputs.context is not ctx or not 0 <= inputs.num_tokens <= 128:
        raise ValueError("math requires this context's decode inputs")
    e, h = ctx.experts_per_rank, ctx.hidden
    if w1.ndim != 3 or w1.shape[0] != e or w1.shape[2] != h:
        raise ValueError("FC1 must have shape [E/16,2I,H]")
    n = w1.shape[1]
    i = n // 2
    if n % 256 or i % 128 or w2.shape != (e, h, i):
        raise ValueError("FC1/FC2 dimensions must match and I must be divisible by 128")
    if s1.shape != (e, n // 128, h // 128) or s2.shape != (e, h // 128, i // 128):
        raise ValueError("weights require per-128x128 scales")
    for tensor, dtype in (
        (w1, torch.float8_e4m3fn),
        (w2, torch.float8_e4m3fn),
        (s1, torch.float32),
        (s2, torch.float32),
    ):
        if (
            tensor.device != ctx.device
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError("weights and scales must be contiguous on the context GPU")
    # Reuse the policy resolved by the public API.
    selected = (
        _selection
        if _selection is not None
        else _select_math_config(
            ctx, inputs, i, num_sms, config=config, return_selection=True
        )
    )
    config, num_sms = selected.launch, selected.grid
    mode = 0 if config.use_swap_ab else (2 if config.block_n == 256 else 1)
    bn, bm = config.block_n, config.block_m
    identity = (w1.data_ptr(), w2.data_ptr(), i, bn, mode, bm)
    if workspace.weight_identity != identity:
        rows, sf_rows = workspace.num_pool_rows, workspace.num_padded_sf_pool_tokens
        if workspace.l2_acts is None or workspace.l2_acts.shape != (rows, i):
            workspace.l2_acts = torch.empty(
                (rows, i), dtype=torch.float8_e4m3fn, device=ctx.device
            )
            workspace.l2_acts_sf_mn_major = torch.empty(
                (i // 64, sf_rows), dtype=torch.float32, device=ctx.device
            )
            workspace.l2_arrival = torch.empty(
                rows // 64, dtype=torch.int32, device=ctx.device
            )

        def descriptor(tensor, shape, dtype):
            layout = gl.NVMMASharedLayout.get_default_for(shape, dtype)
            return TensorDescriptor.from_tensor(tensor, shape, layout)

        workspace.descriptor_set = (
            descriptor(workspace.pool.acts, [bm, 128], gl.float8e4nv),
            descriptor(workspace.pool.acts_sf_mn_major.view(-1), [bm], gl.float32),
            descriptor(w1.view(e * n, h), [bn, 128], gl.float8e4nv),
            descriptor(
                workspace.l2_acts,
                [bm, bn // 2 if mode == 0 else (32 if mode == 1 else 64)],
                gl.float8e4nv,
            ),
            descriptor(workspace.l2_acts, [bm, 128], gl.float8e4nv),
            descriptor(workspace.l2_acts_sf_mn_major.view(-1), [bm], gl.float32),
            descriptor(w2.view(e * h, i), [bn, 128], gl.float8e4nv),
        )
        workspace.weight_identity = identity
    # With very small token capacity and many empty experts, the compact
    # pool can have fewer blocks than experts. The ordinary reset covers its
    # pool-sized prefix; clear any remaining expert counters on the same stream.
    if (
        _initialize
        and workspace.fc2_expert_blocks.numel() > workspace.l2_arrival.numel()
    ):
        workspace.fc2_expert_blocks[workspace.l2_arrival.numel() :].zero_()
    result = (config, mode, num_sms)
    return (*result, selected) if return_selection else result


def _math_arguments(
    ctx, workspace, w1, s1, s2, config, publish, num_sms, activation_clamp, fast_math
):
    """Descriptor, tensor and scalar arguments for the selected math body."""
    e, n, h = w1.shape
    i, bn = n // 2, config.block_n
    capacity = triton.next_power_of_2(e)
    return (
        *workspace.descriptor_set,
        workspace.l2_acts_sf_mn_major,
        workspace.pool.token_src_metadata,
        ctx.peer_combine_target_ptrs,
        workspace.pool.topk_weights,
        workspace.l2_arrival,
        workspace.l1_arrival,
        s1,
        s2,
        workspace.pool.expert_state,
        workspace.control[8:],
        publish,
        n,
        h,
        h,
        i,
        e,
        *s1.stride(),
        *s2.stride(),
        triton.cdiv(n, bn),
        triton.cdiv(h, bn),
        config.num_experts_per_wave,
        num_sms,
        capacity,
        max(triton.cdiv(capacity, 32), 1),
        workspace.num_padded_sf_pool_tokens,
        ctx.max_tokens,
        ctx.topk,
        16,
        ctx.local_participants,
        activation_clamp,
        math.isfinite(activation_clamp),
        fast_math,
        config.fc2_arrival_counter,
        config.fc2_epilogue_requires_full_sync,
    )


def _prepare_dispatch(ctx, inputs, *, workspace=None, num_sms=None, rendezvous=True):
    if ctx.physical_size != 16 or inputs.context is not ctx:
        raise ValueError("network dispatch requires this context's 16-rank inputs")
    workspace = ctx.create_workspace(inputs.epoch) if workspace is None else workspace
    epoch = ctx.stage_epoch("dispatch")
    # Disabled rendezvous has no counter to consume. Keep its zero out of the
    # kernel ABI; the enabled path retains its independent runtime epoch.
    rendezvous_epoch = ctx.barrier_epoch("dispatch") if rendezvous else gl.constexpr(0)
    p = inputs.epoch % 2
    state = ctx.expert_state[epoch % 2]
    workspace.pool = replace(workspace.pool, expert_state=state)
    if not hasattr(workspace, "expert_send_state"):
        workspace.expert_send_state = torch.empty(
            (2, ctx.num_experts), dtype=torch.int64, device=ctx.device
        )
    device_sms = ctx.physical_sms
    num_sms = device_sms if num_sms is None else num_sms
    if type(num_sms) is not int or not 0 < num_sms <= 3 * device_sms:
        raise ValueError("persistent dispatch grid exceeds three CTAs per SM")
    dispatch = (
        ctx.input_topk_idx[p],
        ctx.landing_topk_idx,
        ctx.landing_num_tokens,
        ctx.signals,
        workspace.expert_send_state,
        state,
        ctx.expert_state[1 - epoch % 2],
        ctx.source_routes,
        ctx.recv_count,
        workspace.l1_arrival,
        workspace.actual_num_pool_rows,
        workspace.pool.acts_sf_mn_major,
        workspace.pool.topk_weights,
        workspace.pool.token_src_metadata,
        workspace.control,
    )
    peers = (
        ctx.peer_source_routes_ptrs,
        ctx.peer_recv_count_ptrs,
        ctx.peer_expert_state_ptrs[epoch % 2],
        ctx.peer_input_sf_ptrs[p],
        ctx.peer_input_topk_weights_ptrs[p],
        ctx.peer_data_signal_ptrs,
        ctx.node_dispatch_barrier,
        ctx.peer_node_dispatch_barrier_ptrs,
        ctx.node_dispatch_consumed,
        ctx.peer_node_dispatch_consumed_ptrs,
        rendezvous_epoch,
    )
    network = (
        ctx.input_metadata[p],
        ctx.input_acts[p],
        ctx.input_sf[p],
        ctx.input_topk_weights[p],
        ctx.landing_metadata,
        ctx.landing_acts,
        ctx.landing_sf,
        ctx.landing_topk_weights,
        ctx.signals,
        ctx.credit_payload[epoch % 2],
        workspace.control,
        ctx._stage_epochs["combine"],
    )
    dispatch_args = (
        dispatch,
        peers,
        network,
        *workspace.dispatch_descs[p],
        epoch,
        inputs.num_tokens,
        num_sms,
        ctx.rank,
        ctx.node_id,
        ctx.hidden,
        ctx.topk,
        ctx.num_experts,
        ctx.max_routes,
        workspace.num_padded_sf_pool_tokens,
    )
    return workspace, epoch, dispatch_args


def _combine_state(ctx, inputs, workspace, epoch, control):
    if not hasattr(workspace, "output"):
        workspace.output = torch.empty(
            (ctx.max_tokens, ctx.hidden), dtype=torch.bfloat16, device=ctx.device
        )
    return (
        ctx.combine_stage,
        ctx.combine_reduced[epoch % 2],
        ctx.landing_topk_idx,
        ctx.landing_num_tokens,
        ctx.combine_buffer,
        ctx.combine_recv,
        workspace.output,
        ctx.input_topk_idx[inputs.epoch % 2],
        ctx.signals,
        control,
        ctx.node_combine_consumed,
        ctx.peer_node_combine_consumed_ptrs,
    )


def _launch_fused(
    ctx,
    inputs,
    workspace,
    w1,
    s1,
    w2,
    s2,
    *,
    num_sms=None,
    activation_clamp=None,
    fast_math=None,
    config=None,
    _compile_only=False,
    _selection=None,
):
    """Compile or launch the resident EP16 pipeline."""
    deferred = inputs.deferred is not None
    if ctx.physical_size != 16 or inputs.context is not ctx:
        raise ValueError("dispatch+math requires this context's 16-rank inputs")
    partitions = (
        swap_math_partition,
        bm64_bn128_split_math_partition,
        bm64_bn256_split_math_partition,
    )
    partitions = (*partitions, fused_dispatch0_partition, fused_net_partition)
    (config, mode, num_sms, selected) = _prepare_math(
        ctx,
        inputs,
        workspace,
        w1,
        s1,
        w2,
        s2,
        num_sms,
        config=config,
        return_selection=True,
        _initialize=not _compile_only,
        _selection=_selection,
    )
    chunked = selected.config.combine is CombineMode.CHUNKED
    if chunked:
        partitions = (*partitions[:3], chunked_dispatch_partition, partitions[4])
    if selected.config.fuse_reset != deferred:
        raise ValueError("registration state must match the selected reset policy")
    activation_clamp = (
        selected.config.activation_clamp
        if activation_clamp is None
        else activation_clamp
    )
    fast_math = selected.config.fast_math if fast_math is None else fast_math
    (workspace, dispatch_epoch, dispatch_args) = _prepare_dispatch(
        ctx,
        inputs,
        workspace=workspace,
        num_sms=num_sms,
        rendezvous=selected.config.rendezvous is Rendezvous.D4,
    )
    bn = config.block_n
    blocks = workspace.num_pool_rows // 64
    arrivals = (
        workspace.l1_arrival,
        workspace.l2_arrival,
        workspace.fc2_arrival,
        workspace.fc2_expert_blocks,
    )
    if chunked:
        if blocks < ctx.max_tokens or num_sms < 2:
            raise ValueError(
                "chunked return requires complete row reset coverage and two resident CTAs"
            )
        for name in ("combine_row_claims", "combine_slot_ready"):
            value = getattr(workspace, name)
            if value is None:
                setattr(
                    workspace,
                    name,
                    torch.empty(blocks, dtype=torch.int32, device=ctx.device),
                )
            elif value.numel() != blocks:
                raise ValueError("chunked return workspace differs from reset coverage")
        arrivals = (
            *arrivals,
            workspace.combine_row_claims,
            workspace.combine_slot_ready,
        )
    if not deferred and (not _compile_only):
        workspace.expert_send_state.zero_()
        reset_composed[_reset_grid(workspace, num_sms)](
            workspace.control, arrivals, blocks, 256, num_warps=4
        )
    phase_epoch = ctx.stage_epoch("math")
    publish = MathPublication(
        ctx.node_fused_barrier,
        ctx.peer_node_fused_barrier_ptrs,
        workspace.control[16:24],
        ctx.barrier_epoch("fused"),
        (
            workspace.fc2_arrival,
            workspace.fc2_expert_blocks,
            ctx.fc2_expert_done,
            phase_epoch,
        ),
    )
    combine_epoch = ctx.stage_epoch("combine")
    combine = _combine_state(
        ctx, inputs, workspace, combine_epoch, workspace.control[24:32]
    )
    tail = combine_ready_tail if chunked else combine_tail
    if chunked:
        combine = (
            *combine,
            ctx.peer_fc2_expert_done_ptrs,
            phase_epoch,
            workspace.combine_row_claims,
            (
                ctx.combine_send_records[combine_epoch % 2].view(-1),
                ctx.combine_recv_records.view(-1),
                ctx.combine_chunk_signals,
                workspace.combine_slot_ready,
            ),
            inputs.num_tokens,
        )
    publish = publish._replace(
        tail=CombineTail(
            combine,
            combine_epoch,
            inputs.num_tokens,
            ctx.node_id,
            tail,
        )
    )
    network = (
        *dispatch_args[2],
        (
            combine,
            ctx.credit_payload[combine_epoch % 2, 1:],
            combine_epoch,
            1 if mode == 0 else 2,
            num_sms,
            ctx.combine_reduced.element_size(),
        ),
    )
    dispatch_args = (*dispatch_args[:2], network, *dispatch_args[3:])
    merge_net = chunked or num_sms > ctx.physical_sms or config.num_math_warps == 16
    dispatch_regs = selected.config.dispatch_registers
    tma_regs = 24 if mode == 2 else config.non_epilogue_register_budget
    maxnreg = derive_launch_maxnreg(
        num_math_warps=config.num_math_warps,
        math_register_budget=config.math_register_budget,
        specialized_register_budgets=(tma_regs, tma_regs, dispatch_regs, dispatch_regs)
        + (() if merge_net else (40,)),
    )
    math_args = (
        *_math_arguments(
            ctx,
            workspace,
            w1,
            s1,
            s2,
            config,
            publish,
            num_sms,
            activation_clamp,
            fast_math,
        ),
        config.block_m,
        bn,
        config.num_math_warps,
        config.num_stages,
        config.math_register_budget,
        tma_regs,
        mode,
        partitions,
        tuple((fn.cache_key for fn in partitions)) + (tail.cache_key,),
        *dispatch_args[:20],
        dispatch_epoch,
        inputs.num_tokens,
        ctx.rank,
        ctx.node_id,
        merge_net,
        dispatch_regs,
    )
    launch_options = dict(
        num_warps=config.num_math_warps if mode == 0 else 4, maxnreg=maxnreg
    )
    kernel = internode_fused_kernel
    launch_options.update(
        dispatch_tail_quiet=selected.config.dispatch_handoff is DispatchHandoff.D1,
        dispatch_rendezvous=selected.config.rendezvous is Rendezvous.D4,
    )
    if deferred:
        if mode != 0 or config.num_math_warps != 4:
            raise ValueError("fused registration requires four-warp swap")
        if not hasattr(ctx, "registration_counter"):
            ctx.registration_counter = torch.zeros(
                (), dtype=torch.int64, device=ctx.device
            )
            ctx.registration_target = 0
        registration_target = ctx.registration_target + num_sms
        if registration_target >= 1 << 62:
            raise RuntimeError("recreate context before the registration counter wraps")
        (x, x_sf, idx, weights) = inputs.deferred
        r = inputs.registered
        registration = (
            x,
            r.input_acts_sf if x_sf is None else x_sf,
            idx,
            weights,
            r.input_acts_fp8,
            r.input_acts_sf,
            r.input_topk_idx,
            r.input_topk_weights,
            inputs.num_tokens,
        )
        reset = (workspace.control, arrivals)
        launch_options["api_prologue"] = (
            registration,
            reset,
            ctx.registration_counter,
            registration_target,
            ctx.input_metadata[inputs.epoch % 2, -1:],
            workspace.expert_send_state,
            workspace.expert_send_state.numel(),
        )
    prepared_launch = prepare_launch(
        kernel,
        (num_sms,),
        *math_args,
        extern_libs={"libnvshmem_device": _nvshmem_find_nvshmem_device_bitcode()},
        **launch_options,
    )
    candidate = _nvshmem_initialize_module(prepared_launch.compiled)
    if _compile_only:
        return (candidate, selected)
    _ensure_resident_grid(ctx, workspace, "fused", candidate, num_sms)
    if deferred:
        ctx.registration_target = registration_target
    compiled = prepared_launch()
    return InternodeMoEResult(
        ctx,
        workspace,
        inputs,
        workspace.output[: inputs.num_tokens],
        config,
        compiled,
        num_sms,
    )


def fused_moe(
    ctx,
    x,
    topk_idx,
    topk_weights,
    w1,
    s1,
    w2,
    s2,
    *,
    x_sf=None,
    tokens_bound=None,
    workspace=None,
    num_sms=None,
    activation_clamp=None,
    fast_math=None,
    config: MegaMoEConfig | None = None,
):
    """Register input, reset local workspaces, and launch one persistent EP16 kernel."""

    bound = resolve_tokens_bound(ctx, x.shape[0], tokens_bound)
    selected = _select_internode_config(
        shape=Shape(ctx.hidden, w2.shape[2], ctx.num_experts, ctx.topk),
        tokens_bound=bound,
        num_sms=ctx.physical_sms,
        grid=num_sms,
        override=config,
        return_selection=True,
    )
    config = selected.config
    check_policy_agreement(
        ctx,
        backend="ep16_fp8",
        tokens_bound=bound,
        shape=(ctx.hidden, w2.shape[2], ctx.num_experts, ctx.topk),
        capacity=ctx.max_tokens,
        config=config,
        launch=selected.launch,
        grid=selected.grid,
        activation_clamp=config.activation_clamp
        if activation_clamp is None
        else activation_clamp,
        fast_math=config.fast_math if fast_math is None else fast_math,
    )
    inputs = ctx.register_inputs(
        x,
        topk_idx,
        topk_weights,
        x_sf=x_sf,
        tokens_bound=bound,
        defer=config.fuse_reset,
    )
    workspace = ctx.create_workspace(inputs.epoch) if workspace is None else workspace
    combined = _launch_fused(
        ctx,
        inputs,
        workspace,
        w1,
        s1,
        w2,
        s2,
        num_sms=num_sms,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
        config=config,
        _selection=selected,
    )
    return combined


# internode/prepare.py


@dataclass(frozen=True)
class PreparedMoE:
    """A common configuration and local workspace for the prepared input shape.

    Pass ``**prepared.kwargs`` to ``internode.fused_moe``. Reprepare collectively
    when the token bound, tensor shapes/dtypes, weights, device or configuration
    changes. Preparation is outside CUDA graph capture and the timed operation.
    """

    context: object
    tokens_bound: int
    selection: object
    workspace: object
    attempts: tuple

    @property
    def kwargs(self):
        return dict(
            tokens_bound=self.tokens_bound,
            config=self.selection.config,
            num_sms=self.selection.grid,
            workspace=self.workspace,
        )


def prepare(
    ctx,
    x,
    topk_idx,
    topk_weights,
    w1,
    s1,
    w2,
    s2,
    *,
    x_sf=None,
    tokens_bound=None,
    workspace=None,
    num_sms=None,
    config=None,
    fallback_configs=(),
    activation_clamp=None,
    fast_math=None,
):
    """Agree on a resident configuration before input registration or dispatch.

    Every rank must call this outside graph capture, on an idle context. It
    compiles with a shallow context whose host epochs are independent; no input
    registration, workspace reset, or communication kernel is launched. Local
    allocations and their initialization are allowed. Each candidate first tries
    the requested grid, then the common driver capacity. Explicit stage counts
    are tried down to two before the next supplied fallback configuration.

    Fallback configurations must obey the same activation contract. They should
    come from correctness-validated policies for this shape.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("collective preparation must run outside CUDA graph capture")
    if ctx.physical_size != 16:
        raise ValueError("collective fused preparation requires sixteen ranks")

    def gather(value):
        values = [None] * ctx.physical_size
        dist.all_gather_object(values, value, group=ctx.group)
        return values

    error = None
    try:
        bound = validate_tokens_bound(x.shape[0], tokens_bound, ctx.max_tokens)
        shape = Shape(ctx.hidden, w2.shape[2], ctx.num_experts, ctx.topk)
        device_sms = ctx.physical_sms
        initial = (
            config
            if config is not None
            else _select_internode_config(
                shape=shape,
                tokens_bound=bound,
                num_sms=device_sms,
                grid=num_sms,
                return_selection=True,
            ).config
        )
        configs = (initial, *tuple(fallback_configs))
        if any(not isinstance(value, MegaMoEConfig) for value in configs):
            raise ValueError("preparation candidates must be MegaMoEConfig objects")
        clamp = (
            initial.activation_clamp if activation_clamp is None else activation_clamp
        )
        fast = initial.fast_math if fast_math is None else fast_math
        configs = tuple(
            replace(value, activation_clamp=clamp, fast_math=fast) for value in configs
        )
        contract = dict(
            shape=asdict(shape),
            tokens_bound=bound,
            capacity=ctx.max_tokens,
            physical_sms=device_sms,
            partial_dtype=str(ctx.partial_dtype),
            dtypes=tuple(
                str(t.dtype) for t in (x, topk_idx, topk_weights, w1, s1, w2, s2)
            ),
            x_sf_dtype=None if x_sf is None else str(x_sf.dtype),
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = gather(error)
    if any(errors):
        raise ValueError(f"collective preparation input errors: {errors}")

    # Each compile attempt gets independent host counters, but uses the actual
    # tensor allocations and layouts. Failed attempts consume no peer epochs.
    scratch = workspace

    def resolve(candidate, grid):
        return _select_internode_config(
            shape=shape,
            tokens_bound=bound,
            num_sms=device_sms,
            grid=grid,
            override=candidate,
            return_selection=True,
        )

    def probe(selected):
        nonlocal scratch
        shadow = copy(ctx)
        shadow._stage_epochs = dict(ctx._stage_epochs)
        shadow._barrier_epochs = dict(ctx._barrier_epochs)
        epoch = ctx.epoch + 1
        if epoch >= (1 << 27):
            raise RuntimeError("recreate context before int32 node counters wrap")
        registered = register_inputs(
            shadow.registration_view(epoch),
            x,
            topk_idx,
            topk_weights,
            x_sf=x_sf,
            launch=False,
        )
        deferred = (
            (x, x_sf, topk_idx, topk_weights) if selected.config.fuse_reset else None
        )
        inputs = InternodeRegisteredInputs(
            shadow, registered, epoch, x.shape[0], bound, deferred
        )
        if scratch is None:
            scratch = shadow.create_workspace(epoch)
        compiled, actual = _launch_fused(
            shadow,
            inputs,
            scratch,
            w1,
            s1,
            w2,
            s2,
            num_sms=selected.grid,
            config=selected.config,
            _compile_only=True,
            _selection=selected,
        )
        if actual.config != selected.config or actual.grid != selected.grid:
            raise RuntimeError("compiled policy differs from prepared selection")
        return _resident_grid_proof(shadow, scratch, "fused", compiled)

    candidates = []
    for index, value in enumerate(configs):
        stages = [value.stages]
        if type(value.stages) is int and 2 < value.stages <= 8:
            stages.extend(range(value.stages - 1, 1, -1))
        for count in stages:
            candidate = replace(value, stages=count)
            entry = (candidate, num_sms if index == 0 else None)
            if entry not in candidates:
                candidates.append(entry)
    result = prepare_collectively(
        candidates, resolve=resolve, probe=probe, gather=gather, contract=contract
    )
    return PreparedMoE(ctx, bound, result.selection, scratch, result.attempts)


__all__ = [
    "create_context",
    "prepare_weights",
    "fused_moe",
    "MegaMoEConfig",
    "MathBody",
    "Shape",
    "select",
    "ResourcePreparationError",
    "prepare",
    "PreparedMoE",
    "InternodeContext",
    "InternodeMoEResult",
    "InternodeWorkspace",
]
