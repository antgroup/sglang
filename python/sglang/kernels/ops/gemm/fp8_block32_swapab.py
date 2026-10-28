"""SM90 block-[32, 32] FP8 GEMM for decode-sized M: swap-AB, warp-specialized (Gluon).

    out[m, n] = bf16(sum_g sa[m, g] * sw[n // 32, g] * dot32(x[m, 32g:], w[n, 32g:]))

Per output element the arithmetic is DeepGEMM sm90_fp8_gemm_1d2d's block32
recipe: a fresh fp32 wgmma partial per 32-wide K group, promoted in K order as
acc = fma.rn(mul.rn(sa, sw), partial, acc), then one bf16 rounding. With
split == 1 the output is bitwise DeepGEMM's.

Layout differences from DeepGEMM, all decode-oriented:

* Swap-AB: weight rows are the wgmma M side (64 per warpgroup), token rows the
  N side. Gluon tensors are power-of-two shaped, so a token tile is one or two
  pieces (48 = 32 + 16, 96 = 64 + 32), each with its own accumulator, instead
  of padding 48 / 96 rows to a 64 / 128-row wgmma.
* One producer warp streams weight / token / activation-scale stages with TMA;
  under PDL it issues the weight loads before waiting on the previous kernel.
  The consumer warpgroup alternates two partial banks so the promotion of
  group g overlaps the wgmmas of group g + 1; several small CTAs share an SM.
* split > 1 splits K across CTAs: each writes its fp32 tile partial in
  accumulator-fragment order to scratch, and the last CTA to arrive on the
  tile's counter sums the partials in split order (deterministic), stores
  bf16 and re-zeroes the counter for the next call.
* group_n > 0 runs a block-diagonal (grouped) GEMM in one launch: weight rows
  [g * group_n, (g + 1) * group_n) multiply activation columns [g * K, (g + 1) * K),
  so a CTA only offsets its token loads; the per-element arithmetic is unchanged.

Launch configs come from measured tables, ``configs/block32_swapab,device_name=<GPU>
[,<role>].json``: per weight shape, row counts with the config that beat DeepGEMM
there, or null where DeepGEMM stayed faster; M rows use the entry at the next
listed row count. A role's table replaces whole shapes of the base table. A key
with a group count ("N=1024,K=4096,G=8": per-group N and K) lists the grouped GEMM
over G groups; its entries must be split 1.
"""

import functools
import json
from typing import Dict, Optional, Tuple

import msgspec
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

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.gemm.fp8_kernel import find_tuned_config

# K per TMA stage: four 32-wide scale groups.
_BK = gl.constexpr(128)
# Weight rows per CTA: one consumer warpgroup.
BLOCK_W = 64
_BW = gl.constexpr(BLOCK_W)


@gluon.jit
def _gdc_wait():
    # griddepcontrol.wait; a no-op when the launch is not programmatic-dependent.
    gl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _load_tokens(
    x1_desc,
    x2_desc,
    s1_desc,
    s2_desc,
    x1_smem,
    x2_smem,
    s1_smem,
    s2_smem,
    bar,
    stage,
    kb,
    t0,
    BT1: gl.constexpr,
    BT2: gl.constexpr,
):
    tma.async_copy_global_to_shared(x1_desc, [t0, kb * _BK], bar, x1_smem.index(stage))
    tma.async_copy_global_to_shared(s1_desc, [kb * 4, t0], bar, s1_smem.index(stage))
    if BT2 > 0:
        tma.async_copy_global_to_shared(
            x2_desc, [t0 + BT1, kb * _BK], bar, x2_smem.index(stage)
        )
        tma.async_copy_global_to_shared(
            s2_desc, [kb * 4, t0 + BT1], bar, s2_smem.index(stage)
        )


@gluon.jit
def _producer(
    w_desc,
    x1_desc,
    x2_desc,
    s1_desc,
    s2_desc,
    w_smem,
    x1_smem,
    x2_smem,
    s1_smem,
    s2_smem,
    ready,
    empty,
    w0,
    t0,
    kb_lo,
    n_kb,
    kg0,
    BT1: gl.constexpr,
    BT2: gl.constexpr,
    STAGES: gl.constexpr,
    PDL: gl.constexpr,
):
    NBYTES: gl.constexpr = (_BW + BT1 + BT2) * _BK + 16 * (BT1 + BT2)
    j0 = 0
    if PDL:
        # Weights do not depend on the previous kernel: start streaming them,
        # then wait for it before touching the activation and its scales.
        for s in gl.static_range(STAGES):
            if s < n_kb:
                mbarrier.expect(ready.index(s), NBYTES)
                tma.async_copy_global_to_shared(
                    w_desc, [w0, (kb_lo + s) * _BK], ready.index(s), w_smem.index(s)
                )
        _gdc_wait()
        for s in gl.static_range(STAGES):
            if s < n_kb:
                _load_tokens(
                    x1_desc,
                    x2_desc,
                    s1_desc,
                    s2_desc,
                    x1_smem,
                    x2_smem,
                    s1_smem,
                    s2_smem,
                    ready.index(s),
                    s,
                    kg0 + kb_lo + s,
                    t0,
                    BT1,
                    BT2,
                )
        j0 = STAGES
    for j in range(j0, n_kb):
        stage = j % STAGES
        if j >= STAGES:
            mbarrier.wait(empty.index(stage), ((j // STAGES) - 1) & 1)
            # Generic -> async proxy fence: a consumer warp's ld.shared of this
            # stage's scales may still be pending; the TMA refill must not overtake it.
            fence_async_shared()
        bar = ready.index(stage)
        mbarrier.expect(bar, NBYTES)
        tma.async_copy_global_to_shared(
            w_desc, [w0, (kb_lo + j) * _BK], bar, w_smem.index(stage)
        )
        _load_tokens(
            x1_desc,
            x2_desc,
            s1_desc,
            s2_desc,
            x1_smem,
            x2_smem,
            s1_smem,
            s2_smem,
            bar,
            stage,
            kg0 + kb_lo + j,
            t0,
            BT1,
            BT2,
        )


@gluon.jit
def _mma(w_t, x_t, sub: gl.constexpr, buf):
    # use_acc=False on a recycled bank: a fresh partial per 32-wide K group.
    return warpgroup_mma(
        w_t.slice(sub * 32, 32, dim=1),
        x_t.slice(sub * 32, 32, dim=1).permute((1, 0)),
        buf,
        use_acc=False,
        is_async=True,
    )


@gluon.jit
def _load_sa(s_t, sub: gl.constexpr, col_l: gl.constexpr):
    # [1, BT] row of the staged scales in the accumulator layout; the size-1
    # reduction lands it in the column slice layout without a shuffle.
    return gl.max(s_t.slice(sub, 1, dim=0).load(col_l.parent), axis=0)


@gluon.jit
def _promote(acc, p, sa, sw):
    # Packed per accumulator quad (r, c), (r, c+1), (r+8, c), (r+8, c+1) of the
    # wgmma fragment. Rows r and r+8 always fall in one 32-row weight-scale
    # block (a warp owns 16 rows of a 64-row tile), so one sa * sw product per
    # column serves both rows.
    return gl.inline_asm_elementwise(
        """
        {
        .reg .f32 s0, s1;
        mul.rn.f32 s0, $8, $12;
        mul.rn.f32 s1, $9, $12;
        fma.rn.f32 $0, s0, $4, $16;
        fma.rn.f32 $1, s1, $5, $17;
        fma.rn.f32 $2, s0, $6, $18;
        fma.rn.f32 $3, s1, $7, $19;
        }
        """,
        "=f,=f,=f,=f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f",
        [p, sa[None, :], sw[:, None], acc],
        dtype=gl.float32,
        is_pure=True,
        pack=4,
    )


@gluon.jit
def _issue_group(w_t, x1_t, x2_t, sub: gl.constexpr, q1, q2, BT2: gl.constexpr):
    q1 = _mma(w_t, x1_t, sub, q1)
    if BT2 > 0:
        q2 = _mma(w_t, x2_t, sub, q2)
    return q1, q2


@gluon.jit
def _wait_group(q1, q2, n: gl.constexpr, BT2: gl.constexpr):
    if BT2 > 0:
        q1, q2 = warpgroup_mma_wait(n * 2, deps=[q1, q2])
    else:
        q1 = warpgroup_mma_wait(n, deps=[q1])
    return q1, q2


@gluon.jit
def _promote_group(
    acc1,
    acc2,
    q1,
    q2,
    s1_t,
    s2_t,
    sub: gl.constexpr,
    sw,
    BT2: gl.constexpr,
    col1_l: gl.constexpr,
    col2_l: gl.constexpr,
):
    acc1 = _promote(acc1, q1, _load_sa(s1_t, sub, col1_l), sw)
    if BT2 > 0:
        sw2 = gl.convert_layout(sw, gl.SliceLayout(1, col2_l.parent))
        acc2 = _promote(acc2, q2, _load_sa(s2_t, sub, col2_l), sw2)
    return acc1, acc2


@gluon.jit
def _block(
    acc1,
    acc2,
    a1,
    a2,
    b1,
    b2,
    w_t,
    x1_t,
    x2_t,
    s1_t,
    s2_t,
    sw0,
    sw1,
    sw2,
    sw3,
    NSUB: gl.constexpr,
    BT2: gl.constexpr,
    col1_l: gl.constexpr,
    col2_l: gl.constexpr,
):
    """The first NSUB (1..4) groups of one staged 128-wide K block; banks a / b
    alternate so the promotion of group g overlaps the wgmmas of group g + 1."""
    a1, a2 = _issue_group(w_t, x1_t, x2_t, 0, a1, a2, BT2)
    if NSUB > 1:
        b1, b2 = _issue_group(w_t, x1_t, x2_t, 1, b1, b2, BT2)
        a1, a2 = _wait_group(a1, a2, 1, BT2)
    else:
        a1, a2 = _wait_group(a1, a2, 0, BT2)
    acc1, acc2 = _promote_group(
        acc1, acc2, a1, a2, s1_t, s2_t, 0, sw0, BT2, col1_l, col2_l
    )
    if NSUB > 2:
        a1, a2 = _issue_group(w_t, x1_t, x2_t, 2, a1, a2, BT2)
        b1, b2 = _wait_group(b1, b2, 1, BT2)
        acc1, acc2 = _promote_group(
            acc1, acc2, b1, b2, s1_t, s2_t, 1, sw1, BT2, col1_l, col2_l
        )
        if NSUB > 3:
            b1, b2 = _issue_group(w_t, x1_t, x2_t, 3, b1, b2, BT2)
            a1, a2 = _wait_group(a1, a2, 1, BT2)
            acc1, acc2 = _promote_group(
                acc1, acc2, a1, a2, s1_t, s2_t, 2, sw2, BT2, col1_l, col2_l
            )
            b1, b2 = _wait_group(b1, b2, 0, BT2)
            acc1, acc2 = _promote_group(
                acc1, acc2, b1, b2, s1_t, s2_t, 3, sw3, BT2, col1_l, col2_l
            )
        else:
            a1, a2 = _wait_group(a1, a2, 0, BT2)
            acc1, acc2 = _promote_group(
                acc1, acc2, a1, a2, s1_t, s2_t, 2, sw2, BT2, col1_l, col2_l
            )
    elif NSUB > 1:
        b1, b2 = _wait_group(b1, b2, 0, BT2)
        acc1, acc2 = _promote_group(
            acc1, acc2, b1, b2, s1_t, s2_t, 1, sw1, BT2, col1_l, col2_l
        )
    return acc1, acc2, a1, a2, b1, b2


@gluon.jit
def _ld_sw(sw_ptr, sw_offs, g, KG: gl.constexpr):
    # Clamped: the last block's groups past K are never promoted, but the load
    # must stay inside the scale tensor.
    return gl.load(sw_ptr + sw_offs + gl.minimum(g, KG - 1))


@gluon.jit
def _consumer(
    w_smem,
    x1_smem,
    x2_smem,
    s1_smem,
    s2_smem,
    ready,
    empty,
    sw_ptr,
    sw_offs,
    kb_lo,
    n_kb,
    has_tail,
    KG: gl.constexpr,
    BT1: gl.constexpr,
    BT2: gl.constexpr,
    STAGES: gl.constexpr,
    TAIL: gl.constexpr,
    mma1_l: gl.constexpr,
    mma2_l: gl.constexpr,
):
    col1_l: gl.constexpr = gl.SliceLayout(0, mma1_l)
    col2_l: gl.constexpr = gl.SliceLayout(0, mma2_l)
    BT2S: gl.constexpr = BT2 if BT2 > 0 else 8
    acc1 = gl.zeros([_BW, BT1], gl.float32, layout=mma1_l)
    a1 = acc1
    b1 = acc1
    acc2 = gl.zeros([_BW, BT2S], gl.float32, layout=mma2_l)
    a2 = acc2
    b2 = acc2
    n_full = n_kb - has_tail.to(gl.int32)
    g = kb_lo * 4
    g_last = (kb_lo + n_kb) * 4 - 4
    sw0 = _ld_sw(sw_ptr, sw_offs, g, KG)
    sw1 = _ld_sw(sw_ptr, sw_offs, g + 1, KG)
    sw2 = _ld_sw(sw_ptr, sw_offs, g + 2, KG)
    sw3 = _ld_sw(sw_ptr, sw_offs, g + 3, KG)
    for j in range(n_full):
        stage = j % STAGES
        # Weight scales of the next block, one block ahead.
        gn = gl.minimum(g + 4, g_last)
        nw0 = _ld_sw(sw_ptr, sw_offs, gn, KG)
        nw1 = _ld_sw(sw_ptr, sw_offs, gn + 1, KG)
        nw2 = _ld_sw(sw_ptr, sw_offs, gn + 2, KG)
        nw3 = _ld_sw(sw_ptr, sw_offs, gn + 3, KG)
        mbarrier.wait(ready.index(stage), (j // STAGES) & 1)
        acc1, acc2, a1, a2, b1, b2 = _block(
            acc1,
            acc2,
            a1,
            a2,
            b1,
            b2,
            w_smem.index(stage),
            x1_smem.index(stage),
            x2_smem.index(stage),
            s1_smem.index(stage),
            s2_smem.index(stage),
            sw0,
            sw1,
            sw2,
            sw3,
            4,
            BT2,
            col1_l,
            col2_l,
        )
        # Every warp has issued this stage's scale reads and its last wgmma retired:
        # hand the stage back to the producer, which proxy-fences before the refill.
        gl.barrier()
        mbarrier.arrive(empty.index(stage), count=1)
        sw0 = nw0
        sw1 = nw1
        sw2 = nw2
        sw3 = nw3
        g += 4
    if TAIL > 0:
        if has_tail:
            j = n_kb - 1
            stage = j % STAGES
            mbarrier.wait(ready.index(stage), (j // STAGES) & 1)
            acc1, acc2, a1, a2, b1, b2 = _block(
                acc1,
                acc2,
                a1,
                a2,
                b1,
                b2,
                w_smem.index(stage),
                x1_smem.index(stage),
                x2_smem.index(stage),
                s1_smem.index(stage),
                s2_smem.index(stage),
                sw0,
                sw1,
                sw2,
                sw3,
                TAIL,
                BT2,
                col1_l,
                col2_l,
            )
    return acc1, acc2


@gluon.jit
def _frag_offs(BT: gl.constexpr, mma_l: gl.constexpr):
    """Bijective [64, BT] -> [0, 64 * BT) map in wgmma accumulator fragment order:
    per (warp, register pair) the 32 lanes cover 256 contiguous bytes."""
    r = gl.arange(0, _BW, layout=gl.SliceLayout(1, mma_l))[:, None]
    c = gl.arange(0, BT, layout=gl.SliceLayout(0, mma_l))[None, :]
    lane = (r % 8) * 4 + (c % 8) // 2
    pair = (r // 16) * (BT // 4) + (c // 8) * 2 + (r % 16) // 8
    return (pair * 32 + lane) * 2 + c % 2


@gluon.jit
def _tma_store_tile(
    acc1,
    acc2,
    o1_smem,
    o2_smem,
    o1_desc,
    o2_desc,
    t0,
    w0,
    BT1: gl.constexpr,
    BT2: gl.constexpr,
):
    # Transposed through shared memory into [tokens, rows] tiles; the
    # descriptor clips rows past N and tokens past M.
    o1_smem.store(gl.permute(acc1.to(gl.bfloat16), (1, 0)))
    if BT2 > 0:
        o2_smem.store(gl.permute(acc2.to(gl.bfloat16), (1, 0)))
    fence_async_shared()
    gl.barrier()
    tma.async_copy_shared_to_global(o1_desc, [t0, w0], o1_smem)
    if BT2 > 0:
        tma.async_copy_shared_to_global(o2_desc, [t0 + BT1, w0], o2_smem)
    tma.store_wait(0)


@gluon.jit
def _block32_swapab_kernel(
    w_desc,
    x1_desc,
    x2_desc,
    s1_desc,
    s2_desc,
    o1_desc,
    o2_desc,
    sw_ptr,
    part_ptr,
    cnt_ptr,
    N,
    KG: gl.constexpr,
    BT1: gl.constexpr,
    BT2: gl.constexpr,
    STAGES: gl.constexpr,
    SPLIT: gl.constexpr,
    PRODUCER_REGS: gl.constexpr,
    PDL: gl.constexpr,
    GROUP_N: gl.constexpr,
    num_warps: gl.constexpr,
):
    NKB: gl.constexpr = (KG + 3) // 4
    TAIL: gl.constexpr = KG % 4
    mma1_l: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, BT1, 32]
    )
    if BT2 > 0:
        mma2_l: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, BT2, 32]
        )
    else:
        mma2_l: gl.constexpr = mma1_l

    pid_w = gl.program_id(0)
    pid_t = gl.program_id(1)
    pid_s = gl.program_id(2)
    w0 = pid_w * _BW
    t0 = pid_t * (BT1 + BT2)
    if GROUP_N > 0:
        # First 128-wide token K block of this CTA's group (K % 128 == 0 when grouped).
        kg0 = (w0 // GROUP_N) * NKB
    else:
        kg0 = 0
    kb_lo = pid_s * NKB // SPLIT
    kb_hi = (pid_s + 1) * NKB // SPLIT
    n_kb = kb_hi - kb_lo
    has_tail = (TAIL > 0) & (kb_hi == NKB)

    w_smem = gl.allocate_shared_memory(gl.float8e4nv, [STAGES, _BW, _BK], w_desc.layout)
    x1_smem = gl.allocate_shared_memory(
        gl.float8e4nv, [STAGES, BT1, _BK], x1_desc.layout
    )
    s1_smem = gl.allocate_shared_memory(gl.float32, [STAGES, 4, BT1], s1_desc.layout)
    o1_smem = gl.allocate_shared_memory(gl.bfloat16, [BT1, _BW], o1_desc.layout)
    if BT2 > 0:
        x2_smem = gl.allocate_shared_memory(
            gl.float8e4nv, [STAGES, BT2, _BK], x2_desc.layout
        )
        s2_smem = gl.allocate_shared_memory(
            gl.float32, [STAGES, 4, BT2], s2_desc.layout
        )
        o2_smem = gl.allocate_shared_memory(gl.bfloat16, [BT2, _BW], o2_desc.layout)
    else:
        x2_smem = x1_smem
        s2_smem = s1_smem
        o2_smem = o1_smem
    ready = mbarrier.allocate_mbarrier(batch=STAGES)
    empty = mbarrier.allocate_mbarrier(batch=STAGES)
    for s in gl.static_range(STAGES):
        mbarrier.init(ready.index(s), count=1)
        mbarrier.init(empty.index(s), count=1)

    offs_w = w0 + gl.arange(0, _BW, layout=gl.SliceLayout(1, mma1_l))
    sw_offs = gl.minimum(offs_w, N - 1) // 32 * KG

    acc1, acc2 = gl.warp_specialize(
        [
            (
                _consumer,
                (
                    w_smem,
                    x1_smem,
                    x2_smem,
                    s1_smem,
                    s2_smem,
                    ready,
                    empty,
                    sw_ptr,
                    sw_offs,
                    kb_lo,
                    n_kb,
                    has_tail,
                    KG,
                    BT1,
                    BT2,
                    STAGES,
                    TAIL,
                    mma1_l,
                    mma2_l,
                ),
            ),
            (
                _producer,
                (
                    w_desc,
                    x1_desc,
                    x2_desc,
                    s1_desc,
                    s2_desc,
                    w_smem,
                    x1_smem,
                    x2_smem,
                    s1_smem,
                    s2_smem,
                    ready,
                    empty,
                    w0,
                    t0,
                    kb_lo,
                    n_kb,
                    kg0,
                    BT1,
                    BT2,
                    STAGES,
                    PDL,
                ),
            ),
        ],
        [1],
        [PRODUCER_REGS],
    )

    if SPLIT == 1:
        _tma_store_tile(
            acc1, acc2, o1_smem, o2_smem, o1_desc, o2_desc, t0, w0, BT1, BT2
        )
    else:
        REGION: gl.constexpr = _BW * (BT1 + BT2)
        tile = pid_w * gl.num_programs(1) + pid_t
        tbase = part_ptr + tile.to(gl.int64) * (SPLIT * REGION)
        f1 = _frag_offs(BT1, mma1_l)
        gl.store(tbase + pid_s * REGION + f1, acc1)
        if BT2 > 0:
            f2 = _BW * BT1 + _frag_offs(BT2, mma2_l)
            gl.store(tbase + pid_s * REGION + f2, acc2)
        gl.barrier()
        cnt = gl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
        if cnt == SPLIT - 1:
            gl.barrier()
            tot1 = gl.load(tbase + f1, cache_modifier=".cg")
            for sidx in gl.static_range(1, SPLIT):
                tot1 += gl.load(tbase + sidx * REGION + f1, cache_modifier=".cg")
            if BT2 > 0:
                tot2 = gl.load(tbase + f2, cache_modifier=".cg")
                for sidx in gl.static_range(1, SPLIT):
                    tot2 += gl.load(tbase + sidx * REGION + f2, cache_modifier=".cg")
            else:
                tot2 = tot1
            _tma_store_tile(
                tot1, tot2, o1_smem, o2_smem, o1_desc, o2_desc, t0, w0, BT1, BT2
            )
            gl.store(cnt_ptr + tile, 0)


_layouts = {}


def _smem_layout(rows: int, cols: int, dtype) -> gl.NVMMASharedLayout:
    key = (rows, cols, dtype)
    if key not in _layouts:
        _layouts[key] = gl.NVMMASharedLayout.get_default_for([rows, cols], dtype)
    return _layouts[key]


_SCALE_LAYOUT = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=32, rank=2)


def num_tiles(n: int, m: int, bt1: int, bt2: int) -> int:
    return triton.cdiv(n, BLOCK_W) * triton.cdiv(m, bt1 + bt2)


def block32_swapab_gemm(
    q: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    bt1: int,
    bt2: int,
    stages: int,
    split: int,
    maxnreg: int,
    counters: Optional[torch.Tensor],
    out: Optional[torch.Tensor] = None,
    group_n: int = 0,
) -> torch.Tensor:
    """q [M, K] e4m3 (row stride a multiple of 16 B), x_scale its MN-major fp32
    view [M, K / 32] (strides (1, padded M), padded M % 4 == 0), weight [N, K]
    e4m3, weight_scale [ceil(N / 32), K / 32] fp32 contiguous -> bf16 [M, N]
    (written into ``out`` when given; its row stride may exceed N).

    ``counters``: int32 zeros with at least ``num_tiles`` entries, owned by this
    weight (split > 1 only); the kernel leaves them zero again.

    ``group_n`` > 0 (a multiple of 64, K a multiple of 128, split 1): q is
    [M, N / group_n * K] and x_scale [M, N / group_n * K / 32], one K slice per
    group of ``group_n`` weight rows.
    """
    m = q.shape[0]
    n, k = weight.shape
    if group_n:
        assert (
            n % group_n == 0 and group_n % BLOCK_W == 0 and k % 128 == 0 and split == 1
        )
        assert q.shape[1] == n // group_n * k
    else:
        assert q.shape[1] == k
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=q.device)
    x_scale_t = x_scale.t()
    w_desc = TensorDescriptor.from_tensor(
        weight, [BLOCK_W, 128], _smem_layout(BLOCK_W, 128, gl.float8e4nv)
    )
    x1_desc = TensorDescriptor.from_tensor(
        q, [bt1, 128], _smem_layout(bt1, 128, gl.float8e4nv)
    )
    s1_desc = TensorDescriptor.from_tensor(x_scale_t, [4, bt1], _SCALE_LAYOUT)
    o1_desc = TensorDescriptor.from_tensor(
        out, [bt1, BLOCK_W], _smem_layout(bt1, BLOCK_W, gl.bfloat16)
    )
    if bt2 > 0:
        x2_desc = TensorDescriptor.from_tensor(
            q, [bt2, 128], _smem_layout(bt2, 128, gl.float8e4nv)
        )
        s2_desc = TensorDescriptor.from_tensor(x_scale_t, [4, bt2], _SCALE_LAYOUT)
        o2_desc = TensorDescriptor.from_tensor(
            out, [bt2, BLOCK_W], _smem_layout(bt2, BLOCK_W, gl.bfloat16)
        )
    else:
        x2_desc, s2_desc, o2_desc = x1_desc, s1_desc, o1_desc
    grid = (triton.cdiv(n, BLOCK_W), triton.cdiv(m, bt1 + bt2), split)
    if split > 1:
        assert counters is not None and counters.numel() >= grid[0] * grid[1]
        part = torch.empty(
            (grid[0] * grid[1] * split * BLOCK_W * (bt1 + bt2),),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        part = x_scale
        counters = x_scale
    pdl = is_arch_support_pdl()
    _block32_swapab_kernel[grid](
        w_desc,
        x1_desc,
        x2_desc,
        s1_desc,
        s2_desc,
        o1_desc,
        o2_desc,
        weight_scale,
        part,
        counters,
        n,
        KG=k // 32,
        BT1=bt1,
        BT2=bt2,
        STAGES=stages,
        SPLIT=split,
        PRODUCER_REGS=24,
        PDL=pdl,
        GROUP_N=group_n,
        num_warps=4,
        maxnreg=maxnreg,
        launch_pdl=pdl,
    )
    return out


class SwapABConfig(msgspec.Struct, frozen=True):
    # Token tile = bt1 + bt2 rows, each a power of two (bt2 may be 0).
    bt1: int
    bt2: int
    stages: int
    split: int
    # Register cap per thread; sets how many CTAs share an SM.
    maxnreg: int


# (n, k) or grouped (n, k, groups) -> ((m, config or None for DeepGEMM), ...) by m.
SwapABTable = Dict[Tuple[int, ...], Tuple[Tuple[int, Optional[SwapABConfig]], ...]]


def parse_swapab_table(raw: Dict[str, Dict[str, Optional[dict]]]) -> SwapABTable:
    table = {}
    for shape, rows in raw.items():
        n_part, k_part, *g_part = shape.split(",")
        n = int(n_part.removeprefix("N="))
        k = int(k_part.removeprefix("K="))
        groups = int(g_part[0].removeprefix("G=")) if g_part else 1
        if groups > 1 and (n % BLOCK_W or k % 128):
            raise ValueError(
                f"{shape}: grouped tiles need N % {BLOCK_W} == 0 and K % 128 == 0"
            )
        entries = sorted(
            (int(m), None if cfg is None else msgspec.convert(cfg, SwapABConfig))
            for m, cfg in rows.items()
        )
        for m, cfg in entries:
            if cfg is None:
                continue
            for bt in (cfg.bt1, cfg.bt2):
                if bt and (bt < 8 or bt > 128 or bt & (bt - 1)):
                    raise ValueError(
                        f"{shape} M={m}: token tile {bt} is not a power of two in [8, 128]"
                    )
            if cfg.split < 1 or cfg.split > (k // 32 + 3) // 4:
                raise ValueError(
                    f"{shape} M={m}: split {cfg.split} exceeds the K blocks"
                )
            if groups > 1 and cfg.split != 1:
                raise ValueError(f"{shape} M={m}: grouped entries run split 1")
        table[(n, k) if groups == 1 else (n, k, groups)] = tuple(entries)
    return table


def _read_swapab_table(path: Optional[str]) -> SwapABTable:
    if path is None:
        return {}
    with open(path) as f:
        return parse_swapab_table(json.load(f))


@functools.lru_cache(maxsize=None)
def load_swapab_table(device_name: str, role: str) -> SwapABTable:
    """The device's table with the role's rows (mix / prefill / decode) layered
    over it; empty when the device was not measured."""
    table = _read_swapab_table(find_tuned_config("block32_swapab", device_name))
    rows = _read_swapab_table(
        find_tuned_config("block32_swapab", device_name, suffix=f",{role}")
    )
    return {**table, **rows}


def lookup_swapab(
    table: SwapABTable, n: int, k: int, m: int, groups: int = 1
) -> Optional[SwapABConfig]:
    """The entry at the smallest listed row count >= m; None past the largest
    listed row count, for an unlisted shape, or where DeepGEMM was faster."""
    rows = table.get((n, k) if groups == 1 else (n, k, groups))
    if not rows or m <= 0 or m > rows[-1][0]:
        return None
    return next(cfg for row_m, cfg in rows if row_m >= m)
