"""SM90 FP8 LM-head GEMM, swap-AB, block-[128, 128] recipe (Gluon).

    out[m, n] = bf16(sum_kb sa[m, kb] * sw[n // 128, kb] * dot128(x[m, 128kb:], w[n, 128kb:]))

Per output element the arithmetic is DeepGEMM sm90_fp8_gemm_1d2d's: a fresh
fp32 wgmma partial over each 128-wide K block (four k=32 wgmmas), promoted in
K order as acc = fma.rn(mul.rn(sa, sw), partial, acc), then one bf16 rounding.

Weight rows are the wgmma M side (64 per consumer warpgroup), token rows the
N side, so a token tile is one or two power-of-two pieces (80 = 64 + 16,
160 = 128 + 32) instead of DeepGEMM's 64 / 128 / 256-row tiles. Token tiles of
one weight tile are adjacent in the grid so the weight streams from HBM once.
One producer warp streams weight / token / scale stages with TMA; the consumer
alternates two partial banks so the promotion of block kb overlaps the wgmmas
of block kb + 1.

The block-[32, 32] kernel in fp8_block32_swapab.py promotes every 32-wide K
group instead, which rounds differently; the two share only their helpers.
"""

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
from sglang.kernels.ops.gemm.fp8_block32_swapab import (
    _SCALE_LAYOUT,
    _gdc_wait,
    _smem_layout,
    _tma_store_tile,
)

# K per TMA stage and per scale block.
_BK = gl.constexpr(128)
# TMA writes shared memory at 128-byte granularity: a stage's token scales
# take at least 32 fp32 slots, the extra ones unused.
_MIN_SCALE_COLS = 32


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
    tma.async_copy_global_to_shared(s1_desc, [kb, t0], bar, s1_smem.index(stage))
    if BT2 > 0:
        tma.async_copy_global_to_shared(
            x2_desc, [t0 + BT1, kb * _BK], bar, x2_smem.index(stage)
        )
        tma.async_copy_global_to_shared(
            s2_desc, [kb, t0 + BT1], bar, s2_smem.index(stage)
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
    KB: gl.constexpr,
    BW: gl.constexpr,
    BT1: gl.constexpr,
    BT2: gl.constexpr,
    STAGES: gl.constexpr,
    PDL: gl.constexpr,
):
    SC1: gl.constexpr = s1_desc.block_type.shape[1]
    SC2: gl.constexpr = s2_desc.block_type.shape[1] if BT2 > 0 else 0
    NBYTES: gl.constexpr = (BW + BT1 + BT2) * _BK + 4 * (SC1 + SC2)
    j0 = 0
    if PDL:
        # Weights do not depend on the previous kernel: start streaming them,
        # then wait for it before touching the activation and its scales.
        for s in gl.static_range(STAGES):
            if s < KB:
                mbarrier.expect(ready.index(s), NBYTES)
                tma.async_copy_global_to_shared(
                    w_desc, [w0, s * _BK], ready.index(s), w_smem.index(s)
                )
        _gdc_wait()
        for s in gl.static_range(STAGES):
            if s < KB:
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
                    s,
                    t0,
                    BT1,
                    BT2,
                )
        j0 = STAGES
    for j in range(j0, KB):
        stage = j % STAGES
        if j >= STAGES:
            mbarrier.wait(empty.index(stage), ((j // STAGES) - 1) & 1)
            # Generic -> async proxy fence: a consumer warp's ld.shared of this
            # stage's scales may still be pending; the TMA refill must not overtake it.
            fence_async_shared()
        bar = ready.index(stage)
        mbarrier.expect(bar, NBYTES)
        tma.async_copy_global_to_shared(w_desc, [w0, j * _BK], bar, w_smem.index(stage))
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
            j,
            t0,
            BT1,
            BT2,
        )


@gluon.jit
def _issue_block(w_t, x1_t, x2_t, q1, q2, BT2: gl.constexpr):
    # Four k=32 wgmmas accumulate one 128-wide K block into a fresh bank.
    for sub in gl.static_range(4):
        q1 = warpgroup_mma(
            w_t.slice(sub * 32, 32, dim=1),
            x1_t.slice(sub * 32, 32, dim=1).permute((1, 0)),
            q1,
            use_acc=sub > 0,
            is_async=True,
        )
        if BT2 > 0:
            q2 = warpgroup_mma(
                w_t.slice(sub * 32, 32, dim=1),
                x2_t.slice(sub * 32, 32, dim=1).permute((1, 0)),
                q2,
                use_acc=sub > 0,
                is_async=True,
            )
    return q1, q2


@gluon.jit
def _wait_bank(q1, q2, pending: gl.constexpr, BT2: gl.constexpr):
    if BT2 > 0:
        q1, q2 = warpgroup_mma_wait(pending, deps=[q1, q2])
    else:
        q1 = warpgroup_mma_wait(pending, deps=[q1])
    return q1, q2


@gluon.jit
def _load_sa(s_t, BT: gl.constexpr, col_l: gl.constexpr):
    # [1, BT] row of the staged scales in the accumulator layout; the size-1
    # reduction lands it in the column slice layout without a shuffle.
    return gl.max(s_t.slice(0, BT, dim=1).load(col_l.parent), axis=0)


@gluon.jit
def _promote_block(
    acc1,
    acc2,
    p1,
    p2,
    s1_t,
    s2_t,
    sw,
    BT1: gl.constexpr,
    BT2: gl.constexpr,
    col1_l: gl.constexpr,
    col2_l: gl.constexpr,
):
    # One weight-scale block per CTA: its rows sit in one 128-row block.
    s1, _ = gl.broadcast((_load_sa(s1_t, BT1, col1_l) * sw)[None, :], p1)
    acc1 = gl.fma(s1, p1, acc1)
    if BT2 > 0:
        s2, _ = gl.broadcast((_load_sa(s2_t, BT2, col2_l) * sw)[None, :], p2)
        acc2 = gl.fma(s2, p2, acc2)
    return acc1, acc2


@gluon.jit
def _release(empty, stage):
    # Every warp has issued this stage's scale reads and its wgmmas retired;
    # the producer proxy-fences before the refill.
    gl.barrier()
    mbarrier.arrive(empty.index(stage), count=1)


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
    sw_row,
    KB: gl.constexpr,
    BW: gl.constexpr,
    BT1: gl.constexpr,
    BT2: gl.constexpr,
    STAGES: gl.constexpr,
    mma1_l: gl.constexpr,
    mma2_l: gl.constexpr,
):
    col1_l: gl.constexpr = gl.SliceLayout(0, mma1_l)
    col2_l: gl.constexpr = gl.SliceLayout(0, mma2_l)
    BT2S: gl.constexpr = BT2 if BT2 > 0 else 8
    # Groups in flight behind a bank: one per wgmma call.
    PENDING: gl.constexpr = 4 * (2 if BT2 > 0 else 1)
    acc1 = gl.zeros([BW, BT1], gl.float32, layout=mma1_l)
    a1 = acc1
    b1 = acc1
    acc2 = gl.zeros([BW, BT2S], gl.float32, layout=mma2_l)
    a2 = acc2
    b2 = acc2
    # KB is even: block 2i accumulates in bank a, block 2i + 1 in bank b.
    mbarrier.wait(ready.index(0), 0)
    a1, a2 = _issue_block(
        w_smem.index(0), x1_smem.index(0), x2_smem.index(0), a1, a2, BT2
    )
    sw_a = gl.load(sw_ptr + sw_row)
    for i in range(KB // 2 - 1):
        jb = 2 * i + 1
        ja = jb + 1
        sb = jb % STAGES
        sa = ja % STAGES
        sp = (jb - 1) % STAGES
        sw_b = gl.load(sw_ptr + sw_row + jb)
        mbarrier.wait(ready.index(sb), (jb // STAGES) & 1)
        b1, b2 = _issue_block(
            w_smem.index(sb), x1_smem.index(sb), x2_smem.index(sb), b1, b2, BT2
        )
        a1, a2 = _wait_bank(a1, a2, PENDING, BT2)
        acc1, acc2 = _promote_block(
            acc1,
            acc2,
            a1,
            a2,
            s1_smem.index(sp),
            s2_smem.index(sp),
            sw_a,
            BT1,
            BT2,
            col1_l,
            col2_l,
        )
        _release(empty, sp)
        sw_a = gl.load(sw_ptr + sw_row + ja)
        mbarrier.wait(ready.index(sa), (ja // STAGES) & 1)
        a1, a2 = _issue_block(
            w_smem.index(sa), x1_smem.index(sa), x2_smem.index(sa), a1, a2, BT2
        )
        b1, b2 = _wait_bank(b1, b2, PENDING, BT2)
        acc1, acc2 = _promote_block(
            acc1,
            acc2,
            b1,
            b2,
            s1_smem.index(sb),
            s2_smem.index(sb),
            sw_b,
            BT1,
            BT2,
            col1_l,
            col2_l,
        )
        _release(empty, sb)
    jb = KB - 1
    sb = jb % STAGES
    sp = (jb - 1) % STAGES
    sw_b = gl.load(sw_ptr + sw_row + jb)
    mbarrier.wait(ready.index(sb), (jb // STAGES) & 1)
    b1, b2 = _issue_block(
        w_smem.index(sb), x1_smem.index(sb), x2_smem.index(sb), b1, b2, BT2
    )
    a1, a2 = _wait_bank(a1, a2, PENDING, BT2)
    acc1, acc2 = _promote_block(
        acc1,
        acc2,
        a1,
        a2,
        s1_smem.index(sp),
        s2_smem.index(sp),
        sw_a,
        BT1,
        BT2,
        col1_l,
        col2_l,
    )
    _release(empty, sp)
    b1, b2 = _wait_bank(b1, b2, 0, BT2)
    acc1, acc2 = _promote_block(
        acc1,
        acc2,
        b1,
        b2,
        s1_smem.index(sb),
        s2_smem.index(sb),
        sw_b,
        BT1,
        BT2,
        col1_l,
        col2_l,
    )
    _release(empty, sb)
    return acc1, acc2


@gluon.jit
def _lm_head_swapab_kernel(
    w_desc,
    x1_desc,
    x2_desc,
    s1_desc,
    s2_desc,
    o1_desc,
    o2_desc,
    sw_ptr,
    KB: gl.constexpr,
    BT1: gl.constexpr,
    BT2: gl.constexpr,
    STAGES: gl.constexpr,
    PRODUCER_REGS: gl.constexpr,
    PDL: gl.constexpr,
    num_warps: gl.constexpr,
):
    # One consumer warpgroup per 64 weight rows.
    BW: gl.constexpr = num_warps * 16
    mma1_l: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, BT1, 32]
    )
    if BT2 > 0:
        mma2_l: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, BT2, 32]
        )
    else:
        mma2_l: gl.constexpr = mma1_l

    # Token tiles of one weight tile run back to back.
    pid_t = gl.program_id(0)
    pid_w = gl.program_id(1)
    w0 = pid_w * BW
    t0 = pid_t * (BT1 + BT2)

    w_smem = gl.allocate_shared_memory(gl.float8e4nv, [STAGES, BW, _BK], w_desc.layout)
    x1_smem = gl.allocate_shared_memory(
        gl.float8e4nv, [STAGES, BT1, _BK], x1_desc.layout
    )
    s1_smem = gl.allocate_shared_memory(
        gl.float32, [STAGES, 1, s1_desc.block_type.shape[1]], s1_desc.layout
    )
    o1_smem = gl.allocate_shared_memory(gl.bfloat16, [BT1, BW], o1_desc.layout)
    if BT2 > 0:
        x2_smem = gl.allocate_shared_memory(
            gl.float8e4nv, [STAGES, BT2, _BK], x2_desc.layout
        )
        s2_smem = gl.allocate_shared_memory(
            gl.float32, [STAGES, 1, s2_desc.block_type.shape[1]], s2_desc.layout
        )
        o2_smem = gl.allocate_shared_memory(gl.bfloat16, [BT2, BW], o2_desc.layout)
    else:
        x2_smem = x1_smem
        s2_smem = s1_smem
        o2_smem = o1_smem
    ready = mbarrier.allocate_mbarrier(batch=STAGES)
    empty = mbarrier.allocate_mbarrier(batch=STAGES)
    for s in gl.static_range(STAGES):
        mbarrier.init(ready.index(s), count=1)
        mbarrier.init(empty.index(s), count=1)

    sw_row = (w0 // 128) * KB

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
                    sw_row,
                    KB,
                    BW,
                    BT1,
                    BT2,
                    STAGES,
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
                    KB,
                    BW,
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
    _tma_store_tile(acc1, acc2, o1_smem, o2_smem, o1_desc, o2_desc, t0, w0, BT1, BT2)


def lm_head_swapab_gemm(
    q: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    block_w: int,
    bt1: int,
    bt2: int,
    stages: int,
    maxnreg: int,
) -> torch.Tensor:
    """q [M, K] e4m3, x_scale its MN-major fp32 view [M, K / 128] (strides
    (1, padded M), padded M % 4 == 0), weight [N, K] e4m3 with N % 128 == 0,
    weight_scale [N / 128, K / 128] fp32 contiguous -> ``out`` [M, N] bf16.
    ``block_w``: weight rows per CTA, 64 or 128 (one consumer warpgroup each 64)."""
    m, k = q.shape
    n = weight.shape[0]
    assert block_w in (64, 128) and n % 128 == 0 and k % 256 == 0, (block_w, n, k)
    assert (
        tuple(weight_scale.shape) == (n // 128, k // 128)
        and weight_scale.is_contiguous()
    )
    x_scale_t = x_scale.t()
    w_desc = TensorDescriptor.from_tensor(
        weight, [block_w, 128], _smem_layout(block_w, 128, gl.float8e4nv)
    )
    x1_desc = TensorDescriptor.from_tensor(
        q, [bt1, 128], _smem_layout(bt1, 128, gl.float8e4nv)
    )
    s1_desc = TensorDescriptor.from_tensor(
        x_scale_t, [1, max(bt1, _MIN_SCALE_COLS)], _SCALE_LAYOUT
    )
    o1_desc = TensorDescriptor.from_tensor(
        out, [bt1, block_w], _smem_layout(bt1, block_w, gl.bfloat16)
    )
    if bt2 > 0:
        x2_desc = TensorDescriptor.from_tensor(
            q, [bt2, 128], _smem_layout(bt2, 128, gl.float8e4nv)
        )
        s2_desc = TensorDescriptor.from_tensor(
            x_scale_t, [1, max(bt2, _MIN_SCALE_COLS)], _SCALE_LAYOUT
        )
        o2_desc = TensorDescriptor.from_tensor(
            out, [bt2, block_w], _smem_layout(bt2, block_w, gl.bfloat16)
        )
    else:
        x2_desc, s2_desc, o2_desc = x1_desc, s1_desc, o1_desc
    grid = (triton.cdiv(m, bt1 + bt2), n // block_w)
    pdl = is_arch_support_pdl()
    _lm_head_swapab_kernel[grid](
        w_desc,
        x1_desc,
        x2_desc,
        s1_desc,
        s2_desc,
        o1_desc,
        o2_desc,
        weight_scale,
        KB=k // 128,
        BT1=bt1,
        BT2=bt2,
        STAGES=stages,
        PRODUCER_REGS=24,
        PDL=pdl,
        num_warps=block_w // 16,
        maxnreg=maxnreg,
        launch_pdl=pdl,
    )
    return out
