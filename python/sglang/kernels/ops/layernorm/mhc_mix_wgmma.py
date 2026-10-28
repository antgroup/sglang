"""SM90 hc_mix_stats partial kernel on wgmma (Gluon), bitwise the production partials.

Writes the same ``part_mix`` / ``part_sq`` as ``_hc_mix_stats_partial_kernel`` in
``mhc.py`` (tf32x3 dot, 64-wide K tiles, K slices) for bf16 x:

* Triton (3.7, 3.8) lowers the production tf32 dots to mma.sync m16n8k8 with the bf16 x
  operand loaded through ldmatrix (kWidth 2), so k-step 2b of a 64-wide tile takes
  the even columns of 16-column block b and k-step 2b + 1 the odd ones. The tensor
  core's sum within one k-step does not depend on the order of its 8 products, and
  wgmma m64n32k8 rounds like mma.sync for the same k-step column sets (both
  measured bitwise on H20), so every output element here sees the same k-step sets
  in the same chain order: dot(x, wSmall) from zero, NaN to zero, dot(x, wBig)
  continuing from it, then ``acc = d + acc`` per 64-wide tile.
* x is the wgmma A operand in registers: a 32-bit word of a bf16 pair is the even
  column as ``w << 16`` and the odd one as ``w & 0xffff0000``, exact in fp32, so A
  never goes through shared memory. W is split (cvt.rna.tf32) into shared memory
  with its columns in the same k-step order.
* The row sum of squares repeats the production reduction tree as size-2 sums.
* Persistent CTAs walk (64-row tile, K slice) units; a CTA computes one slice of
  one row tile at a time, so rows stay independent of the batch size.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_wait,
)

_BLOCK_M = 64
_MIX_PAD = 32
# Measured on H20: two CTAs per SM below 1024 rows; three with a 168-register cap
# from 1024 rows (prefill chunks).
_LARGE_M = 1024
_CTAS_PER_SM = 2
_CTAS_PER_SM_LARGE = 3
_MAXNREG_LARGE = 168
# A whole K slice of split W in shared memory: 2 x K / 64 tiles of [32, 64] fp32,
# 64 KiB at 256, three CTAs per SM.
_SLICE_W_MAX_K = 256


@gluon.jit
def _rna_tf32(v):
    return gl.inline_asm_elementwise(
        "cvt.rna.tf32.f32 $0, $1;",
        "=r,r",
        [v],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _load_x_words(xw_ptr, xw_stride, M, row0, kw0, valid, layout: gl.constexpr):
    """[64, 32] int32 words (bf16 pairs) of rows row0.. and word columns kw0.."""
    r = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    q = gl.arange(0, 32, layout=gl.SliceLayout(0, layout))
    rows = (row0 + r)[:, None].to(gl.int64)
    ptrs = xw_ptr + rows * xw_stride + kw0 + q[None, :]
    return gl.load(ptrs, mask=((row0 + r < M) & valid)[:, None], other=0)


@gluon.jit
def _words_to_a(words, a_layout: gl.constexpr):
    # Word q = 8b + 2t + p holds x columns 2q (low half) and 2q + 1 (high half);
    # A column 16b + 8h + 4p + t (k-step 2b + h, slot 4p + t) takes half h of it.
    lo = (words << 16).to(gl.float32, bitcast=True)
    hi = (words & -65536).to(gl.float32, bitcast=True)
    a = gl.reshape(gl.join(lo, hi), [64, 4, 4, 2, 2])
    a = gl.reshape(gl.permute(a, (0, 1, 4, 3, 2)), [64, 64])
    return gl.convert_layout(a, a_layout)


@gluon.jit
def _sq_tree64(a):
    # Same adds as mhc._hc_mix_sq_tree64. A column 16(2g + beta) + 8h + 4p + 2t1 + t0
    # is x column 32g + 16beta + j with j = 8t1 + 4t0 + 2p + h.
    a = gl.permute(gl.reshape(a, [64, 2, 2, 16]), (0, 1, 3, 2))
    own, partner = gl.split(a)
    s = gl.fma(own, own, partner * partner)
    s = gl.reshape(s, [64, 2, 2, 2, 2, 2])  # (g, h, p, t1, t0)
    s = gl.sum(s, axis=4)  # j + (j + 8)
    s = gl.sum(s, axis=4)  # j + (j + 4)
    s = gl.sum(s, axis=3)  # j + (j + 2)
    s = gl.sum(s, axis=2)  # j + (j + 1)
    return gl.sum(s, axis=1)  # columns 0..31 + columns 32..63


@gluon.jit
def _load_w(
    w_ptr,
    w_stride_n,
    k0,
    valid,
    MIX: gl.constexpr,
    MIX_PAD: gl.constexpr,
    layout: gl.constexpr,
):
    n = gl.arange(0, MIX_PAD, layout=gl.SliceLayout(1, layout))
    k = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    ptrs = w_ptr + k0 + n[:, None] * w_stride_n + k[None, :]
    return gl.load(ptrs, mask=((n < MIX) & valid)[:, None], other=0.0)


@gluon.jit
def _stage_w(w, w_small_buf, w_big_buf):
    # W column 16b + 4t + 2p + h goes to B row 16b + 8h + 4p + t, the A column order.
    NPAD: gl.constexpr = w.shape[0]
    w = gl.permute(gl.reshape(w, [NPAD, 4, 4, 2, 2]), (0, 1, 4, 3, 2))
    w = gl.reshape(w, [NPAD, 64])
    w_big = _rna_tf32(w)
    w_small_buf.store(w - w_big)
    w_big_buf.store(w_big)
    fence_async_shared()


@gluon.jit
def _hc_mix_partial_wgmma_kernel(
    xw_ptr,
    w_ptr,
    part_mix_ptr,
    part_sq_ptr,
    M,
    xw_stride,
    w_stride_n,
    n_units,
    n_m_tiles,
    MIX: gl.constexpr,
    MIX_PAD: gl.constexpr,
    K_PER_SLICE: gl.constexpr,
    num_warps: gl.constexpr,
):
    BM: gl.constexpr = 64
    BK: gl.constexpr = 64
    T: gl.constexpr = K_PER_SLICE // BK
    gl.static_assert(num_warps == 4)
    gl.static_assert(K_PER_SLICE % BK == 0)
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, MIX_PAD, 8]
    )
    a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=1
    )
    # a_layout with the two halves of each word (k-steps 2b and 2b + 1) folded,
    # so a thread's bf16 pairs load as 8-byte vectors.
    words_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [8, 0], [0, 8], [0, 16]],
        lane_bases=[[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[16, 0], [32, 0]],
        block_bases=[],
        shape=[64, 32],
    )
    w_layout: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [num_warps, 1], [1, 0])
    b_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [MIX_PAD, BK], gl.float32
    )
    w_small_smem = gl.allocate_shared_memory(gl.float32, [2, MIX_PAD, BK], b_layout)
    w_big_smem = gl.allocate_shared_memory(gl.float32, [2, MIX_PAD, BK], b_layout)

    pid = gl.program_id(0)
    grid = gl.num_programs(0)
    u = pid
    true = u >= 0
    k0 = (u // n_m_tiles) * K_PER_SLICE
    r0 = (u % n_m_tiles) * BM
    xr = _load_x_words(xw_ptr, xw_stride, M, r0, k0 // 2, true, words_layout)
    wr = _load_w(w_ptr, w_stride_n, k0, true, MIX, MIX_PAD, w_layout)
    _stage_w(wr, w_small_smem.index(0), w_big_smem.index(0))
    wr = _load_w(w_ptr, w_stride_n, k0 + BK, true, MIX, MIX_PAD, w_layout)
    d = gl.zeros([BM, MIX_PAD], gl.float32, layout=mma_layout)
    it = 0
    while u < n_units:
        sl = u // n_m_tiles
        r0 = (u % n_m_tiles) * BM
        kb = sl * K_PER_SLICE
        un = u + grid
        nv = un < n_units
        rn = (un % n_m_tiles) * BM
        kn = (un // n_m_tiles) * K_PER_SLICE
        for t in gl.static_range(T):
            buf = it % 2
            a = _words_to_a(xr, a_layout)
            if t + 1 < T:
                kw = (kb + (t + 1) * BK) // 2
                xr = _load_x_words(xw_ptr, xw_stride, M, r0, kw, true, words_layout)
            else:
                xr = _load_x_words(xw_ptr, xw_stride, M, rn, kn // 2, nv, words_layout)
            gl.barrier()
            small = warpgroup_mma(
                a,
                w_small_smem.index(buf).permute((1, 0)),
                d,
                use_acc=False,
                is_async=True,
            )
            tile_sq = _sq_tree64(a)
            if t == 0:
                sq = tile_sq + 0.0
            else:
                sq = sq + tile_sq
            d = warpgroup_mma_wait(0, deps=[small])
            d = gl.where(d != d, 0.0, d)
            big = warpgroup_mma(
                a, w_big_smem.index(buf).permute((1, 0)), d, is_async=True
            )
            # The other buffer's last reader, the previous tile's big chain, retired.
            _stage_w(wr, w_small_smem.index(1 - buf), w_big_smem.index(1 - buf))
            if t + 2 < T:
                kw = kb + (t + 2) * BK
                wr = _load_w(w_ptr, w_stride_n, kw, true, MIX, MIX_PAD, w_layout)
            else:
                kw = kn + (t + 2 - T) * BK
                wr = _load_w(w_ptr, w_stride_n, kw, nv, MIX, MIX_PAD, w_layout)
            d = warpgroup_mma_wait(0, deps=[big])
            if t == 0:
                acc = d + 0.0
            else:
                acc = d + acc
            it += 1
        rows = r0 + gl.arange(0, BM, layout=gl.SliceLayout(1, mma_layout))
        cols = gl.arange(0, MIX_PAD, layout=gl.SliceLayout(0, mma_layout))
        gl.store(
            part_mix_ptr + (sl * M + rows[:, None]) * MIX + cols[None, :],
            acc,
            mask=(rows < M)[:, None] & (cols < MIX)[None, :],
        )
        sq_rows = r0 + gl.arange(0, BM, layout=sq.type.layout)
        gl.store(part_sq_ptr + sl * M + sq_rows, sq, mask=sq_rows < M)
        u = un


@gluon.jit
def _hc_mix_partial_wgmma_slice_w_kernel(
    xw_ptr,
    w_ptr,
    part_mix_ptr,
    part_sq_ptr,
    M,
    xw_stride,
    w_stride_n,
    n_units,
    n_m_tiles,
    units_per_cta,
    MIX: gl.constexpr,
    MIX_PAD: gl.constexpr,
    K_PER_SLICE: gl.constexpr,
    num_warps: gl.constexpr,
):
    """_hc_mix_partial_wgmma_kernel over contiguous unit runs: a K slice's split W
    is staged once per slice, not per 64-row tile; each unit's chains are unchanged."""
    BM: gl.constexpr = 64
    BK: gl.constexpr = 64
    T: gl.constexpr = K_PER_SLICE // BK
    gl.static_assert(num_warps == 4)
    gl.static_assert(K_PER_SLICE % BK == 0)
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, MIX_PAD, 8]
    )
    a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=1
    )
    words_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [8, 0], [0, 8], [0, 16]],
        lane_bases=[[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[16, 0], [32, 0]],
        block_bases=[],
        shape=[64, 32],
    )
    w_layout: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [num_warps, 1], [1, 0])
    b_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [MIX_PAD, BK], gl.float32
    )
    w_small_smem = gl.allocate_shared_memory(gl.float32, [T, MIX_PAD, BK], b_layout)
    w_big_smem = gl.allocate_shared_memory(gl.float32, [T, MIX_PAD, BK], b_layout)

    u = gl.program_id(0) * units_per_cta
    u_end = gl.minimum(u + units_per_cta, n_units)
    true = u >= 0
    k_first = (u // n_m_tiles) * K_PER_SLICE
    r_first = (u % n_m_tiles) * BM
    xr = _load_x_words(
        xw_ptr, xw_stride, M, r_first, k_first // 2, u < u_end, words_layout
    )
    d = gl.zeros([BM, MIX_PAD], gl.float32, layout=mma_layout)
    staged = -1
    while u < u_end:
        sl = u // n_m_tiles
        r0 = (u % n_m_tiles) * BM
        kb = sl * K_PER_SLICE
        if sl != staged:
            # The previous slice's last chains were waited on before this point.
            gl.barrier()
            for t in gl.static_range(T):
                wr = _load_w(
                    w_ptr, w_stride_n, kb + t * BK, true, MIX, MIX_PAD, w_layout
                )
                _stage_w(wr, w_small_smem.index(t), w_big_smem.index(t))
            gl.barrier()
            staged = sl
        un = u + 1
        nv = un < u_end
        rn = (un % n_m_tiles) * BM
        kn = (un // n_m_tiles) * K_PER_SLICE
        for t in gl.static_range(T):
            a = _words_to_a(xr, a_layout)
            if t + 1 < T:
                kw = (kb + (t + 1) * BK) // 2
                xr = _load_x_words(xw_ptr, xw_stride, M, r0, kw, true, words_layout)
            else:
                xr = _load_x_words(xw_ptr, xw_stride, M, rn, kn // 2, nv, words_layout)
            small = warpgroup_mma(
                a,
                w_small_smem.index(t).permute((1, 0)),
                d,
                use_acc=False,
                is_async=True,
            )
            tile_sq = _sq_tree64(a)
            if t == 0:
                sq = tile_sq + 0.0
            else:
                sq = sq + tile_sq
            d = warpgroup_mma_wait(0, deps=[small])
            d = gl.where(d != d, 0.0, d)
            big = warpgroup_mma(
                a, w_big_smem.index(t).permute((1, 0)), d, is_async=True
            )
            d = warpgroup_mma_wait(0, deps=[big])
            if t == 0:
                acc = d + 0.0
            else:
                acc = d + acc
        rows = r0 + gl.arange(0, BM, layout=gl.SliceLayout(1, mma_layout))
        cols = gl.arange(0, MIX_PAD, layout=gl.SliceLayout(0, mma_layout))
        gl.store(
            part_mix_ptr + (sl * M + rows[:, None]) * MIX + cols[None, :],
            acc,
            mask=(rows < M)[:, None] & (cols < MIX)[None, :],
        )
        sq_rows = r0 + gl.arange(0, BM, layout=sq.type.layout)
        gl.store(part_sq_ptr + sl * M + sq_rows, sq, mask=sq_rows < M)
        u = un


_num_sms = {}


def hc_mix_partial_wgmma(
    x_flat: torch.Tensor,
    hc_fn: torch.Tensor,
    part_mix: torch.Tensor,
    part_sq: torch.Tensor,
    num_slices: int,
    *,
    slice_w: bool = False,
) -> None:
    """part_mix [num_slices, M, MIX] / part_sq [num_slices, M] of bf16 x_flat [M, K]
    (even row stride and offset) and fp32 hc_fn [MIX <= 32, K]. ``slice_w`` stages
    each K slice's W once per CTA from 1024 rows (same partials)."""
    m, k = x_flat.shape
    n_m_tiles = triton.cdiv(m, _BLOCK_M)
    n_units = n_m_tiles * num_slices
    device = x_flat.device.index
    if device is None:
        device = torch.cuda.current_device()
    if device not in _num_sms:
        _num_sms[device] = torch.cuda.get_device_properties(
            device
        ).multi_processor_count
    large = m >= _LARGE_M
    ctas = _num_sms[device] * (_CTAS_PER_SM_LARGE if large else _CTAS_PER_SM)
    xw = x_flat.view(torch.int32)
    k_per_slice = k // num_slices
    if slice_w and large and k_per_slice <= _SLICE_W_MAX_K:
        per_cta = triton.cdiv(n_units, min(n_units, ctas))
        _hc_mix_partial_wgmma_slice_w_kernel[(triton.cdiv(n_units, per_cta),)](
            xw,
            hc_fn,
            part_mix,
            part_sq,
            m,
            xw.stride(0),
            hc_fn.stride(0),
            n_units,
            n_m_tiles,
            per_cta,
            MIX=hc_fn.shape[0],
            MIX_PAD=_MIX_PAD,
            K_PER_SLICE=k_per_slice,
            num_warps=4,
            maxnreg=_MAXNREG_LARGE,
        )
        return
    _hc_mix_partial_wgmma_kernel[(min(n_units, ctas),)](
        xw,
        hc_fn,
        part_mix,
        part_sq,
        m,
        xw.stride(0),
        hc_fn.stride(0),
        n_units,
        n_m_tiles,
        MIX=hc_fn.shape[0],
        MIX_PAD=_MIX_PAD,
        K_PER_SLICE=k // num_slices,
        num_warps=4,
        **({"maxnreg": _MAXNREG_LARGE} if large else {}),
    )
