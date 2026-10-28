"""SM90 multi-row hc_mix stats for DeepSeek-V4.1, bitwise the split-K partials of
``mhc.hc_mix_stats`` followed by the TileLang ``hc_split_sinkhorn``.

The partials come from one of three kernels that write the same bits, picked by the
row count: the production ``_hc_mix_stats_partial_kernel``, its tf32x3 dot written
out (``_hc_mix_stats_partial_v2_kernel``), or the wgmma kernel in
``mhc_mix_wgmma``. The reduce and the sinkhorn run in one Gluon kernel
(``mhc_reduce_sinkhorn``). A row's bits therefore do not depend on the batch size,
and the row thresholds, which the caller passes, move only time. The caller
guarantees an SM90 device, where these lowerings were checked.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.layernorm.mhc import (
    _HC_MIX_BLOCK_K,
    _HC_MIX_DOT_PRECISION,
    _HC_MIX_NUM_STAGES,
    _HC_MIX_NUM_WARPS,
    _block_m_for,
    _hc_mix_stats_partial_kernel,
    _num_slices_for,
)

_WGMMA_MAX_MIX = 32
# Launch-only (like BLOCK_M); the fastest measured on H20 below the wgmma rows.
_V2_NUM_WARPS = 4
_V2_NUM_STAGES = 1


@triton.jit
def _tf32_rna(v):
    # The rounding Triton's tf32x3 lowering (F32DotTC) applies to the "big" halves.
    return tl.inline_asm_elementwise(
        "cvt.rna.tf32.f32 $0, $1;",
        "=r,r",
        [v],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _hc_mix_sq_tree64(x_ptr, offs_m, mask_m, x_stride_m, k0, BLOCK_M: tl.constexpr):
    """sum(x[rows, k0:k0 + 64]^2) with the reduction tree the production kernel's
    tl.sum lowers to on SM90 (x tile sizePerThread [1, 1], threadsPerWarp [1, 32],
    two warps along K): per 32-wide half, lane 0's butterfly whose first stage is
    contracted to fma(x_j, x_j, x_{j+16}^2), then offsets 8, 4, 2, 1, then the sum
    of the two halves. Every step below is one fp32 add of two values (a size-2
    sum), so the result does not depend on the layout Triton picks here."""
    h = tl.arange(0, 2)
    j = tl.arange(0, 16)
    ptrs = (
        x_ptr
        + offs_m[:, None, None] * x_stride_m
        + k0
        + h[None, :, None] * 32
        + j[None, None, :]
    )
    msk = mask_m[:, None, None]
    own = tl.load(ptrs, mask=msk, other=0.0).to(tl.float32)
    partner = tl.load(ptrs + 16, mask=msk, other=0.0).to(tl.float32)
    a = tl.fma(own, own, partner * partner)
    b = tl.sum(tl.reshape(a, (BLOCK_M, 2, 2, 8)), axis=2)
    c = tl.sum(tl.reshape(b, (BLOCK_M, 2, 2, 4)), axis=2)
    d = tl.sum(tl.reshape(c, (BLOCK_M, 2, 2, 2)), axis=2)
    e = tl.sum(d, axis=2)
    return tl.sum(e, axis=1)


@triton.jit
def _hc_mix_stats_partial_v2_kernel(
    x_ptr,
    w_ptr,
    part_mix_ptr,
    part_sq_ptr,
    M,
    K,
    x_stride_m,
    w_stride_n,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PER_SLICE: tl.constexpr,
):
    """``_hc_mix_stats_partial_kernel`` with its tf32x3 dot written out, for bf16 x.

    On SM8x/9x Triton lowers dot(a, b, acc, "tf32x3") to
        dot(aBig, bBig, nanz(dot(aBig, bSmall, dot(aSmall, bBig, 0)))) + acc
    with xBig = cvt.rna.tf32(x) and xSmall = x - xBig. x is loaded from bf16, which
    tf32 represents exactly, so aBig == x and aSmall == +0, and dot(+0, bBig, +0) is
    +0 -- the same accumulator the aBig*bSmall chain starts from. Each output element
    therefore sees the same mma.sync operands, chain order and rounding as the
    production kernel for finite x and W (|W| below the tf32 overflow threshold);
    non-finite inputs give non-finite outputs either way, possibly another inf/NaN.

    x and W load through a constexpr slice offset (vectorized; K_PER_SLICE ==
    K // NUM_SLICES, a BLOCK_K multiple), and the sum of squares is
    ``_hc_mix_sq_tree64``, the production reduction tree spelled out.
    """
    tl.static_assert(BLOCK_K == 64, "_hc_mix_sq_tree64 spells out a 64-wide tile")
    pid_m = tl.program_id(0)
    slice_id = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, MIX_PAD)
    mask_m = offs_m < M
    mask_n = offs_n < MIX
    k_per_slice = K // NUM_SLICES
    acc = tl.zeros([BLOCK_M, MIX_PAD], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for kb in tl.range(0, k_per_slice, BLOCK_K):
        offs_k = slice_id * K_PER_SLICE + kb + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * x_stride_m + offs_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        w_tile = tl.load(
            w_ptr + offs_n[None, :] * w_stride_n + offs_k[:, None],
            mask=mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        w_big = _tf32_rna(w_tile)
        w_small = w_tile - w_big
        d = tl.dot(x_tile, w_small, input_precision="tf32")
        d = tl.where(d != d, 0.0, d)
        d = tl.dot(x_tile, w_big, d, input_precision="tf32")
        acc = d + acc
        sq += _hc_mix_sq_tree64(
            x_ptr, offs_m, mask_m, x_stride_m, slice_id * K_PER_SLICE + kb, BLOCK_M
        )
    tl.store(
        part_mix_ptr + (slice_id * M + offs_m[:, None]) * MIX + offs_n[None, :],
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )
    tl.store(part_sq_ptr + slice_id * M + offs_m, sq, mask=mask_m)


def _wgmma_ok(x_flat: torch.Tensor, mix: int) -> bool:
    # The wgmma kernel reads x as int32 words of bf16 pairs.
    return (
        mix <= _WGMMA_MAX_MIX
        and x_flat.stride(0) % 2 == 0
        and x_flat.storage_offset() % 2 == 0
    )


def hc_mix_partial_sm90(
    x_flat: torch.Tensor,
    hc_fn: torch.Tensor,
    part_mix: torch.Tensor,
    part_sq: torch.Tensor,
    *,
    v2_min_rows: Optional[int],
    wgmma_min_rows: Optional[int],
) -> None:
    """Fill the split-K partials part_mix [S, M, MIX] and part_sq [S, M] of bf16
    x_flat [M, K] and fp32 hc_fn [MIX, K], S = ``mhc._num_slices_for(K)``. The v2 and
    wgmma kernels take the batches from their min rows on; None never takes them."""
    m, k = x_flat.shape
    mix = hc_fn.shape[0]
    num_slices = part_mix.shape[0]
    assert x_flat.dtype == torch.bfloat16 and hc_fn.dtype == torch.float32
    assert x_flat.stride(1) == 1 and hc_fn.stride(1) == 1
    assert num_slices == _num_slices_for(k)
    if m == 0:
        return
    exact_slices = k % (num_slices * _HC_MIX_BLOCK_K) == 0
    if (
        wgmma_min_rows is not None
        and m >= wgmma_min_rows
        and exact_slices
        and _wgmma_ok(x_flat, mix)
    ):
        from sglang.kernels.ops.layernorm.mhc_mix_wgmma import hc_mix_partial_wgmma

        hc_mix_partial_wgmma(x_flat, hc_fn, part_mix, part_sq, num_slices, slice_w=True)
        return
    mix_pad = max(16, triton.next_power_of_2(mix))
    block_m = _block_m_for(m)
    grid = (triton.cdiv(m, block_m), num_slices)
    if v2_min_rows is not None and m >= v2_min_rows and exact_slices:
        _hc_mix_stats_partial_v2_kernel[grid](
            x_flat,
            hc_fn,
            part_mix,
            part_sq,
            m,
            k,
            x_flat.stride(0),
            hc_fn.stride(0),
            MIX=mix,
            MIX_PAD=mix_pad,
            NUM_SLICES=num_slices,
            BLOCK_M=block_m,
            BLOCK_K=_HC_MIX_BLOCK_K,
            K_PER_SLICE=k // num_slices,
            num_warps=_V2_NUM_WARPS,
            num_stages=_V2_NUM_STAGES,
        )
        return
    _hc_mix_stats_partial_kernel[grid](
        x_flat,
        hc_fn,
        part_mix,
        part_sq,
        m,
        k,
        x_flat.stride(0),
        hc_fn.stride(0),
        MIX=mix,
        MIX_PAD=mix_pad,
        NUM_SLICES=num_slices,
        BLOCK_M=block_m,
        BLOCK_K=_HC_MIX_BLOCK_K,
        DOT_PRECISION=_HC_MIX_DOT_PRECISION,
        num_warps=_HC_MIX_NUM_WARPS,
        num_stages=_HC_MIX_NUM_STAGES,
    )


def hc_mix_stats_split_sinkhorn(
    x_flat: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_eps: float,
    *,
    v2_min_rows: Optional[int],
    wgmma_min_rows: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """hc_split_sinkhorn(hc_mix_stats(x_flat, hc_fn, rms_eps)[:, None], ...) bit for bit
    with the TileLang sinkhorn (SM90, HC=4, bf16 x_flat [M, K]). Returns pre [M, 4],
    post [M, 4] and comb [M, 4, 4] fp32."""
    from sglang.kernels.ops.layernorm.mhc_reduce_sinkhorn import (
        hc_mix_reduce_sinkhorn,
    )

    assert x_flat.dim() == 2 and hc_fn.dim() == 2
    m, k = x_flat.shape
    mix = hc_fn.shape[0]
    assert hc_mult == 4 and mix == (2 + hc_mult) * hc_mult and hc_fn.shape[1] == k
    num_slices = _num_slices_for(k)
    dev = x_flat.device
    part_mix = torch.empty((num_slices, m, mix), dtype=torch.float32, device=dev)
    part_sq = torch.empty((num_slices, m), dtype=torch.float32, device=dev)
    hc_mix_partial_sm90(
        x_flat,
        hc_fn,
        part_mix,
        part_sq,
        v2_min_rows=v2_min_rows,
        wgmma_min_rows=wgmma_min_rows,
    )
    return hc_mix_reduce_sinkhorn(
        part_mix,
        part_sq,
        hc_scale,
        hc_base,
        1.0 / k,
        rms_eps,
        sinkhorn_iters=sinkhorn_iters,
        hc_eps=hc_eps,
    )
