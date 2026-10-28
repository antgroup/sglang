"""Reduce + split-sinkhorn of hc_mix_stats in one SM90 kernel (Gluon), bitwise equal
to ``_hc_mix_stats_reduce_kernel`` (Triton) followed by the TileLang
``hc_split_sinkhorn_kernel`` (HC=4).

* The slice sums keep the production order (0 + p0 + p1 + ... per element) and its
  rms expression (fma, sqrt.approx, div.full).
* In the sinkhorn each comb row of 4 sits on 4 consecutive lanes and each column on
  lanes 4 apart, so gl.sum lowers to the shfl.bfly xor 2, 1 / xor 8, 4 trees of
  TileLang's AllReduce.
* exp, sigmoid and division are the PTX sequences nvcc emits for the TileLang
  kernel (TileLang 0.1.12, nvcc 13.0, no fast-math), written out inline.
* TileLang's 0 + x before each 4-element sum is skipped: every comb value is +0 or
  more (or NaN), where adding +0 changes no bits.
"""

from typing import Optional, Tuple

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _sigmoid_nvcc(t):
    # nvcc: 1 / (1 + expf(0 - t)); libdevice expf with its final multiply
    # contracted into the "+ 1" (fma.rn(ex2, 2^n, 1)), then rcp.rn.
    return gl.inline_asm_elementwise(
        """
        {
        .reg .f32 x, a, b, c, d, e, f, g;
        .reg .b32 n;
        mov.f32 a, 0f00000000;
        sub.f32 x, a, $1;
        fma.rn.f32 a, x, 0f3BBB989D, 0f3F000000;
        cvt.sat.f32.f32 a, a;
        fma.rm.f32 b, a, 0f437C0000, 0f4B400001;
        add.f32 c, b, 0fCB40007F;
        neg.f32 c, c;
        fma.rn.f32 d, x, 0f3FB8AA3B, c;
        fma.rn.f32 d, x, 0f32A57060, d;
        mov.b32 n, b;
        shl.b32 n, n, 23;
        mov.b32 e, n;
        ex2.approx.ftz.f32 f, d;
        fma.rn.f32 g, f, e, 0f3F800000;
        rcp.rn.f32 $0, g;
        }
        """,
        "=r,r",
        [t],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _expf_nvcc(x):
    # libdevice expf (non-fast-math). mul.rn keeps ptxas from contracting the final
    # multiply into the row-sum add that follows.
    return gl.inline_asm_elementwise(
        """
        {
        .reg .f32 a, b, c, d, e, f;
        .reg .b32 n;
        fma.rn.f32 a, $1, 0f3BBB989D, 0f3F000000;
        cvt.sat.f32.f32 a, a;
        fma.rm.f32 b, a, 0f437C0000, 0f4B400001;
        add.f32 c, b, 0fCB40007F;
        neg.f32 c, c;
        fma.rn.f32 d, $1, 0f3FB8AA3B, c;
        fma.rn.f32 d, $1, 0f32A57060, d;
        mov.b32 n, b;
        shl.b32 n, n, 23;
        mov.b32 e, n;
        ex2.approx.ftz.f32 f, d;
        mul.rn.f32 $0, f, e;
        }
        """,
        "=r,r",
        [x],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _div_rn(a, b):
    return gl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [a, b],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _hc_mix_reduce_sinkhorn_tl_kernel(
    part_mix_ptr,
    part_sq_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    mixes_ptr,
    m,
    inv_k,
    rms_eps,
    NUM_SLICES: gl.constexpr,
    ITERS: gl.constexpr,
    EPS: gl.constexpr,
    ROWS: gl.constexpr,
    STAGE: gl.constexpr,
    CHUNK: gl.constexpr,
    SLOTS: gl.constexpr,
    WRITE_MIXES: gl.constexpr,
    num_warps: gl.constexpr,
):
    MIX: gl.constexpr = 24
    Q: gl.constexpr = ROWS * 8
    # Reduction: one thread per (row, group of 4 mixes), 16-byte loads. Group g < 4
    # is comb row g, g = 4 / 5 are pre / post, g >= 6 idle; this order keeps both
    # 4-group halves aligned for the shared-memory slices below.
    LRED: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [num_warps, 1], [1, 0])
    LQ: gl.constexpr = gl.SliceLayout(1, LRED)
    # Sinkhorn: [row, j, k] with k on lane bits 0-1 and j on lane bits 2-3, so gl.sum
    # lowers to TileLang's AllReduce trees (shfl.bfly xor 2,1 for rows, xor 8,4 for
    # columns). pp holds pre (j=0) and post (j=1).
    L3: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1], [2, 4, 4], [num_warps, 1, 1], [2, 1, 0]
    )
    FLAT2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    FLAT1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    row0 = gl.program_id(0) * ROWS
    q1 = gl.arange(0, Q, layout=LQ)
    c1 = gl.arange(0, 4, layout=gl.SliceLayout(0, LRED))
    g1 = q1 % 8
    rq = row0 + q1 // 8
    red_off = (
        rq[:, None] * MIX + gl.where(g1 < 4, g1 + 2, g1 - 4)[:, None] * 4 + c1[None, :]
    )
    red_mask = (rq < m)[:, None] & (g1 < 6)[:, None] & (c1 < 4)[None, :]

    acc = gl.zeros([Q, 4], gl.float32, LRED)
    sq = gl.zeros([Q], gl.float32, LQ)
    if STAGE:
        # Small M: spread the slices over all threads so every load is in flight at
        # once, then sum each element in slice order from shared memory.
        smem = gl.allocate_shared_memory(gl.float32, [SLOTS * Q, 4], FLAT2)
        smem_sq = gl.allocate_shared_memory(gl.float32, [SLOTS * Q], FLAT1)
        t1 = gl.arange(0, CHUNK * Q, layout=LQ)
        tg = t1 % 8
        tr = row0 + (t1 // 8) % ROWS
        ts = t1 // Q
        t_off = (
            ts[:, None] * (m * MIX)
            + tr[:, None] * MIX
            + gl.where(tg < 4, tg + 2, tg - 4)[:, None] * 4
            + c1[None, :]
        )
        t_mask = (tr < m)[:, None] & (tg < 6)[:, None] & (c1 < 4)[None, :]
        for c in gl.static_range(NUM_SLICES // CHUNK):
            tile = gl.load(
                part_mix_ptr + c * CHUNK * m * MIX + t_off, mask=t_mask, other=0.0
            )
            smem.slice(c * CHUNK * Q, CHUNK * Q).store(tile)
            sqt = gl.load(
                part_sq_ptr + (c * CHUNK + ts) * m + tr, mask=tr < m, other=0.0
            )
            smem_sq.slice(c * CHUNK * Q, CHUNK * Q).store(sqt)
        gl.barrier()
        for s in gl.static_range(NUM_SLICES):
            acc += smem.slice(s * Q, Q).load(LRED)
            sq += smem_sq.slice(s * Q, Q).load(LQ)
    else:
        for s in gl.static_range(NUM_SLICES):
            acc += gl.load(
                part_mix_ptr + s * m * MIX + red_off, mask=red_mask, other=0.0
            )
            sq += gl.load(part_sq_ptr + s * m + rq, mask=rq < m, other=0.0)
    # Same expression as _hc_mix_stats_reduce_kernel (fma, sqrt.approx, div.full).
    rsqrt = 1.0 / gl.sqrt(sq * inv_k + rms_eps)
    mix = acc * rsqrt[:, None]
    if WRITE_MIXES:
        gl.store(mixes_ptr + red_off, mix, mask=red_mask)
    smem_mix = gl.allocate_shared_memory(gl.float32, [Q, 4], FLAT2)
    smem_mix.store(mix)
    gl.barrier()
    comb = smem_mix.reshape([ROWS, 8, 4]).slice(0, 4, dim=1).load(L3)
    pp = smem_mix.reshape([ROWS, 8, 4]).slice(4, 4, dim=1).load(L3)

    LR: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, L3))
    row = row0 + gl.arange(0, ROWS, layout=LR)
    j1 = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(2, L3)))
    k1 = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, L3)))
    r3 = row[:, None, None]
    j3 = j1[None, :, None]
    k3 = k1[None, None, :]
    rmask = r3 < m
    jk = j3 * 4 + k3

    s0 = gl.load(scale_ptr)
    s1 = gl.load(scale_ptr + 1)
    s2 = gl.load(scale_ptr + 2)
    pp_base = gl.load(base_ptr + jk, mask=j3 < 2, other=0.0)
    c_base = gl.load(base_ptr + 8 + jk)

    sig = _sigmoid_nvcc(gl.fma(pp, gl.where(j3 == 0, s0, s1), pp_base))
    pp = gl.where(j3 == 0, sig + EPS, sig + sig)
    out_off = r3 * 4 + j3 * 0 + k3
    gl.store(pre_ptr + out_off, pp, mask=rmask & (j3 == 0))
    gl.store(post_ptr + out_off, pp, mask=rmask & (j3 == 1))

    comb = gl.fma(comb, s2, c_base)
    row_max = gl.max(gl.maximum(comb, float("-inf")), axis=2)
    comb = _expf_nvcc(comb - row_max[:, :, None])
    comb = _div_rn(comb, gl.sum(comb, axis=2)[:, :, None]) + EPS
    comb = _div_rn(comb, gl.sum(comb, axis=1)[:, None, :] + EPS)
    for _ in gl.static_range(ITERS - 1):
        comb = _div_rn(comb, gl.sum(comb, axis=2)[:, :, None] + EPS)
        comb = _div_rn(comb, gl.sum(comb, axis=1)[:, None, :] + EPS)
    gl.store(comb_ptr + r3 * 16 + jk, comb, mask=rmask & (jk < 16))


def _launch_config(m: int):
    """(ROWS, num_warps, STAGE); launch-only, a row's arithmetic does not depend on it.
    Measured on H20: staging the partials in shared memory pays off for small grids."""
    if m <= 128:
        return 2, 4, True
    if m <= 512:
        return 2, 1, True
    if m <= 2048:
        return 2, 1, False
    return 4, 1, False


def hc_mix_reduce_sinkhorn(
    part_mix: torch.Tensor,
    part_sq: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    inv_k: float,
    rms_eps: float,
    sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
    mixes: Optional[torch.Tensor] = None,
    config: Optional[Tuple[int, int, bool]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """pre / post [M, 4] and comb [M, 4, 4] from the split-K partials [S, M, 24] and
    [S, M]; ``mixes`` [M, 24], if given, receives the reduced mixes."""
    num_slices, m, mix = part_mix.shape
    assert mix == 24 and part_sq.shape == (num_slices, m)
    assert part_mix.is_contiguous() and part_sq.is_contiguous()
    assert hc_scale.dtype == hc_base.dtype == torch.float32
    dev = part_mix.device
    pre = torch.empty(m, 4, dtype=torch.float32, device=dev)
    post = torch.empty(m, 4, dtype=torch.float32, device=dev)
    comb = torch.empty(m, 4, 4, dtype=torch.float32, device=dev)
    if m == 0:
        return pre, post, comb
    rows, warps, stage = config or _launch_config(m)
    stage = stage and num_slices % 16 == 0
    _hc_mix_reduce_sinkhorn_tl_kernel[(triton.cdiv(m, rows),)](
        part_mix,
        part_sq,
        hc_scale.contiguous(),
        hc_base.contiguous(),
        pre,
        post,
        comb,
        mixes if mixes is not None else pre,
        m,
        inv_k,
        rms_eps,
        NUM_SLICES=num_slices,
        ITERS=sinkhorn_iters,
        EPS=hc_eps,
        ROWS=rows,
        STAGE=stage,
        CHUNK=16,
        SLOTS=triton.next_power_of_2(num_slices),
        WRITE_MIXES=mixes is not None,
        num_warps=warps,
    )
    return pre, post, comb
