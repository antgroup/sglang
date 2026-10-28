"""Keys-major Hopper prefill index scoring with explicit async wgmma (Gluon).

Computes the same scores as ``_fp8_index_logits_prefill_tiled_kernel`` in
``sm90_fp8_prefill``, bitwise:
every (row, key) goes through the same exact 32-wide partial dots, FP32 adds in
the same order and the same BF16 rounding points. Only the schedule differs:

* A CTA keeps all heads of 32 query rows in shared memory and streams 128-key
  tiles; the key tile is the wgmma A operand (M = 128 keys over two warpgroups),
  each head's queries the B operand (N = 32 rows).
* wgmmas are asynchronous and double-buffered across heads, so the epilogue of
  head h overlaps the MMAs of head h + 1 (Triton waits after every dot).
* The BF16 roundings run on bf16x2 pairs (``cvt.rn.relu.bf16x2.f32``,
  ``mul.rn.bf16x2``) instead of scalar F2F conversions, which saturate the XU
  pipe on H20 and bound the tiled kernel at ~55% of the FP8 tensor peak.
* Persistent CTAs split the visible (row block, key tile) work evenly.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from sglang.kernels.ops.attention.dsv4.fp4_indexer import INDEX_HEAD_DIM

# wgmma N (query rows per CTA). All heads of the rows stay in shared memory:
# 32 heads * 32 rows * 128 B = 128 KiB next to STAGES * 16 KiB of keys.
BLOCK_ROWS = 32
# wgmma M over two warpgroups.
BLOCK_KEYS = 128
STAGES = 3
NUM_WARPS = 8
MAX_HEADS = 32


@triton.jit(do_not_specialize=["B", "L", "n_row_blocks"])
def _visible_tile_prefix_kernel(
    lens_ptr,
    prefix_ptr,
    B,
    L,
    n_row_blocks,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """prefix[rb] = key tiles visible to some row of the row blocks before rb."""
    offs_c = tl.arange(0, CHUNK)
    offs_r = tl.arange(0, BLOCK_R)
    tl.store(prefix_ptr, 0)
    carry = tl.zeros([], dtype=tl.int32)
    for c0 in range(0, n_row_blocks, CHUNK):
        rbs = c0 + offs_c
        rows = rbs[:, None] * BLOCK_R + offs_r[None, :]
        lens = tl.load(lens_ptr + rows, mask=rows < B, other=0)
        vis = tl.maximum(tl.minimum(tl.max(lens, axis=1), L), 0)
        tiles = ((vis + BLOCK_K - 1) // BLOCK_K).to(tl.int32)
        tiles = tl.where(rbs < n_row_blocks, tiles, 0)
        tl.store(
            prefix_ptr + 1 + rbs,
            tl.cumsum(tiles, axis=0) + carry,
            mask=rbs < n_row_blocks,
        )
        carry += tl.sum(tiles, axis=0)


@gluon.jit
def _last_row_block_at(
    prefix_ptr, n_row_blocks, v, n_key_tiles, INVISIBLE: gl.constexpr
):
    """Largest row block whose first tile index is <= v.

    Visible tiles of row block rb start at prefix[rb]; invisible ones at
    rb * n_key_tiles - prefix[rb]. Both are non-decreasing in rb.
    """
    lo = 0
    hi = n_row_blocks
    while hi - lo > 1:
        mid = (lo + hi) // 2
        start = gl.load(prefix_ptr + mid)
        if INVISIBLE:
            start = mid * n_key_tiles - start
        right = start <= v
        lo = gl.where(right, mid, lo)
        hi = gl.where(right, hi, mid)
    return lo


@gluon.jit
def _issue_head(k_tile, q_head, c0, c1, c2, c3):
    # Four independent 32-wide dots, each exact for FP4-valued operands.
    # use_acc=False on recycled registers: no non-wgmma instruction writes an
    # accumulator while other wgmmas are in flight (ptxas would serialize).
    p0 = warpgroup_mma(
        k_tile.slice(0, 32, dim=1),
        q_head.slice(0, 32, dim=1).permute((1, 0)),
        c0,
        use_acc=False,
        is_async=True,
    )
    p1 = warpgroup_mma(
        k_tile.slice(32, 32, dim=1),
        q_head.slice(32, 32, dim=1).permute((1, 0)),
        c1,
        use_acc=False,
        is_async=True,
    )
    p2 = warpgroup_mma(
        k_tile.slice(64, 32, dim=1),
        q_head.slice(64, 32, dim=1).permute((1, 0)),
        c2,
        use_acc=False,
        is_async=True,
    )
    p3 = warpgroup_mma(
        k_tile.slice(96, 32, dim=1),
        q_head.slice(96, 32, dim=1).permute((1, 0)),
        c3,
        use_acc=False,
        is_async=True,
    )
    return p0, p1, p2, p3


@gluon.jit
def _head_epilogue(acc, p0, p1, p2, p3, w):
    # acc += rn_bf16(relu(rn_bf16(((p0 + p1) + p2) + p3)) * w) as in the tiled
    # kernel. cvt.rn.relu.bf16x2 and mul.rn.bf16x2 match its FP32 max / mul /
    # cvt for every input bit pattern (NaN stays NaN): bf16 * bf16 is exact in
    # FP32, so both round the exact product once.
    s = ((p0 + p1) + p2) + p3
    return gl.inline_asm_elementwise(
        """
        {
        .reg .b32 t;
        cvt.rn.relu.bf16x2.f32 t, $3, $2;
        mul.rn.bf16x2 t, t, $4;
        shl.b32 $0, t, 16;
        and.b32 $1, t, 0xffff0000;
        add.f32 $0, $5, $0;
        add.f32 $1, $6, $1;
        }
        """,
        "=r,=r,r,r,r,r,r",
        [s, w[None, :], acc],
        dtype=gl.float32,
        is_pure=True,
        pack=2,
    )


@gluon.jit
def _load_row_block(
    q_desc,
    q_bar,
    q_smem,
    w_ptr,
    w_smem,
    lens_ptr,
    B,
    L,
    stride_wb,
    rb,
    q_phase,
    offs_r0,
    offs_w,
    H: gl.constexpr,
    D: gl.constexpr,
    BLOCK_R: gl.constexpr,
):
    """Queries and weights of row block rb into smem; returns visible lengths."""
    offs_r = rb * BLOCK_R + offs_r0
    n_vis = gl.minimum(gl.load(lens_ptr + offs_r, mask=offs_r < B, other=0), L)
    offs_wr = rb * BLOCK_R + offs_w % BLOCK_R
    w_blk = gl.load(
        w_ptr + offs_wr.to(gl.int64) * stride_wb + offs_w // BLOCK_R,
        mask=offs_wr < B,
        other=0.0,
    )
    w_smem.store(w_blk)
    mbarrier.expect(q_bar, H * BLOCK_R * D)
    for h in gl.static_range(H):
        tma.async_copy_global_to_shared(
            q_desc, [rb * BLOCK_R, h * D], q_bar, q_smem.index(h)
        )
    mbarrier.wait(q_bar, q_phase)
    return n_vis


@gluon.jit
def _score_tile(
    k_tile,
    q_smem,
    w_smem,
    hz,
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    b2,
    b3,
    H: gl.constexpr,
    BLOCK_R: gl.constexpr,
    row_layout: gl.constexpr,
):
    """Scores of one key tile summed over heads; a* / b* are recycled buffers."""
    acc = gl.zeros(a0.shape, gl.float32, layout=a0.type.layout)
    # Fully unrolled and double-buffered over heads: the epilogue of head h
    # overlaps the wgmmas of head h + 1. A runtime head loop copies in-flight
    # accumulators through phis and ptxas serializes every wgmma.
    a0, a1, a2, a3 = _issue_head(k_tile, q_smem.index(hz), a0, a1, a2, a3)
    for h in gl.static_range(0, H - 2, 2):
        b0, b1, b2, b3 = _issue_head(k_tile, q_smem.index(hz + h + 1), b0, b1, b2, b3)
        w = w_smem.slice(h * BLOCK_R, BLOCK_R).load(row_layout)
        a0, a1, a2, a3 = warpgroup_mma_wait(4, deps=[a0, a1, a2, a3])
        acc = _head_epilogue(acc, a0, a1, a2, a3, w)
        a0, a1, a2, a3 = _issue_head(k_tile, q_smem.index(hz + h + 2), a0, a1, a2, a3)
        w = w_smem.slice((h + 1) * BLOCK_R, BLOCK_R).load(row_layout)
        b0, b1, b2, b3 = warpgroup_mma_wait(4, deps=[b0, b1, b2, b3])
        acc = _head_epilogue(acc, b0, b1, b2, b3, w)
    b0, b1, b2, b3 = _issue_head(k_tile, q_smem.index(hz + H - 1), b0, b1, b2, b3)
    w = w_smem.slice((H - 2) * BLOCK_R, BLOCK_R).load(row_layout)
    a0, a1, a2, a3 = warpgroup_mma_wait(4, deps=[a0, a1, a2, a3])
    acc = _head_epilogue(acc, a0, a1, a2, a3, w)
    w = w_smem.slice((H - 1) * BLOCK_R, BLOCK_R).load(row_layout)
    b0, b1, b2, b3 = warpgroup_mma_wait(0, deps=[b0, b1, b2, b3])
    acc = _head_epilogue(acc, b0, b1, b2, b3, w)
    return acc, a0, a1, a2, a3, b0, b1, b2, b3


@gluon.jit
def _prefetch_keys(
    k_desc,
    k_bar,
    k_smem,
    prefix_ptr,
    n_row_blocks,
    n_key_tiles,
    v,
    stage,
    p_lo,
    p_hi,
    BLOCK_K: gl.constexpr,
    D: gl.constexpr,
):
    """TMA the key tile of visible tile v; (p_lo, p_hi) caches its row block."""
    if v >= p_hi:
        rb = _last_row_block_at(prefix_ptr, n_row_blocks, v, n_key_tiles, False)
        p_lo = gl.load(prefix_ptr + rb)
        p_hi = gl.load(prefix_ptr + rb + 1)
    mbarrier.expect(k_bar.index(stage), BLOCK_K * D)
    tma.async_copy_global_to_shared(
        k_desc, [(v - p_lo) * BLOCK_K, 0], k_bar.index(stage), k_smem.index(stage)
    )
    return p_lo, p_hi


@gluon.jit
def _fill_invisible_tiles(
    prefix_ptr,
    out_ptr,
    B,
    OUT_L,
    n_row_blocks,
    n_key_tiles,
    n_vis_tiles,
    BLOCK_R: gl.constexpr,
    BLOCK_K: gl.constexpr,
    num_warps: gl.constexpr,
):
    """-inf for the tiles no row of their block can see, split across CTAs."""
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [num_warps, 1], [1, 0])
    pid = gl.program_id(0)
    grid = gl.num_programs(0)
    n_inv = n_row_blocks * n_key_tiles - n_vis_tiles
    u0 = (pid.to(gl.int64) * n_inv // grid).to(gl.int32)
    u1 = ((pid.to(gl.int64) + 1) * n_inv // grid).to(gl.int32)
    offs_r0 = gl.arange(0, BLOCK_R, layout=gl.SliceLayout(1, layout))
    offs_k0 = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, layout))
    neg_inf = gl.full([BLOCK_R, BLOCK_K], float("-inf"), gl.float32, layout=layout)
    rb = 0
    lo = 0
    hi = 0
    first_kt = 0
    for u in range(u0, u1):
        if u >= hi:
            rb = _last_row_block_at(prefix_ptr, n_row_blocks, u, n_key_tiles, True)
            vis_lo = gl.load(prefix_ptr + rb)
            first_kt = gl.load(prefix_ptr + rb + 1) - vis_lo
            lo = rb * n_key_tiles - vis_lo
            hi = lo + n_key_tiles - first_kt
        offs_r = rb * BLOCK_R + offs_r0
        offs_k = (first_kt + u - lo) * BLOCK_K + offs_k0
        ptrs = out_ptr + offs_r.to(gl.int64)[:, None] * OUT_L + offs_k[None, :]
        gl.store(ptrs, neg_inf, mask=(offs_r < B)[:, None] & (offs_k < OUT_L)[None, :])


# These sizes are runtime loop bounds, including 1-row CP tails. Specializing
# their value/alignment adds binaries without changing the tile layout. Keep
# OUT_L and pointer alignment specialization for vectorized loads/stores.
@gluon.jit(do_not_specialize=["B", "L", "n_row_blocks", "n_key_tiles"])
def _keys_major_scores_kernel(
    q_desc,  # [B, H * D] e4m3, block [BLOCK_R, D]
    k_desc,  # [L, D] e4m3, block [BLOCK_K, D]
    w_ptr,  # [B, H] bf16
    lens_ptr,  # [B] int64
    prefix_ptr,  # [n_row_blocks + 1] int32
    out_ptr,  # [B, OUT_L] fp32
    B,
    L,
    OUT_L,
    stride_wb,
    n_row_blocks,
    n_key_tiles,
    H: gl.constexpr,
    D: gl.constexpr,
    BLOCK_R: gl.constexpr,
    BLOCK_K: gl.constexpr,
    STAGES: gl.constexpr,
    num_warps: gl.constexpr,
):
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, BLOCK_R, 32]
    )
    key_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)
    row_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
    w_layout: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])

    q_smem = gl.allocate_shared_memory(gl.float8e4nv, [H, BLOCK_R, D], q_desc.layout)
    k_smem = gl.allocate_shared_memory(
        gl.float8e4nv, [STAGES, BLOCK_K, D], k_desc.layout
    )
    # [H, BLOCK_R] flattened: per-head slices need constexpr offsets.
    w_smem = gl.allocate_shared_memory(
        gl.bfloat16, [H * BLOCK_R], gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    )
    q_bar = mbarrier.allocate_mbarrier()
    k_bar = mbarrier.allocate_mbarrier(batch=STAGES)
    mbarrier.init(q_bar, count=1)
    for s in gl.static_range(STAGES):
        mbarrier.init(k_bar.index(s), count=1)

    pid = gl.program_id(0)
    grid = gl.num_programs(0)
    n_vis_tiles = gl.load(prefix_ptr + n_row_blocks)
    v0 = (pid.to(gl.int64) * n_vis_tiles // grid).to(gl.int32)
    v1 = ((pid.to(gl.int64) + 1) * n_vis_tiles // grid).to(gl.int32)
    p_lo = 0
    p_hi = 0
    for s in gl.static_range(STAGES):
        if v0 + s < v1:
            p_lo, p_hi = _prefetch_keys(
                k_desc,
                k_bar,
                k_smem,
                prefix_ptr,
                n_row_blocks,
                n_key_tiles,
                v0 + s,
                s,
                p_lo,
                p_hi,
                BLOCK_K,
                D,
            )

    a0 = gl.zeros([BLOCK_K, BLOCK_R], gl.float32, layout=mma_layout)
    a1 = a0
    a2 = a0
    a3 = a0
    b0 = a0
    b1 = a0
    b2 = a0
    b3 = a0
    offs_r0 = gl.arange(0, BLOCK_R, layout=row_layout)
    offs_k0 = gl.arange(0, BLOCK_K, layout=key_layout)
    offs_w = gl.arange(0, H * BLOCK_R, layout=w_layout)
    rb = 0
    lo = 0
    hi = 0
    q_phase = 0
    n_vis = gl.zeros([BLOCK_R], gl.int64, layout=row_layout)
    for v in range(v0, v1):
        if v >= hi:
            # New row block: queries, weights and visible lengths. Every wgmma
            # reading q_smem / w_smem has retired (wait(0) + barrier per tile).
            rb = _last_row_block_at(prefix_ptr, n_row_blocks, v, n_key_tiles, False)
            lo = gl.load(prefix_ptr + rb)
            hi = gl.load(prefix_ptr + rb + 1)
            n_vis = _load_row_block(
                q_desc,
                q_bar,
                q_smem,
                w_ptr,
                w_smem,
                lens_ptr,
                B,
                L,
                stride_wb,
                rb,
                q_phase,
                offs_r0,
                offs_w,
                H,
                D,
                BLOCK_R,
            )
            q_phase ^= 1
            gl.barrier()

        j = v - v0
        stage = j % STAGES
        mbarrier.wait(k_bar.index(stage), (j // STAGES) & 1)
        k_tile = k_smem.index(stage)
        # Opaque per-tile zero: keeps the 4 * H q descriptors from being
        # hoisted out of the tile loop, where they spill.
        hz = gl.minimum(v, 0)
        acc, a0, a1, a2, a3, b0, b1, b2, b3 = _score_tile(
            k_tile,
            q_smem,
            w_smem,
            hz,
            a0,
            a1,
            a2,
            a3,
            b0,
            b1,
            b2,
            b3,
            H,
            BLOCK_R,
            row_layout,
        )
        # Every warp is done with this K stage (and q / w) before refills.
        gl.barrier()
        if v + STAGES < v1:
            p_lo, p_hi = _prefetch_keys(
                k_desc,
                k_bar,
                k_smem,
                prefix_ptr,
                n_row_blocks,
                n_key_tiles,
                v + STAGES,
                stage,
                p_lo,
                p_hi,
                BLOCK_K,
                D,
            )
        offs_r = rb * BLOCK_R + offs_r0
        offs_k = (v - lo) * BLOCK_K + offs_k0
        logit = acc.to(gl.bfloat16).to(gl.float32)
        logit = gl.where(offs_k[:, None] < n_vis[None, :], logit, float("-inf"))
        ptrs = out_ptr + offs_r.to(gl.int64)[None, :] * OUT_L + offs_k[:, None]
        gl.store(ptrs, logit, mask=(offs_r < B)[None, :] & (offs_k < OUT_L)[:, None])

    mbarrier.invalidate(q_bar)
    for s in gl.static_range(STAGES):
        mbarrier.invalidate(k_bar.index(s))
    _fill_invisible_tiles(
        prefix_ptr,
        out_ptr,
        B,
        OUT_L,
        n_row_blocks,
        n_key_tiles,
        n_vis_tiles,
        BLOCK_R,
        BLOCK_K,
        num_warps,
    )


_num_sms = {}


def keys_major_scores_supported(q: torch.Tensor, keys: torch.Tensor) -> bool:
    """Shape / TMA alignment contract of :func:`keys_major_scores`."""
    _, heads, dim = q.shape
    return (
        dim == INDEX_HEAD_DIM
        and heads % 2 == 0
        and 2 <= heads <= MAX_HEADS
        and q.is_contiguous()
        and keys.is_contiguous()
        and q.data_ptr() % 16 == 0
        and keys.data_ptr() % 16 == 0
    )


def keys_major_scores(
    q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    lens: torch.Tensor,
    out: torch.Tensor,
    *,
    grid_size: int = 0,
) -> None:
    """Write exact scores into ``out`` [rows, width] (row stride >= width).

    q [rows, H, 128] e4m3, weights [rows, H] bf16 (contiguous), keys
    [width, 128] e4m3, lens [rows] int64. Columns from width up to the row
    stride are written as -inf, like the tiled kernel. ``grid_size`` 0 means
    one persistent CTA per SM.
    """
    rows, heads, dim = q.shape
    width = keys.shape[0]
    out_width = out.stride(0)
    n_row_blocks = triton.cdiv(rows, BLOCK_ROWS)
    n_key_tiles = triton.cdiv(out_width, BLOCK_KEYS)
    prefix = torch.empty(n_row_blocks + 1, dtype=torch.int32, device=q.device)
    _visible_tile_prefix_kernel[(1,)](
        lens,
        prefix,
        rows,
        width,
        n_row_blocks,
        BLOCK_R=BLOCK_ROWS,
        BLOCK_K=BLOCK_KEYS,
        CHUNK=256,
    )
    q_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_ROWS, dim], gl.float8e4nv)
    k_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_KEYS, dim], gl.float8e4nv)
    q_desc = TensorDescriptor.from_tensor(
        q.view(rows, heads * dim), [BLOCK_ROWS, dim], q_layout
    )
    k_desc = TensorDescriptor.from_tensor(keys, [BLOCK_KEYS, dim], k_layout)
    if grid_size <= 0:
        device = q.device.index
        if device not in _num_sms:
            _num_sms[device] = torch.cuda.get_device_properties(
                device
            ).multi_processor_count
        grid_size = _num_sms[device]
    _keys_major_scores_kernel[(min(n_row_blocks * n_key_tiles, grid_size),)](
        q_desc,
        k_desc,
        weights,
        lens,
        prefix,
        out,
        rows,
        width,
        out_width,
        weights.stride(0),
        n_row_blocks,
        n_key_tiles,
        H=heads,
        D=dim,
        BLOCK_R=BLOCK_ROWS,
        BLOCK_K=BLOCK_KEYS,
        STAGES=STAGES,
        num_warps=NUM_WARPS,
    )


def prewarm_keys_major_scores(device: torch.device, heads: int) -> None:
    """Load the serving variants using small, context-independent allocations.

    The dispatcher uses contiguous FP8 queries/keys, BF16 weights and FP32
    output with a row stride rounded up to four elements. The two remaining
    alignment choices are the output row stride (divisible by 16 or not) and
    an int64 length-vector slice (16- or 8-byte aligned). No context length or
    chunk-size buckets are needed. This bypasses the dispatch size threshold
    deliberately; allocating a long-context score matrix at startup would skew
    the KV budget and unnecessarily increase the load-time peak.
    """
    rows, width = BLOCK_ROWS + 1, BLOCK_KEYS + 1
    q = torch.zeros(
        (rows, heads, INDEX_HEAD_DIM), device=device, dtype=torch.float8_e4m3fn
    )
    keys = torch.zeros(
        (width, INDEX_HEAD_DIM), device=device, dtype=torch.float8_e4m3fn
    )
    if not keys_major_scores_supported(q, keys):
        return
    weights = torch.ones((rows, heads), device=device, dtype=torch.bfloat16)
    lengths = torch.full((rows + 1,), width, device=device, dtype=torch.int64)
    for out_width in (triton.cdiv(width, 16) * 16, triton.cdiv(width, 4) * 4):
        out = torch.empty((rows, out_width), device=device, dtype=torch.float32)
        for offset in (0, 1):
            keys_major_scores(q, weights, keys, lengths[offset : offset + rows], out)
    # Complete the first device load as well as compilation before serving.
    torch.cuda.synchronize(device)
