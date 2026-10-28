"""Hopper wgmma scoring for the decode/verify FP4 indexer (Gluon).

Same logits as ``sm90_fp4_indexer`` with one difference in where the work
runs: keys are the wgmma M dimension, (row, head) pairs the N dimension, so a
request's DSpark verify rows share each decoded key tile and the MMA issues
from shared memory instead of per-row register fragments.

Numerics. The epilogue keeps the per-row kernel's rounding points (bf16 dot,
relu, bf16 weighted score, fp32 head sum, bf16 logit) and its exact head-sum
tree: ``S_g = ((h_g + h_{g+8}) + h_{g+16}) + h_{g+24}``, then pairs over the
bits 2, 1, 0 of g. The dot accumulates the 128 products in natural k order,
which differs from the per-row kernel's register-operand order. Results are
bit-identical whenever every fp32 partial sum is exact, which holds when the
combined block-scale exponents of a query/key pair (four 32-wide blocks) span
at most 2^9; wider spans may round differently.
"""

import torch
import triton
import triton.language as tl
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

# wgmma M: keys per tile (one warpgroup).
BLOCK_K = gl.constexpr(64)
# Heads per query row; a 64-wide wgmma N holds two rows.
HEADS = 32
# Bounded-grid CTA targets; measured on H20 (32 requests x 6 rows, 61K-69K
# context). Per-row candidate programs are lighter; 2048-8192 are within 3%.
_GRID_CTAS = 2048
_CANDIDATE_GRID_CTAS = 4096
# 168 registers keep three CTAs (64 KiB shared memory each) resident per SM;
# measured on H20, the few spills cost less than dropping to two CTAs.
_MAX_REGS = 168


@gluon.jit
def _relu_weight(acc, w):
    # rn_bf16(relu(rn_bf16(acc)) * w) as fp32 on bf16x2 pairs; bf16 * bf16 is
    # exact in fp32, so the paired multiply rounds the same product once.
    return gl.inline_asm_elementwise(
        """
        {
        .reg .b32 t, z;
        mov.b32 z, 0;
        cvt.rn.bf16x2.f32 t, $3, $2;
        max.bf16x2 t, t, z;
        mul.rn.bf16x2 t, t, $4;
        shl.b32 $0, t, 16;
        and.b32 $1, t, 0xffff0000;
        }
        """,
        "=r,=r,r,r,r",
        [acc, w],
        dtype=gl.float32,
        is_pure=True,
        pack=2,
    )


@gluon.jit
def _head_sum(s):
    # s: [keys, 64], column 32 * row + 8a + 2q + e holds head 8a + g, g = 2q + e.
    # Reproduces the per-row kernel's reduction (in-thread over a, then
    # butterflies over g bits 2, 1, 0); each size-2 sum is one commutative add.
    x = gl.reshape(s, [BLOCK_K, 2, 2, 2, 4, 2])  # key, row, a1, a0, q, e
    x = gl.permute(x, [0, 1, 2, 4, 5, 3])
    x0, x1 = gl.split(x)  # a0 = 0 / 1
    x0 = gl.permute(x0, [0, 1, 3, 4, 2])
    x1 = gl.permute(x1, [0, 1, 3, 4, 2])
    h0, h2 = gl.split(x0)
    h1, h3 = gl.split(x1)
    partial = ((h0 + h1) + h2) + h3  # [keys, row, q, e]
    even, odd = gl.split(partial)
    even = gl.sum(gl.sum(gl.reshape(even, [BLOCK_K, 2, 2, 2]), axis=2), axis=2)
    odd = gl.sum(gl.sum(gl.reshape(odd, [BLOCK_K, 2, 2, 2]), axis=2), axis=2)
    return even + odd  # [keys, row]


@gluon.jit
def _row_head_sum(s):
    # One row's [keys, 32] heads; same tree as _head_sum.
    x = gl.reshape(s, [BLOCK_K, 2, 2, 4, 2])  # key, a1, a0, q, e
    x = gl.permute(x, [0, 1, 3, 4, 2])
    x0, x1 = gl.split(x)
    x0 = gl.permute(x0, [0, 2, 3, 1])
    x1 = gl.permute(x1, [0, 2, 3, 1])
    h0, h2 = gl.split(x0)
    h1, h3 = gl.split(x1)
    partial = ((h0 + h1) + h2) + h3  # [keys, q, e]
    even, odd = gl.split(partial)
    even = gl.sum(gl.sum(gl.reshape(even, [BLOCK_K, 2, 2]), axis=1), axis=1)
    odd = gl.sum(gl.sum(gl.reshape(odd, [BLOCK_K, 2, 2]), axis=1), axis=1)
    return even + odd


@gluon.jit
def _max_nan(a, b):
    return gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@gluon.jit
def _tile_slots(
    slots_ptr,
    req,
    tile,
    visible,
    stride_slots,
    RATIO: gl.constexpr,
    key_l: gl.constexpr,
):
    keys = tile * BLOCK_K + gl.arange(0, BLOCK_K, layout=key_l)
    valid = keys < visible
    slot = gl.load(slots_ptr + req * stride_slots + keys * RATIO, mask=valid, other=0)
    return slot.to(gl.int64) // RATIO, valid


@gluon.jit
def _candidate_slots(
    slots_ptr,
    blocks_ptr,
    req,
    tile,
    count,
    stride_slots,
    COMPRESS_RATIO: gl.constexpr,
    CANDIDATE_BLOCK_SIZE: gl.constexpr,
    key_l: gl.constexpr,
):
    # Compact column c holds logical position blocks[c // size] * size + c % size.
    cols = tile * BLOCK_K + gl.arange(0, BLOCK_K, layout=key_l)
    valid = cols < count
    block = gl.load(blocks_ptr + cols // CANDIDATE_BLOCK_SIZE, mask=valid, other=0)
    logical = block.to(gl.int64) * CANDIDATE_BLOCK_SIZE + cols % CANDIDATE_BLOCK_SIZE
    slot = gl.load(
        slots_ptr + req * stride_slots + logical * COMPRESS_RATIO, mask=valid, other=0
    )
    return slot.to(gl.int64) // COMPRESS_RATIO, valid


@gluon.jit
def _tile_payload(table_ptr, slot, valid, page_size, row_stride, L3: gl.constexpr):
    # [keys, 4 scale blocks, 16 bytes] e2m1 payload and [keys, 4] e8m0 scales.
    page_base = (slot // page_size) * row_stride
    off = slot % page_size
    blk = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(2, L3)))
    byte = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, L3)))
    cols = gl.expand_dims(gl.expand_dims(blk * 16, 0), 2) + gl.expand_dims(
        gl.expand_dims(byte, 0), 1
    )
    row = gl.expand_dims(gl.expand_dims(page_base + off * 64, 1), 2)
    payload = gl.load(
        table_ptr + row + cols,
        mask=gl.expand_dims(gl.expand_dims(valid, 1), 2),
        other=0,
    )
    scale_row = gl.expand_dims(page_base + page_size * 64 + off * 4, 1)
    scales = gl.load(
        table_ptr + scale_row + gl.expand_dims(blk, 0),
        mask=gl.expand_dims(valid, 1),
        other=127,
    )
    return payload, scales


@gluon.jit
def _decode_tile(payload, scales):
    # [keys, 128] bf16 in natural element order (2i: low nibble of byte i).
    # e2m1 bits at the fp32 exponent LSBs read as value * 2^-126; the 2^126
    # multiply restores it exactly before the e8m0 scale.
    payload = payload.to(gl.int32)
    scale = gl.expand_dims((scales.to(gl.int32) << 23).to(gl.float32, bitcast=True), 2)
    lo = ((payload & 7) << 22) | ((payload & 8) << 28)
    hi = ((payload & 0x70) << 18) | ((payload & 0x80) << 24)
    lo = lo.to(gl.float32, bitcast=True) * 8.507059173023462e37 * scale
    hi = hi.to(gl.float32, bitcast=True) * 8.507059173023462e37 * scale
    return gl.reshape(gl.join(lo, hi), [BLOCK_K, 128]).to(gl.bfloat16)


@gluon.jit
def _store_pair(
    acc,
    w_ptr,
    lens_ptr,
    out_ptr,
    block_scores_ptr,
    row0,
    rows_left,
    tile,
    L,
    stride_wb,
    stride_out,
    stride_block_scores,
    WRITE_BLOCK_SCORES: gl.constexpr,
    CANDIDATE_BLOCK_SIZE: gl.constexpr,
):
    cols = gl.arange(0, 64, layout=gl.SliceLayout(0, acc.type.layout))
    w = gl.load(
        w_ptr + (row0 + cols // 32) * stride_wb + cols % 32,
        mask=cols // 32 < rows_left,
        other=0.0,
    )
    s = _relu_weight(acc, gl.broadcast(gl.expand_dims(w, 0), acc)[0])
    logit = _head_sum(s).to(gl.bfloat16).to(gl.float32)
    LL: gl.constexpr = logit.type.layout
    keys = tile * BLOCK_K + gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, LL))
    pair = gl.arange(0, 2, layout=gl.SliceLayout(0, LL))
    live = pair < rows_left
    n_vis = gl.load(lens_ptr + row0 + pair, mask=live, other=0)
    n_vis = gl.minimum(gl.maximum(n_vis, 0), L)
    logit = gl.where(
        gl.expand_dims(keys, 1) < gl.expand_dims(n_vis, 0), logit, float("-inf")
    )
    row = gl.expand_dims((row0 + pair).to(gl.int64), 0)
    gl.store(
        out_ptr + row * stride_out + gl.expand_dims(keys, 1),
        logit,
        mask=gl.expand_dims(live, 0) & gl.expand_dims(keys < stride_out, 1),
    )
    if WRITE_BLOCK_SCORES:
        blocks_per_tile: gl.constexpr = BLOCK_K // CANDIDATE_BLOCK_SIZE
        maxima = gl.reduce(
            gl.reshape(logit, [blocks_per_tile, CANDIDATE_BLOCK_SIZE, 2]), 1, _max_nan
        )
        BL: gl.constexpr = maxima.type.layout
        blocks = tile * blocks_per_tile + gl.arange(
            0, blocks_per_tile, layout=gl.SliceLayout(1, BL)
        )
        pair_b = gl.arange(0, 2, layout=gl.SliceLayout(0, BL))
        n_vis_b = gl.convert_layout(n_vis, gl.SliceLayout(0, BL))
        last = gl.expand_dims((n_vis_b - 1) // CANDIDATE_BLOCK_SIZE, 0)
        maxima = gl.where(
            gl.expand_dims(n_vis_b > 0, 0) & (gl.expand_dims(blocks, 1) == last),
            float("inf"),
            maxima,
        )
        gl.store(
            block_scores_ptr
            + gl.expand_dims((row0 + pair_b).to(gl.int64), 0) * stride_block_scores
            + gl.expand_dims(blocks, 1),
            maxima,
            mask=gl.expand_dims(pair_b < rows_left, 0)
            & gl.expand_dims(blocks < stride_block_scores, 1),
        )


@gluon.jit
def _score_tile(
    k_tile,
    q_smem,
    acc_a,
    acc_b,
    w_ptr,
    lens_ptr,
    out_ptr,
    block_scores_ptr,
    first,
    num_rows,
    tile,
    L,
    stride_wb,
    stride_out,
    stride_block_scores,
    NUM_PAIRS: gl.constexpr,
    WRITE_BLOCK_SCORES: gl.constexpr,
    CANDIDATE_BLOCK_SIZE: gl.constexpr,
):
    # Row pair p + 1 multiplies while pair p runs its epilogue. acc_a / acc_b
    # are recycled registers (use_acc=False): no non-wgmma write to an
    # accumulator while another wgmma is in flight.
    acc_a = warpgroup_mma(
        k_tile, q_smem.index(0).permute((1, 0)), acc_a, use_acc=False, is_async=True
    )
    for p in gl.static_range(NUM_PAIRS):
        if p + 1 < NUM_PAIRS:
            acc_b = warpgroup_mma(
                k_tile,
                q_smem.index(p + 1).permute((1, 0)),
                acc_b,
                use_acc=False,
                is_async=True,
            )
            acc_a = warpgroup_mma_wait(1, deps=[acc_a])
        else:
            acc_a = warpgroup_mma_wait(0, deps=[acc_a])
        if 2 * p < num_rows:
            _store_pair(
                acc_a,
                w_ptr,
                lens_ptr,
                out_ptr,
                block_scores_ptr,
                first + 2 * p,
                num_rows - 2 * p,
                tile,
                L,
                stride_wb,
                stride_out,
                stride_block_scores,
                WRITE_BLOCK_SCORES,
                CANDIDATE_BLOCK_SIZE,
            )
        if p + 1 < NUM_PAIRS:
            acc_a, acc_b = acc_b, acc_a
    return acc_a, acc_b


@gluon.jit
def _grouped_wgmma_kernel(
    q_desc,  # [B * HEADS, 128] bf16, blocks of two rows
    w_ptr,  # [B, HEADS] bf16
    slots_ptr,  # [num_reqs, context] request-to-token table
    req_ptr,  # [B] request IDs
    lens_ptr,  # [B] int64 visible compressed positions per row
    table_ptr,
    out_ptr,
    block_scores_ptr,
    B,
    L,
    page_size,
    row_stride,
    stride_wb,
    stride_slots,
    stride_out,
    stride_block_scores,
    COMPRESS_RATIO: gl.constexpr,
    GROUP: gl.constexpr,
    NUM_PAIRS: gl.constexpr,
    WRITE_BLOCK_SCORES: gl.constexpr,
    CANDIDATE_BLOCK_SIZE: gl.constexpr,
):
    # Scores groups of GROUP rows that hold one request; groups mixing requests
    # are left to the Triton grouped kernel in sm90_fp4_indexer (SKIP_UNIFORM).
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 16]
    )
    L3: gl.constexpr = gl.BlockedLayout([1, 1, 16], [8, 4, 1], [4, 1, 1], [2, 1, 0])
    key_l: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, L3))
    group_l: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    k_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_K, 128], gl.bfloat16
    )

    q_smem = gl.allocate_shared_memory(gl.bfloat16, [NUM_PAIRS, 64, 128], q_desc.layout)
    k_smem = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_K, 128], k_layout)
    bar = mbarrier.allocate_mbarrier()

    first = gl.program_id(0) * GROUP
    lane = gl.program_id(1)
    stride = gl.num_programs(1)
    offs_g = gl.arange(0, 8, layout=group_l)
    in_group = (offs_g < GROUP) & (first + offs_g < B)
    group_req = gl.load(req_ptr + first + offs_g, mask=in_group, other=-1)
    req = gl.load(req_ptr + first)
    mixed = gl.max(gl.where(in_group & (group_req != req), 1, 0), axis=0)
    num_rows = gl.minimum(B - first, GROUP)
    visible = gl.load(lens_ptr + first + offs_g, mask=in_group, other=0)
    visible = gl.max(gl.minimum(gl.maximum(visible, 0), L), axis=0).to(gl.int32)
    # Groups mixing requests score no tiles here (see SKIP_UNIFORM).
    visible = gl.where(mixed == 0, visible, 0)
    req = req.to(gl.int64)

    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, NUM_PAIRS * 64 * 128 * 2)
    for p in gl.static_range(NUM_PAIRS):
        tma.async_copy_global_to_shared(
            q_desc, [(first + 2 * p) * 32, 0], bar, q_smem.index(p)
        )
    mbarrier.wait(bar, 0)
    mbarrier.invalidate(bar)

    acc_a = gl.zeros([BLOCK_K, 64], gl.float32, layout=mma_layout)
    acc_b = gl.zeros([BLOCK_K, 64], gl.float32, layout=mma_layout)
    # Two-deep prefetch: the payload of the next tile, the slots after it.
    slot, valid = _tile_slots(
        slots_ptr, req, lane, visible, stride_slots, COMPRESS_RATIO, key_l
    )
    payload, scales = _tile_payload(table_ptr, slot, valid, page_size, row_stride, L3)
    slot, valid = _tile_slots(
        slots_ptr, req, lane + stride, visible, stride_slots, COMPRESS_RATIO, key_l
    )
    for tile in range(lane, gl.cdiv(visible, BLOCK_K), stride):
        k_tile = _decode_tile(payload, scales)
        payload, scales = _tile_payload(
            table_ptr, slot, valid, page_size, row_stride, L3
        )
        slot, valid = _tile_slots(
            slots_ptr,
            req,
            tile + 2 * stride,
            visible,
            stride_slots,
            COMPRESS_RATIO,
            key_l,
        )
        gl.barrier()  # the previous tile's wgmmas have retired
        k_smem.store(k_tile)
        fence_async_shared()
        gl.barrier()
        acc_a, acc_b = _score_tile(
            k_smem,
            q_smem,
            acc_a,
            acc_b,
            w_ptr,
            lens_ptr,
            out_ptr,
            block_scores_ptr,
            first,
            num_rows,
            tile,
            L,
            stride_wb,
            stride_out,
            stride_block_scores,
            NUM_PAIRS,
            WRITE_BLOCK_SCORES,
            CANDIDATE_BLOCK_SIZE,
        )


def fp4_index_logits_grouped_wgmma(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    req_indices: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    out: torch.Tensor,
    block_scores: torch.Tensor,
    *,
    compress_ratio: int,
    rows_per_group: int,
    write_block_scores: bool,
    candidate_block_size: int,
) -> None:
    """Prefix-only logits (and block maxima) for groups that hold one request.

    out / block_scores follow ``fp4_index_logits_decode``'s prefix contract;
    groups mixing requests are skipped and must be scored by the caller.
    """
    B = q.shape[0]
    L = out.shape[1]
    q_layout = gl.NVMMASharedLayout.get_default_for([64, 128], gl.bfloat16)
    q_desc = TensorDescriptor.from_tensor(q.view(B * HEADS, 128), [64, 128], q_layout)
    groups = triton.cdiv(B, rows_per_group)
    lanes = min(triton.cdiv(L, 64), triton.cdiv(_GRID_CTAS, groups))
    _grouped_wgmma_kernel[(groups, lanes)](
        q_desc,
        weights,
        slots,
        req_indices,
        lens,
        table,
        out,
        block_scores,
        B,
        L,
        page_size,
        table.stride(0),
        weights.stride(0),
        slots.stride(0),
        out.stride(0),
        block_scores.stride(0),
        COMPRESS_RATIO=compress_ratio,
        GROUP=rows_per_group,
        NUM_PAIRS=triton.cdiv(rows_per_group, 2),
        WRITE_BLOCK_SCORES=write_block_scores,
        CANDIDATE_BLOCK_SIZE=candidate_block_size,
        num_warps=4,
        maxnreg=_MAX_REGS,
    )


@gluon.jit
def _candidate_wgmma_kernel(
    q_desc,  # [B * HEADS, 128] bf16, blocks of one row
    w_ptr,  # [B, HEADS] bf16
    slots_ptr,  # [num_reqs, context] request-to-token table
    req_ptr,  # [B] request IDs
    blocks_ptr,  # [B, K] sorted logical candidate blocks
    counts_ptr,  # [B] int64 valid compact columns per row
    table_ptr,
    out_ptr,  # [B, K * block size] fp32 compact logits
    L,
    page_size,
    row_stride,
    stride_wb,
    stride_slots,
    stride_blocks,
    stride_out,
    COMPRESS_RATIO: gl.constexpr,
    CANDIDATE_BLOCK_SIZE: gl.constexpr,
):
    # One row per program: its candidates are its own, so K tiles are not
    # shared; wgmma (N = 32 heads) replaces the per-row register MMA.
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 16]
    )
    L3: gl.constexpr = gl.BlockedLayout([1, 1, 16], [8, 4, 1], [4, 1, 1], [2, 1, 0])
    key_l: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, L3))
    k_layout: gl.constexpr = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_K, 128], gl.bfloat16
    )
    q_smem = gl.allocate_shared_memory(gl.bfloat16, [32, 128], q_desc.layout)
    k_smem = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_K, 128], k_layout)
    bar = mbarrier.allocate_mbarrier()

    row = gl.program_id(0)
    lane = gl.program_id(1)
    stride = gl.num_programs(1)
    req = gl.load(req_ptr + row).to(gl.int64)
    count = gl.load(counts_ptr + row)
    count = gl.minimum(gl.maximum(count, 0), L).to(gl.int32)
    blocks_row = blocks_ptr + row.to(gl.int64) * stride_blocks
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, 32 * 128 * 2)
    tma.async_copy_global_to_shared(q_desc, [row * 32, 0], bar, q_smem)
    mbarrier.wait(bar, 0)
    mbarrier.invalidate(bar)

    cols = gl.arange(0, 32, layout=gl.SliceLayout(0, mma_layout))
    w = gl.load(w_ptr + row * stride_wb + cols)
    acc = gl.zeros([BLOCK_K, 32], gl.float32, layout=mma_layout)
    w = gl.broadcast(gl.expand_dims(w, 0), acc)[0]
    slot, valid = _candidate_slots(
        slots_ptr,
        blocks_row,
        req,
        lane,
        count,
        stride_slots,
        COMPRESS_RATIO,
        CANDIDATE_BLOCK_SIZE,
        key_l,
    )
    payload, scales = _tile_payload(table_ptr, slot, valid, page_size, row_stride, L3)
    slot, valid = _candidate_slots(
        slots_ptr,
        blocks_row,
        req,
        lane + stride,
        count,
        stride_slots,
        COMPRESS_RATIO,
        CANDIDATE_BLOCK_SIZE,
        key_l,
    )
    for tile in range(lane, gl.cdiv(count, BLOCK_K), stride):
        k_tile = _decode_tile(payload, scales)
        payload, scales = _tile_payload(
            table_ptr, slot, valid, page_size, row_stride, L3
        )
        slot, valid = _candidate_slots(
            slots_ptr,
            blocks_row,
            req,
            tile + 2 * stride,
            count,
            stride_slots,
            COMPRESS_RATIO,
            CANDIDATE_BLOCK_SIZE,
            key_l,
        )
        gl.barrier()  # the previous tile's wgmma has retired
        k_smem.store(k_tile)
        fence_async_shared()
        gl.barrier()
        acc = warpgroup_mma(k_smem, q_smem.permute((1, 0)), acc, use_acc=False)
        logit = _row_head_sum(_relu_weight(acc, w)).to(gl.bfloat16).to(gl.float32)
        LL: gl.constexpr = logit.type.layout
        keys = tile * BLOCK_K + gl.arange(0, BLOCK_K, layout=LL)
        logit = gl.where(keys < count, logit, float("-inf"))
        gl.store(
            out_ptr + row.to(gl.int64) * stride_out + keys,
            logit,
            mask=keys < stride_out,
        )


def fp4_index_logits_candidates_wgmma(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    req_indices: torch.Tensor,
    candidate_blocks: torch.Tensor,
    counts: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    out: torch.Tensor,
    *,
    compress_ratio: int,
    candidate_block_size: int,
) -> None:
    """Prefix-only compact logits of candidate consumers, one row per program."""
    B = q.shape[0]
    q_layout = gl.NVMMASharedLayout.get_default_for([32, 128], gl.bfloat16)
    q_desc = TensorDescriptor.from_tensor(q.view(B * HEADS, 128), [32, 128], q_layout)
    lanes = min(triton.cdiv(out.shape[1], 64), triton.cdiv(_CANDIDATE_GRID_CTAS, B))
    _candidate_wgmma_kernel[(B, lanes)](
        q_desc,
        weights,
        slots,
        req_indices,
        candidate_blocks,
        counts,
        table,
        out,
        out.shape[1],
        page_size,
        table.stride(0),
        weights.stride(0),
        slots.stride(0),
        candidate_blocks.stride(0),
        out.stride(0),
        COMPRESS_RATIO=compress_ratio,
        CANDIDATE_BLOCK_SIZE=candidate_block_size,
        num_warps=4,
    )
