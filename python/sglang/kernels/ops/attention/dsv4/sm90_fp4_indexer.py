"""Decode-time low-ratio indexer logits for Hopper.

DeepGEMM's fp8_fp4 mqa-logits kernels need SM100/SM120. These Triton kernels
cover the same decode step on SM90: each query row (or each DSpark verify row of
a request, sharing its decoded keys) is scored against the request's visible
compressed positions read straight out of the fp4 indexer pool (e2m1 payload +
e8m0 per-32 block scales, page layout of store_fp4_index_k_cache), summed over
heads with relu and the per-head weights.

Numerics follow the torch reference path (bf16 dot, bf16 relu/weight product,
bf16 head reduction) with the rounding points and dot shapes of
``fp4_indexer.fp4_index_logits_paged``, so the Triton kernels here return its
logits bit for bit.
"""

import torch
import triton
import triton.language as tl

INDEX_HEAD_DIM = 128
# Grouped-row launch tuned on H20 for 32 requests x 6 verify rows at 61K-69K
# context; uncapped, the kernel takes 163 registers and runs ~15% slower.
_GROUPED_CTAS = 4096
_GROUPED_MAX_REGS = 80
PAYLOAD_BYTES = tl.constexpr(64)
SCALE_BYTES = tl.constexpr(4)


@triton.jit
def _e2m1_decode(code, FAST: tl.constexpr = False):
    # code: uint 0..15 -> e2m1 value. exp = bits 2..1, mantissa = bit 0, sign = bit 3.
    e = (code >> 1) & 3
    m = (code & 1).to(tl.float32)
    sub = m * 0.5
    if FAST:
        exponent = tl.where(e == 3, 4.0, tl.where(e == 2, 2.0, 1.0))
    else:
        exponent = tl.exp2((e - 1).to(tl.float32))
    nor = (1.0 + m * 0.5) * exponent
    v = tl.where(e == 0, sub, nor)
    return tl.where((code >> 3) == 1, -v, v)


@triton.jit
def _e2m1_decode_bits(code):
    # Same fp32 bits as _e2m1_decode for every code, without float selects:
    # e2m1 bits at the fp32 exponent LSBs read as value * 2^-126 (subnormal
    # when e == 0), and the 2^126 multiply restores the value exactly. Adding
    # +0.0 turns code 8 into +0.0, as the float path yields.
    code = code.to(tl.int32)
    bits = ((code & 7) << 22) | ((code & 8) << 28)
    return bits.to(tl.float32, bitcast=True) * 8.507059173023462e37 + 0.0


@triton.jit
def _e8m0_decode(code):
    # Indexer scales are clamped to [1, 254] by the writer. Match this kernel's
    # historical CUDA exp2 behavior at the unused endpoints too: 0 -> flushed
    # zero, 255 -> infinity. This is not a general float8_e8m0fnu decoder.
    return (code.to(tl.int32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _maximum_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _fp4_index_logits_kernel(
    q_ptr,  # [B, H, D] bf16, fq4 queries (already rope'd)
    w_ptr,  # [B, H] bf16 head weights (softmax scale folded in)
    slots_ptr,  # [B, L] slots, or [num_reqs, context] request-to-token table
    req_ptr,  # [B] request IDs when INDIRECT_SLOTS is true
    blocks_ptr,  # [B, K] sorted logical candidate blocks
    lens_ptr,  # [B] int64 visible compressed positions per request
    table_ptr,  # [num_pages, page_size * 64 + page_size * 4] uint8
    out_ptr,  # [B, L] fp32 logits, -inf beyond lens
    block_scores_ptr,
    L,
    page_size,
    row_stride,
    stride_qb,
    stride_qh,
    stride_wb,
    stride_slots,
    stride_out,
    stride_blocks,
    stride_block_scores,
    INDIRECT_SLOTS: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SKIP_INVISIBLE: tl.constexpr,
    BOUNDED_GRID: tl.constexpr,
    PREFIX_ONLY: tl.constexpr,
    USE_CANDIDATES: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    WRITE_BLOCK_SCORES: tl.constexpr,
    BITS_DECODE: tl.constexpr,
    H: tl.constexpr,
    HALF_D: tl.constexpr,  # D // 2 == 64 nibble-pairs per row
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)
    lane = tl.program_id(1)
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(
        0, HALF_D
    )  # byte index i holds elements 2i (low nibble), 2i+1 (high nibble)

    n_vis = tl.load(lens_ptr + b)
    visible = tl.minimum(tl.maximum(n_vis, 0), L)
    if BOUNDED_GRID:
        steps = tl.cdiv(
            tl.maximum(tl.cdiv(visible, BLOCK_L) - lane, 0), tl.num_programs(1)
        )
    else:
        steps = 1
    if not PREFIX_ONLY or lane * BLOCK_L < visible:
        if PREFIX_ONLY:
            # queries: even / odd elements, [H, HALF_D] bf16
            q_even = tl.load(
                q_ptr
                + b * stride_qb
                + offs_h[:, None] * stride_qh
                + 2 * offs_i[None, :]
            )
            q_odd = tl.load(
                q_ptr
                + b * stride_qb
                + offs_h[:, None] * stride_qh
                + 2 * offs_i[None, :]
                + 1
            )
            w = tl.load(w_ptr + b * stride_wb + offs_h).to(tl.float32)
        for step in range(steps):
            lb = lane + step * tl.num_programs(1)
            offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
            valid = offs_l < tl.minimum(n_vis, L)
            if not SKIP_INVISIBLE or lb * BLOCK_L < tl.minimum(n_vis, L):
                if INDIRECT_SLOTS:
                    logical = offs_l
                    if USE_CANDIDATES:
                        block = tl.load(
                            blocks_ptr
                            + b * stride_blocks
                            + offs_l // CANDIDATE_BLOCK_SIZE,
                            mask=valid,
                            other=0,
                        )
                        logical = (
                            block * CANDIDATE_BLOCK_SIZE + offs_l % CANDIDATE_BLOCK_SIZE
                        )
                    req = tl.load(req_ptr + b).to(tl.int64)
                    slot = (
                        tl.load(
                            slots_ptr + req * stride_slots + logical * COMPRESS_RATIO,
                            mask=valid,
                            other=0,
                        ).to(tl.int64)
                        // COMPRESS_RATIO
                    )
                else:
                    slot = tl.load(
                        slots_ptr + b * stride_slots + offs_l, mask=valid, other=0
                    ).to(tl.int64)
                page = slot // page_size
                off = slot % page_size
                row_base = page * row_stride

                # K payload: [BLOCK_L, HALF_D] uint8
                pay = tl.load(
                    table_ptr
                    + row_base[:, None]
                    + off[:, None] * PAYLOAD_BYTES
                    + offs_i[None, :],
                    mask=valid[:, None],
                    other=0,
                )
                if BITS_DECODE:
                    low = _e2m1_decode_bits(pay & 0x0F)
                    high = _e2m1_decode_bits(pay >> 4)
                else:
                    low = _e2m1_decode(pay & 0x0F, PREFIX_ONLY)
                    high = _e2m1_decode((pay >> 4) & 0x0F, PREFIX_ONLY)
                # e8m0 block scales: element j uses block j // 32 -> byte i uses block i // 16.
                if PREFIX_ONLY:
                    sc_idx = tl.arange(0, 4)
                else:
                    sc_idx = offs_i // 16
                exps = tl.load(
                    table_ptr
                    + row_base[:, None]
                    + page_size * PAYLOAD_BYTES
                    + off[:, None] * SCALE_BYTES
                    + sc_idx[None, :],
                    mask=valid[:, None],
                    other=127,
                )
                if PREFIX_ONLY:
                    scale = _e8m0_decode(exps)[:, :, None]
                    k_low = tl.reshape(
                        tl.reshape(low, (BLOCK_L, 4, 16)) * scale, (BLOCK_L, HALF_D)
                    ).to(tl.bfloat16)
                    k_high = tl.reshape(
                        tl.reshape(high, (BLOCK_L, 4, 16)) * scale, (BLOCK_L, HALF_D)
                    ).to(tl.bfloat16)
                else:
                    scale = tl.exp2(exps.to(tl.float32) - 127.0)
                    k_low = (low * scale).to(
                        tl.bfloat16
                    )  # [BLOCK_L, HALF_D] elements 2i
                    k_high = (high * scale).to(tl.bfloat16)  # elements 2i+1

                if not PREFIX_ONLY:
                    # queries: even / odd elements, [H, HALF_D] bf16
                    q_even = tl.load(
                        q_ptr
                        + b * stride_qb
                        + offs_h[:, None] * stride_qh
                        + 2 * offs_i[None, :]
                    )
                    q_odd = tl.load(
                        q_ptr
                        + b * stride_qb
                        + offs_h[:, None] * stride_qh
                        + 2 * offs_i[None, :]
                        + 1
                    )
                acc = tl.dot(q_even, tl.trans(k_low))  # [H, BLOCK_L] fp32
                acc += tl.dot(q_odd, tl.trans(k_high))
                # reference rounding points: bf16 dot -> relu -> * bf16 weight -> bf16 -> sum -> bf16
                s = acc.to(tl.bfloat16).to(tl.float32)
                s = tl.maximum(s, 0.0)
                if not PREFIX_ONLY:
                    w = tl.load(w_ptr + b * stride_wb + offs_h).to(tl.float32)
                s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
                logit = tl.sum(s, axis=0).to(tl.bfloat16).to(tl.float32)
                logit = tl.where(valid, logit, float("-inf"))
            else:
                logit = tl.full((BLOCK_L,), float("-inf"), tl.float32)
            tl.store(
                out_ptr + b * stride_out + offs_l,
                logit,
                mask=offs_l < (stride_out if PREFIX_ONLY else L),
            )
            if WRITE_BLOCK_SCORES:
                blocks_per_tile: tl.constexpr = BLOCK_L // CANDIDATE_BLOCK_SIZE
                values = tl.reshape(logit, (blocks_per_tile, CANDIDATE_BLOCK_SIZE))
                scores = tl.reduce(values, 1, _maximum_with_nan)
                blocks = lb * blocks_per_tile + tl.arange(0, blocks_per_tile)
                scores = tl.where(
                    (n_vis > 0) & (blocks == (n_vis - 1) // CANDIDATE_BLOCK_SIZE),
                    float("inf"),
                    scores,
                )
                tl.store(
                    block_scores_ptr + b * stride_block_scores + blocks,
                    scores,
                    blocks
                    < (
                        stride_block_scores
                        if PREFIX_ONLY
                        else tl.cdiv(L, CANDIDATE_BLOCK_SIZE)
                    ),
                )
    if BOUNDED_GRID and not PREFIX_ONLY:
        # Full output contract, including a shrinking CUDA graph replay: compute
        # tiles write through the last partial tile, then coalesced wide stores
        # initialize the rest. No host read of lens or extra scratch allocation.
        TAIL_TILE: tl.constexpr = 4096
        tail = tl.cdiv(visible, BLOCK_L) * BLOCK_L
        for offset in range(tail + lane * TAIL_TILE, L, tl.num_programs(1) * TAIL_TILE):
            columns = offset + tl.arange(0, TAIL_TILE)
            tl.store(out_ptr + b * stride_out + columns, -float("inf"), columns < L)
        if WRITE_BLOCK_SCORES:
            first_block = tail // CANDIDATE_BLOCK_SIZE
            num_blocks = tl.cdiv(L, CANDIDATE_BLOCK_SIZE)
            for offset in range(
                first_block + lane * TAIL_TILE,
                num_blocks,
                tl.num_programs(1) * TAIL_TILE,
            ):
                columns = offset + tl.arange(0, TAIL_TILE)
                tl.store(
                    block_scores_ptr + b * stride_block_scores + columns,
                    -float("inf"),
                    columns < num_blocks,
                )


@triton.jit
def _fp4_shared_k_tile(
    table_ptr,
    slots_ptr,
    req,
    offs_l,
    valid,
    page_size,
    row_stride,
    stride_slots,
    COMPRESS_RATIO: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # Same bf16 K values as the per-row kernel's decode, for every code.
    offs_i = tl.arange(0, HALF_D)
    slot = (
        tl.load(
            slots_ptr + req * stride_slots + offs_l * COMPRESS_RATIO,
            mask=valid,
            other=0,
        ).to(tl.int64)
        // COMPRESS_RATIO
    )
    page = slot // page_size
    off = slot % page_size
    row_base = page * row_stride
    pay = tl.load(
        table_ptr + row_base[:, None] + off[:, None] * PAYLOAD_BYTES + offs_i[None, :],
        mask=valid[:, None],
        other=0,
    )
    exps = tl.load(
        table_ptr
        + row_base[:, None]
        + page_size * PAYLOAD_BYTES
        + off[:, None] * SCALE_BYTES
        + tl.arange(0, 4)[None, :],
        mask=valid[:, None],
        other=127,
    )
    scale = _e8m0_decode(exps)[:, :, None]
    low = _e2m1_decode_bits(pay & 0x0F)
    high = _e2m1_decode_bits(pay >> 4)
    k_low = tl.reshape(tl.reshape(low, (BLOCK_L, 4, 16)) * scale, (BLOCK_L, HALF_D)).to(
        tl.bfloat16
    )
    k_high = tl.reshape(
        tl.reshape(high, (BLOCK_L, 4, 16)) * scale, (BLOCK_L, HALF_D)
    ).to(tl.bfloat16)
    return k_low, k_high


@triton.jit
def _fp4_grouped_row_tile(
    q_ptr,
    w_ptr,
    lens_ptr,
    out_ptr,
    block_scores_ptr,
    row,
    k_low,
    k_high,
    offs_l,
    lb,
    L,
    stride_qb,
    stride_wb,
    stride_out,
    stride_block_scores,
    PREFIX_ONLY: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    WRITE_BLOCK_SCORES: tl.constexpr,
    H: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    n_vis = tl.load(lens_ptr + row)
    valid = offs_l < tl.minimum(n_vis, L)
    offs_h = tl.arange(0, H)
    q_row = q_ptr + row * stride_qb + offs_h[:, None] * HALF_D + tl.arange(0, HALF_D)
    # Dot shapes, register-resident K operands and rounding points are those of
    # the per-row kernel, so each row's logits come out bit-identical.
    acc = tl.dot(tl.load(q_row), tl.trans(k_low))  # [H, BLOCK_L] fp32
    acc += tl.dot(tl.load(q_row + H * HALF_D), tl.trans(k_high))
    s = acc.to(tl.bfloat16).to(tl.float32)
    s = tl.maximum(s, 0.0)
    w = tl.load(w_ptr + row * stride_wb + offs_h).to(tl.float32)
    s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
    logit = tl.sum(s, axis=0).to(tl.bfloat16).to(tl.float32)
    logit = tl.where(valid, logit, float("-inf"))
    tl.store(
        out_ptr + row * stride_out + offs_l,
        logit,
        mask=offs_l < (stride_out if PREFIX_ONLY else L),
    )
    if WRITE_BLOCK_SCORES:
        blocks_per_tile: tl.constexpr = BLOCK_L // CANDIDATE_BLOCK_SIZE
        values = tl.reshape(logit, (blocks_per_tile, CANDIDATE_BLOCK_SIZE))
        scores = tl.reduce(values, 1, _maximum_with_nan)
        blocks = lb * blocks_per_tile + tl.arange(0, blocks_per_tile)
        scores = tl.where(
            (n_vis > 0) & (blocks == (n_vis - 1) // CANDIDATE_BLOCK_SIZE),
            float("inf"),
            scores,
        )
        tl.store(
            block_scores_ptr + row * stride_block_scores + blocks,
            scores,
            blocks
            < (
                stride_block_scores if PREFIX_ONLY else tl.cdiv(L, CANDIDATE_BLOCK_SIZE)
            ),
        )


@triton.jit
def _fp4_grouped_tail(
    out_ptr,
    block_scores_ptr,
    row,
    tail,
    lane,
    L,
    stride_out,
    stride_block_scores,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    WRITE_BLOCK_SCORES: tl.constexpr,
):
    TAIL_TILE: tl.constexpr = 4096
    for offset in range(tail + lane * TAIL_TILE, L, tl.num_programs(1) * TAIL_TILE):
        columns = offset + tl.arange(0, TAIL_TILE)
        tl.store(out_ptr + row * stride_out + columns, -float("inf"), columns < L)
    if WRITE_BLOCK_SCORES:
        num_blocks = tl.cdiv(L, CANDIDATE_BLOCK_SIZE)
        for offset in range(
            tail // CANDIDATE_BLOCK_SIZE + lane * TAIL_TILE,
            num_blocks,
            tl.num_programs(1) * TAIL_TILE,
        ):
            columns = offset + tl.arange(0, TAIL_TILE)
            tl.store(
                block_scores_ptr + row * stride_block_scores + columns,
                -float("inf"),
                columns < num_blocks,
            )


@triton.jit
def _fp4_index_logits_grouped_kernel(
    q_ptr,  # [B, 2, H, D // 2] bf16: even / odd elements of each query
    w_ptr,  # [B, H] bf16
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
    stride_qb,
    stride_wb,
    stride_slots,
    stride_out,
    stride_block_scores,
    COMPRESS_RATIO: tl.constexpr,
    PREFIX_ONLY: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    WRITE_BLOCK_SCORES: tl.constexpr,
    GROUP: tl.constexpr,
    SKIP_UNIFORM: tl.constexpr,
    GROUP_P2: tl.constexpr,
    H: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # A program owns GROUP consecutive rows and walks their runs of equal
    # request IDs; each run decodes a K tile once for all of its rows. DSpark
    # verify groups are one run. Lanes stride over the run's live tiles.
    first = tl.program_id(0) * GROUP
    lane = tl.program_id(1)
    offs_g = tl.arange(0, GROUP_P2)
    in_group = (offs_g < GROUP) & (first + offs_g < B)
    group_req = tl.load(req_ptr + first + offs_g, mask=in_group, other=-1)
    group_vis = tl.load(lens_ptr + first + offs_g, mask=in_group, other=0)
    group_vis = tl.minimum(tl.maximum(group_vis, 0), L)
    num_rows = tl.minimum(B - first, GROUP)
    if SKIP_UNIFORM:
        # Groups of a single request were scored by the wgmma kernel.
        mixed = tl.max(
            tl.where(in_group & (group_req != tl.load(req_ptr + first)), 1, 0), axis=0
        )
        num_rows = tl.where(mixed > 0, num_rows, 0)
    start = 0
    while start < num_rows:
        req = tl.load(req_ptr + first + start)
        breaks = (offs_g > start) & in_group & (group_req != req)
        end = tl.minimum(tl.min(tl.where(breaks, offs_g, GROUP_P2), axis=0), num_rows)
        in_run = (offs_g >= start) & (offs_g < end)
        visible = tl.max(tl.where(in_run, group_vis, 0), axis=0)
        steps = tl.cdiv(
            tl.maximum(tl.cdiv(visible, BLOCK_L) - lane, 0), tl.num_programs(1)
        )
        for step in range(steps):
            lb = lane + step * tl.num_programs(1)
            offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
            k_low, k_high = _fp4_shared_k_tile(
                table_ptr,
                slots_ptr,
                req.to(tl.int64),
                offs_l,
                offs_l < visible,
                page_size,
                row_stride,
                stride_slots,
                COMPRESS_RATIO,
                HALF_D,
                BLOCK_L,
            )
            for r in range(start, end):
                _fp4_grouped_row_tile(
                    q_ptr,
                    w_ptr,
                    lens_ptr,
                    out_ptr,
                    block_scores_ptr,
                    first + r,
                    k_low,
                    k_high,
                    offs_l,
                    lb,
                    L,
                    stride_qb,
                    stride_wb,
                    stride_out,
                    stride_block_scores,
                    PREFIX_ONLY,
                    CANDIDATE_BLOCK_SIZE,
                    WRITE_BLOCK_SCORES,
                    H,
                    HALF_D,
                    BLOCK_L,
                )
        if not PREFIX_ONLY:
            for r in range(start, end):
                _fp4_grouped_tail(
                    out_ptr,
                    block_scores_ptr,
                    first + r,
                    tl.cdiv(visible, BLOCK_L) * BLOCK_L,
                    lane,
                    L,
                    stride_out,
                    stride_block_scores,
                    CANDIDATE_BLOCK_SIZE,
                    WRITE_BLOCK_SCORES,
                )
        start = end


def _wgmma_ok(q: torch.Tensor) -> bool:
    # TMA reads two-row query blocks: 16-byte aligned, 32 heads x 128 dims.
    return q.is_cuda and q.shape[1] == 32 and q.data_ptr() % 16 == 0


def fp4_index_logits_decode(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    *,
    req_indices: torch.Tensor | None = None,
    compress_ratio: int = 1,
    max_seq_len: int | None = None,
    skip_invisible: bool = True,
    bounded_grid: bool = False,
    prefix_only: bool = False,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    return_block_scores: bool = False,
    rows_per_group: int = 1,
    bits_decode: bool = False,
    use_wgmma: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """q [B, H, 128] bf16, weights [B, H], slots [B, L] int64, lens [B] int64,
    table = the layer's fp4 index-K page buffer (uint8, 2D). Returns [B, L] fp32
    logits with -inf at positions >= lens.

    When req_indices is given, slots is the request-to-token table, and the
    kernel resolves compressed slots directly. max_seq_len is a static replay
    bound, never a host read of the live lens. skip_invisible=False retains the
    original full-grid arithmetic for correctness/performance comparisons.
    bounded_grid=True limits the launch to roughly 2048 CTAs, loops over live
    tiles on device, and initializes the invisible tail with wide stores. It
    preserves the full -inf tail contract; it cannot be combined with
    skip_invisible=False. The launch depends only on shapes, never live lengths.

    prefix_only=True leaves the invisible tail uninitialized. EVERY consumer
    must use device lengths, including block-score Top-K and final slot mapping;
    torch.topk over the allocation is not permitted. The final partial tile is
    initialized through vector-load padding. This path also reuses queries in
    bounded loops and decodes FP4 scales without exp2 (PRs #40945 / #42861).

    A source can also return fused block maxima while retaining full logits for
    its global Top-K. Consumers address sorted candidate blocks directly, with
    lens giving their valid-prefix counts. Both paths retain query chunking.

    use_wgmma=True (prefix_only, 32 heads) scores grouped rows and candidate
    consumers with the Gluon wgmma kernels of sm90_fp4_indexer_wgmma: the same
    rounding points and head-sum tree, natural-order dot accumulation. Groups
    mixing requests stay on the Triton grouped kernel.

    bits_decode=True decodes e2m1 with integer bit placement instead of float
    selects: the same values, measurably cheaper on the ALU pipe that bounds
    this kernel on H20. The grouped path always does.

    rows_per_group > 1 lets one program score that many consecutive rows and
    decode each K tile once per run of equal request IDs among them (DSpark
    verify rows). Logits and block maxima stay bit-identical for any request
    layout; the launch always uses the bounded grid. Candidate consumers keep
    one row per program, since their rows read different positions.
    """
    assert not (bounded_grid or prefix_only) or skip_invisible
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    B, H, _ = q.shape
    assert weights.shape == (B, H) and lens.shape == (B,)
    indirect = req_indices is not None
    assert slots.dim() == 2 and slots.stride(1) == 1
    assert slots.dtype in (torch.int32, torch.int64)
    if indirect:
        assert compress_ratio in (1, 2)
        assert req_indices.shape == (B,) and req_indices.is_contiguous()
        assert req_indices.dtype in (torch.int32, torch.int64)
        assert (
            max_seq_len is not None
            and 0 <= max_seq_len <= slots.shape[1] // compress_ratio
        )
        L = max_seq_len
    else:
        assert slots.shape[0] == B
        L = slots.shape[1]
    assert table.dtype == torch.uint8 and table.dim() == 2
    assert table.stride(1) == 1 and table.shape[1] == page_size * 68
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    if candidate_blocks is not None or return_block_scores:
        assert indirect and candidate_block_size in (1, 2, 4, 8, 16, 32, 64)
    if candidate_blocks is not None:
        assert not return_block_scores
        assert candidate_blocks.shape[0] == B and candidate_blocks.stride(1) == 1
        assert candidate_blocks.dtype in (torch.int32, torch.int64)
        L = candidate_blocks.shape[1] * candidate_block_size
    # Top-K v2 uses 16-byte vector loads, including for non-aligned live widths.
    storage = torch.empty(
        (B, triton.cdiv(L, 4) * 4), dtype=torch.float32, device=q.device
    )
    out = storage[:, :L]
    block_scores = out
    if return_block_scores:
        num_blocks = triton.cdiv(L, candidate_block_size)
        block_storage = torch.empty(
            (B, triton.cdiv(num_blocks, 4) * 4), dtype=torch.float32, device=q.device
        )
        block_scores = block_storage[:, :num_blocks]
    if L == 0 or B == 0:
        return (out, block_scores) if return_block_scores else out
    BLOCK_L = 64
    tiles = triton.cdiv(L, BLOCK_L)
    if use_wgmma and prefix_only and candidate_blocks is not None and _wgmma_ok(q):
        from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer_wgmma import (
            fp4_index_logits_candidates_wgmma,
        )

        fp4_index_logits_candidates_wgmma(
            q,
            weights,
            slots,
            req_indices,
            candidate_blocks,
            lens.to(torch.int64).contiguous(),
            table,
            page_size,
            out,
            compress_ratio=compress_ratio,
            candidate_block_size=candidate_block_size,
        )
        return out
    if rows_per_group > 1 and indirect and skip_invisible and candidate_blocks is None:
        groups = triton.cdiv(B, rows_per_group)
        wgmma = use_wgmma and prefix_only and rows_per_group <= 8 and _wgmma_ok(q)
        if wgmma:
            from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer_wgmma import (
                fp4_index_logits_grouped_wgmma,
            )

            fp4_index_logits_grouped_wgmma(
                q,
                weights,
                slots,
                req_indices,
                lens.to(torch.int64).contiguous(),
                table,
                page_size,
                out,
                block_scores,
                compress_ratio=compress_ratio,
                rows_per_group=rows_per_group,
                write_block_scores=return_block_scores,
                candidate_block_size=candidate_block_size or 1,
            )
        # [B, 2, H, 64]: even / odd query elements as contiguous planes.
        q_split = q.view(B, H, INDEX_HEAD_DIM // 2, 2).permute(0, 3, 1, 2).contiguous()
        _fp4_index_logits_grouped_kernel[
            (groups, min(tiles, triton.cdiv(_GROUPED_CTAS, groups)))
        ](
            q_split,
            weights,
            slots,
            req_indices,
            lens.to(torch.int64).contiguous(),
            table,
            out,
            block_scores,
            B,
            L,
            page_size,
            table.stride(0),
            q_split.stride(0),
            weights.stride(0),
            slots.stride(0),
            out.stride(0),
            block_scores.stride(0),
            COMPRESS_RATIO=compress_ratio,
            PREFIX_ONLY=prefix_only,
            CANDIDATE_BLOCK_SIZE=candidate_block_size or 1,
            WRITE_BLOCK_SCORES=return_block_scores,
            GROUP=rows_per_group,
            SKIP_UNIFORM=wgmma,
            GROUP_P2=triton.next_power_of_2(rows_per_group),
            H=H,
            HALF_D=INDEX_HEAD_DIM // 2,
            BLOCK_L=BLOCK_L,
            num_warps=4,
            maxnreg=_GROUPED_MAX_REGS,
        )
        return (out, block_scores) if return_block_scores else out
    if bounded_grid:
        tiles = min(tiles, max(1, triton.cdiv(2048, B)))
    grid = (B, tiles)
    _fp4_index_logits_kernel[grid](
        q,
        weights,
        slots,
        req_indices,
        candidate_blocks,
        lens.to(torch.int64).contiguous(),
        table,
        out,
        block_scores,
        L,
        page_size,
        table.stride(0),
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        slots.stride(0),
        out.stride(0),
        candidate_blocks.stride(0) if candidate_blocks is not None else 0,
        block_scores.stride(0),
        INDIRECT_SLOTS=indirect,
        COMPRESS_RATIO=compress_ratio,
        SKIP_INVISIBLE=skip_invisible,
        BOUNDED_GRID=bounded_grid,
        PREFIX_ONLY=prefix_only,
        USE_CANDIDATES=candidate_blocks is not None,
        CANDIDATE_BLOCK_SIZE=candidate_block_size or 1,
        WRITE_BLOCK_SCORES=return_block_scores,
        BITS_DECODE=bits_decode,
        H=H,
        HALF_D=INDEX_HEAD_DIM // 2,
        BLOCK_L=BLOCK_L,
        num_warps=4,
    )
    return (out, block_scores) if return_block_scores else out
