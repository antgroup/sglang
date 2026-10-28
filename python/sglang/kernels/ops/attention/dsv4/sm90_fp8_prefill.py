"""Hopper V4.1 prefill index scoring, adapted from sgl-project/sglang#39086.

Unpack a request's visible FP4 index-K rows once into E4M3, cast its fp4-grid
BF16 queries to E4M3, then score them with FP8 tensor-core dots that keep the
BF16 post-dot rounding points of the torch reference.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    INDEX_HEAD_DIM,
    INDEX_K_PAYLOAD_BYTES,
    INDEX_K_SCALE_BYTES,
    _e2m1_decode,
)

FP8_E4M3_MAX = 448.0


@triton.jit
def _unpack_fp4_index_keys_to_fp8_kernel(
    slots_ptr,
    table_ptr,
    out_ptr,
    page_size,
    row_stride,
    HALF_D: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Decode block-scaled E2M1 values directly into E4M3."""
    row = tl.program_id(0)
    offs_i = tl.arange(0, HALF_D)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    page = slot // page_size
    off = slot % page_size
    row_base = page * row_stride
    pay = tl.load(table_ptr + row_base + off * INDEX_K_PAYLOAD_BYTES + offs_i)
    scale_block = offs_i // 16
    exps = tl.load(
        table_ptr
        + row_base
        + page_size * INDEX_K_PAYLOAD_BYTES
        + off * INDEX_K_SCALE_BYTES
        + scale_block
    )
    scale = tl.exp2(exps.to(tl.float32) - 127.0)
    low = tl.clamp(_e2m1_decode(pay & 0x0F) * scale, -FP8_MAX, FP8_MAX).to(
        out_ptr.dtype.element_ty
    )
    high = tl.clamp(
        _e2m1_decode((pay >> 4) & 0x0F) * scale,
        -FP8_MAX,
        FP8_MAX,
    ).to(out_ptr.dtype.element_ty)
    out = out_ptr + row * (HALF_D * 2)
    tl.store(out + 2 * offs_i, low)
    tl.store(out + 2 * offs_i + 1, high)


@triton.jit
def _prmt(a, b, selectors):
    """PTX byte permute: output byte i is byte (selectors >> 4i) & 7 of the eight
    bytes a[0..3], b[0..3]; selector bit 4i + 3 must be clear."""
    return tl.inline_asm_elementwise(
        "prmt.b32 $0, $1, $2, $3;",
        "=r,r,r,r",
        [a, b, selectors],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp4_codes_to_fp8(codes, pos_lo, pos_hi, neg_lo, neg_hi):
    """Four FP4 codes (nibbles of bits 0..15) to four E4M3 bytes, looked up in the
    16-byte table pos_lo, pos_hi, neg_lo, neg_hi (codes 0..15) of their scale byte."""
    magnitude = codes & 0x7777
    pos = _prmt(pos_lo, pos_hi, magnitude)
    neg = _prmt(neg_lo, neg_hi, magnitude)
    # 0xFF in the bytes whose code has the sign bit set.
    sign = _prmt(
        tl.full(codes.shape, 0xFF00, tl.int32),
        tl.zeros(codes.shape, tl.int32),
        (codes >> 3) & 0x1111,
    )
    return (pos & ~sign) | (neg & sign)


@triton.jit
def _broadcast_blocks(per_block, BLOCK_R: tl.constexpr):
    """[BLOCK_R, 4] per scale block to [BLOCK_R, 16] per payload word."""
    return tl.reshape(
        tl.broadcast_to(tl.reshape(per_block, (BLOCK_R, 4, 1)), (BLOCK_R, 4, 4)),
        (BLOCK_R, 16),
    )


@triton.jit
def _unpack_fp4_index_keys_to_fp8_lut_kernel(
    slots_ptr,
    table_ptr,
    lut_ptr,
    out_ptr,
    num_rows,
    row_stride_words,
    PAGE_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """_unpack_fp4_index_keys_to_fp8_kernel through _fp4_index_key_lut, BLOCK_R rows
    per program, in 32-bit words: a payload word holds 8 codes, a row 16 words
    and one word of 4 scale bytes."""
    rows = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    row_ok = rows < num_rows
    slot = tl.load(slots_ptr + rows, mask=row_ok, other=0).to(tl.int64)
    page_base = (slot // PAGE_SIZE) * row_stride_words
    off = slot % PAGE_SIZE
    mask = row_ok[:, None]
    table_words = table_ptr.to(tl.pointer_type(tl.int32))
    payload = tl.load(
        table_words + (page_base + off * 16)[:, None] + tl.arange(0, 16)[None, :],
        mask=mask,
        other=0,
    )
    scale_word = tl.load(
        table_words + page_base + PAGE_SIZE * 16 + off, mask=row_ok, other=0
    )
    scale = (scale_word[:, None] >> (tl.arange(0, 4) * 8)[None, :]) & 0xFF
    lut_row = lut_ptr + scale * 4
    pos_lo = _broadcast_blocks(tl.load(lut_row, mask=mask, other=0), BLOCK_R)
    pos_hi = _broadcast_blocks(tl.load(lut_row + 1, mask=mask, other=0), BLOCK_R)
    neg_lo = _broadcast_blocks(tl.load(lut_row + 2, mask=mask, other=0), BLOCK_R)
    neg_hi = _broadcast_blocks(tl.load(lut_row + 3, mask=mask, other=0), BLOCK_R)
    # Payload bytes i, i + 1 hold output bytes 2i .. 2i + 3 as their nibbles, low first.
    first = _fp4_codes_to_fp8(payload & 0xFFFF, pos_lo, pos_hi, neg_lo, neg_hi)
    second = _fp4_codes_to_fp8((payload >> 16) & 0xFFFF, pos_lo, pos_hi, neg_lo, neg_hi)
    out = tl.reshape(tl.join(first, second), (BLOCK_R, 32))
    tl.store(
        out_ptr.to(tl.pointer_type(tl.int32))
        + rows[:, None] * 32
        + tl.arange(0, 32)[None, :],
        out,
        mask=mask,
    )


_fp4_index_key_luts: dict = {}


def _fp4_index_key_lut(device: torch.device) -> torch.Tensor:
    """[256, 4] int32: word w of row s packs the E4M3 bytes that
    _unpack_fp4_index_keys_to_fp8_kernel writes for codes 4w .. 4w + 3 under scale
    byte s. Each output byte is a function of its code and scale byte only, and
    the table is that kernel's own output, so the LUT path is bitwise it."""
    key = (device.type, device.index)
    lut = _fp4_index_key_luts.get(key)
    if lut is not None:
        return lut
    assert not torch.cuda.is_current_stream_capturing()
    # Row s: scale bytes s; payload byte c = c | c << 4 for codes c of every block.
    page_size = 256
    table = torch.empty((1, page_size * 68), dtype=torch.uint8, device=device)
    codes = torch.arange(16, dtype=torch.uint8, device=device)
    table[0, : page_size * 64] = (codes | (codes << 4)).repeat(page_size * 4)
    table[0, page_size * 64 :] = torch.arange(
        page_size, dtype=torch.uint8, device=device
    ).repeat_interleave(4)
    slots = torch.arange(page_size, device=device)
    out = unpack_fp4_index_keys_to_fp8(slots, table, page_size, row_tiled=False)
    out = out.view(torch.uint8)
    # [scale byte, block, code, low / high nibble]
    out = out.view(page_size, 4, 16, 2)
    lut = out[:, 0, :, 0].contiguous()
    assert torch.equal(out, lut[:, None, :, None].expand_as(out))
    lut = lut.view(torch.int32)
    _fp4_index_key_luts[key] = lut
    return lut


def unpack_fp4_index_keys_to_fp8(
    slots: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    *,
    row_tiled: bool = True,
) -> torch.Tensor:
    """Gather the index-K rows at ``slots`` of a layer's page buffer and decode
    them into E4M3 ``[len(slots), 128]``. ``row_tiled`` decodes many rows per
    program through a per-scale-byte table of the per-row kernel's own outputs:
    bitwise the same."""
    assert slots.dim() == 1
    assert table.dtype == torch.uint8 and table.dim() == 2
    assert table.stride(1) == 1 and table.shape[1] == page_size * 68
    slots = slots.to(torch.int64).contiguous()
    values = torch.empty(
        (slots.shape[0], INDEX_HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device=slots.device,
    )
    if slots.numel() > 0 and row_tiled:
        assert table.data_ptr() % 4 == 0
        block_r = 32
        _unpack_fp4_index_keys_to_fp8_lut_kernel[
            (triton.cdiv(slots.shape[0], block_r),)
        ](
            slots,
            table,
            _fp4_index_key_lut(table.device),
            values,
            slots.shape[0],
            table.stride(0) // 4,
            PAGE_SIZE=page_size,
            BLOCK_R=block_r,
            num_warps=2,
        )
    elif slots.numel() > 0:
        _unpack_fp4_index_keys_to_fp8_kernel[(slots.shape[0],)](
            slots,
            table,
            values,
            page_size,
            table.stride(0),
            HALF_D=INDEX_HEAD_DIM // 2,
            FP8_MAX=FP8_E4M3_MAX,
            num_warps=4,
        )
    return values


@triton.jit
def _quantize_bf16_index_queries_fp8_kernel(
    q_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_ob,
    stride_oh,
    H: tl.constexpr,
    D: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    row = tl.program_id(0)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    q = tl.load(
        q_ptr + row * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :]
    ).to(tl.float32)
    q_fp8 = tl.clamp(q, -FP8_MAX, FP8_MAX).to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + row * stride_ob + offs_h[:, None] * stride_oh + offs_d[None, :],
        q_fp8,
    )


def quantize_bf16_index_queries_fp8(q: torch.Tensor) -> torch.Tensor:
    """Cast each BF16 query head to E4M3 for one K=128 tensor-core dot; fp4-grid
    queries are exact in E4M3."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    q = q.contiguous()
    rows, heads, _ = q.shape
    values = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    if rows > 0:
        _quantize_bf16_index_queries_fp8_kernel[(rows,)](
            q,
            values,
            q.stride(0),
            q.stride(1),
            values.stride(0),
            values.stride(1),
            H=heads,
            D=INDEX_HEAD_DIM,
            FP8_MAX=FP8_E4M3_MAX,
            num_warps=8,
        )
    return values


@triton.jit
def _fp8_index_logits_prefill_tiled_kernel(
    q_ptr,  # [B, H, D] e4m3
    w_ptr,  # [B, H] bf16
    k_ptr,  # [L, D] e4m3
    lens_ptr,  # [B] int64
    out_ptr,  # [B, OUT_L] fp32
    B,
    L,
    OUT_L,
    stride_qb,
    stride_qh,
    stride_kl,
    stride_wb,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """BLOCK_M query rows x BLOCK_N key rows per CTA, one head at a time; the K
    tile is loaded once for every row and head of the block.

    Each head is four independent 32-wide dots summed in FP32 in order. Index q
    and k are FP4 values with one ue8m0 scale per 32 elements, so every 32-wide
    partial dot is exact even inside the MMA and the scores do not depend on
    the tile config, the row split or the row count. Triton's own exact mode
    (max_num_imprecise_acc=0) serializes MMA and adds and is ~2x slower on H20.
    """
    tl.static_assert(D == 128, "index head dim is 128 (four 32-wide blocks)")
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < B
    n_vis = tl.minimum(tl.load(lens_ptr + offs_m, mask=mask_m, other=0), L)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Causal chunks leave whole tiles invisible to every row of the block.
    if pid_n * BLOCK_N < tl.max(n_vis, axis=0):
        k_rows = k_ptr + offs_n[:, None].to(tl.int64) * stride_kl
        k_mask = (offs_n < L)[:, None]
        q_rows = q_ptr + offs_m[:, None].to(tl.int64) * stride_qb
        w_rows = w_ptr + offs_m.to(tl.int64) * stride_wb
        d32 = tl.arange(0, 32)
        k0 = tl.trans(tl.load(k_rows + d32[None, :], mask=k_mask, other=0.0))
        k1 = tl.trans(tl.load(k_rows + 32 + d32[None, :], mask=k_mask, other=0.0))
        k2 = tl.trans(tl.load(k_rows + 64 + d32[None, :], mask=k_mask, other=0.0))
        k3 = tl.trans(tl.load(k_rows + 96 + d32[None, :], mask=k_mask, other=0.0))
        for h in range(H):
            qh = q_rows + h * stride_qh + d32[None, :]
            qm = mask_m[:, None]
            p0 = tl.dot(tl.load(qh, mask=qm, other=0.0), k0, out_dtype=tl.float32)
            p1 = tl.dot(tl.load(qh + 32, mask=qm, other=0.0), k1, out_dtype=tl.float32)
            p2 = tl.dot(tl.load(qh + 64, mask=qm, other=0.0), k2, out_dtype=tl.float32)
            p3 = tl.dot(tl.load(qh + 96, mask=qm, other=0.0), k3, out_dtype=tl.float32)
            w = tl.load(w_rows + h, mask=mask_m, other=0.0).to(tl.float32)
            s = (((p0 + p1) + p2) + p3).to(tl.bfloat16).to(tl.float32)
            s = tl.maximum(s, 0.0, propagate_nan=tl.PropagateNan.ALL)
            acc += (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
    logit = acc.to(tl.bfloat16).to(tl.float32)
    logit = tl.where(offs_n[None, :] < n_vis[:, None], logit, float("-inf"))
    tl.store(
        out_ptr + offs_m[:, None].to(tl.int64) * OUT_L + offs_n[None, :],
        logit,
        mask=mask_m[:, None] & (offs_n < OUT_L)[None, :],
    )


# (BLOCK_M, BLOCK_N, num_warps, num_stages) of the tiled kernel, from an H20
# sweep at 1024 rows x 64K/256K keys; small calls take a smaller row tile.
_TILED_CONFIG = (64, 128, 4, 3)
_TILED_SMALL_CONFIG = (16, 128, 4, 3)
_TILED_SMALL_ROWS = 32
# The keys-major kernel from this many scores (rows * width) up; below it its
# launch overhead (TMA descriptors, prefix pass) loses to the tiled kernel.
# H20: 0.5-0.9x under 1M scores, 0.9-1.6x at 2M, 1.2-1.6x from 4M up. Other
# SM90 parts keep the tiled kernel until measured.
_KEYS_MAJOR_MIN_SCORES = {"NVIDIA H20": 1 << 21, "NVIDIA H20-3e": 1 << 21}
_keys_major_min_scores_by_device: dict = {}


def keys_major_min_scores(device: torch.device) -> Optional[int]:
    """The keys-major dispatch threshold measured for this device, or None."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _keys_major_min_scores_by_device:
        name = torch.cuda.get_device_name(index)
        _keys_major_min_scores_by_device[index] = _KEYS_MAJOR_MIN_SCORES.get(name)
    return _keys_major_min_scores_by_device[index]


def _use_keys_major(q: torch.Tensor, keys: torch.Tensor) -> bool:
    threshold = keys_major_min_scores(q.device)
    if threshold is None or q.shape[0] * keys.shape[0] < threshold:
        return False
    # Lazy: the Gluon Hopper API is experimental; import it only when it runs.
    from sglang.kernels.ops.attention.dsv4.sm90_fp8_prefill_keys_major import (
        keys_major_scores_supported,
    )

    return keys_major_scores_supported(q, keys)


def fp8_index_logits_prefill(
    q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    lens: torch.Tensor,
    *,
    keys_major: bool = True,
    tiled_config: Optional[tuple] = None,
) -> torch.Tensor:
    """Score E4M3 queries ``[rows, H, 128]`` against E4M3 keys ``[width, 128]``,
    -inf at columns ``>= lens``: ``[rows, width]`` fp32 whose rows have a
    four-float-aligned stride for ragged top-k; the backing padding is -inf.

    ``keys_major`` lets large calls take the Gluon keys-major kernel, bitwise the
    tiled one; ``tiled_config`` overrides the tiled kernel's tile config.
    """
    assert q.dtype == torch.float8_e4m3fn and q.shape[-1] == INDEX_HEAD_DIM
    rows, heads, _ = q.shape
    width = keys.shape[0]
    assert keys.dtype == torch.float8_e4m3fn and keys.shape[1:] == (INDEX_HEAD_DIM,)
    assert weights.shape == (rows, heads)
    assert lens.shape == (rows,)
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    keys = keys.contiguous()
    out_width = ((width + 3) // 4) * 4
    storage = torch.empty((rows, out_width), dtype=torch.float32, device=q.device)
    out = storage[:, :width]
    if rows == 0 or width == 0:
        return out
    lens = lens.to(torch.int64).contiguous()
    if keys_major and tiled_config is None and _use_keys_major(q, keys):
        from sglang.kernels.ops.attention.dsv4.sm90_fp8_prefill_keys_major import (
            keys_major_scores,
        )

        keys_major_scores(q, weights, keys, lens, out)
        return out
    if tiled_config is not None:
        config = tiled_config
    elif rows <= _TILED_SMALL_ROWS:
        config = _TILED_SMALL_CONFIG
    else:
        config = _TILED_CONFIG
    block_m, block_n, num_warps, num_stages = config
    _fp8_index_logits_prefill_tiled_kernel[
        (triton.cdiv(rows, block_m), triton.cdiv(out_width, block_n))
    ](
        q,
        weights,
        keys,
        lens,
        out,
        rows,
        width,
        out_width,
        q.stride(0),
        q.stride(1),
        keys.stride(0),
        weights.stride(0),
        H=heads,
        D=INDEX_HEAD_DIM,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# LENS is a row slice whose 16-byte alignment follows the chunk's row offset;
# specialized on it, a deep 512K chunk compiled a new variant mid-serving (H20
# TP8). One int load per row, so dropping the alignment hint is free.
@triton.jit(
    do_not_specialize=["k", "num_slots", "score_width"],
    do_not_specialize_on_alignment=["LENS"],
)
def _finalize_prefill_topk_kernel(
    SELECTED,
    SCORES,
    LENS,
    SLOTS,
    PAGE_INDICES,
    RAW_INDICES,
    selected_stride,
    score_stride,
    page_stride,
    raw_stride,
    k,
    num_slots,
    score_width,
    K_PAD: tl.constexpr,
    MASK_SCORES: tl.constexpr,
    HAS_PAGES: tl.constexpr,
    HAS_RAW: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, K_PAD)
    in_k = offsets < k
    selected = tl.load(SELECTED + row * selected_stride + offsets, in_k, -1)
    if MASK_SCORES:
        column = tl.minimum(tl.maximum(selected, 0), score_width - 1)
        score = tl.load(
            SCORES + row * score_stride + column, mask=in_k, other=-float("inf")
        )
        keep = (selected >= 0) & (selected < score_width) & (score > -float("inf"))
        selected = tl.where(keep, selected, -1)
    # Holes and the lanes past k sort last as INT32_MAX.
    unselected = 2147483647
    selected = tl.sort(tl.where(selected < 0, unselected, selected))
    reach = (selected != unselected) & (selected < tl.load(LENS + row))
    if HAS_PAGES:
        slot = tl.load(SLOTS + tl.minimum(selected, num_slots - 1), mask=reach, other=0)
        tl.store(
            PAGE_INDICES + row * page_stride + offsets,
            tl.where(reach, slot, -1).to(tl.int32),
            in_k,
        )
    if HAS_RAW:
        tl.store(
            RAW_INDICES + row * raw_stride + offsets,
            tl.where(reach, selected, -1),
            in_k,
        )


def finalize_prefill_topk(
    *,
    selected: torch.Tensor,
    scores: torch.Tensor,
    lens: torch.Tensor,
    slots: torch.Tensor,
    page_indices: Optional[torch.Tensor],
    raw_indices: Optional[torch.Tensor],
    mask_scores: bool,
) -> None:
    """One request's ragged prefill top-k rows, sorted ascending with -1 last:
    ``slots[id]`` into ``page_indices``, ``id`` into ``raw_indices``, the first k
    columns only.

    selected: [rows, k] int32 (-1 holes); scores: [rows, >= 1] fp32;
    lens: [rows] int32 visible positions; slots: [num_slots] int64.
    mask_scores drops ids whose score is -inf or outside scores.
    """
    rows, k = selected.shape
    if rows == 0:
        return
    assert selected.dtype == torch.int32 and selected.stride(1) == 1
    assert scores.dtype == torch.float32 and scores.stride(1) == 1
    assert lens.dtype == torch.int32 and lens.shape == (rows,)
    assert slots.dim() == 1 and slots.stride(0) == 1 and slots.numel() > 0
    for out in (page_indices, raw_indices):
        if out is not None:
            assert out.dtype == torch.int32 and out.stride(1) == 1
            assert out.shape[0] == rows and out.shape[1] >= k
    _finalize_prefill_topk_kernel[(rows,)](
        selected,
        scores,
        lens,
        slots,
        page_indices,
        raw_indices,
        selected.stride(0),
        scores.stride(0),
        page_indices.stride(0) if page_indices is not None else 0,
        raw_indices.stride(0) if raw_indices is not None else 0,
        k,
        slots.shape[0],
        scores.shape[1],
        K_PAD=triton.next_power_of_2(k),
        MASK_SCORES=mask_scores,
        HAS_PAGES=page_indices is not None,
        HAS_RAW=raw_indices is not None,
        # Arbitrary; sweep with the prefill finish microbenchmark when retuning.
        num_warps=4,
    )
