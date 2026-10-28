"""Kernels of the DeepSeek-V4.1 CP prefill collectives over NVLink symmetric memory.

The buffers and the barriers belong to ``sglang.srt.layers.communication.cp_prefill_symm``;
these kernels only read or write rows of them:

* ``sum_slots``: the reduce half of the CP MoE reduce-scatter, an fp32 sum of the
  ranks' slots in rank order with one rounding.
* ``embed_owner_scatter``: each token's vocab-shard owner writes its embedding row
  into the row of the CP rank that holds the token.
* ``engram_owner_scatter``: the same owner write for an engram hash-table lookup.

Both scatters write ``x + 0.0``, so -0 lands as +0, as the sum of the owner's row
with the other ranks' zero rows does: their output is bitwise the all-reduce or
reduce-scatter of the zero-filled sharded lookup.
"""

import torch
import triton
import triton.language as tl

_SUM_BLOCK = 2048
_EMBED_BLOCK_H = 1024
# e8m0 has no zero: the exponent byte 0 encodes 2**-127.
_E8M0_ZERO = 2.0**-127


@triton.jit
def _sum_slots_kernel(
    src_ptr, out_ptr, numel, WORLD: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    acc = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    for s in tl.static_range(1, WORLD):
        acc += tl.load(src_ptr + s * numel + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def sum_slots(slots: torch.Tensor) -> torch.Tensor:
    """``[W, rows, hidden]`` -> ``[rows, hidden]``: the fp32 sum of the W slots in
    slot order, rounded once to the slots' dtype."""
    world, rows, hidden = slots.shape
    assert slots.is_contiguous()
    out = torch.empty((rows, hidden), dtype=slots.dtype, device=slots.device)
    numel = rows * hidden
    if numel:
        _sum_slots_kernel[(triton.cdiv(numel, _SUM_BLOCK),)](
            slots, out, numel, WORLD=world, BLOCK=_SUM_BLOCK, num_warps=4
        )
    return out


@triton.jit
def _embed_owner_scatter_kernel(
    ids_ptr,
    weight_ptr,
    dst_table_ptr,
    vocab_start,
    vocab_end,
    WORLD: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)
    tok = tl.load(ids_ptr + t).to(tl.int64)
    if (tok >= vocab_start) & (tok < vocab_end):
        dest = t % WORLD
        row = (t // WORLD).to(tl.int64)
        base = tl.load(dst_table_ptr + dest).to(tl.pointer_type(tl.bfloat16))
        src = weight_ptr + (tok - vocab_start) * HIDDEN
        for h in tl.static_range(0, HIDDEN, BLOCK_H):
            offs = h + tl.arange(0, BLOCK_H)
            mask = offs < HIDDEN
            v = tl.load(src + offs, mask=mask).to(tl.float32) + 0.0
            tl.store(base + row * HIDDEN + offs, v.to(tl.bfloat16), mask=mask)


def embed_owner_scatter(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    dst_table: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
) -> None:
    """Write row ``weight[id - vocab_start]`` of every token whose id is in
    ``[vocab_start, vocab_end)`` to row ``t // W`` of buffer ``t % W``.

    ``input_ids``: [T] int; ``weight``: [vocab_end - vocab_start, hidden] bf16,
    contiguous; ``dst_table``: [W] int64 addresses of [T // W, hidden] bf16
    buffers (this rank's and its peers'). Tokens of other shards write nothing.
    """
    assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
    t = input_ids.shape[0]
    if t:
        _embed_owner_scatter_kernel[(t,)](
            input_ids,
            weight,
            dst_table,
            vocab_start,
            vocab_end,
            WORLD=dst_table.shape[0],
            HIDDEN=weight.shape[1],
            BLOCK_H=_EMBED_BLOCK_H,
            num_warps=4,
        )


@triton.jit
def _engram_owner_scatter_kernel(
    w_ptr,
    s_ptr,
    sorted_ids_ptr,
    src_pos_ptr,
    dst_table_ptr,
    owned_pos_ptr,
    row_lo,
    num_programs,
    ROWS,
    COLS: tl.constexpr,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
):
    # Flat position p of the [W * ROWS, COLS] ids goes to rank (p // COLS) // ROWS,
    # row (p // COLS) % ROWS, column p % COLS.
    w = w_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    s = s_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    offs = tl.arange(0, DIM)
    lo = tl.load(owned_pos_ptr)
    hi = tl.load(owned_pos_ptr + 1)
    for pos in range(lo + tl.program_id(0), hi, num_programs):
        local = tl.load(sorted_ids_ptr + pos).to(tl.int64) - row_lo
        src = tl.load(src_pos_ptr + pos).to(tl.int64)
        g = src // COLS
        base = tl.load(dst_table_ptr + g // ROWS).to(tl.pointer_type(tl.bfloat16))
        vals = tl.load(w + local * DIM + offs).to(tl.float32)
        exps = tl.load(s + local * (DIM // BLK) + offs // BLK).to(tl.int32)
        # 2**(e - 127) from the exponent bits, as engram_gather computes it.
        scale = (exps << 23).to(tl.float32, bitcast=True)
        scale = tl.where(exps == 0, E8M0_ZERO, scale)
        out = vals * scale + 0.0
        dst = base + ((g % ROWS) * COLS + src % COLS) * DIM + offs
        tl.store(dst, out.to(tl.bfloat16))


def engram_owner_scatter(
    weight_ptr: int,
    scale_ptr: int,
    ids: torch.Tensor,
    dst_table: torch.Tensor,
    dim: int,
    block_size: int,
    row_lo: int,
    row_hi: int,
    *,
    rows: int,
    ids_fit_int32: bool,
    num_programs: int,
) -> None:
    """The engram lookup of ``ids`` ([W * rows, cols], every CP rank's ids
    rank-major) for the rows this rank's shard holds, ``[row_lo, row_hi)``, each
    written to its CP rank's buffer (``dst_table``: [W] int64 addresses of
    [rows, cols, dim] bf16 buffers) instead of a zero-filled local tensor.

    ``weight_ptr`` / ``scale_ptr`` address the shard's [row_hi - row_lo, dim] fp8
    e4m3 rows and their e8m0 block scales (device or host memory). The owned ids
    are read in sorted order (host-table translation reach), ``num_programs``
    persistent programs over them (sorted as int32 when ``ids_fit_int32``). Every
    id has one owner, so the W buffers end up holding the reduce-scatter of the
    sharded lookup.
    """
    assert dim > 0 and dim & (dim - 1) == 0 and dim % block_size == 0
    assert ids.dim() == 2 and ids.shape[0] == dst_table.shape[0] * rows
    n = ids.numel()
    if n == 0:
        return
    keys = ids.reshape(-1)
    sorted_ids, src_pos = torch.sort(keys.int() if ids_fit_int32 else keys.long())
    # Filled on the device: an H2D copy here would sync the host.
    bounds = torch.full((2,), row_lo, dtype=sorted_ids.dtype, device=ids.device)
    bounds[1:].fill_(row_hi)
    owned_pos = torch.searchsorted(sorted_ids, bounds)
    num_programs = max(1, min(n, num_programs))
    _engram_owner_scatter_kernel[(num_programs,)](
        weight_ptr,
        scale_ptr,
        sorted_ids,
        src_pos,
        dst_table,
        owned_pos,
        row_lo,
        num_programs,
        rows,
        COLS=ids.shape[1],
        DIM=dim,
        BLK=block_size,
        E8M0_ZERO=_E8M0_ZERO,
        num_warps=4,
    )
