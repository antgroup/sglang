"""Single-launch index arithmetic for the DeepSeek-V4.1 decode / verify metadata.
Pure integer arithmetic, so the results are bitwise those of the torch ops they
replace."""

import torch
import triton
import triton.language as tl


@triton.jit
def _expand_page_table_kernel(
    PT,
    OUT,
    stride_pt,
    out_cols,
    BLOCKS_PER_PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < out_cols
    page = tl.load(PT + row * stride_pt + col // BLOCKS_PER_PAGE, mask=mask)
    block = page.to(tl.int64) * BLOCKS_PER_PAGE + col % BLOCKS_PER_PAGE
    tl.store(OUT + row * out_cols + col, block.to(tl.int32), mask=mask)


def expand_page_table(page_table: torch.Tensor, blocks_per_page: int) -> torch.Tensor:
    """[bs, n] -> int32 [bs, n * blocks_per_page] with entry
    page_table[b, j // bpp] * bpp + j % bpp, computed in int64 and truncated
    like (page_table.to(int64) * bpp + arange(bpp)).to(int32)."""
    assert page_table.dim() == 2 and page_table.stride(1) == 1
    bs, n = page_table.shape
    out_cols = n * blocks_per_page
    out = torch.empty(bs, out_cols, dtype=torch.int32, device=page_table.device)
    if out.numel() == 0:
        return out
    block = 1024
    _expand_page_table_kernel[(bs, triton.cdiv(out_cols, block))](
        page_table,
        out,
        page_table.stride(0),
        out_cols,
        BLOCKS_PER_PAGE=blocks_per_page,
        BLOCK=block,
        num_warps=4,
    )
    return out
