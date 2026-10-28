"""Pack a vision grid with start/newline/end embeddings, without index syncs."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "width"])
def _pack_vision_span(
    F, START, NEWLINE, END, OUT, rows, width, D: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    inner = row - 1
    patch = (row > 0) & (row < rows - 1) & (inner % (width + 1) < width)
    src = (inner // (width + 1)) * width + inner % (width + 1)
    value = tl.load(F + src * D + col, patch & (col < D), 0)
    start = tl.load(START + col, (row == 0) & (col < D), 0)
    end = tl.load(END + col, (row == rows - 1) & (col < D), 0)
    newline = tl.load(
        NEWLINE + col, (~patch) & (row > 0) & (row < rows - 1) & (col < D), 0
    )
    value = tl.where(
        row == 0, start, tl.where(row == rows - 1, end, tl.where(patch, value, newline))
    )
    tl.store(OUT + row * D + col, value, col < D)


def pack_vision_span(
    features: torch.Tensor,
    start: torch.Tensor,
    newline: torch.Tensor,
    end: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Pack [height * width, hidden] features in reading order, one newline per row.

    The output is [2 + height * (width + 1), hidden]: ``start``, then each grid
    row followed by ``newline``, then ``end``; it owns its storage.
    """
    if not (
        height > 0
        and width > 0
        and features.is_cuda
        and features.ndim == 2
        and features.is_contiguous()
        and features.shape[0] == height * width
        and all(
            v.device == features.device
            and v.dtype == features.dtype
            and v.is_contiguous()
            and v.numel() == features.shape[1]
            for v in (start, newline, end)
        )
    ):
        raise ValueError("Unsupported vision span layout/grid")
    rows, dim = 2 + height * (width + 1), features.shape[1]
    out = torch.empty((rows, dim), dtype=features.dtype, device=features.device)
    with torch.cuda.device(features.device):
        _pack_vision_span[(rows, triton.cdiv(dim, 512))](
            features, start, newline, end, out, rows, width, dim, 512
        )
    return out
