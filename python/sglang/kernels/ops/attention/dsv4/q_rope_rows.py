"""Forward RoPE of DeepSeek-V4.1 queries in place, several heads per program.

Bitwise ``fused_rope_inplace(q[..., -64:], None, freqs_cis, positions)`` (the
expressions of deepseek_rope_kernel with NVCC's contraction and IEEE negation,
signed zeros included), with one program per (row, block of heads) instead of a
warp per (row, head).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _q_rope_rows_kernel(X, F, POS, SX, HEADS: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    heads = block * HEADS + tl.arange(0, HEADS)
    offsets = heads[:, None] * 512 + 448 + tl.arange(0, 64)[None, :]
    base = X + row * SX
    x = tl.load(base + offsets).to(tl.float32)
    position = tl.load(POS + row)
    pair = tl.arange(0, 32)
    cos = tl.load(F + position * 64 + pair * 2)[None, :]
    sin = tl.load(F + position * 64 + pair * 2 + 1)[None, :]
    real, imag = tl.split(tl.reshape(x, (HEADS, 32, 2)))
    # Triton's unary minus is 0 - x and would turn -0 into +0.
    even = tl.fma(real, cos, (imag * sin) * -1.0)
    odd = tl.fma(real, sin, imag * cos)
    tl.store(
        base + offsets, tl.reshape(tl.join(even, odd), (HEADS, 64)).to(tl.bfloat16)
    )


def can_use_q_rope_rows(
    q: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor
) -> bool:
    return (
        q.is_cuda
        and q.dim() == 3
        and q.shape[2] == 512
        and q.shape[1] % 8 == 0
        and q.dtype == torch.bfloat16
        and q.stride(2) == 1
        and q.stride(1) == 512
        and freqs_cis.dtype == torch.complex64
        and freqs_cis.is_contiguous()
        and freqs_cis.shape[1] == 32
        and positions.shape == (q.shape[0],)
        and positions.dtype in (torch.int32, torch.int64)
        and positions.is_contiguous()
    )


def q_rope_rows_inplace(
    q: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor
) -> None:
    """q [T, H, 512] bf16 (H % 8 == 0): forward RoPE of q[..., 448:] in place."""
    assert can_use_q_rope_rows(q, freqs_cis, positions)
    t, h, _ = q.shape
    if t == 0:
        return
    # 32 heads per program measured fastest at 4096 rows x 64 heads (H20-3e).
    heads = next(b for b in (32, 16, 8) if h % b == 0)
    _q_rope_rows_kernel[(t, h // heads)](
        q,
        torch.view_as_real(freqs_cis),
        positions,
        q.stride(0),
        HEADS=heads,
        num_warps=4,
    )
