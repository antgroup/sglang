"""Clamped SwiGLU with per-row FP8 quantization in one pass over the gate/up rows.

Bitwise ``act_and_mul_triton(swiglu_limit=...)`` (BF16 output) followed by Humming's
dynamic float8e4m3 input quant with one float32 scale per row
(``humming.ops.input.quant_input``, group_size 0); the expressions below are copied
from both kernels, so the same Triton build rounds them identically.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.fused_moe_triton_kernels import _apply_activation


@triton.jit
def _swiglu(
    gate_up_ptr,
    row,
    row_ok,
    stride,
    col0,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
):
    cols = col0 + tl.arange(0, BLOCK)
    mask = row_ok[:, None] & (cols < N)[None, :]
    gate_ptrs = gate_up_ptr + row[:, None] * stride + cols[None, :]
    gate = tl.load(gate_ptrs, mask=mask, other=0.0)
    up = tl.load(gate_ptrs + N, mask=mask, other=0.0)
    # act_and_mul_kernel, HAS_SWIGLU_LIMIT, silu.
    gate = tl.minimum(gate, SWIGLU_LIMIT)
    up = tl.maximum(tl.minimum(up, SWIGLU_LIMIT), -SWIGLU_LIMIT)
    act = _apply_activation(gate, "silu").to(tl.bfloat16)
    # Padding lanes hold +0 and cannot raise the row's absmax.
    return (act * up).to(tl.bfloat16).to(tl.float32), cols, mask


@triton.jit
def _silu_mul_fp8_row_quant_kernel(
    gate_up_ptr,
    q_ptr,
    scale_ptr,
    rows,
    stride_gate_up,
    stride_q,
    N: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
    ROWS: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
):
    # Each row is two column tiles, [0, BLOCK_A) and [BLOCK_A, N) padded to BLOCK_B,
    # so a row of 576 runs 512 + 64 lanes instead of 1024.
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    row_ok = row < rows
    a, cols_a, mask_a = _swiglu(
        gate_up_ptr, row, row_ok, stride_gate_up, 0, N, BLOCK_A, SWIGLU_LIMIT
    )
    absmax = tl.max(tl.abs(a), axis=1)
    if BLOCK_B > 0:
        b, cols_b, mask_b = _swiglu(
            gate_up_ptr, row, row_ok, stride_gate_up, BLOCK_A, N, BLOCK_B, SWIGLU_LIMIT
        )
        absmax = tl.maximum(absmax, tl.max(tl.abs(b), axis=1))
    # humming _quant_tensor_kernel: float8e4m3, dynamic, float32 scale.
    absmax = tl.maximum(absmax, 1e-30)
    scale = absmax / 448
    inv_scale = 1 / scale
    q_row = q_ptr + row[:, None] * stride_q
    q_a = (a * inv_scale[:, None]).to(tl.float8e4nv)
    tl.store(q_row + cols_a[None, :], q_a, mask=mask_a)
    if BLOCK_B > 0:
        q_b = (b * inv_scale[:, None]).to(tl.float8e4nv)
        tl.store(q_row + cols_b[None, :], q_b, mask=mask_b)
    tl.store(scale_ptr + row, scale, mask=row_ok)


def silu_mul_fp8_row_quant(
    gate_up: torch.Tensor,
    *,
    swiglu_limit: float,
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """gate_up [M, 2N] bf16 (gate | up) -> (out [M, N] float8_e4m3fn, scale [M, 1]
    float32), the down-projection input and its per-row scale."""
    assert gate_up.dtype == torch.bfloat16 and gate_up.stride(-1) == 1
    rows, two_n = gate_up.shape
    n = two_n // 2
    assert out.shape == (rows, n) and out.dtype == torch.float8_e4m3fn
    assert out.stride(-1) == 1
    scale = torch.empty((rows, 1), dtype=torch.float32, device=gate_up.device)
    if rows == 0:
        return out, scale
    block_a = 1 << (n.bit_length() - 1)
    block_b = triton.next_power_of_2(n - block_a) if n > block_a else 0
    # Measured on H20-3e at 98304 x 576: 4 rows x 4 warps 118 us; 1-16 rows and
    # 1-8 warps are no faster (SiLU's exp and divide bound it, not HBM).
    rows_per_program = 4
    _silu_mul_fp8_row_quant_kernel[(triton.cdiv(rows, rows_per_program),)](
        gate_up,
        out,
        scale,
        rows,
        gate_up.stride(0),
        out.stride(0),
        N=n,
        BLOCK_A=block_a,
        BLOCK_B=block_b,
        ROWS=rows_per_program,
        SWIGLU_LIMIT=float(swiglu_limit),
        num_warps=4,
    )
    return out, scale
