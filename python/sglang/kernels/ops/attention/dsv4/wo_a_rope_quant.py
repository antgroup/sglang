"""Inverse RoPE + per-32 FP8 rounding of the DeepSeek-V4.1 wo_a input, in one launch.

For ``o`` = the attention output's real heads [T, H, 512] (rope in the last 64,
possibly a strided view of the head-padded buffer), each function is bitwise

    fused_rope_inplace(o[..., -64:], None, freqs_cis, positions, inverse=True)

followed by ``fp8_block32_quant_mn_major(o.view(T, G, -1))`` (the ``quant`` forms)
or ``fp8_block32_quant_dequant`` (the ``qdq`` forms). The rotation repeats
deepseek_rope_kernel's inverse with NVCC's contraction (first product fused,
second rounded) and rounds to bf16; 32-value blocks never cross a head since
512 % 32 == 0. Unlike the replaced kernels they leave ``o`` unmodified. The
per-head form runs one program per (row, head); the ``rows`` form handles four
heads per program, for prefill-sized row counts.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_block32_qdq import mn_major_scale_buffer

# Columns per program of the rows kernel: 2048 (four heads) with 2 warps, the
# fastest on H20-3e at 4096 rows x 64 heads, within noise of 4096.
_ROWS_BLOCK = 2048


@triton.jit
def _wo_a_rope_qdq_kernel(
    X,
    F,
    POS,
    Y,
    S,
    SX_T,
    SX_H,
    S_T,
    H: tl.constexpr,
    D: tl.constexpr,
    ROPE: tl.constexpr,
    GROUP: tl.constexpr,
    QUANT: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    NB: tl.constexpr = D // GROUP
    r = tl.arange(0, NB)[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    base = X + row * SX_T + head * SX_H
    value = tl.load(base + r).to(tl.float32)
    partner = tl.load(base + (r ^ 1)).to(tl.float32)
    position = tl.load(POS + row)
    is_rope = r >= D - ROPE
    pair = (r - (D - ROPE)) // 2
    cos = tl.load(F + position * ROPE + pair * 2, is_rope, 0)
    sin = tl.load(F + position * ROPE + pair * 2 + 1, is_rope, 0)
    # deepseek_rope_kernel, inverse: real = xr*c + xi*s, imag = xi*c - xr*s.
    real = tl.fma(value, cos, partner * sin)  # even r: value = xr, partner = xi
    # IEEE negation: Triton's unary minus is 0 - x and would turn -0 into +0.
    imag = tl.fma(value, cos, (partner * sin) * -1.0)  # odd r: value = xi
    rotated = tl.where((r & 1) == 0, real, imag).to(tl.bfloat16).to(tl.float32)
    x = tl.where(is_rope, rotated, value)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-4)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    q = tl.div_rn(x, scale[:, None])
    q = tl.minimum(tl.maximum(q, -448.0), 448.0).to(tl.float8e4nv)
    if QUANT:
        tl.store(Y + row * (H * D) + head * D + r, q)
        tl.store(S + (head * NB + tl.arange(0, NB)) * S_T + row, scale)
    else:
        y = q.to(tl.float32) * scale[:, None]
        tl.store(Y + row * (H * D) + head * D + r, y.to(tl.bfloat16))


def can_use_wo_a_rope_quant(
    o: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor
) -> bool:
    return (
        o.is_cuda
        and o.dim() == 3
        and o.shape[2] == 512
        and o.dtype == torch.bfloat16
        and o.stride(2) == 1
        and o.stride(1) == 512
        and freqs_cis.dtype == torch.complex64
        and freqs_cis.is_contiguous()
        and freqs_cis.shape[1] == 32
        and positions.shape == (o.shape[0],)
        and positions.dtype in (torch.int32, torch.int64)
        and positions.is_contiguous()
    )


def _launch_per_head(o, freqs_cis, positions, out, scale, *, quant: bool) -> None:
    t, h, d = o.shape
    _wo_a_rope_qdq_kernel[(t, h)](
        o,
        torch.view_as_real(freqs_cis),
        positions,
        out,
        scale,
        o.stride(0),
        o.stride(1),
        scale.stride(1) if quant else 0,
        H=h,
        D=d,
        ROPE=64,
        GROUP=32,
        QUANT=quant,
        num_warps=4,
    )


def wo_a_rope_qdq(
    o: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """o [T, H, 512] bf16 -> contiguous rounded bf16 [T, H * 512]."""
    assert can_use_wo_a_rope_quant(o, freqs_cis, positions)
    t, h, d = o.shape
    out = torch.empty((t, h * d), dtype=o.dtype, device=o.device)
    if t and h:
        _launch_per_head(o, freqs_cis, positions, out, out, quant=False)
    return out


def wo_a_rope_quant_block32(
    o: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """o [T, H, 512] bf16 -> (e4m3 [T, H * 512], fp32 MN-major scales [T, H * 16])."""
    assert can_use_wo_a_rope_quant(o, freqs_cis, positions)
    t, h, d = o.shape
    out = torch.empty((t, h * d), dtype=torch.float8_e4m3fn, device=o.device)
    scale = mn_major_scale_buffer(t, h * d // 32, o.device)
    if t and h:
        _launch_per_head(o, freqs_cis, positions, out, scale, quant=True)
    return out, scale


@triton.jit
def _block32_quant_store(x, Y, S, row, blocks, offsets, N: tl.constexpr, S_T, store):
    # The quotient by the exact reciprocal of the power-of-two scale: bitwise div_rn.
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-4)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    recip = tl.div_rn(tl.full(scale.shape, 1.0, tl.float32), scale)
    q = tl.minimum(tl.maximum(x * recip[:, None], -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Y + row * N + offsets, q, store[:, None])
    tl.store(S + blocks * S_T + row, scale, store)


@triton.jit
def _block32_qdq_store(x, Y, row, offsets, N: tl.constexpr, store):
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-4)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    q = tl.div_rn(x, scale[:, None])
    q = tl.minimum(tl.maximum(q, -448.0), 448.0).to(tl.float8e4nv)
    y = q.to(tl.float32) * scale[:, None]
    tl.store(Y + row * N + offsets, y.to(tl.bfloat16), store[:, None])


@triton.jit
def _wo_a_rope_quant_rows_kernel(
    X,
    F,
    POS,
    Y,
    S,
    SX_T,
    S_T,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    DEQUANT: tl.constexpr,
):
    row, chunk = tl.program_id(0), tl.program_id(1)
    NB: tl.constexpr = BLOCK // 32
    HEADS: tl.constexpr = BLOCK // 512
    blocks = chunk * NB + tl.arange(0, NB)
    cols = tl.arange(0, 32)
    offsets = blocks[:, None] * 32 + cols[None, :]
    x = tl.load(X + row * SX_T + offsets).to(tl.float32)
    # Blocks 14 and 15 of every 512-wide head hold its rope columns; they are
    # quantized from the rotated values below.
    if DEQUANT:
        _block32_qdq_store(x, Y, row, offsets, N, (blocks % 16) < 14)
    else:
        _block32_quant_store(x, Y, S, row, blocks, offsets, N, S_T, (blocks % 16) < 14)
    rope = tl.arange(0, HEADS * 2)
    rope_blocks = chunk * NB + (rope // 2) * 16 + 14 + (rope % 2)
    rope_offsets = rope_blocks[:, None] * 32 + cols[None, :]
    value = tl.load(X + row * SX_T + rope_offsets).to(tl.float32)
    partner = tl.load(X + row * SX_T + (rope_offsets ^ 1)).to(tl.float32)
    position = tl.load(POS + row)
    pair = ((rope_offsets % 512) - 448) // 2
    cos = tl.load(F + position * 64 + pair * 2)
    sin = tl.load(F + position * 64 + pair * 2 + 1)
    # The expressions of _wo_a_rope_qdq_kernel.
    real = tl.fma(value, cos, partner * sin)
    imag = tl.fma(value, cos, (partner * sin) * -1.0)
    rotated = tl.where((cols[None, :] & 1) == 0, real, imag)
    rotated = rotated.to(tl.bfloat16).to(tl.float32)
    if DEQUANT:
        _block32_qdq_store(rotated, Y, row, rope_offsets, N, rope_blocks >= 0)
    else:
        _block32_quant_store(
            rotated, Y, S, row, rope_blocks, rope_offsets, N, S_T, rope_blocks >= 0
        )


def can_use_wo_a_rope_quant_rows(
    o: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor
) -> bool:
    return o.shape[1] % (_ROWS_BLOCK // 512) == 0 and can_use_wo_a_rope_quant(
        o, freqs_cis, positions
    )


def _launch_rows(o, freqs_cis, positions, out, scale, *, dequant: bool) -> None:
    t, h, d = o.shape
    n = h * d
    _wo_a_rope_quant_rows_kernel[(t, n // _ROWS_BLOCK)](
        o,
        torch.view_as_real(freqs_cis),
        positions,
        out,
        scale,
        o.stride(0),
        0 if dequant else scale.stride(1),
        N=n,
        BLOCK=_ROWS_BLOCK,
        DEQUANT=dequant,
        num_warps=2,
    )


def wo_a_rope_quant_block32_rows(
    o: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``wo_a_rope_quant_block32`` for many rows; H must be a multiple of 4."""
    assert can_use_wo_a_rope_quant_rows(o, freqs_cis, positions)
    t, h, d = o.shape
    out = torch.empty((t, h * d), dtype=torch.float8_e4m3fn, device=o.device)
    scale = mn_major_scale_buffer(t, h * d // 32, o.device)
    if t:
        _launch_rows(o, freqs_cis, positions, out, scale, dequant=False)
    return out, scale


def wo_a_rope_qdq_rows(
    o: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """``wo_a_rope_qdq`` with four heads per program; H must be a multiple of 4."""
    assert can_use_wo_a_rope_quant_rows(o, freqs_cis, positions)
    t, h, d = o.shape
    out = torch.empty((t, h * d), dtype=o.dtype, device=o.device)
    if t:
        _launch_rows(o, freqs_cis, positions, out, out, dequant=True)
    return out
