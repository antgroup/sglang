"""Per-32 power-of-two FP8 rounding of the DeepSeek-V4.1 wo_a input.

The V4.1 release reference rounds the attention output through FP8 e4m3 before
its grouped wo_a projection: per 32 values, scale = 2^ceil(log2(max(amax, 1e-4)
/ 448)) taken from the fp32 bits (exact powers of two included), and q =
e4m3(clamp(x / scale, -448, 448)). ``fp8_block32_quant_dequant`` returns q * scale
in bf16, the input of the BF16 projection; ``fp8_block32_quant_mn_major`` keeps
q and the scales, the operands of the FP8 block-[32, 32] projection, with the
scales in DeepGEMM's MN-major layout. The amax floor and the division differ
from ``block32_act_quant_ue8m0_mn_major`` (floor 1e-10), so the two are not
interchangeable on small blocks.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

# Columns per program; sized like the round-trip kernel's 1024 values.
_QUANT_BLOCK = 1024
# Rows from which the quant multiplies by the exact reciprocal with 2 warps,
# bitwise the division; H20-3e 2048 rows: 94 -> 71 us (4096: 186 -> 141 us).
_RECIP_MIN_ROWS = 1024


def fp8_block32_quant_dequant_native(x: torch.Tensor) -> torch.Tensor:
    """Torch reference of ``fp8_block32_quant_dequant`` for finite inputs."""
    values = x.float().reshape(-1, 32)
    amax = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    bits = (amax * (1.0 / 448.0)).view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()
    scale = (exponent << 23).view(torch.float32)
    quantized = (values / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (quantized.float() * scale).to(x.dtype).reshape(x.shape)


@triton.jit
def _fp8_block32_qdq_kernel(X, Y, N, GROUPS: tl.constexpr):
    groups = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    offsets = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + offsets, offsets < N, other=0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-4)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    q = tl.div_rn(x, scale[:, None])
    q = tl.minimum(tl.maximum(q, -448.0), 448.0).to(tl.float8e4nv)
    y = q.to(tl.float32) * scale[:, None]
    tl.store(Y + offsets, y.to(tl.bfloat16), offsets < N)


def fp8_block32_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """bf16 [..., D] (D % 32 == 0) -> a new contiguous bf16 tensor rounded via FP8."""
    assert x.dtype == torch.bfloat16 and x.dim() > 0 and x.shape[-1] % 32 == 0
    x = x.contiguous()
    out = torch.empty_like(x)
    if x.numel():
        groups = _QUANT_BLOCK // 32
        _fp8_block32_qdq_kernel[(triton.cdiv(x.numel(), _QUANT_BLOCK),)](
            x, out, x.numel(), groups, num_warps=4
        )
    return out


def mn_major_scale_buffer(
    num_rows: int, num_blocks: int, device: torch.device
) -> torch.Tensor:
    """fp32 [num_rows, num_blocks] with strides (1, num_rows padded to 4): the
    MN-major, 16-byte aligned activation scales DeepGEMM and swap-AB read in place."""
    padded = triton.cdiv(num_rows, 4) * 4
    storage = torch.empty((num_blocks, padded), dtype=torch.float32, device=device)
    return storage.t()[:num_rows]


@triton.jit
def _fp8_block32_quant_kernel(
    X,
    Y,
    S,
    SX_T,
    S_T,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    RECIP: tl.constexpr,
):
    row, chunk = tl.program_id(0), tl.program_id(1)
    NB: tl.constexpr = BLOCK // 32
    blocks = chunk * NB + tl.arange(0, NB)
    offsets = blocks[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + row * SX_T + offsets).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-4)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    if RECIP:
        # scale is a power of two, so 1 / scale is exact and x times it rounds the
        # same real quotient as the division.
        recip = tl.div_rn(tl.full(scale.shape, 1.0, tl.float32), scale)
        q = x * recip[:, None]
    else:
        q = tl.div_rn(x, scale[:, None])
    q = tl.minimum(tl.maximum(q, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Y + row * N + offsets, q)
    tl.store(S + blocks * S_T + row, scale)


def fp8_block32_quant_mn_major(o: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """o [T, G, D] bf16 (rows may be strided, G * D % 1024 == 0) -> (e4m3 [T, G * D],
    fp32 scales [T, G * D / 32] from ``mn_major_scale_buffer``)."""
    assert o.dtype == torch.bfloat16 and o.dim() == 3
    if not (o.stride(2) == 1 and o.stride(1) == o.shape[2]):
        o = o.contiguous()
    t = o.shape[0]
    n = o.shape[1] * o.shape[2]
    assert n % _QUANT_BLOCK == 0
    out = torch.empty((t, n), dtype=torch.float8_e4m3fn, device=o.device)
    scale = mn_major_scale_buffer(t, n // 32, o.device)
    if t:
        recip = t >= _RECIP_MIN_ROWS
        _fp8_block32_quant_kernel[(t, n // _QUANT_BLOCK)](
            o,
            out,
            scale,
            o.stride(0),
            scale.stride(1),
            N=n,
            BLOCK=_QUANT_BLOCK,
            RECIP=recip,
            num_warps=2 if recip else 4,
        )
    return out, scale
