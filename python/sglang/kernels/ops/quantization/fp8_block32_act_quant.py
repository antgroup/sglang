"""Block-32 ue8m0 activation quantization that writes DeepGEMM's SM90 scale layout.

sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True) writes row-major fp32
power-of-two scales, which deep_gemm.fp8_gemm_nt(recipe=(1, 32, 32)) then copies
into an MN-major, TMA-aligned buffer with a separate transpose launch. This kernel
writes that buffer directly (and the same FP8 values), so DeepGEMM and the swap-AB
GEMM use the scales as given. The per-group arithmetic repeats
per_token_group_quant_8bit_v2 (float32 scale storage, ue8m0 rounding) and is
bitwise identical to it.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl

# DeepGEMM's TMA needs 16-byte aligned scale columns: 4 fp32 rows.
_TMA_ALIGN_ROWS = 4


@triton.jit
def _block32_act_quant_ue8m0_kernel(
    X,
    Q,
    S,
    M,
    NUM_GROUPS,
    stride_xm,
    stride_qm,
    stride_sg,
    BLOCK_M: tl.constexpr,
    BLOCK_G: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    groups = tl.program_id(1) * BLOCK_G + tl.arange(0, BLOCK_G)
    cols = groups[None, :, None] * 32 + tl.arange(0, 32)[None, None, :]
    mask = (rows < M)[:, None] & (groups < NUM_GROUPS)[None, :]

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(
        X + rows[:, None, None] * stride_xm + cols, mask=mask[:, :, None], other=0.0
    ).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    # Same steps as the v2 CUDA kernel: amax floor 1e-10, e = log2_ceil(amax / 448)
    # from the fp32 bits, multiply by 2^-e, clamp, satfinite e4m3; store 2^e.
    amax = tl.maximum(tl.max(tl.abs(x), axis=2), 1e-10)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = ((127 - e) << 23).to(tl.float32, bitcast=True)
    scale_inv = ((e + 127) << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(x * scale[:, :, None], -448.0), 448.0)

    tl.store(
        Q + rows[:, None, None] * stride_qm + cols,
        q.to(tl.float8e4nv),
        mask=mask[:, :, None],
    )
    tl.store(S + rows[:, None] + groups[None, :] * stride_sg, scale_inv, mask=mask)


def _launch_shape(m: int, num_groups: int) -> Tuple[int, int, int]:
    # (BLOCK_M, BLOCK_G, num_warps), measured on H20-3e for K in 288..6144;
    # prefill rows: 16384 x 5120 101 -> 72 us, 2048 x 8192 22 -> 17 us.
    # The tiling does not change the values.
    if m >= 2048:
        return 8, 8, 2
    if m >= 192:
        return 8, 4, 1
    if m >= 24:
        return 8, 8, 4
    return 2, min(16, triton.next_power_of_2(num_groups)), 4


def block32_act_quant_ue8m0_mn_major(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """bf16/fp16 [M, K] -> (fp8 [M, K], fp32 power-of-two scales [M, K // 32]).

    The scales are MN-major with the row count padded to 4: storage
    [K // 32, align(M, 4)] viewed as [M, K // 32] with strides (1, align(M, 4)).
    """
    assert x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)
    assert x.dim() == 2 and x.is_contiguous(), "expects a contiguous 2D activation"
    m, k = x.shape
    assert k % 32 == 0, f"K={k} must be a multiple of 32"
    num_groups = k // 32
    aligned_m = triton.cdiv(m, _TMA_ALIGN_ROWS) * _TMA_ALIGN_ROWS
    q = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(
        (num_groups, aligned_m), device=x.device, dtype=torch.float32
    ).transpose(0, 1)[:m]
    if m == 0:
        return q, s
    block_m, block_g, num_warps = _launch_shape(m, num_groups)
    use_pdl = is_arch_support_pdl()
    grid = (triton.cdiv(m, block_m), triton.cdiv(num_groups, block_g))
    _block32_act_quant_ue8m0_kernel[grid](
        x,
        q,
        s,
        m,
        num_groups,
        x.stride(0),
        q.stride(0),
        s.stride(1),
        BLOCK_M=block_m,
        BLOCK_G=block_g,
        num_warps=num_warps,
        USE_PDL=use_pdl,
        launch_pdl=use_pdl,
    )
    return q, s
