"""Split-K grouped FP8 block-[32, 32] wo_a projection of DeepSeek-V4.1 for a few rows.

SM90 DeepGEMM has no split-K, so a few rows leave most SMs idle. This kernel
splits K = 4096 into 8 slices per (group, 64 output columns) tile and sums the
fp32 slice partials in slice order (deterministic).
"""

import torch
import triton
import triton.language as tl

_BLOCK_M, _BLOCK_N, _BLOCK_K, _SPLIT_K = 16, 64, 256, 8


@triton.jit
def _wo_a_small_rows_kernel(
    A,
    AS,
    B,
    BS,
    C,
    P,
    M,
    S_T,
    K: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # Power-of-two scales make the bf16 dequant of both FP8 tiles exact, so the bf16
    # dot over wide K tiles forms the same products as the 32-wide scaled FP8 dots.
    pid_n, pid_m, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    group = pid_n * BLOCK_N // R
    KB: tl.constexpr = BLOCK_K // 32
    NB: tl.constexpr = BLOCK_N // 32
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_begin = pid_k * (K // SPLIT_K)
    offs_k = k_begin + tl.arange(0, BLOCK_K)
    rows = offs_m < M
    a_ptrs = A + offs_m[:, None] * (N // R * K) + group * K + offs_k[None, :]
    b_ptrs = B + offs_n[:, None] * K + offs_k[None, :]
    k_blocks = k_begin // 32 + tl.arange(0, KB)
    as_ptrs = AS + (group * (K // 32) + k_blocks)[None, :] * S_T + offs_m[:, None]
    n_blocks = pid_n * NB + tl.arange(0, NB)
    bs_ptrs = BS + n_blocks[:, None] * (K // 32) + k_blocks[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(K // SPLIT_K // BLOCK_K):
        a = tl.load(a_ptrs, mask=rows[:, None], other=0.0).to(tl.float32)
        b = tl.load(b_ptrs).to(tl.float32)
        a_s = tl.load(as_ptrs, mask=rows[:, None], other=0.0)
        b_s = tl.load(bs_ptrs)
        a = tl.reshape(
            tl.reshape(a, (BLOCK_M, KB, 32)) * a_s[:, :, None], (BLOCK_M, BLOCK_K)
        )
        b = tl.reshape(
            tl.reshape(b, (NB, 32, KB, 32)) * b_s[:, None, :, None], (BLOCK_N, BLOCK_K)
        )
        acc = tl.dot(a.to(tl.bfloat16), tl.trans(b.to(tl.bfloat16)), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
        as_ptrs += KB * S_T
        bs_ptrs += KB
    tl.store(
        P + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=rows[:, None],
    )


@triton.jit
def _split_k_sum_kernel(P, C, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.load(P + offs, mask=mask, other=0.0)
    for split in tl.static_range(1, SPLIT_K):
        acc += tl.load(P + split * MN + offs, mask=mask, other=0.0)
    tl.store(C + offs, acc.to(tl.bfloat16), mask=mask)


def can_use_wo_a_block32_small_gemm(*, group_n: int, k: int) -> bool:
    return group_n % _BLOCK_N == 0 and k % (_SPLIT_K * _BLOCK_K) == 0


def wo_a_block32_small_gemm(
    o_fp8: torch.Tensor,
    o_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """o_fp8 [T, G * K] e4m3 with o_scale its MN-major fp32 view [T, G * K / 32];
    weight [G * R, K] e4m3 with weight_scale [G * R / 32, K / 32] fp32; writes
    out [T, G, R] bf16 contiguous."""
    t = o_fp8.shape[0]
    n, k = weight.shape
    r = out.shape[2]
    assert t > 0 and out.shape == (t, n // r, r) and out.is_contiguous()
    assert can_use_wo_a_block32_small_gemm(group_n=r, k=k)
    assert o_fp8.is_contiguous() and o_scale.stride(0) == 1
    partials = torch.empty((_SPLIT_K, t, n), dtype=torch.float32, device=o_fp8.device)
    grid = (n // _BLOCK_N, triton.cdiv(t, _BLOCK_M), _SPLIT_K)
    _wo_a_small_rows_kernel[grid](
        o_fp8,
        o_scale,
        weight,
        weight_scale,
        out,
        partials,
        t,
        o_scale.stride(1),
        K=k,
        R=r,
        N=n,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        SPLIT_K=_SPLIT_K,
        num_warps=4,
        num_stages=2,
    )
    _split_k_sum_kernel[(triton.cdiv(t * n, 1024),)](
        partials, out, t * n, SPLIT_K=_SPLIT_K, BLOCK=1024, num_warps=4
    )
