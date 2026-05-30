# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def _qknorm_across_heads_kernel(
    q,
    k,
    q_weight,
    k_weight,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    q_row_stride: tl.constexpr,
    k_row_stride: tl.constexpr,
    q_weight_stride: tl.constexpr,
    k_weight_stride: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < hidden

    q_ptrs = q + row * q_row_stride + offsets
    k_ptrs = k + row * k_row_stride + offsets
    qw_ptrs = q_weight + offsets * q_weight_stride
    kw_ptrs = k_weight + offsets * k_weight_stride

    q_vals = tl.load(q_ptrs, mask=mask, other=0.0).to(tl.float32)
    k_vals = tl.load(k_ptrs, mask=mask, other=0.0).to(tl.float32)
    q_w = tl.load(qw_ptrs, mask=mask, other=0.0).to(tl.float32)
    k_w = tl.load(kw_ptrs, mask=mask, other=0.0).to(tl.float32)

    q_var = tl.sum(q_vals * q_vals, axis=0) / hidden
    k_var = tl.sum(k_vals * k_vals, axis=0) / hidden
    q_vals = q_vals * tl.rsqrt(q_var + eps) * q_w
    k_vals = k_vals * tl.rsqrt(k_var + eps) * k_w

    tl.store(q_ptrs, q_vals, mask=mask)
    tl.store(k_ptrs, k_vals, mask=mask)


def _flattened_row_stride(tensor: torch.Tensor) -> int:
    if tensor.ndim == 1:
        return tensor.stride(0)
    if tensor.ndim > 2:
        expected_outer_stride = tensor.shape[-2] * tensor.stride(-2)
        for dim in range(tensor.ndim - 2):
            expected = expected_outer_stride
            for inner_dim in range(dim + 1, tensor.ndim - 2):
                expected *= tensor.shape[inner_dim]
            if tensor.stride(dim) != expected:
                raise ValueError("q/k tensors must have a regular flattened row layout")
    return tensor.stride(-2)


def fused_qknorm_across_heads_(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply in-place Q/K RMSNorm across the full hidden dimension.

    LingBotWorld applies RMSNorm before reshaping Q/K into heads. Existing
    fused QK norm kernels generally normalize per head_dim after reshape, so
    they do not preserve this model's original normalization axis.
    """

    if q.shape != k.shape:
        raise ValueError(f"q/k shape mismatch: {tuple(q.shape)} vs {tuple(k.shape)}")
    if q.ndim == 0:
        raise ValueError("q/k tensors must have at least one dimension")
    if not q.is_cuda or not k.is_cuda:
        raise ValueError("fused_qknorm_across_heads_ requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported q dtype: {q.dtype}")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("q/k last dimension must be contiguous")

    hidden = q.shape[-1]
    if hidden <= 0:
        return q, k
    if q_weight.numel() != hidden or k_weight.numel() != hidden:
        raise ValueError(
            f"weight size mismatch: hidden={hidden}, "
            f"q_weight={q_weight.numel()}, k_weight={k_weight.numel()}"
        )

    rows = q.numel() // hidden
    block_size = triton.next_power_of_2(hidden)
    if block_size > 131072:
        raise ValueError(f"hidden dimension too large for Triton qknorm: {hidden}")

    _qknorm_across_heads_kernel[(rows,)](
        q,
        k,
        q_weight,
        k_weight,
        rows,
        hidden,
        _flattened_row_stride(q),
        _flattened_row_stride(k),
        q_weight.stride(0),
        k_weight.stride(0),
        float(eps),
        block_size,
        num_warps=8,
    )
    return q, k
