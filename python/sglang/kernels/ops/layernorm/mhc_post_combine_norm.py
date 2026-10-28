"""HC=4 sublayer-input collapse and RMSNorm in one launch (DeepSeek-V4.1, H=5120).

``mhc_post_combine_norm(x, residual, post, comb, pre, weight, eps)`` returns
``(R', rmsnorm(mhc_combine(R', pre)))`` with
``R' = mhc_post(x, residual, post, comb)``, bitwise ``mhc_post_combine`` followed by
``RMSNorm.forward_cuda`` (FlashInfer's CuTe-DSL kernel).
``mhc_combine_norm(residual, pre, weight, eps)`` is the same without the post-mix:
bitwise ``mhc_combine`` followed by ``RMSNorm.forward_cuda``.
"""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

# Rows per CTA and vector blocks loaded together; any values give the same bits.
# H20, 4096 rows post+combine+norm (cold L2): (2, 1) 148.6 us, (1, 5) 138.6 us.
_ROWS_PER_CTA = 1
_LOAD_GROUP = 5


@cache_once
def _module(rows_per_cta: int, load_group: int):
    args = make_cpp_args(rows_per_cta, load_group)
    return load_jit(
        "dsv41_mhc_collapse_norm",
        *args,
        cuda_files=["deepseek_v4/mhc_post_combine_norm.cuh"],
        cuda_wrappers=[
            ("post_combine_norm", f"MhcCollapseNorm<{args}>::post_combine_norm"),
            ("combine_norm", f"MhcCollapseNorm<{args}>::combine_norm"),
        ],
    )


def can_use_mhc_post_combine_norm(x, residual, post, comb, pre, weight) -> bool:
    return (
        x.is_cuda
        and x.dim() == 2
        and x.shape[1] == 5120
        and residual.shape == (x.shape[0], 4, 5120)
        and x.dtype == residual.dtype == weight.dtype == torch.bfloat16
        and post.dtype == comb.dtype == pre.dtype == torch.float32
        and post.numel() == x.shape[0] * 4
        and comb.shape == (x.shape[0], 4, 4)
        and pre.shape == (x.shape[0], 4)
        and weight.shape == (5120,)
        and all(t.is_contiguous() for t in (x, residual, post, comb, weight))
        and all(t.data_ptr() % 16 == 0 for t in (x, residual, weight))
    )


@register_custom_op(mutates_args=["residual_out", "normalized"])
def _post_combine_norm_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    weight: torch.Tensor,
    residual_out: torch.Tensor,
    normalized: torch.Tensor,
    eps: float,
) -> None:
    _module(_ROWS_PER_CTA, _LOAD_GROUP).post_combine_norm(
        x, residual, post, comb, pre, weight, residual_out, normalized, eps
    )


def mhc_post_combine_norm(x, residual, post, comb, pre, weight, eps: float):
    assert can_use_mhc_post_combine_norm(x, residual, post, comb, pre, weight)
    residual_out = torch.empty_like(residual)
    normalized = torch.empty_like(x)
    _post_combine_norm_op(
        x,
        residual,
        post.view(x.shape[0], 4),
        comb,
        pre,
        weight,
        residual_out,
        normalized,
        eps,
    )
    return residual_out, normalized


def can_use_mhc_combine_norm(residual, pre, weight) -> bool:
    return (
        residual.is_cuda
        and residual.dim() == 3
        and residual.shape[1:] == (4, 5120)
        and residual.dtype == weight.dtype == torch.bfloat16
        and pre.dtype == torch.float32
        and pre.shape == (residual.shape[0], 4)
        and weight.shape == (5120,)
        and residual.is_contiguous()
        and weight.is_contiguous()
        and residual.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    )


@register_custom_op(mutates_args=["normalized"])
def _combine_norm_op(
    residual: torch.Tensor,
    pre: torch.Tensor,
    weight: torch.Tensor,
    normalized: torch.Tensor,
    eps: float,
) -> None:
    _module(_ROWS_PER_CTA, _LOAD_GROUP).combine_norm(
        residual, pre, weight, normalized, eps
    )


def mhc_combine_norm(residual, pre, weight, eps: float):
    assert can_use_mhc_combine_norm(residual, pre, weight)
    normalized = residual.new_empty(residual.shape[0], 5120)
    _combine_norm_op(residual, pre, weight, normalized, eps)
    return normalized
