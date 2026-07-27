# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible import path for the LingBot QKNorm Triton kernel."""

from sglang.kernels.ops.diffusion.triton.qknorm_across_heads import (
    _flattened_row_stride as _flattened_row_stride,
)
from sglang.kernels.ops.diffusion.triton.qknorm_across_heads import (
    _qknorm_across_heads_kernel as _qknorm_across_heads_kernel,
)
from sglang.kernels.ops.diffusion.triton.qknorm_across_heads import (
    fused_qknorm_across_heads_ as fused_qknorm_across_heads_,
)

__all__ = ["fused_qknorm_across_heads_"]
