from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


@cache_once
def _jit_gated_residual_module(dtype: torch.dtype) -> Module:
    """Compile and cache the gated-residual module for one input dtype."""
    args = make_cpp_args(dtype)
    return load_jit(
        "gated_residual",
        *args,
        cuda_files=["elementwise/gated_residual.cuh"],
        cuda_wrappers=[
            ("gated_residual", f"GatedResidualKernel<{args}>::run"),
        ],
    )


def _validate_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    ndim: int | None = None,
    require_contiguous: bool = True,
) -> None:
    if not tensor.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if require_contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if dtype is not None and tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(tensor.shape)}")


@debug_kernel_api
def gated_residual(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a frame-gated residual update in FP32 and cast back.

    The operation is::

        out[b, s, d] = (
            residual[b, s, d].float()
            + x[b, s, d].float() * gate[b, s // (S / F), 0, d]
        ).to(residual.dtype)

    ``residual`` and ``x`` must be contiguous CUDA tensors with shape
    ``[B, S, D]`` and dtype FP16 or BF16. ``gate`` must be CUDA FP32 with shape
    ``[B, F, D]`` or ``[B, F, 1, D]``, where ``S % F == 0``. Chunk/squeeze
    views used by LingBot are normalized to contiguous ``[B, F, 1, D]`` before
    launch. When supplied, ``out`` must not share storage with ``residual``,
    ``x``, or ``gate``. The D=5120 path is compile-time specialized and uses 128-bit
    vectorized loads/stores when every pointer is suitably aligned.
    """
    _validate_tensor("residual", residual, ndim=3)
    if residual.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"residual must have dtype float16 or bfloat16, got {residual.dtype}"
        )
    _validate_tensor("x", x, dtype=residual.dtype, ndim=3)
    _validate_tensor("gate", gate, dtype=torch.float32, require_contiguous=False)
    if gate.ndim == 3:
        gate = gate.unsqueeze(2)
    elif gate.ndim != 4:
        raise ValueError(
            "gate must be 3D [B, F, D] or 4D [B, F, 1, D], got "
            f"shape {tuple(gate.shape)}"
        )
    if not gate.is_contiguous():
        gate = gate.contiguous()

    if x.shape != residual.shape:
        raise ValueError(
            f"x shape {tuple(x.shape)} must match residual shape {tuple(residual.shape)}"
        )
    if x.device != residual.device or gate.device != residual.device:
        raise ValueError("residual, x, and gate must be on the same CUDA device")

    batch_size, sequence_length, hidden_size = residual.shape
    if batch_size <= 0 or sequence_length <= 0 or hidden_size <= 0:
        raise ValueError(
            f"residual dimensions must be positive, got {tuple(residual.shape)}"
        )
    if (
        gate.shape[0] != batch_size
        or gate.shape[2] != 1
        or gate.shape[3] != hidden_size
    ):
        raise ValueError(
            "gate must have shape [B, F, 1, D] matching residual; got "
            f"gate={tuple(gate.shape)}, residual={tuple(residual.shape)}"
        )
    num_frames = gate.shape[1]
    if num_frames <= 0 or sequence_length % num_frames != 0:
        raise ValueError(
            f"sequence length {sequence_length} must be divisible by "
            f"positive num_frames {num_frames}"
        )

    if out is None:
        out = torch.empty_like(residual)
    else:
        _validate_tensor("out", out, dtype=residual.dtype, ndim=3)
        if out.shape != residual.shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} must match residual shape "
                f"{tuple(residual.shape)}"
            )
        if out.device != residual.device:
            raise ValueError("out must be on the same CUDA device as residual")
        out_storage = out.untyped_storage().data_ptr()
        if out_storage in {
            residual.untyped_storage().data_ptr(),
            x.untyped_storage().data_ptr(),
            gate.untyped_storage().data_ptr(),
        }:
            raise ValueError("out must not share storage with residual, x, or gate")

    module = _jit_gated_residual_module(residual.dtype)
    module.gated_residual(out, residual, x, gate)
    return out
