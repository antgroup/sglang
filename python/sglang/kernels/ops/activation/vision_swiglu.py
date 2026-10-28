"""Packed vision SwiGLU with the eager SiLU intermediate rounding."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit(do_not_specialize=["numel"])
def _vision_swiglu(X, Y, numel, D: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    mask = i < numel
    ix = (i // D) * (2 * D) + i % D
    gate = tl.load(X + ix, mask, 0).to(tl.float32)
    up = tl.load(X + ix + D, mask, 0).to(tl.float32)
    # Aten silu computes x / (1 + exp(-x)) and stores it in the input dtype;
    # keep that rounding point and the separate rounding of the multiply.
    activated = libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate))
    activated = activated.to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + i, activated * up, mask)


def vision_swiglu(x: torch.Tensor) -> torch.Tensor:
    """Return ``silu(x[:, :D]) * x[:, D:]`` for a contiguous [rows, 2 * D] input.

    Bitwise ``F.silu(gate) * up``: SiLU is rounded to the input dtype before
    the multiply, as the two eager kernels do.
    """
    if not (
        x.is_cuda
        and x.ndim == 2
        and x.is_contiguous()
        and x.dtype in (torch.bfloat16, torch.float32)
        and x.shape[-1] % 2 == 0
    ):
        raise ValueError("Unsupported vision SwiGLU layout/dtype")
    out = torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=x.dtype)
    if out.numel():
        with torch.cuda.device(x.device):
            _vision_swiglu[(triton.cdiv(out.numel(), 512),)](
                x, out, out.numel(), out.shape[-1], 512, enable_fp_fusion=False
            )
    return out
