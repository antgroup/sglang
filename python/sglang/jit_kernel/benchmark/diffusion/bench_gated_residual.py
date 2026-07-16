"""Benchmark LingBot's gated residual update at its production tensor shape."""

import torch
import triton.testing

from sglang.jit_kernel.benchmark.utils import run_benchmark_no_cudagraph
from sglang.jit_kernel.gated_residual import gated_residual
from sglang.multimodal_gen.runtime.layers.layernorm import (
    ScaleResidualLayerNormScaleShift,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-benchmark-1-gpu-large")

BATCH_SIZE = 1
SEQUENCE_LENGTH = 585
HIDDEN_SIZE = 5120
NUM_FRAMES = 1
DTYPE = torch.bfloat16
DEVICE = "cuda"
EPS = 1e-6

PROVIDERS = [
    "jit_alloc",
    "jit_preallocated",
    "existing_fused_residual_norm",
    "torch",
]
LINE_NAMES = [
    "JIT Gated Residual (allocate)",
    "JIT Gated Residual (preallocated)",
    "Existing Fused Residual + LayerNorm",
    "PyTorch FP32 Reference",
]
STYLES = [("blue", "-"), ("cyan", "--"), ("green", "-."), ("red", ":")]


def torch_reference(
    residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    tokens_per_frame = SEQUENCE_LENGTH // NUM_FRAMES
    expanded_gate = gate.expand(
        BATCH_SIZE, NUM_FRAMES, tokens_per_frame, HIDDEN_SIZE
    ).reshape_as(residual)
    return (residual.float() + x.float() * expanded_gate).to(residual.dtype)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "S", "D", "F"],
        x_vals=[(BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE, NUM_FRAMES)],
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="lingbot-gated-residual-b1-s585-d5120",
        args={},
    )
)
def benchmark_gated_residual(
    B: int, S: int, D: int, F: int, provider: str
) -> tuple[float, float, float]:
    residual = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE)
    x = torch.randn_like(residual)
    gate = torch.randn(B, F, 1, D, device=DEVICE, dtype=torch.float32)

    if provider == "jit_alloc":

        def fn():
            return gated_residual(residual, x, gate)

    elif provider == "jit_preallocated":
        out = torch.empty_like(residual)

        def fn():
            return gated_residual(residual, x, gate, out=out)

    elif provider == "existing_fused_residual_norm":
        layer = ScaleResidualLayerNormScaleShift(
            D,
            eps=EPS,
            elementwise_affine=True,
            dtype=torch.float32,
        ).to(DEVICE)
        layer.requires_grad_(False)
        null_shift = torch.zeros(1, device=DEVICE, dtype=DTYPE)
        null_scale = torch.zeros(1, device=DEVICE, dtype=DTYPE)

        def fn():
            return layer.forward_cuda(residual, x, gate, null_shift, null_scale)

    else:

        def fn():
            return torch_reference(residual, x, gate)

    return run_benchmark_no_cudagraph(fn)


if __name__ == "__main__":
    benchmark_gated_residual.run(print_data=True)
