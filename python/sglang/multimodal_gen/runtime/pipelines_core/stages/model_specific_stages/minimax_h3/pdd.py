# SPDX-License-Identifier: Apache-2.0
"""Validation for MiniMax-H3's offline-fused PDD output heads."""

import json

import torch


def shard_pdd_heads(heads, *, video_width, audio_width, hidden_size, tp_size, tp_rank):
    expected = {
        f"{name}.{kind}"
        for name in ("video_out", "audio_out")
        for kind in ("weight", "bias")
    }
    if set(heads) != expected:
        raise ValueError(
            f"PDD requires exactly {sorted(expected)}, got {sorted(heads)}"
        )
    steps = None
    result = {}
    for name, width in (("video_out", video_width), ("audio_out", audio_width)):
        weight, bias = heads[f"{name}.weight"], heads[f"{name}.bias"]
        if weight.ndim != 3 or bias.ndim != 2:
            raise ValueError(f"PDD {name} requires rank-3 weight and rank-2 bias")
        if steps is None:
            steps = weight.shape[0]
        if (
            steps <= 0
            or tuple(weight.shape) != (steps, width * tp_size, hidden_size)
            or tuple(bias.shape) != (steps, width * tp_size)
        ):
            raise ValueError(
                f"PDD {name} shape mismatch: {tuple(weight.shape)}, {tuple(bias.shape)}"
            )
        for kind, tensor in (("weight", weight), ("bias", bias)):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(
                    f"PDD {name}.{kind} must contain finite floating-point values"
                )
            result[f"{name}.{kind}"] = (
                tensor[:, tp_rank * width : (tp_rank + 1) * width].float().contiguous()
            )
    return result


def validate_pdd_schedule(steps, metadata, sigmas, *, warmup=False):
    for modality in ("video", "audio"):
        actual = torch.as_tensor(sigmas[modality], dtype=torch.float64, device="cpu")
        nfe = actual.numel() - 1
        if (not warmup and nfe != steps) or not 0 < nfe <= steps:
            raise ValueError(
                f"MiniMax-H3 PDD requires {steps} evaluations; use --num-inference-steps {steps + 1}, got {nfe} evaluations"
            )
        key = f"{modality}_sigmas"
        # Legacy fused files did not record their grid. They still get NFE checks.
        if key in metadata and not warmup:
            expected = torch.tensor(json.loads(metadata[key]), dtype=torch.float64)
            if expected.shape != actual.shape or not torch.allclose(
                actual, expected, atol=1e-6, rtol=1e-5
            ):
                raise ValueError(
                    f"PDD {modality} sigma grid differs from the grid used to fuse heads; regenerate heads with matching shifts"
                )


def project_pdd_head(heads, h: torch.Tensor, name: str, step: int) -> torch.Tensor:
    """Project with a temporary step-local copy; keep the source bank on CPU.

    The bank is a sidecar, outside module parameters/buffers and residency
    management. Never retain accelerator tensors in it across forwards.
    """
    stack = heads[f"{name}.weight"]
    if not 0 <= step < stack.shape[0]:
        raise ValueError(
            f"MiniMax-H3 PDD has {stack.shape[0]} fused heads but the loop is at "
            f"step {step}; run with --num-inference-steps {stack.shape[0] + 1} "
            "(H3 counts sigma grid points, so that is one more than the steps)."
        )
    weight = stack[step].to(device=h.device, dtype=h.dtype)
    bias = heads[f"{name}.bias"][step].to(device=h.device, dtype=h.dtype)
    return torch.nn.functional.linear(h, weight, bias)
