import sys

import pytest
import torch

from sglang.jit_kernel.gated_residual import gated_residual
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


def torch_reference(
    residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    if gate.ndim == 3:
        gate = gate.unsqueeze(2)
    batch_size, sequence_length, hidden_size = residual.shape
    num_frames = gate.shape[1]
    tokens_per_frame = sequence_length // num_frames
    expanded_gate = gate.expand(
        batch_size, num_frames, tokens_per_frame, hidden_size
    ).reshape_as(residual)
    return (residual.float() + x.float() * expanded_gate).to(residual.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch_size,num_frames,sequence_length,hidden_size",
    [
        (1, 1, 1, 8),
        (1, 2, 6, 16),
        (2, 3, 9, 128),
        (2, 5, 15, 37),  # exercises the scalar tail/fallback path
        (1, 3, 585, 5120),  # LingBot production shape
    ],
)
def test_gated_residual_correctness(
    dtype: torch.dtype,
    batch_size: int,
    num_frames: int,
    sequence_length: int,
    hidden_size: int,
) -> None:
    residual = torch.randn(
        batch_size, sequence_length, hidden_size, device="cuda", dtype=dtype
    )
    x = torch.randn_like(residual)
    gate = torch.randn(
        batch_size, num_frames, 1, hidden_size, device="cuda", dtype=torch.float32
    )

    actual = gated_residual(residual, x, gate)
    expected = torch_reference(residual, x, gate)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gated_residual_out_parameter(dtype: torch.dtype) -> None:
    residual = torch.randn(2, 12, 5120, device="cuda", dtype=dtype)
    x = torch.randn_like(residual)
    gate = torch.randn(2, 3, 1, 5120, device="cuda", dtype=torch.float32)
    out = torch.empty_like(residual)

    result = gated_residual(residual, x, gate, out=out)

    assert result is out
    torch.testing.assert_close(
        out, torch_reference(residual, x, gate), rtol=1e-2, atol=1e-2
    )


@pytest.mark.parametrize("squeeze_gate", [False, True])
def test_gated_residual_accepts_lingbot_gate_view(squeeze_gate: bool) -> None:
    residual = torch.randn(1, 9, 128, device="cuda", dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    modulation = torch.randn(1, 3, 6, 128, device="cuda", dtype=torch.float32)
    gate = modulation.chunk(6, dim=2)[2]
    if squeeze_gate:
        gate = gate.squeeze(2)
    assert not gate.is_contiguous()

    actual = gated_residual(residual, x, gate)
    expected = torch_reference(residual, x, gate)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_gated_residual_accepts_misaligned_contiguous_input() -> None:
    shape = (1, 6, 16)
    numel = torch.Size(shape).numel()
    residual_storage = torch.randn(numel + 1, device="cuda", dtype=torch.bfloat16)
    x_storage = torch.randn(numel + 1, device="cuda", dtype=torch.bfloat16)
    residual = residual_storage[1:].view(shape)
    x = x_storage[1:].view(shape)
    gate = torch.randn(1, 2, 1, 16, device="cuda", dtype=torch.float32)
    assert residual.is_contiguous() and x.is_contiguous()
    assert residual.data_ptr() % 16 != 0 and x.data_ptr() % 16 != 0

    actual = gated_residual(residual, x, gate)
    expected = torch_reference(residual, x, gate)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_gated_residual_rejects_cpu_input() -> None:
    residual = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(1, 1, 1, 8, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="CUDA"):
        gated_residual(residual, x, gate)


def test_gated_residual_rejects_unsupported_input_dtype() -> None:
    residual = torch.randn(1, 4, 8, device="cuda", dtype=torch.float32)
    x = torch.randn_like(residual)
    gate = torch.randn(1, 1, 1, 8, device="cuda", dtype=torch.float32)
    with pytest.raises(TypeError, match="float16 or bfloat16"):
        gated_residual(residual, x, gate)


def test_gated_residual_rejects_mismatched_x() -> None:
    residual = torch.randn(1, 6, 8, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(1, 5, 8, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(1, 2, 1, 8, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="x shape"):
        gated_residual(residual, x, gate)


def test_gated_residual_rejects_bad_gate_dtype() -> None:
    residual = torch.randn(1, 6, 8, device="cuda", dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(1, 2, 1, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="gate must have dtype"):
        gated_residual(residual, x, gate)


def test_gated_residual_rejects_bad_gate_shape() -> None:
    residual = torch.randn(1, 6, 8, device="cuda", dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(1, 2, 2, 8, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match=r"\[B, F, 1, D\]"):
        gated_residual(residual, x, gate)


def test_gated_residual_rejects_nondivisible_frames() -> None:
    residual = torch.randn(1, 7, 8, device="cuda", dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(1, 2, 1, 8, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="must be divisible"):
        gated_residual(residual, x, gate)


def test_gated_residual_rejects_noncontiguous_input() -> None:
    residual = torch.randn(1, 8, 16, device="cuda", dtype=torch.bfloat16)[..., ::2]
    x = torch.randn_like(residual)
    gate = torch.randn(1, 2, 1, 8, device="cuda", dtype=torch.float32)
    assert not residual.is_contiguous()
    with pytest.raises(ValueError, match="residual must be contiguous"):
        gated_residual(residual, x, gate)


def test_gated_residual_rejects_invalid_out() -> None:
    residual = torch.randn(1, 6, 8, device="cuda", dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(1, 2, 1, 8, device="cuda", dtype=torch.float32)
    out = torch.empty(1, 5, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="out shape"):
        gated_residual(residual, x, gate, out=out)


@pytest.mark.parametrize("alias", ["residual", "x", "gate"])
def test_gated_residual_rejects_aliased_out(alias: str) -> None:
    residual = torch.randn(1, 4, 8, device="cuda", dtype=torch.bfloat16)
    x = torch.randn_like(residual)
    gate = torch.randn(1, 2, 1, 8, device="cuda", dtype=torch.float32)
    aliases = {
        "residual": residual,
        "x": x,
        "gate": gate.view(torch.bfloat16).view_as(residual),
    }

    with pytest.raises(ValueError, match="must not share storage"):
        gated_residual(residual, x, gate, out=aliases[alias])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
