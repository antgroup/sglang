# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Copyright (c) 2025 DeepSeek
"""Manual EP8 MXFP4 checks and benchmark; CLI matches test_mega_moe_gluon.py.

The reference decodes the original checkpoint nibble and E8M0 values directly.
Processed packed weights and scales are independently decoded and compared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import torch

MXFP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode_checkpoint(packed, scale):
    import torch

    byte = packed.view(torch.uint8)
    codes = torch.stack((byte & 15, byte >> 4), dim=-1).flatten(-2)
    lut = torch.tensor(MXFP4_E2M1_VALUES, device=packed.device)
    value = lut[(codes & 7).long()] * torch.where(codes & 8 != 0, -1.0, 1.0)
    return value * torch.exp2(scale.float() - 127).repeat_interleave(32, -1)


def restore_mxfp4_scales(sf: torch.Tensor) -> torch.Tensor:
    """Restore the processed ``[E, K/128, N, 4]`` scale payload."""
    import torch

    if sf.ndim != 3 or sf.dtype != torch.uint8:
        raise ValueError("processed MXFP4 scale must be uint8 [E, N, K/32]")
    num_experts, num_rows, num_k32_groups = sf.shape
    if num_k32_groups % 4:
        raise ValueError("processed MXFP4 K/32 groups must be divisible by 4")
    return (
        sf.view(num_experts, num_k32_groups // 4, num_rows, 4)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view_as(sf)
    )


def deinterleave_mxfp4_gate_up_rows(
    tensor: torch.Tensor,
    granularity: int = 8,
) -> torch.Tensor:
    """Undo the FC1 ``gate8, up8`` row interleave used by MegaMoE."""
    import torch

    if tensor.ndim != 3 or tensor.shape[1] % (2 * granularity):
        raise ValueError("interleaved MXFP4 tensor has an invalid row shape")
    num_experts, num_rows, *tail = tensor.shape
    chunks = tensor.reshape(
        num_experts,
        num_rows // (2 * granularity),
        2,
        granularity,
        *tail,
    )
    return (
        torch.cat((chunks[:, :, 0], chunks[:, :, 1]), dim=1)
        .reshape_as(tensor)
        .contiguous()
    )


def restore_mxfp4_sign_bits(weight: torch.Tensor) -> torch.Tensor:
    """Undo packed-weight's packed-word sign permutation."""
    import torch

    if weight.dtype not in (torch.uint8, torch.int8) or weight.shape[-1] % 4:
        raise ValueError("processed MXFP4 weight has an invalid packed shape")
    original_dtype = weight.dtype
    source = weight.contiguous().view(torch.uint8).reshape(*weight.shape[:-1], -1, 4)
    restored = source & 0x77
    restored[..., 0] |= (source[..., 0] & 0x08) | ((source[..., 1] & 0x08) << 4)
    restored[..., 1] |= (source[..., 2] & 0x08) | ((source[..., 3] & 0x08) << 4)
    restored[..., 2] |= ((source[..., 0] & 0x80) >> 4) | (source[..., 1] & 0x80)
    restored[..., 3] |= ((source[..., 2] & 0x80) >> 4) | (source[..., 3] & 0x80)
    return restored.reshape(weight.shape).view(original_dtype)


def dequantize_processed_mxfp4(
    processed: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    fc1_gate_up_interleaved: bool = False,
) -> torch.Tensor:
    """Decode the processed SM90 MXFP4 ABI to FP32 for correctness checks."""
    import torch

    if not isinstance(processed, tuple) or len(processed) != 3:
        raise TypeError("processed MXFP4 weights must be a tensor triple")
    packed, sf, secondary = processed
    if packed.dtype != torch.int8 or sf.dtype != torch.uint8:
        raise ValueError("processed MXFP4 requires int8 weights and uint8 scales")
    if packed.ndim != 3 or sf.ndim != 3:
        raise ValueError("processed MXFP4 weights and scales must be 3D")
    if packed.shape[:2] != sf.shape[:2] or packed.shape[2] != sf.shape[2] * 16:
        raise ValueError("processed MXFP4 packed and scale shapes do not match")
    if secondary.dtype != torch.float32 or secondary.shape != (packed.shape[0],):
        raise ValueError("processed MXFP4 secondary scale must be float32 [E]")

    hidden = packed.shape[2] * 2 if fc1_gate_up_interleaved else packed.shape[1]
    logical_sf = restore_mxfp4_scales(sf) if hidden <= 8192 else sf
    logical_packed = restore_mxfp4_sign_bits(packed)
    if fc1_gate_up_interleaved:
        logical_packed = deinterleave_mxfp4_gate_up_rows(logical_packed)
        logical_sf = deinterleave_mxfp4_gate_up_rows(logical_sf)

    byte_values = logical_packed.view(torch.uint8)
    low = byte_values & 0x0F
    high = byte_values >> 4
    codes = torch.stack((low, high), dim=-1).flatten(-2)
    magnitude_lut = torch.tensor(
        MXFP4_E2M1_VALUES,
        dtype=torch.float32,
        device=packed.device,
    )
    magnitude_indices = (codes & 0x07).to(torch.long)
    magnitudes = magnitude_lut[magnitude_indices]
    values = torch.where((codes & 0x08) != 0, -magnitudes, magnitudes)
    relative = torch.exp2(logical_sf.to(torch.float32))
    scale = relative.repeat_interleave(32, dim=-1)
    return values * scale * secondary[:, None, None]


def make_mxfp4_weights(backend, shape, world, rank, seed):
    import torch

    h, intermediate, experts, _ = shape
    checkpoints = []
    for layer, (n, k) in enumerate(((2 * intermediate, h), (h, intermediate))):
        gen = torch.Generator(device="cuda").manual_seed(seed + rank * 1000003 + layer)
        packed = torch.randint(
            0,
            256,
            (experts // world, n, k // 2),
            dtype=torch.uint8,
            device="cuda",
            generator=gen,
        ).view(torch.int8)
        scales = torch.randint(
            120,
            128,
            (experts // world, n, k // 32),
            dtype=torch.uint8,
            device="cuda",
            generator=gen,
        )
        checkpoints.append((packed, scales))
    processed = backend.prepare_weights(*checkpoints)
    for layer in range(2):
        for e in range(experts // world):
            decoded = dequantize_processed_mxfp4(
                tuple(t[e : e + 1] for t in processed[layer]),
                fc1_gate_up_interleaved=layer == 0,
            )
            expected = decode_checkpoint(*(t[e : e + 1] for t in checkpoints[layer]))
            torch.testing.assert_close(decoded, expected, rtol=0, atol=0)

    def weight_at(expert):
        return tuple(
            decode_checkpoint(*(t[expert : expert + 1] for t in pair))[0]
            for pair in checkpoints
        )

    return processed, weight_at


from test_mega_moe_gluon import run_test


def main(argv=None):
    run_test("mxfp4", make_mxfp4_weights, argv)


if __name__ == "__main__":
    main()
