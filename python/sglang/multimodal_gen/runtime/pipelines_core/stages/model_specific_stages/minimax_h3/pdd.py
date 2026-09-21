# SPDX-License-Identifier: Apache-2.0
"""Online loading, fusion, and validation for MiniMax-H3 PDD adapters."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

PDD_CONFIG_METADATA_KEY = "sglang_pdd_config"


@dataclass(frozen=True)
class PDDModalityConfig:
    interval_sigmas: tuple[float, ...]
    groups: tuple[tuple[int, int], ...]
    weight_key: str
    bias_key: str

    @property
    def step_sigmas(self) -> list[float]:
        return [self.interval_sigmas[start] for start, _ in self.groups] + [
            self.interval_sigmas[-1]
        ]


@dataclass(frozen=True)
class PDDConfig:
    """Version 1: shared features, linear velocity heads, deterministic Euler.

    Groups are half-open interval ranges [start, stop), covering every interval
    exactly once in order. Both H3 modalities must have the same number of groups.
    Keys identify the tensors in the accompanying safetensors file.
    """

    modalities: dict[str, PDDModalityConfig]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PDDConfig:
        required = {
            "format_version",
            "prediction_type",
            "update_rule",
            "shared_backbone_features",
            "modalities",
        }
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError(f"PDD config requires exactly {sorted(required)}")
        if type(data["format_version"]) is not int or data["format_version"] != 1:
            raise ValueError("Unsupported PDD format_version; expected 1")
        if (
            data["prediction_type"] != "velocity"
            or data["update_rule"] != "euler_eta0"
            or data["shared_backbone_features"] is not True
        ):
            raise ValueError(
                "PDD supports only shared backbone features, velocity heads and euler_eta0"
            )
        raw = data["modalities"]
        if not isinstance(raw, dict) or set(raw) != {"video", "audio"}:
            raise ValueError("H3 PDD requires video and audio modalities")
        modalities = {}
        keys = set()
        for name, spec in raw.items():
            if not isinstance(spec, dict) or set(spec) != {
                "interval_sigmas",
                "groups",
                "head_keys",
            }:
                raise ValueError(f"Invalid PDD {name} config fields")
            sigma = spec["interval_sigmas"]
            if (
                not isinstance(sigma, list)
                or len(sigma) < 2
                or any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in sigma
                )
            ):
                raise ValueError(f"PDD {name} sigmas must be finite numbers")
            if (
                sigma[0] != 1.0
                or sigma[-1] != 0.0
                or any(a <= b for a, b in zip(sigma, sigma[1:]))
            ):
                raise ValueError(
                    f"PDD {name} sigmas must strictly decrease from 1 to 0"
                )
            groups = spec["groups"]
            if not isinstance(groups, list) or not groups:
                raise ValueError(f"PDD {name} groups must not be empty")
            end = 0
            for group in groups:
                if (
                    not isinstance(group, list)
                    or len(group) != 2
                    or any(type(x) is not int for x in group)
                ):
                    raise ValueError("PDD groups must be integer [start, stop) ranges")
                start, stop = group
                if start != end or stop <= start or stop > len(sigma) - 1:
                    raise ValueError(
                        "PDD groups must cover consecutive intervals without gaps or overlap"
                    )
                end = stop
            if end != len(sigma) - 1:
                raise ValueError("PDD groups must cover all intervals")
            head_keys = spec["head_keys"]
            if not isinstance(head_keys, dict) or set(head_keys) != {"weight", "bias"}:
                raise ValueError("PDD head_keys must specify weight and bias")
            for key in head_keys.values():
                if not isinstance(key, str) or not key or key in keys:
                    raise ValueError("PDD tensor keys must be nonempty and unique")
                keys.add(key)
            modalities[name] = PDDModalityConfig(
                tuple(map(float, sigma)),
                tuple(map(tuple, groups)),
                head_keys["weight"],
                head_keys["bias"],
            )
        if len(modalities["video"].groups) != len(modalities["audio"].groups):
            raise ValueError("PDD video/audio must have the same number of fused steps")
        return cls(modalities)

    @property
    def nfe(self) -> int:
        return len(self.modalities["video"].groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "prediction_type": "velocity",
            "update_rule": "euler_eta0",
            "shared_backbone_features": True,
            "modalities": {
                name: {
                    "interval_sigmas": list(spec.interval_sigmas),
                    "groups": [list(g) for g in spec.groups],
                    "head_keys": {"weight": spec.weight_key, "bias": spec.bias_key},
                }
                for name, spec in self.modalities.items()
            },
        }

    def with_canonical_keys(self) -> PDDConfig:
        data = self.to_dict()
        for name, spec in data["modalities"].items():
            spec["head_keys"] = {
                kind: f"{name}_out.{kind}" for kind in ("weight", "bias")
            }
        return self.from_dict(data)

    def request_sigmas(
        self, num_points: int, *, warmup: bool = False
    ) -> dict[str, list[float]]:
        if type(num_points) is not int or num_points <= 0:
            raise ValueError("PDD num_inference_steps must be a positive integer")
        if not warmup and num_points != self.nfe + 1:
            raise ValueError(f"PDD requires --num-inference-steps {self.nfe + 1}")
        # Server warmup defaults to one; execute at least one real interval.
        count = min(max(num_points, 2), self.nfe + 1) if warmup else num_points
        return {
            name: spec.step_sigmas[:count] for name, spec in self.modalities.items()
        }


def pdd_config_from_metadata(metadata: dict[str, str]) -> PDDConfig | None:
    if PDD_CONFIG_METADATA_KEY not in metadata:
        return None
    return PDDConfig.from_dict(json.loads(metadata[PDD_CONFIG_METADATA_KEY]))


def canonicalize_pdd_heads(
    heads: dict[str, torch.Tensor], config: PDDConfig
) -> dict[str, torch.Tensor]:
    result = {}
    for name, spec in config.modalities.items():
        count = len(spec.interval_sigmas) - 1
        if spec.weight_key not in heads or spec.bias_key not in heads:
            raise ValueError(f"Missing PDD {name} head tensors")
        weight, bias = heads[spec.weight_key], heads[spec.bias_key]
        if (
            weight.ndim != 3
            or bias.ndim != 2
            or weight.shape[:2] != bias.shape
            or weight.shape[0] != count
            or min(weight.shape) <= 0
        ):
            raise ValueError(f"PDD {name} tensor shapes do not match config")
        for kind, tensor in (("weight", weight), ("bias", bias)):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(
                    f"PDD {name} heads must be finite floating-point tensors"
                )
            result[f"{name}_out.{kind}"] = tensor
    return result


def shard_pdd_heads(
    heads: dict[str, torch.Tensor],
    *,
    video_width: int,
    audio_width: int,
    hidden_size: int,
    tp_size: int,
    tp_rank: int,
) -> dict[str, torch.Tensor]:
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


def validate_pdd_schedule(
    steps: int,
    metadata: dict[str, str],
    sigmas: Mapping[str, Sequence[float] | torch.Tensor],
    *,
    warmup: bool = False,
) -> None:
    config = pdd_config_from_metadata(metadata)
    if config is not None:
        if steps != config.nfe:
            raise ValueError("PDD head count disagrees with config")
        for name, spec in config.modalities.items():
            actual = torch.as_tensor(sigmas[name], dtype=torch.float64, device="cpu")
            expected = torch.tensor(spec.step_sigmas, dtype=torch.float64, device="cpu")
            if warmup and actual.ndim == 1 and 2 <= actual.numel() <= expected.numel():
                expected = expected[: actual.numel()]
            if actual.shape != expected.shape or not torch.allclose(
                actual, expected, atol=1e-6, rtol=1e-5
            ):
                raise ValueError(
                    f"PDD {name} schedule differs from the configured group boundaries"
                )
        return
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


def project_pdd_head(
    heads: dict[str, torch.Tensor], h: torch.Tensor, name: str, step: int
) -> torch.Tensor:
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


def legacy_pdd_config(
    num_steps: int, block_size: int, video_shift: float = 12.0, audio_shift: float = 3.0
) -> PDDConfig:
    """Interpret the original released adapter's metadata and shifted grids."""
    if (
        type(num_steps) is not int
        or type(block_size) is not int
        or num_steps <= 0
        or block_size <= 0
        or num_steps % block_size
    ):
        raise ValueError(
            "Legacy PDD steps must be positive and divisible by block size"
        )
    modalities = {}
    for name, prefix, shift in (
        ("video", "proj_out", video_shift),
        ("audio", "audio_proj_out", audio_shift),
    ):
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("PDD shift must be positive and finite")
        base = torch.linspace(
            1.0, 0.0, num_steps + 1, dtype=torch.float64, device="cpu"
        )
        modalities[name] = {
            "interval_sigmas": (shift * base / (1 + (shift - 1) * base)).tolist(),
            "groups": [
                [start, start + block_size] for start in range(0, num_steps, block_size)
            ],
            "head_keys": {"weight": f"{prefix}.weight", "bias": f"{prefix}.bias"},
        }
    return PDDConfig.from_dict(
        {
            "format_version": 1,
            "prediction_type": "velocity",
            "update_rule": "euler_eta0",
            "shared_backbone_features": True,
            "modalities": modalities,
        }
    )


def fuse_groups(
    heads: torch.Tensor, sigmas: torch.Tensor, groups: tuple[tuple[int, int], ...]
) -> torch.Tensor:
    out = []
    for start, stop in groups:
        widths = sigmas[start:stop] - sigmas[start + 1 : stop + 1]
        coefficients = (widths / widths.sum()).float()
        chunk = heads[start:stop].float()
        out.append(
            (
                chunk
                * coefficients.to(chunk.device).view(-1, *([1] * (chunk.dim() - 1)))
            ).sum(0)
        )
    return torch.stack(out)


def fuse_configured_heads(
    heads: dict[str, torch.Tensor], config: PDDConfig
) -> dict[str, torch.Tensor]:
    canonical = canonicalize_pdd_heads(heads, config)
    return {
        f"{name}_out.{kind}": fuse_groups(
            canonical[f"{name}_out.{kind}"],
            torch.tensor(spec.interval_sigmas, dtype=torch.float64, device="cpu"),
            spec.groups,
        )
        for name, spec in config.modalities.items()
        for kind in ("weight", "bias")
    }


def load_pdd_adapter(
    path: str, *, video_shift: float = 12.0, audio_shift: float = 3.0
) -> tuple[PDDConfig, dict[str, torch.Tensor], int | None] | None:
    """Return (config, fused CPU heads, alpha) for a local PDD adapter, else None.

    Versioned adapters embed their raw-head config in safetensors metadata.
    The original release is recognized by its four output-head tensors.
    """
    from pathlib import Path

    from safetensors import safe_open

    local = Path(path).expanduser()
    if not local.is_file() or local.suffix.lower() != ".safetensors":
        return None
    with safe_open(str(local), "pt", device="cpu") as f:
        metadata = f.metadata() or {}
        config = pdd_config_from_metadata(metadata)
        legacy_keys = {
            f"{prefix}.{kind}"
            for prefix in ("proj_out", "audio_proj_out")
            for kind in ("weight", "bias")
        }
        keys = set(f.keys())
        if config is None:
            if not keys & legacy_keys:
                return None
            if not legacy_keys <= keys:
                raise ValueError("PDD adapter is missing video/audio output heads")
            config = legacy_pdd_config(
                int(metadata.get("pdd_num_steps", 32)),
                int(metadata.get("pdd_block_size", 4)),
                video_shift,
                audio_shift,
            )
        required = {
            key
            for spec in config.modalities.values()
            for key in (spec.weight_key, spec.bias_key)
        }
        if not required <= keys:
            raise ValueError("PDD adapter is missing configured head tensors")
        heads = {key: f.get_tensor(key) for key in required}
        alpha = metadata.get("lora_alpha")
        if alpha is not None:
            value = float(alpha)
            if not math.isfinite(value) or value <= 0 or not value.is_integer():
                raise ValueError("PDD LoRA alpha must be a positive integer")
            alpha = int(value)
    return config, fuse_configured_heads(heads, config), alpha
