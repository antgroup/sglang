# SPDX-License-Identifier: Apache-2.0
# Adapted from NVIDIA FlashDreams TAEHV streaming decoder.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.utils.hf_file_utils import resolve_hf_file_reference
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

DEFAULT_TAEHV_CHECKPOINT_FILENAME = "lighttaew2_1.pth"


@dataclass
class TAEHVCache:
    dec_state: dict[int, torch.Tensor] = field(default_factory=dict)


def _conv(n_in: int, n_out: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, act_func: nn.Module):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in * 2, n_out),
            act_func,
            _conv(n_out, n_out),
            act_func,
            _conv(n_out, n_out),
        )
        self.skip = (
            nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        )
        self.act = act_func

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

    def cache_step(
        self, x: torch.Tensor, state: dict[int, torch.Tensor], batch: int
    ) -> torch.Tensor:
        key = id(self)
        bt, c, h, w = x.shape
        t = bt // batch
        x5 = x.view(batch, t, c, h, w)
        prev = state.get(key)
        if prev is None:
            prev = torch.zeros(batch, 1, c, h, w, dtype=x.dtype, device=x.device)
        past = torch.cat([prev, x5[:, :-1]], dim=1)
        out = self.forward(x, past.reshape(bt, c, h, w))

        value = x5[:, -1:]
        if prev.shape == value.shape:
            prev.copy_(value)
            state[key] = prev
        else:
            state[key] = value.clone()
        return out


class TGrow(nn.Module):
    def __init__(self, n_f: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _nt, c, h, w = x.shape
        return self.conv(x).reshape(-1, c, h, w)


class Decoder(nn.Module):
    def __init__(
        self,
        n_f: tuple[int, int, int, int],
        latent_channels: int,
        image_channels: int,
        patch_size: int,
        decoder_time_upscale: tuple[bool, bool],
        decoder_space_upscale: tuple[bool, bool, bool],
        act_func: nn.Module,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            Clamp(),
            _conv(latent_channels, n_f[0]),
            act_func,
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            MemBlock(n_f[0], n_f[0], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            _conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            MemBlock(n_f[1], n_f[1], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            _conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            MemBlock(n_f[2], n_f[2], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            _conv(n_f[2], n_f[3], bias=False),
            act_func,
            _conv(n_f[3], image_channels * patch_size**2),
        )

    def forward(
        self, z: torch.Tensor, state: dict[int, torch.Tensor], batch: int
    ) -> torch.Tensor:
        b, t, c, h, w = z.shape
        x = z.reshape(b * t, c, h, w)
        for block in self.blocks:
            if isinstance(block, MemBlock):
                x = block.cache_step(x, state, batch)
            else:
                x = block(x)
        bt, c_out, h_out, w_out = x.shape
        return x.reshape(b, bt // b, c_out, h_out, w_out)


def _extract_state_dict(checkpoint) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"TAEHV checkpoint must be a mapping, got {type(checkpoint)}")
    for key in ("state_dict", "model", "module"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


class TAEHV(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        *,
        decoder_time_upscale: tuple[bool, bool] = (True, True),
        decoder_space_upscale: tuple[bool, bool, bool] = (True, True, True),
        patch_size: int = 1,
        latent_channels: int = 16,
        channels: tuple[int, int, int, int] = (256, 128, 64, 64),
        clamp_output: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
        self.clamp_output = clamp_output
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1
        self.decoder = Decoder(
            n_f=channels,
            latent_channels=latent_channels,
            image_channels=self.image_channels,
            patch_size=patch_size,
            decoder_time_upscale=decoder_time_upscale,
            decoder_space_upscale=decoder_space_upscale,
            act_func=nn.ReLU(inplace=True),
        )
        self.load_from_checkpoint(checkpoint_path)

    def _transform_checkpoint_key(self, key: str) -> str:
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("taehv."):
            key = key[len("taehv.") :]
        if key.startswith("decoder."):
            parts = key.split(".", 2)
            if len(parts) == 3 and parts[1].isdigit():
                key = f"decoder.blocks.{parts[1]}.{parts[2]}"
        return key

    def _maybe_truncate_tgrow_weight(
        self, key: str, value: torch.Tensor
    ) -> torch.Tensor:
        parts = key.split(".")
        if len(parts) != 5 or parts[0:2] != ["decoder", "blocks"]:
            return value
        if parts[3:] != ["conv", "weight"] or not parts[2].isdigit():
            return value

        block = self.decoder.blocks[int(parts[2])]
        if not isinstance(block, TGrow):
            return value
        expected_shape = block.conv.weight.shape
        if value.shape == expected_shape:
            return value
        if (
            value.ndim == len(expected_shape)
            and value.shape[0] >= expected_shape[0]
            and value.shape[1:] == expected_shape[1:]
        ):
            return value[-expected_shape[0] :]
        return value

    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_path = resolve_hf_file_reference(
            checkpoint_path,
            default_filename=DEFAULT_TAEHV_CHECKPOINT_FILENAME,
            description="TAEHV checkpoint",
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        decoder_state_dict = {}
        for key, value in state_dict.items():
            key = self._transform_checkpoint_key(key)
            if not key.startswith("decoder."):
                continue
            decoder_state_dict[key] = self._maybe_truncate_tgrow_weight(key, value)

        incompatible = self.load_state_dict(decoder_state_dict, strict=False)
        if incompatible.missing_keys:
            preview = ", ".join(incompatible.missing_keys[:8])
            raise RuntimeError(
                "TAEHV checkpoint is missing decoder keys: "
                f"{preview}; total_missing={len(incompatible.missing_keys)}"
            )
        self.eval().requires_grad_(False)
        logger.info(
            "Loaded TAEHV decoder from %s with %d decoder tensors",
            checkpoint_path,
            len(decoder_state_dict),
        )

    def prepare_cache(self) -> TAEHVCache:
        return TAEHVCache()

    @torch.inference_mode()
    def decode(self, z: torch.Tensor, cache: TAEHVCache | None = None) -> torch.Tensor:
        if cache is None:
            cache = self.prepare_cache()
        first_decode = not cache.dec_state
        x = self.decoder(z, cache.dec_state, z.shape[0])
        if self.clamp_output:
            x = x.clamp(0, 1)
        if self.patch_size > 1:
            n, t, c, h, w = x.shape
            x = F.pixel_shuffle(x.reshape(n * t, c, h, w), self.patch_size)
            x = x.reshape(n, t, x.shape[1], x.shape[2], x.shape[3])
        if first_decode:
            x = x[:, self.frames_to_trim :]
        return x


class LingBotTAEHVDecoder(nn.Module):
    def __init__(self, checkpoint_path: str, *, dtype: torch.dtype):
        super().__init__()
        self.decode_dtype = dtype
        self.taehv = TAEHV(checkpoint_path)
        mean = torch.tensor(
            [
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            ],
            dtype=dtype,
        ).view(1, 1, 16, 1, 1)
        std = torch.tensor(
            [
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.9160,
            ],
            dtype=dtype,
        ).view(1, 1, 16, 1, 1)
        self.register_buffer("latent_mean", mean, persistent=False)
        self.register_buffer("latent_std", std, persistent=False)
        self.to(dtype=dtype)
        self.eval().requires_grad_(False)

    def prepare_cache(self) -> TAEHVCache:
        return self.taehv.prepare_cache()

    @torch.inference_mode()
    def causal_decode(
        self, latents: torch.Tensor, *, cache: TAEHVCache | None = None
    ) -> torch.Tensor:
        z = latents.permute(0, 2, 1, 3, 4).to(dtype=self.decode_dtype)
        z = z * self.latent_std + self.latent_mean
        frames = self.taehv.decode(z, cache=cache)
        frames = frames.mul(2).sub(1)
        return frames.permute(0, 2, 1, 3, 4).contiguous()
