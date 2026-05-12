# SPDX-License-Identifier: Apache-2.0
"""Self-contained FlashVSR Tiny Long inference runner.

This module vendors only the model/runtime pieces needed by the official
``infer_flashvsr_v1.1_tiny_long_video.py`` path.  It deliberately avoids the
FlashVSR example project's ``ModelManager`` and ``BasePipeline`` wrappers so
SGLang only needs the v1.1 weights at runtime.
"""

from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

from .lq_projector import Causal_LQ4x_Proj
from .model_utils import init_weights_on_device
from .tc_decoder import build_tcdecoder
from .wan_video_dit import WanModel, WanModelStateDictConverter, sinusoidal_embedding_1d

_DIT_WEIGHT_KEYS = {
    "dim",
    "in_dim",
    "ffn_dim",
    "out_dim",
    "text_dim",
    "freq_dim",
    "eps",
    "patch_size",
    "num_heads",
    "num_layers",
    "has_image_input",
}


def load_flashvsr_tiny_long_runner(
    weights_dir: str,
    prompt_tensor_path: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> "FlashVSRTinyLongRunner":
    weights_dir = os.path.abspath(os.path.expanduser(weights_dir))
    dit_path = os.path.join(
        weights_dir, "diffusion_pytorch_model_streaming_dmd.safetensors"
    )
    lq_proj_path = os.path.join(weights_dir, "LQ_proj_in.ckpt")
    tc_decoder_path = os.path.join(weights_dir, "TCDecoder.ckpt")
    for required_path in (dit_path, lq_proj_path, tc_decoder_path, prompt_tensor_path):
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Missing FlashVSR Tiny Long file: {required_path}")

    device = torch.device(device)
    dit = _load_dit(dit_path, device=device, dtype=dtype)

    lq_proj = Causal_LQ4x_Proj(in_dim=3, out_dim=dit.dim, layer_num=1)
    lq_proj.load_state_dict(_torch_load(lq_proj_path), strict=True)
    dit.LQ_proj_in = lq_proj.to(device=device, dtype=dtype).eval()

    tc_decoder = build_tcdecoder(
        new_channels=[512, 256, 128, 128],
        new_latent_channels=16 + 768,
        device=str(device),
        dtype=dtype,
    )
    tc_decoder.load_state_dict(_torch_load(tc_decoder_path), strict=False)
    tc_decoder = tc_decoder.to(device=device, dtype=dtype).eval()

    runner = FlashVSRTinyLongRunner(
        dit=dit, tc_decoder=tc_decoder, device=device, dtype=dtype
    )
    context = torch.load(prompt_tensor_path, map_location=device)
    runner.init_cross_kv(context)
    return runner


class FlashVSRTinyLongRunner(nn.Module):
    def __init__(
        self,
        *,
        dit: WanModel,
        tc_decoder: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.dit = dit
        self.tc_decoder = tc_decoder
        self.device = device
        self.dtype = dtype
        self.color_corrector = TorchColorCorrectorWavelet(levels=5)
        self.prompt_context = None
        self.timestep = None
        self.t = None
        self.t_mod = None

    @torch.no_grad()
    def init_cross_kv(self, context_tensor: torch.Tensor):
        context_tensor = context_tensor.to(device=self.device, dtype=self.dtype)
        self.prompt_context = context_tensor
        self.dit.reinit_cross_kv(context_tensor)
        self.timestep = torch.tensor([1000.0], device=self.device, dtype=self.dtype)
        self.t = self.dit.time_embedding(
            sinusoidal_embedding_1d(self.dit.freq_dim, self.timestep)
        )
        self.t_mod = self.dit.time_projection(self.t).unflatten(1, (6, self.dit.dim))

    @torch.no_grad()
    def forward(
        self,
        *,
        lq_video: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        topk_ratio: float = 2.0,
        kv_ratio: float = 3.0,
        local_range: int = 11,
        color_fix: bool = True,
    ) -> torch.Tensor:
        if self.prompt_context is None:
            raise RuntimeError(
                "FlashVSR Tiny Long cross-attention KV is not initialized"
            )
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
        if height % 128 != 0 or width % 128 != 0:
            raise ValueError(
                "FlashVSR Tiny Long expects height and width to be multiples of 128, "
                f"got {height}x{width}."
            )

        lq_video = lq_video.to(device=self.device, dtype=self.dtype)
        latents = torch.randn(
            (1, 16, (num_frames - 1) // 4, height // 8, width // 8),
            device=self.device,
            dtype=self.dtype,
        )
        process_total_num = (num_frames - 1) // 8 - 2
        if process_total_num <= 0:
            raise ValueError(
                "FlashVSR Tiny Long requires at least 25 padded frames for one "
                f"streaming step, got {num_frames}."
            )

        self.dit.LQ_proj_in.clear_cache()
        self.tc_decoder.clean_mem()
        lq_pre_idx = 0
        lq_cur_idx = 0
        frames_total = []
        pre_cache_k = None
        pre_cache_v = None

        for cur_process_idx in range(process_total_num):
            if cur_process_idx == 0:
                pre_cache_k = [None] * len(self.dit.blocks)
                pre_cache_v = [None] * len(self.dit.blocks)
                lq_latents = self._collect_lq_latents(lq_video, start=None, loops=7)
                lq_cur_idx = 21
                cur_latents = latents[:, :, :6, :, :]
            else:
                start = cur_process_idx * 8 + 17
                lq_latents = self._collect_lq_latents(lq_video, start=start, loops=2)
                lq_cur_idx = cur_process_idx * 8 + 21
                cur_latents = latents[
                    :, :, 4 + cur_process_idx * 2 : 6 + cur_process_idx * 2
                ]

            noise_pred, pre_cache_k, pre_cache_v = model_fn_wan_video(
                self.dit,
                x=cur_latents,
                timestep=self.timestep,
                context=None,
                lq_latents=lq_latents,
                is_stream=True,
                pre_cache_k=pre_cache_k,
                pre_cache_v=pre_cache_v,
                topk_ratio=topk_ratio,
                kv_ratio=kv_ratio,
                cur_process_idx=cur_process_idx,
                t_mod=self.t_mod,
                t=self.t,
                local_range=local_range,
            )
            cur_latents = cur_latents - noise_pred
            cur_lq_frame = lq_video[:, :, lq_pre_idx:lq_cur_idx, :, :]
            cur_frames = (
                self.tc_decoder.decode_video(
                    cur_latents.transpose(1, 2),
                    parallel=False,
                    show_progress_bar=False,
                    cond=cur_lq_frame,
                )
                .transpose(1, 2)
                .mul_(2)
                .sub_(1)
            )

            if color_fix:
                cur_frames = self.color_corrector(
                    cur_frames.to(device=self.device),
                    cur_lq_frame,
                    clip_range=(-1, 1),
                    chunk_size=None,
                    method="adain",
                )

            frames_total.append(cur_frames.cpu())
            lq_pre_idx = lq_cur_idx

        return torch.cat(frames_total, dim=2)[0]

    def _collect_lq_latents(
        self, lq_video: torch.Tensor, *, start: int | None, loops: int
    ) -> list[torch.Tensor] | None:
        lq_latents = None
        for inner_idx in range(loops):
            if start is None:
                clip_start = max(0, inner_idx * 4 - 3)
                clip_end = (inner_idx + 1) * 4 - 3
            else:
                clip_start = start + inner_idx * 4
                clip_end = start + inner_idx * 4 + 4
            cur = self.dit.LQ_proj_in.stream_forward(
                lq_video[:, :, clip_start:clip_end, :, :].to(self.device)
            )
            if cur is None:
                continue
            if lq_latents is None:
                lq_latents = cur
            else:
                for layer_idx in range(len(lq_latents)):
                    lq_latents[layer_idx] = torch.cat(
                        [lq_latents[layer_idx], cur[layer_idx]], dim=1
                    )
        return lq_latents


def model_fn_wan_video(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor | None,
    *,
    lq_latents: list[torch.Tensor] | None = None,
    is_full_block: bool = False,
    is_stream: bool = False,
    pre_cache_k: list[torch.Tensor] | None = None,
    pre_cache_v: list[torch.Tensor] | None = None,
    topk_ratio: float = 2.0,
    kv_ratio: float = 3.0,
    cur_process_idx: int = 0,
    t_mod: torch.Tensor,
    t: torch.Tensor,
    local_range: int = 11,
):
    x, (f, h, w) = dit.patchify(x)

    win = (2, 8, 8)
    seqlen = f // win[0]
    local_num = seqlen
    window_size = win[0] * h * w // 128
    square_num = window_size * window_size
    topk = max(int(square_num * topk_ratio) - 1, 1)
    kv_len = max(int(kv_ratio), 1)

    if cur_process_idx == 0:
        freqs = (
            torch.cat(
                [
                    dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
    else:
        f_start = 4 + cur_process_idx * 2
        freqs = (
            torch.cat(
                [
                    dit.freqs[0][f_start : f_start + f]
                    .view(f, 1, 1, -1)
                    .expand(f, h, w, -1),
                    dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )

    for block_id, block in enumerate(dit.blocks):
        if lq_latents is not None and block_id < len(lq_latents):
            x = x + lq_latents[block_id]
        x, last_pre_cache_k, last_pre_cache_v = block(
            x,
            context,
            t_mod,
            freqs,
            f,
            h,
            w,
            local_num,
            topk,
            block_id=block_id,
            kv_len=kv_len,
            is_full_block=is_full_block,
            is_stream=is_stream,
            pre_cache_k=pre_cache_k[block_id] if pre_cache_k is not None else None,
            pre_cache_v=pre_cache_v[block_id] if pre_cache_v is not None else None,
            local_range=local_range,
        )
        if pre_cache_k is not None:
            pre_cache_k[block_id] = last_pre_cache_k
        if pre_cache_v is not None:
            pre_cache_v[block_id] = last_pre_cache_v

    x = dit.head(x, t)
    return dit.unpatchify(x, (f, h, w)), pre_cache_k, pre_cache_v


class TorchColorCorrectorWavelet(nn.Module):
    def __init__(self, levels: int = 5):
        super().__init__()
        self.levels = levels

    @staticmethod
    def _flatten_time(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch, _channels, frames, _height, _width = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, 3, x.shape[-2], x.shape[-1]
        )
        return y, batch, frames

    @staticmethod
    def _unflatten_time(y: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
        return y.reshape(batch, frames, 3, y.shape[-2], y.shape[-1]).permute(
            0, 2, 1, 3, 4
        )

    def forward(
        self,
        hq_image: torch.Tensor,
        lq_image: torch.Tensor,
        clip_range: tuple[float, float] = (-1.0, 1.0),
        method: Literal["wavelet", "adain"] = "wavelet",
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if hq_image.shape != lq_image.shape:
            raise ValueError(
                "Color correction expects HQ and LQ tensors with matching shapes"
            )
        if hq_image.dim() != 5 or hq_image.shape[1] != 3:
            raise ValueError(
                "Color correction expects tensors shaped as (B, 3, F, H, W)"
            )

        if chunk_size is None or chunk_size >= hq_image.shape[2]:
            hq4, batch, frames = self._flatten_time(hq_image)
            lq4, _batch, _frames = self._flatten_time(lq_image)
            out4 = _match_color(hq4, lq4, method=method, levels=self.levels)
            out4 = torch.clamp(out4, *clip_range)
            return self._unflatten_time(out4, batch, frames)

        outs = []
        for start in range(0, hq_image.shape[2], chunk_size):
            end = min(start + chunk_size, hq_image.shape[2])
            outs.append(
                self(
                    hq_image[:, :, start:end],
                    lq_image[:, :, start:end],
                    clip_range=clip_range,
                    method=method,
                    chunk_size=None,
                )
            )
        return torch.cat(outs, dim=2)


def _match_color(
    content: torch.Tensor,
    style: torch.Tensor,
    *,
    method: Literal["wavelet", "adain"],
    levels: int,
) -> torch.Tensor:
    if method == "wavelet":
        return _wavelet_reconstruct(content, style, levels=levels)
    if method == "adain":
        return _adain(content, style)
    raise ValueError(f"Unknown color correction method: {method}")


def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    var = feat.view(feat.shape[0], feat.shape[1], -1).var(dim=2, unbiased=False) + eps
    std = var.sqrt().view(feat.shape[0], feat.shape[1], 1, 1)
    mean = (
        feat.view(feat.shape[0], feat.shape[1], -1)
        .mean(dim=2)
        .view(feat.shape[0], feat.shape[1], 1, 1)
    )
    return mean, std


def _adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    style_mean, style_std = _calc_mean_std(style_feat)
    content_mean, content_std = _calc_mean_std(content_feat)
    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean


def _wavelet_reconstruct(
    content: torch.Tensor, style: torch.Tensor, *, levels: int = 5
) -> torch.Tensor:
    content_high, _content_low = _wavelet_decompose(content, levels=levels)
    _style_high, style_low = _wavelet_decompose(style, levels=levels)
    return content_high + style_low


def _wavelet_decompose(x: torch.Tensor, *, levels: int = 5):
    high = torch.zeros_like(x)
    low = x
    for level in range(levels):
        radius = 2**level
        blurred = _wavelet_blur(low, radius)
        high = high + (low - blurred)
        low = blurred
    return high, low


def _wavelet_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    base = torch.tensor(
        [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]],
        dtype=x.dtype,
        device=x.device,
    )
    weight = base.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    x_pad = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(
        x_pad, weight, stride=1, padding=0, dilation=radius, groups=x.shape[1]
    )


def _load_dit(path: str, *, device: torch.device, dtype: torch.dtype) -> WanModel:
    state_dict = _load_state_dict(path)
    model_state_dict, config = _convert_dit_state_dict(state_dict)
    init_kwargs = {
        key: value
        for key, value in config.items()
        if key in _DIT_WEIGHT_KEYS and key != "has_image_input"
    }
    init_kwargs["has_image_input"] = bool(config.get("has_image_input", False))
    try:
        with init_weights_on_device():
            model = WanModel(**init_kwargs).eval()
        model.load_state_dict(model_state_dict, assign=True)
    except TypeError:
        model = WanModel(**init_kwargs).eval()
        model.load_state_dict(model_state_dict)
    return model.to(device=device, dtype=dtype).eval()


def _convert_dit_state_dict(state_dict: dict[str, torch.Tensor]):
    converter = WanModelStateDictConverter()
    for convert in (converter.from_civitai, converter.from_diffusers):
        model_state_dict, config = convert(state_dict)
        if config:
            return model_state_dict, config
    raise ValueError(
        "Unrecognized FlashVSR DiT checkpoint. Expected the v1.1 Tiny Long "
        "diffusion_pytorch_model_streaming_dmd.safetensors file."
    )


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        return load_safetensors(path, device="cpu")
    return _torch_load(path)


def _torch_load(path: str):
    obj = torch.load(path, map_location="cpu")
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        return obj["state_dict"]
    return obj
