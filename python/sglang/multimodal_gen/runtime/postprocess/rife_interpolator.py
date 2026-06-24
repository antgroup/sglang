# SPDX-License-Identifier: Apache-2.0
"""
RIFE frame interpolation for SGLang diffusion pipelines.

RIFE model code is vendored and adapted from:
  - https://github.com/hzwer/ECCV2022-RIFE  (MIT License)
  - https://github.com/hzwer/Practical-RIFE  (MIT License)
  Copyright (c) 2021 Zhewei Huang

The FrameInterpolator wrapper and integration code are original work.
"""

import math
import os
from glob import glob
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Default HuggingFace repo for RIFE 4.22.lite weights
_DEFAULT_RIFE_HF_REPO = "elfgum/RIFE-4.22.lite"
_DEFAULT_RIFE_BLOCK_CHANNELS = (192, 128, 64, 32)
_RIFE_PATH_ENV_VARS = (
    "SGLANG_RIFE_MODEL_PATH",
    "SGLANG_FRAME_INTERPOLATION_MODEL_PATH",
    "SGLANG_LINGBOT_RIFE_MODEL_PATH",
    "SGLANG_RIFE_SR_ASSET_DIR",
)
_RIFE_LOCAL_CANDIDATES = (
    "Practical-RIFE/train_log",
    "train_log",
    ".",
)

# Module-level cache: model_path -> Model instance
_MODEL_CACHE: dict[str, "Model"] = {}


# ---------------------------------------------------------------------------
# Vendored RIFE model code
# (IFBlock, IFNet_HDv3 backbone, Model wrapper)
# ---------------------------------------------------------------------------


def warp(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    """Warp tenInput by tenFlow using grid_sample."""
    # Build base grid for the current size
    tenHorizontal = (
        torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device)
        .view(1, 1, 1, tenFlow.shape[3])
        .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
    )
    tenVertical = (
        torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device)
        .view(1, 1, tenFlow.shape[2], 1)
        .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
    )
    tenGrid = torch.cat([tenHorizontal, tenVertical], dim=1)

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        dim=1,
    )

    grid = (tenGrid + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=tenInput,
        grid=grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def _conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    """Conv2d + LeakyReLU helper (matches RIFE 4.22 conv())."""
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


class ResConv(nn.Module):
    """Residual convolution block with learnable beta scaling (RIFE 4.22)."""

    def __init__(self, c: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    """Single-scale optical flow + mask + feature block (RIFE 4.22)."""

    def __init__(self, in_planes: int, c: int = 64):
        super().__init__()
        self.conv0 = nn.Sequential(
            _conv(in_planes, c // 2, 3, 2, 1),
            _conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(
        self,
        x: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        if flow is not None:
            flow = (
                F.interpolate(
                    flow,
                    scale_factor=1.0 / scale,
                    mode="bilinear",
                    align_corners=False,
                )
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(
            tmp, scale_factor=scale, mode="bilinear", align_corners=False
        )
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat


class Head(nn.Module):
    """Feature encoder producing 4-channel features at full resolution (RIFE 4.22)."""

    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        return x3


class IFNet(nn.Module):
    """Multi-scale IFNet optical flow network."""

    def __init__(self, block_channels: tuple[int, ...] = _DEFAULT_RIFE_BLOCK_CHANNELS):
        super().__init__()
        self.block_channels = tuple(block_channels)
        for idx, channels in enumerate(self.block_channels):
            in_planes = 7 + 8 if idx == 0 else 8 + 4 + 8 + 8
            setattr(self, f"block{idx}", IFBlock(in_planes, c=channels))
        self.encode = Head()

    def _blocks(self) -> list[IFBlock]:
        return [getattr(self, f"block{i}") for i in range(len(self.block_channels))]

    def default_scale_list(self, scale: float = 1.0) -> list[float]:
        if len(self.block_channels) == 5:
            base = [32, 16, 8, 4, 1]
        elif len(self.block_channels) == 4:
            base = [8, 4, 2, 1]
        else:
            base = [2**i for i in range(len(self.block_channels) - 1, -1, -1)]
        return [value / scale for value in base]

    def forward(
        self,
        x: torch.Tensor,
        timestep: float = 0.5,
        scale_list: Optional[list] = None,
    ) -> tuple[list, torch.Tensor, list]:
        if scale_list is None:
            scale_list = self.default_scale_list()

        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]

        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])

        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None

        block = self._blocks()
        for i in range(len(block)):
            if flow is None:
                flow, mask, feat = block[i](
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                    None,
                    scale=scale_list[i],
                )
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, m0, feat = block[i](
                    torch.cat(
                        (
                            warped_img0[:, :3],
                            warped_img1[:, :3],
                            wf0,
                            wf1,
                            timestep,
                            mask,
                            feat,
                        ),
                        1,
                    ),
                    flow,
                    scale=scale_list[i],
                )
                mask = m0
                flow = flow + fd

            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))

        mask = torch.sigmoid(mask)
        merged[-1] = warped_img0 * mask + warped_img1 * (1 - mask)

        return flow_list, mask_list[-1], merged


class Model:
    """Wraps IFNet, provides load_model() and inference() API."""

    def __init__(self):
        self.flownet = IFNet()
        self.device_type: str = "cpu"

    def eval(self) -> "Model":
        self.flownet.eval()
        return self

    def device(self) -> torch.device:
        return next(self.flownet.parameters()).device

    def load_model(self, path: str, strip_module_prefix: bool = True) -> None:
        """Load weights from {path}/flownet.pkl.

        Args:
            path: Directory containing ``flownet.pkl``.
            strip_module_prefix: If True, strip the ``module.`` prefix that
                ``DataParallel`` / ``DistributedDataParallel`` adds to keys.
        """
        flownet_path = os.path.join(path, "flownet.pkl")
        if not os.path.isfile(flownet_path):
            raise FileNotFoundError(
                f"RIFE weight file not found: {flownet_path}\n"
                "Expected layout: <model_path>/flownet.pkl"
            )

        def convert(param):
            if not strip_module_prefix:
                return dict(param)
            return {
                (k.removeprefix("module.") if k.startswith("module.") else k): v
                for k, v in param.items()
            }

        state = torch.load(flownet_path, map_location="cpu", weights_only=False)
        state = convert(state)
        self._rebuild_flownet_for_state_dict(state)
        missing, unexpected = self.flownet.load_state_dict(state, strict=False)
        unexpected = [
            key for key in unexpected if not key.startswith(("teacher.", "caltime."))
        ]
        if missing or unexpected:
            logger.warning(
                "Loaded RIFE weights with missing keys=%s unexpected keys=%s",
                missing,
                unexpected,
            )
        logger.info("Loaded RIFE weights from %s", flownet_path)

    @staticmethod
    def _infer_block_channels(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
        channels = []
        for idx in range(16):
            key = f"block{idx}.conv0.1.0.weight"
            weight = state_dict.get(key)
            if weight is None:
                break
            channels.append(int(weight.shape[0]))
        return tuple(channels) or _DEFAULT_RIFE_BLOCK_CHANNELS

    def _rebuild_flownet_for_state_dict(self, state_dict: dict[str, torch.Tensor]):
        block_channels = self._infer_block_channels(state_dict)
        if block_channels == self.flownet.block_channels:
            return

        device = next(self.flownet.parameters()).device
        dtype = next(self.flownet.parameters()).dtype
        logger.info("Detected RIFE IFNet block channels: %s", block_channels)
        self.flownet = IFNet(block_channels=block_channels).to(
            device=device, dtype=dtype
        )

    @staticmethod
    def _pad_multiple_for_scale_list(scale_list: list[float]) -> int:
        # Each IFBlock downsamples twice internally and then upsamples back.
        # The largest external scale therefore needs input dimensions divisible
        # by scale * 4 to avoid round-off growth such as 960 -> 1024.
        return max(32, int(math.ceil(max(scale_list) * 4)))

    def inference(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        scale: float = 1.0,
        timestep: float = 0.5,
    ) -> torch.Tensor:
        """Interpolate a single intermediate frame between img0 and img1."""
        n, c, h, w = img0.shape

        scale_list = self.flownet.default_scale_list(scale)
        pad_multiple = self._pad_multiple_for_scale_list(scale_list)
        # Pad so that RIFE's downsample/upsample round-trips preserve spatial
        # dimensions exactly for both 4-block and 5-block IFNet variants.
        ph = ((h - 1) // pad_multiple + 1) * pad_multiple
        pw = ((w - 1) // pad_multiple + 1) * pad_multiple
        pad = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, pad)
        img1 = F.pad(img1, pad)

        imgs = torch.cat((img0, img1), 1)
        with torch.no_grad():
            flow_list, mask, merged = self.flownet(
                imgs,
                timestep=timestep,
                scale_list=scale_list,
            )

        # Crop back to original resolution
        return merged[-1][:, :, :h, :w]


# ---------------------------------------------------------------------------
# FrameInterpolator public class
# ---------------------------------------------------------------------------


class FrameInterpolator:
    """
    Lazy-loaded RIFE 4.22.lite frame interpolator.

    Weights are loaded on first call to `.interpolate()` and cached globally
    per model_path to avoid reloading across requests.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path
        self._resolved_path: Optional[str] = None

    def _ensure_model_loaded(self) -> Model:
        """Load RIFE model weights.

        Accepts a local directory **or** a HuggingFace repo ID.  When *None*
        (the default) the weights are downloaded (and cached) automatically
        from ``elfgum/RIFE-4.22.lite`` via ``maybe_download_model()``.
        """
        from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
            maybe_download_model,
        )

        model_path = self._model_path or _default_rife_model_path()

        # Resolve: local asset/checkpoint path pass-through, HF repo ID → download & cache
        if _looks_like_local_rife_path(model_path):
            model_path = _resolve_local_rife_dir(model_path)
        else:
            model_path = maybe_download_model(model_path)

        self._resolved_path = model_path

        if model_path in _MODEL_CACHE:
            return _MODEL_CACHE[model_path]

        device = current_platform.get_local_torch_device()
        model = Model()
        model.load_model(model_path, strip_module_prefix=True)
        model.eval()
        model.flownet = model.flownet.to(device)
        _MODEL_CACHE[model_path] = model
        logger.info("RIFE model loaded on device: %s", device)
        return model

    @staticmethod
    def _frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert uint8 HWC numpy frame to float32 CHW tensor on device."""
        t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return t.to(device)

    @staticmethod
    def _tensor_to_frame(t: torch.Tensor) -> np.ndarray:
        """Convert float32 CHW tensor (batch=1) to uint8 HWC numpy frame."""
        arr = t.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
        return (arr * 255.0).astype(np.uint8)

    def _make_inference(
        self, model: Model, I0: torch.Tensor, I1: torch.Tensor, n: int, scale: float
    ) -> list[torch.Tensor]:
        """
        Recursively generate n-1 intermediate frames between I0 and I1.

        Returns a list of intermediate frame tensors (not including I0 or I1).
        """
        if n == 1:
            return [model.inference(I0, I1, scale=scale)]
        mid = model.inference(I0, I1, scale=scale)
        return (
            self._make_inference(model, I0, mid, n // 2, scale)
            + [mid]
            + self._make_inference(model, mid, I1, n // 2, scale)
        )

    def interpolate(
        self,
        frames: list[np.ndarray],
        exp: int = 1,
        scale: float = 1.0,
    ) -> tuple[list[np.ndarray], int]:
        """
        Interpolate frames using RIFE.

        Args:
            frames: List of uint8 numpy arrays with shape [H, W, 3].
            exp:    Exponent for interpolation factor. 1 → 2×, 2 → 4×.
            scale:  RIFE inference scale. Use 0.5 for high-resolution inputs.

        Returns:
            (interpolated_frames, multiplier) where multiplier = 2**exp.
        """
        if len(frames) < 2:
            logger.warning(
                "Frame interpolation requires at least 2 frames; returning input unchanged."
            )
            return frames, 1

        model = self._ensure_model_loaded()
        device = model.device()

        if exp == 1 and all(frame.shape == frames[0].shape for frame in frames):
            input_tensors = torch.cat(
                [self._frame_to_tensor(frame, device) for frame in frames], dim=0
            ).contiguous()
            middle_tensors = model.inference(
                input_tensors[:-1],
                input_tensors[1:],
                scale=scale,
            )

            result: list[np.ndarray] = []
            for i in range(len(frames) - 1):
                result.append(frames[i])
                result.append(self._tensor_to_frame(middle_tensors[i : i + 1]))
            result.append(frames[-1])
            return result, 2

        n_intermediate = 2**exp // 2  # intermediates per adjacent pair

        result: list[np.ndarray] = []
        for i in range(len(frames) - 1):
            I0 = self._frame_to_tensor(frames[i], device)
            I1 = self._frame_to_tensor(frames[i + 1], device)

            intermediate_tensors = self._make_inference(
                model, I0, I1, n_intermediate, scale
            )

            result.append(frames[i])
            for t in intermediate_tensors:
                result.append(self._tensor_to_frame(t))

        result.append(frames[-1])
        multiplier = 2**exp
        return result, multiplier

    def interpolate_tensor(
        self,
        frames: torch.Tensor,
        exp: int = 1,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, int]:
        """
        Interpolate an NCHW RGB tensor without converting through numpy.

        Args:
            frames: Tensor with shape [N, 3, H, W], either float in [0, 1] or
                    uint8 in [0, 255].
            exp:    Exponent for interpolation factor. 1 → 2×, 2 → 4×.
            scale:  RIFE inference scale. Use 0.5 for high-resolution inputs.

        Returns:
            (interpolated_frames, multiplier) where frames are float32 NCHW on
            the RIFE model device and multiplier = 2**exp.
        """
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(
                f"expected NCHW RGB tensor with shape [N, 3, H, W], got {tuple(frames.shape)}"
            )
        if frames.shape[0] < 2 or exp <= 0:
            return frames, 1

        model = self._ensure_model_loaded()
        device = model.device()
        if frames.dtype == torch.uint8:
            frames = frames.to(device=device, dtype=torch.float32).mul(1.0 / 255.0)
        else:
            frames = frames.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        frames = frames.contiguous()

        if exp == 1:
            middle_frames = model.inference(
                frames[:-1],
                frames[1:],
                scale=scale,
            )
            result = torch.empty(
                (frames.shape[0] * 2 - 1, *frames.shape[1:]),
                device=frames.device,
                dtype=middle_frames.dtype,
            )
            result[0::2] = frames
            result[1::2] = middle_frames
            return result.contiguous(), 2

        n_intermediate = 2**exp // 2  # intermediates per adjacent pair

        result: list[torch.Tensor] = []
        for i in range(frames.shape[0] - 1):
            I0 = frames[i : i + 1]
            I1 = frames[i + 1 : i + 2]
            intermediate_tensors = self._make_inference(
                model, I0, I1, n_intermediate, scale
            )

            result.append(I0)
            result.extend(intermediate_tensors)

        result.append(frames[-1:])
        multiplier = 2**exp
        return torch.cat(result, dim=0).contiguous(), multiplier


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def interpolate_video_frames(
    frames: list[np.ndarray],
    exp: int = 1,
    scale: float = 1.0,
    model_path: Optional[str] = None,
) -> tuple[list[np.ndarray], int]:
    """
    Convenience wrapper around FrameInterpolator.

    Args:
        frames:     List of uint8 HWC numpy frames.
        exp:        Interpolation exponent (1=2×, 2=4×).
        scale:      RIFE inference scale (default 1.0; use 0.5 for high-res).
        model_path: Local directory or HuggingFace repo ID containing
                    ``flownet.pkl``.  *None* → default ``elfgum/RIFE-4.22.lite``.

    Returns:
        (interpolated_frames, multiplier)
    """
    interpolator = FrameInterpolator(model_path=model_path)
    return interpolator.interpolate(frames, exp=exp, scale=scale)


def interpolate_video_tensor(
    frames: torch.Tensor,
    exp: int = 1,
    scale: float = 1.0,
    model_path: Optional[str] = None,
) -> tuple[torch.Tensor, int]:
    """
    Convenience wrapper around FrameInterpolator for NCHW RGB tensors.

    Args:
        frames:     Tensor with shape [N, 3, H, W], either float in [0, 1]
                    or uint8 in [0, 255].
        exp:        Interpolation exponent (1=2×, 2=4×).
        scale:      RIFE inference scale (default 1.0; use 0.5 for high-res).
        model_path: Local directory or HuggingFace repo ID containing
                    ``flownet.pkl``.  *None* → default ``elfgum/RIFE-4.22.lite``.

    Returns:
        (interpolated_frames, multiplier)
    """
    interpolator = FrameInterpolator(model_path=model_path)
    return interpolator.interpolate_tensor(frames, exp=exp, scale=scale)


def _default_rife_model_path() -> str:
    for env_var in _RIFE_PATH_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            return value
    return _DEFAULT_RIFE_HF_REPO


def _looks_like_local_rife_path(model_path: str) -> bool:
    model_path = os.path.expanduser(model_path)
    return (
        os.path.exists(model_path)
        or model_path.endswith((".pkl", ".onnx", ".engine", ".plan"))
        or os.path.isabs(model_path)
        or model_path.startswith(".")
    )


def _resolve_local_rife_dir(model_path: str) -> str:
    model_path = os.path.expanduser(model_path)

    if os.path.isfile(model_path):
        if os.path.basename(model_path) == "flownet.pkl":
            return os.path.dirname(model_path)
        if model_path.endswith((".engine", ".plan")):
            raise ValueError(
                "TensorRT RIFE engine files are not supported by the PyTorch "
                "interpolator path yet. Provide a directory containing "
                "flownet.pkl, such as Practical-RIFE/train_log, or pass the "
                "combined asset root directory."
            )
        raise ValueError(
            f"Unsupported RIFE model file: {model_path}. Expected flownet.pkl "
            "or a directory containing it."
        )

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"RIFE model path does not exist: {model_path}")

    for rel_path in _RIFE_LOCAL_CANDIDATES:
        candidate = os.path.join(model_path, rel_path, "flownet.pkl")
        if os.path.isfile(candidate):
            return os.path.dirname(candidate)

    flownet_matches = sorted(
        glob(os.path.join(model_path, "**", "flownet.pkl"), recursive=True)
    )
    if flownet_matches:
        return os.path.dirname(flownet_matches[0])

    engine_matches = sorted(
        glob(os.path.join(model_path, "**", "*.engine"), recursive=True)
        + glob(os.path.join(model_path, "**", "*.plan"), recursive=True)
    )
    if engine_matches:
        raise ValueError(
            "Found TensorRT RIFE engine files but no flownet.pkl under "
            f"{model_path}. This code path currently loads PyTorch checkpoints; "
            "use Practical-RIFE/train_log or pass a directory containing "
            "flownet.pkl."
        )

    raise FileNotFoundError(
        "Could not find RIFE flownet.pkl under "
        f"{model_path}. Expected one of: "
        + ", ".join(os.path.join(p, "flownet.pkl") for p in _RIFE_LOCAL_CANDIDATES)
    )
