# SPDX-License-Identifier: Apache-2.0
"""FlashVSR-backed video upscaling for diffusion postprocess.

Only the FlashVSR v1.1 Tiny Long path is supported.  The small runtime pieces
needed for inference live under ``postprocess.flashvsr`` so callers only need
to provide the v1.1 weights and prompt tensor at runtime.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_MODEL_CACHE: dict[tuple[Any, ...], "FlashVSRUpscaler"] = {}
_CACHE_LOCK = threading.Lock()

_DEFAULT_WEIGHTS_DIR = "FlashVSR-v1.1"
_DIT_WEIGHTS = "diffusion_pytorch_model_streaming_dmd.safetensors"
_LQ_PROJ_WEIGHTS = "LQ_proj_in.ckpt"
_TC_DECODER_WEIGHTS = "TCDecoder.ckpt"
_PROMPT_TENSOR = os.path.join("examples", "WanVSR", "prompt_tensor", "posi_prompt.pth")


@dataclass(frozen=True)
class FlashVSRSpec:
    weights_dir: str
    prompt_tensor_path: str
    scale: float = 2.0
    sparse_ratio: float = 2.0
    kv_ratio: float = 3.0
    local_range: int = 11
    color_fix: bool = True
    target_multiple: int = 128


def is_flashvsr_model_path(model_path: str | None) -> bool:
    """Return true when an upscaling model path should be handled by FlashVSR."""
    if not model_path:
        return False
    path = _strip_flashvsr_prefix(model_path.split("?", 1)[0])
    if model_path.lower().startswith("flashvsr:"):
        return True
    if not os.path.isdir(os.path.expanduser(path)):
        return False
    base = os.path.abspath(os.path.expanduser(path))
    return (
        os.path.isdir(os.path.join(base, _DEFAULT_WEIGHTS_DIR))
        or os.path.isdir(os.path.join(base, "examples", "WanVSR", _DEFAULT_WEIGHTS_DIR))
        or os.path.isdir(os.path.join(base, "diffsynth"))
        or os.path.isfile(os.path.join(base, _DIT_WEIGHTS))
    )


def upscale_frames_flashvsr(
    frames: list[np.ndarray],
    model_path: str | None = None,
    scale: int | float = 2,
) -> list[np.ndarray]:
    """Upscale one video chunk using FlashVSR.

    ``model_path`` accepts a local ``FlashVSR-v1.1`` weights directory, a local
    FlashVSR checkout for path discovery only, or the explicit
    ``flashvsr:/path/to/FlashVSR-v1.1`` form.  Optional query parameters are
    supported, for example:

    ``flashvsr:/opt/FlashVSR-v1.1?prompt_tensor=/opt/posi_prompt.pth&scale=2``.

    The returned list always matches the input chunk length.  Short LingBot
    realtime chunks are padded internally only to satisfy FlashVSR's streaming
    frame-count requirements.
    """
    if not frames:
        return frames

    spec = _resolve_flashvsr_spec(model_path, scale=scale)
    cache_key = (
        spec.weights_dir,
        spec.prompt_tensor_path,
        spec.scale,
        spec.sparse_ratio,
        spec.kv_ratio,
        spec.local_range,
        spec.color_fix,
        spec.target_multiple,
        str(current_platform.get_local_torch_device()),
    )
    with _CACHE_LOCK:
        upscaler = _MODEL_CACHE.get(cache_key)
        if upscaler is None:
            upscaler = FlashVSRUpscaler(spec)
            _MODEL_CACHE[cache_key] = upscaler
    return upscaler.upscale(frames)


class FlashVSRUpscaler:
    """Lazy single-process FlashVSR Tiny Long pipeline wrapper."""

    def __init__(self, spec: FlashVSRSpec):
        self.spec = spec
        self._pipe = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._pipe is not None:
            return self._pipe

        try:
            from .flashvsr import load_flashvsr_tiny_long_runner
        except ImportError as e:
            raise ImportError(
                "FlashVSR Tiny Long runtime dependencies are not importable. "
                "Install the required packages, including block_sparse_attn and "
                "safetensors, before enabling FlashVSR upscaling."
            ) from e

        weights_dir = self.spec.weights_dir
        dit_path = os.path.join(weights_dir, _DIT_WEIGHTS)
        lq_proj_path = os.path.join(weights_dir, _LQ_PROJ_WEIGHTS)
        tc_decoder_path = os.path.join(weights_dir, _TC_DECODER_WEIGHTS)
        for required_path in (dit_path, lq_proj_path, tc_decoder_path):
            if not os.path.exists(required_path):
                raise FileNotFoundError(
                    f"Missing FlashVSR weight file: {required_path}. "
                    "Expected diffusion_pytorch_model_streaming_dmd.safetensors, "
                    "LQ_proj_in.ckpt, and TCDecoder.ckpt in the weights directory."
                )

        device = current_platform.get_local_torch_device()
        dtype = torch.bfloat16
        logger.info(
            "Loading FlashVSR Tiny Long pipeline from %s on %s (scale=%sx)",
            weights_dir,
            device,
            self.spec.scale,
        )
        pipe = load_flashvsr_tiny_long_runner(
            weights_dir,
            self.spec.prompt_tensor_path,
            device=device,
            dtype=dtype,
        )
        self._pipe = pipe
        return pipe

    def upscale(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        if not frames:
            return frames
        start_time = time.perf_counter()
        pipe = self._ensure_loaded()
        with self._lock:
            lq_video, height, width, num_frames, requested_frames = _prepare_lq_video(
                frames,
                scale=self.spec.scale,
                target_multiple=self.spec.target_multiple,
                dtype=torch.bfloat16,
                device=current_platform.get_local_torch_device(),
            )
            topk_ratio = self.spec.sparse_ratio * 768 * 1280 / float(height * width)
            output = pipe(
                lq_video=lq_video,
                num_frames=num_frames,
                height=height,
                width=width,
                topk_ratio=topk_ratio,
                kv_ratio=self.spec.kv_ratio,
                local_range=self.spec.local_range,
                color_fix=self.spec.color_fix,
            )
            output_frames = _tensor_to_frames(output, requested_frames)

        logger.info(
            "FlashVSR upscale completed in %.3f seconds for %d frames (%sx%s -> %sx%s)",
            time.perf_counter() - start_time,
            len(frames),
            frames[0].shape[1],
            frames[0].shape[0],
            width,
            height,
        )
        return output_frames


def _resolve_flashvsr_spec(model_path: str | None, scale: int | float) -> FlashVSRSpec:
    path_spec, options = _split_path_options(model_path)
    path_spec = path_spec or os.environ.get("FLASHVSR_MODEL_PATH")
    if not path_spec:
        raise ValueError(
            "FlashVSR upscaling requires upscaling_model_path, for example "
            "'flashvsr:/path/to/FlashVSR-v1.1'."
        )
    path = _strip_flashvsr_prefix(path_spec)
    if not path:
        path = os.environ.get("FLASHVSR_MODEL_PATH")
    if not path:
        raise ValueError(
            "FlashVSR upscaling requires a non-empty model path, for example "
            "'flashvsr:/path/to/FlashVSR-v1.1'."
        )
    path = os.path.abspath(os.path.expanduser(path))
    _validate_tiny_long_options(options)
    weights_dir = _resolve_weights_dir(path)
    prompt_tensor_path = _resolve_prompt_tensor_path(
        options=options,
        input_path=path,
        weights_dir=weights_dir,
    )
    if prompt_tensor_path is None:
        raise FileNotFoundError(
            "Could not locate FlashVSR posi_prompt.pth. Put it in the weights "
            "directory, pass '?prompt_tensor=/path/to/posi_prompt.pth', or set "
            "FLASHVSR_PROMPT_TENSOR_PATH."
        )

    return FlashVSRSpec(
        weights_dir=weights_dir,
        prompt_tensor_path=prompt_tensor_path,
        scale=float(options.get("scale", scale)),
        sparse_ratio=float(options.get("sparse_ratio", 2.0)),
        kv_ratio=float(options.get("kv_ratio", 3.0)),
        local_range=int(options.get("local_range", 11)),
        color_fix=_parse_bool(options.get("color_fix", True)),
        target_multiple=int(options.get("target_multiple", 128)),
    )


def _split_path_options(model_path: str | None) -> tuple[str | None, dict[str, str]]:
    if not model_path:
        return None, {}
    path, _, query = model_path.partition("?")
    parsed = parse_qs(query, keep_blank_values=False)
    options = {key: values[-1] for key, values in parsed.items() if values}
    return path, options


def _strip_flashvsr_prefix(path: str) -> str:
    return path[len("flashvsr:") :] if path.lower().startswith("flashvsr:") else path


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _validate_tiny_long_options(options: dict[str, str]) -> None:
    pipeline = options.get("pipeline", options.get("mode"))
    if pipeline is None:
        return
    value = str(pipeline).strip().lower().replace("_", "-")
    if value not in {"long", "tiny-long", "tinylong"}:
        raise ValueError(
            "Only FlashVSR Tiny Long is supported in this integration; "
            f"got pipeline={pipeline!r}."
        )


def _resolve_weights_dir(path: str) -> str:
    candidates = [
        path,
        os.path.join(path, _DEFAULT_WEIGHTS_DIR),
        os.path.join(path, "examples", "WanVSR", _DEFAULT_WEIGHTS_DIR),
        os.path.join(path, "examples", "WanVSR", "FlashVSR"),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, _DIT_WEIGHTS)):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "Could not locate FlashVSR Tiny Long weights. Expected "
        f"{_DIT_WEIGHTS}, {_LQ_PROJ_WEIGHTS}, and {_TC_DECODER_WEIGHTS} in "
        "the model directory or its FlashVSR-v1.1 child."
    )


def _ceil_8n_minus_3(num_frames: int) -> int:
    # FlashVSR Tiny's streaming loop uses ``(num_frames - 1) // 8 - 2``.
    # The smallest usable input length is therefore 25 frames, which
    # corresponds to 21 returned frames after its 4-frame lookahead.
    if num_frames <= 21:
        return 21
    remainder = (num_frames + 3) % 8
    return num_frames if remainder == 0 else num_frames + (8 - remainder)


def _prepare_lq_video(
    frames: list[np.ndarray],
    *,
    scale: float,
    target_multiple: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, int, int, int, int]:
    first = frames[0]
    if first.ndim != 3 or first.shape[-1] not in (3, 4):
        raise ValueError(
            f"FlashVSR expects HWC RGB/RGBA uint8 frames, got {tuple(first.shape)}"
        )
    input_height, input_width = first.shape[:2]
    scaled_width = int(round(input_width * scale))
    scaled_height = int(round(input_height * scale))
    target_width = (scaled_width // target_multiple) * target_multiple
    target_height = (scaled_height // target_multiple) * target_multiple
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"FlashVSR target size is too small after {scale}x scaling: "
            f"{scaled_width}x{scaled_height}."
        )

    requested_frames = len(frames)
    output_frames = _ceil_8n_minus_3(requested_frames)
    padded_frames = list(frames)
    padded_frames.extend([frames[-1]] * (output_frames + 4 - len(padded_frames)))

    tensors = [
        _frame_to_tensor(
            frame,
            scaled_width=scaled_width,
            scaled_height=scaled_height,
            target_width=target_width,
            target_height=target_height,
            dtype=dtype,
            device=device,
        )
        for frame in padded_frames
    ]
    video = torch.stack(tensors, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    return video, target_height, target_width, len(padded_frames), requested_frames


def _frame_to_tensor(
    frame: np.ndarray,
    *,
    scaled_width: int,
    scaled_height: int,
    target_width: int,
    target_height: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    image = Image.fromarray(frame)
    image = image.resize((scaled_width, scaled_height), Image.BICUBIC)
    left = (scaled_width - target_width) // 2
    top = (scaled_height - target_height) // 2
    image = image.crop((left, top, left + target_width, top + target_height))
    tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8)).to(
        device=device, dtype=torch.float32
    )
    tensor = tensor.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    return tensor.to(dtype)


def _tensor_to_frames(video: torch.Tensor, requested_frames: int) -> list[np.ndarray]:
    if video.dim() != 4:
        raise ValueError(f"Unexpected FlashVSR output shape: {tuple(video.shape)}")
    video = video[:, :requested_frames]
    arr = video.permute(1, 2, 3, 0)
    arr = ((arr.float() + 1.0) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
    frames = [arr[i] for i in range(arr.shape[0])]
    if len(frames) < requested_frames and frames:
        frames.extend([frames[-1]] * (requested_frames - len(frames)))
    return frames


def _resolve_prompt_tensor_path(
    *,
    options: dict[str, str],
    input_path: str,
    weights_dir: str,
) -> str | None:
    explicit_path = (
        options.get("prompt_tensor")
        or options.get("prompt_tensor_path")
        or options.get("prompt_path")
        or os.environ.get("FLASHVSR_PROMPT_TENSOR_PATH")
    )
    candidates = [
        explicit_path,
        os.path.join(weights_dir, "posi_prompt.pth"),
        os.path.join(weights_dir, "prompt_tensor", "posi_prompt.pth"),
        os.path.join(weights_dir, _PROMPT_TENSOR),
        os.path.join(os.path.dirname(weights_dir), "prompt_tensor", "posi_prompt.pth"),
        os.path.join(input_path, "prompt_tensor", "posi_prompt.pth"),
        os.path.join(input_path, _PROMPT_TENSOR),
    ]
    for candidate in candidates:
        if candidate:
            candidate = os.path.abspath(os.path.expanduser(candidate))
            if os.path.exists(candidate):
                return candidate
    return None
