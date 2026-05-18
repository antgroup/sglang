# SPDX-License-Identifier: Apache-2.0
"""Frame interpolation and upscaling support for SGLang diffusion pipelines."""

from sglang.multimodal_gen.runtime.postprocess.realesrgan_upscaler import (
    ImageUpscaler,
    upscale_frames,
    upscale_tensor,
)
from sglang.multimodal_gen.runtime.postprocess.rife_interpolator import (
    FrameInterpolator,
    interpolate_video_frames,
    interpolate_video_tensor,
)

__all__ = [
    "FrameInterpolator",
    "interpolate_video_frames",
    "interpolate_video_tensor",
    "ImageUpscaler",
    "upscale_frames",
    "upscale_tensor",
]
