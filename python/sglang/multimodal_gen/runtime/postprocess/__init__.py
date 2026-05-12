# SPDX-License-Identifier: Apache-2.0
"""Frame interpolation and upscaling support for SGLang diffusion pipelines."""

from sglang.multimodal_gen.runtime.postprocess.flashvsr_upscaler import (
    FlashVSRUpscaler,
    is_flashvsr_model_path,
    upscale_frames_flashvsr,
)
from sglang.multimodal_gen.runtime.postprocess.realesrgan_upscaler import (
    ImageUpscaler,
    upscale_frames,
)
from sglang.multimodal_gen.runtime.postprocess.rife_interpolator import (
    FrameInterpolator,
    interpolate_video_frames,
)

__all__ = [
    "FlashVSRUpscaler",
    "FrameInterpolator",
    "interpolate_video_frames",
    "ImageUpscaler",
    "is_flashvsr_model_path",
    "upscale_frames_flashvsr",
    "upscale_frames",
]
