# SPDX-License-Identifier: Apache-2.0
"""
LingBot-World realtime input validation.

The generic input validation stage has to support all diffusion pipelines.  The
LingBot realtime path repeatedly sends chunks for the same session, so the
conditioning image and resolved dimensions can be reused after the first chunk.
"""

from __future__ import annotations

from PIL import Image

from sglang.multimodal_gen.runtime.pipelines_core.realtime_session import (
    BaseRealtimeState,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _cover_resize_pil_to_wh(image: Image.Image, target_width: int, target_height: int):
    image = image.convert("RGB")
    target_width, target_height = int(target_width), int(target_height)
    width, height = image.size
    if width == target_width and height == target_height:
        return image

    target_aspect = target_width / max(target_height, 1)
    source_aspect = width / max(height, 1)
    resample = getattr(Image, "Resampling", Image).LANCZOS

    if abs(source_aspect - target_aspect) >= 1e-4:
        if source_aspect > target_aspect:
            new_width = max(1, int(round(height * target_aspect)))
            left = (width - new_width) // 2
            image = image.crop((left, 0, left + new_width, height))
        else:
            new_height = max(1, int(round(width / target_aspect)))
            top = (height - new_height) // 2
            image = image.crop((0, top, width, top + new_height))

    return image.resize((target_width, target_height), resample)


class LingBotWorldInputValidationState(BaseRealtimeState):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.condition_image = None
        self.original_condition_image_size = None
        self.height = None
        self.width = None
        self.generator = None
        self.seeds = None
        self.generator_seed = None
        self.generator_device = None
        self.num_outputs_per_prompt = None

    def dispose(self):
        super().dispose()
        self.image_path = None
        self.condition_image = None
        self.original_condition_image_size = None
        self.height = None
        self.width = None
        self.generator = None
        self.seeds = None
        self.generator_seed = None
        self.generator_device = None
        self.num_outputs_per_prompt = None


class LingBotWorldInputValidationStage(InputValidationStage):
    """Reuse LingBot realtime conditioning image validation across chunks."""

    def preprocess_condition_image(
        self,
        batch: Req,
        server_args: ServerArgs,
        condition_image_width,
        condition_image_height,
    ):
        del server_args, condition_image_width, condition_image_height

        if batch.condition_image is None:
            return
        if isinstance(batch.condition_image, list):
            batch.condition_image = batch.condition_image[0]

        width = int(batch.width or 832)
        height = int(batch.height or 480)
        batch.condition_image = _cover_resize_pil_to_wh(
            batch.condition_image, width, height
        )
        batch.width = width
        batch.height = height

    def _cache_batch(self, batch: Req, state: LingBotWorldInputValidationState) -> None:
        state.image_path = batch.image_path
        state.condition_image = batch.condition_image
        state.original_condition_image_size = batch.original_condition_image_size
        state.height = batch.height
        state.width = batch.width

    def _can_reuse_cached_image(
        self, batch: Req, state: LingBotWorldInputValidationState
    ) -> bool:
        if batch.block_idx == 0 or state.condition_image is None:
            return False
        return (
            batch.image_path in (None, state.image_path)
            and batch.height in (None, state.height)
            and batch.width in (None, state.width)
        )

    def _cache_generator(
        self, batch: Req, state: LingBotWorldInputValidationState
    ) -> None:
        state.generator = batch.generator
        state.seeds = batch.seeds
        state.generator_seed = batch.seed
        state.generator_device = batch.generator_device
        state.num_outputs_per_prompt = batch.num_outputs_per_prompt

    def _can_reuse_generator(
        self, batch: Req, state: LingBotWorldInputValidationState
    ) -> bool:
        if batch.block_idx == 0 or state.generator is None:
            return False
        return (
            state.generator_seed == batch.seed
            and state.generator_device == batch.generator_device
            and state.num_outputs_per_prompt == batch.num_outputs_per_prompt
        )

    def _reuse_or_cache_generator(
        self, batch: Req, state: LingBotWorldInputValidationState
    ) -> None:
        if self._can_reuse_generator(batch, state):
            batch.generator = state.generator
            batch.seeds = state.seeds
            return

        self._cache_generator(batch, state)

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        if batch.session is None:
            return super().forward(batch, server_args)

        state = batch.session.get_or_create_state(LingBotWorldInputValidationState)
        if self._can_reuse_cached_image(batch, state):
            original_image_path = batch.image_path
            batch.image_path = None
            batch.condition_image = state.condition_image
            batch.original_condition_image_size = state.original_condition_image_size
            if batch.height is None:
                batch.height = state.height
            if batch.width is None:
                batch.width = state.width
            try:
                batch = super().forward(batch, server_args)
                self._reuse_or_cache_generator(batch, state)
                return batch
            finally:
                batch.image_path = original_image_path

        batch = super().forward(batch, server_args)
        self._cache_batch(batch, state)
        self._reuse_or_cache_generator(batch, state)
        return batch
