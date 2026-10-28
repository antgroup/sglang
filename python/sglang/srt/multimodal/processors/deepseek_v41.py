"""DeepSeek-V4.1 image preprocessing, preserving raw token IDs for Engram."""

import asyncio
import logging

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.multimodal.deepseek_v41_image_processing import (
    GPU_PLAN_KEY,
    image_token_types,
    prepare_image_gpu,
)
from sglang.srt.multimodal.deepseek_v41_mm_inputs import (
    build_epd_mm_output,
    resolve_image_backend,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)


class DeepseekV41ImageProcessor(BaseMultimodalProcessor):
    models = [DeepseekV4ForCausalLM]
    preserve_processor_input_ids = True
    prefer_tokenized_input = True
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        # Resolved before super().__init__(), which fingerprints the backend.
        self.preprocess_backend, self.cpu_patchify = resolve_image_backend()
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.image_token_id = hf_config.image_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=_processor.convert_ids_to_tokens(self.image_token_id),
            image_token_id=self.image_token_id,
        ).build(_processor)

    def preprocess_fingerprint_payload(self):
        payload = super().preprocess_fingerprint_payload()
        payload["dsv41_preprocess_backend"] = self.preprocess_backend
        return payload

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, *args, **kwargs
    ):
        base = await self.load_mm_data(
            input_text, image_data=image_data, multimodal_tokens=self.mm_tokens
        )
        ids = (
            input_text
            if isinstance(input_text, list)
            else self._processor.encode(input_text)
        )
        if ids.count(self.image_token_id) != len(base.images):
            raise ValueError("Image placeholders and images must match")
        images = iter(base.images)
        tokens, items = [], []
        for token in ids:
            if token != self.image_token_id:
                tokens.append(token)
                continue
            image = next(images)
            plan = None
            if self.preprocess_backend == "gpu":
                patches, plan, lh, lw = await asyncio.to_thread(
                    prepare_image_gpu, image, self.hf_config
                )
                h = plan["height"] // plan["patch_size"]
                w = plan["width"] // plan["patch_size"]
            else:
                patches, h, w, lh, lw = await asyncio.to_thread(
                    self.cpu_patchify, image, self.hf_config
                )
            if self.keep_mm_features_on_device:
                patches = patches.to(torch.device("cuda", self.server_args.base_gpu_id))
            metadata = {"n_vit_h": h, "n_vit_w": w}
            if plan is not None:
                metadata[GPU_PLAN_KEY] = plan
            count = len(image_token_types(lh, lw))
            start = len(tokens)
            tokens.extend([self.image_token_id] * count)
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=patches,
                    offsets=[(start, start + count - 1)],
                    model_specific_data=metadata,
                )
            )
        return MultimodalProcessorOutput(
            input_ids=tokens,
            mm_items=self._prepare_mm_items_for_transport(items),
            im_token_id=self.image_token_id,
        )

    def get_mm_data(self, prompt, embeddings, *, img_grid_thw=None, **kwargs):
        """Rebuild an EPD prompt from the encoder's grids; see build_epd_mm_output."""
        return build_epd_mm_output(
            prompt if isinstance(prompt, list) else self._processor.encode(prompt),
            embeddings=embeddings,
            img_grid_thw=img_grid_thw,
            image_token_id=self.image_token_id,
            downsample_ratio=self.hf_config.vision_downsample_ratio,
        )
