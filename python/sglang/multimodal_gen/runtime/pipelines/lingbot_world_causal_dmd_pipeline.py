# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
LingBot-World realtime causal DMD pipeline.
"""

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    CausalDMDDenoisingStage,
    ImageEncodingStage,
    ImageVAEEncodingStage,
    InputValidationStage,
    WorldConditioningStage,
)


class LingBotWorldCausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "LingBotWorldCausalDMDPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
        "image_encoder",
        "image_processor",
    ]

    def create_pipeline_stages(self, server_args) -> None:
        self.add_stage(InputValidationStage())
        self.add_standard_text_encoding_stage()

        image_encoder = self.get_module("image_encoder", None)
        image_processor = self.get_module("image_processor", None)
        self.add_stage_if(
            image_encoder is not None and image_processor is not None,
            ImageEncodingStage(
                image_encoder=image_encoder,
                image_processor=image_processor,
            ),
        )

        self.add_stage(WorldConditioningStage())
        self.add_standard_latent_preparation_stage()
        self.add_stage(
            ImageVAEEncodingStage(
                vae=self.get_module("vae"),
            )
        )
        self.add_stage(
            CausalDMDDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_standard_decoding_stage()


EntryClass = LingBotWorldCausalDMDPipeline
