# SPDX-License-Identifier: Apache-2.0
"""
Krea Wan Causal pipeline implementation.

This module wires the causal DMD denoising stage into the modular pipeline.
"""

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.krea_realtime_video import (
    KreaRealtimeVideoBeforeDenoisingStage,
    KreaRealtimeVideoDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# isort: on

logger = init_logger(__name__)


class KreaWanCausalPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "KreaWanCausalPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(
            KreaRealtimeVideoBeforeDenoisingStage(
                textencoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                transformer=self.get_module("transformer"),
                vae=self.get_module("vae"),
            ),
        )
        self.add_stage(
            KreaRealtimeVideoDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
            ),
        )


EntryClass = KreaWanCausalPipeline
