# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from: https://github.com/Robbyant/lingbot-world

# SPDX-License-Identifier: Apache-2.0
"""
LingBot-World realtime causal DMD pipeline.
"""

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    AuxiliaryConditionEncodingStage,
    DMDTimestepPreparationStage,
    ImageEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_world.lingbot_world_causal_denoising import (
    LingBotWorldCausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime import (
    CausalVaeDecodingStage,
    RealtimeChunkLatentPreparationStage,
    RealtimeImageVAEEncodingStage,
    RealtimeInputValidationStage,
    RealtimeTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


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

    def __init__(self, model_path, server_args, **kwargs):
        self._causal_decode_only = False
        self._causal_runner = None
        if server_args.causal_pipeline_mode != "off":
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                get_world_rank,
            )
            from sglang.multimodal_gen.runtime.realtime.causal_pipeline_config import (
                placement_from_args,
            )

            placement = placement_from_args(server_args)
            self._causal_decode_only = get_world_rank() >= placement.denoising_ranks
            if self._causal_decode_only:
                kwargs["required_config_modules"] = ["vae"]
        super().__init__(model_path=model_path, server_args=server_args, **kwargs)

    def _load_config(self):
        config = super()._load_config()
        if self._causal_decode_only:
            # The base loader expands MoE denoisers when a boundary is present,
            # even when required_config_modules contains only the VAE.
            config.pop("boundary_ratio", None)
        return config

    def forward_causal_window(self, batches, server_args):
        if server_args.causal_pipeline_mode == "off":
            raise ValueError("Enable --causal-pipeline-mode to submit a chunk window")
        if self._causal_runner is None:
            from sglang.multimodal_gen.runtime.realtime.lingbot_causal_pipeline import (
                LingBotCausalPipelineRunner,
            )

            self._causal_runner = LingBotCausalPipelineRunner(self, server_args)
        return self._causal_runner.forward(batches)

    def forward(self, batch, server_args):
        if server_args.causal_pipeline_mode != "off":
            return self.forward_causal_window([batch], server_args)[0]
        return super().forward(batch, server_args)

    def initialize_pipeline(self, server_args: ServerArgs):
        if self._causal_decode_only:
            return
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            shift=server_args.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

    def create_pipeline_stages(self, server_args) -> None:
        if self._causal_decode_only:
            self.add_stage(
                CausalVaeDecodingStage(vae=self.get_module("vae"), pipeline=self)
            )
            return
        self.add_stage(RealtimeInputValidationStage())
        self.add_stage(
            RealtimeTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )

        image_encoder = self.get_module("image_encoder", None)
        image_processor = self.get_module("image_processor", None)
        self.add_stage_if(
            image_encoder is not None and image_processor is not None,
            ImageEncodingStage(
                image_encoder=image_encoder,
                image_processor=image_processor,
            ),
        )

        self.add_stage(AuxiliaryConditionEncodingStage())
        self.add_stage(
            RealtimeImageVAEEncodingStage(
                vae=self.get_module("vae"),
            )
        )
        self.add_stage(DMDTimestepPreparationStage(self.get_module("scheduler")))
        self.add_stage(
            RealtimeChunkLatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            )
        )
        self.add_stage(
            LingBotWorldCausalDMDDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_stage(
            CausalVaeDecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
            )
        )


EntryClass = LingBotWorldCausalDMDPipeline
