# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.wan import Wan2_2_I2V_A14B_SamplingParam


@dataclass
class LingBotWorldSamplingParams(Wan2_2_I2V_A14B_SamplingParam):
    actions: list[list[str]] | None = None
    chunk_size: int | None = None
    guidance_scale: float = 5.0
    guidance_scale_2: float = 5.0
    num_inference_steps: int = 70
    fps: int = 16

    def _adjust(self, server_args):
        super()._adjust(server_args)
        if self.chunk_size is None:
            self.chunk_size = max(
                1,
                int(
                    server_args.pipeline_config.dit_config.arch_config.num_frames_per_block
                ),
            )

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        if self.actions is not None:
            extra["actions"] = self.actions
        if self.chunk_size is not None:
            extra["chunk_size"] = self.chunk_size
        return extra
