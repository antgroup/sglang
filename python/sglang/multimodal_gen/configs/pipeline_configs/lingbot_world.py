# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig
from sglang.multimodal_gen.configs.models.dits import LingBotWorldVideoConfig
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_I2V_A14B_Config
from sglang.multimodal_gen.runtime.utils.lingbot_world import (
    prepare_lingbot_world_condition,
)


@dataclass
class LingBotWorldI2VConfig(Wan2_2_I2V_A14B_Config):
    dit_config: DiTConfig = field(default_factory=LingBotWorldVideoConfig)
    flow_shift: float | None = 10.0
    boundary_ratio: float | None = 0.947

    def prepare_world_condition(self, batch, device, dtype):
        # Normalize offline and realtime controls into c2ws_plucker_emb.
        c2ws_plucker_emb, resolved_num_frames = prepare_lingbot_world_condition(
            batch=batch,
            pipeline_config=self,
            device=device,
            dtype=dtype,
        )
        if resolved_num_frames is not None:
            batch.num_frames = resolved_num_frames
        if c2ws_plucker_emb is not None:
            batch.c2ws_plucker_emb = c2ws_plucker_emb

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        kwargs = super().prepare_pos_cond_kwargs(batch, device, rotary_emb, dtype)
        if batch.c2ws_plucker_emb is not None:
            kwargs["c2ws_plucker_emb"] = batch.c2ws_plucker_emb.to(
                device=device, dtype=dtype
            )
        return kwargs
