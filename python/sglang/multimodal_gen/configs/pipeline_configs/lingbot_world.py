# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig
from sglang.multimodal_gen.configs.models.dits import LingBotWorldVideoConfig
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_I2V_A14B_Config


@dataclass
class LingBotWorldI2VConfig(Wan2_2_I2V_A14B_Config):
    dit_config: DiTConfig = field(default_factory=LingBotWorldVideoConfig)
    flow_shift: float | None = 10.0
    boundary_ratio: float | None = 0.947

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        kwargs = super().prepare_pos_cond_kwargs(batch, device, rotary_emb, dtype)
        if batch.c2ws_plucker_emb is not None:
            kwargs["c2ws_plucker_emb"] = batch.c2ws_plucker_emb.to(
                device=device, dtype=dtype
            )
        return kwargs


@dataclass
class LingBotWorldCausalDMDConfig(LingBotWorldI2VConfig):
    is_causal: bool = True
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 821, 642, 321]
    )
    warp_denoising_step: bool = True

    def postprocess_image_latent(self, latent_condition, batch):
        """Build condition tensor aligned to chunk_size (num_frames_per_block).

        Matches lingbot_fast_server's _prepare_latents_causal:
        condition = [mask(temporal_ratio ch), latent(z_dim ch)] -> 20ch total,
        with temporal dim aligned to chunk_size.
        """
        vae_arch = self.vae_config.arch_config
        temporal_ratio = vae_arch.temporal_compression_ratio
        spatial_ratio = vae_arch.spatial_compression_ratio
        chunk_size = self.dit_config.arch_config.num_frames_per_block

        latent_height = batch.height // spatial_ratio
        latent_width = batch.width // spatial_ratio

        # Align num_latent_frames to chunk_size
        num_latent_frames = latent_condition.shape[2]
        num_latent_frames = num_latent_frames - (num_latent_frames % chunk_size)
        latent_condition = latent_condition[:, :, :num_latent_frames, :, :]

        # Number of initial frames that have actual image content
        # (latent_condition from VAE encode of [image, zeros...])
        # First frame is real, rest are zero-padded
        initial_latent_frames = 1  # single image -> 1 latent frame

        # Build mask: [B, temporal_ratio, num_latent_frames, H, W]
        mask = torch.ones(
            1, temporal_ratio, num_latent_frames,
            latent_height, latent_width,
            dtype=latent_condition.dtype,
            device=latent_condition.device,
        )
        # Zero out mask for frames beyond the initial image
        if initial_latent_frames < num_latent_frames:
            mask[:, :, initial_latent_frames:] = 0

        return torch.cat([mask, latent_condition], dim=1)
