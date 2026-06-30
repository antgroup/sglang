# SPDX-License-Identifier: Apache-2.0
"""
LingBot-World realtime VAE stages.

These stages align the realtime LingBot path with lingbot_fast_server:
- cache the conditioning image latent per realtime session instead of re-encoding
  it on every chunk
- decode chunk latents with a persistent causal VAE cache
"""

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.realtime_session import (
    BaseRealtimeState,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    DecodingStage,
    _ensure_tensor_decode_output,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class LingBotWorldRealtimeVAEState(BaseRealtimeState):
    def __init__(self):
        super().__init__()
        self.image_latent: torch.Tensor | None = None

    def dispose(self):
        super().dispose()
        self.image_latent = None


class LingBotWorldRealtimeImageVAEEncodingStage(ImageVAEEncodingStage):
    """Reuse the first chunk's conditioning image latent across a realtime session."""

    @staticmethod
    def _target_latent_num_frames(
        *,
        num_frames: int,
        temporal_compression_ratio: int,
        chunk_size: int,
    ) -> int:
        latent_frames = (num_frames - 1) // temporal_compression_ratio + 1
        return latent_frames - (latent_frames % chunk_size)

    @staticmethod
    def _lazy_encode_num_frames(
        *,
        initial_num_frames: int,
        lazy_black_frames: int,
        temporal_compression_ratio: int,
    ) -> int:
        encode_frames = initial_num_frames + int(lazy_black_frames or 0)
        if (encode_frames - 1) % temporal_compression_ratio != 0:
            encode_frames = (
                (encode_frames - 1) // temporal_compression_ratio + 1
            ) * temporal_compression_ratio + 1
        return encode_frames

    @staticmethod
    def _repeat_last_latent_to_frames(
        latent_condition: torch.Tensor, target_latent_frames: int
    ) -> torch.Tensor:
        current_latent_frames = int(latent_condition.shape[2])
        if current_latent_frames < target_latent_frames:
            tail = latent_condition[:, :, -1:, :, :].repeat(
                1,
                1,
                target_latent_frames - current_latent_frames,
                1,
                1,
            )
            return torch.cat([latent_condition, tail], dim=2)
        if current_latent_frames > target_latent_frames:
            return latent_condition[:, :, :target_latent_frames, :, :]
        return latent_condition

    @torch.no_grad()
    def _forward_lazy_condition(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        self.load_model()
        num_frames = batch.num_frames

        images = (
            batch.vae_image if batch.vae_image is not None else batch.condition_image
        )
        if not isinstance(images, list):
            images = [images]

        pipeline_config = server_args.pipeline_config
        temporal_ratio = int(
            pipeline_config.vae_config.arch_config.temporal_compression_ratio
        )
        chunk_size = int(pipeline_config.dit_config.arch_config.num_frames_per_block)
        target_latent_frames = self._target_latent_num_frames(
            num_frames=num_frames,
            temporal_compression_ratio=temporal_ratio,
            chunk_size=chunk_size,
        )
        lazy_black_frames = int(
            getattr(pipeline_config, "lazy_vae_encode_black_frames", 0) or 0
        )

        vae_dtype = PRECISION_TO_TYPE[pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast
        device = get_local_torch_device()

        all_image_latents = []
        prepare_condition_image_latent_ids = getattr(
            pipeline_config, "prepare_condition_image_latent_ids", None
        )
        condition_latents = [] if callable(prepare_condition_image_latent_ids) else None

        for image in images:
            image = self.preprocess(image).to(device, dtype=torch.float32)
            image = image.unsqueeze(2)
            initial_num_frames = int(image.shape[2])
            encode_num_frames = self._lazy_encode_num_frames(
                initial_num_frames=initial_num_frames,
                lazy_black_frames=lazy_black_frames,
                temporal_compression_ratio=temporal_ratio,
            )
            n_pad = max(0, encode_num_frames - initial_num_frames)
            video_condition = torch.cat(
                [
                    image,
                    image.new_zeros(
                        image.shape[0],
                        image.shape[1],
                        n_pad,
                        image.shape[3],
                        image.shape[4],
                    ),
                ],
                dim=2,
            ).to(device=device, dtype=torch.float32)

            with torch.autocast(
                device_type=current_platform.device_type,
                dtype=vae_dtype,
                enabled=vae_autocast_enabled,
            ):
                if pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
                if not vae_autocast_enabled:
                    video_condition = video_condition.to(vae_dtype)
                latent_dist: DiagonalGaussianDistribution = self.vae.encode(
                    video_condition
                )
                if isinstance(latent_dist, AutoencoderKLOutput):
                    latent_dist = latent_dist.latent_dist

            generator = batch.generator
            if generator is None:
                raise ValueError("Generator must be provided")

            sample_mode = pipeline_config.vae_config.encode_sample_mode()
            latent_condition = self.retrieve_latents(
                latent_dist, generator, sample_mode=sample_mode
            )
            short_latent_frames = int(latent_condition.shape[2])
            latent_condition = pipeline_config.postprocess_vae_encode(
                latent_condition, self.vae
            )
            latent_condition = self._repeat_last_latent_to_frames(
                latent_condition, target_latent_frames
            )

            scaling_factor, shift_factor = pipeline_config.get_decode_scale_and_shift(
                device=latent_condition.device,
                dtype=latent_condition.dtype,
                vae=self.vae,
            )
            if isinstance(shift_factor, torch.Tensor):
                shift_factor = shift_factor.to(latent_condition.device)
            if isinstance(scaling_factor, torch.Tensor):
                scaling_factor = scaling_factor.to(latent_condition.device)

            latent_condition -= shift_factor
            latent_condition = latent_condition * scaling_factor

            if condition_latents is not None:
                condition_latents.append(latent_condition)

            image_latent = pipeline_config.postprocess_image_latent(
                latent_condition, batch
            )
            all_image_latents.append(image_latent)

            logger.info(
                "LingBot lazy VAE encode: encode_pixel_frames=%s "
                "(first=%s black=%s pad=%s) -> latent_frames=%s repeat_last_to=%s",
                encode_num_frames,
                initial_num_frames,
                lazy_black_frames,
                n_pad,
                short_latent_frames,
                target_latent_frames,
            )

        batch.image_latent = torch.cat(all_image_latents, dim=1)
        if condition_latents is not None:
            prepare_condition_image_latent_ids(condition_latents, batch)

        self.offload_model()
        return batch

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        state = None
        if batch.session is not None:
            state = batch.session.get_or_create_state(LingBotWorldRealtimeVAEState)
            if batch.block_idx == 0:
                state.image_latent = None
            elif state.image_latent is not None:
                batch.image_latent = state.image_latent
                return batch

        if batch.condition_image is None:
            if state is not None and state.image_latent is not None:
                batch.image_latent = state.image_latent
            return batch

        if (
            int(
                getattr(server_args.pipeline_config, "lazy_vae_encode_black_frames", 0)
                or 0
            )
            > 0
        ):
            batch = self._forward_lazy_condition(batch, server_args)
        else:
            batch = super().forward(batch, server_args)

        if state is not None and batch.image_latent is not None:
            state.image_latent = batch.image_latent
        return batch


class LingBotWorldCausalDecodingStage(DecodingStage):
    """Decode LingBot realtime chunks with a persistent causal Wan VAE cache."""

    @torch.no_grad()
    def decode_causal(
        self, latents: torch.Tensor, server_args: ServerArgs
    ) -> torch.Tensor:
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        self.vae = self.vae.to(device=get_local_torch_device(), dtype=vae_dtype)
        latents = latents.to(get_local_torch_device())
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        latents = self.scale_and_shift(latents, server_args)
        latents = server_args.pipeline_config.preprocess_decoding(
            latents, server_args, vae=self.vae
        )

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=vae_autocast_enabled,
        ):
            try:
                if server_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
            except Exception:
                pass

            if not vae_autocast_enabled:
                latents = latents.to(vae_dtype)

            decode_fn = getattr(self.vae, "causal_decode", None)
            if decode_fn is None:
                decode_output = self.vae.decode(latents)
            else:
                decode_output = decode_fn(latents)
            image = _ensure_tensor_decode_output(decode_output)

        return (image / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        if batch.session is None:
            return super().forward(batch, server_args)

        self.load_model()

        reset_causal_state = getattr(self.vae, "reset_causal_decode_state", None)
        if batch.block_idx == 0 and callable(reset_causal_state):
            reset_causal_state()

        frames = self.decode_causal(batch.latents, server_args)
        frames = server_args.pipeline_config.post_decoding(frames, server_args)

        output_batch = OutputBatch(
            output=frames,
            asyn_post_process=server_args.realtime_async_postprocess,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            trajectory_decoded=None,
            metrics=batch.metrics,
            noise_pred=None,
        )

        return output_batch
