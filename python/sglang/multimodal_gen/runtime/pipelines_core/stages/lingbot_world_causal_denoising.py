# SPDX-License-Identifier: Apache-2.0
"""
LingBot-World causal DMD denoising stage.

Extends CausalDMDDenoisingStage with I2V condition concatenation:
the transformer expects [noise_latent, condition] concatenated along
the channel dimension, matching the lingbot_fast_server inference loop.
"""

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.utils import pred_noise_to_pred_video
from sglang.multimodal_gen.runtime.pipelines_core.realtime_session import (
    BaseRealtimeState,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import dict_to_3d_list

logger = init_logger(__name__)


class LingBotWorldCausalDMDRealtimeState(BaseRealtimeState):
    """Persists KV cache across chunks in a realtime session."""

    pass


class LingBotWorldCausalDMDDenoisingStage(CausalDMDDenoisingStage):
    """Causal DMD denoising with I2V condition concatenation for LingBot-World.

    The LingBot-World transformer has ``in_channels = 36`` and expects the
    input hidden_states to be ``[noise(16ch), condition(16ch+4ch mask)]``
    concatenated along the channel dimension.  This stage injects that
    concatenation before every transformer call (denoising loop + KV cache
    context update), while reusing all other logic from the base class.
    """

    def _get_cache_state(
        self,
        batch: Req,
    ) -> tuple[LingBotWorldCausalDMDRealtimeState, bool]:
        if batch.session is not None:
            state = batch.session.get_or_create_state(
                LingBotWorldCausalDMDRealtimeState
            )
            return state, True
        return LingBotWorldCausalDMDRealtimeState(), False

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        target_dtype = torch.bfloat16
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not server_args.disable_autocast

        latent_seq_length = batch.latents.shape[-1] * batch.latents.shape[-2]
        patch_ratio = (
            self.transformer.config.arch_config.patch_size[-1]
            * self.transformer.config.arch_config.patch_size[-2]
        )
        self.frame_seq_length = latent_seq_length // patch_ratio

        # Timesteps
        timesteps = torch.tensor(
            server_args.pipeline_config.dmd_denoising_steps, dtype=torch.long
        ).cpu()
        if server_args.pipeline_config.warp_denoising_step:
            scheduler_timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            )
            timesteps = scheduler_timesteps[1000 - timesteps]
        timesteps = timesteps.to(get_local_torch_device())
        logger.info("Using timesteps: %s", timesteps)

        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            image_embeds = [
                image_embed.to(target_dtype) for image_embed in image_embeds
            ]

        image_kwargs = self.prepare_extra_func_kwargs(
            getattr(self.transformer, "forward", self.transformer),
            {
                "encoder_hidden_states_image": image_embeds,
                "mask_strategy": dict_to_3d_list(None, t_max=50, l_max=60, h_max=24),
            },
        )

        pos_cond_kwargs = self.prepare_extra_func_kwargs(
            getattr(self.transformer, "forward", self.transformer),
            {
                "encoder_hidden_states_2": batch.clip_embedding_pos,
                "encoder_attention_mask": batch.prompt_attention_mask,
            }
            | server_args.pipeline_config.prepare_pos_cond_kwargs(
                batch,
                self.device,
                getattr(self.transformer, "rotary_emb", None),
                dtype=target_dtype,
            ),
        )

        if self.attn_backend.get_enum() == AttentionBackendEnum.SLIDING_TILE_ATTN:
            self.prepare_sta_param(batch, server_args)

        # Latents [B, C_noise, T, H, W] and condition [B, C_cond, T, H, W]
        assert batch.latents is not None, "latents must be provided"
        latents = batch.latents
        b, c, t, h, w = latents.shape
        prompt_embeds = server_args.pipeline_config.get_pos_prompt_embeds(batch)

        # Condition from ImageVAEEncodingStage (image_latent includes image + mask channels)
        condition = batch.image_latent
        assert condition is not None, (
            "LingBot-World causal DMD requires image_latent as condition. "
            "Ensure ImageVAEEncodingStage runs before this stage."
        )

        # Split condition into per-block chunks along temporal dim
        condition_chunks = condition.split(self.num_frames_per_block, dim=2)

        # KV cache
        cache_state, persist_cache_state = self._get_cache_state(batch)
        kv_cache1 = cache_state.kv_cache
        crossattn_cache = cache_state.crossattn_cache

        should_reset_cache = (
            batch.block_idx == 0
            or kv_cache1 is None
            or crossattn_cache is None
            or len(kv_cache1) != self.num_transformer_blocks
            or len(crossattn_cache) != self.num_transformer_blocks
        )

        if should_reset_cache:
            self._initialize_kv_cache(
                batch_size=b, dtype=target_dtype, device=latents.device
            )
            self._initialize_crossattn_cache(
                batch_size=b,
                max_text_len=server_args.pipeline_config.text_encoder_configs[
                    0
                ].arch_config.text_len,
                dtype=target_dtype,
                device=latents.device,
            )
            kv_cache1 = cache_state.kv_cache = self.kv_cache1
            crossattn_cache = cache_state.crossattn_cache = self.crossattn_cache
        else:
            for block_index in range(self.num_transformer_blocks):
                crossattn_cache[block_index]["is_init"] = False

        # Block decomposition
        if t % self.num_frames_per_block != 0:
            raise ValueError(
                "num_frames must be divisible by num_frames_per_block"
            )
        num_blocks = t // self.num_frames_per_block
        current_start_frame = 0

        with self.progress_bar(total=num_blocks * len(timesteps)) as progress_bar:
            for block_idx in range(num_blocks):
                current_latents = latents[
                    :, :, current_start_frame : current_start_frame + self.num_frames_per_block, :, :
                ]
                cond_idx = min(block_idx, len(condition_chunks) - 1)
                current_condition = condition_chunks[cond_idx]

                # BTCHW for DMD math
                noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
                video_raw_latent_shape = noise_latents_btchw.shape

                for i, t_cur in enumerate(timesteps):
                    noise_latents = noise_latents_btchw.clone()

                    # KEY: concat [noise, condition] along channel dim -> in_channels=36
                    latent_model_input = torch.cat(
                        [current_latents, current_condition], dim=1
                    ).to(target_dtype)

                    t_expand = t_cur.repeat(latent_model_input.shape[0])

                    # Attention metadata
                    if (
                        self.attn_backend.get_enum()
                        == AttentionBackendEnum.VIDEO_SPARSE_ATTN
                    ):
                        self.attn_metadata_builder_cls = (
                            self.attn_backend.get_builder_cls()
                        )
                        if self.attn_metadata_builder_cls is not None:
                            self.attn_metadata_builder = (
                                self.attn_metadata_builder_cls()
                            )
                            attn_metadata = self.attn_metadata_builder.build(
                                current_timestep=i,
                                raw_latent_shape=(
                                    self.num_frames_per_block, h, w,
                                ),
                                patch_size=server_args.pipeline_config.dit_config.patch_size,
                                STA_param=batch.STA_param,
                                VSA_sparsity=server_args.attention_backend_config.VSA_sparsity,
                                device=get_local_torch_device(),
                            )
                        else:
                            attn_metadata = None
                    else:
                        attn_metadata = None

                    with (
                        torch.autocast(
                            device_type=current_platform.device_type,
                            dtype=target_dtype,
                            enabled=autocast_enabled,
                        ),
                        set_forward_context(
                            current_timestep=i,
                            attn_metadata=attn_metadata,
                            forward_batch=batch,
                        ),
                    ):
                        t_expanded = t_cur * torch.ones(
                            (latent_model_input.shape[0], 1),
                            device=latent_model_input.device,
                            dtype=torch.long,
                        )
                        # Transformer output is [B, C_out, T, H, W]
                        pred_noise = self.transformer(
                            latent_model_input,
                            prompt_embeds,
                            t_expanded,
                            kv_cache=kv_cache1,
                            crossattn_cache=crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            start_frame=current_start_frame,
                            **image_kwargs,
                            **pos_cond_kwargs,
                        )

                    # Convert flow pred -> x0
                    pred_noise_btchw = pred_noise.permute(0, 2, 1, 3, 4)
                    pred_video_btchw = pred_noise_to_pred_video(
                        pred_noise=pred_noise_btchw.flatten(0, 1),
                        noise_input_latent=noise_latents.flatten(0, 1),
                        timestep=t_expand,
                        scheduler=self.scheduler,
                    ).unflatten(0, pred_noise_btchw.shape[:2])

                    if i < len(timesteps) - 1:
                        next_timestep = timesteps[i + 1] * torch.ones(
                            [1], dtype=torch.long, device=pred_video_btchw.device
                        )
                        noise = torch.randn(
                            video_raw_latent_shape,
                            dtype=pred_video_btchw.dtype,
                            generator=(
                                batch.generator[0]
                                if isinstance(batch.generator, list)
                                else batch.generator
                            ),
                            device=self.device,
                        )
                        noise_latents_btchw = self.scheduler.add_noise(
                            pred_video_btchw.flatten(0, 1),
                            noise.flatten(0, 1),
                            next_timestep,
                        ).unflatten(0, pred_video_btchw.shape[:2])
                        current_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)
                    else:
                        current_latents = pred_video_btchw.permute(0, 2, 1, 3, 4)

                    if progress_bar is not None:
                        progress_bar.update()

                # Write denoised result back
                latents[:, :, current_start_frame : current_start_frame + self.num_frames_per_block, :, :] = (
                    current_latents
                )

                # KV cache update: forward with clean context (concat x0 + condition)
                context_noise = getattr(server_args.pipeline_config, "context_noise", 0)
                t_context = torch.ones(
                    [b], device=latents.device, dtype=torch.long
                ) * int(context_noise)
                context_input = torch.cat(
                    [current_latents, current_condition], dim=1
                ).to(target_dtype)
                with (
                    torch.autocast(
                        device_type=current_platform.device_type,
                        dtype=target_dtype,
                        enabled=autocast_enabled,
                    ),
                    set_forward_context(
                        current_timestep=0,
                        attn_metadata=attn_metadata,
                        forward_batch=batch,
                    ),
                ):
                    t_expanded_context = t_context.unsqueeze(1)
                    _ = self.transformer(
                        context_input,
                        prompt_embeds,
                        t_expanded_context,
                        kv_cache=kv_cache1,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        start_frame=current_start_frame,
                        **image_kwargs,
                        **pos_cond_kwargs,
                    )
                current_start_frame += self.num_frames_per_block

        batch.latents = latents
        if not persist_cache_state:
            cache_state.dispose()
        return batch
