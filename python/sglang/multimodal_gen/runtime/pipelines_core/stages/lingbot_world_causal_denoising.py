# SPDX-License-Identifier: Apache-2.0
"""
LingBot-World causal DMD denoising stage.

Extends CausalDMDDenoisingStage with:
- I2V condition concatenation ([noise, condition] along channel dim)
- Per-chunk noise generation (no LatentPreparationStage needed)
- Session-persistent KV cache with cumulative frame position tracking
"""

import torch
from diffusers.utils.torch_utils import randn_tensor

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
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LingBotWorldCausalDMDRealtimeState(BaseRealtimeState):
    """Persists KV cache and frame position across chunks in a realtime session."""

    def __init__(self):
        super().__init__()
        self.current_chunk_start_frame: int = 0
        self.chunk_idx: int = 0


class LingBotWorldCausalDMDDenoisingStage(CausalDMDDenoisingStage):
    """Causal DMD denoising with I2V condition concatenation for LingBot-World.

    The LingBot-World transformer has ``in_channels = 36`` and expects
    ``[noise(16ch), condition(20ch)]`` concatenated along channel dim.
    Each call processes one chunk (num_frames_per_block frames).
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

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """LingBot generates latents internally; only require image_latent."""
        result = VerificationResult()
        result.add_check(
            "image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)]
        )
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        return result

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        target_dtype = torch.bfloat16
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not server_args.disable_autocast
        device = get_local_torch_device()

        # --- Condition: take current chunk's slice ---
        condition_full = batch.image_latent
        assert condition_full is not None, (
            "LingBot-World causal DMD requires image_latent as condition. "
            "Ensure ImageVAEEncodingStage runs before this stage."
        )

        b = condition_full.shape[0]
        h = condition_full.shape[3]
        w = condition_full.shape[4]
        t = self.num_frames_per_block
        num_channels_latents = self.transformer.config.arch_config.out_channels

        # frame_seq_length from spatial dims and patch size
        patch_ratio = (
            self.transformer.config.arch_config.patch_size[-1]
            * self.transformer.config.arch_config.patch_size[-2]
        )
        self.frame_seq_length = (h * w) // patch_ratio

        # --- Generate noise latent for this chunk ---
        latents = randn_tensor(
            (b, num_channels_latents, t, h, w),
            generator=batch.generator,
            device=device,
            dtype=condition_full.dtype,
        )

        # --- Timesteps ---
        timesteps = torch.tensor(
            server_args.pipeline_config.dmd_denoising_steps, dtype=torch.long
        ).cpu()
        if server_args.pipeline_config.warp_denoising_step:
            scheduler_timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            )
            timesteps = scheduler_timesteps[1000 - timesteps]
        timesteps = timesteps.to(device)
        logger.info("Using timesteps: %s", timesteps)

        # --- Transformer kwargs ---
        # Note: bypass prepare_extra_func_kwargs because
        # CausalWanTransformer3DModel.forward uses (*args, **kwargs) which
        # causes inspect-based filtering to drop all keyword arguments.
        # The underlying _forward_inference accepts these explicitly.
        image_embeds = getattr(batch, "image_embeds", [])
        if len(image_embeds) > 0:
            image_embeds = [ie.to(target_dtype) for ie in image_embeds]

        image_kwargs = {
            "encoder_hidden_states_image": image_embeds,
        }

        pos_cond_kwargs = server_args.pipeline_config.prepare_pos_cond_kwargs(
            batch,
            self.device,
            getattr(self.transformer, "rotary_emb", None),
            dtype=target_dtype,
        )

        if self.attn_backend.get_enum() == AttentionBackendEnum.SLIDING_TILE_ATTN:
            self.prepare_sta_param(batch, server_args)

        prompt_embeds = server_args.pipeline_config.get_pos_prompt_embeds(batch)

        # --- KV cache from session state ---
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
                batch_size=b, dtype=target_dtype, device=device
            )
            self._initialize_crossattn_cache(
                batch_size=b,
                max_text_len=server_args.pipeline_config.text_encoder_configs[
                    0
                ].arch_config.text_len,
                dtype=target_dtype,
                device=device,
            )
            kv_cache1 = cache_state.kv_cache = self.kv_cache1
            crossattn_cache = cache_state.crossattn_cache = self.crossattn_cache
            # Reset frame position on cache reset
            cache_state.current_chunk_start_frame = 0
            cache_state.chunk_idx = 0
        else:
            for block_index in range(self.num_transformer_blocks):
                crossattn_cache[block_index]["is_init"] = False

        current_start_frame = cache_state.current_chunk_start_frame

        # Slice condition to current chunk
        condition_chunks = condition_full.split(t, dim=2)
        cond_idx = min(cache_state.chunk_idx, len(condition_chunks) - 1)
        condition = condition_chunks[cond_idx]

        # --- Denoising loop (single chunk) ---
        current_latents = latents
        noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
        video_raw_latent_shape = noise_latents_btchw.shape
        attn_metadata = None

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t_cur in enumerate(timesteps):
                noise_latents = noise_latents_btchw.clone()

                # Concat [noise, condition] along channel dim
                latent_model_input = torch.cat(
                    [current_latents, condition], dim=1
                ).to(target_dtype)

                t_expand = t_cur.repeat(b)

                # Attention metadata
                if (
                    self.attn_backend.get_enum()
                    == AttentionBackendEnum.VIDEO_SPARSE_ATTN
                ):
                    builder_cls = self.attn_backend.get_builder_cls()
                    if builder_cls is not None:
                        attn_metadata = builder_cls().build(
                            current_timestep=i,
                            raw_latent_shape=(t, h, w),
                            patch_size=server_args.pipeline_config.dit_config.patch_size,
                            STA_param=batch.STA_param,
                            VSA_sparsity=server_args.attention_backend_config.VSA_sparsity,
                            device=device,
                        )

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
                        (b, 1), device=device, dtype=torch.long,
                    )
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
                x0_btchw = pred_noise_to_pred_video(
                    pred_noise=pred_noise_btchw.flatten(0, 1),
                    noise_input_latent=noise_latents.flatten(0, 1),
                    timestep=t_expand,
                    scheduler=self.scheduler,
                ).unflatten(0, pred_noise_btchw.shape[:2])

                if i < len(timesteps) - 1:
                    next_timestep = timesteps[i + 1] * torch.ones(
                        [1], dtype=torch.long, device=device
                    )
                    noise = torch.randn(
                        video_raw_latent_shape,
                        dtype=x0_btchw.dtype,
                        generator=(
                            batch.generator[0]
                            if isinstance(batch.generator, list)
                            else batch.generator
                        ),
                        device=device,
                    )
                    noise_latents_btchw = self.scheduler.add_noise(
                        x0_btchw.flatten(0, 1),
                        noise.flatten(0, 1),
                        next_timestep,
                    ).unflatten(0, x0_btchw.shape[:2])
                    current_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)
                else:
                    current_latents = x0_btchw.permute(0, 2, 1, 3, 4)

                if progress_bar is not None:
                    progress_bar.update()

        # --- KV cache update: forward with clean x0 + condition ---
        context_noise = getattr(server_args.pipeline_config, "context_noise", 0)
        t_context = torch.ones([b], device=device, dtype=torch.long) * int(
            context_noise
        )
        context_input = torch.cat(
            [current_latents, condition], dim=1
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
            _ = self.transformer(
                context_input,
                prompt_embeds,
                t_context.unsqueeze(1),
                kv_cache=kv_cache1,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                start_frame=current_start_frame,
                **image_kwargs,
                **pos_cond_kwargs,
            )

        # Advance cumulative frame position
        cache_state.current_chunk_start_frame += t
        cache_state.chunk_idx += 1

        # Output denoised latents for decoder
        batch.latents = current_latents
        batch.raw_latent_shape = current_latents.shape
        if not persist_cache_state:
            cache_state.dispose()
        return batch
