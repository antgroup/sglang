# SPDX-License-Identifier: Apache-2.0
"""
LingBot-World causal DMD denoising stage.

Extends CausalDMDDenoisingStage with:
- I2V condition concatenation ([noise, condition] along channel dim)
- Per-chunk noise generation (no LatentPreparationStage needed)
- Session-persistent KV cache with cumulative frame position tracking
- Optional CUDA graph acceleration after KV window warmup
"""

from __future__ import annotations

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.lingbot_world import (
    KVCacheSteadyStateIndices,
)
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
from sglang.multimodal_gen.runtime.utils.lingbot_world_cuda_graph import (
    LingBotWorldCudaGraphRunner,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LingBotWorldCausalDMDRealtimeState(BaseRealtimeState):
    """Persists KV cache and frame position across chunks in a realtime session."""

    def __init__(self):
        super().__init__()
        self.current_chunk_start_frame: int = 0
        self.chunk_idx: int = 0
        self.cuda_graph_runner: LingBotWorldCudaGraphRunner | None = None


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

    def _initialize_kv_cache(
        self,
        batch_size,
        dtype,
        device,
        *,
        sequence_shard_enabled: bool = False,
    ) -> None:
        kv_cache1 = []
        num_attention_heads = self.transformer.num_attention_heads
        if sequence_shard_enabled:
            ulysses_world_size = get_ulysses_parallel_world_size()
            if ulysses_world_size <= 1:
                raise ValueError(
                    "LingBot causal sequence sharding requires ulysses_degree > 1."
                )
            if get_ring_parallel_world_size() > 1:
                raise NotImplementedError(
                    "LingBot causal sequence sharding currently supports ring_degree = 1 only."
                )
            if num_attention_heads % ulysses_world_size != 0:
                raise ValueError(
                    f"num_attention_heads ({num_attention_heads}) must be divisible by ulysses_degree ({ulysses_world_size})."
                )
            num_attention_heads //= ulysses_world_size
        attention_head_dim = self.transformer.attention_head_dim
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = self.frame_seq_length * self.sliding_window_num_frames

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [
                            batch_size,
                            kv_cache_size,
                            num_attention_heads,
                            attention_head_dim,
                        ],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [
                            batch_size,
                            kv_cache_size,
                            num_attention_heads,
                            attention_head_dim,
                        ],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "local_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "global_end_index_int": 0,
                    "local_end_index_int": 0,
                }
            )

        self.kv_cache1 = kv_cache1

    def _should_use_cuda_graph(self, cache_state, batch) -> bool:
        """Check if CUDA graph mode should be used for this chunk."""
        from sglang.multimodal_gen import envs

        if not envs.SGLANG_LINGBOT_CUDA_GRAPH_ENABLED:
            return False
        # NCCL all-to-all in CUDA graph is unreliable; disable for ulysses SP.
        if (
            getattr(batch, "enable_sequence_shard", False)
            and get_ulysses_parallel_world_size() > 1
        ):
            return False
        return True

    def _compute_kv_indices(
        self, cache_state, kv_cache1: list[dict],
    ) -> tuple[KVCacheSteadyStateIndices, bool, KVCacheSteadyStateIndices]:
        """Compute KV cache indices for the current chunk.

        Returns (first_call_idx, first_needs_roll, subsequent_idx).
        - Growing phase (cache not full): no roll, indices advance per chunk.
        - Steady phase (cache full): first call rolls, subsequent calls overwrite.
        """
        from sglang.multimodal_gen.runtime.models.dits.lingbot_world import (
            KVCacheSteadyStateIndices,
        )
        arch = self.transformer.config.arch_config
        kv_cache_size = kv_cache1[0]["k"].shape[1]
        num_new_tokens = self.num_frames_per_block * self.frame_seq_length
        sink_tokens = arch.sink_size * self.frame_seq_length
        max_attention_size = (
            32760 if arch.local_attn_size == -1
            else arch.local_attn_size * self.frame_seq_length
        )
        window_chunks = arch.sliding_window_num_frames // self.num_frames_per_block
        chunk_idx = cache_state.chunk_idx

        if chunk_idx < window_chunks:
            # Growing phase: no eviction, write at advancing position
            write_start = chunk_idx * num_new_tokens
            write_end = write_start + num_new_tokens
            attn_end = write_end
            attn_start = max(0, attn_end - max_attention_size)
            idx = KVCacheSteadyStateIndices.__new__(KVCacheSteadyStateIndices)
            idx.roll_dst_start = 0
            idx.roll_dst_end = 0
            idx.roll_src_start = 0
            idx.roll_src_end = 0
            idx.write_start = write_start
            idx.write_end = write_end
            idx.attn_start = attn_start
            idx.attn_end = attn_end
            return idx, False, idx  # no roll for any call
        else:
            # Steady phase: eviction + roll on first call
            first_idx = KVCacheSteadyStateIndices(
                kv_cache_size=kv_cache_size,
                num_new_tokens=num_new_tokens,
                sink_tokens=sink_tokens,
                max_attention_size=max_attention_size,
            )
            # Subsequent calls: same write/attn but no roll
            sub_idx = KVCacheSteadyStateIndices.__new__(KVCacheSteadyStateIndices)
            sub_idx.roll_dst_start = 0
            sub_idx.roll_dst_end = 0
            sub_idx.roll_src_start = 0
            sub_idx.roll_src_end = 0
            sub_idx.write_start = first_idx.write_start
            sub_idx.write_end = first_idx.write_end
            sub_idx.attn_start = first_idx.attn_start
            sub_idx.attn_end = first_idx.attn_end
            return first_idx, True, sub_idx

    def _forward_cuda_graph(
        self,
        batch: Req,
        server_args: ServerArgs,
        cache_state: LingBotWorldCausalDMDRealtimeState,
        condition: torch.Tensor,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds,
        kv_cache1: list[dict],
        image_embeds: list,
        c2ws_plucker_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """Denoising loop using CUDA graph replay."""
        target_dtype = torch.bfloat16
        device = latents.device
        b = latents.shape[0]
        t = self.num_frames_per_block
        h = latents.shape[3]
        w = latents.shape[4]
        current_start_frame = cache_state.current_chunk_start_frame

        # Lazy-init the graph runner (shared across chunks & sessions)
        runner = cache_state.cuda_graph_runner
        if runner is None:
            runner = LingBotWorldCudaGraphRunner(self.transformer)
            cache_state.cuda_graph_runner = runner

        # Compute KV indices for this chunk
        first_idx, first_needs_roll, sub_idx = self._compute_kv_indices(
            cache_state, kv_cache1,
        )

        # Pre-compute ROPE for this chunk
        freqs_cis = self.transformer.compute_rope(
            num_frames=t, height=h, width=w,
            start_frame=current_start_frame, device=device,
        )

        enc_hs = prompt_embeds[0] if isinstance(prompt_embeds, list) else prompt_embeds
        enc_img = image_embeds[0] if len(image_embeds) > 0 else None
        common_replay_kwargs = dict(
            encoder_hidden_states=enc_hs,
            kv_cache=kv_cache1,
            c2ws_plucker_emb=c2ws_plucker_emb,
            encoder_hidden_states_image=enc_img,
            autocast_dtype=target_dtype,
        )

        # Denoising loop with graph replay
        current_latents = latents
        noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
        video_raw_latent_shape = noise_latents_btchw.shape

        for i, t_cur in enumerate(timesteps):
            noise_latents = noise_latents_btchw.clone()
            latent_model_input = torch.cat([current_latents, condition], dim=1).to(
                target_dtype
            )
            t_expand = t_cur.repeat(b)
            t_expanded = t_cur * torch.ones((b, 1), device=device, dtype=torch.long)

            # First call may need roll; subsequent calls never roll
            if i == 0:
                pred_noise = runner.replay(
                    latent_model_input, t_expanded, freqs_cis,
                    kv_idx=first_idx, with_roll=first_needs_roll,
                    **common_replay_kwargs,
                )
            else:
                pred_noise = runner.replay(
                    latent_model_input, t_expanded, freqs_cis,
                    kv_idx=sub_idx, with_roll=False,
                    **common_replay_kwargs,
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

        # Context update (no roll, overwrite same positions)
        context_noise = getattr(server_args.pipeline_config, "context_noise", 0)
        t_context_val = int(context_noise)
        t_context = torch.ones((b, 1), device=device, dtype=torch.long) * t_context_val
        context_input = torch.cat([current_latents, condition], dim=1).to(target_dtype)
        runner.replay(
            context_input, t_context, freqs_cis,
            kv_idx=sub_idx, with_roll=False,
            **common_replay_kwargs,
        )

        return current_latents

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
        logger.debug("Using timesteps: %s", timesteps)

        # --- Transformer kwargs ---
        # Note: bypass prepare_extra_func_kwargs because
        # CausalWanTransformer3DModel.forward uses (*args, **kwargs) which
        # causes inspect-based filtering to drop all keyword arguments.
        # The underlying _forward_inference accepts these explicitly.
        image_embeds = getattr(batch, "image_embeds", [])
        if len(image_embeds) > 0:
            image_embeds = [ie.to(target_dtype) for ie in image_embeds]

        prompt_embeds = server_args.pipeline_config.get_pos_prompt_embeds(batch)

        # --- KV cache from session state ---
        cache_state, persist_cache_state = self._get_cache_state(batch)
        kv_cache1 = cache_state.kv_cache
        crossattn_cache = cache_state.crossattn_cache
        sequence_shard_enabled = bool(
            getattr(batch, "enable_sequence_shard", False)
            and get_ulysses_parallel_world_size() > 1
        )
        expected_cache_heads = self.transformer.num_attention_heads
        if sequence_shard_enabled:
            if get_ring_parallel_world_size() > 1:
                raise NotImplementedError(
                    "LingBot causal sequence sharding currently supports ulysses_degree > 1 with ring_degree = 1 only."
                )
            expected_cache_heads //= get_ulysses_parallel_world_size()

        should_reset_cache = (
            batch.block_idx == 0
            or kv_cache1 is None
            or crossattn_cache is None
            or len(kv_cache1) != self.num_transformer_blocks
            or len(crossattn_cache) != self.num_transformer_blocks
            or kv_cache1[0]["k"].shape[2] != expected_cache_heads
        )

        if should_reset_cache:
            self._initialize_kv_cache(
                batch_size=b,
                dtype=target_dtype,
                device=device,
                sequence_shard_enabled=sequence_shard_enabled,
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
            # Invalidate CUDA graph on session reset
            if cache_state.cuda_graph_runner is not None:
                cache_state.cuda_graph_runner.dispose()
                cache_state.cuda_graph_runner = None
        else:
            for block_index in range(self.num_transformer_blocks):
                crossattn_cache[block_index]["is_init"] = False

        current_start_frame = cache_state.current_chunk_start_frame

        # Slice condition to current chunk
        condition_chunks = condition_full.split(t, dim=2)
        cond_idx = min(cache_state.chunk_idx, len(condition_chunks) - 1)
        condition = condition_chunks[cond_idx]

        # Camera conditioning
        c2ws_plucker_emb = getattr(batch, "c2ws_plucker_emb", None)

        # ====== CUDA Graph path ======
        if self._should_use_cuda_graph(cache_state, batch):
            current_latents = self._forward_cuda_graph(
                batch=batch,
                server_args=server_args,
                cache_state=cache_state,
                condition=condition,
                latents=latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                kv_cache1=kv_cache1,
                image_embeds=image_embeds,
                c2ws_plucker_emb=c2ws_plucker_emb,
            )

            # Advance cumulative frame position
            cache_state.current_chunk_start_frame += t
            cache_state.chunk_idx += 1

            batch.latents = current_latents
            batch.raw_latent_shape = current_latents.shape
            if not persist_cache_state:
                cache_state.dispose()
            return batch

        # ====== Eager path (original) ======
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

        # --- Denoising loop (single chunk) ---
        current_latents = latents
        noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
        video_raw_latent_shape = noise_latents_btchw.shape
        attn_metadata = None

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t_cur in enumerate(timesteps):
                noise_latents = noise_latents_btchw

                # Concat [noise, condition] along channel dim
                latent_model_input = torch.cat([current_latents, condition], dim=1).to(
                    target_dtype
                )

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
                        (b, 1),
                        device=device,
                        dtype=torch.long,
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
        context_input = torch.cat([current_latents, condition], dim=1).to(target_dtype)
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
