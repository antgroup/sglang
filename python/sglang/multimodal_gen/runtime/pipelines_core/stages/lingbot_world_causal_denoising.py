# SPDX-License-Identifier: Apache-2.0
"""
LingBot-World causal DMD denoising stage.

Extends CausalDMDDenoisingStage with:
- I2V condition concatenation ([noise, condition] along channel dim)
- Per-chunk noise generation (no LatentPreparationStage needed)
- Session-persistent KV cache with cumulative frame position tracking
"""

from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_ulysses_parallel_world_size,
)
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
        self.interactive_kv_consecutive_still_chunks: int = 0
        self.interactive_kv_sample_num_frames: int | None = None
        self.kv_cache_valid_local_start_frame: int = 0
        self.kv_reset_replay_chunks: dict[int, dict[str, Any]] = {}

    def dispose(self):
        super().dispose()
        self.kv_reset_replay_chunks.clear()


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
        server_args: ServerArgs,
        interactive_kv_window_active: bool = False,
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
            effective_sliding_window_num_frames = self.sliding_window_num_frames
        else:
            effective_sliding_window_num_frames = (
                self._effective_sliding_window_num_frames(
                    server_args,
                    interactive_kv_window_active=interactive_kv_window_active,
                )
            )
            kv_cache_size = self.frame_seq_length * effective_sliding_window_num_frames
        logger.info(
            "LingBot KV cache init: batch=%s layers=%s frame_seq_length=%s "
            "num_frames_per_block=%s sliding_window_num_frames=%s effective_sliding_window_num_frames=%s local_attn_size=%s "
            "sink_size=%s kv_cache_tokens=%s heads=%s head_dim=%s sequence_shard=%s",
            batch_size,
            self.num_transformer_blocks,
            self.frame_seq_length,
            self.num_frames_per_block,
            self.sliding_window_num_frames,
            effective_sliding_window_num_frames,
            self.local_attn_size,
            self.transformer.config.arch_config.sink_size,
            kv_cache_size,
            num_attention_heads,
            attention_head_dim,
            sequence_shard_enabled,
        )

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

    def _uses_interactive_kv_window(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> bool:
        if not self._interactive_kv_window_enabled(server_args):
            return False
        batch_extra = getattr(batch, "extra", None)
        return bool(batch_extra is not None and "actions" in batch_extra)

    @staticmethod
    def _interactive_kv_window_enabled(server_args: ServerArgs) -> bool:
        return bool(
            getattr(
                server_args.pipeline_config,
                "interactive_kv_window_enable",
                False,
            )
        )

    def _effective_sliding_window_num_frames(
        self,
        server_args: ServerArgs,
        *,
        interactive_kv_window_active: bool,
    ) -> int:
        window = int(self.sliding_window_num_frames)
        if self.local_attn_size != -1:
            return window

        pipeline_config = server_args.pipeline_config
        sink_size = int(getattr(self.transformer.config.arch_config, "sink_size", 0))
        current_chunk = int(self.num_frames_per_block)
        if interactive_kv_window_active:
            moving_window = max(
                0, int(getattr(pipeline_config, "interactive_kv_moving_window", 0))
            )
            window = max(window, sink_size + moving_window + current_chunk)
        if self._kv_cache_reset_enabled(server_args):
            reset_replay_chunks = self._kv_cache_reset_keep_prev_chunks(server_args) + 1
            reset_replay_frames = reset_replay_chunks * current_chunk
            reset_gap = self._resolve_kv_cache_reset_rope_gap_latent_frames(server_args)
            window = max(window, sink_size + reset_gap + reset_replay_frames)
        return window

    def _base_kv_sample_num_frames(self) -> int | None:
        sink_size = int(getattr(self.transformer.config.arch_config, "sink_size", 0))
        sample_frames = (
            int(self.sliding_window_num_frames)
            - sink_size
            - int(self.num_frames_per_block)
        )
        return sample_frames if sample_frames > 0 else None

    def _kv_cache_reset_enabled(self, server_args: ServerArgs) -> bool:
        return (
            bool(getattr(server_args.pipeline_config, "kv_cache_reset_enable", False))
            and self.local_attn_size == -1
        )

    def _kv_cache_reset_max_window_latent_frames(self, server_args: ServerArgs) -> int:
        return int(
            getattr(
                server_args.pipeline_config,
                "kv_cache_reset_max_window_latent_frames",
                88,
            )
        )

    def _kv_cache_reset_keep_prev_chunks(self, server_args: ServerArgs) -> int:
        return max(
            0,
            int(
                getattr(
                    server_args.pipeline_config,
                    "kv_cache_reset_keep_prev_chunks",
                    1,
                )
            ),
        )

    def _resolve_kv_cache_reset_rope_gap_latent_frames(
        self,
        server_args: ServerArgs,
    ) -> int:
        configured = int(
            getattr(
                server_args.pipeline_config,
                "kv_cache_reset_rope_gap_latent_frames",
                -1,
            )
        )
        if configured >= 0:
            return configured
        chunk_frames = max(1, int(self.num_frames_per_block))
        max_window = max(0, self._kv_cache_reset_max_window_latent_frames(server_args))
        return max(chunk_frames, min(max_window // 4, 2 * chunk_frames))

    def _should_reset_kv_rope_for_next_chunk(
        self,
        server_args: ServerArgs,
        *,
        current_start_frame: int,
        current_chunk_frames: int,
    ) -> bool:
        if not self._kv_cache_reset_enabled(server_args):
            return False
        max_window = self._kv_cache_reset_max_window_latent_frames(server_args)
        if max_window <= 0:
            return False
        next_chunk_frames = int(self.num_frames_per_block)
        return (
            int(current_start_frame) + int(current_chunk_frames) + next_chunk_frames
            > max_window
        )

    @staticmethod
    def _detach_replay_value(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, tuple):
            return tuple(
                LingBotWorldCausalDMDDenoisingStage._detach_replay_value(item)
                for item in value
            )
        if isinstance(value, list):
            return [
                LingBotWorldCausalDMDDenoisingStage._detach_replay_value(item)
                for item in value
            ]
        if isinstance(value, dict):
            return {
                key: LingBotWorldCausalDMDDenoisingStage._detach_replay_value(item)
                for key, item in value.items()
            }
        return value

    def _remember_kv_replay_chunk(
        self,
        cache_state: LingBotWorldCausalDMDRealtimeState,
        server_args: ServerArgs,
        *,
        chunk_idx: int,
        latents: torch.Tensor,
        condition: torch.Tensor,
        pos_cond_kwargs: dict[str, Any],
    ) -> None:
        if not self._kv_cache_reset_enabled(server_args):
            return
        chunk_idx = int(chunk_idx)
        cache_state.kv_reset_replay_chunks[chunk_idx] = {
            "latents": latents.detach(),
            "condition": condition.detach(),
            "pos_cond_kwargs": self._detach_replay_value(pos_cond_kwargs),
        }
        min_keep_chunk = chunk_idx - self._kv_cache_reset_keep_prev_chunks(server_args)
        for old_chunk_idx in list(cache_state.kv_reset_replay_chunks):
            if old_chunk_idx < min_keep_chunk:
                del cache_state.kv_reset_replay_chunks[old_chunk_idx]

    @staticmethod
    def _kv_replay_condition_mask_channels(
        server_args: ServerArgs,
        *,
        latents: torch.Tensor,
        condition: torch.Tensor,
    ) -> int:
        pipeline_config = getattr(server_args, "pipeline_config", None)
        vae_config = getattr(pipeline_config, "vae_config", None)
        vae_arch_config = getattr(vae_config, "arch_config", None)
        mask_channels = getattr(vae_arch_config, "temporal_compression_ratio", None)
        if mask_channels is None:
            mask_channels = int(condition.shape[1]) - int(latents.shape[1])
        return max(0, int(mask_channels))

    def _anchor_kv_replay_first_frame(
        self,
        server_args: ServerArgs,
        *,
        latents: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if latents.shape[2] <= 0 or condition.shape[2] <= 0:
            return condition

        latent_channels = int(latents.shape[1])
        mask_channels = self._kv_replay_condition_mask_channels(
            server_args,
            latents=latents,
            condition=condition,
        )
        if (
            mask_channels <= 0
            or latent_channels <= 0
            or int(condition.shape[1]) < mask_channels + latent_channels
        ):
            return condition

        anchored_condition = condition.clone()
        anchored_condition[:, :mask_channels, 0:1] = 1.0
        anchored_condition[
            :, mask_channels : mask_channels + latent_channels, 0:1
        ] = latents[:, :, 0:1].to(
            device=anchored_condition.device,
            dtype=anchored_condition.dtype,
        )
        return anchored_condition

    @staticmethod
    def _set_kv_cache_indices(kv_cache, end_tokens: int) -> None:
        end_tokens = int(end_tokens)
        for block_cache in kv_cache:
            block_cache["global_end_index"].fill_(end_tokens)
            block_cache["local_end_index"].fill_(end_tokens)
            block_cache["global_end_index_int"] = end_tokens
            block_cache["local_end_index_int"] = end_tokens

    def _copy_kv_sink_prefix(self, old_kv_cache, new_kv_cache) -> int:
        if old_kv_cache is None:
            return 0
        sink_tokens = int(
            getattr(self.transformer.config.arch_config, "sink_size", 0)
        ) * int(self.frame_seq_length)
        if sink_tokens <= 0:
            return 0
        old_local_end = int(old_kv_cache[0].get("local_end_index_int", 0))
        sink_tokens = min(
            sink_tokens,
            old_local_end,
            int(old_kv_cache[0]["k"].shape[1]),
            int(new_kv_cache[0]["k"].shape[1]),
        )
        if sink_tokens <= 0:
            return 0
        for old_block, new_block in zip(old_kv_cache, new_kv_cache):
            new_block["k"][:, :sink_tokens].copy_(old_block["k"][:, :sink_tokens])
            new_block["v"][:, :sink_tokens].copy_(old_block["v"][:, :sink_tokens])
        self._set_kv_cache_indices(new_kv_cache, sink_tokens)
        return sink_tokens

    def _write_kv_context_chunk(
        self,
        batch: Req,
        *,
        latents: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        kv_cache,
        crossattn_cache,
        current_start_frame: int,
        kv_cache_sample_num_frames: int | None,
        kv_cache_valid_local_start_frame: int,
        t_context: torch.Tensor,
        attn_metadata,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
    ) -> None:
        context_input = torch.cat([latents, condition], dim=1).to(target_dtype)
        with (
            torch.autocast(
                device_type=current_platform.device_type,
                dtype=target_dtype,
                enabled=autocast_enabled,
            ),
            set_forward_context(
                current_timestep=-1,
                attn_metadata=attn_metadata,
                forward_batch=batch,
            ),
        ):
            _ = self.transformer(
                context_input,
                prompt_embeds,
                t_context.unsqueeze(1),
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=int(current_start_frame) * self.frame_seq_length,
                start_frame=int(current_start_frame),
                kv_cache_sample_num_frames=kv_cache_sample_num_frames,
                kv_cache_valid_local_start_frame=kv_cache_valid_local_start_frame,
                skip_final_projection=True,
                **image_kwargs,
                **pos_cond_kwargs,
            )

    def _maybe_reset_kv_rope_for_next_chunk(
        self,
        cache_state: LingBotWorldCausalDMDRealtimeState,
        batch: Req,
        server_args: ServerArgs,
        *,
        old_kv_cache,
        crossattn_cache,
        current_chunk_idx: int,
        current_start_frame: int,
        current_chunk_frames: int,
        batch_size: int,
        device,
        target_dtype: torch.dtype,
        sequence_shard_enabled: bool,
        prompt_embeds,
        image_kwargs: dict[str, Any],
        kv_cache_sample_num_frames: int | None,
        t_context: torch.Tensor,
        attn_metadata,
        autocast_enabled: bool,
    ) -> None:
        if not self._should_reset_kv_rope_for_next_chunk(
            server_args,
            current_start_frame=current_start_frame,
            current_chunk_frames=current_chunk_frames,
        ):
            return

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=target_dtype,
            device=device,
            server_args=server_args,
            interactive_kv_window_active=self._interactive_kv_window_enabled(
                server_args
            ),
            sequence_shard_enabled=sequence_shard_enabled,
        )
        new_kv_cache = self.kv_cache1
        preserved_sink_tokens = self._copy_kv_sink_prefix(old_kv_cache, new_kv_cache)
        preserved_sink_frames = preserved_sink_tokens // int(self.frame_seq_length)
        rope_gap_frames = self._resolve_kv_cache_reset_rope_gap_latent_frames(
            server_args
        )
        replay_start_frame = int(preserved_sink_frames) + int(rope_gap_frames)
        valid_local_start_frame = (
            replay_start_frame if replay_start_frame > int(preserved_sink_frames) else 0
        )
        cache_state.kv_cache = new_kv_cache
        cache_state.kv_cache_valid_local_start_frame = valid_local_start_frame

        keep_prev_chunks = self._kv_cache_reset_keep_prev_chunks(server_args)
        replay_start_chunk = max(0, int(current_chunk_idx) - keep_prev_chunks)
        replay_chunk_ids = [
            chunk_idx
            for chunk_idx in range(replay_start_chunk, int(current_chunk_idx) + 1)
            if chunk_idx in cache_state.kv_reset_replay_chunks
        ]

        replay_frame = replay_start_frame
        for replay_index, replay_chunk_id in enumerate(replay_chunk_ids):
            replay_entry = cache_state.kv_reset_replay_chunks[replay_chunk_id]
            replay_latents = replay_entry["latents"]
            replay_condition = replay_entry["condition"]
            if replay_index == 0:
                replay_condition = self._anchor_kv_replay_first_frame(
                    server_args,
                    latents=replay_latents,
                    condition=replay_condition,
                )
            self._write_kv_context_chunk(
                batch,
                latents=replay_latents,
                condition=replay_condition,
                prompt_embeds=prompt_embeds,
                image_kwargs=image_kwargs,
                pos_cond_kwargs=replay_entry["pos_cond_kwargs"],
                kv_cache=new_kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_frame=replay_frame,
                kv_cache_sample_num_frames=kv_cache_sample_num_frames,
                kv_cache_valid_local_start_frame=valid_local_start_frame,
                t_context=t_context,
                attn_metadata=attn_metadata,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
            )
            replay_frame += int(replay_latents.shape[2])

        cache_state.current_chunk_start_frame = replay_frame
        logger.info(
            "LingBot KV/RoPE reset: session_id=%s block_idx=%s "
            "replay_chunks=%s local_start_before=%s local_start_after=%s "
            "max_window=%s keep_prev_chunks=%s sink_frames=%s "
            "rope_gap_frames=%s valid_local_start=%s",
            (
                getattr(batch, "extra", {}).get("realtime_session_id")
                if isinstance(getattr(batch, "extra", None), dict)
                else None
            ),
            getattr(batch, "block_idx", None),
            replay_chunk_ids,
            current_start_frame,
            replay_frame,
            self._kv_cache_reset_max_window_latent_frames(server_args),
            keep_prev_chunks,
            preserved_sink_frames,
            rope_gap_frames,
            valid_local_start_frame,
        )

    @staticmethod
    def _chunk_has_camera_motion(actions) -> bool:
        if not actions:
            return False
        for frame_actions in actions:
            if frame_actions:
                return True
        return False

    def _get_interactive_kv_sample_num_frames(
        self,
        cache_state: LingBotWorldCausalDMDRealtimeState,
        batch: Req,
        server_args: ServerArgs,
    ) -> int | None:
        pipeline_config = server_args.pipeline_config
        if not bool(getattr(pipeline_config, "interactive_kv_window_enable", False)):
            return None
        if not self._uses_interactive_kv_window(batch, server_args):
            kv_cache_sample_num_frames = self._base_kv_sample_num_frames()
            batch_extra = getattr(batch, "extra", None)
            logger.info(
                "LingBot interactive KV window: session_id=%s block_idx=%s "
                "active=false kv_cache_sample_num_frames=%s",
                (
                    batch_extra.get("realtime_session_id")
                    if isinstance(batch_extra, dict)
                    else None
                ),
                getattr(batch, "block_idx", None),
                kv_cache_sample_num_frames,
            )
            return kv_cache_sample_num_frames

        moving_window = int(
            getattr(pipeline_config, "interactive_kv_moving_window", 12)
        )
        still_window = int(getattr(pipeline_config, "interactive_kv_still_window", 3))
        still_chunks_threshold = max(
            1, int(getattr(pipeline_config, "interactive_kv_still_chunks", 2))
        )
        if cache_state.interactive_kv_sample_num_frames is None:
            cache_state.interactive_kv_sample_num_frames = moving_window

        batch_extra = getattr(batch, "extra", None)
        assert batch_extra is not None
        actions = batch_extra.get("actions")
        has_motion = self._chunk_has_camera_motion(actions)
        if has_motion:
            cache_state.interactive_kv_consecutive_still_chunks = 0
            cache_state.interactive_kv_sample_num_frames = moving_window
        else:
            cache_state.interactive_kv_consecutive_still_chunks += 1
            if (
                cache_state.interactive_kv_consecutive_still_chunks
                >= still_chunks_threshold
            ):
                cache_state.interactive_kv_sample_num_frames = still_window

        mode = (
            "moving"
            if has_motion
            else (
                "still"
                if cache_state.interactive_kv_sample_num_frames == still_window
                else "pending_still"
            )
        )
        logger.info(
            "LingBot interactive KV window: session_id=%s block_idx=%s "
            "active=true has_motion=%s mode=%s still_chunks=%s/%s "
            "moving_window=%s still_window=%s kv_cache_sample_num_frames=%s",
            batch_extra.get("realtime_session_id"),
            getattr(batch, "block_idx", None),
            has_motion,
            mode,
            cache_state.interactive_kv_consecutive_still_chunks,
            still_chunks_threshold,
            moving_window,
            still_window,
            cache_state.interactive_kv_sample_num_frames,
        )
        return cache_state.interactive_kv_sample_num_frames

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

        interactive_kv_window_enabled = self._interactive_kv_window_enabled(server_args)
        required_kv_cache_size = None
        if self.local_attn_size == -1 and (
            interactive_kv_window_enabled or self._kv_cache_reset_enabled(server_args)
        ):
            required_kv_cache_size = (
                self.frame_seq_length
                * self._effective_sliding_window_num_frames(
                    server_args,
                    interactive_kv_window_active=interactive_kv_window_enabled,
                )
            )

        should_reset_cache = (
            batch.block_idx == 0
            or kv_cache1 is None
            or crossattn_cache is None
            or len(kv_cache1) != self.num_transformer_blocks
            or len(crossattn_cache) != self.num_transformer_blocks
            or kv_cache1[0]["k"].shape[2] != expected_cache_heads
            or (
                required_kv_cache_size is not None
                and kv_cache1[0]["k"].shape[1] < required_kv_cache_size
            )
        )

        if should_reset_cache:
            self._initialize_kv_cache(
                batch_size=b,
                dtype=target_dtype,
                device=device,
                server_args=server_args,
                interactive_kv_window_active=interactive_kv_window_enabled,
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
            cache_state.interactive_kv_consecutive_still_chunks = 0
            cache_state.interactive_kv_sample_num_frames = None
            cache_state.kv_cache_valid_local_start_frame = 0
            cache_state.kv_reset_replay_chunks.clear()
        elif getattr(batch, "update_prompt_embeds", False):
            for block_cache in crossattn_cache:
                block_cache["is_init"] = False
            # Match torchtitan: keep replay chunks across prompt switches so a later
            # KV/RoPE reset can rebuild the local window from the same history.
            logger.info(
                "LingBot cross-attention cache reset for prompt embedding change: "
                "session_id=%s block_idx=%s event_ids=%s",
                batch.extra.get("realtime_session_id"),
                batch.block_idx,
                batch.extra.get("active_prompt_event_ids"),
            )
        # Keep cross-attention K/V cache across realtime chunks; LingBot text/image
        # conditions are session-static and are invalidated by the cache reset above.

        current_start_frame = cache_state.current_chunk_start_frame
        kv_cache_valid_local_start_frame = cache_state.kv_cache_valid_local_start_frame
        kv_cache_sample_num_frames = self._get_interactive_kv_sample_num_frames(
            cache_state,
            batch,
            server_args,
        )

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
                        kv_cache_sample_num_frames=kv_cache_sample_num_frames,
                        kv_cache_valid_local_start_frame=kv_cache_valid_local_start_frame,
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
        self._write_kv_context_chunk(
            batch,
            latents=current_latents,
            condition=condition,
            prompt_embeds=prompt_embeds,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            kv_cache=kv_cache1,
            crossattn_cache=crossattn_cache,
            current_start_frame=current_start_frame,
            kv_cache_sample_num_frames=kv_cache_sample_num_frames,
            kv_cache_valid_local_start_frame=kv_cache_valid_local_start_frame,
            t_context=t_context,
            attn_metadata=attn_metadata,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
        )

        # Advance cumulative frame position
        current_chunk_idx = cache_state.chunk_idx
        self._remember_kv_replay_chunk(
            cache_state,
            server_args,
            chunk_idx=current_chunk_idx,
            latents=current_latents,
            condition=condition,
            pos_cond_kwargs=pos_cond_kwargs,
        )
        cache_state.current_chunk_start_frame += t
        cache_state.chunk_idx += 1
        self._maybe_reset_kv_rope_for_next_chunk(
            cache_state,
            batch,
            server_args,
            old_kv_cache=kv_cache1,
            crossattn_cache=crossattn_cache,
            current_chunk_idx=current_chunk_idx,
            current_start_frame=current_start_frame,
            current_chunk_frames=t,
            batch_size=b,
            device=device,
            target_dtype=target_dtype,
            sequence_shard_enabled=sequence_shard_enabled,
            prompt_embeds=prompt_embeds,
            image_kwargs=image_kwargs,
            kv_cache_sample_num_frames=kv_cache_sample_num_frames,
            t_context=t_context,
            attn_metadata=attn_metadata,
            autocast_enabled=autocast_enabled,
        )

        # Output denoised latents for decoder
        batch.latents = current_latents
        batch.raw_latent_shape = current_latents.shape
        if not persist_cache_state:
            cache_state.dispose()
        return batch
