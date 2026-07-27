# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/Robbyant/lingbot-world

"""LingBot-World causal DMD denoising stage."""

from typing import Any

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDCachePolicy,
    CausalDMDDenoisingStage,
    CausalDMDForwardContext,
    CausalDMDRealtimeCacheContext,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_world.constants import (
    LINGBOT_C2WS_PLUCKER_EMB_CACHE,
    LINGBOT_CAM_CONDITIONER_CACHE,
    LINGBOT_CAMERA_ACTIONS_CONDITION,
    LINGBOT_INTERACTIVE_KV_WINDOW_CACHE,
    LINGBOT_KV_RESET_REPLAY_CACHE,
    LINGBOT_KV_RESET_SAMPLE_TOKENS_CACHE,
    LINGBOT_PROMPT_UPDATED_CONDITION,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LingBotWorldCausalDMDDenoisingStage(CausalDMDDenoisingStage):
    """Causal DMD denoising with I2V condition concatenation for LingBot-World.

    The LingBot-World transformer has ``in_channels = 36`` and expects
    ``[noise(16ch), condition(20ch)]`` concatenated along channel dim.
    Each call processes one chunk (num_frames_per_block frames).
    """

    def _get_causal_kv_cache_size(
        self,
        *,
        sequence_shard_enabled: bool = False,
    ) -> int:
        if self.local_attn_size != -1:
            return self.local_attn_size * self.num_token_per_frame

        return self.sliding_window_num_frames * self.num_token_per_frame

    def _causal_sequence_shard_enabled(self, batch: Req) -> bool:
        return bool(
            getattr(batch, "enable_sequence_shard", False)
            and get_ulysses_parallel_world_size() > 1
        )

    def _num_causal_cache_attention_heads(
        self,
        *,
        sequence_shard_enabled: bool,
    ) -> int:
        num_attention_heads = self.transformer.num_attention_heads
        if not sequence_shard_enabled:
            return num_attention_heads

        ulysses_world_size = get_ulysses_parallel_world_size()
        if get_ring_parallel_world_size() > 1:
            raise NotImplementedError(
                "LingBot causal sequence sharding currently supports ulysses_degree > 1 with ring_degree = 1 only."
            )
        if ulysses_world_size <= 1:
            raise ValueError(
                "LingBot causal sequence sharding requires ulysses_degree > 1."
            )
        if num_attention_heads % ulysses_world_size != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by ulysses_degree ({ulysses_world_size})."
            )
        return num_attention_heads // ulysses_world_size

    def _causal_kv_cache_kwargs(
        self,
        policy: CausalDMDCachePolicy,
    ) -> dict[str, Any]:
        return {
            "sequence_shard_enabled": policy.sequence_shard_enabled,
            "kv_cache_size": policy.expected_cache_tokens,
        }

    def _use_causal_cache_int_indices(
        self,
        *,
        sequence_shard_enabled: bool,
    ) -> bool:
        return True

    @staticmethod
    def _chunk_has_camera_motion(actions) -> bool:
        if not actions:
            return False
        for frame_actions in actions:
            if frame_actions:
                return True
        return False

    def _uses_interactive_kv_window(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> bool:
        if not self._interactive_kv_window_enabled(server_args):
            return False
        condition_inputs = getattr(batch, "condition_inputs", None) or {}
        return LINGBOT_CAMERA_ACTIONS_CONDITION in condition_inputs

    @staticmethod
    def _interactive_kv_window_enabled(server_args: ServerArgs) -> bool:
        config_enabled = bool(
            getattr(
                server_args.pipeline_config,
                "interactive_kv_window_enable",
                False,
            )
        )
        return config_enabled or envs.SGLANG_LINGBOT_ENABLE_INTERACTIVE_KV_WINDOW

    def _apply_causal_cache_overrides(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        self._reset_causal_cache_config_defaults()
        super()._apply_causal_cache_overrides(batch, server_args)
        self._sync_interactive_kv_cache_window(server_args)

    def _reset_causal_cache_config_defaults(self) -> None:
        arch_config = getattr(
            getattr(self.transformer, "config", None), "arch_config", None
        )
        if arch_config is None:
            return
        if hasattr(arch_config, "sink_size"):
            self.sink_size = int(arch_config.sink_size)
        if hasattr(arch_config, "sliding_window_num_frames"):
            self.sliding_window_num_frames = int(arch_config.sliding_window_num_frames)

    def _sync_interactive_kv_cache_window(self, server_args: ServerArgs) -> None:
        if self.local_attn_size != -1:
            return
        self.sliding_window_num_frames = self._effective_lingbot_kv_cache_num_frames(
            server_args
        )

    def _effective_lingbot_kv_cache_num_frames(
        self,
        server_args: ServerArgs,
    ) -> int:
        cache_window = int(self.sliding_window_num_frames)
        if self.local_attn_size != -1:
            return cache_window

        if self._interactive_kv_window_enabled(server_args):
            moving_window = self._moving_kv_sample_num_frames(server_args) or 0
            still_window = self._still_kv_sample_num_frames(server_args) or 0
            cache_window = max(
                cache_window,
                int(self.sink_size)
                + max(moving_window, still_window)
                + int(self.num_frames_per_block),
            )
        if self._kv_cache_reset_enabled(server_args):
            replay_chunks = self._kv_cache_reset_keep_prev_chunks(server_args) + 1
            replay_frames = replay_chunks * int(self.num_frames_per_block)
            cache_window = max(
                cache_window,
                int(self.sink_size)
                + self._resolve_kv_cache_reset_rope_gap_latent_frames(server_args)
                + replay_frames,
            )
        return cache_window

    def _build_realtime_causal_cache_policy(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> CausalDMDCachePolicy:
        policy = super()._build_realtime_causal_cache_policy(batch, server_args)
        if self._interactive_kv_window_enabled(server_args):
            policy.expected_cache_tokens = (
                self._effective_lingbot_kv_cache_num_frames(server_args)
                * self.num_token_per_frame
            )
        return policy

    @staticmethod
    def _should_reset_lingbot_crossattn_cache(batch: Req) -> bool:
        condition_inputs = getattr(batch, "condition_inputs", None) or {}
        return bool(condition_inputs.get(LINGBOT_PROMPT_UPDATED_CONDITION))

    def _sync_lingbot_crossattn_cache(
        self,
        batch: Req,
        cache_ctx: CausalDMDRealtimeCacheContext,
    ) -> None:
        if self._should_reset_lingbot_crossattn_cache(batch):
            self._reset_crossattn_cache(cache_ctx.crossattn_cache)

    def _prepare_realtime_causal_caches(
        self,
        batch: Req,
        server_args: ServerArgs,
        ctx: CausalDMDForwardContext,
    ) -> CausalDMDRealtimeCacheContext:
        cache_ctx = super()._prepare_realtime_causal_caches(batch, server_args, ctx)
        if batch.block_idx == 0:
            cache_ctx.cache_state.runtime_cache.pop(
                LINGBOT_KV_RESET_REPLAY_CACHE,
                None,
            )
            cache_ctx.cache_state.runtime_cache.pop(
                LINGBOT_KV_RESET_SAMPLE_TOKENS_CACHE,
                None,
            )
        self._sync_lingbot_crossattn_cache(batch, cache_ctx)
        return cache_ctx

    def _base_kv_sample_num_frames(self) -> int | None:
        sample_frames = (
            int(self.sliding_window_num_frames)
            - int(self.sink_size)
            - int(self.num_frames_per_block)
        )
        return sample_frames if sample_frames > 0 else None

    def _kv_cache_reset_enabled(self, server_args: ServerArgs) -> bool:
        return (
            bool(
                getattr(
                    server_args.pipeline_config,
                    "kv_cache_reset_enable",
                    False,
                )
            )
            and self.local_attn_size == -1
        )

    @staticmethod
    def _kv_cache_reset_max_window_latent_frames(
        server_args: ServerArgs,
    ) -> int:
        return int(
            getattr(
                server_args.pipeline_config,
                "kv_cache_reset_max_window_latent_frames",
                88,
            )
        )

    @staticmethod
    def _kv_cache_reset_keep_prev_chunks(server_args: ServerArgs) -> int:
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
        max_window = max(
            0,
            self._kv_cache_reset_max_window_latent_frames(server_args),
        )
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
        return (
            int(current_start_frame)
            + int(current_chunk_frames)
            + int(self.num_frames_per_block)
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
        cache_state,
        server_args: ServerArgs,
        *,
        chunk_idx: int,
        latents: torch.Tensor,
        condition: torch.Tensor,
        pos_cond_kwargs: dict[str, Any],
    ) -> None:
        if not self._kv_cache_reset_enabled(server_args):
            return
        replay_chunks = cache_state.runtime_cache.setdefault(
            LINGBOT_KV_RESET_REPLAY_CACHE,
            {},
        )
        chunk_idx = int(chunk_idx)
        replay_chunks[chunk_idx] = {
            "latents": latents.detach(),
            "condition": condition.detach(),
            "pos_cond_kwargs": self._detach_replay_value(pos_cond_kwargs),
        }
        min_keep_chunk = chunk_idx - self._kv_cache_reset_keep_prev_chunks(server_args)
        for old_chunk_idx in list(replay_chunks):
            if old_chunk_idx < min_keep_chunk:
                del replay_chunks[old_chunk_idx]

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
    def _set_kv_cache_indices(
        kv_cache,
        *,
        global_end_tokens: int,
        local_end_tokens: int,
    ) -> None:
        for block_cache in kv_cache:
            block_cache.reset_pinned_sink()
            block_cache._write_indices(
                global_end_index=int(global_end_tokens),
                local_end_index=int(local_end_tokens),
            )

    def _allocate_reset_kv_cache(self, old_kv_cache):
        template = old_kv_cache[0]
        return self._allocate_causal_kv_cache(
            batch_size=int(template.k.shape[0]),
            kv_cache_size=int(template.k.shape[1]),
            num_attention_heads=int(template.k.shape[2]),
            attention_head_dim=int(template.k.shape[3]),
            dtype=template.k.dtype,
            device=template.k.device,
            use_int_indices=template.global_end_index_int is not None,
            sink_tokens=int(template.sink_tokens),
            global_sink_tokens=int(template.global_sink_tokens),
            attention_window_size=int(template.attention_window_size),
            allow_growth=bool(template.allow_growth),
        )

    def _copy_kv_sink_prefix(self, old_kv_cache, new_kv_cache) -> int:
        if not old_kv_cache:
            return 0
        _, old_local_end = old_kv_cache[0]._read_indices()
        sink_tokens = min(
            int(self.sink_size) * int(self.num_token_per_frame),
            int(old_local_end),
            int(old_kv_cache[0].k.shape[1]),
            int(new_kv_cache[0].k.shape[1]),
        )
        if sink_tokens <= 0:
            return 0
        for old_block, new_block in zip(old_kv_cache, new_kv_cache):
            new_block.k[:, :sink_tokens].copy_(old_block.k[:, :sink_tokens])
            new_block.v[:, :sink_tokens].copy_(old_block.v[:, :sink_tokens])
        return sink_tokens

    def _write_kv_replay_context_chunk(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        latents: torch.Tensor,
        condition: torch.Tensor,
        prompt_embeds,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        kv_cache,
        crossattn_cache,
        current_start_frame: int,
        sample_tokens: int | None,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
    ) -> None:
        previous_sample_tokens = getattr(
            batch,
            "realtime_causal_kv_sample_tokens",
            None,
        )
        batch.realtime_causal_kv_sample_tokens = sample_tokens
        try:
            self._update_causal_context_cache(
                batch,
                server_args,
                context_input=torch.cat([latents, condition], dim=1),
                prompt_embeds=prompt_embeds,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_tokens=(
                    int(current_start_frame) * int(self.num_token_per_frame)
                ),
                start_frame=int(current_start_frame),
                image_kwargs=image_kwargs,
                pos_cond_kwargs=pos_cond_kwargs,
                attn_metadata=None,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
            )
        finally:
            batch.realtime_causal_kv_sample_tokens = previous_sample_tokens

    def _maybe_reset_kv_rope_for_next_chunk(
        self,
        cache_ctx: CausalDMDRealtimeCacheContext,
        batch: Req,
        server_args: ServerArgs,
        *,
        old_kv_cache,
        current_chunk_idx: int,
        current_start_frame: int,
        current_chunk_frames: int,
        prompt_embeds,
        image_kwargs: dict[str, Any],
        target_dtype: torch.dtype,
        autocast_enabled: bool,
    ) -> None:
        if not self._should_reset_kv_rope_for_next_chunk(
            server_args,
            current_start_frame=current_start_frame,
            current_chunk_frames=current_chunk_frames,
        ):
            return

        new_kv_cache = self._allocate_reset_kv_cache(old_kv_cache)
        preserved_sink_tokens = self._copy_kv_sink_prefix(
            old_kv_cache,
            new_kv_cache,
        )
        preserved_sink_frames = preserved_sink_tokens // int(self.num_token_per_frame)
        rope_gap_frames = self._resolve_kv_cache_reset_rope_gap_latent_frames(
            server_args
        )
        replay_start_frame = preserved_sink_frames + rope_gap_frames
        self._set_kv_cache_indices(
            new_kv_cache,
            global_end_tokens=replay_start_frame * int(self.num_token_per_frame),
            local_end_tokens=preserved_sink_tokens,
        )

        cache_state = cache_ctx.cache_state
        cache_state.kv_cache = new_kv_cache
        replay_chunks = cache_state.runtime_cache.get(
            LINGBOT_KV_RESET_REPLAY_CACHE,
            {},
        )
        keep_prev_chunks = self._kv_cache_reset_keep_prev_chunks(server_args)
        replay_start_chunk = max(0, int(current_chunk_idx) - keep_prev_chunks)
        replay_chunk_ids = [
            chunk_idx
            for chunk_idx in range(replay_start_chunk, int(current_chunk_idx) + 1)
            if chunk_idx in replay_chunks
        ]
        sample_tokens = cache_state.runtime_cache.get(
            LINGBOT_KV_RESET_SAMPLE_TOKENS_CACHE
        )

        replay_frame = replay_start_frame
        for replay_index, replay_chunk_id in enumerate(replay_chunk_ids):
            replay_entry = replay_chunks[replay_chunk_id]
            replay_latents = replay_entry["latents"]
            replay_condition = replay_entry["condition"]
            if replay_index == 0:
                replay_condition = self._anchor_kv_replay_first_frame(
                    server_args,
                    latents=replay_latents,
                    condition=replay_condition,
                )
            self._write_kv_replay_context_chunk(
                batch,
                server_args,
                latents=replay_latents,
                condition=replay_condition,
                prompt_embeds=prompt_embeds,
                image_kwargs=image_kwargs,
                pos_cond_kwargs=replay_entry["pos_cond_kwargs"],
                kv_cache=new_kv_cache,
                crossattn_cache=cache_ctx.crossattn_cache,
                current_start_frame=replay_frame,
                sample_tokens=sample_tokens,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
            )
            replay_frame += int(replay_latents.shape[2])

        cache_state.current_chunk_start_frame = replay_frame
        self._clear_lingbot_dynamic_condition_cache(cache_state)
        logger.info(
            "LingBot KV/RoPE reset: session_id=%s block_idx=%s "
            "replay_chunks=%s local_start_before=%s local_start_after=%s "
            "max_window=%s keep_prev_chunks=%s sink_frames=%s "
            "rope_gap_frames=%s valid_local_start=%s",
            getattr(batch, "realtime_session_id", None)
            or getattr(batch, "extra", {}).get("realtime_session_id"),
            getattr(batch, "block_idx", None),
            replay_chunk_ids,
            current_start_frame,
            replay_frame,
            self._kv_cache_reset_max_window_latent_frames(server_args),
            keep_prev_chunks,
            preserved_sink_frames,
            rope_gap_frames,
            replay_start_frame if rope_gap_frames > 0 else 0,
        )

    @staticmethod
    def _optional_non_negative_int(value: Any) -> int | None:
        if value is None:
            return None
        return max(0, int(value))

    def _moving_kv_sample_num_frames(
        self,
        server_args: ServerArgs,
    ) -> int | None:
        return self._optional_non_negative_int(
            getattr(
                server_args.pipeline_config,
                "interactive_kv_moving_window",
                None,
            )
        )

    def _still_kv_sample_num_frames(
        self,
        server_args: ServerArgs,
    ) -> int | None:
        return self._optional_non_negative_int(
            getattr(
                server_args.pipeline_config,
                "interactive_kv_still_window",
                3,
            )
        )

    def _get_interactive_kv_sample_num_frames(
        self,
        cache_state,
        batch: Req,
        server_args: ServerArgs,
    ) -> int | None:
        pipeline_config = server_args.pipeline_config
        if not self._interactive_kv_window_enabled(server_args):
            return None
        if not self._uses_interactive_kv_window(batch, server_args):
            return self._base_kv_sample_num_frames()

        dynamic_state = cache_state.runtime_cache.setdefault(
            LINGBOT_INTERACTIVE_KV_WINDOW_CACHE,
            {
                "consecutive_still_chunks": 0,
                "sample_num_frames": None,
            },
        )
        if cache_state.chunk_idx == 0:
            dynamic_state["consecutive_still_chunks"] = 0
            dynamic_state["sample_num_frames"] = None

        moving_window = self._moving_kv_sample_num_frames(server_args)
        if moving_window is None:
            return None
        still_window = self._still_kv_sample_num_frames(server_args)
        still_chunks_threshold = max(
            1, int(getattr(pipeline_config, "interactive_kv_still_chunks", 2))
        )
        if dynamic_state["sample_num_frames"] is None:
            dynamic_state["sample_num_frames"] = moving_window

        condition_inputs = getattr(batch, "condition_inputs", None) or {}
        if self._chunk_has_camera_motion(
            condition_inputs.get(LINGBOT_CAMERA_ACTIONS_CONDITION)
        ):
            dynamic_state["consecutive_still_chunks"] = 0
            dynamic_state["sample_num_frames"] = moving_window
        else:
            dynamic_state["consecutive_still_chunks"] += 1
            if (
                still_window is not None
                and dynamic_state["consecutive_still_chunks"] >= still_chunks_threshold
            ):
                dynamic_state["sample_num_frames"] = still_window

        return int(dynamic_state["sample_num_frames"])

    def _log_lingbot_kv_window(
        self,
        cache_state,
        batch: Req,
        server_args: ServerArgs,
        *,
        sample_frames: int | None,
    ) -> None:
        if not self._interactive_kv_window_enabled(server_args):
            return

        mode = "base"
        still_chunks = None
        if self._uses_interactive_kv_window(batch, server_args):
            dynamic_state = cache_state.runtime_cache.get(
                LINGBOT_INTERACTIVE_KV_WINDOW_CACHE, {}
            )
            still_chunks = dynamic_state.get("consecutive_still_chunks")
            condition_inputs = getattr(batch, "condition_inputs", None) or {}
            if self._chunk_has_camera_motion(
                condition_inputs.get(LINGBOT_CAMERA_ACTIONS_CONDITION)
            ):
                mode = "moving"
            else:
                still_window = self._still_kv_sample_num_frames(server_args)
                still_chunks_threshold = max(
                    1,
                    int(
                        getattr(
                            server_args.pipeline_config,
                            "interactive_kv_still_chunks",
                            2,
                        )
                    ),
                )
                if (
                    still_window is not None
                    and sample_frames == still_window
                    and still_chunks is not None
                    and still_chunks >= still_chunks_threshold
                ):
                    mode = "still"
                else:
                    mode = "moving"

        window_frames = (
            int(self.sliding_window_num_frames)
            if sample_frames is None
            else int(self.sink_size)
            + int(sample_frames)
            + int(self.num_frames_per_block)
        )
        sample_tokens = (
            None
            if sample_frames is None
            else int(sample_frames) * int(self.num_token_per_frame)
        )
        logger.info(
            "LingBot interactive KV window: session_id=%s request_id=%s "
            "chunk_idx=%s mode=%s window_frames=%s sample_frames=%s "
            "cache_frames=%s sink_frames=%s current_frames=%s sample_tokens=%s "
            "cache_tokens=%s still_chunks=%s",
            getattr(batch, "realtime_session_id", None),
            getattr(batch, "request_id", None),
            getattr(batch, "block_idx", None),
            mode,
            window_frames,
            sample_frames,
            int(self.sliding_window_num_frames),
            int(self.sink_size),
            int(self.num_frames_per_block),
            sample_tokens,
            int(self.sliding_window_num_frames) * int(self.num_token_per_frame),
            still_chunks,
        )

    def _set_lingbot_kv_sample_tokens(
        self,
        cache_state,
        batch: Req,
        server_args: ServerArgs,
    ) -> int | None:
        self._sync_interactive_kv_cache_window(server_args)
        sample_frames = self._get_interactive_kv_sample_num_frames(
            cache_state,
            batch,
            server_args,
        )
        sample_tokens = (
            None
            if sample_frames is None
            else int(sample_frames) * self.num_token_per_frame
        )
        cache_state.runtime_cache[LINGBOT_KV_RESET_SAMPLE_TOKENS_CACHE] = sample_tokens
        self._log_lingbot_kv_window(
            cache_state,
            batch,
            server_args,
            sample_frames=sample_frames,
        )
        previous = getattr(batch, "realtime_causal_kv_sample_tokens", None)
        batch.realtime_causal_kv_sample_tokens = sample_tokens
        return previous

    @staticmethod
    def _clear_lingbot_dynamic_condition_cache(cache_state) -> None:
        runtime_cache = getattr(cache_state, "runtime_cache", None)
        if runtime_cache is None:
            return
        runtime_cache.pop(LINGBOT_C2WS_PLUCKER_EMB_CACHE, None)
        runtime_cache.pop(LINGBOT_CAM_CONDITIONER_CACHE, None)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)]
        )
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.with_dims(1)])
        result.add_check("scheduler", batch.scheduler, V.not_none)
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        return result

    def _denoise_realtime_causal_chunk(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        ctx,
        cache_ctx,
        chunk_latents: torch.Tensor,
        prepare_model_input,
        prepare_context_input,
    ) -> torch.Tensor:
        previous_sample_tokens = self._set_lingbot_kv_sample_tokens(
            cache_ctx.cache_state,
            batch,
            server_args,
        )
        try:
            return super()._denoise_realtime_causal_chunk(
                batch,
                server_args,
                ctx=ctx,
                cache_ctx=cache_ctx,
                chunk_latents=chunk_latents,
                prepare_model_input=prepare_model_input,
                prepare_context_input=prepare_context_input,
            )
        finally:
            batch.realtime_causal_kv_sample_tokens = previous_sample_tokens

    def _get_causal_dmd_latents(self, batch: Req) -> torch.Tensor:
        latents = batch.latents
        assert latents is not None, (
            "LingBot-World causal DMD requires prepared chunk latents. "
            "Ensure RealtimeChunkLatentPreparationStage runs before this stage."
        )
        return latents

    def _get_causal_dmd_scheduler(self, batch: Req, server_args: ServerArgs):
        scheduler = batch.scheduler
        assert scheduler is not None, (
            "LingBot-World causal DMD requires prepared DMD timesteps. "
            "Ensure DMDTimestepPreparationStage runs before this stage."
        )
        return scheduler

    def _prepare_causal_dmd_timesteps(
        self,
        batch: Req,
        server_args: ServerArgs,
        scheduler,
        device: torch.device,
    ) -> torch.Tensor:
        timesteps = batch.timesteps
        assert timesteps is not None
        return timesteps.to(device)

    def _prepare_causal_dmd_image_kwargs(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ) -> dict:
        image_embeds = getattr(batch, "image_embeds", [])
        if len(image_embeds) > 0:
            image_embeds = [ie.to(target_dtype) for ie in image_embeds]
        return {
            "encoder_hidden_states_image": image_embeds,
        }

    def _prepare_causal_dmd_pos_cond_kwargs(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ) -> dict:
        # lingbot transformer forward uses varargs, so inspect filtering drops valid kwargs
        return server_args.pipeline_config.prepare_pos_cond_kwargs(
            batch,
            self.device,
            getattr(self.transformer, "rotary_emb", None),
            dtype=target_dtype,
        )

    def _prepare_causal_dmd_prompt_embeds(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ):
        return server_args.pipeline_config.get_pos_prompt_embeds(batch)

    def _update_causal_context_cache(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        context_input: torch.Tensor,
        prompt_embeds,
        kv_cache,
        crossattn_cache,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        attn_metadata,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
    ) -> None:
        context_noise = getattr(server_args.pipeline_config, "context_noise", 0)
        timestep = torch.full(
            (context_input.shape[0], 1),
            int(context_noise),
            device=context_input.device,
            dtype=torch.long,
        )
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
            self.transformer(
                context_input.to(target_dtype),
                prompt_embeds,
                timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_tokens,
                start_frame=start_frame,
                skip_final_projection=True,
                **image_kwargs,
                **pos_cond_kwargs,
            )

    @staticmethod
    def _select_i2v_condition_chunk(
        condition_full: torch.Tensor,
        chunk_idx: int,
        chunk_size: int,
    ) -> torch.Tensor:
        condition_chunks = condition_full.split(chunk_size, dim=2)
        condition = condition_chunks[min(chunk_idx, len(condition_chunks) - 1)]

        if condition.shape[2] == chunk_size:
            return condition
        pad_frames = chunk_size - condition.shape[2]
        return torch.cat(
            [
                condition,
                condition.new_zeros(
                    condition.shape[0],
                    condition.shape[1],
                    pad_frames,
                    condition.shape[3],
                    condition.shape[4],
                ),
            ],
            dim=2,
        )

    @staticmethod
    def _build_i2v_model_input_writer(
        *,
        latents: torch.Tensor,
        condition: torch.Tensor,
        target_dtype: torch.dtype,
        device: torch.device,
    ):
        b, latent_channels, t, h, w = latents.shape
        condition = condition.to(device=device, dtype=target_dtype)
        model_input = torch.empty(
            (
                b,
                latent_channels + condition.shape[1],
                t,
                h,
                w,
            ),
            dtype=target_dtype,
            device=device,
        )
        model_input[:, latent_channels:].copy_(condition)

        def write(current_latents: torch.Tensor) -> torch.Tensor:
            model_input[:, :latent_channels].copy_(current_latents)
            return model_input

        return write

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # --- Condition: take current chunk's slice ---
        condition_full = batch.image_latent
        assert condition_full is not None, (
            "LingBot-World causal DMD requires image_latent as condition. "
            "Ensure ImageVAEEncodingStage runs before this stage."
        )
        ctx = self._prepare_causal_dmd_forward_context(batch, server_args)
        latents = ctx.latents
        cache_ctx = self._prepare_realtime_causal_caches(batch, server_args, ctx)

        # Keep cross-attention K/V cache across realtime chunks; LingBot text/image
        # conditions are session-static and are invalidated by cache reset.

        # Slice condition to current chunk
        condition = self._select_i2v_condition_chunk(
            condition_full,
            cache_ctx.chunk_idx,
            ctx.num_frames,
        )

        # --- Denoising loop (single chunk) ---
        current_latents = latents
        prepare_model_input = self._build_i2v_model_input_writer(
            latents=current_latents,
            condition=condition,
            target_dtype=ctx.target_dtype,
            device=ctx.device,
        )

        try:
            current_latents = self._denoise_realtime_causal_chunk(
                batch,
                server_args,
                ctx=ctx,
                cache_ctx=cache_ctx,
                chunk_latents=current_latents,
                prepare_model_input=prepare_model_input,
                prepare_context_input=prepare_model_input,
            )
        finally:
            self._clear_lingbot_dynamic_condition_cache(cache_ctx.cache_state)

        current_chunk_idx = cache_ctx.chunk_idx
        self._remember_kv_replay_chunk(
            cache_ctx.cache_state,
            server_args,
            chunk_idx=current_chunk_idx,
            latents=current_latents,
            condition=condition,
            pos_cond_kwargs=ctx.pos_cond_kwargs,
        )

        # Advance cumulative frame position, then compact/replay before the next
        # request when the configured RoPE window would be exceeded.
        self._advance_realtime_causal_cache(cache_ctx, num_frames=ctx.num_frames)
        self._maybe_reset_kv_rope_for_next_chunk(
            cache_ctx,
            batch,
            server_args,
            old_kv_cache=cache_ctx.kv_cache,
            current_chunk_idx=current_chunk_idx,
            current_start_frame=cache_ctx.current_start_frame,
            current_chunk_frames=ctx.num_frames,
            prompt_embeds=ctx.prompt_embeds,
            image_kwargs=ctx.image_kwargs,
            target_dtype=ctx.target_dtype,
            autocast_enabled=ctx.autocast_enabled,
        )

        # Output denoised latents for decoder
        batch.latents = current_latents
        batch.raw_latent_shape = current_latents.shape
        if not cache_ctx.persist_state:
            cache_ctx.cache_state.dispose()
        return batch
