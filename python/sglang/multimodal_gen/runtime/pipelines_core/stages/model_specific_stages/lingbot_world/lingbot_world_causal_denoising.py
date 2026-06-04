# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/Robbyant/lingbot-world

"""LingBot-World causal DMD denoising stage."""

import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_group,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.utils import pred_noise_to_pred_video
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDCachePolicy,
    CausalDMDDenoisingStage,
    CausalDMDForwardContext,
    CausalDMDRealtimeCacheContext,
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


class _LingBotSteadyDenoiseCudaGraphState:
    def __init__(self) -> None:
        self.signature = None
        self.warmup_chunks = 0
        self.disabled_reason: str | None = None
        self.graphs: list[torch.cuda.CUDAGraph | None] = []
        self.outputs: list[torch.Tensor | None] = []
        self.static_model_input: torch.Tensor | None = None
        self.static_timestep: torch.Tensor | None = None
        self.static_c2ws_plucker_emb: torch.Tensor | None = None
        self.static_freqs_cis: tuple[torch.Tensor, ...] | None = None
        self.graph_pool = None
        self.ready_logged = False
        self.replay_logged = False

    def _clear_graphs(self) -> None:
        self.signature = None
        self.graphs = []
        self.outputs = []
        self.static_model_input = None
        self.static_timestep = None
        self.static_c2ws_plucker_emb = None
        self.static_freqs_cis = None
        self.graph_pool = None
        self.ready_logged = False
        self.replay_logged = False

    def dispose(self) -> None:
        self._clear_graphs()
        self.warmup_chunks = 0
        self.disabled_reason = None

    @property
    def captured(self) -> bool:
        return bool(self.graphs) and all(graph is not None for graph in self.graphs)

    def reset_for_signature(self, signature) -> None:
        if self.signature == signature:
            return
        self._clear_graphs()
        self.signature = signature

    def ensure_buffers(
        self,
        *,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        c2ws_plucker_emb: torch.Tensor | None,
        freqs_cis: tuple[torch.Tensor, ...],
        num_steps: int,
    ) -> None:
        if self.static_model_input is None:
            self.static_model_input = torch.empty_like(latent_model_input)
        if self.static_timestep is None:
            self.static_timestep = torch.empty_like(timestep)
        if c2ws_plucker_emb is None:
            self.static_c2ws_plucker_emb = None
        elif self.static_c2ws_plucker_emb is None:
            self.static_c2ws_plucker_emb = torch.empty_like(c2ws_plucker_emb)
        if self.static_freqs_cis is None:
            self.static_freqs_cis = tuple(torch.empty_like(freq) for freq in freqs_cis)
        if not self.graphs:
            self.graphs = [None] * num_steps
            self.outputs = [None] * num_steps
        elif len(self.graphs) != num_steps:
            raise RuntimeError(
                f"CUDA graph step count changed: {len(self.graphs)} -> {num_steps}"
            )

    def copy_inputs(
        self,
        *,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        c2ws_plucker_emb: torch.Tensor | None,
        freqs_cis: tuple[torch.Tensor, ...],
    ) -> None:
        assert self.static_model_input is not None
        assert self.static_timestep is not None
        assert self.static_freqs_cis is not None
        self.static_model_input.copy_(latent_model_input)
        self.static_timestep.copy_(timestep)
        if c2ws_plucker_emb is not None:
            assert self.static_c2ws_plucker_emb is not None
            self.static_c2ws_plucker_emb.copy_(c2ws_plucker_emb)
        for static_freq, freq in zip(self.static_freqs_cis, freqs_cis, strict=True):
            static_freq.copy_(freq)


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
    ) -> dict[str, bool]:
        return {"sequence_shard_enabled": policy.sequence_shard_enabled}

    def _use_causal_cache_int_indices(
        self,
        *,
        sequence_shard_enabled: bool,
    ) -> bool:
        return True

    @staticmethod
    def _cuda_graph_mode(server_args: ServerArgs) -> str:
        return str(
            getattr(server_args.pipeline_config, "realtime_cuda_graph_mode", "off")
            or "off"
        ).lower()

    @staticmethod
    def _cuda_graph_fallback_on_error(server_args: ServerArgs) -> bool:
        return bool(
            getattr(
                server_args.pipeline_config,
                "realtime_cuda_graph_fallback_on_error",
                True,
            )
        )

    def _cuda_graph_warmup_chunks(self, server_args: ServerArgs) -> int:
        return max(
            0,
            int(
                getattr(
                    server_args.pipeline_config,
                    "realtime_cuda_graph_warmup_chunks",
                    2,
                )
            ),
        )

    @staticmethod
    def _all_crossattn_caches_initialized(crossattn_cache) -> bool:
        return crossattn_cache is not None and all(
            cache.is_init for cache in crossattn_cache
        )

    @staticmethod
    def _get_or_create_cuda_graph_state(
        cache_ctx: CausalDMDRealtimeCacheContext,
    ) -> _LingBotSteadyDenoiseCudaGraphState:
        state = getattr(cache_ctx.cache_state, "cuda_graph_state", None)
        if state is None:
            state = _LingBotSteadyDenoiseCudaGraphState()
            cache_ctx.cache_state.cuda_graph_state = state
        return state

    def _steady_cuda_graph_eligible(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        ctx: CausalDMDForwardContext,
        cache_ctx: CausalDMDRealtimeCacheContext,
        timesteps: torch.Tensor,
    ) -> tuple[bool, str]:
        mode = self._cuda_graph_mode(server_args)
        if mode not in ("on", "steady", "steady_denoise"):
            return False, "disabled"
        if not current_platform.is_cuda() or not torch.cuda.is_available():
            return False, "cuda unavailable"
        if batch.session is None or not cache_ctx.persist_state:
            return False, "missing realtime session"
        if getattr(server_args, "enable_torch_compile", False):
            return False, "torch.compile enabled"
        if self.attn_backend.get_enum() == AttentionBackendEnum.VIDEO_SPARSE_ATTN:
            return False, "video sparse attention metadata is dynamic"
        if self.local_attn_size != -1:
            return False, "local attention cache is not supported"
        if ctx.num_frames != self.num_frames_per_block:
            return False, "non-standard chunk size"
        if int(timesteps.numel()) == 0:
            return False, "empty timesteps"
        if not self._all_crossattn_caches_initialized(cache_ctx.crossattn_cache):
            return False, "cross-attention cache is not initialized"

        expected_cache_tokens = self._get_causal_kv_cache_size(
            sequence_shard_enabled=self._causal_sequence_shard_enabled(batch)
        )
        first_cache = cache_ctx.kv_cache[0]
        if first_cache.cache_size != expected_cache_tokens:
            return False, "unexpected KV cache size"
        if first_cache.local_end_index_int is not None:
            local_end_index = first_cache.local_end_index_int
        else:
            local_end_index = int(first_cache.local_end_index.item())
        if local_end_index != first_cache.cache_size:
            return False, "KV cache is not full"
        if cache_ctx.current_start_frame < self.sliding_window_num_frames:
            return False, "not in steady-state"
        return True, "eligible"

    def _cuda_graph_signature(
        self,
        *,
        batch: Req,
        ctx: CausalDMDForwardContext,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        c2ws_plucker_emb: torch.Tensor | None,
        freqs_cis: tuple[torch.Tensor, ...],
        num_steps: int,
    ) -> tuple:
        return (
            tuple(latent_model_input.shape),
            latent_model_input.dtype,
            latent_model_input.device.type,
            latent_model_input.device.index,
            tuple(timestep.shape),
            timestep.dtype,
            tuple(c2ws_plucker_emb.shape) if c2ws_plucker_emb is not None else None,
            c2ws_plucker_emb.dtype if c2ws_plucker_emb is not None else None,
            tuple((tuple(freq.shape), freq.dtype) for freq in freqs_cis),
            num_steps,
            bool(getattr(batch, "enable_sequence_shard", False)),
            ctx.num_frames,
            ctx.height,
            ctx.width,
            self.sink_size,
            self.sliding_window_num_frames,
        )

    def _prepare_cuda_graph_freqs_cis(
        self,
        *,
        batch: Req,
        ctx: CausalDMDForwardContext,
        start_frame: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        p_t, p_h, p_w = self.transformer.patch_size
        return self.transformer.prepare_freqs_cis(
            forward_batch=batch,
            post_patch_num_frames=ctx.num_frames // p_t,
            post_patch_height=ctx.height // p_h,
            post_patch_width=ctx.width // p_w,
            start_frame=start_frame,
            device=device,
            sequence_shard_enabled=self._causal_sequence_shard_enabled(batch),
        )

    @staticmethod
    def _refresh_steady_kv_indices(
        kv_cache,
        *,
        global_end_index: int,
    ) -> None:
        for cache_block in kv_cache:
            cache_block._write_indices(
                global_end_index=global_end_index,
                local_end_index=cache_block.cache_size,
            )

    def _run_cuda_graph_transformer_step(
        self,
        batch: Req,
        *,
        graph_state: _LingBotSteadyDenoiseCudaGraphState,
        step_idx: int,
        graph_kv_mode: str,
        latent_model_input: torch.Tensor,
        prompt_embeds,
        timestep: torch.Tensor,
        kv_cache,
        crossattn_cache,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        freqs_cis: tuple[torch.Tensor, ...],
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        skip_final_projection: bool = False,
    ) -> torch.Tensor:
        c2ws_plucker_emb = pos_cond_kwargs.get("c2ws_plucker_emb")
        graph_state.copy_inputs(
            latent_model_input=latent_model_input,
            timestep=timestep,
            c2ws_plucker_emb=c2ws_plucker_emb,
            freqs_cis=freqs_cis,
        )

        if graph_state.graphs[step_idx] is not None:
            graph_state.graphs[step_idx].replay()
            output = graph_state.outputs[step_idx]
            assert output is not None
            return output

        if graph_state.graph_pool is None:
            graph_state.graph_pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        graph_pos_cond_kwargs = {
            key: value
            for key, value in pos_cond_kwargs.items()
            if key != "c2ws_plucker_emb"
        }
        if graph_state.static_c2ws_plucker_emb is not None:
            graph_pos_cond_kwargs["c2ws_plucker_emb"] = (
                graph_state.static_c2ws_plucker_emb
            )
        assert graph_state.static_model_input is not None
        assert graph_state.static_timestep is not None
        assert graph_state.static_freqs_cis is not None

        torch.cuda.synchronize()
        if get_ulysses_parallel_world_size() > 1:
            get_sp_group().barrier()
        with torch.cuda.graph(graph, pool=graph_state.graph_pool):
            with (
                torch.autocast(
                    device_type=current_platform.device_type,
                    dtype=target_dtype,
                    enabled=autocast_enabled,
                ),
                set_forward_context(
                    current_timestep=step_idx,
                    attn_metadata=None,
                    forward_batch=batch,
                ),
            ):
                graph_state.outputs[step_idx] = self.transformer(
                    graph_state.static_model_input,
                    prompt_embeds,
                    graph_state.static_timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_tokens,
                    start_frame=start_frame,
                    skip_final_projection=skip_final_projection,
                    freqs_cis=graph_state.static_freqs_cis,
                    graph_kv_mode=graph_kv_mode,
                    disable_c2ws_plucker_cache=True,
                    **image_kwargs,
                    **graph_pos_cond_kwargs,
                )
        graph_state.graphs[step_idx] = graph
        output = graph_state.outputs[step_idx]
        assert output is not None
        return output

    def _update_causal_context_cache_steady(
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
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        freqs_cis: tuple[torch.Tensor, ...] | None = None,
        graph_state: _LingBotSteadyDenoiseCudaGraphState | None = None,
        graph_idx: int | None = None,
    ) -> None:
        context_noise = getattr(server_args.pipeline_config, "context_noise", 0)
        timestep = torch.full(
            (context_input.shape[0], 1),
            int(context_noise),
            device=context_input.device,
            dtype=torch.long,
        )
        if graph_state is not None:
            assert graph_idx is not None
            assert freqs_cis is not None
            self._run_cuda_graph_transformer_step(
                batch,
                graph_state=graph_state,
                step_idx=graph_idx,
                graph_kv_mode="steady_overwrite",
                latent_model_input=context_input.to(target_dtype),
                prompt_embeds=prompt_embeds,
                timestep=timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_tokens=current_start_tokens,
                start_frame=start_frame,
                image_kwargs=image_kwargs,
                pos_cond_kwargs=pos_cond_kwargs,
                freqs_cis=freqs_cis,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                skip_final_projection=True,
            )
            return

        with (
            torch.autocast(
                device_type=current_platform.device_type,
                dtype=target_dtype,
                enabled=autocast_enabled,
            ),
            set_forward_context(
                current_timestep=0,
                attn_metadata=None,
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
                freqs_cis=freqs_cis,
                graph_kv_mode="steady_overwrite",
                disable_c2ws_plucker_cache=True,
                **image_kwargs,
                **pos_cond_kwargs,
            )

    def _denoise_and_update_causal_block_with_cuda_graph(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        ctx: CausalDMDForwardContext,
        cache_ctx: CausalDMDRealtimeCacheContext,
        chunk_latents: torch.Tensor,
        prepare_model_input,
        prepare_context_input,
        progress_bar=None,
    ) -> torch.Tensor | None:
        eligible, reason = self._steady_cuda_graph_eligible(
            batch,
            server_args,
            ctx=ctx,
            cache_ctx=cache_ctx,
            timesteps=ctx.timesteps,
        )
        graph_state = self._get_or_create_cuda_graph_state(cache_ctx)
        if not eligible:
            return None
        if graph_state.disabled_reason is not None:
            return None
        if graph_state.warmup_chunks < self._cuda_graph_warmup_chunks(server_args):
            graph_state.warmup_chunks += 1
            return None

        num_denoise_steps = int(ctx.timesteps.numel())
        num_graphs = num_denoise_steps + 1
        current_start_tokens = cache_ctx.current_start_frame * self.num_token_per_frame
        current_latents = chunk_latents
        noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
        raw_latent_shape = noise_latents_btchw.shape
        freqs_cis = self._prepare_cuda_graph_freqs_cis(
            batch=batch,
            ctx=ctx,
            start_frame=cache_ctx.current_start_frame,
            device=ctx.device,
        )
        was_captured = graph_state.captured

        for i, timestep in enumerate(ctx.timesteps):
            noise_latents = noise_latents_btchw
            latent_model_input = prepare_model_input(current_latents).to(
                ctx.target_dtype
            )
            timestep_2d = self._expand_timestep(
                timestep, latent_model_input.shape[0], latent_model_input.device
            )
            timestep_input = timestep_2d.unsqueeze(1)
            if i == 0:
                signature = self._cuda_graph_signature(
                    batch=batch,
                    ctx=ctx,
                    latent_model_input=latent_model_input,
                    timestep=timestep_input,
                    c2ws_plucker_emb=ctx.pos_cond_kwargs.get("c2ws_plucker_emb"),
                    freqs_cis=freqs_cis,
                    num_steps=num_graphs,
                )
                graph_state.reset_for_signature(signature)
                graph_state.ensure_buffers(
                    latent_model_input=latent_model_input,
                    timestep=timestep_input,
                    c2ws_plucker_emb=ctx.pos_cond_kwargs.get("c2ws_plucker_emb"),
                    freqs_cis=freqs_cis,
                    num_steps=num_graphs,
                )
                was_captured = graph_state.captured

            pred_noise = self._run_cuda_graph_transformer_step(
                batch,
                graph_state=graph_state,
                step_idx=i,
                graph_kv_mode="steady_append" if i == 0 else "steady_overwrite",
                latent_model_input=latent_model_input,
                prompt_embeds=ctx.prompt_embeds,
                timestep=timestep_input,
                kv_cache=cache_ctx.kv_cache,
                crossattn_cache=cache_ctx.crossattn_cache,
                current_start_tokens=current_start_tokens,
                start_frame=cache_ctx.current_start_frame,
                image_kwargs=ctx.image_kwargs,
                pos_cond_kwargs=ctx.pos_cond_kwargs,
                freqs_cis=freqs_cis,
                target_dtype=ctx.target_dtype,
                autocast_enabled=ctx.autocast_enabled,
            )
            pred_noise_btchw = pred_noise.permute(0, 2, 1, 3, 4)
            x0_btchw = pred_noise_to_pred_video(
                pred_noise=pred_noise_btchw.flatten(0, 1),
                noise_input_latent=noise_latents.flatten(0, 1),
                timestep=timestep_2d,
                scheduler=ctx.scheduler,
            ).unflatten(0, pred_noise_btchw.shape[:2])

            if i < int(ctx.timesteps.numel()) - 1:
                next_timestep = ctx.timesteps[i + 1 : i + 2]
                noise_latents_btchw = self._add_noise_for_next_timestep(
                    batch,
                    x0_btchw=x0_btchw,
                    raw_latent_shape=raw_latent_shape,
                    next_timestep=next_timestep,
                    scheduler=ctx.scheduler,
                    device=ctx.device,
                )
                current_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)
            else:
                current_latents = x0_btchw.permute(0, 2, 1, 3, 4)

            if progress_bar is not None:
                progress_bar.update()

        self._update_causal_context_cache_steady(
            batch,
            server_args,
            context_input=prepare_context_input(current_latents),
            prompt_embeds=ctx.prompt_embeds,
            kv_cache=cache_ctx.kv_cache,
            crossattn_cache=cache_ctx.crossattn_cache,
            current_start_tokens=current_start_tokens,
            start_frame=cache_ctx.current_start_frame,
            image_kwargs=ctx.image_kwargs,
            pos_cond_kwargs=ctx.pos_cond_kwargs,
            target_dtype=ctx.target_dtype,
            autocast_enabled=ctx.autocast_enabled,
            freqs_cis=freqs_cis,
            graph_state=graph_state,
            graph_idx=num_denoise_steps,
        )
        self._refresh_steady_kv_indices(
            cache_ctx.kv_cache,
            global_end_index=current_start_tokens
            + ctx.num_frames * self.num_token_per_frame,
        )
        if graph_state.captured and not graph_state.ready_logged:
            logger.info(
                "LingBot realtime CUDA graph captured: denoise_steps=%s graphs=%s "
                "sequence_shard=%s",
                num_denoise_steps,
                num_graphs,
                self._causal_sequence_shard_enabled(batch),
            )
            graph_state.ready_logged = True
        elif was_captured and not graph_state.replay_logged:
            logger.info(
                "LingBot realtime CUDA graph replay active: graphs=%s",
                num_graphs,
            )
            graph_state.replay_logged = True
        return current_latents

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
                current_timestep=0,
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

        graph_latents = None
        if self._cuda_graph_mode(server_args) not in ("off", "false", "0"):
            try:
                graph_latents = self._denoise_and_update_causal_block_with_cuda_graph(
                    batch,
                    server_args,
                    ctx=ctx,
                    cache_ctx=cache_ctx,
                    chunk_latents=current_latents,
                    prepare_model_input=prepare_model_input,
                    prepare_context_input=prepare_model_input,
                )
            except Exception as e:
                graph_state = self._get_or_create_cuda_graph_state(cache_ctx)
                graph_state.disabled_reason = str(e)
                if not self._cuda_graph_fallback_on_error(server_args):
                    raise
                logger.exception(
                    "Disabling LingBot realtime CUDA graph for this session: %s",
                    e,
                )

        if graph_latents is None:
            current_latents = self._denoise_realtime_causal_chunk(
                batch,
                server_args,
                ctx=ctx,
                cache_ctx=cache_ctx,
                chunk_latents=current_latents,
                prepare_model_input=prepare_model_input,
                prepare_context_input=prepare_model_input,
            )
        else:
            current_latents = graph_latents

        # Advance cumulative frame position
        self._advance_realtime_causal_cache(cache_ctx, num_frames=ctx.num_frames)

        # Output denoised latents for decoder
        batch.latents = current_latents
        batch.raw_latent_shape = current_latents.shape
        if not cache_ctx.persist_state:
            cache_ctx.cache_state.dispose()
        return batch
