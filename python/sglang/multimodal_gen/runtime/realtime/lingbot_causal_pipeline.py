# SPDX-License-Identifier: Apache-2.0
"""Experimental LingBot adapter for timestep and transformer-block pipelines.

This executes a bounded window of one session. Preparation and delivery are
windowed; dedicated VAE decoding overlaps denoising inside the window. Shared
VAE decoding runs after denoising, on the selected rank. This is not a dynamic
multi-stream scheduler or a quality-equivalent replacement for clean history.
"""

from contextlib import ExitStack
from dataclasses import dataclass, field

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.cache.conditioning import conditioning_weights_epoch
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_group
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
    CrossAttentionKVCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    pred_noise_to_pred_video,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.latent_preparation import (
    RealtimeChunkLatentPreparationStage,
)
from sglang.multimodal_gen.runtime.realtime.causal_pipeline import (
    CausalPipelinePlan,
    TorchCausalPipelineTransport,
    causal_pipeline_seed,
    run_causal_pipeline,
)
from sglang.multimodal_gen.runtime.realtime.causal_pipeline_config import (
    placement_from_args,
)
from sglang.multimodal_gen.runtime.realtime.lingbot_world import (
    LINGBOT_PROMPT_UPDATED_CONDITION,
)
from sglang.multimodal_gen.runtime.realtime.session import (
    BaseRealtimeState,
    RealtimeSession,
)
from sglang.multimodal_gen.runtime.realtime.states import RealtimeCausalDiTState
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.runtime.utils.profiler import maybe_record_function

logger = init_logger(__name__)


@dataclass
class LingBotPipelineState(BaseRealtimeState):
    next_chunk: int = 0
    next_frame: int = 0
    signature: tuple | None = None
    # Each rank stores only the (step, layer) histories it executes.
    histories: dict = field(default_factory=dict)

    def dispose(self):
        self.histories.clear()
        self.signature = None
        self.next_chunk = self.next_frame = 0


def validate_chunk_window(batches, limit):
    if not 1 <= len(batches) <= limit:
        raise ValueError(f"Causal pipeline expects 1..{limit} chunks")
    first = batches[0]
    for offset, batch in enumerate(batches):
        if (
            batch.realtime_session_id != first.realtime_session_id
            or batch.block_idx != first.block_idx + offset
        ):
            raise ValueError(
                "Causal pipeline requires consecutive chunks of one session"
            )
        if batch.num_outputs_per_prompt != 1 or batch.do_classifier_free_guidance:
            raise ValueError("Causal pipeline requires one output and no CFG")
        if batch.rollout:
            raise ValueError("Causal pipeline does not support rollout")
        if not isinstance(batch.seed, int):
            raise ValueError("Causal pipeline requires one integer seed")


class LingBotCausalPipelineRunner:
    def __init__(self, pipeline, server_args):
        self.pipeline = pipeline
        self.args = server_args
        self.placement = placement_from_args(server_args)
        self.group = get_world_group()
        self.rank = self.group.rank_in_group
        self.transport = TorchCausalPipelineTransport(self.group)
        self.is_denoiser = self.rank < self.placement.denoising_ranks
        self.decode = pipeline.stages[-1]
        self.denoise = pipeline.stages[-2] if self.is_denoiser else None

    def _local_stage(self, stage, index, batch):
        return self.pipeline.executor._run_stage_with_executor_hooks(
            stage,
            index,
            batch,
            self.args,
            lambda current, req: current(req, self.args),
            self.pipeline.executor._should_use_stage_nvtx(batch, self.args),
        )

    def _prepare(self, batches):
        """All ranks agree on preparation failure before neighbor transfers."""
        error = None
        contexts = []
        state = None
        plan = None
        try:
            validate_chunk_window(batches, self.args.causal_pipeline_window)
            session = batches[0].session
            state = session.get_or_create_state(LingBotPipelineState)
            if batches[0].block_idx == 0:
                state.dispose()
            if batches[0].block_idx != state.next_chunk:
                raise ValueError(
                    f"Expected chunk {state.next_chunk}, got {batches[0].block_idx}"
                )
            if self.is_denoiser:
                for batch in batches:
                    for index, stage in enumerate(self.pipeline.stages[:-2]):
                        if (
                            isinstance(stage, RealtimeChunkLatentPreparationStage)
                            and batch.latents is None
                        ):
                            batch.generator = torch.Generator(
                                device=self.denoise._single_generator(batch).device
                            ).manual_seed(
                                causal_pipeline_seed(batch.seed, batch.block_idx, -1)
                            )
                        batch = self._local_stage(stage, index, batch)
                    contexts.append(
                        self.denoise._prepare_causal_dmd_forward_context(
                            batch, self.args
                        )
                    )
                ctx = contexts[0]
                if ctx.batch_size != 1:
                    raise ValueError(
                        "Causal pipeline currently supports a single video stream"
                    )
                plan = CausalPipelinePlan(
                    len(ctx.timesteps),
                    self.denoise.num_transformer_blocks,
                    self.placement.denoising_ranks,
                    self.args.causal_pipeline_mode,
                )
                policy = self.denoise._build_realtime_causal_cache_policy(
                    batches[0], self.args
                )
                signature = (
                    tuple(ctx.latents.shape),
                    tuple(ctx.timesteps.tolist()),
                    policy.expected_cache_tokens,
                    policy.expected_sink_tokens,
                    plan,
                    ctx.target_dtype,
                    conditioning_weights_epoch(),
                )
                if state.signature is not None and signature != state.signature:
                    raise ValueError(
                        "Restart the session before changing causal pipeline geometry, schedule or weights"
                    )
                for batch, other in zip(batches[1:], contexts[1:]):
                    other_policy = self.denoise._build_realtime_causal_cache_policy(
                        batch, self.args
                    )
                    if (
                        other.latents.shape != ctx.latents.shape
                        or not torch.equal(other.timesteps, ctx.timesteps)
                        or other_policy.expected_cache_tokens
                        != policy.expected_cache_tokens
                        or other_policy.expected_sink_tokens
                        != policy.expected_sink_tokens
                    ):
                        raise ValueError(
                            "A causal pipeline window requires fixed shapes, cache policy and timesteps"
                        )
                state.signature = signature
                if not state.histories:
                    self._allocate_histories(state, ctx, plan, policy)
        except Exception as exc:
            error = f"Causal pipeline preparation on rank {self.rank}: {exc}"
        self.transport.finish(error)
        return state, contexts, plan

    def _allocate_histories(self, state, ctx, plan, policy):
        shape = (
            ctx.batch_size,
            policy.expected_cache_tokens,
            self.denoise.transformer.num_attention_heads,
            self.denoise.transformer.attention_head_dim,
        )
        for part in plan.slices(self.rank):
            kv = [None] * plan.num_blocks
            cross = [None] * plan.num_blocks
            for layer in range(part.block_start, part.block_end):
                kv[layer] = CausalSelfAttentionKVCache(
                    k=torch.zeros(shape, dtype=ctx.target_dtype, device=ctx.device),
                    v=torch.zeros(shape, dtype=ctx.target_dtype, device=ctx.device),
                    global_end_index=torch.zeros(
                        1, dtype=torch.long, device=ctx.device
                    ),
                    local_end_index=torch.zeros(1, dtype=torch.long, device=ctx.device),
                    global_end_index_int=0,
                    local_end_index_int=0,
                    cache_size=policy.expected_cache_tokens,
                    sink_tokens=policy.expected_sink_tokens,
                    attention_window_size=policy.expected_cache_tokens,
                )
                empty = torch.empty(0, dtype=ctx.target_dtype, device=ctx.device)
                cross[layer] = CrossAttentionKVCache(k=empty, v=empty)
            state.histories[part.timestep] = (kv, cross)

    def _execute(self, batch, ctx, state, start_frame, part, payload):
        kv, cross = state.histories[part.timestep]
        if batch.condition_inputs.get(LINGBOT_PROMPT_UPDATED_CONDITION, False):
            for cache in cross:
                if cache is not None:
                    cache.reset()
        stage = self.denoise
        latents = payload["latents"]
        condition = stage._select_i2v_condition_chunk(
            batch.image_latent, batch.block_idx, ctx.num_frames
        )
        model_input = torch.cat(
            (latents, condition.to(ctx.device, ctx.target_dtype)), dim=1
        ).to(ctx.target_dtype)
        timestep = ctx.timesteps[part.timestep]
        timestep_2d = stage._expand_timestep(timestep, ctx.batch_size, ctx.device)
        metadata = stage._build_causal_attn_metadata(
            batch,
            self.args,
            current_timestep=part.timestep,
            raw_latent_shape=(ctx.num_frames, ctx.height, ctx.width),
            device=ctx.device,
        )
        pred = stage._forward_causal_transformer(
            batch,
            latent_model_input=model_input,
            prompt_embeds=ctx.prompt_embeds,
            timestep=timestep_2d.unsqueeze(1),
            kv_cache=kv,
            crossattn_cache=cross,
            current_start_tokens=start_frame * stage.num_token_per_frame,
            start_frame=start_frame,
            image_kwargs=ctx.image_kwargs,
            pos_cond_kwargs={
                **ctx.pos_cond_kwargs,
                "block_range": (part.block_start, part.block_end),
                "block_hidden_states": payload.get("hidden_states"),
            },
            current_timestep=part.timestep,
            attn_metadata=metadata,
            target_dtype=ctx.target_dtype,
            autocast_enabled=ctx.autocast_enabled,
        )
        if part.block_end != stage.num_transformer_blocks:
            return {"latents": latents, "hidden_states": pred.contiguous()}
        noise = latents.permute(0, 2, 1, 3, 4)
        prediction = pred.permute(0, 2, 1, 3, 4)
        x0 = pred_noise_to_pred_video(
            pred_noise=prediction.flatten(0, 1),
            noise_input_latent=noise.flatten(0, 1),
            timestep=timestep_2d,
            scheduler=ctx.scheduler,
        ).unflatten(0, prediction.shape[:2])
        if part.timestep + 1 < len(ctx.timesteps):
            generator = torch.Generator(
                device=stage._single_generator(batch).device
            ).manual_seed(
                causal_pipeline_seed(batch.seed, batch.block_idx, part.timestep)
            )
            fresh_noise = randn_tensor(
                x0.shape, dtype=x0.dtype, device=ctx.device, generator=generator
            )
            x0 = ctx.scheduler.add_noise(
                x0.flatten(0, 1),
                fresh_noise.flatten(0, 1),
                ctx.timesteps[part.timestep + 1 : part.timestep + 2],
            ).unflatten(0, x0.shape[:2])
        return {"latents": x0.permute(0, 2, 1, 3, 4).contiguous()}

    def _decode_payload(self, batch, payload):
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        if payload.get("chunk_index") != batch.block_idx:
            raise ValueError("Decoder received a different chunk than expected")
        batch.latents = payload["latents"]
        batch.raw_latent_shape = batch.latents.shape
        result = self._local_stage(self.decode, len(self.pipeline.stages) - 1, batch)
        # Bound device memory even when HTTP delivery waits for a full window.
        return result.output.detach().cpu(memory_format=torch.contiguous_format)

    def _receive_and_decode(self, batches):
        outputs, error = [], None
        for batch in batches:
            payload = self.group.recv_tensor_dict(
                src=self.placement.denoising_ranks - 1
            )
            try:
                if error is None:
                    outputs.append(self._decode_payload(batch, payload))
            except Exception as exc:
                error = f"Causal VAE rank {self.rank}: {exc}"
        self.transport.finish(error)
        return outputs

    @torch.no_grad()
    def forward(self, batches):
        ephemeral = batches[0].session is None
        if ephemeral:
            session = RealtimeSession()
            for batch in batches:
                batch.session = session
        try:
            error = None
            outputs = None
            try:
                with ExitStack() as stack:
                    try:
                        stack.enter_context(
                            self.pipeline.executor.profile_execution(
                                batches[0], dump_rank=0
                            )
                        )
                        self.pipeline._install_component_residency_manager(self.args)
                        stack.enter_context(
                            self.pipeline.executor._component_residency_request(
                                self.pipeline.stages, batches, self.args
                            )
                        )
                    except Exception as exc:
                        error = f"Causal pipeline residency setup on rank {self.rank}: {exc}"
                    self.transport.finish(error)
                    outputs = self._forward_window(batches)
            except Exception as exc:
                error = f"Causal pipeline request on rank {self.rank}: {exc}"
            # Context teardown (offload and profiler shutdown) can fail after the
            # window drains. No rank may return success before all have exited.
            self.transport.finish(error)
            return outputs
        except Exception:
            # Partial KV writes and camera state must never be reused after failure.
            batches[0].session.dispose()
            raise
        finally:
            if ephemeral:
                session.dispose()
                for batch in batches:
                    batch.session = None

    def _forward_window(self, batches):
        state, contexts, plan = self._prepare(batches)
        frames = None
        manager = self.pipeline.executor.component_residency_manager
        setup_error = None
        if self.is_denoiser:
            try:
                self.pipeline.executor.before_stage(
                    self.denoise, len(self.pipeline.stages) - 2, batches, self.args
                )
                manager.begin_stage()
            except Exception as exc:
                setup_error = f"Causal denoiser setup on rank {self.rank}: {exc}"
        # A failed model placement must not leave another rank waiting in recv.
        self.transport.finish(setup_error)
        if self.is_denoiser:
            payloads, cleanup_error = self._denoise_window(
                batches=batches,
                contexts=contexts,
                state=state,
                plan=plan,
                manager=manager,
            )
        else:
            frames = self._receive_and_decode(batches)
            cleanup_error = None

        # Offload/cleanup can fail locally after the execution collective succeeds.
        # Agree on that failure before any rank starts the next transfer phase.
        self.transport.finish(cleanup_error)
        if self.is_denoiser:
            state.next_frame += len(batches) * contexts[0].num_frames

        if self.placement.vae_placement == "shared":
            frames = self._decode_shared_window(batches=batches, payloads=payloads)

        state.next_chunk += len(batches)
        return self._collect_outputs(batches=batches, frames=frames)

    def _denoise_window(self, *, batches, contexts, state, plan, manager):
        cleanup_error = None
        first = batches[0].block_idx
        start_frame = state.next_frame

        def initialize(index):
            return {"latents": contexts[index - first].latents.contiguous()}

        def execute(index, part, payload):
            batch, ctx = batches[index - first], contexts[index - first]
            cache_state = batch.session.get_or_create_state(RealtimeCausalDiTState)
            # Position-dependent conditioning cannot leak between chunks.
            self.denoise._clear_lingbot_dynamic_condition_cache(cache_state)
            name = f"causal_step_{part.timestep}_blocks_{part.block_start}_{part.block_end}"
            with (
                maybe_record_function(name),
                StageProfiler(
                    name,
                    logger=logger,
                    metrics=batch.metrics,
                    perf_dump_path_provided=batch.perf_dump_path is not None,
                    record_as_step=True,
                ),
            ):
                result = self._execute(
                    batch,
                    ctx,
                    state,
                    start_frame + (index - first) * ctx.num_frames,
                    part,
                    payload,
                )
            self.denoise.step_profile()
            return result

        try:
            payloads = run_causal_pipeline(
                plan,
                chunk_indices=[b.block_idx for b in batches],
                transport=self.transport,
                initialize=initialize,
                execute=execute,
                consume=(
                    lambda payload: self.group.send_tensor_dict(
                        payload, dst=self.placement.decoder_rank
                    )
                )
                if self.placement.vae_placement == "dedicated"
                else None,
            )
        finally:
            try:
                manager.end_stage()
            except Exception as exc:
                cleanup_error = f"Causal denoiser cleanup on rank {self.rank}: {exc}"
        return payloads, cleanup_error

    def _decode_shared_window(self, *, batches, payloads):
        frames = []
        # All denoisers have drained. Any denoising rank can now decode
        # without creating a communication cycle through a busy earlier rank.
        error = None
        for i, batch in enumerate(batches):
            source = self.placement.denoising_ranks - 1
            target = self.placement.decoder_rank
            payload = payloads[i] if self.rank == source else None
            if source != target:
                if self.rank == source:
                    self.group.send_tensor_dict(payload, dst=target)
                elif self.rank == target:
                    payload = self.group.recv_tensor_dict(src=source)
            if self.rank == target and error is None:
                try:
                    frames.append(self._decode_payload(batch, payload))
                except Exception as exc:
                    error = f"Shared causal VAE: {exc}"
        self.transport.finish(error)
        return frames

    def _collect_outputs(self, *, batches, frames):
        outputs = []
        for index, batch in enumerate(batches):
            frame = frames[index] if self.rank == self.placement.decoder_rank else None
            if self.placement.decoder_rank != 0:
                if self.rank == self.placement.decoder_rank:
                    self.group.send_tensor_dict({"frames": frame}, dst=0)
                elif self.rank == 0:
                    frame = self.group.recv_tensor_dict(
                        src=self.placement.decoder_rank
                    )["frames"]
            outputs.append(
                OutputBatch(
                    output=frame if self.rank == 0 else None, metrics=batch.metrics
                )
            )
        return outputs
