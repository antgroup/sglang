# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for LingBot block boundaries and bounded realtime submission."""

import asyncio
import copy
import json
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.causal_window import (
    generate_causal_windows,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.models.dits.lingbot_world import (
    CausalLingBotWorldTransformer3DModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.latent_preparation import (
    RealtimeChunkLatentPreparationStage,
)
from sglang.multimodal_gen.runtime.realtime.causal_pipeline import (
    CausalPipelinePlacement,
    CausalPipelinePlan,
    causal_pipeline_seed,
)
from sglang.multimodal_gen.runtime.realtime.lingbot_causal_pipeline import (
    LingBotCausalPipelineRunner,
    LingBotPipelineState,
)
from sglang.multimodal_gen.runtime.realtime.session import (
    RealtimeSession,
    RealtimeSessionCache,
)
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    MemorySnapshot,
    RequestMetrics,
)
from sglang.multimodal_gen.test.causal_pipeline_test_utils import (
    _Links,
    _parallel,
    _Transport,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _HistoryBlock(torch.nn.Module):
    """Small deterministic attention substitute; the production range driver runs."""

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(
        self,
        hidden,
        context,
        timestep_proj,
        freqs,
        *,
        kv_cache,
        crossattn_cache,
        **kwargs,
    ):
        previous = kv_cache.k.mean().clone()
        kv_cache.k.fill_(hidden.mean())
        if not crossattn_cache.is_init:
            crossattn_cache.v = context.mean().clone()
            crossattn_cache.is_init = True
        return hidden * (1 + self.index / 10) + previous + crossattn_cache.v


class _Norm(torch.nn.Module):
    def forward(self, hidden, shift, scale):
        return hidden * (1 + scale.flatten(1, 2)) + shift.flatten(1, 2)


def _small_lingbot():
    model = CausalLingBotWorldTransformer3DModel.__new__(
        CausalLingBotWorldTransformer3DModel
    )
    torch.nn.Module.__init__(model)
    model.sp_size = 1
    model.patch_size = (1, 1, 1)
    model.hidden_size = 4
    model.block_mask = None
    model.patch_embedding = torch.nn.Conv3d(2, 4, 1)
    model.blocks = torch.nn.ModuleList([_HistoryBlock(index) for index in range(4)])
    model.condition_embedder = SimpleNamespace(
        text_embedder=lambda x: x * 2,
        time_embedder=lambda t: t.float().unsqueeze(-1).repeat(1, 4) / 1000,
        time_modulation=lambda t: t.repeat(1, 6),
    )
    model.scale_shift_table = torch.zeros(1, 2, 4)
    model.norm_out = _Norm()
    model.proj_out = torch.nn.Linear(4, 2)
    return model


def _caches():
    return (
        [SimpleNamespace(k=torch.tensor(0.0)) for _ in range(4)],
        [SimpleNamespace(is_init=False, v=None) for _ in range(4)],
    )


def _make_window_runner(
    *, rank, placement, plan, session, links, failure_rank, cleanup_phase
):
    transport = _Transport(links, rank)
    runner = LingBotCausalPipelineRunner.__new__(LingBotCausalPipelineRunner)
    runner.rank, runner.group, runner.transport = (
        rank,
        transport,
        transport,
    )
    runner.placement = placement
    runner.is_denoiser = rank < plan.degree
    runner.args = None
    runner.decode = object()
    runner.denoise = SimpleNamespace(
        _clear_lingbot_dynamic_condition_cache=lambda state: None,
        step_profile=Mock(),
    )

    def end_stage():
        if rank == failure_rank and cleanup_phase == "stage":
            raise RuntimeError("injected cleanup failure")

    @contextmanager
    def request_context(phase):
        yield
        if rank == failure_rank and cleanup_phase == phase:
            raise RuntimeError("injected cleanup failure")

    manager = SimpleNamespace(begin_stage=lambda: None, end_stage=end_stage)
    runner.pipeline = SimpleNamespace(
        stages=[None, None],
        _install_component_residency_manager=lambda args: None,
        executor=SimpleNamespace(
            before_stage=lambda *args: None,
            component_residency_manager=manager,
            profile_execution=lambda *a, **kw: request_context("profile"),
            _component_residency_request=lambda *a: request_context("request"),
        ),
    )
    state = session.get_or_create_state(LingBotPipelineState)
    contexts = [
        SimpleNamespace(
            latents=torch.tensor([float(i + 10)]),
            device="cpu",
            num_frames=1,
        )
        for i in range(5)
    ]
    runner._prepare = lambda batches: (state, contexts, plan)
    runner._execute = lambda batch, ctx, state, start, part, payload: {
        "latents": payload["latents"] + (part.block_end - part.block_start)
    }
    runner._local_stage = lambda stage, index, batch: SimpleNamespace(
        output=batch.latents * 2
    )
    batches = [
        SimpleNamespace(
            block_idx=i,
            seed=42,
            session=session,
            metrics=None,
            perf_dump_path=None,
        )
        for i in range(5)
    ]
    return runner, batches


class TestLingBotCausalPipeline(CustomTestCase):
    def test_noisy_sampling_is_independent_of_partition(self):
        """Exercise the production step adapter, KV allocation and noise generation."""
        torch.manual_seed(123)
        template = _small_lingbot()
        template.num_attention_heads, template.attention_head_dim = 1, 2
        policy = SimpleNamespace(expected_cache_tokens=6, expected_sink_tokens=2)
        ctx = SimpleNamespace(
            batch_size=1,
            target_dtype=torch.float32,
            device=torch.device("cpu"),
            num_frames=1,
            height=2,
            width=2,
            autocast_enabled=False,
            timesteps=torch.tensor([1000, 500]),
            prompt_embeds=torch.ones(1, 2, 4),
            image_kwargs={},
            pos_cond_kwargs={},
            scheduler=SimpleNamespace(
                timesteps=torch.tensor([1000, 500]),
                sigmas=torch.tensor([1.0, 0.5]),
                add_noise=lambda x, noise, timestep: (x + noise) / 2,
            ),
        )

        def simulate(mode, degree):
            plan = CausalPipelinePlan(2, 4, degree, mode)
            workers = []
            for rank in range(degree):
                runner = LingBotCausalPipelineRunner.__new__(
                    LingBotCausalPipelineRunner
                )
                runner.rank, runner.args = rank, None
                model = copy.deepcopy(template)
                model._prepare_cached_rope = lambda **kw: ()

                def forward(
                    batch,
                    *,
                    latent_model_input,
                    timestep,
                    kv_cache,
                    crossattn_cache,
                    pos_cond_kwargs,
                    current_timestep,
                    model=model,
                    **kw,
                ):
                    with set_forward_context(current_timestep, None, batch):
                        return model(
                            latent_model_input,
                            ctx.prompt_embeds,
                            timestep,
                            kv_cache=kv_cache,
                            crossattn_cache=crossattn_cache,
                            **pos_cond_kwargs,
                        )

                runner.denoise = SimpleNamespace(
                    transformer=model,
                    num_transformer_blocks=4,
                    num_token_per_frame=4,
                    _select_i2v_condition_chunk=lambda condition, *args: condition,
                    _expand_timestep=lambda timestep, batch_size, device: (
                        timestep.reshape(1).expand(batch_size)
                    ),
                    _build_causal_attn_metadata=lambda *a, **kw: None,
                    _forward_causal_transformer=forward,
                    _single_generator=lambda batch: batch.generator,
                )
                state = LingBotPipelineState()
                runner._allocate_histories(state, ctx, plan, policy)
                workers.append((runner, state))
            outputs = []
            with torch.no_grad():
                for chunk in range(3):
                    batch = SimpleNamespace(
                        seed=42,
                        block_idx=chunk,
                        condition_inputs={},
                        image_latent=torch.empty(1, 0, 1, 2, 2),
                        session=None,
                        extra={},
                        generator=torch.Generator(device="cpu"),
                    )
                    generator = torch.Generator().manual_seed(
                        causal_pipeline_seed(42, chunk, -1)
                    )
                    payload = {
                        "latents": torch.randn(1, 2, 1, 2, 2, generator=generator)
                    }
                    for rank, (runner, state) in enumerate(workers):
                        for part in plan.slices(rank):
                            payload = runner._execute(
                                batch, ctx, state, chunk, part, payload
                            )
                    outputs.append(payload["latents"])
            return outputs

        expected = simulate("timestep", 1)
        for mode, degree in (("timestep", 2), ("block", 3)):
            for actual, reference in zip(simulate(mode, degree), expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_rank_local_histories_do_not_alias_steps_or_unowned_layers(self):
        runner = LingBotCausalPipelineRunner.__new__(LingBotCausalPipelineRunner)
        runner.denoise = SimpleNamespace(
            transformer=SimpleNamespace(num_attention_heads=1, attention_head_dim=2)
        )
        ctx = SimpleNamespace(batch_size=1, target_dtype=torch.float32, device="cpu")
        policy = SimpleNamespace(expected_cache_tokens=6, expected_sink_tokens=2)
        plan = CausalPipelinePlan(4, 3, 2, "block")
        pointers = set()
        held_tensors = []
        for rank in range(2):
            runner.rank = rank
            state = LingBotPipelineState()
            runner._allocate_histories(state, ctx, plan, policy)
            for part in plan.slices(rank):
                kv, cross = state.histories[part.timestep]
                for layer in range(3):
                    if part.block_start <= layer < part.block_end:
                        self.assertNotIn(kv[layer].k.data_ptr(), pointers)
                        pointers.add(kv[layer].k.data_ptr())
                        held_tensors.append(kv[layer].k)
                        self.assertIsNotNone(cross[layer])
                    else:
                        self.assertIsNone(kv[layer])
                        self.assertIsNone(cross[layer])
            # Release is per session and must invalidate its frontier too.
            state.next_chunk = 9
            state.dispose()
            self.assertEqual((state.histories, state.next_chunk), ({}, 0))

    def test_window_routes_decode_to_shared_or_dedicated_rank(self):
        """In particular, a shared rank upstream of the final stage must not deadlock."""
        cases = (
            (mode, degree, cleanup_rank, cleanup_phase)
            for mode, degree in (("timestep", 2), ("block", 3))
            for cleanup_rank, cleanup_phase in (
                (-1, "stage"),
                (0, "stage"),
                (degree - 1, "stage"),
                (0, "request"),
                (degree - 1, "request"),
                (0, "profile"),
                (degree - 1, "profile"),
            )
        )
        for mode, degree, cleanup_rank, cleanup_phase in cases:
            for placement in (
                CausalPipelinePlacement(degree),
                CausalPipelinePlacement(degree, "shared", 0),
                CausalPipelinePlacement(degree, "dedicated"),
            ):
                failure_rank = (
                    placement.decoder_rank
                    if cleanup_rank == degree - 1 and cleanup_phase != "stage"
                    else cleanup_rank
                )
                with self.subTest(
                    mode=mode,
                    placement=placement,
                    cleanup_rank=failure_rank,
                    cleanup_phase=cleanup_phase,
                ):
                    world = placement.world_size
                    links = _Links(world)
                    sessions = [RealtimeSession() for _ in range(world)]
                    plan = CausalPipelinePlan(2, 3, degree, mode)

                    def worker(rank):
                        session = sessions[rank]
                        runner, batches = _make_window_runner(
                            rank=rank,
                            placement=placement,
                            plan=plan,
                            session=session,
                            links=links,
                            failure_rank=failure_rank,
                            cleanup_phase=cleanup_phase,
                        )
                        state = session.get_state(LingBotPipelineState)
                        outputs = runner.forward(batches)
                        self.assertEqual(state.next_chunk, 5)
                        if runner.is_denoiser:
                            self.assertEqual(
                                runner.denoise.step_profile.call_count,
                                5 * len(plan.slices(rank)),
                            )
                        return outputs

                    results, errors = _parallel(
                        [lambda rank=rank: worker(rank) for rank in range(world)]
                    )
                    if cleanup_rank >= 0:
                        for error in errors:
                            self.assertIsInstance(error, RuntimeError)
                            self.assertIn("injected cleanup failure", str(error))
                        for session in sessions:
                            self.assertIsNone(session.get_state(LingBotPipelineState))
                        continue
                    self.assertEqual(errors, [None] * world)
                    for index, output in enumerate(results[0]):
                        torch.testing.assert_close(
                            output.output, torch.tensor([(index + 16.0) * 2])
                        )
                    self.assertTrue(
                        all(
                            item.output is None
                            for outputs in results[1:]
                            for item in outputs
                        )
                    )

    def test_decode_normalizes_channels_last_output_for_cpu_transfer(self):
        runner = LingBotCausalPipelineRunner.__new__(LingBotCausalPipelineRunner)
        frames = torch.randn(1, 3, 2, 4, 4).to(memory_format=torch.channels_last_3d)
        self.assertFalse(frames.is_contiguous())
        runner.decode = object()
        runner.pipeline = SimpleNamespace(stages=[runner.decode])
        runner._local_stage = lambda *args: SimpleNamespace(output=frames)
        output = runner._decode_payload(
            SimpleNamespace(block_idx=0),
            {"chunk_index": 0, "latents": torch.zeros(1)},
        )
        self.assertTrue(output.is_contiguous())
        self.assertEqual(output.device.type, "cpu")
        torch.testing.assert_close(output, frames)

    def test_preparation_seeds_existing_stage_and_preserves_supplied_latents(self):
        runner = LingBotCausalPipelineRunner.__new__(LingBotCausalPipelineRunner)
        runner.rank, runner.is_denoiser = 0, True
        runner.args = SimpleNamespace(
            causal_pipeline_window=2, causal_pipeline_mode="timestep"
        )
        runner.placement = CausalPipelinePlacement(1)
        runner.transport = SimpleNamespace(
            finish=lambda error: self.assertIsNone(error)
        )
        stage = RealtimeChunkLatentPreparationStage.__new__(
            RealtimeChunkLatentPreparationStage
        )
        stage.transformer = SimpleNamespace(config=SimpleNamespace(out_channels=2))
        stage.scheduler = object()
        runner.pipeline = SimpleNamespace(stages=[stage, None, None])
        runner._local_stage = lambda stage, index, batch: stage.forward(
            batch, runner.args
        )
        runner._allocate_histories = Mock()
        runner.denoise = SimpleNamespace(
            _single_generator=lambda batch: batch.generator,
            num_transformer_blocks=2,
            _build_realtime_causal_cache_policy=lambda *args: SimpleNamespace(
                expected_cache_tokens=4, expected_sink_tokens=0
            ),
            _prepare_causal_dmd_forward_context=lambda batch, args: SimpleNamespace(
                batch_size=1,
                latents=batch.latents,
                timesteps=torch.tensor([1000, 500]),
                target_dtype=torch.float32,
            ),
        )
        supplied = torch.full((1, 2, 1, 2, 2), 7.0)
        session = RealtimeSession()
        batches = [
            SimpleNamespace(
                realtime_session_id="stream",
                session=session,
                block_idx=i,
                num_outputs_per_prompt=1,
                do_classifier_free_guidance=False,
                rollout=False,
                seed=42,
                generator=torch.Generator(device="cpu"),
                batch_size=1,
                latents=None if i == 0 else supplied,
                realtime_chunk_size=1,
                image_latent=torch.zeros(1, 2, 1, 2, 2),
                height=None,
                width=None,
            )
            for i in range(2)
        ]
        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            _, contexts, _ = runner._prepare(batches)
        expected = torch.randn(
            supplied.shape,
            generator=torch.Generator(device="cpu").manual_seed(
                causal_pipeline_seed(42, 0, -1)
            ),
        )
        torch.testing.assert_close(contexts[0].latents, expected, rtol=0, atol=0)
        self.assertEqual(batches[0].generator.device.type, "cpu")
        self.assertIs(contexts[1].latents, supplied)

    def test_window_enters_and_closes_existing_profile_context(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                events = []

                @contextmanager
                def profile(batch, dump_rank):
                    self.assertTrue(batch.profile)
                    self.assertEqual(dump_rank, 0)
                    events.append("start")
                    try:
                        yield
                    finally:
                        events.append("stop")

                def forward(batches):
                    events.append("execute")
                    if fail:
                        raise RuntimeError("injected execution failure")
                    return []

                runner = LingBotCausalPipelineRunner.__new__(
                    LingBotCausalPipelineRunner
                )
                runner.args, runner.rank = None, 0

                def finish(error):
                    if error:
                        raise RuntimeError(error)

                runner.transport = SimpleNamespace(finish=finish)
                runner.pipeline = SimpleNamespace(
                    stages=[],
                    _install_component_residency_manager=lambda args: None,
                    executor=SimpleNamespace(
                        profile_execution=profile,
                        _component_residency_request=lambda *args: nullcontext(),
                    ),
                )
                runner._forward_window = forward
                batches = [SimpleNamespace(session=RealtimeSession(), profile=True)]
                if fail:
                    with self.assertRaisesRegex(
                        RuntimeError, "injected execution failure"
                    ):
                        runner.forward(batches)
                else:
                    self.assertEqual(runner.forward(batches), [])
                self.assertEqual(events, ["start", "execute", "stop"])

    def test_worker_reports_window_time_and_materializes_every_chunk(self):
        """Cached outputs must not look like near-zero-time generation in reports."""
        worker = GPUWorker.__new__(GPUWorker)
        worker.server_args = SimpleNamespace(
            causal_pipeline_mode="timestep", causal_pipeline_window=3, model_path="test"
        )
        worker.is_output_rank = True
        worker.memory_occupation = None
        worker._realtime_sessions = RealtimeSessionCache(max_sessions=1)
        worker._release_warmup_pool_before_serving = False
        worker._runtime_peak_reserved_mb = worker._runtime_peak_allocated_mb = 0
        clock = [0.0]

        def forward(batches, args):
            clock[0] += 0.055
            return [
                OutputBatch(
                    output=torch.full((1, 3, 1, 2, 2), i / 2), metrics=batch.metrics
                )
                for i, batch in enumerate(batches)
            ]

        worker.pipeline = SimpleNamespace(forward_causal_window=forward)
        module = "sglang.multimodal_gen.runtime.managers.gpu_worker"
        perf = "sglang.multimodal_gen.runtime.utils.perf_logger"
        with tempfile.TemporaryDirectory() as directory:
            batches = [
                SimpleNamespace(
                    realtime_session_id="stream",
                    request_id=f"chunk-{i}",
                    block_idx=i,
                    session=None,
                    num_outputs_per_prompt=1,
                    do_classifier_free_guidance=False,
                    rollout=False,
                    seed=42,
                    is_warmup=False,
                    extra={},
                    trace_ctx=None,
                    metrics=RequestMetrics(f"chunk-{i}"),
                    log=lambda **kw: None,
                    perf_dump_path=str(Path(directory) / f"chunk-{i}.json"),
                    suppress_logs=True,
                    return_raw_frames=True,
                    enable_frame_interpolation=False,
                    enable_upscaling=False,
                )
                for i in range(3)
            ]
            device = Mock()
            with (
                patch(
                    f"{module}.current_platform",
                    SimpleNamespace(
                        is_cpu=lambda: False,
                        is_mps=lambda: False,
                        is_cuda=lambda: False,
                    ),
                ),
                patch(f"{module}.torch.get_device_module", return_value=device),
                patch(
                    f"{module}.capture_memory_snapshot",
                    return_value=MemorySnapshot(1, 2, 3, 4),
                ),
                patch(f"{module}.time.monotonic", side_effect=lambda: clock[0]),
                patch(f"{module}.trace_slice", return_value=nullcontext()),
                patch(f"{perf}.get_diffusion_perf_log_dir", return_value=directory),
                patch(f"{perf}.get_is_main_process", return_value=True),
                patch(f"{perf}.get_git_commit_hash", return_value="test"),
            ):
                result = worker.execute_causal_window(batches)
            self.assertIsNone(result.error)
            self.assertEqual(device.reset_peak_memory_stats.call_count, 1)
            self.assertEqual(clock[0], 0.055)
            for i, output in enumerate(result.chunk_outputs):
                self.assertIsNone(output.output)
                self.assertEqual(
                    output.raw_frame_batches, [[bytes([int(i / 2 * 255)]) * 12]]
                )
                self.assertAlmostEqual(output.metrics.total_duration_ms, 55)
                metadata = output.metrics.to_dict()
                report = json.loads(Path(batches[i].perf_dump_path).read_text())
                for item in (metadata, report):
                    self.assertEqual(item["total_duration_scope"], "window")
                    self.assertEqual(item["window_chunks"], 3)
                    self.assertAlmostEqual(item["total_duration_ms"], 55)
            records = [
                json.loads(line)
                for line in (Path(directory) / "performance.log")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(records), 3)
            for record in records:
                self.assertEqual(record["total_duration_scope"], "window")
                self.assertEqual(record["window_chunks"], 3)
                self.assertAlmostEqual(record["total_duration_ms"], 55)
            self.assertNotIn(
                "total_duration_scope", RequestMetrics("ordinary").to_dict()
            )

    def test_partial_forwards_preserve_full_forward_and_history(self):
        """A split must neither re-embed intermediates nor project them early."""
        torch.manual_seed(123)
        full = _small_lingbot()
        split = copy.deepcopy(full)
        full_kv, full_cross = _caches()
        split_kv, split_cross = _caches()
        batch = SimpleNamespace(extra={}, enable_sequence_shard=False)
        with (
            patch.object(full, "_prepare_cached_rope", return_value=()),
            patch.object(split, "_prepare_cached_rope", return_value=()),
            set_forward_context(0, None, batch),
            torch.no_grad(),
        ):
            for chunk in range(3):
                inputs = dict(
                    hidden_states=torch.randn(1, 2, 1, 2, 2),
                    encoder_hidden_states=torch.ones(1, 2, 4),
                    timestep=torch.tensor([[500]]),
                    start_frame=chunk,
                )
                expected = full(
                    **inputs,
                    kv_cache=full_kv,
                    crossattn_cache=full_cross,
                    block_range=(0, 4),
                )
                intermediate = split(
                    **inputs,
                    kv_cache=split_kv,
                    crossattn_cache=split_cross,
                    block_range=(0, 1),
                )
                self.assertEqual(intermediate.shape, (1, 4, 4))
                # Unowned caches are absent on a real rank.
                actual = split(
                    **inputs,
                    kv_cache=[None] + split_kv[1:],
                    crossattn_cache=[None] + split_cross[1:],
                    block_range=(1, 4),
                    block_hidden_states=intermediate,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for left, right in zip(full_kv, split_kv):
                    torch.testing.assert_close(left.k, right.k, rtol=0, atol=0)

    def test_noninitial_range_cannot_silently_reembed_latents(self):
        model = _small_lingbot()
        with self.assertRaisesRegex(ValueError, "intermediate"):
            model(
                torch.zeros(1, 2, 1, 2, 2),
                torch.zeros(1, 2, 4),
                torch.tensor([[500]]),
                block_range=(1, 3),
            )

    def test_window_is_bounded_and_delivery_precedes_next_submission(self):
        async def scenario():
            events = []
            session = SimpleNamespace(
                id="stream",
                generate_chunk_cnt=0,
                current_chunk=None,
                request=SimpleNamespace(max_chunks=5),
            )
            session.reached_max_chunks = lambda: session.generate_chunk_cnt >= 5

            class Adapter:
                async def wait_for_next_chunk(self, current):
                    pass

                def prepare_next_request(self, current, args, chunk):
                    events.append(("prepare", chunk.index))
                    return SimpleNamespace(block_idx=chunk.index, extra={})

                def on_chunk_complete(self, current, result):
                    current.generate_chunk_cnt += 1
                    current.current_chunk = None

            class Client:
                async def forward(self, request):
                    indices = [batch.block_idx for batch in request.chunks]
                    events.append(("submit", indices))
                    return SimpleNamespace(
                        error=None,
                        chunk_outputs=[SimpleNamespace(error=None) for _ in indices],
                    )

            async def send(ws, current, chunk, batch, output, *timings):
                await asyncio.sleep(0)
                self.assertEqual(
                    batch.extra["causal_pipeline_window_size"],
                    3 if chunk.index < 3 else 2,
                )
                events.append(("send", chunk.index))

            session.adapter = Adapter()
            await generate_causal_windows(
                None, session, SimpleNamespace(causal_pipeline_window=3), Client(), send
            )
            return events

        events = asyncio.run(scenario())
        self.assertEqual(
            [value for kind, value in events if kind == "submit"], [[0, 1, 2], [3, 4]]
        )
        self.assertEqual(
            [value for kind, value in events if kind == "send"], list(range(5))
        )
        self.assertLess(events.index(("send", 2)), events.index(("prepare", 3)))

    def test_cancellation_does_not_admit_another_window(self):
        async def scenario():
            started = asyncio.Event()
            submissions = []
            session = SimpleNamespace(
                id="stream",
                generate_chunk_cnt=0,
                request=SimpleNamespace(max_chunks=None),
            )
            session.reached_max_chunks = lambda: False

            class Adapter:
                async def wait_for_next_chunk(self, session):
                    pass

                def prepare_next_request(self, session, args, chunk):
                    return SimpleNamespace(block_idx=chunk.index, extra={})

            class Client:
                async def forward(self, request):
                    submissions.append(request)
                    started.set()
                    await asyncio.Event().wait()

            session.adapter = Adapter()
            task = asyncio.create_task(
                generate_causal_windows(
                    None,
                    session,
                    SimpleNamespace(causal_pipeline_window=3),
                    Client(),
                    None,
                )
            )
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(len(submissions), 1)
            self.assertEqual(session.generate_chunk_cnt, 0)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
