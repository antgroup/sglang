import asyncio
import contextlib
import pickle
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import sglang.srt.disaggregation.encoder.server as encoder_server
from sglang.srt.disaggregation.encoder import http_server
from sglang.srt.disaggregation.encoder import runtime as encoder_runtime
from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor
from sglang.srt.disaggregation.encoder.receiver import EmbeddingData
from sglang.srt.disaggregation.encoder.runtime import (
    _DP_RELEASE_AFTER_ENCODE,
    DPDispatcher,
    EncoderScheduler,
    _retire_abandoned_encode,
)
from sglang.srt.disaggregation.encoder.server import (
    EncoderMetaRegistry,
    MMEncoder,
    MMError,
    meta_registry,
)
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import unwrap_from_pickle
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.encoder_preprocessing import EncoderMediaProcessorConfig
from sglang.srt.runtime_context import get_context, get_flags
from sglang.srt.utils.common import safe_pickle_loads
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestEncoderRouteSelection(CustomTestCase):
    def test_preprocess_meta_route_requires_epd_in_the_http_process(self):
        """Importing HTTP used to expose metadata preprocessing even when off.
        Exercise startup as well: DP children cannot initialize the HTTP parent.
        """
        for dp_size in (1, 2):
            for enabled, disabled, expected in (
                (False, "", 404),
                (True, "", 200),
                (True, "epd", 404),
            ):
                with self.subTest(dp_size=dp_size, enabled=enabled, disabled=disabled):
                    # Preserve the actual import-time routes, including a leaked
                    # decorator on pre-fix code; isolate each launch's app state.
                    app = FastAPI()
                    app.router.routes.extend(
                        route
                        for route in http_server.app.router.routes
                        if isinstance(route, APIRoute)
                    )
                    content = {"req_id": "meta-route", "items": []}
                    backend = SimpleNamespace(
                        preprocess_meta=AsyncMock(return_value=content)
                    )
                    runtime = SimpleNamespace(
                        encoder=backend, scheduler=None, send_sockets=[]
                    )
                    dispatcher = SimpleNamespace(
                        dispatch=AsyncMock(return_value={"content": content})
                    )
                    model_config = SimpleNamespace(
                        hf_config=SimpleNamespace(model_type="deepseek_v41")
                    )
                    with (
                        get_context().override_server_args(
                            encoder_only=True,
                            dp_size=dp_size,
                            enable_metrics=False,
                            encoder_register_urls=[],
                            _model_config=model_config,
                        ) as server_args,
                        get_flags().dsv41_sm90.override(
                            subsystems=frozenset(), lossy=frozenset(), role=""
                        ),
                        envs.SGLANG_OPT_DSV41_SM90.override(enabled),
                        envs.SGLANG_OPT_DSV41_SM90_DISABLE.override(disabled),
                        envs.SGLANG_OPT_DSV41_SM90_LOSSY.override(""),
                        patch(
                            "sglang.srt.layers.dsv41_sm90.is_sm90_supported",
                            return_value=True,
                        ),
                        patch.multiple(
                            http_server,
                            app=app,
                            encoder=None,
                            encoder_scheduler=None,
                            local_runtime=None,
                            send_sockets=[],
                            dp_dispatcher=None,
                        ),
                        patch.object(http_server, "configure_logger"),
                        patch.object(
                            http_server, "launch_local_runtime", return_value=runtime
                        ),
                        patch.object(
                            http_server, "launch_dp_runtime", return_value=dispatcher
                        ),
                        patch.object(http_server.uvicorn, "run"),
                    ):
                        http_server.launch_server(server_args)
                        with TestClient(app) as client:
                            response = client.post("/preprocess_meta", json=content)
                            self.assertEqual(response.status_code, expected)
                            if expected == 200:
                                self.assertEqual(response.json(), content)
                            # OpenAPI is also a client-visible surface.
                            paths = client.get("/openapi.json").json()["paths"]
                            self.assertEqual(
                                "/preprocess_meta" in paths, expected == 200
                            )
                        if expected == 200:
                            # Rebuilding in the same process must remove both
                            # the route and its cached OpenAPI entry.
                            with envs.SGLANG_OPT_DSV41_SM90.override(False):
                                http_server.launch_server(server_args)
                                with TestClient(app) as client:
                                    self.assertEqual(
                                        client.post(
                                            "/preprocess_meta", json=content
                                        ).status_code,
                                        404,
                                    )
                                    self.assertNotIn(
                                        "/preprocess_meta",
                                        client.get("/openapi.json").json()["paths"],
                                    )


class TestEncoderDPErrorHandling(CustomTestCase):
    def test_encode_control_field_cannot_enable_metadata_preprocessing(self):
        """The /encode dict can carry _dp_type; that must not bypass EPD opt-out."""
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                enc = SimpleNamespace(
                    dsv41_epd=enabled,
                    preprocess_meta=AsyncMock(return_value={"metadata": True}),
                )
                send = AsyncMock()

                async def run():
                    with (
                        patch.object(encoder_runtime, "async_sock_send", send),
                        patch.object(
                            encoder_runtime,
                            "execute_encode_pipeline",
                            AsyncMock(return_value={"embedding": True}),
                        ),
                    ):
                        await encoder_runtime._dp_worker_handle_request(
                            enc,
                            None,
                            object(),
                            asyncio.Lock(),
                            0,
                            {"req_id": "injected", "_dp_type": "preprocess_meta"},
                            "preprocess_meta",
                        )

                asyncio.run(run())
                envelope = unwrap_from_pickle(send.await_args.args[1])
                self.assertEqual(
                    envelope["content"],
                    {"metadata": True} if enabled else {"embedding": True},
                )

    @staticmethod
    async def _run_registration_error(error):
        encoder = SimpleNamespace(
            register_embedding_destinations=AsyncMock(side_effect=error)
        )
        send = AsyncMock()
        request = {
            "req_id": "req",
            "receive_count": 1,
            "receive_url": "tcp://127.0.0.1:1",
        }
        with patch.object(encoder_runtime, "async_sock_send", send):
            await encoder_runtime._dp_worker_handle_request(
                encoder,
                None,
                object(),
                asyncio.Lock(),
                0,
                request,
                "register_destinations",
            )
        return unwrap_from_pickle(send.await_args.args[1])

    def test_worker_reports_third_party_exception_with_callable_code(self):
        class RpcLikeError(Exception):
            def code(self):
                return "INTERNAL"

        envelope = asyncio.run(
            self._run_registration_error(RpcLikeError("registration failed"))
        )
        self.assertEqual(envelope["_error"], "registration failed")
        self.assertEqual(envelope["_error_code"], 500)

    def test_worker_preserves_mm_error_status(self):
        envelope = asyncio.run(
            self._run_registration_error(MMError("bad destination", code=400))
        )
        self.assertEqual(envelope["_error_code"], 400)

    def test_dispatcher_drops_malformed_result_without_stopping_listener(self):
        async def run():
            dispatcher = encoder_runtime.DPDispatcher(
                dp_size=1,
                dispatch_sockets=[object()],
                release_sockets=[object()],
                result_socket=object(),
                worker_processes=[],
            )
            future = asyncio.get_running_loop().create_future()
            dispatcher.pending_futures[0]["req"] = future
            dispatcher.req_id_to_rank["req"] = 0
            valid = {"req_id": "req", "_dp_type": "encode", "content": None}
            recv = AsyncMock(
                side_effect=[
                    ["not", "an", "envelope"],
                    valid,
                    asyncio.CancelledError(),
                ]
            )

            with patch.object(encoder_runtime, "async_sock_recv", recv):
                listener = asyncio.create_task(dispatcher._result_listener())
                await asyncio.wait_for(future, timeout=1)
                listener.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await listener

            self.assertEqual(future.result(), valid)

        asyncio.run(run())


class TestEncoderMetaRegistry(CustomTestCase):
    def test_stale_releases_do_not_block_each_other(self):
        async def run():
            registry = EncoderMetaRegistry(wait_timeout=1, sweep_timeout=1)
            blocked_started = asyncio.Event()
            unblock = asyncio.Event()
            fast_released = asyncio.Event()

            async def release(req_id):
                if req_id == "blocked":
                    blocked_started.set()
                    await unblock.wait()
                else:
                    fast_released.set()

            registry.on_release = release
            registry._pending_at.update(blocked=0, fast=0)

            blocked_task = registry._schedule_stale_release("blocked")
            await asyncio.wait_for(blocked_started.wait(), timeout=1)
            fast_task = registry._schedule_stale_release("fast")
            await asyncio.wait_for(fast_released.wait(), timeout=1)
            await fast_task

            self.assertIn("blocked", registry._pending_at)
            self.assertNotIn("fast", registry._pending_at)

            unblock.set()
            await blocked_task
            self.assertNotIn("blocked", registry._pending_at)

        asyncio.run(run())

    def test_failed_stale_release_is_retried(self):
        async def run():
            registry = EncoderMetaRegistry(wait_timeout=1, sweep_timeout=1)
            attempts = 0

            async def release(_req_id):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("transient cleanup failure")

            registry.on_release = release
            registry._pending_at["req"] = 0

            await registry._release_stale("req")
            self.assertIn("req", registry._pending_at)
            retry_at = registry._pending_at["req"]
            self.assertGreater(retry_at, 0)

            await registry._release_stale("req")
            self.assertNotIn("req", registry._pending_at)
            self.assertEqual(attempts, 2)

        asyncio.run(run())

    def test_send_retries_do_not_release_before_all_destinations_finish(self):
        async def run():
            registry = EncoderMetaRegistry(wait_timeout=1, sweep_timeout=1)
            released = AsyncMock()
            registry.on_release = released

            await registry.note_send_done("req", 2, "10.0.0.1:5000")
            await registry.note_send_done("req", 2, "10.0.0.1:5000")
            released.assert_not_awaited()

            await registry.note_send_done("req", 2, "10.0.0.2:5000")
            released.assert_awaited_once_with("req")

        asyncio.run(run())

    def test_http_send_counts_the_normalized_destination(self):
        async def run():
            send = AsyncMock(return_value=True)
            note_send_done = AsyncMock()
            request = {
                "req_id": "req",
                "prefill_host": "127.0.0.1",
                "embedding_port": 5000,
                "session_id": "session",
                "buffer_address": 1234,
                "receive_count": 2,
            }
            with (
                patch.object(http_server, "dp_dispatcher", None),
                patch.object(http_server, "encoder", SimpleNamespace(send=send)),
                patch.object(
                    encoder_server.meta_registry,
                    "note_send_done",
                    note_send_done,
                ),
            ):
                response = await http_server.handle_send_request(request)

            self.assertEqual(response.status_code, 200)
            note_send_done.assert_awaited_once_with("req", 2, "127.0.0.1:5000")

        asyncio.run(run())

    def test_dp_send_counts_the_normalized_destination(self):
        async def run():
            encoder = SimpleNamespace(send=AsyncMock(return_value=True))
            note_send_done = AsyncMock()
            request = {
                "req_id": "req",
                "prefill_host": "127.0.0.1",
                "embedding_port": 5000,
                "session_id": "session",
                "buffer_address": 1234,
                "receive_count": 2,
            }
            with (
                patch.object(encoder_runtime, "async_sock_send", AsyncMock()),
                patch.object(
                    encoder_server.meta_registry,
                    "note_send_done",
                    note_send_done,
                ),
            ):
                await encoder_runtime._dp_worker_handle_request(
                    encoder,
                    None,
                    object(),
                    asyncio.Lock(),
                    0,
                    request,
                    "send",
                )

            note_send_done.assert_awaited_once_with("req", 2, "127.0.0.1:5000")

        asyncio.run(run())


class TestEncoderPreprocessorOffEventLoop(CustomTestCase):
    @staticmethod
    def _make_preprocessor():
        preprocessor = EncoderPreprocessor.__new__(EncoderPreprocessor)
        preprocessor.model_type = "minimax_m3_vl"
        preprocessor.image_processor = None
        preprocessor.use_image_processor_gpu = False
        preprocessor.vision_config = {"image": {}, "video": {}}
        preprocessor.encoder_media_processor_config = EncoderMediaProcessorConfig(
            preserve_media_metadata=True, preprocess_off_event_loop=True
        )
        preprocessor.io_executor = ThreadPoolExecutor(max_workers=2)
        preprocessor.preproc_executor = ThreadPoolExecutor(max_workers=1)
        return preprocessor

    @staticmethod
    def _run_with_loop_handshake(make_awaitable):
        """The hook blocks until a coroutine on the event loop answers it, so a hook
        that holds the loop thread starves that coroutine and times out."""
        started, answered = threading.Event(), threading.Event()

        def wait_for_loop():
            started.set()
            if not answered.wait(timeout=5):
                raise TimeoutError("the preprocess hook blocked the event loop")

        async def answer():
            while not started.is_set():
                await asyncio.sleep(0.001)
            answered.set()

        async def run():
            responder = asyncio.create_task(answer())
            result = await make_awaitable(wait_for_loop)
            await asyncio.wait_for(responder, timeout=5)
            return result

        return asyncio.run(run())

    def test_video_hook_runs_off_the_event_loop(self):
        preprocessor = self._make_preprocessor()
        preprocessor._load_single_item = lambda data, modality: "decoded"

        def process(wait_for_loop):
            def hook(videos, modality, config):
                wait_for_loop()
                return {"videos": videos}

            return preprocessor._process_video_items([{"url": "a.mp4"}], hook)

        output = self._run_with_loop_handshake(process)
        self.assertEqual(output, {"videos": ["decoded"]})

    def test_image_hook_runs_off_the_event_loop(self):
        preprocessor = self._make_preprocessor()
        preprocessor._flatten_and_load_images = AsyncMock(
            return_value=[torch.zeros(3, 4, 4)]
        )
        grid = torch.tensor([[1, 2, 2]])

        def process(wait_for_loop):
            def hook(images, modality, config):
                wait_for_loop()
                return {"image_grid_thw": grid}

            return preprocessor._process_image_items(["img"], hook)

        output = self._run_with_loop_handshake(process)
        self.assertIs(output["image_grid_thw"], grid)


class TestEncoderPreprocessorKimiGrid(CustomTestCase):
    @staticmethod
    def _make_preprocessor(model_type="kimi_vl"):
        preprocessor = EncoderPreprocessor.__new__(EncoderPreprocessor)
        preprocessor.model_type = model_type
        preprocessor.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                vision_config=SimpleNamespace(merge_kernel_size=(2, 2))
            )
        )
        preprocessor.image_processor = SimpleNamespace(merge_size=2)
        preprocessor._model_preprocessor = None
        return preprocessor

    @staticmethod
    def _make_encoder():
        return MMEncoder.__new__(MMEncoder)

    def test_kimi_vl_prefers_and_normalizes_hw_grid(self):
        mm_inputs = {
            "image_grid_hws": np.array([[40, 60]], dtype=np.int64),
            "image_grid_thw": torch.tensor([[1, 20, 30]]),
            "grid_thws": torch.tensor([[1, 10, 15]]),
        }

        grid = self._make_preprocessor()._get_mm_grid_dim(mm_inputs, Modality.IMAGE)

        self.assertIsInstance(grid, torch.Tensor)
        torch.testing.assert_close(grid, torch.tensor([[40, 60]]))

    def test_kimi_k25_keeps_thw_grid_preference(self):
        mm_inputs = {
            "image_grid_hws": np.array([[40, 60]], dtype=np.int64),
            "grid_thws": np.array([[1, 10, 15]], dtype=np.int64),
        }

        grid = self._make_preprocessor("kimi_k25")._get_mm_grid_dim(
            mm_inputs, Modality.IMAGE
        )

        torch.testing.assert_close(grid, torch.tensor([[1, 10, 15]]))

    def test_kimi_vl_2d_grid_counting_and_slicing(self):
        preprocessor = self._make_preprocessor()
        encoder = self._make_encoder()
        grids = torch.tensor([[40, 60], [20, 40]])
        embedding = torch.arange(800 * 2).reshape(800, 2)

        self.assertEqual(
            preprocessor.get_num_patches(grids[0], Modality.IMAGE),
            2400,
        )
        self.assertEqual(
            preprocessor.get_num_tokens(grids[0], Modality.IMAGE),
            600,
        )

        slices = encoder.slice_embedding(embedding, [600, 200])

        self.assertEqual([item.shape for item in slices], [(600, 2), (200, 2)])
        torch.testing.assert_close(slices[0], embedding[:600])
        torch.testing.assert_close(slices[1], embedding[600:])

    def test_kimi_3d_grid_remains_supported(self):
        preprocessor = self._make_preprocessor()
        grid = torch.tensor([1, 40, 60])

        self.assertEqual(preprocessor.get_num_patches(grid, Modality.IMAGE), 2400)
        self.assertEqual(preprocessor.get_num_tokens(grid, Modality.IMAGE), 600)

    def test_kimi_k25_3d_patch_counting_is_unchanged(self):
        preprocessor = self._make_preprocessor("kimi_k25")
        grid = torch.tensor([2, 12, 16])

        self.assertEqual(preprocessor.get_num_patches(grid, Modality.IMAGE), 384)
        self.assertEqual(preprocessor.get_num_tokens(grid, Modality.IMAGE), 48)

    def test_grid_metadata_is_safe_to_deserialize(self):
        grid = self._make_preprocessor()._get_mm_grid_dim(
            {"image_grid_hws": np.array([[40, 60]], dtype=np.int64)},
            Modality.IMAGE,
        )
        embedding_data = EmbeddingData(
            req_id="test-request",
            num_parts=1,
            part_idx=0,
            grid_dim=grid,
            modality=Modality.IMAGE,
            embedding=torch.zeros((600, 4)),
        )

        restored = safe_pickle_loads(
            pickle.dumps(embedding_data.copy_without_embedding())
        )

        torch.testing.assert_close(restored.grid_dim, torch.tensor([[40, 60]]))


class TestEncoderDPAbandonedRequest(CustomTestCase):
    @staticmethod
    def _make_dispatcher():
        return DPDispatcher(
            dp_size=1,
            dispatch_sockets=[object()],
            release_sockets=[object()],
            result_socket=object(),
            worker_processes=[],
        )

    def test_dispatch_timeout_notifies_worker_to_release(self):
        async def run():
            dispatcher = self._make_dispatcher()
            sent = []
            release_sent = asyncio.Event()

            async def send(socket, payload):
                message = unwrap_from_pickle(payload)
                sent.append((socket, message))
                if message.get("_dp_type") == _DP_RELEASE_AFTER_ENCODE:
                    release_sent.set()

            request = {"req_id": "timed-out", "modality": "image"}
            with (
                patch(
                    "sglang.srt.disaggregation.encoder.runtime.async_sock_send",
                    side_effect=send,
                ),
                patch(
                    "sglang.srt.disaggregation.encoder.runtime.server_module.ENCODER_REQ_TIMEOUT",
                    0.01,
                ),
            ):
                result = await dispatcher.dispatch(request)
                await asyncio.wait_for(release_sent.wait(), timeout=1)

            self.assertEqual(result["_error_type"], "TimeoutError")
            self.assertIs(sent[0][0], dispatcher.dispatch_sockets[0])
            self.assertEqual(sent[0][1], request)
            self.assertIs(sent[1][0], dispatcher.release_sockets[0])
            self.assertEqual(
                sent[1][1],
                {
                    "_dp_type": _DP_RELEASE_AFTER_ENCODE,
                    "req_id": "timed-out",
                },
            )
            self.assertEqual(dispatcher.pending_counts, [0])
            self.assertNotIn("timed-out", dispatcher.req_id_to_rank)

        asyncio.run(run())

    def test_timed_out_preprocess_meta_sends_no_release(self):
        """/preprocess_meta leaves no encode state on the worker; a release for
        its req_id would sit in the worker's abandoned set forever."""

        async def run():
            dispatcher = self._make_dispatcher()
            sent = []

            async def send(_socket, payload):
                sent.append(unwrap_from_pickle(payload))

            request = {"req_id": "meta-timeout", "_dp_type": "preprocess_meta"}
            with (
                get_flags().dsv41_sm90.override(subsystems=frozenset({"epd"})),
                patch(
                    "sglang.srt.disaggregation.encoder.runtime.async_sock_send",
                    side_effect=send,
                ),
                patch(
                    "sglang.srt.disaggregation.encoder.runtime.server_module.ENCODER_REQ_TIMEOUT",
                    0.01,
                ),
            ):
                result = await dispatcher.dispatch(request)
                await asyncio.sleep(0.05)

            self.assertEqual(result["_error_type"], "TimeoutError")
            self.assertEqual(result["_dp_type"], "preprocess_meta")
            self.assertEqual(sent, [request])
            self.assertEqual(dispatcher.pending_counts, [0])

        asyncio.run(run())

    def test_dispatch_cancellation_notifies_worker_to_release(self):
        async def run():
            dispatcher = self._make_dispatcher()
            encode_sent = asyncio.Event()
            release_sent = asyncio.Event()

            async def send(_socket, payload):
                message = unwrap_from_pickle(payload)
                if message.get("_dp_type") == _DP_RELEASE_AFTER_ENCODE:
                    release_sent.set()
                else:
                    encode_sent.set()

            with patch(
                "sglang.srt.disaggregation.encoder.runtime.async_sock_send",
                side_effect=send,
            ):
                task = asyncio.create_task(
                    dispatcher.dispatch({"req_id": "cancelled", "modality": "image"})
                )
                await encode_sent.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.wait_for(release_sent.wait(), timeout=1)

            self.assertEqual(dispatcher.pending_counts, [0])
            self.assertNotIn("cancelled", dispatcher.req_id_to_rank)

        asyncio.run(run())

    def test_worker_marks_running_encode_abandoned(self):
        async def run():
            async def encode():
                await asyncio.Event().wait()

            encode_task = asyncio.create_task(encode())
            encoder = SimpleNamespace(
                abandon_request=AsyncMock(),
                release_request=AsyncMock(),
            )
            await _retire_abandoned_encode(encoder, encode_task, "abandoned")

            encoder.abandon_request.assert_awaited_once_with("abandoned")
            encoder.release_request.assert_not_awaited()
            encode_task.cancel()
            await asyncio.gather(encode_task, return_exceptions=True)

        asyncio.run(run())

    def test_worker_preserves_release_before_encode_task_exists(self):
        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 0
            encoder.req_states = {}
            encoder.abandoned_req_ids = set()
            encoder.delivery = SimpleNamespace(release=AsyncMock())

            await _retire_abandoned_encode(encoder, None, "abandoned")
            self.assertIn("abandoned", encoder.abandoned_req_ids)

            state = encoder._acquire_encode_ref("abandoned")
            self.assertTrue(state.release_requested)
            with patch.object(meta_registry, "discard", AsyncMock()):
                await encoder._release_encode_ref(state)

            encoder.delivery.release.assert_awaited_once_with(state)
            self.assertNotIn("abandoned", encoder.req_states)

        asyncio.run(run())

    def test_worker_release_survives_encode_failure(self):
        async def run():
            async def encode():
                raise RuntimeError("bad image")

            encode_task = asyncio.create_task(encode())
            await asyncio.sleep(0)
            encoder = SimpleNamespace(
                abandon_request=AsyncMock(),
                release_request=AsyncMock(),
            )
            await _retire_abandoned_encode(
                encoder,
                encode_task,
                "failed",
            )

            encoder.release_request.assert_awaited_once_with("failed")
            encoder.abandon_request.assert_not_awaited()
            await asyncio.gather(encode_task, return_exceptions=True)

        asyncio.run(run())


class _StagedEncoder:
    """batch_encode that preprocesses, then runs a gated forward under compute_lock."""

    def __init__(self, *, dsv41_epd: bool):
        self.dsv41_epd = dsv41_epd
        self.encode_dispatch_lock = asyncio.Lock()
        self.events = []
        self.forward_started = asyncio.Event()
        self.finish_forward = asyncio.Event()

    async def batch_encode(self, requests, _modality, compute_lock=None):
        req_id = requests[0]["req_id"]
        self.events.append(("preprocess", req_id))
        await asyncio.sleep(0)
        async with compute_lock or contextlib.nullcontext():
            self.events.append(("forward", req_id))
            self.forward_started.set()
            await self.finish_forward.wait()
            self.events.append(("forward_done", req_id))
        return [(1, 2, 3, None, None)]


def _image_request(req_id):
    return {
        "req_id": req_id,
        "modality": "image",
        "mm_items": [object()],
        "num_parts": 1,
        "part_idx": 0,
    }


class TestEncoderSchedulerPipelining(CustomTestCase):
    """With the V4.1 switch and no TP followers, the next batch preprocesses
    while the previous one runs the forward (1E c16 3.70 -> 5.80 img/s on H20)."""

    async def _run_two(self, scheduler, encoder):
        scheduler.start()
        try:
            first = asyncio.create_task(scheduler.submit(_image_request("a")))
            await asyncio.wait_for(encoder.forward_started.wait(), timeout=1)
            second = asyncio.create_task(scheduler.submit(_image_request("b")))
            for _ in range(10):
                await asyncio.sleep(0)
            overlapped = ("preprocess", "b") in encoder.events
            forward_overlapped = ("forward", "b") in encoder.events
            encoder.finish_forward.set()
            await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        finally:
            await scheduler.stop()
        return overlapped, forward_overlapped

    def test_completion_time_advances_only_after_successful_staging(self):
        """Failed embedding publication must not refresh the health fast path."""
        encoder = MMEncoder.__new__(MMEncoder)
        encoder.dsv41_epd, encoder.rank = True, 0
        ctx = SimpleNamespace(
            req_id="request",
            items_per_req=[1],
            preprocess_result=SimpleNamespace(token_counts=[2], grid_thw=[[1, 1, 2]]),
            aux_data={},
            modality=Modality.IMAGE,
        )
        requests = [{"req_id": "request", "num_parts": 1, "part_idx": 0}]
        embedding = torch.zeros(2, 4)
        with (
            patch.object(encoder, "_stage_embedding_batch") as stage,
            patch.object(
                encoder_server, "time", SimpleNamespace(monotonic=lambda: 100.0)
            ),
        ):
            self.assertIsNone(encoder.last_embedding_completed_at)
            result = encoder._stage_embeddings(
                ctx, requests, embedding, keep_on_gpu=False
            )
            self.assertEqual(result, [(32, 2, 4, None, None)])
            self.assertEqual(encoder.last_embedding_completed_at, 100.0)
            stage.side_effect = RuntimeError("stage failed")
            encoder.last_embedding_completed_at = 50.0
            with self.assertRaisesRegex(RuntimeError, "stage failed"):
                encoder._stage_embeddings(ctx, requests, embedding, keep_on_gpu=False)
            self.assertEqual(encoder.last_embedding_completed_at, 50.0)

    def test_next_batch_preprocesses_while_previous_forward_runs(self):
        async def run():
            encoder = _StagedEncoder(dsv41_epd=True)
            scheduler = EncoderScheduler(
                encoder=encoder, send_sockets=[], max_batch_size=1
            )
            overlapped, forward_overlapped = await self._run_two(scheduler, encoder)
            self.assertTrue(overlapped)
            self.assertFalse(forward_overlapped)
            forwards = [e for e in encoder.events if e[0] != "preprocess"]
            self.assertEqual(
                forwards,
                [
                    ("forward", "a"),
                    ("forward_done", "a"),
                    ("forward", "b"),
                    ("forward_done", "b"),
                ],
            )

        asyncio.run(run())

    def test_whole_batches_stay_serial_with_tp_followers_or_switch_off(self):
        """TP followers replay the broadcast order; switch off is upstream."""

        async def run(dsv41_epd, send_sockets):
            encoder = _StagedEncoder(dsv41_epd=dsv41_epd)
            scheduler = EncoderScheduler(
                encoder=encoder, send_sockets=send_sockets, max_batch_size=1
            )
            with patch("sglang.srt.disaggregation.encoder.runtime.sock_send"):
                overlapped, _ = await self._run_two(scheduler, encoder)
            self.assertFalse(scheduler.pipelined)
            self.assertFalse(overlapped)

        for dsv41_epd, send_sockets in ((True, [object()]), (False, [])):
            with self.subTest(dsv41_epd=dsv41_epd, followers=len(send_sockets)):
                asyncio.run(run(dsv41_epd, send_sockets))

    def test_busy_encoder_answers_health_without_waiting_for_dispatch_lock(self):
        """A pipelined encoder holds the dispatch lock for whole forwards; the
        bootstrap probe times out after 2 s and evicts it if /health waits."""

        async def run():
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.dsv41_epd = True
            encoder.last_embedding_completed_at = http_server.time.monotonic()
            encoder.encode_dispatch_lock = asyncio.Lock()
            encoder.req_states = {"real-request": object()}
            encoder.supports_modality = lambda modality: modality == Modality.IMAGE
            await encoder.encode_dispatch_lock.acquire()
            with (
                patch.object(http_server, "encoder", encoder),
                patch.object(http_server, "dp_dispatcher", None),
            ):
                response = await asyncio.wait_for(
                    http_server.health_generate(), timeout=1
                )
            self.assertEqual(response.status_code, 200)

        asyncio.run(run())

    def test_pending_without_recent_completion_waits_for_upstream_health_check(self):
        """A wedged forward with pending work must not get the fast health 200."""

        async def run(last_completed):
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.dsv41_epd = True
            encoder.last_embedding_completed_at = last_completed
            encoder.encode_dispatch_lock = asyncio.Lock()
            encoder.req_states = {"real-request": object()}
            encoder.supports_modality = lambda modality: modality == Modality.IMAGE
            await encoder.encode_dispatch_lock.acquire()
            with (
                patch.object(http_server, "encoder", encoder),
                patch.object(http_server, "dp_dispatcher", None),
                patch.object(
                    http_server, "time", SimpleNamespace(monotonic=lambda: 1000.0)
                ),
            ):
                task = asyncio.create_task(http_server.health_generate())
                try:
                    await asyncio.sleep(0)
                    self.assertFalse(task.done())
                finally:
                    encoder.encode_dispatch_lock.release()
                    await asyncio.wait_for(task, timeout=1)

        for last_completed in (None, 1000.0 - http_server.HEALTH_CHECK_TIMEOUT):
            with self.subTest(last_completed=last_completed):
                asyncio.run(run(last_completed))


class _StopRefresh(Exception):
    pass


class TestEncoderRegistrationRefresh(CustomTestCase):
    def test_encoder_registers_again_with_a_restarted_bootstrap(self):
        """Registration lives in the bootstrap's memory: a prefill that
        restarted listed no encoder until this encoder came back."""
        posts = []
        listed = []

        def post(url, json, timeout):
            posts.append(url)
            return SimpleNamespace(status_code=200, text="")

        def get(url, timeout):
            if url.endswith("/list_encoder_urls"):
                listed.append(url)
                return SimpleNamespace(
                    status_code=200, json=lambda: {"encoder_urls": []}
                )
            return SimpleNamespace(status_code=200)  # the encoder's /health

        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) > 1:
                raise _StopRefresh

        def run_inline(target, daemon, name):
            return SimpleNamespace(start=target)

        with (
            get_context().override_server_args(
                host="10.0.0.1",
                port=8288,
                encoder_register_urls=["http://boot:8997"],
            ),
            get_flags().dsv41_sm90.override(subsystems=frozenset({"epd"})),
            patch.object(http_server.http_requests, "post", side_effect=post),
            patch.object(http_server.http_requests, "get", side_effect=get),
            patch.object(http_server.time, "sleep", side_effect=sleep),
            patch.object(http_server.threading, "Thread", side_effect=run_inline),
        ):
            with self.assertRaises(_StopRefresh):
                http_server._register_encoder_url_with_bootstrap()

        # The startup registration, then one refresh after the bootstrap
        # lost it; the refresh runs on a fixed interval.
        register = "http://boot:8997/register_encoder_url"
        self.assertEqual(posts, [register, register])
        self.assertEqual(listed, ["http://boot:8997/list_encoder_urls"])
        self.assertEqual(sleeps, [http_server._REGISTER_REFRESH_INTERVAL_S] * 2)


class TestMooncakeRegistration(CustomTestCase):
    def setUp(self):
        self.engine = MooncakeTransferEngine.__new__(MooncakeTransferEngine)
        self.engine.engine = unittest.mock.Mock()

    def test_register_raises_on_nonzero_status(self):
        self.engine.engine.register_memory.return_value = -1

        with self.assertRaisesRegex(RuntimeError, "registration failed.*ret=-1"):
            self.engine.register(1234, 4096)

    def test_deregister_preserves_backend_failure(self):
        backend_error = OSError("backend failed")
        self.engine.engine.unregister_memory.side_effect = backend_error

        with self.assertRaisesRegex(RuntimeError, "deregistration failed") as ctx:
            self.engine.deregister(1234)

        self.assertIs(ctx.exception.__cause__, backend_error)


if __name__ == "__main__":
    unittest.main()
