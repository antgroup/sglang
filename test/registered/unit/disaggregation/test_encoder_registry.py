"""Encoder discovery for a language-only server with several tokenizer workers.

With --tokenizer-worker-num 4 on P and 8 on D in a 2E3P1D setup on H20,
per-worker encoder bootstrap servers compete for one port. Only one can bind it;
the others log "address already in use", and encoders register with one worker.
Other workers find no encoder and return 503 for multimodal requests.
Under the DeepSeek-V4.1 SM90 ``epd`` switch, the router owns the registry;
it pushes versioned snapshots to every worker.
"""

import asyncio
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import zmq
from fastapi.testclient import TestClient

from sglang.srt.disaggregation.encoder.receiver import EncoderBootstrapServer
from sglang.srt.managers.io_struct import (
    EncoderRegistryUpdateReq,
    GenerateReqInput,
    sock_recv,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiTokenizerRouter,
    SocketMapping,
    TokenizerWorker,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _worker():
    worker = TokenizerWorker.__new__(TokenizerWorker)
    worker.encoder_urls = []
    worker.encoder_registry_ready = asyncio.Event()
    worker._encoder_registry_version = -1
    worker._epd_should_dispatch = lambda obj: True
    worker.mm_receiver = SimpleNamespace(encode_urls=worker.encoder_urls)
    return worker


def _registry(**kwargs):
    # The real HTTP handlers and health loop, without binding a port.
    with patch.object(EncoderBootstrapServer, "_run_server"):
        registry = EncoderBootstrapServer(
            "127.0.0.1", 0, health_check_interval=0, **kwargs
        )
        registry.thread.join()
    return registry


class TestEncoderRegistry(CustomTestCase):
    def test_registration_reaches_every_worker_and_a_late_joiner(self):
        """An encoder registered over HTTP with the router must reach every
        worker's receiver list, and a worker that joins later must get the
        current set."""
        router = MultiTokenizerRouter.__new__(MultiTokenizerRouter)
        router.socket_mapping = SocketMapping()
        router.all_worker_ipcs = set()
        router._loop = asyncio.new_event_loop()
        thread = threading.Thread(target=router._loop.run_forever, daemon=True)
        thread.start()
        router.encoder_bootstrap_server = _registry(
            urls=["http://static-e"], on_change=router._on_encoder_registry_change
        )
        registry = router.encoder_bootstrap_server
        context = zmq.Context()
        receivers, workers = [], []
        with tempfile.TemporaryDirectory() as tmp:
            try:

                def join(index):
                    ipc = f"ipc://{tmp}/worker-{index}"
                    recv = context.socket(zmq.PULL)
                    recv.setsockopt(zmq.RCVTIMEO, 2000)
                    recv.bind(ipc)
                    receivers.append(recv)
                    worker = _worker()
                    workers.append(worker)

                    def register_worker():
                        router.all_worker_ipcs.add(ipc)
                        router._send_encoder_registry(ipc)

                    router._loop.call_soon_threadsafe(register_worker)
                    worker._handle_encoder_registry_update(sock_recv(recv))
                    return worker

                def drain(expected):
                    for worker, recv in zip(workers, receivers):
                        worker._handle_encoder_registry_update(sock_recv(recv))
                        self.assertEqual(worker.mm_receiver.encode_urls, expected)

                join(0)
                join(1)
                with TestClient(registry.app) as client:
                    client.post(
                        "/register_encoder_url", json={"url": "http://dynamic-e"}
                    ).raise_for_status()
                    drain(["http://static-e", "http://dynamic-e"])
                    self.assertEqual(
                        join(2).mm_receiver.encode_urls, registry.list_urls()
                    )
                    client.request(
                        "DELETE",
                        "/unregister_encoder_url",
                        json={"url": "http://static-e"},
                    ).raise_for_status()
                    drain(["http://dynamic-e"])
            finally:
                for recv in receivers:
                    recv.close(linger=0)
                router._loop.call_soon_threadsafe(
                    router.socket_mapping.clear_all_sockets
                )
                router._loop.call_soon_threadsafe(router._loop.stop)
                thread.join(timeout=3)
                router._loop.close()
                router.socket_mapping._zmq_context.term()
                context.term()

    def test_worker_binds_no_registration_port_under_the_switch(self):
        """Under ``epd`` a tokenizer worker leaves the registration port to the
        router; without it the upstream per-process server stays."""
        cfg = SimpleNamespace(
            language_only=True,
            encoder_urls=["http://static-e"],
            encoder_bootstrap_port=0,
            disaggregation_mode="decode",
        )
        for epd in (False, True):
            with self.subTest(epd=epd):
                manager = TokenizerManager.__new__(TokenizerManager)
                manager.server_args = object()
                manager.model_config = SimpleNamespace(
                    dtype="float16", hf_config=object()
                )
                with (
                    get_flags().dsv41_sm90.override(
                        subsystems=frozenset({"epd"} if epd else ())
                    ),
                    patch(
                        "sglang.srt.managers.tokenizer_manager.get_disagg",
                        return_value=cfg,
                    ),
                    patch(
                        "sglang.srt.managers.tokenizer_manager.get_serving",
                        return_value=SimpleNamespace(host="127.0.0.1"),
                    ),
                    patch("sglang.srt.managers.tokenizer_manager.start_disagg_service"),
                    patch(
                        "sglang.srt.disaggregation.encoder.receiver.EncoderBootstrapServer"
                    ) as encoder_bootstrap,
                    patch(
                        "sglang.srt.managers.tokenizer_manager.create_mm_receiver"
                    ) as receiver,
                    patch(
                        "sglang.srt.managers.tokenizer_manager.epd_decode_uses_meta_only",
                        return_value=True,
                    ),
                ):
                    manager.init_disaggregation(start_pd_bootstrap_service=False)
                self.assertIs(
                    receiver.call_args.kwargs["encode_urls"], manager.encoder_urls
                )
                if epd:
                    encoder_bootstrap.assert_not_called()
                    self.assertFalse(manager.encoder_registry_ready.is_set())
                else:
                    encoder_bootstrap.assert_called_once()
                    self.assertIsNone(manager.encoder_registry_ready)

    def test_out_of_order_snapshot_cannot_restore_an_unregistered_encoder(self):
        worker = _worker()
        for version, urls in ((2, []), (1, ["http://old-e"]), (2, ["http://old-e"])):
            worker._handle_encoder_registry_update(
                EncoderRegistryUpdateReq(version=version, urls=urls)
            )
        self.assertEqual(worker.mm_receiver.encode_urls, [])
        self.assertTrue(worker.encoder_registry_ready.is_set())

    def test_worker_subscribes_before_its_first_request(self):
        """An idle worker must apply the router's snapshot without a request
        having started its receive loop."""
        from sglang.utils import TypeBasedDispatcher

        for epd in (False, True):
            with self.subTest(epd=epd):

                def base_init(worker, *args, **kwargs):
                    worker.encoder_urls = []
                    worker.encoder_registry_ready = asyncio.Event() if epd else None
                    worker._result_dispatcher = TypeBasedDispatcher([])

                def start(worker):
                    worker._result_dispatcher(
                        EncoderRegistryUpdateReq(version=1, urls=["http://e"])
                    )

                with (
                    patch.object(TokenizerManager, "__init__", base_init),
                    patch.object(TokenizerWorker, "_dispatch_to_scheduler"),
                    patch.object(
                        TokenizerWorker,
                        "auto_create_handle_loop",
                        autospec=True,
                        side_effect=start,
                    ) as receive,
                    patch(
                        "sglang.srt.managers.multi_tokenizer_mixin.get_disagg",
                        return_value=SimpleNamespace(
                            disaggregation_transfer_backend="mooncake"
                        ),
                    ),
                    patch(
                        "sglang.srt.managers.multi_tokenizer_mixin.setproctitle.setproctitle"
                    ),
                    patch("torch.set_num_threads"),
                ):
                    worker = TokenizerWorker(
                        object(), SimpleNamespace(tokenizer_ipc_name="ipc://worker")
                    )
                if epd:
                    receive.assert_called_once_with(worker)
                    self.assertEqual(worker.encoder_urls, ["http://e"])
                else:
                    receive.assert_not_called()

    def test_first_multimodal_request_waits_for_the_snapshot_text_does_not(self):
        async def run():
            worker = _worker()
            await worker._wait_for_encoder_registry(GenerateReqInput(text="hello"))
            request = GenerateReqInput(text="read", image_data=["http://image"])
            waiting = asyncio.create_task(worker._wait_for_encoder_registry(request))
            await asyncio.sleep(0)
            self.assertFalse(waiting.done())
            worker._handle_encoder_registry_update(
                EncoderRegistryUpdateReq(version=0, urls=["http://e"])
            )
            await asyncio.wait_for(waiting, timeout=1)

        asyncio.run(run())

    def test_health_changes_publish_snapshots_and_unregister_wins_a_probe(self):
        updates = []
        registry = _registry(urls=["http://e"], on_change=lambda *s: updates.append(s))
        registry._health_fail_threshold = 1
        registry._probe = AsyncMock(side_effect=[False, True])

        async def run_until(n, settle=0):
            task = asyncio.create_task(registry._health_check_loop())
            try:

                async def wait():
                    while len(updates) < n:
                        await asyncio.sleep(0)
                    # Let an in-flight probe result get applied.
                    for _ in range(settle):
                        await asyncio.sleep(0)

                await asyncio.wait_for(wait(), timeout=2)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        asyncio.run(run_until(2))
        self.assertEqual(updates, [(1, []), (2, ["http://e"])])

        # An unregister while a failing probe is in flight must not let the
        # probe's result evict the URL into the revival set.
        updates.clear()

        async def probe(*args):
            registry.unregister("http://e")
            return False

        registry._probe = probe
        asyncio.run(run_until(1, settle=5))
        self.assertEqual(registry.list_urls(), [])
        self.assertEqual(registry._evicted_urls, {})
        self.assertEqual(updates, [(3, [])])


if __name__ == "__main__":
    unittest.main()
