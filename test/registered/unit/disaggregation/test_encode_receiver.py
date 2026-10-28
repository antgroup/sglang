"""Unit tests for the encode-disaggregation receiver."""

import asyncio
import threading
import time
import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock, patch

import aiohttp
import fastapi
import torch

from sglang.srt.disaggregation.encoder import receiver as receiver_module
from sglang.srt.disaggregation.encoder.receiver import (
    EmbeddingPool,
    MMReceiverBase,
    WaitingListSync,
    WaitingMMRequestStatus,
    WaitingRDMARequest,
    WaitingZmqRequest,
    _select_mm_processor_prompt,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestSelectMMProcessorPrompt(CustomTestCase):
    def test_tokens_in_prompt_is_list(self):
        """A tokens-in prompt reaches the processor as list[int], not array("q");
        HTTP sends input_text=None and the gRPC servicer sends ""."""
        processor = SimpleNamespace(prefer_tokenized_input=False)
        for input_text in (None, ""):
            with self.subTest(input_text=input_text):
                recv_req = SimpleNamespace(
                    input_text=input_text, input_ids=array("q", [1, 2, 3])
                )

                prompt = _select_mm_processor_prompt(recv_req, processor)

                self.assertIsInstance(prompt, list)
                self.assertEqual(prompt, [1, 2, 3])


class _RecordingGroup:
    def __init__(self, name, world_size, peer_values=None):
        self.name = name
        self.world_size = world_size
        self.cpu_group = name
        self.peer_values = peer_values

    def all_gather_object(self, values):
        return [values] + [self.peer_values] * (self.world_size - 1)


class _CompletedRequest:
    def __init__(self, rid):
        self.rid = rid
        self.recv_req = SimpleNamespace(rid=rid, encoder_req_id=None)
        self.status = WaitingMMRequestStatus.SUCCESS
        self.start_time = time.time()

    def _try_recv_mm_data(self):
        pass

    def close_recv_socket(self):
        pass


def _scheduler_rank(waiting_list, waiting_list_sync):
    return SimpleNamespace(
        waiting_list=waiting_list,
        waiting_by_rid={request.rid: request for request in waiting_list},
        wait_timeout=float("inf"),
        waiting_list_sync=waiting_list_sync,
        _drain_scheduler_embeddings=lambda: None,
        _sync_fail_info_across_tp=lambda request: None,
        create_req=lambda recv_req: recv_req,
    )


class TestDpAttentionWaitingList(CustomTestCase):
    def test_dp_ranks_with_their_own_requests_sync_only_their_shard(self):
        """With DP attention each DP rank of a TP group holds different
        requests; a status all-reduce over the whole TP group got tensors of
        different lengths from its ranks and aborted gloo, and the encoder
        waited for registrations from every TP rank."""
        calls = []

        def record(tensor, op=None, group=None):
            calls.append((group, tensor.numel()))

        # TP4 x DP4 with attn_tp = attn_cp = 1: every waiting list is one rank's.
        parallel = SimpleNamespace(
            attn_dp_enabled=True,
            attn_tp_group=_RecordingGroup("attn_tp", 1),
            attn_cp_group=_RecordingGroup("attn_cp", 1),
            tp_group=_RecordingGroup("full_tp", 4),
        )
        busy_sync = WaitingListSync.for_scheduler(parallel)
        self.assertEqual(busy_sync.size, 1)
        busy = _scheduler_rank([_CompletedRequest("a")], busy_sync)
        idle = _scheduler_rank([], WaitingListSync.for_scheduler(parallel))
        with patch("torch.distributed.all_reduce", record):
            ready_busy, _ = MMReceiverBase._process_waiting_requests(
                busy, [], waiting_cls=None
            )
            ready_idle, _ = MMReceiverBase._process_waiting_requests(
                idle, [], waiting_cls=None
            )

        self.assertEqual(calls, [])
        self.assertEqual([req.rid for req in ready_busy], ["a"])
        self.assertEqual(ready_idle, [])
        # Without DP attention the whole TP group shares one waiting list.
        parallel.attn_dp_enabled = False
        self.assertEqual(WaitingListSync.for_scheduler(parallel).size, 4)

    def test_shard_collectives_run_along_attn_tp_then_attn_cp(self):
        calls = []

        def record(tensor, op=None, group=None):
            calls.append(group)

        shard = WaitingListSync(
            (
                _RecordingGroup("attn_tp", 2, peer_values=["tp-peer"]),
                _RecordingGroup("attn_cp", 2, peer_values=["cp-a", "cp-b"]),
            )
        )
        rank = _scheduler_rank([_CompletedRequest("a")], shard)
        with patch("torch.distributed.all_reduce", record):
            MMReceiverBase._process_waiting_requests(rank, [], waiting_cls=None)

        self.assertEqual(calls, ["attn_tp", "attn_cp"])
        self.assertEqual(shard.size, 4)
        self.assertEqual(
            shard.all_gather_object("me"), ["me", "tp-peer", "cp-a", "cp-b"]
        )


class _PendingRequest(_CompletedRequest):
    def __init__(self, rid):
        super().__init__(rid)
        self.status = WaitingMMRequestStatus.PENDING
        self.start_time = 0.0

    def release_resources(self):
        pass


class TestWaitingRequestFailures(CustomTestCase):
    def test_embedding_wait_timeout_aborts_with_gateway_timeout(self):
        """The prefill gave up on its encoder: a server-side upstream timeout is
        504; 408 tells the client it sent the request too slowly."""
        rank = _scheduler_rank([_PendingRequest("a")], WaitingListSync())
        rank.wait_timeout = 1.0
        _, abort_reqs = MMReceiverBase._process_waiting_requests(
            rank, [], waiting_cls=None
        )
        self.assertEqual(
            [code for *_, code in abort_reqs], [HTTPStatus.GATEWAY_TIMEOUT]
        )

    def test_failed_staging_copy_releases_the_unpublished_slot(self):
        """The caller only learns the slot id on success, so a failed copy
        leaked the pool slot for good."""
        slot_view = Mock()
        slot_view.__getitem__ = Mock(
            return_value=SimpleNamespace(copy_=Mock(side_effect=RuntimeError("copy")))
        )
        pool = SimpleNamespace(
            try_alloc=Mock(return_value=(slot_view, 4096, 7)), release=Mock()
        )
        with self.assertRaisesRegex(RuntimeError, "copy"):
            EmbeddingPool.try_stage(pool, [torch.zeros(4)])
        pool.release.assert_called_once_with(7)

    def test_terminal_request_stops_its_metadata_pull(self):
        """The encoder holds /scheduler_receive_meta_data until the part
        publishes (up to its request timeout); an aborted request must not
        keep its receive thread parked there."""
        request = WaitingRDMARequest.__new__(WaitingRDMARequest)
        request.rid = "rdma-1"
        request.encoder_urls = ["http://encoder"]
        request._buffer_lock = threading.Lock()
        request._terminal = False
        request._receive_running = True
        request._cancel_meta_pull = None
        request.embeddings_buffer = None
        request._pool_slot_id = None
        in_flight = threading.Event()
        outcome = []

        class StalledSession:
            async def post(self, url, json):
                in_flight.set()
                await asyncio.Event().wait()

        def pull():
            try:
                asyncio.run(
                    request._pull_meta_until_terminal(
                        StalledSession(),
                        [{"encoder_idx": 0, "req_id": "r", "part_idx": 0}],
                    )
                )
            except asyncio.CancelledError:
                outcome.append("cancelled")

        thread = threading.Thread(target=pull, daemon=True)
        thread.start()
        self.assertTrue(in_flight.wait(timeout=5))
        with patch.object(WaitingRDMARequest, "_release_buffer_locked"):
            request._cleanup_gpu_buffer()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcome, ["cancelled"])
        self.assertIsNone(request._cancel_meta_pull)


class TestEncoderDispatchIdentity(CustomTestCase):
    def test_each_dispatch_has_its_own_wire_id_and_errors_keep_both(self):
        """P and D (decode's metadata fallback) encode one client request, and
        a client may reuse an rid: the encoder keyed both by the rid."""
        receiver = SimpleNamespace(
            encode_urls=["http://e0"],
            _extract_url_data=lambda obj: ["image"],
            _assign_items_by_modality=lambda mm_data, n: {"image": [1]},
            _run_encode_in_thread=None,
        )
        started = []

        def fake_thread(target, args, daemon):
            started.append(args)
            return SimpleNamespace(start=lambda: None)

        objs = [SimpleNamespace(rid="client-rid") for _ in range(2)]
        with patch("threading.Thread", side_effect=fake_thread):
            for obj in objs:
                MMReceiverBase._send_encode_request(receiver, obj)
        wire_ids = [obj.encoder_req_id for obj in objs]
        self.assertNotEqual(wire_ids[0], wire_ids[1])
        self.assertEqual([args[0] for args in started], wire_ids)
        self.assertEqual([args[-1] for args in started], ["client-rid"] * 2)

        # A failed dispatch reports the client rid and its own wire id.
        error = SimpleNamespace(rid=wire_ids[0], encoder_req_id=None)
        delivered = []

        async def failing_encode(**kwargs):
            return error

        receiver.encode = failing_encode
        ready = threading.Event()
        ready.set()
        MMReceiverBase._run_encode_in_thread(
            receiver,
            wire_ids[0],
            ["image"],
            "encode",
            {"image": [1]},
            scheduler_dispatch_ready=ready,
            on_dispatch_error=delivered.append,
            logical_req_id="client-rid",
        )
        self.assertEqual(delivered, [error])
        self.assertEqual((error.rid, error.encoder_req_id), ("client-rid", wire_ids[0]))

    def test_pool_slot_outlives_the_processor_output(self):
        """The scheduler replaces the processor output with MultimodalInputs;
        a pool slot bound to the output was freed under live tensor views."""
        waiting = WaitingZmqRequest.__new__(WaitingZmqRequest)
        waiting.recv_req = SimpleNamespace(rid="r", input_ids=[1], input_text=None)
        processor_output = SimpleNamespace(input_ids=[1, 2])
        waiting.mm_processor = SimpleNamespace(
            get_validated_mm_data=lambda *args, **kwargs: processor_output
        )
        waiting.recv_embedding_data = SimpleNamespace(get_mm_extra_meta=lambda: {})
        waiting._pool_slot_id = 3
        waiting.embedding_pool = object()
        bound = []
        final = object()
        waiting._bind_pool_slot_to_mm_inputs = bound.append
        with (
            patch(
                "sglang.srt.disaggregation.encoder.receiver._select_mm_processor_prompt",
                return_value=[1],
            ),
            patch(
                "sglang.srt.disaggregation.encoder.receiver.MultimodalInputs.from_processor_output",
                return_value=final,
            ) as materialize,
        ):
            waiting._finish_assemble(recv_embedding=None)
        materialize.assert_called_once_with(processor_output)
        self.assertEqual(bound, [final])
        self.assertIs(waiting.recv_req.mm_inputs, final)
        self.assertEqual(list(waiting.recv_req.input_ids), [1, 2])

    def test_v41_image_request_without_encoder_is_rejected(self):
        """With every encoder evicted (a saturated encoder misses its health
        probes), the request fell back to local vision; a V4.1 language-only
        prefill has no tower, and the scheduler crash took the server down."""
        manager = SimpleNamespace(
            enable_trace=False,
            encoder_dispatch_ready={},
            mm_receiver=SimpleNamespace(
                dsv41_epd=True, send_encode_request=lambda obj, **kwargs: None
            ),
            _schedule_encoder_dispatch_error=None,
        )
        obj = SimpleNamespace(rid="r", need_wait_for_mm_inputs=None)
        with get_context().override_server_args(
            encoder_transfer_backend="zmq_to_scheduler"
        ):
            with self.assertRaises(fastapi.HTTPException) as caught:
                TokenizerManager._dispatch_epd_encode(manager, obj)
            self.assertEqual(caught.exception.status_code, 503)
            # Switch off: upstream's local fallback stays.
            manager.mm_receiver.dsv41_epd = False
            TokenizerManager._dispatch_epd_encode(manager, obj)
        self.assertEqual(manager.encoder_dispatch_ready, {})


class _ConnectFlakySession:
    """aiohttp-like session whose first ``failures`` posts fail to connect."""

    def __init__(self, failures: int):
        self.failures = failures
        self.posts = 0

    async def post(self, url, json):
        self.posts += 1
        if self.posts <= self.failures:
            raise aiohttp.ClientConnectorError(
                SimpleNamespace(host="encoder", port=8087, ssl=True),
                ConnectionRefusedError(111, "Connection refused"),
            )
        return _OkResponse()


class _OkResponse:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    async def text(self):
        return "ok"


class TestReceiveUrlRegistrationRetry(CustomTestCase):
    def test_epd_retries_connect_failure(self):
        """A refused connect to the encoder failed the image request outright;
        under epd the post is retried, as a failed connect sent nothing."""
        session = _ConnectFlakySession(failures=2)
        with (
            patch.object(receiver_module, "dsv41_sm90_on", return_value=True),
            patch.object(receiver_module, "_CONNECT_BACKOFF_S", 0.0),
        ):
            text = asyncio.run(
                receiver_module._post_receive_url(session, "http://e/x", {})
            )
        self.assertEqual((text, session.posts), ("ok", 3))

    def test_switch_off_keeps_single_attempt(self):
        """Off the switch the registration stays upstream's single post."""
        session = _ConnectFlakySession(failures=1)
        with patch.object(receiver_module, "dsv41_sm90_on", return_value=False):
            with self.assertRaises(aiohttp.ClientConnectorError):
                asyncio.run(
                    receiver_module._post_receive_url(session, "http://e/x", {})
                )
        self.assertEqual(session.posts, 1)


if __name__ == "__main__":
    unittest.main()
