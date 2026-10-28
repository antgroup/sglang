"""A PD prefill that retires a request before the KV handoff must fail the
room on decode at once, with the prefill's reason.

Otherwise decode never hears about a request that prefill aborted before
creating its KV sender (an EPD encoder failure, an intake reject, a KV capacity
reject): it waits for KV until SGLANG_DISAGGREGATION_WAITING_TIMEOUT and the
client gets an error ~300 s later. Decode can register its destination before
or after prefill gives up, so both orders are covered.
"""

import threading
import time
import unittest
from collections import OrderedDict, defaultdict
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.nixl.conn import GUARD as NIXL_GUARD
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROOM = 6359277859203974196
REASON = "Timeout waiting for image embedding after 180.0s"
DECODE_IP, DECODE_PORT = "10.0.0.2", 23456
DECODE_TCP = f"tcp://{DECODE_IP}:{DECODE_PORT}"


class _StopLoop(Exception):
    pass


def _prefill_manager(bootstrap_timeout=300, manager_cls=MooncakeKVManager):
    mgr = object.__new__(manager_cls)
    mgr.request_status = {}
    mgr.transfer_infos = {}
    mgr.req_to_decode_prefix_len = {}
    mgr._failed_room_tombstones = OrderedDict()
    mgr.bootstrap_timeout = bootstrap_timeout
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr.is_mla_backend = False
    mgr.enable_deferred_decode_kv_release = True
    mgr.enable_staging = False
    mgr._staging_outstanding = defaultdict(int)
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_poisoned_rooms = set()
    mgr.attn_tp_rank = mgr.attn_cp_rank = mgr.pp_rank = 0
    mgr.attn_cp_size = mgr.pp_size = 1
    mgr.server_args = SimpleNamespace(enable_pd_role_switch=False)
    mgr._stopped = False
    mgr._worker_threads = []
    mgr.sent = []
    mgr._send_multipart_locked = lambda endpoint, parts, is_ipv6=False: mgr.sent.append(
        (endpoint, parts)
    )
    return mgr


def _registration(room=ROOM):
    """What decode's MooncakeKVManager.build_receiver_metadata_message sends."""
    return [
        str(room).encode("ascii"),
        DECODE_IP.encode("ascii"),
        str(DECODE_PORT).encode("ascii"),
        b"decode-session",
        np.array([3, 4], dtype=np.int32).tobytes(),
        b"0",
        pack_int_lists([], "i"),
        b"1",
        b"0",
        b"",
    ]


def _abort(room=ROOM, generation=b"7"):
    return [
        b"ABORT",
        str(room).encode("ascii"),
        DECODE_IP.encode("ascii"),
        str(DECODE_PORT).encode("ascii"),
        generation,
    ]


def _run_mooncake_prefill_thread(mgr, messages):
    """Run the mooncake prefill control loop over ``messages`` on this thread."""
    mgr.server_socket = MagicMock()
    mgr.server_socket.recv_multipart.side_effect = [*messages, _StopLoop()]
    with patch("sglang.srt.disaggregation.mooncake.conn.threading.Thread") as thread:
        mgr.start_prefill_thread()
    try:
        thread.call_args.kwargs["target"]()
    except _StopLoop:
        pass


def _run_mooncake_decode_thread(mgr, messages):
    """Run the mooncake decode control loop over ``messages`` on this thread."""
    mgr.server_args = SimpleNamespace(enable_pd_role_switch=False)
    mgr._stopped = False
    mgr._worker_threads = []
    mgr._start_heartbeat_checker_thread = MagicMock()
    mgr.server_socket = MagicMock()
    mgr.server_socket.recv_multipart.side_effect = [*messages, _StopLoop()]
    with patch("sglang.srt.disaggregation.mooncake.conn.threading.Thread") as thread:
        mgr.start_decode_thread()
    try:
        thread.call_args.kwargs["target"]()
    except _StopLoop:
        pass


def _nixl_registration(room=ROOM, agent=b"decode-agent", required=b"1"):
    """What decode's NIXL receiver sends a prefill rank for one room."""
    return [
        NIXL_GUARD,
        str(room).encode("ascii"),
        DECODE_IP.encode("ascii"),
        str(DECODE_PORT).encode("ascii"),
        agent,
        np.array([3, 4], dtype=np.int32).tobytes(),
        b"0",
        required,
    ]


def _run_nixl_prefill_thread(mgr, messages):
    # Read by the thread's crash log, which the end-of-input sentinel triggers.
    mgr.kv_args = SimpleNamespace(engine_rank=0)
    mgr.server_socket = MagicMock()
    mgr.server_socket.recv_multipart.side_effect = [*messages, _StopLoop()]
    with patch("sglang.srt.disaggregation.nixl.conn.threading.Thread") as thread:
        mgr._start_bootstrap_thread()
    try:
        thread.call_args.kwargs["target"]()
    except _StopLoop:
        pass


class _PopAfterFirstLook(dict):
    """A table whose room the scheduler fails right after the control thread
    first looks it up."""

    def __init__(self, *args, on_first_look):
        super().__init__(*args)
        self._on_first_look = on_first_look

    def _looked(self):
        hook, self._on_first_look = self._on_first_look, None
        if hook is not None:
            hook()

    def __contains__(self, key):
        found = super().__contains__(key)
        self._looked()
        return found

    def get(self, key, default=None):
        value = super().get(key, default)
        self._looked()
        return value


def _decode_manager():
    mgr = object.__new__(MooncakeKVManager)
    mgr.request_status = {}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr.required_prefill_response_num_table = {}
    mgr.prefill_response_tracker = {}
    mgr.addr_to_rooms_tracker = defaultdict(set)
    mgr.connection_lock = threading.Lock()
    mgr.connection_pool = {}
    mgr.enable_staging = False
    mgr._staging_outstanding = defaultdict(int)
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_poisoned_rooms = set()
    mgr.waiting_timeout = 300
    return mgr


def _waiting_decode_receiver(room=ROOM):
    """A decode receiver that sent its destination and now waits for KV."""
    mgr = _decode_manager()
    receiver = MooncakeKVReceiver(mgr, "prefill:8998", room)
    mgr.required_prefill_response_num_table[room] = 1
    mgr.update_status(room, KVPoll.WaitingForInput)
    receiver.init_time = time.time()
    return receiver


def _deliver(prefill_mgr, receiver):
    """Feed what prefill sent to decode through decode's control-thread path."""
    for endpoint, parts in prefill_mgr.sent:
        assert endpoint == DECODE_TCP, endpoint
        room, status, prefill_rank, reason = receiver.kv_mgr.parse_kv_status_message(
            parts
        )
        receiver.kv_mgr.apply_prefill_status(
            bootstrap_room=room,
            status=status,
            prefill_rank=prefill_rank,
            failure_reason=reason,
        )


class TestPrefillFailureBeforeHandoff(CustomTestCase):
    def assertDecodeFailedWith(self, receiver, reason):
        self.assertEqual(receiver.poll(), KVPoll.Failed)
        with self.assertRaises(KVTransferError) as ctx:
            receiver.failure_exception()
        self.assertEqual(ctx.exception.failure_reason, reason)

    def assertNoPrefillRoomState(self, mgr, room=ROOM):
        self.assertNotIn(room, mgr.request_status)
        self.assertNotIn(room, mgr.transfer_infos)
        self.assertNotIn(room, mgr.req_to_decode_prefix_len)

    def test_registered_decode_fails_at_once_with_the_prefill_reason(self):
        prefill = _prefill_manager()
        _run_mooncake_prefill_thread(prefill, [_registration()])
        # Registration alone opens the room on prefill.
        self.assertEqual(prefill.request_status[ROOM], KVPoll.WaitingForInput)
        receiver = _waiting_decode_receiver()

        prefill.fail_room_before_handoff(bootstrap_room=ROOM, failure_reason=REASON)

        self.assertNoPrefillRoomState(prefill)
        _deliver(prefill, receiver)
        self.assertDecodeFailedWith(receiver, REASON)

    def test_decode_registering_after_the_failure_is_failed_not_registered(self):
        prefill = _prefill_manager()
        prefill.fail_room_before_handoff(bootstrap_room=ROOM, failure_reason=REASON)
        self.assertEqual(prefill.sent, [])
        receiver = _waiting_decode_receiver()

        _run_mooncake_prefill_thread(prefill, [_registration()])

        self.assertNoPrefillRoomState(prefill)
        self.assertEqual(len(prefill.sent), 1)
        _deliver(prefill, receiver)
        self.assertDecodeFailedWith(receiver, REASON)

    def test_tombstone_expires_with_the_bootstrap_timeout(self):
        prefill = _prefill_manager(bootstrap_timeout=0)
        prefill.fail_room_before_handoff(bootstrap_room=ROOM, failure_reason=REASON)

        _run_mooncake_prefill_thread(prefill, [_registration()])

        # Past the window a live sender would wait, the room registers normally.
        self.assertEqual(prefill._failed_room_tombstones, OrderedDict())
        self.assertEqual(prefill.request_status[ROOM], KVPoll.WaitingForInput)
        self.assertEqual(prefill.sent, [])

    def test_abort_for_a_failed_room_is_acked_at_once(self):
        """Decode holds the room's KV pages until every prefill rank acks."""
        prefill = _prefill_manager()
        _run_mooncake_prefill_thread(prefill, [_registration()])
        prefill.fail_room_before_handoff(bootstrap_room=ROOM, failure_reason=REASON)
        prefill.sent.clear()

        _run_mooncake_prefill_thread(prefill, [_abort()])

        self.assertEqual(
            prefill.sent,
            [(DECODE_TCP, [b"ABORT_ACK", str(ROOM).encode("ascii"), b"0", b"7"])],
        )
        self.assertNoPrefillRoomState(prefill)

    def test_abort_racing_the_room_failure_is_acked(self):
        """The scheduler can fail a room while the control thread handles that
        room's ABORT; the ABORT must still be acked and the loop keep serving."""
        prefill = _prefill_manager()
        _run_mooncake_prefill_thread(prefill, [_registration()])
        prefill.request_status = _PopAfterFirstLook(
            prefill.request_status,
            on_first_look=lambda: prefill.fail_room_before_handoff(
                bootstrap_room=ROOM, failure_reason=REASON
            ),
        )
        prefill.sent.clear()
        other_room = ROOM + 1

        _run_mooncake_prefill_thread(prefill, [_abort(), _registration(other_room)])

        acks = [parts for _, parts in prefill.sent if parts[0] == b"ABORT_ACK"]
        self.assertEqual(acks, [[b"ABORT_ACK", str(ROOM).encode("ascii"), b"0", b"7"]])
        self.assertEqual(prefill.request_status[other_room], KVPoll.WaitingForInput)

    def test_a_failing_message_does_not_stop_the_control_loops(self):
        """Otherwise one bad message stops the rank from serving every later
        room until restart."""
        prefill = _prefill_manager()
        _run_mooncake_prefill_thread(prefill, [[b"7", b"short"], _registration()])
        self.assertEqual(prefill.request_status[ROOM], KVPoll.WaitingForInput)

        receiver = _waiting_decode_receiver()
        failed = prefill._encode_kv_status_message(
            bootstrap_room=ROOM, status=KVPoll.Failed, failure_reason=REASON
        )
        _run_mooncake_decode_thread(receiver.kv_mgr, [[b"STAGING_REQ"], failed])
        self.assertDecodeFailedWith(receiver, REASON)

    def test_nixl_decode_registering_after_the_failure_is_failed(self):
        prefill = _prefill_manager(manager_cls=NixlKVManager)
        prefill.fail_room_before_handoff(bootstrap_room=ROOM, failure_reason=REASON)

        _run_nixl_prefill_thread(prefill, [_nixl_registration()])

        self.assertNoPrefillRoomState(prefill)
        ((endpoint, parts),) = prefill.sent
        self.assertEqual(endpoint, DECODE_TCP)
        self.assertEqual(
            prefill.parse_kv_status_message(parts),
            (ROOM, KVPoll.Failed, 0, REASON),
        )

    def test_nixl_registration_racing_the_room_failure_keeps_the_loop(self):
        """The scheduler can fail a room while the control thread registers a
        second decode destination for it; the thread must keep serving."""
        prefill = _prefill_manager(manager_cls=NixlKVManager)
        first, second = b"decode-agent-0", b"decode-agent-1"
        _run_nixl_prefill_thread(
            prefill, [_nixl_registration(agent=first, required=b"2")]
        )
        prefill.transfer_infos = _PopAfterFirstLook(
            prefill.transfer_infos,
            on_first_look=lambda: prefill.fail_room_before_handoff(
                bootstrap_room=ROOM, failure_reason=REASON
            ),
        )
        other_room = ROOM + 1

        _run_nixl_prefill_thread(
            prefill,
            [
                _nixl_registration(agent=second, required=b"2"),
                _nixl_registration(room=other_room),
            ],
        )

        self.assertEqual(prefill.request_status[other_room], KVPoll.WaitingForInput)

    def test_sender_failure_before_registration_answers_a_late_decode(self):
        """A sender aborted while decode has not registered yet has nobody to
        tell; a later registration must still learn the room failed."""
        prefill = _prefill_manager()
        prefill.update_status(ROOM, KVPoll.Bootstrapping)
        prefill.conclude_failure(bootstrap_room=ROOM, failure_reason="Aborted")
        self.assertEqual(prefill.sent, [])
        receiver = _waiting_decode_receiver()

        _run_mooncake_prefill_thread(prefill, [_registration()])

        self.assertNotIn(ROOM, prefill.transfer_infos)
        _deliver(prefill, receiver)
        self.assertDecodeFailedWith(receiver, "Aborted")

    def test_prefill_abort_after_the_sender_exists_fails_the_room(self):
        """An abort on prefill stops the room's chunks, so decode never gets the
        last one; it must hear the failure instead of waiting for its timeout."""
        prefill = _prefill_manager()
        _run_mooncake_prefill_thread(prefill, [_registration()])
        receiver = _waiting_decode_receiver()
        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.kv_mgr = prefill
        sender.bootstrap_room = ROOM
        sender.conclude_state = None
        sender._owns_bootstrap_room = True
        sender.trace_ctx = MagicMock()

        sender.abort()
        sender.abort()

        self.assertEqual(len(prefill.sent), 1)
        _deliver(prefill, receiver)
        self.assertDecodeFailedWith(receiver, "Aborted by AbortReq.")

    def test_only_rooms_with_a_real_decode_are_failed(self):
        for bootstrap_host, bootstrap_room, expected in (
            ("10.0.0.1", ROOM, True),
            (FAKE_BOOTSTRAP_HOST, ROOM, False),
            ("10.0.0.1", None, False),
        ):
            with self.subTest(host=bootstrap_host, room=bootstrap_room):
                prefill = _prefill_manager()
                queue = SimpleNamespace(kv_manager=prefill)
                req = SimpleNamespace(
                    bootstrap_host=bootstrap_host, bootstrap_room=bootstrap_room
                )
                PrefillBootstrapQueue.fail_before_handoff(queue, req, REASON)
                self.assertEqual(
                    prefill._failed_room_reason(ROOM),
                    REASON if expected else None,
                )

    def test_kv_capacity_reject_fails_the_room(self):
        prefill = _prefill_manager()
        _run_mooncake_prefill_thread(prefill, [_registration()])
        receiver = _waiting_decode_receiver()
        queue = SimpleNamespace(
            max_input_len=4,
            kv_manager=prefill,
            scheduler=SimpleNamespace(output_streamer=MagicMock()),
        )
        queue.fail_before_handoff = lambda req, reason: (
            PrefillBootstrapQueue.fail_before_handoff(queue, req, reason)
        )
        req = SimpleNamespace(
            rid="too-long",
            origin_input_ids=[0] * 8,
            bootstrap_host="10.0.0.1",
            bootstrap_room=ROOM,
            time_stats=SimpleNamespace(trace_ctx=MagicMock()),
            return_logprob=False,
            finished_reason=None,
        )
        with patch("sglang.srt.disaggregation.prefill.prepare_abort"):
            self.assertTrue(
                PrefillBootstrapQueue._check_if_req_exceed_kv_capacity(queue, req)
            )
        _deliver(prefill, receiver)
        self.assertEqual(receiver.poll(), KVPoll.Failed)

    def test_epd_encoder_failure_fails_the_room(self):
        """The encoder wait failed while the request was still in the multimodal
        waiting list, before any KV sender existed."""
        req = SimpleNamespace(return_logprob=False, finished_reason=None)
        mm_receiver = MagicMock()
        mm_receiver.process_waiting_requests.return_value = (
            [],
            [(req, REASON, HTTPStatus.GATEWAY_TIMEOUT)],
        )
        events = []
        receiver = SimpleNamespace(
            mm_receiver=mm_receiver,
            stream_output=lambda reqs, *_: events.append(("stream", reqs[0])),
            fail_before_handoff=lambda r, reason: events.append(("fail", r, reason)),
        )
        disagg = SimpleNamespace(
            language_only=True, encoder_transfer_backend="mooncake"
        )
        base = "sglang.srt.managers.scheduler_components.request_receiver"
        with (
            patch(f"{base}.get_disagg", return_value=disagg),
            patch(f"{base}.get_parallel", return_value=SimpleNamespace(pp_rank=0)),
        ):
            SchedulerRequestReceiver._apply_mm_receiver(receiver, [])

        self.assertEqual(events, [("stream", req), ("fail", req, REASON)])
        self.assertEqual(req.finished_reason.status_code, HTTPStatus.GATEWAY_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
