"""Unit tests for the scheduler's per-request error isolation and in-place
event-loop recovery (scheduler_components/error_isolation.py).

Every path here is CPU-only bookkeeping over fake scheduler state -- no model,
no GPU, no distributed init (the recovery consensus barrier short-circuits
when torch.distributed is not initialized). The e2e side (an injected crash
returns HTTP 500 for the affected request while the server keeps serving) is
covered by scheduler/test_scheduler_recovery.py.
"""

import unittest
from collections import deque
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components import error_isolation
from sglang.srt.managers.scheduler_components.error_isolation import (
    filter_failed_materialization,
    guard_request_admission,
    is_recoverable_exception,
    run_event_loop_with_recovery,
    try_recover_from_loop_crash,
)
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _FakeReq:
    """Must stay hashable: collect_inflight_reqs builds a set of requests."""

    def __init__(self, rid, *, finished=False, grammar=None):
        self.rid = rid
        self.output_ids = []
        self.weight_version_events = []
        self.grammar = grammar
        self._finished = finished

    def finished(self):
        return self._finished


def _admission_scheduler():
    s = Scheduler.__new__(Scheduler)
    s.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
    s.waiting_queue = []
    s.grammar_manager = SimpleNamespace(grammar_queue=[])
    s.disaggregation_mode = DisaggregationMode.NULL
    s.abort_request = MagicMock()
    return s


def _recovery_scheduler(**overrides):
    """A scheduler with every structure the recovery path touches, all empty
    or idle by default. collect_inflight_reqs is the real method."""
    s = Scheduler.__new__(Scheduler)
    s.disaggregation_mode = DisaggregationMode.NULL
    s.ps = SimpleNamespace(pp_size=1)
    s.enable_pdmux = False
    s.enable_hicache_storage = False
    s.mm_receiver = None
    s.dllm_manager = None
    s.chunked_req = None
    s._pending_chunked_abort_req = None
    s.running_batch = SimpleNamespace(reqs=[])
    s.last_batch = None
    s.cur_batch_for_debug = None
    s.result_queue = deque()
    s.waiting_queue = []
    s.grammar_manager = SimpleNamespace(grammar_queue=[])
    s.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
    s.beam_coordinator = MagicMock()
    s.device_module = MagicMock()
    s.is_fully_idle = MagicMock(return_value=True)
    s.flush_cache = MagicMock(return_value=True)
    for name, value in overrides.items():
        setattr(s, name, value)
    return s


def _sent_aborts(scheduler):
    return [
        call.args[0]
        for call in scheduler.ipc_channels.send_to_tokenizer.send_output.call_args_list
    ]


class TestExceptionClassification(CustomTestCase):
    def test_ordinary_request_driven_exceptions_are_recoverable(self):
        for exc in (
            RuntimeError("shape mismatch between embeddings and input_ids"),
            ValueError("invalid multimodal payload"),
            KeyError("missing_session"),
            AssertionError("batch invariant violated"),
            IndexError("list index out of range"),
        ):
            self.assertTrue(is_recoverable_exception(exc), repr(exc))

    def test_process_level_failures_are_not_recoverable(self):
        self.assertFalse(is_recoverable_exception(MemoryError()))
        self.assertFalse(is_recoverable_exception(SystemExit(0)))
        self.assertFalse(is_recoverable_exception(KeyboardInterrupt()))

    def test_device_oom_is_recoverable(self):
        # A failed allocation is clean, and the recovery flush frees memory.
        oom_type = getattr(torch, "OutOfMemoryError", None) or getattr(
            torch.cuda, "OutOfMemoryError"
        )
        oom = oom_type("CUDA out of memory. Tried to allocate 20.00 GiB")
        self.assertTrue(is_recoverable_exception(oom))

    def test_poisoned_device_or_comm_errors_are_not_recoverable(self):
        for message in (
            "CUDA error: an illegal memory access was encountered",
            "CUDA error: device-side assert triggered",
            "CUDA error: unspecified launch failure",
            "NCCL communicator was aborted on rank 1",
            "cuBLAS error: CUBLAS_STATUS_EXECUTION_FAILED",
            "HIP error: uncorrectable ECC error encountered",
        ):
            self.assertFalse(is_recoverable_exception(RuntimeError(message)), message)

    def test_dist_backend_error_is_not_recoverable(self):
        err_type = getattr(torch.distributed, "DistBackendError", None)
        if err_type is None:
            self.skipTest("this torch build has no DistBackendError")
        self.assertFalse(is_recoverable_exception(err_type("connection reset")))

    def test_accelerator_error_is_not_recoverable(self):
        err_type = getattr(torch, "AcceleratorError", None)
        if err_type is None:
            self.skipTest("this torch build has no AcceleratorError")
        self.assertFalse(is_recoverable_exception(err_type("device fault")))


class TestAdmissionGuard(CustomTestCase):
    def test_a_successful_admission_has_no_side_effects(self):
        s = _admission_scheduler()
        handler = MagicMock()
        recv = SimpleNamespace(rid="ok")

        guard_request_admission(s, handler, recv)

        handler.assert_called_once_with(recv)
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()
        s.abort_request.assert_not_called()

    def test_an_admission_error_aborts_only_that_request(self):
        s = _admission_scheduler()
        recv = SimpleNamespace(rid="bad")

        def handler(req):
            raise ValueError("malformed multimodal payload")

        guard_request_admission(s, handler, recv)  # must not raise

        out, echoed = s.ipc_channels.send_to_tokenizer.send_output.call_args.args
        self.assertIsInstance(out, AbortReq)
        self.assertEqual(out.rid, "bad")
        self.assertEqual(
            out.finished_reason["status_code"], HTTPStatus.INTERNAL_SERVER_ERROR
        )
        self.assertIn("malformed multimodal payload", out.finished_reason["message"])
        self.assertIs(echoed, recv)
        # The request never reached a queue, so no queue-side abort is needed.
        s.abort_request.assert_not_called()

    def test_a_partially_admitted_request_is_cleaned_up(self):
        s = _admission_scheduler()
        recv = SimpleNamespace(rid="partial")

        def handler(req):
            s.waiting_queue.append(_FakeReq("partial"))
            raise RuntimeError("failure after enqueueing")

        guard_request_admission(s, handler, recv)

        s.abort_request.assert_called_once()
        self.assertEqual(s.abort_request.call_args.args[0].rid, "partial")

    def test_non_recoverable_errors_keep_the_crash_behavior(self):
        s = _admission_scheduler()

        def handler(req):
            raise RuntimeError("CUDA error: device-side assert triggered")

        with self.assertRaisesRegex(RuntimeError, "device-side assert"):
            guard_request_admission(s, handler, SimpleNamespace(rid="r"))
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_requests_without_a_rid_keep_the_crash_behavior(self):
        s = _admission_scheduler()

        def handler(req):
            raise ValueError("boom")

        with self.assertRaisesRegex(ValueError, "boom"):
            guard_request_admission(s, handler, SimpleNamespace())

    def test_a_failing_abort_reraises_the_admission_error(self):
        s = _admission_scheduler()
        s.ipc_channels.send_to_tokenizer.send_output.side_effect = RuntimeError(
            "ipc channel down"
        )

        def handler(req):
            raise ValueError("original admission bug")

        with self.assertRaisesRegex(ValueError, "original admission bug"):
            guard_request_admission(s, handler, SimpleNamespace(rid="r"))


class TestFilterFailedMaterialization(CustomTestCase):
    def test_only_the_failing_request_is_dropped(self):
        s = _admission_scheduler()
        r1, r2, r3 = (SimpleNamespace(rid=f"r{i}") for i in (1, 2, 3))

        def materialize(req):
            if req is r2:
                raise ValueError("bad multimodal feature handle")

        kept = filter_failed_materialization(s, [r1, r2, r3], materialize)

        self.assertEqual(kept, [r1, r3])
        self.assertEqual([out.rid for out in _sent_aborts(s)], ["r2"])

    def test_non_recoverable_errors_propagate(self):
        s = _admission_scheduler()

        def materialize(req):
            raise RuntimeError("NCCL error: unhandled system error")

        with self.assertRaisesRegex(RuntimeError, "NCCL"):
            filter_failed_materialization(s, [SimpleNamespace(rid="r1")], materialize)


class TestRunEventLoopWithRecovery(CustomTestCase):
    def _publish(self, enabled: bool):
        override = get_context().override_server_args(enable_scheduler_recovery=enabled)
        override.install()
        self.addCleanup(override.restore)

    def test_disabled_recovery_keeps_the_crash_behavior(self):
        self._publish(False)
        dispatch = MagicMock(side_effect=RuntimeError("loop bug"))

        with self.assertRaisesRegex(RuntimeError, "loop bug"):
            run_event_loop_with_recovery(MagicMock(), dispatch)
        dispatch.assert_called_once()

    def test_enabled_recovery_resumes_the_loop(self):
        self._publish(True)
        dispatch = MagicMock(side_effect=[RuntimeError("loop bug"), None])

        with patch.object(
            error_isolation, "try_recover_from_loop_crash", return_value=True
        ) as recover:
            run_event_loop_with_recovery(MagicMock(), dispatch)

        self.assertEqual(dispatch.call_count, 2)
        recover.assert_called_once()

    def test_failed_recovery_reraises_the_loop_error(self):
        self._publish(True)
        dispatch = MagicMock(side_effect=RuntimeError("loop bug"))

        with patch.object(
            error_isolation, "try_recover_from_loop_crash", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "loop bug"):
                run_event_loop_with_recovery(MagicMock(), dispatch)
        dispatch.assert_called_once()


class TestLoopCrashRecovery(CustomTestCase):
    def setUp(self):
        # _abort_output_for_req reads the serving weight version off the bag.
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def test_recovery_aborts_all_inflight_requests_and_resets_state(self):
        running = _FakeReq("running")
        waiting = _FakeReq("waiting")
        finished = _FakeReq("finished", finished=True)
        chunked = _FakeReq("chunked")
        grammar_req = _FakeReq("grammar", grammar=MagicMock())
        queued = _FakeReq("queued")
        s = _recovery_scheduler(
            chunked_req=chunked,
            # `running` appears in three places and must be aborted once.
            running_batch=SimpleNamespace(reqs=[running]),
            last_batch=SimpleNamespace(reqs=[running]),
            cur_batch_for_debug=SimpleNamespace(reqs=[running]),
            result_queue=deque([(SimpleNamespace(reqs=[queued]), None)]),
            waiting_queue=[waiting, finished],
            grammar_manager=SimpleNamespace(grammar_queue=[grammar_req]),
        )

        ok = try_recover_from_loop_crash(s, RuntimeError("model forward bug"))

        self.assertTrue(ok)
        outs = _sent_aborts(s)
        self.assertEqual(len(outs), 5, "one abort per unfinished unique request")
        self.assertEqual(
            {out.rid for out in outs},
            {"running", "waiting", "chunked", "grammar", "queued"},
        )
        for out in outs:
            self.assertEqual(
                out.finished_reason["status_code"], HTTPStatus.INTERNAL_SERVER_ERROR
            )
            self.assertIn("model forward bug", out.finished_reason["message"])
            self.assertTrue(out.weight_versions)

        self.assertEqual(s.waiting_queue, [])
        self.assertEqual(list(s.grammar_manager.grammar_queue), [])
        grammar_req.grammar.cancel.assert_called_once()
        self.assertIsNone(s.chunked_req)
        self.assertIsNone(s.last_batch)
        self.assertIsNone(s.cur_batch_for_debug)
        self.assertEqual(len(s.result_queue), 0)
        self.assertTrue(s.running_batch.is_empty())
        s.device_module.synchronize.assert_called_once()
        s.flush_cache.assert_called_once()

    def test_non_recoverable_exceptions_keep_the_crash_behavior(self):
        s = _recovery_scheduler(waiting_queue=[_FakeReq("r")])
        exc = RuntimeError("CUDA error: an illegal memory access was encountered")

        self.assertFalse(try_recover_from_loop_crash(s, exc))
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()
        s.flush_cache.assert_not_called()

    def test_unsupported_modes_keep_the_crash_behavior(self):
        for field, value in (
            ("disaggregation_mode", DisaggregationMode.PREFILL),
            ("disaggregation_mode", DisaggregationMode.DECODE),
            ("ps", SimpleNamespace(pp_size=2)),
            ("enable_pdmux", True),
        ):
            s = _recovery_scheduler(**{field: value})
            self.assertFalse(
                try_recover_from_loop_crash(s, RuntimeError("bug")),
                f"{field}={value} must not be recovered in place",
            )
            s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()
            s.flush_cache.assert_not_called()

    def test_a_failed_cache_flush_keeps_the_crash_behavior(self):
        s = _recovery_scheduler(flush_cache=MagicMock(return_value=False))
        self.assertFalse(try_recover_from_loop_crash(s, RuntimeError("bug")))

    def test_undrainable_background_work_keeps_the_crash_behavior(self):
        s = _recovery_scheduler(is_fully_idle=MagicMock(return_value=False))
        with envs.SGLANG_SCHEDULER_RECOVERY_TIMEOUT.override(0.05):
            self.assertFalse(try_recover_from_loop_crash(s, RuntimeError("bug")))
        s.flush_cache.assert_not_called()

    def test_client_notification_is_best_effort(self):
        s = _recovery_scheduler(waiting_queue=[_FakeReq("r")])
        s.ipc_channels.send_to_tokenizer.send_output.side_effect = RuntimeError(
            "ipc channel down"
        )

        self.assertTrue(try_recover_from_loop_crash(s, RuntimeError("bug")))
        s.flush_cache.assert_called_once()


if __name__ == "__main__":
    unittest.main()
