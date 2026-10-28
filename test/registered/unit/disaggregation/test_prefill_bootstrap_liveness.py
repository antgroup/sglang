"""The prefill bootstrap /health, which the decode heartbeat polls, must fail
for a dead or hung prefill scheduler rank (DeepSeek-V4.1 PD switch).

It used to answer 200 while a prefill rank was dead (kill -9 of one TP rank) or
wedged, so decode never declared that prefill lost and its rooms were never
failed or reclaimed.
"""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from aiohttp.test_utils import TestClient, TestServer

from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dsv41_sm90 import initialize_dsv41_sm90
from sglang.srt.runtime_context import get_context, get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CONN = "sglang.srt.disaggregation.common.conn"


def _make_server():
    with (
        get_flags().dsv41_sm90.override(subsystems=frozenset({"pd"})),
        patch.object(CommonKVBootstrapServer, "run"),
    ):
        return CommonKVBootstrapServer("127.0.0.1", 0)


def _rank(tp, loop_age):
    return {
        "dp_group": 0,
        "cp_rank": 0,
        "tp_rank": tp,
        "pp_rank": 0,
        "loop_age": loop_age,
    }


class TestPrefillBootstrapHealth(CustomTestCase):
    def test_liveness_route_requires_pd_selection(self):
        """An opt-out server used to accept rank reports at a non-upstream URL.
        A stale report must affect /health only while the PD subsystem is on.
        """
        for enabled, disabled, expected in (
            (False, "", 404),
            (True, "", 200),
            (True, "pd", 404),
        ):
            with (
                self.subTest(enabled=enabled, disabled=disabled),
                get_context().override_server_args(disaggregation_mode="prefill"),
                get_flags().dsv41_sm90.override(
                    subsystems=frozenset(), lossy=frozenset(), role=""
                ),
                envs.SGLANG_OPT_DSV41_SM90.override(enabled),
                envs.SGLANG_OPT_DSV41_SM90_DISABLE.override(disabled),
                envs.SGLANG_OPT_DSV41_SM90_LOSSY.override(""),
                patch(
                    "sglang.srt.layers.dsv41_sm90.is_sm90_supported", return_value=True
                ),
                patch.object(CommonKVBootstrapServer, "run"),
            ):
                initialize_dsv41_sm90(
                    model_config=SimpleNamespace(
                        hf_config=SimpleNamespace(model_type="deepseek_v41")
                    )
                )
                server = CommonKVBootstrapServer("127.0.0.1", 0)

                async def check():
                    async with TestClient(TestServer(server.app)) as client:
                        response = await client.put(
                            "/rank_liveness", json=_rank(0, 200)
                        )
                        self.assertEqual(response.status, expected)
                        health = await client.get("/health")
                        self.assertEqual(health.status, 503 if expected == 200 else 200)
                        if expected != 200:
                            self.assertEqual(await health.text(), "OK")

                asyncio.run(check())

    def _run(self, server, steps):
        """steps: (monotonic_now, [rank reports]); returns /health (status, text)."""

        async def go():
            results = []
            async with TestClient(TestServer(server.app)) as client:
                for now, reports in steps:
                    # conn's own time binding only: the event loop keeps its clock.
                    with patch(f"{CONN}.time") as clock:
                        clock.monotonic.return_value = now
                        for report in reports:
                            resp = await client.put("/rank_liveness", json=report)
                            self.assertEqual(resp.status, 200)
                        resp = await client.get("/health")
                        results.append((resp.status, await resp.text()))
            return results

        with get_flags().dsv41_sm90.override(subsystems=frozenset({"pd"})):
            return asyncio.run(go())

    def test_health_fails_for_a_silent_or_stalled_rank(self):
        results = self._run(
            _make_server(),
            [
                (1000.0, []),  # nothing reported: switch off, or loop not started
                (1000.0, [_rank(0, 1.0), _rank(1, 1.0)]),
                (1100.0, [_rank(0, 1.0)]),  # TP1 silent for 100 s: within 180 s
                (1200.0, [_rank(0, 1.0)]),  # TP1 silent for 200 s: dead rank
                (1200.0, [_rank(1, 200.0)]),  # TP1 reports, but its loop is wedged
                (1205.0, [_rank(1, 0.5)]),  # TP1 loop moving again
            ],
        )
        self.assertEqual(
            [status for status, _ in results], [200, 200, 200, 503, 503, 200]
        )
        self.assertIn("DP0 CP0 TP1 PP0", results[3][1])
        self.assertNotIn("TP0", results[3][1])


class _StopReporter(Exception):
    pass


def _make_manager(manager_type=CommonKVManager):
    mgr = manager_type.__new__(manager_type)
    mgr.dist_init_addr = None
    mgr.bootstrap_host = "0.0.0.0"
    mgr.bootstrap_port = 8998
    mgr.attn_dp_rank, mgr.system_dp_size, mgr.system_dp_rank = 0, 1, 0
    mgr.attn_cp_rank, mgr.attn_tp_rank, mgr.pp_rank = 0, 3, 0
    mgr._scheduler_tick = None
    return mgr


class TestPrefillLivenessReporter(CustomTestCase):
    def test_teardown_stops_reporter_and_closes_session(self):
        """A role switch must stop the old prefill reporter instead of leaking it."""
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        mgr = _make_manager(MooncakeKVManager)
        mgr._worker_threads = []
        mgr._socket_lock = threading.Lock()
        mgr._socket_cache, mgr._monitor_cache = {}, {}
        mgr.server_socket = Mock()
        mgr._zmq_ctx = Mock()
        mgr.deregister_buffer_to_engine = Mock()
        mgr.disaggregation_mode = SimpleNamespace(value="prefill")
        started, closed = threading.Event(), threading.Event()
        session = Mock()
        session.close.side_effect = closed.set

        def make_session():
            started.set()
            return session

        with patch(f"{CONN}.requests.Session", side_effect=make_session):
            mgr._start_liveness_reporter()
            try:
                self.assertTrue(started.wait(timeout=1))
                thread = mgr._liveness_thread
                self.assertTrue(thread.is_alive())
                mgr.teardown()
                self.assertFalse(thread.is_alive())
                self.assertTrue(closed.is_set())
            finally:
                mgr._stop_liveness_reporter()
        session.put.assert_not_called()
        session.close.assert_called_once_with()

    def test_reports_loop_age_only_after_the_loop_starts(self):
        mgr = _make_manager()
        with patch(f"{CONN}.threading.Thread") as thread:
            mgr._start_liveness_reporter()
        report = thread.call_args.kwargs["target"]
        # Wake 1: loop not started; wake 2: the scheduler ticked at 40; wake 3: stop.
        wakes = iter([lambda: None, lambda: mgr.note_scheduler_tick(), None])

        def sleep(_):
            step = next(wakes)
            if step is None:
                raise _StopReporter
            step()

        with (
            get_flags().dsv41_sm90.override(subsystems=frozenset({"pd"})),
            patch(f"{CONN}.time") as clock,
            patch(f"{CONN}.requests.Session") as session,
        ):
            stop_wait = patch.object(mgr._liveness_stop, "wait", side_effect=sleep)
            stop_wait.start()
            self.addCleanup(stop_wait.stop)
            clock.monotonic.side_effect = [40.0, 52.5]
            with self.assertRaises(_StopReporter):
                report()
        session.return_value.close.assert_called_once_with()
        puts = session.return_value.put.call_args_list
        self.assertEqual(len(puts), 1)  # nothing before the first scheduler tick
        self.assertEqual(puts[0].args[0], "http://127.0.0.1:8998/rank_liveness")
        body = puts[0].kwargs["json"]
        self.assertEqual((body["tp_rank"], body["loop_age"]), (3, 12.5))


if __name__ == "__main__":
    unittest.main()
