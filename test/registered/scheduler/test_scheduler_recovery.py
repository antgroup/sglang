"""E2E: with --enable-scheduler-recovery, an unexpected exception in the
scheduler's forward path fails only the affected request with HTTP 500 and
the server keeps serving.

The failure is injected with SGLANG_TEST_SCHEDULER_RECOVERY_CRASH:
Scheduler.run_batch raises for any batch containing a request whose rid
carries TEST_RECOVERY_CRASH_RID_MARKER, which exercises the full recovery
path (abort every in-flight request with an internal error response, reset
the KV cache and scheduling state, resume the event loop). The bookkeeping
details of recovery are covered by
unit/managers/scheduler_components/test_error_isolation.py.
"""

import unittest
import uuid

import requests

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.error_isolation import (
    TEST_RECOVERY_CRASH_RID_MARKER,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")


class TestSchedulerRecovery(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_TEST_SCHEDULER_RECOVERY_CRASH.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--enable-scheduler-recovery"],
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _generate(self, rid=None):
        payload = {
            "text": "The capital of France is",
            "sampling_params": {"temperature": 0, "max_new_tokens": 8},
        }
        if rid is not None:
            payload["rid"] = rid
        return requests.post(f"{self.base_url}/generate", json=payload, timeout=120)

    def _assert_healthy_generation(self):
        resp = self._generate()
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text", resp.json())

    def test_a_crashing_request_gets_500_and_the_server_keeps_serving(self):
        self._assert_healthy_generation()

        # Recovery must work repeatedly, not just once per process.
        for round_idx in range(2):
            # The poisoned request crashes the scheduler event loop; recovery
            # turns that into an internal error response for it alone.
            rid = f"{TEST_RECOVERY_CRASH_RID_MARKER}-{uuid.uuid4().hex}"
            resp = self._generate(rid=rid)
            self.assertEqual(resp.status_code, 500, f"round {round_idx}")
            body = resp.json()
            self.assertEqual(body["object"], "error")
            self.assertEqual(body["code"], 500)
            self.assertIn("please retry", body["message"])

            # The server stayed up and keeps serving new requests.
            self._assert_healthy_generation()

        health = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(health.status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=3)
