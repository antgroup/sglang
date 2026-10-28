"""Prefill admission against the free req_to_token rows: a request that already holds
a row is not counted twice (#38118, #35537), and the min-free-slots delay."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.min_free_slots_delayer import MinFreeSlotsDelayer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context, get_flags

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

NUM_ROWS = 4


class _Req:
    def __init__(self, holds_row):
        self.kv = SimpleNamespace(holds_kv=holds_row)
        self.beam_group = None
        self.token_indices_to_pool = None
        self.inflight_middle_chunks = 0

    def init_next_round_input(self, tree_cache=None):
        pass


class _Adder:
    """Admits whatever the gate lets through; the final chunk joins first."""

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = 0

    def add_chunked_req(self, req):
        self.can_run_list.append(req)
        return None

    def budget_state(self):
        return AddReqResult.CONTINUE

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


class _BatchBuilt(Exception):
    pass


def _scheduler(disaggregation_mode, running, final_chunk, waiting):
    s = Scheduler.__new__(Scheduler)
    s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
    s.enable_priority_preemption = False
    s.is_hybrid_swa = False
    s.waiting_queue = waiting
    s.chunked_req = final_chunk
    s.min_free_slots_delayer = None
    s.policy = Mock()
    s.policy.shortest_prefill_chunk_limit.return_value = None
    s.processed_tokens_counter = 0
    s.chunked_prefill_size = 8192
    s.dynamic_chunk_sizer = None
    s.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
    )
    s.get_prefill_adder_class = lambda: _Adder
    s.page_size = 1
    s.tree_cache = SimpleNamespace(buffer_pipeline=None, storage_prefetch_retries=None)
    s.token_to_kv_pool_allocator = None
    s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
    s.max_prefill_tokens = 1 << 20
    s.is_mixed_chunk = False
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs = 0
    s.max_running_requests = NUM_ROWS
    s.dllm_config = None
    s.enable_lora = False
    s.disaggregation_mode = disaggregation_mode
    s.enable_hicache_storage = False
    s.enable_lmcache = False
    s.truncation_align_size = None
    s.model_config = None
    s.enable_overlap = False
    s.spec_algorithm = None
    # The final chunk and the running requests hold rows; one row is free.
    s.req_to_token_pool = SimpleNamespace(
        available_size=lambda: NUM_ROWS - len(running) - 1
    )
    s.beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)
    return s


class TestHeldRowAdmission(CustomTestCase):
    def _admitted(self, disaggregation_mode):
        running = [_Req(holds_row=True), _Req(holds_row=True)]
        final_chunk = _Req(holds_row=True)
        waiting = _Req(holds_row=False)
        running_batch = ScheduleBatch(reqs=running)
        s = _scheduler(disaggregation_mode, running, final_chunk, [waiting])

        def capture(can_run_list, *args, **kwargs):
            raise _BatchBuilt(list(can_run_list))

        with (
            get_context().override_server_args(pp_max_micro_batch_size=NUM_ROWS),
            patch.object(ScheduleBatch, "init_new", side_effect=capture),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            with self.assertRaises(_BatchBuilt) as built:
                s._get_new_batch_prefill_raw(
                    prefill_delayer_single_pass=None, running_batch=running_batch
                )
        return (
            built.exception.args[0],
            running_batch.batch_is_full,
            final_chunk,
            waiting,
        )

    def test_final_chunk_leaves_the_free_row_to_a_waiting_request(self):
        for mode in (DisaggregationMode.NULL, DisaggregationMode.PREFILL):
            with self.subTest(mode=mode):
                can_run_list, batch_is_full, final_chunk, waiting = self._admitted(mode)
                self.assertEqual(can_run_list, [final_chunk, waiting])
                self.assertFalse(batch_is_full)


class TestMinFreeSlotsAdmission(CustomTestCase):
    """64 rows, the DSpark auto threshold of 4 free slots."""

    def _admit(self, *, running, waiting, switch_on, rounds=1):
        """Requests admitted in the first of ``rounds`` scheduler rounds that admits."""
        running_reqs = [_Req(holds_row=True) for _ in range(running)]
        waiting_reqs = [_Req(holds_row=False) for _ in range(waiting)]
        running_batch = ScheduleBatch(reqs=running_reqs)
        s = _scheduler(DisaggregationMode.NULL, running_reqs, None, waiting_reqs)
        s.max_running_requests = 64
        s.req_to_token_pool = SimpleNamespace(available_size=lambda: 64 - running)
        with get_flags().dsv41_sm90.override(
            subsystems=frozenset({"runtime"} if switch_on else ())
        ):
            s.min_free_slots_delayer = MinFreeSlotsDelayer(min_free_slots=4)

        def capture(can_run_list, *args, **kwargs):
            raise _BatchBuilt(list(can_run_list))

        with (
            get_context().override_server_args(pp_max_micro_batch_size=64),
            patch.object(ScheduleBatch, "init_new", side_effect=capture),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            for _ in range(rounds):
                try:
                    batch, _ = s._get_new_batch_prefill_raw(
                        prefill_delayer_single_pass=None, running_batch=running_batch
                    )
                except _BatchBuilt as built:
                    return built.args[0]
                self.assertIsNone(batch)
        return []

    def test_waiting_that_fits_the_free_slots_waits_a_bounded_number_of_rounds(self):
        """The last 3 requests of a 64-request wave (3 rows free) must not wait a
        whole decode round for a fourth free row; a hold that new arrivals could
        still grow lasts at most 32 rounds."""
        self.assertEqual(
            self._admit(running=61, waiting=3, switch_on=True, rounds=32), []
        )
        self.assertEqual(
            len(self._admit(running=61, waiting=3, switch_on=True, rounds=33)), 3
        )

    def test_more_waiting_than_free_slots_still_waits(self):
        self.assertEqual(
            self._admit(running=61, waiting=5, switch_on=True, rounds=100), []
        )

    def test_switch_off_keeps_the_threshold(self):
        self.assertEqual(
            self._admit(running=61, waiting=3, switch_on=False, rounds=100), []
        )
        self.assertEqual(len(self._admit(running=60, waiting=3, switch_on=False)), 3)


if __name__ == "__main__":
    unittest.main()
