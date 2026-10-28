"""The DSpark draft runner's metadata prep through the glue graph. Only a backend that declares its prep host-free accepts it
(the DSV4 backend, then taking the draft block's extend lengths from device
constants), and the runner hands the graph the registry's out_cache_loc slot instead
of the step's own tensor."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention import deepseek_v4_backend as dsv4
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _backend(**overrides):
    backend = dsv4.DeepseekV4AttnBackend.__new__(dsv4.DeepseekV4AttnBackend)
    fields = dict(
        is_dspark_draft=True,
        trtllm_attn=False,
        online_c128_mtp=SimpleNamespace(enabled=lambda: False),
        token_to_kv_pool=SimpleNamespace(request_window=None),
        draft_metadata_glued=False,
        _draft_extend_lens={},
    )
    fields.update(overrides)
    for name, value in fields.items():
        setattr(backend, name, value)
    return backend


def _parallel(cp=1):
    return mock.patch.object(
        dsv4, "get_parallel", return_value=SimpleNamespace(attn_cp_size=cp)
    )


class TestHostFreeDraftPrep(CustomTestCase):
    def test_the_audited_setup_accepts_and_glues(self):
        backend = _backend()
        with _parallel():
            self.assertEqual(backend.glue_draft_metadata(), "")
        self.assertTrue(backend.draft_metadata_glued)

    def test_every_other_setup_says_why(self):
        cases = {
            "not the DSpark draft runner": dict(is_dspark_draft=False),
            "trtllm-gen attention": dict(trtllm_attn=True),
            "online c128 MTP": dict(
                online_c128_mtp=SimpleNamespace(enabled=lambda: True)
            ),
            "a request window": dict(
                token_to_kv_pool=SimpleNamespace(request_window=object())
            ),
        }
        for why, overrides in cases.items():
            backend = _backend(**overrides)
            with _parallel(), self.subTest(why=why):
                self.assertEqual(backend.glue_draft_metadata(), why)
                self.assertFalse(backend.draft_metadata_glued)
        with _parallel(cp=2):
            self.assertEqual(_backend().glue_draft_metadata(), "context parallelism")


class TestDraftExtendLengths(CustomTestCase):
    def _moving(self, backend):
        moved = []

        def move(values):
            moved.append(list(values))
            return torch.tensor(values, dtype=torch.int32)

        backend._move_to_device = move
        return moved

    def test_glued_keeps_one_constant_per_batch_size(self):
        backend = _backend(draft_metadata_glued=True)
        moved = self._moving(backend)
        first = backend._draft_block_extend_lens([6, 6])
        self.assertIs(backend._draft_block_extend_lens([6, 6]), first)
        backend._draft_block_extend_lens([6])
        self.assertEqual(moved, [[6, 6], [6]])
        self.assertEqual(first.tolist(), [6, 6])

    def test_glued_refuses_a_new_constant_inside_a_capture(self):
        backend = _backend(draft_metadata_glued=True)
        self._moving(backend)
        backend._draft_block_extend_lens([6])
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
        ):
            # a warm-up made this one: the capture reads it by address
            backend._draft_block_extend_lens([6])
            with self.assertRaisesRegex(RuntimeError, "first seen inside"):
                backend._draft_block_extend_lens([6, 6])


class _DeviceLens(torch.Tensor):
    """Lengths that fail the step if read on the host (a device->host wait)."""

    def tolist(self):
        raise AssertionError("draft block prep copied the device lengths")


class TestUnGluedDraftBlockPrep(CustomTestCase):
    def _prep(self, backend, seq_lens):
        seen = {}
        backend._draft_block_extend_lens = lambda lens: torch.tensor(lens)
        backend.init_forward_metadata_prefill = lambda **kw: seen.update(kw)
        backend.page_size = 256
        backend.init_forward_metadata_dspark_draft_block(
            max_seq_len=1 << 20,
            req_pool_indices=torch.arange(2),
            seq_lens=seq_lens,
            seq_lens_cpu=None,
            out_cache_loc=torch.zeros(10, dtype=torch.int64),
            block_size=5,
        )
        return seen

    def test_switch_on_reads_no_host_lengths(self):
        """Under prefill CP the glue is off, and the per-step host copy of the draft
        lengths waited for the last verify on every decode step (no CPU run-ahead).
        Under the switch the prep must use the device lengths only."""
        backend = _backend(sm90_decode=True)
        seq_lens = torch.tensor([100, 7]).as_subclass(_DeviceLens)
        seen = self._prep(backend, seq_lens)
        self.assertIsNone(seen["seq_lens_cpu"])
        self.assertEqual(seen["extend_seq_lens_cpu"], [5, 5])
        self.assertEqual(seen["num_tokens"], 10)
        self.assertEqual(seen["seq_lens"].as_subclass(torch.Tensor).tolist(), [105, 12])
        # Off: the upstream host copy.
        seen = self._prep(_backend(sm90_decode=False), torch.tensor([100, 7]))
        self.assertEqual(seen["seq_lens_cpu"], [105, 12])


def _runner(*, draft=True, backend=None):
    runner = mod.DecodeCudaGraphRunner.__new__(mod.DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(is_draft_worker=draft)
    runner.attn_backend = backend if backend is not None else _backend()
    return runner


class TestRunnerGluesTheDraft(CustomTestCase):
    def test_the_glue_reads_the_registry_slot(self):
        runner = _runner()
        runner.buffers = SimpleNamespace(out_cache_loc=torch.arange(16))
        step_own = torch.arange(100, 106)
        view = SimpleNamespace(out_cache_loc=step_own)
        runner._draft_metadata_glue = False
        runner._glue_draft_inputs(view)
        self.assertIs(view.out_cache_loc, step_own)
        runner._draft_metadata_glue = True
        runner._glue_draft_inputs(view)
        slot = view.out_cache_loc
        self.assertEqual(slot.data_ptr(), runner.buffers.out_cache_loc.data_ptr())
        self.assertEqual(slot.shape, step_own.shape)
        idle = SimpleNamespace(out_cache_loc=None)
        runner._glue_draft_inputs(idle)
        self.assertIsNone(idle.out_cache_loc)


if __name__ == "__main__":
    unittest.main()
