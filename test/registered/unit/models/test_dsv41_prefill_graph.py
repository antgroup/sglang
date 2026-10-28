"""DeepSeek-V4.1 prefill-CP layer-range graphs (prefill_graph subsystem). CPU only.

A graph replays what its range body recorded, so the body must compute what the
model's eager layer loop computes over the same layers; and the runner's capture
policy must keep every rank on the same path.
"""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace as NS
from typing import NamedTuple
from unittest.mock import patch

import torch

from sglang.srt.layers.aux_hidden_states import AuxHiddenStatePacker
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.models import deepseek_v4 as v4
from sglang.srt.models import deepseek_v4_boundary as boundary
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models import deepseek_v4_prefill_graph as pg
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

ROWS, HC, H, CP = 4, 2, 4, 2


class _Engram:
    def __init__(self, idx, log):
        self.layer_hash_index = idx
        self.log = log

    def __call__(self, x, hash_ids, forward_batch, *, input_ids, cp_all_tokens):
        self.log.append(("engram", self.layer_hash_index, x.clone(), cp_all_tokens))
        return x + hash_ids[:, None, None].to(x.dtype) * 0.1


class _Layer(torch.nn.Module):
    boundary_mhc_fusions = False
    boundary_pre_from_prev = True
    hc_cfg = mhc.HcConfig(HC, 4, 1e-5, 1e-6, H, True, True)

    def __init__(self, i, log, engram=None):
        super().__init__()
        self.layer_id = i
        self.log = log
        self.engram = engram
        self.attn_hc = None
        self.self_attn = NS(layer_id=i)

    def forward(self, hidden_states, forward_batch, **kw):
        state = boundary.materialize(hidden_states, forward_batch)
        self.log.append(
            (
                "layer",
                self.layer_id,
                state.residual.clone(),
                None if state.pre is None else state.pre.clone(),
                kw["positions"].clone(),
                kw["input_ids"].clone(),
                kw["input_ids_global"].clone(),
            )
        )
        update = boundary.HcUpdate(self, False)
        # A post that leaves the next sublayer's normed input and stats behind,
        # as a fused next-boundary post does.
        update.post = lambda value, st, **_: mhc.HcState(
            st.residual * 0.75 + value,
            value.sum(-1),
            mhc.HcNormed(value[:, 0] * 0.5),
            (value.sum(-1) * 2,),
        )
        out = state.residual * 0.5 + kw["positions"][:, None, None] * 0.01 + 1
        return residual_batch.stream_of(forward_batch).record(out, update)


class _Tail:
    """Rows 1 and 3 of this rank's shard, padded by one row as under CP."""

    scatter_indices = None
    cp_metadata = "cp"
    skip_layers = False

    def __init__(self, positions):
        self.token_indices = torch.tensor([1, 3])
        self.positions = self.rows(positions)
        self.extend_seq_lens_cpu = [2]

    def rows(self, t):
        r = t[self.token_indices]
        return torch.cat([r, r.new_zeros((1, *r.shape[1:]))])


class _Backend:
    def __init__(self, tail, log):
        self.tail_forward_metadata = NS(late_layer_tail=tail)
        self.log = log

    def write_tail_source_kv(self, *, layer, x, positions, forward_batch):
        self.log.append(("source_kv", layer.layer_id, x.clone(), positions.clone()))

    def enter_late_layer_tail(self, forward_batch):
        self.log.append(("enter",))
        return "saved"

    def exit_late_layer_tail(self, saved, forward_batch):
        self.log.append(("exit", saved))


def _combine(hc, state):
    if isinstance(state.input, mhc.HcNormed):
        return state.input.rows
    return (state.residual * state.pre[..., None]).sum(1)


def _stream_mean(x, out=None):
    return x.mean(1) if out is None else out.copy_(x.mean(1))


def _model(log, late, runner):
    m = v4.DeepseekV4Model.__new__(v4.DeepseekV4Model)
    torch.nn.Module.__init__(m)
    engrams = (1, 4)
    m.layers = torch.nn.ModuleList(
        _Layer(i, log, _Engram(engrams.index(i), log) if i in engrams else None)
        for i in range(6)
    )
    m.start_layer, m.end_layer, m.late_layer_start = 0, 6, late
    m.tail_writes_source_kv = late is not None
    m.pp_group = NS(world_size=1)
    m.dspark_layers_to_capture = [1, 3, 5]
    m.decoder_replay_tail = None
    m.engram_hasher = lambda ids, fb: torch.stack([ids * 3 + 1, ids * 5 + 2], dim=-1)
    m.prefill_graph = runner
    return m


def _forward(*, late, runner, capturing=False):
    torch.manual_seed(0)
    log, regions = [], []
    m = _model(log, late, runner)
    hidden = torch.randn(ROWS, HC, H)
    positions = torch.tensor([0, 2, 4, 6])
    input_ids = torch.tensor([5, 9, 7, 8])
    input_ids_global = torch.arange(ROWS * CP) + 20
    tail = _Tail(positions) if late is not None else None
    backend = _Backend(tail, log)
    fb = NS(
        forward_mode=ForwardMode.EXTEND,
        attn_cp_metadata=NS(total_seq_lens=ROWS * CP),
        input_ids=torch.arange(ROWS * CP) + 10,
        extend_seq_lens_cpu=[ROWS * CP],
        num_token_non_padded=None,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_logprob=False,
        residual_stream=None,
    )
    if runner is not None:

        def replay_body(
            self, *, region, key, inputs, body, forward_batch, tail, packer
        ):
            # A graph replays what its body recorded: run the body once.
            regions.append((region, inputs[0].shape[0]))
            with self._scope(forward_batch, tail, packer, capturing=capturing):
                return body(list(inputs), capturing)

        runner._run = replay_body.__get__(runner)
    recorder = NS(with_current_layer=lambda i: nullcontext(), recording=False)
    parallel = NS(attn_cp_rank=0, attn_cp_size=CP, attn_dp_size=1, attn_tp_size=1)
    aux = AuxHiddenStatePacker(3)
    with (
        patch.object(v4, "_is_cuda", True),
        patch.object(v4, "_is_hip", False),
        patch.object(v4, "get_attn_backend", return_value=backend),
        patch.object(v4, "is_cp_active", return_value=True),
        patch.object(v4, "get_parallel", return_value=parallel),
        patch.object(v4, "get_forward", return_value=NS(sp_active=False)),
        patch.object(pg, "get_forward", return_value=NS(sp_active=False)),
        patch.object(v4, "check_cuda_graph_backend", return_value=False),
        patch.object(pg, "check_cuda_graph_backend", return_value=False),
        patch.object(
            v4, "get_global_expert_distribution_recorder", return_value=recorder
        ),
        patch.object(
            pg, "get_global_expert_distribution_recorder", return_value=recorder
        ),
        patch.object(v4.mhc, "combine", _combine),
        patch("sglang.kernels.ops.layernorm.mhc.mhc_stream_mean", _stream_mean),
    ):
        out, pre, got_tail = m._forward_layers_hc_pre_from_prev(
            positions, hidden, fb, input_ids, input_ids_global, True, aux
        )
    return out, pre, got_tail, aux.finalize(), log, regions, fb


def _runner():
    return pg.PrefillGraphRunner(max_rows={pg.FULL: 64, pg.HEAD: 64, pg.TAIL: 64})


class TestRangesMatchEagerLoop(CustomTestCase):
    def _check(self, late, regions):
        ref = _forward(late=late, runner=None)
        for capturing in (False, True):
            with self.subTest(late=late, capturing=capturing):
                got = _forward(late=late, runner=_runner(), capturing=capturing)
                self.assertEqual(got[5], regions)
                for a, b in zip(got[:2] + got[3:4], ref[:2] + ref[3:4]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertIs(got[2] is None, ref[2] is None)
                self.assertEqual(len(got[4]), len(ref[4]))
                for a, b in zip(got[4], ref[4]):
                    self.assertEqual(len(a), len(b))
                    for x, y in zip(a, b):
                        if torch.is_tensor(x):
                            torch.testing.assert_close(x, y, rtol=0, atol=0)
                        else:
                            self.assertEqual(x, y)
                self.assertIsNone(got[6].residual_stream)

    def test_all_layers_without_bounded_replay(self):
        self._check(late=None, regions=[(pg.FULL, ROWS)])

    def test_head_and_tail_around_the_switch(self):
        # The switch (source KV write, tail rows) runs eagerly between the graphs;
        # the head hands over the normed input and stats its last post left.
        self._check(late=3, regions=[(pg.HEAD, ROWS), (pg.TAIL, 3)])


class _Pair(NamedTuple):
    data: torch.Tensor
    scale: torch.Tensor


class TestStateRoundTrip(CustomTestCase):
    def test_mhc_input_and_stats_cross_a_range_boundary(self):
        # The next combine short-circuits on state.input, so dropping it at a
        # range boundary would change the tail's numerics.
        t = [torch.full((2, 3), float(i)) for i in range(6)]
        for hc_input in (
            mhc.HcQuantized(t[2], _Pair(t[3], t[4])),
            mhc.HcQuantized(t[2], (t[3], t[4])),
            mhc.HcNormed(t[2]),
            # The SM90 fused post below the norm-fold rows leaves the rows unnormed.
            mhc.HcCombined(t[2]),
            None,
        ):
            state = mhc.HcState(t[0], t[1], hc_input, (t[5],))
            leaves, spec = pg.flatten_state(state)
            got = pg.rebuild_state(leaves, spec)
            self.assertEqual(type(got.input), type(hc_input))
            if hc_input is not None:
                self.assertIs(got.input.rows, t[2])
            if isinstance(hc_input, mhc.HcQuantized):
                self.assertIs(type(got.input.pair), type(hc_input.pair))
                self.assertIs(got.input.pair[1], t[4])
            self.assertIs(got.stats[0], t[5])
            self.assertIs(got.pre, t[1])


class TestCaptureFailure(CustomTestCase):
    def _capture(self, *, failure=None, comm_kind="v2"):
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        events, registered = [], []
        if comm_kind == "v2":
            comm = CustomAllReduceV2.__new__(CustomAllReduceV2)
            comm.disabled = False
            comm._graph_inputs = [(123, 16)]
            comm._graph_counter = 7
        elif comm_kind == "none":
            comm = None
        else:
            comm = NS(disabled=comm_kind == "disabled")
        capturing = False

        @contextmanager
        def graph_capture(*, stream):
            events.append("enter")
            yield NS(stream=stream)
            events.append("exit")
            if comm_kind == "v2":
                registered.extend(comm._graph_inputs)

        @contextmanager
        def stream_capture(**kwargs):
            nonlocal capturing
            capturing = True
            try:
                yield
            finally:
                capturing = False

        def vote(ok):
            self.assertFalse(capturing)
            events.append(("vote", ok))
            return ok and failure != "peer"

        def static_inputs(*args):
            if failure == "inputs":
                raise RuntimeError("input allocation failed")
            return ["static"]

        def pool(*args):
            if failure == "pool":
                raise RuntimeError("pool setup failed")
            return "pool"

        def body(*args):
            if failure == "body":
                raise RuntimeError("capture body failed")
            return ["output"], "spec"

        runner = _runner()
        key = (pg.FULL, 4)
        with (
            patch.object(
                pg, "get_parallel", return_value=NS(tp_group=NS(ca_comm=comm))
            ),
            patch.object(pg, "graph_capture", graph_capture),
            patch.object(pg, "BreakableCUDAGraphCapture", stream_capture),
            patch.object(pg, "BreakableCUDAGraph", return_value=NS(_segments=[])),
            patch.object(pg, "get_or_create_global_graph_memory_pool", pool),
            patch.object(pg, "set_graph_pool_id"),
            patch.object(
                pg, "get_or_create_global_graph_capture_stream", return_value="stream"
            ),
            patch.object(pg, "graph_pool_capture_scope", nullcontext),
            patch.object(pg, "_all_ranks", vote),
            patch.object(pg, "logger"),
            patch.object(runner, "_static_inputs", static_inputs),
            patch.object(torch.cuda, "mem_get_info", return_value=(8 << 30, 96 << 30)),
            patch.object(
                torch.cuda,
                "current_stream",
                return_value=NS(wait_stream=lambda _: events.append("wait")),
            ),
        ):
            result = runner._capture(key, [], body, None, None)
        return result, runner, key, comm, events, registered

    def test_failure_votes_before_registration_and_discards_all_pending_inputs(self):
        """A failed rank must join the vote before any peer registers CA graph inputs."""
        for failure in ("body", "inputs", "pool", "peer"):
            with self.subTest(failure=failure):
                result, runner, key, comm, events, registered = self._capture(
                    failure=failure
                )
                self.assertIsNone(result)
                self.assertEqual(
                    events, ["enter", ("vote", failure == "peer"), "exit", "wait"]
                )
                self.assertEqual(comm._graph_inputs, [])
                self.assertEqual(comm._graph_counter, 7)
                self.assertEqual(registered, [])
                self.assertIn(key, runner._failed)
                self.assertEqual(runner.stats.failed, 1)
                self.assertEqual(runner.graphs, {})

    def test_success_keeps_inputs_for_registration(self):
        result, runner, key, comm, events, registered = self._capture()
        self.assertIs(result, runner.graphs[key])
        self.assertEqual(result.outputs, ["output"])
        self.assertEqual(registered, [(123, 16)])
        self.assertEqual(comm._graph_inputs, registered)
        self.assertEqual(events, ["enter", ("vote", True), "exit", "wait"])
        self.assertEqual(runner.stats.captured, 1)

    def test_legacy_ca_runs_eagerly_but_disabled_or_absent_ca_can_capture(self):
        """Legacy CA cannot cancel its C++ registration after a failed capture."""
        result, runner, _, _, events, _ = self._capture(comm_kind="legacy")
        self.assertIsNone(result)
        self.assertEqual(events, [])
        self.assertEqual(runner.graphs, {})
        for comm_kind in ("none", "disabled"):
            for failure in (None, "body"):
                with self.subTest(comm_kind=comm_kind, failure=failure):
                    result, _, _, _, events, _ = self._capture(
                        comm_kind=comm_kind, failure=failure
                    )
                    self.assertEqual(result is None, failure is not None)
                    self.assertEqual(
                        events, ["enter", ("vote", failure is None), "exit", "wait"]
                    )


class TestRunnerPolicy(CustomTestCase):
    def _runner(self, max_graphs=2):
        r = pg.PrefillGraphRunner(
            max_rows={pg.FULL: 512, pg.HEAD: 512, pg.TAIL: 128},
            max_graphs=max_graphs,
            warmup=1,
        )
        captured = []

        def capture(graph_key, inputs, body, forward_batch, tail):
            captured.append(graph_key[1])
            entry = r.graphs[graph_key] = pg._Graph(None, [], [], None)
            return entry

        r._capture = capture
        r._replay = lambda graph, inputs, *a: ("replay", inputs[0].shape[0])
        return r, captured

    def _call(self, r, region, rows, cols=4):
        inputs = [torch.zeros(rows, cols), None]
        return r._run(
            region=region,
            key=(),
            inputs=inputs,
            body=None,
            forward_batch=None,
            tail=None,
            packer=None,
        )

    def test_exact_rows_second_sight_and_lru(self):
        r, captured = self._runner()
        with (
            patch("torch.cuda.mem_get_info", return_value=(8 << 30, 96 << 30)),
            patch.object(pg, "_all_ranks", side_effect=lambda ok: ok),
        ):
            self.assertIsNone(self._call(r, pg.TAIL, 129))  # above the cap
            self.assertIsNone(self._call(r, pg.FULL, 64))  # first sight: eager
            self.assertEqual(self._call(r, pg.FULL, 64), ("replay", 64))
            self.assertIsNone(self._call(r, pg.FULL, 65))  # no padding to 64
            self.assertEqual(self._call(r, pg.FULL, 65), ("replay", 65))
            self.assertEqual(self._call(r, pg.FULL, 64), ("replay", 64))
            self.assertIsNone(self._call(r, pg.FULL, 66))
            self.assertEqual(self._call(r, pg.FULL, 66), ("replay", 66))  # evicts 65
            self.assertEqual(sorted(k[1] for k in r.graphs), [64, 66])
            # Another range keeps its own cap.
            self.assertIsNone(self._call(r, pg.TAIL, 16))
            self.assertEqual(self._call(r, pg.TAIL, 16), ("replay", 16))
        self.assertEqual(captured, [64, 65, 66, 16])

    def test_low_memory_on_any_rank_runs_eagerly_without_evicting(self):
        """Capturing while a rank was out of device memory segfaulted in NCCL: a
        shape not captured yet runs eagerly and the cached graphs stay."""
        r, captured = self._runner(max_graphs=1)
        other_rank_ok = {"ok": True}
        with (
            patch("torch.cuda.mem_get_info", return_value=(8 << 30, 96 << 30)),
            patch.object(
                pg, "_all_ranks", side_effect=lambda ok: ok and other_rank_ok["ok"]
            ),
        ):
            self._call(r, pg.FULL, 64)
            self.assertEqual(self._call(r, pg.FULL, 64), ("replay", 64))
            other_rank_ok["ok"] = False
            self._call(r, pg.FULL, 128)
            self.assertIsNone(self._call(r, pg.FULL, 128))
            self.assertEqual([k[1] for k in r.graphs], [64])
            self.assertEqual(self._call(r, pg.FULL, 64), ("replay", 64))
            other_rank_ok["ok"] = True
            self.assertEqual(self._call(r, pg.FULL, 128), ("replay", 128))
        self.assertEqual(captured, [64, 128])


if __name__ == "__main__":
    unittest.main()
