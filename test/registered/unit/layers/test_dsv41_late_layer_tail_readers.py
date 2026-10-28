"""Decoder SWA bounded replay: the rows the late layers run (whole extends for
requests that read prompt rows) and how DP ranks learn each other's. CPU only."""

import random
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.kernels.ops.attention.dsv4.attn_metadata_kernels import (
    late_layer_tail_layout,
)
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import Mxfp8Activation
from sglang.srt.layers.attention import deepseek_v4_backend
from sglang.srt.layers.attention.deepseek_v4_backend import (
    SWA_WINDOW,
    DeepseekV4AttnBackend,
    DSV4Metadata,
    LateLayerTail,
    late_layer_whole_extend,
)
from sglang.srt.layers.aux_hidden_states import AuxHiddenStatePacker
from sglang.srt.layers.cp.interleave import (
    InterleaveContextParallelMetadata,
    InterleaveCPStrategy,
)
from sglang.srt.managers.schedule_policy import late_tail_safe_chunk_len
from sglang.srt.model_executor import forward_batch_info
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.runner import eager_runner
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.models.deepseek_common import decoder_swa_bounded_replay
from sglang.srt.models.deepseek_common.decoder_swa_bounded_replay import (
    late_layer_dp_counts,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4Model
from sglang.srt.models.deepseek_v4_mhc import HcState
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

CPU = torch.device("cpu")


def _batch(extend_lens, starts=None, capture=CaptureHiddenMode.NULL):
    return SimpleNamespace(
        extend_seq_lens_cpu=extend_lens,
        extend_logprob_start_lens_cpu=starts,
        return_logprob=starts is not None,
        capture_hidden_mode=capture,
    )


class TestWholeExtendRequests(CustomTestCase):
    def test_no_prompt_reader_keeps_the_window_tail(self):
        # No logprobs and output-only logprobs (start == extend) read no
        # prompt row.
        for batch in (
            _batch([300, 50]),
            _batch([300, 50], starts=[300, 50]),
            _batch([300], capture=CaptureHiddenMode.LAST),
        ):
            self.assertIsNone(late_layer_whole_extend(batch, dspark=False))

    def test_dspark_capture_reads_the_tail_only(self):
        # DSpark's target prefill, and its graph capture batches, capture FULL
        # hidden states, but its draft reads tail rows only.
        batch = _batch([300], capture=CaptureHiddenMode.FULL)
        self.assertIsNone(late_layer_whole_extend(batch, dspark=True))

    def test_any_prompt_logprob_reads_the_whole_extend(self):
        # Inside the last window too, and in a prompt shorter than a window:
        # those rows would otherwise get the window floor.
        batch = _batch([300, 300, 300, 50], starts=[0, 300, 300 - SWA_WINDOW, 0])
        whole = late_layer_whole_extend(batch, dspark=False)
        self.assertEqual(whole, [True, False, True, True])

    def test_full_hidden_capture_reads_every_row(self):
        batch = _batch([300, 50], capture=CaptureHiddenMode.FULL)
        self.assertEqual(late_layer_whole_extend(batch, dspark=False), [True, True])


class TestWholeExtendLayout(CustomTestCase):
    def test_other_requests_match_the_window_layout(self):
        extend_lens, seq_lens = [200, 300, 50], [300, 600, 50]
        token_indices, tail_lens, floor = late_layer_tail_layout(
            extend_lens_cpu=extend_lens,
            seq_lens_cpu=seq_lens,
            tail_len=SWA_WINDOW,
            device=CPU,
            whole_extend=[False, True, False],
        )
        base_indices, base_lens, base_floor = late_layer_tail_layout(
            extend_lens_cpu=extend_lens,
            seq_lens_cpu=seq_lens,
            tail_len=SWA_WINDOW,
            device=CPU,
        )
        self.assertEqual(tail_lens, [128, 300, 50])
        self.assertEqual(base_lens, [128, 128, 50])
        # The first and last requests keep the window layout row for row.
        torch.testing.assert_close(token_indices[:128], base_indices[:128])
        torch.testing.assert_close(floor[:128], base_floor[:128])
        torch.testing.assert_close(token_indices[428:], base_indices[256:])
        torch.testing.assert_close(floor[428:], base_floor[256:])
        # The whole request: every row of its extend, each with its plain window
        # (a floor a window below the extend start).
        torch.testing.assert_close(token_indices[128:428], torch.arange(200, 500))
        expected = torch.full((300,), 600 - 300 - SWA_WINDOW, dtype=torch.int32)
        torch.testing.assert_close(floor[128:428], expected)

    def test_short_whole_extend_keeps_its_plain_window(self):
        # A chunk shorter than a window is floored one window back otherwise.
        _, tail_lens, floor = late_layer_tail_layout(
            extend_lens_cpu=[40],
            seq_lens_cpu=[300],
            tail_len=SWA_WINDOW,
            device=CPU,
            whole_extend=[True],
        )
        self.assertEqual(tail_lens, [40])
        expected = torch.full((40,), 300 - 40 - SWA_WINDOW, dtype=torch.int32)
        torch.testing.assert_close(floor, expected)


class TestSkipUnreadLateLayers(CustomTestCase):
    """A CP prefill extend that leaves every request at least a window of fill
    skips its late layers; PrefillAdder's chunk guard keeps every late-layer row
    a later extend or decode reads written by an extend that ran them."""

    def _unread(
        self,
        remaining,
        *,
        whole_extend=None,
        capture=CaptureHiddenMode.NULL,
        dspark=False,
        cp_active=True,
        encoder_swa_replay=False,
    ):
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.skip_unread_late_layer_tail = True
        backend.encoder_row_floor = None
        backend.model_runner = SimpleNamespace(
            spec_algorithm=(
                SpeculativeAlgorithm.DSPARK if dspark else SpeculativeAlgorithm.NONE
            )
        )
        forward_batch = SimpleNamespace(
            extend_remaining_lens_cpu=remaining,
            capture_hidden_mode=capture,
            encoder_swa_replay=encoder_swa_replay,
        )
        with patch.object(deepseek_v4_backend, "is_cp_active", return_value=cp_active):
            return backend._late_layer_tail_unread(
                forward_batch, whole_extend=whole_extend
            )

    def test_every_request_must_leave_a_window(self):
        self.assertTrue(self._unread([SWA_WINDOW, 5000]))
        self.assertFalse(self._unread([SWA_WINDOW - 1, 5000]))
        # Not a plain EXTEND from host lists.
        self.assertFalse(self._unread(None))

    def test_readers_of_the_extend_keep_the_late_layers(self):
        self.assertFalse(self._unread([5000], whole_extend=[True]))
        self.assertFalse(self._unread([5000], capture=CaptureHiddenMode.LAST))
        self.assertFalse(self._unread([5000], capture=CaptureHiddenMode.FULL))
        self.assertTrue(
            self._unread([5000], capture=CaptureHiddenMode.FULL, dspark=True)
        )
        self.assertFalse(self._unread([5000], cp_active=False))
        self.assertFalse(self._unread([5000], encoder_swa_replay=True))

    def test_chunk_sequences_read_only_written_late_rows(self):
        """A forward that runs the late layers writes its tail (last <= window
        rows) and reads the rows from seq_len - window up to its extend; decode
        reads the last window. Under the guard every such row was written for any
        budgets; without it, a short chunk after a skipped one reads rows the
        skipped chunk never wrote."""
        page, window = 256, SWA_WINDOW
        rng = random.Random(0)

        def unwritten_reads(fill, guard):
            written, start, misses = set(), 0, 0
            while start < fill:
                budget = rng.choice((rng.randint(1, 2 * window), rng.randint(1, 4096)))
                if start == 0:
                    # A first chunk is cut on a page boundary (PrefillAdder).
                    ext = fill if fill <= budget else budget // page * page
                elif guard:
                    ext = late_tail_safe_chunk_len(
                        remaining=fill - start, budget=budget, window=window
                    )
                else:
                    ext = budget
                if ext <= 0:
                    continue
                end = min(fill, start + ext)
                if fill - end < window:
                    reads = range(max(end - window, 0), start)
                    misses += sum(row not in written for row in reads)
                    written.update(range(max(end - window, start), end))
                start = end
            decode_reads = range(max(fill - window, 0), fill)
            return misses + sum(row not in written for row in decode_reads)

        fills = [rng.randint(1, 3000) for _ in range(3000)]
        self.assertEqual(sum(unwritten_reads(f, guard=True) for f in fills), 0)
        self.assertGreater(sum(unwritten_reads(f, guard=False) for f in fills), 0)

    def test_skipped_forward_fills_every_late_capture(self):
        """The skipped forward still hands DSpark one aux slot per captured layer
        (finalize raises on a missing one) and the tail rows of the switch
        layer's streams, padded like the tail; the late slots are zeros."""
        rows, hc, hidden = 10, 4, 6
        model = DeepseekV4Model.__new__(DeepseekV4Model)
        torch.nn.Module.__init__(model)
        model.late_layer_start, model.end_layer = 3, 6
        model.dspark_layers_to_capture = [1, 4, 5]
        streams = torch.randn(rows, hc, hidden)
        pre = torch.randn(rows, hc)
        tail = LateLayerTail(
            token_indices=torch.tensor([7, 8, 9]),
            positions=torch.tensor([7, 8, 9]),
            extend_seq_lens=torch.tensor([3]),
            extend_seq_lens_cpu=[3],
            swa_out_cache_loc=torch.empty(0, dtype=torch.int32),
            pad_rows=2,
            skip_layers=True,
        )
        packer = AuxHiddenStatePacker(len(model.dspark_layers_to_capture))
        early = tail.rows(streams)[:, 0]
        packer.append(early)
        backend = Mock()
        with patch(
            "sglang.srt.models.deepseek_v4.get_attn_backend", return_value=backend
        ):
            residual, last_pre, out_tail = model._skip_late_layers(
                HcState(streams, pre),
                tail,
                forward_batch=SimpleNamespace(residual_stream=None),
                capture_dspark=True,
                dspark_aux_hidden_states=packer,
            )
        backend.skip_late_layer_tail.assert_called_once_with()
        self.assertIs(out_tail, tail)
        torch.testing.assert_close(residual, tail.rows(streams))
        torch.testing.assert_close(last_pre, tail.rows(pre))
        aux = packer.finalize()
        torch.testing.assert_close(aux[:, :hidden], early)
        self.assertEqual(aux.shape, (5, 3 * hidden))
        self.assertTrue(bool((aux[:, hidden:] == 0).all()))

    def test_host_cp_layout_is_the_device_layout(self):
        """The host build (no step syncs) gives the device build's layout for
        every rank, single-request, mixed and whole-extend batches."""
        cases = (
            ([5000], [5000], None),
            ([300, 40, 129, 1], [300, 1064, 129, 2049], None),
            ([300, 40, 129, 1], [300, 1064, 129, 2049], [False, True, False, False]),
        )
        for extend_lens, seq_lens, whole_extend in cases:
            token_indices, tail_lens_cpu, _ = late_layer_tail_layout(
                extend_lens_cpu=extend_lens,
                seq_lens_cpu=seq_lens,
                tail_len=SWA_WINDOW,
                device=CPU,
                whole_extend=whole_extend,
            )
            forward_batch = SimpleNamespace(
                batch_size=len(extend_lens),
                extend_seq_lens_cpu=extend_lens,
                positions=torch.cat(
                    [torch.arange(s - n, s) for n, s in zip(extend_lens, seq_lens)]
                ),
            )
            for cp_size in (1, 2, 4, 8):
                for rank in range(cp_size):
                    parallel = SimpleNamespace(attn_cp_rank=rank, attn_cp_size=cp_size)
                    layouts = []
                    for host in (False, True):
                        backend = object.__new__(DeepseekV4AttnBackend)
                        backend.cp_prefill_async_h2d = host
                        with patch.object(
                            deepseek_v4_backend, "get_parallel", return_value=parallel
                        ):
                            layouts.append(
                                backend._late_layer_tail_cp_layout(
                                    forward_batch,
                                    token_indices=token_indices,
                                    tail_lens=torch.tensor(
                                        tail_lens_cpu, dtype=torch.int32
                                    ),
                                    tail_lens_cpu=tail_lens_cpu,
                                )
                            )
                    device_layout, host_layout = layouts
                    with self.subTest(lens=extend_lens, cp=cp_size, rank=rank):
                        for key in ("local_token_indices", "local_positions"):
                            torch.testing.assert_close(
                                host_layout[key], device_layout[key], rtol=0, atol=0
                            )
                        for key in ("local_lens_cpu", "pad_rows"):
                            self.assertEqual(host_layout[key], device_layout[key])
                        host_cp, device_cp = (
                            host_layout["cp_metadata"],
                            device_layout["cp_metadata"],
                        )
                        for key in ("gather_index", "local_index"):
                            torch.testing.assert_close(
                                getattr(host_cp, key),
                                getattr(device_cp, key),
                                rtol=0,
                                atol=0,
                            )
                        for key in (
                            "per_rank_actual_token",
                            "per_rank_logical_token",
                            "total_seq_lens",
                        ):
                            self.assertEqual(
                                getattr(host_cp, key), getattr(device_cp, key)
                            )


class TestPrefillGraphTailCapacity(CustomTestCase):
    """The backend's graph check and the DP vote's view of it agree: no DP rank
    replays the prefill graph while another runs the step eagerly."""

    def _backend(self, capacity):
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.enable_decoder_swa_bounded_replay = True
        backend._prefill_graph_tail_capacity = lambda: capacity
        backend.model_runner = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.NONE)
        return backend

    def _both(self, extend_lens, capacity, starts=None, full=False):
        hidden = CaptureHiddenMode.FULL if full else CaptureHiddenMode.NULL
        forward_batch = _batch(extend_lens, starts, capture=hidden)
        forward_batch.seq_lens_cpu = torch.tensor([400] * len(extend_lens))
        schedule_batch = SimpleNamespace(
            extend_lens=extend_lens,
            extend_logprob_start_lens=starts,
            return_logprob=starts is not None,
            reqs=[SimpleNamespace(return_hidden_states_mode=hidden)] * len(extend_lens),
        )
        backend = self._backend(capacity)
        with (
            patch.object(
                deepseek_v4_backend, "_prefill_graph_max_seq_len", return_value=16384
            ),
            patch.object(
                deepseek_v4_backend,
                "get_server_return_hidden_states_mode",
                return_value=CaptureHiddenMode.NULL,
            ),
        ):
            can_run = backend.can_run_prefill_cuda_graph(forward_batch)
            vote = backend.vote_prefill_cuda_graph(schedule_batch)
        self.assertEqual(can_run, vote)
        return vote

    def test_whole_extend_counts_against_the_tail_block(self):
        self.assertTrue(self._both([300, 300], capacity=256))
        # A prompt-logprob request's tail no longer fits: run the step eagerly.
        self.assertFalse(self._both([300, 300], 256, starts=[0, 300]))
        self.assertTrue(self._both([300, 300], 428, starts=[0, 300]))
        self.assertFalse(self._both([300], 256, full=True))


class TestTailReadersCheck(CustomTestCase):
    def _check(self, batch, tail_lens, dspark=False):
        model = SimpleNamespace(dspark_layers_to_capture=[1] if dspark else None)
        tail = SimpleNamespace(extend_seq_lens_cpu=tail_lens)
        DeepseekV4Model._check_late_layer_tail_readers(model, batch, tail)

    def test_tail_covering_the_readers_passes(self):
        self._check(_batch([300, 50], starts=[300, 50]), [128, 50])
        self._check(_batch([300, 50], starts=[200, 0]), [128, 50])
        self._check(_batch([300, 50], starts=[0, 0]), [300, 50])
        self._check(_batch([300], capture=CaptureHiddenMode.FULL), [300])
        self._check(_batch([300], capture=CaptureHiddenMode.FULL), [128], dspark=True)

    def test_tail_missing_rows_raises(self):
        with self.assertRaises(ValueError):
            self._check(_batch([300], starts=[0]), [128])
        with self.assertRaises(ValueError):
            self._check(_batch([300], capture=CaptureHiddenMode.FULL), [128])


class TestLateLayerRowsUnderDpAttention(CustomTestCase):
    """A rank replaying the prefill graph never runs the model's late-layer code,
    so its eager peers learn its late-layer rows before the forward."""

    def setUp(self):
        override = get_context().override_server_args(
            tp_size=2,
            dp_size=2,
            enable_dp_attention=True,
            cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(bs=[])),
        )
        override.install()
        self.addCleanup(override.restore)

    def _dp_rank(self, rank):
        return get_parallel().override(
            attn_tp_size=1,
            attn_dp_size=2,
            attn_dp_rank=rank,
            tp_group=SimpleNamespace(cpu_group=None),
        )

    def test_rows_are_exchanged_before_the_forward(self):
        # Rank 0 prefills 300 rows; rank 1 is idle.
        n = 300
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch.arange(n),
            req_pool_indices=torch.tensor([1]),
            seq_lens=torch.tensor([n]),
            out_cache_loc=torch.arange(n),
            seq_lens_sum=n,
            positions=torch.arange(n),
            seq_lens_cpu=torch.tensor([n]),
            extend_num_tokens=n,
            extend_seq_lens=torch.tensor([n]),
            extend_seq_lens_cpu=[n],
            extend_prefix_lens_cpu=[0],
            is_extend_in_batch=True,
        )
        fb.init_mlp_sync_metadata(
            SimpleNamespace(
                global_num_tokens=[n, 0],
                global_num_tokens_for_logprob=[1, 0],
                dp_spec_prefill_coordination_applied=False,
                draft_global_num_tokens=None,
                can_run_decode_cuda_graph=False,
                can_run_dp_draft_cuda_graph=False,
            ),
            CPU,
        )
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.enable_decoder_swa_bounded_replay = True
        backend.model_runner = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.NONE)
        backend.get_cpu_graph_seq_len_fill_value = lambda: 1
        runner = MagicMock(attn_backend=backend, is_draft_worker=False)
        runner.attn_tp_sequence_sharded.return_value = False

        def idle_peer(out, local, group):
            out.copy_(torch.cat([local, local.new_zeros(1)]))

        with (
            self._dp_rank(0),
            # CPU runners have no driver to pin the synced token counts with.
            patch.object(forward_batch_info, "_is_cpu", True),
            patch.object(
                forward_batch_info, "all_gather_single", side_effect=idle_peer
            ),
        ):
            fb.prepare_mlp_sync_batch(runner)
        self.assertEqual(fb.global_late_layer_num_tokens_cpu, [SWA_WINDOW, 0])

        for rank, local_rows in ((0, SWA_WINDOW), (1, 0)):
            with (
                self.subTest(rank=rank),
                self._dp_rank(rank),
                patch.object(
                    decoder_swa_bounded_replay,
                    "all_gather_single",
                    side_effect=AssertionError("a collective inside the forward"),
                ),
            ):
                counts = late_layer_dp_counts(fb, local_rows, in_graph=False)
            self.assertEqual(counts, [SWA_WINDOW, 0])


def _prefill_graph_replay(*, ratios, seq_len):
    """The backend's default replayed-context limit and the widths a replay gives
    the full-row and tail metadata, for one request of ``seq_len`` tokens."""
    seen = {}

    def build_full(forward_batch, *, max_seq_len_override, use_prefill_cuda_graph):
        seen["full"] = max_seq_len_override

    def build_tail(forward_batch, max_seq_len):
        seen["tail"] = max_seq_len

    backend = object.__new__(DeepseekV4AttnBackend)
    backend.MAX_SEQ_LEN_FOR_CAPTURE = 4096
    backend.low_ratios = tuple(ratio for ratio in ratios if ratio in (1, 2))
    backend.has_c4 = 4 in ratios
    backend.has_c128 = 128 in ratios
    backend._build_forward_metadata = build_full
    backend._prefill_graph_translates_swa_loc = lambda metadata: False
    backend._prefill_graph_tail_metadata = build_tail
    backend._activate_request_window = lambda: None
    capture = object.__new__(DSV4Metadata)
    capture.prefill_graph_tail = True
    with patch.object(DSV4Metadata, "refresh_for_breakable_cuda_graph_replay_"):
        seen["limit"] = backend.prefill_cuda_graph_default_max_seq_len
        DeepseekV4AttnBackend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
            backend,
            capture,
            SimpleNamespace(
                seq_lens_cpu=torch.tensor([seq_len]), max_seq_len_override=1 << 20
            ),
        )
    return seen


class TestPrefillGraphReplayContext(CustomTestCase):
    """DeepSeek-V4.1's prefill graph replays any context, its metadata spanning the
    live context; with c4/c128 compress metadata the width stays fixed."""

    def test_limit_and_metadata_width(self):
        seq_len = 155648 + 1024
        for ratios, width in (((1, 2), seq_len), ((4, 128), 1 << 20)):
            with self.subTest(ratios=ratios):
                self.assertEqual(
                    _prefill_graph_replay(ratios=ratios, seq_len=seq_len),
                    {"limit": None, "full": width, "tail": width},
                )


class TestDSV4LateLayerTailLowRatioRows(CustomTestCase):
    """An eager low-ratio break inside the prefill graph's late-layer tail must read
    the hoisted rows of the tail, not rows rebuilt from the whole extend."""

    def _run(self, hoisted_req, hoisted_pos, positions, q_lora=None):
        backend = object.__new__(deepseek_v4_backend.DeepseekV4AttnBackend)
        backend.tail_source_kv_layer_id = None
        backend.forward_metadata = SimpleNamespace(
            low_ratio_req_indices=hoisted_req, low_ratio_pos_i64=hoisted_pos
        )
        backend._low_ratio_compress = MagicMock()
        backend._low_ratio_index_topk = MagicMock()
        # Two requests extend 4 and 6 tokens; the tail keeps their last 2 and 3.
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            encoder_swa_replay=False,
            req_pool_indices=torch.tensor([7, 9]),
            extend_seq_lens=torch.tensor([4, 6]),
        )
        layer = SimpleNamespace(layer_id=21, compressor=object(), indexer=object())
        with (
            patch.object(deepseek_v4_backend, "dsa_use_prefill_cp", return_value=False),
            patch.object(
                deepseek_v4_backend.DeepseekV4AttnBackend,
                "_low_ratio_in_prefill_graph",
                return_value=False,
            ),
        ):
            backend.forward_low_ratio_sources(
                layer=layer,
                x=torch.zeros(positions.shape[0], 4),
                q_lora=torch.zeros(positions.shape[0], 4) if q_lora is None else q_lora,
                positions=positions,
                forward_batch=forward_batch,
            )
        compress_args = backend._low_ratio_compress.call_args.args
        index_args = backend._low_ratio_index_topk.call_args.args
        self.index_q_lora = index_args[2]
        return compress_args[2], compress_args[3], index_args[3], index_args[4]

    def test_tail_break_uses_live_prefix_of_hoisted_rows(self):
        # Bucket of 8 rows, the 5 live tail rows first.
        hoisted_req = torch.tensor([7, 7, 9, 9, 9, 0, 0, 0])
        hoisted_pos = torch.tensor([2, 3, 3, 4, 5, 0, 0, 0])
        positions = hoisted_pos[:5].clone()
        req, pos, index_req, index_pos = self._run(hoisted_req, hoisted_pos, positions)
        self.assertEqual(req.tolist(), [7, 7, 9, 9, 9])
        self.assertEqual(pos.tolist(), [2, 3, 3, 4, 5])
        self.assertEqual(index_req.tolist(), [7, 7, 9, 9, 9])
        self.assertEqual(index_pos.tolist(), [2, 3, 3, 4, 5])

    def test_dp_padding_cuts_a_quantized_q_lora_by_rows(self):
        """HIP's quantized q_lora raises a TypeError on a plain slice; DP padding past the
        hoisted rows must cut it field by field."""
        q_lora = Mxfp8Activation(
            q=torch.arange(320).view(5, 64), scale=torch.arange(10).view(5, 2)
        )
        positions = torch.tensor([2, 3, 3, 0, 0])
        self._run(torch.tensor([7, 7, 9]), torch.tensor([2, 3, 3]), positions, q_lora)
        self.assertTrue(torch.equal(self.index_q_lora.q, q_lora.q[:3]))
        self.assertTrue(torch.equal(self.index_q_lora.scale, q_lora.scale[:3]))


class TestDSV41CpCompressorProjectsLocalRows(CustomTestCase):
    """CP prefill, ratio-1/2 compressor without a pre-gathered projection (the tail
    source write, eager breaks): under dsv41_sm90 "cp" each rank projects its own
    rows and gathers the projections; switch off it gathers the hidden rows and
    projects all of them on every rank (upstream)."""

    def _run(self, *, project_local_rows):
        cp_size, local, hidden, width = 4, 3, 16, 5
        backend = object.__new__(deepseek_v4_backend.DeepseekV4AttnBackend)
        backend.forward_metadata = SimpleNamespace(late_layer_tail=None)
        backend.cp_compressor_projects_local_rows = project_local_rows
        backend._low_ratio_compress_torch = MagicMock()
        projected_rows = []

        def project(x):
            projected_rows.append(x.shape)
            return x[:, :width], None

        gathered = []

        def gather(value, forward_batch, stream):
            gathered.append(value.shape)
            return value.repeat(cp_size, 1)

        layer = SimpleNamespace(
            layer_id=3, compressor=SimpleNamespace(project=project), indexer=None
        )
        forward_batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(total_seq_lens=cp_size * local),
            extend_seq_lens_cpu=[cp_size * local],
            positions=torch.arange(cp_size * local),
        )
        with (
            get_parallel().override(attn_cp_size=cp_size, attn_cp_rank=1),
            patch.object(
                deepseek_v4_backend, "cp_materialize_global_token_order", gather
            ),
            patch.object(torch.cuda, "current_stream", return_value=None),
            patch.object(
                deepseek_v4_backend,
                "token_req_indices",
                return_value=torch.zeros(cp_size * local, dtype=torch.int64),
            ),
        ):
            backend._forward_low_ratio_sources_cp(
                layer=layer,
                x=torch.randn(local, hidden),
                q_lora=None,
                positions=torch.arange(local),
                forward_batch=forward_batch,
                run_compressor=True,
                run_indexer=False,
            )
        args, kwargs = backend._low_ratio_compress_torch.call_args
        return projected_rows, gathered, args[1], kwargs.get("projected")

    def test_projects_before_the_gather_only_under_the_switch(self):
        projected_rows, gathered, x, projected = self._run(project_local_rows=True)
        self.assertEqual(projected_rows, [torch.Size([3, 16])])
        self.assertEqual(gathered, [torch.Size([3, 5])])
        self.assertIsNone(x)
        self.assertEqual(projected[0].shape, torch.Size([12, 5]))

        projected_rows, gathered, x, projected = self._run(project_local_rows=False)
        self.assertEqual(projected_rows, [])
        self.assertEqual(gathered, [torch.Size([3, 16])])
        self.assertEqual(x.shape, torch.Size([12, 16]))
        self.assertIsNone(projected)


class _TailBody:
    """DeepseekV4Model stand-in: returns this rank's share of the tail rows."""

    late_layer_start = 21

    def __init__(self, outputs):
        self.outputs = outputs

    def __call__(self, input_ids, positions, forward_batch, **kwargs):
        hidden, pre, aux = self.outputs
        return (hidden, pre), aux


class TestPrefillCpDSparkTailLogits(CustomTestCase):
    """Prefill CP + DSpark under decoder SWA bounded replay: each rank holds its share
    of the late-layer tail rows. The logits step must get every tail row in extend
    order, and DSpark the extend row of each (the full-extend gather raised)."""

    def test_tail_rows_reach_logits_in_extend_order_on_every_rank(self):
        cp_size = 8
        extend_lens = [300, 40]
        total = sum(extend_lens)
        token_indices, tail_lens_cpu, _ = late_layer_tail_layout(
            extend_lens_cpu=extend_lens,
            seq_lens_cpu=extend_lens,
            tail_len=SWA_WINDOW,
            device=CPU,
        )
        tail_lens = torch.tensor(tail_lens_cpu, dtype=torch.int32)
        owner = token_indices % cp_size
        extend_rows = token_indices.float()[:, None]
        # Tail-order outputs: hidden, pre-hc-head, packed DSpark aux, extend rows.
        global_outputs = (
            extend_rows + torch.arange(4) / 8,
            extend_rows + torch.arange(16) / 32,
            extend_rows - torch.arange(12) / 16,
            token_indices,
        )
        extend_counts = torch.bincount(
            torch.arange(total) % cp_size, minlength=cp_size
        ).tolist()

        for rank in range(cp_size):
            parallel = SimpleNamespace(
                attn_cp_rank=rank, attn_cp_size=cp_size, attn_cp_group=None
            )
            positions = torch.cat([torch.arange(n) for n in extend_lens])
            backend = object.__new__(DeepseekV4AttnBackend)
            backend.cp_prefill_async_h2d = False
            with patch.object(
                deepseek_v4_backend, "get_parallel", return_value=parallel
            ):
                layout = backend._late_layer_tail_cp_layout(
                    SimpleNamespace(batch_size=len(extend_lens), positions=positions),
                    token_indices=token_indices,
                    tail_lens=tail_lens,
                    tail_lens_cpu=tail_lens_cpu,
                )
            tail = LateLayerTail(
                token_indices=layout["local_token_indices"],
                positions=layout["local_positions"],
                extend_seq_lens=tail_lens,
                extend_seq_lens_cpu=tail_lens_cpu,
                swa_out_cache_loc=torch.empty(0),
                pad_rows=layout["pad_rows"],
                cp_metadata=layout["cp_metadata"],
                local_lens_cpu=layout["local_lens_cpu"],
            )
            physical = layout["cp_metadata"].per_rank_actual_token[rank]

            def share(value, r, physical=physical):
                rows = value[owner == r]
                pad = rows.new_zeros((physical - rows.shape[0], *rows.shape[1:]))
                return torch.cat([rows, pad])

            def all_gather(out, local, rank=rank, share=share):
                (value,) = [
                    v
                    for v in global_outputs
                    if v.dtype == local.dtype and v.shape[1:] == local.shape[1:]
                ]
                torch.testing.assert_close(local, share(value, rank))
                out.copy_(torch.cat([share(value, r) for r in range(cp_size)]))

            extend_cp = InterleaveContextParallelMetadata(
                total_seq_lens=total,
                bs=len(extend_lens),
                per_rank_actual_token=[max(extend_counts)] * cp_size,
                per_rank_logical_token=extend_counts,
            )
            fb = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=len(extend_lens),
                input_ids=torch.arange(total) + 1000,
                req_pool_indices=torch.tensor([1, 2]),
                seq_lens=torch.tensor(extend_lens),
                out_cache_loc=torch.arange(total),
                seq_lens_sum=total,
                positions=positions,
                seq_lens_cpu=torch.tensor(extend_lens),
                extend_num_tokens=total,
                extend_seq_lens=torch.tensor(extend_lens),
                extend_seq_lens_cpu=extend_lens,
                extend_prefix_lens_cpu=[0, 0],
                attn_cp_metadata=extend_cp,
            )
            logits = Mock(return_value=SimpleNamespace())
            model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
            torch.nn.Module.__init__(model)
            model.model = _TailBody(tuple(share(v, rank) for v in global_outputs[:3]))
            model.pp_group = SimpleNamespace(is_last_rank=True)
            model.capture_aux_hidden_states = True
            model.logits_processor = logits
            model.lm_head = object()

            @contextmanager
            def unsharded(
                input_embeds, positions, forward_batch, input_ids, sharded_hidden_states
            ):
                assert sharded_hidden_states is None
                yield input_embeds, positions, input_ids

            strategy = InterleaveCPStrategy(cp_size=cp_size)
            interleave = "sglang.srt.layers.cp.interleave"
            with (
                self.subTest(rank=rank),
                patch(
                    "sglang.srt.models.deepseek_v4.get_attn_backend",
                    return_value=SimpleNamespace(
                        tail_forward_metadata=SimpleNamespace(late_layer_tail=tail)
                    ),
                ),
                patch(
                    "sglang.srt.models.deepseek_v4.get_parallel", return_value=parallel
                ),
                patch("sglang.srt.layers.cp.base.get_parallel", return_value=parallel),
                patch(f"{interleave}.get_parallel", return_value=parallel),
                patch(f"{interleave}.use_symmetric_memory", return_value=nullcontext()),
                patch(f"{interleave}.is_allocation_symmetric", return_value=False),
                patch(
                    f"{interleave}.attn_cp_all_gather_into_tensor",
                    side_effect=all_gather,
                ),
                patch("sglang.srt.layers.cp.utils.is_cp_active", return_value=True),
                patch(
                    "sglang.srt.layers.cp.utils.get_cp_strategy", return_value=strategy
                ),
                patch.object(eager_runner, "cp_shard_model_inputs", unsharded),
                patch("torch.cuda.current_stream", return_value=None),
            ):
                out = EagerRunner._execute_extend_cp(
                    SimpleNamespace(model_runner=SimpleNamespace(model=model)),
                    fb,
                    {"input_embeds": torch.zeros(total, 4)},
                )
                self.assertIs(fb.attn_cp_metadata, extend_cp)
                args, kwargs = logits.call_args
                torch.testing.assert_close(args[0], fb.input_ids[token_indices])
                torch.testing.assert_close(args[1], global_outputs[0])
                self.assertEqual(args[3].extend_seq_lens_cpu, tail_lens_cpu)
                torch.testing.assert_close(args[4], global_outputs[2])
                self.assertIsNone(kwargs.get("hidden_states_before_norm"))
                torch.testing.assert_close(
                    out.hidden_states_token_indices, token_indices
                )


if __name__ == "__main__":
    unittest.main()


class TestCpTailVisionRoutingIds(CustomTestCase):
    """Under prefill CP the late layers' TP MoE routes every CP rank's tail rows in
    rank order; vision routing must read those rows' token ids, not a slice of the
    full-prompt ids (a language-only EPD prefill crashed in its first prefill)."""

    def setUp(self):
        override = get_context().override_server_args(tp_size=2)
        override.install()
        self.addCleanup(override.restore)

    def test_vision_routing_reads_the_gathered_tail_ids(self):
        cp_size = 2
        model = object.__new__(DeepseekV4Model)
        torch.nn.Module.__init__(model)
        tail = SimpleNamespace(rows=lambda x: x[:3])
        prompt_ids = torch.arange(100, 110)
        local_tail_ids = torch.tensor([7, 8, 9])

        def gather(local):
            # Rank 1 contributes its own tail rows after rank 0's.
            return torch.cat([local, local + 1000])

        for vl_routing, expected in (
            (True, torch.tensor([7, 8, 9, 1007, 1008, 1009])),
            # Without vision routing no late layer reads token ids.
            (False, prompt_ids[:3]),
        ):
            model.late_layer_vl_routing = vl_routing
            with (
                self.subTest(vl_routing=vl_routing),
                # The whole topology: an earlier test in the process may have
                # built real (world size 1) groups.
                get_parallel().override(
                    tp_size=cp_size,
                    tp_rank=0,
                    attn_tp_size=1,
                    attn_tp_rank=0,
                    attn_cp_size=cp_size,
                    attn_cp_rank=0,
                    attn_dp_size=1,
                    attn_dp_rank=0,
                    moe_tp_size=cp_size,
                    moe_ep_size=1,
                    moe_dp_size=1,
                    pp_size=1,
                    tp_group=SimpleNamespace(world_size=cp_size),
                    attn_tp_group=SimpleNamespace(world_size=1),
                    attn_cp_group=SimpleNamespace(world_size=cp_size),
                    moe_ep_group=SimpleNamespace(world_size=1),
                    pp_group=SimpleNamespace(world_size=1),
                ),
                patch(
                    "sglang.srt.models.deepseek_v4.attn_cp_interleave_gather",
                    side_effect=gather,
                ),
            ):
                ids = model._late_layer_ids_global(
                    tail=tail,
                    tail_input_ids=local_tail_ids,
                    input_ids_global=prompt_ids,
                    cp_extend=True,
                )
            self.assertTrue(torch.equal(ids, expected))
