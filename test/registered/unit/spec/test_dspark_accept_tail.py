import unittest
from types import SimpleNamespace

import msgspec
import torch

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
    TargetVerifyResult,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

BS, STRIDE, VOCAB = 3, 5, 50


class _RecordingTpSync:
    def __init__(self):
        self.calls = []

    def sync(self, site, values):
        self.calls.append((site, values.clone()))
        return values


def _greedy_inputs():
    torch.manual_seed(0)
    candidates = torch.randint(0, VOCAB, (BS, STRIDE))
    target_predict = torch.randint(0, VOCAB, (BS, STRIDE))
    # Rows match the drafts for 0, 2 and all positions.
    target_predict[1, :2] = candidates[1, 1:3]
    target_predict[2, :-1] = candidates[2, 1:]
    logits = torch.randn(BS * STRIDE, VOCAB)
    logits.scatter_(1, target_predict.view(-1, 1), 100.0)
    prefix_lens = torch.tensor([7, 11, 13], dtype=torch.int64)
    return candidates, logits, prefix_lens


class TestAcceptTailParity(CustomTestCase):
    def _run(self, simulated_correct_len, verify_lens):
        candidates, logits, prefix_lens = _greedy_inputs()
        draft_tokens = candidates[:, 1:].contiguous()

        graph_sync = _RecordingTpSync()
        epilogue = DsparkVerifyEpilogue(
            max_bs=BS + 1,
            verify_num_draft_tokens=STRIDE,
            device="cpu",
            tp_sync=graph_sync,
            simulated_correct_len=simulated_correct_len,
        )
        epilogue._accept(
            candidates=candidates,
            logits=logits,
            draft_tokens=draft_tokens,
            seq_lens=prefix_lens,
            cutoff_verify_lens=verify_lens,
        )
        graph = epilogue.read_accept(BS)

        eager_sync = _RecordingTpSync()
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(model_type="test")
                )
            )
        )
        executor = TargetVerifyExecutor(
            target_worker=target_worker,
            gamma=STRIDE - 1,
            verify_num_draft_tokens=STRIDE,
            model_runner=None,
            kv_injector=None,
            tp_sync=eager_sync,
            simulated_correct_len=simulated_correct_len,
        )
        eager = executor.accept_and_finalize(
            folded_accept=False,
            bs=BS,
            verify_ids_2d=candidates,
            target_logits=logits,
            draft_block=DraftBlockResult(
                draft_tokens=draft_tokens,
                corrected_logits=None,
                greedy_mask=torch.ones(BS, dtype=torch.bool),
                temperatures=torch.ones(BS),
                greedy_draft=True,
            ),
            sampling_info=None,
            draft_input=None,
            layout=None
            if verify_lens is None
            else SimpleNamespace(verify_lens=verify_lens),
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
        )

        got, want = msgspec.structs.asdict(graph), msgspec.structs.asdict(eager)
        for field, value in want.items():
            self.assertTrue(
                torch.equal(got[field].cpu().long(), value.cpu().long()), field
            )
        self.assertEqual(
            [site for site, _ in graph_sync.calls],
            [SpecTpSyncSite.DSPARK_ACCEPT_GRAPH] * 3,
        )
        self.assertEqual(
            [site for site, _ in eager_sync.calls],
            [SpecTpSyncSite.DSPARK_ACCEPT_GREEDY] * 3,
        )
        for (_, graph_values), (_, eager_values) in zip(
            graph_sync.calls, eager_sync.calls
        ):
            self.assertTrue(torch.equal(graph_values, eager_values))

    def test_graph_accept_matches_eager_greedy(self):
        # Row 1 has a 2-token budget, so its 2 correct drafts are capped to 1.
        ragged = torch.tensor([STRIDE, 2, 3], dtype=torch.int64)
        for simulated in (None, 1):
            for verify_lens in (None, ragged):
                with self.subTest(simulated=simulated, ragged=verify_lens is not None):
                    simulated_correct_len = (
                        None if simulated is None else torch.tensor([simulated])
                    )
                    self._run(simulated_correct_len, verify_lens)


class _NoSiteTpSync(_RecordingTpSync):
    def enabled(self, site):
        return False


def _verify_out(logits):
    # A static verify graph's output always carries the target hidden states.
    return SimpleNamespace(
        next_token_logits=logits,
        hidden_states=logits.new_zeros(logits.shape[0], 4),
    )


def _verify_batch(candidates, prefix_lens):
    rows = candidates.numel()
    return SimpleNamespace(
        batch_size=BS,
        input_ids=candidates.view(-1),
        seq_lens=prefix_lens,
        req_pool_indices=torch.arange(BS, device=candidates.device),
        out_cache_loc=candidates.new_zeros(rows),
        positions=candidates.new_zeros(rows),
    )


class TestFusedStaticAcceptTail(CustomTestCase):
    """SGLANG_DSPARK_FUSED_ACCEPT_TAIL: the one-launch accept tail (the Triton kernel on
    CUDA) writes what the static epilogue's separate ops write."""

    def _static_accept(self, *, fused, simulated, tp_sync, device="cpu"):
        candidates, logits, prefix_lens = (t.to(device) for t in _greedy_inputs())
        epilogue = DsparkVerifyEpilogue(
            max_bs=BS + 1,
            verify_num_draft_tokens=STRIDE,
            device=device,
            tp_sync=tp_sync,
            simulated_correct_len=simulated,
            fused_accept_tail=fused,
        )
        epilogue._static_epilogue(
            _verify_out(logits), _verify_batch(candidates, prefix_lens)
        )
        return epilogue

    def test_fused_matches_separate_ops(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for simulated in (None, torch.tensor([1], device=device)):
                with self.subTest(device=device, simulated=simulated is not None):
                    kw = dict(simulated=simulated, device=device)
                    fused = self._static_accept(
                        fused=True, tp_sync=_NoSiteTpSync(), **kw
                    )
                    self.assertTrue(fused._serves_accept_tail(BS))
                    separate = self._static_accept(
                        fused=False, tp_sync=_NoSiteTpSync(), **kw
                    )
                    got = msgspec.structs.asdict(fused.read_accept(BS))
                    ref = msgspec.structs.asdict(separate.read_accept(BS))
                    for field, value in ref.items():
                        self.assertTrue(torch.equal(got[field], value), field)

    def test_accept_site_broadcast_keeps_separate_ops(self):
        # Rank 0's results are broadcast between the ops; one kernel would skip it.
        class _AllSites(_RecordingTpSync):
            def enabled(self, site):
                return True

        sync = _AllSites()
        epilogue = self._static_accept(fused=True, simulated=None, tp_sync=sync)
        self.assertFalse(epilogue._serves_accept_tail(BS))
        self.assertEqual(
            [site for site, _ in sync.calls], [SpecTpSyncSite.DSPARK_ACCEPT_GRAPH] * 3
        )


def _devices():
    return ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


class TestVocabShardVerify(CustomTestCase):
    """A target that defers its verify gather hands the step vocab shards: only a
    replayed fold may skip the gather, and the fold argmaxes through the target."""

    def test_only_a_replayed_fold_skips_the_gather(self):
        shard = torch.randn(BS * STRIDE, VOCAB // 2)
        full = torch.randn(BS * STRIDE, VOCAB)
        processor = SimpleNamespace(
            is_deferred_shard=lambda logits: logits.shape[-1] < VOCAB,
            finish_deferred_gather=lambda logits: full,
        )
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(model_type="test")
                )
            )
        )
        for deferring in (processor, None):
            executor = TargetVerifyExecutor(
                target_worker=target_worker,
                gamma=STRIDE - 1,
                verify_num_draft_tokens=STRIDE,
                model_runner=None,
                kv_injector=None,
                tp_sync=_RecordingTpSync(),
                target_logits_processor=deferring,
            )
            for keep in (False, True):
                for replayed in (False, True):
                    out = SimpleNamespace(next_token_logits=shard)
                    dropped = executor._drop_or_gather_vocab_shard(
                        TargetVerifyResult(
                            logits_output=out, can_run_cuda_graph=replayed
                        ),
                        keep=keep,
                    )
                    with self.subTest(
                        deferring=deferring is not None, keep=keep, replayed=replayed
                    ):
                        if deferring is None:
                            self.assertFalse(dropped)
                            self.assertIs(out.next_token_logits, shard)
                        elif keep and replayed:
                            self.assertTrue(dropped)
                            self.assertIsNone(out.next_token_logits)
                        else:
                            self.assertFalse(dropped)
                            self.assertIs(out.next_token_logits, full)

    def test_fold_argmaxes_the_shard_through_the_target(self):
        # Rank 1's slice: its local argmax is not the vocab's.
        for device in _devices():
            candidates, logits, prefix_lens = (t.to(device) for t in _greedy_inputs())
            shard = logits[:, VOCAB // 2 :].contiguous()
            seen = []

            def shard_argmax(local):
                seen.append(local)
                return logits.argmax(dim=-1)

            for fused in (False, True):
                with self.subTest(device=device, fused=fused):
                    accepts = []
                    for step_logits, argmax in ((logits, None), (shard, shard_argmax)):
                        epilogue = DsparkVerifyEpilogue(
                            max_bs=BS + 1,
                            verify_num_draft_tokens=STRIDE,
                            device=device,
                            tp_sync=_NoSiteTpSync(),
                            fused_accept_tail=fused,
                        )
                        epilogue.vocab_shard_argmax = argmax
                        epilogue._static_epilogue(
                            _verify_out(step_logits),
                            _verify_batch(candidates, prefix_lens),
                        )
                        accepts.append(epilogue.read_accept(BS))
                    self.assertIs(seen[-1], shard)
                    ref, got = (msgspec.structs.asdict(a) for a in accepts)
                    for field, value in ref.items():
                        self.assertTrue(torch.equal(got[field], value), field)


class TestPlainVerifyGatherDeferral(CustomTestCase):
    """LogitsProcessor.defer_plain_verify_gather: only a plain TP vocab split
    defers, and an unfolded step's gather returns what the forward would have."""

    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="test")

    def _processor(self, **overrides):
        processor = LogitsProcessor(SimpleNamespace(vocab_size=VOCAB))
        # A TP vocab split; the test context has one rank, so force the gather.
        processor.do_tensor_parallel_all_gather = True
        for name, value in overrides.items():
            setattr(processor, name, value)
        return processor

    def test_only_a_plain_tp_split_defers(self):
        lm_head = SimpleNamespace(
            tp_size=get_parallel().tp_size, num_embeddings_per_partition=32
        )
        other_group = SimpleNamespace(
            tp_size=lm_head.tp_size + 1, num_embeddings_per_partition=32
        )
        refusals = {
            "no tp gather": (dict(do_tensor_parallel_all_gather=False), lm_head),
            "dp attention": (dict(do_tensor_parallel_all_gather_dp_attn=True), lm_head),
            "attn-tp lm head": (dict(use_attn_tp_group=True), lm_head),
            "lm-head all-to-all": (dict(use_tp_lm_head_all_to_all=True), lm_head),
            "softcapping": (dict(final_logit_softcapping=30.0), lm_head),
            "lm head on another group": ({}, other_group),
        }
        for name, (overrides, head) in refusals.items():
            with self.subTest(name):
                processor = self._processor(**overrides)
                self.assertFalse(
                    processor.defer_plain_verify_gather(head, prefer_nvlink=False)
                )
                self.assertFalse(processor.defer_verify_gather)
        processor = self._processor()
        self.assertTrue(
            processor.defer_plain_verify_gather(lm_head, prefer_nvlink=False)
        )
        self.assertTrue(processor.defer_verify_gather)

    def test_unfolded_gather_matches_the_undeferred_forward(self):
        rows, hidden_size, partition = BS * STRIDE, 16, 32
        for device in _devices():
            gen = torch.Generator(device=device).manual_seed(0)
            lm_head = torch.nn.Linear(
                hidden_size, partition, bias=False, device=device, dtype=torch.bfloat16
            )
            lm_head.tp_size = get_parallel().tp_size
            lm_head.num_embeddings_per_partition = partition
            hidden = torch.randn(rows, hidden_size, generator=gen, device=device).to(
                torch.bfloat16
            )
            # Rank 0's view; the other rank's slice ends in vocab padding.
            other = torch.randn(rows, partition, generator=gen, device=device).to(
                torch.bfloat16
            )

            def gather(local):
                return torch.cat([local, other], dim=-1)

            plain, deferring = self._processor(), self._processor()
            self.assertTrue(
                deferring.defer_plain_verify_gather(lm_head, prefer_nvlink=False)
            )
            for processor in (plain, deferring):
                processor._logits_gatherer = gather
            for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DECODE):
                with self.subTest(device=device, mode=mode):
                    # The graph runner's FP32 buffer, which the forward casts into.
                    metadata = SimpleNamespace(
                        forward_mode=mode,
                        next_token_logits_buffer=torch.empty(
                            rows, VOCAB, device=device
                        ),
                    )
                    with torch.no_grad():
                        ref = plain._get_logits(hidden, lm_head, metadata).clone()
                        got = deferring._get_logits(hidden, lm_head, metadata)
                    if mode.is_target_verify():
                        self.assertEqual(got.shape, (rows, partition))
                        self.assertTrue(deferring.is_deferred_shard(got))
                        got = deferring.finish_deferred_gather(got)
                    self.assertEqual(got.dtype, torch.float32)
                    self.assertTrue(torch.equal(got, ref))


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestSimBonusSharedLength(CustomTestCase):
    def test_a_stride_0_view_of_the_shared_length_reads_like_a_full_tensor(self):
        """The simulated bonus takes the one shared length as expand(bs) of its [1]
        buffer (no copy in the graph); the partials equal those of a [bs] tensor."""
        from sglang.kernels.ops.speculative.dspark.simulated_bonus import (
            simulated_bonus_partials_graph,
        )

        bs, rows = 8, 6
        shared = torch.tensor([2], dtype=torch.int64, device="cuda")
        args = dict(
            target_logits=torch.randn(bs * rows, 9000, device="cuda"),
            temperatures=torch.ones(bs, device="cuda"),
            seed=torch.tensor([77], dtype=torch.int32, device="cuda"),
            bs=bs,
            rows_per_request=rows,
        )
        view = simulated_bonus_partials_graph(correct_len=shared.expand(bs), **args)
        full = simulated_bonus_partials_graph(correct_len=shared.repeat(bs), **args)
        self.assertTrue(torch.equal(view[0], full[0]))
        self.assertTrue(torch.equal(view[1], full[1]))


if __name__ == "__main__":
    unittest.main()
