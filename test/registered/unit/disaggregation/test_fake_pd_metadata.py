"""A fake PD handoff (health check, warmup) must not consume a previous
request's metadata slot: no prefill writes it, and a recycled slot still holds
the last owner's cache stats, logprobs, spec hidden states and DSA top-k
positions, which can point past the fake request's own KV."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, MetadataBuffers
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _buffers():
    with envs.SGLANG_ENABLE_DISAGG_SAMPLING_MASK.override(True):
        return MetadataBuffers(
            size=2,
            hidden_size=4,
            hidden_states_dtype=torch.float32,
            max_sampling_mask_tokens=4,
            max_top_logprobs_num=2,
            output_dsa_topk_indices_dim=3,
        )


def _dirty(buffers, cached_tokens):
    """What a released real transfer leaves behind: every field written, only
    the readiness marker cleared (MetadataBuffers.clear_slot)."""
    for buf in vars(buffers).values():
        if isinstance(buf, torch.Tensor):
            buf.fill_(2)
    buffers.cached_tokens[0, 0] = cached_tokens
    buffers.clear_slot(0)


def _decode_request(*, fake: bool):
    return SimpleNamespace(
        metadata_buffer_index=0,
        is_rebootstrap=False,
        kv_receiver=SimpleNamespace(clear=Mock()),
        req=SimpleNamespace(
            rid="HEALTH_CHECK_reused_slot" if fake else "real_pd_request",
            bootstrap_host=FAKE_BOOTSTRAP_HOST if fake else "10.0.0.1",
            bootstrap_room=0 if fake else 2,
            vocab_size=32000,
            output_ids=[],
            return_logprob=True,
            return_sampling_mask=False,
            logprob=SimpleNamespace(
                top_logprobs_num=2,
                token_ids_logprob=None,
                output_token_logprobs_val=[],
                output_token_logprobs_idx=[],
                output_top_logprobs_val=[],
                output_top_logprobs_idx=[],
            ),
            time_stats=SimpleNamespace(set_wait_queue_entry_time=Mock()),
        ),
    )


def _queue(buffers):
    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue.metadata_buffers = buffers
    queue.scheduler = SimpleNamespace(
        kv_checksum_computer=None,
        batch_result_processor=SimpleNamespace(_maybe_update_reasoning_tokens=Mock()),
    )
    queue.spec_algorithm = SimpleNamespace(is_none=lambda: False)
    queue._commit_hicache_local_restore_to_req = Mock()
    return queue


class TestFakePDMetadata(CustomTestCase):
    def test_reused_slot_gives_a_fake_request_none_of_the_old_owner(self):
        buffers = _buffers()
        _dirty(buffers, cached_tokens=281984)
        other_slot = buffers.get_buf(1)
        request = _decode_request(fake=True)

        _queue(buffers)._commit_transfer_to_req(request)

        req = request.req
        self.assertEqual(req.cached_tokens, 0)
        self.assertEqual(req.already_computed, 0)
        self.assertEqual(req.cached_tokens_device, 0)
        # Stale top-k positions would index past this request's KV.
        self.assertIsNone(req.output_dsa_topk_indices)
        for tensor in (
            req.output_topk_p,
            req.output_topk_index,
            req.hidden_states_tensor,
        ):
            self.assertEqual(torch.count_nonzero(tensor).item(), 0)
        self.assertEqual(req.logprob.output_token_logprobs_val, [0.0])
        self.assertEqual(req.logprob.output_top_logprobs_idx, [[0, 0]])
        self.assertEqual(len(req.output_ids), 1)
        self.assertIsNone(request.kv_receiver)
        # Only the fake request's slot is reset.
        for before, after in zip(other_slot, buffers.get_buf(1)):
            torch.testing.assert_close(before, after)

    def test_real_transfer_still_reads_what_prefill_wrote(self):
        buffers = _buffers()
        _dirty(buffers, cached_tokens=1024)
        buffers.bootstrap_room[0, 0] = 2  # prefill published the slot
        request = _decode_request(fake=False)

        _queue(buffers)._commit_transfer_to_req(request)

        self.assertEqual(request.req.cached_tokens, 1024)
        self.assertEqual(request.req.output_ids, [2])
        self.assertEqual(request.req.output_dsa_topk_indices.tolist(), [2, 2, 2])


if __name__ == "__main__":
    unittest.main()
