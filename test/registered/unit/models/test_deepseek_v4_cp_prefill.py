"""The CP-local router of the DeepSeek-V4.1 CP prefill MoE (deepseek_v4_cp_prefill)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.models import deepseek_v4_cp_prefill as cp_prefill
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TOP_K = 6


IMAGE_TOKEN = 3


class _Gate:
    def __init__(self, *, vision: bool, seen: list):
        self.seen = seen
        # Only its presence matters to the CP prefill; vision_topk reads it.
        self.e_score_correction_bias_vl = torch.zeros(8) if vision else None

    def __call__(self, x):
        self.seen.append(x.shape[0])
        return x[:, :8].float() * 3 - 1


def _top_k(logits):
    weights, ids = logits.topk(TOP_K, dim=1)
    return StandardTopKOutput(weights.float(), ids.to(torch.int32), logits)


def _vision_topk(moe, logits, input_ids):
    """vision_topk stand-in: image-token rows take another expert bias."""
    image = (input_ids == IMAGE_TOKEN)[:, None]
    return _top_k(logits + image * torch.arange(8).float())


def _row_router(*, is_hash: bool, vision: bool, seen: list):
    """A router that is a function of each row alone (and of its token id when
    hash-routed or VL-biased), recording the rows it was given."""

    def topk(x, logits, input_ids=None):
        if input_ids is not None:
            logits = logits + input_ids[:, None].float()
        return _top_k(logits)

    return SimpleNamespace(
        gate=_Gate(vision=vision, seen=seen), topk=topk, is_hash=is_hash
    )


def _gathered_routing(mlp, gathered, ids_global):
    """How DeepseekV2MoE.forward_normal routes the gathered rows."""
    logits = mlp.gate(gathered)
    if mlp.gate.e_score_correction_bias_vl is not None:
        return _vision_topk(mlp, logits, ids_global)
    return mlp.topk(
        gathered, logits, **({"input_ids": ids_global} if mlp.is_hash else {})
    )


class TestCpLocalRouter(CustomTestCase):
    def test_side_payload_round_trips_ids_and_weight_bits(self):
        ids = torch.randint(0, 384, (37, TOP_K), dtype=torch.int32)
        weights = torch.randn(37, TOP_K)
        weights[0, 0] = -0.0
        weights[1, 1] = float("nan")
        got = cp_prefill._unpack_topk(cp_prefill._pack_topk(ids, weights), TOP_K)
        self.assertTrue(torch.equal(got.topk_ids, ids))
        self.assertTrue(
            torch.equal(got.topk_weights.view(torch.int32), weights.view(torch.int32))
        )

    def test_repeated_rows_route_like_the_gathered_rank_block(self):
        """Rows repeated up to the validated GEMM row count, and the rank's block of
        the rank-major token ids, must give exactly this rank's rows of the
        gathered routing, also on the VL-biased layers of a checkpoint with a
        vision tower; the repeated rows' top-k must not survive."""
        cp, rank, rows = 4, 2, 5
        gathered = torch.randn(cp * rows, 16)
        ids_global = torch.arange(cp * rows) * 7 % 11
        for is_hash, vision in ((False, False), (True, False), (False, True)):
            with self.subTest(is_hash=is_hash, vision=vision):
                seen = []
                mlp = _row_router(is_hash=is_hash, vision=vision, seen=seen)
                full = _gathered_routing(mlp, gathered, ids_global)
                parallel = SimpleNamespace(attn_cp_size=cp, attn_cp_rank=rank)
                with (
                    patch.object(cp_prefill, "get_parallel", return_value=parallel),
                    patch.object(cp_prefill, "vision_topk", _vision_topk),
                ):
                    ids, weights = cp_prefill._route_local_rows(
                        mlp,
                        gathered[rank * rows : (rank + 1) * rows],
                        ids_global,
                        route_rows=2 * rows,
                    )
                own = slice(rank * rows, (rank + 1) * rows)
                self.assertEqual(seen[-1], 2 * rows)
                self.assertTrue(torch.equal(ids, full.topk_ids[own]))
                self.assertTrue(torch.equal(weights, full.topk_weights[own]))


if __name__ == "__main__":
    unittest.main()
