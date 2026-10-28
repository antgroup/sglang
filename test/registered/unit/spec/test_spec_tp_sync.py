import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.spec_tp_sync import (
    SpecTpSync,
    SpecTpSyncSite,
    parse_spec_tp_sync,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

DRAFT = SpecTpSyncSite.DSPARK_DRAFT


class TestParseSpecTpSync(CustomTestCase):
    def test_retired_draft_handles_select_the_merged_site(self):
        for token in ("dspark-graph-greedy", "dspark-draft-sample", "2", "6"):
            with self.subTest(token=token):
                self.assertEqual(parse_spec_tp_sync(token), frozenset({DRAFT}))

    def test_negating_a_retired_handle_drops_the_merged_site(self):
        sites = parse_spec_tp_sync("all,-6")
        self.assertNotIn(DRAFT, sites)
        self.assertEqual(sites, frozenset(SpecTpSyncSite) - {DRAFT})

    def test_draft_noise_is_not_an_rng_site(self):
        # Ranks with the same logits draw the same draft noise.
        self.assertNotIn(DRAFT, parse_spec_tp_sync("rng"))
        self.assertIn(DRAFT, parse_spec_tp_sync("dspark-draft"))


class TestSpecTpSyncMemory(CustomTestCase):
    def test_attention_dp_capture_memory_is_reduced_over_the_graph_group(self):
        """With attention DP the sampling group is one rank while the draft
        graph still spans the MoE TP group; ranks deciding on local memory
        diverge on capture and hang its collectives."""
        peers_gb = [4.0, 0.5]
        for selector, distributed in (("all", True), ("off", False)):
            with (
                self.subTest(selector=selector),
                envs.SGLANG_SPEC_TP_SYNC.override(selector),
            ):
                sampling = SimpleNamespace(world_size=1, rank_in_group=0)
                graph = SimpleNamespace(world_size=2, cpu_group=object())
                sync = SpecTpSync(sampling)
                self.assertFalse(sync.enabled(SpecTpSyncSite.DSPARK_TARGET))

                def reduce_min(packet, op, group):
                    self.assertIs(group, graph.cpu_group)
                    packet.fill_(min(peers_gb) * (1 << 30))

                with (
                    patch(
                        "sglang.srt.utils.common.get_free_cpu_memory",
                        return_value=peers_gb[0] * (1 << 30),
                    ),
                    patch(
                        "sglang.srt.utils.common.get_cpu_ids_by_node",
                        return_value=[[0]],
                    ),
                    patch.object(torch.distributed, "all_reduce", reduce_min),
                ):
                    available = sync.available_memory_gb(
                        SpecTpSyncSite.DSPARK_MEM, "cpu", 0, group=graph
                    )
                self.assertEqual(
                    available, min(peers_gb) if distributed else peers_gb[0]
                )


if __name__ == "__main__":
    unittest.main()
