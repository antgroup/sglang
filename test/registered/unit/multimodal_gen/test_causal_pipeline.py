# SPDX-License-Identifier: Apache-2.0
"""Pipeline dependency, placement, failure-drain and reproducibility contracts."""

import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.realtime.causal_pipeline import (
    CausalPipelinePlacement,
    CausalPipelinePlan,
    causal_pipeline_seed,
    run_causal_pipeline,
)
from sglang.multimodal_gen.runtime.realtime.causal_pipeline_config import (
    configure_causal_pipeline_parallelism,
    validate_causal_pipeline_args,
)
from sglang.multimodal_gen.test.causal_pipeline_test_utils import (
    _Links,
    _parallel,
    _Transport,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestCausalPipeline(CustomTestCase):
    def test_partition_covers_each_step_layer_once_in_order(self):
        for mode, degrees in (("timestep", (1, 2, 3)), ("block", (1, 2, 3, 4, 7, 15))):
            for degree in degrees:
                with self.subTest(mode=mode, degree=degree):
                    plan = CausalPipelinePlan(3, 5, degree, mode)
                    visited = [
                        (part.timestep, layer)
                        for rank in range(degree)
                        for part in plan.slices(rank)
                        for layer in range(part.block_start, part.block_end)
                    ]
                    self.assertEqual(
                        visited,
                        [(step, layer) for step in range(3) for layer in range(5)],
                    )
        for mode, degree in (("timestep", 4), ("block", 16)):
            with self.assertRaises(ValueError):
                CausalPipelinePlan(3, 5, degree, mode)

    def test_pipeline_preserves_both_activation_and_history_dependencies(self):
        """Overwriting history across timesteps or reordering chunks changes the oracle."""
        expected, history = [], {}
        for chunk in range(7):
            value = chunk + 1
            for stage in range(12):
                previous = history.get(stage, 0)
                history[stage] = value
                value = 2 * value + previous + stage
            expected.append(value)

        for mode, degree in (
            ("timestep", 1),
            ("timestep", 4),
            ("block", 5),
            ("block", 12),
        ):
            with self.subTest(mode=mode, degree=degree):
                plan = CausalPipelinePlan(4, 3, degree, mode)
                links = _Links(degree)

                def worker(rank):
                    local_history = {}

                    def execute(chunk, part, payload):
                        value = payload["value"]
                        for layer in range(part.block_start, part.block_end):
                            stage = part.timestep * 3 + layer
                            previous = local_history.get(stage, 0)
                            local_history[stage] = value
                            value = 2 * value + previous + stage
                        return {"value": value}

                    return run_causal_pipeline(
                        plan,
                        chunk_indices=list(range(7)),
                        transport=_Transport(links, rank),
                        initialize=lambda chunk: {"value": chunk + 1},
                        execute=execute,
                    )

                results, errors = _parallel(
                    [lambda rank=rank: worker(rank) for rank in range(degree)]
                )
                self.assertEqual(errors, [None] * degree)
                self.assertEqual([item["value"] for item in results[-1]], expected)
                self.assertTrue(all(result == [] for result in results[:-1]))

    def test_model_failure_drains_all_ranks_and_dedicated_decoder(self):
        for failing_rank in (0, 1, 2, 3):
            with self.subTest(failing_rank=failing_rank):
                plan = CausalPipelinePlan(3, 2, 3, "timestep")
                links = _Links(4)
                received = []

                def worker(rank):
                    transport = _Transport(links, rank)

                    def execute(chunk, part, payload):
                        if rank == failing_rank and chunk == 1:
                            raise ValueError("injected model failure")
                        return payload

                    return run_causal_pipeline(
                        plan,
                        chunk_indices=list(range(5)),
                        transport=transport,
                        initialize=lambda chunk: {"value": chunk},
                        execute=execute,
                        consume=transport.send,
                    )

                def decoder():
                    transport = _Transport(links, 3)
                    error = None
                    for index in range(5):
                        payload = transport.receive()
                        received.append(payload["chunk_index"])
                        error = error or payload.get("error")
                        if failing_rank == 3 and index == 1:
                            error = "injected decoder failure"
                    transport.finish(error)

                _, errors = _parallel(
                    [lambda rank=rank: worker(rank) for rank in range(3)] + [decoder]
                )
                self.assertEqual(received, list(range(5)))
                self.assertTrue(
                    all(isinstance(error, RuntimeError) for error in errors)
                )
                self.assertTrue(all("injected" in str(error) for error in errors))

    def test_vae_placement_counts_only_denoising_ranks_in_degree(self):
        shared = CausalPipelinePlacement(4)
        dedicated = CausalPipelinePlacement(4, "dedicated")
        self.assertEqual((shared.world_size, shared.decoder_rank), (4, 3))
        self.assertEqual((dedicated.world_size, dedicated.decoder_rank), (5, 4))
        self.assertEqual(CausalPipelinePlacement(4, "shared", 0).decoder_rank, 0)
        for placement, rank in (
            ("shared", 4),
            ("shared", -1),
            ("dedicated", 3),
            ("dedicated", 5),
        ):
            with self.assertRaises(ValueError):
                CausalPipelinePlacement(4, placement, rank)

    def test_noise_identity_is_window_and_rank_independent(self):
        contiguous = [
            causal_pipeline_seed(42, chunk, step)
            for chunk in range(8)
            for step in range(-1, 4)
        ]
        windowed = [
            causal_pipeline_seed(42, chunk, step)
            for window in (range(0, 3), range(3, 6), range(6, 8))
            for chunk in window
            for step in range(-1, 4)
        ]
        self.assertEqual(contiguous, windowed)
        self.assertEqual(len(set(contiguous)), len(contiguous))
        self.assertNotEqual(
            causal_pipeline_seed(41, 0, 0), causal_pipeline_seed(42, 0, 0)
        )

    def test_parallel_defaults_do_not_spend_decoder_rank_on_sequence_parallelism(self):
        args = SimpleNamespace(
            causal_pipeline_mode="timestep",
            causal_pipeline_parallel_size=4,
            causal_vae_placement="dedicated",
            causal_vae_rank=None,
            num_gpus=5,
            tp_size=None,
            sp_degree=None,
            ulysses_degree=None,
            ring_degree=None,
            kv_gather_degree=None,
            dp_size=1,
            enable_cfg_parallel=None,
            cfg_parallel_degree=None,
            encoder_parallel="auto",
        )
        configure_causal_pipeline_parallelism(args)
        self.assertEqual(
            (args.sp_degree, args.tp_size, args.cfg_parallel_degree), (1, 1, 1)
        )
        self.assertEqual(args.encoder_parallel, "replicate")
        args.num_gpus = 4
        with self.assertRaisesRegex(ValueError, "must be 5"):
            configure_causal_pipeline_parallelism(args)

    def test_disabled_mode_cannot_silently_accept_dedicated_vae(self):
        args = SimpleNamespace(
            causal_pipeline_mode="off",
            causal_pipeline_parallel_size=1,
            causal_vae_placement="dedicated",
            causal_vae_rank=None,
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_causal_pipeline_args(args)


if __name__ == "__main__":
    unittest.main()
