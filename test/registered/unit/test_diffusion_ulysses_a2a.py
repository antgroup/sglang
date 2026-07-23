import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_p2p_a2a as p2p_module
from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a import (
    A2AMode,
    A2ASlot,
    UlyssesA2AConfig,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a.base import (
    A2ASpec,
    UlyssesA2ACommitError,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a.router import (
    UlyssesA2ARouter,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_p2p_a2a import (
    UlyssesP2PAllToAll,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.distributed.device_communicators import custom_all_reduce
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="stage-a-test-cpu")


class TestDiffusionUlyssesA2A(CustomTestCase):
    @staticmethod
    def _resolve_args(
        *,
        backend: str | None = None,
        transfer: str | None = None,
        overlap: str | None = None,
    ) -> ServerArgs:
        args = object.__new__(ServerArgs)
        args.ulysses_a2a_backend = backend
        args.ulysses_a2a_transfer = transfer
        args.ulysses_a2a_qkv_overlap = overlap
        args.ulysses_a2a_legacy_prefer_p2p = False
        args._adjust_ulysses_a2a()
        args._validate_ulysses_a2a()
        return args

    def test_config_and_environment_precedence(self):
        self.assertEqual(UlyssesA2AConfig().backend, "nccl")
        with patch.dict(os.environ, {}, clear=True):
            defaults = self._resolve_args()
        self.assertEqual(defaults.ulysses_a2a_backend, "nccl")
        self.assertEqual(defaults.ulysses_a2a_transfer, "auto")
        self.assertEqual(defaults.ulysses_a2a_qkv_overlap, "off")

        with patch.dict(
            os.environ,
            {
                "SGLANG_ULYSSES_A2A_BACKEND": "auto",
                "SGLANG_ENABLE_ULYSSES_P2P_A2A": "1",
            },
            clear=True,
        ):
            explicit = self._resolve_args(backend="nccl")
        self.assertEqual(explicit.ulysses_a2a_backend, "nccl")

    def test_legacy_enable_maps_to_auto(self):
        with patch.dict(
            os.environ,
            {
                "SGLANG_ENABLE_ULYSSES_P2P_A2A": "1",
                "LINGBOT_FORCE_P2P": "1",
                "LINGBOT_ULYSSES_JIT": "1",
            },
            clear=True,
        ):
            args = self._resolve_args()

        self.assertEqual(args.ulysses_a2a_backend, "auto")
        self.assertTrue(args.ulysses_a2a_legacy_prefer_p2p)

    def test_conflicting_legacy_flags_fail_closed(self):
        with patch.dict(
            os.environ,
            {
                "SGLANG_ENABLE_ULYSSES_P2P_A2A": "1",
                "LINGBOT_FORCE_P2P": "0",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "conflict"):
                self._resolve_args()

    def test_unimplemented_overlap_modes_fail_closed(self):
        for overlap in ("auto", "on"):
            with self.subTest(overlap=overlap):
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaisesRegex(ValueError, "not implemented"):
                        self._resolve_args(backend="auto", overlap=overlap)
                with self.assertRaisesRegex(ValueError, "not implemented"):
                    UlyssesA2AConfig(qkv_overlap=overlap)

    def test_transfer_engine_is_rejected_for_non_fast_backend(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "only valid"):
                self._resolve_args(backend="nccl", transfer="sm")
        with self.assertRaisesRegex(ValueError, "only"):
            UlyssesA2AConfig(backend="nccl", transfer="sm")

    def test_free_shared_buffer_closes_peer_ipc_mappings(self):
        library = Mock()
        group = object()
        pointers = [101, 202, 303, 404]
        with (
            patch.object(
                custom_all_reduce,
                "CudaRTLibrary",
                return_value=library,
            ),
            patch.object(
                custom_all_reduce.dist,
                "get_rank",
                return_value=2,
            ) as get_rank,
            patch.object(custom_all_reduce, "_is_musa", False),
        ):
            custom_all_reduce.CustomAllreduce.free_shared_buffer(
                pointers,
                group=group,
            )

        get_rank.assert_called_once_with(group=group)
        closed = [
            call.args[0].value for call in library.cudaIpcCloseMemHandle.call_args_list
        ]
        self.assertEqual(closed, [101, 202, 404])
        self.assertEqual(library.cudaFree.call_args.args[0].value, 303)

    def test_free_shared_buffer_skips_unavailable_musa_ipc_close(self):
        library = Mock()
        pointers = [101, 202]
        with (
            patch.object(
                custom_all_reduce,
                "CudaRTLibrary",
                return_value=library,
            ),
            patch.object(custom_all_reduce.dist, "get_rank", return_value=0),
            patch.object(custom_all_reduce, "_is_musa", True),
        ):
            custom_all_reduce.CustomAllreduce.free_shared_buffer(pointers)

        library.cudaIpcCloseMemHandle.assert_not_called()
        self.assertEqual(library.cudaFree.call_args.args[0].value, 101)

    def test_p2p_runtime_failures_are_post_commit_errors(self):
        backend = object.__new__(UlyssesP2PAllToAll)
        backend._launch_lock = threading.Lock()
        backend._prepare_call = Mock(
            side_effect=RuntimeError("injected launch failure")
        )

        with self.assertRaisesRegex(UlyssesA2ACommitError, "after backend commit"):
            backend.all_to_all(Mock(), mode=0)

    def test_p2p_preflight_checks_selected_kernel_style_and_lifecycle_ops(self):
        fake_module = SimpleNamespace(
            meta_size=Mock(),
            init_ulysses_a2a=Mock(),
            dispose_ulysses_a2a=Mock(),
            ulysses_a2a=Mock(),
        )
        fake_ops = SimpleNamespace(
            meta_size=object(),
            init_ulysses_a2a=object(),
            dispose_ulysses_a2a=object(),
            ulysses_a2a=object(),
        )
        with (
            patch.object(p2p_module, "_is_cuda", True),
            patch.dict(sys.modules, {"sgl_kernel": fake_module}),
            patch.object(p2p_module.torch.ops, "sgl_kernel", fake_ops),
        ):
            self.assertTrue(p2p_module._ops_available(tk_style=False))
            self.assertFalse(p2p_module._ops_available(tk_style=True))

            fake_module.ulysses_a2a_tk = Mock()
            fake_ops.ulysses_a2a_tk = object()
            self.assertTrue(p2p_module._ops_available(tk_style=True))

            del fake_ops.dispose_ulysses_a2a
            self.assertFalse(p2p_module._ops_available(tk_style=True))

    def test_variable_consensus_normalizes_only_valid_local_sequence_dim(self):
        router_rank0 = object.__new__(UlyssesA2ARouter)
        router_rank0.nccl = SimpleNamespace(rank=0, world_size=2)
        router_rank1 = object.__new__(UlyssesA2ARouter)
        router_rank1.nccl = SimpleNamespace(rank=1, world_size=2)

        def make_spec(shape, seq_lens):
            return A2ASpec(
                mode=A2AMode.INPUT,
                shape=shape,
                dtype=p2p_module.torch.bfloat16,
                device=p2p_module.torch.device("cuda"),
                head_dim=2,
                seq_lens=seq_lens,
                capturing=False,
                slot=A2ASlot.PACKED_QKV,
            )

        rank0_descriptor, rank0_error = router_rank0._consensus_descriptor(
            make_spec((1, 3, 4, 8), (3, 5))
        )
        rank1_descriptor, rank1_error = router_rank1._consensus_descriptor(
            make_spec((1, 5, 4, 8), (3, 5))
        )
        self.assertEqual(rank0_error, "")
        self.assertEqual(rank1_error, "")
        self.assertEqual(rank0_descriptor, rank1_descriptor)

        _, invalid_error = router_rank1._consensus_descriptor(
            make_spec((1, 4, 4, 8), (3, 5))
        )
        self.assertIn("sequence_dim", invalid_error)


if __name__ == "__main__":
    unittest.main(verbosity=3)
