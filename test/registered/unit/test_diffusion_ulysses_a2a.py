import os
from unittest.mock import Mock, patch

from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a import (
    UlyssesA2AConfig,
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
                "SGLANG_ULYSSES_A2A_BACKEND": "fast_ulysses",
                "SGLANG_ENABLE_ULYSSES_P2P_A2A": "1",
            },
            clear=True,
        ):
            explicit = self._resolve_args(backend="nccl")
        self.assertEqual(explicit.ulysses_a2a_backend, "nccl")

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
