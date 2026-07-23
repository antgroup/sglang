import unittest
from unittest.mock import Mock, patch

from sglang.srt.distributed.device_communicators import custom_all_reduce


class TestSharedBufferLifecycle(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
