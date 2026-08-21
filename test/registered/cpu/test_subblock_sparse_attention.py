# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
    _get_subblock_sparse_attention_runner,
    _sm90_sparse_attention,
    _sm100_sparse_attention,
    _sm120_sparse_attention,
)
from sglang.multimodal_gen.runtime.platforms.cuda import (
    _SubBlockSparseAttentionBackendResolver,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-b-test-cpu")


class TestSubBlockSparseAttentionDispatch(CustomTestCase):
    def setUp(self):
        _get_subblock_sparse_attention_runner.cache_clear()
        self.addCleanup(_get_subblock_sparse_attention_runner.cache_clear)

    def test_dispatch_is_resolved_once_per_device(self):
        device = torch.device("cuda:0")
        with patch(
            "torch.cuda.get_device_capability", return_value=(9, 0)
        ) as get_capability:
            first = _get_subblock_sparse_attention_runner(device)
            second = _get_subblock_sparse_attention_runner(device)

        self.assertIs(first, _sm90_sparse_attention)
        self.assertIs(second, first)
        get_capability.assert_called_once_with(device)

    def test_dispatches_sm100(self):
        device = torch.device("cuda:0")
        with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
            runner = _get_subblock_sparse_attention_runner(device)

        self.assertIs(runner, _sm100_sparse_attention)

    def test_dispatches_sm120(self):
        device = torch.device("cuda:0")
        with patch("torch.cuda.get_device_capability", return_value=(12, 0)):
            runner = _get_subblock_sparse_attention_runner(device)

        self.assertIs(runner, _sm120_sparse_attention)

    def test_platform_resolver_loads_sm120_dependency(self):
        capability = Mock(major=12, minor=0)
        capability.as_version_str.return_value = "12.0"
        platform = Mock()
        platform.get_device_capability.return_value = capability

        with patch(
            "sglang.multimodal_gen.runtime.layers.attention.backends."
            "subblock_sparse.load_bsa_attn_sm120_blk64_fwd"
        ) as load_sm120:
            resolved = _SubBlockSparseAttentionBackendResolver.resolve(platform)

        self.assertEqual(
            resolved,
            "sglang.multimodal_gen.runtime.layers.attention.backends."
            "subblock_sparse_attn.SubBlockSparseAttentionBackend",
        )
        load_sm120.assert_called_once_with()

    def test_sm120_adapter_forwards_subblock_plan(self):
        q = torch.empty((1, 64, 2, 128), dtype=torch.bfloat16)
        k = torch.empty((1, 65, 2, 128), dtype=torch.bfloat16)
        v = torch.empty_like(k)
        q2k_block_index = torch.zeros((1, 2, 1, 2), dtype=torch.int32)
        expected = torch.empty_like(q)
        kernel = Mock(return_value=(expected, None))

        with patch(
            "sglang.multimodal_gen.runtime.layers.attention.backends."
            "subblock_sparse_attn.load_bsa_attn_sm120_blk64_fwd",
            return_value=kernel,
        ):
            result = _sm120_sparse_attention(
                q, k, v, q2k_block_index, topk=2, softmax_scale=0.125
            )

        self.assertIs(result, expected)
        kernel.assert_called_once()
        args, kwargs = kernel.call_args
        self.assertIs(args[0], q)
        self.assertIs(args[1], k)
        self.assertIs(args[2], v)
        self.assertIs(args[3], q2k_block_index)
        self.assertEqual(args[4], 2)
        torch.testing.assert_close(
            kwargs["block_sizes"], torch.tensor([64, 1], dtype=torch.int32)
        )
        self.assertIsNone(kwargs["q2k_block_nums"])
        self.assertEqual(kwargs["softmax_scale"], 0.125)

    def test_rejects_unsupported_compute_capability(self):
        device = torch.device("cuda:0")
        with patch("torch.cuda.get_device_capability", return_value=(10, 3)):
            with self.assertRaisesRegex(
                RuntimeError,
                "supports compute capability 9.0, 10.0, or 12.0;.*10.3 device",
            ):
                _get_subblock_sparse_attention_runner(device)


if __name__ == "__main__":
    unittest.main(verbosity=3)
