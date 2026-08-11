# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn import (
    SolAttentionConfig,
    SolAttentionImpl,
    _SolAttentionKernels,
)
from sglang.multimodal_gen.runtime.platforms.cuda import CudaPlatformBase
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_SOL_BACKEND_CLS_STR = (
    "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn."
    "SolAttentionBackend"
)


class _FakeDenseImpl:
    def __init__(self, fill: float = -1.0):
        self.fill = fill
        self.varlen_calls = []

    def forward_varlen(self, query, key, value, **kwargs):
        self.varlen_calls.append((query, key, value, kwargs))
        return torch.full_like(query, self.fill)

    def forward(self, query, key, value, attn_metadata):
        return torch.full_like(query, self.fill)


def _make_impl(*, dense=None, strict=False, kv_splits="auto"):
    return SolAttentionImpl(
        num_heads=2,
        head_size=128,
        causal=False,
        softmax_scale=128**-0.5,
        num_kv_heads=2,
        tau=1.25,
        thresh_type="diag",
        kv_splits=kv_splits,
        strict=strict,
        dense_impl=dense or _FakeDenseImpl(),
    )


class TestSolAttentionConfig(CustomTestCase):
    def test_defaults_and_string_values(self):
        self.assertEqual(SolAttentionConfig.from_mapping(None), SolAttentionConfig())
        self.assertTrue(SolAttentionConfig().strict)
        self.assertEqual(SolAttentionConfig.from_mapping({"tau": 0}).tau, 0.0)
        self.assertEqual(
            SolAttentionConfig.from_mapping(
                {
                    "tau": "1.5",
                    "thresh_type": "exact",
                    "kv_splits": "4",
                    "strict": "true",
                }
            ),
            SolAttentionConfig(
                tau=1.5,
                thresh_type="exact",
                kv_splits=4,
                strict=True,
            ),
        )

    def test_invalid_values_are_rejected(self):
        for config in (
            {"tau": float("nan")},
            {"thresh_type": ""},
            {"thresh_type": "approx"},
            {"kv_splits": 0},
            {"kv_splits": 3},
            {"kv_splits": "3"},
            {"kv_splits": "many"},
            {"strict": 1},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                SolAttentionConfig.from_mapping(config)


class TestSolAttentionSelection(CustomTestCase):
    def test_enum_name_and_sparse_classification(self):
        self.assertEqual(str(AttentionBackendEnum.SOL_ATTN), "sol_attn")
        self.assertTrue(AttentionBackendEnum.SOL_ATTN.is_sparse)
        self.assertIn(
            AttentionBackendEnum.SOL_ATTN,
            MiniMaxH3DiTArchConfig()._supported_attention_backends,
        )

    def test_cuda_resolver_does_not_import_optional_package(self):
        with patch("importlib.import_module") as import_module:
            resolved = CudaPlatformBase.get_attn_backend_cls_str(
                AttentionBackendEnum.SOL_ATTN,
                head_size=128,
                dtype=torch.bfloat16,
            )

        self.assertEqual(resolved, _SOL_BACKEND_CLS_STR)
        import_module.assert_not_called()


class TestSolAttentionImpl(CustomTestCase):
    def setUp(self):
        self.bounds = (0, 2, 5)
        self.cu_seqlens = torch.tensor(self.bounds, dtype=torch.int32)
        base = torch.arange(5 * 2 * 256, dtype=torch.float32).reshape(5, 2, 256)
        self.query = base[..., ::2]
        self.key = self.query + 1
        self.value = self.query + 2

    def _forward(self, impl, *, max_seqlen=3):
        return impl.forward_varlen(
            self.query,
            self.key,
            self.value,
            cu_seqlens=self.cu_seqlens,
            cu_seqlens_host=self.bounds,
            max_seqlen=max_seqlen,
        )

    def test_backend_owns_global_config_parsing(self):
        server_args = SimpleNamespace(
            attention_backend_config={
                "tau": 1.75,
                "thresh_type": "exact",
                "kv_splits": 2,
                "strict": False,
            }
        )
        with patch(
            "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
            return_value=server_args,
        ):
            impl = SolAttentionImpl(
                num_heads=2,
                head_size=128,
                causal=False,
                softmax_scale=128**-0.5,
                num_kv_heads=2,
                dense_impl=_FakeDenseImpl(),
            )

        self.assertEqual(impl.config.tau, 1.75)
        self.assertEqual(impl.config.thresh_type, "exact")
        self.assertEqual(impl.config.kv_splits, 2)
        self.assertFalse(impl.config.strict)

    def test_strict_mode_rejects_ineligible_inputs(self):
        with self.assertRaisesRegex(ValueError, "unsupported inputs"):
            self._forward(_make_impl(strict=True))

    def test_ineligible_input_uses_dense_without_loading_sol_attn(self):
        dense = _FakeDenseImpl(fill=3.0)
        impl = _make_impl(dense=dense)
        with patch(
            "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn."
            "_load_sol_attn_kernels"
        ) as load_kernels:
            output = self._forward(impl)

        torch.testing.assert_close(output, torch.full_like(self.query, 3.0))
        self.assertEqual(len(dense.varlen_calls), 1)
        load_kernels.assert_not_called()

    def test_native_varlen_kernel_receives_contiguous_packed_inputs(self):
        impl = _make_impl()
        kernel = Mock(side_effect=lambda query, key, value, **_: query + value)
        kernels = _SolAttentionKernels(
            varlen=kernel,
            bthd=None,
            load_error=None,
        )
        with (
            patch.object(impl, "_eligible", return_value=True),
            patch.object(impl, "_resolve_kv_splits", return_value=1),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn."
                "_load_sol_attn_kernels",
                return_value=kernels,
            ),
        ):
            output = self._forward(impl)

        torch.testing.assert_close(output, self.query + self.value)
        kernel_query, kernel_key, kernel_value = kernel.call_args.args
        self.assertTrue(kernel_query.is_contiguous())
        self.assertTrue(kernel_key.is_contiguous())
        self.assertTrue(kernel_value.is_contiguous())
        self.assertEqual(kernel.call_args.kwargs["max_seqlen"], 3)
        self.assertEqual(kernel.call_args.kwargs["tau"], 1.25)
        self.assertEqual(kernel.call_args.kwargs["kv_splits"], 1)

    def test_official_bthd_adapter_keeps_segments_isolated(self):
        impl = _make_impl(kv_splits=2)
        call_count = 0

        def kernel(query, key, value, **kwargs):
            nonlocal call_count
            call_count += 1
            self.assertEqual(query.shape[0], 1)
            self.assertEqual(kwargs["kv_splits"], 2)
            return query + call_count

        kernels = _SolAttentionKernels(
            varlen=None,
            bthd=kernel,
            load_error=None,
        )
        with (
            patch.object(impl, "_eligible", return_value=True),
            patch.object(impl, "_resolve_kv_splits", return_value=2),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn."
                "_load_sol_attn_kernels",
                return_value=kernels,
            ),
        ):
            output = self._forward(impl)

        self.assertEqual(call_count, 2)
        torch.testing.assert_close(output[:2], self.query[:2] + 1)
        torch.testing.assert_close(output[2:], self.query[2:] + 2)

    def test_load_failure_falls_back_to_dense(self):
        dense = _FakeDenseImpl(fill=7.0)
        impl = _make_impl(dense=dense)
        kernels = _SolAttentionKernels(
            varlen=None,
            bthd=None,
            load_error=ImportError("not installed"),
        )
        with (
            patch.object(impl, "_eligible", return_value=True),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn."
                "_load_sol_attn_kernels",
                return_value=kernels,
            ) as load_kernels,
        ):
            output = self._forward(impl)
            output_again = self._forward(impl, max_seqlen=4)

        torch.testing.assert_close(output, torch.full_like(self.query, 7.0))
        torch.testing.assert_close(output_again, torch.full_like(self.query, 7.0))
        self.assertEqual(len(dense.varlen_calls), 2)
        load_kernels.assert_called_once()

    def test_recoverable_runtime_failure_disables_sol_for_impl(self):
        kernel = Mock(side_effect=RuntimeError("kernel failed"))
        kernels = _SolAttentionKernels(
            varlen=kernel,
            bthd=None,
            load_error=None,
        )
        non_strict = _make_impl(dense=_FakeDenseImpl(fill=9.0))

        with (
            patch.object(non_strict, "_eligible", return_value=True),
            patch.object(non_strict, "_resolve_kv_splits", return_value=1),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn."
                "_load_sol_attn_kernels",
                return_value=kernels,
            ) as load_kernels,
        ):
            output = self._forward(non_strict)
            output_again = self._forward(non_strict)
            output_other_signature = self._forward(non_strict, max_seqlen=4)
        torch.testing.assert_close(output, torch.full_like(self.query, 9.0))
        torch.testing.assert_close(output_again, torch.full_like(self.query, 9.0))
        torch.testing.assert_close(
            output_other_signature, torch.full_like(self.query, 9.0)
        )
        self.assertEqual(kernel.call_count, 1)
        load_kernels.assert_called_once()
        self.assertEqual(len(non_strict._dense.varlen_calls), 3)
        self.assertEqual(
            non_strict._sol_unavailable_reason, "RuntimeError: kernel failed"
        )

    def test_runtime_failure_obeys_strict(self):
        kernel = Mock(side_effect=RuntimeError("kernel failed"))
        kernels = _SolAttentionKernels(
            varlen=kernel,
            bthd=None,
            load_error=None,
        )
        strict = _make_impl(strict=True)

        with (
            patch.object(strict, "_eligible", return_value=True),
            patch.object(strict, "_resolve_kv_splits", return_value=1),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn."
                "_load_sol_attn_kernels",
                return_value=kernels,
            ),
            self.assertRaisesRegex(RuntimeError, "kernel failed"),
        ):
            self._forward(strict)

    def test_fatal_cuda_errors_do_not_fallback(self):
        errors = (
            torch.cuda.OutOfMemoryError("CUDA out of memory"),
            RuntimeError("CUDA error: an illegal memory access was encountered"),
        )
        for error in errors:
            with self.subTest(error=error):
                dense = _FakeDenseImpl(fill=11.0)
                impl = _make_impl(dense=dense)
                kernels = _SolAttentionKernels(
                    varlen=Mock(side_effect=error),
                    bthd=None,
                    load_error=None,
                )
                with (
                    patch.object(impl, "_eligible", return_value=True),
                    patch.object(impl, "_resolve_kv_splits", return_value=1),
                    patch(
                        "sglang.multimodal_gen.runtime.layers.attention.backends."
                        "sol_attn._load_sol_attn_kernels",
                        return_value=kernels,
                    ),
                    self.assertRaises(type(error)),
                ):
                    self._forward(impl)
                self.assertEqual(dense.varlen_calls, [])

    def test_kv_splits_capability_gates(self):
        auto = _make_impl(kv_splits="auto")
        explicit_two = _make_impl(kv_splits=2)

        with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            self.assertEqual(auto._resolve_kv_splits(self.query, 65536), 1)
            with self.assertRaisesRegex(RuntimeError, "only on SM90"):
                explicit_two._resolve_kv_splits(self.query, 8192)

        with (
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn."
                "_cute_runtime_available",
                return_value=False,
            ),
            self.assertRaisesRegex(RuntimeError, "requires the SM90 CuTe runtime"),
        ):
            explicit_two._resolve_kv_splits(self.query, 8192)

        with (
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn."
                "_cute_runtime_available",
                return_value=True,
            ),
        ):
            self.assertEqual(auto._resolve_kv_splits(self.query, 65536), 4)
            self.assertEqual(explicit_two._resolve_kv_splits(self.query, 8192), 2)
            with self.assertRaisesRegex(ValueError, "N64 route group"):
                explicit_two._resolve_kv_splits(self.query, 4096)


if __name__ == "__main__":
    unittest.main(verbosity=3)
