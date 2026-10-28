"""CPU checks for DeepGEMM warmup buffers and scale conversion workspace."""

import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.deep_gemm_wrapper.compile_utils import (
    DeepGemmKernelType,
    _BaseWarmupExecutor,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _allocate_on_meta(factory):
    def allocate(*args, **kwargs):
        kwargs["device"] = "meta"
        return factory(*args, **kwargs)

    return allocate


def _storage_bytes(*tensors):
    return sum(tensor.untyped_storage().nbytes() for tensor in tensors)


def _pack_scales_on_meta(scales):
    scales = scales.unsqueeze(0) if scales.ndim == 2 else scales
    groups, rows, cols = scales.shape
    aligned_rows = (rows + 3) // 4 * 4
    packed_cols = (cols + 3) // 4
    initial = torch.empty_strided(
        (groups, rows, packed_cols),
        (aligned_rows * packed_cols, 1, aligned_rows),
        dtype=torch.int32,
        device="meta",
    )
    peak = _storage_bytes(initial)
    if groups > 1 and rows * cols % 4:
        shifted = scales.view(torch.int32) >> 23
        unpacked = shifted.to(torch.uint8)
        peak = max(peak, _storage_bytes(initial, shifted, unpacked))
        padded = torch.zeros(
            (groups, aligned_rows, packed_cols * 4), dtype=torch.uint8, device="meta"
        )
        packed = torch.empty_strided(
            (groups, aligned_rows, packed_cols),
            (aligned_rows * packed_cols, 1, aligned_rows),
            dtype=torch.int32,
            device="meta",
        )
        peak = max(peak, _storage_bytes(initial, unpacked, padded, packed))
        return packed, peak
    return initial, peak


def _scale_workspace_on_meta(executor, architecture):
    if not hasattr(executor, "lhs_s") or architecture == "musa":
        return 0
    scales = executor.lhs_s
    if architecture == "hopper":
        scales = scales.unsqueeze(0) if scales.ndim == 2 else scales
        _, rows, cols = scales.shape
        aligned_rows = (rows + 3) // 4 * 4
        workspace = torch.empty_strided(
            scales.shape,
            (aligned_rows * cols, 1, aligned_rows),
            dtype=scales.dtype,
            device="meta",
        )
        return _storage_bytes(workspace)
    lhs_packed, lhs_peak = _pack_scales_on_meta(scales)
    rows = executor.rhs_q.shape[-2]
    indices = torch.arange(rows, device="meta") // 128
    broadcast = executor.rhs_s.index_select(-2, indices)
    _, rhs_peak = _pack_scales_on_meta(broadcast)
    return max(
        lhs_peak,
        _storage_bytes(lhs_packed, broadcast) + max(_storage_bytes(indices), rhs_peak),
    )


class TestWarmupMemory(CustomTestCase):
    def test_budget_covers_executor_buffers(self):
        for architecture in ("hopper", "blackwell", "musa"):
            for max_m, n, k, num_groups in (
                (9, 131, 259, 3),  # Ceil division and grouped packing fallback.
                (16384, 4096, 7168, 32),  # V3 DP8/EP8 startup OOM shape.
            ):
                for kernel_type in DeepGemmKernelType:
                    if (
                        kernel_type == DeepGemmKernelType.GEMM_NT_F8F8BF16_BLOCK32
                        and architecture == "musa"
                    ):
                        continue  # Native block32 is only dispatched on SM90/SM100.
                    with self.subTest(
                        architecture=architecture, kernel_type=kernel_type, max_m=max_m
                    ):
                        with ExitStack() as stack:
                            stack.enter_context(
                                patch.object(
                                    compile_utils,
                                    "DEEPGEMM_NEED_TMA_ALIGNED_SCALES",
                                    architecture == "hopper",
                                )
                            )
                            stack.enter_context(
                                patch.object(
                                    compile_utils,
                                    "DEEPGEMM_SCALE_UE8M0",
                                    architecture == "blackwell",
                                )
                            )
                            for name in ("empty", "ones", "zeros"):
                                factory = getattr(torch, name)
                                stack.enter_context(
                                    patch.object(
                                        torch,
                                        name,
                                        side_effect=_allocate_on_meta(factory),
                                    )
                                )
                            executor = _BaseWarmupExecutor.create(
                                kernel_type,
                                max_m=max_m,
                                n=n,
                                k=k,
                                num_groups=num_groups,
                            )
                            budget_gib = _BaseWarmupExecutor.get_memory_requirement(
                                kernel_type,
                                max_m=max_m,
                                n=n,
                                k=k,
                                num_groups=num_groups,
                            )
                        allocated_bytes = _storage_bytes(
                            *[
                                tensor
                                for tensor in vars(executor).values()
                                if isinstance(tensor, torch.Tensor)
                            ]
                        )
                        peak_bytes = allocated_bytes + _scale_workspace_on_meta(
                            executor, architecture
                        )
                        self.assertGreaterEqual(budget_gib * (1 << 30), peak_bytes)
                        # Upper bounds include the last row's TMA padding.
                        self.assertLess(budget_gib * (1 << 30) - peak_bytes, 64)


class TestPrecompileDefaults(CustomTestCase):
    def _resolve(self, *, startup: bool, precompile=None, fast_warmup=None):
        """compile_utils state after update_deep_gemm_config, starting from the
        import-time values the envs give (None = unset)."""
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ))
            for env, value in (
                (envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE, precompile),
                (envs.SGLANG_JIT_DEEPGEMM_FAST_WARMUP, fast_warmup),
            ):
                os.environ.pop(env.name, None)
                if value is not None:
                    env.set(value)
            for name, value in (
                (
                    "_ENABLE_JIT_DEEPGEMM_PRECOMPILE",
                    envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.get(),
                ),
                ("_FAST_WARMUP", envs.SGLANG_JIT_DEEPGEMM_FAST_WARMUP.get()),
                ("_BUILTIN_M_LIST", []),
                ("_DO_COMPILE_ALL", True),
                ("_IS_FIRST_RANK_ON_NODE", False),
                ("dsv41_sm90_on", lambda subsystem: startup),
                ("get_schedule", lambda: SimpleNamespace(chunked_prefill_size=16384)),
                ("get_device", lambda: SimpleNamespace(base_gpu_id=0)),
            ):
                stack.enter_context(patch.object(compile_utils, name, value))
            compile_utils.update_deep_gemm_config(0)
            return (
                compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE,
                compile_utils._FAST_WARMUP,
                len(compile_utils._BUILTIN_M_LIST),
            )

    def test_startup_switch_changes_only_the_defaults(self):
        """The V4.1 SM90 startup skips the all-M precompile and samples the Ms if it
        runs; an explicit env still wins, and off the switch the upstream defaults stay."""
        # Fast warmup at chunk 16384: Ms 1..1024, then 512 sampled Ms per doubling, then the chunk.
        self.assertEqual(self._resolve(startup=True), (False, True, 3072))
        # All Ms up to twice a chunk above 8192.
        self.assertEqual(
            self._resolve(startup=True, precompile=True, fast_warmup=False),
            (True, False, 32768),
        )
        self.assertEqual(self._resolve(startup=False), (True, False, 32768))


if __name__ == "__main__":
    unittest.main()
