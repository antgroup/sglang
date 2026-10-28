"""SM90 block32 FP8 GEMMs of the DeepSeek-V4.1 routes: swap-AB and split-K Triton."""

import glob
import json
import os
import unittest

import torch
import triton

from sglang.kernels.ops.gemm import fp8_kernel
from sglang.kernels.ops.gemm.fp8_block32_swapab import block32_swapab_gemm, num_tiles
from sglang.kernels.ops.quantization.fp8_block32_act_quant import (
    block32_act_quant_ue8m0_mn_major,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=150, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_CONFIGS = os.path.join(os.path.dirname(fp8_kernel.__file__), "configs")


def _block32_weight(n, k, gen):
    w = torch.randn(n, k, device="cuda", generator=gen) * 0.02
    blocks = w.view(n // 32, 32, k // 32, 32)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448)))
    q = (blocks / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(n, k), scale.float().contiguous()


def _activation(m, k, gen):
    # Every 32-wide group's power-of-two scale differs from those of the groups
    # one and two 128-wide K blocks later, so a stale scale changes the output.
    x = torch.randn(m, k, device="cuda", generator=gen)
    group_exp = torch.arange(k // 32, device="cuda") % 5 - 2
    x = x * torch.exp2(group_exp.float()).repeat_interleave(32)
    return block32_act_quant_ue8m0_mn_major(x.to(torch.bfloat16))


def _shipped_split1_rows():
    """(n, k, groups, m, launch config) of every distinct split-1 launch config in
    the shipped tables, at its listed row count."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(_CONFIGS, "block32_swapab,*.json"))):
        with open(path) as f:
            for shape, entries in json.load(f).items():
                parts = dict(p.split("=") for p in shape.split(","))
                k, groups = int(parts["K"]), int(parts.get("G", 1))
                for m, cfg in entries.items():
                    if cfg is not None and cfg["split"] == 1:
                        key = (k, groups, cfg["bt1"], cfg["bt2"], cfg["stages"])
                        rows.setdefault(key, (int(parts["N"]), k, groups, int(m), cfg))
    return list(rows.values())


class TestBlock32SwapAB(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            raise unittest.SkipTest("Requires SM90 (Hopper)")

    def test_split1_rows_are_bitwise_deepgemm(self):
        """Split 1 promotes the same fp32 partials in the same order as DeepGEMM's
        block32 recipe, for every launch config the tables ship."""
        import deep_gemm

        gen = torch.Generator(device="cuda").manual_seed(1027)
        cases = _shipped_split1_rows()
        self.assertGreater(len(cases), 20)
        for n, k, groups, m, cfg in cases:
            with self.subTest(n=n, k=k, groups=groups, m=m, **cfg):
                # Two weight tiles per group keep the DeepGEMM compile small.
                group_n = 128 if groups > 1 else n
                w, w_scale = _block32_weight(group_n * groups, k, gen)
                q, x_scale = _activation(m, groups * k, gen)
                got = block32_swapab_gemm(
                    q,
                    x_scale,
                    w,
                    w_scale,
                    counters=None,
                    group_n=group_n if groups > 1 else 0,
                    **cfg,
                )
                ref = torch.empty_like(got)
                for g in range(groups):
                    rows = slice(g * group_n, (g + 1) * group_n)
                    deep_gemm.fp8_gemm_nt(
                        (
                            q[:, g * k : (g + 1) * k],
                            x_scale[:, g * k // 32 : (g + 1) * k // 32],
                        ),
                        (w[rows], w_scale[g * group_n // 32 : (g + 1) * group_n // 32]),
                        ref[:, rows],
                        recipe=(1, 32, 32),
                    )
                self.assertTrue(
                    torch.equal(got.view(torch.int16), ref.view(torch.int16))
                )

    def test_back_to_back_launches_match_an_isolated_launch(self):
        """Back-to-back launches must give an isolated launch's bits: a refilled TMA
        stage must not reach a consumer warp's still-pending activation-scale read."""
        cases = (
            # The DSpark draft fc at its decode rows (split-K), a split-1 table row,
            # and a 3-stage ring; each failed on the unfenced kernel.
            (5120, 15360, 60, dict(bt1=32, bt2=16, stages=2, split=5, maxnreg=64)),
            (5120, 15360, 96, dict(bt1=32, bt2=16, stages=3, split=5, maxnreg=64)),
            (8192, 1280, 320, dict(bt1=64, bt2=0, stages=3, split=1, maxnreg=80)),
        )
        gen = torch.Generator(device="cuda").manual_seed(1022)
        for n, k, m, cfg in cases:
            with self.subTest(n=n, k=k, m=m, **cfg):
                w, w_scale = _block32_weight(n, k, gen)
                inputs = [_activation(m, k, gen) for _ in range(2)]
                counters = None
                if cfg["split"] > 1:
                    tiles = num_tiles(n, m, cfg["bt1"], cfg["bt2"])
                    counters = torch.zeros(tiles, dtype=torch.int32, device="cuda")

                def launch(i):
                    q, x_scale = inputs[i % 2]
                    return block32_swapab_gemm(
                        q, x_scale, w, w_scale, counters=counters, **cfg
                    )

                goldens = []
                for i in range(2):
                    torch.cuda.synchronize()
                    goldens.append(launch(i))
                    torch.cuda.synchronize()
                mismatches = 0
                for start in range(0, 3000, 64):
                    outs = [launch(i) for i in range(start, start + 64)]
                    torch.cuda.synchronize()
                    mismatches += sum(
                        not torch.equal(out, goldens[(start + j) % 2])
                        for j, out in enumerate(outs)
                    )
                self.assertEqual(mismatches, 0)
                if counters is not None:
                    self.assertEqual(int(counters.count_nonzero()), 0)


class TestFixedOrderSplitK(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Requires CUDA")

    def test_sums_even_then_odd_partials_in_split_order(self):
        """The split-K sum must not follow the Triton version's tl.sum lowering:
        The reference summation order is (p0 + p2 + p4 + p6) + (p1 + p3 + p5 + p7)."""
        torch.manual_seed(5)
        for splits, elements in ((8, 1536 * 6), (8, 1000), (4, 24576)):
            with self.subTest(splits=splits, elements=elements):
                # Spread exponents so that every other order gives other fp32 bits.
                parts = torch.randn(splits, elements, device="cuda")
                exps = torch.randint(-20, 20, parts.shape, device="cuda")
                parts = parts * torch.exp2(exps.float())
                out = torch.empty(elements, device="cuda")
                fp8_kernel._reduce_block_fp8_split_k_fixed_order[
                    (triton.cdiv(elements, 256),)
                ](parts, out, elements, splits, 256)
                even, odd = parts[0].clone(), parts[1].clone()
                for split in range(2, splits, 2):
                    even += parts[split]
                    odd += parts[split + 1]
                self.assertTrue(torch.equal(out, even + odd))


if __name__ == "__main__":
    unittest.main()
