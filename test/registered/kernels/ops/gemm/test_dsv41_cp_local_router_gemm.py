"""The DeepSeek-V4.1 CP-local router GEMM shapes (deepseek_v4_cp_prefill).

Under prefill CP the router may run on a rank's own rows instead of the gathered
CP * rows; routing is per row, so that is bitwise the gathered routing exactly
when the 384 x 5120 bf16 -> fp32 router GEMM gives every row the same logits at
both row counts. Each listed (rows, CP size) must, on each listed device, with
the GEMMs linear_bf16_fp32 dispatches to (cuBLAS, DeepGEMM). The VL-biased layers
of a checkpoint with a vision tower then take the per-row vision top-k.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.gemm.bf16_fp32 import _linear_bf16_fp32_cublas
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import MoeRunnerBackend, initialize_moe_config
from sglang.srt.models import deepseek_v4_cp_prefill as cp_prefill
from sglang.srt.models.deepseek_v4_cp_prefill import (
    _LOCAL_ROUTER_DEVICES,
    _LOCAL_ROUTER_ROWS,
    _LOCAL_ROUTER_WEIGHT_SHAPE,
)
from sglang.srt.multimodal.dsv41.vl_routing import vision_topk
from sglang.srt.runtime_context import get_context, get_model
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-c", runner_config="8-gpu-h20")


def _deep_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    from sglang.srt.layers import deep_gemm_wrapper

    out = torch.empty(x.shape[0], w.shape[0], dtype=torch.float32, device=x.device)
    deep_gemm_wrapper.gemm_nt_bf16bf16f32(x, w, out)
    return out


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() in _LOCAL_ROUTER_DEVICES,
    "the local-router shapes are validated per device",
)
class TestCpLocalRouterGemm(CustomTestCase):
    def test_local_rows_give_the_gathered_logits(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        n, k = _LOCAL_ROUTER_WEIGHT_SHAPE
        w = (torch.randn(n, k, device="cuda", generator=gen) * k**-0.5).bfloat16()
        for name, gemm in (
            ("cublas", _linear_bf16_fp32_cublas),
            ("deep_gemm", _deep_gemm),
        ):
            for (rows, cp), route_rows in _LOCAL_ROUTER_ROWS.items():
                x = torch.randn(cp * rows, k, device="cuda", generator=gen).bfloat16()
                gathered = gemm(x, w)
                for r in range(cp):
                    local = x[r * rows : (r + 1) * rows]
                    local = local.repeat(route_rows // rows, 1)
                    with self.subTest(gemm=name, rows=rows, cp=cp, rank=r):
                        self.assertTrue(
                            torch.equal(
                                gemm(local, w)[:rows].view(torch.int32),
                                gathered[r * rows : (r + 1) * rows].view(torch.int32),
                            )
                        )

    def test_vl_biased_rows_route_like_the_gathered_rows(self):
        """A VL-biased MoE's local routing (router GEMM, vision top-k over the
        rank's block of the rank-major ids) gives this rank's rows of the gathered
        routing bit for bit, image-token rows included."""
        gen = torch.Generator(device="cuda").manual_seed(1)
        n, k = _LOCAL_ROUTER_WEIGHT_SHAPE
        image_token = 129264
        w = (torch.randn(n, k, device="cuda", generator=gen) * k**-0.5).bfloat16()
        gate = _Gate(
            weight=w,
            bias=torch.randn(n, device="cuda", generator=gen),
            bias_vl=torch.randn(n, device="cuda", generator=gen),
        )
        config = get_context().override_server_args()
        config.install()
        self.addCleanup(config.restore)
        initialize_moe_config()
        with get_model().override(
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=0)
        ):
            # V4.1-Flash's router (text_config): 6 of 384, renormalized, x1.5.
            topk = TopK(
                top_k=6,
                layer_id=3,
                renormalize=True,
                scoring_func="sqrtsoftplus",
                correction_bias=gate.e_score_correction_bias,
                routed_scaling_factor=1.5,
            )
        topk.bind_runner_backend(MoeRunnerBackend.HUMMING)
        mlp = SimpleNamespace(
            gate=gate,
            is_hash=False,
            layer_id=3,
            config=SimpleNamespace(image_token_id=image_token),
            topk=topk,
        )
        for (rows, cp), route_rows in _LOCAL_ROUTER_ROWS.items():
            x = torch.randn(cp * rows, k, device="cuda", generator=gen).bfloat16()
            ids = torch.randint(0, 129000, (cp * rows,), device="cuda", generator=gen)
            image = torch.rand(cp * rows, device="cuda", generator=gen) < 0.3
            ids = torch.where(image, image_token, ids)
            full = vision_topk(mlp, gate(x), ids)
            for r in range(cp):
                parallel = SimpleNamespace(attn_cp_size=cp, attn_cp_rank=r)
                with (
                    self.subTest(rows=rows, cp=cp, rank=r),
                    patch.object(cp_prefill, "get_parallel", return_value=parallel),
                ):
                    local_ids, weights = cp_prefill._route_local_rows(
                        mlp, x[r * rows : (r + 1) * rows], ids, route_rows=route_rows
                    )
                    own = slice(r * rows, (r + 1) * rows)
                    self.assertTrue(torch.equal(local_ids, full.topk_ids[own]))
                    self.assertTrue(
                        torch.equal(
                            weights.view(torch.int32),
                            full.topk_weights[own].view(torch.int32),
                        )
                    )


class _Gate:
    """The router gate as the MoE calls it: the cuBLAS bf16 -> fp32 GEMM."""

    def __init__(self, *, weight, bias, bias_vl):
        self.weight = weight
        self.e_score_correction_bias = bias
        self.e_score_correction_bias_vl = bias_vl

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _linear_bf16_fp32_cublas(x, self.weight)


if __name__ == "__main__":
    unittest.main()
