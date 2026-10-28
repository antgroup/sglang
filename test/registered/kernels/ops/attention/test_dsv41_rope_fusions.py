"""DeepSeek-V4.1 SM90 RoPE fusions are bitwise the kernels they replace.

* the wo_a input's inverse RoPE + per-32 FP8 rounding (per head and four heads per
  program, as FP8 operands or rounded bf16) == inverse ``fused_rope_inplace`` +
  ``fp8_block32_quant_mn_major`` / ``fp8_block32_quant_dequant``;
* the row-tiled q RoPE == forward ``fused_rope_inplace``;
* the K norm/RoPE/FlashMLA store with q == ``fused_rope_inplace(q)`` + the store.

Signed zeros of (0, 0) rope pairs are part of the contract.
"""

import types
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

ROWS = (1, 6, 9, 64, 192, 384, 1024, 2051)


def _bits(t):
    return t.view(torch.int16)


def _freqs(n=1 << 14):
    return torch.polar(
        torch.ones(n, 32, device="cuda"), torch.randn(n, 32, device="cuda")
    )


def _attn_out(rows, padded_heads):
    """Spread exponents, -0 entries and (0, 0) rope pairs of every sign."""
    x = torch.randn(rows, padded_heads, 512, device="cuda")
    x = x * torch.exp2(torch.randint(-24, 16, x.shape, device="cuda").float())
    x[:, :, 1::41] = -0.0
    x[:, :, 448:452] = torch.tensor([0.0, -0.0, -0.0, 0.0], device="cuda")
    x[:, :, 508:512] = torch.tensor([0.0, 0.0, -0.0, -0.0], device="cuda")
    return x.bfloat16()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0),
    "needs SM90+",
)
class TestDsv41RopeFusions(CustomTestCase):
    def test_wo_a_rope_rounding_forms(self):
        from sglang.kernels.ops.attention.dsv4 import fused_rope_inplace
        from sglang.kernels.ops.attention.dsv4.wo_a_rope_quant import (
            wo_a_rope_qdq,
            wo_a_rope_qdq_rows,
            wo_a_rope_quant_block32,
            wo_a_rope_quant_block32_rows,
        )
        from sglang.kernels.ops.quantization.fp8_block32_qdq import (
            fp8_block32_quant_dequant,
            fp8_block32_quant_mn_major,
        )

        torch.manual_seed(3)
        freqs = _freqs()
        # 2051: scale rows padded to 4; from 1024 rows the quant takes the
        # reciprocal form, which the fused kernels always use.
        for m in ROWS:
            for heads in (8, 64):
                with self.subTest(m=m, heads=heads):
                    # The real heads are a strided view of a head-padded buffer.
                    buf = _attn_out(m, 72)
                    o = buf[:, :heads]
                    before = buf.clone()
                    pos = torch.randint(0, freqs.shape[0], (m,), device="cuda")
                    ref = o.clone()
                    fused_rope_inplace(
                        ref[..., 448:], None, freqs, positions=pos, inverse=True
                    )
                    groups = heads // 8
                    ref_qdq = fp8_block32_quant_dequant(ref.view(m, groups, -1))
                    ref_q, ref_s = fp8_block32_quant_mn_major(ref.view(m, groups, -1))
                    for qdq in (wo_a_rope_qdq, wo_a_rope_qdq_rows):
                        got = qdq(o, freqs, pos).view(m, groups, -1)
                        self.assertTrue(torch.equal(_bits(got), _bits(ref_qdq)))
                    for quant in (
                        wo_a_rope_quant_block32,
                        wo_a_rope_quant_block32_rows,
                    ):
                        got_q, got_s = quant(o, freqs, pos)
                        self.assertTrue(
                            torch.equal(
                                got_q.view(torch.uint8), ref_q.view(torch.uint8)
                            )
                        )
                        self.assertTrue(torch.equal(got_s, ref_s))
                        self.assertEqual(got_s.stride(), ref_s.stride())
                    self.assertTrue(torch.equal(_bits(buf), _bits(before)))

    def test_q_rope_rows(self):
        from sglang.kernels.ops.attention.dsv4 import fused_rope_inplace
        from sglang.kernels.ops.attention.dsv4.q_rope_rows import (
            q_rope_rows_inplace,
        )

        torch.manual_seed(4)
        freqs = _freqs()
        for m in ROWS:
            for heads in (8, 16, 64):
                with self.subTest(m=m, heads=heads):
                    q = _attn_out(m, heads)
                    pos = torch.randint(0, freqs.shape[0], (m,), device="cuda")
                    ref = q.clone()
                    fused_rope_inplace(ref[..., 448:], None, freqs, positions=pos)
                    q_rope_rows_inplace(q, freqs, pos)
                    self.assertTrue(torch.equal(_bits(q), _bits(ref)))

    def test_k_store_with_q_rope(self):
        from sglang.kernels.ops.attention.dsv4 import (
            fused_k_norm_rope_flashmla,
            fused_rope_inplace,
        )
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

        torch.manual_seed(6)
        freqs = _freqs()
        page_bytes = KVLayout.V4.page_bytes(128)
        for m in (1, 9, 192, 384):
            for heads in (8, 64):
                for pos_dtype in (torch.int32, torch.int64):
                    with self.subTest(m=m, heads=heads, pos_dtype=pos_dtype):
                        # kv is the strided tail of the fused q/kv projection.
                        kv = torch.randn(m, 1280 + 512, device="cuda").bfloat16()
                        kv = kv[:, 1280:]
                        weight = (torch.rand(512, device="cuda") + 0.5).bfloat16()
                        q = _attn_out(m, heads)
                        pos = torch.randint(0, freqs.shape[0], (m,), device="cuda")
                        pos = pos.to(pos_dtype)
                        # A row without a KV slot still ropes its query heads.
                        loc = torch.randperm(16 * 128, device="cuda")[:m].int()
                        loc[m // 2] = -1
                        store = dict(
                            kv=kv,
                            kv_weight=weight,
                            eps=1e-6,
                            freqs_cis=freqs,
                            positions=pos,
                            out_loc=loc,
                            page_size=128,
                            layout=KVLayout.V4,
                        )
                        ref_cache = torch.zeros(
                            16, page_bytes, dtype=torch.uint8, device="cuda"
                        )
                        cache = ref_cache.clone()
                        ref_q = q.clone()
                        fused_rope_inplace(ref_q[..., 448:], None, freqs, positions=pos)
                        fused_k_norm_rope_flashmla(**store, kvcache=ref_cache)
                        fused_k_norm_rope_flashmla(**store, kvcache=cache, q=q)
                        self.assertTrue(torch.equal(cache, ref_cache))
                        self.assertTrue(torch.equal(_bits(q), _bits(ref_q)))

    def test_draft_wo_a_input_takes_the_reference_rounding(self):
        """The V4.1 reference rounds the DSpark draft's wo_a input through FP8 too;
        with attn_proj off the draft keeps upstream's unrounded input."""
        from sglang.kernels.ops.attention.dsv4 import fused_rope_inplace
        from sglang.kernels.ops.quantization.fp8_block32_qdq import (
            fp8_block32_quant_dequant,
        )
        from sglang.srt.models.deepseek_v4_attn_proj import AttnProjPlan
        from sglang.srt.models.deepseek_v4_dspark import DSparkAttention

        torch.manual_seed(5)
        freqs = _freqs()
        for plan_role in (None, "mix", "decode"):
            plan = None
            if plan_role is not None:
                plan = AttnProjPlan(
                    wo_a_fp8_block32=False,
                    decode_rope_rows=plan_role == "decode",
                    kv_store_q_rope=False,
                    routes=None,
                    streams=(),
                )
            for m in (5, 192):
                with self.subTest(role=plan_role, m=m):
                    layer = types.SimpleNamespace(
                        rope_head_dim=64,
                        _use_fast_kernel=True,
                        attn_proj=plan,
                        freqs_cis=freqs,
                        n_local_groups=8,
                    )
                    o = _attn_out(m, 64)
                    pos = torch.randint(0, freqs.shape[0], (m,), device="cuda")
                    ref = o.clone()
                    fused_rope_inplace(
                        ref[..., 448:], None, freqs, positions=pos, inverse=True
                    )
                    ref = ref.view(m, 8, -1)
                    if plan is not None:
                        ref = fp8_block32_quant_dequant(ref)
                    got = DSparkAttention._wo_a_input(layer, o, pos)
                    self.assertTrue(torch.equal(_bits(got), _bits(ref)))


if __name__ == "__main__":
    unittest.main()
