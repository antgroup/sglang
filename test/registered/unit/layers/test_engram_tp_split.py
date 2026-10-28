"""Engram TP split (DeepSeek-V4.1 SM90 mix decode): exchange bookkeeping, mode
choice and gating (CPU).

Ranks are threads over in-memory collectives; each holds one shard of the table.
wkv is an elementwise stand-in whose column slices are a rank's columns, and the
gate kernels are replaced by engram_gate's torch arithmetic, so every split mode
and every row slice must rebuild the replicated lookup + wkv + gate bit for bit."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import engram as engram_mod
from sglang.srt.layers.engram import (
    Engram,
    EngramEmbedding,
    _EngramSplit,
    _tp_split_on,
    _wkv_splits_bitwise,
    engram_gate,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

WORLD, HC, DIM, COLS, HEAD_DIM, BLK = 4, 3, 32, 3, 32, 32
KV_COLS = (HC + 1) * DIM  # 128: one 32-column block per rank
# Three hash columns of unequal prime sizes; 3 does not divide over 4 ranks, so
# shard_range takes the equal-row split and one rank owns rows of two columns.
COLUMN_BOUNDS = (0, 17, 36, 53)
TABLE_ROWS = COLUMN_BOUNDS[-1]
IMAGE_ID = 7
EPS, CLAMP = 1e-6, 1e-6


class _Mailbox:
    def __init__(self):
        self.slots = [None] * WORLD
        self.barrier = threading.Barrier(WORLD, timeout=30)

    def exchange(self, rank, tensor):
        self.slots[rank] = tensor.clone()
        self.barrier.wait()
        slots = list(self.slots)
        self.barrier.wait()
        return slots


class _FakeTp:
    def __init__(self, box, rank):
        self.box, self.rank_in_group, self.world_size = box, rank, WORLD

    def all_reduce(self, inp):
        total = sum(s.float() for s in self.box.exchange(self.rank_in_group, inp))
        return total.to(inp.dtype)

    def reduce_scatter_tensor(self, out, inp):
        total = self.all_reduce(inp)
        n = out.shape[0]
        out.copy_(total[self.rank_in_group * n : (self.rank_in_group + 1) * n])

    def all_gather_into_tensor(self, out, inp):
        out.copy_(torch.cat(self.box.exchange(self.rank_in_group, inp)))

    def all_to_all_single(self, out, inp):
        slots = self.box.exchange(self.rank_in_group, inp)
        n = inp.numel() // WORLD
        r = self.rank_in_group
        out.copy_(torch.cat([s[r * n : (r + 1) * n] for s in slots]))


def _table():
    g = torch.Generator().manual_seed(0)
    weight = (torch.randn(TABLE_ROWS, HEAD_DIM, generator=g) * 3).to(
        torch.float8_e4m3fn
    )
    scale = torch.randint(
        110, 135, (TABLE_ROWS, HEAD_DIM // BLK), dtype=torch.uint8, generator=g
    ).view(torch.float8_e8m0fnu)
    return weight, scale


def _elementwise_linear(
    input, weight, block_size=None, weight_scale=None, input_scale=None, bias=None
):
    """Column n is input column weight[n, 2] scaled by weight[n, 0] and shifted by
    weight[n, 1]: exact at any row count and for any column slice."""
    out = input.float()[:, weight[:, 2].long()] * weight[:, 0] + weight[:, 1]
    return out.to(torch.bfloat16)


class _ElementwiseWkv:
    """Stand-in for the block32 wkv; _wkv_columns slices its weight and scales."""

    def __init__(self):
        g = torch.Generator().manual_seed(1)
        k = COLS * HEAD_DIM
        self.weight = torch.stack(
            [
                torch.rand(KV_COLS, generator=g) + 0.5,
                torch.randn(KV_COLS, generator=g),
                torch.randint(0, k, (KV_COLS,), generator=g).float(),
            ],
            dim=1,
        )
        self.weight_scale_inv = torch.ones(KV_COLS // BLK, k // BLK)
        self.quant_method = SimpleNamespace(
            w8a8_block_fp8_linear=_elementwise_linear, weight_block_size=[BLK, BLK]
        )

    def __call__(self, emb_flat):
        return _elementwise_linear(input=emb_flat, weight=self.weight), None


def _pack(x, kv, q_weight, k_weight, eps, clamp_value, out):
    """engram_gate's torch arithmetic, stored as [value | fp32 gate per hc copy];
    kv is [rows, cols] or column chunks [chunks, rows, cols / chunks]."""
    t = x.shape[0]
    if kv.ndim == 3:
        kv = kv[:, :t].permute(1, 0, 2).reshape(-1, KV_COLS)
    kv = kv[:t]
    key = kv[:, : HC * DIM].float().unflatten(-1, (HC, DIM))
    h, weight = x.float(), q_weight.float() * k_weight.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * DIM**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    out[:, :DIM] = kv[:, HC * DIM :]
    out[:, DIM:].view(torch.float32)[:, :HC] = gate
    return out


def _apply(x, packed, input_ids, image_token_id):
    gate = packed[:, DIM:].view(torch.float32)[:, :HC]
    value = packed[:, :DIM].float()
    out = (x.float() + gate.unsqueeze(-1) * value.unsqueeze(-2)).to(x.dtype)
    if input_ids is not None:
        out = torch.where((input_ids == image_token_id)[:, None, None], x, out)
    return out


def _layer(rank, weight, scale, q_weight, k_weight, *, row_split_ok=True):
    """Rank `rank`'s Engram: its real table shard, the stand-in wkv."""
    embed = EngramEmbedding(TABLE_ROWS, HEAD_DIM, 1, COLUMN_BOUNDS)
    embed.weight.weight_loader(embed.weight, weight)
    embed.scale.weight_loader(embed.scale, scale)
    layer = Engram.__new__(Engram)
    torch.nn.Module.__init__(layer)
    layer.embed, layer.wkv = embed, _ElementwiseWkv()
    layer.q_weight, layer.k_weight = q_weight, k_weight
    layer.eps, layer.clamp_value, layer.image_token_id = EPS, CLAMP, IMAGE_ID
    layer._row_split_ok, layer._col_split_ok = row_split_ok, True
    return layer


def _inputs(rows):
    torch.manual_seed(rows)
    x = torch.randn(rows, HC, DIM).to(torch.bfloat16)
    # The model passes a strided [rows, cols] slice of [rows, n_layers, cols].
    ids = torch.randint(0, TABLE_ROWS, (rows, 2, COLS))[:, 1]
    input_ids = torch.randint(0, 12, (rows,))
    input_ids[::3] = IMAGE_ID
    q_weight, k_weight = torch.randn(2, HC, DIM).bfloat16()
    return x, ids, input_ids, q_weight, k_weight


def _run_ranks(rank_fn):
    box, local = _Mailbox(), threading.local()

    def run(rank):
        local.tp = _FakeTp(box, rank)
        local.par = SimpleNamespace(
            tp_size=WORLD, tp_rank=rank, tp_group=local.tp, attn_dp_size=1
        )
        return rank_fn(rank)

    with (
        patch.object(engram_mod, "get_parallel", side_effect=lambda: local.par),
        patch.object(
            engram_mod,
            "tensor_model_parallel_all_reduce",
            side_effect=lambda t: local.tp.all_reduce(t),
        ),
        patch.object(engram_mod, "fused_engram_gate_pack", _pack),
        patch.object(engram_mod, "fused_engram_gate_apply", _apply),
    ):
        with ThreadPoolExecutor(WORLD) as pool:
            # result() re-raises a rank's exception; the barrier timeout frees the rest.
            return [f.result() for f in [pool.submit(run, r) for r in range(WORLD)]]


def _reference(x, ids, input_ids, q_weight, k_weight):
    """The replicated path: whole table on one rank, every row projected and gated."""
    weight, scale = _table()
    kv, _ = _ElementwiseWkv()(_dequant(weight, scale, ids).flatten(-2))
    return engram_gate(
        x, kv, q_weight, k_weight, EPS, CLAMP, input_ids, image_token_id=IMAGE_ID
    )


def _bits(t):
    return t.view(torch.int16)


def _dequant(weight, scale, ids):
    rows = weight[ids].float().unflatten(-1, (-1, BLK))
    return (rows * scale[ids].float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


class TestEngramSplitExchange(CustomTestCase):
    def test_every_mode_rebuilds_the_replicated_rows_on_every_rank(self):
        # Divisible, uneven, and fewer rows than ranks (some ranks own none);
        # every third row is an image token and must keep x.
        for mode in _EngramSplit:
            for rows in (64, 65, 102, 5, 1):
                with self.subTest(mode=mode.name, rows=rows):
                    x, ids, input_ids, qw, kw = _inputs(rows)
                    weight, scale = _table()

                    def rank_fn(rank):
                        layer = _layer(rank, weight, scale, qw, kw)
                        return layer._forward_split(
                            x, ids, input_ids=input_ids, mode=mode
                        )

                    ref = _reference(x, ids, input_ids, qw, kw)
                    self.assertTrue(torch.equal(_bits(ref[::3]), _bits(x[::3])))
                    for rank, out in enumerate(_run_ranks(rank_fn)):
                        self.assertTrue(
                            torch.equal(_bits(out), _bits(ref)), f"rank {rank}"
                        )

    def test_row_slices_rebuild_the_replicated_rows(self):
        """forward_row_slice on each rank's [start, start + chunk) rows, with padding
        rows past the batch (a rank whose slice is all padding included), with the
        row-invariant lookup slice and with the project-every-row fallback."""
        for row_split_ok in (True, False):
            for rows, chunk in ((64, 16), (65, 17), (5, 2), (3, 1)):
                with self.subTest(row_split_ok=row_split_ok, rows=rows, chunk=chunk):
                    x, ids, input_ids, qw, kw = _inputs(rows)
                    weight, scale = _table()
                    x_pad = torch.cat([x, torch.randn(chunk * WORLD - rows, HC, DIM)])
                    x_pad = x_pad.to(torch.bfloat16)

                    def rank_fn(rank):
                        layer = _layer(
                            rank, weight, scale, qw, kw, row_split_ok=row_split_ok
                        )
                        start = rank * chunk
                        return layer.forward_row_slice(
                            x_pad[start : start + chunk].clone(),
                            ids,
                            start=start,
                            chunk=chunk,
                            world=WORLD,
                            input_ids=input_ids,
                        )

                    out = torch.cat(_run_ranks(rank_fn))
                    ref = _reference(x, ids, input_ids, qw, kw)
                    self.assertTrue(torch.equal(_bits(out[:rows]), _bits(ref)))
                    self.assertTrue(torch.equal(_bits(out[rows:]), _bits(x_pad[rows:])))


def _mode(rows, *, row_ok=True, col_ok=True):
    layer = SimpleNamespace(_row_split_ok=row_ok, _col_split_ok=col_ok)
    layer._wkv_split_ok = partial(Engram._wkv_split_ok, layer)
    x = SimpleNamespace(
        shape=(rows, HC, DIM),
        is_cuda=True,
        dtype=torch.bfloat16,
        is_contiguous=lambda: True,
    )
    fb = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    return Engram._split_mode(layer, x, fb)


class TestEngramSplitMode(CustomTestCase):
    def test_mode_by_row_count(self):
        # H20-3e TP4 measurements: column all-gather below 64 rows, column
        # all-to-all up to 512, the row split from 65 rows otherwise.
        S = _EngramSplit
        self.assertEqual(_mode(1), S.COL_GATHER)
        self.assertEqual(_mode(63), S.COL_GATHER)
        self.assertEqual(_mode(64), S.COL_A2A)
        self.assertEqual(_mode(512), S.COL_A2A)
        self.assertEqual(_mode(513), S.ROW)
        # wkv columns not splittable: the row split from 65 rows, else replicated.
        self.assertIsNone(_mode(64, col_ok=False))
        self.assertEqual(_mode(65, col_ok=False), S.ROW)
        self.assertIsNone(_mode(513, row_ok=False))
        self.assertIsNone(_mode(384, row_ok=False, col_ok=False))

    def test_only_replicated_decode_and_verify_rows_split(self):
        layer = SimpleNamespace(_row_split_ok=True, _col_split_ok=True)
        layer._wkv_split_ok = partial(Engram._wkv_split_ok, layer)

        def mode(forward_mode, *, rows=96, dtype=torch.bfloat16, cuda=True):
            x = SimpleNamespace(
                shape=(rows, HC, DIM),
                is_cuda=cuda,
                dtype=dtype,
                is_contiguous=lambda: True,
            )
            fb = (
                None
                if forward_mode is None
                else SimpleNamespace(forward_mode=forward_mode)
            )
            return Engram._split_mode(layer, x, fb)

        self.assertIsNotNone(mode(ForwardMode.DECODE))
        self.assertIsNotNone(mode(ForwardMode.TARGET_VERIFY))
        for forward_mode in (ForwardMode.EXTEND, ForwardMode.IDLE, None):
            self.assertIsNone(mode(forward_mode), forward_mode)
        self.assertIsNone(mode(ForwardMode.DECODE, rows=0))
        # The fused gate kernels would not serve these rows.
        self.assertIsNone(mode(ForwardMode.DECODE, dtype=torch.float16))
        self.assertIsNone(mode(ForwardMode.DECODE, cuda=False))


class TestEngramSplitGate(CustomTestCase):
    def test_only_sm90_mix_tp_without_all_m_precompile(self):
        def on(*, subsystem=True, role="mix", precompile=False, tp=8, dp=1):
            with (
                patch.object(engram_mod, "dsv41_sm90_on", return_value=subsystem),
                patch.object(engram_mod, "dsv41_sm90_role", return_value=role),
                patch(
                    "sglang.srt.layers.deep_gemm_wrapper.compile_utils."
                    "_ENABLE_JIT_DEEPGEMM_PRECOMPILE",
                    precompile,
                ),
                patch.object(
                    engram_mod,
                    "get_parallel",
                    return_value=SimpleNamespace(tp_rank=1),
                ),
            ):
                return _tp_split_on(tp_size=tp, attn_dp_size=dp)

        self.assertTrue(on())
        self.assertFalse(on(subsystem=False))
        self.assertFalse(on(role="decode"))
        self.assertFalse(on(role="prefill"))
        # The cross-rank compile lock of the all-M precompile vs the collectives.
        self.assertFalse(on(precompile=True))
        self.assertFalse(on(tp=1))
        # DP attention: ranks hold different rows.
        self.assertFalse(on(dp=8))


def _fp8_wkv(n, k, linear):
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    method.block_quant, method.weight_block_size = True, [32, 32]
    method.use_marlin = method.use_mxfp8 = method.block_fp8_as_mxfp8 = False
    method.w8a8_block_fp8_linear = linear
    return SimpleNamespace(
        quant_method=method,
        bias=None,
        weight=torch.empty(n, k, dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.empty(n // 32, k // 32),
    )


class TestWkvSplitEligibility(CustomTestCase):
    def test_routes_that_change_bits_refuse_the_split(self):
        from sglang.kernels.ops.gemm.fp8_block32_swapab import parse_swapab_table
        from sglang.srt.layers.quantization.fp8_block32_sm90 import (
            Block32Routes,
            load_block32_routes,
        )
        from sglang.srt.layers.quantization.fp8_utils import (
            deepgemm_w8a8_block_fp8_linear_with_fallback as dg,
        )

        def ok(n, world, routes, k=6144):
            linear = partial(dg, act_scale_ue8m0=True, block32_routes=routes)
            return _wkv_splits_bitwise(_fp8_wkv(n, k, linear), world)

        # The shipped H20 mix tables keep the V4.1 wkv and its TP2/4/8 column
        # slices on DeepGEMM bits; TP16 slices (1600 columns) take small-M Triton.
        for device in ("NVIDIA H20", "NVIDIA H20-3e"):
            routes = load_block32_routes(device_name=device, role="mix")
            for world in (1, 2, 4, 8):
                self.assertTrue(ok(25600, world, routes), (device, world))
            self.assertFalse(ok(25600, 16, routes), device)
        # A split-K swap-AB row for the TP4 slice sums K partials in another order.
        split_k = {"bt1": 64, "bt2": 0, "stages": 3, "split": 4, "maxnreg": 80}
        routes = Block32Routes(
            swapab=parse_swapab_table({"N=6400,K=6144": {"48": split_k}}),
            triton_small_m={},
            triton_max_m=16,
        )
        self.assertFalse(ok(25600, 4, routes))
        self.assertTrue(ok(25600, 1, routes))
        self.assertTrue(ok(25600, 8, routes))
        # Upstream's SM90 DeepGEMM route without the tables; a non-DeepGEMM linear.
        self.assertTrue(ok(25600, 4, None))
        wkv = _fp8_wkv(25600, 6144, partial(lambda **kw: None))
        self.assertFalse(_wkv_splits_bitwise(wkv, 4))
        # Column slices must cover whole 32-row scale blocks.
        self.assertFalse(ok(25600 + 32, 4, None))


if __name__ == "__main__":
    unittest.main()
