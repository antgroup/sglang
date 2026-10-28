"""The SM90 decode attention head<->row all-to-all (deepseek_v4_head_a2a) on CPU:
the gate, the exchange, the row-sliced metadata and the row-q projections,
without FlashMLA.

One thread per rank runs the exchange over an in-memory all-to-all. The stand-in
attention is a function of the global row, the global head, that head's sink and
the row's metadata, so the reassembled output must equal applying it to all rows
and heads at once."""

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention import deepseek_v4_backend
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v4_head_a2a as head_a2a
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

DIM = 16
_DUMMY = -7  # pad-row metadata value


class _FakeGroup:
    """all_to_all_single over flattened tensors, as pynccl does it."""

    def __init__(self, slots, barrier, rank):
        self.slots, self.barrier = slots, barrier
        self.world_size, self.rank_in_group = len(slots), rank
        self.pynccl_comm = object()

    def all_to_all_single(self, output, input):
        assert output.dim() == input.dim() == 1
        self.slots[self.rank_in_group] = input.clone()
        self.barrier.wait()
        n = input.numel() // self.world_size
        r = self.rank_in_group
        for s in range(self.world_size):
            output[s * n : (s + 1) * n] = self.slots[s][r * n : (r + 1) * n]
        self.barrier.wait()


def _attn(q, rows, sink, meta):
    """Stand-in attention for rows `rows` with all heads."""
    heads = torch.arange(q.shape[1], dtype=q.dtype)
    return (
        q * 2
        + rows.to(q.dtype)[:, None, None] * 1000
        + heads[None, :, None] * 10
        + sink[None, :, None]
        + meta.to(q.dtype)[:, None, None] * 0.5
    )


class _FakeBackend:
    def __init__(self, meta, num_heads):
        self.meta, self.num_heads = meta, num_heads
        self.calls = []

    def forward(
        self,
        *,
        q,
        k,
        v,
        layer,
        forward_batch,
        compress_ratio,
        attn_sink,
        save_kv_cache,
        row_slice,
    ):
        assert k is v and not save_kv_cache
        assert q.shape[1] == self.num_heads and attn_sink.shape == (self.num_heads,)
        start, end, num_rows = row_slice
        assert end - start == q.shape[0]
        meta = deepseek_v4_backend._slice_query_rows(self.meta, row_slice, value=_DUMMY)
        self.calls.append((row_slice, meta))
        return _attn(q, torch.arange(start, end), attn_sink, meta)


def _run(world, heads, rows, meta_rows=None):
    """Every rank's output and the backend calls, plus the all-heads reference."""
    torch.manual_seed(rows * world)
    num_heads = world * heads
    q_all = torch.randn(rows, num_heads, DIM)
    sink = torch.randn(num_heads)
    # A static metadata buffer may hold more or fewer rows than the batch.
    meta = torch.arange(meta_rows or rows, dtype=torch.int32) * 3 + 1
    slots, barrier = [None] * world, threading.Barrier(world)
    outs, backends = [None] * world, [None] * world

    def rank_fn(r):
        layer = SimpleNamespace(
            attn_mqa=None, compress_ratio=4, attn_sink=sink, n_local_heads=heads
        )
        group = _FakeGroup(slots, barrier, r)
        start, end = head_a2a.head_a2a_rows(rows, group)
        plan = head_a2a.HeadA2APlan(
            group=group, start=start, end=end, num_rows=rows, row_q=False
        )
        x = torch.empty(rows, 1)
        q = head_a2a.new_q_buffer(
            layer=SimpleNamespace(n_local_heads=heads, head_dim=DIM), x=x, plan=plan
        )
        q[:rows] = q_all[:, r * heads : (r + 1) * heads]
        backends[r] = _FakeBackend(meta, num_heads)
        outs[r] = head_a2a.attend_rows(
            layer=layer, q=q, plan=plan, attn_backend=backends[r], forward_batch=None
        )

    with mock.patch.object(head_a2a, "_symm_exchange", return_value=None):
        threads = [threading.Thread(target=rank_fn, args=(r,)) for r in range(world)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    ref_meta = deepseek_v4_backend._slice_query_rows(
        meta, (0, rows, rows), value=_DUMMY
    )
    ref = _attn(q_all, torch.arange(rows), sink, ref_meta)
    return outs, backends, ref


class TestHeadA2aExchange(unittest.TestCase):
    def _check(self, world, heads, rows, meta_rows=None):
        outs, backends, ref = _run(world, heads, rows, meta_rows)
        chunk = -(-rows // world)
        for r in range(world):
            # Every rank gets its heads of every real row back, exactly.
            torch.testing.assert_close(
                outs[r], ref[:, r * heads : (r + 1) * heads], rtol=0, atol=0
            )
            # Each rank attended its own row slice, pad rows with dummy metadata.
            ((row_slice, meta),) = backends[r].calls
            self.assertEqual(row_slice, (r * chunk, (r + 1) * chunk, rows))
            real = max(0, min(chunk, rows - r * chunk))
            self.assertTrue(torch.all(meta[real:] == _DUMMY))

    def test_rows_divisible(self):
        for rows in (96, 192, 384):
            self._check(8, 8, rows)
        self._check(4, 16, 96)

    def test_rows_padded(self):
        for rows in (97, 102, 366):
            self._check(8, 8, rows)
        self._check(4, 16, 99)

    def test_static_metadata_rows(self):
        # Larger buffers are cut to the batch, never read past it; smaller ones
        # are padded like match_num_queries does.
        self._check(8, 8, 102, meta_rows=128)
        self._check(8, 8, 102, meta_rows=100)


class TestSliceQueryRows(unittest.TestCase):
    def test_slices(self):
        meta = torch.arange(10, dtype=torch.int32).view(5, 2)
        f = deepseek_v4_backend._slice_query_rows
        torch.testing.assert_close(f(meta, (1, 3, 5), value=-1), meta[1:3])
        # Rows from num_rows on take the pad value, also past the buffer.
        torch.testing.assert_close(
            f(meta, (2, 6, 4), value=-1),
            torch.tensor([[4, 5], [6, 7], [-1, -1], [-1, -1]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            f(meta, (4, 8, 8), value=1),
            torch.tensor([[8, 9], [1, 1], [1, 1], [1, 1]], dtype=torch.int32),
        )


def _group(world, rank, pynccl=True):
    return SimpleNamespace(
        world_size=world, rank_in_group=rank, pynccl_comm=object() if pynccl else None
    )


def _plan(
    mode=ForwardMode.TARGET_VERIFY,
    *,
    rows=192,
    tp=8,
    n_heads=64,
    attn_tp_group=None,
    tp_group=None,
    batch_invariant=False,
    full_wq_b=True,
):
    layer = SimpleNamespace(
        attn_tp_size=tp, attn_tp_rank=3, n_heads=n_heads, n_local_heads=n_heads // tp
    )
    parallel = SimpleNamespace(
        attn_tp_group=attn_tp_group or _group(tp, 3),
        tp_group=tp_group or _group(tp, 3),
    )
    with (
        mock.patch.object(head_a2a, "get_parallel", return_value=parallel),
        mock.patch.object(
            head_a2a, "full_wq_b", return_value=object() if full_wq_b else None
        ),
        mock.patch(
            "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
            return_value=batch_invariant,
        ),
    ):
        plan = head_a2a.plan_head_a2a(
            layer=layer, num_rows=rows, forward_batch=SimpleNamespace(forward_mode=mode)
        )
    return plan, parallel


class TestHeadA2aGate(unittest.TestCase):
    def test_enabled(self):
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            plan, parallel = _plan(mode)
            self.assertIs(plan.group, parallel.attn_tp_group)
            self.assertEqual((plan.start, plan.end, plan.num_rows), (72, 96, 192))
        plan, _ = _plan(rows=96, tp=4)  # 16 local heads
        self.assertIsNotNone(plan)
        plan, _ = _plan(n_heads=128)  # 16 of 128 heads
        self.assertIsNotNone(plan)

    def test_cp_decode_attn_tp_uses_tp_group(self):
        # attn_tp_* are the decode (CP) ranks; the attention TP group is size 1.
        plan, parallel = _plan(attn_tp_group=_group(1, 0))
        self.assertIs(plan.group, parallel.tp_group)

    def test_row_q_threshold(self):
        self.assertFalse(_plan(rows=287)[0].row_q)
        self.assertTrue(_plan(rows=288)[0].row_q)
        # Without the replicated wq_b (plain TP) the slice keeps the exchange.
        self.assertFalse(_plan(rows=384, full_wq_b=False)[0].row_q)

    def test_disabled(self):
        cases = {
            "few rows": dict(rows=95),
            "no attention TP": dict(tp=1),
            "no head padding": dict(tp=2, n_heads=128),
            "kernel head count": dict(n_heads=32, tp=4),
            "batch invariant": dict(batch_invariant=True),
            "extend": dict(mode=ForwardMode.EXTEND),
            "draft extend": dict(mode=ForwardMode.DRAFT_EXTEND_V2),
            "idle": dict(mode=ForwardMode.IDLE),
            "rank order": dict(attn_tp_group=_group(8, 4), tp_group=_group(8, 5)),
            "no pynccl": dict(
                attn_tp_group=_group(8, 3, pynccl=False),
                tp_group=_group(8, 3, pynccl=False),
            ),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                self.assertIsNone(_plan(**kwargs)[0])


class _BlockLinear(Fp8LinearMethod):
    """A block-FP8 linear method: dequantized weight, per-[32, 32] block scale."""

    block_quant = True
    weight_block_size = [32, 32]

    def __init__(self):
        self.w8a8_block_fp8_linear = self._linear

    @staticmethod
    def _linear(input, weight, block_size, weight_scale, input_scale, bias, out=None):
        w = weight.float() * weight_scale.repeat_interleave(32, 0).repeat_interleave(
            32, 1
        )
        y = input.float() @ w.T
        if out is None:
            return y
        out.copy_(y)
        return out


def _fp8(*shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen).to(torch.float8_e4m3fn)


def _scales(rows, cols, seed):
    gen = torch.Generator().manual_seed(seed)
    return 2.0 ** torch.randint(-3, 3, (rows // 32, cols // 32), generator=gen).float()


class TestRowQProjections(CustomTestCase):
    HEADS, HEAD_DIM, Q_RANK, KV_DIM, HIDDEN = 4, 32, 64, 32, 96

    def test_unquantized_layer_falls_back_without_reading_fp8_scales(self):
        """A non-FP8 head must fall back even when its layer has no weight_scale_inv."""
        layer = SimpleNamespace(
            wqkv_a=SimpleNamespace(
                quant_method=UnquantizedLinearMethod(), weight=object()
            )
        )
        for row_q in (False, True):
            with self.subTest(row_q=row_q):
                self.assertIsNone(
                    head_a2a.wqkv_a_row_q(
                        layer=layer, x=None, plan=SimpleNamespace(row_q=row_q)
                    )
                )

    def _plan(self, rank, rows=10, world=4):
        chunk = -(-rows // world)
        return head_a2a.HeadA2APlan(
            group=SimpleNamespace(world_size=world),
            start=rank * chunk,
            end=(rank + 1) * chunk,
            num_rows=rows,
            row_q=True,
        )

    def test_project_row_q_slices_rows_with_every_head(self):
        method = _BlockLinear()
        weight = _fp8(self.HEADS * self.HEAD_DIM, self.Q_RANK, seed=1)
        scale = _scales(*weight.shape, seed=2)
        layer = SimpleNamespace(
            wq_b=SimpleNamespace(quant_method=method),
            n_heads=self.HEADS,
            head_dim=self.HEAD_DIM,
        )
        q_lora = torch.randn(10, self.Q_RANK)
        positions = torch.arange(10) + 100
        full = method._linear(q_lora, weight, None, scale, None, None)
        full = full.view(10, self.HEADS, self.HEAD_DIM)
        with mock.patch.object(head_a2a, "full_wq_b", return_value=(weight, scale)):
            for rank in range(4):
                plan = self._plan(rank)
                q, pos = head_a2a.project_row_q(
                    layer=layer, q_lora=q_lora, positions=positions, plan=plan
                )
                real = plan.real_end - plan.start
                self.assertEqual(tuple(q.shape), (3, self.HEADS, self.HEAD_DIM))
                torch.testing.assert_close(q[:real], full[plan.start : plan.real_end])
                self.assertTrue((q[real:] == 0).all())
                torch.testing.assert_close(
                    pos[:real], positions[plan.start : plan.real_end]
                )
                self.assertTrue((pos[real:] == 0).all())

    def test_wqkv_a_row_q_halves(self):
        for stores_into_slices in (False, True):
            method = _BlockLinear()
            weight = _fp8(self.Q_RANK + self.KV_DIM, self.HIDDEN, seed=3)
            scale = _scales(*weight.shape, seed=4)
            layer = SimpleNamespace(
                wqkv_a=SimpleNamespace(
                    quant_method=method, weight=weight, weight_scale_inv=scale
                ),
                q_lora_rank=self.Q_RANK,
            )
            x = torch.randn(10, self.HIDDEN)
            full = method._linear(x, weight, None, scale, None, None)
            for rank in range(4):
                with self.subTest(stores_into_slices=stores_into_slices, rank=rank):
                    plan = self._plan(rank)
                    with mock.patch.object(
                        head_a2a,
                        "_stores_into_slices",
                        return_value=stores_into_slices,
                    ):
                        out = head_a2a.wqkv_a_row_q(layer=layer, x=x, plan=plan)
                    kv, q = out[:, self.Q_RANK :], out[:, : self.Q_RANK]
                    # Every row's kv (each rank stores the whole KV).
                    torch.testing.assert_close(kv, full[:, self.Q_RANK :])
                    rows = slice(plan.start, plan.real_end)
                    torch.testing.assert_close(q[rows], full[rows, : self.Q_RANK])
                    others = torch.ones(10, dtype=torch.bool)
                    others[rows] = False
                    self.assertTrue((q[others] == 0).all())

    def test_wqkv_a_row_q_needs_a_block_aligned_split(self):
        method = _BlockLinear()
        weight = _fp8(48 + self.KV_DIM, self.HIDDEN, seed=5)
        layer = SimpleNamespace(
            wqkv_a=SimpleNamespace(
                quant_method=method,
                weight=weight,
                weight_scale_inv=torch.ones(3, self.HIDDEN // 32),
            ),
            q_lora_rank=48,
        )
        plan = self._plan(0)
        self.assertIsNone(head_a2a.wqkv_a_row_q(layer=layer, x=None, plan=plan))


if __name__ == "__main__":
    unittest.main()
