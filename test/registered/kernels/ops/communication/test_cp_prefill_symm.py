"""The DeepSeek-V4.1 CP prefill collectives over symmetric memory.

Each gather must be bitwise the NCCL collective it replaces (the rank-major
all-gather, the interleave CP gather in global token order, column parts of
mixed dtypes, the top-k side payload); the reduce-scatter must be the fp32 sum of
the ranks' partials in rank order, rounded once; the buffers must survive a rank
that lags into the next layer; and the owner scatters must equal the all-reduce
of the zero-filled sharded lookups, -0 turned into +0 as that sum does.

Every rank regenerates all ranks' inputs from seeds, so expectations need no
collective. Usage::

    python test/registered/kernels/ops/communication/test_cp_prefill_symm.py --num-gpu 8
"""

import atexit
import os
from types import SimpleNamespace
from typing import List

import pytest
import torch
import torch.distributed as dist

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication.cp_prefill_symm import (
    embed_owner_scatter,
    engram_owner_scatter,
)
from sglang.kernels.ops.embeddings.engram_gather import engram_gather
from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum
from sglang.srt.layers.communication.cp_prefill_symm import (
    CpOwnerScatter,
    CpPrefillSymm,
)
from sglang.srt.layers.cp.interleave import (
    InterleaveContextParallelMetadata,
    interleave_gather_layout,
    interleave_global_order,
)
from sglang.srt.layers.moe.moe_runner.humming import HummingDeferredMulSum
from sglang.srt.models.deepseek_v4_cp_prefill import (
    reduce_scatter_with_shared,
    shared_on_landed_shards,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=60, stage="extra-b", runner_config="4-gpu-h100")

HIDDEN = 5120  # DeepSeek-V4.1 hidden size
KV_DIM = 512  # DeepSeek-V4.1 SWA KV width
TOP_K = 6
MAX_LOCAL_ROWS = 2048  # 16K-token chunk under CP8
# A ragged last chunk, the MoE threshold and a full 16K chunk under CP8.
ROWS = (1, 37, 1024, 2048)


def _device() -> torch.device:
    return torch.device(
        "cuda", int(os.environ["LOCAL_RANK"]) % torch.cuda.device_count()
    )


@cache_once
def _group() -> SimpleNamespace:
    torch.cuda.set_device(_device())
    dist.init_process_group(backend="gloo")
    atexit.register(dist.destroy_process_group)
    return SimpleNamespace(
        world_size=dist.get_world_size(),
        rank_in_group=dist.get_rank(),
        cpu_group=dist.group.WORLD,
        unique_name="test",
    )


def _symm(
    width: int, *, with_reduce: bool, side_cols: int = 0, gather_cols: int = 0
) -> CpPrefillSymm:
    group = _group()
    return CpPrefillSymm(
        cpu_group=group.cpu_group,
        rank=group.rank_in_group,
        world_size=group.world_size,
        width=width,
        max_rows=group.world_size * MAX_LOCAL_ROWS,
        with_reduce=with_reduce,
        side_cols=side_cols,
        gather_cols=gather_cols,
    )


@cache_once
def _moe_comm() -> CpPrefillSymm:
    # As the MoE binds it: room for an fp8 gather's two fp8 planes and scales.
    return _symm(
        HIDDEN,
        with_reduce=True,
        side_cols=2 * TOP_K + 4,
        gather_cols=HIDDEN + HIDDEN // 32 * 2,
    )


@cache_once
def _kv_comm() -> CpPrefillSymm:
    return _symm(KV_DIM, with_reduce=False)


@cache_once
def _compressor_comm() -> CpPrefillSymm:
    return _symm(4 * KV_DIM, with_reduce=False)


def _randn(*shape: int, seed: int, rank: int, specials: bool = False) -> torch.Tensor:
    gen = torch.Generator(device=_device()).manual_seed(seed * 131 + rank)
    x = (torch.randn(*shape, device=_device(), generator=gen) * 4).bfloat16()
    if specials:
        x.view(-1)[::7] = -0.0
        x.view(-1)[1::97] = float("nan")
    return x


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16)


def _sum_in_rank_order(parts: List[torch.Tensor]) -> torch.Tensor:
    acc = parts[0].float()
    for part in parts[1:]:
        acc = acc + part.float()
    return acc.bfloat16()


def _sync() -> None:
    torch.cuda.synchronize()
    dist.barrier()


def _lag(rank: int) -> None:
    # Hold this rank's stream so its peers run ahead into the next collective.
    if dist.get_rank() == rank:
        torch.cuda._sleep(3_000_000)


@pytest.mark.parametrize("rows", ROWS)
def test_gather_is_the_rank_major_all_gather(rows: int) -> None:
    comm = _moe_comm()
    world, rank = comm.world_size, comm.rank
    x = _randn(rows, HIDDEN, seed=rows, rank=rank, specials=True)
    expected = [
        _randn(rows, HIDDEN, seed=rows, rank=r, specials=True) for r in range(world)
    ]
    pending = comm.start_gather(x)
    runs = pending.runs()
    blocks = {run: pending.rows_of(*run).clone() for run in runs}
    gathered = pending.finish().clone()
    _sync()
    assert runs[0] == (rank, rank + 1)
    assert sorted(p for lo, hi in runs for p in range(lo, hi)) == list(range(world))
    for (lo, hi), block in blocks.items():
        assert torch.equal(_bits(block), _bits(torch.cat(expected[lo:hi])))
    assert torch.equal(_bits(gathered), _bits(torch.cat(expected)))


@pytest.mark.parametrize("rows", ROWS)
def test_side_payload_is_the_all_gather(rows: int) -> None:
    comm = _moe_comm()
    world, rank = comm.world_size, comm.rank

    def side_of(r: int) -> torch.Tensor:
        gen = torch.Generator(device=_device()).manual_seed(900 + rows * 7 + r)
        return torch.randint(
            -(2**31), 2**31 - 1, (rows, 2 * TOP_K), generator=gen, device=_device()
        ).to(torch.int32)

    x = _randn(rows, HIDDEN, seed=rows + 3, rank=rank)
    side = side_of(rank)
    assert comm.fits_side(side, rows)
    pending = comm.start_gather(x, side=side)
    got = pending.finish_side().clone()
    pending.finish()
    _sync()
    assert torch.equal(got, torch.cat([side_of(r) for r in range(world)]))


@pytest.mark.parametrize("rows", ROWS)
def test_reduce_scatter_sums_partials_in_rank_order(rows: int) -> None:
    comm = _moe_comm()
    world, rank = comm.world_size, comm.rank
    partial = _randn(world * rows, HIDDEN, seed=rows + 1, rank=rank)

    def write_rows(p: int, out: torch.Tensor) -> None:
        out.copy_(partial[p * rows : (p + 1) * rows])

    got = comm.reduce_scatter(rows, write_rows)
    own = slice(rank * rows, (rank + 1) * rows)
    expected = _sum_in_rank_order(
        [_randn(world * rows, HIDDEN, seed=rows + 1, rank=r)[own] for r in range(world)]
    )
    _sync()
    assert torch.equal(_bits(got), _bits(expected))


def _layers(
    comm: CpPrefillSymm, xs: List[torch.Tensor], lagging: int
) -> List[torch.Tensor]:
    """Per layer: gather, a row-local op per landed block, reduce-scatter of the
    gathered rows scaled by (rank + 1); the given rank lags at every step."""
    rows, rank = xs[0].shape[0], comm.rank
    outs = []
    for x in xs:
        pending = comm.start_gather(x)
        shards = {}
        for lo, hi in pending.runs():
            _lag(lagging)
            block = pending.rows_of(lo, hi) * 2
            for p in range(lo, hi):
                shards[p] = block[(p - lo) * rows : (p - lo + 1) * rows]
        pending.finish()

        def write_rows(p: int, out: torch.Tensor) -> None:
            _lag(lagging)
            out.copy_(shards[p] * (rank + 1))

        outs.append(comm.reduce_scatter(rows, write_rows))
    return outs


@pytest.mark.parametrize("lagging", [0, 1])
def test_layer_sequence_with_a_lagging_rank(lagging: int) -> None:
    """A rank late to read its gathered rows or its slots must still see this
    layer's data while its peers start the next layer's gather and reduce."""
    comm = _moe_comm()
    world, rank = comm.world_size, comm.rank
    rows = 1024

    def inputs(r: int) -> List[torch.Tensor]:
        return [_randn(rows, HIDDEN, seed=500 + i, rank=r) for i in range(3)]

    got = [t.clone() for t in _layers(comm, inputs(rank), lagging)]
    # Rank r adds its partial (r + 1) * 2 * x of this rank's rows.
    own = inputs(rank)
    expected = [
        _sum_in_rank_order([own[i] * 2 * (r + 1) for r in range(world)])
        for i in range(3)
    ]
    _sync()
    for g, e in zip(got, expected):
        assert torch.equal(_bits(g), _bits(e))


def test_shared_rows_pair_with_their_shards() -> None:
    """The shared expert runs on blocks of landed shards; each shard's output must
    be added to that shard's routed rows before the sum, as the MoE's
    routed += shared does on the gathered rows."""
    comm = _moe_comm()
    world, rank = comm.world_size, comm.rank
    rows = 1000

    def shared_fn(x: torch.Tensor) -> torch.Tensor:
        # Row-local, rank-dependent like a TP shard of the shared expert.
        return x * (rank + 3) - 1

    def routed_of(r: int, gathered: torch.Tensor) -> torch.Tensor:
        return gathered.flip(1) * (r + 1)

    x = _randn(rows, HIDDEN, seed=77, rank=rank)
    pending = comm.start_gather(x)
    shared = shared_on_landed_shards(shared_fn, pending)
    gathered = pending.finish()
    got = reduce_scatter_with_shared(
        comm, routed=routed_of(rank, gathered), shared=shared, rows=rows
    )
    full = torch.cat([_randn(rows, HIDDEN, seed=77, rank=r) for r in range(world)])
    own = slice(rank * rows, (rank + 1) * rows)
    expected = _sum_in_rank_order(
        [(routed_of(r, full) + (full * (r + 3) - 1))[own] for r in range(world)]
    )
    _sync()
    assert torch.equal(_bits(got), _bits(expected))


@pytest.mark.parametrize("rows", [37, 2048])
def test_fp8_planes_are_their_all_gathers(rows: int) -> None:
    """The fp8 gather moves two fp8 planes and an fp32 scale plane, each into its
    own dense plane; every plane, block by block and whole, must be the
    all-gather of that part."""
    comm = _moe_comm()
    world, rank = comm.world_size, comm.rank

    def parts_of(r: int) -> List[torch.Tensor]:
        gen = torch.Generator(device=_device()).manual_seed(60 + rows * 3 + r)
        a = torch.randn(rows, HIDDEN, device=_device(), generator=gen)
        b = torch.randn(rows, HIDDEN, device=_device(), generator=gen) * 7
        c = torch.rand(rows, HIDDEN // 32, device=_device(), generator=gen)
        return [a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn), c]

    local = parts_of(rank)
    assert comm.fits(local)
    pending = comm.start_gather_planar(local)
    blocks = {
        run: [p.clone() for p in pending.planes_of(*run)] for run in pending.runs()
    }
    whole = [p.clone() for p in pending.finish_parts()]
    everyone = [parts_of(r) for r in range(world)]
    _sync()
    for i, got in enumerate(whole):
        expected = torch.cat([everyone[r][i] for r in range(world)])
        assert got.dtype == expected.dtype and got.is_contiguous()
        assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
        for (lo, hi), parts in blocks.items():
            block = torch.cat([everyone[r][i] for r in range(lo, hi)])
            assert torch.equal(parts[i].view(torch.uint8), block.view(torch.uint8))


@pytest.mark.parametrize("rows", [37, 2048])
def test_deferred_reduce_is_the_merged_reduce(rows: int) -> None:
    """With the humming weighted sum deferred, one launch writes every owner's
    routed + shared rows into the owners' slots; the reduce-scatter must be bitwise
    the one of the merged routed + shared rows."""
    comm = _moe_comm()
    world, rank = comm.world_size, comm.rank
    gen = torch.Generator(device=_device()).manual_seed(80 + rows * 5 + rank)
    values = torch.randn(world * rows, TOP_K, HIDDEN, device=_device(), generator=gen)
    values = values.to(torch.bfloat16)
    weights = torch.rand(world * rows, TOP_K, device=_device(), generator=gen)
    ids = torch.randint(0, 64, (world * rows, TOP_K), device=_device(), generator=gen)
    ids = ids.to(torch.int32)
    shared = _randn(world * rows, HIDDEN, seed=90 + rows, rank=rank)
    kwargs = dict(
        inputs=values, topk_weights=weights, topk_ids=ids, routed_scaling_factor=2.5
    )
    routed = moe_fused_mul_sum(**kwargs)
    deferred = HummingDeferredMulSum()
    deferred.mul_sum_kwargs = dict(
        kwargs, outputs=torch.empty_like(routed), is_ep=False
    )
    blocks = list(shared.split(rows))
    merged = reduce_scatter_with_shared(
        comm, routed=routed, shared=blocks, rows=rows
    ).clone()
    _sync()
    got = reduce_scatter_with_shared(
        comm, routed=routed, shared=blocks, rows=rows, deferred=deferred
    ).clone()
    _sync()
    assert torch.equal(_bits(got), _bits(merged))


def _interleave_case(per_rank: int, short: int, align: int, world: int):
    total = per_rank * world - short
    base, extra = divmod(total, world)
    logical = [base + (r < extra) for r in range(world)]
    slot = -(-max(logical) // align) * align
    metadata = InterleaveContextParallelMetadata(
        per_rank_actual_token=[slot] * world,
        per_rank_logical_token=logical,
        max_rank_len=[slot] * world,
        total_seq_lens=total,
    )
    return total, logical, slot, metadata


def _token_order(local: List[torch.Tensor], total: int) -> torch.Tensor:
    # Interleave CP: token t is row t // W of rank t % W.
    world = len(local)
    return torch.stack([local[t % world][t // world] for t in range(total)])


@pytest.mark.parametrize("per_rank,short,align", [(1000, 3, 1), (1024, 0, 8)])
def test_kv_gather_is_the_interleave_gather(per_rank, short, align) -> None:
    """Ranks send their logical rows only (rows past them are garbage) into
    zero-padded slots; the slots in global token order are the chunk's KV."""
    comm = _kv_comm()
    world, rank = comm.world_size, comm.rank
    total, logical, slot, metadata = _interleave_case(per_rank, short, align, world)
    local = [_randn(slot, KV_DIM, seed=per_rank, rank=r) for r in range(world)]
    send, rows = interleave_gather_layout(
        metadata, cp_rank=rank, cp_size=world, local_rows=slot
    )
    assert (send, rows) == (logical[rank], slot)
    gathered = comm.start_gather(local[rank][:send], rows=rows).finish()
    got = interleave_global_order(
        gathered, metadata, cp_size=world, physical_rank_len=rows
    ).clone()
    _sync()
    assert torch.equal(_bits(got), _bits(_token_order(local, total)))


@pytest.mark.parametrize("ratio", [1, 2])
def test_compressor_parts_are_their_interleave_gathers(ratio: int) -> None:
    """The ratio-2 compressor's fp32 kv and score (ratio 1: bf16 kv) travel side
    by side in one slot; each part must come back as its own gather."""
    comm = _compressor_comm()
    world, rank = comm.world_size, comm.rank
    total, logical, slot, metadata = _interleave_case(1000, 3, 1, world)

    def parts_of(r: int) -> List[torch.Tensor]:
        if ratio == 1:
            return [_randn(slot, KV_DIM, seed=40, rank=r)]
        return [
            _randn(slot, KV_DIM, seed=41, rank=r).float() / 3,
            _randn(slot, KV_DIM, seed=42, rank=r).float() * 7,
        ]

    local = parts_of(rank)
    send, rows = interleave_gather_layout(
        metadata, cp_rank=rank, cp_size=world, local_rows=slot
    )
    assert comm.fits([p[:send] for p in local], rows=rows)
    pending = comm.start_gather([p[:send] for p in local], rows=rows)
    got = [
        interleave_global_order(
            g, metadata, cp_size=world, physical_rank_len=rows
        ).clone()
        for g in pending.finish_parts()
    ]
    everyone = [parts_of(r) for r in range(world)]
    _sync()
    for i, g in enumerate(got):
        expected = _token_order([everyone[r][i] for r in range(world)], total)
        assert g.dtype == expected.dtype
        assert torch.equal(g.view(torch.int16), expected.view(torch.int16))


def test_fits_rejects_what_the_buffers_cannot_hold() -> None:
    comm = _moe_comm()
    rows = MAX_LOCAL_ROWS
    ok = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=_device())
    assert comm.fits(ok)
    assert not comm.fits(ok[:0])
    assert not comm.fits(ok, rows=rows - 1)
    assert not comm.fits(torch.empty(rows + 1, HIDDEN, dtype=torch.bfloat16))
    assert not comm.fits(ok.float())
    fp8 = ok.to(torch.float8_e4m3fn)
    assert comm.fits([fp8, fp8, ok[:, : HIDDEN // 32].float()])
    assert not comm.fits(fp8[:, :-1])
    assert not comm.fits([fp8, fp8, ok[:, : HIDDEN // 16].float()])
    assert not comm.fits(ok.to(torch.float16))
    assert not comm.fits(ok.t())
    side = torch.empty(rows, 2 * TOP_K + 5, dtype=torch.int32, device=_device())
    assert not comm.fits_side(side, rows)
    assert not _kv_comm().fits_side(side[:, : 2 * TOP_K], rows)


VOCAB_PER_RANK = 997
ENGRAM_COLS, ENGRAM_DIM, ENGRAM_ROWS_PER_RANK = 24, 256, 4093


@cache_once
def _embed_comm() -> CpOwnerScatter:
    return CpOwnerScatter(group=_group(), width=HIDDEN, max_rows=MAX_LOCAL_ROWS)


@cache_once
def _engram_comm() -> CpOwnerScatter:
    return CpOwnerScatter(
        group=_group(), width=ENGRAM_COLS * ENGRAM_DIM, max_rows=MAX_LOCAL_ROWS
    )


@pytest.mark.parametrize("rows", [1, 37, 2048])
def test_embed_owner_scatter_is_the_all_reduced_shard(rows: int) -> None:
    comm = _embed_comm()
    world, rank = comm.world_size, comm.rank
    gen = torch.Generator(device=_device()).manual_seed(7)
    table = torch.randn(world * VOCAB_PER_RANK, HIDDEN, device=_device(), generator=gen)
    table = table.bfloat16()
    table.view(-1)[::13] = -0.0
    ids = torch.randint(
        0, world * VOCAB_PER_RANK, (world * rows,), device=_device(), generator=gen
    )
    lo, hi = rank * VOCAB_PER_RANK, (rank + 1) * VOCAB_PER_RANK
    embed_owner_scatter(ids, table[lo:hi].contiguous(), comm.dst_table, lo, hi)
    got = comm.finish(rows)
    # The default path: zero-filled shard embeddings summed, then this rank's
    # interleaved rows.
    acc = torch.zeros(world * rows, HIDDEN, device=_device())
    for r in range(world):
        owned = (ids >= r * VOCAB_PER_RANK) & (ids < (r + 1) * VOCAB_PER_RANK)
        acc = acc + torch.where(owned[:, None], table[ids], 0).float()
    expected = acc.bfloat16()[rank::world]
    _sync()
    assert torch.equal(_bits(got), _bits(expected))


@pytest.mark.parametrize("rows", [1, 37, 2048])
def test_engram_owner_scatter_is_the_reduce_scattered_lookup(rows: int) -> None:
    comm = _engram_comm()
    world, rank = comm.world_size, comm.rank
    gen = torch.Generator(device=_device()).manual_seed(11)
    n_table = world * ENGRAM_ROWS_PER_RANK
    table = (torch.randn(n_table, ENGRAM_DIM, device=_device(), generator=gen) * 3).to(
        torch.float8_e4m3fn
    )
    table.view(torch.uint8).view(-1)[::17] = 0x80  # fp8 -0
    scales = torch.randint(
        110, 135, (n_table, ENGRAM_DIM // 32), device=_device(), generator=gen
    ).to(torch.uint8)
    lo, hi = rank * ENGRAM_ROWS_PER_RANK, (rank + 1) * ENGRAM_ROWS_PER_RANK
    w, s = table[lo:hi].contiguous(), scales[lo:hi].contiguous()
    ids = torch.randint(
        0, n_table, (world * rows, ENGRAM_COLS), device=_device(), generator=gen
    )
    engram_owner_scatter(
        w.data_ptr(),
        s.data_ptr(),
        ids,
        comm.dst_table,
        ENGRAM_DIM,
        32,
        lo,
        hi,
        rows=rows,
        ids_fit_int32=True,
        num_programs=2 * -(-ids.numel() // world),
    )
    got = comm.finish(rows).view(rows, ENGRAM_COLS, ENGRAM_DIM)
    # The default path: every rank's zero-filled shard lookup, summed.
    acc = torch.zeros(world * rows * ENGRAM_COLS, ENGRAM_DIM, device=_device())
    for r in range(world):
        rlo, rhi = r * ENGRAM_ROWS_PER_RANK, (r + 1) * ENGRAM_ROWS_PER_RANK
        part = torch.empty(
            ids.numel(), ENGRAM_DIM, dtype=torch.bfloat16, device=_device()
        )
        wr, sr = table[rlo:rhi].contiguous(), scales[rlo:rhi].contiguous()
        engram_gather(
            wr.data_ptr(),
            sr.data_ptr(),
            ids.reshape(-1),
            part,
            ENGRAM_DIM,
            32,
            row_lo=rlo,
            row_hi=rhi,
        )
        acc = acc + part.float()
    expected = acc.bfloat16().view(world, rows, ENGRAM_COLS, ENGRAM_DIM)[rank]
    _sync()
    assert torch.equal(_bits(got), _bits(expected))


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4, 8))
