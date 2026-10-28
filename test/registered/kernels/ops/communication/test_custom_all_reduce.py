"""Correctness test for the JIT custom all-reduce (v2) kernel.

Compares the JIT custom all-reduce output against NCCL all-reduce for a sweep
of tensor sizes, dtypes, and algorithms, in both eager and CUDA-graph modes.

Usage::

    # Run the test on the default world sizes (2, 4, 8 GPUs):
    python tests/test_custom_all_reduce.py
    # Pick a specific world size (or comma-separated list), e.g. the rarer
    # odd / non-power-of-two counts that the default sweep skips:
    python tests/test_custom_all_reduce.py --num-gpu 3
    python tests/test_custom_all_reduce.py --num-gpu 2,4,6,8
"""

from __future__ import annotations

import atexit
import itertools
import logging
import multiprocessing
import os
from multiprocessing.context import SpawnProcess
from typing import List

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once, get_ci_test_range
from sglang.kernels.ops.communication.all_reduce import (
    AllReduceAlgo,
    get_all_reduce_module,
    get_pull_collective_module,
)
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(
    est_time=300,
    stage="extra-b",
    runner_config="8-gpu-h200",
)
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=110, stage="nightly", runner_config="8-gpu-h200")

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

TEST_SIZES = [
    16,
    32,
    512,
    1024,
    1024 + 16,  # weird case
    4 * 1024,
    32 * 1024,
    256 * 1024,
    2 * 1024 * 1024,
    3 * 1024 * 1024 + 24,  # chunks not a multiple of world_size
    4 * 1024 * 1024,
]
TEST_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
TEST_ALGOS = [
    AllReduceAlgo.ONE_SHOT_PULL,
    AllReduceAlgo.ONE_SHOT_PUSH,
    AllReduceAlgo.TWO_SHOT_PUSH,
    AllReduceAlgo.TWO_SHOT_PULL,
]
USE_GRAPH_OPTIONS = [False, True]
TEST_LAYERS = 4
TEST_LOOP = 16

TEST_SIZES = get_ci_test_range(TEST_SIZES, [16, 1024, 32 * 1024, 2 * 1024 * 1024])
TEST_DTYPES = get_ci_test_range(TEST_DTYPES, [torch.bfloat16])

# ---------------------------------------------------------------------------
# Parallel JIT precompile (outer process, before any torchrun child starts)
# ---------------------------------------------------------------------------


def _compile_one(dtype: torch.dtype, world_size: int) -> None:
    """Compile the all-reduce modules for a single (dtype, world_size).

    Top-level so it survives ``spawn`` pickling. Compiled artifacts are
    cached on disk by ``tvm_ffi``; torchrun children will reuse them.
    """
    get_all_reduce_module(dtype, world_size)
    get_pull_collective_module(dtype, world_size)


def _precompile_kernels(num_gpus: List[int]) -> None:
    """Fan out one process per (dtype, world_size) to warm the JIT cache.

    Without this, every torchrun child serial-compiles its kernels on first
    use, multiplying the wall-clock cost of the run by ~(#dtypes * #ranks).
    """
    ctx = multiprocessing.get_context("spawn")
    procs: list[tuple[torch.dtype, int, SpawnProcess]] = []
    for dtype, world_size in itertools.product(TEST_DTYPES, num_gpus):
        p = ctx.Process(target=_compile_one, args=(dtype, world_size))
        p.start()
        procs.append((dtype, world_size, p))
    for dtype, world_size, p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"Custom-all-reduce precompile failed for "
                f"{dtype=} {world_size=} (exit {p.exitcode})"
            )


# ---------------------------------------------------------------------------
# Per-rank distributed setup (run once per torchrun worker)
# ---------------------------------------------------------------------------


@cache_once
def _init_cpu_group_once() -> dist.ProcessGroup:
    """Initialize gloo world group + cuda device for this rank."""
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    cpu_group = coord.cpu_group
    assert isinstance(cpu_group, dist.ProcessGroup)
    # Suppress chatty internal logging for cleaner test output.
    logging.disable(logging.INFO)
    # Use a non-default stream (mirrors prior behavior).
    torch.cuda.set_stream(torch.cuda.Stream())
    return cpu_group


@cache_once
def _init_nccl_group_once() -> dist.ProcessGroup:
    # Reference NCCL group allocated independently of the parallel_state
    # world group, so the test does not couple to framework internals.
    _init_cpu_group_once()
    local_rank = int(os.environ["LOCAL_RANK"])
    device_group = dist.new_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    assert isinstance(device_group, dist.ProcessGroup)
    return device_group


@cache_once
def _init_comm_once(max_push_blocks: int | None = None) -> CustomAllReduceV2:
    cpu_group = _init_cpu_group_once()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    max_size = max(TEST_SIZES) * max(
        torch.tensor([], dtype=d).element_size() for d in TEST_DTYPES
    )
    comm = CustomAllReduceV2(
        cpu_group,
        device,
        max_pull_size=max_size,
        max_push_size=max_size,
        max_push_blocks=max_push_blocks,
    )
    if comm.disabled:
        raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")
    register_comm_cleanup(comm)
    return comm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_graph", USE_GRAPH_OPTIONS)
@pytest.mark.parametrize("algo", TEST_ALGOS)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("size", TEST_SIZES)
@torch.inference_mode()
def test_custom_all_reduce(
    size: int,
    dtype: torch.dtype,
    algo: AllReduceAlgo,
    use_graph: bool,
) -> None:
    _check_custom_all_reduce(size, dtype, algo, use_graph, _init_comm_once())


@pytest.mark.parametrize("use_graph", USE_GRAPH_OPTIONS)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("nbytes", [32, 288 * 1024])
@torch.inference_mode()
def test_two_shot_push_single_block(
    nbytes: int, dtype: torch.dtype, use_graph: bool
) -> None:
    # At TP8 a 128-thread block has only four warps for seven peers. Both
    # sizes used to hang; the larger one also falls in SM100's tuned range.
    size = nbytes // torch.tensor([], dtype=dtype).element_size()
    _check_custom_all_reduce(
        size, dtype, AllReduceAlgo.TWO_SHOT_PUSH, use_graph, _init_comm_once(1)
    )


def _check_custom_all_reduce(
    size: int,
    dtype: torch.dtype,
    algo: AllReduceAlgo,
    use_graph: bool,
    comm: CustomAllReduceV2,
) -> None:
    nccl_group = _init_nccl_group_once()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    comm.override_algo = algo

    if use_graph:
        graph = torch.cuda.CUDAGraph()
        graph_inp = torch.zeros((TEST_LAYERS, size), dtype=dtype, device=device)
        outs: list[torch.Tensor] = []
        with comm.capture():
            with torch.cuda.graph(graph):
                for i in range(TEST_LAYERS):
                    outs.append(comm.custom_all_reduce(graph_inp[i]))
                out_jit_stack = torch.stack(outs)
        torch.cuda.synchronize()

        def run(x: torch.Tensor) -> torch.Tensor:
            graph_inp.copy_(x)
            graph.replay()
            return out_jit_stack.clone()

    else:

        def run(x: torch.Tensor) -> torch.Tensor:
            eager_inp = x.clone()
            outs = []
            for i in range(TEST_LAYERS):
                outs.append(comm.custom_all_reduce(eager_inp[i]))
                torch.cuda.synchronize()
            return torch.stack(outs)

    for _ in range(TEST_LOOP):
        # NOTE: 15 * 8 < 128, which is the precision limit for bf16
        inp = torch.randint(0, 16, (TEST_LAYERS, size), dtype=dtype, device=device)
        assert comm.should_custom_ar(inp[0])
        out_ref = inp.clone()
        dist.all_reduce(out_ref, group=nccl_group)
        out_jit = run(inp)
        # Exact equality, since values are small integers within bf16 precision.
        # NOTE: use torch's assert_close: it compares on device (~2 ms here),
        # while triton's converts to numpy on the host (~0.6 s per 32 MB
        # tensor) and would dominate the test wall time.
        torch.testing.assert_close(out_ref, out_jit, atol=0, rtol=0)


def _eager_all_reduce(comm: CustomAllReduceV2, x: torch.Tensor) -> torch.Tensor:
    outs = []
    for row in x.clone():
        outs.append(comm.custom_all_reduce(row))
        torch.cuda.synchronize()
    return torch.stack(outs)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("size", TEST_SIZES)
@torch.inference_mode()
def test_custom_all_reduce_tuned_eager(size: int, dtype: torch.dtype) -> None:
    """Without an override, eager calls launch the tuned algorithm (cached per size)."""
    nccl_group = _init_nccl_group_once()
    comm = _init_comm_once()
    comm.override_algo = None
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    inp = torch.randint(0, 16, (TEST_LAYERS, size), dtype=dtype, device=device)
    nbytes = inp[0].numel() * inp.element_size()
    tuned = comm._pick_config(nbytes, False) is not None
    assert comm.should_custom_ar(inp[0]) == tuned
    if not tuned:
        return
    out_ref = inp.clone()
    dist.all_reduce(out_ref, group=nccl_group)
    for _ in range(2):  # the second pass runs from the cached launch
        torch.testing.assert_close(
            out_ref, _eager_all_reduce(comm, inp), atol=0, rtol=0
        )


@torch.inference_mode()
def test_custom_all_reduce_eager_plan_follows_config() -> None:
    """The cached eager launch is rebuilt when the config object is replaced."""
    nccl_group = _init_nccl_group_once()
    comm = _init_comm_once()
    comm.override_algo = None
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    inp = torch.randint(0, 16, (TEST_LAYERS, 1024), dtype=torch.bfloat16, device=device)
    assert comm.should_custom_ar(inp[0])  # cached under the tuned config
    tuned = comm.config
    try:
        eager = tuned.eager._replace(
            one_shot_push_threshold=0,
            one_shot_pull_threshold=0,
            two_shot_pull_threshold=0,
            mc=tuned.eager.mc._replace(min_bytes=0, max_bytes=0),
        )
        comm.config = tuned._replace(eager=eager)
        assert not comm.should_custom_ar(inp[0])
    finally:
        comm.config = tuned
    assert comm.should_custom_ar(inp[0])
    out_ref = inp.clone()
    dist.all_reduce(out_ref, group=nccl_group)
    torch.testing.assert_close(out_ref, _eager_all_reduce(comm, inp), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Reduce-scatter / all-gather halves of 2shot_pull (the TP-SP decode rows)
# ---------------------------------------------------------------------------

HIDDEN = 5120  # DeepSeek-V4.1 hidden size
HC_HIDDEN = 4 * HIDDEN  # its hc_mult = 4 residual streams of a row
RS_SHAPES = [(6, HIDDEN), (64, HIDDEN), (384, HIDDEN), (6, HC_HIDDEN), (96, HC_HIDDEN)]
RS_SHAPES = get_ci_test_range(RS_SHAPES, [(64, HIDDEN)])
# Rows of a decode/verify batch; 512 x HC_HIDDEN bf16 exceeds the default 16 MB cap.
TUNED_SHAPES = [
    (rows, width) for width in (HIDDEN, HC_HIDDEN) for rows in (6, 16, 48, 64, 192, 384)
] + [(512, HC_HIDDEN)]
TUNED_SHAPES = get_ci_test_range(TUNED_SHAPES, [(16, HIDDEN), (192, HIDDEN)])
AG_SHAPES = [(1, HIDDEN), (48, HIDDEN), (12, HC_HIDDEN)]


@cache_once
def _init_default_comm_once() -> CustomAllReduceV2:
    """An instance with the production sizes, so `_pick_config` sees the tuned
    thresholds (including multicast where the fabric has it)."""
    comm = CustomAllReduceV2(
        _init_cpu_group_once(), torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    )
    if comm.disabled:
        raise RuntimeError("JIT CustomAllReduceV2 is disabled on this system")
    register_comm_cleanup(comm)
    return comm


def _partials(numel: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """This rank's partial sums, with the zeros a sum must keep: (+0, +0) pairs
    on every rank (the lamport empty marker), -0.0 on every rank, x + (-x)
    cancellation across ranks, sums that depend on the rank order, and an inf."""
    world, rank = dist.get_world_size(), dist.get_rank()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    gen = torch.Generator(device=device).manual_seed(seed * 131 + rank)
    x = torch.randn(numel, device=device, generator=gen).to(dtype)
    x[0::64] = 0.0
    x[1::64] = 0.0
    x[2::64] = -0.0
    x[3::64] = -0.0
    x[4::64] = 1.5 if rank % 2 == 0 else -1.5
    if world >= 3:
        # Ranks (i, j, k) hold (2^15, 2^-10, -2^15); a rank-ordered fp32 sum
        # keeps the small term only when it comes after both big ones.
        triples = torch.tensor(
            list(itertools.permutations(range(world), 3)), device=device
        )
        slots = x[6::64]
        i, j, k = triples[torch.arange(slots.numel(), device=device) % len(triples)].T
        zero = torch.zeros_like(slots, dtype=torch.float32)
        slots.copy_(
            torch.where(i == rank, 2.0**15, zero)
            + torch.where(j == rank, 2.0**-10, zero)
            + torch.where(k == rank, -(2.0**15), zero)
        )
    if rank == 0:
        x[5::4099] = float("inf")
    return x


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16 if x.element_size() == 2 else torch.int32)


def _bitwise_on_all_ranks(got: torch.Tensor, ref: torch.Tensor) -> bool:
    # Every rank must take the same assert branch, or its peers block in the
    # next collective.
    ok = torch.tensor([int(torch.equal(_bits(got), _bits(ref)))])
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    return ok.item() == 1


def _replay_one(comm: CustomAllReduceV2, fn, use_graph: bool):
    """``fn()`` run eagerly, or captured and replayed once; the graph is
    returned too, as it owns any tensor ``fn`` allocated."""
    if not use_graph:
        return fn(), None
    graph = torch.cuda.CUDAGraph()
    with comm.capture():
        with torch.cuda.graph(graph):
            result = fn()
    graph.replay()
    return result, graph


def _all_reduce_and_reduce_scatter(
    comm: CustomAllReduceV2, full: torch.Tensor, use_graph: bool
):
    """(predicate, this rank's shard of the custom all-reduce or None when it
    would not run, the reduce-scatter shard or None when it does not fit), all
    taken in one mode: eager, or one CUDA graph (the predicate inside capture)."""
    world, rank = dist.get_world_size(), dist.get_rank()
    shard_numel = full.numel() // world
    # Separate inputs: the graph-mode 2shot_pull all-reduce works in place.
    ar_in, rs_in = full.clone(), full.clone()
    rs_out = torch.full((shard_numel,), 7.0, dtype=full.dtype, device=full.device)

    def body():
        pred = comm.all_reduce_sums_in_rank_order(numel=full.numel(), dtype=full.dtype)
        ar = comm.custom_all_reduce(ar_in) if comm.should_custom_ar(ar_in) else None
        rs_fits = comm.can_reduce_scatter(rs_in, rs_out)
        if rs_fits:
            comm.reduce_scatter(rs_in, rs_out)
        return pred, ar, rs_fits

    (pred, ar, rs_fits), _graph = _replay_one(comm, body, use_graph)
    torch.cuda.synchronize()
    ar_shard = (
        None
        if ar is None
        else ar[rank * shard_numel : (rank + 1) * shard_numel].clone()
    )
    return pred, ar_shard, rs_out if rs_fits else None


@pytest.mark.parametrize("use_graph", USE_GRAPH_OPTIONS)
@pytest.mark.parametrize("algo", TEST_ALGOS)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("rows, width", RS_SHAPES)
@torch.inference_mode()
def test_reduce_scatter_matches_all_reduce_iff_rank_order(
    rows: int, width: int, dtype: torch.dtype, algo: AllReduceAlgo, use_graph: bool
) -> None:
    """Every all-reduce algorithm, forced: the reduce-scatter shard is bitwise
    the all-reduce shard exactly when ``all_reduce_sums_in_rank_order`` holds.
    The push kernels' lamport remap returns the (+0, +0) pairs as -0.0, so the
    predicate must refuse them while accepting both pull kernels."""
    comm = _init_comm_once()
    comm.override_algo = algo
    try:
        pred, ar_shard, rs_shard = _all_reduce_and_reduce_scatter(
            comm, _partials(rows * width, dtype, seed=rows), use_graph
        )
    finally:
        comm.override_algo = None
    assert ar_shard is not None and rs_shard is not None
    assert _bitwise_on_all_ranks(rs_shard, ar_shard) == pred


@pytest.mark.parametrize("use_graph", USE_GRAPH_OPTIONS)
@pytest.mark.parametrize("rows, width", TUNED_SHAPES)
@torch.inference_mode()
def test_rank_order_predicate_follows_tuned_all_reduce(
    rows: int, width: int, use_graph: bool
) -> None:
    """With the tuned algorithm choice (graph and eager differ): a pull kernel
    must be accepted and match the reduce-scatter bitwise, a push kernel must
    be refused (it does not match), and NVLS multicast or the NCCL fallback
    above the custom-AR sizes must be refused (they sum in their own order)."""
    comm = _init_default_comm_once()
    full = _partials(rows * width, torch.bfloat16, seed=rows + width)
    config = comm._pick_config(full.numel() * full.element_size(), use_graph)
    pred, ar_shard, rs_shard = _all_reduce_and_reduce_scatter(comm, full, use_graph)
    if config is None or config.use_multicast:
        assert not pred
    elif config.algo.is_push():
        assert not pred
        assert not _bitwise_on_all_ranks(rs_shard, ar_shard)
    else:
        assert pred
        assert _bitwise_on_all_ranks(rs_shard, ar_shard)


@pytest.mark.parametrize("use_graph", USE_GRAPH_OPTIONS)
@pytest.mark.parametrize("rows, width", AG_SHAPES)
@torch.inference_mode()
def test_all_gather_matches_nccl(rows: int, width: int, use_graph: bool) -> None:
    nccl_group = _init_nccl_group_once()
    comm = _init_comm_once()
    world = dist.get_world_size()
    local = _partials(rows * width, torch.bfloat16, seed=rows).view(rows, width)
    local.view(-1)[3::997] = float("nan")  # a copy must carry any bit pattern
    ref = torch.empty(world * rows, width, dtype=local.dtype, device=local.device)
    dist.all_gather_into_tensor(ref, local, group=nccl_group)
    out = torch.full_like(ref, 5.0)
    assert comm.can_all_gather(local, out)
    _replay_one(comm, lambda: comm.all_gather(local, out), use_graph)
    torch.cuda.synchronize()
    assert _bitwise_on_all_ranks(out, ref)


def _two_shot_pull_graph(comm: CustomAllReduceV2, full: torch.Tensor) -> torch.Tensor:
    """This rank's shard of the graph-mode in-place 2shot_pull all-reduce."""
    world, rank = dist.get_world_size(), dist.get_rank()
    rows = full.shape[0] // world
    comm.override_algo = AllReduceAlgo.TWO_SHOT_PULL
    graph_inp = torch.empty_like(full)
    graph = torch.cuda.CUDAGraph()
    try:
        with comm.capture():
            with torch.cuda.graph(graph):
                result = comm.custom_all_reduce(graph_inp)
    finally:
        comm.override_algo = None
    graph_inp.copy_(full)
    graph.replay()
    torch.cuda.synchronize()
    return result[rank * rows : (rank + 1) * rows].clone()


@torch.inference_mode()
def test_gather_scatter_chain_with_lagging_rank() -> None:
    """All-gather, a GEMM stand-in, reduce-scatter, layer after layer in one CUDA
    graph, with rank 0 producing its reduce-scatter input late in every layer:
    no rank may load a peer's registered input before that peer has written it,
    and the interleaved graph rows must map to the right tensors."""
    nccl_group = _init_nccl_group_once()
    comm = _init_comm_once()
    world, rank = dist.get_world_size(), dist.get_rank()
    chunk_rows, num_layers = 24, 3
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    scales = [0.5, -1.25, 2.0]

    def partial(gathered: torch.Tensor, layer: int) -> torch.Tensor:
        if rank == 0:
            torch.cuda._sleep(2_000_000)
        return gathered * scales[layer] + float(rank)

    def run_nvlink(x: torch.Tensor) -> torch.Tensor:
        for layer in range(num_layers):
            gathered = torch.empty(
                world * chunk_rows, HIDDEN, dtype=x.dtype, device=device
            )
            comm.all_gather(x, gathered)
            x = torch.empty_like(x)
            comm.reduce_scatter(partial(gathered, layer), x)
        return x

    x0 = _partials(chunk_rows * HIDDEN, torch.bfloat16, seed=5).view(chunk_rows, -1)
    ref = x0
    for layer in range(num_layers):
        gathered = torch.empty(
            world * chunk_rows, HIDDEN, dtype=x0.dtype, device=device
        )
        dist.all_gather_into_tensor(gathered, ref, group=nccl_group)
        ref = _two_shot_pull_graph(comm, partial(gathered, layer))

    graph_in = torch.empty_like(x0)
    graph = torch.cuda.CUDAGraph()
    with comm.capture():
        with torch.cuda.graph(graph):
            graph_out = run_nvlink(graph_in)
    for _ in range(2):
        graph_in.copy_(x0)
        graph.replay()
        torch.cuda.synchronize()
        assert _bitwise_on_all_ranks(graph_out, ref)


@torch.inference_mode()
def test_pull_collective_fit_is_shape_only() -> None:
    """The fit checks gate a collective every rank must enter together, so an
    unfit shape must be refused (NCCL fallback) rather than launched."""
    comm = _init_comm_once()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    bf16 = dict(dtype=torch.bfloat16, device=device)
    local = torch.empty(4, HIDDEN, **bf16)
    assert comm.can_all_gather(local, torch.empty(world * 4, HIDDEN, **bf16))
    assert not comm.can_all_gather(local, torch.empty(world * 4 + 1, HIDDEN, **bf16))
    assert not comm.can_all_gather(
        local.int(), torch.empty(world * 4, HIDDEN, device=device, dtype=torch.int32)
    )
    empty = torch.empty(0, HIDDEN, **bf16)
    assert not comm.can_reduce_scatter(empty, empty)
    # 3 bf16 per shard is not a whole 16-byte vector.
    assert not comm.can_reduce_scatter_input(numel=3 * world, dtype=torch.bfloat16)
    # Eager stages the input through the pull workspace, which bounds it.
    too_big = comm.max_pull_size // 2 + 16 * world
    assert not comm.can_reduce_scatter_input(numel=too_big, dtype=torch.bfloat16)


if __name__ == "__main__":
    # Only sweep the common world sizes (2, 4, 8, 16) by default: testing every
    # count in 2..16 serially overruns the per-file CI time budget, and numbers
    # in the middle are rare in practice. Use --num-gpu to exercise them
    # explicitly.
    multigpu_pytest_main(
        __name__,
        __file__,
        num_gpus=(2, 4, 8, 16),
        pre_launch_fn=_precompile_kernels,
    )
