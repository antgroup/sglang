"""NVLink symmetric-memory collectives of the DeepSeek-V4.1 interleave CP prefill.

Under prefill context parallelism each of the W CP ranks holds ``rows`` rows of a
chunk. ``CpPrefillSymm`` replaces the NCCL all-gathers of those rows (the MoE
input, the SWA KV, the compressor projections) and the MoE reduce-scatter:

* ``start_gather``: a rank stores its rows into its own symmetric buffer, a
  device barrier publishes them, and copy-engine pulls on a side stream fetch the
  peers' shards, one event each. No SM runs the transfer, so the caller computes
  on every shard as it lands. The gathered rows are bitwise the NCCL all-gather.
* ``reduce_scatter``: the caller writes its partial sums of rank p's rows into
  rank p's slot, then after a barrier each rank adds its W slots in fp32 in rank
  order and rounds once (NCCL's ring rounds to bf16 on every hop instead).

``CpOwnerScatter`` serves a sharded lookup (the vocab-parallel input embedding,
the engram table): each row's owner writes it into the buffer of the CP rank that
holds its token, in place of a zero-filled all-reduce or reduce-scatter.

A rank overwrites a buffer only after a collective that every peer enters after
its last read of it: each user (one instance per user) runs a collective on the
group between two of its gathers and between two of its reduce-scatters, which
one use per layer, with the layer's other collectives in between, satisfies.
"""

from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.cp_prefill_symm import sum_slots
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.runtime_context import (
    get_parallel,
    get_resources,
    max_prefill_buffer_tokens,
)

_GATHER_CHANNEL = 0
_REDUCE_CHANNEL = 1
_SCATTER_CHANNEL = 0
# H20-3e CP8, 2048 rows per rank: the shared expert on a block of 2 landed shards
# outlasts pulling the next 2; one block of all later shards left ranks waiting
# ~190 us, which the reduce barrier then paid (layer 2372 -> 2057 us).
_RUN_SHARDS = 2
# fp32 columns travel as bf16 pairs and fp8 ones (an even number) as bf16 halves;
# the transfer only moves bytes.
_GATHER_DTYPES = (torch.bfloat16, torch.float32, torch.float8_e4m3fn)

GatherInput = Union[torch.Tensor, Sequence[torch.Tensor]]


def _allocate(nbytes: int, *, device: torch.device, cpu_group):
    # Imported on first use: custom_all_reduce_v2 needs tvm_ffi, which XPU lacks.
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        _allocate_symmetric_memory,
    )

    return _allocate_symmetric_memory(nbytes, device=device, group=cpu_group)


def _bf16_view(part: torch.Tensor) -> torch.Tensor:
    return part if part.dtype == torch.bfloat16 else part.view(torch.bfloat16)


def _as_dtype(bf16: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return bf16 if dtype == torch.bfloat16 else bf16.view(dtype)


class CpPendingGather:
    """A gather in flight. Shard p is rows ``[p * rows, (p + 1) * rows)`` of
    ``gathered``; shards land in the order own, later ranks, earlier ranks."""

    def __init__(
        self,
        *,
        gathered: torch.Tensor,
        rows: int,
        rank: int,
        events: List[torch.cuda.Event],
        stream: torch.cuda.Stream,
        part_dtypes: List[torch.dtype],
        part_widths: List[int],
        side_gathered: Optional[torch.Tensor],
        planes: Optional[List[torch.Tensor]] = None,
    ) -> None:
        self.gathered = gathered
        self.rows = rows
        self.rank = rank
        self._events = events
        self._stream = stream
        self._part_dtypes = part_dtypes
        self._part_widths = part_widths
        self._side_gathered = side_gathered
        # start_gather_planar: one [W * rows, width] bf16 plane per part.
        self._planes = planes

    @property
    def world_size(self) -> int:
        return len(self._events)

    def runs(self) -> List[Tuple[int, int]]:
        """Shard ranges ``[lo, hi)`` covering every shard in landing order: this
        rank's shard, then the later and the earlier ranks' in blocks of at most
        ``_RUN_SHARDS``."""
        r, world = self.rank, self.world_size
        runs = [(r, r + 1)]
        for lo, hi in ((r + 1, world), (0, r)):
            runs += [
                (start, min(start + _RUN_SHARDS, hi))
                for start in range(lo, hi, _RUN_SHARDS)
            ]
        return runs

    def _wait_shards(self, lo: int, hi: int) -> None:
        stream = torch.cuda.current_stream()
        for p in range(lo, hi):
            if p != self.rank:
                stream.wait_event(self._events[p])

    def rows_of(self, lo: int, hi: int) -> torch.Tensor:
        """The rows of shards ``[lo, hi)``, valid on the current stream from here on."""
        self._wait_shards(lo, hi)
        return self.gathered[lo * self.rows : hi * self.rows]

    def planes_of(self, lo: int, hi: int) -> Tuple[torch.Tensor, ...]:
        """start_gather_planar: every part's rows of shards ``[lo, hi)`` in its dtype,
        valid on the current stream from here on."""
        self._wait_shards(lo, hi)
        rows = slice(lo * self.rows, hi * self.rows)
        return tuple(
            _as_dtype(plane[rows], dtype)
            for plane, dtype in zip(self._planes, self._part_dtypes)
        )

    def finish(self) -> torch.Tensor:
        """All ``W * rows`` rows, valid on the current stream from here on."""
        torch.cuda.current_stream().wait_stream(self._stream)
        return self.gathered

    def finish_parts(self) -> Tuple[torch.Tensor, ...]:
        """finish(), split back into the gathered parts' columns and dtypes."""
        gathered = self.finish()
        if self._planes is not None:
            return tuple(
                _as_dtype(plane, dtype)
                for plane, dtype in zip(self._planes, self._part_dtypes)
            )
        parts, col = [], 0
        for dtype, width in zip(self._part_dtypes, self._part_widths):
            part = gathered[:, col : col + width]
            parts.append(part if dtype == torch.bfloat16 else part.view(dtype))
            col += width
        return tuple(parts)

    def finish_side(self) -> torch.Tensor:
        """The gathered side payload, rank-major like an all-gather, valid on the
        current stream from here on."""
        assert self._side_gathered is not None
        torch.cuda.current_stream().wait_stream(self._stream)
        return self._side_gathered


class CpPrefillSymm:
    """Symmetric gather (and optionally reduce) buffers of one CP group for one
    user: room for ``max_rows`` gathered rows of ``max(gather_cols, width)`` bf16
    columns, reduce slots of ``width`` columns, and ``side_cols`` int32 columns of
    a side payload per row."""

    def __init__(
        self,
        *,
        cpu_group,
        rank: int,
        world_size: int,
        width: int,
        max_rows: int,
        with_reduce: bool,
        side_cols: int = 0,
        gather_cols: int = 0,
    ) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        self.rank = rank
        self.world_size = world_size
        self.width = width
        self.max_rows = max_rows
        self.side_cols = side_cols
        self.gather_cols = max(gather_cols, width)
        gather_region = max_rows * self.gather_cols
        reduce_region = max_rows * width if with_reduce else 0
        # [gather | reduce slots] bf16, then the side payload's int32 columns.
        self._symm_tensor, self._handle = _allocate(
            (gather_region + reduce_region) * 2 + max_rows * side_cols * 4,
            device=device,
            cpu_group=cpu_group,
        )
        self._gather_flat = [
            self._handle.get_buffer(r, [gather_region], torch.bfloat16, 0)
            for r in range(world_size)
        ]
        self._reduce_bufs = (
            [
                self._handle.get_buffer(
                    r, [max_rows, width], torch.bfloat16, gather_region
                )
                for r in range(world_size)
            ]
            if with_reduce
            else None
        )
        self._side_flat = (
            [
                # storage_offset counts int32 elements.
                self._handle.get_buffer(
                    r,
                    [max_rows * side_cols],
                    torch.int32,
                    (gather_region + reduce_region) // 2,
                )
                for r in range(world_size)
            ]
            if side_cols
            else None
        )
        self._stream = torch.cuda.Stream(device=device)
        self._events = [torch.cuda.Event() for _ in range(world_size)]
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)

    def fits(self, x: GatherInput, rows: Optional[int] = None) -> bool:
        """Whether start_gather(x, rows) fits. Depends on shapes and dtypes only, so
        rank-uniform inputs give every rank the same answer."""
        parts = [x] if isinstance(x, torch.Tensor) else list(x)
        if not parts or any(
            p.dim() != 2
            or p.dtype not in _GATHER_DTYPES
            or p.stride(-1) != 1
            or (p.element_size() == 1 and p.shape[1] % 2)
            for p in parts
        ):
            return False
        rows = parts[0].shape[0] if rows is None else rows
        width = sum(p.shape[1] * p.element_size() // 2 for p in parts)
        # The reduce slots and the side payload hold max_rows rows.
        return (
            all(p.shape[0] <= rows for p in parts)
            and rows * self.world_size <= self.max_rows
            and 0 < rows * self.world_size * width <= self.max_rows * self.gather_cols
        )

    def fits_side(self, side: torch.Tensor, rows: int) -> bool:
        """Whether ``side`` ([rows, c] int32, c <= side_cols) can ride a gather."""
        return (
            self._side_flat is not None
            and side.dim() == 2
            and side.dtype == torch.int32
            and side.shape[0] == rows
            and side.shape[1] <= self.side_cols
            and rows * self.world_size <= self.max_rows
        )

    def start_gather(
        self,
        x: GatherInput,
        rows: Optional[int] = None,
        side: Optional[torch.Tensor] = None,
    ) -> CpPendingGather:
        """Gather every rank's ``x``, one tensor or column parts side by side in a
        row, into ``rows``-row slots (rank-uniform, default the first part's rows);
        rows past a part's own are zero. ``side`` (see fits_side) rides the same
        barrier, gathered as an all-gather would."""
        dtypes = [x.dtype] if isinstance(x, torch.Tensor) else [p.dtype for p in x]
        parts = [_bf16_view(p) for p in ([x] if isinstance(x, torch.Tensor) else x)]
        rows = parts[0].shape[0] if rows is None else rows
        widths = [p.shape[1] for p in parts]
        width = sum(widths)
        world, rank = self.world_size, self.rank
        size = world * rows * width
        gathered = self._gather_flat[rank][:size].view(world * rows, width)
        own = gathered[rank * rows : (rank + 1) * rows]
        col = 0
        for part in parts:
            n, w = part.shape
            own[:n, col : col + w].copy_(part)
            if n < rows:
                own[n:, col : col + w].zero_()
            col += w
        side_gathered = None
        if side is not None:
            sc = side.shape[1]
            side_gathered = self._side_flat[rank][: world * rows * sc].view(
                world * rows, sc
            )
            side_gathered[rank * rows : (rank + 1) * rows].copy_(side)
        self._handle.barrier(channel=_GATHER_CHANNEL)
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            for i in range(1, world):
                p = (rank + i) % world
                rows_p = slice(p * rows, (p + 1) * rows)
                if side_gathered is not None:
                    peer_side = self._side_flat[p][: world * rows * sc].view(
                        world * rows, sc
                    )
                    side_gathered[rows_p].copy_(peer_side[rows_p])
                peer = self._gather_flat[p][:size].view(world * rows, width)
                gathered[rows_p].copy_(peer[rows_p])
                self._events[p].record()
        return CpPendingGather(
            gathered=gathered,
            rows=rows,
            rank=rank,
            events=self._events,
            stream=self._stream,
            part_dtypes=dtypes,
            part_widths=widths,
            side_gathered=side_gathered,
        )

    def start_gather_planar(
        self, parts: Sequence[torch.Tensor], side: Optional[torch.Tensor] = None
    ) -> CpPendingGather:
        """start_gather of column parts with the same rows, each gathered into its
        own contiguous [W * rows, width] plane so that an fp8 part stays a dense
        row-major matrix (``planes_of`` / ``finish_parts``); same barrier, one
        copy-engine pull per part and peer."""
        bf16 = [_bf16_view(p) for p in parts]
        rows = bf16[0].shape[0]
        assert all(p.shape[0] == rows for p in bf16)
        world, rank = self.world_size, self.rank
        offsets, off = [], 0
        for part in bf16:
            offsets.append(off)
            off += world * rows * part.shape[1]
        assert off <= self._gather_flat[rank].numel()

        def planes_on(r: int) -> List[torch.Tensor]:
            return [
                self._gather_flat[r][o : o + world * rows * p.shape[1]].view(
                    world * rows, p.shape[1]
                )
                for o, p in zip(offsets, bf16)
            ]

        mine = planes_on(rank)
        own = slice(rank * rows, (rank + 1) * rows)
        for plane, part in zip(mine, bf16):
            plane[own].copy_(part)
        side_gathered = None
        if side is not None:
            sc = side.shape[1]
            side_gathered = self._side_flat[rank][: world * rows * sc].view(
                world * rows, sc
            )
            side_gathered[own].copy_(side)
        self._handle.barrier(channel=_GATHER_CHANNEL)
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            for i in range(1, world):
                p = (rank + i) % world
                rows_p = slice(p * rows, (p + 1) * rows)
                if side_gathered is not None:
                    peer_side = self._side_flat[p][: world * rows * sc].view(
                        world * rows, sc
                    )
                    side_gathered[rows_p].copy_(peer_side[rows_p])
                for plane, peer in zip(mine, planes_on(p)):
                    plane[rows_p].copy_(peer[rows_p])
                self._events[p].record()
        return CpPendingGather(
            gathered=mine[0],
            rows=rows,
            rank=rank,
            events=self._events,
            stream=self._stream,
            part_dtypes=[p.dtype for p in parts],
            part_widths=[p.shape[1] for p in bf16],
            side_gathered=side_gathered,
            planes=mine,
        )

    def reduce_scatter(
        self,
        rows: int,
        write_rows: Optional[Callable[[int, torch.Tensor], None]] = None,
        *,
        write_all: Optional[Callable[[List[torch.Tensor]], None]] = None,
    ) -> torch.Tensor:
        """The sum over ranks of their partials of this rank's ``rows`` rows.

        ``write_rows(p, out)`` writes this rank's partial sums of rank p's rows into
        ``out`` ([rows, width], contiguous, in rank p's memory); ``write_all(outs)``
        instead writes rank p's into ``outs[p]`` for every p at once."""
        assert self._reduce_bufs is not None
        assert (write_rows is None) != (write_all is None)
        world, rank = self.world_size, self.rank
        own = slice(rank * rows, (rank + 1) * rows)
        if write_all is not None:
            write_all([self._reduce_bufs[p][own] for p in range(world)])
        else:
            # Own slot last: the peers' rows go out first, spread over the NVLink ports.
            for i in range(1, world + 1):
                p = (rank + i) % world
                write_rows(p, self._reduce_bufs[p][own])
        self._handle.barrier(channel=_REDUCE_CHANNEL)
        slots = self._reduce_bufs[rank][: world * rows]
        return sum_slots(slots.view(world, rows, self.width))


class CpOwnerScatter:
    """Per-rank symmetric buffers of ``max_rows`` x ``width`` bf16 that the owners
    of a sharded lookup write into (``dst_table``), one CP rank's rows each."""

    def __init__(self, *, group, width: int, max_rows: int) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        self.world_size = group.world_size
        self.rank = group.rank_in_group
        self.width = width
        self.max_rows = max_rows
        self._symm, self._handle = _allocate(
            max_rows * width * 2, device=device, cpu_group=group.cpu_group
        )
        self._bufs = [
            self._handle.get_buffer(r, [max_rows, width], torch.bfloat16, 0)
            for r in range(self.world_size)
        ]
        # [W] int64: every rank's buffer address, for the kernels that write peers.
        self.dst_table = torch.tensor(
            [b.data_ptr() for b in self._bufs], dtype=torch.int64, device=device
        )
        torch.cuda.synchronize()
        dist.barrier(group=group.cpu_group)

    def finish(self, rows: int) -> torch.Tensor:
        """After this rank's owner-write kernel: barrier, then a copy of this rank's
        first ``rows`` rows. Peers write the buffer again in the next scatter."""
        self._handle.barrier(channel=_SCATTER_CHANNEL)
        return self._bufs[self.rank][:rows].clone()


def cp_group_is_tp_group() -> bool:
    """Whether the CP group and the TP group hold the same ranks in the same
    order, so the owner of a TP-sharded row can write it to a CP rank."""
    parallel = get_parallel()
    return (
        parallel.attn_tp_size == 1
        and parallel.attn_dp_size == 1
        and parallel.tp_size == parallel.attn_cp_size
        and parallel.tp_rank == parallel.attn_cp_rank
    )


def _usable(group) -> bool:
    return (
        group.world_size in (2, 4, 8)
        and max_prefill_buffer_tokens() > 0
        and all(in_the_same_node_as(group.cpu_group, source_rank=0))
    )


def get_cp_prefill_symm(
    group,
    *,
    user: str,
    width: int,
    with_reduce: bool = False,
    side_cols: int = 0,
    gather_cols: int = 0,
) -> Optional[CpPrefillSymm]:
    """``user``'s buffers on ``group``, built on first use; None means the NCCL
    path. Building is collective and allocates, so a call under CUDA-graph
    capture returns None and a later eager call builds them."""
    name = (
        f"cp_prefill_symm:{group.unique_name}:{user}:{width}:{side_cols}:{gather_cols}"
    )
    buffers = get_resources().buffers
    if name not in buffers:
        if torch.cuda.is_current_stream_capturing():
            return None
        world = group.world_size
        buffers[name] = (
            CpPrefillSymm(
                cpu_group=group.cpu_group,
                rank=group.rank_in_group,
                world_size=world,
                width=width,
                max_rows=-(-max_prefill_buffer_tokens() // world) * world,
                with_reduce=with_reduce,
                side_cols=side_cols,
                gather_cols=gather_cols,
            )
            if _usable(group)
            else None
        )
    return buffers[name]


def get_cp_owner_scatter(group, *, user: str, width: int) -> Optional[CpOwnerScatter]:
    """``user``'s owner-scatter buffers on ``group`` (rows of ``width`` bf16), built
    on first eager use like get_cp_prefill_symm; None means the collective path."""
    name = f"cp_owner_scatter:{group.unique_name}:{user}:{width}"
    buffers = get_resources().buffers
    if name not in buffers:
        if torch.cuda.is_current_stream_capturing():
            return None
        buffers[name] = (
            CpOwnerScatter(
                group=group,
                width=width,
                max_rows=-(-max_prefill_buffer_tokens() // group.world_size),
            )
            if _usable(group)
            else None
        )
    return buffers[name]
