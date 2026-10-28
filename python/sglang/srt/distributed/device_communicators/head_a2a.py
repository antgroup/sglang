"""Head <-> row all-to-all of decode attention over NVLink symmetric memory.

Under the DSV4.1 head<->row exchange every TP rank holds ``[rows, H, D]``
(its ``H`` heads of every row) and the attention wants ``[chunk, W * H, D]``
(``chunk = ceil(rows / W)`` rows with all ``W * H`` heads), and back. Instead
of the NCCL all-to-all plus a permute on each side, one push kernel per
direction stores every segment straight into the receiver's symmetric buffer
at its final layout. Results are views of that buffer: each stays valid until
the next exchange on the same instance, so its consumers must run on the
calling stream before that exchange (the attention reads q, the output
projection reads o). Bytes are moved unmodified, so outputs are bitwise those
of the NCCL path, zero row padding included.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.all_reduce import Communicator, PullPlane
from sglang.kernels.ops.communication.segment_a2a import segment_a2a
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
    _allocate_symmetric_memory,
)
from sglang.srt.runtime_context import get_resources

logger = logging.getLogger(__name__)

_SEMAPHORE_BYTES = 128  # sizeof(device::distributed::Semaphore)
_ALIGN_BYTES = 1024
# Measured on H20-3e TP4 (3-6 MB per rank): within 5% of the best over 16-128
# blocks x 256-1024 threads x unroll 2-8. Fixed, as every rank must launch the same grid.
_NUM_BLOCKS = 64
_NUM_THREADS = 512
_UNROLL = 4
# Semaphore slots: the cap on any launch's grid.
_MAX_BLOCKS = 128
# Fallback row capacity when the scheduler config is not published (offline use).
_DEFAULT_CAPACITY_ROWS = 1024
_MAX_CAPACITY_BYTES = 64 << 20


class SymmHeadA2A:
    """The two directions of the exchange on one TP group (rank order = head order)."""

    def __init__(self, *, group, capacity_bytes: int) -> None:
        cpu_group = group.cpu_group
        self.world_size = int(group.world_size)
        self.rank = int(group.rank_in_group)
        assert dist.get_rank(group=cpu_group) == self.rank
        device = torch.device("cuda", torch.cuda.current_device())
        self.capacity_bytes = -(-capacity_bytes // _ALIGN_BYTES) * _ALIGN_BYTES
        sem_bytes = _MAX_BLOCKS * _SEMAPHORE_BYTES
        total = self.capacity_bytes + sem_bytes
        self._symm_tensor, symm_mem = _allocate_symmetric_memory(
            total, device=device, group=cpu_group
        )
        slabs = [
            symm_mem.get_buffer(i, [total], torch.uint8) for i in range(self.world_size)
        ]
        # The semaphores must start zeroed on every rank before any peer signals.
        slabs[self.rank].zero_()
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        self._buffer = slabs[self.rank][: self.capacity_bytes]
        self.comm = Communicator(
            pull=PullPlane(
                self.rank,
                self.world_size,
                workspaces=[slab[: self.capacity_bytes] for slab in slabs],
                semaphores=[
                    slab[self.capacity_bytes :].view(_MAX_BLOCKS, _SEMAPHORE_BYTES)
                    for slab in slabs
                ],
            )
        )
        if self.rank == 0:
            logger.info(
                "Symmetric-memory head<->row a2a: %.2f MB receive buffer, world %d",
                self.capacity_bytes / (1 << 20),
                self.world_size,
            )

    def to_row_layout(self, q_local: torch.Tensor) -> Optional[torch.Tensor]:
        """``[rows, H, D]`` -> ``[chunk, W * H, D]``: rows ``[rank * chunk, (rank +
        1) * chunk)`` with every rank's heads, rows past ``rows`` zero. None when
        the input does not fit (the caller then takes the NCCL path)."""
        if q_local.dim() != 3:
            return None
        rows, heads, dim = q_local.shape
        world = self.world_size
        chunk = -(-rows // world)
        seg = heads * dim * q_local.element_size()
        if not self._fits(q_local, seg=seg, out_bytes=chunk * world * seg):
            return None
        self._exchange(
            q_local,
            src_peer_stride=chunk * seg,
            src_row_stride=seg,
            dst_rank_stride=seg,
            dst_row_stride=world * seg,
            seg_bytes=seg,
            rows=chunk,
            valid_rows=rows,
        )
        return self._result(chunk * world * seg, q_local.dtype).view(
            chunk, world * heads, dim
        )

    def to_head_layout(self, o_rows: torch.Tensor) -> Optional[torch.Tensor]:
        """``[chunk, W * H, D]`` (this rank's rows) -> ``[W * chunk, H, D]`` (this
        rank's heads of every rank's rows). None when the input does not fit."""
        if o_rows.dim() != 3 or o_rows.shape[1] % self.world_size:
            return None
        chunk, all_heads, dim = o_rows.shape
        world = self.world_size
        heads = all_heads // world
        seg = heads * dim * o_rows.element_size()
        if not self._fits(o_rows, seg=seg, out_bytes=chunk * world * seg):
            return None
        self._exchange(
            o_rows,
            src_peer_stride=seg,
            src_row_stride=world * seg,
            dst_rank_stride=chunk * seg,
            dst_row_stride=seg,
            seg_bytes=seg,
            rows=chunk,
            valid_rows=chunk * world,
        )
        return self._result(chunk * world * seg, o_rows.dtype).view(
            world * chunk, heads, dim
        )

    def _fits(self, x: torch.Tensor, *, seg: int, out_bytes: int) -> bool:
        # Every operand is shape-derived, so all ranks take the same branch.
        return (
            x.is_cuda
            and x.is_contiguous()
            and x.numel() > 0
            and x.data_ptr() % 16 == 0
            and seg % 16 == 0
            and out_bytes <= self.capacity_bytes
        )

    def _exchange(self, src: torch.Tensor, **geometry: int) -> None:
        segment_a2a(
            self.comm,
            src,
            dst_offset=0,
            num_blocks=_NUM_BLOCKS,
            num_threads=_NUM_THREADS,
            unroll=_UNROLL,
            **geometry,
        )

    def _result(self, nbytes: int, dtype: torch.dtype) -> torch.Tensor:
        return self._buffer[:nbytes].view(dtype)


def _capacity_bytes(*, world_size: int, local_heads: int, head_bytes: int) -> int:
    """Room for the largest decode/verify batch the scheduler runs, padded to
    whole chunks."""
    from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
        recommended_max_tokens,
    )

    rows = recommended_max_tokens(include_prefill=False, floor=0)
    rows = rows if rows > 0 else _DEFAULT_CAPACITY_ROWS
    padded = -(-rows // world_size) * world_size
    return min(padded * local_heads * head_bytes, _MAX_CAPACITY_BYTES)


def get_symm_head_a2a(
    group, *, local_heads: int, head_bytes: int
) -> Optional[SymmHeadA2A]:
    """The group's exchange, built on first use; None means take the NCCL path.

    Building is collective over ``group`` (every rank reaches the same call
    with the same shapes) and allocates, so it never happens under CUDA-graph
    capture: such a call returns None and a later eager call builds it."""
    name = f"symm_head_a2a:{group.unique_name}"
    buffers = get_resources().buffers
    if name not in buffers:
        if torch.cuda.is_current_stream_capturing():
            return None
        # A group whose custom all-reduce v2 is up is a single-node full-NVLink
        # clique, which is what the peer stores need.
        ca_comm = group.ca_comm
        usable = (
            isinstance(ca_comm, CustomAllReduceV2)
            and not ca_comm.disabled
            and group.world_size in (2, 4, 8)
        )
        buffers[name] = (
            SymmHeadA2A(
                group=group,
                capacity_bytes=_capacity_bytes(
                    world_size=group.world_size,
                    local_heads=local_heads,
                    head_bytes=head_bytes,
                ),
            )
            if usable
            else None
        )
    return buffers[name]
