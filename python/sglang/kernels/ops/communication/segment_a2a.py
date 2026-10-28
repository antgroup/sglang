"""NVLink push all-to-all of strided byte segments (``distributed/segment_a2a.cuh``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from sglang.kernels.ops.communication.all_reduce import Communicator


@cache_once
def _module(world_size: int, unroll: int):
    args = make_cpp_args(world_size, unroll, is_arch_support_pdl())
    return load_jit(
        "segment_a2a",
        *args,
        cuda_files=["distributed/segment_a2a.cuh"],
        cuda_wrappers=[("segment_a2a", f"SegmentA2A<{args}>::run")],
    )


def segment_a2a(
    comm: Communicator,
    src: torch.Tensor,
    *,
    dst_offset: int,
    src_peer_stride: int,
    src_row_stride: int,
    dst_rank_stride: int,
    dst_row_stride: int,
    seg_bytes: int,
    rows: int,
    valid_rows: int,
    num_blocks: int,
    num_threads: int,
    unroll: int,
) -> None:
    """Segment (p, t) of ``src`` (byte offset ``p * src_peer_stride + t *
    src_row_stride``, ``seg_bytes`` long) lands in rank p's pull workspace at
    ``dst_offset + rank * dst_rank_stride + t * dst_row_stride``; segments with
    ``p * rows + t >= valid_rows`` are sent as zeros. Collective over the
    communicator's ranks, which must pass identical geometry and grid."""
    _module(comm.world_size, unroll).segment_a2a(
        comm,
        src,
        dst_offset,
        src_peer_stride,
        src_row_stride,
        dst_rank_stride,
        dst_row_stride,
        seg_bytes,
        rows,
        valid_rows,
        num_blocks,
        num_threads,
    )
