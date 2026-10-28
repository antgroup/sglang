"""Tile-scheduler metadata for the FlashMLA SM90 sparse decode kernel.

FlashMLA's get_mla_metadata_kernel runs its greedy split on one thread
(~16-40 us per call on H20, 3-4 calls per decode step) and charges every
split piece 5 blocks of overhead, which on H20 (MMA-bound, ~5 us per 64-token
block) can leave SMs idle: 48 rows x 10 blocks run on 48 of 78 SMs. The JIT
kernel (csrc/deepseek_v4/flashmla_sparse_sched.cuh) runs the same greedy for a
few candidate overheads, one warp each, scores each schedule with a cost
model fitted on H20 and writes the best; candidates (5,) is the stock
schedule bit for bit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module

SCHED_META_INTS = 8
STOCK_OVERHEAD = 5
# Cost model in 0.1 us: per 64-token block, per piece, per split piece, once
# if any request splits. Fitted on H20 to FlashMLA sparse decode wall times
# (kernel + combine; 6..320 rows, 2..10 blocks/row, uniform, padded and
# random lengths x overheads 0..5; picks the oracle schedule within 0.1%);
# re-fit when the kernel or the GPU changes.
DEFAULT_COST = (50, 20, 20, 40)
DEFAULT_OVERHEADS = (STOCK_OVERHEAD, 3, 2, 1)


@cache_once
def _jit_module() -> Module:
    return load_jit(
        make_name("flashmla_sparse_sched"),
        cuda_files=["deepseek_v4/flashmla_sparse_sched.cuh"],
        cuda_wrappers=[("run", "FlashMLASparseSchedKernel::run")],
        extra_cuda_cflags=["-O3"],
    )


def flashmla_sparse_sched_meta(
    *,
    topk_length: torch.Tensor,
    extra_topk_length: Optional[torch.Tensor],
    topk: int,
    extra_topk: int,
    num_sm_parts: int,
    overheads: Sequence[int] = DEFAULT_OVERHEADS,
    cost: Tuple[int, int, int, int] = DEFAULT_COST,
    per_part: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(tile_scheduler_metadata [num_sm_parts, 8], num_splits [b + 1]) int32
    for flash_mla_with_kvcache sparse decode with s_q = 1. topk_length [b]
    int32 is required; extra_topk = 0 means no extra KV cache. On ties the
    earlier candidate wins, so list the stock overhead first. per_part walks
    the greedy one partition per step (same output, fewer serial steps)."""
    assert 1 <= len(overheads) <= 4
    b = topk_length.shape[0]
    device = topk_length.device
    meta = torch.empty(
        (num_sm_parts, SCHED_META_INTS), dtype=torch.int32, device=device
    )
    num_splits = torch.empty((b + 1,), dtype=torch.int32, device=device)
    ovs = list(overheads) + [STOCK_OVERHEAD] * (4 - len(overheads))
    _jit_module().run(
        topk_length,
        topk_length if extra_topk_length is None else extra_topk_length,
        meta,
        num_splits,
        topk,
        extra_topk,
        int(extra_topk_length is not None),
        len(overheads),
        *ovs,
        *cost,
        int(per_part),
    )
    return meta, num_splits
