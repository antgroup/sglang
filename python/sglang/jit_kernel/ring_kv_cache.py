from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_ring_kv_cache_module(row_bytes: int) -> Module:
    args = make_cpp_args(row_bytes)
    return load_jit(
        "ring_kv_cache",
        *args,
        cuda_files=["elementwise/ring_kv_cache.cuh"],
        cuda_wrappers=[("gather_ring_kv", f"GatherRingKVKernel<{args}>::run")],
    )


@cache_once
def can_use_ring_kv_gather(row_bytes: int) -> bool:
    if row_bytes <= 0 or row_bytes % 16 != 0:
        return False
    try:
        _jit_ring_kv_cache_module(row_bytes)
        return True
    except Exception as error:
        logging.getLogger(__name__).warning(
            "Failed to load JIT ring KV gather for row_bytes=%s: %s",
            row_bytes,
            error,
        )
        return False


@register_custom_op(mutates_args=["k_out", "v_out"])
def gather_ring_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    *,
    sink_tokens: int,
    tail_start: int,
    first_start: int,
    first_length: int,
    second_start: int = 0,
    second_length: int = 0,
) -> None:
    """Gather up to two logical ranges from a sink-plus-ring KV cache.

    All tensors are shaped ``[batch, tokens, row_elements]``. Logical indices
    below ``sink_tokens`` address the fixed prefix; later indices address the
    circular tail whose oldest entry is at ``tail_start``.
    """
    row_bytes = k_cache.shape[-1] * k_cache.element_size()
    module = _jit_ring_kv_cache_module(row_bytes)
    module.gather_ring_kv(
        k_cache,
        v_cache,
        k_out,
        v_out,
        sink_tokens,
        tail_start,
        first_start,
        first_length,
        second_start,
        second_length,
    )
