import itertools
import sys

import pytest
import torch

from sglang.jit_kernel.ring_kv_cache import gather_ring_kv
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=12, suite="stage-b-kernel-unit-1-gpu-large")


def _physical_cache(logical: torch.Tensor, sink_tokens: int, tail_start: int):
    cache = torch.empty_like(logical)
    cache[:, :sink_tokens].copy_(logical[:, :sink_tokens])
    tail_capacity = logical.shape[1] - sink_tokens
    for logical_idx in range(sink_tokens, logical.shape[1]):
        physical_idx = (
            sink_tokens + (tail_start + logical_idx - sink_tokens) % tail_capacity
        )
        cache[:, physical_idx].copy_(logical[:, logical_idx])
    return cache


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "tail_start,ranges",
    list(
        itertools.product(
            [0, 2, 6],
            [
                [(0, 3), (5, 4)],
                [(1, 8)],
                [(3, 7)],
            ],
        )
    ),
)
def test_gather_ring_kv_is_bitwise_exact(dtype, tail_start, ranges):
    batch_size = 2
    cache_tokens = 10
    sink_tokens = 3
    row_elements = 64
    logical_k = torch.randn(
        batch_size,
        cache_tokens,
        row_elements,
        dtype=dtype,
        device="cuda",
    )
    logical_v = torch.randn_like(logical_k)
    k_cache = _physical_cache(logical_k, sink_tokens, tail_start)
    v_cache = _physical_cache(logical_v, sink_tokens, tail_start)

    output_tokens = sum(length for _, length in ranges)
    # Slice a larger allocation so batch stride differs from output_tokens.
    k_backing = torch.empty_like(k_cache)
    v_backing = torch.empty_like(v_cache)
    k_out = k_backing[:, :output_tokens]
    v_out = v_backing[:, :output_tokens]
    first_start, first_length = ranges[0]
    second_start, second_length = ranges[1] if len(ranges) > 1 else (0, 0)
    gather_ring_kv(
        k_cache,
        v_cache,
        k_out,
        v_out,
        sink_tokens=sink_tokens,
        tail_start=tail_start,
        first_start=first_start,
        first_length=first_length,
        second_start=second_start,
        second_length=second_length,
    )

    logical_indices = torch.cat(
        [torch.arange(start, start + length, device="cuda") for start, length in ranges]
    )
    torch.testing.assert_close(k_out, logical_k[:, logical_indices], atol=0, rtol=0)
    torch.testing.assert_close(v_out, logical_v[:, logical_indices], atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
