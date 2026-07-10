from typing import Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import DEFAULT_QUANTILES
from sglang.jit_kernel.ring_kv_cache import gather_ring_kv
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(est_time=8, suite="stage-b-kernel-benchmark-1-gpu-large")

DTYPE = torch.bfloat16
DEVICE = "cuda"
BATCH_SIZE = 1
CACHE_TOKENS = 37_440
SINK_TOKENS = 14_040
ROW_ELEMENTS = 5 * 128
TAIL_START = 4_680

CONFIGS = [(14_040, 9_360)] if is_in_ci() else [(14_040, 9_360), (14_040, 23_400)]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["sink_length", "recent_length"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=["jit", "torch"],
        line_names=["SGL JIT fused K/V gather", "torch.cat K + V"],
        styles=[("blue", "--"), ("green", "-.")],
        ylabel="us",
        plot_name="lingbot-ring-kv-gather-performance",
        args={},
    )
)
def benchmark(
    sink_length: int, recent_length: int, provider: str
) -> Tuple[float, float, float]:
    k_cache = torch.randn(
        BATCH_SIZE,
        CACHE_TOKENS,
        ROW_ELEMENTS,
        dtype=DTYPE,
        device=DEVICE,
    )
    v_cache = torch.randn_like(k_cache)
    output_tokens = sink_length + recent_length
    k_out = torch.empty_like(k_cache)[:, :output_tokens]
    v_out = torch.empty_like(v_cache)[:, :output_tokens]
    recent_start = CACHE_TOKENS - recent_length
    tail_capacity = CACHE_TOKENS - SINK_TOKENS
    recent_relative = (TAIL_START + recent_start - SINK_TOKENS) % tail_capacity
    recent_first = min(recent_length, tail_capacity - recent_relative)
    recent_ranges = [(SINK_TOKENS + recent_relative, recent_first)]
    if recent_first < recent_length:
        recent_ranges.append((SINK_TOKENS, recent_length - recent_first))
    k_parts = [
        k_cache[:, :sink_length],
        *[k_cache[:, start : start + length] for start, length in recent_ranges],
    ]
    v_parts = [
        v_cache[:, :sink_length],
        *[v_cache[:, start : start + length] for start, length in recent_ranges],
    ]

    if provider == "jit":

        def fn():
            gather_ring_kv(
                k_cache,
                v_cache,
                k_out,
                v_out,
                sink_tokens=SINK_TOKENS,
                tail_start=TAIL_START,
                first_start=0,
                first_length=sink_length,
                second_start=recent_start,
                second_length=recent_length,
            )

    else:

        def fn():
            return torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)

    return tuple(
        value * 1000
        for value in triton.testing.do_bench_cudagraph(fn, quantiles=DEFAULT_QUANTILES)
    )


if __name__ == "__main__":
    benchmark.run(print_data=True)
