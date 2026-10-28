"""DeepSeek-V4.1 FP8 block-[32, 32] grouped wo_a on SM90 (``attn_proj``).

Upstream dequantizes a [32, 32] wo_a to bf16 at load. On this path wo_a keeps its
FP8 weight and 32x32 scales and multiplies the per-32 FP8 rounding of its input
(``fp8_block32_quant_mn_major``, the V4.1 reference's activation rounding), so
it multiplies the operands the BF16 path would, accumulated in fp32 per 32-wide
K block. Routes, measured on H20-3e:

* up to 16 rows, the split-K Triton kernel (DeepGEMM has no split-K);
* one group (attention TP8): DeepGEMM;
* a row count the device's swap-AB table lists for the grouped shape: one
  block-diagonal swap-AB launch over all groups (split 1, bitwise DeepGEMM);
* otherwise one DeepGEMM 1x32x32 GEMM per group, or the per-group swap-AB GEMM
  where the table lists it, groups forked over side streams. DeepGEMM's einsum
  and grouped GEMMs take 128-wide K scales only.
"""

from collections.abc import Callable, Sequence

import torch

from sglang.kernels.ops.gemm.dsv41_wo_a_fp8 import (
    can_use_wo_a_block32_small_gemm,
    wo_a_block32_small_gemm,
)
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.quantization.fp8_block32_sm90 import (
    Block32Routes,
    block32_swapab_grouped_matmul,
    block32_swapab_matmul,
)

# Rows up to which the split-K Triton kernel beats DeepGEMM; measured on H20-3e
# (G = 1: 6.9 vs 10.2 us at 5 rows, still ahead at 40).
_SMALL_ROWS = 16
# Concurrent groups; measured on H20-3e, 4 and 8 streams time the same.
NUM_SIDE_STREAMS = 4


def _group_gemm(
    group: int,
    *,
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    routes: Block32Routes,
) -> None:
    k_blocks = a.shape[2] // 32
    r = out.shape[2]
    lhs = (a[:, group], a_scale[:, group * k_blocks : (group + 1) * k_blocks])
    rhs = (
        weight[group * r : (group + 1) * r],
        weight_scale[group * r // 32 : (group + 1) * r // 32],
    )
    # The swap-AB table lists wo_a's per-group shape at decode row counts.
    if (
        block32_swapab_matmul(*lhs, *rhs, table=routes.swapab, out=out[:, group])
        is not None
    ):
        return
    deep_gemm_wrapper.gemm_nt_f8f8bf16(lhs, rhs, out[:, group], recipe=(1, 32, 32))


def _for_each_group(
    num_groups: int, fn: Callable[[int], None], streams: Sequence[torch.cuda.Stream]
) -> None:
    if num_groups == 1 or not streams:
        for group in range(num_groups):
            fn(group)
        return
    main = torch.cuda.current_stream()
    used = streams[:num_groups]
    for stream in used:
        stream.wait_stream(main)
    for group in range(num_groups):
        with torch.cuda.stream(used[group % len(used)]):
            fn(group)
    # Every tensor here is allocated on main and next read there; joining before
    # the last reference drops also closes the fork inside a CUDA graph capture.
    for stream in used:
        main.wait_stream(stream)


def wo_a_fp8_block32_gemm(
    *,
    o_fp8: torch.Tensor,
    o_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    num_groups: int,
    routes: Block32Routes,
    streams: Sequence[torch.cuda.Stream] = (),
) -> torch.Tensor:
    """o_fp8 [T, G * K] e4m3 with o_scale its MN-major fp32 view [T, G * K / 32];
    weight [G * R, K] e4m3 with weight_scale [G * R / 32, K / 32] fp32 -> [T, G, R]
    bf16."""
    t = o_fp8.shape[0]
    n, k = weight.shape
    r = n // num_groups
    assert o_fp8.shape[1] == num_groups * k and r % 32 == 0 and k % 32 == 0
    out = torch.empty((t, num_groups, r), dtype=torch.bfloat16, device=o_fp8.device)
    if t == 0:
        return out
    if t <= _SMALL_ROWS and can_use_wo_a_block32_small_gemm(group_n=r, k=k):
        wo_a_block32_small_gemm(o_fp8, o_scale, weight, weight_scale, out)
        return out
    if num_groups == 1:
        # Bitwise the split-1 swap-AB GEMM and faster on H20 at 17-256 rows (11.0
        # vs 18.5 us at 64 rows, cold L2); the table's split-3 rows change the bits.
        deep_gemm_wrapper.gemm_nt_f8f8bf16(
            (o_fp8, o_scale), (weight, weight_scale), out.view(t, n), recipe=(1, 32, 32)
        )
        return out
    if (
        block32_swapab_grouped_matmul(
            o_fp8,
            o_scale,
            weight,
            weight_scale,
            groups=num_groups,
            table=routes.swapab,
            out=out.view(t, n),
        )
        is not None
    ):
        return out
    _for_each_group(
        num_groups,
        lambda group: _group_gemm(
            group,
            a=o_fp8.view(t, num_groups, k),
            a_scale=o_scale,
            weight=weight,
            weight_scale=weight_scale,
            out=out,
            routes=routes,
        ),
        streams,
    )
    return out
