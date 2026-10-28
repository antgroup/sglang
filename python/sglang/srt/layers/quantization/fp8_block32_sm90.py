"""H20-measured routes of SM90 block-[32, 32] FP8 linears (DeepSeek-V4.1, ``fp8_gemm``).

Upstream quantizes the activation row-major and runs DeepGEMM at every row count.
With ``dsv41_sm90_on("fp8_gemm")`` a ue8m0 checkpoint's linear instead:

* quantizes straight into DeepGEMM's MN-major scale layout (the same FP8 values
  and scales, without DeepGEMM's transpose launch);
* takes the swap-AB Gluon GEMM at the row counts the device's table lists for the
  shape (split 1 is bitwise DeepGEMM; split > 1 sums fp32 K partials in a fixed
  order);
* takes the measured split-K Triton GEMM at decode-sized M on narrow-N/long-K or
  short-K shapes, which SM90 DeepGEMM (no split-K) leaves mostly idle.

The tables are looked up by device name and role. A device without them keeps
DeepGEMM at every row count; only the MN-major quantization stays.
"""

import logging
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.gemm.fp8_block32_swapab import (
    SwapABTable,
    block32_swapab_gemm,
    load_swapab_table,
    lookup_swapab,
    num_tiles,
)
from sglang.kernels.ops.gemm.fp8_kernel import (
    get_w8a8_block32_small_m_configs,
    w8a8_block_fp8_matmul_deepgemm,
    w8a8_block_fp8_matmul_triton,
)
from sglang.kernels.ops.quantization.fp8_block32_act_quant import (
    block32_act_quant_ue8m0_mn_major,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.utils.common import log_info_on_rank0

logger = logging.getLogger(__name__)

_BLOCK = [32, 32]
# Measured on H20-3e at M = 5, Triton vs DeepGEMM: 1792x5120 12.7 vs 14.5 us,
# 576x5120 9.1 vs 13.7 us, 5120x288 3.9 vs 5.0 us; 4096x1280 and 5120x1024 lose.
_TRITON_MAX_M = 16
# A Triton-routed shape the device table lacks: 16 is the smallest FP8 MMA M tile,
# with the same K accumulation and scale boundaries as the default config.
_TRITON_SMALL_M_DEFAULT = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 3,
}


class Block32Routes(msgspec.Struct, frozen=True):
    swapab: SwapABTable
    triton_small_m: Dict[Tuple[int, int], Dict[int, Dict[str, Any]]]
    # Rows up to which the Triton shapes leave DeepGEMM; 0 on unmeasured devices.
    triton_max_m: int


@lru_cache(maxsize=None)
def load_block32_routes(*, device_name: str, role: str) -> Block32Routes:
    swapab = load_swapab_table(device_name, role)
    triton_small_m = get_w8a8_block32_small_m_configs(device_name)
    routes = Block32Routes(
        swapab=swapab,
        triton_small_m=triton_small_m,
        triton_max_m=_TRITON_MAX_M if triton_small_m else 0,
    )
    log_info_on_rank0(
        logger,
        f"SM90 block32 FP8 linears on {device_name} ({role}): swap-AB rows for "
        f"{len(swapab)} shapes, Triton up to M={routes.triton_max_m} for "
        "narrow-N/long-K and short-K shapes, DeepGEMM otherwise.",
    )
    return routes


def _prefers_triton(*, routes: Block32Routes, m: int, n: int, k: int) -> bool:
    if m > routes.triton_max_m or not ((n <= 2048 and k >= 4096) or k <= 512):
        return False
    # A row count the swap-AB table lists takes it over Triton.
    return torch._dynamo.is_compiling() or lookup_swapab(routes.swapab, n, k, m) is None


def _triton_config(*, routes: Block32Routes, m: int, n: int, k: int) -> dict:
    configs = routes.triton_small_m.get((n, k))
    if not configs:
        return _TRITON_SMALL_M_DEFAULT
    return configs[min(configs.keys(), key=lambda x: abs(x - m))]


def _operands_fit(
    q: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: Optional[torch.Tensor],
) -> bool:
    # TMA: 16-byte aligned base addresses and row strides, unit inner stride.
    n, k = weight.shape
    return (
        q.dtype == torch.float8_e4m3fn
        and weight.dtype == torch.float8_e4m3fn
        and q.stride(1) == 1
        and q.stride(0) % 16 == 0
        and q.data_ptr() % 16 == 0
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
        and k % 32 == 0
        and n % 8 == 0
        and x_scale.dtype == torch.float32
        and x_scale.stride(0) == 1
        and x_scale.stride(1) % 4 == 0
        and x_scale.data_ptr() % 16 == 0
        and weight_scale.dtype == torch.float32
        and weight_scale.is_contiguous()
        and tuple(weight_scale.shape) == ((n + 31) // 32, k // 32)
        and (
            out is None
            or (
                out.dtype == torch.bfloat16
                and out.stride(1) == 1
                and out.stride(0) % 8 == 0
                and out.data_ptr() % 16 == 0
            )
        )
    )


# (device index, weight data_ptr) -> int32 tile counters, zero between calls.
_split_counters: Dict[Tuple[int, int], torch.Tensor] = {}
# Replaced counter buffers stay alive: captured graphs keep their addresses.
_retired_counters = []


def _max_split_tiles(table: SwapABTable, n: int, k: int) -> int:
    # Tiles grow with m for a fixed config, and m never exceeds its entry's key.
    return max(
        (
            num_tiles(n, m, cfg.bt1, cfg.bt2)
            for m, cfg in table.get((n, k), ())
            if cfg and cfg.split > 1
        ),
        default=0,
    )


def _counters_for(
    weight: torch.Tensor, tiles: int, table: SwapABTable
) -> Optional[torch.Tensor]:
    # One buffer per weight: two concurrent split GEMMs on the same weight
    # would share counters, which no decode graph does.
    key = (weight.device.index, weight.data_ptr())
    buf = _split_counters.get(key)
    if buf is None or buf.numel() < tiles:
        if torch.cuda.is_current_stream_capturing():
            # A buffer allocated here would live in this graph's pool.
            return None
        if buf is not None:
            _retired_counters.append(buf)
        size = max(tiles, _max_split_tiles(table, *weight.shape))
        buf = torch.zeros(size, dtype=torch.int32, device=weight.device)
        _split_counters[key] = buf
    return buf


def block32_swapab_matmul(
    q: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    table: SwapABTable,
    out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """bf16 [M, N] from the swap-AB GEMM, or None to keep DeepGEMM.

    q [M, K] e4m3 with x_scale its MN-major fp32 view [M, K / 32] (as from
    block32_act_quant_ue8m0_mn_major); weight [N, K] e4m3 with weight_scale
    [ceil(N / 32), K / 32] fp32.
    """
    if torch._dynamo.is_compiling():
        return None
    m, k = q.shape
    n = weight.shape[0]
    cfg = lookup_swapab(table, n, k, m)
    if cfg is None or not _operands_fit(q, x_scale, weight, weight_scale, out):
        return None
    counters = None
    if cfg.split > 1:
        counters = _counters_for(weight, num_tiles(n, m, cfg.bt1, cfg.bt2), table)
        if counters is None:
            return None
    return block32_swapab_gemm(
        q,
        x_scale,
        weight,
        weight_scale,
        bt1=cfg.bt1,
        bt2=cfg.bt2,
        stages=cfg.stages,
        split=cfg.split,
        maxnreg=cfg.maxnreg,
        counters=counters,
        out=out,
    )


def block32_swapab_grouped_matmul(
    q: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    groups: int,
    table: SwapABTable,
    out: torch.Tensor,
) -> Optional[torch.Tensor]:
    """``out`` [M, N] bf16 from one swap-AB launch, or None to keep the per-group GEMMs.

    Block-diagonal: out[:, g * N / G:][:, :N / G] = q[:, g * K:][:, :K] @
    weight[g * N / G:][:N / G].T with q [M, G * K] e4m3, x_scale its MN-major fp32
    view [M, G * K / 32], weight [N, K] e4m3 and weight_scale [N / 32, K / 32] fp32.
    """
    if torch._dynamo.is_compiling():
        return None
    m = q.shape[0]
    n, k = weight.shape
    cfg = lookup_swapab(table, n // groups, k, m, groups)
    if (
        cfg is None
        or q.shape[1] != groups * k
        or not _operands_fit(q, x_scale, weight, weight_scale, out)
    ):
        return None
    return block32_swapab_gemm(
        q,
        x_scale,
        weight,
        weight_scale,
        bt1=cfg.bt1,
        bt2=cfg.bt2,
        stages=cfg.stages,
        split=1,
        maxnreg=cfg.maxnreg,
        counters=None,
        out=out,
        group_n=n // groups,
    )


def _prequantized_linear(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
    routes: Block32Routes,
) -> torch.Tensor:
    # Only MN-major input scales fit the swap-AB GEMM; DeepGEMM takes either layout.
    input_2d = input.view(-1, input.shape[-1])
    output = block32_swapab_matmul(
        input_2d, input_scale, weight, weight_scale, table=routes.swapab
    )
    if output is None:
        output = w8a8_block_fp8_matmul_deepgemm(
            input_2d,
            weight,
            input_scale,
            weight_scale,
            _BLOCK,
            output_dtype=torch.bfloat16,
        )
    return output


def _triton_linear(
    *,
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    routes: Block32Routes,
) -> torch.Tensor:
    m, k = input_2d.shape
    q_input, x_scale = sglang_per_token_group_quant_fp8(
        input_2d, _BLOCK[1], scale_ue8m0=True
    )
    return w8a8_block_fp8_matmul_triton(
        q_input,
        weight,
        x_scale,
        weight_scale,
        _BLOCK,
        output_dtype=input_2d.dtype,
        config=_triton_config(routes=routes, m=m, n=weight.shape[0], k=k),
        fixed_order_split_k=True,
    )


def block32_routed_linear(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    routes: Block32Routes,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """A DeepGEMM-servable bf16 block32 linear with ue8m0 activation scales.

    ``out`` (2D bf16 input only): a [M, N] row-major view, e.g. a column slice of a
    wider buffer, that receives the result; DeepGEMM and swap-AB write it directly.
    """
    n, k = weight.shape
    output_shape = [*input.shape[:-1], n]
    if input_scale is not None:
        assert out is None and input.dtype == torch.float8_e4m3fn
        output = _prequantized_linear(
            input=input,
            weight=weight,
            weight_scale=weight_scale,
            input_scale=input_scale,
            routes=routes,
        )
    else:
        assert out is None or input.dim() == 2
        input_2d = input.view(-1, k).contiguous()
        if _prefers_triton(routes=routes, m=input_2d.shape[0], n=n, k=k):
            output = _triton_linear(
                input_2d=input_2d,
                weight=weight,
                weight_scale=weight_scale,
                routes=routes,
            )
            if out is not None:
                out.copy_(output)
                output = out
        else:
            q_input, x_scale = block32_act_quant_ue8m0_mn_major(input_2d)
            output = block32_swapab_matmul(
                q_input, x_scale, weight, weight_scale, table=routes.swapab, out=out
            )
            if output is None:
                output = w8a8_block_fp8_matmul_deepgemm(
                    q_input,
                    weight,
                    x_scale,
                    weight_scale,
                    _BLOCK,
                    output_dtype=input.dtype,
                    out=out,
                )
    if bias is not None:
        output += bias
    return output if out is not None else output.view(*output_shape)
