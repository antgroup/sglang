"""DeepSeek-V4.1 SM90 attention projections (``dsv41_sm90_on("attn_proj")``).

The V4.1 release reference rounds the attention output through per-32 FP8
(``fp8_block32_quant_dequant``) before its grouped wo_a projection, in the target
and in the DSpark draft; upstream SM90 multiplies it unrounded. On this path:

* the target keeps a [32, 32] FP8 wo_a and runs the FP8 grouped GEMM on the
  rounded operands (``wo_a_fp8_block32``); a bf16 wo_a (the DSpark draft, or a
  checkpoint the FP8 GEMM cannot take) multiplies the rounded bf16 values;
* the inverse RoPE of the wo_a input runs inside that rounding: per head for
  decode and verify up to 384 rows, four heads per program from 1024 rows;
* the forward q RoPE without a padded buffer runs several heads per program
  from 1024 rows;
* on the decode role the row-tiled kernels also serve fewer rows, and the q RoPE
  of the non-CP K store path rides in the K norm/RoPE/FlashMLA store launch.

Every fused kernel is bitwise the kernels it replaces. The layer owns weights and
modes; this module owns the dispatch.
"""

from typing import Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4 import fused_rope_inplace
from sglang.kernels.ops.attention.dsv4.q_rope_rows import (
    can_use_q_rope_rows,
    q_rope_rows_inplace,
)
from sglang.kernels.ops.attention.dsv4.wo_a_rope_quant import (
    can_use_wo_a_rope_quant,
    can_use_wo_a_rope_quant_rows,
    wo_a_rope_qdq,
    wo_a_rope_qdq_rows,
    wo_a_rope_quant_block32,
    wo_a_rope_quant_block32_rows,
)
from sglang.kernels.ops.quantization.fp8_block32_qdq import (
    fp8_block32_quant_dequant,
    fp8_block32_quant_mn_major,
)
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.dsv4.wo_a_fp8_block32 import (
    NUM_SIDE_STREAMS,
    wo_a_fp8_block32_gemm,
)
from sglang.srt.layers.dsv41_sm90 import dsv41_sm90_on, dsv41_sm90_role
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_block32_sm90 import (
    Block32Routes,
    load_block32_routes,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_stream

# Rows from which the row-tiled RoPE kernels serve every role. Arbitrary; they were
# measured at 2048-4096 rows (wo_a input 226.8 -> 166.0 us, q RoPE 85.2 -> 26.9 us
# per layer, H20-3e) and win below that only on the decode role's measurement.
_ROPE_ROWS_MIN_PREFILL = 1024
# Decode/verify rows up to which the per-head wo_a RoPE + rounding runs.
_ROPE_PER_HEAD_MAX_ROWS = 384


class AttnProjPlan(msgspec.Struct, frozen=True):
    # FP8 block-[32, 32] grouped wo_a; else the bf16 one on the rounded input.
    wo_a_fp8_block32: bool
    # Decode role: the row-tiled RoPE kernels below _ROPE_ROWS_MIN_PREFILL rows too.
    decode_rope_rows: bool
    # Decode role: the q RoPE of the non-CP K store path rides in the store launch.
    kv_store_q_rope: bool
    # The device's and role's swap-AB rows, and the side streams of the per-group
    # GEMMs (leased before graph capture); None / () off the FP8 GEMM.
    routes: Optional[Block32Routes]
    streams: Tuple[torch.cuda.Stream, ...]


def wo_a_fp8_block32_supported(quant_config: Optional[QuantizationConfig]) -> bool:
    return (
        isinstance(quant_config, Fp8Config)
        and quant_config.weight_block_size == [32, 32]
        and quant_config.scale_fmt == "ue8m0"
        and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
    )


def wo_a_fp8_block32_enabled(quant_config: Optional[QuantizationConfig]) -> bool:
    """Whether the target's attention layers keep wo_a in FP8 (the weight loader's
    side of ``resolve_attn_proj``)."""
    return dsv41_sm90_on("attn_proj") and wo_a_fp8_block32_supported(quant_config)


def resolve_attn_proj(
    *, quant_config: Optional[QuantizationConfig], fp8_wo_a_allowed: bool
) -> Optional[AttnProjPlan]:
    """The layer's plan, or None (upstream path) with the subsystem off.
    ``fp8_wo_a_allowed`` is False for a layer that pins its wo_a to bf16."""
    if not dsv41_sm90_on("attn_proj"):
        return None
    decode = dsv41_sm90_role() == "decode"
    block32 = fp8_wo_a_allowed and wo_a_fp8_block32_supported(quant_config)
    return AttnProjPlan(
        wo_a_fp8_block32=block32,
        decode_rope_rows=decode,
        # The bf16-sourced K store has no query RoPE launch to join.
        kv_store_q_rope=decode and not envs.SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE.get(),
        routes=(
            load_block32_routes(
                device_name=torch.cuda.get_device_name(), role=dsv41_sm90_role()
            )
            if block32
            else None
        ),
        streams=(
            tuple(get_stream(f"dsv41_wo_a_{i}") for i in range(NUM_SIDE_STREAMS))
            if block32
            else ()
        ),
    )


def _rope_rows(plan: AttnProjPlan, rows: int) -> bool:
    return rows >= _ROPE_ROWS_MIN_PREFILL or (plan.decode_rope_rows and rows > 0)


def _draft_rope_rows(plan: AttnProjPlan, rows: int) -> bool:
    # The draft's row-tiled kernels serve the decode role's rows only.
    return plan.decode_rope_rows and 0 < rows < _ROPE_ROWS_MIN_PREFILL


def _rope_per_head(rows: int, forward_mode: ForwardMode) -> bool:
    return (
        forward_mode.is_decode() or forward_mode.is_target_verify()
    ) and 0 < rows <= _ROPE_PER_HEAD_MAX_ROWS


def q_rope_rows(
    plan: AttnProjPlan,
    *,
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    draft: bool = False,
) -> bool:
    """Forward RoPE of q [T, H, 512] in place with the row-tiled kernel, bitwise
    ``fused_rope_inplace``; False leaves q for the caller's launch."""
    rows = q.shape[0]
    if not (
        (_draft_rope_rows(plan, rows) if draft else _rope_rows(plan, rows))
        and can_use_q_rope_rows(q, freqs_cis, positions)
    ):
        return False
    q_rope_rows_inplace(q, freqs_cis, positions)
    return True


def _inverse_rope(o: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor):
    # freqs_cis holds one complex value per rope column pair.
    fused_rope_inplace(
        o[..., -freqs_cis.shape[1] * 2 :],
        None,
        freqs_cis,
        positions=positions,
        inverse=True,
    )


def wo_a_fp8_block32(
    plan: AttnProjPlan,
    *,
    o: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    forward_mode: ForwardMode,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Inverse RoPE + per-32 FP8 quant of o [T, H, Dh], then the grouped FP8 wo_a
    -> [T, G, R] bf16."""
    rows = o.shape[0]
    if _rope_rows(plan, rows) and can_use_wo_a_rope_quant_rows(o, freqs_cis, positions):
        o_fp8, o_scale = wo_a_rope_quant_block32_rows(o, freqs_cis, positions)
    elif _rope_per_head(rows, forward_mode) and can_use_wo_a_rope_quant(
        o, freqs_cis, positions
    ):
        o_fp8, o_scale = wo_a_rope_quant_block32(o, freqs_cis, positions)
    else:
        _inverse_rope(o, freqs_cis, positions)
        o_fp8, o_scale = fp8_block32_quant_mn_major(o.view(rows, num_groups, -1))
    return wo_a_fp8_block32_gemm(
        o_fp8=o_fp8,
        o_scale=o_scale,
        weight=weight,
        weight_scale=weight_scale,
        num_groups=num_groups,
        routes=plan.routes,
        streams=plan.streams,
    )


def wo_a_bf16_input(
    *,
    o: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    forward_mode: ForwardMode,
    num_groups: int,
) -> torch.Tensor:
    """Inverse RoPE + per-32 FP8 rounding of o [T, H, Dh] -> the bf16 wo_a input
    [T, G, H * Dh / G]."""
    rows = o.shape[0]
    if _rope_per_head(rows, forward_mode) and can_use_wo_a_rope_quant(
        o, freqs_cis, positions
    ):
        return wo_a_rope_qdq(o, freqs_cis, positions).view(rows, num_groups, -1)
    _inverse_rope(o, freqs_cis, positions)
    return fp8_block32_quant_dequant(o.view(rows, num_groups, -1))


def draft_wo_a_rope_qdq(
    plan: AttnProjPlan,
    *,
    o: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    num_groups: int,
) -> Optional[torch.Tensor]:
    """The DSpark draft's wo_a input from the row-tiled kernel, or None to keep
    the inverse RoPE + ``fp8_block32_quant_dequant``."""
    rows = o.shape[0]
    if not (
        _draft_rope_rows(plan, rows)
        and can_use_wo_a_rope_quant_rows(o, freqs_cis, positions)
    ):
        return None
    return wo_a_rope_qdq_rows(o, freqs_cis, positions).view(rows, num_groups, -1)
