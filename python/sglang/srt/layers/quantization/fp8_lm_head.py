"""FP8 copy of a bf16 LM head for decode-sized logits GEMMs (DeepSeek-V4.1, SM90).

The head is re-quantized at load time to e4m3 with fp32 scales per 128x128
weight block; activations get dynamic per-row 1x128-group scales. DeepGEMM's
sm90 1d2d kernel promotes the FP8 partial sums to fp32 every 128 K, so the
logits error is the e4m3 rounding of the two operands rather than the reduced
precision tensor-core accumulator of the fast-accum CUTLASS FP8 GEMMs. The
swap-AB Gluon kernel takes the batch sizes DeepGEMM pads to a much larger tile,
with DeepGEMM's per-element arithmetic (bitwise the same logits).

The FP8 logits are not bitwise the bf16 head's. ``dsv41_sm90_on("lm_head")`` puts
the DSpark draft logits on the copy: they only choose the proposed tokens, which
the target verifies (greedy outputs stay the target's argmax, rejection sampling
keeps the target's distribution). ``dsv41_sm90_lossy("fp8_lm_head_target")`` also
puts the target logits on it, which changes outputs and logprobs.
"""

import logging
from typing import List, Optional, Tuple

import msgspec
import torch
from torch import nn

from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dsv41_sm90 import (
    dsv41_sm90_lossy,
    dsv41_sm90_on,
    dsv41_sm90_role,
)
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.runtime_context import get_lora, get_spec
from sglang.srt.utils.common import log_info_on_rank0

if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
    import deep_gemm

logger = logging.getLogger(__name__)

_BLOCK = 128
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
# The row-split and swap-AB thresholds and launch configs below were measured on
# H20-3e and H20 at K=5120 (N=129280 and the TP8 shard N=16256); another device
# keeps the bf16 head.
_MEASURED_DEVICES = frozenset({"NVIDIA H20", "NVIDIA H20-3e"})
# Roles whose decode / verify logits were measured on the FP8 head (mix, PD decode).
_ROLES = frozenset({"mix", "decode"})
# Measured on H20, K=5120, N=129280: DeepGEMM pads a 129..191-row batch to two
# 128-row tiles (192 rows: 1296 us), while a separate <=64-row tail launch gets
# its 64-row tile (128 + 64 rows: 985 us).
_MAX_SPLIT_TAIL_ROWS = 64
# Measured on H20: on a TP8 vocab shard (N=16256) the single launch stayed faster.
_MIN_SPLIT_WEIGHT_ROWS = 65536
# Rows quantized per step at load time; bounds the fp32 staging copy.
_QUANT_ROWS_PER_STEP = 8192
# (stages, maxnreg) per swap-AB token tile (bt1, bt2); measured on H20, K=5120,
# N=129280 (bitwise the DeepGEMM path at every listed shape).
_SWAPAB_SMALL_LAUNCH = (4, 128)
_SWAPAB_LAUNCH = {
    (64, 8): (4, 128),
    (64, 16): (3, 112),
    (64, 32): (3, 112),
    (128, 8): (3, 128),
    (128, 16): (3, 128),
    # Re-measured on H20 under Triton 3.8, 146-160 rows: maxnreg 112 vs 128 takes
    # 913 vs 941 us at N=129280 and 140 vs 148 us at N=16160.
    (128, 32): (3, 112),
}


class Fp8LmHeadUse(msgspec.Struct, frozen=True):
    """Which logits of this process come from ``lm_head.fp8_head``."""

    target: bool = False
    # The DSpark draft's base logits over the target's shared head.
    draft: bool = False
    swapab: bool = False


def fp8_lm_head_supported() -> bool:
    return deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and not (
        deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
    )


def resolve_fp8_lm_head_use() -> Fp8LmHeadUse:
    """Read once at model construction; the target and its DSpark draft must agree,
    since the target builds the shared copy before the KV pool is sized."""
    if dsv41_sm90_role() not in _ROLES:
        return Fp8LmHeadUse()
    if get_lora().enable_lora:
        logger.info("FP8 LM head disabled for target and draft because LoRA is enabled")
        return Fp8LmHeadUse()
    if (
        torch.cuda.get_device_name() not in _MEASURED_DEVICES
        or not fp8_lm_head_supported()
    ):
        return Fp8LmHeadUse()
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    lm_head_on = dsv41_sm90_on("lm_head")
    dspark = SpeculativeAlgorithm.from_string(
        get_spec().speculative_algorithm
    ).is_dspark()
    return Fp8LmHeadUse(
        target=dsv41_sm90_lossy("fp8_lm_head_target"),
        # An FP32 draft head that SGLANG_DSPARK_FP32_LM_HEAD asks for stays FP32.
        draft=lm_head_on and dspark and not envs.SGLANG_DSPARK_FP32_LM_HEAD.get(),
        swapab=lm_head_on,
    )


def quantize_lm_head_weight(
    weight: torch.Tensor, *, out_q: torch.Tensor, out_scale: torch.Tensor
) -> None:
    """bf16 [N, K] -> e4m3 ``out_q`` [align(N, 128), K] and fp32 ``out_scale``
    [align(N, 128) / 128, K / 128]; rows past N are zero."""
    n, k = weight.shape
    assert k % _BLOCK == 0, f"K={k} must be a multiple of {_BLOCK}"
    out_q.zero_()
    out_scale.fill_(1.0)
    for r0 in range(0, n, _QUANT_ROWS_PER_STEP):
        r1 = min(r0 + _QUANT_ROWS_PER_STEP, n)
        rows = r1 - r0
        padded = -(-rows // _BLOCK) * _BLOCK
        block = torch.zeros(padded, k, dtype=torch.float32, device=weight.device)
        block[:rows] = weight[r0:r1].float()
        block = block.view(padded // _BLOCK, _BLOCK, k // _BLOCK, _BLOCK)
        scale = block.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / _FP8_MAX
        q = (block / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        out_q[r0:r1] = q.view(padded, k)[:rows]
        out_scale[r0 // _BLOCK : r0 // _BLOCK + padded // _BLOCK] = scale.view(
            padded // _BLOCK, k // _BLOCK
        )


def split_rows(num_rows: int, num_weight_rows: int) -> List[Tuple[int, int]]:
    """Row ranges, one DeepGEMM launch each."""
    head = num_rows - num_rows % _BLOCK
    tail = num_rows - head
    if (
        head == 0
        or tail == 0
        or tail > _MAX_SPLIT_TAIL_ROWS
        or num_weight_rows < _MIN_SPLIT_WEIGHT_ROWS
    ):
        return [(0, num_rows)]
    return [(0, head), (head, num_rows)]


def _pow2_at_least(rows: int) -> int:
    return max(8, 1 << (rows - 1).bit_length())


def swapab_tile(num_rows: int) -> Optional[Tuple[int, int]]:
    """Token pieces (bt1, bt2) of the one swap-AB tile that covers ``num_rows``, or
    None where DeepGEMM's own tile is as fast (measured on H20: 49-64, 97-128 and
    over 160 rows)."""
    if 0 < num_rows <= 48:
        rows = -(-num_rows // 8) * 8
        bt1 = 1 << (rows.bit_length() - 1)
        return bt1, 0 if rows == bt1 else _pow2_at_least(rows - bt1)
    if 64 < num_rows <= 96:
        return 64, _pow2_at_least(num_rows - 64)
    if 128 < num_rows <= 160:
        return 128, _pow2_at_least(num_rows - 128)
    return None


def swapab_launch(tile: Tuple[int, int]) -> Tuple[int, int]:
    """(stages, maxnreg) of a tile ``swapab_tile`` returns."""
    if tile[0] <= 32:
        return _SWAPAB_SMALL_LAUNCH
    return _SWAPAB_LAUNCH[tile]


class Fp8LmHead(nn.Module):
    """``forward(hidden [..., K] bf16) -> logits [..., N] bf16``."""

    def __init__(self, weight: torch.Tensor, *, use_swapab: bool):
        super().__init__()
        assert weight.dim() == 2 and weight.dtype == torch.bfloat16, (
            f"FP8 LM head expects a bf16 [N, K] weight, got {weight.dtype} "
            f"{tuple(weight.shape)}"
        )
        if use_swapab:
            from sglang.kernels.ops.gemm.fp8_lm_head_swapab import (
                lm_head_swapab_gemm,
            )

            self._swapab_gemm = lm_head_swapab_gemm
        else:
            self._swapab_gemm = None
        n, k = weight.shape
        padded_n = -(-n // _BLOCK) * _BLOCK
        self.out_features = n
        self.register_buffer(
            "weight",
            torch.empty(padded_n, k, dtype=torch.float8_e4m3fn, device=weight.device),
            persistent=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(
                padded_n // _BLOCK,
                k // _BLOCK,
                dtype=torch.float32,
                device=weight.device,
            ),
            persistent=False,
        )
        self.requantize_(weight)

    @torch.no_grad()
    def requantize_(self, weight: torch.Tensor) -> None:
        """In place, so captured CUDA graphs keep reading the same buffers."""
        assert tuple(weight.shape) == (self.out_features, self.weight.shape[1])
        quantize_lm_head_weight(weight, out_q=self.weight, out_scale=self.weight_scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        lead_shape = hidden_states.shape[:-1]
        x = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
        padded_n = self.weight.shape[0]
        out = torch.empty((x.shape[0], padded_n), dtype=torch.bfloat16, device=x.device)
        tile = swapab_tile(x.shape[0]) if self._swapab_gemm is not None else None
        if tile is not None:
            self._swapab_forward(x=x, tile=tile, out=out)
            row_ranges = []
        else:
            # DP-attention idle ranks run a zero-row forward; DeepGEMM rejects M == 0.
            row_ranges = split_rows(x.shape[0], padded_n) if x.shape[0] else []
        for r0, r1 in row_ranges:
            q, scale = sglang_per_token_group_quant_fp8(
                x[r0:r1], _BLOCK, column_major_scales=True, scale_tma_aligned=True
            )
            # Not deep_gemm_wrapper.gemm_nt_f8f8bf16: its precompile hook would run
            # this vocab-wide GEMM at every M up to 2x the chunked prefill size.
            deep_gemm.fp8_gemm_nt(
                (q, scale), (self.weight, self.weight_scale), out[r0:r1]
            )
        if padded_n != self.out_features:
            out = out[:, : self.out_features].contiguous()
        return out.view(*lead_shape, self.out_features)

    def _swapab_forward(
        self, *, x: torch.Tensor, tile: Tuple[int, int], out: torch.Tensor
    ) -> None:
        q, scale = sglang_per_token_group_quant_fp8(
            x, _BLOCK, column_major_scales=True, scale_tma_aligned=True
        )
        stages, maxnreg = swapab_launch(tile)
        self._swapab_gemm(
            q,
            scale,
            self.weight,
            self.weight_scale,
            out,
            block_w=64,
            bt1=tile[0],
            bt2=tile[1],
            stages=stages,
            maxnreg=maxnreg,
        )


def attach_fp8_lm_head(lm_head: ParallelLMHead, *, use: Fp8LmHeadUse) -> None:
    """Build ``lm_head.fp8_head`` from its bf16 weight, or refresh it in place;
    the target and the DSpark draft share the one copy through the shared head."""
    assert isinstance(lm_head, ParallelLMHead), type(lm_head)
    if lm_head.fp8_head is not None:
        lm_head.fp8_head.requantize_(lm_head.weight)
        return
    lm_head.fp8_head = Fp8LmHead(lm_head.weight, use_swapab=use.swapab)
    log_info_on_rank0(
        logger,
        f"FP8 LM head copy of {tuple(lm_head.weight.shape)} built: target logits "
        f"{use.target}, DSpark draft logits {use.draft}, swap-AB {use.swapab}.",
    )
