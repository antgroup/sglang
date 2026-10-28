"""Fused FP32 Engram gate with a single final cast to the activation dtype."""

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# _engram_gate_pack_kernel + _engram_gate_apply_kernel repeat this arithmetic and
# must stay bitwise equal to it (the Engram TP split).
@triton.jit
def _engram_gate_kernel(
    X,
    KV,
    QW,
    KW,
    IDS,
    O,
    IMAGE_ID,
    D: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
    HAS_IDS: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HC
    hc = row % HC
    col = tl.arange(0, B)
    mask = col < D
    x = tl.load(X + row * D + col, mask, 0).to(tl.float32)
    key = tl.load(KV + token * (HC + 1) * D + hc * D + col, mask, 0).to(tl.float32)
    q_weight = tl.load(QW + hc * D + col, mask, 0).to(tl.float32)
    k_weight = tl.load(KW + hc * D + col, mask, 0).to(tl.float32)
    weight = q_weight * k_weight
    rstd = tl.rsqrt(tl.sum(x * x, 0) / D + EPS) * tl.rsqrt(
        tl.sum(key * key, 0) / D + EPS
    )
    dot = tl.sum((x * weight) * key, 0) * rstd * (D**-0.5)
    gate = tl.sigmoid(libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), CLAMP)), dot))
    value = tl.load(KV + token * (HC + 1) * D + HC * D + col, mask, 0).to(tl.float32)
    out = x + gate * value
    if HAS_IDS:
        out = tl.where(tl.load(IDS + token) == IMAGE_ID, x, out)
    tl.store(O + row * D + col, out, mask)


def fused_engram_gate(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
    input_ids: Optional[torch.Tensor] = None,
    image_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Rows whose ``input_ids`` entry is ``image_token_id`` keep ``x`` unchanged."""
    assert x.ndim == 3 and kv.ndim == 2
    t, hc, d = x.shape
    assert kv.shape == (t, (hc + 1) * d)
    assert q_weight.shape == k_weight.shape == (hc, d)
    assert input_ids is None or (
        input_ids.shape == (t,) and input_ids.is_cuda and image_token_id is not None
    )
    assert all(
        a.is_cuda and a.is_contiguous() and a.device == x.device
        for a in (x, kv, q_weight, k_weight)
    )
    assert all(
        a.dtype in (torch.bfloat16, torch.float32) for a in (x, kv, q_weight, k_weight)
    )
    out = torch.empty_like(x)
    if t:
        _engram_gate_kernel[(t * hc,)](
            x,
            kv,
            q_weight,
            k_weight,
            input_ids,
            out,
            image_token_id,
            d,
            hc,
            eps,
            clamp_value,
            triton.next_power_of_2(d),
            input_ids is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _engram_gate_pack_kernel(
    X,
    KV,
    QW,
    KW,
    P,
    D: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
    PW: tl.constexpr,
    KV_CW: tl.constexpr,
    KV_CS,
    KV_RS,
):
    # kv column n of token t sits at (n // KV_CW) * KV_CS + t * KV_RS + n % KV_CW.
    row = tl.program_id(0)
    token = row // HC
    hc = row % HC
    col = tl.arange(0, B)
    mask = col < D
    x = tl.load(X + row * D + col, mask, 0).to(tl.float32)
    n = hc * D + col
    kv_row = KV + token * KV_RS
    key = tl.load(kv_row + (n // KV_CW) * KV_CS + n % KV_CW, mask, 0).to(tl.float32)
    q_weight = tl.load(QW + hc * D + col, mask, 0).to(tl.float32)
    k_weight = tl.load(KW + hc * D + col, mask, 0).to(tl.float32)
    weight = q_weight * k_weight
    rstd = tl.rsqrt(tl.sum(x * x, 0) / D + EPS) * tl.rsqrt(
        tl.sum(key * key, 0) / D + EPS
    )
    dot = tl.sum((x * weight) * key, 0) * rstd * (D**-0.5)
    gate = tl.sigmoid(libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), CLAMP)), dot))
    gates = (P + token * PW + D).to(tl.pointer_type(tl.float32), bitcast=True)
    tl.store(gates + hc, gate)
    if hc == 0:
        n = HC * D + col
        value = tl.load(kv_row + (n // KV_CW) * KV_CS + n % KV_CW, mask, 0)
        tl.store(P + token * PW + col, value, mask)


@triton.jit
def _engram_gate_apply_kernel(
    X,
    P,
    IDS,
    O,
    IMAGE_ID,
    D: tl.constexpr,
    HC: tl.constexpr,
    B: tl.constexpr,
    PW: tl.constexpr,
    HAS_IDS: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HC
    hc = row % HC
    col = tl.arange(0, B)
    mask = col < D
    x = tl.load(X + row * D + col, mask, 0).to(tl.float32)
    gates = (P + token * PW + D).to(tl.pointer_type(tl.float32), bitcast=True)
    gate = tl.load(gates + hc)
    value = tl.load(P + token * PW + col, mask, 0).to(tl.float32)
    out = x + gate * value
    if HAS_IDS:
        out = tl.where(tl.load(IDS + token) == IMAGE_ID, x, out)
    tl.store(O + row * D + col, out, mask)


def engram_gate_pack_width(hc: int, dim: int, dtype: torch.dtype) -> int:
    """Elements of ``dtype`` per packed row: the value row, then hc fp32 gates."""
    gate_elems, rem = divmod(hc * 4, dtype.itemsize)
    assert rem == 0 and dim * dtype.itemsize % 4 == 0, (hc, dim, dtype)
    return dim + gate_elems


def fused_engram_gate_pack(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
    out: torch.Tensor,
) -> torch.Tensor:
    """x [t, hc, d] -> out [t, pack width] in kv's dtype: the value row followed by
    the fp32 gate fused_engram_gate gives each copy. kv holds the (hc + 1) * d
    columns of rows 0..t-1 either whole, [>= t, cols], or as equal column chunks
    of all rows, [chunks, >= t, cols / chunks] (a column-split projection)."""
    t, hc, d = x.shape
    cols = (hc + 1) * d
    if kv.ndim == 2:
        kv = kv.unsqueeze(0)
    chunks, kv_rows, chunk_cols = kv.shape
    assert chunks * chunk_cols == cols and kv_rows >= t, (kv.shape, x.shape)
    assert q_weight.shape == k_weight.shape == (hc, d)
    assert out.shape == (t, engram_gate_pack_width(hc, d, kv.dtype))
    assert out.dtype == kv.dtype
    assert all(
        a.is_cuda and a.is_contiguous() and a.device == x.device
        for a in (x, kv, q_weight, k_weight, out)
    )
    if t:
        _engram_gate_pack_kernel[(t * hc,)](
            x,
            kv,
            q_weight,
            k_weight,
            out,
            d,
            hc,
            eps,
            clamp_value,
            triton.next_power_of_2(d),
            out.shape[1],
            chunk_cols,
            kv_rows * chunk_cols,
            chunk_cols,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def fused_engram_gate_apply(
    x: torch.Tensor,
    packed: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None,
    image_token_id: Optional[int] = None,
) -> torch.Tensor:
    """x [t, hc, d] plus each copy's gate times the value, from packed rows [t, width];
    bitwise fused_engram_gate when the rows came from fused_engram_gate_pack. Rows
    whose ``input_ids`` entry is ``image_token_id`` keep ``x`` unchanged."""
    t, hc, d = x.shape
    assert packed.shape == (t, engram_gate_pack_width(hc, d, packed.dtype))
    assert input_ids is None or (
        input_ids.shape == (t,) and input_ids.is_cuda and image_token_id is not None
    )
    assert all(
        a.is_cuda and a.is_contiguous() and a.device == x.device for a in (x, packed)
    )
    out = torch.empty_like(x)
    if t:
        _engram_gate_apply_kernel[(t * hc,)](
            x,
            packed,
            input_ids,
            out,
            image_token_id,
            d,
            hc,
            triton.next_power_of_2(d),
            packed.shape[1],
            input_ids is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
