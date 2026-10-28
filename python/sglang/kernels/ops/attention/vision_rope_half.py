"""Half-split vision Q/K RoPE in FP32, rounding each op as eager torch does."""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["pairs"])
def _vision_qk_rope_half(
    Q,
    K,
    C,
    S,
    OQ,
    OK,
    pairs,
    H: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    mask = i < pairs
    row = i // (D // 2)
    col = i % (D // 2)
    token, head = row // H, row % H
    c = tl.load(C + token * (D // 2) + col, mask, 0).to(tl.float32)
    s = tl.load(S + token * (D // 2) + col, mask, 0).to(tl.float32)
    qi = token * QS + head * D + col
    ki = token * KS + head * D + col
    q1 = tl.load(Q + qi, mask, 0).to(tl.float32)
    q2 = tl.load(Q + qi + D // 2, mask, 0).to(tl.float32)
    k1 = tl.load(K + ki, mask, 0).to(tl.float32)
    k2 = tl.load(K + ki + D // 2, mask, 0).to(tl.float32)
    out = row * D + col
    tl.store(OQ + out, q1 * c - q2 * s, mask)
    tl.store(OQ + out + D // 2, q2 * c + q1 * s, mask)
    tl.store(OK + out, k1 * c - k2 * s, mask)
    tl.store(OK + out + D // 2, k2 * c + k1 * s, mask)


def vision_qk_rope_half(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate [tokens, heads, dim] Q/K, which may be token-strided QKV views.

    ``cos``/``sin`` are FP32 [tokens, 1, dim / 2]; the outputs are contiguous.
    Each multiply and add rounds in FP32 (no FMA), only the stores cast to the
    input dtype, so the result is bitwise the eager half-split rotation.
    """
    if not (
        q.is_cuda
        and q.device == k.device == cos.device == sin.device
        and q.dtype == k.dtype
        and q.dtype in (torch.bfloat16, torch.float32)
        and q.ndim == 3
        and q.shape == k.shape
        and q.shape[-1] % 2 == 0
        and q.stride(-1) == k.stride(-1) == 1
        and q.stride(-2) == k.stride(-2) == q.shape[-1]
        and cos.dtype == sin.dtype == torch.float32
        and cos.shape == sin.shape == (q.shape[0], 1, q.shape[-1] // 2)
        and cos.is_contiguous()
        and sin.is_contiguous()
    ):
        raise ValueError("Unsupported half-split vision RoPE layout/dtype")
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    pairs = q.numel() // 2
    if pairs:
        with torch.cuda.device(q.device):
            _vision_qk_rope_half[(triton.cdiv(pairs, 256),)](
                q,
                k,
                cos,
                sin,
                oq,
                ok,
                pairs,
                q.shape[1],
                q.shape[2],
                q.stride(0),
                k.stride(0),
                256,
                enable_fp_fusion=False,
            )
    return oq, ok
