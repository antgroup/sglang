"""HC=4 post-mix fused with the next sublayer's pre-collapse (DeepSeek-V4.1).

One launch computes ``R' = mhc_post(x, residual, post, comb)`` and
``y = mhc_combine(R', pre)``. Each output element follows the exact operation
sequence of the two kernels it replaces, so both outputs are bitwise identical:

* ``R'`` repeats ``mhc_post_split_h`` (itself matched to the TileLang post kernel):
  round ``comb[0] * residual[0]`` first, FMA ``post * x`` into it, then FMA the
  remaining ``comb[i] * residual[i]`` terms in order, all in fp32; store BF16.
* ``y`` repeats ``_mhc_combine_kernel`` on the BF16-rounded ``R'``: an fp32
  accumulator starting at zero, ``acc += pre[k] * R'[k]`` for k = 0..3 in order,
  compiled with the same (default) fp-fusion setting; store BF16.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mhc_post_combine_kernel(
    X,
    R,
    P,
    C,
    PRE,
    Y,
    OUT,
    pre_stride_m,
    pre_stride_k,
    H: tl.constexpr,
    B: tl.constexpr,
):
    token = tl.program_id(0)
    h = tl.program_id(1) * B + tl.arange(0, B)
    mask = h < H
    x = tl.load(X + token * H + h, mask, 0).to(tl.float32)
    r0 = tl.load(R + (token * 4 + 0) * H + h, mask, 0).to(tl.float32)
    r1 = tl.load(R + (token * 4 + 1) * H + h, mask, 0).to(tl.float32)
    r2 = tl.load(R + (token * 4 + 2) * H + h, mask, 0).to(tl.float32)
    r3 = tl.load(R + (token * 4 + 3) * H + h, mask, 0).to(tl.float32)
    acc_y = tl.zeros([B], dtype=tl.float32)
    for o in tl.static_range(4):
        # comb is [token, i, o]: residual stream i feeds output stream o.
        acc = tl.fma(
            tl.load(P + token * 4 + o),
            x,
            tl.load(C + token * 16 + 0 * 4 + o) * r0,
        )
        acc = tl.fma(tl.load(C + token * 16 + 1 * 4 + o), r1, acc)
        acc = tl.fma(tl.load(C + token * 16 + 2 * 4 + o), r2, acc)
        acc = tl.fma(tl.load(C + token * 16 + 3 * 4 + o), r3, acc)
        out = acc.to(tl.bfloat16)
        tl.store(OUT + (token * 4 + o) * H + h, out, mask)
        pk = tl.load(PRE + token * pre_stride_m + o * pre_stride_k).to(tl.float32)
        acc_y += pk * out.to(tl.float32)
    tl.store(Y + token * H + h, acc_y, mask)


def can_use_mhc_post_combine(x, residual, post, comb, pre) -> bool:
    return (
        x.is_cuda
        and x.dim() == 2
        and x.shape[1] == 5120
        and residual.shape == (x.shape[0], 4, x.shape[1])
        and x.dtype == residual.dtype == torch.bfloat16
        and post.dtype == comb.dtype == pre.dtype == torch.float32
        and post.numel() == x.shape[0] * 4
        and comb.shape == (x.shape[0], 4, 4)
        and pre.shape == (x.shape[0], 4)
        and all(t.is_contiguous() for t in (x, residual, post, comb))
    )


def mhc_post_combine(x, residual, post, comb, pre):
    """Returns (mhc_post(x, residual, post, comb), mhc_combine(that, pre)), bitwise."""
    assert can_use_mhc_post_combine(x, residual, post, comb, pre)
    m, hidden = x.shape
    output = torch.empty_like(residual)
    y = torch.empty((m, hidden), dtype=x.dtype, device=x.device)
    if m == 0:
        return output, y
    block = 128 if m <= 8 else 1024
    _mhc_post_combine_kernel[(m, triton.cdiv(hidden, block))](
        x,
        residual,
        post,
        comb,
        pre,
        y,
        output,
        pre.stride(0),
        pre.stride(1),
        H=hidden,
        B=block,
        num_warps=4,
    )
    return output, y
