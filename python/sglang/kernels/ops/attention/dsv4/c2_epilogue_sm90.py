"""Hopper C2 post-pooling RMSNorm, RoPE, fake FP4 and FlashMLA cache store.

Keep the existing pair-pooling kernel and its request-ring ownership rules.
This epilogue consumes its FP32 pooled rows and returns the BF16 pre-RoPE
latent plus gathered frequencies for the index-K branch. No SM100 FP4
instructions or changes to compressor projection weights are required.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


def can_use_c2_epilogue(pooled, weight, freqs, slots, group_pos, cache, page_size):
    """Restrict the direct store to the standard 584-byte FlashMLA layout."""
    return (
        pooled.is_cuda
        and torch.version.cuda is not None
        and pooled.ndim == 2
        and pooled.shape[1] == 512
        and pooled.dtype == torch.float32
        and pooled.is_contiguous()
        and weight.shape == (512,)
        and weight.dtype in (torch.bfloat16, torch.float32)
        and weight.is_contiguous()
        and freqs.ndim == 2
        and freqs.shape[1] == 32
        and freqs.dtype == torch.complex64
        and freqs.is_contiguous()
        and all(
            t.shape == (pooled.shape[0],)
            and t.dtype in (torch.int32, torch.int64)
            and t.is_contiguous()
            for t in (slots, group_pos)
        )
        and page_size > 0
        and page_size & (page_size - 1) == 0
        and cache is not None
        and cache.ndim == 2
        and cache.shape[1] == triton.cdiv(584 * page_size, 576) * 576
        and cache.dtype in (torch.uint8, torch.float8_e4m3fn)
        and cache.is_contiguous()
        and all(
            t.device == pooled.device for t in (weight, freqs, slots, group_pos, cache)
        )
    )


@triton.jit
def _c2_epilogue_kernel(
    Pooled,
    Weight,
    Freqs,
    GroupPos,
    Slots,
    Latent,
    GatheredFreqs,
    Cache,
    EPS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, 512)

    # finish() rounds pooled FP32 to BF16 *before* computing the statistic.
    x = tl.load(Pooled + row * 512 + d).to(tl.bfloat16).to(tl.float32)
    w = tl.load(Weight + d).to(tl.float32)
    # RMSNorm uses enable_fp_fusion=False. Pin these rounding boundaries while
    # allowing the RoPE expression below the same FMA contraction as _rope_fq4.
    sq = libdevice.mul_rn(x, x)
    mean = tl.sum(sq, 0) * (1.0 / 512)
    inv_rms = tl.rsqrt(libdevice.add_rn(mean, EPS))
    normalized = libdevice.mul_rn(libdevice.mul_rn(x, inv_rms), w)
    latent = normalized.to(tl.bfloat16)
    tl.store(Latent + row * 512 + d, latent)

    # Gather once for both the main-KV and index-K branches, without a separate
    # torch advanced-indexing launch. GroupPos is produced by c2_decode_pool.
    position = tl.load(GroupPos + row).to(tl.int64)
    f = tl.arange(0, 64)
    frequency = tl.load(Freqs + position * 64 + f)
    tl.store(GatheredFreqs + row * 64 + f, frequency)

    value = latent.to(tl.float32)
    tail = d >= 448
    pair = tl.maximum((d - 448) // 2, 0)
    re = tl.gather(value, 448 + 2 * pair, 0)
    im = tl.gather(value, 449 + 2 * pair, 0)
    fr = tl.load(Freqs + position * 64 + 2 * pair, tail, 1.0)
    fi = tl.load(Freqs + position * 64 + 2 * pair + 1, tail, 0.0)
    rotated = tl.where((d % 2) == 1, re * fi + im * fr, re * fr - im * fi)
    value = tl.where(tail, rotated.to(tl.bfloat16).to(tl.float32), value)

    # Match rope_tail_fake_quant_fp4(compressed_kv=True): per-16 E4M3
    # scales, RNE software FP4, followed by the BF16 materialization boundary.
    blocks = tl.reshape(value, (32, 16))
    amax = tl.max(tl.abs(blocks), 1)
    scale = tl.minimum(tl.maximum(amax * (1.0 / 6.0), 2.0**-9), 448.0)
    scale = scale.to(tl.float8e4nv).to(tl.float32)
    scaled = tl.div_rn(blocks, scale[:, None])
    scaled = tl.minimum(tl.maximum(scaled, -6.0), 6.0)
    magnitude = tl.abs(scaled)
    step = tl.where(magnitude < 2.0, 0.5, tl.where(magnitude < 4.0, 1.0, 2.0))
    sign = tl.where(scaled > 0, 1.0, tl.where(scaled < 0, -1.0, 0.0))
    rounded = libdevice.rint(magnitude / step) * step * sign
    quantized = tl.reshape(rounded * scale[:, None], (512,))
    quantized = quantized.to(tl.bfloat16).to(tl.float32)

    # FlashMLA: 448 E4M3 bytes + 64 BF16 values per token; seven UE8M0
    # scales live after the page's value array. Leave the eighth scale byte and
    # page padding untouched, exactly like fused_store_flashmla_cache.
    fp8_blocks = tl.reshape(quantized, (8, 64))
    abs_max = tl.max(tl.abs(fp8_blocks), 1)
    raw_scale = tl.maximum(abs_max, 1.0e-4) / 448.0
    bits = raw_scale.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    inv_scale = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    fp8 = tl.minimum(tl.maximum(fp8_blocks * inv_scale[:, None], -448.0), 448.0)
    payload = tl.reshape(fp8.to(tl.float8e4nv).to(tl.uint8, bitcast=True), (512,))

    slot = tl.load(Slots + row).to(tl.int64)
    live = slot > 0
    page = slot // PAGE_SIZE
    offset = slot % PAGE_SIZE
    base = page * PAGE_BYTES + offset * 576
    tl.store(Cache + base + d, payload, (d < 448) & live)
    rope_ptr = (Cache + base + 448).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_ptr + d - 448, quantized.to(tl.bfloat16), tail & live)
    scale_id = tl.arange(0, 8)
    tl.store(
        Cache + page * PAGE_BYTES + PAGE_SIZE * 576 + offset * 8 + scale_id,
        exponent.to(tl.uint8),
        (scale_id < 7) & live,
    )


def c2_epilogue_sm90(pooled, weight, eps, freqs, slots, group_pos, cache, page_size):
    """Return (pre-RoPE latent, gathered complex frequencies), and write KV.

    Live output slots must be unique. Slots <= 0 are padding/incomplete pairs
    and never write the cache, including its sentinel slot 0. Latent/frequencies
    are initialized for *all* rows because index-K consumes the full row batch.
    Does not touch pair state or mutate pooled rows, positions or locations.
    """
    assert can_use_c2_epilogue(
        pooled, weight, freqs, slots, group_pos, cache, page_size
    )
    latent = torch.empty_like(pooled, dtype=torch.bfloat16)
    gathered = torch.empty(
        (pooled.shape[0], 32), dtype=torch.complex64, device=pooled.device
    )
    if pooled.shape[0]:
        _c2_epilogue_kernel[(pooled.shape[0],)](
            pooled,
            weight,
            torch.view_as_real(freqs),
            group_pos,
            slots,
            latent,
            torch.view_as_real(gathered),
            cache.view(torch.uint8),
            EPS=eps,
            PAGE_SIZE=page_size,
            PAGE_BYTES=cache.stride(0),
            num_warps=4,
            enable_fp_fusion=True,
        )
    return latent, gathered
