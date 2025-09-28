from typing import Any, Dict, List, Optional, Tuple, Union

import pytest
import torch
from sgl_kernel import FusedSetKVBufferArg, apply_rope_with_cos_sin_cache_inplace

from sglang.srt.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
    RotaryEmbedding,
)

@pytest.mark.parametrize(
    "head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, device, batch_size, seq_len, num_q_heads, num_kv_heads, save_kv_cache",
    [
        # GPT-OSS cases
        *[
            (
                64,
                64,
                4096,
                8000,
                True,
                torch.bfloat16,
                "cuda",
                batch_size,
                seq_len,
                64,
                8,
                save_kv_cache,
            )
            for batch_size, seq_len in (
                (1, 1),
                (32, 1),
                (128, 1),
                (512, 1),
                (2, 512),
                (4, 4096),
            )
            for save_kv_cache in (False, True)
        ],
        # Other cases
        (64, 64, 32, 8000, True, torch.bfloat16, "cuda", 32, 32, 1, 1, False),
        (256, 128, 4096, 10000, True, torch.bfloat16, "cuda", 2, 512, 4, 2, False),
        (512, 128, 311, 10000, True, torch.bfloat16, "cuda", 3, 39, 4, 2, False),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cuda", 2, 512, 32, 8, False),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cuda", 2, 512, 16, 4, False),
        (512, 128, 311, 10000, False, torch.bfloat16, "cuda", 3, 39, 4, 2, False),
    ],
)
def test_correctness(
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: int,
    is_neox_style: bool,
    dtype: torch.dtype,
    device: str,
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    save_kv_cache: bool,
):
    config1 = dict(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
        scaling_factor=1.0,
        reference=False,
    )

    config2 = dict(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
        scaling_factor=1.0,
        reference=True,
    )

    rot = DeepseekScalingRotaryEmbedding(**config1).to(device)
    rot_ref = DeepseekScalingRotaryEmbedding(**config2).to(device)

    positions = torch.randint(0, max_position_embeddings, (batch_size,), device=device)
    # query is [batch, num_heads, head_size]
    # key is [batch, 1, head_size]
    # cos_sin is [batch, head_size]
    query = torch.randn(
        batch_size, num_q_heads, head_size, dtype=torch.float32, device=device
    )
    key = torch.randn(batch_size, 1, head_size, dtype=torch.float32, device=device)
    ref_query, ref_key = rot_ref.forward(positions, query, key)
    out_query, out_key = rot.forward(positions, query, key)
    torch.testing.assert_close(out_key.cpu(), ref_key.cpu(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_query.cpu(), ref_query.cpu(), atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
