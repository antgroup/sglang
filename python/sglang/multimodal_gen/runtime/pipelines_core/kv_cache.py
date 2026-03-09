"""
General-purpose KV cache abstractions for causal video diffusion inference.

Provides typed cache classes that replace raw List[Dict] patterns:
- SelfAttentionKVCache: per-layer self-attention buffer with position tracking
  and sliding window eviction (streaming-specific)
- CrossAttentionKVCache: per-layer cross-attention buffer with compute-once
  semantics (general optimization, also managed here for convenience)
- KVCacheManager: container managing per-layer caches for all transformer blocks
"""

from __future__ import annotations

import torch
from torch import Tensor


class SelfAttentionKVCache:
    """Per-layer self-attention KV buffer with position tracking and sliding window eviction.

    The buffer is pre-allocated to a fixed max_size. Two write modes are supported:
    - bulk_write: overwrite from position 0 (used during KV recompute with block_mask)
    - append: incremental write with automatic sliding window eviction when full
    """

    __slots__ = ("k", "v", "_global_end", "_local_end")

    def __init__(
        self,
        batch_size: int,
        max_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.k = torch.zeros(
            batch_size, max_size, num_heads, head_dim,
            dtype=dtype, device=device,
        ).contiguous()
        self.v = torch.zeros(
            batch_size, max_size, num_heads, head_dim,
            dtype=dtype, device=device,
        ).contiguous()
        self._global_end: int = 0
        self._local_end: int = 0

    @property
    def max_size(self) -> int:
        return self.k.shape[1]

    @property
    def global_end_index(self) -> int:
        return self._global_end

    @property
    def local_end_index(self) -> int:
        return self._local_end

    def reset(self) -> None:
        """Zero buffers and reset indices. Reuses allocated memory."""
        self.k.zero_()
        self.v.zero_()
        self._global_end = 0
        self._local_end = 0

    def bulk_write(self, key: Tensor, value: Tensor) -> None:
        """Write KV from position 0 and update indices.

        Used during the recompute phase (with block_mask / flex_attention).
        """
        seq_len = key.shape[1]
        self.k[:, :seq_len] = key
        self.v[:, :seq_len] = value
        self._global_end = seq_len
        self._local_end = seq_len

    def append(
        self,
        key: Tensor,
        value: Tensor,
        current_start: int,
        sink_tokens: int = 0,
    ) -> None:
        """Append new KV entries, with sliding window eviction if the buffer is full.

        When the buffer cannot fit the new tokens:
          1. Compute how many old tokens to evict
          2. Preserve the first ``sink_tokens`` positions (attention sink)
          3. Shift remaining entries left to make room
          4. Write new tokens at the end

        When there is room, detach the buffer from the computation graph
        (to avoid gradient accumulation across denoising steps) and write directly.
        """
        current_end = current_start + key.shape[1]
        num_new = key.shape[1]
        buf_size = self.max_size

        needs_eviction = (
            current_end > self._global_end
            and num_new + self._local_end > buf_size
        )

        if needs_eviction:
            num_evicted = num_new + self._local_end - buf_size
            num_rolled = self._local_end - num_evicted - sink_tokens
            self.k[:, sink_tokens : sink_tokens + num_rolled] = self.k[
                :,
                sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled,
            ].clone()
            self.v[:, sink_tokens : sink_tokens + num_rolled] = self.v[
                :,
                sink_tokens + num_evicted : sink_tokens + num_evicted + num_rolled,
            ].clone()
            local_end = (
                self._local_end + current_end - self._global_end - num_evicted
            )
        else:
            self.k = self.k.detach()
            self.v = self.v.detach()
            local_end = self._local_end + current_end - self._global_end

        local_start = local_end - num_new
        self.k[:, local_start:local_end] = key
        self.v[:, local_start:local_end] = value
        self._global_end = current_end
        self._local_end = local_end

    def get_active_kv(self, max_attention_size: int) -> tuple[Tensor, Tensor]:
        """Return the (k, v) window the attention layer should attend to.

        Sliced to at most ``max_attention_size`` tokens from the end of
        the valid range.
        """
        start = max(0, self._local_end - max_attention_size)
        return (
            self.k[:, start : self._local_end],
            self.v[:, start : self._local_end],
        )


class CrossAttentionKVCache:
    """Per-layer cross-attention KV cache with compute-once semantics.

    On the first call the caller computes K,V from encoder hidden states
    and stores them via ``update()``. All subsequent calls retrieve the
    cached tensors via ``get()``.
    """

    __slots__ = ("k", "v", "_is_initialized")

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.k = torch.zeros(
            batch_size, seq_len, num_heads, head_dim,
            dtype=dtype, device=device,
        ).contiguous()
        self.v = torch.zeros(
            batch_size, seq_len, num_heads, head_dim,
            dtype=dtype, device=device,
        ).contiguous()
        self._is_initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def reset(self) -> None:
        """Zero buffers and mark uninitialized. Reuses allocated memory."""
        self.k.zero_()
        self.v.zero_()
        self._is_initialized = False

    def update(self, k: Tensor, v: Tensor) -> None:
        """Store computed KV tensors and mark as initialized.

        The stored references replace the pre-allocated buffers
        (projections may produce different shapes).
        """
        self.k = k
        self.v = v
        self._is_initialized = True

    def get(self) -> tuple[Tensor, Tensor]:
        """Return cached (k, v)."""
        return self.k, self.v


class KVCacheManager:
    """Container managing per-layer KV caches for all transformer blocks.

    Holds separate lists of ``SelfAttentionKVCache`` and ``CrossAttentionKVCache``,
    one per transformer block. Provides convenience methods to reset all caches.

    Typical lifecycle (per streaming block):
      1. First block: ``session.kv_cache_manager = KVCacheManager(...)``
      2. Subsequent blocks: ``manager.reset_self_attn()``
         (and ``manager.reset_cross_attn()`` if the prompt changed)
    """

    def __init__(
        self,
        num_blocks: int,
        sa_batch_size: int,
        sa_max_size: int,
        sa_num_heads: int,
        sa_head_dim: int,
        ca_batch_size: int,
        ca_seq_len: int,
        ca_num_heads: int,
        ca_head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.self_attn_caches: list[SelfAttentionKVCache] = [
            SelfAttentionKVCache(
                sa_batch_size, sa_max_size, sa_num_heads, sa_head_dim, dtype, device,
            )
            for _ in range(num_blocks)
        ]
        self.cross_attn_caches: list[CrossAttentionKVCache] = [
            CrossAttentionKVCache(
                ca_batch_size, ca_seq_len, ca_num_heads, ca_head_dim, dtype, device,
            )
            for _ in range(num_blocks)
        ]

    def reset_self_attn(self) -> None:
        """Reset all self-attention caches (zero buffers, reset indices)."""
        for cache in self.self_attn_caches:
            cache.reset()

    def reset_cross_attn(self) -> None:
        """Reset all cross-attention caches (zero buffers, mark uninitialized)."""
        for cache in self.cross_attn_caches:
            cache.reset()
