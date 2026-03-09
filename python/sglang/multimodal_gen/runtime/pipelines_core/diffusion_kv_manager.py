"""
KV cache management for streaming diffusion transformers (e.g. KREA).

In streaming video generation, each block is processed by a separate pipeline
invocation.  Between invocations the KV cache is reset, so context from
previous blocks must be reconstructed.  The current KREA approach re-runs a
full context forward pass at the start of every block — even though the clean
KV it produces is identical to what was generated (or could have been
generated) during the previous block.

This module provides:

  1. **Structured KV buffers** (``LayerKVBuffer`` / ``CrossAttnBuffer``) that
     wrap the raw ``dict`` interface expected by existing transformers, adding
     lifecycle tracking without breaking compatibility.

  2. **CleanKVStore** — a persistent cache that stores the clean KV produced
     after each block, so subsequent blocks can load context directly instead
     of re-running the expensive context forward.

  3. **DiffusionKVManager** — a bookkeeper that orchestrates buffer resets,
     context loading, lifecycle state transitions, and clean KV persistence
     across streaming blocks.

Lifecycle per block::

    EMPTY  ──(load context / block_idx==0)──►  CONTEXT_READY
    CONTEXT_READY  ──(begin_denoising)──►  DENOISING
    DENOISING  ──(denoising done)──►  CLEAN_COMMITTED
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DiffusionKVConfig:
    """Parameters for sizing and managing streaming diffusion KV buffers."""

    num_layers: int
    num_heads: int
    head_dim: int
    max_context_frames: int
    num_frames_per_block: int
    frame_seq_length: int
    local_attn_size: int = -1
    max_text_length: int = 512
    enable_clean_kv_store: bool = True

    @property
    def total_kv_tokens(self) -> int:
        """Total token slots in the self-attention KV buffer."""
        if self.local_attn_size != -1:
            return self.local_attn_size * self.frame_seq_length
        return (self.max_context_frames + self.num_frames_per_block) * self.frame_seq_length


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class KVLifecycleState(enum.Enum):
    EMPTY = "empty"
    CONTEXT_READY = "context_ready"
    DENOISING = "denoising"
    CLEAN_COMMITTED = "clean_committed"


# ---------------------------------------------------------------------------
# Per-layer buffer wrappers
# ---------------------------------------------------------------------------

class LayerKVBuffer:
    """Self-attention KV buffer for one transformer layer.

    ``as_dict()`` returns a mutable dict backed by the **same underlying
    tensors** so that existing transformer code can read/write through it
    with zero-copy overhead.
    """

    __slots__ = ("k", "v", "global_end_index", "local_end_index", "_dict_view")

    def __init__(
        self,
        batch_size: int,
        kv_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        shape = [batch_size, kv_size, num_heads, head_dim]
        self.k = torch.zeros(shape, dtype=dtype, device=device).contiguous()
        self.v = torch.zeros(shape, dtype=dtype, device=device).contiguous()
        self.global_end_index: int = 0
        self.local_end_index: int = 0
        self._dict_view: dict[str, Any] | None = None

    def reset(self) -> None:
        self.k.zero_()
        self.v.zero_()
        self.global_end_index = 0
        self.local_end_index = 0
        if self._dict_view is not None:
            self._dict_view["global_end_index"] = 0
            self._dict_view["local_end_index"] = 0

    def as_dict(self) -> dict[str, Any]:
        """Return a stable mutable dict backed by the same tensors."""
        if self._dict_view is None:
            self._dict_view = {
                "k": self.k,
                "v": self.v,
                "global_end_index": self.global_end_index,
                "local_end_index": self.local_end_index,
            }
        else:
            self._dict_view["global_end_index"] = self.global_end_index
            self._dict_view["local_end_index"] = self.local_end_index
        return self._dict_view

    def sync_from_dict(self, d: dict[str, Any]) -> None:
        """Pull back index values that the transformer may have updated."""
        self.global_end_index = d["global_end_index"]
        self.local_end_index = d["local_end_index"]


class CrossAttnBuffer:
    """Cross-attention KV buffer for one transformer layer.

    Cross-attn KV depends only on the text prompt, not on timestep or block
    index, so it is computed once and reused until the prompt changes.
    """

    __slots__ = ("k", "v", "is_init", "_dict_view")

    def __init__(
        self,
        batch_size: int,
        max_text_length: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        shape = [batch_size, max_text_length, num_heads, head_dim]
        self.k = torch.zeros(shape, dtype=dtype, device=device).contiguous()
        self.v = torch.zeros(shape, dtype=dtype, device=device).contiguous()
        self.is_init: bool = False
        self._dict_view: dict[str, Any] | None = None

    def reset(self) -> None:
        self.k.zero_()
        self.v.zero_()
        self.is_init = False
        if self._dict_view is not None:
            self._dict_view["is_init"] = False

    def as_dict(self) -> dict[str, Any]:
        if self._dict_view is None:
            self._dict_view = {"k": self.k, "v": self.v, "is_init": self.is_init}
        else:
            self._dict_view["is_init"] = self.is_init
        return self._dict_view

    def sync_from_dict(self, d: dict[str, Any]) -> None:
        self.is_init = d["is_init"]


# ---------------------------------------------------------------------------
# Clean KV store — persists clean KV across streaming blocks
# ---------------------------------------------------------------------------

class CleanKVStore:
    """Persistent cache for clean KV produced by context-warmup forward passes.

    After each block's denoising completes and a context warmup generates clean
    KV, the relevant portion is saved here.  On the *next* block, the manager
    can load clean KV directly from this store into the working buffer,
    potentially skipping the expensive context forward entirely.

    Supports sliding-window eviction when stored frames exceed ``max_frames``.
    """

    def __init__(
        self,
        num_layers: int,
        max_frames: int,
        frame_seq_length: int,
        num_heads: int,
        head_dim: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_layers = num_layers
        self.frame_seq_length = frame_seq_length
        self.max_frames = max_frames
        self.num_stored_frames: int = 0

        total_tokens = max_frames * frame_seq_length
        shape = [batch_size, total_tokens, num_heads, head_dim]
        self.k_store = [
            torch.zeros(shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_store = [
            torch.zeros(shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

    @property
    def num_stored_tokens(self) -> int:
        return self.num_stored_frames * self.frame_seq_length

    def store_clean_kv(
        self,
        layer_idx: int,
        k_clean: torch.Tensor,
        v_clean: torch.Tensor,
    ) -> None:
        """Append clean KV for one layer at the current write cursor."""
        start = self.num_stored_tokens
        end = start + k_clean.shape[1]
        self.k_store[layer_idx][:, start:end] = k_clean
        self.v_store[layer_idx][:, start:end] = v_clean

    def advance_frames(self, num_frames: int) -> None:
        """Advance the write cursor. Call once after all layers are stored."""
        self.num_stored_frames += num_frames

    def load_into_buffer(
        self,
        layer_buf: LayerKVBuffer,
        layer_idx: int,
        max_context_frames: int,
    ) -> int:
        """Copy cached clean KV into *layer_buf*. Returns token count loaded."""
        n_frames = min(self.num_stored_frames, max_context_frames)
        if n_frames == 0:
            return 0
        n_tokens = n_frames * self.frame_seq_length
        layer_buf.k[:, :n_tokens] = self.k_store[layer_idx][:, :n_tokens]
        layer_buf.v[:, :n_tokens] = self.v_store[layer_idx][:, :n_tokens]
        layer_buf.global_end_index = n_tokens
        layer_buf.local_end_index = n_tokens
        return n_tokens

    def evict_oldest(self, frames_to_evict: int) -> None:
        """Slide out the oldest frames to make room for new ones."""
        if frames_to_evict <= 0 or self.num_stored_frames == 0:
            return
        frames_to_evict = min(frames_to_evict, self.num_stored_frames)
        tok_evict = frames_to_evict * self.frame_seq_length
        remaining = self.num_stored_tokens - tok_evict
        for i in range(self.num_layers):
            self.k_store[i][:, :remaining] = self.k_store[i][:, tok_evict:tok_evict + remaining].clone()
            self.v_store[i][:, :remaining] = self.v_store[i][:, tok_evict:tok_evict + remaining].clone()
        self.num_stored_frames -= frames_to_evict

    def reset(self) -> None:
        self.num_stored_frames = 0
        for i in range(self.num_layers):
            self.k_store[i].zero_()
            self.v_store[i].zero_()


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class DiffusionKVSession:
    """Per-stream KV state, stored on ``RealtimeSession.kv_session``.

    Holds all per-layer KV buffers, lifecycle state, and an optional
    ``CleanKVStore`` for context caching across blocks.
    """

    def __init__(
        self,
        config: DiffusionKVConfig,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.config = config
        self.state = KVLifecycleState.EMPTY
        self.current_block_idx: int = -1

        self.self_attn: list[LayerKVBuffer] = [
            LayerKVBuffer(
                batch_size, config.total_kv_tokens,
                config.num_heads, config.head_dim, dtype, device,
            )
            for _ in range(config.num_layers)
        ]
        self.cross_attn: list[CrossAttnBuffer] = [
            CrossAttnBuffer(
                batch_size, config.max_text_length,
                config.num_heads, config.head_dim, dtype, device,
            )
            for _ in range(config.num_layers)
        ]

        self.clean_store: CleanKVStore | None = None
        if config.enable_clean_kv_store:
            self.clean_store = CleanKVStore(
                num_layers=config.num_layers,
                max_frames=config.max_context_frames + config.num_frames_per_block,
                frame_seq_length=config.frame_seq_length,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                batch_size=batch_size,
                dtype=dtype,
                device=device,
            )

        self.context_end_tokens: int = 0

    # -- transformer-compatible dict views ------------------------------------

    def self_attn_dicts(self) -> list[dict[str, Any]]:
        return [buf.as_dict() for buf in self.self_attn]

    def cross_attn_dicts(self) -> list[dict[str, Any]]:
        return [buf.as_dict() for buf in self.cross_attn]

    def sync_self_attn_from_dicts(self, dicts: list[dict[str, Any]]) -> None:
        for buf, d in zip(self.self_attn, dicts):
            buf.sync_from_dict(d)

    def sync_cross_attn_from_dicts(self, dicts: list[dict[str, Any]]) -> None:
        for buf, d in zip(self.cross_attn, dicts):
            buf.sync_from_dict(d)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class DiffusionKVManager:
    """Orchestrates the KV cache lifecycle for streaming diffusion blocks.

    The manager is a *bookkeeper* — it manages buffer state and data but never
    calls the transformer itself.  Pipeline stages call the transformer using
    the ``list[dict]`` views returned by ``get_transformer_caches()``.

    Typical per-block flow (KREA streaming)::

        manager.begin_block(session, block_idx)

        # Context warmup (block_idx > 0)
        if block_idx > 0:
            if not manager.try_load_cached_context(session):
                kv, xa = manager.get_transformer_caches(session)
                transformer(context_frames, t=0, kv_cache=kv, ...)
                manager.sync_after_forward(session, kv, xa)
            manager.mark_context_ready(session)

        # Denoising loop
        manager.begin_denoising(session)
        for t in timesteps:
            kv, xa = manager.get_transformer_caches(session)
            transformer(noisy, t, kv_cache=kv, ...)

        # After denoising — save clean KV from context warmup for next block
        manager.save_context_as_clean(session)
    """

    def __init__(self, config: DiffusionKVConfig):
        self.config = config

    # -- block lifecycle ------------------------------------------------------

    def begin_block(self, session: DiffusionKVSession, block_idx: int) -> None:
        """Reset self-attn KV buffers and prepare for a new block."""
        session.current_block_idx = block_idx
        for buf in session.self_attn:
            buf.reset()
        session.context_end_tokens = 0
        session.state = (
            KVLifecycleState.CONTEXT_READY if block_idx == 0
            else KVLifecycleState.EMPTY
        )

    def try_load_cached_context(self, session: DiffusionKVSession) -> bool:
        """Try to populate the Context Zone from CleanKVStore.

        Returns True if context was loaded (caller can skip context forward),
        False if a context forward is still required.
        """
        if session.clean_store is None or session.clean_store.num_stored_frames == 0:
            return False

        n_tokens = 0
        for layer_idx, buf in enumerate(session.self_attn):
            n_tokens = session.clean_store.load_into_buffer(
                buf, layer_idx, self.config.max_context_frames,
            )

        session.context_end_tokens = n_tokens
        session.state = KVLifecycleState.CONTEXT_READY
        logger.debug(
            "Loaded %d cached context tokens for block %d",
            n_tokens, session.current_block_idx,
        )
        return True

    def mark_context_ready(self, session: DiffusionKVSession) -> None:
        """Call after a context forward pass fills the Context Zone."""
        if session.self_attn:
            session.context_end_tokens = session.self_attn[0].global_end_index
        session.state = KVLifecycleState.CONTEXT_READY

    def begin_denoising(self, session: DiffusionKVSession) -> None:
        """Transition from CONTEXT_READY to DENOISING."""
        if session.state != KVLifecycleState.CONTEXT_READY:
            raise RuntimeError(
                f"Cannot begin denoising in state {session.state}; "
                "expected CONTEXT_READY"
            )
        session.state = KVLifecycleState.DENOISING

    def save_context_as_clean(
        self,
        session: DiffusionKVSession,
        num_new_frames: int | None = None,
    ) -> None:
        """Persist the context-zone KV into CleanKVStore for future reuse.

        In KREA's flow, the context warmup at the start of each block generates
        clean KV for all context frames.  After denoising completes, we save
        the *new* block's portion so it becomes part of the context for the
        next block.

        Args:
            num_new_frames: Number of newly denoised frames in this block.
                Defaults to ``config.num_frames_per_block``.
        """
        if session.clean_store is None:
            session.state = KVLifecycleState.CLEAN_COMMITTED
            return

        if num_new_frames is None:
            num_new_frames = self.config.num_frames_per_block

        overflow = (
            session.clean_store.num_stored_frames + num_new_frames
            - session.clean_store.max_frames
        )
        if overflow > 0:
            session.clean_store.evict_oldest(overflow)

        ctx_end = session.context_end_tokens
        new_tokens = num_new_frames * self.config.frame_seq_length
        for layer_idx, buf in enumerate(session.self_attn):
            session.clean_store.store_clean_kv(
                layer_idx,
                buf.k[:, ctx_end:ctx_end + new_tokens],
                buf.v[:, ctx_end:ctx_end + new_tokens],
            )
        session.clean_store.advance_frames(num_new_frames)
        session.state = KVLifecycleState.CLEAN_COMMITTED

    # -- transformer interface ------------------------------------------------

    def get_transformer_caches(
        self, session: DiffusionKVSession,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return ``(kv_cache, crossattn_cache)`` dict lists for the transformer."""
        return session.self_attn_dicts(), session.cross_attn_dicts()

    def sync_after_forward(
        self,
        session: DiffusionKVSession,
        kv_dicts: list[dict[str, Any]],
        xa_dicts: list[dict[str, Any]] | None = None,
    ) -> None:
        """Pull transformer-updated index/flag values back into the buffers."""
        session.sync_self_attn_from_dicts(kv_dicts)
        if xa_dicts is not None:
            session.sync_cross_attn_from_dicts(xa_dicts)

    # -- cross-attention helpers ----------------------------------------------

    def reset_cross_attn(self, session: DiffusionKVSession) -> None:
        """Reset all cross-attention buffers (e.g. on prompt change)."""
        for buf in session.cross_attn:
            buf.reset()

    # -- factory --------------------------------------------------------------

    @classmethod
    def from_transformer(
        cls,
        transformer,
        *,
        max_context_frames: int,
        num_frames_per_block: int,
        frame_seq_length: int,
        local_attn_size: int = -1,
        max_text_length: int = 512,
        enable_clean_kv_store: bool = True,
    ) -> DiffusionKVManager:
        """Create a manager by reading sizes from a KREA transformer."""
        config = DiffusionKVConfig(
            num_layers=len(transformer.blocks),
            num_heads=transformer.num_attention_heads,
            head_dim=transformer.attention_head_dim,
            max_context_frames=max_context_frames,
            num_frames_per_block=num_frames_per_block,
            frame_seq_length=frame_seq_length,
            local_attn_size=local_attn_size,
            max_text_length=max_text_length,
            enable_clean_kv_store=enable_clean_kv_store,
        )
        return cls(config)

    def create_session(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> DiffusionKVSession:
        """Allocate a new KV session for a streaming generation."""
        return DiffusionKVSession(self.config, batch_size, dtype, device)
