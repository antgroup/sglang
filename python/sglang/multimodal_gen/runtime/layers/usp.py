# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import logging
from typing import TYPE_CHECKING

import torch
from torch.distributed.tensor.experimental._attention import _cp_options

from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a import (
    A2AMode,
    A2ASlot,
    UlyssesA2ATransaction,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
)
from sglang.srt.utils.common import torch_release

_cp_options.enable_load_balance = False

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
        AttentionImpl,
    )

logger = logging.getLogger(__name__)


def _coerce_slot(slot: A2ASlot | str) -> A2ASlot:
    return slot if isinstance(slot, A2ASlot) else A2ASlot(slot)


def _usp_input_all_to_all(
    x: torch.Tensor,
    head_dim: int = 1,
    *,
    transaction: UlyssesA2ATransaction | None = None,
    slot: A2ASlot | str = A2ASlot.COMPAT,
) -> torch.Tensor:
    """Gather sequence and shard heads through the configured A2A backend."""
    semantic_slot = _coerce_slot(slot)
    if transaction is not None:
        return transaction.input_all_to_all(x, head_dim=head_dim, slot=semantic_slot)
    router = get_sp_group().get_ulysses_a2a_router(x.device)
    if router.config.backend in {"nccl", "fast_ulysses"}:
        return router.fixed_backend_all_to_all(
            x,
            mode=A2AMode.INPUT,
            head_dim=head_dim,
            seq_lens=None,
            slot=semantic_slot,
        )
    return router.input_all_to_all(x, head_dim=head_dim, slot=semantic_slot)


def _usp_input_all_to_all_variable(
    x: torch.Tensor,
    seq_lens: list[int],
    head_dim: int = 1,
    *,
    transaction: UlyssesA2ATransaction | None = None,
    slot: A2ASlot | str = A2ASlot.COMPAT,
) -> torch.Tensor:
    """Variable-split input redistribution; V1 routes this to NCCL."""
    semantic_slot = _coerce_slot(slot)
    if transaction is not None:
        return transaction.input_all_to_all(
            x,
            head_dim=head_dim,
            seq_lens=seq_lens,
            slot=semantic_slot,
        )
    router = get_sp_group().get_ulysses_a2a_router(x.device)
    if router.config.backend in {"nccl", "fast_ulysses"}:
        return router.fixed_backend_all_to_all(
            x,
            mode=A2AMode.INPUT,
            head_dim=head_dim,
            seq_lens=seq_lens,
            slot=semantic_slot,
        )
    return router.input_all_to_all(
        x,
        head_dim=head_dim,
        seq_lens=seq_lens,
        slot=semantic_slot,
    )


def _usp_output_all_to_all(
    x: torch.Tensor,
    head_dim: int = 1,
    *,
    transaction: UlyssesA2ATransaction | None = None,
    slot: A2ASlot | str = A2ASlot.OUT,
) -> torch.Tensor:
    """Shard sequence and gather heads through the configured A2A backend."""
    semantic_slot = _coerce_slot(slot)
    if transaction is not None:
        return transaction.output_all_to_all(x, head_dim=head_dim, slot=semantic_slot)
    router = get_sp_group().get_ulysses_a2a_router(x.device)
    if router.config.backend in {"nccl", "fast_ulysses"}:
        return router.fixed_backend_all_to_all(
            x,
            mode=A2AMode.OUTPUT,
            head_dim=head_dim,
            seq_lens=None,
            slot=semantic_slot,
        )
    return router.output_all_to_all(x, head_dim=head_dim, slot=semantic_slot)


def _usp_output_all_to_all_variable(
    x: torch.Tensor,
    seq_lens: list[int],
    head_dim: int = 1,
    *,
    transaction: UlyssesA2ATransaction | None = None,
    slot: A2ASlot | str = A2ASlot.OUT,
) -> torch.Tensor:
    """Variable-split output redistribution; V1 routes this to NCCL."""
    semantic_slot = _coerce_slot(slot)
    if transaction is not None:
        return transaction.output_all_to_all(
            x,
            head_dim=head_dim,
            seq_lens=seq_lens,
            slot=semantic_slot,
        )
    router = get_sp_group().get_ulysses_a2a_router(x.device)
    if router.config.backend in {"nccl", "fast_ulysses"}:
        return router.fixed_backend_all_to_all(
            x,
            mode=A2AMode.OUTPUT,
            head_dim=head_dim,
            seq_lens=seq_lens,
            slot=semantic_slot,
        )
    return router.output_all_to_all(
        x,
        head_dim=head_dim,
        seq_lens=seq_lens,
        slot=semantic_slot,
    )


def ring_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_impl: "AttentionImpl",
    is_causal: bool = False,
    dropout_p: float = 0.0,
):
    """Run ring attention inside the configured ring subgroup."""
    # torch.distributed.tensor.experimental._attention is not a public API,
    from torch.distributed.tensor.experimental._attention import (
        _templated_ring_attention,
    )

    ring_pg = get_sp_group().ring_group
    assert ring_pg is not None, "Ring process group is not initialized."

    # Ring primitives expect [B, H, S, D].
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    def attn_callable_adapter(q, k, v, *args, **kwargs):
        q = torch.permute(q, [0, 2, 1, 3])
        k = torch.permute(k, [0, 2, 1, 3])
        v = torch.permute(v, [0, 2, 1, 3])
        output, softmax_lse, *rest = attn_impl.forward(
            q,
            k,
            v,
            attn_metadata=None,
            return_softmax_lse=True,
        )
        output = torch.permute(output, [0, 2, 1, 3])
        return output, softmax_lse, *rest

    use_segment_id = torch_release >= (2, 6)
    attn_kwargs = dict(
        op=attn_callable_adapter,
        dropout_p=dropout_p,
        is_causal=is_causal,
        query=query,
        key=key,
        value=value,
        group=ring_pg,
    )

    if use_segment_id:
        out, *_ = _templated_ring_attention(
            seq_dim=1,
            **attn_kwargs,
        )
    else:
        out, *_ = _templated_ring_attention(
            **attn_kwargs,
        )

    return torch.permute(out, [0, 2, 1, 3])
