# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.models.dits.causal_wan_common import (
    BaseCausalWanSelfAttention,
    BaseCausalWanTransformer3DModel,
    BaseCausalWanTransformerBlock,
)
from sglang.multimodal_gen.runtime.pipelines_core.kv_cache import (
    CrossAttentionKVCache,
    SelfAttentionKVCache,
)


class KreaCausalWanSelfAttention(BaseCausalWanSelfAttention):

    def _should_use_flex_attention(self, block_mask, kv_cache) -> bool:
        return kv_cache is None or block_mask is not None

    def _prepare_flex_cache(
        self,
        roped_key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: SelfAttentionKVCache | None,
    ) -> None:
        if kv_cache is not None:
            # Bulk write mode: populate cache from position 0 and use flex_attention.
            kv_cache.bulk_write(roped_key, value)

    def _incremental_attention(
        self,
        roped_query: torch.Tensor,
        roped_key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: SelfAttentionKVCache,
        current_start: int,
        cache_start: int,
    ) -> torch.Tensor:
        kv_cache.append(roped_key, value, current_start)
        active_k, active_v = kv_cache.get_active_kv(self.max_attention_size)
        return self.attn(roped_query, active_k, active_v)


class KreaCausalWanTransformerBlock(BaseCausalWanTransformerBlock):
    self_attn_cls = KreaCausalWanSelfAttention


class KreaCausalWanTransformer3DModel(BaseCausalWanTransformer3DModel):
    block_cls = KreaCausalWanTransformerBlock

    def _get_rope_embed_kwargs(self, hidden_states: torch.Tensor) -> dict:
        # Krea path keeps rotary embeddings on-device to avoid extra transfers.
        return {"device": hidden_states.device}

    def _use_gradient_checkpointing_inference(self) -> bool:
        # Keep Krea inference path simple and avoid extra graph machinery.
        return False

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        kv_cache: list[SelfAttentionKVCache] = None,
        crossattn_cache: list[CrossAttentionKVCache] = None,
        current_start: int = 0,
        cache_start: int = 0,
        start_frame: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        return super()._forward_inference(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_hidden_states_image=encoder_hidden_states_image,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
            start_frame=start_frame,
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        return self._forward_inference(*args, **kwargs)


EntryClass = KreaCausalWanTransformer3DModel
