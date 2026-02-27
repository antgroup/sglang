# Copied and adapted from: https://github.com/krea-ai/realtime-video/blob/main/wan/modules/causal_model.py and https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/models/dits/causal_wanvideo.py

import functools
import math
from typing import Any

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from sglang.multimodal_gen.configs.models.dits import WanVideoConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_world_size
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    RMSNorm,
    ScaleResidualLayerNormScaleShift,
)
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    _apply_rotary_emb,
    get_rotary_pos_embed,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import PatchEmbed
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanT2VCrossAttention,
    WanTimeTextImageEmbedding,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
)
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@functools.lru_cache(maxsize=32)
def get_block_mask(
    device: str,
    num_frames: int = 21,
    frame_seqlen: int = 1560,
    num_frame_per_block=3,
    local_attn_size=-1,
):
    print("Generating block mask")
    total_length = num_frames * frame_seqlen

    # we do right padding to get to a multiple of 128
    padded_length = math.ceil(total_length / 128) * 128 - total_length

    ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

    # Block-wise causal mask will attend to all elements that are before the end of the current chunk
    frame_indices = torch.arange(
        start=0,
        end=total_length,
        step=frame_seqlen * num_frame_per_block,
        device=device,
    )

    for tmp in frame_indices:
        ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
            tmp + frame_seqlen * num_frame_per_block
        )

    def attention_mask(b, h, q_idx, kv_idx):
        if local_attn_size == -1:
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
        else:
            return (
                (kv_idx < ends[q_idx])
                & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
            ) | (q_idx == kv_idx)

    block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=False,
        device=device,
    )
    return block_mask


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][start_frame : start_frame + f]
                .view(f, 1, 1, -1)
                .expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class KreaCausalWanSelfAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm=True,
        eps=1e-6,
        parallel_attention=False,
    ) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.parallel_attention = parallel_attention
        self.max_attention_size = (
            32760 if local_attn_size == -1 else local_attn_size * 1560
        )

        # Scaled dot product attention
        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
            ),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        grid_sizes,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask,
        kv_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        if cache_start is None:
            cache_start = current_start

        if kv_cache is None or block_mask is not None:
            cos, sin = freqs_cis
            roped_query = _apply_rotary_emb(q, cos, sin, is_neox_style=False).type_as(v)
            roped_key = _apply_rotary_emb(k, cos, sin, is_neox_style=False).type_as(v)
            local_end_index = roped_key.shape[1]
            kv_cache["k"][:, :local_end_index] = roped_key
            kv_cache["v"][:, :local_end_index] = v

            kv_cache["global_end_index"] = local_end_index
            kv_cache["local_end_index"] = local_end_index
            # Padding for flex attention
            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [
                    roped_query,
                    torch.zeros(
                        [q.shape[0], padded_length, q.shape[2], q.shape[3]],
                        device=q.device,
                        dtype=v.dtype,
                    ),
                ],
                dim=1,
            )

            padded_roped_key = torch.cat(
                [
                    roped_key,
                    torch.zeros(
                        [k.shape[0], padded_length, k.shape[2], k.shape[3]],
                        device=k.device,
                        dtype=v.dtype,
                    ),
                ],
                dim=1,
            )

            padded_v = torch.cat(
                [
                    v,
                    torch.zeros(
                        [v.shape[0], padded_length, v.shape[2], v.shape[3]],
                        device=v.device,
                        dtype=v.dtype,
                    ),
                ],
                dim=1,
            )

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1).contiguous(),
                key=padded_roped_key.transpose(2, 1).contiguous(),
                value=padded_v.transpose(2, 1).contiguous(),
                block_mask=block_mask,
                kernel_options={
                    "BLOCKS_ARE_CONTIGUOUS": True,
                },
            )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = 1560
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs_cis, start_frame=current_start_frame
            ).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs_cis, start_frame=current_start_frame
            ).type_as(v)
            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            global_end_index = (
                int(kv_cache["global_end_index"].item())
                if isinstance(kv_cache["global_end_index"], torch.Tensor)
                else int(kv_cache["global_end_index"])
            )
            local_end_index_prev = (
                int(kv_cache["local_end_index"].item())
                if isinstance(kv_cache["local_end_index"], torch.Tensor)
                else int(kv_cache["local_end_index"])
            )
            if (
                self.local_attn_size != -1
                and (current_end > global_end_index)
                and (num_new_tokens + local_end_index_prev > kv_cache_size)
            ):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = (
                    num_new_tokens + local_end_index_prev - kv_cache_size
                )
                num_rolled_tokens = (
                    local_end_index_prev - num_evicted_tokens - sink_tokens
                )
                kv_cache["k"][
                    :, sink_tokens : sink_tokens + num_rolled_tokens
                ] = kv_cache["k"][
                    :,
                    sink_tokens
                    + num_evicted_tokens : sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()
                kv_cache["v"][
                    :, sink_tokens : sink_tokens + num_rolled_tokens
                ] = kv_cache["v"][
                    :,
                    sink_tokens
                    + num_evicted_tokens : sink_tokens
                    + num_evicted_tokens
                    + num_rolled_tokens,
                ].clone()
                # Insert the new keys/values at the end
                local_end_index = (
                    local_end_index_prev
                    + current_end
                    - global_end_index
                    - num_evicted_tokens
                )
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = local_end_index_prev + current_end - global_end_index
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"] = kv_cache["k"].detach()
                kv_cache["v"] = kv_cache["v"].detach()
                # logger.info("kv_cache['k'] is in comp graph: %s", kv_cache["k"].requires_grad or kv_cache["k"].grad_fn is not None)
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            x = self.attn(
                roped_query,
                kv_cache["k"][
                    :,
                    max(0, local_end_index - self.max_attention_size) : local_end_index,
                ],
                kv_cache["v"][
                    :,
                    max(0, local_end_index - self.max_attention_size) : local_end_index,
                ],
            )
            if isinstance(kv_cache["global_end_index"], torch.Tensor):
                kv_cache["global_end_index"].fill_(current_end)
            else:
                kv_cache["global_end_index"] = current_end
            if isinstance(kv_cache["local_end_index"], torch.Tensor):
                kv_cache["local_end_index"].fill_(local_end_index)
            else:
                kv_cache["local_end_index"] = local_end_index

        return x


class KreaCausalWanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
        prefix: str = "",
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)

        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        self.attn1 = KreaCausalWanSelfAttention(
            dim,
            num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        self.local_attn_size = local_attn_size
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            print("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
        )

        # 2. Cross-attention
        # Only T2V for now
        self.attn2 = WanT2VCrossAttention(dim, num_heads, qk_norm=qk_norm, eps=eps)
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = MulAdd()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask,
        kv_cache: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        # hidden_states.shape: [batch_size, seq_length, inner_dim]
        # temb.shape: [batch_size, num_frames, 6, inner_dim]
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        num_frames = temb.shape[1]
        frame_seqlen = hidden_states.shape[1] // num_frames
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        # assert orig_dtype != torch.float32
        e = self.scale_shift_table + temb
        # e.shape: [batch_size, num_frames, 6, inner_dim]
        assert e.shape == (bs, num_frames, 6, self.hidden_dim)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=2
        )
        # *_msa.shape: [batch_size, num_frames, 1, inner_dim]
        # assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            * (1 + scale_msa)
            + shift_msa
        ).flatten(1, 2)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q.forward_native(query)
        if self.norm_k is not None:
            key = self.norm_k.forward_native(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        attn_output = self.attn1(
            query,
            key,
            value,
            freqs_cis,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
        )
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(
            norm_hidden_states,
            context=encoder_hidden_states,
            context_lens=None,
            crossattn_cache=crossattn_cache,
        )
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
        )

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)

        return hidden_states


class KreaCausalWanTransformer3DModel(CachableDiT, OffloadableDiTMixin):
    """
    Real-time Wan2.1-T2V-14B model implementation for SGLang.

    This class extends the base Wan model with real-time optimization features:
    - KV cache for attention layers
    - Optimized memory management for streaming
    - Real-time inference optimizations
    """

    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig()._supported_attention_backends
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    def __init__(self, config: WanVideoConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.attention_head_dim
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.local_attn_size = config.local_attn_size

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                KreaCausalWanTransformerBlock(
                    inner_dim,
                    config.ffn_dim,
                    config.num_attention_heads,
                    config.local_attn_size,
                    config.sink_size,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                    config.added_kv_proj_dim,
                    self._supported_attention_backends,
                    prefix=f"{config.prefix}.blocks.{i}",
                )
                for i in range(config.num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(
            inner_dim,
            norm_type="layer",
            eps=config.eps,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size)
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False

        # Causal-specific
        self.block_mask = None
        self.num_frame_per_block = config.arch_config.num_frames_per_block
        assert self.num_frame_per_block <= 3
        self.independent_first_frame = False

        self.__post_init__()

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block=1,
        local_attn_size=-1,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        block_mask = get_block_mask(
            str(device), num_frames, frame_seqlen, num_frame_per_block, local_attn_size
        )
        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block=1,
        local_attn_size=-1,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(
            total_length + padded_length, device=device, dtype=torch.long
        )

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device,
        )

        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
                tmp + frame_seqlen * num_frame_per_block
            )

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return (
                    (kv_idx < ends[q_idx])
                    & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))
                ) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        return block_mask

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        start_frame: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)
        """

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if (
            isinstance(encoder_hidden_states_image, list)
            and len(encoder_hidden_states_image) > 0
        ):
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (
                post_patch_num_frames * get_sp_world_size(),
                post_patch_height,
                post_patch_width,
            ),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32,
            rope_theta=10000,
            start_frame=start_frame,  # Assume that start_frame is 0 when kv_cache is None
        )
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.stack(
            [torch.tensor(hidden_states[0].shape[1:], dtype=torch.long)]
        )
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        encoder_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                encoder_hidden_states.new_zeros(
                    1,
                    self.text_len - encoder_hidden_states.size(1),
                    encoder_hidden_states.size(2),
                ),
            ],
            dim=1,
        )

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep.flatten(), encoder_hidden_states, encoder_hidden_states_image
            )
        )
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size)).unflatten(
            dim=0, sizes=timestep.shape
        )

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        assert encoder_hidden_states.dtype == orig_dtype

        # 4. Transformer blocks
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                causal_kwargs = {
                    "kv_cache": kv_cache[block_index],
                    "current_start": current_start,
                    "cache_start": cache_start,
                    "block_mask": self.block_mask,
                }
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs_cis,
                    **causal_kwargs,
                )
            else:
                causal_kwargs = {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": crossattn_cache[block_index],
                    "current_start": current_start,
                    "cache_start": cache_start,
                    "block_mask": self.block_mask,
                }
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs_cis,
                    **causal_kwargs,
                )

        # 5. Output norm, projection & unpatchify
        temb = temb.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        shift, scale = (self.scale_shift_table.unsqueeze(1) + temb).chunk(2, dim=2)
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        output = self.unpatchify(hidden_states, grid_sizes)

        return torch.stack(output)

    def forward(self, *args, **kwargs):
        assert kwargs.get("kv_cache") is not None
        return self._forward_inference(*args, **kwargs)
