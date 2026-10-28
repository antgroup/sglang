"""DeepSeek-V4.1 vision tower and aligner."""

from collections import OrderedDict
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.activation.vision_swiglu import vision_swiglu
from sglang.kernels.ops.attention.vision_rope_half import vision_qk_rope_half
from sglang.srt.layers.attention.vision import (
    VisionAttention,
    VisionAttentionMetadata,
    prepare_vision_attention_metadata,
)
from sglang.srt.layers.dsv41_sm90 import dsv41_sm90_on, dsv41_sm90_role
from sglang.srt.layers.layernorm import RMSNorm

# The dtypes the fused vision kernels take.
_FUSED_DTYPES = (torch.bfloat16, torch.float32)


def _rms_norm(dim: int) -> RMSNorm:
    # The fused CUDA kernels do not take an fp32 weight with a bf16 input.
    return RMSNorm(dim, eps=1e-6, weight_dtype=torch.float32, force_native=True)


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.proj = nn.Linear(3 * args.vision_patch_size**2, args.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


def apply_vision_rotary(q, k, position_embeddings, x_shape):
    # The reference pairs the two halves of each head, with FP32 arithmetic.
    cos, sin = position_embeddings
    return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


def apply_vision_rotary_fused(q, k, position_embeddings, x_shape):
    # RL on-policy targets hand over cos/sin cast to the activation dtype.
    if any(p.dtype != torch.float32 for p in position_embeddings):
        return apply_vision_rotary(q, k, position_embeddings, x_shape)
    return vision_qk_rope_half(q, k, *position_embeddings)


class Attention(VisionAttention):
    def __init__(self, args, *, fused: bool):
        super().__init__(
            embed_dim=args.vision_dim,
            num_heads=args.vision_n_heads,
            projection_size=args.vision_dim,
            use_qkv_parallel=True,
            use_data_parallel=True,
            customized_position_embedding_applier=(
                apply_vision_rotary_fused if fused else apply_vision_rotary
            ),
        )
        # The fused RoPE reads token-strided Q/K views and writes dense outputs,
        # and FA3 takes the strided V view as is; other backends stay dense.
        if fused and self.qkv_backend_name == "fa3":
            self.pass_strided_qkv = True

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: VisionAttentionMetadata,
    ) -> torch.Tensor:
        return (
            super()
            .forward(
                x,
                position_embeddings=(cos, sin),
                forward_metadata=metadata,
            )
            .squeeze(0)
        )


class MLP(nn.Module):
    def __init__(self, args, *, fused: bool):
        super().__init__()
        self.fused = fused
        self.w1 = nn.Linear(args.vision_dim, 2 * args.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(args.vision_inter_dim, args.vision_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return self.w2(vision_swiglu(self.w1(x)))
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, args, *, fused: bool):
        super().__init__()
        self.norm1 = _rms_norm(args.vision_dim)
        self.attn = Attention(args, fused=fused)
        self.norm2 = _rms_norm(args.vision_dim)
        self.mlp = MLP(args, fused=fused)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: VisionAttentionMetadata,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, metadata)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, args):
        super().__init__()
        self.rope_dim = args.vision_dim // args.vision_n_heads // 2
        self.rope_theta = args.vision_rope_theta
        self.patch_embed = PatchEmbed(args)
        # E only, where the gain was measured; each fused op is bitwise the
        # eager op it replaces.
        self.fused = (
            dsv41_sm90_on("epd")
            and dsv41_sm90_role() == "encoder"
            and self.patch_embed.proj.weight.dtype in _FUSED_DTYPES
        )
        # (n_h, n_w, device) -> (cos, sin, metadata); filled only when fused.
        self._device_metadata = OrderedDict() if self.fused else None
        self.blocks = nn.ModuleList(
            [Block(args, fused=self.fused) for _ in range(args.vision_n_layers)]
        )
        self.norm = _rms_norm(args.vision_dim)

    @torch.no_grad()
    def prewarm_fused_kernels(self) -> None:
        """Compile the fused kernels before the first image arrives."""
        weight = self.patch_embed.proj.weight
        attn = self.blocks[0].attn
        heads, head_dim = attn.num_attention_heads_per_partition, attn.head_size
        cos = torch.ones(
            (8, 1, head_dim // 2), device=weight.device, dtype=torch.float32
        )
        q = torch.zeros((8, heads, head_dim), device=weight.device, dtype=weight.dtype)
        vision_qk_rope_half(q, q, cos, cos)
        # The token stride is a kernel specialization, so warm the QKV view too.
        if attn.pass_strided_qkv:
            qkv = torch.zeros(
                (8, 3, heads, head_dim), device=weight.device, dtype=weight.dtype
            )
            vision_qk_rope_half(qkv[:, 0], qkv[:, 1], cos, cos)
        packed = torch.zeros(
            (8, self.blocks[0].mlp.w1.out_features),
            device=weight.device,
            dtype=weight.dtype,
        )
        vision_swiglu(packed)
        torch.cuda.synchronize(weight.device)

    def _rope_and_metadata(self, x: torch.Tensor, n_h: int, n_w: int):
        key = (n_h, n_w, x.device)
        if self.fused and key in self._device_metadata:
            self._device_metadata.move_to_end(key)
            return self._device_metadata[key]
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        cos, sin = cos.to(x.device), sin.to(x.device)
        # Passing the known length avoids device-to-host length discovery per layer.
        metadata = prepare_vision_attention_metadata(
            torch.tensor([0, x.shape[0]], dtype=torch.int32),
            x.device,
            max_seqlen=x.shape[0],
        )
        if self.fused:
            self._device_metadata[key] = (cos, sin, metadata)
            # Arbitrary bound, the same as the host-side get_vision_cos_sin cache.
            if len(self._device_metadata) > 8:
                self._device_metadata.popitem(last=False)
        return cos, sin, metadata

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin, metadata = self._rope_and_metadata(x, n_h, n_w)
        for block in self.blocks:
            x = block(x, cos, sin, metadata)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.downsample_ratio = args.vision_downsample_ratio
        in_dim = args.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, args.dim)
        self.w2 = nn.Linear(args.dim, args.dim)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))
