# copy and modify from https://github.com/thu-ml/TurboDiffusion/blob/main/turbodiffusion/rcm/utils/a2a_cp.py and https://github.com/thu-ml/TurboDiffusion/blob/main/turbodiffusion/SLA/core.py

import torch
from einops import rearrange
from torch import Tensor

from sglang.multimodal_gen.runtime.layers.attention.backends.sparse_linear_attn import (
    SageSparseLinearAttentionBackend,
    SparseLinearAttentionBackend,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.managers.forward_context import (
    ForwardContext,
    get_forward_context,
)
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import get_compute_dtype

logger = init_logger(__name__)


class MinimalA2AAttnOp(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        attention_type: str,
        topk: float,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__()
        dtype = get_compute_dtype()
        attn_backend = get_attn_backend(
            head_size, dtype, supported_attention_backends=supported_attention_backends
        )
        # Maintained for compatibility purposes; can be removed when CI allows setting Attention_backend or when TurboWan supports FA.
        if attn_backend not in (
            SparseLinearAttentionBackend,
            SageSparseLinearAttentionBackend,
        ):
            logger.warning(
                "TurboWan now only supports `sla_attn` or `sage_sla_attn` and has been automatically set to attention_type. Please set --attention-backend to `sla_attn` or `sage_sla_attn`."
            )
            if attention_type == "sagesla":
                attn_backend = SageSparseLinearAttentionBackend
            else:
                attn_backend = SparseLinearAttentionBackend

        self.local_attn = USPAttention(
            num_heads=num_heads,
            head_size=head_size,
            supported_attention_backends=supported_attention_backends,
            redirected_attention_backend=attn_backend,
            topk_ratio=topk,
        )

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata
        results = super().forward(query, key, value, ctx_attn_metadata)
        return rearrange(results, "b ... h l -> b ... (h l)")
