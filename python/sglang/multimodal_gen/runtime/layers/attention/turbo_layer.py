# copy and modify from https://github.com/thu-ml/TurboDiffusion/blob/main/turbodiffusion/rcm/utils/a2a_cp.py and https://github.com/thu-ml/TurboDiffusion/blob/main/turbodiffusion/SLA/core.py

from typing import Any, Callable, Type, Union

import torch
from torch import Tensor
from torch.nn import Module

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionImpl,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sparse_linear_attn import (
    SageSparseLinearAttentionBackend,
    SparseLinearAttentionBackend,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    ForwardContext,
    get_forward_context,
)
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import get_compute_dtype

logger = init_logger(__name__)


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
    """

    def __init__(self, local_attention: Union[Module, Callable]) -> None:
        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, ctx_attn_metadata
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer

        Returns:
            * output (Tensor): context output
        """
        if get_sequence_parallel_world_size() < 2:
            return self.local_attn(query, key, value, ctx_attn_metadata)

        if get_ulysses_parallel_world_size() > 1:
            # -> [B, S, H_local, D]
            query = _usp_input_all_to_all(query, head_dim=2)
            key = _usp_input_all_to_all(key, head_dim=2)
            value = _usp_input_all_to_all(value, head_dim=2)
            out = self.local_attn(query, key, value, ctx_attn_metadata)

            # -> [B, S_local, H, D]
            output = _usp_output_all_to_all(out, head_dim=2)
        else:
            output = self.local_attn(query, key, value, ctx_attn_metadata)
        return output


class MinimalA2AAttnOp(DistributedAttention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        attention_type: str,
        topk: float,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
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
        impl_cls: Type["AttentionImpl"] = attn_backend.get_impl_cls()
        local_attn = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            topk_ratio=topk,
        )
        super(MinimalA2AAttnOp, self).__init__(local_attn)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs
    ) -> Tensor:
        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata
        results = super().forward(query, key, value, ctx_attn_metadata)
        return results
