import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class WorldConditioningStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        prepare_world_condition = getattr(
            server_args.pipeline_config, "prepare_world_condition", None
        )
        if callable(prepare_world_condition):
            # Map request-specific world inputs to the transformer conditioning tensor.
            prepare_world_condition(batch, self.device, torch.float32)
        return batch

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        del server_args
        result = VerificationResult()
        result.add_check(
            "c2ws_plucker_emb", batch.c2ws_plucker_emb, V.none_or_tensor_with_dims(5)
        )
        return result
