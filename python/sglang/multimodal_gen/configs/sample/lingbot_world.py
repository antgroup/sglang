# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.wan import Wan2_2_I2V_A14B_SamplingParam


@dataclass
class LingBotWorldSamplingParams(Wan2_2_I2V_A14B_SamplingParam):
    diffusers_kwargs: dict[str, Any] = field(default_factory=dict)
    guidance_scale: float = 5.0
    guidance_scale_2: float = 5.0
    num_inference_steps: int = 70
    fps: int = 16

    @staticmethod
    def _default_chunk_size(server_args) -> int:
        return max(
            1,
            int(
                server_args.pipeline_config.dit_config.arch_config.num_frames_per_block
            ),
        )

    @classmethod
    def _normalize_chunk_size(cls, server_args, kwargs: dict[str, Any]) -> None:
        chunk_size = kwargs.get("chunk_size")
        if chunk_size is None:
            chunk_size = kwargs.pop("control_chunk_size", None)
        else:
            kwargs.pop("control_chunk_size", None)

        if chunk_size is None:
            chunk_size = cls._default_chunk_size(server_args)
        kwargs["chunk_size"] = max(1, int(chunk_size))

    @staticmethod
    def _normalize_actions(kwargs: dict[str, Any]) -> None:
        actions = kwargs.get("actions")
        if actions is None:
            actions = kwargs.pop("action_sequence", None)
        else:
            kwargs.pop("action_sequence", None)
        if actions is not None:
            kwargs["actions"] = actions

    @classmethod
    def normalize_diffusers_kwargs(
        cls, server_args, diffusers_kwargs: Any
    ) -> dict[str, Any]:
        if not isinstance(diffusers_kwargs, dict):
            diffusers_kwargs = {}
        else:
            diffusers_kwargs = dict(diffusers_kwargs)

        effective_chunk_size = None
        lingbot_kwargs = diffusers_kwargs.get("lingbot_world")
        if isinstance(lingbot_kwargs, dict):
            lingbot_kwargs = dict(lingbot_kwargs)
            cls._normalize_chunk_size(server_args, lingbot_kwargs)
            cls._normalize_actions(lingbot_kwargs)
            effective_chunk_size = lingbot_kwargs["chunk_size"]
            diffusers_kwargs["lingbot_world"] = lingbot_kwargs

        cls._normalize_chunk_size(server_args, diffusers_kwargs)
        cls._normalize_actions(diffusers_kwargs)
        if effective_chunk_size is not None:
            diffusers_kwargs["chunk_size"] = effective_chunk_size

        return diffusers_kwargs

    def _adjust(self, server_args):
        super()._adjust(server_args)
        self.diffusers_kwargs = self.normalize_diffusers_kwargs(
            server_args, self.diffusers_kwargs
        )
