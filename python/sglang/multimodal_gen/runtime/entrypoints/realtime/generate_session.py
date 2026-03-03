from collections import deque
from typing import Any
from uuid import uuid4

import torch

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeAction,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params


class GenerateSession:

    def __init__(self):
        self.id = uuid4().hex
        self.request_id = None
        self.request = None
        self.action_queue = deque(maxlen=3)
        self.generate_chunk_cnt = 0
        self.kv_cache: Any = None
        self.crossattn_cache: Any = None
        self.current_denoised_latents: torch.Tensor = None
        self.frame_cache_context: deque = None
        self.decoder_cache: Any = None
        self.input_frames_cache: deque = None

    def setRequest(self, request: RealtimeVideoGenerationsRequest):
        self.request = request

    def dispose(self):
        self.action_queue.clear()

    def new_request(self):
        self.request_id = f"{self.id}_{uuid4().hex}"

    def generate_chunk_completed(self):
        self.generate_chunk_cnt += 1

    def append_action(self, action: RealtimeAction):
        self.action_queue.append(action)

    def sample_action(self) -> RealtimeAction:
        return self.action_queue.popleft()

    def build_sampling_params(self):
        if self.generate_chunk_cnt == 0:
            prompt = self.request.prompt
        elif len(self.action_queue) > 0:
            realtime_action = self.action_queue.popleft()
            # only support prompt action
            if realtime_action.type == "prompt":
                prompt = realtime_action.action_content
        else:
            # TODO(@puf147): generate with empty action
            prompt = self.request.prompt

        return build_sampling_params(
            self.request_id,
            prompt=prompt,
            size=self.request.size,
            num_frames=self.request.num_frames,
            fps=self.request.fps,
            image_path=self.request.first_frame,
            output_file_name=self.request_id,
            seed=self.request.seed,
            generator_device=self.request.generator_device,
            num_inference_steps=self.request.num_inference_steps,
            guidance_scale=self.request.guidance_scale,
            guidance_scale_2=self.request.guidance_scale_2,
            negative_prompt=self.request.negative_prompt,
            enable_teacache=self.request.enable_teacache,
            output_path=self.request.output_path,
            output_compression=self.request.output_compression,
            output_quality=self.request.output_quality,
        )
