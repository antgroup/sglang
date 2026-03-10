from collections import deque
from uuid import uuid4

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeAction,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import RealtimeSession


class GenerateSession:

    def __init__(self):
        self.id = uuid4().hex
        self.request_id = None
        self.request = None
        self.action_queue = deque(maxlen=1)
        self.video_chunk_queue = deque(maxlen=64)
        self.generate_chunk_cnt = 0
        self.realtime_session = RealtimeSession()

    def setRequest(self, request: RealtimeVideoGenerationsRequest):
        self.request = request

    def dispose(self):
        self.action_queue.clear()
        self.video_chunk_queue.clear()

    def new_request(self):
        self.request_id = f"{self.id}_{uuid4().hex}"

    def generate_chunk_completed(self):
        self.generate_chunk_cnt += 1

    def append_action(self, action: RealtimeAction):
        self.action_queue.append(action)

    def append_video_frames(self, frames: list):
        if len(frames) > 0:
            # Keep client chunk boundaries so each generation step consumes one chunk.
            self.video_chunk_queue.append(frames)

    def has_pending_video_frames(self) -> bool:
        return len(self.video_chunk_queue) > 0

    def sample_action(self) -> RealtimeAction:
        return self.action_queue.popleft()

    def sample_video_frames(self):
        if len(self.video_chunk_queue) == 0:
            return None
        return self.video_chunk_queue.popleft()

    def build_sampling_params(self):
        if self.generate_chunk_cnt == 0:
            prompt = self.request.prompt
        elif (
            not self.realtime_session.interpolated_embeds and len(self.action_queue) > 0
        ):
            realtime_action = self.sample_action()
            # only support prompt action for now
            if realtime_action.type == "prompt":
                prompt = realtime_action.action_content
                self.request.prompt = prompt
        else:
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
            profile=self.request.profile,
            num_profiled_timesteps=self.request.num_profiled_timesteps,
            profile_all_stages=self.request.profile_all_stages,
            perf_dump_path=self.request.perf_dump_path,
            output_path=self.request.output_path,
            output_compression=self.request.output_compression,
            output_quality=self.request.output_quality,
        )
