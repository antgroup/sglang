import json
from collections import OrderedDict, deque
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeAction,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.pipelines_core.realtime_session import (
    RealtimeSession,
)


class RealtimeVideoMode(str, Enum):
    T2V = "t2v"
    V2V = "v2v"


SUPPORTED_CONTROL_KEYS = {"w", "a", "s", "d", "i", "j", "k", "l"}


class GenerateSession:
    _FIRST_BLOCK_ENCODE_FRAMES = 9
    _NEXT_BLOCK_ENCODE_FRAMES = 12

    def __init__(self):
        self.id = uuid4().hex
        self.request_id = None
        self.request = None
        self.mode: RealtimeVideoMode | None = None
        self.action_queue = deque(maxlen=1)
        self.control_queue = deque(maxlen=512)
        self.video_frame_queue = deque(maxlen=256)
        self.generate_chunk_cnt = 0
        self.current_keys: set[str] = set()
        self.last_control_actions: list[str] = []
        self.has_control_state = False
        self.realtime_session = RealtimeSession()
        self.prompt_events: dict[str, str] = {}
        self.prompt_event_order: list[str] = []
        self.prompt_event_mode: str = "overwrite"
        self.prompt_event_chunk: int = 1
        self.active_prompt_events: OrderedDict[str, int] = OrderedDict()

    def setRequest(self, request: RealtimeVideoGenerationsRequest):
        self.request = request
        self.configure_prompt_events(request)

    def set_mode(self, mode: RealtimeVideoMode | None):
        self.mode = mode

    def dispose(self):
        self.action_queue.clear()
        self.control_queue.clear()
        self.video_frame_queue.clear()
        self.mode = None
        self.request = None
        self.request_id = None
        self.generate_chunk_cnt = 0
        self.current_keys.clear()
        self.last_control_actions = []
        self.has_control_state = False
        self.realtime_session.dispose()
        self.prompt_events = {}
        self.prompt_event_order = []
        self.prompt_event_mode = "overwrite"
        self.prompt_event_chunk = 1
        self.active_prompt_events.clear()

    def new_request(self):
        self.request_id = f"{self.id}_{uuid4().hex}"

    def generate_chunk_completed(self):
        self.generate_chunk_cnt += 1

    def append_action(self, action: RealtimeAction):
        self.action_queue.append(action)

    def _append_control_frame(self, actions: list[str]):
        normalized = list(actions)
        self.control_queue.append(normalized)
        self.last_control_actions = normalized
        self.has_control_state = True

    def append_control_chunk(self, control_chunk: list[list[str]]):
        for actions in control_chunk:
            self._append_control_frame(actions)

    def validate_control_key_action(self, key: str, action: str) -> tuple[str, str]:
        normalized_key = key.lower()
        if normalized_key not in SUPPORTED_CONTROL_KEYS:
            raise ValueError(f"unsupported control key: {key}")
        if action not in ("down", "up"):
            raise ValueError(f"unsupported control action: {action}")
        return normalized_key, action

    def set_key_state(self, key: str, action: str) -> None:
        normalized_key, action = self.validate_control_key_action(key, action)
        before = set(self.current_keys)
        if action == "down":
            self.current_keys.add(normalized_key)
        elif action == "up":
            self.current_keys.discard(normalized_key)

        if self.current_keys != before:
            self._append_control_frame(sorted(self.current_keys))

    def append_video_frames(self, frames: list):
        if len(frames) > 0:
            self.video_frame_queue.extend(frames)

    def has_pending_video_frames(self) -> bool:
        return len(self.video_frame_queue) >= self.required_video_frames()

    def is_v2v_enabled(self) -> bool:
        if self.request is None:
            return False
        if self.mode is not None:
            return self.mode == RealtimeVideoMode.V2V
        # Auto mode only checks first_frame.
        return self.request.first_frame is not None

    def required_video_frames(self) -> int:
        # todo make _FIRST_BLOCK_ENCODE_FRAMES and _NEXT_BLOCK_ENCODE_FRAMES config
        if self.generate_chunk_cnt == 0:
            return self._FIRST_BLOCK_ENCODE_FRAMES
        return self._NEXT_BLOCK_ENCODE_FRAMES

    def sample_action(self) -> RealtimeAction:
        return self.action_queue.popleft()

    def configure_prompt_events(
        self, request: RealtimeVideoGenerationsRequest | None
    ) -> None:
        if request is None:
            self.prompt_events = {}
            self.prompt_event_order = []
            self.prompt_event_mode = "overwrite"
            self.prompt_event_chunk = 1
            self.active_prompt_events.clear()
            return

        raw_events: Any = getattr(request, "events", None) or {}
        if isinstance(raw_events, str):
            try:
                raw_events = json.loads(raw_events)
            except json.JSONDecodeError as exc:
                raise ValueError("events must be a JSON object") from exc
        if not isinstance(raw_events, dict):
            raise ValueError("events must be a JSON object")

        events: dict[str, str] = {}
        for key, value in raw_events.items():
            if value is None:
                continue
            event_id = str(key).strip()
            if not event_id:
                continue
            events[event_id] = str(value)

        event_mode = getattr(request, "event_mode", None) or "overwrite"
        if event_mode not in ("overwrite", "append"):
            raise ValueError("event_mode must be 'overwrite' or 'append'")

        event_chunk = getattr(request, "event_chunk", None)
        event_chunk = 1 if event_chunk is None else int(event_chunk)
        if event_chunk <= 0:
            raise ValueError("event_chunk must be a positive integer")

        self.prompt_events = events
        self.prompt_event_order = list(events.keys())
        self.prompt_event_mode = event_mode
        self.prompt_event_chunk = event_chunk
        self.active_prompt_events.clear()

    def resolve_action_event_id(self, action: RealtimeAction) -> str | None:
        event_ids = self.resolve_action_event_ids(action)
        if len(event_ids) == 0:
            return None
        return event_ids[0]

    def _normalize_event_ids(self, raw_event_ids: Any) -> list[str]:
        if raw_event_ids is None:
            return []
        if isinstance(raw_event_ids, (list, tuple, set)):
            values = raw_event_ids
        else:
            values = [raw_event_ids]

        event_ids: list[str] = []
        for raw_event_id in values:
            if raw_event_id is None:
                continue
            event_id = str(raw_event_id).strip()
            if event_id:
                event_ids.append(event_id)
        return event_ids

    def resolve_event_ids(
        self,
        *,
        event_id: Any = None,
        event: Any = None,
        event_ids: Any = None,
        events: Any = None,
        fallback_event: Any = None,
    ) -> list[str]:
        ordered_event_ids: list[str] = []
        seen_event_ids: set[str] = set()

        has_explicit_event = any(
            value is not None for value in (event_id, event, event_ids, events)
        )
        raw_values = [event_id, event, event_ids, events]
        if not has_explicit_event:
            raw_values.append(fallback_event)

        for raw_value in raw_values:
            for normalized_event_id in self._normalize_event_ids(raw_value):
                if normalized_event_id in seen_event_ids:
                    continue
                ordered_event_ids.append(normalized_event_id)
                seen_event_ids.add(normalized_event_id)
        return ordered_event_ids

    def resolve_action_event_ids(self, action: RealtimeAction) -> list[str]:
        fallback_event = action.action_content if action.type == "control" else None
        return self.resolve_event_ids(
            event_id=action.event_id,
            event=action.event,
            event_ids=action.event_ids,
            events=action.events,
            fallback_event=fallback_event,
        )

    def trigger_prompt_event(self, event_id: str) -> None:
        self.trigger_prompt_events([event_id])

    def validate_prompt_event_ids(self, event_ids: list[str]) -> list[str]:
        if len(event_ids) == 0:
            raise ValueError("event id cannot be empty")

        normalized_event_ids: list[str] = []
        seen_event_ids: set[str] = set()
        for event_id in event_ids:
            event_id = str(event_id).strip()
            if not event_id:
                raise ValueError("event id cannot be empty")
            if event_id not in self.prompt_events:
                raise ValueError(f"unknown prompt event: {event_id}")
            if event_id in seen_event_ids:
                continue
            normalized_event_ids.append(event_id)
            seen_event_ids.add(event_id)
        return normalized_event_ids

    def trigger_prompt_events(self, event_ids: list[str]) -> None:
        normalized_event_ids = self.validate_prompt_event_ids(event_ids)
        for event_id in normalized_event_ids:
            if event_id in self.active_prompt_events:
                del self.active_prompt_events[event_id]
            self.active_prompt_events[event_id] = self.prompt_event_chunk

    def apply_prompt_event_to_batch(self, batch) -> None:
        if not self.prompt_events:
            return

        batch.extra["prompt_events"] = self.prompt_events
        batch.extra["prompt_event_mode"] = self.prompt_event_mode
        batch.extra["prompt_event_chunk"] = self.prompt_event_chunk

        active_event_ids = []
        for event_id in self.prompt_event_order:
            remaining_chunks = self.active_prompt_events.get(event_id, 0)
            if remaining_chunks > 0:
                active_event_ids.append(event_id)
        if len(active_event_ids) == 0:
            self.active_prompt_events.clear()
            return

        batch.extra["prompt_event_ids"] = active_event_ids
        if len(active_event_ids) == 1:
            batch.extra["prompt_event_id"] = active_event_ids[0]

        expired_event_ids: list[str] = []
        for event_id in active_event_ids:
            remaining_chunks = self.active_prompt_events[event_id] - 1
            if remaining_chunks <= 0:
                expired_event_ids.append(event_id)
            else:
                self.active_prompt_events[event_id] = remaining_chunks
        for event_id in expired_event_ids:
            self.active_prompt_events.pop(event_id, None)

    def sample_control_chunk(self, chunk_size: int) -> list[list[str]] | None:
        if chunk_size <= 0:
            return None

        chunk: list[list[str]] = []
        while len(chunk) < chunk_size and len(self.control_queue) > 0:
            chunk.append(list(self.control_queue.popleft()))

        if len(chunk) == 0 and not self.has_control_state:
            # Keep emitting an explicit no-op control chunk so LingBot-style
            # camera conditioning stays active before any user control arrives.
            return [[] for _ in range(chunk_size)]

        pad_actions = list(self.last_control_actions)
        while len(chunk) < chunk_size:
            chunk.append(list(pad_actions))
        return chunk

    def sample_video_frames(self):
        required = self.required_video_frames()
        if len(self.video_frame_queue) < required:
            return None

        pending_frames = []
        while len(self.video_frame_queue) > 0:
            pending_frames.append(self.video_frame_queue.popleft())
        if len(pending_frames) < required:
            return None
        if len(pending_frames) == required:
            return pending_frames

        # TODO more sampling strategy.
        indices = np.round(np.linspace(0, len(pending_frames) - 1, required)).astype(
            int
        )
        return [pending_frames[i] for i in indices]

    def build_sampling_params(self):
        if self.generate_chunk_cnt == 0:
            prompt = self.request.prompt
        elif len(self.action_queue) > 0:
            realtime_action = self.sample_action()
            # TODO more sampling strategy.
            # only support prompt action now
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
            enable_frame_interpolation=self.request.enable_frame_interpolation,
            frame_interpolation_exp=self.request.frame_interpolation_exp,
            frame_interpolation_scale=self.request.frame_interpolation_scale,
            frame_interpolation_model_path=self.request.frame_interpolation_model_path,
            enable_upscaling=self.request.enable_upscaling,
            upscaling_model_path=self.request.upscaling_model_path,
            upscaling_scale=self.request.upscaling_scale,
            diffusers_kwargs=self.request.diffusers_kwargs,
            profile=self.request.profile,
            num_profiled_timesteps=self.request.num_profiled_timesteps,
            profile_all_stages=self.request.profile_all_stages,
            perf_dump_path=self.request.perf_dump_path,
            output_path=self.request.output_path,
            output_compression=self.request.output_compression,
            output_quality=self.request.output_quality,
        )
