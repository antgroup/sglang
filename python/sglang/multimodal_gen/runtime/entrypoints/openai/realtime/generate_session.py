# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeAction,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.realtime.session import (
    RealtimeSession,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
        BaseRealtimeModelAdapter,
    )

logger = init_logger(__name__)


class RealtimeVideoMode(str, Enum):
    """Legacy realtime mode retained for pre-mainline clients."""

    T2V = "t2v"
    V2V = "v2v"


SUPPORTED_CONTROL_KEYS = {"w", "a", "s", "d", "i", "j", "k", "l"}


@dataclass(frozen=True, slots=True)
class RealtimeChunkContext:
    session_id: str
    index: int
    request_id: str


class GenerateSession:
    """A realtime generation session"""

    _FIRST_BLOCK_ENCODE_FRAMES = 9
    _NEXT_BLOCK_ENCODE_FRAMES = 12

    def __init__(self):
        self.id = uuid4().hex
        self.request_id: str | None = None
        self.request: RealtimeVideoGenerationsRequest | None = None
        self.mode: RealtimeVideoMode | None = None
        self.action_queue: deque[RealtimeAction] = deque(maxlen=1)
        self.control_queue: deque[list[str]] = deque(maxlen=512)
        self.video_frame_queue: deque[Any] = deque(maxlen=256)
        self.current_keys: set[str] = set()
        self.last_control_actions: list[str] = []
        self.has_control_state = False
        self.prompt_events: dict[str, str] = {}
        self.prompt_event_order: list[str] = []
        self.prompt_event_mode: str = "overwrite"
        self.prompt_event_chunk: int = 1
        self.active_prompt_events: OrderedDict[str, int] = OrderedDict()
        self._last_effective_prompt: str | tuple[str, ...] | None = None
        self.legacy_protocol = False
        self.input_temp_dir: str | None = None
        self.generate_chunk_cnt = 0
        self.current_chunk: RealtimeChunkContext | None = None
        self.realtime_session = RealtimeSession()
        self.adapter: BaseRealtimeModelAdapter | None = None
        self.adapter_state: Any = None
        self.output_pace_next_send_at: float | None = None
        self.output_pace_last_event_id: int | None = None

    def set_adapter(self, adapter: BaseRealtimeModelAdapter):
        self.adapter = adapter
        self.adapter_state = adapter.create_state()

    def set_request(self, request: RealtimeVideoGenerationsRequest):
        self.request = request
        self.configure_prompt_events(request)

    def setRequest(self, request: RealtimeVideoGenerationsRequest):
        """Compatibility alias retained for existing realtime clients."""
        self.set_request(request)

    def set_stream_id(self, stream_id: Any | None):
        if stream_id is None:
            return
        if self.request_id is not None or self.generate_chunk_cnt != 0:
            raise ValueError("stream_id must be set before generation starts")
        if isinstance(stream_id, bytes):
            stream_id = stream_id.decode("utf-8")
        stream_id = str(stream_id).strip()
        if not stream_id:
            raise ValueError("stream_id cannot be empty")
        if any(ch in stream_id for ch in ("/", "\\", "\x00")):
            raise ValueError("stream_id cannot contain path separators")
        self.id = stream_id

    def set_mode(self, mode: RealtimeVideoMode | None):
        self.mode = mode

    def dispose(self):
        if self.adapter is not None:
            self.adapter.dispose(self)
        self.action_queue.clear()
        self.control_queue.clear()
        self.video_frame_queue.clear()
        self.mode = None
        self.request = None
        self.request_id = None
        self.current_keys.clear()
        self.last_control_actions = []
        self.has_control_state = False
        self.prompt_events = {}
        self.prompt_event_order = []
        self.prompt_event_mode = "overwrite"
        self.prompt_event_chunk = 1
        self.active_prompt_events.clear()
        self._last_effective_prompt = None
        self.legacy_protocol = False
        self.input_temp_dir = None
        self.generate_chunk_cnt = 0
        self.current_chunk = None
        self.adapter = None
        self.adapter_state = None
        self.output_pace_next_send_at = None
        self.output_pace_last_event_id = None
        self.realtime_session.dispose()

    def new_request(self):
        """Create the request id used by the legacy generation loop."""
        self.request_id = f"{self.id}_{uuid4().hex}"

    def new_chunk(self) -> RealtimeChunkContext:
        if self.current_chunk is not None:
            raise RuntimeError("previous realtime chunk is still active")
        chunk = RealtimeChunkContext(
            session_id=self.id,
            index=self.generate_chunk_cnt,
            request_id=f"{self.id}_{uuid4().hex}",
        )
        self.current_chunk = chunk
        return chunk

    def generate_chunk_completed(self):
        self.generate_chunk_cnt += 1
        self.current_chunk = None

    def reached_max_chunks(self) -> bool:
        return (
            self.request is not None
            and self.request.max_chunks is not None
            and self.generate_chunk_cnt >= self.request.max_chunks
        )

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
        else:
            self.current_keys.discard(normalized_key)
        if self.current_keys != before:
            self._append_control_frame(sorted(self.current_keys))

    def append_video_frames(self, frames: list[Any]):
        if frames:
            self.video_frame_queue.extend(frames)

    def has_pending_video_frames(self) -> bool:
        return len(self.video_frame_queue) >= self.required_video_frames()

    def is_v2v_enabled(self) -> bool:
        if self.request is None:
            return False
        if self.mode is not None:
            return self.mode == RealtimeVideoMode.V2V
        return self.request.first_frame is not None

    def required_video_frames(self) -> int:
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
            if event_id:
                events[event_id] = str(value)

        event_mode = getattr(request, "event_mode", None) or "overwrite"
        if event_mode not in ("overwrite", "append"):
            raise ValueError("event_mode must be 'overwrite' or 'append'")
        event_chunk = getattr(request, "event_chunk", None)
        event_chunk = 1 if event_chunk is None else int(event_chunk)
        if event_chunk <= 0:
            raise ValueError("event_chunk must be a positive integer")

        self.prompt_events = events
        self.prompt_event_order = list(events)
        self.prompt_event_mode = event_mode
        self.prompt_event_chunk = event_chunk
        self.active_prompt_events.clear()

    @staticmethod
    def _normalize_event_ids(raw_event_ids: Any) -> list[str]:
        if raw_event_ids is None:
            return []
        values = (
            raw_event_ids
            if isinstance(raw_event_ids, (list, tuple, set))
            else [raw_event_ids]
        )
        return [
            event_id
            for value in values
            if value is not None and (event_id := str(value).strip())
        ]

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
                if normalized_event_id not in seen_event_ids:
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

    def resolve_action_event_id(self, action: RealtimeAction) -> str | None:
        event_ids = self.resolve_action_event_ids(action)
        return event_ids[0] if event_ids else None

    def validate_prompt_event_ids(self, event_ids: list[str]) -> list[str]:
        if not event_ids:
            raise ValueError("event id cannot be empty")
        normalized_event_ids: list[str] = []
        seen_event_ids: set[str] = set()
        for event_id in event_ids:
            event_id = str(event_id).strip()
            if not event_id:
                raise ValueError("event id cannot be empty")
            if event_id not in self.prompt_events:
                raise ValueError(f"unknown prompt event: {event_id}")
            if event_id not in seen_event_ids:
                normalized_event_ids.append(event_id)
                seen_event_ids.add(event_id)
        return normalized_event_ids

    def trigger_prompt_event(self, event_id: str) -> None:
        self.trigger_prompt_events([event_id])

    def trigger_prompt_events(self, event_ids: list[str]) -> None:
        for event_id in self.validate_prompt_event_ids(event_ids):
            if event_id in self.active_prompt_events:
                del self.active_prompt_events[event_id]
            self.active_prompt_events[event_id] = self.prompt_event_chunk

    @staticmethod
    def _append_prompt_suffix(prompt: str, suffix: Any) -> str:
        if suffix is None:
            return prompt
        suffix = str(suffix)
        if not suffix:
            return prompt
        return f"{prompt}{suffix}" if prompt else suffix

    @staticmethod
    def _actions_have_movement(actions: Any) -> bool:
        return bool(actions) and any(bool(frame_actions) for frame_actions in actions)

    def build_movement_prompt(self, actions: Any | None = None) -> str:
        if self.request is None:
            return ""
        prompt = str(self.request.prompt)
        suffix = (
            getattr(self.request, "movement_dynamic", None)
            if self._actions_have_movement(actions)
            else getattr(self.request, "movement_static", None)
        )
        return self._append_prompt_suffix(prompt, suffix)

    def get_movement_prompt_mode(self, actions: Any | None = None) -> str:
        return "dynamic" if self._actions_have_movement(actions) else "static"

    def build_movement_prompt_variants(self) -> list[str]:
        prompts: list[str] = []
        for actions in ([], [["__movement__"]]):
            prompt = self.build_movement_prompt(actions)
            if prompt not in prompts:
                prompts.append(prompt)
        return prompts

    def apply_movement_prompt_to_batch(self, batch) -> None:
        if self.request is None or "actions" not in batch.extra:
            return
        actions = batch.extra.get("actions")
        batch.prompt = self.build_movement_prompt(actions)
        batch.extra["movement_prompt_mode"] = self.get_movement_prompt_mode(actions)
        batch.extra["movement_prompt_variants"] = self.build_movement_prompt_variants()

    @staticmethod
    def _prompt_cache_value(prompt: Any) -> str | tuple[str, ...]:
        if isinstance(prompt, list):
            return tuple(str(item) for item in prompt)
        return str(prompt)

    def _mark_effective_prompt(self, batch, active_event_ids: list[str]) -> None:
        effective_prompt = self._prompt_cache_value(batch.prompt)
        prompt_updated = self._last_effective_prompt != effective_prompt
        self._last_effective_prompt = effective_prompt
        batch.update_prompt_embeds = prompt_updated
        condition_inputs = getattr(batch, "condition_inputs", None)
        if prompt_updated and isinstance(condition_inputs, dict):
            condition_inputs["lingbot_prompt_updated"] = True
        if os.environ.get("SGLANG_LOG_FULL_PROMPT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            logger.info(
                "LingBot effective prompt: session_id=%s block_idx=%s "
                "event_mode=%s event_ids=%s prompt=%r",
                getattr(batch, "realtime_session_id", None)
                or batch.extra.get("realtime_session_id"),
                getattr(batch, "block_idx", None),
                self.prompt_event_mode,
                active_event_ids,
                batch.prompt,
            )

    def apply_prompt_event_to_batch(self, batch) -> None:
        if not self.prompt_events:
            return

        batch.extra["prompt_events"] = self.prompt_events
        batch.extra["prompt_event_mode"] = self.prompt_event_mode
        batch.extra["prompt_event_chunk"] = self.prompt_event_chunk

        active_event_ids = [
            event_id
            for event_id in self.prompt_event_order
            if self.active_prompt_events.get(event_id, 0) > 0
        ]
        if not active_event_ids:
            self.active_prompt_events.clear()
            return

        batch.extra["prompt_event_ids"] = active_event_ids
        if len(active_event_ids) == 1:
            batch.extra["prompt_event_id"] = active_event_ids[0]

        for event_id in active_event_ids:
            remaining_chunks = self.active_prompt_events[event_id] - 1
            if remaining_chunks <= 0:
                self.active_prompt_events.pop(event_id, None)
            else:
                self.active_prompt_events[event_id] = remaining_chunks

    def sample_control_chunk(self, chunk_size: int) -> list[list[str]] | None:
        if chunk_size <= 0:
            return None
        chunk: list[list[str]] = []
        while len(chunk) < chunk_size and self.control_queue:
            chunk.append(list(self.control_queue.popleft()))
        if not chunk and not self.has_control_state:
            return [[] for _ in range(chunk_size)]
        pad_actions = list(self.last_control_actions)
        while len(chunk) < chunk_size:
            chunk.append(list(pad_actions))
        return chunk

    def sample_video_frames(self):
        required = self.required_video_frames()
        if len(self.video_frame_queue) < required:
            return None
        pending_frames = list(self.video_frame_queue)
        self.video_frame_queue.clear()
        if len(pending_frames) == required:
            return pending_frames
        indices = np.round(np.linspace(0, len(pending_frames) - 1, required)).astype(
            int
        )
        return [pending_frames[index] for index in indices]

    def build_sampling_params(self):
        if self.request is None or self.request_id is None:
            raise RuntimeError("realtime request has not been initialized")
        if self.generate_chunk_cnt > 0 and self.action_queue:
            realtime_action = self.sample_action()
            if realtime_action.type == "prompt":
                self.request.prompt = realtime_action.action_content or ""

        return build_sampling_params(
            self.request_id,
            prompt=self.request.prompt,
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
