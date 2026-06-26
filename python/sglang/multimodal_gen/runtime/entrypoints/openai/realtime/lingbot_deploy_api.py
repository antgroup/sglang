import asyncio
import io
import json
import os
import queue
import random
import struct
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
    RealtimeVideoMode,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    ReleaseRealtimeSessionReq,
    prepare_request,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    init_logger,
    log_session_context,
)

logger = init_logger(__name__)
router = APIRouter(tags=["lingbot-realtime"])
_WEBUI_HTML_PATH = Path(__file__).with_name("lingbot_webui.html")


class _ControlEvent(NamedTuple):
    event_time: float
    key: str
    action: str
    actions: list[str]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_preview_transport(value: Any, *, default: str = "raw") -> str:
    aliases = {
        "": default,
        "jpeg": "ctk1",
        "jpg": "ctk1",
        "mjpeg": "ctk1",
        "h264": "ctv1",
        "h264_zerolatency": "ctv1",
        "h264-zero-latency": "ctv1",
    }
    normalized = str(value if value is not None else default).strip().lower()
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"raw", "ctk1", "ctv1", "auto"}:
        return default
    return normalized


DEFAULT_PROMPT = "A cinematic video with smooth camera motion"
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 16
DEFAULT_NUM_FRAMES = 117
DEFAULT_MOVE_SPEED = 0.05
DEFAULT_ROTATE_SPEED_DEG_IK = 4.0
DEFAULT_ROTATE_SPEED_DEG_JL = 6.0
DEFAULT_ENABLE_UPSCALING = True
DEFAULT_UPSCALING_SCALE = 2
DEFAULT_UPSCALING_MODEL_PATH = (
    "/home/admin/realesr-general-x4v3/realesr-general-x4v3.pth"
)
DEFAULT_ENABLE_RIFE = _env_bool("SGLANG_LINGBOT_ENABLE_RIFE", False)
DEFAULT_RIFE_MODEL_PATH = os.environ.get(
    "SGLANG_LINGBOT_RIFE_MODEL_PATH",
    "/home/admin/RIFE-4.22.lite",
)
DEFAULT_RIFE_EXP = int(os.environ.get("SGLANG_LINGBOT_RIFE_EXP", "1"))
DEFAULT_RIFE_SCALE = float(os.environ.get("SGLANG_LINGBOT_RIFE_SCALE", "1.0"))
DEFAULT_DEBUG_SAVE_VIDEO = os.environ.get(
    "SGLANG_LINGBOT_DEBUG_SAVE_VIDEO", "0"
).lower() in {"1", "true", "yes", "on"}
DEFAULT_DEBUG_VIDEO_DIR = os.environ.get(
    "SGLANG_LINGBOT_DEBUG_VIDEO_DIR", "/tmp/sglang_lingbot_debug"
)
DEFAULT_DEBUG_VIDEO_QUEUE_SIZE = max(
    0, int(os.environ.get("SGLANG_LINGBOT_DEBUG_VIDEO_QUEUE_SIZE", "0"))
)
DEFAULT_PINNED_D2H = _env_bool("SGLANG_LINGBOT_PINNED_D2H", False)
DEFAULT_SEND_MEMORYVIEW = _env_bool("SGLANG_LINGBOT_SEND_MEMORYVIEW", False)
DEFAULT_PREVIEW_TRANSPORT = _normalize_preview_transport(
    os.environ.get("SGLANG_LINGBOT_PREVIEW_TRANSPORT", "raw"),
    default="raw",
)
DEFAULT_PREVIEW_JPEG_QUALITY = max(
    1, min(100, int(os.environ.get("SGLANG_LINGBOT_PREVIEW_JPEG_QUALITY", "85")))
)
DEFAULT_PREVIEW_H264_QP = max(
    0, min(51, int(os.environ.get("SGLANG_LINGBOT_PREVIEW_H264_QP", "23")))
)
STARTUP_WARMUP_CHUNKS = int(os.environ.get("SGLANG_LINGBOT_STARTUP_WARMUP_CHUNKS", "0"))
STARTUP_WARMUP_IMAGE_PATH = os.environ.get("SGLANG_LINGBOT_STARTUP_WARMUP_IMAGE_PATH")
STARTUP_WARMUP_SIZES = os.environ.get("SGLANG_LINGBOT_STARTUP_WARMUP_SIZES")
STARTUP_WARMUP_WIDTH = int(
    os.environ.get(
        "SGLANG_LINGBOT_STARTUP_WARMUP_WIDTH",
        str(DEFAULT_WIDTH * DEFAULT_UPSCALING_SCALE),
    )
)
STARTUP_WARMUP_HEIGHT = int(
    os.environ.get(
        "SGLANG_LINGBOT_STARTUP_WARMUP_HEIGHT",
        str(DEFAULT_HEIGHT * DEFAULT_UPSCALING_SCALE),
    )
)
STARTUP_WARMUP_PROMPT = os.environ.get(
    "SGLANG_LINGBOT_STARTUP_WARMUP_PROMPT", DEFAULT_PROMPT
)
_CHUNK_PREVIEW_MAGIC_MAIN = b"CTK1"
_CHUNK_PREVIEW_MAGIC_VIDEO = b"CTV1"


class H264EncoderUnavailable(RuntimeError):
    """Raised when PyAV/libx264 cannot provide the low-latency preview path."""


def _annexb_has_idr(buf: bytes) -> bool:
    if not buf:
        return False
    n = len(buf)
    i = 0
    while i + 4 < n:
        if buf[i] == 0 and buf[i + 1] == 0 and buf[i + 2] == 1:
            if (buf[i + 3] & 0x1F) == 5:
                return True
            i += 4
        elif buf[i] == 0 and buf[i + 1] == 0 and buf[i + 2] == 0 and buf[i + 3] == 1:
            if (buf[i + 4] & 0x1F) == 5:
                return True
            i += 5
        else:
            i += 1
    return False


class H264ZerolatencyStreamEncoder:
    """Persistent libx264 zerolatency encoder for CTV1 preview packets."""

    def __init__(self, width: int, height: int, fps: int, qp: int):
        try:
            import av  # type: ignore
            from av.video.frame import PictureType  # type: ignore
        except ImportError as exc:
            raise H264EncoderUnavailable(
                "PyAV is not installed; falling back to CTK1 JPEG preview"
            ) from exc
        self._av = av
        self._picture_type = PictureType
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._qp = int(qp)
        self._frames_total = 0
        self._ctx = None
        self._open()

    def _open(self) -> None:
        from fractions import Fraction

        try:
            codec = self._av.codec.Codec("libx264", "w")
        except Exception as exc:
            raise H264EncoderUnavailable(
                f"libx264 is not available in this FFmpeg build: {exc}"
            ) from exc

        option_candidates = [
            {
                "crf": str(self._qp),
                "g": "1000",
                "annexb": "1",
                "preset": "ultrafast",
                "tune": "zerolatency",
                "bf": "0",
                "bframes": "0",
                "rc-lookahead": "0",
                "sync-lookahead": "0",
                "x264-params": "repeat-headers=1:forced-idr=1",
            },
            {
                "crf": str(self._qp),
                "g": "1000",
                "annexb": "1",
                "preset": "ultrafast",
                "tune": "zerolatency",
                "bf": "0",
                "bframes": "0",
                "rc-lookahead": "0",
                "sync-lookahead": "0",
            },
            {
                "crf": str(self._qp),
                "g": "1000",
                "annexb": "1",
                "preset": "ultrafast",
                "tune": "zerolatency",
                "bf": "0",
            },
        ]
        errors: list[str] = []
        for options in option_candidates:
            ctx = codec.create()
            ctx.width = self._width
            ctx.height = self._height
            ctx.pix_fmt = "yuv420p"
            ctx.framerate = Fraction(self._fps, 1)
            ctx.time_base = Fraction(1, self._fps)
            ctx.options = options
            try:
                ctx.open()
                self._ctx = ctx
                return
            except Exception as exc:
                errors.append(f"options={options}: {exc}")
        raise H264EncoderUnavailable(
            "libx264 zerolatency open failed: " + " | ".join(errors)
        )

    def close(self) -> None:
        try:
            if self._ctx is not None:
                self._ctx.close()
        except Exception:
            pass
        self._ctx = None
        self._frames_total = 0

    def encode_chunk(
        self,
        frames_hwc_uint8: list[np.ndarray],
        *,
        force_first_keyframe: bool = True,
    ) -> list[tuple[bytes, bool]]:
        if self._ctx is None:
            self._open()
        out: list[tuple[bytes, bool]] = []
        for i, frame in enumerate(frames_hwc_uint8):
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"expected HWC RGB uint8, got shape {frame.shape}")
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            av_frame = self._av.VideoFrame.from_ndarray(frame, format="rgb24")
            av_frame = av_frame.reformat(format="yuv420p")
            av_frame.pts = self._frames_total
            if i == 0 and force_first_keyframe:
                av_frame.pict_type = self._picture_type.I
            else:
                av_frame.pict_type = self._picture_type.NONE
            packets = self._ctx.encode(av_frame)
            buf = b"".join(bytes(packet) for packet in packets)
            out.append((buf, i == 0 or _annexb_has_idr(buf)))
            self._frames_total += 1
        return out


class LingBotDebugVideoRecorder:
    _STOP = object()

    def __init__(self, path: str, fps: int):
        self.path = path
        self.fps = fps
        self.writer = None
        self.frame_count = 0
        self.dropped_frames = 0
        self._closed = False
        self._last_drop_log_time = 0.0
        self._queue: queue.Queue[list[np.ndarray] | object] = queue.Queue(
            maxsize=DEFAULT_DEBUG_VIDEO_QUEUE_SIZE
        )
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"lingbot-debug-video-{os.path.basename(path) or 'writer'}",
            daemon=True,
        )
        self._worker.start()

    def append(self, frames: list[np.ndarray] | np.ndarray) -> None:
        if self._closed or len(frames) == 0:
            return

        if isinstance(frames, np.ndarray):
            frame_batch = [frames] if frames.ndim == 3 else list(frames)
        else:
            frame_batch = list(frames)
        if not frame_batch:
            return

        try:
            self._queue.put_nowait(frame_batch)
        except queue.Full:
            dropped = len(frame_batch)
            self.dropped_frames += dropped
            now = time.perf_counter()
            if now - self._last_drop_log_time >= 5.0:
                self._last_drop_log_time = now
                logger.warning(
                    "dropping LingBot deploy debug video frames: path=%s dropped=%d total_dropped=%d queue_size=%d",
                    self.path,
                    dropped,
                    self.dropped_frames,
                    self._queue.qsize(),
                )

    def _ensure_writer(self):
        if self.writer is not None:
            return

        import imageio

        output_dir = os.path.dirname(self.path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.writer = imageio.get_writer(
            self.path,
            fps=self.fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        logger.info("LingBot deploy debug video opened: %s", self.path)

    def _worker_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is self._STOP:
                        return
                    self._ensure_writer()
                    for frame in item:
                        self.writer.append_data(np.ascontiguousarray(frame))
                        self.frame_count += 1
                except Exception as e:
                    logger.warning(
                        "failed to write LingBot deploy debug video frames, path=%s, error=%s",
                        self.path,
                        e,
                        exc_info=True,
                    )
                finally:
                    self._queue.task_done()
        finally:
            if self.writer is not None:
                self.writer.close()
                self.writer = None
                logger.info(
                    "LingBot deploy debug video saved: path=%s frames=%d dropped_frames=%d",
                    self.path,
                    self.frame_count,
                    self.dropped_frames,
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._STOP)
        self._queue.join()
        self._worker.join()


class LingBotDeployCompatSession(GenerateSession):
    """Realtime session that speaks lingbot_fast_server's deploy WebSocket protocol."""

    def __init__(self):
        super().__init__()
        self.current_keys: set[str] = set()
        self.selected_image_id: str | None = None
        self.stop_requested = False
        self.started = False
        self.move_speed = DEFAULT_MOVE_SPEED
        self.rotate_speed_deg_ik = DEFAULT_ROTATE_SPEED_DEG_IK
        self.rotate_speed_deg_jl = DEFAULT_ROTATE_SPEED_DEG_JL
        self.debug_video_path: str | None = None
        self.debug_video_recorder: LingBotDebugVideoRecorder | None = None
        self.output_width = DEFAULT_WIDTH
        self.output_height = DEFAULT_HEIGHT
        self.model_width = DEFAULT_WIDTH
        self.model_height = DEFAULT_HEIGHT
        self.enable_rife = DEFAULT_ENABLE_RIFE
        self.rife_model_path = DEFAULT_RIFE_MODEL_PATH
        self.rife_exp = DEFAULT_RIFE_EXP
        self.rife_scale = DEFAULT_RIFE_SCALE
        self.preview_transport = DEFAULT_PREVIEW_TRANSPORT
        self.preview_jpeg_quality = DEFAULT_PREVIEW_JPEG_QUALITY
        self.preview_fps = DEFAULT_FPS
        self._h264_encoder: H264ZerolatencyStreamEncoder | None = None
        self._h264_encoder_shape: tuple[int, int, int] | None = None
        self._h264_fallback_logged = False
        self._control_events: deque[_ControlEvent] = deque(maxlen=512)
        self._control_sample_cursor: float | None = None
        self._control_sample_base_actions: list[str] = []
        self.reset_control_sampler()

    def dispose(self):
        self.close_debug_video()
        self.close_preview_encoder()
        super().dispose()
        self.current_keys.clear()
        self.selected_image_id = None
        self.stop_requested = False
        self.started = False
        self.move_speed = DEFAULT_MOVE_SPEED
        self.rotate_speed_deg_ik = DEFAULT_ROTATE_SPEED_DEG_IK
        self.rotate_speed_deg_jl = DEFAULT_ROTATE_SPEED_DEG_JL
        self.debug_video_path = None
        self.output_width = DEFAULT_WIDTH
        self.output_height = DEFAULT_HEIGHT
        self.model_width = DEFAULT_WIDTH
        self.model_height = DEFAULT_HEIGHT
        self.enable_rife = DEFAULT_ENABLE_RIFE
        self.rife_model_path = DEFAULT_RIFE_MODEL_PATH
        self.rife_exp = DEFAULT_RIFE_EXP
        self.rife_scale = DEFAULT_RIFE_SCALE
        self.preview_transport = DEFAULT_PREVIEW_TRANSPORT
        self.preview_jpeg_quality = DEFAULT_PREVIEW_JPEG_QUALITY
        self.preview_fps = DEFAULT_FPS
        self._h264_fallback_logged = False
        self.reset_control_sampler()

    def build_sampling_params(self):
        sampling_params = super().build_sampling_params()
        sampling_params.save_output = False
        sampling_params.return_file_paths_only = False
        return sampling_params

    def set_camera_config(
        self,
        *,
        move_speed: float,
        rotate_speed_deg_ik: float,
        rotate_speed_deg_jl: float,
    ) -> None:
        self.move_speed = move_speed
        self.rotate_speed_deg_ik = rotate_speed_deg_ik
        self.rotate_speed_deg_jl = rotate_speed_deg_jl

    def set_resolution_config(
        self,
        *,
        output_width: int,
        output_height: int,
        model_width: int,
        model_height: int,
    ) -> None:
        self.output_width = output_width
        self.output_height = output_height
        self.model_width = model_width
        self.model_height = model_height

    def set_rife_config(
        self,
        *,
        enable_rife: bool,
        rife_model_path: str | None,
        rife_exp: int,
        rife_scale: float,
    ) -> None:
        self.enable_rife = bool(enable_rife)
        self.rife_model_path = rife_model_path
        self.rife_exp = int(rife_exp)
        self.rife_scale = float(rife_scale)

    def set_preview_config(
        self,
        *,
        transport: str,
        jpeg_quality: int,
        fps: int,
    ) -> None:
        normalized = _normalize_preview_transport(
            transport, default=DEFAULT_PREVIEW_TRANSPORT
        )
        if normalized != self.preview_transport:
            self.close_preview_encoder()
            self._h264_fallback_logged = False
        self.preview_transport = normalized
        self.preview_jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self.preview_fps = max(1, int(fps))

    def close_preview_encoder(self) -> None:
        encoder = self._h264_encoder
        self._h264_encoder = None
        self._h264_encoder_shape = None
        if encoder is not None:
            encoder.close()

    def mark_h264_unavailable(self, reason: Exception) -> None:
        self.close_preview_encoder()
        if self._h264_fallback_logged:
            return
        self._h264_fallback_logged = True
        logger.warning(
            "LingBot deploy CTV1 preview unavailable; falling back to CTK1 JPEG: session_id=%s error=%s",
            self.id,
            reason,
        )

    def h264_encoder_for(
        self, *, width: int, height: int, fps: int
    ) -> H264ZerolatencyStreamEncoder:
        shape = (int(width), int(height), int(fps))
        if self._h264_encoder is None or self._h264_encoder_shape != shape:
            self.close_preview_encoder()
            self._h264_encoder = H264ZerolatencyStreamEncoder(
                width=width,
                height=height,
                fps=fps,
                qp=DEFAULT_PREVIEW_H264_QP,
            )
            self._h264_encoder_shape = shape
        return self._h264_encoder

    def apply_camera_config(self, batch) -> None:
        batch.extra["move_speed"] = self.move_speed
        batch.extra["rotate_speed_deg_ik"] = self.rotate_speed_deg_ik
        batch.extra["rotate_speed_deg_jl"] = self.rotate_speed_deg_jl

    def reset_control_sampler(self, *, timestamp: float | None = None) -> None:
        self.control_queue.clear()
        self._control_events.clear()
        self._control_sample_cursor = (
            time.perf_counter() if timestamp is None else timestamp
        )
        self._control_sample_base_actions = sorted(self.current_keys)
        self.last_control_actions = list(self._control_sample_base_actions)
        self.has_control_state = bool(self._control_sample_base_actions)

    def set_debug_video(self, path: str | None, fps: int) -> None:
        self.close_debug_video()
        self.debug_video_path = path
        if path:
            self.debug_video_recorder = LingBotDebugVideoRecorder(path, fps=fps)
        else:
            self.debug_video_recorder = None

    def append_debug_video(self, frames: list[np.ndarray] | np.ndarray) -> None:
        if self.debug_video_recorder is None:
            return
        try:
            self.debug_video_recorder.append(frames)
        except Exception as e:
            logger.warning(
                "failed to append LingBot deploy debug video, path=%s, error=%s",
                self.debug_video_path,
                e,
                exc_info=True,
            )

    def close_debug_video(self) -> None:
        if self.debug_video_recorder is None:
            return
        try:
            self.debug_video_recorder.close()
        except Exception as e:
            logger.warning(
                "failed to close LingBot deploy debug video, path=%s, error=%s",
                self.debug_video_path,
                e,
                exc_info=True,
            )
        finally:
            self.debug_video_recorder = None

    def set_key_state(
        self, key: str, action: str, *, timestamp: float | None = None
    ) -> None:
        normalized_key, action = self.validate_control_key_action(key, action)
        before = set(self.current_keys)
        if action == "down":
            self.current_keys.add(normalized_key)
        elif action == "up":
            self.current_keys.discard(normalized_key)

        if self.current_keys == before:
            return

        event_time = time.perf_counter() if timestamp is None else timestamp
        self._control_events.append(
            _ControlEvent(
                event_time=event_time,
                key=normalized_key,
                action=action,
                actions=sorted(self.current_keys),
            )
        )

    def _log_dropped_control_events(
        self,
        dropped_events: list[_ControlEvent],
        *,
        start: float,
        now: float,
        chunk_size: int,
    ) -> None:
        if not dropped_events:
            return

        max_events_to_log = 16
        dropped_summary: list[dict[str, Any]] = [
            {
                "key": event.key,
                "action": event.action,
                "offset_ms": round((event.event_time - start) * 1000.0, 3),
                "actions": event.actions,
            }
            for event in dropped_events[:max_events_to_log]
        ]
        if len(dropped_events) > max_events_to_log:
            dropped_summary.append(
                {"truncated_count": len(dropped_events) - max_events_to_log}
            )

        logger.info(
            "drop LingBot deploy control events during sampling, "
            "session_id=%s, chunk_size=%s, window_ms=%.3f, "
            "dropped_count=%s, dropped_events=%s",
            self.id,
            chunk_size,
            (now - start) * 1000.0,
            len(dropped_events),
            dropped_summary,
        )

    def sample_control_chunk(
        self, chunk_size: int, *, timestamp: float | None = None
    ) -> list[list[str]] | None:
        if chunk_size <= 0:
            return None

        now = time.perf_counter() if timestamp is None else timestamp
        start = self._control_sample_cursor
        if start is None:
            start = now

        events = [
            control_event
            for control_event in self._control_events
            if control_event.event_time <= now
        ]
        base_actions = list(self._control_sample_base_actions)
        state = list(base_actions)
        chunk: list[list[str]] = []
        duration = max(0.0, now - start)
        dropped_events: list[_ControlEvent] = []

        if duration == 0.0:
            chunk = [list(state) for _ in range(chunk_size)]
            dropped_events.extend(events)
        else:
            event_idx = 0
            for frame_idx in range(chunk_size):
                sample_time = start + duration * (frame_idx + 1) / chunk_size
                sampled_events: list[_ControlEvent] = []
                while (
                    event_idx < len(events)
                    and events[event_idx].event_time <= sample_time
                ):
                    sampled_events.append(events[event_idx])
                    state = list(events[event_idx].actions)
                    event_idx += 1
                dropped_events.extend(sampled_events[:-1])
                chunk.append(list(state))

        for event in events:
            state = list(event.actions)
        self._control_sample_cursor = now
        self._control_sample_base_actions = list(state)
        self.last_control_actions = list(self._control_sample_base_actions)
        if events:
            self.has_control_state = True
        self._log_dropped_control_events(
            dropped_events,
            start=start,
            now=now,
            chunk_size=chunk_size,
        )
        while self._control_events and self._control_events[0].event_time <= now:
            self._control_events.popleft()

        return chunk


def _json_message(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _format_prompt_for_log(prompt: Any) -> str:
    prompt_text = str(prompt)
    if _env_bool("SGLANG_LOG_FULL_PROMPT", False):
        return repr(prompt_text)
    return f"<redacted, len={len(prompt_text)}>"


def _log_chunk_movement_prompt(
    session: LingBotDeployCompatSession,
    batch,
    *,
    chunk_size: int,
) -> None:
    logger.info(
        "LingBot chunk movement prompt: session_id=%s block_idx=%s "
        "chunk_size=%s mode=%s prompt=%s",
        session.id,
        batch.block_idx,
        chunk_size,
        batch.extra.get("movement_prompt_mode"),
        _format_prompt_for_log(batch.prompt),
    )


async def _send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_text(_json_message(payload))


async def _send_error(ws: WebSocket, message: str) -> None:
    await _send_json(ws, {"type": "ERROR", "message": message})


def _parse_int_field(data: dict[str, Any], name: str, default: int) -> int:
    value = data.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _parse_float_field(data: dict[str, Any], name: str, default: float) -> float:
    value = data.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


def _parse_bool_field(data: dict[str, Any], name: str, default: bool) -> bool:
    value = data.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, int):
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


def _parse_optional_str_alias(data: dict[str, Any], *names: str) -> str | None:
    for name in names:
        if name in data:
            value = data.get(name)
            return None if value is None else str(value)
    return None


def _resolve_enable_upscaling(data: dict[str, Any], width: int, height: int) -> bool:
    if "enable_upscaling" in data:
        return _parse_bool_field(data, "enable_upscaling", DEFAULT_ENABLE_UPSCALING)
    return width * height > DEFAULT_WIDTH * DEFAULT_HEIGHT


def _parse_resolution(data: dict[str, Any]) -> tuple[int, int]:
    width = data.get("width")
    height = data.get("height")
    size = data.get("size")

    if (width is None or height is None) and isinstance(size, str):
        try:
            w, h = size.lower().replace(" ", "").split("x", 1)
            width = width if width is not None else int(w)
            height = height if height is not None else int(h)
        except Exception as exc:
            raise ValueError("size must use WxH format") from exc

    resolved_width = int(width if width is not None else DEFAULT_WIDTH)
    resolved_height = int(height if height is not None else DEFAULT_HEIGHT)
    if resolved_width <= 0 or resolved_height <= 0:
        raise ValueError("width and height must be positive")
    return resolved_width, resolved_height


def _build_start_request(
    data: dict[str, Any], session: LingBotDeployCompatSession
) -> tuple[RealtimeVideoGenerationsRequest, int]:
    output_width, output_height = _parse_resolution(data)
    seed = data.get("seed")
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    else:
        seed = int(seed)

    move_speed = _parse_float_field(data, "move_speed", DEFAULT_MOVE_SPEED)
    rotate_speed_deg_ik = _parse_float_field(
        data, "rotate_speed_deg_ik", DEFAULT_ROTATE_SPEED_DEG_IK
    )
    rotate_speed_deg_jl = _parse_float_field(
        data, "rotate_speed_deg_jl", DEFAULT_ROTATE_SPEED_DEG_JL
    )
    session.set_camera_config(
        move_speed=move_speed,
        rotate_speed_deg_ik=rotate_speed_deg_ik,
        rotate_speed_deg_jl=rotate_speed_deg_jl,
    )

    first_frame = data.get("image_path")
    session.selected_image_id = None
    if not first_frame:
        raise ValueError("START requires image_path")
    if not isinstance(first_frame, str):
        raise ValueError("image_path must be a string")
    if not first_frame.startswith(("http://", "https://")) and not os.path.isfile(
        first_frame
    ):
        raise ValueError(f"image_path is not a valid file: {first_frame}")

    fps = _parse_int_field(data, "fps", DEFAULT_FPS)
    preview_transport = _normalize_preview_transport(
        data.get("preview_transport", DEFAULT_PREVIEW_TRANSPORT),
        default=DEFAULT_PREVIEW_TRANSPORT,
    )
    preview_jpeg_quality = _parse_int_field(
        data, "preview_jpeg_quality", DEFAULT_PREVIEW_JPEG_QUALITY
    )
    session.set_preview_config(
        transport=preview_transport,
        jpeg_quality=preview_jpeg_quality,
        fps=fps,
    )
    upscaling_scale = _parse_int_field(data, "upscaling_scale", DEFAULT_UPSCALING_SCALE)
    enable_upscaling = _resolve_enable_upscaling(data, output_width, output_height)
    if upscaling_scale <= 0:
        raise ValueError("upscaling_scale must be positive")

    enable_rife = _parse_bool_field(data, "enable_rife", DEFAULT_ENABLE_RIFE)
    rife_exp = _parse_int_field(data, "rife_exp", DEFAULT_RIFE_EXP)
    rife_scale = _parse_float_field(data, "rife_scale", DEFAULT_RIFE_SCALE)
    rife_model_path = data.get("rife_model_path", DEFAULT_RIFE_MODEL_PATH)
    if rife_exp < 0:
        raise ValueError("rife_exp must be non-negative")
    if rife_scale <= 0:
        raise ValueError("rife_scale must be positive")
    if rife_model_path is not None and not isinstance(rife_model_path, str):
        raise ValueError("rife_model_path must be a string")
    session.set_rife_config(
        enable_rife=enable_rife,
        rife_model_path=rife_model_path,
        rife_exp=rife_exp,
        rife_scale=rife_scale,
    )

    model_width = output_width
    model_height = output_height
    if enable_upscaling:
        if output_width % upscaling_scale != 0 or output_height % upscaling_scale != 0:
            raise ValueError(
                "width and height must be divisible by upscaling_scale "
                "when enable_upscaling is true"
            )
        model_width = output_width // upscaling_scale
        model_height = output_height // upscaling_scale

    session.set_resolution_config(
        output_width=output_width,
        output_height=output_height,
        model_width=model_width,
        model_height=model_height,
    )
    logger.info(
        "LingBot START resolution: output=%sx%s model=%sx%s enable_upscaling=%s scale=%s",
        output_width,
        output_height,
        model_width,
        model_height,
        enable_upscaling,
        upscaling_scale,
    )

    debug_video_path = data.get("debug_video_path")
    debug_save_video = _parse_bool_field(
        data,
        "debug_save_video",
        DEFAULT_DEBUG_SAVE_VIDEO or debug_video_path is not None,
    )
    if debug_save_video and not debug_video_path:
        debug_video_dir = str(data.get("debug_video_dir") or DEFAULT_DEBUG_VIDEO_DIR)
        debug_video_path = os.path.join(
            debug_video_dir,
            f"lingbot_{session.id}_{int(time.time() * 1000)}.mp4",
        )
    if debug_video_path is not None and not isinstance(debug_video_path, str):
        raise ValueError("debug_video_path must be a string")
    debug_video_fps = fps * (2**rife_exp if enable_rife else 1)
    session.set_debug_video(
        debug_video_path if debug_save_video else None,
        fps=debug_video_fps,
    )

    prompt = str(data.get("prompt") or DEFAULT_PROMPT)
    logger.info(
        "LingBot START request: session_id=%s prompt=%r image_path=%s "
        "seed=%s output=%sx%s model=%sx%s rife=%s rife_exp=%s rife_scale=%s "
        "preview_transport=%s preview_jpeg_quality=%s",
        session.id,
        prompt,
        first_frame,
        seed,
        output_width,
        output_height,
        model_width,
        model_height,
        session.enable_rife,
        session.rife_exp,
        session.rife_scale,
        session.preview_transport,
        session.preview_jpeg_quality,
    )

    request = RealtimeVideoGenerationsRequest(
        stream_id=session.id if data.get("stream_id") is not None else None,
        prompt=prompt,
        first_frame=first_frame,
        width=model_width,
        height=model_height,
        size=f"{model_width}x{model_height}",
        fps=fps,
        num_frames=_parse_int_field(data, "num_frames", DEFAULT_NUM_FRAMES),
        seed=seed,
        generator_device=data.get("generator_device"),
        guidance_scale=data.get("guidance_scale"),
        guidance_scale_2=data.get("guidance_scale_2"),
        negative_prompt=data.get("negative_prompt"),
        num_inference_steps=data.get("num_inference_steps"),
        enable_teacache=data.get("enable_teacache"),
        enable_upscaling=enable_upscaling,
        upscaling_model_path=data.get(
            "upscaling_model_path", DEFAULT_UPSCALING_MODEL_PATH
        ),
        upscaling_scale=upscaling_scale,
        diffusers_kwargs=data.get("diffusers_kwargs"),
        events=data.get("events"),
        event_mode=data.get("event_mode"),
        event_chunk=data.get("event_chunk"),
        movement_static=_parse_optional_str_alias(
            data, "movement static", "movement_static"
        ),
        movement_dynamic=_parse_optional_str_alias(
            data, "movement dynamic", "movement_dynamic"
        ),
    )
    return request, seed


def _normalize_rgb_array(
    arr: np.ndarray, *, prefer_channel_first: bool = False
) -> np.ndarray:
    if arr.ndim == 5:
        arr = arr[0]

    if arr.ndim == 4:
        if prefer_channel_first and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr[:3], (1, 2, 3, 0))
        elif arr.shape[-1] in (1, 3, 4):
            arr = arr[..., :3]
        elif arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr[:3], (1, 2, 3, 0))
        elif arr.shape[1] in (1, 3, 4):
            arr = np.transpose(arr[:, :3], (0, 2, 3, 1))
        else:
            raise ValueError(f"unsupported video tensor shape: {arr.shape}")
    elif arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):
            arr = arr[None, ..., :3]
        elif arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr[:3], (1, 2, 0))[None, ...]
        else:
            raise ValueError(f"unsupported image tensor shape: {arr.shape}")
    else:
        raise ValueError(f"unsupported output tensor shape: {arr.shape}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"expected RGB frames, got shape: {arr.shape}")
    return np.ascontiguousarray(arr)


def _normalize_rgb_tensor(
    tensor: torch.Tensor, *, prefer_channel_first: bool = False
) -> torch.Tensor:
    if tensor.ndim == 5:
        tensor = tensor[0]

    if tensor.ndim == 4:
        if (
            prefer_channel_first
            and tensor.shape[0] in (1, 3, 4)
            and tensor.shape[1] not in (1, 3, 4)
        ):
            tensor = tensor[:3].permute(1, 0, 2, 3)
        elif tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3].permute(0, 3, 1, 2)
        elif tensor.shape[1] in (1, 3, 4):
            tensor = tensor[:, :3]
        elif tensor.shape[0] in (1, 3, 4):
            tensor = tensor[:3].permute(1, 0, 2, 3)
        else:
            raise ValueError(f"unsupported video tensor shape: {tuple(tensor.shape)}")
    elif tensor.ndim == 3:
        if tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3].permute(2, 0, 1).unsqueeze(0)
        elif tensor.shape[0] in (1, 3, 4):
            tensor = tensor[:3].unsqueeze(0)
        else:
            raise ValueError(f"unsupported image tensor shape: {tuple(tensor.shape)}")
    else:
        raise ValueError(f"unsupported output tensor shape: {tuple(tensor.shape)}")

    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    if tensor.shape[1] != 3:
        raise ValueError(f"expected RGB frames, got shape: {tuple(tensor.shape)}")
    return tensor.contiguous()


def output_to_rgb_tensor(output: Any) -> torch.Tensor:
    """Normalize model output to an NCHW RGB tensor on the existing device."""
    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise ValueError("empty model output")
        output = output[0]

    if isinstance(output, torch.Tensor):
        tensor = _normalize_rgb_tensor(output.detach(), prefer_channel_first=True)
    elif isinstance(output, np.ndarray):
        tensor = _normalize_rgb_tensor(torch.from_numpy(output))
    else:
        raise TypeError(f"unsupported model output type: {type(output)}")

    if tensor.dtype == torch.uint8:
        return tensor
    if not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.clamp(0.0, 1.0).contiguous()


def rgb_tensor_to_uint8_array(frames: torch.Tensor) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(
            f"expected NCHW RGB tensor with shape [N, 3, H, W], got {tuple(frames.shape)}"
        )
    if frames.dtype == torch.uint8:
        out = frames
    else:
        out = frames.clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
    out = out.permute(0, 2, 3, 1).contiguous()
    if DEFAULT_PINNED_D2H and out.device.type == "cuda":
        cpu_out = torch.empty(
            out.shape,
            dtype=out.dtype,
            device="cpu",
            pin_memory=True,
        )
        cpu_out.copy_(out, non_blocking=True)
        torch.cuda.current_stream(out.device).synchronize()
        return cpu_out.numpy()
    return out.cpu().numpy()


def rgb_tensor_hwc_shape(frames: torch.Tensor) -> tuple[int, int, int] | None:
    if frames.numel() == 0:
        return None
    return (int(frames.shape[-2]), int(frames.shape[-1]), int(frames.shape[1]))


def _cuda_device_from_output(output: Any) -> torch.device | None:
    if isinstance(output, torch.Tensor) and output.device.type == "cuda":
        return output.device
    if isinstance(output, (list, tuple)) and output:
        return _cuda_device_from_output(output[0])
    return None


def _cuda_device_from_tensor(tensor: torch.Tensor) -> torch.device | None:
    if tensor.device.type == "cuda":
        return tensor.device
    return None


def _start_cuda_stage_timer(
    device: torch.device | None,
) -> tuple[float, Any, Any, torch.device | None]:
    wall_start = time.perf_counter()
    if device is None or not torch.cuda.is_available():
        return wall_start, None, None, None

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(torch.cuda.current_stream(device))
    return wall_start, start_event, end_event, device


def _stop_cuda_stage_timer(
    timings: dict[str, float] | None,
    stage_name: str,
    timer: tuple[float, Any, Any, torch.device | None],
    cuda_events: dict[str, tuple[Any, Any]] | None,
    *,
    wall_timing_key: str | None = None,
    fallback_timing_key: str | None = None,
) -> None:
    if timings is None:
        return

    wall_start, start_event, end_event, device = timer
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    timing_key = f"{stage_name}_ms"
    if (
        start_event is None
        or end_event is None
        or device is None
        or cuda_events is None
    ):
        timings[fallback_timing_key or timing_key] = wall_ms
        return

    end_event.record(torch.cuda.current_stream(device))
    timings[wall_timing_key or f"{stage_name}_enqueue_ms"] = wall_ms
    cuda_events[stage_name] = (start_event, end_event)


def _resolve_cuda_stage_timings(
    timings: dict[str, float] | None,
    cuda_events: dict[str, tuple[Any, Any]] | None,
) -> None:
    if timings is None or not cuda_events:
        return

    for stage_name, (start_event, end_event) in cuda_events.items():
        end_event.synchronize()
        timings[f"{stage_name}_ms"] = start_event.elapsed_time(end_event)
    cuda_events.clear()


def output_to_rgb_frames(output: Any) -> list[np.ndarray]:
    if isinstance(output, torch.Tensor):
        return list(rgb_tensor_to_uint8_array(output_to_rgb_tensor(output)))
    if (
        isinstance(output, (list, tuple))
        and output
        and isinstance(output[0], torch.Tensor)
    ):
        return list(rgb_tensor_to_uint8_array(output_to_rgb_tensor(output)))

    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise ValueError("empty model output")
        output = output[0]
    if not isinstance(output, np.ndarray):
        raise TypeError(f"unsupported model output type: {type(output)}")

    arr = output
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return list(_normalize_rgb_array(arr))


def rgb_frames_to_bytes(frames: list[np.ndarray] | np.ndarray) -> bytes | memoryview:
    if isinstance(frames, np.ndarray):
        arr = np.ascontiguousarray(frames)
    else:
        arr = np.ascontiguousarray(np.stack(frames, axis=0))
    if DEFAULT_SEND_MEMORYVIEW:
        return memoryview(arr).cast("B")
    return arr.tobytes()


def _payload_nbytes(payload: bytes | bytearray | memoryview) -> int:
    if isinstance(payload, memoryview):
        return payload.nbytes
    return len(payload)


class _PreviewPayload(NamedTuple):
    payload: bytes | memoryview
    transport: str
    payload_bytes: int


def _frames_as_hwc_uint8_array(frames: list[np.ndarray] | np.ndarray) -> np.ndarray:
    if isinstance(frames, np.ndarray):
        arr = frames
    else:
        arr = np.stack(frames, axis=0)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return _normalize_rgb_array(arr)


def _encode_jpeg_frame(frame: np.ndarray, *, quality: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame), "RGB").save(
        buf,
        format="JPEG",
        quality=max(1, min(100, int(quality))),
    )
    return buf.getvalue()


def _build_chunk_jpeg_packet(
    *,
    eval_idx: int,
    chunk_id: int,
    chunk_index: int | None,
    chunk_total: int | None,
    chunk_progress: str | None,
    jpeg_by_frame: list[tuple[int, bytes]],
) -> bytes:
    version = 1
    flags = 0
    ci = int(chunk_index) if chunk_index is not None else 0xFFFFFFFF
    ct = int(chunk_total) if chunk_total is not None else 0xFFFFFFFF
    prog_bytes = (chunk_progress or "").encode("utf-8")
    if len(prog_bytes) > 65535:
        prog_bytes = prog_bytes[:65535]
    nframes = len(jpeg_by_frame)
    if nframes > 65535:
        raise ValueError("CTK1 preview chunk has too many frames")
    header = struct.pack(
        "<4sBBiiIIHH",
        _CHUNK_PREVIEW_MAGIC_MAIN,
        version,
        flags,
        int(eval_idx),
        int(chunk_id),
        ci,
        ct,
        nframes,
        len(prog_bytes),
    )
    parts: list[bytes] = [header, prog_bytes]
    for frame_id, raw in jpeg_by_frame:
        if len(raw) > 0xFFFFFFFF:
            raise ValueError("CTK1 preview frame is too large")
        parts.append(struct.pack("<HI", int(frame_id), len(raw)))
        parts.append(raw)
    return b"".join(parts)


def _build_chunk_h264_packet(
    *,
    eval_idx: int,
    chunk_id: int,
    chunk_index: int | None,
    chunk_total: int | None,
    chunk_progress: str | None,
    frames: list[tuple[int, bytes, bool]],
) -> bytes:
    version = 1
    flags = 0
    ci = int(chunk_index) if chunk_index is not None else 0xFFFFFFFF
    ct = int(chunk_total) if chunk_total is not None else 0xFFFFFFFF
    prog_bytes = (chunk_progress or "").encode("utf-8")
    if len(prog_bytes) > 65535:
        prog_bytes = prog_bytes[:65535]
    nframes = len(frames)
    if nframes > 65535:
        raise ValueError("CTV1 preview chunk has too many frames")
    header = struct.pack(
        "<4sBBiiIIHH",
        _CHUNK_PREVIEW_MAGIC_VIDEO,
        version,
        flags,
        int(eval_idx),
        int(chunk_id),
        ci,
        ct,
        nframes,
        len(prog_bytes),
    )
    parts: list[bytes] = [header, prog_bytes]
    for frame_id, nal, is_key in frames:
        if len(nal) > 0xFFFFFFFF:
            raise ValueError("CTV1 preview frame is too large")
        parts.append(struct.pack("<HIB3x", int(frame_id), len(nal), 1 if is_key else 0))
        parts.append(nal)
    return b"".join(parts)


def _pack_ctk1_preview_payload(
    frame_array: np.ndarray,
    *,
    chunk_idx: int,
    jpeg_quality: int,
) -> bytes:
    jpeg_by_frame = [
        (frame_id, _encode_jpeg_frame(frame, quality=jpeg_quality))
        for frame_id, frame in enumerate(frame_array)
    ]
    return _build_chunk_jpeg_packet(
        eval_idx=0,
        chunk_id=chunk_idx,
        chunk_index=None,
        chunk_total=None,
        chunk_progress=None,
        jpeg_by_frame=jpeg_by_frame,
    )


def _pack_ctv1_preview_payload(
    session: LingBotDeployCompatSession,
    frame_array: np.ndarray,
    *,
    chunk_idx: int,
) -> bytes:
    if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
        raise H264EncoderUnavailable(
            f"unsupported CTV1 frame shape {frame_array.shape}"
        )
    height = int(frame_array.shape[1])
    width = int(frame_array.shape[2])
    if width % 2 != 0 or height % 2 != 0:
        raise H264EncoderUnavailable(
            f"CTV1 requires even dimensions, got {width}x{height}"
        )
    encoder = session.h264_encoder_for(
        width=width,
        height=height,
        fps=session.preview_fps,
    )
    encoded = encoder.encode_chunk(
        [np.ascontiguousarray(frame) for frame in frame_array],
        force_first_keyframe=True,
    )
    if any(len(nal) == 0 for nal, _is_key in encoded):
        raise H264EncoderUnavailable("libx264 returned delayed empty packets")
    h264_by_frame = [
        (frame_id, nal, is_key) for frame_id, (nal, is_key) in enumerate(encoded)
    ]
    return _build_chunk_h264_packet(
        eval_idx=0,
        chunk_id=chunk_idx,
        chunk_index=None,
        chunk_total=None,
        chunk_progress=None,
        frames=h264_by_frame,
    )


def _build_preview_payload(
    session: LingBotDeployCompatSession,
    frames: list[np.ndarray] | np.ndarray,
    *,
    chunk_idx: int,
) -> _PreviewPayload:
    frame_array = _frames_as_hwc_uint8_array(frames)
    requested_transport = session.preview_transport

    if requested_transport in {"auto", "ctv1"}:
        try:
            payload = _pack_ctv1_preview_payload(
                session,
                frame_array,
                chunk_idx=chunk_idx,
            )
            return _PreviewPayload(
                payload=payload,
                transport="ctv1",
                payload_bytes=len(payload),
            )
        except H264EncoderUnavailable as exc:
            session.mark_h264_unavailable(exc)
        except Exception as exc:
            session.mark_h264_unavailable(exc)
            logger.warning(
                "LingBot deploy CTV1 preview encode failed; falling back to CTK1 JPEG: session_id=%s chunk_idx=%s",
                session.id,
                chunk_idx,
                exc_info=True,
            )

    if requested_transport in {"auto", "ctv1", "ctk1"}:
        try:
            payload = _pack_ctk1_preview_payload(
                frame_array,
                chunk_idx=chunk_idx,
                jpeg_quality=session.preview_jpeg_quality,
            )
            return _PreviewPayload(
                payload=payload,
                transport="ctk1",
                payload_bytes=len(payload),
            )
        except Exception:
            logger.warning(
                "LingBot deploy CTK1 preview encode failed; falling back to raw RGB: session_id=%s chunk_idx=%s",
                session.id,
                chunk_idx,
                exc_info=True,
            )

    payload = rgb_frames_to_bytes(frame_array)
    return _PreviewPayload(
        payload=payload,
        transport="raw",
        payload_bytes=_payload_nbytes(payload),
    )


def output_to_rgb_tensor_for_send(
    output: Any,
    *,
    enable_upscaling: bool = False,
    upscaling_model_path: str | None = None,
    upscaling_scale: int = 4,
    half_precision: bool = False,
    enable_rife: bool = DEFAULT_ENABLE_RIFE,
    rife_model_path: str | None = DEFAULT_RIFE_MODEL_PATH,
    rife_exp: int = DEFAULT_RIFE_EXP,
    rife_scale: float = DEFAULT_RIFE_SCALE,
    timings: dict[str, float] | None = None,
    cuda_events: dict[str, tuple[Any, Any]] | None = None,
) -> torch.Tensor:
    stage_timer = _start_cuda_stage_timer(_cuda_device_from_output(output))
    frames = output_to_rgb_tensor(output)
    _stop_cuda_stage_timer(
        timings,
        "rgb_convert",
        stage_timer,
        cuda_events,
        fallback_timing_key="rgb_convert_ms",
    )

    input_shape = rgb_tensor_hwc_shape(frames)
    if enable_upscaling:
        from sglang.multimodal_gen.runtime.postprocess import upscale_tensor

        logger.info(
            "LingBot deploy SR enabled: frames=%d input_shape=%s model_path=%s scale=%s half_precision=%s",
            frames.shape[0],
            input_shape,
            upscaling_model_path,
            upscaling_scale,
            half_precision,
        )
        stage_timer = _start_cuda_stage_timer(_cuda_device_from_tensor(frames))
        frames = upscale_tensor(
            frames,
            model_path=upscaling_model_path,
            scale=upscaling_scale,
            half_precision=half_precision,
        )
        _stop_cuda_stage_timer(
            timings,
            "sr",
            stage_timer,
            cuda_events,
            fallback_timing_key="sr_ms",
        )
    else:
        if timings is not None:
            timings["sr_ms"] = 0.0
        logger.info(
            "LingBot deploy SR disabled: frames=%d input_shape=%s",
            frames.shape[0],
            input_shape,
        )

    if enable_rife:
        from sglang.multimodal_gen.runtime.postprocess import (
            interpolate_video_tensor,
        )

        rife_input_shape = rgb_tensor_hwc_shape(frames)
        logger.info(
            "LingBot deploy RIFE enabled after SR: frames=%d input_shape=%s model_path=%s exp=%s scale=%s",
            frames.shape[0],
            rife_input_shape,
            rife_model_path,
            rife_exp,
            rife_scale,
        )
        stage_timer = _start_cuda_stage_timer(_cuda_device_from_tensor(frames))
        frames, multiplier = interpolate_video_tensor(
            frames,
            exp=rife_exp,
            scale=rife_scale,
            model_path=rife_model_path,
        )
        _stop_cuda_stage_timer(
            timings,
            "rife",
            stage_timer,
            cuda_events,
            fallback_timing_key="rife_ms",
        )
        if timings is not None:
            timings["rife_multiplier"] = float(multiplier)
    else:
        if timings is not None:
            timings["rife_ms"] = 0.0
            timings["rife_multiplier"] = 1.0
        logger.info("LingBot deploy RIFE disabled: frames=%d", frames.shape[0])

    logger.info(
        "LingBot deploy RGB frames prepared: frames=%d output_shape=%s",
        frames.shape[0],
        rgb_tensor_hwc_shape(frames),
    )
    return frames


def output_to_rgb_array_for_send(
    output: Any,
    *,
    enable_upscaling: bool = False,
    upscaling_model_path: str | None = None,
    upscaling_scale: int = 4,
    half_precision: bool = False,
    enable_rife: bool = DEFAULT_ENABLE_RIFE,
    rife_model_path: str | None = DEFAULT_RIFE_MODEL_PATH,
    rife_exp: int = DEFAULT_RIFE_EXP,
    rife_scale: float = DEFAULT_RIFE_SCALE,
    timings: dict[str, float] | None = None,
) -> np.ndarray:
    cuda_events: dict[str, tuple[Any, Any]] | None = {} if timings is not None else None
    frames = output_to_rgb_tensor_for_send(
        output,
        enable_upscaling=enable_upscaling,
        upscaling_model_path=upscaling_model_path,
        upscaling_scale=upscaling_scale,
        half_precision=half_precision,
        enable_rife=enable_rife,
        rife_model_path=rife_model_path,
        rife_exp=rife_exp,
        rife_scale=rife_scale,
        timings=timings,
        cuda_events=cuda_events,
    )
    stage_timer = _start_cuda_stage_timer(_cuda_device_from_tensor(frames))
    frame_array = rgb_tensor_to_uint8_array(frames)
    _stop_cuda_stage_timer(
        timings,
        "finalize_cuda",
        stage_timer,
        cuda_events,
        wall_timing_key="finalize_ms",
        fallback_timing_key="finalize_ms",
    )
    _resolve_cuda_stage_timings(timings, cuda_events)
    return frame_array


def output_to_rgb_frames_for_send(
    output: Any,
    *,
    enable_upscaling: bool = False,
    upscaling_model_path: str | None = None,
    upscaling_scale: int = 4,
    half_precision: bool = False,
    enable_rife: bool = DEFAULT_ENABLE_RIFE,
    rife_model_path: str | None = DEFAULT_RIFE_MODEL_PATH,
    rife_exp: int = DEFAULT_RIFE_EXP,
    rife_scale: float = DEFAULT_RIFE_SCALE,
    timings: dict[str, float] | None = None,
) -> list[np.ndarray]:
    return list(
        output_to_rgb_array_for_send(
            output,
            enable_upscaling=enable_upscaling,
            upscaling_model_path=upscaling_model_path,
            upscaling_scale=upscaling_scale,
            half_precision=half_precision,
            enable_rife=enable_rife,
            rife_model_path=rife_model_path,
            rife_exp=rife_exp,
            rife_scale=rife_scale,
            timings=timings,
        )
    )


def output_to_rgb_bytes(
    output: Any,
    *,
    enable_upscaling: bool = False,
    upscaling_model_path: str | None = None,
    upscaling_scale: int = 4,
    half_precision: bool = False,
) -> bytes | memoryview:
    frames = output_to_rgb_array_for_send(
        output,
        enable_upscaling=enable_upscaling,
        upscaling_model_path=upscaling_model_path,
        upscaling_scale=upscaling_scale,
        half_precision=half_precision,
        enable_rife=DEFAULT_ENABLE_RIFE,
        rife_model_path=DEFAULT_RIFE_MODEL_PATH,
        rife_exp=DEFAULT_RIFE_EXP,
        rife_scale=DEFAULT_RIFE_SCALE,
    )
    payload = rgb_frames_to_bytes(frames)
    logger.info("LingBot deploy raw chunk prepared: bytes=%d", _payload_nbytes(payload))
    return payload


def _ensure_startup_warmup_image(width: int, height: int) -> str:
    if STARTUP_WARMUP_IMAGE_PATH:
        if not os.path.isfile(STARTUP_WARMUP_IMAGE_PATH):
            raise FileNotFoundError(
                f"SGLANG_LINGBOT_STARTUP_WARMUP_IMAGE_PATH does not exist: "
                f"{STARTUP_WARMUP_IMAGE_PATH}"
            )
        return STARTUP_WARMUP_IMAGE_PATH

    from PIL import Image

    path = os.path.join(
        tempfile.gettempdir(), f"sglang_lingbot_startup_warmup_{width}x{height}.png"
    )
    if not os.path.isfile(path):
        Image.new("RGB", (max(1, width), max(1, height)), (24, 24, 24)).save(path)
    return path


def _parse_startup_warmup_sizes() -> list[tuple[int, int]]:
    if not STARTUP_WARMUP_SIZES:
        return [(STARTUP_WARMUP_WIDTH, STARTUP_WARMUP_HEIGHT)]

    sizes: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_size in STARTUP_WARMUP_SIZES.replace(";", ",").split(","):
        token = raw_size.strip().lower()
        if not token:
            continue
        try:
            width_text, height_text = token.split("x", 1)
            width = int(width_text)
            height = int(height_text)
        except ValueError as exc:
            raise ValueError(
                "SGLANG_LINGBOT_STARTUP_WARMUP_SIZES must use comma-separated "
                "WIDTHxHEIGHT entries, for example: 1664x960,960x1664"
            ) from exc
        if width <= 0 or height <= 0:
            raise ValueError(
                "SGLANG_LINGBOT_STARTUP_WARMUP_SIZES entries must be positive "
                f"WIDTHxHEIGHT values, got: {token}"
            )
        size = (width, height)
        if size in seen:
            continue
        seen.add(size)
        sizes.append(size)

    if not sizes:
        raise ValueError(
            "SGLANG_LINGBOT_STARTUP_WARMUP_SIZES did not contain any valid "
            "WIDTHxHEIGHT entries"
        )
    return sizes


async def run_startup_warmup_if_enabled(server_args) -> None:
    if STARTUP_WARMUP_CHUNKS <= 0:
        return

    warmup_sizes = _parse_startup_warmup_sizes()
    start_time = time.perf_counter()

    for size_idx, (warmup_width, warmup_height) in enumerate(warmup_sizes, start=1):
        session = LingBotDeployCompatSession()
        session_log_context = log_session_context(session.id)
        session_log_context.__enter__()
        try:
            warmup_image = _ensure_startup_warmup_image(
                warmup_width,
                warmup_height,
            )
            request, seed = _build_start_request(
                {
                    "type": "START",
                    "prompt": STARTUP_WARMUP_PROMPT,
                    "image_path": warmup_image,
                    "width": warmup_width,
                    "height": warmup_height,
                    "seed": 0,
                    "debug_save_video": False,
                },
                session,
            )
            session.stop_requested = False
            session.set_mode(RealtimeVideoMode.V2V)
            session.setRequest(request)
            session.current_keys.clear()
            session.reset_control_sampler()
            session.started = True

            logger.info(
                "LingBot startup warmup begin: size=%d/%d chunks=%d "
                "output=%sx%s model=%sx%s image_path=%s seed=%s",
                size_idx,
                len(warmup_sizes),
                STARTUP_WARMUP_CHUNKS,
                session.output_width,
                session.output_height,
                session.model_width,
                session.model_height,
                warmup_image,
                seed,
            )

            for chunk_idx in range(STARTUP_WARMUP_CHUNKS):
                chunk_start = time.perf_counter()
                session.new_request()
                sampling_params = session.build_sampling_params()
                sampling_params.suppress_logs = True
                batch = prepare_request(
                    server_args=server_args,
                    sampling_params=sampling_params,
                )
                batch.session = session.realtime_session
                batch.extra["realtime_session_id"] = session.id
                session.apply_camera_config(batch)
                batch.block_idx = session.generate_chunk_cnt
                chunk_size = batch.extra.get("chunk_size", 1)
                batch.extra["actions"] = [[] for _ in range(chunk_size)]
                session.apply_movement_prompt_to_batch(batch)
                _log_chunk_movement_prompt(session, batch, chunk_size=chunk_size)
                session.apply_prompt_event_to_batch(batch)

                result = await async_scheduler_client.forward([batch])
                if result.output is None:
                    error_msg = (
                        result.error or "LingBot startup warmup returned no output"
                    )
                    raise RuntimeError(error_msg)

                frames = output_to_rgb_array_for_send(
                    result.output,
                    enable_upscaling=bool(batch.enable_upscaling),
                    upscaling_model_path=batch.upscaling_model_path,
                    upscaling_scale=int(batch.upscaling_scale),
                    half_precision=server_args.realesrgan_half_precision,
                )
                logger.info(
                    "LingBot startup warmup size %d/%d chunk %d/%d done: "
                    "%.2fms frames=%d shape=%s",
                    size_idx,
                    len(warmup_sizes),
                    chunk_idx + 1,
                    STARTUP_WARMUP_CHUNKS,
                    (time.perf_counter() - chunk_start) * 1000.0,
                    len(frames),
                    tuple(frames[0].shape) if len(frames) else None,
                )
                session.generate_chunk_completed()
        finally:
            session.started = False
            session.current_keys.clear()
            session.reset_control_sampler()
            session.close_debug_video()
            await _release_realtime_session(session)
            session.dispose()
            session_log_context.__exit__(None, None, None)

    logger.info(
        "LingBot startup warmup complete: sizes=%d chunks_per_size=%d "
        "total_chunks=%d elapsed=%.2fs",
        len(warmup_sizes),
        STARTUP_WARMUP_CHUNKS,
        STARTUP_WARMUP_CHUNKS * len(warmup_sizes),
        time.perf_counter() - start_time,
    )


async def _release_realtime_session(session: LingBotDeployCompatSession) -> None:
    try:
        await async_scheduler_client.forward(
            ReleaseRealtimeSessionReq(session_id=session.id)
        )
    except Exception as e:
        logger.warning(
            "failed to release LingBot realtime session, session_id=%s, error=%s",
            session.id,
            e,
        )


def _raise_if_sender_failed(sender_task: asyncio.Task) -> None:
    if not sender_task.done():
        return
    if sender_task.cancelled():
        raise asyncio.CancelledError()
    exc = sender_task.exception()
    if exc is not None:
        raise exc
    raise RuntimeError("LingBot deploy sender task exited unexpectedly")


async def _await_cancelled_task(task: asyncio.Task) -> None:
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _enqueue_send_item(
    send_queue: asyncio.Queue[dict[str, Any]],
    sender_task: asyncio.Task,
    item: dict[str, Any],
) -> None:
    put_task = asyncio.create_task(send_queue.put(item))
    done, pending = await asyncio.wait(
        {put_task, sender_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    if sender_task in done:
        for task in pending:
            task.cancel()
            await _await_cancelled_task(task)
        _raise_if_sender_failed(sender_task)
    await put_task


async def _join_send_queue(
    send_queue: asyncio.Queue[dict[str, Any]],
    sender_task: asyncio.Task,
) -> None:
    join_task = asyncio.create_task(send_queue.join())
    done, pending = await asyncio.wait(
        {join_task, sender_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    if sender_task in done:
        for task in pending:
            task.cancel()
            await _await_cancelled_task(task)
        _raise_if_sender_failed(sender_task)
    await join_task


async def _send_chunk_item(
    ws: WebSocket,
    session: LingBotDeployCompatSession,
    item: dict[str, Any],
) -> None:
    frames = item["frames"]
    timings = item["timings"]
    chunk_idx = item["chunk_idx"]
    start = item["start"]
    queue_size = item["queue_size"]
    frame_count = len(frames)
    frame_shape = tuple(frames[0].shape) if frame_count else None

    stage_start = time.perf_counter()
    preview_payload = await asyncio.to_thread(
        _build_preview_payload,
        session,
        frames,
        chunk_idx=chunk_idx,
    )
    item["frames"] = None
    del frames
    timings["pack_bytes_ms"] = (time.perf_counter() - stage_start) * 1000.0
    payload = preview_payload.payload
    payload_bytes = preview_payload.payload_bytes
    timings["preview_transport"] = preview_payload.transport
    logger.info(
        "LingBot deploy preview chunk prepared: transport=%s bytes=%d",
        preview_payload.transport,
        payload_bytes,
    )

    stage_start = time.perf_counter()
    await ws.send_bytes(payload)
    timings["ws_send_ms"] = (time.perf_counter() - stage_start) * 1000.0

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    timings["total_ms"] = elapsed_ms
    logger.info(
        "LingBot deploy stage timing: session_id=%s chunk_idx=%s "
        "prepare=%.2fms scheduler=%.2fms rgb_convert=%.2fms sr=%.2fms rife=%.2fms "
        "finalize=%.2fms finalize_cuda=%.2fms postprocess=%.2fms "
        "debug_video=%.2fms send_queue_wait=%.2fms pack_bytes=%.2fms "
        "ws_send=%.2fms produce=%.2fms total=%.2fms frames=%d frame_shape=%s "
        "payload_bytes=%d preview_transport=%s rife_multiplier=%.1f",
        session.id,
        chunk_idx,
        timings["prepare_ms"],
        timings["scheduler_ms"],
        timings["rgb_convert_ms"],
        timings["sr_ms"],
        timings["rife_ms"],
        timings["finalize_ms"],
        timings.get("finalize_cuda_ms", 0.0),
        timings["postprocess_ms"],
        timings["debug_video_ms"],
        timings.get("send_queue_wait_ms", 0.0),
        timings["pack_bytes_ms"],
        timings["ws_send_ms"],
        timings["produce_ms"],
        elapsed_ms,
        frame_count,
        frame_shape,
        payload_bytes,
        preview_payload.transport,
        timings["rife_multiplier"],
    )
    await _send_json(
        ws,
        {
            "type": "TIMING",
            "chunk_idx": chunk_idx,
            "chunk_time_ms": elapsed_ms,
            "chunks_per_sec": 1000.0 / elapsed_ms if elapsed_ms > 0 else 0,
            "produce_time_ms": timings["produce_ms"],
            "producer_chunks_per_sec": (
                1000.0 / timings["produce_ms"] if timings["produce_ms"] > 0 else 0
            ),
            "queue_size": queue_size,
            "frames": frame_count,
            "frame_shape": list(frame_shape) if frame_shape is not None else None,
            "preview_transport": preview_payload.transport,
            "payload_bytes": payload_bytes,
            "rife_multiplier": timings["rife_multiplier"],
            "stage_timings_ms": {
                key: round(value, 3)
                for key, value in timings.items()
                if isinstance(value, (int, float))
            },
        },
    )


async def _send_chunk_loop(
    ws: WebSocket,
    session: LingBotDeployCompatSession,
    send_queue: asyncio.Queue[dict[str, Any]],
) -> None:
    while True:
        item = await send_queue.get()
        try:
            await _send_chunk_item(ws, session, item)
        finally:
            send_queue.task_done()


async def _generate_loop(ws: WebSocket, session: LingBotDeployCompatSession) -> None:
    session_log_context = log_session_context(session.id)
    session_log_context.__enter__()
    send_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
    sender_task = asyncio.create_task(_send_chunk_loop(ws, session, send_queue))
    try:
        while not session.stop_requested:
            _raise_if_sender_failed(sender_task)
            session.new_request()

            start = time.perf_counter()
            timings: dict[str, float] = {}
            stage_start = time.perf_counter()
            server_args = get_global_server_args()
            sampling_params = session.build_sampling_params()
            batch = prepare_request(
                server_args=server_args,
                sampling_params=sampling_params,
            )
            batch.session = session.realtime_session
            batch.extra["realtime_session_id"] = session.id
            session.apply_camera_config(batch)
            batch.block_idx = session.generate_chunk_cnt
            timings["prepare_ms"] = (time.perf_counter() - stage_start) * 1000.0

            chunk_size = batch.extra.get("chunk_size", 1)
            control_chunk = session.sample_control_chunk(chunk_size)
            if control_chunk is not None:
                logger.info(
                    "consume LingBot deploy control, session_id=%s, block_idx=%s, chunk_size=%s, control_chunk=%s",
                    session.id,
                    batch.block_idx,
                    chunk_size,
                    control_chunk,
                )
                batch.extra["actions"] = control_chunk

            session.apply_movement_prompt_to_batch(batch)
            _log_chunk_movement_prompt(session, batch, chunk_size=chunk_size)
            session.apply_prompt_event_to_batch(batch)

            stage_start = time.perf_counter()
            result = await async_scheduler_client.forward([batch])
            timings["scheduler_ms"] = (time.perf_counter() - stage_start) * 1000.0
            if result.output is None:
                error_msg = result.error or "Model generation returned no raw frames"
                raise RuntimeError(error_msg)

            stage_start = time.perf_counter()
            frames = output_to_rgb_array_for_send(
                result.output,
                enable_upscaling=bool(batch.enable_upscaling),
                upscaling_model_path=batch.upscaling_model_path,
                upscaling_scale=int(batch.upscaling_scale),
                half_precision=server_args.realesrgan_half_precision,
                enable_rife=session.enable_rife,
                rife_model_path=session.rife_model_path,
                rife_exp=session.rife_exp,
                rife_scale=session.rife_scale,
                timings=timings,
            )
            timings["postprocess_ms"] = (time.perf_counter() - stage_start) * 1000.0
            stage_start = time.perf_counter()
            session.append_debug_video(frames)
            timings["debug_video_ms"] = (time.perf_counter() - stage_start) * 1000.0

            timings["produce_ms"] = (time.perf_counter() - start) * 1000.0
            chunk_idx = session.generate_chunk_cnt
            queue_wait_start = time.perf_counter()
            await _enqueue_send_item(
                send_queue,
                sender_task,
                {
                    "chunk_idx": chunk_idx,
                    "frames": frames,
                    "timings": timings,
                    "start": start,
                    "queue_size": send_queue.qsize(),
                },
            )
            timings["send_queue_wait_ms"] = (
                time.perf_counter() - queue_wait_start
            ) * 1000.0

            session.generate_chunk_completed()

        await _join_send_queue(send_queue, sender_task)
        _raise_if_sender_failed(sender_task)
        await _send_json(ws, {"type": "FINISH"})
    except asyncio.CancelledError:
        raise
    except WebSocketDisconnect:
        logger.info(
            "LingBot deploy client disconnected during generation, session_id=%s",
            session.id,
        )
    except Exception as e:
        err_msg = str(e).splitlines()[0]
        logger.error(
            "error during LingBot deploy generate loop, session_id=%s, error=%s",
            session.id,
            err_msg,
            exc_info=True,
        )
        try:
            await _send_error(ws, err_msg)
        except Exception:
            pass
    finally:
        if not sender_task.done():
            sender_task.cancel()
        await asyncio.gather(sender_task, return_exceptions=True)
        session.started = False
        session.current_keys.clear()
        session.reset_control_sampler()
        session.close_debug_video()
        await _release_realtime_session(session)
        session_log_context.__exit__(None, None, None)


async def _handle_message(
    ws: WebSocket,
    session: LingBotDeployCompatSession,
    generate_task: asyncio.Task | None,
    raw_message: str,
) -> asyncio.Task | None:
    try:
        data = json.loads(raw_message)
    except json.JSONDecodeError as exc:
        await _send_error(ws, f"invalid JSON message: {exc.msg}")
        return generate_task

    if not isinstance(data, dict):
        await _send_error(ws, "message must be a JSON object")
        return generate_task

    msg_type = str(data.get("type", "")).upper()

    if msg_type == "IMAGE_SELECT":
        image_id = data.get("image_id")
        if not image_id:
            await _send_error(ws, "IMAGE_SELECT requires image_id")
            return generate_task
        session.selected_image_id = str(image_id)
        await _send_json(ws, {"type": "IMAGE_SELECTED", "image_id": image_id})
        return generate_task

    if msg_type == "START":
        if generate_task is not None and generate_task.done():
            generate_task = None
            session.dispose()
        if generate_task is not None:
            await _send_error(ws, "session already started")
            return generate_task

        try:
            session.set_stream_id(data.get("stream_id"))
            request, seed = _build_start_request(data, session)
        except Exception as exc:
            await _send_error(ws, str(exc))
            return generate_task

        session.stop_requested = False
        session.set_mode(RealtimeVideoMode.V2V)
        session.setRequest(request)
        session.current_keys.clear()
        session.reset_control_sampler()
        session.started = True
        await _send_json(
            ws,
            {
                "type": "STARTED",
                "prompt": request.prompt[:100],
                "seed": seed,
                "move_speed": session.move_speed,
                "rotate_speed_deg_ik": session.rotate_speed_deg_ik,
                "rotate_speed_deg_jl": session.rotate_speed_deg_jl,
                "enable_upscaling": request.enable_upscaling,
                "upscaling_scale": request.upscaling_scale,
                "upscaling_model_path": request.upscaling_model_path,
                "enable_rife": session.enable_rife,
                "rife_model_path": session.rife_model_path,
                "rife_exp": session.rife_exp,
                "rife_scale": session.rife_scale,
                "preview_transport": session.preview_transport,
                "preview_jpeg_quality": session.preview_jpeg_quality,
                "width": session.output_width,
                "height": session.output_height,
                "output_width": session.output_width,
                "output_height": session.output_height,
                "model_width": session.model_width,
                "model_height": session.model_height,
                "debug_video_path": session.debug_video_path,
            },
        )
        return asyncio.create_task(_generate_loop(ws, session))

    if msg_type == "CONTROL":
        key = data.get("key")
        action = data.get("action")
        event_ids = session.resolve_event_ids(
            event_id=data.get("event_id"),
            event=data.get("event"),
            event_ids=data.get("event_ids"),
            events=data.get("events"),
        )
        if event_ids:
            try:
                event_ids = session.validate_prompt_event_ids(event_ids)
            except Exception as exc:
                await _send_error(ws, str(exc))
                return generate_task
        if key is not None or action is not None:
            try:
                key, action = session.validate_control_key_action(
                    str(key or ""), str(action or "")
                )
            except Exception as exc:
                await _send_error(ws, str(exc))
                return generate_task
        did_handle = False

        if key is not None or action is not None:
            try:
                session.set_key_state(key, action)
                did_handle = True
            except Exception as exc:
                await _send_error(ws, str(exc))
                return generate_task

        if event_ids:
            try:
                session.trigger_prompt_events(event_ids)
                did_handle = True
            except Exception as exc:
                await _send_error(ws, str(exc))
                return generate_task

        if not did_handle:
            await _send_error(ws, "CONTROL requires key/action or event(s)")
            return generate_task

        logger.info(
            "LingBot CONTROL request: session_id=%s key=%s action=%s "
            "event_ids=%s current_keys=%s started=%s",
            session.id,
            key,
            action,
            event_ids,
            sorted(session.current_keys),
            session.started,
        )
        return generate_task

    if msg_type == "STOP":
        session.stop_requested = True
        if generate_task is None or generate_task.done():
            await _send_json(ws, {"type": "FINISH"})
        return generate_task

    await _send_error(ws, f"unsupported message type: {msg_type}")
    return generate_task


@router.get("/webui", response_class=HTMLResponse)
@router.get("/v1/lingbot/realtime/webui", response_class=HTMLResponse)
async def lingbot_realtime_webui():
    return HTMLResponse(_WEBUI_HTML_PATH.read_text(encoding="utf-8"))


@router.websocket("/")
@router.websocket("/v1/lingbot/realtime")
async def generate(websocket: WebSocket):
    await websocket.accept()
    session = LingBotDeployCompatSession()
    session_log_context = log_session_context(session.id)
    session_log_context.__enter__()
    generate_task = None
    try:
        async for message in websocket.iter_text():
            generate_task = await _handle_message(
                websocket, session, generate_task, message
            )
    except WebSocketDisconnect:
        logger.info("LingBot deploy client disconnected, session_id=%s", session.id)
    finally:
        session.stop_requested = True
        if generate_task is not None and not generate_task.done():
            generate_task.cancel()
        if generate_task is not None:
            await asyncio.gather(generate_task, return_exceptions=True)
        await _release_realtime_session(session)
        session.dispose()
        session_log_context.__exit__(None, None, None)
