import asyncio
import json
import os
import random
import time
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

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
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(tags=["lingbot-realtime"])

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
DEFAULT_DEBUG_SAVE_VIDEO = os.environ.get(
    "SGLANG_LINGBOT_DEBUG_SAVE_VIDEO", "0"
).lower() in {"1", "true", "yes", "on"}
DEFAULT_DEBUG_VIDEO_DIR = os.environ.get(
    "SGLANG_LINGBOT_DEBUG_VIDEO_DIR", "/tmp/sglang_lingbot_debug"
)
SUPPORTED_CONTROL_KEYS = {"w", "a", "s", "d", "i", "j", "k", "l"}


class LingBotDebugVideoRecorder:
    def __init__(self, path: str, fps: int):
        self.path = path
        self.fps = fps
        self.writer = None
        self.frame_count = 0

    def append(self, frames: list[np.ndarray]) -> None:
        if not frames:
            return

        import imageio

        output_dir = os.path.dirname(self.path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if self.writer is None:
            self.writer = imageio.get_writer(
                self.path,
                fps=self.fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
            logger.info("LingBot deploy debug video opened: %s", self.path)

        for frame in frames:
            self.writer.append_data(np.ascontiguousarray(frame))
            self.frame_count += 1

    def close(self) -> None:
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None
        logger.info(
            "LingBot deploy debug video saved: path=%s frames=%d",
            self.path,
            self.frame_count,
        )


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

    def dispose(self):
        self.close_debug_video()
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

    def apply_camera_config(self, batch) -> None:
        batch.extra["move_speed"] = self.move_speed
        batch.extra["rotate_speed_deg_ik"] = self.rotate_speed_deg_ik
        batch.extra["rotate_speed_deg_jl"] = self.rotate_speed_deg_jl

    def set_debug_video(self, path: str | None, fps: int) -> None:
        self.close_debug_video()
        self.debug_video_path = path
        if path:
            self.debug_video_recorder = LingBotDebugVideoRecorder(path, fps=fps)
        else:
            self.debug_video_recorder = None

    def append_debug_video(self, frames: list[np.ndarray]) -> None:
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

    def set_key_state(self, key: str, action: str) -> None:
        normalized_key = key.lower()
        if normalized_key not in SUPPORTED_CONTROL_KEYS:
            raise ValueError(f"unsupported control key: {key}")
        if action == "down":
            self.current_keys.add(normalized_key)
        elif action == "up":
            self.current_keys.discard(normalized_key)
        else:
            raise ValueError(f"unsupported control action: {action}")

        self._append_control_frame(sorted(self.current_keys))


def _json_message(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


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
    enable_upscaling = _parse_bool_field(
        data, "enable_upscaling", DEFAULT_ENABLE_UPSCALING
    )
    upscaling_scale = _parse_int_field(data, "upscaling_scale", DEFAULT_UPSCALING_SCALE)
    if upscaling_scale <= 0:
        raise ValueError("upscaling_scale must be positive")

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
    session.set_debug_video(debug_video_path if debug_save_video else None, fps=fps)

    request = RealtimeVideoGenerationsRequest(
        prompt=str(data.get("prompt") or DEFAULT_PROMPT),
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


def output_to_rgb_frames(output: Any) -> list[np.ndarray]:
    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise ValueError("empty model output")
        output = output[0]

    if isinstance(output, torch.Tensor):
        tensor = output.detach()
        if tensor.dtype != torch.uint8:
            tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)
        arr = tensor.cpu().numpy()
        return list(_normalize_rgb_array(arr, prefer_channel_first=True))
    elif isinstance(output, np.ndarray):
        arr = output
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        raise TypeError(f"unsupported model output type: {type(output)}")

    return list(_normalize_rgb_array(arr))


def rgb_frames_to_bytes(frames: list[np.ndarray]) -> bytes:
    return np.ascontiguousarray(np.stack(frames, axis=0)).tobytes()


def output_to_rgb_frames_for_send(
    output: Any,
    *,
    enable_upscaling: bool = False,
    upscaling_model_path: str | None = None,
    upscaling_scale: int = 4,
    half_precision: bool = False,
) -> bytes:
    frames = output_to_rgb_frames(output)
    input_shape = tuple(frames[0].shape) if frames else None
    if enable_upscaling:
        from sglang.multimodal_gen.runtime.postprocess import upscale_frames

        logger.info(
            "LingBot deploy SR enabled: frames=%d input_shape=%s model_path=%s scale=%s half_precision=%s",
            len(frames),
            input_shape,
            upscaling_model_path,
            upscaling_scale,
            half_precision,
        )
        frames = upscale_frames(
            frames,
            model_path=upscaling_model_path,
            scale=upscaling_scale,
            half_precision=half_precision,
        )
    else:
        logger.info(
            "LingBot deploy SR disabled: frames=%d input_shape=%s",
            len(frames),
            input_shape,
        )

    logger.info(
        "LingBot deploy RGB frames prepared: frames=%d output_shape=%s",
        len(frames),
        tuple(frames[0].shape) if frames else None,
    )
    return frames


def output_to_rgb_bytes(
    output: Any,
    *,
    enable_upscaling: bool = False,
    upscaling_model_path: str | None = None,
    upscaling_scale: int = 4,
    half_precision: bool = False,
) -> bytes:
    frames = output_to_rgb_frames_for_send(
        output,
        enable_upscaling=enable_upscaling,
        upscaling_model_path=upscaling_model_path,
        upscaling_scale=upscaling_scale,
        half_precision=half_precision,
    )
    payload = rgb_frames_to_bytes(frames)
    logger.info("LingBot deploy raw chunk prepared: bytes=%d", len(payload))
    return payload


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


async def _generate_loop(ws: WebSocket, session: LingBotDeployCompatSession) -> None:
    try:
        while not session.stop_requested:
            session.new_request()

            start = time.perf_counter()
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

            result = await async_scheduler_client.forward([batch])
            if result.output is None:
                error_msg = result.error or "Model generation returned no raw frames"
                raise RuntimeError(error_msg)

            frames = output_to_rgb_frames_for_send(
                result.output,
                enable_upscaling=bool(batch.enable_upscaling),
                upscaling_model_path=batch.upscaling_model_path,
                upscaling_scale=int(batch.upscaling_scale),
                half_precision=server_args.realesrgan_half_precision,
            )
            session.append_debug_video(frames)
            payload = rgb_frames_to_bytes(frames)
            logger.info("LingBot deploy raw chunk prepared: bytes=%d", len(payload))
            await ws.send_bytes(payload)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            await _send_json(
                ws,
                {
                    "type": "TIMING",
                    "chunk_idx": session.generate_chunk_cnt,
                    "chunk_time_ms": elapsed_ms,
                    "chunks_per_sec": 1000.0 / elapsed_ms if elapsed_ms > 0 else 0,
                    "queue_size": 0,
                },
            )

            session.generate_chunk_completed()

        await _send_json(ws, {"type": "FINISH"})
    except asyncio.CancelledError:
        raise
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
        session.started = False
        session.close_debug_video()
        await _release_realtime_session(session)


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
            request, seed = _build_start_request(data, session)
        except Exception as exc:
            await _send_error(ws, str(exc))
            return generate_task

        session.stop_requested = False
        session.set_mode(RealtimeVideoMode.V2V)
        session.setRequest(request)
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
        try:
            session.set_key_state(str(data.get("key", "")), str(data.get("action", "")))
        except Exception as exc:
            await _send_error(ws, str(exc))
        return generate_task

    if msg_type == "STOP":
        session.stop_requested = True
        if generate_task is None or generate_task.done():
            await _send_json(ws, {"type": "FINISH"})
        return generate_task

    await _send_error(ws, f"unsupported message type: {msg_type}")
    return generate_task


@router.websocket("/")
@router.websocket("/v1/lingbot/realtime")
async def generate(websocket: WebSocket):
    await websocket.accept()
    session = LingBotDeployCompatSession()
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
            await asyncio.gather(generate_task, return_exceptions=True)
        await _release_realtime_session(session)
        session.dispose()
