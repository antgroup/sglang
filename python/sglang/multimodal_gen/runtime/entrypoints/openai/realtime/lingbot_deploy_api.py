import asyncio
import json
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
SUPPORTED_CONTROL_KEYS = {"w", "a", "s", "d", "i", "j", "k", "l"}


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

    def dispose(self):
        super().dispose()
        self.current_keys.clear()
        self.selected_image_id = None
        self.stop_requested = False
        self.started = False
        self.move_speed = DEFAULT_MOVE_SPEED
        self.rotate_speed_deg_ik = DEFAULT_ROTATE_SPEED_DEG_IK
        self.rotate_speed_deg_jl = DEFAULT_ROTATE_SPEED_DEG_JL

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

    def apply_camera_config(self, batch) -> None:
        batch.extra["move_speed"] = self.move_speed
        batch.extra["rotate_speed_deg_ik"] = self.rotate_speed_deg_ik
        batch.extra["rotate_speed_deg_jl"] = self.rotate_speed_deg_jl

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
    width, height = _parse_resolution(data)
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

    request = RealtimeVideoGenerationsRequest(
        prompt=str(data.get("prompt") or DEFAULT_PROMPT),
        first_frame=first_frame,
        width=width,
        height=height,
        size=f"{width}x{height}",
        fps=_parse_int_field(data, "fps", DEFAULT_FPS),
        num_frames=_parse_int_field(data, "num_frames", DEFAULT_NUM_FRAMES),
        seed=seed,
        generator_device=data.get("generator_device"),
        guidance_scale=data.get("guidance_scale"),
        guidance_scale_2=data.get("guidance_scale_2"),
        negative_prompt=data.get("negative_prompt"),
        num_inference_steps=data.get("num_inference_steps"),
        enable_teacache=data.get("enable_teacache"),
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


def output_to_rgb_bytes(output: Any) -> bytes:
    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise ValueError("empty model output")
        output = output[0]

    if isinstance(output, torch.Tensor):
        tensor = output.detach()
        if tensor.dtype != torch.uint8:
            tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)
        arr = tensor.cpu().numpy()
        return _normalize_rgb_array(arr, prefer_channel_first=True).tobytes()
    elif isinstance(output, np.ndarray):
        arr = output
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        raise TypeError(f"unsupported model output type: {type(output)}")

    return _normalize_rgb_array(arr).tobytes()


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

            await ws.send_bytes(output_to_rgb_bytes(result.output))

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
