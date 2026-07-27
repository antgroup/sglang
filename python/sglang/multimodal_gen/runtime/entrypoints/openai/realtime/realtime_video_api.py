# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import math
import shutil
import time
from typing import TYPE_CHECKING

import msgspec.msgpack
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeAction,
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
    RealtimeChunkContext,
    RealtimeVideoMode,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_output_adapter import (
    RealtimeFrameSendStats,
    empty_frame_send_stats,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    get_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.timer import (
    RealtimeStageTimer,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    ReleaseRealtimeSessionReq,
    save_outputs,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/realtime_video", tags=["realtime"])
_ACTIVE_SESSION_IDS: set[str] = set()
_ACTIVE_SESSION_WAIT_SECONDS = 1.0
_ACTIVE_SESSION_WAIT_INTERVAL_SECONDS = 0.1
_LEGACY_FRAME_PART_SIZE = 512 * 1024


def _transport_ms(value: float) -> int:
    return max(0, int(value + 0.5))


async def _wait_for_active_session_slot(
    *,
    timeout_s: float = _ACTIVE_SESSION_WAIT_SECONDS,
    interval_s: float = _ACTIVE_SESSION_WAIT_INTERVAL_SECONDS,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while _ACTIVE_SESSION_IDS and time.monotonic() < deadline:
        await asyncio.sleep(interval_s)
    return not _ACTIVE_SESSION_IDS


def _log_realtime_chunk_timing(
    session: GenerateSession,
    chunk: RealtimeChunkContext,
    batch: "Req",
    request_prepare_ms: float,
    scheduler_forward_ms: float,
    chunk_total_ms: float,
    send_stats: RealtimeFrameSendStats,
) -> None:
    logger.info(
        "realtime chunk timing: session_id=%s request_id=%s "
        "chunk_idx=%s event_id=%s condition_kinds=%s "
        "request_prepare=%.2fms scheduler_forward=%.2fms "
        "output_pace=%.2fms "
        "header_pack=%.2fms "
        "header_write=%.2fms raw_payload_build=%.2fms raw_write=%.2fms "
        "ws_write=%.2fms chunk_total=%.2fms batches=%d frames=%d "
        "frame_shape=%s raw_bytes=%d ws_payload_bytes=%d content_type=%s",
        session.id,
        chunk.request_id,
        batch.block_idx,
        getattr(batch, "realtime_event_id", None),
        sorted(batch.condition_inputs) if batch.condition_inputs else [],
        request_prepare_ms,
        scheduler_forward_ms,
        send_stats["pace_wait_ms"],
        send_stats["header_pack_ms"],
        send_stats["header_write_ms"],
        send_stats["raw_payload_build_ms"],
        send_stats["raw_write_ms"],
        send_stats["ws_write_ms"],
        chunk_total_ms,
        send_stats["num_batches"],
        send_stats["num_frames"],
        send_stats["frame_shape"],
        send_stats["raw_bytes"],
        send_stats["ws_payload_bytes"],
        send_stats["content_type"],
    )


async def _send_realtime_chunk_stats(
    ws: WebSocket,
    session: GenerateSession,
    chunk: RealtimeChunkContext,
    batch: "Req",
    request_prepare_ms: float,
    scheduler_forward_ms: float,
    chunk_total_ms: float,
    send_stats: RealtimeFrameSendStats,
) -> None:
    await ws.send_bytes(
        msgspec.msgpack.encode(
            {
                "type": "chunk_stats",
                "session_id": session.id,
                "request_id": chunk.request_id,
                "chunk_index": batch.block_idx,
                "event_id": getattr(batch, "realtime_event_id", None),
                "request_prepare_ms": _transport_ms(request_prepare_ms),
                "scheduler_forward_ms": _transport_ms(scheduler_forward_ms),
                "pace_wait_ms": _transport_ms(send_stats["pace_wait_ms"]),
                "header_write_ms": _transport_ms(send_stats["header_write_ms"]),
                "raw_payload_build_ms": _transport_ms(
                    send_stats["raw_payload_build_ms"]
                ),
                "raw_write_ms": _transport_ms(send_stats["raw_write_ms"]),
                "ws_write_ms": _transport_ms(send_stats["ws_write_ms"]),
                "chunk_total_ms": _transport_ms(chunk_total_ms),
                "num_batches": send_stats["num_batches"],
                "num_frames": send_stats["num_frames"],
                "raw_bytes": send_stats["raw_bytes"],
                "ws_payload_bytes": send_stats["ws_payload_bytes"],
                "content_type": send_stats["content_type"],
            }
        )
    )


async def _generate_loop(ws: WebSocket, session: GenerateSession):
    adapter = session.adapter
    if adapter is None:
        raise ValueError("realtime adapter is not initialized")

    pending_send_task = None
    while not session.reached_max_chunks():
        try:
            if (
                session.legacy_protocol
                and session.request is not None
                and session.is_v2v_enabled()
                and session.request.first_frame is None
            ):
                while not session.has_pending_video_frames():
                    await asyncio.sleep(0.01)

            if pending_send_task is not None and pending_send_task.done():
                await pending_send_task
                pending_send_task = None

            # send to scheduler and generate video chunk
            server_args = get_global_server_args()

            await adapter.wait_for_next_chunk(session)

            timer = RealtimeStageTimer()
            chunk_started = time.perf_counter()

            chunk = session.new_chunk()
            batch = adapter.prepare_next_request(
                session,
                server_args,
                chunk,
            )
            if not hasattr(batch, "extra"):
                batch.extra = {}
            camera_actions = (batch.condition_inputs or {}).get("camera_actions")
            if camera_actions is not None:
                batch.extra["actions"] = camera_actions
            if session.legacy_protocol and session.is_v2v_enabled():
                batch.input_video = session.sample_video_frames()
            session.apply_movement_prompt_to_batch(batch)
            session.apply_prompt_event_to_batch(batch)
            if session.legacy_protocol:
                # Preserve the pre-mainline transport: save each generated
                # chunk and send its bytes using frame_start/part/end messages.
                batch.return_raw_frames = False
                batch.save_output = True
                batch.return_file_paths_only = False
            if batch.condition_inputs:
                logger.debug(
                    "consume realtime conditions, session_id=%s, block_idx=%s, kinds=%s",
                    session.id,
                    batch.block_idx,
                    sorted(batch.condition_inputs),
                )
            request_prepare_ms = timer.mark_ms()

            legacy_async_postprocess = (
                session.legacy_protocol and server_args.realtime_async_postprocess
            )
            if legacy_async_postprocess:
                result = await async_scheduler_client.forward([batch])
                if (
                    result.output is None
                    and result.output_file_paths is None
                    and result.raw_frame_batches is None
                ):
                    error_msg = result.error or "Unknown error"
                    raise RuntimeError(
                        "Model generation returned no output. "
                        f"Error from scheduler: {error_msg}"
                    )
                save_file_paths = list(result.output_file_paths or [])
            else:
                save_file_paths, result = await process_generation_batch(
                    async_scheduler_client, batch
                )
            scheduler_forward_ms = timer.mark_ms()

            # finish
            adapter.on_chunk_complete(session, result)
            if pending_send_task is not None:
                await pending_send_task
            if getattr(batch, "realtime_output_pacing", False):
                await _send_output_and_log(
                    ws,
                    session,
                    chunk,
                    batch,
                    result,
                    request_prepare_ms,
                    scheduler_forward_ms,
                    chunk_started,
                    save_file_paths,
                )
                pending_send_task = None
            else:
                pending_send_task = asyncio.create_task(
                    _send_output_and_log(
                        ws,
                        session,
                        chunk,
                        batch,
                        result,
                        request_prepare_ms,
                        scheduler_forward_ms,
                        chunk_started,
                        save_file_paths,
                    )
                )

        except asyncio.CancelledError:
            if pending_send_task is not None:
                pending_send_task.cancel()
                await _await_realtime_task(pending_send_task)
            logger.info("generation completed, session_id=%s", session.id)
            break
        except WebSocketDisconnect:
            if pending_send_task is not None:
                pending_send_task.cancel()
                await _await_realtime_task(pending_send_task)
            logger.info(
                "client disconnected during generation, session_id=%s", session.id
            )
            break
        except Exception as e:
            if pending_send_task is not None:
                pending_send_task.cancel()
                await _await_realtime_task(pending_send_task)
            err_msg = str(e).splitlines()[0]
            logger.error("error during generate loop: %s", err_msg)
            try:
                await write_error_msg(f"error during generate loop: {err_msg}", ws)
            except Exception as send_error:
                logger.error(
                    "error during sending complete msg: %s",
                    send_error,
                )
            break
    else:
        if pending_send_task is not None:
            await pending_send_task
        logger.info(
            "generation reached max chunks, session_id=%s, max_chunks=%s",
            session.id,
            session.request.max_chunks if session.request is not None else None,
        )


async def _send_output_and_log(
    ws: WebSocket,
    session: GenerateSession,
    chunk: RealtimeChunkContext,
    batch: "Req",
    result,
    request_prepare_ms: float,
    scheduler_forward_ms: float,
    chunk_started: float,
    save_file_paths: list[str] | None = None,
) -> RealtimeFrameSendStats:
    if session.adapter is None:
        raise ValueError("realtime adapter is not initialized")
    save_file_paths = list(save_file_paths or [])
    pace_wait_ms = await _wait_for_realtime_output_slot(session, batch, result)
    if session.legacy_protocol:
        if not save_file_paths and result.output is not None:
            postprocess_started = time.perf_counter()
            save_file_paths = await asyncio.to_thread(
                _save_legacy_output_paths,
                batch,
                result,
            )
            logger.info(
                "Async postprocess completed in %.3f seconds for request %s (files=%s)",
                time.perf_counter() - postprocess_started,
                chunk.request_id,
                save_file_paths,
            )
        send_stats = await _send_legacy_file_paths(
            ws,
            save_file_paths,
            chunk_index_start=batch.block_idx,
            request_id=chunk.request_id,
        )
    else:
        send_stats = await session.adapter.send_output(
            ws,
            session,
            result,
            batch,
        )
    send_stats["pace_wait_ms"] = pace_wait_ms
    chunk_total_ms = (time.perf_counter() - chunk_started) * 1000
    _log_realtime_chunk_timing(
        session,
        chunk,
        batch,
        request_prepare_ms,
        scheduler_forward_ms,
        chunk_total_ms,
        send_stats,
    )
    if not session.legacy_protocol:
        await _send_realtime_chunk_stats(
            ws,
            session,
            chunk,
            batch,
            request_prepare_ms,
            scheduler_forward_ms,
            chunk_total_ms,
            send_stats,
        )
    return send_stats


def _save_legacy_output_paths(batch: "Req", result) -> list[str]:
    if result.output is None:
        return list(result.output_file_paths or [])
    num_outputs = len(result.output)
    return save_outputs(
        result.output,
        batch.data_type,
        batch.fps,
        batch.save_output,
        lambda idx: str(batch.output_file_path(num_outputs, idx)),
        audio=result.audio,
        audio_sample_rate=result.audio_sample_rate,
        output_compression=batch.output_compression,
        enable_frame_interpolation=batch.enable_frame_interpolation,
        frame_interpolation_exp=batch.frame_interpolation_exp,
        frame_interpolation_scale=batch.frame_interpolation_scale,
        frame_interpolation_model_path=batch.frame_interpolation_model_path,
        enable_upscaling=batch.enable_upscaling,
        upscaling_model_path=batch.upscaling_model_path,
        upscaling_scale=batch.upscaling_scale,
    )


async def _send_legacy_file_paths(
    ws: WebSocket,
    file_paths: list[str],
    *,
    chunk_index_start: int,
    request_id: str,
) -> RealtimeFrameSendStats:
    stats = empty_frame_send_stats("application/octet-stream")
    chunk_index = chunk_index_start
    for file_path in file_paths:
        frame_bytes = await asyncio.to_thread(_read_file_bytes, file_path)
        started = time.perf_counter()
        payload_bytes = await _write_legacy_frame_msg(
            frame_bytes,
            ws,
            chunk_index=chunk_index,
            request_id=request_id,
        )
        stats["raw_write_ms"] += (time.perf_counter() - started) * 1000
        stats["raw_bytes"] += len(frame_bytes)
        stats["ws_payload_bytes"] += payload_bytes
        stats["num_frames"] += 1
        stats["num_batches"] += 1
        chunk_index += 1
    stats["ws_write_ms"] = stats["raw_write_ms"]
    return stats


def _read_file_bytes(file_path: str) -> bytes:
    with open(file_path, "rb") as file:
        return file.read()


async def _write_legacy_frame_msg(
    content: bytes,
    websocket: WebSocket,
    *,
    chunk_index: int,
    request_id: str,
) -> int:
    num_parts = max(1, math.ceil(len(content) / _LEGACY_FRAME_PART_SIZE))
    payload_bytes = 0
    messages = [
        {
            "type": "frame_start",
            "request_id": request_id,
            "chunk_index": chunk_index,
            "total_size": len(content),
            "num_parts": num_parts,
        }
    ]
    for part_index in range(num_parts):
        start = part_index * _LEGACY_FRAME_PART_SIZE
        end = min(len(content), start + _LEGACY_FRAME_PART_SIZE)
        messages.append(
            {
                "type": "frame_part",
                "request_id": request_id,
                "chunk_index": chunk_index,
                "part_index": part_index,
                "content": content[start:end],
            }
        )
    messages.append(
        {
            "type": "frame_end",
            "request_id": request_id,
            "chunk_index": chunk_index,
        }
    )
    for message in messages:
        payload = msgspec.msgpack.encode(message)
        payload_bytes += len(payload)
        await websocket.send_bytes(payload)
    return payload_bytes


def _result_num_frames(result) -> int:
    if result.raw_frame_batches is None:
        return 0
    return sum(len(frames) for frames in result.raw_frame_batches)


def _output_pacing_fps(batch: "Req") -> float:
    fps = float(batch.fps or 0)
    if batch.enable_frame_interpolation:
        fps *= 2 ** int(batch.frame_interpolation_exp or 1)
    return fps


async def _wait_for_realtime_output_slot(
    session: GenerateSession,
    batch: "Req",
    result,
) -> float:
    if not getattr(batch, "realtime_output_pacing", False):
        return 0.0

    frame_count = _result_num_frames(result)
    output_fps = _output_pacing_fps(batch)
    if frame_count <= 0 or output_fps <= 0:
        return 0.0

    now = time.perf_counter()
    next_send_at = session.output_pace_next_send_at
    if next_send_at is None:
        next_send_at = now
    if (
        batch.realtime_event_id is not None
        and batch.realtime_event_id != session.output_pace_last_event_id
    ):
        next_send_at = min(next_send_at, now)
        session.output_pace_last_event_id = batch.realtime_event_id

    wait_s = max(0.0, next_send_at - now)
    if wait_s > 0:
        await asyncio.sleep(wait_s)

    send_started_at = time.perf_counter()
    session.output_pace_next_send_at = (
        max(next_send_at, send_started_at) + frame_count / output_fps
    )
    return wait_s * 1000


async def _await_realtime_task(task: asyncio.Task | None) -> None:
    if task is None:
        return
    try:
        await task
    except (asyncio.CancelledError, WebSocketDisconnect):
        pass
    except Exception as e:
        logger.debug("realtime task exited with error: %s", e)


async def _listen_events(ws: WebSocket, session: GenerateSession):
    """listen for user events: usually condition inputs"""
    async for message in ws.iter_bytes():
        data = None
        try:
            data = msgspec.msgpack.decode(message)
            if not isinstance(data, dict):
                raise ValueError("realtime event must be a map")
            event_type = data.get("type")
            if isinstance(event_type, bytes):
                event_type = event_type.decode("utf-8")
            if event_type in {"prompt", "video", "control"}:
                event_log = _ingest_legacy_action(
                    session,
                    RealtimeAction.model_validate(data),
                )
                event_id = None
            else:
                realtime_event = RealtimeEvent.model_validate(data)
                if session.adapter is None:
                    raise ValueError("realtime adapter is not initialized")
                event_log = session.adapter.ingest_event(session, realtime_event)
                event_id = realtime_event.event_id
            logger.info(
                "receive realtime event, session_id=%s, event_id=%s, %s",
                session.id,
                event_id,
                event_log,
            )
        except Exception as e:
            event_kind = data.get("kind") if isinstance(data, dict) else None
            logger.warning("invalid event, kind=%s, error=%s", event_kind, e)
            await write_error_msg("invalid event", ws)
            continue


def _ingest_legacy_action(
    session: GenerateSession,
    realtime_action: RealtimeAction,
) -> str:
    if session.adapter is None:
        raise ValueError("realtime adapter is not initialized")

    if realtime_action.type == "video":
        if not session.is_v2v_enabled():
            raise ValueError(
                "video action requires mode=v2v (or first_frame in auto mode)"
            )
        encoded_frames = list(realtime_action.video_frames or [])
        if realtime_action.video_frame is not None:
            encoded_frames.append(realtime_action.video_frame)
        if not encoded_frames:
            raise ValueError("video action requires video_frame or video_frames")
        frames = [
            Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            for frame_bytes in encoded_frames
        ]
        session.append_video_frames(frames)
        return (
            f"type=video, num_frames={len(encoded_frames)}, "
            f"total_bytes={sum(len(frame) for frame in encoded_frames)}"
        )

    if realtime_action.type == "prompt":
        if not realtime_action.action_content:
            raise ValueError("prompt action requires action_content")
        return session.adapter.ingest_event(
            session,
            RealtimeEvent(
                type="event",
                kind="prompt",
                payload=realtime_action.action_content,
            ),
        )

    action_log_parts: list[str] = []
    if realtime_action.control_chunk is not None:
        session.append_control_chunk(realtime_action.control_chunk)
        session.adapter.ingest_event(
            session,
            RealtimeEvent(
                type="event",
                kind="camera_actions",
                payload=realtime_action.control_chunk,
            ),
        )
        action_log_parts.append(
            f"mode=chunk, chunk_len={len(realtime_action.control_chunk)}"
        )

    if realtime_action.key is not None or realtime_action.action is not None:
        if realtime_action.key is None or realtime_action.action is None:
            raise ValueError("control key and action must be set together")
        key, key_action = session.validate_control_key_action(
            realtime_action.key,
            realtime_action.action,
        )
        session.set_key_state(key, key_action)
        session.adapter.ingest_event(
            session,
            RealtimeEvent(
                type="event",
                kind="camera_actions",
                payload={
                    "mode": "state",
                    "transitions": [{"actions": sorted(session.current_keys)}],
                },
            ),
        )
        action_log_parts.append(f"key={key}, action={key_action}")

    event_ids = session.resolve_action_event_ids(realtime_action)
    if event_ids:
        session.trigger_prompt_events(event_ids)
        action_log_parts.append(
            f"events={event_ids}, event_chunk={session.prompt_event_chunk}"
        )
    if not action_log_parts:
        raise ValueError(
            "control action requires control_chunk, key/action, or event(s)"
        )
    return "type=control, " + ", ".join(action_log_parts)


async def _listen_generate_request(ws: WebSocket, session: GenerateSession):
    while True:
        try:
            data = msgspec.msgpack.decode(await ws.receive_bytes())
            if not isinstance(data, dict):
                raise ValueError("generate request must be a map")

            session.legacy_protocol = "type" not in data
            mode_raw = data.get("mode")
            if mode_raw is None:
                mode = None
            else:
                if isinstance(mode_raw, bytes):
                    mode_raw = mode_raw.decode("utf-8")
                try:
                    mode = RealtimeVideoMode(mode_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError("mode must be one of: t2v, v2v") from exc
            if mode == RealtimeVideoMode.T2V and data.get("first_frame") is not None:
                raise ValueError("first_frame is not allowed when mode=t2v")

            realtime_req = RealtimeVideoGenerationsRequest.model_validate(data)
            session.set_stream_id(realtime_req.stream_id)
            session.set_mode(mode)
            adapter = get_realtime_model_adapter(get_global_server_args())
            session.set_adapter(adapter)
            await adapter.on_init(session, realtime_req)

            # Keep session state update atomic with validated request.
            session.set_request(realtime_req)
            break
        except WebSocketDisconnect:
            raise
        except Exception as e:
            logger.warning(
                "invalid generate request, session_id=%s, error=%s",
                session.id,
                e,
            )
            await write_error_msg("invalid generate request", ws)
            continue


async def _cleanup_realtime_session(
    session: GenerateSession,
    generate_task: asyncio.Task | None,
    listen_task: asyncio.Task | None,
) -> None:
    logger.info("terminating session, session_id=%s", session.id)
    for task in (generate_task, listen_task):
        if task and not task.done():
            task.cancel()
    for task in (generate_task, listen_task):
        if task is None:
            continue
        await _await_realtime_task(task)
    try:
        await async_scheduler_client.forward(
            ReleaseRealtimeSessionReq(session_id=session.id)
        )
    except Exception as e:
        logger.warning(
            "failed to release realtime session on scheduler, session_id=%s, error=%s",
            session.id,
            e,
        )
    if session.input_temp_dir is not None:
        shutil.rmtree(session.input_temp_dir, ignore_errors=True)
    session.dispose()


async def _close_realtime_websocket(
    websocket: WebSocket,
    *,
    code: int,
    reason: str,
) -> None:
    try:
        await websocket.close(code=code, reason=reason)
    except (RuntimeError, WebSocketDisconnect):
        pass


async def _wait_for_server_warmup(websocket: WebSocket) -> None:
    warmup_done = getattr(websocket.app.state, "server_warmup_done", None)
    if warmup_done is not None and not warmup_done.is_set():
        await warmup_done.wait()


@router.websocket("/generate")
async def generate(websocket: WebSocket):
    """endpoint for creating a new realtime session"""
    await websocket.accept()
    await _wait_for_server_warmup(websocket)
    if _ACTIVE_SESSION_IDS and not await _wait_for_active_session_slot():
        logger.warning(
            "reject realtime session because another session is active: %s",
            sorted(_ACTIVE_SESSION_IDS),
        )
        try:
            await write_error_msg(
                "another realtime session is already active", websocket
            )
        finally:
            await websocket.close(code=1008)
        return

    session = GenerateSession()
    _ACTIVE_SESSION_IDS.add(session.id)
    generate_task = None
    listen_task = None
    try:
        # receive new generate request
        await _listen_generate_request(websocket, session)

        # continuously generate video chunk
        generate_task = asyncio.create_task(_generate_loop(websocket, session))
        # continuously listen for user events
        listen_task = asyncio.create_task(_listen_events(websocket, session))

        wait_tasks = [generate_task, listen_task]
        await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
        if generate_task.done() and session.reached_max_chunks():
            await _close_realtime_websocket(
                websocket,
                code=1000,
                reason="generation complete",
            )

    except WebSocketDisconnect:
        logger.info("client disconnected, session_id=%s", session.id)
    finally:
        try:
            await _cleanup_realtime_session(session, generate_task, listen_task)
        finally:
            _ACTIVE_SESSION_IDS.discard(session.id)


async def write_error_msg(error_msg: str, websocket: WebSocket):
    await websocket.send_bytes(
        msgspec.msgpack.encode({"type": "error", "content": error_msg})
    )
