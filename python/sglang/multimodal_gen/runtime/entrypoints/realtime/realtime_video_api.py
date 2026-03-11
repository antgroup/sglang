import asyncio
import io
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from msgpack import packb, unpackb
from PIL import Image
from pydantic import ValidationError

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeAction,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    process_generation_batch,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    ReleaseRealtimeSessionReq,
    prepare_request,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/realtime_video", tags=["realtime"])


async def _generate_loop(ws: WebSocket, session: GenerateSession):

    while True:
        try:
            # If the request is video-stream driven (no first_frame), give the
            # client a short grace period to push the first video chunk.
            if (
                session.generate_chunk_cnt == 0
                and session.request is not None
                and session.request.first_frame is None
                and not session.has_pending_video_frames()
            ):
                for _ in range(30):  # ~300ms max
                    await asyncio.sleep(0.01)
                    if session.has_pending_video_frames():
                        break

            session.new_request()

            # send to scheduler and generate video chunk
            batch = prepare_request(
                server_args=get_global_server_args(),
                sampling_params=session.build_sampling_params(),
            )
            batch.session = session.realtime_session
            batch.extra["realtime_session_id"] = session.id
            batch.block_idx = session.generate_chunk_cnt
            batch.input_video = session.sample_video_frames()
            save_file_path_list, result = await process_generation_batch(
                async_scheduler_client, batch
            )

            # send to client
            save_file_path = save_file_path_list[0]
            with open(save_file_path, "rb") as f:
                frame_bytes = f.read()
            await write_frame_msg(frame_bytes, ws)

            session.generate_chunk_completed()

            logger.info(
                f"generate video chunk, "
                f"request_id: {session.request_id},"
                f"chunk_cnt: {session.generate_chunk_cnt},"
                f"save_file_path: {save_file_path}"
            )

        except asyncio.CancelledError:
            logger.info(f"generation completed, session_id: {session.id}")
            break
        except Exception as e:
            err_msg = str(e).splitlines()[0]
            logger.error(f"error during generate loop: {err_msg}")
            try:
                await write_error_msg(f"error during generate loop: {err_msg}", ws)
            except Exception as e:
                logger.error(f"error during sending complete msg: {e}")
                pass
            break


async def _listen_actions(ws: WebSocket, session: GenerateSession):
    async for data in ws.iter_bytes():
        data = unpackb(data)
        try:
            realtime_action = RealtimeAction.model_validate(data)
            if realtime_action.type == "video":
                encoded_frames = list(realtime_action.video_frames or [])
                if realtime_action.video_frame is not None:
                    encoded_frames.append(realtime_action.video_frame)
                if len(encoded_frames) == 0:
                    raise ValueError(
                        "video action requires video_frame or video_frames"
                    )

                frames = []
                for frame_bytes in encoded_frames:
                    image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                    frames.append(image)
                session.append_video_frames(frames)
                total_bytes = sum(len(frame) for frame in encoded_frames)
                action_log = (
                    f"type=video, num_frames={len(encoded_frames)}, "
                    f"total_bytes={total_bytes}"
                )
            else:
                if not realtime_action.action_content:
                    raise ValueError("prompt action requires action_content")
                session.append_action(realtime_action)
                action_log = (
                    f"type=prompt, content_len={len(realtime_action.action_content)}"
                )
            logger.info(
                f"receive realtime action, session_id: {session.id}, {action_log}"
            )
        except (ValidationError, ValueError) as e:
            action_type = data.get("type") if isinstance(data, dict) else None
            logger.warning(f"invalid action, type={action_type}, error={e}")
            await write_error_msg("invalid action", ws)
            continue


async def _listen_generate_request(ws: WebSocket, session: GenerateSession):
    while True:
        try:
            data = unpackb(await ws.receive_bytes())
            realtime_req = RealtimeVideoGenerationsRequest.model_validate(data)
            # TODO(puf147): convert RGB for krea
            # params.start_frame = Image.open(params.start_frame).convert("RGB")
            if realtime_req.first_frame is not None:
                uploads_dir = os.path.join("inputs", "uploads")
                os.makedirs(uploads_dir, exist_ok=True)

                target_path = os.path.join(uploads_dir, f"{session.id}_first_frame")
                image_path = await save_image_to_path(
                    realtime_req.first_frame, target_path
                )
                realtime_req.first_frame = image_path

            session.setRequest(realtime_req)
            break
        except Exception as e:
            logger.warning(
                f"invalid generate request, session_id: {session.id}, error={e}"
            )
            await write_error_msg("invalid generate request", ws)
            continue


@router.websocket("/generate")
async def generate(websocket: WebSocket):
    await websocket.accept()
    session = GenerateSession()
    generate_task = None
    listen_task = None
    try:
        # receive new generate request
        await _listen_generate_request(websocket, session)

        # generate video chunk
        generate_task = asyncio.create_task(_generate_loop(websocket, session))
        # listen for actions
        listen_task = asyncio.create_task(_listen_actions(websocket, session))

        await asyncio.wait(
            [generate_task, listen_task], return_when=asyncio.FIRST_COMPLETED
        )

    except WebSocketDisconnect:
        logger.info(f"client disconnected, session_id: {session.id}")
    finally:
        logger.info(f"terminating session, session_id: {session.id}")
        if generate_task and not generate_task.done():
            generate_task.cancel()
        if listen_task and not listen_task.done():
            listen_task.cancel()
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
        if session:
            session.dispose()


async def write_error_msg(error_msg: str, websocket: WebSocket):
    await websocket.send_bytes(packb({"type": "error", "content": error_msg}))


async def write_status_msg(status: str, websocket: WebSocket):
    await websocket.send_bytes(packb({"type": "status", "content": status}))


async def write_frame_msg(content: bytes, websocket: WebSocket):
    await websocket.send_bytes(packb({"type": "frame", "content": content}))
