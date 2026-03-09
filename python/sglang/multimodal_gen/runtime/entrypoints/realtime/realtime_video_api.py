import asyncio
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from msgpack import packb, unpackb
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
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/realtime_video", tags=["realtime"])


async def _generate_loop(ws: WebSocket, session: GenerateSession):

    while True:
        try:
            session.new_request()

            # send to scheduler and generate video chunk
            batch = prepare_request(
                server_args=get_global_server_args(),
                sampling_params=session.build_sampling_params(),
            )
            batch.session = session.realtime_session
            batch.block_idx = session.generate_chunk_cnt
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
            logger.error(f"error during generate loop: {e}")
            try:
                await write_error_msg(f"error during generate loop: {e}", ws)
            except Exception as e:
                logger.error(f"error during sending complete msg: {e}")
                pass
            break


async def _listen_actions(ws: WebSocket, session: GenerateSession):
    async for data in ws.iter_bytes():
        data = unpackb(data)
        try:
            realtime_action = RealtimeAction.model_validate(data)
            session.append_action(realtime_action)
            logger.info(
                f"receive realtime action, session_id: {session.id}, realtime_action: {realtime_action}"
            )
        except ValidationError as e:
            logger.warning(f"invalid action, data={data}, error={e}")
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
        if listen_task and not generate_task.done():
            listen_task.cancel()
        if session:
            session.dispose()


async def write_error_msg(error_msg: str, websocket: WebSocket):
    await websocket.send_bytes(packb({"type": "error", "content": error_msg}))


async def write_status_msg(status: str, websocket: WebSocket):
    await websocket.send_bytes(packb({"type": "status", "content": status}))


async def write_frame_msg(content: bytes, websocket: WebSocket):
    await websocket.send_bytes(packb({"type": "frame", "content": content}))
