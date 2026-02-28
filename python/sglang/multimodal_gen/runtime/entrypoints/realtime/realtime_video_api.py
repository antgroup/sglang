import asyncio
import os
from collections import deque

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from msgpack import unpackb
from pydantic import ValidationError

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeAction,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/realtime_video", tags=["realtime"])


class GenerateSession:

    def __init__(self, id: str):
        self.id = id
        self.action_queue = deque(maxlen=3)

    def dispose(self):
        self.action_queue.clear()

    def sample_action(self):
        return self.action_queue.popleft()


async def _generate_loop(ws: WebSocket, session: GenerateSession):
    while True:
        try:
            # TODO: send to scheduler
            # rt_act = session.sample_action()
            # from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
            # save_file_path_list, result = await process_generation_batch(
            #     async_scheduler_client, batch
            # )
            # save_file_path = save_file_path_list[0]
            # TODO: receive bytes stream or file path?
            # websocket.send_bytes(next_frame)
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info(f"generation completed, session_id: {session.id}")
            try:
                await ws.send_json({"session_id": session.id, "status": "completed"})
            except Exception as e:
                logger.error(f"error during sending complete msg: {e}")
                pass
            break
        except Exception as e:
            logger.error(f"error during generate loop: {e}")


async def _listen_actions(ws: WebSocket, session: GenerateSession):
    async for data in ws.iter_bytes():
        data = unpackb(data)
        try:
            realtime_action = RealtimeAction.model_validate(data)
            session.action_queue.append(realtime_action)
        except ValidationError as e:
            logger.warning(f"invalid action, data={data}, error={e}")
            await ws.send_json({"error": e.errors()})
            continue


async def _handle_generate_request(data, session_id: str):
    realtime_req = RealtimeVideoGenerationsRequest.model_validate(data)
    # TODO: convert RGB for krea
    # params.start_frame = Image.open(params.start_frame).convert("RGB")
    uploads_dir = os.path.join("inputs", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    target_path = os.path.join(uploads_dir, f"{session_id}_first_frame")
    image_path = await save_image_to_path(realtime_req.first_frame, target_path)

    realtime_req.first_frame = image_path
    return realtime_req


@router.websocket("/generate/{id}")
async def generate(websocket: WebSocket, id: str):
    await websocket.accept()
    try:
        while True:
            # receive new generate request
            try:
                data = unpackb(await websocket.receive_bytes())
                realtime_req = await _handle_generate_request(data, id)
            except Exception as e:
                logger.warning(f"invalid generate request, data={data}, error={e}")
                await websocket.send_json({"error": e.errors()})
                continue

            # TODO: init session
            session = GenerateSession(id)

            # generate video chunk
            generate_task = asyncio.create_task(_generate_loop(websocket, session))
            # listen for actions
            _listen_actions(websocket, session)

    except WebSocketDisconnect:
        logger.info(f"client disconnected, session_id: {id}")
    finally:
        logger.info(f"terminating session, session_id: {id}")
        if generate_task:
            generate_task.cancel()
        if session:
            session.dispose()
        try:
            await websocket.send_json({"session_id": id, "status": "completed"})
        except:
            pass
