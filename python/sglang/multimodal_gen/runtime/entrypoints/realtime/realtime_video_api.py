import asyncio
import os
from collections import deque
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from msgpack import packb, unpackb
from pydantic import ValidationError

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeAction,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    build_sampling_params,
    process_generation_batch,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/realtime_video", tags=["realtime"])


class GenerateSession:

    def __init__(self):
        self.id = uuid4().hex
        self.request_id = None
        self.request = None
        self.action_queue = deque(maxlen=3)
        self.generate_chunk_cnt = 0

    def setRequest(self, request: RealtimeVideoGenerationsRequest):
        self.request = request

    def dispose(self):
        self.action_queue.clear()

    def new_request(self):
        self.request_id = f"{self.id}_{uuid4().hex}"

    def generate_chunk_completed(self):
        self.generate_chunk_cnt += 1

    def append_action(self, action: RealtimeAction):
        self.action_queue.append(action)

    def sample_action(self) -> RealtimeAction:
        return self.action_queue.popleft()

    def build_sampling_params(self):
        if self.generate_chunk_cnt == 0:
            prompt = self.request.prompt
        else:
            realtime_action = self.action_queue.popleft()
            # only support prompt action
            if realtime_action.type == "prompt":
                prompt = realtime_action.action_content

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
            output_path=self.request.output_path,
            output_compression=self.request.output_compression,
            output_quality=self.request.output_quality,
        )


async def _generate_loop(ws: WebSocket, session: GenerateSession):

    while True:
        try:
            session.new_request()

            # send to scheduler and generate video chunk
            batch = prepare_request(
                server_args=get_global_server_args(),
                sampling_params=session.build_sampling_params(),
            )
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
            try:
                await write_status_msg("generation canceled.", ws)
            except Exception as e:
                logger.error(f"error during sending complete msg: {e}")
                pass
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
    data = unpackb(await ws.receive_bytes())
    realtime_req = RealtimeVideoGenerationsRequest.model_validate(data)
    # TODO: convert RGB for krea
    # params.start_frame = Image.open(params.start_frame).convert("RGB")
    uploads_dir = os.path.join("inputs", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    target_path = os.path.join(uploads_dir, f"{session.id}_first_frame")
    image_path = await save_image_to_path(realtime_req.first_frame, target_path)

    realtime_req.first_frame = image_path
    return realtime_req


@router.websocket("/generate")
async def generate(websocket: WebSocket):
    await websocket.accept()
    session = GenerateSession()
    try:
        # receive new generate request
        while True:
            try:
                realtime_req = await _listen_generate_request(websocket, session)
                session.setRequest(realtime_req)
                break
            except Exception as e:
                logger.warning(f"invalid generate request, session_id={id}, error={e}")
                await write_error_msg("invalid generate request", websocket)
                continue

        # generate video chunk
        generate_task = asyncio.create_task(_generate_loop(websocket, session))
        # listen for actions
        listen_task = asyncio.create_task(_listen_actions(websocket, session))

        await asyncio.wait(
            [generate_task, listen_task], return_when=asyncio.FIRST_COMPLETED
        )

    except WebSocketDisconnect:
        logger.info(f"client disconnected, session_id: {id}")
    finally:
        logger.info(f"terminating session, session_id: {id}")
        if generate_task:
            generate_task.cancel()
        if listen_task:
            listen_task.cancel()
        if session:
            session.dispose()


async def write_error_msg(error_msg: str, websocket: WebSocket):
    await websocket.send_bytes(packb({"type": "error", "content": error_msg}))


async def write_status_msg(status: str, websocket: WebSocket):
    await websocket.send_bytes(packb({"type": "status", "content": status}))


async def write_frame_msg(content: bytes, websocket: WebSocket):
    await websocket.send_bytes(packb({"type": "frame", "content": content}))
