import argparse
import asyncio
import glob
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, Iterator, Literal, Optional

import msgpack
import websockets
from websockets.exceptions import ConnectionClosed


def merge_mp4_chunks(chunk_dir: str, output_path: str) -> None:
    """Merge streamed mp4 chunks into one mp4 using ffmpeg concat."""
    chunk_files = sorted(
        [f for f in os.listdir(chunk_dir) if f.endswith(".mp4")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    if not chunk_files:
        print(f"[Merge] No mp4 files found in {chunk_dir}")
        return

    list_file = os.path.join(chunk_dir, "file_list.txt")
    with open(list_file, "w") as f:
        for chunk_file in chunk_files:
            f.write(f"file '{chunk_file}'\n")

    cmd = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file,
        "-c",
        "copy",
        output_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[Merge] Merged {len(chunk_files)} chunks -> {output_path}")
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)


def iter_frame_chunks(
    video_path: str,
    target_fps: float,
    frames_per_chunk: int,
    jpeg_quality: int,
    first_chunk_frames: int = 9,
) -> Iterator[list[bytes]]:
    """Extract frames via ffmpeg and yield frame chunks as jpeg bytes."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    qscale = max(2, min(31, int(round((101 - jpeg_quality) / 3.3))))
    with tempfile.TemporaryDirectory(prefix="krea_stream_frames_") as tmp_dir:
        frame_pattern = os.path.join(tmp_dir, "frame_%08d.jpg")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vf",
            f"fps={target_fps}",
            "-q:v",
            str(qscale),
            frame_pattern,
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        frame_paths = sorted(glob.glob(os.path.join(tmp_dir, "frame_*.jpg")))
        chunk: list[bytes] = []
        chunk_target = max(1, first_chunk_frames)
        first_chunk_sent = False
        for frame_path in frame_paths:
            with open(frame_path, "rb") as f:
                chunk.append(f.read())
            if len(chunk) >= chunk_target:
                yield chunk
                chunk = []
                if not first_chunk_sent:
                    first_chunk_sent = True
                    chunk_target = frames_per_chunk

        if chunk:
            yield chunk


@dataclass
class RealtimeAction:
    type: Literal["prompt", "video"]
    action_content: Optional[str] = None
    video_frame: Optional[bytes] = None
    video_frames: Optional[list[bytes]] = None

    def to_dict(self):
        data = {"type": self.type}
        if self.action_content is not None:
            data["action_content"] = self.action_content
        if self.video_frame is not None:
            data["video_frame"] = self.video_frame
        if self.video_frames is not None:
            data["video_frames"] = self.video_frames
        return data


class RealtimeVideoClient:
    def __init__(self, ws_url: str):
        self.ws_url = ws_url.rstrip("/")
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._listen_task: Optional[asyncio.Task] = None
        self.on_frame: Optional[Callable[[bytes], None]] = None

    async def connect(self) -> None:
        self.ws = await websockets.connect(self.ws_url)
        print(f"[Client] Connected to {self.ws_url}")

    async def disconnect(self) -> None:
        if self._listen_task:
            self._listen_task.cancel()
            self._listen_task = None
        if self.ws:
            await self.ws.close()
            self.ws = None
        print("[Client] Disconnected")

    async def generate_video(self, request_data: dict) -> None:
        if not self.ws:
            raise RuntimeError("Not connected. Call connect() first.")

        await self.ws.send(msgpack.packb(request_data))
        print(f"[Client] Sent generation request: {request_data.get('prompt')}")
        self._listen_task = asyncio.create_task(self._listen_loop())

    async def send_action(self, action: RealtimeAction) -> None:
        if not self.ws:
            raise RuntimeError("Not connected. Call connect() first.")
        await self.ws.send(msgpack.packb(action.to_dict()))

    async def send_prompt_action(self, prompt: str) -> None:
        await self.send_action(
            RealtimeAction(type="prompt", action_content=prompt),
        )

    async def send_video_frames(self, frames: list[bytes]) -> None:
        await self.send_action(
            RealtimeAction(type="video", video_frames=frames),
        )

    async def _listen_loop(self) -> None:
        try:
            async for message in self.ws:
                data = msgpack.unpackb(message)
                if data["type"] == "frame":
                    if self.on_frame:
                        self.on_frame(data["content"])
                else:
                    print(f"[Client] Received: {data}")
        except ConnectionClosed:
            print("[Client] Connection closed")
        except asyncio.CancelledError:
            print("[Client] Listen loop cancelled")
        except Exception as e:
            print(f"[Client] Error in listen loop: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Krea realtime v2v stream client")
    parser.add_argument(
        "--ws-url",
        default="ws://127.0.0.1:30000/v1/realtime_video/generate",
        help="Realtime websocket endpoint",
    )
    parser.add_argument("--input-video", required=True, help="Input video path")
    parser.add_argument("--prompt", required=True, help="Initial prompt")
    parser.add_argument("--size", default="832x480")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=6)
    parser.add_argument("--target-fps", type=float, default=6.0)
    parser.add_argument("--frames-per-chunk", type=int, default=12)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--stream-seconds", type=int, default=60)
    parser.add_argument("--chunk-dir", default="./receive_chunks")
    parser.add_argument("--output", default="output_video.mp4")
    args = parser.parse_args()

    chunk_dir = args.chunk_dir
    output_path = args.output

    if os.path.exists(chunk_dir):
        shutil.rmtree(chunk_dir)
    os.makedirs(chunk_dir, exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    client = RealtimeVideoClient(args.ws_url)
    chunk_index = 0

    def on_frame(frame_bytes: bytes):
        nonlocal chunk_index
        chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_index}.mp4")
        with open(chunk_file, "wb") as f:
            f.write(frame_bytes)
        print(f"[Callback] Chunk {chunk_index}, bytes={len(frame_bytes)}")
        chunk_index += 1

    client.on_frame = on_frame

    frame_chunks = iter_frame_chunks(
        args.input_video,
        target_fps=args.target_fps,
        frames_per_chunk=args.frames_per_chunk,
        jpeg_quality=args.jpeg_quality,
    )
    first_chunk = next(frame_chunks, None)
    if not first_chunk:
        raise RuntimeError("No frames sampled from input video")

    try:
        await client.connect()
        await client.generate_video(
            {
                "prompt": args.prompt,
                "size": args.size,
                "seed": args.seed,
                "num_inference_steps": args.num_inference_steps,
            }
        )

        chunk_interval = max(args.frames_per_chunk / max(args.target_fps, 0.1), 0.01)
        await client.send_video_frames(first_chunk)
        await asyncio.sleep(40)
        start = asyncio.get_event_loop().time()
        for frames in frame_chunks:
            await client.send_video_frames(frames)
            await asyncio.sleep(chunk_interval)
            await asyncio.sleep(20)
            if asyncio.get_event_loop().time() - start > args.stream_seconds:
                break

        await asyncio.sleep(10)
    finally:
        await client.disconnect()
        merge_mp4_chunks(chunk_dir, output_path)


if __name__ == "__main__":
    asyncio.run(main())
