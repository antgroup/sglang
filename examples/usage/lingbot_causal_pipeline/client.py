# SPDX-License-Identifier: Apache-2.0
"""Collect a bounded LingBot realtime response as a video and chunk-stat JSON."""

import argparse
import asyncio
import io
import json
from pathlib import Path

import imageio.v2 as imageio
import msgspec.msgpack
import numpy as np
import websockets
from PIL import Image


async def collect(args):
    payload = {
        "type": "init",
        "prompt": args.prompt,
        "first_frame": args.image.read_bytes(),
        "max_chunks": args.chunks,
        "size": args.size,
        "seed": args.seed,
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
        "realtime_output_format": "jpeg",
        "realtime_output_pacing": False,
        "fps": args.fps,
    }
    stats = []
    frames = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(args.output), fps=args.fps, codec="libx264") as writer:
        async with websockets.connect(args.url, max_size=None, ping_timeout=None) as ws:
            await ws.send(msgspec.msgpack.encode(payload))
            async for message in ws:
                header = msgspec.msgpack.decode(message)
                kind = header.get("type")
                if kind == "error":
                    raise RuntimeError(header.get("content", "Generation failed"))
                if kind == "chunk_stats":
                    stats.append(header)
                elif kind in ("frame_batch", "frame_batch_header"):
                    data = (
                        header["payload"] if kind == "frame_batch" else await ws.recv()
                    )
                    offset = 0
                    for length in header["payload_lengths"]:
                        with Image.open(
                            io.BytesIO(data[offset : offset + length])
                        ) as image:
                            writer.append_data(np.asarray(image.convert("RGB")))
                        offset += length
                        frames += 1
    if len(stats) != args.chunks:
        raise RuntimeError(f"Expected {args.chunks} chunks, received {len(stats)}")
    args.output.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Saved {frames} frames in {len(stats)} chunks to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default="ws://127.0.0.1:30000/v1/realtime_video/generate"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--size", default="832x480")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("lingbot-pipeline.mp4"))
    args = parser.parse_args()
    if args.chunks < 1:
        parser.error("--chunks must be positive")
    asyncio.run(collect(args))
