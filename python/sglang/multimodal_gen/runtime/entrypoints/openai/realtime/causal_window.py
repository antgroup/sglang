# SPDX-License-Identifier: Apache-2.0
"""Bounded realtime submission for the experimental LingBot causal pipeline."""

import time
from uuid import uuid4

from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
    CausalChunkWindowReq,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    RealtimeChunkContext,
)


async def generate_causal_windows(
    ws, session, server_args, scheduler_client, send_output
):
    """Snapshot at most one window of conditions; preserve ordered delivery.

    User input arriving while a window runs applies to the next window. A slow
    client prevents the next submission, bounding both decoded and latent queues.
    """
    adapter = session.adapter
    while not session.reached_max_chunks():
        count = server_args.causal_pipeline_window
        if session.request.max_chunks is not None:
            count = min(count, session.request.max_chunks - session.generate_chunk_cnt)
        chunks, batches, starts, prepare_times = [], [], [], []
        for offset in range(count):
            await adapter.wait_for_next_chunk(session)
            started = time.perf_counter()
            chunk = RealtimeChunkContext(
                session_id=session.id,
                index=session.generate_chunk_cnt + offset,
                request_id=f"{session.id}_{uuid4().hex}",
            )
            batch = adapter.prepare_next_request(session, server_args, chunk)
            batch.extra["causal_pipeline_window_size"] = count
            chunks.append(chunk)
            batches.append(batch)
            starts.append(started)
            prepare_times.append((time.perf_counter() - started) * 1000)

        forward_start = time.perf_counter()
        result = await scheduler_client.forward(CausalChunkWindowReq(chunks=batches))
        elapsed_ms = (time.perf_counter() - forward_start) * 1000
        if result.error:
            raise RuntimeError(result.error)
        outputs = result.chunk_outputs
        if outputs is None or len(outputs) != count:
            raise RuntimeError("Causal pipeline returned an incomplete chunk window")
        for offset, output in enumerate(outputs):
            if output.error:
                raise RuntimeError(output.error)
            session.current_chunk = chunks[offset]
            adapter.on_chunk_complete(session, output)
            await send_output(
                ws,
                session,
                chunks[offset],
                batches[offset],
                output,
                prepare_times[offset],
                elapsed_ms,
                starts[offset],
            )
