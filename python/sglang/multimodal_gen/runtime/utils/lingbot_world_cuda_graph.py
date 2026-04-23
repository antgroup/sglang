# SPDX-License-Identifier: Apache-2.0
"""CUDA graph capture/replay for LingBot-World-Fast realtime transformer.

Graphs are lazily captured and cached by KV cache index signature.  Each
unique combination of (write_start, write_end, attn_start, attn_end, with_roll)
produces one captured graph.  Typically ~16 graphs are needed:
- 15 growing-phase graphs (chunks 0-14, no roll, varying window size)
- 1 steady-state no-roll graph (chunks 15+, same indices every time)
- 1 steady-state with-roll graph (first denoising call of chunks 15+)

Subsequent sessions with the same resolution reuse all captured graphs.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.models.dits.lingbot_world import (
    KVCacheSteadyStateIndices,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class _CapturedGraph:
    """One captured CUDA graph with its static output."""

    __slots__ = ("graph", "s_output")

    def __init__(self, graph: torch.cuda.CUDAGraph, s_output: torch.Tensor) -> None:
        self.graph = graph
        self.s_output = s_output


class LingBotWorldCudaGraphRunner:
    """Lazily captures and replays CUDA graphs keyed by KV cache indices.

    Static input buffers are shared across all graphs.  On the first call
    with a new index signature the graph is captured; subsequent calls with
    the same signature replay instantly.
    """

    def __init__(self, transformer) -> None:
        self.transformer = transformer
        self._pool = torch.cuda.graph_pool_handle()
        self._graphs: dict[tuple, _CapturedGraph] = {}

        # Shared static input buffers (allocated on first capture)
        self._s_hidden_states: torch.Tensor | None = None
        self._s_encoder_hidden_states: torch.Tensor | None = None
        self._s_timestep: torch.Tensor | None = None
        self._s_freqs_cos: torch.Tensor | None = None
        self._s_freqs_sin: torch.Tensor | None = None
        self._s_c2ws_plucker_emb: torch.Tensor | None = None
        self._s_encoder_hidden_states_image: torch.Tensor | None = None
        self._kv_cache: list[dict] | None = None
        self._initialized = False

    @staticmethod
    def _make_key(kv_idx: KVCacheSteadyStateIndices, with_roll: bool) -> tuple:
        return (
            kv_idx.write_start,
            kv_idx.write_end,
            kv_idx.attn_start,
            kv_idx.attn_end,
            with_roll,
        )

    def _ensure_buffers(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        kv_cache: list[dict],
        c2ws_plucker_emb: torch.Tensor | None,
        encoder_hidden_states_image: torch.Tensor | None,
    ) -> None:
        """Allocate shared static input buffers on first use."""
        if self._initialized:
            return
        self._s_hidden_states = hidden_states.clone()
        self._s_encoder_hidden_states = encoder_hidden_states.clone()
        self._s_timestep = timestep.clone()
        self._s_freqs_cos = freqs_cis[0].clone()
        self._s_freqs_sin = freqs_cis[1].clone()
        if c2ws_plucker_emb is not None:
            self._s_c2ws_plucker_emb = c2ws_plucker_emb.clone()
        if encoder_hidden_states_image is not None:
            self._s_encoder_hidden_states_image = encoder_hidden_states_image.clone()
        self._kv_cache = kv_cache
        self._initialized = True

    def _copy_inputs(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        c2ws_plucker_emb: torch.Tensor | None,
    ) -> None:
        self._s_hidden_states.copy_(hidden_states)
        self._s_timestep.copy_(timestep)
        self._s_freqs_cos.copy_(freqs_cis[0])
        self._s_freqs_sin.copy_(freqs_cis[1])
        if c2ws_plucker_emb is not None and self._s_c2ws_plucker_emb is not None:
            self._s_c2ws_plucker_emb.copy_(c2ws_plucker_emb)

    def _capture_one(
        self,
        kv_idx: KVCacheSteadyStateIndices,
        with_roll: bool,
    ) -> _CapturedGraph:
        forward_fn = (
            self.transformer.forward_cuda_graph_with_roll
            if with_roll
            else self.transformer.forward_cuda_graph_no_roll
        )
        s_freqs_cis = (self._s_freqs_cos, self._s_freqs_sin)
        common = dict(
            encoder_hidden_states_image=self._s_encoder_hidden_states_image,
            c2ws_plucker_emb=self._s_c2ws_plucker_emb,
        )
        call_args = (
            self._s_hidden_states,
            self._s_encoder_hidden_states,
            self._s_timestep,
            s_freqs_cis,
            self._kv_cache,
            kv_idx,
        )

        device = self._s_hidden_states.device

        # Warmup
        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            forward_fn(*call_args, **common)
        torch.cuda.current_stream(device).wait_stream(s)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            s_output = forward_fn(*call_args, **common)

        key = self._make_key(kv_idx, with_roll)
        logger.info(
            "Captured CUDA graph: key=%s (total cached: %d)",
            key,
            len(self._graphs) + 1,
        )
        return _CapturedGraph(graph, s_output)

    def replay(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        kv_cache: list[dict],
        kv_idx: KVCacheSteadyStateIndices,
        with_roll: bool,
        c2ws_plucker_emb: torch.Tensor | None = None,
        encoder_hidden_states_image: torch.Tensor | None = None,
        autocast_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Get-or-capture a graph, then copy inputs and replay.

        Returns a clone of the static output buffer.
        """
        self._ensure_buffers(
            hidden_states,
            encoder_hidden_states,
            timestep,
            freqs_cis,
            kv_cache,
            c2ws_plucker_emb,
            encoder_hidden_states_image,
        )

        key = self._make_key(kv_idx, with_roll)
        cg = self._graphs.get(key)
        if cg is None:
            if autocast_dtype is not None:
                with torch.autocast(
                    device_type="cuda", dtype=autocast_dtype, enabled=True
                ):
                    cg = self._capture_one(kv_idx, with_roll)
            else:
                cg = self._capture_one(kv_idx, with_roll)
            self._graphs[key] = cg

        self._copy_inputs(hidden_states, timestep, freqs_cis, c2ws_plucker_emb)
        cg.graph.replay()
        return cg.s_output.clone()

    def update_session_inputs(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
    ) -> None:
        """Update session-constant inputs (e.g. after prompt change)."""
        if self._s_encoder_hidden_states is not None:
            self._s_encoder_hidden_states.copy_(encoder_hidden_states)
        if (
            encoder_hidden_states_image is not None
            and self._s_encoder_hidden_states_image is not None
        ):
            self._s_encoder_hidden_states_image.copy_(encoder_hidden_states_image)

    def dispose(self) -> None:
        """Release all graphs and static buffers."""
        for cg in self._graphs.values():
            del cg.graph
        self._graphs.clear()
        self._s_hidden_states = None
        self._s_encoder_hidden_states = None
        self._s_timestep = None
        self._s_freqs_cos = None
        self._s_freqs_sin = None
        self._s_c2ws_plucker_emb = None
        self._s_encoder_hidden_states_image = None
        self._kv_cache = None
        self._initialized = False
