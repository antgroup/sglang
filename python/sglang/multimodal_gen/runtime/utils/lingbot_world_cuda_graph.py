# SPDX-License-Identifier: Apache-2.0
"""CUDA graph capture/replay for LingBot-World-Fast realtime transformer.

After the KV cache sliding window fills up (~15 chunks), all tensor shapes
and slice indices become fixed constants.  This module captures the
transformer forward as two CUDA graphs and replays them for all subsequent
chunks, eliminating ~5000 kernel launches per chunk.

Two graphs are needed because the first call per chunk rolls the KV cache
(evict oldest tokens), while subsequent calls overwrite the same positions.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.models.dits.lingbot_world import (
    KVCacheSteadyStateIndices,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class _SingleGraph:
    """Wraps a single CUDAGraph with its static input/output buffers."""

    def __init__(self) -> None:
        self.graph: torch.cuda.CUDAGraph | None = None
        self.s_output: torch.Tensor | None = None


class LingBotWorldCudaGraphRunner:
    """Manages two CUDA graphs for the CausalLingBotWorld transformer forward.

    - **graph_roll**: first call per chunk (rolls KV cache, writes, attends)
    - **graph_no_roll**: subsequent calls per chunk (overwrites, attends)

    Both graphs share the same static input buffers.
    """

    def __init__(
        self,
        transformer,
        kv_idx: KVCacheSteadyStateIndices,
    ) -> None:
        self.transformer = transformer
        self.kv_idx = kv_idx
        self._pool = torch.cuda.graph_pool_handle()

        # Two graphs: with/without KV cache roll
        self._graph_roll = _SingleGraph()
        self._graph_no_roll = _SingleGraph()

        # Shared static input buffers (allocated during capture)
        self._s_hidden_states: torch.Tensor | None = None
        self._s_encoder_hidden_states: torch.Tensor | None = None
        self._s_timestep: torch.Tensor | None = None
        self._s_freqs_cos: torch.Tensor | None = None
        self._s_freqs_sin: torch.Tensor | None = None
        self._s_c2ws_plucker_emb: torch.Tensor | None = None
        self._s_encoder_hidden_states_image: torch.Tensor | None = None

        # KV cache references (persistent, graph uses their addresses)
        self._kv_cache: list[dict] | None = None

    @property
    def is_captured(self) -> bool:
        return self._graph_roll.graph is not None

    def capture(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        kv_cache: list[dict],
        c2ws_plucker_emb: torch.Tensor | None = None,
        encoder_hidden_states_image: torch.Tensor | None = None,
    ) -> None:
        """Capture both graph variants.

        Runs warmup iterations on a side stream, then captures each graph.
        """
        assert not self.is_captured, "Graphs already captured"
        device = hidden_states.device
        self._kv_cache = kv_cache

        # Create shared static input buffers
        self._s_hidden_states = hidden_states.clone()
        self._s_encoder_hidden_states = encoder_hidden_states.clone()
        self._s_timestep = timestep.clone()
        self._s_freqs_cos = freqs_cis[0].clone()
        self._s_freqs_sin = freqs_cis[1].clone()
        if c2ws_plucker_emb is not None:
            self._s_c2ws_plucker_emb = c2ws_plucker_emb.clone()
        if encoder_hidden_states_image is not None:
            self._s_encoder_hidden_states_image = encoder_hidden_states_image.clone()

        s_freqs_cis = (self._s_freqs_cos, self._s_freqs_sin)
        common_kwargs = dict(
            encoder_hidden_states_image=self._s_encoder_hidden_states_image,
            c2ws_plucker_emb=self._s_c2ws_plucker_emb,
        )

        for name, forward_fn, sg in [
            ("with_roll", self.transformer.forward_cuda_graph_with_roll, self._graph_roll),
            ("no_roll", self.transformer.forward_cuda_graph_no_roll, self._graph_no_roll),
        ]:
            # Warmup
            logger.info("CUDA graph warmup (%s)", name)
            s = torch.cuda.Stream(device=device)
            s.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(s):
                forward_fn(
                    self._s_hidden_states,
                    self._s_encoder_hidden_states,
                    self._s_timestep,
                    s_freqs_cis,
                    self._kv_cache,
                    self.kv_idx,
                    **common_kwargs,
                )
            torch.cuda.current_stream(device).wait_stream(s)

            # Capture
            logger.info("Capturing CUDA graph (%s)", name)
            sg.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(sg.graph, pool=self._pool):
                sg.s_output = forward_fn(
                    self._s_hidden_states,
                    self._s_encoder_hidden_states,
                    self._s_timestep,
                    s_freqs_cis,
                    self._kv_cache,
                    self.kv_idx,
                    **common_kwargs,
                )

        logger.info("Both CUDA graphs captured successfully")

    def _copy_inputs(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        c2ws_plucker_emb: torch.Tensor | None,
    ) -> None:
        """Copy per-call changing inputs to static buffers."""
        self._s_hidden_states.copy_(hidden_states)
        self._s_timestep.copy_(timestep)
        self._s_freqs_cos.copy_(freqs_cis[0])
        self._s_freqs_sin.copy_(freqs_cis[1])
        if c2ws_plucker_emb is not None and self._s_c2ws_plucker_emb is not None:
            self._s_c2ws_plucker_emb.copy_(c2ws_plucker_emb)

    def replay_with_roll(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        c2ws_plucker_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """First call per chunk: copy inputs, replay graph with KV roll."""
        self._copy_inputs(hidden_states, timestep, freqs_cis, c2ws_plucker_emb)
        self._graph_roll.graph.replay()
        return self._graph_roll.s_output.clone()

    def replay_no_roll(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        c2ws_plucker_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Subsequent calls per chunk: copy inputs, replay graph without KV roll."""
        self._copy_inputs(hidden_states, timestep, freqs_cis, c2ws_plucker_emb)
        self._graph_no_roll.graph.replay()
        return self._graph_no_roll.s_output.clone()

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
        """Release graphs and static buffers."""
        for sg in (self._graph_roll, self._graph_no_roll):
            if sg.graph is not None:
                del sg.graph
                sg.graph = None
            sg.s_output = None
        self._s_hidden_states = None
        self._s_encoder_hidden_states = None
        self._s_timestep = None
        self._s_freqs_cos = None
        self._s_freqs_sin = None
        self._s_c2ws_plucker_emb = None
        self._s_encoder_hidden_states_image = None
        self._kv_cache = None
