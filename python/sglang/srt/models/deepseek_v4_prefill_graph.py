# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Breakable CUDA graphs of DeepSeek-V4.1 prefill-CP layer ranges.

Under prefill context parallelism a decoder layer launches about 60 kernels and
costs about 4 ms of host time whatever its row count, so short prompts are
host-bound. With the ``prefill_graph`` subsystem on, the decoder layers run from
breakable CUDA graphs keyed on the exact CP-local row count: every layer, or under
decoder SWA bounded replay the layers before the switch (head) and from it (tail).
The ops that read per-forward metadata run eagerly between graph segments against
the live batch: in every layer the CP KV gather, the window-KV store, the
compressor and indexer and the attention core (one break), the Engram lookup, and
the DSpark aux capture.

Rows are never padded: a range replays only when its row count equals a captured
one, so the graph runs the kernels eager prefill runs on the same shapes and is
bitwise identical to it. A shape is captured on its second forward, by every rank
in lockstep (interleave CP shards are rank-uniform); breaks are no-op stubs while
capturing, and the first replay computes the forward that triggered the capture.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.cp.utils import cp_materialize_global_token_order
from sglang.srt.layers.dsv41_sm90 import dsv41_sm90_on
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_capture_stream,
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)
from sglang.srt.models import deepseek_v4_boundary as hc_boundary
from sglang.srt.models.deepseek_v4_cp_prefill import (
    start_cp_compressor_gather,
    start_cp_kv_gather,
)
from sglang.srt.models.deepseek_v4_mhc import (
    HcCombined,
    HcNormed,
    HcQuantized,
    HcState,
)
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

FULL = "full"
HEAD = "head"
TAIL = "tail"

# Largest CP-local rows a range is captured for, by device. Measured on H20 and
# H20-3e (TP8 + prefill CP8, with and without PD): a 2048-row head cost 2.5 GiB
# with no TTFT gain. A device without an entry keeps the eager path.
_MAX_ROWS = {
    "NVIDIA H20": {FULL: 512, HEAD: 512, TAIL: 128},
    "NVIDIA H20-3e": {FULL: 512, HEAD: 512, TAIL: 128},
}
# Graphs kept per range, least recently used evicted first; measured on H20-3e.
_MAX_GRAPHS = 32
# Eager forwards of a shape before it is captured: they JIT-compile its kernels.
_WARMUP = 1
# Capture needs this much free device memory on every rank; a capture under
# exhausted memory segfaulted inside NCCL.
_MIN_FREE_BYTES = 1 << 30
_LOG_EVERY = 200


class _Live:
    """What the eager breaks read on replay; they run with their capture-time
    arguments, so per-forward state comes from here."""

    capturing: bool = False
    forward_batch: Optional[ForwardBatch] = None
    tail: Optional[LateLayerTail] = None
    # The DSpark aux packer the breaks write into (AuxHiddenStatePacker.slot).
    packer: Any = None


_live = _Live()


def is_capturing() -> bool:
    """True while a layer range is being captured: the attention defers its CP
    span to `cp_attention` and the range body routes Engram and DSpark aux
    through their breaks."""
    return _live.capturing


class CpAttentionInputs(NamedTuple):
    """The activations the CP attention span reads, deferred to its break."""

    x: torch.Tensor
    q_lora: torch.Tensor
    kv: torch.Tensor
    positions: torch.Tensor


def _cp_attention(layer, x, q_lora, kv, positions, q, attn_sink, o) -> None:
    # MQALayer._forward_prepare's CP branch from the gather on, its sources and
    # MQALayer.forward's attention core, on the live batch. The cp gathers start
    # as MQALayer._start_cp_prefill_gathers starts them before q_b.
    forward_batch = _live.forward_batch
    attn_backend = get_attn_backend()
    finish_kv = finish_projected = None
    if layer.cp_prefill_symm:
        finish_kv = start_cp_kv_gather(kv=kv, forward_batch=forward_batch)
        if layer.compress_ratio in (1, 2) and attn_backend.cp_low_ratio_compressor_runs(
            layer, forward_batch
        ):
            finish_projected = start_cp_compressor_gather(
                compressor=layer.compressor, x=x, forward_batch=forward_batch
            )
    kv = (
        finish_kv()
        if finish_kv is not None
        else cp_materialize_global_token_order(
            kv, forward_batch, torch.cuda.current_stream()
        )
    )
    tail = attn_backend.forward_metadata.late_layer_tail
    global_positions = (
        tail.pos_global if tail is not None else forward_batch.positions[: kv.shape[0]]
    )
    get_token_to_kv_pool().set_swa_key_buffer_radix_fused_norm_rope(
        layer_id=layer.layer_id,
        swa_loc=attn_backend.get_swa_out_cache_loc(forward_batch),
        kv=kv,
        kv_weight=layer.kv_norm.weight.data,
        eps=layer.eps,
        freqs_cis=layer.freqs_cis,
        positions=global_positions,
    )
    del kv
    if layer.compress_ratio in (1, 2) and (
        layer.compressor is not None or layer.indexer is not None
    ):
        attn_backend.forward_low_ratio_sources(
            layer=layer,
            x=x,
            q_lora=q_lora,
            positions=positions,
            forward_batch=forward_batch,
            cp_projected=None if finish_projected is None else finish_projected(),
        )
    else:
        if layer.indexer is not None:
            layer.indexer(
                x=x,
                q_lora=q_lora,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
                q_lora_quant=None,
            )
        if layer.compressor is not None:
            attn_backend.forward_core_compressor(
                x, forward_batch, layer.layer_id, layer.compressor
            )
    out = attn_backend.forward(
        q=q,
        k=q,
        v=q,
        layer=layer.attn_mqa,
        forward_batch=forward_batch,
        compress_ratio=layer.compress_ratio,
        attn_sink=attn_sink,
        save_kv_cache=False,
    )
    if out.dtype != o.dtype or out.numel() != o.numel():
        raise RuntimeError(
            f"CP attention returned {out.dtype} {tuple(out.shape)}, the graph "
            f"holds {o.dtype} {tuple(o.shape)}"
        )
    o.view(out.shape).copy_(out)


def _no_op(*args, **kwargs) -> None:
    return None


cp_attention = eager_on_graph(_cp_attention, capture_stub=_no_op)


def _engram_lookup(engram, x, hash_ids, input_ids, out) -> None:
    out.copy_(
        engram(
            x, hash_ids, _live.forward_batch, input_ids=input_ids, cp_all_tokens=True
        )
    )


engram_lookup = eager_on_graph(_engram_lookup, capture_stub=_no_op)


def _write_dspark_aux(residual, tail, packer) -> None:
    from sglang.kernels.ops.layernorm.mhc import mhc_stream_mean

    aux = residual if tail is None else tail.rows(residual)
    mhc_stream_mean(aux, out=packer.slot(aux[:, 0]))


def _dspark_aux(residual, use_tail_rows: bool) -> None:
    _write_dspark_aux(residual, _live.tail if use_tail_rows else None, _live.packer)


dspark_aux = eager_on_graph(_dspark_aux, capture_stub=_no_op)


# ---------------------------------------------------------------------------
# The mHC state crossing a range boundary, as graph inputs and outputs
# ---------------------------------------------------------------------------


def flatten_state(state: HcState) -> tuple:
    """The state's tensors and the structure `rebuild_state` needs; the structure
    is part of the graph key."""
    assert isinstance(state.streams, torch.Tensor), "prefill graphs are CUDA only"
    leaves = [state.streams]
    if state.pre is not None:
        leaves.append(state.pre)
    input_kind = None
    if isinstance(state.input, (HcNormed, HcCombined)):
        input_kind = type(state.input)
        leaves.append(state.input.rows)
    elif isinstance(state.input, HcQuantized):
        pair = state.input.pair
        input_kind = (HcQuantized, type(pair), len(pair))
        leaves.extend((state.input.rows, *pair))
    elif state.input is not None:
        raise TypeError(f"unsupported mHC input {type(state.input)}")
    num_stats = None if state.stats is None else len(state.stats)
    if num_stats is not None:
        leaves.extend(state.stats)
    return leaves, (state.pre is not None, input_kind, num_stats)


def rebuild_state(leaves: List[torch.Tensor], spec: tuple) -> HcState:
    has_pre, input_kind, num_stats = spec
    it = iter(leaves)
    streams = next(it)
    pre = next(it) if has_pre else None
    hc_input = None
    if input_kind in (HcNormed, HcCombined):
        hc_input = input_kind(next(it))
    elif input_kind is not None:
        _, pair_type, pair_len = input_kind
        rows = next(it)
        pair = [next(it) for _ in range(pair_len)]
        hc_input = HcQuantized(
            rows, pair_type(*pair) if pair_type is not tuple else tuple(pair)
        )
    stats = None if num_stats is None else tuple(next(it) for _ in range(num_stats))
    return HcState(streams, pre, hc_input, stats)


# ---------------------------------------------------------------------------
# The layers of one range, as the model's layer loop runs them
# ---------------------------------------------------------------------------


class _RangeArgs(NamedTuple):
    start: int
    end: int
    # The range ends at the bounded-replay switch, which reads the state the way
    # the loop's iteration at the switch materializes it.
    ends_at_switch: bool
    state_spec: tuple
    num_state: int


def _run_range(
    model,
    args: _RangeArgs,
    tensors: List[Optional[torch.Tensor]],
    *,
    forward_batch: ForwardBatch,
    tail: Optional[LateLayerTail],
    capture_dspark: bool,
    packer,
    capturing: bool,
) -> tuple:
    """Layers [start, end) of a prefill-CP forward, as
    DeepseekV4Model._forward_layers_hc_pre_from_prev runs them on CUDA; with
    ``capturing`` the Engram lookup and the DSpark aux capture are breaks."""
    n = args.num_state
    state = rebuild_state(tensors[:n], args.state_spec)
    positions, input_ids, input_ids_global, hash_ids = tensors[n:]
    if input_ids_global is None:
        input_ids_global = input_ids
    hidden_states = residual_batch.set_written(state, forward_batch)
    try:
        for i in range(args.start, args.end):
            layer = model.layers[i]
            dspark = capture_dspark and i in model.dspark_layers_to_capture
            if layer.engram is not None or dspark:
                hidden_states = state = _materialize(
                    model, i, hidden_states, input_ids, input_ids_global, forward_batch
                )
            if layer.engram is not None:
                hash_cols = hash_ids[:, layer.engram.layer_hash_index]
                if capturing:
                    residual = torch.empty_like(state.residual)
                    engram_lookup(
                        layer.engram, state.residual, hash_cols, input_ids, residual
                    )
                else:
                    residual = layer.engram(
                        state.residual,
                        hash_cols,
                        forward_batch,
                        input_ids=input_ids,
                        cp_all_tokens=True,
                    )
                state = state.with_residual(residual)
                hidden_states = residual_batch.set_written(state, forward_batch)
            if dspark:
                use_tail_rows = tail is not None and i < model.late_layer_start
                if capturing:
                    dspark_aux(state.residual, use_tail_rows)
                else:
                    _write_dspark_aux(
                        state.residual, tail if use_tail_rows else None, packer
                    )
            ctx = (
                nullcontext()
                if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                else get_global_expert_distribution_recorder().with_current_layer(i)
            )
            with ctx:
                hidden_states = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    forward_batch=forward_batch,
                    input_ids_global=input_ids_global,
                )
        if args.ends_at_switch:
            state = _materialize(
                model,
                args.end,
                hidden_states,
                input_ids,
                input_ids_global,
                forward_batch,
            )
        else:
            state = residual_batch.read_output(
                hidden_states, forward_batch, hc_boundary.FinalRead(), None
            )
            state = state.materialized(model.layers[args.end - 1].hc_cfg)
    finally:
        residual_batch.clear(forward_batch)
    return flatten_state(state)


def _materialize(model, i, hidden_states, input_ids, input_ids_global, batch):
    call = hc_boundary.LayerCall(input_ids, input_ids_global)
    next_boundary = (
        hc_boundary.HcRead(model.layers[i], True).bind(batch, call).next_boundary
    )
    return hc_boundary.materialize(hidden_states, batch, next=next_boundary)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class _Graph:
    __slots__ = ("graph", "inputs", "outputs", "out_spec", "replays", "last_use")

    def __init__(self, graph, inputs, outputs, out_spec):
        self.graph = graph
        self.inputs = inputs
        self.outputs = outputs
        self.out_spec = out_spec
        self.replays = 0
        self.last_use = 0


class _Stats:
    __slots__ = ("eager", "captured", "replayed", "failed", "low_memory")

    def __init__(self):
        self.eager = self.captured = self.replayed = 0
        self.failed = self.low_memory = 0

    def __repr__(self):
        return " ".join(f"{name}={getattr(self, name)}" for name in self.__slots__)


# body(tensors, capturing) -> (output state leaves, output state spec)
Body = Callable[[List[Optional[torch.Tensor]], bool], tuple]


class PrefillGraphRunner:
    """Per-model cache of layer-range graphs. Every decision is a function of the
    shapes, the key and collective votes, so all ranks take the same path."""

    def __init__(
        self,
        *,
        max_rows: Dict[str, int],
        max_graphs: int = _MAX_GRAPHS,
        warmup: int = _WARMUP,
        min_free_bytes: int = _MIN_FREE_BYTES,
    ) -> None:
        self.max_rows = max_rows
        self.max_graphs = max_graphs
        self.warmup = warmup
        self.min_free_bytes = min_free_bytes
        self.graphs: Dict[tuple, _Graph] = {}
        self._seen: Dict[tuple, int] = {}
        self._failed: set = set()
        self._buffers: Dict[tuple, torch.Tensor] = {}
        self._clock = 0
        self.stats = _Stats()

    @classmethod
    def maybe_create(cls, model) -> Optional[PrefillGraphRunner]:
        if not (dsv41_sm90_on("prefill_graph") and is_cuda()):
            return None
        max_rows = _MAX_ROWS.get(torch.cuda.get_device_name())
        parallel = get_parallel()
        if (
            max_rows is None
            or parallel.attn_cp_size == 1
            or parallel.attn_tp_size != 1
            or parallel.attn_dp_size != 1
            or model.pp_group.world_size != 1
            or not model.hc_pre_from_prev_sublayer
            or not get_moe_a2a_backend().is_none()
        ):
            return None
        if parallel.tp_rank == 0:
            logger.info("DSV4 prefill graph on: CP-local row caps %s", max_rows)
        return cls(max_rows=max_rows)

    @staticmethod
    def eligible(forward_batch: ForwardBatch) -> bool:
        """Whether this CP extend may run its layers from graphs."""
        return (
            forward_batch.forward_mode == ForwardMode.EXTEND
            and not get_forward().sp_active
            and not get_global_expert_distribution_recorder().recording
        )

    # -- entry points ------------------------------------------------------------

    def run_layers(
        self,
        model,
        *,
        region: str,
        start: int,
        end: int,
        state: HcState,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_global: torch.Tensor,
        hash_ids: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        tail: Optional[LateLayerTail],
        capture_dspark: bool,
        packer,
    ) -> Optional[HcState]:
        """The state after layers [start, end) from a graph, or None when the
        caller runs them eagerly."""
        leaves, spec = flatten_state(state)
        args = _RangeArgs(start, end, region == HEAD, spec, len(leaves))
        ids_global = None if input_ids_global is input_ids else input_ids_global
        inputs = [*leaves, positions, input_ids, ids_global, hash_ids]
        key = (
            args,
            capture_dspark,
            forward_batch.num_token_non_padded is None,
        )

        def body(tensors, capturing):
            return _run_range(
                model,
                args,
                tensors,
                forward_batch=forward_batch,
                tail=tail,
                capture_dspark=capture_dspark,
                packer=packer,
                capturing=capturing,
            )

        out = self._run(
            region=region,
            key=key,
            inputs=inputs,
            body=body,
            forward_batch=forward_batch,
            tail=tail,
            packer=packer,
        )
        return None if out is None else rebuild_state(*out)

    def _run(self, *, region, key, inputs, body, forward_batch, tail, packer):
        rows = inputs[0].shape[0]
        if not 0 < rows <= self.max_rows[region]:
            self.stats.eager += 1
            return None
        graph_key = (
            region,
            rows,
            key,
            tuple(None if t is None else (tuple(t.shape), t.dtype) for t in inputs),
        )
        graph = self.graphs.get(graph_key)
        if graph is None:
            graph = self._maybe_capture(graph_key, inputs, body, forward_batch, tail)
            if graph is None:
                self.stats.eager += 1
                return None
        self._clock += 1
        graph.last_use = self._clock
        return self._replay(graph, inputs, forward_batch, tail, packer)

    # -- capture -----------------------------------------------------------------

    def _maybe_capture(self, graph_key, inputs, body, forward_batch, tail):
        if graph_key in self._failed:
            return None
        seen = self._seen.get(graph_key, 0)
        self._seen[graph_key] = seen + 1
        if seen < self.warmup or not self._every_rank_has_capture_room():
            return None
        region = graph_key[0]
        if sum(1 for k in self.graphs if k[0] == region) >= self.max_graphs:
            self._evict_lru(region)
        return self._capture(graph_key, inputs, body, forward_batch, tail)

    def _every_rank_has_capture_room(self) -> bool:
        # Replay needs every rank's graph (the segments hold collectives), so the
        # ranks capture together or not at all.
        free, _ = torch.cuda.mem_get_info()
        if _all_ranks(free >= self.min_free_bytes):
            return True
        self.stats.low_memory += 1
        if self.stats.low_memory % _LOG_EVERY == 1:
            logger.warning(
                "DSV4 prefill graph: a rank has under %d MiB free (this rank "
                "%.1f GiB); shapes not captured yet run eagerly (%d so far)",
                self.min_free_bytes >> 20,
                free / 2**30,
                self.stats.low_memory,
            )
        return False

    def _evict_lru(self, region: str) -> None:
        victim = min(
            (k for k in self.graphs if k[0] == region),
            key=lambda k: self.graphs[k].last_use,
        )
        graph = self.graphs.pop(victim)
        logger.info(
            "DSV4 prefill graph evicted %s rows %d (%d replays)",
            victim[0],
            victim[1],
            graph.replays,
        )

    def _static_inputs(self, region, inputs) -> List[Optional[torch.Tensor]]:
        """Fixed-address inputs of a new graph. The graphs of one range never
        replay concurrently and load their inputs right before replay, so they
        share one growing buffer per input slot (views of its prefix)."""
        static = []
        for slot, t in enumerate(inputs):
            if t is None:
                static.append(None)
                continue
            buf_key = (region, slot, t.dtype)
            buf = self._buffers.get(buf_key)
            if buf is None or buf.numel() < t.numel():
                # Earlier graphs keep the old buffer alive through their views.
                buf = self._buffers[buf_key] = torch.empty(
                    t.numel(), dtype=t.dtype, device=t.device
                )
            s = buf[: t.numel()].view(t.shape)
            s.copy_(t)
            static.append(s)
        return static

    def _capture(self, graph_key, inputs, body, forward_batch, tail):
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        ca_comm = get_parallel().tp_group.ca_comm
        if (
            ca_comm is not None
            and not ca_comm.disabled
            and not isinstance(ca_comm, CustomAllReduceV2)
        ):
            # Legacy CA keeps pending graph buffers in C++ and cannot discard a failed capture.
            return None
        start = time.perf_counter()
        free_before = torch.cuda.mem_get_info()[0]
        ok, leaves, out_spec = True, [], None
        stream = get_or_create_global_graph_capture_stream()
        with graph_capture(stream=stream) as ctx:
            try:
                static = self._static_inputs(graph_key[0], inputs)
                pool = get_or_create_global_graph_memory_pool(torch.cuda)
                set_graph_pool_id(pool)
                graph = BreakableCUDAGraph()
                with (
                    self._scope(forward_batch, tail, None, capturing=True),
                    graph_pool_capture_scope(),
                    BreakableCUDAGraphCapture(
                        cuda_graph=graph,
                        pool=pool,
                        stream=ctx.stream,
                        capture_error_mode="thread_local",
                    ),
                ):
                    leaves, out_spec = body(static, True)
            except Exception:  # noqa: BLE001 - every rank votes on the outcome below
                logger.exception("DSV4 prefill graph capture failed for %s", graph_key)
                ok = False
            # CA input registration on context exit must follow the same vote on every rank.
            ok = _all_ranks(ok)
            if not ok:
                if isinstance(ca_comm, CustomAllReduceV2) and not ca_comm.disabled:
                    ca_comm.discard_pending_graph_inputs()
        torch.cuda.current_stream().wait_stream(stream)
        if not ok:
            self._failed.add(graph_key)
            self.stats.failed += 1
            return None
        entry = self.graphs[graph_key] = _Graph(graph, static, leaves, out_spec)
        self.stats.captured += 1
        free_after = torch.cuda.mem_get_info()[0]
        logger.info(
            "DSV4 prefill graph captured %s rows %d: %d segments in %.0f ms, "
            "%.0f MiB, %d graphs, %.1f GiB free",
            graph_key[0],
            graph_key[1],
            len(graph._segments),
            (time.perf_counter() - start) * 1e3,
            (free_before - free_after) / 2**20,
            len(self.graphs),
            free_after / 2**30,
        )
        return entry

    @contextmanager
    def _scope(self, forward_batch, tail, packer, *, capturing: bool):
        saved = (_live.capturing, _live.forward_batch, _live.tail, _live.packer)
        _live.capturing = capturing
        _live.forward_batch = forward_batch
        _live.tail = tail
        _live.packer = packer
        try:
            yield
        finally:
            _live.capturing, _live.forward_batch, _live.tail, _live.packer = saved

    # -- replay ------------------------------------------------------------------

    def _replay(self, graph, inputs, forward_batch, tail, packer):
        for s, t in zip(graph.inputs, inputs):
            if t is not None and s.data_ptr() != t.data_ptr():
                s.copy_(t)
        with self._scope(forward_batch, tail, packer, capturing=False):
            with graph_pool_replay_scope():
                graph.graph.replay()
        graph.replays += 1
        self.stats.replayed += 1
        if self.stats.replayed % _LOG_EVERY == 0:
            logger.info(
                "DSV4 prefill graph stats: %s, graphs %d", self.stats, len(self.graphs)
            )
        # Pool blocks are rewritten by the next replay of any graph in the pool.
        return [None if o is None else o.clone() for o in graph.outputs], graph.out_spec


def _all_ranks(ok: bool) -> bool:
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    torch.distributed.all_reduce(
        flag,
        op=torch.distributed.ReduceOp.MIN,
        group=get_parallel().tp_group.cpu_group,
    )
    return bool(flag.item())
