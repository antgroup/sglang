"""DeepSeek-V4.1 interleave CP prefill collectives on SM90 (``dsv41_sm90_on("cp")``).

* Attention: the SWA KV all-gather and the low-ratio compressor's projection
  gather start on the copy engines before q_b and finish after it. The
  compressor projects this rank's rows and gathers the projections, not the
  hidden rows.
* MoE (``CpMoeSymmTransport``): the input rows travel over symmetric memory
  while the TP-sharded shared expert runs on the shards that have landed; the
  router may run on the CP-local rows, its top-k riding the same gather, and
  then each rank sends its rows already quantized for both experts; one launch
  writes the deferred humming weighted sum plus the shared output of every
  owner's rows into its slot, summed there in fp32 (one rounding).
* Input embedding and engram lookup: the owner of each row writes it to the CP
  rank that holds the token instead of an all-reduce of zero-filled rows.

Every gather and owner write is bitwise the NCCL collective it replaces; the
reduce-scatter is not (one fp32 rounding instead of one per ring hop). The router
runs at CP-local row counts only where its GEMM gives the gathered logits
(``_LOCAL_ROUTER_ROWS``). The compressor projects CP-local rows at any count; on
H20-3e that is bitwise the gathered projection except at 1024 rows per rank.
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from sglang.kernels.ops.communication.cp_prefill_symm import embed_owner_scatter
from sglang.kernels.ops.quantization.fp8_block32_act_quant import (
    block32_act_quant_ue8m0_mn_major,
)
from sglang.srt.environ import envs
from sglang.srt.layers.communication.cp_prefill_symm import (
    CpPendingGather,
    CpPrefillSymm,
    cp_group_is_tp_group,
    get_cp_owner_scatter,
    get_cp_prefill_symm,
)
from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.layers.cp.interleave import (
    InterleaveCPStrategy,
    interleave_gather_layout,
    interleave_global_order,
)
from sglang.srt.layers.cp.utils import cp_materialize_global_token_order
from sglang.srt.layers.layer_boundary.layout import TokenAxis
from sglang.srt.layers.layer_boundary.row_transport import ComputeTransport
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.models.deepseek_common.moe_transport import _every_row_routed
from sglang.srt.multimodal.dsv41.vl_routing import vision_topk
from sglang.srt.runtime_context import get_exec, get_forward, get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.humming import HummingDeferredMulSum

# CP-local slot rows from which the KV and compressor gathers use copy engines;
# measured on H20-3e, where shorter chunks keep NCCL.
_ATTN_SYMM_MIN_ROWS = 1024
# CP-local rows from which the MoE takes symmetric memory. H20-3e CP4 layer GPU
# time: -546 us at 4096 rows, -281 at 2048, -136 at 1024, where the extra
# launches cost more than that unless the host runs ahead.
_MOE_SYMM_MIN_ROWS = 2048
# (CP-local rows, CP size) -> rows to route them as: the 384 x 5120 router GEMM
# gives these the logits of the gathered CP * rows GEMM bit for bit, with cuBLAS
# and DeepGEMM, on the devices below (test_dsv41_cp_local_router_gemm.py). 1024
# rows under CP8 do not, so they route as 2048 repeated rows; other shapes route
# the gathered rows.
_LOCAL_ROUTER_ROWS = {
    (1024, 8): 2048,
    (2048, 4): 2048,
    (2048, 8): 2048,
    (4096, 4): 4096,
}
_LOCAL_ROUTER_DEVICES = ("NVIDIA H20", "NVIDIA H20-3e")
# int32 side-payload columns behind the top-k for the fp8 gather's humming scales
# (one per row with per-row fp8; arbitrary room for a few groups).
_MAX_SCALE_COLS = 4
_LOCAL_ROUTER_WEIGHT_SHAPE = (384, 5120)


def _interleave_plan(
    *,
    local_rows: int,
    forward_batch,
    user: str,
    width: int,
) -> Optional[Tuple[CpPrefillSymm, int, int]]:
    """(buffers, rows sent, rows per slot) of a symmetric interleave CP all-gather
    of ``local_rows``-row parts, or None for the NCCL path."""
    if (
        not isinstance(get_cp_strategy(), InterleaveCPStrategy)
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    parallel = get_parallel()
    send_rows, slot_rows = interleave_gather_layout(
        forward_batch.attn_cp_metadata,
        cp_rank=parallel.attn_cp_rank,
        cp_size=parallel.attn_cp_size,
        local_rows=local_rows,
    )
    # slot_rows is the largest rank's row count, so every rank agrees here.
    if slot_rows < _ATTN_SYMM_MIN_ROWS:
        return None
    comm = get_cp_prefill_symm(parallel.attn_cp_group, user=user, width=width)
    return None if comm is None else (comm, send_rows, slot_rows)


def _start_interleave_gather(
    plan: Tuple[CpPrefillSymm, int, int], parts: List[torch.Tensor], forward_batch
) -> Optional[Callable[[], Tuple[torch.Tensor, ...]]]:
    """Start the planned gather of ``parts`` (this rank's rows); the returned call
    gives each part's rows of every rank in global token order."""
    comm, send_rows, slot_rows = plan
    parts = [part[:send_rows] for part in parts]
    # Shapes and dtypes only, so every rank takes the same branch.
    if not comm.fits(parts, rows=slot_rows):
        return None
    pending = comm.start_gather(parts, rows=slot_rows)
    metadata = forward_batch.attn_cp_metadata
    cp_size = get_parallel().attn_cp_size
    return lambda: tuple(
        interleave_global_order(
            part, metadata, cp_size=cp_size, physical_rank_len=slot_rows
        )
        for part in pending.finish_parts()
    )


def start_cp_kv_gather(
    *, kv: torch.Tensor, forward_batch
) -> Optional[Callable[[], torch.Tensor]]:
    """Start the CP all-gather of this rank's SWA KV rows on the copy engines; the
    returned call finishes it, bitwise cp_materialize_global_token_order. None:
    take that path instead."""
    plan = _interleave_plan(
        local_rows=kv.shape[0],
        forward_batch=forward_batch,
        user="kv",
        width=kv.shape[1],
    )
    if plan is None:
        return None
    finish = _start_interleave_gather(plan, [kv], forward_batch)
    return None if finish is None else lambda: finish()[0]


def start_cp_compressor_gather(
    *, compressor, x: torch.Tensor, forward_batch
) -> Callable[[], Tuple[Optional[torch.Tensor], ...]]:
    """Project this rank's rows for the low-ratio compressor and start their CP
    all-gather; the returned call gives the projections of every CP row in global
    token order (``forward_low_ratio_sources(cp_projected=...)``)."""
    projected = compressor.project(x)
    parts = [value for value in projected if value is not None]
    total = int(forward_batch.attn_cp_metadata.total_seq_lens)
    plan = _interleave_plan(
        local_rows=x.shape[0],
        forward_batch=forward_batch,
        user="compressor",
        # The widest row, ratio 2's fp32 kv and score, in bf16 columns; ratio 1
        # shares these buffers.
        width=4 * parts[0].shape[1],
    )
    finish = (
        _start_interleave_gather(plan, parts, forward_batch)
        if plan is not None
        else None
    )
    if finish is None:

        def finish() -> Tuple[torch.Tensor, ...]:
            return tuple(
                cp_materialize_global_token_order(
                    part.contiguous(), forward_batch, torch.cuda.current_stream()
                )
                for part in parts
            )

    def finish_projected() -> Tuple[Optional[torch.Tensor], ...]:
        gathered = iter(finish())
        return tuple(
            next(gathered)[:total] if value is not None else None for value in projected
        )

    return finish_projected


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


def cp_moe_symm_supported(mlp) -> bool:
    """Whether ``mlp`` splits into what CpMoeSymmTransport runs: routed experts
    on the Humming runner and a separate, TP-sharded shared expert whose output
    the MoE adds to the routed rows before the TP sum."""
    return (
        mlp.shared_experts is not None
        and mlp.num_fused_shared_experts == 0
        and not mlp._shared_expert_tp1
        and not mlp._fuse_shared_experts_inside_sbo
        and not mlp._enable_a2a_moe
        and not mlp.reduce_results
        and get_moe_runner_backend().is_humming()
        and not envs.SGLANG_OPT_MOE_QUANT_ONCE.get()
    )


def cp_local_router_supported(mlp) -> bool:
    """Whether ``mlp`` may route a CP rank's own rows and take their gathered
    top-k: a Humming runner reads only the ids and weights, and nothing else
    observes the rows routed together (EPLB, recorders, expert simulation)."""
    gate = mlp.gate
    return (
        get_moe_runner_backend().is_humming()
        and torch.cuda.get_device_name() in _LOCAL_ROUTER_DEVICES
        and gate.weight.dtype == torch.bfloat16
        and tuple(gate.weight.shape) == _LOCAL_ROUTER_WEIGHT_SHAPE
        and envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get() in ("cublas", "deep_gemm")
        and not get_exec().deterministic.enable_deterministic_inference
        and not get_exec().moe.enable_eplb
        and get_exec().moe.expert_distribution_recorder_mode is None
        and not get_exec().features.enable_return_routed_experts
        and not envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
        and not envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get()
    )


def _pack_topk(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
    """[rows, 2 * topk] int32: the ids, then the fp32 weights' bits."""
    return torch.cat([topk_ids, topk_weights.view(torch.int32)], dim=1)


def _unpack_topk(packed: torch.Tensor, topk: int) -> StandardTopKOutput:
    return StandardTopKOutput(
        packed[:, topk : 2 * topk].contiguous().view(torch.float32),
        packed[:, :topk].contiguous(),
        None,
    )


def _routes_by_token_id(mlp) -> bool:
    # Hash routing, and the VL bias that image-token rows take, read each row's id.
    return mlp.is_hash or mlp.gate.e_score_correction_bias_vl is not None


def _route_local_rows(
    mlp, hidden_states: torch.Tensor, input_ids_global, *, route_rows: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(top-k ids, weights) of this rank's CP rows, routed as ``route_rows`` rows
    (the rows repeated), or None when their dtypes cannot ride the side payload."""
    rows = hidden_states.shape[0]
    parallel = get_parallel()
    local_ids = None
    if _routes_by_token_id(mlp):
        # Rank-major like the gathered rows: this rank's ids are one block.
        local_ids = input_ids_global.tensor_split(parallel.attn_cp_size)[
            parallel.attn_cp_rank
        ]
    route_states = hidden_states
    if route_rows != rows:
        repeats = route_rows // rows
        route_states = hidden_states.repeat(repeats, 1)
        if local_ids is not None:
            local_ids = local_ids.repeat(repeats)
    logits = mlp.gate(route_states)
    # The top-k the gathered rows take in DeepseekV2MoE.forward_normal.
    if mlp.gate.e_score_correction_bias_vl is not None:
        topk = vision_topk(mlp, logits, local_ids)
    else:
        kwargs = {"input_ids": local_ids} if mlp.is_hash else {}
        topk = mlp.topk(route_states, logits, **kwargs)
    ids, weights = topk.topk_ids[:rows], topk.topk_weights[:rows]
    if ids.dtype != torch.int32 or weights.dtype != torch.float32:
        return None
    return ids, weights


def shared_on_landed_shards(
    shared_fn: Callable[[torch.Tensor], torch.Tensor], pending: CpPendingGather
) -> List[torch.Tensor]:
    """``shared_fn`` (the TP-sharded shared expert) of every shard's rows, one call
    per block of landed shards; entry p holds shard p's rows. Bitwise one call on
    all the rows only where its GEMMs give a row the same bits at every row count."""
    rows = pending.rows
    shared: List[Optional[torch.Tensor]] = [None] * pending.world_size
    for lo, hi in pending.runs():
        out = shared_fn(pending.rows_of(lo, hi))
        for p in range(lo, hi):
            shared[p] = out[(p - lo) * rows : (p - lo + 1) * rows]
    return shared


def reduce_scatter_with_shared(
    comm: CpPrefillSymm,
    *,
    routed: torch.Tensor,
    shared: List[torch.Tensor],
    rows: int,
    deferred: Optional["HummingDeferredMulSum"] = None,
) -> torch.Tensor:
    """This rank's ``rows`` rows of the TP sum of ``routed + shared`` ([W * rows,
    hidden] partials on every rank): the MoE's ``routed += shared`` (one bf16
    rounding) is written straight into each owner's slot, then summed there.

    With ``deferred`` (the humming run left its weighted sum unwritten, ``routed``
    is not read) one launch computes every owner's rows and writes them there."""
    if deferred is not None:

        def write_all(outs: List[torch.Tensor]) -> None:
            deferred.write_all_rows_with_shared_add(
                rows=rows, rank=comm.rank, outputs=outs, shared=shared
            )

        return comm.reduce_scatter(rows, write_all=write_all)

    def write_rows(p: int, out: torch.Tensor) -> None:
        torch.add(routed[p * rows : (p + 1) * rows], shared[p], out=out)

    return comm.reduce_scatter(rows, write_rows)


def _fp8_gather_cols(hidden: int) -> int:
    # bf16 columns of an fp8-gathered row: two fp8 copies of it (the humming w13
    # input, the shared gate_up input) and its fp32 per-32-column scales.
    return hidden + hidden // 32 * 2


def _shared_gate_up_takes_prequant(mlp, rows: int) -> bool:
    """Whether the TP-sharded shared expert's gate_up GEMM, run on blocks of ``rows``
    and ``2 * rows`` gathered rows, takes their block32 ue8m0 quant (row-major
    scales) to the GEMM its bf16 input would take: the SM90 block32 routes put
    neither row count on the swap-AB or Triton GEMM, so both are DeepGEMM."""
    from sglang.kernels.ops.gemm.fp8_block32_swapab import lookup_swapab
    from sglang.srt.layers.quantization.fp8 import (
        Fp8Config,
        Fp8LinearMethod,
        _dsv41_block32_routes,
    )
    from sglang.srt.layers.quantization.fp8_utils import (
        _deepgemm_block_fp8_supported,
    )

    gate_up = mlp.shared_experts.gate_up_proj
    method = gate_up.quant_method
    if not (
        isinstance(method, Fp8LinearMethod)
        and isinstance(method.quant_config, Fp8Config)
        and method.block_quant
        and not method.use_mxfp8
        and not method.block_fp8_as_mxfp8
        and list(method.weight_block_size) == [32, 32]
        and method.quant_config.scale_fmt == "ue8m0"
    ):
        return False
    routes = _dsv41_block32_routes(weight_block_size=[32, 32], act_scale_ue8m0=True)
    n, k = gate_up.weight.shape
    return (
        routes is not None
        and _deepgemm_block_fp8_supported((n, k), [32, 32], torch.bfloat16)
        and all(
            m > routes.triton_max_m and lookup_swapab(routes.swapab, n, k, m) is None
            for m in (rows, 2 * rows)
        )
    )


def _start_fp8_gather(
    comm: CpPrefillSymm,
    mlp,
    hidden_states: torch.Tensor,
    topk: Tuple[torch.Tensor, torch.Tensor],
) -> Optional[Tuple[CpPendingGather, int]]:
    """Gather this rank's rows already quantized for both consumers instead of the
    bf16 rows they would each quantize again on every rank: the humming w13 input
    (fp8, per-row scales riding the side payload behind the top-k) and the shared
    gate_up input (block32 fp8 + scales), about the bf16 bytes. Both quants are
    row-local, so the consumers' results are bitwise. None: not this layer."""
    rows = hidden_states.shape[0]
    if not _shared_gate_up_takes_prequant(mlp, rows):
        return None
    from sglang.srt.layers.moe.moe_runner.humming import humming_quant_input_rows

    quant = humming_quant_input_rows(mlp.experts, hidden_states)
    if quant is None:
        return None
    q, scale = quant
    q_shared, scale_shared = block32_act_quant_ue8m0_mn_major(hidden_states)
    side = torch.cat([_pack_topk(*topk), scale.view(torch.int32)], dim=1)
    parts = [q, q_shared, scale_shared.contiguous()]
    if not (comm.fits(parts) and comm.fits_side(side, rows)):
        return None
    return comm.start_gather_planar(parts, side=side), scale.shape[1]


def _shared_on_landed_planes(mlp, pending: CpPendingGather) -> List[torch.Tensor]:
    """shared_on_landed_shards of an fp8 gather: the shared expert takes each
    block's gathered block32 quant as its gate_up input."""
    rows = pending.rows
    shared: List[Optional[torch.Tensor]] = [None] * pending.world_size
    for lo, hi in pending.runs():
        _, q, scale = pending.planes_of(lo, hi)
        out = mlp._forward_shared_experts(q, pre_quant_input=(q, scale))
        for p in range(lo, hi):
            shared[p] = out[(p - lo) * rows : (p - lo + 1) * rows]
    return shared


class CpMoeSymmTransport(ComputeTransport):
    """The CP rows of one V4.1 MoE invocation over symmetric memory, in place of
    MoeTransport's NCCL all-gather and all-reduce; same source rows and sum."""

    def __init__(
        self,
        *,
        layer,
        hidden_states: torch.Tensor,
        forward_batch,
        call,
        rows,
        comm: CpPrefillSymm,
        local_router: bool,
    ):
        mlp = layer.mlp
        self.rows = rows
        self.source_rows = hidden_states.shape[0]
        self.input_ids, self.input_ids_global = call.input_ids, call.input_ids_global
        # The shared expert runs here, on the shards as they land.
        self.do_shared_local = True
        self.return_moe_output = False
        self.sum_in_return = True
        self._comm = comm
        self._batch = forward_batch
        self._deferred: Optional[HummingDeferredMulSum] = None
        # (fp8 gathered rows, their humming scales) of an fp8 gather, else None.
        self._pre_quant: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        top_k = mlp.config.num_experts_per_tok
        topk = self._route_local_rows(
            mlp=mlp, hidden_states=hidden_states, local_router=local_router
        )
        fp8 = (
            _start_fp8_gather(comm, mlp, hidden_states, topk)
            if topk is not None
            else None
        )
        if fp8 is not None:
            pending, scale_cols = fp8
            self._shared = _shared_on_landed_planes(mlp, pending)
            q = pending.finish_parts()[0]
            side = pending.finish_side()
            scale = side[:, 2 * top_k : 2 * top_k + scale_cols].contiguous()
            self._pre_quant = (q, scale.view(torch.float32))
            self.input = q
            self.topk_output = _unpack_topk(side, top_k)
        else:
            side = _pack_topk(*topk) if topk is not None else None
            if side is not None and not comm.fits_side(side, self.source_rows):
                side = None
            pending = comm.start_gather(hidden_states, side=side)
            self._shared = shared_on_landed_shards(mlp._forward_shared_experts, pending)
            self.input = pending.finish()
            self.topk_output = (
                _unpack_topk(pending.finish_side(), top_k) if side is not None else None
            )
        self._gathered_rows = _every_row_routed(forward_batch, self.input.shape[0])

    def _route_local_rows(
        self, *, mlp, hidden_states: torch.Tensor, local_router: bool
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        rows = hidden_states.shape[0]
        route_rows = _LOCAL_ROUTER_ROWS.get((rows, get_parallel().attn_cp_size))
        if not (
            local_router
            and route_rows is not None
            and rows > mlp.gate.tiny_router_gemm_max_tokens
            and (self.input_ids_global is not None or not _routes_by_token_id(mlp))
        ):
            return None
        return _route_local_rows(
            mlp, hidden_states, self.input_ids_global, route_rows=route_rows
        )

    def run_experts(self, mlp) -> torch.Tensor:
        """The routed experts' rows of the gathered input, scaled, not TP-summed
        and without the shared expert's output. An indexed humming run leaves its
        weighted sum to return_output (the returned rows are then unwritten)."""
        from sglang.srt.layers.moe.moe_runner.humming import (
            humming_deferred_mul_sum_context,
        )

        # The MoE's own deferral is for the shared add it skips here, so the
        # humming run sees this context, also through mlp().
        with humming_deferred_mul_sum_context() as deferred:
            if self.topk_output is not None:
                routed = mlp.experts(
                    self.input, self.topk_output, pre_quant_input=self._pre_quant
                )
            else:
                routed = mlp(
                    self.input,
                    self._batch,
                    input_ids=self.input_ids,
                    input_ids_global=self.input_ids_global,
                    skip_shared_experts=True,
                    leave_output_sum=True,
                ).routed
        if deferred.mul_sum_kwargs is not None:
            self._deferred = deferred
        return routed

    @contextmanager
    def scope(self):
        with (
            get_forward().scoped(mlp_reduce_scatter=True, defer_moe_finalize=False),
            self._gathered_rows,
        ):
            yield

    def prepare_output(self, routed: torch.Tensor) -> torch.Tensor:
        self.input = self._pre_quant = None
        return routed

    def return_output(self, routed: torch.Tensor) -> torch.Tensor:
        value = reduce_scatter_with_shared(
            self._comm,
            routed=routed,
            shared=self._shared,
            rows=self.source_rows,
            deferred=self._deferred,
        )
        self._shared = self._deferred = None
        return value

    def release(self):
        self.input = self._shared = self.topk_output = None
        self._deferred = self._pre_quant = None


def bind_cp_moe_symm_transport(
    layer, hidden_states: torch.Tensor, forward_batch, call, rows
) -> Optional[CpMoeSymmTransport]:
    """The symmetric-memory transport for this invocation, or None for
    MoeTransport. Every input of the decision is rank-uniform under CP."""
    parallel = get_parallel()
    if (
        TokenAxis.ATTN_CP not in rows.source.sharded
        or not get_moe_a2a_backend().is_none()
        or parallel.attn_dp_size != 1
        or parallel.attn_tp_size != 1
        or parallel.tp_size != parallel.attn_cp_size
        or hidden_states.dim() != 2
        or hidden_states.shape[0] < _MOE_SYMM_MIN_ROWS
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    comm = get_cp_prefill_symm(
        parallel.attn_cp_group,
        user="moe",
        width=hidden_states.shape[1],
        with_reduce=True,
        side_cols=2 * layer.mlp.config.num_experts_per_tok + _MAX_SCALE_COLS,
        gather_cols=_fp8_gather_cols(hidden_states.shape[1]),
    )
    if comm is None or not comm.fits(hidden_states):
        return None
    return CpMoeSymmTransport(
        layer=layer,
        hidden_states=hidden_states,
        forward_batch=forward_batch,
        call=call,
        rows=rows,
        comm=comm,
        local_router=layer.cp_local_router,
    )


# ---------------------------------------------------------------------------
# Owner-scattered lookups
# ---------------------------------------------------------------------------


def cp_shard_input_embeds(
    embed, input_ids: torch.Tensor, forward_batch
) -> Optional[torch.Tensor]:
    """This CP rank's interleaved rows of the vocab-parallel embedding of the
    chunk, each written by its id's vocab-shard owner: the rows the full-chunk
    all-reduce and shard would give. None: take that path."""
    if not isinstance(embed, VocabParallelEmbedding):
        return None
    metadata = forward_batch.attn_cp_metadata
    parallel = get_parallel()
    cp, num_tokens = parallel.attn_cp_size, input_ids.shape[0]
    shard = embed.shard_indices
    if (
        not isinstance(get_cp_strategy(), InterleaveCPStrategy)
        or not cp_group_is_tp_group()
        or embed.tp_size != cp
        or embed.weight.dtype != torch.bfloat16
        or not embed.weight.is_contiguous()
        or shard.added_vocab_end_index != shard.added_vocab_start_index
        or num_tokens % cp != 0
        or int(metadata.total_seq_lens) != num_tokens
        or metadata.per_rank_logical_token not in (None, metadata.per_rank_actual_token)
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    comm = get_cp_owner_scatter(
        parallel.attn_cp_group, user="embed", width=embed.embedding_dim
    )
    rows = num_tokens // cp
    if comm is None or rows > comm.max_rows:
        return None
    embed_owner_scatter(
        input_ids,
        embed.weight,
        comm.dst_table,
        shard.org_vocab_start_index,
        shard.org_vocab_end_index,
    )
    return comm.finish(rows)
