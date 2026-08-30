"""Per-request error isolation and in-place crash recovery for the scheduler.

A single bad request must not take down the whole server. This module provides
two layers of protection:

1. Request-admission guard (always on): an unexpected exception raised while
   admitting one tokenized request (``handle_generate_request`` /
   ``handle_embedding_request``) aborts only that request with an internal
   error response. Admission runs the same deterministic, content-driven code
   on every TP rank (requests are broadcast from attn-TP rank 0), so a local
   abort decision is rank-consistent, matching the existing whitelisted
   ``set_finish_with_abort`` validation paths.

2. Event-loop recovery (opt-in via ``--enable-scheduler-recovery``): an
   unexpected exception escaping the scheduler event loop (e.g. from the model
   forward path) aborts every in-flight request with an internal error
   response, resets the KV cache and scheduling state, and resumes serving.
   This trades the affected requests for the server's uptime: clients receive
   HTTP 500 and can retry immediately instead of losing the server for a full
   restart and weight reload. Errors that indicate a corrupted process (CUDA
   context poison, communicator failures, host OOM) are never recovered from;
   they keep the existing crash behavior.
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from http import HTTPStatus
from typing import TYPE_CHECKING, Callable, Dict, Optional

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.runtime_context import get_schedule, get_serving
from sglang.srt.utils.weight_versions import compute_weight_version_spans
from sglang.utils import get_exception_traceback

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

# Message substrings that indicate the CUDA/HIP context or the communication
# layer is likely poisoned. Recovering in-place after one of these would risk
# hangs or silent corruption, so they keep the crash behavior.
_FATAL_ERROR_PATTERNS = (
    "cuda error",
    "hip error",
    "illegal memory access",
    "device-side assert",
    "ecc error",
    "uncorrectable",
    "unspecified launch failure",
    "context is destroyed",
    "nccl",
    "rccl",
    "cudnn error",
    "cublas",
)


def is_recoverable_exception(exc: BaseException) -> bool:
    """Return True if aborting requests and resetting state can safely handle
    ``exc``; False if the process must crash (existing behavior)."""
    if not isinstance(exc, Exception):
        # SystemExit / KeyboardInterrupt / GeneratorExit must propagate.
        return False
    if isinstance(exc, MemoryError):
        # Host memory exhaustion: the process itself is not trustworthy.
        return False
    # A failed device allocation is clean: nothing is corrupted, and the
    # post-recovery cache flush frees memory. Check before the message
    # patterns ("CUDA out of memory" is not a context poison).
    oom_types = tuple(
        t
        for t in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(torch.cuda, "OutOfMemoryError", None),
        )
        if isinstance(t, type)
    )
    if oom_types and isinstance(exc, oom_types):
        return True
    dist_error = getattr(torch.distributed, "DistBackendError", None)
    if isinstance(dist_error, type) and isinstance(exc, dist_error):
        return False
    accelerator_error = getattr(torch, "AcceleratorError", None)
    if isinstance(accelerator_error, type) and isinstance(exc, accelerator_error):
        return False
    message = str(exc).lower()
    return not any(pattern in message for pattern in _FATAL_ERROR_PATTERNS)


def _internal_error_finish_reason(message: str) -> Dict:
    return {
        "type": "abort",
        "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
        "message": message,
    }


def _abort_output_for_req(req: Req, message: str) -> AbortReq:
    return AbortReq(
        rid=req.rid,
        finished_reason=_internal_error_finish_reason(message),
        weight_versions=compute_weight_version_spans(
            req.weight_version_events,
            current_version=get_serving().weight_version,
            num_output_tokens=len(req.output_ids),
        ),
    )


# =============================================================================
# Layer 1: request-admission guard
# =============================================================================


def guard_request_admission(
    scheduler: Scheduler, handler: Callable, recv_req
) -> None:
    """Run one tokenized request through ``handler``; on an unexpected
    exception, abort only this request instead of crashing the scheduler."""
    try:
        handler(recv_req)
    except Exception as exc:
        if not is_recoverable_exception(exc):
            raise
        rid = getattr(recv_req, "rid", None)
        if rid is None:
            raise
        logger.error(
            "Unexpected error while admitting request %s; aborting only this "
            "request: %s",
            rid,
            get_exception_traceback(),
        )
        try:
            abort_failed_admission(scheduler, recv_req, exc)
        except Exception:
            logger.error(
                "Failed to abort request %s after an admission error: %s",
                rid,
                get_exception_traceback(),
            )
            raise exc


def filter_failed_materialization(
    scheduler: Scheduler, recv_reqs: list, materialize: Callable
) -> list:
    """Run per-request input materialization (e.g. CUDA-VMM multimodal
    feature unwrapping), aborting only the requests whose payload cannot be
    materialized instead of crashing the scheduler."""
    kept = []
    for recv_req in recv_reqs:
        try:
            materialize(recv_req)
        except Exception as exc:
            if not is_recoverable_exception(exc):
                raise
            if getattr(recv_req, "rid", None) is None:
                raise
            logger.error(
                "Unexpected error while materializing the inputs of request "
                "%s; aborting only this request: %s",
                recv_req.rid,
                get_exception_traceback(),
            )
            abort_failed_admission(scheduler, recv_req, exc)
            continue
        kept.append(recv_req)
    return kept


def abort_failed_admission(scheduler: Scheduler, recv_req, exc: Exception) -> None:
    """Return an internal error to the client of a request whose admission
    raised, and clean up any partially-admitted state for it."""
    rid = recv_req.rid
    message = (
        f"Unexpected internal error while admitting request {rid}: "
        f"{type(exc).__name__}: {exc}. The request was rejected; other "
        "requests are unaffected."
    )
    # Finish the client-side state first so the client gets the real error
    # message; a later abort echo for the same rid is ignored harmlessly.
    scheduler.ipc_channels.send_to_tokenizer.send_output(
        AbortReq(rid=rid, finished_reason=_internal_error_finish_reason(message)),
        recv_req,
    )
    if _find_partially_admitted(scheduler, rid):
        # The exception hit after the request reached a queue. Reuse the
        # standard abort machinery, which knows how to remove a request from
        # every queue (waiting/grammar/disaggregation) and release resources.
        scheduler.abort_request(AbortReq(rid=rid))


def _find_partially_admitted(scheduler: Scheduler, rid: str) -> bool:
    for req in scheduler.waiting_queue:
        if req.rid == rid:
            return True
    for req in scheduler.grammar_manager.grammar_queue:
        if req.rid == rid:
            return True
    if scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
        for req in scheduler.disagg_prefill_bootstrap_queue.queue:
            if req.rid == rid:
                return True
    elif scheduler.disaggregation_mode == DisaggregationMode.DECODE:
        for decode_req in scheduler.disagg_decode_prealloc_queue.queue:
            if decode_req.req.rid == rid:
                return True
    return False


# =============================================================================
# Layer 2: event-loop recovery
# =============================================================================


def run_event_loop_with_recovery(
    scheduler: Scheduler, dispatch: Callable[[], None]
) -> None:
    """Run the event loop; when enabled, recover in place from recoverable
    exceptions instead of letting them kill the scheduler process."""
    if not get_schedule().enable_scheduler_recovery:
        dispatch()
        return
    while True:
        try:
            dispatch()
            return  # Graceful exit (ShutdownReq set gracefully_exit).
        except Exception as exc:
            if not try_recover_from_loop_crash(scheduler, exc):
                raise


def try_recover_from_loop_crash(scheduler: Scheduler, exc: Exception) -> bool:
    """Attempt in-place recovery. Returns True when the scheduler is safe to
    keep serving; False re-raises the original exception (process crash)."""
    logger.error(
        "Scheduler event loop hit an exception; attempting in-place "
        "recovery: %s",
        get_exception_traceback(),
    )
    if not is_recoverable_exception(exc):
        logger.error(
            "The exception is classified as non-recoverable (process or "
            "device state may be corrupted); crashing the server."
        )
        return False
    if not _recovery_supported(scheduler):
        return False
    timeout_s = envs.SGLANG_SCHEDULER_RECOVERY_TIMEOUT.get()
    if not _consensus_barrier(timeout_s):
        return False
    try:
        num_aborted = _abort_all_inflight_requests(scheduler, exc)
        _reset_scheduler_state(scheduler, timeout_s)
    except Exception:
        logger.error(
            "In-place recovery failed; crashing the server: %s",
            get_exception_traceback(),
        )
        return False
    logger.error(
        "Scheduler recovered in place from an unexpected exception. "
        "%d in-flight request(s) were aborted with an internal error "
        "response; the KV cache was reset and serving resumed. "
        "Root cause: %r",
        num_aborted,
        exc,
    )
    return True


def _recovery_supported(scheduler: Scheduler) -> bool:
    if scheduler.disaggregation_mode != DisaggregationMode.NULL:
        # PD queues can hold requests with in-flight remote KV transfers
        # (e.g. RDMA writes) that a local reset cannot fence. Keep the crash
        # behavior there; per-request KV transfer failures are already
        # handled by the disaggregation queues themselves.
        logger.error(
            "In-place recovery is not supported in disaggregation mode; "
            "crashing the server."
        )
        return False
    if scheduler.ps.pp_size > 1:
        logger.error(
            "In-place recovery is not supported with pipeline parallelism; "
            "crashing the server."
        )
        return False
    if scheduler.enable_pdmux:
        logger.error(
            "In-place recovery is not supported with PD multiplexing; "
            "crashing the server."
        )
        return False
    return True


def _consensus_barrier(timeout_s: float) -> bool:
    """Make sure every rank of this scheduler's distributed world is also in
    recovery before any rank resets its state.

    A deterministic, request-driven exception raises on all ranks at the same
    loop iteration (request intake is broadcast from attn-TP rank 0), so all
    ranks arrive here and recover consistently. If any rank did NOT fail --
    e.g. a rank-local error while peers wait inside a collective -- the
    barrier times out and the crash behavior is kept, because a one-sided
    reset would desynchronize the ranks. The hard watchdog remains the
    backstop for any divergence that slips through.
    """
    try:
        if not torch.distributed.is_initialized():
            return True
        from sglang.srt.distributed import get_world_group

        world_group = get_world_group()
        if world_group.world_size <= 1:
            return True
        cpu_group = world_group.cpu_group
    except Exception:
        logger.error(
            "Could not determine the distributed topology for recovery "
            "consensus; crashing the server: %s",
            get_exception_traceback(),
        )
        return False
    try:
        torch.distributed.monitored_barrier(
            group=cpu_group, timeout=timedelta(seconds=timeout_s)
        )
        return True
    except Exception:
        logger.error(
            "Recovery consensus barrier failed (some ranks did not reach "
            "recovery within %.1fs); crashing the server: %s",
            timeout_s,
            get_exception_traceback(),
        )
        return False


def _collect_inflight_requests(scheduler: Scheduler) -> Dict[str, Req]:
    """All not-yet-finished requests the scheduler currently owns, deduped by
    rid. Their KV and pool resources are reclaimed wholesale by the cache
    flush, so no per-request release is attempted here."""
    reqs: Dict[str, Req] = {}

    def add(req: Optional[Req]) -> None:
        if req is None or req.finished():
            return
        reqs.setdefault(req.rid, req)

    add(scheduler.chunked_req)
    for batch in (scheduler.cur_batch_for_debug, scheduler.last_batch):
        if batch is not None:
            for req in batch.reqs:
                add(req)
    for req in scheduler.collect_inflight_reqs():
        add(req)
    if getattr(scheduler, "result_queue", None) is not None:
        for batch, _ in scheduler.result_queue:
            for req in batch.reqs:
                add(req)
    for req in scheduler.waiting_queue:
        add(req)
    for req in scheduler.grammar_manager.grammar_queue:
        add(req)
    if scheduler.dllm_manager is not None:
        for req in scheduler.dllm_manager.pop_aborted_reqs(True, ""):
            add(req)
    return reqs


def _abort_all_inflight_requests(scheduler: Scheduler, exc: Exception) -> int:
    message = (
        "The server recovered from an unexpected internal error while this "
        "request was in flight. The request was aborted; please retry. "
        f"Root cause: {type(exc).__name__}: {exc}"
    )
    reqs = _collect_inflight_requests(scheduler)
    for req in reqs.values():
        # Notification is best-effort per request: one bad request object
        # must not block the recovery of the server.
        try:
            scheduler.ipc_channels.send_to_tokenizer.send_output(
                _abort_output_for_req(req, message), req
            )
        except Exception:
            logger.error(
                "Failed to notify the client of aborted request %s: %s",
                req.rid,
                get_exception_traceback(),
            )
        try:
            scheduler.beam_coordinator.retire_group(req)
        except Exception:
            pass
        if scheduler.enable_hicache_storage:
            try:
                scheduler.tree_cache.release_aborted_request(req.rid)
            except Exception:
                pass
    if scheduler.mm_receiver is not None:
        # EPD: requests parked while waiting for encoder embeddings.
        scheduler.mm_receiver.abort_waiting_requests(AbortReq(abort_all=True))
    return len(reqs)


def _reset_scheduler_state(scheduler: Scheduler, timeout_s: float) -> None:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    # Drop every scheduling structure without touching per-request KV
    # bookkeeping; the cache flush below rebuilds the pools from scratch.
    scheduler.waiting_queue.clear()
    for req in scheduler.grammar_manager.grammar_queue:
        grammar = req.grammar
        if grammar is not None and hasattr(grammar, "cancel"):
            try:
                grammar.cancel()
            except Exception:
                pass
    scheduler.grammar_manager.grammar_queue.clear()
    scheduler.chunked_req = None
    scheduler._pending_chunked_abort_req = None
    scheduler.running_batch = ScheduleBatch(reqs=[], batch_is_full=False)
    scheduler.last_batch = None
    scheduler.cur_batch_for_debug = None
    if getattr(scheduler, "result_queue", None) is not None:
        scheduler.result_queue.clear()

    # Wait for already-launched device work (forward/sampling streams) to
    # drain before resetting the pools it reads and writes.
    try:
        scheduler.device_module.synchronize()
    except Exception:
        logger.warning(
            "Device synchronize during recovery failed: %s",
            get_exception_traceback(),
        )

    # Asynchronous subsystems (e.g. HiCache host offload) must finish before
    # a destructive reset. If they cannot drain in time, give up and crash.
    deadline = time.monotonic() + timeout_s
    while not scheduler.is_fully_idle():
        if time.monotonic() > deadline:
            raise RuntimeError(
                "The scheduler still has in-flight background work after "
                f"aborting all requests (waited {timeout_s:.1f}s); "
                "a safe in-place reset is not possible."
            )
        time.sleep(0.01)

    if not scheduler.flush_cache():
        raise RuntimeError("Cache flush failed during scheduler recovery.")
