from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

ENV_ENABLE = "SGLANG_MAMBA_SLOT_COPY_VALIDATOR"
LOG_PREFIX = "MAMBA_SLOT_COPY_PROOF"


@dataclass(frozen=True)
class SlotSelection:
    rid: str
    result_forward_iter: Optional[int]
    result_seq_len: int
    request_seq_len: int
    kv_committed_len: int
    cache_len: int
    selected_logical_slot: int
    selected_slot_id: int
    ping_pong_slot_ids: tuple[int, ...]


def build_copy_evidence(
    selection: SlotSelection,
    d2h_src_slot_id: int,
    *,
    io_backend: str,
    layout: str,
) -> dict:
    copied_selected_slot = selection.selected_slot_id == d2h_src_slot_id
    selected_future_checkpoint = (
        selection.cache_len > selection.result_seq_len
        and selection.cache_len == selection.kv_committed_len
    )
    wrong_slot_copied = copied_selected_slot and selected_future_checkpoint
    return {
        "status": (
            "WRONG_SLOT_COPIED" if wrong_slot_copied else "NO_WRONG_SLOT_EVIDENCE"
        ),
        **asdict(selection),
        "d2h_src_slot_id": d2h_src_slot_id,
        "copied_selected_slot": copied_selected_slot,
        "selected_future_checkpoint": selected_future_checkpoint,
        "wrong_slot_copied": wrong_slot_copied,
        "io_backend": io_backend,
        "layout": layout,
        "transfer_kernel": "transfer_kv_mamba_lf_pf",
    }


class MambaSlotCopyValidator:
    """Match a suspicious Mamba cache selection to the staged D2H source slot."""

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = (
            os.environ.get(ENV_ENABLE, "0") == "1" if enabled is None else enabled
        )
        self._lock = threading.Lock()
        self._results: dict[int, tuple[Optional[int], int]] = {}
        self._selections: dict[int, SlotSelection] = {}

    def record_result(self, req, batch, index: int) -> None:
        if (
            not self.enabled
            or req.mamba_ping_pong_track_buffer is None
            or batch.seq_lens_cpu is None
        ):
            return
        result = (
            getattr(batch, "forward_iter", None),
            int(batch.seq_lens_cpu[index].item()),
        )
        with self._lock:
            self._results[id(req)] = result

    def record_selection(
        self,
        *,
        req,
        cache_len: int,
        selected_logical_slot: int,
        selected_slot_tensor: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return

        with self._lock:
            result = self._results.pop(id(req), None)
        if result is None:
            return
        result_forward_iter, result_seq_len = result
        if cache_len <= result_seq_len:
            return

        selection = SlotSelection(
            rid=str(req.rid),
            result_forward_iter=result_forward_iter,
            result_seq_len=result_seq_len,
            request_seq_len=int(req.seqlen),
            kv_committed_len=int(req.kv_committed_len),
            cache_len=int(cache_len),
            selected_logical_slot=int(selected_logical_slot),
            selected_slot_id=int(selected_slot_tensor.item()),
            ping_pong_slot_ids=tuple(
                int(slot)
                for slot in req.mamba_ping_pong_track_buffer.detach().cpu().tolist()
            ),
        )
        with self._lock:
            self._selections[selection.selected_slot_id] = selection

    def record_staged_d2h(
        self,
        device_indices: torch.Tensor,
        *,
        io_backend: str,
        layout: str,
    ) -> None:
        if not self.enabled or io_backend != "kernel" or layout != "page_first":
            return

        slot_ids = [
            int(slot) for slot in device_indices.detach().cpu().flatten().tolist()
        ]
        with self._lock:
            selections = [self._selections.pop(slot_id, None) for slot_id in slot_ids]

        for slot_id, selection in zip(slot_ids, selections):
            if selection is None:
                continue
            evidence = build_copy_evidence(
                selection,
                slot_id,
                io_backend=io_backend,
                layout=layout,
            )
            logger.error(
                "%s %s",
                LOG_PREFIX,
                json.dumps(evidence, sort_keys=True),
            )


mamba_slot_copy_validator = MambaSlotCopyValidator()
