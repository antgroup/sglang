"""Coordinator-owned router for Ulysses all-to-all backends."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .base import (
    A2AMode,
    A2ASlot,
    A2ASpec,
    UlyssesA2AConfig,
    UlyssesA2AStats,
    UlyssesA2ATransaction,
    UlyssesA2ATransactionError,
    UlyssesA2AUnsupportedError,
)
from .nccl import NCCLUlyssesA2ABackend

logger = logging.getLogger(__name__)


class UlyssesA2ARouter:
    """Selects a transport while preserving one semantic Ulysses API."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        config: UlyssesA2AConfig,
    ) -> None:
        self.group = group
        self.device = device
        self.config = config
        self.nccl = NCCLUlyssesA2ABackend(group)
        self.sgl_p2p = None
        self.fast_ulysses = None
        self.stats = UlyssesA2AStats()
        self.closed = False
        self._logged_backends: set[str] = set()
        self._p2p_route_cache: dict[tuple[object, ...], tuple[bool, str]] = {}
        self._fixed_fast_signatures: set[tuple[object, ...]] = set()
        self._validate_config_consensus()

    def begin_transaction(self) -> UlyssesA2ATransaction:
        self._ensure_open()
        return UlyssesA2ATransaction(self)

    def input_all_to_all(
        self,
        x: torch.Tensor,
        *,
        head_dim: int = 1,
        seq_lens: list[int] | None = None,
        slot: A2ASlot = A2ASlot.COMPAT,
        transaction: UlyssesA2ATransaction | None = None,
    ) -> torch.Tensor:
        return self._run(
            x,
            mode=A2AMode.INPUT,
            head_dim=head_dim,
            seq_lens=seq_lens,
            slot=slot,
            transaction=transaction,
        )

    def output_all_to_all(
        self,
        x: torch.Tensor,
        *,
        head_dim: int = 1,
        seq_lens: list[int] | None = None,
        slot: A2ASlot = A2ASlot.OUT,
        transaction: UlyssesA2ATransaction | None = None,
    ) -> torch.Tensor:
        return self._run(
            x,
            mode=A2AMode.OUTPUT,
            head_dim=head_dim,
            seq_lens=seq_lens,
            slot=slot,
            transaction=transaction,
        )

    def fixed_backend_all_to_all(
        self,
        x: torch.Tensor,
        *,
        mode: A2AMode,
        head_dim: int,
        seq_lens: Optional[list[int]],
        slot: A2ASlot,
    ) -> torch.Tensor:
        """Low-overhead path for model code pinned to one strict backend.

        NCCL needs no per-call routing. fast_ulysses validates and reaches
        consensus once for each semantic signature, then calls the transport
        directly on subsequent transformer layers and denoising steps.
        """
        self._ensure_open()
        self._validate_device(x)
        if self.config.backend == "nccl":
            self.stats.selected_backend = "nccl"
            self.stats.nccl_calls += 1
            self._log_selection_once("nccl")
            if mode == A2AMode.INPUT:
                return self.nccl.input_all_to_all(
                    x, head_dim=head_dim, seq_lens=seq_lens
                )
            return self.nccl.output_all_to_all(x, head_dim=head_dim, seq_lens=seq_lens)

        if self.config.backend != "fast_ulysses":
            return self._run(
                x,
                mode=mode,
                head_dim=head_dim,
                seq_lens=seq_lens,
                slot=slot,
                transaction=None,
            )

        spec = A2ASpec.from_tensor(
            x,
            mode=mode,
            head_dim=head_dim,
            seq_lens=seq_lens,
            slot=slot,
        )
        if spec.signature not in self._fixed_fast_signatures:
            supported, reason = self._accelerated_supports("fast_ulysses", spec)
            supported, reason = self._p2p_support_consensus(spec, supported, reason)
            if not supported:
                self.stats.strict_backend_violation_count += 1
                raise UlyssesA2AUnsupportedError(
                    f"forced fast_ulysses backend rejected operation: {reason}"
                )
            self._get_fast_ulysses()
            self._fixed_fast_signatures.add(spec.signature)

        self.stats.selected_backend = "fast_ulysses"
        self.stats.fast_ulysses_calls += 1
        self._log_selection_once("fast_ulysses")
        assert self.fast_ulysses is not None
        return self.fast_ulysses.all_to_all(x, int(mode), slot=slot)

    def _run(
        self,
        x: torch.Tensor,
        *,
        mode: A2AMode,
        head_dim: int,
        seq_lens: Optional[list[int]],
        slot: A2ASlot,
        transaction: UlyssesA2ATransaction | None,
    ) -> torch.Tensor:
        self._ensure_open()
        self._validate_device(x)
        spec = A2ASpec.from_tensor(
            x,
            mode=mode,
            head_dim=head_dim,
            seq_lens=seq_lens,
            slot=slot,
        )
        backend_name = self._select_backend(spec, transaction)
        if transaction is not None:
            if transaction.backend_name is None:
                transaction.backend_name = backend_name
            elif transaction.backend_name != backend_name:
                raise UlyssesA2ATransactionError(
                    "Ulysses A2A backend cannot change inside a transaction: "
                    f"{transaction.backend_name} -> {backend_name}"
                )
            transaction.calls += 1

        self.stats.selected_backend = backend_name
        self.stats.bytes_by_mode["input" if mode == A2AMode.INPUT else "output"] += (
            x.numel() * x.element_size()
        )
        self._log_selection_once(backend_name)

        if backend_name == "nccl":
            self.stats.nccl_calls += 1
            if mode == A2AMode.INPUT:
                return self.nccl.input_all_to_all(
                    x, head_dim=head_dim, seq_lens=seq_lens
                )
            return self.nccl.output_all_to_all(x, head_dim=head_dim, seq_lens=seq_lens)

        if backend_name == "sgl_p2p":
            self.stats.sgl_p2p_calls += 1
            assert self.sgl_p2p is not None
            out = self.sgl_p2p.all_to_all(x, int(mode))
            if out is None:
                raise UlyssesA2ATransactionError(
                    "sgl_p2p passed preflight but rejected after backend commit"
                )
            return out

        self.stats.fast_ulysses_calls += 1
        assert self.fast_ulysses is not None
        return self.fast_ulysses.all_to_all(
            x,
            int(mode),
            slot=slot,
        )

    def _validate_device(self, x: torch.Tensor) -> None:
        if x.device.type != self.device.type or (
            self.device.index is not None and x.device.index != self.device.index
        ):
            raise UlyssesA2ATransactionError(
                f"Ulysses A2A router is bound to {self.device}, got {x.device}"
            )

    def _select_backend(
        self,
        spec: A2ASpec,
        transaction: UlyssesA2ATransaction | None,
    ) -> str:
        if transaction is not None and transaction.backend_name is not None:
            backend_name = transaction.backend_name
            if backend_name != "nccl":
                supported, reason = self._accelerated_supports(backend_name, spec)
                if backend_name == "sgl_p2p":
                    supported, reason = self._p2p_support_consensus(
                        spec, supported, reason
                    )
                if not supported:
                    raise UlyssesA2ATransactionError(
                        f"{backend_name} transaction cannot serve {spec.signature}: "
                        f"{reason}"
                    )
            return backend_name

        requested = self.config.backend
        if requested == "nccl":
            return "nccl"
        if requested == "fast_ulysses":
            supported, reason = self._accelerated_supports("fast_ulysses", spec)
            supported, reason = self._p2p_support_consensus(spec, supported, reason)
            if not supported:
                self.stats.strict_backend_violation_count += 1
                raise UlyssesA2AUnsupportedError(
                    f"forced fast_ulysses backend rejected operation: {reason}"
                )
            self._get_fast_ulysses()
            return "fast_ulysses"
        if requested == "sgl_p2p":
            supported, reason = self._accelerated_supports("sgl_p2p", spec)
            supported, reason = self._p2p_support_consensus(spec, supported, reason)
            if not supported:
                self.stats.strict_backend_violation_count += 1
                raise UlyssesA2AUnsupportedError(
                    f"forced sgl_p2p backend rejected operation: {reason}"
                )
            self._get_sgl_p2p()
            return "sgl_p2p"

        supported, reason = self._accelerated_supports("sgl_p2p", spec)
        supported, reason = self._p2p_support_consensus(spec, supported, reason)
        if reason == "rank_descriptor_mismatch":
            self.stats.strict_backend_violation_count += 1
            raise UlyssesA2AUnsupportedError(
                "rank_descriptor_mismatch: refusing unsafe NCCL fallback"
            )
        if supported:
            try:
                self._get_sgl_p2p()
            except UlyssesA2AUnsupportedError as exc:
                self.stats.record_fallback(f"sgl_p2p_init:{exc}")
            else:
                self.stats.eligible_calls += 1
                return "sgl_p2p"
        else:
            self.stats.semantic_exclusion_calls += 1
            self.stats.record_fallback(reason)
        return "nccl"

    def _accelerated_supports(
        self, backend_name: str, spec: A2ASpec
    ) -> tuple[bool, str]:
        if spec.is_variable:
            return False, "variable_split"
        if spec.head_dim != 2:
            return False, "head_dim"
        if spec.capturing:
            return False, "cuda_graph_capture"
        if backend_name == "fast_ulysses":
            from .fast_ulysses import FastUlyssesA2ABackend

            return FastUlyssesA2ABackend.supports_semantics(
                spec.shape,
                spec.dtype,
                spec.device,
                int(spec.mode),
                self.nccl.world_size,
                head_dim=spec.head_dim,
            )

        from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_p2p_a2a import (
            UlyssesP2PAllToAll,
        )

        supported, reason = UlyssesP2PAllToAll.supports_semantics(
            spec.shape,
            spec.dtype,
            spec.device,
            int(spec.mode),
            self.nccl.world_size,
        )
        if not supported:
            return supported, reason
        return UlyssesP2PAllToAll.runtime_available(tk_style=self.config.p2p_tk_style)

    def _p2p_support_consensus(
        self, spec: A2ASpec, local_supported: bool, local_reason: str
    ) -> tuple[bool, str]:
        # Capture cannot contain selector/control collectives. Accelerated
        # backends are deterministically excluded before this point.
        if spec.capturing:
            return local_supported, local_reason

        cache_key = spec.signature
        cached = self._p2p_route_cache.get(cache_key)
        if cached is not None:
            self.stats.route_cache_hits += 1
            return cached

        self.stats.route_cache_misses += 1
        self.stats.control_consensus_calls += 1
        descriptor, descriptor_error = self._consensus_descriptor(spec)
        local_descriptor = (
            descriptor,
            bool(local_supported),
            local_reason,
            descriptor_error,
        )
        gathered: list[object] = [None] * self.nccl.world_size
        dist.all_gather_object(gathered, local_descriptor, group=self.group)

        support_flags = [bool(item[1]) for item in gathered]
        reasons = [str(item[2]) for item in gathered if not bool(item[1])]
        descriptors = [item[0] for item in gathered]
        descriptor_errors = [str(item[3]) for item in gathered if item[3]]
        if descriptor_errors or len(set(descriptors)) != 1:
            # Do not cache a mismatch: if ranks realign on a later call, each
            # rank must participate in a fresh consensus.
            return False, "rank_descriptor_mismatch"

        if not any(support_flags) and set(reasons) == {"variable_split"}:
            # A valid variable-split collective has different local sequence
            # dimensions by construction. _consensus_descriptor normalizes
            # that dimension while still requiring every rank to agree on the
            # complete seq_lens vector and all other semantics.
            decision = (False, "variable_split")
            self._p2p_route_cache[cache_key] = decision
            return decision

        if all(support_flags):
            decision = (True, "")
        elif not any(support_flags) and len(set(reasons)) == 1:
            decision = (False, reasons[0])
        else:
            decision = (False, "rank_capability_miss")
        self._p2p_route_cache[cache_key] = decision
        return decision

    def _consensus_descriptor(self, spec: A2ASpec) -> tuple[str, str]:
        """Return a rank-comparable descriptor and a local validation error."""
        if not spec.is_variable:
            return repr(spec.signature), ""

        shape = list(spec.shape)
        errors: list[str] = []
        if len(shape) != 4:
            errors.append(f"rank_not_four:{len(shape)}")
        if spec.head_dim not in (1, 2):
            errors.append(f"invalid_head_dim:{spec.head_dim}")

        seq_lens = spec.seq_lens
        if seq_lens is None or len(seq_lens) != self.nccl.world_size:
            actual = 0 if seq_lens is None else len(seq_lens)
            errors.append(f"seq_lens_size:{actual}:expected:{self.nccl.world_size}")
        elif any(seq_len < 0 for seq_len in seq_lens):
            errors.append("negative_seq_len")

        if not errors:
            assert seq_lens is not None
            seq_dim = 2 if spec.head_dim == 1 else 1
            expected_seq = (
                seq_lens[self.nccl.rank]
                if spec.mode == A2AMode.INPUT
                else sum(seq_lens)
            )
            if shape[seq_dim] != expected_seq:
                errors.append(f"sequence_dim:{shape[seq_dim]}:expected:{expected_seq}")
            # Local input sequence lengths legitimately differ by rank. Replace
            # the dimension only after validating it against the shared vector.
            shape[seq_dim] = "<sequence_from_seq_lens>"

        descriptor = (
            int(spec.mode),
            tuple(shape),
            str(spec.dtype),
            spec.device.type,
            spec.head_dim,
            seq_lens,
            spec.capturing,
            spec.slot.value,
        )
        return repr(descriptor), ",".join(errors)

    def _validate_config_consensus(self) -> None:
        descriptor = (
            self.config.backend,
            self.config.transfer,
            self.config.qkv_overlap,
            self.config.p2p_tk_style,
            self.config.legacy_prefer_p2p,
        )
        gathered: list[object] = [None] * self.nccl.world_size
        dist.all_gather_object(gathered, descriptor, group=self.group)
        if any(item != descriptor for item in gathered):
            self.stats.strict_backend_violation_count += 1
            raise UlyssesA2AUnsupportedError(
                f"Ulysses A2A config differs across ranks: {gathered}"
            )

    def _get_sgl_p2p(self):
        if self.sgl_p2p is None:
            from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_p2p_a2a import (
                UlyssesP2PAllToAll,
            )

            candidate = UlyssesP2PAllToAll(
                self.group,
                self.device,
                tk_style=self.config.p2p_tk_style,
            )
            if not candidate.enabled:
                raise UlyssesA2AUnsupportedError(candidate.disabled_reason)
            self.sgl_p2p = candidate
        return self.sgl_p2p

    def _get_fast_ulysses(self):
        if self.fast_ulysses is None:
            from .fast_ulysses import FastUlyssesA2ABackend

            self.fast_ulysses = FastUlyssesA2ABackend(
                self.group,
                self.device,
                transfer=self.config.transfer,
            )
        return self.fast_ulysses

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("Ulysses A2A router is closed")

    def _log_selection_once(self, backend_name: str) -> None:
        if backend_name in self._logged_backends:
            return
        logger.info(
            "Ulysses A2A router selected backend=%s transfer=%s "
            "qkv_overlap=%s p2p_tk_style=%s legacy_prefer_p2p=%s",
            backend_name,
            self.config.transfer,
            self.config.qkv_overlap,
            self.config.p2p_tk_style,
            self.config.legacy_prefer_p2p,
        )
        self._logged_backends.add(backend_name)

    def close(self) -> None:
        if self.closed:
            return
        if self.fast_ulysses is not None:
            self.fast_ulysses.close()
            self.fast_ulysses = None
        if self.sgl_p2p is not None:
            self.sgl_p2p.close()
            self.sgl_p2p = None
        self.nccl.close()
        self._p2p_route_cache.clear()
        self._fixed_fast_signatures.clear()
        self.closed = True
