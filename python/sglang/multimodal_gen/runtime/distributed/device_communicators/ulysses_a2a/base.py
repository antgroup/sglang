"""Common contracts for Ulysses all-to-all backends.

The model-facing API is semantic: callers describe input/output Ulysses
redistribution and never handle transport tags, barriers, or shared buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from .router import UlyssesA2ARouter


class A2AMode(IntEnum):
    INPUT = 0
    OUTPUT = 1


class A2ASlot(str, Enum):
    Q = "q"
    K = "k"
    V = "v"
    PACKED_QKV = "packed_qkv"
    OUT = "out"
    COMPAT = "compat"


class UlyssesA2AUnsupportedError(RuntimeError):
    """The forced backend cannot implement the requested semantic operation."""


class UlyssesA2ACommitError(RuntimeError):
    """A backend failed after collective initialization or allocation began."""


class UlyssesA2ATransactionError(RuntimeError):
    """A transaction was misused or would switch backend after its first call."""


@dataclass(frozen=True)
class UlyssesA2AConfig:
    backend: str = "nccl"
    transfer: str = "auto"
    qkv_overlap: str = "off"
    p2p_tk_style: bool = True
    legacy_prefer_p2p: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"nccl", "sgl_p2p", "fast_ulysses", "auto"}:
            raise ValueError(f"Unsupported Ulysses A2A backend: {self.backend}")
        if self.transfer not in {"auto", "sm", "tma", "ce"}:
            raise ValueError(f"Unsupported Ulysses A2A transfer: {self.transfer}")
        if self.qkv_overlap not in {"off", "auto", "on"}:
            raise ValueError(
                f"Unsupported Ulysses A2A QKV overlap mode: {self.qkv_overlap}"
            )


@dataclass(frozen=True)
class A2ASpec:
    mode: A2AMode
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    head_dim: int
    seq_lens: Optional[tuple[int, ...]]
    capturing: bool
    slot: A2ASlot

    @classmethod
    def from_tensor(
        cls,
        x: torch.Tensor,
        *,
        mode: A2AMode,
        head_dim: int,
        seq_lens: Optional[list[int]],
        slot: A2ASlot,
    ) -> "A2ASpec":
        capturing = bool(
            x.is_cuda
            and torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        )
        return cls(
            mode=mode,
            shape=tuple(int(v) for v in x.shape),
            dtype=x.dtype,
            device=x.device,
            head_dim=head_dim,
            seq_lens=None if seq_lens is None else tuple(seq_lens),
            capturing=capturing,
            slot=slot,
        )

    @property
    def is_variable(self) -> bool:
        return self.seq_lens is not None

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            int(self.mode),
            self.shape,
            str(self.dtype),
            self.device.type,
            self.head_dim,
            self.seq_lens,
            self.capturing,
            self.slot.value,
        )


@dataclass
class UlyssesA2AStats:
    selected_backend: str = "uninitialized"
    nccl_calls: int = 0
    sgl_p2p_calls: int = 0
    fast_ulysses_calls: int = 0
    eligible_calls: int = 0
    semantic_exclusion_calls: int = 0
    bytes_by_mode: dict[str, int] = field(
        default_factory=lambda: {"input": 0, "output": 0}
    )
    fallback_calls_by_reason: dict[str, int] = field(default_factory=dict)
    route_cache_hits: int = 0
    route_cache_misses: int = 0
    control_consensus_calls: int = 0
    strict_backend_violation_count: int = 0

    def record_fallback(self, reason: str) -> None:
        self.fallback_calls_by_reason[reason] = (
            self.fallback_calls_by_reason.get(reason, 0) + 1
        )


class UlyssesA2ATransaction:
    """Pins one attention operation to a single transport backend."""

    def __init__(self, router: "UlyssesA2ARouter") -> None:
        self._router = router
        self.backend_name: str | None = None
        self.closed = False
        self.calls = 0

    def input_all_to_all(
        self,
        x: torch.Tensor,
        *,
        head_dim: int = 1,
        seq_lens: list[int] | None = None,
        slot: A2ASlot = A2ASlot.COMPAT,
    ) -> torch.Tensor:
        self._ensure_open()
        return self._router.input_all_to_all(
            x,
            head_dim=head_dim,
            seq_lens=seq_lens,
            slot=slot,
            transaction=self,
        )

    def output_all_to_all(
        self,
        x: torch.Tensor,
        *,
        head_dim: int = 1,
        seq_lens: list[int] | None = None,
        slot: A2ASlot = A2ASlot.OUT,
    ) -> torch.Tensor:
        self._ensure_open()
        return self._router.output_all_to_all(
            x,
            head_dim=head_dim,
            seq_lens=seq_lens,
            slot=slot,
            transaction=self,
        )

    def _ensure_open(self) -> None:
        if self.closed:
            raise UlyssesA2ATransactionError("Ulysses A2A transaction is closed")

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "UlyssesA2ATransaction":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
