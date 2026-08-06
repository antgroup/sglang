# SPDX-License-Identifier: Apache-2.0
"""Optional Sol-Attn backend for MiniMax H3 packed varlen attention."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_SOL_ATTN_HEAD_SIZE = 128
_SOL_ATTN_SOFTMAX_SCALE = _SOL_ATTN_HEAD_SIZE**-0.5
_SOL_ATTN_KV_SPLITS = {1, 2, 4}
_FATAL_CUDA_ERROR_MARKERS = (
    "out of memory",
    "cublas_status_alloc_failed",
    "illegal memory access",
    "device-side assert",
    "device side assert",
    "misaligned address",
    "unspecified launch failure",
)


def _is_fatal_cuda_error(exc: Exception) -> bool:
    """Whether falling back after this error could hide a broken CUDA context."""

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _FATAL_CUDA_ERROR_MARKERS)


@dataclass(frozen=True)
class SolAttentionConfig:
    """Static Sol-Attn options accepted through attention_backend_config."""

    tau: float = 1.0
    thresh_type: str = "diag"
    kv_splits: int | str = "auto"
    strict: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> SolAttentionConfig:
        config = config or {}
        tau = float(config.get("tau", cls.tau))
        if not math.isfinite(tau):
            raise ValueError(f"Sol-Attn tau must be finite, got {tau!r}.")

        thresh_type = str(config.get("thresh_type", cls.thresh_type)).strip().lower()
        if thresh_type not in {"diag", "exact"}:
            raise ValueError(
                f"Sol-Attn thresh_type must be 'diag' or 'exact', got {thresh_type!r}."
            )

        kv_splits = config.get("kv_splits", cls.kv_splits)
        if isinstance(kv_splits, str):
            kv_splits = kv_splits.strip().lower()
            if kv_splits != "auto":
                try:
                    kv_splits = int(kv_splits)
                except ValueError as exc:
                    raise ValueError(
                        "Sol-Attn kv_splits must be 1, 2, 4, or 'auto', "
                        f"got {kv_splits!r}."
                    ) from exc
        if kv_splits != "auto" and (
            not isinstance(kv_splits, int)
            or isinstance(kv_splits, bool)
            or kv_splits not in _SOL_ATTN_KV_SPLITS
        ):
            raise ValueError(
                f"Sol-Attn kv_splits must be 1, 2, 4, or 'auto', got {kv_splits!r}."
            )

        strict = config.get("strict", cls.strict)
        if isinstance(strict, str):
            normalized = strict.strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(f"Sol-Attn strict must be a boolean, got {strict!r}.")
            strict = normalized == "true"
        elif not isinstance(strict, bool):
            raise ValueError(f"Sol-Attn strict must be a boolean, got {strict!r}.")

        return cls(
            tau=tau,
            thresh_type=thresh_type,
            kv_splits=kv_splits,
            strict=strict,
        )


class _SolAttentionKernels(NamedTuple):
    varlen: Callable[..., torch.Tensor] | None
    bthd: Callable[..., torch.Tensor] | None
    load_error: Exception | None


@lru_cache(maxsize=1)
def _load_sol_attn_kernels() -> _SolAttentionKernels:
    """Load optional kernels once, only when an eligible call is executed."""

    try:
        module = importlib.import_module("sol_attn")
        varlen = getattr(module, "sol_attn_varlen", None)
        bthd = getattr(module, "sol_attn", None)
        if not callable(varlen):
            varlen = None
        if not callable(bthd):
            bthd = None
        if varlen is None and bthd is None:
            raise ImportError(
                "the sol_attn package exposes neither sol_attn_varlen nor sol_attn"
            )
        return _SolAttentionKernels(varlen=varlen, bthd=bthd, load_error=None)
    except Exception as exc:
        return _SolAttentionKernels(varlen=None, bthd=None, load_error=exc)


@lru_cache(maxsize=1)
def _cute_runtime_available() -> bool:
    """Whether the optional CuTe runtime can be imported."""

    try:
        importlib.import_module("cuda.bindings.driver")
        importlib.import_module("cutlass.cute")
    except Exception:
        return False
    return True


class SolAttentionBackend(AttentionBackend):
    """Backend descriptor for the MiniMax H3 Sol-Attn integration."""

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [_SOL_ATTN_HEAD_SIZE]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SOL_ATTN

    @staticmethod
    def get_impl_cls() -> type[SolAttentionImpl]:
        return SolAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        raise NotImplementedError("Sol-Attn only supports MiniMax H3 varlen attention")

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        raise NotImplementedError("Sol-Attn does not use AttentionMetadata")


class SolAttentionImpl(AttentionImpl):
    """Packed THD Sol-Attn with a dense implementation as a safe fallback."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = float(extra_impl_args.pop("dropout_p", 0.0))
        self.prefix = prefix
        self.config = SolAttentionConfig.from_mapping(
            {
                "tau": extra_impl_args.pop("tau", SolAttentionConfig.tau),
                "thresh_type": extra_impl_args.pop(
                    "thresh_type", SolAttentionConfig.thresh_type
                ),
                "kv_splits": extra_impl_args.pop(
                    "kv_splits", SolAttentionConfig.kv_splits
                ),
                "strict": extra_impl_args.pop("strict", SolAttentionConfig.strict),
            }
        )
        self._dense = extra_impl_args.pop("dense_impl", None)
        self._dense_was_injected = self._dense is not None
        self._dense_key: tuple[str, torch.dtype] | None = None
        self._sol_unavailable_reason: str | None = None
        self._failed_sol_signatures: set[tuple[Any, ...]] = set()

    def _get_dense_impl(self, query: torch.Tensor) -> AttentionImpl:
        if self._dense_was_injected:
            return self._dense
        dense_key = (query.device.type, query.dtype)
        if self._dense is not None and self._dense_key == dense_key:
            return self._dense

        from sglang.multimodal_gen.runtime.layers.attention.selector import (
            get_attn_backend,
        )

        if (
            query.device.type == "cuda"
            and current_platform.is_cuda()
            and query.dtype in {torch.float16, torch.bfloat16}
        ):
            selected_backend = AttentionBackendEnum.FA
            supported_backends = {
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            }
        elif (
            query.device.type == "cuda"
            and current_platform.is_rocm()
            and query.dtype in {torch.float16, torch.bfloat16}
        ):
            selected_backend = AttentionBackendEnum.AITER
            supported_backends = {
                AttentionBackendEnum.AITER,
                AttentionBackendEnum.TORCH_SDPA,
            }
        else:
            selected_backend = AttentionBackendEnum.TORCH_SDPA
            supported_backends = {AttentionBackendEnum.TORCH_SDPA}

        backend = get_attn_backend(
            self.head_size,
            query.dtype,
            supported_attention_backends=supported_backends,
            selected_attention_backend=selected_backend,
        )
        self._dense = backend.get_impl_cls()(
            num_heads=self.num_heads,
            head_size=self.head_size,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
            num_kv_heads=self.num_kv_heads,
            prefix=f"{self.prefix}.dense" if self.prefix else "dense",
            dropout_p=self.dropout_p,
        )
        self._dense_key = dense_key
        return self._dense

    def _eligible(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> bool:
        if self.causal or self.dropout_p != 0.0:
            return False
        if not math.isclose(self.softmax_scale, _SOL_ATTN_SOFTMAX_SCALE):
            return False
        if query.device.type != "cuda" or query.dtype != torch.bfloat16:
            return False
        if query.ndim != 3 or query.shape[-1] != _SOL_ATTN_HEAD_SIZE:
            return False
        if query.shape != key.shape or query.shape != value.shape:
            return False
        return (
            key.device == query.device
            and value.device == query.device
            and key.dtype == query.dtype
            and value.dtype == query.dtype
        )

    def _resolve_kv_splits(self, query: torch.Tensor, seqlen: int) -> int:
        capability = tuple(torch.cuda.get_device_capability(query.device))
        if capability[0] < 8:
            raise RuntimeError(
                "Sol-Attn requires NVIDIA compute capability 8.0 or newer; "
                f"got SM{capability[0]}{capability[1]}."
            )

        cute_sm90_available = capability == (9, 0) and _cute_runtime_available()
        if self.config.kv_splits == "auto":
            kv_splits = 4 if cute_sm90_available and seqlen >= 65536 else 1
        else:
            kv_splits = int(self.config.kv_splits)

        if kv_splits > 1:
            if capability != (9, 0):
                raise RuntimeError(
                    "Sol-Attn kv_splits=2/4 is currently supported only on SM90; "
                    f"got SM{capability[0]}{capability[1]}."
                )
            if not cute_sm90_available:
                raise RuntimeError(
                    "Sol-Attn kv_splits=2/4 requires the SM90 CuTe runtime."
                )
            route_groups = ((int(seqlen) + 63) // 64 + 63) // 64
            if kv_splits > route_groups:
                raise ValueError(
                    "Sol-Attn requires each KV split to contain at least one "
                    f"N64 route group; got seqlen={seqlen}, kv_splits={kv_splits}."
                )
        return kv_splits

    @staticmethod
    def _execution_signature(
        query: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None,
    ) -> tuple[Any, ...]:
        bounds = (
            tuple(int(item) for item in cu_seqlens_host)
            if cu_seqlens_host is not None
            else ("segments", int(cu_seqlens.numel()))
        )
        return (
            str(query.device),
            query.dtype,
            tuple(query.shape),
            int(max_seqlen),
            bounds,
        )

    @staticmethod
    def _validate_output(output: Any, query: torch.Tensor) -> torch.Tensor:
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"Sol-Attn returned {type(output).__name__}, expected torch.Tensor"
            )
        if output.shape != query.shape:
            raise ValueError(
                "Sol-Attn returned an invalid shape: "
                f"got {tuple(output.shape)}, expected {tuple(query.shape)}."
            )
        if output.dtype != query.dtype or output.device != query.device:
            raise ValueError(
                "Sol-Attn output must preserve input dtype and device: "
                f"got {output.dtype} on {output.device}, expected "
                f"{query.dtype} on {query.device}."
            )
        return output

    def _run_varlen_kernel(
        self,
        kernel: Callable[..., torch.Tensor],
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        output = kernel(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            cu_seqlens=cu_seqlens.to(
                device=query.device, dtype=torch.int32
            ).contiguous(),
            max_seqlen=int(max_seqlen),
            tau=self.config.tau,
            thresh_type=self.config.thresh_type,
            kv_splits=self._resolve_kv_splits(query, max_seqlen),
        )
        return self._validate_output(output, query)

    def _run_segment_adapter(
        self,
        kernel: Callable[..., torch.Tensor],
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None,
    ) -> torch.Tensor:
        """POC adapter for the official BTHD API until native varlen lands."""

        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != query.shape[0]:
            raise ValueError(
                "cu_seqlens must start at 0 and end at the packed token count"
            )
        if any(start > stop for start, stop in zip(bounds[:-1], bounds[1:])):
            raise ValueError("cu_seqlens must be non-decreasing")

        output = torch.empty_like(query)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start == stop:
                continue
            segment_query = query[start:stop].unsqueeze(0).contiguous()
            segment_output = kernel(
                segment_query,
                key[start:stop].unsqueeze(0).contiguous(),
                value[start:stop].unsqueeze(0).contiguous(),
                tau=self.config.tau,
                thresh_type=self.config.thresh_type,
                kv_splits=self._resolve_kv_splits(query, stop - start),
            )
            segment_output = self._validate_output(segment_output, segment_query)
            output[start:stop].copy_(segment_output[0])
        return output

    def _dense_forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None,
    ) -> torch.Tensor:
        return self._get_dense_impl(query).forward_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cu_seqlens_host=cu_seqlens_host,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return self._get_dense_impl(query).forward(query, key, value, attn_metadata)

    @torch.compiler.disable
    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if not self._eligible(query, key, value):
            return self._dense_forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )

        signature = self._execution_signature(
            query,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cu_seqlens_host=cu_seqlens_host,
        )
        if (
            self._sol_unavailable_reason is not None
            or signature in self._failed_sol_signatures
        ):
            return self._dense_forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )

        kernels = _load_sol_attn_kernels()
        if kernels.load_error is not None:
            if self.config.strict or _is_fatal_cuda_error(kernels.load_error):
                raise kernels.load_error
            self._sol_unavailable_reason = (
                f"{type(kernels.load_error).__name__}: {kernels.load_error}"
            )
            logger.warning_once(
                "Sol-Attn is unavailable for this process and will use dense "
                f"packed attention: {self._sol_unavailable_reason}"
            )
            return self._dense_forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )

        try:
            if kernels.varlen is not None:
                return self._run_varlen_kernel(
                    kernels.varlen,
                    query,
                    key,
                    value,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
            if kernels.bthd is None:
                raise RuntimeError("no usable Sol-Attn kernel was loaded")
            logger.warning_once(
                "The installed Sol-Attn package has no sol_attn_varlen kernel; "
                "using the segment-by-segment BTHD POC adapter."
            )
            return self._run_segment_adapter(
                kernels.bthd,
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
            )
        except Exception as exc:
            if self.config.strict or _is_fatal_cuda_error(exc):
                raise
            self._failed_sol_signatures.add(signature)
            logger.warning_once(
                "Sol-Attn failed for this execution signature, which is now "
                "disabled and will use dense packed attention: "
                f"{type(exc).__name__}: {exc}"
            )
        return self._dense_forward_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cu_seqlens_host=cu_seqlens_host,
        )


__all__ = [
    "SolAttentionBackend",
    "SolAttentionConfig",
    "SolAttentionImpl",
]
