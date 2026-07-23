#!/usr/bin/env python3
"""CPU-side smoke checks for the Ulysses A2A backend wiring."""

from __future__ import annotations

import os
from contextlib import contextmanager

from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a import (
    UlyssesA2AConfig,
    UlyssesA2AUnsupportedError,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_a2a.fast_ulysses import (
    FastUlyssesA2ABackend,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

ENV_KEYS = (
    "SGLANG_ULYSSES_A2A_BACKEND",
    "SGLANG_ULYSSES_A2A_TRANSFER",
    "SGLANG_ULYSSES_A2A_ASYNC_QKV",
    "SGLANG_ENABLE_ULYSSES_P2P_A2A",
    "SGLANG_ENABLE_ULYSSES_P2P_A2A_TK_STYLE",
    "LINGBOT_FORCE_P2P",
    "LINGBOT_ULYSSES_JIT",
)


@contextmanager
def clean_a2a_env(**values: str):
    previous = {key: os.environ.get(key) for key in ENV_KEYS}
    try:
        for key in ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def resolve(
    *,
    backend: str | None = None,
    transfer: str | None = None,
    overlap: str | None = None,
) -> ServerArgs:
    args = object.__new__(ServerArgs)
    args.ulysses_a2a_backend = backend
    args.ulysses_a2a_transfer = transfer
    args.ulysses_a2a_qkv_overlap = overlap
    args.ulysses_a2a_legacy_prefer_p2p = False
    args._adjust_ulysses_a2a()
    args._validate_ulysses_a2a()
    return args


def main() -> None:
    assert UlyssesA2AConfig().backend == "nccl"

    with clean_a2a_env():
        args = resolve()
        assert args.ulysses_a2a_backend == "nccl"
        assert args.ulysses_a2a_transfer == "auto"
        assert args.ulysses_a2a_qkv_overlap == "off"

    with clean_a2a_env(SGLANG_ULYSSES_A2A_BACKEND="fast_ulysses"):
        args = resolve()
        assert args.ulysses_a2a_backend == "fast_ulysses"

    with clean_a2a_env(
        SGLANG_ULYSSES_A2A_BACKEND="fast_ulysses",
        SGLANG_ENABLE_ULYSSES_P2P_A2A="1",
    ):
        args = resolve(backend="nccl")
        assert args.ulysses_a2a_backend == "nccl"

    with clean_a2a_env(SGLANG_ENABLE_ULYSSES_P2P_A2A="1"):
        args = resolve()
        assert args.ulysses_a2a_backend == "auto"
        assert args.ulysses_a2a_legacy_prefer_p2p

    with clean_a2a_env(
        SGLANG_ENABLE_ULYSSES_P2P_A2A="1",
        LINGBOT_FORCE_P2P="0",
    ):
        try:
            resolve()
        except ValueError as exc:
            assert "conflict" in str(exc).lower()
        else:
            raise AssertionError("conflicting legacy flags were accepted")

    with clean_a2a_env():
        try:
            resolve(backend="nccl", transfer="sm")
        except ValueError as exc:
            assert "only valid" in str(exc)
        else:
            raise AssertionError("non-fast transfer engine was accepted")

    try:
        FastUlyssesA2ABackend()
    except UlyssesA2AUnsupportedError as exc:
        assert "not installed" in str(exc) or "reclaim" in str(exc)
    else:
        raise AssertionError("unsafe fast_ulysses adapter unexpectedly enabled")

    import sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_p2p_a2a as p2p

    assert not hasattr(p2p, "_INSTANCES")
    assert not hasattr(p2p, "get_ulysses_p2p_a2a")
    print("STATIC_SMOKE_OK")


if __name__ == "__main__":
    main()
