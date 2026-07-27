#!/usr/bin/env python3
"""Compare fast-ulysses, native NCCL, and event_test3's sgl_p2p A2A."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from statistics import median
from typing import Any

import torch
import torch.distributed as dist
from fast_ulysses import UlyssesGroup

IMPLEMENTATIONS = (
    "fast_ulysses",
    "native_nccl",
    "event_test3_p2p_fused",
    "event_test3_p2p_tk",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-seq", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def nccl_mode0(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """[B, N/ws, H, D] -> [B, N, H/ws, D]."""
    batch, local_seq, global_heads, head_dim = x.shape
    local_heads = global_heads // world_size
    send = (
        x.view(batch, local_seq, world_size, local_heads, head_dim)
        .permute(2, 0, 1, 3, 4)
        .contiguous()
    )
    output = torch.empty_like(send)
    dist.all_to_all_single(output, send)
    return (
        output.permute(1, 0, 2, 3, 4)
        .contiguous()
        .view(batch, world_size * local_seq, local_heads, head_dim)
    )


def nccl_mode1(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """[B, N, H/ws, D] -> [B, N/ws, H, D]."""
    batch, global_seq, local_heads, head_dim = x.shape
    local_seq = global_seq // world_size
    send = (
        x.view(batch, world_size, local_seq, local_heads, head_dim)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )
    output = torch.empty_like(send)
    dist.all_to_all_single(output, send)
    return (
        output.permute(1, 2, 0, 3, 4)
        .contiguous()
        .view(batch, local_seq, world_size * local_heads, head_dim)
    )


class FastUlyssesCall:
    """One production-safe fast-ulysses tag.

    The branch backend inserts a stream-ordered pre-write barrier whenever a
    tag is reused. Keep that safety cost in the benchmark instead of measuring
    an unsafe raw-buffer reuse loop.
    """

    def __init__(
        self,
        group: UlyssesGroup,
        *,
        mode: int,
        tag: str,
    ) -> None:
        self.group = group
        self.mode = mode
        self.tag = tag
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.calls:
            self.group.pre_write_barrier(x)
        self.calls += 1
        return self.group.all_to_all_single_4d(
            x,
            mode=self.mode,
            tag=self.tag,
            use_tma=None,
        )


def event_test3_p2p(
    backend: Any,
    x: torch.Tensor,
    mode: int,
) -> torch.Tensor:
    """Launch one original event_test3 P2P hot path.

    The current branch's communicator is used only to allocate the same IPC
    buffers and handle. Bypassing its newer lock/event wrapper preserves the
    original event_test3 launch path being compared:
    empty(output) + sgl_kernel.ulysses_a2a[_tk](...).
    """
    prepared = backend._prepare_call(x, mode)
    if prepared is None:
        raise RuntimeError(
            "event_test3 sgl_p2p does not support this shape, dtype, or topology"
        )
    x, out_shape, batch, local_seq, global_heads, head_dim = prepared
    if backend.fa is None:
        raise RuntimeError("event_test3 sgl_p2p handle was not initialized")

    import sgl_kernel

    output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    kernel = sgl_kernel.ulysses_a2a_tk if backend.tk_style else sgl_kernel.ulysses_a2a
    kernel(
        backend.fa,
        x,
        output,
        batch,
        local_seq,
        global_heads,
        head_dim,
        mode,
    )
    return output


def timed_implementations_us(
    implementations: dict[str, Callable[[], torch.Tensor]],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, list[float]]:
    """Interleave implementations and rotate order to reduce timing drift."""
    if tuple(implementations) != IMPLEMENTATIONS:
        raise ValueError(
            f"expected implementations {IMPLEMENTATIONS}, got {tuple(implementations)}"
        )

    for _ in range(warmup):
        for name in IMPLEMENTATIONS:
            implementations[name]()
    torch.cuda.synchronize()

    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        name: [] for name in IMPLEMENTATIONS
    }
    for index in range(iterations):
        offset = index % len(IMPLEMENTATIONS)
        order = IMPLEMENTATIONS[offset:] + IMPLEMENTATIONS[:offset]
        for name in order:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            implementations[name]()
            end.record()
            events[name].append((begin, end))
    torch.cuda.synchronize()
    return {
        name: [float(begin.elapsed_time(end) * 1000) for begin, end in events[name]]
        for name in IMPLEMENTATIONS
    }


def local_mode_result(
    *,
    mode: int,
    x: torch.Tensor,
    fast_call: FastUlyssesCall,
    event_test3_backends: dict[str, Any],
    world_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    nccl_fn = nccl_mode0 if mode == 0 else nccl_mode1

    expected = nccl_fn(x, world_size)
    fast_output = fast_call(x)
    torch.testing.assert_close(fast_output, expected, rtol=0, atol=0)
    for backend in event_test3_backends.values():
        event_test3_output = event_test3_p2p(backend, x, mode)
        torch.testing.assert_close(event_test3_output, expected, rtol=0, atol=0)
    torch.cuda.synchronize()
    dist.barrier()

    samples = timed_implementations_us(
        {
            "fast_ulysses": lambda: fast_call(x),
            "native_nccl": lambda: nccl_fn(x, world_size),
            "event_test3_p2p_fused": lambda: event_test3_p2p(
                event_test3_backends["event_test3_p2p_fused"], x, mode
            ),
            "event_test3_p2p_tk": lambda: event_test3_p2p(
                event_test3_backends["event_test3_p2p_tk"], x, mode
            ),
        },
        warmup=warmup,
        iterations=iterations,
    )
    dist.barrier()

    remote_bytes = x.numel() * x.element_size() * (world_size - 1) / world_size
    return {
        "input_shape": list(x.shape),
        "remote_bytes_per_rank": remote_bytes,
        "samples_us": samples,
    }


def local_roundtrip_result(
    *,
    x: torch.Tensor,
    fast_mode0: FastUlyssesCall,
    fast_mode1: FastUlyssesCall,
    event_test3_backends: dict[str, Any],
    world_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    """Measure one real mode0 -> mode1 round trip, not a sum of medians."""

    def fast_roundtrip() -> torch.Tensor:
        return fast_mode1(fast_mode0(x))

    def nccl_roundtrip() -> torch.Tensor:
        return nccl_mode1(nccl_mode0(x, world_size), world_size)

    def event_test3_roundtrip(backend: Any) -> torch.Tensor:
        return event_test3_p2p(
            backend,
            event_test3_p2p(backend, x, 0),
            1,
        )

    expected = nccl_roundtrip()
    fast_output = fast_roundtrip()
    torch.testing.assert_close(expected, x, rtol=0, atol=0)
    torch.testing.assert_close(fast_output, expected, rtol=0, atol=0)
    for backend in event_test3_backends.values():
        event_test3_output = event_test3_roundtrip(backend)
        torch.testing.assert_close(event_test3_output, expected, rtol=0, atol=0)
    torch.cuda.synchronize()
    dist.barrier()

    samples = timed_implementations_us(
        {
            "fast_ulysses": fast_roundtrip,
            "native_nccl": nccl_roundtrip,
            "event_test3_p2p_fused": lambda: event_test3_roundtrip(
                event_test3_backends["event_test3_p2p_fused"]
            ),
            "event_test3_p2p_tk": lambda: event_test3_roundtrip(
                event_test3_backends["event_test3_p2p_tk"]
            ),
        },
        warmup=warmup,
        iterations=iterations,
    )
    dist.barrier()

    one_way_remote_bytes = x.numel() * x.element_size() * (world_size - 1) / world_size
    return {
        "input_shape": list(x.shape),
        "remote_bytes_per_rank": 2 * one_way_remote_bytes,
        "samples_us": samples,
    }


def summarize_case(
    gathered: list[object],
    case_name: str,
    iterations: int,
) -> dict[str, object]:
    """Take the median of each iteration's slowest-rank latency."""
    case = gathered[0][case_name]
    remote_bytes = case["remote_bytes_per_rank"]
    implementations: dict[str, object] = {}
    for implementation in IMPLEMENTATIONS:
        critical_rank_samples = [
            max(
                float(rank_result[case_name]["samples_us"][implementation][index])
                for rank_result in gathered
            )
            for index in range(iterations)
        ]
        median_us = median(critical_rank_samples)
        implementations[implementation] = {
            "median_us": median_us,
            "remote_gbps": remote_bytes / (median_us * 1000),
        }

    native_nccl_us = implementations["native_nccl"]["median_us"]
    for implementation in IMPLEMENTATIONS:
        implementations[implementation]["speedup_vs_native_nccl"] = (
            native_nccl_us / implementations[implementation]["median_us"]
        )

    return {
        "input_shape_per_rank": case["input_shape"],
        "remote_bytes_per_rank": remote_bytes,
        "implementations": implementations,
    }


def print_summary(
    summary: dict[str, object],
    *,
    global_seq: int,
    global_heads: int,
    head_dim: int,
    world_size: int,
) -> None:
    print(
        "\n"
        "case    implementation          median us   remote GB/s   "
        "speedup vs native NCCL",
        flush=True,
    )
    for case_name in ("mode0", "mode1", "roundtrip"):
        for implementation in IMPLEMENTATIONS:
            result = summary[case_name]["implementations"][implementation]
            print(
                f"{case_name:<7} {implementation:<24}"
                f"{result['median_us']:9.1f}"
                f"   {result['remote_gbps']:11.1f}"
                f"   {result['speedup_vs_native_nccl']:20.3f}x",
                flush=True,
            )
    local_seq = global_seq // world_size
    local_heads = global_heads // world_size
    print(
        "\n"
        "case shape paths (same for all implementations):\n"
        f"mode0:     [1, {local_seq}, {global_heads}, {head_dim}]"
        f" -> [1, {global_seq}, {local_heads}, {head_dim}]\n"
        f"mode1:     [1, {global_seq}, {local_heads}, {head_dim}]"
        f" -> [1, {local_seq}, {global_heads}, {head_dim}]\n"
        f"roundtrip: [1, {local_seq}, {global_heads}, {head_dim}]"
        f" --mode0--> [1, {global_seq}, {local_heads}, {head_dim}]"
        f" --mode1--> [1, {local_seq}, {global_heads}, {head_dim}]\n"
        "\n"
        "packed-QKV attention shape path (reference only; not timed):\n"
        f"[1, {local_seq}, {global_heads}, {3 * head_dim}]"
        f" --mode0--> [1, {global_seq}, {local_heads}, {3 * head_dim}]"
        f" --attention--> [1, {global_seq}, {local_heads}, {head_dim}]"
        f" --mode1--> [1, {local_seq}, {global_heads}, {head_dim}]\n"
        "\n"
        "event_test3_p2p_fused and event_test3_p2p_tk use the same shapes and "
        "produce the same values;\n"
        "they call different sgl-kernel CUDA entry points: ulysses_a2a vs "
        "ulysses_a2a_tk.",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.global_seq <= 0 or args.heads <= 0 or args.head_dim <= 0:
        raise ValueError("N, H, and D must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if args.global_seq % world_size:
        raise ValueError(
            f"N={args.global_seq} must be divisible by Ulysses degree={world_size}"
        )
    if args.heads % world_size:
        raise ValueError(
            f"H={args.heads} must be divisible by Ulysses degree={world_size}"
        )
    if args.head_dim * torch.bfloat16.itemsize % 16:
        raise ValueError(
            "D * sizeof(bf16) must be 16-byte aligned; D must be a multiple of 8"
        )

    local_seq = args.global_seq // world_size
    local_heads = args.heads // world_size
    generator = torch.Generator(device=device)
    generator.manual_seed(20260724 + rank)
    mode0_input = torch.randn(
        1,
        local_seq,
        args.heads,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    mode1_input = torch.randn(
        1,
        args.global_seq,
        local_heads,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    initial_pool_bytes = int(
        os.environ.get("FAST_ULYSSES_BENCH_POOL_BYTES", str(12 << 30))
    )
    fast_group = None
    event_test3_backends: dict[str, Any] = {}
    try:
        fast_group = UlyssesGroup(
            process_group=dist.group.WORLD,
            device=device,
            initial_pool_bytes=initial_pool_bytes,
        )
        if not hasattr(fast_group, "pre_write_barrier"):
            raise RuntimeError(
                "fast_ulysses lacks pre_write_barrier; run "
                "scripts/fast_ulysses/install.sh before benchmarking"
            )

        # Import lazily so distributed initialization and CUDA device selection
        # are complete before SGLang probes topology and allocates CUDA IPC.
        from sglang.multimodal_gen.runtime.distributed.device_communicators.ulysses_p2p_a2a import (
            UlyssesP2PAllToAll,
        )

        for implementation, tk_style in (
            ("event_test3_p2p_fused", False),
            ("event_test3_p2p_tk", True),
        ):
            backend = UlyssesP2PAllToAll(
                dist.group.WORLD,
                device,
                tk_style=tk_style,
            )
            event_test3_backends[implementation] = backend
            if not backend.enabled:
                raise RuntimeError(
                    f"{implementation} is unavailable: {backend.disabled_reason}"
                )

        fast_calls = {
            "mode0": FastUlyssesCall(
                fast_group,
                mode=0,
                tag="bhd_mode0",
            ),
            "mode1": FastUlyssesCall(
                fast_group,
                mode=1,
                tag="bhd_mode1",
            ),
            "roundtrip_mode0": FastUlyssesCall(
                fast_group,
                mode=0,
                tag="bhd_roundtrip_mode0",
            ),
            "roundtrip_mode1": FastUlyssesCall(
                fast_group,
                mode=1,
                tag="bhd_roundtrip_mode1",
            ),
        }

        local_results = {
            "rank": rank,
            "gpu": torch.cuda.get_device_name(device),
            "mode0": local_mode_result(
                mode=0,
                x=mode0_input,
                fast_call=fast_calls["mode0"],
                event_test3_backends=event_test3_backends,
                world_size=world_size,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            "mode1": local_mode_result(
                mode=1,
                x=mode1_input,
                fast_call=fast_calls["mode1"],
                event_test3_backends=event_test3_backends,
                world_size=world_size,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            "roundtrip": local_roundtrip_result(
                x=mode0_input,
                fast_mode0=fast_calls["roundtrip_mode0"],
                fast_mode1=fast_calls["roundtrip_mode1"],
                event_test3_backends=event_test3_backends,
                world_size=world_size,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
        }
        gathered: list[object] = [None] * world_size
        dist.all_gather_object(gathered, local_results)

        if rank == 0:
            summary = {
                case_name: summarize_case(gathered, case_name, args.iterations)
                for case_name in ("mode0", "mode1", "roundtrip")
            }
            report = {
                "correctness": "bitwise_pass",
                "dtype": "bfloat16",
                "batch": 1,
                "ulysses_degree": world_size,
                "N": args.global_seq,
                "H": args.heads,
                "D": args.head_dim,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "fast_ulysses_transfer": "auto",
                "fast_ulysses_safety": "stream_ordered_pre_write_barrier",
                "event_test3_backend": "sgl_p2p",
                "event_test3_kernels": {
                    "fused": "sgl_kernel.ulysses_a2a",
                    "tk": "sgl_kernel.ulysses_a2a_tk",
                },
                "roundtrip_definition": (
                    "direct mode0 output -> mode1 input in one timed region"
                ),
                "case_shape_paths": {
                    "mode0": [
                        [1, local_seq, args.heads, args.head_dim],
                        [1, args.global_seq, local_heads, args.head_dim],
                    ],
                    "mode1": [
                        [1, args.global_seq, local_heads, args.head_dim],
                        [1, local_seq, args.heads, args.head_dim],
                    ],
                    "roundtrip": [
                        [1, local_seq, args.heads, args.head_dim],
                        [1, args.global_seq, local_heads, args.head_dim],
                        [1, local_seq, args.heads, args.head_dim],
                    ],
                },
                "packed_qkv_attention_shape_path": [
                    [1, local_seq, args.heads, 3 * args.head_dim],
                    [1, args.global_seq, local_heads, 3 * args.head_dim],
                    [1, args.global_seq, local_heads, args.head_dim],
                    [1, local_seq, args.heads, args.head_dim],
                ],
                "event_test3_p2p_variants": {
                    "same_shape_and_values": True,
                    "fused_kernel": "sgl_kernel.ulysses_a2a",
                    "tk_kernel": "sgl_kernel.ulysses_a2a_tk",
                },
                **summary,
            }
            print_summary(
                summary,
                global_seq=args.global_seq,
                global_heads=args.heads,
                head_dim=args.head_dim,
                world_size=world_size,
            )
            print(
                "FAST_ULYSSES_BHD_JSON=" + json.dumps(report, sort_keys=True),
                flush=True,
            )
    finally:
        for backend in reversed(tuple(event_test3_backends.values())):
            backend.close()
        if fast_group is not None:
            fast_group.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
