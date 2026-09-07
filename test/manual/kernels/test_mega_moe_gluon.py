# SPDX-License-Identifier: Apache-2.0

"""Standalone SM90 Gluon MegaMoE correctness and performance matrix.

The default kernel is sglang.kernels.ops.moe.mega_moe_gluon. Run from the
SGLang repository root with this checkout installed, on a single node with
eight SM90 GPUs (PyTorch symmetric memory, Triton >=3.6):

    torchrun --standalone --nproc-per-node=8 \
        test/manual/kernels/test_mega_moe_gluon.py --bench
    torchrun --standalone --nproc-per-node=8 \
        test/manual/kernels/test_mega_moe_gluon.py \
        --check-correctness --models flash --batches 1 128 129 255 256 --baseline none

For standalone use, only this file and mega_moe_gluon.py are required. Copy
both beside each other to load the sibling kernel automatically, or pass
--gluon-path to a kernel file explicitly.

--check-correctness and --bench are independent switches; either or both may
be selected. Correctness runs routing edge cases and checks every model/batch
against the independent PyTorch reference below. Benchmarking measures Gluon
and any available DeepGEMM MegaMoE baseline. No baseline is required. Missing
baseline modules/APIs are reported as skipped; broken imports and execution
errors fail the run. Gluon numerical failures are collected across the matrix
and produce a nonzero exit after its report is written. Baseline numerical
errors are retained as diagnostics and do not affect the exit status, matching
the historical correctness runner. --baseline none
explicitly selects Gluon only.

Performance uses identical preconditioning for both backends: flush L2,
register inputs, enqueue a device barrier, then launch. Kineto measures only
the selected MegaMoE compute kernel(s), excluding registration, control reset,
L2 flush, barriers, and host overhead. Each observation takes the slowest rank's
mean kernel time; the final latency is the median of observations. This is
kernel latency, not end-to-end serving latency. The historical protocol uses
8,000,000,000 bytes of fresh L2 flush per iteration, a Kineto warmup/active
schedule, 20 observations for M <= 128 and 3 for larger M. Every observation
retains all active kernel starts and durations. --trace-dir additionally
captures a separate CPU/CUDA diagnostic window after formal timing; its
barrier gaps and rank start skew are not included in benchmark latency.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

FP8_MAX = 448.0
FP8_E4M3_MAX = FP8_MAX
GROUP_N = GROUP_K = 128
MODEL_CONFIGS = {
    "flash": dict(hidden=4096, intermediate_hidden=2048, num_experts=256, num_topk=6),
    "pro": dict(hidden=7168, intermediate_hidden=3072, num_experts=384, num_topk=6),
}
DEFAULT_BATCHES = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    129,
    255,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
)
GLUON_KERNEL_NAMES = (
    "_sm90_fused_dispatch_1d2d_compact_3d_oob_tma_kernel",
    "_sm90_fused_dispatch_bm64_bn256_large_kernel",
)
BASELINE_APIS = {
    "sgl": ("get_symm_buffer_for_mega_moe", ("sm90_fp8_mega_moe_impl",)),
    "pr383": (
        "get_symm_buffer_for_sm90_mega_moe",
        ("sm90_fp8_mega_moe_l1_impl", "sm90_fp8_mega_moe_l2_impl"),
    ),
}


@dataclass
class _Inputs:
    x_fp8: torch.Tensor
    x_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    l1_weight: torch.Tensor
    l1_weight_sf: torch.Tensor
    l2_weight: torch.Tensor
    l2_weight_sf: torch.Tensor
    x_bf16: torch.Tensor


@dataclass
class _Backend:
    prepare: Callable
    barrier: Callable
    run_kernel: Callable
    kernel_names: tuple[str, ...]
    config: dict
    close: Callable = lambda: None

    def run(self):
        self.prepare()
        return self.run_kernel()


def _per_token_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    assert n % GROUP_K == 0
    x_view = x.view(m, n // GROUP_K, GROUP_K)
    x_amax = x_view.abs().float().amax(dim=2).clamp(1e-4)
    sf = x_amax / FP8_MAX
    x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn)
    return x_fp8.view(m, n).contiguous(), sf.contiguous()


def _quantize_grouped_fp8_block_128_128(
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_groups, n, k = weights.shape
    assert n % GROUP_N == 0 and k % GROUP_K == 0

    weights_fp8 = torch.empty_like(weights, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (num_groups, n // GROUP_N, k // GROUP_K),
        dtype=torch.float,
        device=weights.device,
    )
    for start in range(0, num_groups, 4):
        end = min(start + 4, num_groups)
        block = (
            weights[start:end]
            .view(
                end - start,
                n // GROUP_N,
                GROUP_N,
                k // GROUP_K,
                GROUP_K,
            )
            .float()
        )
        block_scales = block.abs().amax(dim=(-1, -3)).clamp(1e-4) / FP8_MAX
        weights_fp8[start:end].copy_(
            (block / block_scales.unsqueeze(-1).unsqueeze(-3))
            .to(torch.float8_e4m3fn)
            .view(end - start, n, k)
        )
        scales[start:end].copy_(block_scales)
    return weights_fp8, scales.contiguous()


def _interleave_l1_weight(
    weight: torch.Tensor,
    intermediate_hidden: int,
) -> torch.Tensor:
    experts, _, hidden = weight.shape
    gate = weight[:, :intermediate_hidden].view(
        experts,
        intermediate_hidden // 8,
        8,
        hidden,
    )
    up = weight[:, intermediate_hidden:].view(
        experts,
        intermediate_hidden // 8,
        8,
        hidden,
    )
    return torch.stack((gate, up), dim=2).reshape_as(weight).contiguous()


def _dequantize_weight_block_128x128(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    n, k = weight.shape
    return (
        weight.float().view(n // 128, 128, k // 128, 128)
        * scale.unsqueeze(-1).unsqueeze(-3)
    ).view(n, k)


def _dequantize_per_token_per_128(
    x_fp8: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    m, k = x_fp8.shape
    return (x_fp8.float().view(m, k // 128, 128) * scale.unsqueeze(-1)).view(m, k)


def _gather_rank_copies(value: torch.Tensor) -> list[torch.Tensor]:
    copies = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(copies, value)
    return copies


def _independent_reference(
    inputs,
    model: dict,
    activation_clamp: float,
) -> torch.Tensor:
    """Compute the local-token result without either fused MegaMoE kernel."""
    experts_per_rank = model["num_experts"] // dist.get_world_size()
    intermediate_hidden = model["intermediate_hidden"]
    topk = model["num_topk"]
    tokens = inputs.x_fp8.shape[0]

    l1_weights = _gather_rank_copies(inputs.l1_weight)
    l1_scales = _gather_rank_copies(inputs.l1_weight_sf)
    l2_weights = _gather_rank_copies(inputs.l2_weight)
    l2_scales = _gather_rank_copies(inputs.l2_weight_sf)
    x_fp32 = _dequantize_per_token_per_128(inputs.x_fp8, inputs.x_sf)
    flat_experts = inputs.topk_idx.reshape(-1)
    flat_route_weights = inputs.topk_weights.reshape(-1)
    combine = torch.zeros(
        (tokens * topk, model["hidden"]),
        dtype=torch.float32,
        device=inputs.x_fp8.device,
    )

    for expert in range(model["num_experts"]):
        route_positions = torch.nonzero(
            flat_experts == expert,
            as_tuple=False,
        ).flatten()
        if route_positions.numel() == 0:
            continue
        source_tokens = torch.div(
            route_positions,
            topk,
            rounding_mode="floor",
        )
        owner = expert // experts_per_rank
        local_expert = expert % experts_per_rank
        l1_weight = _dequantize_weight_block_128x128(
            l1_weights[owner][local_expert],
            l1_scales[owner][local_expert],
        )
        gate_up = torch.einsum(
            "mk,nk->mn",
            x_fp32[source_tokens],
            l1_weight,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        if math.isfinite(activation_clamp):
            gate = gate.clamp(max=activation_clamp)
            up = up.clamp(min=-activation_clamp, max=activation_clamp)
        activation = torch.nn.functional.silu(gate) * up
        activation *= flat_route_weights[route_positions].unsqueeze(-1)
        activation_view = activation.view(
            route_positions.numel(),
            intermediate_hidden // 64,
            64,
        )
        activation_scale = (
            activation_view.abs().amax(dim=-1).clamp(1.0e-10) / FP8_E4M3_MAX
        )
        activation_fp8 = (activation_view / activation_scale.unsqueeze(-1)).to(
            torch.float8_e4m3fn
        )
        l2_input = (activation_fp8.float() * activation_scale.unsqueeze(-1)).view(
            route_positions.numel(), intermediate_hidden
        )
        l2_weight = _dequantize_weight_block_128x128(
            l2_weights[owner][local_expert],
            l2_scales[owner][local_expert],
        )
        l2_output = torch.einsum("mk,nk->mn", l2_input, l2_weight)
        combine[route_positions] = l2_output.to(torch.bfloat16).float()

    output = (
        combine.view(tokens, topk, model["hidden"])
        .to(torch.bfloat16)
        .sum(dim=1, dtype=torch.float32)
        .to(torch.bfloat16)
        .contiguous()
    )
    del l1_weights, l1_scales, l2_weights, l2_scales, x_fp32, combine
    torch.cuda.empty_cache()
    return output


def _load_gluon(path: Path | None):
    if path is None:
        return importlib.import_module("sglang.kernels.ops.moe.mega_moe_gluon")
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("mega_moe_gluon", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Gluon kernel from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _select_baseline(requested: str) -> tuple[str | None, str]:
    """Agree before any baseline allocation or rank rendezvous takes place."""
    choice, reason, error = None, "disabled by --baseline none", None
    if requested != "none":
        try:
            if importlib.util.find_spec("deep_gemm") is None:
                reason = "deep_gemm is not installed"
            else:
                module = importlib.import_module("deep_gemm")
                candidates = ("sgl", "pr383") if requested == "auto" else (requested,)
                for name in candidates:
                    required = (
                        BASELINE_APIS[name][0],
                        "transform_weights_for_mega_moe_sm90",
                        "fp8_mega_moe",
                    )
                    if all(callable(getattr(module, api, None)) for api in required):
                        choice = name
                        reason = f"{name}: {module.__file__}"
                        break
                if choice is None:
                    reason = "deep_gemm has no compatible SM90 MegaMoE API"
        except Exception as exc:
            # A broken installation must not masquerade as an absent baseline.
            error = f"{type(exc).__name__}: {exc}"
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, (choice, reason, error))
    if any(report[2] is not None for report in reports):
        raise RuntimeError(f"baseline detection failed: {reports}")
    if len({report[0] for report in reports}) != 1:
        reason = f"baseline unavailable on some ranks: {reports}"
        choice = None
    if requested not in ("auto", "none") and choice is None:
        raise RuntimeError(f"requested baseline {requested} is unavailable: {reason}")
    return choice, reason


def _stable_seed(name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(name)) & 0x7FFFFFFF


def _make_inputs(args, model, tokens: int, model_name: str) -> _Inputs:
    rank, world = dist.get_rank(), dist.get_world_size()
    case_seed = args.seed + rank * 1000003 + _stable_seed(f"{model_name}:{tokens}")
    torch.manual_seed(case_seed)
    random.seed(case_seed)
    h, ih = model["hidden"], model["intermediate_hidden"]
    local_experts = model["num_experts"] // world
    x = torch.randn((tokens, h), dtype=torch.bfloat16, device="cuda")
    l1 = (
        torch.randn((local_experts, 2 * ih, h), dtype=torch.bfloat16, device="cuda")
        * 0.05
    )
    l2 = torch.randn((local_experts, h, ih), dtype=torch.bfloat16, device="cuda") * 0.05
    scores = torch.randn((tokens, model["num_experts"]), device="cuda")
    weights, indices = scores.topk(model["num_topk"], dim=-1, sorted=False)
    x_fp8, x_sf = _per_token_cast_to_fp8(x)
    l1_fp8, l1_sf = _quantize_grouped_fp8_block_128_128(l1)
    l2_fp8, l2_sf = _quantize_grouped_fp8_block_128_128(l2)
    return _Inputs(x_fp8, x_sf, indices, weights, l1_fp8, l1_sf, l2_fp8, l2_sf, x)


def _prepare_gluon(
    gluon,
    args,
    inputs,
    model,
    *,
    context=None,
    workspace=None,
    interleaved_l1=None,
    input_dtype="fp8",
    routed_scaling_factor=1.0,
):
    if context is None:
        context = gluon.create_sm90_mega_moe_symmetric_context(
            max_tokens=args.max_tokens,
            hidden=model["hidden"],
            num_experts=model["num_experts"],
            topk=model["num_topk"],
        )
    interleaved = (
        interleaved_l1
        if interleaved_l1 is not None
        else _interleave_l1_weight(inputs.l1_weight, model["intermediate_hidden"])
    )
    state = {"registered": None, "workspace": workspace}

    def prepare():
        state["registered"] = gluon.run_sm90_mega_moe_pre_dispatch(
            context,
            inputs.x_bf16 if input_dtype == "bf16" else inputs.x_fp8,
            inputs.topk_idx,
            inputs.topk_weights,
            x_sf=None if input_dtype == "bf16" else inputs.x_sf,
            routed_scaling_factor=routed_scaling_factor,
        )

    def run_kernel():
        result = gluon.run_sm90_fused_dispatch_1d2d_compact_symmetric(
            context,
            inputs.x_bf16 if input_dtype == "bf16" else inputs.x_fp8,
            None if input_dtype == "bf16" else inputs.x_sf,
            inputs.topk_idx,
            inputs.topk_weights,
            interleaved,
            inputs.l1_weight_sf,
            inputs.l2_weight,
            inputs.l2_weight_sf,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
            workspace=state["workspace"],
            pre_dispatch_result=state["registered"],
            routed_scaling_factor=routed_scaling_factor,
        )
        if (
            state["workspace"] is not None
            and result.workspace is not state["workspace"]
        ):
            raise AssertionError("Gluon failed to reuse the supplied workspace")
        state["workspace"] = result.workspace
        state["config"] = result.config
        return result.output

    prepare()
    run_kernel()
    config = asdict(state["config"])
    backend = _Backend(
        prepare,
        context.barrier,
        run_kernel,
        (GLUON_KERNEL_NAMES[int(config["use_large_normal_partition"])],),
        config,
    )
    return backend, context, state["workspace"]


def _prepare_baseline(name, args, inputs, model):
    module = importlib.import_module("deep_gemm")
    getter, kernel_names = BASELINE_APIS[name]
    buffer = getattr(module, getter)(
        dist.group.WORLD,
        model["num_experts"],
        args.max_tokens,
        model["num_topk"],
        model["hidden"],
        model["intermediate_hidden"],
    )
    l1, l2 = module.transform_weights_for_mega_moe_sm90(
        (inputs.l1_weight, inputs.l1_weight_sf),
        (inputs.l2_weight, inputs.l2_weight_sf),
    )
    output = torch.empty_like(inputs.x_fp8, dtype=torch.bfloat16)
    recv_stats = torch.zeros(
        model["num_experts"] // dist.get_world_size(), dtype=torch.int32, device="cuda"
    )

    def prepare():
        rows = inputs.x_fp8.shape[0]
        buffer.x[:rows].copy_(inputs.x_fp8)
        buffer.x_sf[:rows].copy_(inputs.x_sf)
        buffer.topk_idx[:rows].copy_(inputs.topk_idx)
        buffer.topk_weights[:rows].copy_(inputs.topk_weights)

    def run_kernel():
        module.fp8_mega_moe(
            output,
            l1,
            l2,
            buffer,
            cumulative_local_expert_recv_stats=recv_stats,
            recipe=(128, 128, 128),
            activation="swiglu",
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
        )
        return output

    return _Backend(
        prepare,
        buffer.handle.barrier,
        run_kernel,
        kernel_names,
        {"module": module.__file__, "api": getter},
        buffer.destroy,
    )


def _check(actual, expected, args, label, *, fail_fast=True):
    """All ranks observe a failed check, including a nonfinite or zero reference."""
    valid = actual.shape == expected.shape and actual.dtype == torch.bfloat16
    valid = valid and bool(torch.isfinite(actual).all())
    metrics = {"valid": valid}
    if valid:
        delta = actual.float() - expected.float()
        norm, error_norm = expected.double().norm().item(), delta.double().norm().item()
        relative_l2 = (
            error_norm / norm if norm else (0.0 if error_norm == 0 else math.inf)
        )
        mismatched = int(
            (delta.abs() > args.atol + args.rtol * expected.float().abs()).sum()
        )
        metrics.update(
            relative_l2=relative_l2,
            max_abs=float(delta.abs().max()),
            mismatched=mismatched,
            elements=delta.numel(),
        )
        metrics["valid"] = (
            relative_l2 <= args.max_relative_l2
            and metrics["max_abs"] <= args.max_abs_error
            and mismatched
            <= max(
                args.min_mismatch_budget,
                math.ceil(args.max_mismatch_ratio * delta.numel()),
            )
        )
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, metrics)
    passed = all(item["valid"] for item in gathered)
    if fail_fast and not passed:
        raise AssertionError(f"{label} failed independent reference: {gathered}")
    summary = {
        "passed": passed,
        "failed_ranks": [
            rank for rank, item in enumerate(gathered) if not item["valid"]
        ],
    }
    for key, aggregate in (
        ("relative_l2", max),
        ("max_abs", max),
        ("mismatched", sum),
        ("elements", sum),
    ):
        values = [item.get(key) for item in gathered]
        summary[key] = (
            aggregate(values)
            if all(value is not None and math.isfinite(value) for value in values)
            else None
        )
    return summary


class _SuppressStdoutStderr:
    """Suppress Python and native profiler logging only during capture."""

    def __enter__(self):
        self.streams = (sys.stdout, sys.stderr)
        for stream in self.streams:
            stream.flush()
        self.fds = tuple(stream.fileno() for stream in self.streams)
        self.saved = tuple(os.dup(fd) for fd in self.fds)
        self.null = open(os.devnull, "w")
        for fd in self.fds:
            os.dup2(self.null.fileno(), fd)
        sys.stdout = sys.stderr = self.null
        return self

    def __exit__(self, *_):
        self.null.flush()
        sys.stdout, sys.stderr = self.streams
        for fd, saved in zip(self.fds, self.saved):
            os.dup2(saved, fd)
            os.close(saved)
        self.null.close()


def _iteration(backend, args):
    # Keep the historical allocation/zero_ sequence, including its GPU runway.
    torch.empty(args.flush_l2_bytes // 4, dtype=torch.int, device="cuda").zero_()
    backend.prepare()
    backend.barrier()
    backend.run_kernel()


def _capture(backend, args, *, cpu=False):
    backend.run()  # Historical per-observation unprofiled prelude and launch.
    activities = [torch.profiler.ProfilerActivity.CUDA]
    if cpu:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    profiler = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
        acc_events=True,
    )
    with _SuppressStdoutStderr():
        with profiler:
            for count in (args.warmup, args.iters):
                for _ in range(count):
                    _iteration(backend, args)
                torch.cuda.synchronize()
                profiler.step()
    return profiler


def _cuda_events(profiler):
    return sorted(
        (
            event
            for event in profiler.profiler.kineto_results.events()
            if "CUDA" in str(event.device_type()) and event.duration_ns() > 0
        ),
        key=lambda event: event.start_ns(),
    )


def _active_samples(events, kernel_names, count):
    phases = {}
    for name in kernel_names:
        matches = [event for event in events if name in event.name()]
        if len(matches) != count:
            raise RuntimeError(
                f"expected {count} active CUDA events for {name}, got {len(matches)}"
            )
        phases[name] = [
            {"start_ns": event.start_ns(), "duration_us": event.duration_ns() / 1e3}
            for event in matches
        ]
    samples = [
        sum(phases[name][i]["duration_us"] for name in kernel_names)
        for i in range(count)
    ]
    return {
        "mean_us": statistics.mean(samples),
        "samples_us": samples,
        "phases": phases,
    }


def _measure(backend, args, tokens):
    """Historical schedule; retain every rank/observation/active iteration."""
    repeats = args.small_observations if tokens <= 128 else args.large_observations
    observations, rank_means, raw_observations = [], [], []
    backend.run()
    torch.cuda.synchronize()
    dist.barrier()
    for _ in range(repeats):
        profiler = _capture(backend, args)
        local = _active_samples(
            _cuda_events(profiler), backend.kernel_names, args.iters
        )
        per_rank = [None] * dist.get_world_size()
        # All host/device synchronization here is outside profiler capture.
        dist.all_gather_object(per_rank, local)
        means = [rank["mean_us"] for rank in per_rank]
        raw_observations.append(per_rank)
        rank_means.append(means)
        observations.append(max(means))
        dist.barrier()
    return {
        "us": statistics.median(observations),
        "observation_count": repeats,
        "observations_us": observations,
        "rank_means_us": rank_means,
        "raw_observations": raw_observations,
    }


def _trace_backend(backend, args, model_name, tokens, name):
    """Separate diagnostic capture; absolute Kineto timestamps on one host."""
    dist.barrier()
    profiler = _capture(backend, args, cpu=True)
    path = args.trace_dir / f"{model_name}-m{tokens}-{name}-rank{dist.get_rank()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(path))
    events = _cuda_events(profiler)
    samples = _active_samples(events, backend.kernel_names, args.iters)
    launches = []
    for event in events:
        if not any(kernel in event.name() for kernel in backend.kernel_names):
            continue
        preceding = [
            prior
            for prior in events
            if prior.device_resource_id() == event.device_resource_id()
            and prior.start_ns() + prior.duration_ns() <= event.start_ns()
        ]
        prior = max(
            preceding,
            key=lambda item: item.start_ns() + item.duration_ns(),
            default=None,
        )
        barriers = [prior for prior in preceding if "barrier" in prior.name().lower()]
        barrier = max(barriers, key=lambda item: item.start_ns(), default=None)
        launches.append(
            {
                "kernel": event.name(),
                "start_ns": event.start_ns(),
                "duration_us": event.duration_ns() / 1e3,
                "previous_event": prior.name() if prior else None,
                "previous_event_gap_us": (
                    event.start_ns() - prior.start_ns() - prior.duration_ns()
                )
                / 1e3
                if prior
                else None,
                "preceding_barrier": barrier.name() if barrier else None,
                "barrier_to_kernel_us": (
                    event.start_ns() - barrier.start_ns() - barrier.duration_ns()
                )
                / 1e3
                if barrier
                else None,
            }
        )
    per_rank = [None] * dist.get_world_size()
    dist.all_gather_object(
        per_rank, {"path": str(path), "samples": samples, "launches": launches}
    )
    start_skew = {
        kernel: [
            (max(starts) - min(starts)) / 1e3
            for starts in zip(
                *[
                    [launch["start_ns"] for launch in rank["samples"]["phases"][kernel]]
                    for rank in per_rank
                ]
            )
        ]
        for kernel in backend.kernel_names
    }
    return {
        "timing_included": False,
        "clock": "Kineto start_ns; single host",
        "rank_kernel_start_skew_us": start_skew,
        "ranks": per_rank,
    }


def _edge_cases(gluon, args):
    """Exercise routing, BF16 registration and reuse independently of any baseline."""
    rank, world = dist.get_rank(), dist.get_world_size()
    model = dict(
        hidden=768,
        intermediate_hidden=3072,
        num_experts=2 * world,
        num_topk=min(6, 2 * world),
    )
    edge_args = argparse.Namespace(**vars(args))
    edge_args.max_tokens = 264
    weights = _make_inputs(edge_args, model, 1, "edge")
    interleaved_l1 = _interleave_l1_weight(
        weights.l1_weight, model["intermediate_hidden"]
    )
    context, workspace = None, None
    cases = [
        (f"uniform_m{m}", m, "uniform", "fp8")
        for m in (1, 21, 22, 64, 128, 129, 255, 256, 1)
    ]
    cases += [
        ("irregular_empty_expert", 13, "irregular", "fp8"),
        ("variable_tokens", 5 + rank, "uniform", "fp8"),
        ("mixed_swap_threshold", 253 + rank, "uniform", "fp8"),
        ("invalid_token_after_reuse", 7, "invalid_token", "fp8"),
        ("empty_source_rank", 7, "empty_rank", "fp8"),
        ("bf16_registration", 31, "uniform", "bf16"),
    ]
    for case_id, (name, m, routing, dtype) in enumerate(cases):
        torch.manual_seed(args.seed + 3031 + rank * 97 + case_id)
        x = torch.randn((m, model["hidden"]), dtype=torch.bfloat16, device="cuda") * 0.1
        x_fp8, x_sf = _per_token_cast_to_fp8(x)
        tokens = torch.arange(m, device="cuda")[:, None]
        slots = torch.arange(model["num_topk"], device="cuda")[None, :]
        indices = (((rank + tokens + 1) % world) * 2 + slots) % model["num_experts"]
        if routing == "irregular":
            indices = (rank + tokens * 3 + slots * 7 + 1) % (model["num_experts"] - 1)
            indices[1, 1] = -1
        elif routing == "invalid_token":
            indices[0] = -1
        elif routing == "empty_rank" and rank == 0:
            indices.fill_(-1)
        route_weights = torch.rand(indices.shape, device="cuda")
        route_weights[indices < 0] = 0
        inputs = replace(
            weights,
            x_bf16=x,
            x_fp8=x_fp8,
            x_sf=x_sf,
            topk_idx=indices.contiguous(),
            topk_weights=route_weights,
        )
        scaling = 1.25 if dtype == "bf16" else 1.0
        reference_inputs = replace(inputs, topk_weights=route_weights * scaling)
        expected = _independent_reference(
            reference_inputs, model, args.activation_clamp
        )
        backend, context, workspace = _prepare_gluon(
            gluon,
            edge_args,
            inputs,
            model,
            context=context,
            workspace=workspace,
            interleaved_l1=interleaved_l1,
            input_dtype=dtype,
            routed_scaling_factor=scaling,
        )
        for reuse in range(args.reuse_runs):
            actual = backend.run()
            _check(actual, expected, args, f"edge/{name}/reuse{reuse}")
        # All ranks participate even when only one source has invalid tokens.
        invalid = (indices < 0).all(dim=1)
        exact_zero = (actual[invalid] == 0).all().to(torch.int32)
        dist.all_reduce(exact_zero, op=dist.ReduceOp.MIN)
        if not bool(exact_zero):
            raise AssertionError(f"{name}: invalid token has nonzero output")
        if rank == 0:
            print(f"CHECK edge/{name}: PASS", flush=True)
        del backend, actual, expected, reference_inputs, inputs
    del context, workspace, weights, interleaved_l1
    gc.collect()
    torch.cuda.empty_cache()
    return len(cases)


def _matrix_case(gluon, baseline, args, model_name, tokens):
    model = MODEL_CONFIGS[model_name]
    inputs = _make_inputs(args, model, tokens, model_name)
    expected = (
        _independent_reference(inputs, model, args.activation_clamp)
        if args.check_correctness
        else None
    )
    row = {
        "model": model_name,
        "tokens": tokens,
        "case_seed_rank0": args.seed + _stable_seed(f"{model_name}:{tokens}"),
        "baseline": baseline,
        "correctness": "not_run" if expected is None else "PASS",
        "backends": {},
    }
    for name in ("gluon", baseline):
        if name is None:
            continue
        if name == "gluon":
            backend, context, workspace = _prepare_gluon(gluon, args, inputs, model)
        else:
            backend = _prepare_baseline(name, args, inputs, model)
        try:
            result = {"config": backend.config, "correctness": "not_run"}
            if expected is not None:
                checks = [
                    _check(
                        backend.run(),
                        expected,
                        args,
                        f"{model_name}/M{tokens}/{name}/reuse{i}",
                        fail_fast=False,
                    )
                    for i in range(args.reuse_runs)
                ]
                result["checks"] = checks
                result["correctness"] = (
                    "PASS" if all(check["passed"] for check in checks) else "FAIL"
                )
            if args.bench:
                result["timing"] = _measure(backend, args, tokens)
                if args.trace_dir:
                    result["trace"] = _trace_backend(
                        backend, args, model_name, tokens, name
                    )
            row["backends"][name] = result
        finally:
            torch.cuda.synchronize()
            dist.barrier()
            backend.close()
            del backend
            if name == "gluon":
                del context, workspace
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
    row["correctness"] = row["backends"]["gluon"]["correctness"]
    if baseline is not None:
        row["baseline_correctness"] = row["backends"][baseline]["correctness"]
    if baseline is not None and args.bench and row["correctness"] != "FAIL":
        row["speedup"] = (
            row["backends"][baseline]["timing"]["us"]
            / row["backends"]["gluon"]["timing"]["us"]
        )
        row["speedup_correctness_checked"] = args.check_correctness
        row["speedup_both_backends_passed"] = (
            args.check_correctness and row["baseline_correctness"] == "PASS"
        )
    return row


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sibling = Path(__file__).with_name("mega_moe_gluon.py")
    parser.add_argument(
        "--gluon-path",
        type=Path,
        default=sibling if sibling.is_file() else None,
        help="kernel file override (default: sibling file or SGLang module)",
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="run independent-reference and routing checks",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="run performance observations; does not imply correctness",
    )
    parser.add_argument(
        "--baseline", choices=("auto", "none", "sgl", "pr383"), default="auto"
    )
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_CONFIGS), default=list(MODEL_CONFIGS)
    )
    parser.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--skip-edge-cases", action="store_true")
    parser.add_argument("--reuse-runs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--small-observations",
        type=int,
        default=20,
        help="observations per backend for M <= 128",
    )
    parser.add_argument(
        "--large-observations",
        type=int,
        default=3,
        help="observations per backend for M > 128",
    )
    parser.add_argument("--flush-l2-bytes", type=int, default=8_000_000_000)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="separate CPU/CUDA diagnostic traces, one file per rank/case/backend",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, choices=(0, 1), default=1)
    # FP8 rescaling approximations use both norm and elementwise error budgets.
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=1.5)
    parser.add_argument("--max-relative-l2", type=float, default=0.006)
    parser.add_argument("--max-mismatch-ratio", type=float, default=6.5e-3)
    parser.add_argument("--min-mismatch-budget", type=int, default=32)
    parser.add_argument("--max-abs-error", type=float, default=10.0)
    parser.add_argument(
        "--output", type=Path, help="optional JSON report written by rank 0"
    )
    args = parser.parse_args()
    if not (args.check_correctness or args.bench):
        parser.error("select --check-correctness, --bench, or both")
    if args.trace_dir and not args.bench:
        parser.error("--trace-dir requires --bench")
    if min(args.batches) < 1 or args.max_tokens < max(args.batches):
        parser.error("positive --batches must fit within --max-tokens")
    if (
        args.reuse_runs < 2
        or min(
            args.warmup,
            args.iters,
            args.small_observations,
            args.large_observations,
            args.flush_l2_bytes,
        )
        < 1
    ):
        parser.error(
            "reuse-runs must be >=2; warmup/iters/observations/flush-l2-bytes must be positive"
        )
    if args.flush_l2_bytes % 4:
        parser.error("flush-l2-bytes must be divisible by 4")
    for name in (
        "rtol",
        "atol",
        "max_relative_l2",
        "max_mismatch_ratio",
        "max_abs_error",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if args.min_mismatch_budget < 0 or args.max_mismatch_ratio > 1:
        parser.error("invalid mismatch budget")
    if math.isnan(args.activation_clamp) or args.activation_clamp <= 0:
        parser.error("activation-clamp must be positive")
    if args.gluon_path is not None and not args.gluon_path.is_file():
        parser.error(f"kernel not found: {args.gluon_path}; use --gluon-path")
    return args


def main():
    args = _parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(
        "nccl",
        timeout=timedelta(minutes=10),
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    try:
        world = dist.get_world_size()
        if not 1 < world <= 8 or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != world:
            raise RuntimeError(
                "use torchrun with 2-8 ranks on one node; the default matrix targets 8 ranks"
            )
        if any(MODEL_CONFIGS[name]["num_experts"] % world for name in args.models):
            raise RuntimeError(
                "model expert counts must be divisible by the number of ranks"
            )
        if torch.cuda.get_device_capability() != (9, 0):
            raise RuntimeError("Gluon MegaMoE requires SM90 GPUs")
        torch.backends.cuda.matmul.allow_tf32 = False
        baseline, baseline_reason = _select_baseline(args.baseline)
        gluon = _load_gluon(args.gluon_path)
        report = {
            "status": "running",
            "world_size": world,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "triton": importlib.import_module("triton").__version__,
            "kernel": str(Path(gluon.__file__).resolve()),
            "check_correctness": args.check_correctness,
            "correctness_gate": "gluon_only; baseline numerical errors are diagnostic",
            "bench": args.bench,
            "baseline": baseline,
            "baseline_status": "available" if baseline else "skipped",
            "baseline_reason": baseline_reason,
            "timing_scope": "selected_compute_kernels_only",
            "warmup": args.warmup,
            "iters": args.iters,
            "small_observations": args.small_observations,
            "large_observations": args.large_observations,
            "flush_l2_bytes": args.flush_l2_bytes,
            "profiler_schedule": {"wait": 0, "warmup": 1, "active": 1, "repeat": 1},
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "seed_formula": "seed + rank * 1000003 + stable_seed(model:tokens)",
            "cases": [],
        }
        if dist.get_rank() == 0:
            print(
                f"BASELINE {report['baseline_status']}: {baseline_reason}", flush=True
            )
        edge_count = 0
        if args.check_correctness and not args.skip_edge_cases:
            edge_count = _edge_cases(gluon, args)
        report["edge_cases_passed"] = edge_count
        if dist.get_rank() == 0:
            print(
                "model  tokens  gluon_check  baseline_check  gluon_us  baseline_us  speedup",
                flush=True,
            )
        for model in args.models:
            for tokens in args.batches:
                row = _matrix_case(gluon, baseline, args, model, tokens)
                report["cases"].append(row)
                if dist.get_rank() == 0:
                    timings = [
                        f"{row['backends'][name]['timing']['us']:.3f}"
                        if name and args.bench
                        else "N/A"
                        for name in ("gluon", baseline)
                    ]
                    speedup = f"{row['speedup']:.3f}x" if "speedup" in row else "N/A"
                    print(
                        f"{model:5s} {tokens:6d} {row['correctness']:11s} "
                        f"{row.get('baseline_correctness', 'skipped'):14s} "
                        f"{timings[0]:>9s} {timings[1]:>12s} {speedup:>8s}",
                        flush=True,
                    )
                    if args.output:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["failed_cases"] = [
            {"model": row["model"], "tokens": row["tokens"], "backends": ["gluon"]}
            for row in report["cases"]
            if row["correctness"].startswith("FAIL")
        ]
        report["baseline_failed_cases"] = [
            {"model": row["model"], "tokens": row["tokens"], "backend": baseline}
            for row in report["cases"]
            if row.get("baseline_correctness") == "FAIL"
        ]
        report["status"] = "failed" if report["failed_cases"] else "completed"
        if dist.get_rank() == 0:
            if args.output:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"MEGA_MOE_GLUON_SUMMARY status={'FAIL' if report['failed_cases'] else 'PASS'} "
                f"check_correctness={args.check_correctness} bench={args.bench} "
                f"matrix_cases={len(report['cases'])} edge_cases={edge_count} "
                f"baseline={baseline or 'skipped'} failed_cases={len(report['failed_cases'])}",
                flush=True,
            )
        if report["failed_cases"]:
            raise SystemExit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
