# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Copyright (c) 2025 DeepSeek
"""Manual SM90 EP8 FP8 correctness and full-API benchmark.

Examples (requires eight SM90 GPUs, PyTorch symmetric memory and Triton >=3.6):
  torchrun --standalone --nproc-per-node=8 test_mega_moe_gluon.py --check-correctness
  torchrun --standalone --nproc-per-node=8 test_mega_moe_gluon.py --bench --tokens 128

The independent PyTorch reference reproduces FP8 quantization, route weighting,
BF16 FC2 slots and BF16 output. Importing this file starts no CLI or GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import statistics
import sys
import types
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    import torch

FP8_E4M3_MAX = 448.0
DIFF_TOLERANCE = 0.07
SHAPES = {"flash": (4096, 2048, 256, 6), "pro": (7168, 3072, 384, 6)}


def load_backend(name="fp8", kernel_dir=None):
    """Load the three-file operator set without importing the serving stack."""
    directory = (
        Path(kernel_dir)
        if kernel_dir
        else Path(__file__).resolve().parents[3] / "python/sglang/kernels/ops/moe"
    )
    package = (
        "_sglang_gluon_manual_"
        + hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:12]
    )
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(directory.resolve())]
        sys.modules[package] = module
    suffix = {"fp8": "", "mxfp4": "_mxfp4", "internode": "_internode"}[name]
    return importlib.import_module(package + ".mega_moe_gluon" + suffix)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def quantize_per_token_per_128(
    x: torch.Tensor,
    min_amax: float = 1.0e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match SM90 pre-dispatch's continuous per-token/per-128 FP32 scale."""
    import torch

    if x.ndim != 2 or x.shape[1] % 128:
        raise ValueError("x must have shape [M, K] with K divisible by 128")
    view = x.float().view(x.shape[0], x.shape[1] // 128, 128)
    scale = view.abs().amax(dim=-1).clamp_min(min_amax) / FP8_E4M3_MAX
    quantized = (view / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return quantized.view_as(x).contiguous(), scale.contiguous()


def dequantize_per_token_per_128(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] % 128:
        raise ValueError("x must have shape [M, K] with K divisible by 128")
    view = x.float().view(x.shape[0], x.shape[1] // 128, 128)
    return (view * scale.unsqueeze(-1)).view(x.shape[0], x.shape[1])


def quantize_weight_block_128x128(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize [E,N,K] weights with DeepGEMM's FP32 block scales."""
    import torch

    if weight.ndim != 3 or weight.shape[-2] % 128 or weight.shape[-1] % 128:
        raise ValueError("weight must be [E, N, K], with N and K divisible by 128")
    e, n, k = weight.shape
    view = weight.float().view(e, n // 128, 128, k // 128, 128)
    scale = view.abs().amax(dim=(-1, -3)).clamp_min(1.0e-4) / FP8_E4M3_MAX
    quantized = (view / scale.unsqueeze(-1).unsqueeze(-3)).to(torch.float8_e4m3fn)
    return quantized.view(e, n, k).contiguous(), scale.contiguous()


def dequantize_weight_block_128x128(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    if weight.ndim != 3 or weight.shape[-2] % 128 or weight.shape[-1] % 128:
        raise ValueError("weight must be [E, N, K], with N and K divisible by 128")
    e, n, k = weight.shape
    view = weight.float().view(e, n // 128, 128, k // 128, 128)
    return (view * scale.unsqueeze(-1).unsqueeze(-3)).view(e, n, k)


def swiglu_and_quantize_per_64(
    gate_up: torch.Tensor,
    topk_weight: torch.Tensor,
    activation_clamp: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match the fused L1 epilogue, including route weight before quantize."""
    import torch

    half = gate_up.shape[-1] // 2
    gate = gate_up[..., :half].float()
    up = gate_up[..., half:].float()
    if activation_clamp is not None:
        gate = gate.clamp(max=activation_clamp)
        up = up.clamp(min=-activation_clamp, max=activation_clamp)
    activation = torch.nn.functional.silu(gate) * up
    activation = activation * topk_weight.float().unsqueeze(-1)
    if half % 64:
        raise ValueError("intermediate_hidden must be divisible by 64")
    view = activation.view(*activation.shape[:-1], half // 64, 64)
    scale = view.abs().amax(dim=-1).clamp_min(1.0e-4) / FP8_E4M3_MAX
    quantized = (view / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return quantized.view_as(activation).contiguous(), scale.contiguous()


def calc_diff(actual, expected):
    """Match deep_gemm.testing.numeric.calc_diff, including the all-zero case."""
    actual, expected = actual.double(), expected.double()
    denominator = (actual * actual + expected * expected).sum()
    if denominator == 0:
        return 0.0
    return float(1 - 2 * (actual * expected).sum() / denominator)


def check_bf16_output(actual, expected, *, tolerance=DIFF_TOLERANCE):
    """Check the complete local output and return the measured error."""
    import math

    import torch

    if actual.shape != expected.shape:
        raise AssertionError(
            f"output shapes differ: {actual.shape} and {expected.shape}"
        )
    if actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
        raise AssertionError("actual and reference outputs must both be BF16")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("actual and reference outputs must be finite")
    diff = calc_diff(actual, expected)
    if not math.isfinite(diff) or not diff < tolerance:
        raise AssertionError(f"BF16 calc_diff={diff} must be below {tolerance}")
    return diff


def reference_local_output(
    x, scales, ids, weights, weight_at, local_experts, *, clamp=10.0
):
    """Independent full-output oracle; transport only rows with valid routes.

    Each expert computes FP32 GEMMs, quantizes SwiGLU per 64, and rounds each
    FC2 slot to BF16. The final sum is accumulated in FP32 and rounded to BF16.
    No producer, epilogue, dispatch kernel or processed-weight helper is called.
    """
    import torch
    import torch.distributed as dist

    rank, world = dist.get_rank(), dist.get_world_size()
    valid = ((ids >= 0) & (ids < local_experts * world)).any(1)
    positions = torch.where(valid)[0]
    local = (
        positions.cpu(),
        dequantize_per_token_per_128(x[valid], scales[valid]).cpu(),
        ids[valid].cpu(),
        weights[valid].cpu(),
        len(x),
    )
    peers = [None] * world
    dist.all_gather_object(peers, local)
    sizes = [len(peer[0]) for peer in peers]
    tokens = torch.cat([p[1] for p in peers]).to(x.device)
    routes = torch.cat([p[2] for p in peers]).to(x.device)
    route_weights = torch.cat([p[3] for p in peers]).to(x.device)
    # Keep top-k slots to follow the reference's exact BF16 sum order.
    slots = torch.zeros((sum(sizes), ids.shape[1], x.shape[1]), device=x.device)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for expert in range(local_experts):
            rows, columns = torch.where(routes == rank * local_experts + expert)
            if rows.numel() == 0:
                continue
            w1, w2 = weight_at(expert)
            fc1 = tokens[rows] @ w1.T
            q, sf = swiglu_and_quantize_per_64(fc1, route_weights[rows, columns], clamp)
            act = (q.float().reshape(len(rows), -1, 64) * sf.unsqueeze(-1)).flatten(1)
            slots[rows, columns] = (act @ w2.T).to(torch.bfloat16).float()
        transported = slots.cpu()
        dist.all_reduce(transported)
        offset = sum(sizes[:rank])
        output = torch.zeros(
            (len(x), x.shape[1]), device=x.device, dtype=torch.bfloat16
        )
        output[positions] = (
            transported[offset : offset + sizes[rank]]
            .to(x.device)
            .to(torch.bfloat16)
            .sum(1)
        )
        return output
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def make_fp8_weights(backend, shape, world, rank, seed):
    import torch

    h, intermediate, experts, _ = shape
    parts = []
    for layer, (n, k) in enumerate(((2 * intermediate, h), (h, intermediate))):
        ws, ss = [], []
        for expert in range(experts // world):
            gen = torch.Generator(device="cuda").manual_seed(
                seed + 10007 * (rank * (experts // world) + expert) + layer
            )
            raw = torch.randn((1, n, k), device="cuda", generator=gen) * 0.08
            w, sf = quantize_weight_block_128x128(raw)
            ws.append(w)
            ss.append(sf)
        parts.append((torch.cat(ws), torch.cat(ss)))
    (w1, s1), (w2, s2) = parts

    def weight_at(expert):
        return (
            dequantize_weight_block_128x128(
                w1[expert : expert + 1], s1[expert : expert + 1]
            )[0],
            dequantize_weight_block_128x128(
                w2[expert : expert + 1], s2[expert : expert + 1]
            )[0],
        )

    packed1, sf1 = backend.prepare_weights(w1, s1)
    return (packed1, sf1, w2, s2), weight_at


def make_inputs(shape, bound, rank, world, pattern, seed, sparse=False, generation=0):
    import torch

    h, _, experts, topk = shape
    m = (
        (0 if rank == 0 else max(0, bound - rank % 3))
        if pattern == "unequal"
        else bound
    )
    generator = torch.Generator(device="cuda").manual_seed(
        seed + rank * 1000003 + bound * 101 + generation
    )
    x = torch.randn((m, h), dtype=torch.bfloat16, device="cuda", generator=generator)
    row = torch.arange(m, device="cuda")[:, None]
    col = torch.arange(topk, device="cuda")[None, :]
    ids = ((row * topk + col + rank * 17 + generation) % experts).to(torch.int32)
    if pattern == "hot":
        ids = col.expand(m, -1).to(torch.int32).contiguous()
    elif pattern in ("local", "remote"):
        owner = rank if pattern == "local" else (rank + world // 2) % world
        ids = (owner * (experts // world) + (row + col) % (experts // world)).to(
            torch.int32
        )
    elif pattern == "invalid":
        ids[:, ::2] = -1
        ids[:, 1::4] = -7  # Any negative sentinel disables a route.
    if sparse and m > 8:
        ids[8:] = -1
    weights = torch.rand((m, topk), device="cuda", generator=generator)
    weights /= weights.sum(1, keepdim=True).clamp_min(1e-10)
    return x, ids, weights


def measure_api(run, *, observations=20, iterations=20, flush_bytes=8_000_000_000):
    """CUDA Event full API; median of max-rank active means, every sample retained."""
    import torch
    import torch.distributed as dist

    raw = []
    for observation in range(observations):
        dist.barrier()
        events = []
        for iteration in range(2 * iterations):
            torch.empty(flush_bytes // 4, dtype=torch.int32, device="cuda").zero_()
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            run()
            end.record()
            if iteration >= iterations:
                events.append((begin, end))
        torch.cuda.synchronize()
        local = [begin.elapsed_time(end) * 1000 for begin, end in events]
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, local)
        raw.append(
            dict(
                observation=observation,
                per_rank_active_us=ranks,
                observation_us=max(statistics.fmean(row) for row in ranks),
            )
        )
    return dict(
        median_us=statistics.median(r["observation_us"] for r in raw),
        observations=raw,
        warmup=iterations,
        active=iterations,
        flush_bytes=flush_bytes,
        aggregation="median observations of max rank active mean",
        scope="BF16 input through final BF16 output; compile/collective prepare excluded",
    )


def make_parser(backend, *, prepare=False):
    parser = argparse.ArgumentParser(
        description=f"Manual {backend} Gluon correctness and full API timing"
    )
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--model", choices=tuple(SHAPES), default="flash")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 128])
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=("spread", "hot", "local", "remote", "invalid", "unequal"),
        default=["spread"],
    )
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observations", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--sparse-routes",
        action="store_true",
        help="Only first eight rows route; correctness only",
    )
    parser.add_argument(
        "--prequantized",
        action="store_true",
        help="Also verify FP8 inputs; benchmarks still start at BF16",
    )
    parser.add_argument("--kernel-dir", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/mega-moe-gluon-manual")
    )
    if prepare:
        parser.add_argument("--prepare-only", action="store_true")
    return parser


def run_test(backend_name, make_weights, argv=None, *, prepare_check=None):
    """Run common fixture, output and timing checks for one Gluon backend."""
    args = make_parser(backend_name, prepare=prepare_check is not None).parse_args(argv)
    if not (
        args.check_correctness or args.bench or getattr(args, "prepare_only", False)
    ):
        raise SystemExit("select --check-correctness and/or --bench")
    if (
        min(args.tokens) < 0
        or min(args.generations, args.observations, args.iterations) < 1
    ):
        raise SystemExit("tokens must be nonnegative and repetition counts positive")
    if args.bench and args.sparse_routes:
        raise SystemExit("--sparse-routes is a correctness-only option")
    internode = backend_name == "internode"
    if internode and max(args.tokens) > 128:
        raise ValueError("EP16 supports at most 128 tokens per rank")
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        required = 16 if internode else 8
        if world != required:
            raise ValueError(f"{backend_name} requires {required} ranks")
        backend = load_backend(backend_name, args.kernel_dir)
        if getattr(args, "prepare_only", False):
            prepare_check(backend, rank, args.output)
            return
        shape = SHAPES[args.model]
        h, intermediate, experts, topk = shape
        weights, weight_at = make_weights(backend, shape, world, rank, args.seed)
        ctx = backend.create_context(
            max_tokens=max(1, max(args.tokens)),
            hidden=h,
            num_experts=experts,
            topk=topk,
        )
        workspace = None
        for bound in args.tokens:
            for pattern in args.patterns:
                for generation in range(args.generations):
                    x, ids, routes = make_inputs(
                        shape,
                        bound,
                        rank,
                        world,
                        pattern,
                        args.seed,
                        args.sparse_routes,
                        generation,
                    )
                    q, sf = quantize_per_token_per_128(x)
                    kwargs = dict(tokens_bound=bound)
                    if backend_name == "mxfp4":
                        kwargs.update(activation_clamp=10.0, fast_math=True)
                    selected = backend.select(
                        topology="ep16" if internode else "ep8",
                        fmt="mxfp4" if backend_name == "mxfp4" else "fp8",
                        shape=args.model,
                        tokens_bound=bound,
                        num_sms=torch.cuda.get_device_properties(
                            "cuda"
                        ).multi_processor_count,
                    )
                    if internode:
                        ready = backend.prepare(
                            ctx, x, ids, routes, *weights, workspace=workspace, **kwargs
                        )
                        assert (
                            ready.selection.config == selected.config
                            and ready.selection.grid == selected.grid
                        )
                        workspace = ready.workspace

                    def run():
                        nonlocal workspace
                        result = backend.fused_moe(
                            ctx, x, ids, routes, *weights, workspace=workspace, **kwargs
                        )
                        workspace = result.workspace
                        return result

                    actual = run()
                    torch.cuda.synchronize()
                    record = dict(
                        rank=rank,
                        bound=bound,
                        local_tokens=len(x),
                        pattern=pattern,
                        generation=generation,
                    )
                    if args.check_correctness:
                        expected = reference_local_output(
                            q,
                            sf,
                            ids,
                            routes,
                            weight_at,
                            experts // world,
                            clamp=selected.config.activation_clamp,
                        )
                        record["calc_diff"] = check_bf16_output(actual.output, expected)
                        saved = actual.output.clone()
                        again = run()
                        torch.cuda.synchronize()
                        record["repeat_diff"] = check_bf16_output(
                            again.output, expected
                        )
                        record["repeat_bitwise"] = torch.equal(saved, again.output)
                        if args.prequantized:
                            pre = backend.fused_moe(
                                ctx,
                                q,
                                ids,
                                routes,
                                *weights,
                                x_sf=sf,
                                workspace=workspace,
                                **kwargs,
                            )
                            torch.cuda.synchronize()
                            record["prequantized_diff"] = check_bf16_output(
                                pre.output, expected
                            )
                    case = (
                        args.output / f"{args.model}-m{bound}-{pattern}-g{generation}"
                    )
                    if args.bench:
                        timing = measure_api(
                            run,
                            observations=args.observations,
                            iterations=args.iterations,
                        )
                        if rank == 0:
                            write_json(case / "benchmark.json", timing)
                    record["passed"] = True
                    write_json(case / f"rank{rank}.json", record)
                    if rank == 0:
                        print(
                            json.dumps(
                                dict(stage="case_passed", model=args.model, **record)
                            ),
                            flush=True,
                        )
                    dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    run_test("fp8", make_fp8_weights, argv)


if __name__ == "__main__":
    main()
