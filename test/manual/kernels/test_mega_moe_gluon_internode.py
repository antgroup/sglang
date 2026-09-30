# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
"""Manual two-node EP16 FP8 checks and full-API benchmark.

Launch with torchrun --nnodes=2 --nproc-per-node=8 and the appropriate
--node-rank, --master-addr and --master-port. The shared CLI supports
--check-correctness / --bench independently, --patterns local remote unequal,
--tokens 0 1 2 32 33 64 65 127 128, and --generations for workspace reuse.
Collective prepare and actual driver residency checks run before timing;
the timed API omits config and num_sms and retains the selected default grid.
"""

from __future__ import annotations

import json

from test_mega_moe_gluon import (
    check_bf16_output,
    make_fp8_weights,
    make_inputs,
    quantize_per_token_per_128,
    reference_local_output,
    run_test,
    write_json,
)


def check_collective_preparation(backend, rank, output):
    """Check read-only preparation, actual occupancy, fallback and peer failure."""
    from dataclasses import asdict, replace

    import torch
    import torch.distributed as dist

    shape = (768, 512, 32, 6)
    ctx = backend.create_context(hidden=768, num_experts=32, topk=6, max_tokens=128)
    weights, weight_at = make_fp8_weights(backend, shape, 16, rank, 913)
    x, ids, routes = make_inputs(shape, 8, rank, 16, "unequal", 913)
    q, sf = quantize_per_token_per_128(x)
    expected = reference_local_output(q, sf, ids, routes, weight_at, 2, clamp=10.0)
    base = backend.select(
        topology="ep16",
        fmt="fp8",
        shape=backend.Shape(*shape),
        tokens_bound=8,
        num_sms=ctx.physical_sms,
    ).config
    swap = replace(
        base,
        body=backend.MathBody.SWAP_BN128,
        block_m=64,
        math_warps=4,
        stages=3,
        ctas_per_sm=2,
        math_registers=168,
        producer_registers=40,
        dispatch_handoff=backend.DispatchHandoff.D1,
        rendezvous=backend.Rendezvous.NONE,
    )
    cases = [
        ("default", base),
        ("register", replace(swap, fuse_reset=True)),
        ("bm16", replace(swap, block_m=16, math_registers=104, ctas_per_sm=3)),
        (
            "chunked",
            replace(swap, combine=backend.CombineMode.CHUNKED, dispatch_registers=80),
        ),
        ("stage_fallback", replace(swap, stages=8)),
    ]
    workspace = None

    def snapshot():
        torch.cuda.synchronize()
        host = (
            ctx.epoch,
            dict(ctx._stage_epochs),
            dict(ctx._barrier_epochs),
            getattr(ctx, "registration_target", None),
        )
        names = (
            "input_metadata",
            "input_acts",
            "expert_state",
            "signals",
            "node_dispatch_barrier",
            "node_fused_barrier",
            "credit_payload",
        )
        return host, {
            name: getattr(ctx, name).view(torch.uint8).clone() for name in names
        }

    def unchanged(state):
        host, tensors = state
        torch.cuda.synchronize()
        assert host == (
            ctx.epoch,
            ctx._stage_epochs,
            ctx._barrier_epochs,
            getattr(ctx, "registration_target", None),
        )
        for name, saved in tensors.items():
            assert torch.equal(getattr(ctx, name).view(torch.uint8), saved), name
        dist.barrier()

    for label, config in cases:
        state = snapshot()
        ready = backend.prepare(
            ctx,
            x,
            ids,
            routes,
            *weights,
            tokens_bound=8,
            config=config,
            workspace=workspace,
        )
        unchanged(state)
        if label == "stage_fallback":
            assert len(ready.attempts) > 1 and ready.selection.adjustments
            assert ready.selection.config.stages < 8
        workspace = ready.workspace
        diffs = []
        for _ in range(2):
            result = backend.fused_moe(ctx, x, ids, routes, *weights, **ready.kwargs)
            torch.cuda.synchronize()
            diffs.append(check_bf16_output(result.output, expected))
            dist.barrier()
        write_json(
            output / f"preparation-{label}-rank{rank}.json",
            dict(
                passed=True,
                selection=asdict(ready.selection),
                attempts=ready.attempts,
                calc_diff=diffs,
            ),
        )
    w1, s1, w2, s2 = weights
    bad_weight = w1.transpose(1, 2).contiguous().transpose(1, 2) if rank == 7 else w1
    state = snapshot()
    try:
        backend.prepare(
            ctx,
            x,
            ids,
            routes,
            bad_weight,
            s1,
            w2,
            s2,
            tokens_bound=8,
            config=swap,
            workspace=workspace,
        )
    except backend.ResourcePreparationError as error:
        attempts = error.attempts
    else:
        raise AssertionError("all ranks must observe the peer layout failure")
    unchanged(state)
    write_json(
        output / f"preparation-peer-failure-rank{rank}.json",
        dict(passed=True, attempts=attempts),
    )
    if rank == 0:
        print(
            json.dumps(dict(stage="collective_preparation_passed", cases=6)), flush=True
        )


def main(argv=None):
    run_test(
        "internode", make_fp8_weights, argv, prepare_check=check_collective_preparation
    )


if __name__ == "__main__":
    main()
