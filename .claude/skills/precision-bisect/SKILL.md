---
name: precision-bisect
description: Automated kernel precision bisection — compare baseline vs target dumps, align inputs, drill into intermediate ops to find the root cause of precision divergence. Use when debugging numerical differences between two implementations (e.g., FP16 vs BF16, custom kernel vs reference, different GPU architectures, or before/after a code change).
---

# Precision Bisect: Automated Kernel Accuracy Comparison

This skill guides you through an automated, iterative precision bisection workflow.
The goal is to find **which specific operator** and **which specific invocation** first
introduces a significant numerical difference between a baseline and a target run.

## Prerequisites

The precision debug decorator must already be applied to the kernel functions you want
to inspect. See `python/sglang/kernel_precision_debug.py` for the decorator and its
environment variable documentation.

```python
from sglang.kernel_precision_debug import precision_debug

@precision_debug
def my_kernel(input, weight, ...):
    ...
```

When `SGLANG_PRECISION_DEBUG=0` (default), the decorator is a no-op with zero overhead.

## Quick Reference: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_PRECISION_DEBUG` | `0` | 0=off, 1=shapes(no sync), 2=+stats(sync), 3=+dump, 4=align mode |
| `SGLANG_PRECISION_DUMP_DIR` | `sglang_precision_dumps` | `%i`=PID, `%r`=rank |
| `SGLANG_PRECISION_TAG` | `""` | Subdirectory tag: "baseline" or "target" |
| `SGLANG_PRECISION_OP_FILTER` | `""` | fnmatch op patterns, comma-separated |
| `SGLANG_PRECISION_INDEX_FILTER` | `<10` | Invocation filter (0-based): `5-10`, `>=100`, `3,7` |
| `SGLANG_PRECISION_SHAPE_FILTER` | `""` | e.g. `dim0>=1024,dim-1<256` |
| `SGLANG_PRECISION_RANK_FILTER` | `""` | Ranks to capture. Empty = all |
| `SGLANG_PRECISION_LOG_DEST` | `stderr` | stdout/stderr/filepath |
| `SGLANG_PRECISION_RANGE_START` | `""` | Range start anchor: `"*attn*"` or `"*attn*#3"` |
| `SGLANG_PRECISION_RANGE_END` | `""` | Range end anchor (same syntax). Both required |
| `SGLANG_PRECISION_ALIGN_DIR` | `""` | Level 4: path to baseline dump tag dir |
| `SGLANG_PRECISION_ALIGN_THRESHOLD` | `1e-3` | Level 4: rel_diff PASS/FAIL threshold |

## Workflow Overview

```
Phase 1: Collect dumps (baseline + target)
    ↓
Phase 2: Global comparison — find which ops diverge
    ↓
Phase 3: Input alignment — verify inputs match
    ↓  (if inputs match but outputs don't → this op is the culprit)
    ↓  (if inputs don't match → trace back to the upstream op)
    ↓
Phase 4: Narrow down — re-dump with finer filters
    ↓
Phase 5: Root cause — identify the exact op + invocation
```

---

## Phase 1: Collect Dumps

Run the same workload twice — once for baseline, once for target — with tensor dumping enabled.

### Step 1a: Baseline run

```bash
SGLANG_PRECISION_DEBUG=3 \
SGLANG_PRECISION_TAG=baseline \
SGLANG_PRECISION_DUMP_DIR=/path/to/dumps \
SGLANG_PRECISION_RANK_FILTER=0 \
SGLANG_PRECISION_INDEX_FILTER="<=10" \
python your_script.py [baseline args...]
```

### Step 1b: Target run

```bash
SGLANG_PRECISION_DEBUG=3 \
SGLANG_PRECISION_TAG=target \
SGLANG_PRECISION_DUMP_DIR=/path/to/dumps \
SGLANG_PRECISION_RANK_FILTER=0 \
SGLANG_PRECISION_INDEX_FILTER="<=10" \
python your_script.py [target args...]
```

### Step 1c: Verify dump structure

```bash
# Check that both sides have matching ops
ls /path/to/dumps/baseline/rank0/
ls /path/to/dumps/target/rank0/
```

Expected structure:
```
/path/to/dumps/
  baseline/rank0/{op_name}/call_000001/{inputs,outputs,inputs_after}.pt + meta.json
  target/rank0/{op_name}/call_000001/{inputs,outputs,inputs_after}.pt + meta.json
```

---

## Phase 2: Global Comparison

Run the comparison CLI to get an overview of which ops diverge.

```bash
python -m sglang.kernel_precision_debug compare \
    --baseline /path/to/dumps/baseline \
    --target /path/to/dumps/target \
    --rank 0 \
    --diff-threshold 1e-3
```

### Reading the output

```
[PASS] call_000001/inputs.pt/input: shape=[4, 128] rel_diff=1.2e-7 ...
[PASS] call_000001/inputs.pt/weight: shape=[128] rel_diff=0 ...
[FAIL] call_000001/outputs.pt/return: shape=[4, 128] rel_diff=0.015 ...
```

- **PASS**: rel_diff ≤ threshold — tensors match
- **FAIL**: rel_diff > threshold — tensors diverge

### Key insight

For each op, look at this pattern:
- **Inputs PASS, Outputs FAIL** → This op itself introduces the divergence. It is a
  **root cause candidate**.
- **Inputs FAIL, Outputs FAIL** → Divergence is inherited from upstream. Trace back.
- **Inputs FAIL, Outputs PASS** → This op is somehow convergent (rare, e.g. argmax).
- **All PASS** → This op is not involved.

### Filter to specific ops

```bash
python -m sglang.kernel_precision_debug compare \
    --baseline /path/to/dumps/baseline \
    --target /path/to/dumps/target \
    --op-filter "rmsnorm*,*attention*" \
    --diff-threshold 1e-3
```

---

## Phase 3: Input Alignment (Level 4 — Align Mode)

When you find an op where **outputs diverge but you're not sure about inputs**, use
**level 4** to automatically load baseline inputs, replace the actual call inputs, execute,
and compare outputs — all inline, no separate script needed.

### Step 3a: Run target with align mode

First, you need baseline dumps from Phase 1. Then run the target with level 4:

```bash
# Step 1 was: dump baseline with level 3
# Step 2: run target with level 4, pointing ALIGN_DIR to the baseline dumps
SGLANG_PRECISION_DEBUG=4 \
SGLANG_PRECISION_ALIGN_DIR=/path/to/dumps/baseline \
SGLANG_PRECISION_ALIGN_THRESHOLD=1e-3 \
SGLANG_PRECISION_INDEX_FILTER="<=10" \
python your_script.py [target args...]
```

### Step 3b: Read the output

For each decorated op call, the decorator prints three comparison sections:

```
[ALIGN] my_op #1 — replacing inputs from /path/to/dumps/baseline/rank0/my_op/call_000001
  [ALIGN FAIL] my_op #1 input_diff/input: rel_diff=0.95 ...    ← inputs were different
  [ALIGN PASS] my_op #1 input_diff/weight: rel_diff=0 ...      ← weights matched
  [ALIGN PASS] my_op #1 output_aligned/return: rel_diff=1e-8 ... ← outputs match with aligned inputs!
  [ALIGN PASS] my_op #1 inplace_aligned/input: rel_diff=0 ...  ← inplace state matches
```

**What each section means:**

| Section | What it compares |
|---------|-----------------|
| `input_diff` | Actual inputs vs baseline inputs (shows how much inputs diverged) |
| `output_aligned` | Output from target(baseline_inputs) vs baseline_outputs |
| `inplace_aligned` | Inplace-modified tensors from target vs baseline `inputs_after.pt` |

### Step 3c: Interpret results

- **`output_aligned` PASS** → The op produces the same results given the same inputs.
  The divergence comes from **upstream**. Trace back to find which earlier op diverged.
- **`output_aligned` FAIL** → The op itself behaves differently even with identical inputs.
  This is the **root cause**. Investigate: different kernel? wrong dtype? numerical instability?
- **`input_diff` PASS + `output_aligned` FAIL** → Inputs were already aligned naturally,
  yet outputs differ. Confirms this op is the culprit without needing replacement.

### Step 3d: Align mode also dumps (for chaining)

When `SGLANG_PRECISION_DUMP_DIR` and `SGLANG_PRECISION_TAG` are set, level 4 also saves
the aligned outputs to disk. This lets you chain: run op A aligned, then use its aligned
outputs as the new baseline for op B.

```bash
SGLANG_PRECISION_DEBUG=4 \
SGLANG_PRECISION_ALIGN_DIR=/path/to/dumps/baseline \
SGLANG_PRECISION_DUMP_DIR=/path/to/dumps \
SGLANG_PRECISION_TAG=aligned_target \
python your_script.py [target args...]
```

### Fallback: Manual alignment script

For cases where level 4 cannot be used (e.g., the op is not decorated, or you need custom
preprocessing), you can still write a manual alignment script:

```python
import torch

baseline_inputs = torch.load(
    "/path/to/dumps/baseline/rank0/my_op/call_000001/inputs.pt",
    weights_only=True, map_location="cuda"
)
baseline_outputs = torch.load(
    "/path/to/dumps/baseline/rank0/my_op/call_000001/outputs.pt",
    weights_only=True, map_location="cuda"
)

# Run target implementation with baseline inputs
result = target_my_op(baseline_inputs["input"], baseline_inputs["weight"])

diff = (result.float() - baseline_outputs["return"].float()).abs()
print(f"max_abs_diff={diff.max().item():.6g}")
```

---

## Phase 4: Narrow Down with Finer Filters

Once you identify a suspect op, re-dump with more invocations or specific shape filters
to get a detailed view.

### Dump more invocations of a specific op

```bash
SGLANG_PRECISION_DEBUG=3 \
SGLANG_PRECISION_TAG=baseline_detailed \
SGLANG_PRECISION_DUMP_DIR=/path/to/dumps \
SGLANG_PRECISION_OP_FILTER="jit_kernel.norm.rmsnorm" \
SGLANG_PRECISION_INDEX_FILTER="1-100" \
python your_script.py [baseline args...]
```

### Dump only a range of ops (between two anchors)

```bash
# Only capture ops between the 2nd attention call and 2nd mlp call
SGLANG_PRECISION_DEBUG=3 \
SGLANG_PRECISION_TAG=baseline_range \
SGLANG_PRECISION_DUMP_DIR=/path/to/dumps \
SGLANG_PRECISION_RANGE_START="*attention*#2" \
SGLANG_PRECISION_RANGE_END="*mlp*#2" \
python your_script.py [baseline args...]
```

The `#N` uses the same per-op call counter as `INDEX_FILTER`. Filter order:
rank → op_name → shape → (increment counter) → range start/end → index_filter.

### Dump only for large inputs

```bash
SGLANG_PRECISION_DEBUG=3 \
SGLANG_PRECISION_TAG=baseline_large \
SGLANG_PRECISION_DUMP_DIR=/path/to/dumps \
SGLANG_PRECISION_SHAPE_FILTER="dim0>=1024" \
python your_script.py [baseline args...]
```

### Compare with per-side slicing

When baseline and target have different batch sizes or sequence lengths on dim0:

```bash
# Baseline has batch_size=8, target has batch_size=4
# Compare the first 4 samples from baseline against all of target
python -m sglang.kernel_precision_debug compare \
    --baseline /path/to/dumps/baseline \
    --target /path/to/dumps/target \
    --baseline-slice "0:4"

# Compare specific token positions
python -m sglang.kernel_precision_debug compare \
    --baseline /path/to/dumps/baseline \
    --target /path/to/dumps/target \
    --baseline-index "0,5,10,15" \
    --target-index "0,5,10,15"

# Different ranges for each side
python -m sglang.kernel_precision_debug compare \
    --baseline /path/to/dumps/baseline \
    --target /path/to/dumps/target \
    --baseline-slice "100:200" \
    --target-slice "50:150"
```

---

## Phase 5: Root Cause Analysis

Once you've identified the divergent op and invocation, investigate further.

### Check for numerical issues

```bash
# Quick stats check (level 2, no dump)
SGLANG_PRECISION_DEBUG=2 \
SGLANG_PRECISION_OP_FILTER="the_divergent_op" \
SGLANG_PRECISION_INDEX_FILTER="<=5" \
python your_script.py
```

Look for:
- `nan>0` or `inf>0` in input stats → bad upstream values
- Very large `max` or very small `min` → potential overflow/underflow
- `mean` wildly different between baseline and target

### Common root causes

| Symptom | Likely Cause |
|---------|-------------|
| Inputs match, outputs differ by ~1e-3 | Different CUDA kernel implementation (e.g., FlashAttention v2 vs v3) |
| Inputs match, outputs differ by ~1e-1 | Wrong dtype (FP16 vs BF16) or wrong eps value |
| Inputs match, outputs completely different | Wrong kernel selected, or index/offset bug |
| First op diverges | Weight loading or initialization difference |
| Divergence grows layer-by-layer | Numerical accumulation; check residual connections |
| Only specific invocations diverge | Shape-dependent code path (e.g., warp vs block kernel) |

### Inplace ops

For inplace ops (e.g., `fused_add_rmsnorm` which modifies `input` and `residual` in place),
compare `inputs_after.pt` instead of `outputs.pt`:

```bash
python -m sglang.kernel_precision_debug compare \
    --baseline /path/to/dumps/baseline \
    --target /path/to/dumps/target \
    --op-filter "*fused_add_rmsnorm*"
```

The comparison automatically includes `inputs_after.pt` alongside `inputs.pt` and `outputs.pt`.

---

## Automated Bisection Recipe

For a systematic layer-by-layer bisection, follow this recipe:

### 1. Broad sweep: dump all ops, first few calls

```bash
# Both runs: capture first 5 calls of every op
SGLANG_PRECISION_INDEX_FILTER="<=5"
SGLANG_PRECISION_DEBUG=3
```

### 2. Global compare: find first divergent op

```bash
python -m sglang.kernel_precision_debug compare \
    --baseline .../baseline --target .../target \
    --diff-threshold 1e-3
```

Scan the output **top-to-bottom** (ops are sorted by name, calls by index). The first
`FAIL` on `outputs.pt` where `inputs.pt` is `PASS` is your **primary suspect**.

### 3. Confirm with level-4 align mode

Re-run the target with `SGLANG_PRECISION_DEBUG=4` and `SGLANG_PRECISION_ALIGN_DIR`
pointing to the baseline dumps. For each op, the decorator replaces inputs with baseline
data and compares outputs. If `output_aligned` is FAIL, you've confirmed the root cause.

### 4. If the root cause op is a compound/fused op

- Temporarily replace the fused op with its unfused equivalent
- Re-instrument the sub-operations with `@precision_debug`
- Re-run the bisection on the sub-operations
- This recursively narrows down to the atomic operation that diverges

### 5. Document findings

Record:
- Which op diverges
- At which invocation index
- Input shapes and dtypes
- The magnitude of divergence (rel_diff, max_abs_diff)
- The root cause (wrong kernel, dtype mismatch, numerical instability, etc.)

---

## Multi-Rank Comparison

For distributed inference, compare the same rank across baseline and target:

```bash
# Compare rank 0
python -m sglang.kernel_precision_debug compare \
    --baseline .../baseline --target .../target --rank 0

# Compare rank 1
python -m sglang.kernel_precision_debug compare \
    --baseline .../baseline --target .../target --rank 1
```

To dump all ranks:

```bash
SGLANG_PRECISION_RANK_FILTER="" \  # empty = all ranks
SGLANG_PRECISION_DEBUG=3 \
...
```

---

## Tips

1. **Start with level 1** (`SGLANG_PRECISION_DEBUG=1`) to quickly see which ops are
   called and in what order — no GPU sync overhead.

2. **Use level 2** (`SGLANG_PRECISION_DEBUG=2`) to spot NaN/Inf without generating
   dump files.

3. **Only use level 3** when you need to compare actual tensor values offline.

4. **Use level 4** for automated input alignment — it loads baseline inputs, replaces
   actual inputs, executes, and compares outputs inline. No separate script needed.

5. **Default `INDEX_FILTER=<=10`** means only the first 10 invocations per op are
   captured. Increase this if the divergence only appears later.

5. **Default `RANK_FILTER=0`** means only rank 0 is captured. Set to `""` for all ranks.

6. **For inplace ops**, always check `inputs_after.pt` — it captures the tensor state
   after the op modifies its inputs.

7. **Torch compile**: The decorator automatically skips during `torch.compile` tracing.

8. **CUDA graph capture**: Disable CUDA graph for the debug run to ensure tensors are
   accessible for dumping.

---

## Comparison CLI Full Reference

```bash
python -m sglang.kernel_precision_debug compare \
    --baseline <path>           # Baseline dump directory (required)
    --target <path>             # Target dump directory (required)
    --rank <int>                # Rank to compare (default: 0)
    --op-filter <patterns>      # fnmatch op filter (default: all)
    --diff-threshold <float>    # PASS/FAIL threshold (default: 1e-3)
    --slice <start:stop>        # Dim0 slice for both sides
    --baseline-slice <start:stop>  # Dim0 slice for baseline only
    --target-slice <start:stop>    # Dim0 slice for target only
    --index <i,j,k>             # Dim0 indices for both sides
    --baseline-index <i,j,k>    # Dim0 indices for baseline only
    --target-index <i,j,k>      # Dim0 indices for target only
```
