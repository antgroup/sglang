---
name: precision-bisect
description: Automated kernel precision bisection — instrument kernel call sites, dump baseline, run level-4 align to find root cause of precision divergence. Use when debugging numerical differences between two implementations (e.g., FP16 vs BF16, custom kernel vs reference, different GPU architectures, or before/after a code change).
---

# Precision Bisect: Automated Kernel Accuracy Comparison

This skill guides you through an automated precision bisection workflow:
1. **Instrument** — add `@precision_debug` to kernel call sites (auto or manual)
2. **Dump baseline** — run with level 3
3. **Align & compare** — run target with level 4 against baseline → root cause in one pass

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

---

## Phase 0: Instrument Kernel Call Sites

Before dumping or comparing, the kernel functions must be wrapped with `@precision_debug`.
When `SGLANG_PRECISION_DEBUG=0` (default), the decorator is a no-op with zero overhead.

### Auto-Instrumentation (recommended for this skill)

When the user asks to debug precision, **you should automatically add instrumentation** to
the relevant kernel call sites. The key principle: **wrap the lowest-level kernel functions,
not the model forward methods**. All layers share the same kernel function, so the per-op
call counter naturally distinguishes layers (call #0 = layer 0, call #1 = layer 1, ...).

#### Two instrumentation patterns

The decorator supports both standalone functions and `nn.Module` methods. It automatically
detects and skips `self`/`cls` parameters — only tensor arguments are logged/dumped.

**Pattern A: Standalone wrapper** — for bare kernel calls (e.g., `sgl_kernel.rmsnorm`):
```python
from sglang.kernel_precision_debug import precision_debug

@precision_debug(op_name="sgl_kernel.rmsnorm")
def _traced_rmsnorm(x, weight, eps):
    return rmsnorm(x, weight, eps)

# Replace call site: rmsnorm(x, ...) → _traced_rmsnorm(x, ...)
```

**Pattern B: Direct method decoration** — for `nn.Module` methods (e.g., `forward_normal`):
```python
class DeepseekV2MoE(nn.Module):
    @precision_debug(op_name="moe.forward_normal")
    def forward_normal(self, hidden_states, ...):
        # self is automatically skipped — only hidden_states etc. are captured
        router_logits = self.gate(hidden_states)
        ...
```

Both patterns produce identical dump structure. Use Pattern A for wrapping external
kernel calls (sgl_kernel, triton), Pattern B for instrumenting existing class methods.

#### Instrumentation targets

**Normalization** — `python/sglang/srt/layers/layernorm.py` (Pattern A, shared by all models):
```python
from sglang.kernel_precision_debug import precision_debug

@precision_debug(op_name="sgl_kernel.rmsnorm")
def _traced_rmsnorm(x, weight, eps):
    return rmsnorm(x, weight, eps)

@precision_debug(op_name="sgl_kernel.fused_add_rmsnorm")
def _traced_fused_add_rmsnorm(x, residual, weight, eps):
    fused_add_rmsnorm(x, residual, weight, eps)
    return x  # inplace — decorator captures inputs_after.pt automatically

# Replace in RMSNorm.forward_cuda():
#   rmsnorm(x, ...) → _traced_rmsnorm(x, ...)
#   fused_add_rmsnorm(x, ...) → _traced_fused_add_rmsnorm(x, ...)
```

**Attention** — `python/sglang/srt/layers/attention/` (Pattern A):
```python
@precision_debug(op_name="attn.extend")
def _traced_attn_extend(q, k, v, ...):
    return original_extend(q, k, v, ...)

@precision_debug(op_name="attn.decode")
def _traced_attn_decode(q, k, v, ...):
    return original_decode(q, k, v, ...)
```

**Linear / GEMM** — `python/sglang/srt/layers/linear.py` (Pattern A, shared by all models):
```python
@precision_debug(op_name="qkv_proj")
def _traced_qkv(x, weight, bias): ...

@precision_debug(op_name="o_proj")
def _traced_o_proj(x, weight, bias): ...

@precision_debug(op_name="gate_up_proj")
def _traced_gate_up(x, weight, bias): ...

@precision_debug(op_name="down_proj")
def _traced_down_proj(x, weight, bias): ...
```

**Activation** — `python/sglang/srt/layers/activation.py` (Pattern A):
```python
@precision_debug(op_name="silu_and_mul")
def _traced_silu_and_mul(x):
    return SiluAndMul.forward(x)
```

**ROPE** — `python/sglang/srt/layers/rotary_embedding.py` (Pattern A):
```python
@precision_debug(op_name="rotary_emb")
def _traced_rope(positions, q, k):
    return original_rope(positions, q, k)
```

#### Model-specific instrumentation

The three primary target models share the common layer ops above, plus model-specific ops:

**Qwen** (`python/sglang/srt/models/qwen.py`) — Dense model:
```
QWenBlock.forward():
  ln_1 (RMSNorm) → attn [qkv_proj → rope → attn → o_proj] → residual add
  ln_2 (RMSNorm) → mlp  [gate_up_proj → silu_and_mul → down_proj] → residual add
```
- Standard dense architecture, no extra ops. Uses the common instrumentation above.
- Note: Qwen uses explicit `residual + hidden_states` (not fused_add_rmsnorm).

**DeepSeek V2** (`python/sglang/srt/models/deepseek_v2.py`) — MoE model:
```
DeepseekV2DecoderLayer.forward():
  input_layernorm (RMSNorm) → self_attn (MLA) → post_attention_layernorm (RMSNorm)
  → MoE: gate → topk → experts (FusedMoE) → shared_experts → combine
```
Use **Pattern B** — directly decorate the `nn.Module` methods in `DeepseekV2MoE`:
```python
class DeepseekV2MoE(nn.Module):
    @precision_debug(op_name="moe.forward_normal")
    def forward_normal(self, hidden_states, ...):
        # self is skipped — only hidden_states etc. captured
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        ...
```
For finer granularity, also wrap sub-components inside `forward_normal`:
```python
class DeepseekV2MoE(nn.Module):
    @precision_debug(op_name="moe.gate")
    def _traced_gate(self, hidden_states):
        return self.gate(hidden_states)

    @precision_debug(op_name="moe.topk")
    def _traced_topk(self, hidden_states, router_logits):
        return self.topk(hidden_states, router_logits)

    @precision_debug(op_name="moe.experts")
    def _traced_experts(self, hidden_states, topk_output):
        return self.experts(hidden_states, topk_output)

    @precision_debug(op_name="moe.shared_experts")
    def _traced_shared_experts(self, hidden_states):
        return self._forward_shared_experts(hidden_states)

    def forward_normal(self, hidden_states, ...):
        router_logits = self._traced_gate(hidden_states)
        topk_output = self._traced_topk(hidden_states, router_logits)
        final = self._traced_experts(hidden_states, topk_output)
        shared = self._traced_shared_experts(hidden_states)
        return final + shared
```
- MLA attention: also wrap `kv_a_proj_with_mqa` and `kv_b_proj` for compressed KV precision.
- DeepEP path: decorate `forward_deepep(self, ...)` with Pattern B for A2A dispatch tracking.
- `dsv3_router_gemm` / `dsv3_fused_a_gemm`: wrap with Pattern A if investigating router kernel precision.

**Bailing MoE** (`python/sglang/srt/models/bailing_moe.py`) — MoE model:
```
BailingMoEBlock.forward():
  input_layernorm (RMSNorm) → attention [qkv_proj → rope → attn → o_proj]
  → post_attention_layernorm (RMSNorm)
  → MoE: gate → topk → experts (FusedMoE) → shared_experts → combine
```
Same Pattern B approach as DeepSeek V2:
```python
class BailingMoESparseMoeBlock(nn.Module):
    @precision_debug(op_name="moe.forward_normal")
    def forward_normal(self, hidden_states, ...):
        ...

    # Or finer-grained:
    @precision_debug(op_name="moe.gate")      # BailingMoEGate (F.linear)
    def _traced_gate(self, hidden_states): ...

    @precision_debug(op_name="moe.experts")   # FusedMoE dispatch+GEMM+combine
    def _traced_experts(self, hidden_states, topk_output): ...
```
- Optional QK norm: decorate `self.q_norm` / `self.k_norm` methods in `BailingMoEAttention`.
- DeepEP path: decorate `forward_deepep(self, ...)` with Pattern B.

#### How call counter maps to layers

Example: DeepSeek V2 with 60 layers (some dense, some MoE):

```
sgl_kernel.rmsnorm       #0~#59  → layer 0~59 input_layernorm
sgl_kernel.fused_add_rmsnorm #0~#59 → layer 0~59 post_attn_layernorm
qkv_proj / kv_a_proj     #0~#59  → layer 0~59 attention projections
attn.decode               #0~#59  → layer 0~59 attention
o_proj                    #0~#59  → layer 0~59 output projection
moe.gate                  #0~#N   → MoE layers only (counter only for MoE layers)
moe.topk                  #0~#N   → MoE layers only
moe.experts               #0~#N   → MoE layers only
gate_up_proj              #0~#M   → dense layers only (if model has dense+MoE mix)
```

Note: MoE ops (`moe.gate`, `moe.experts`) have their own counter, separate from dense
ops (`gate_up_proj`). The counter only increments for layers that actually call that op.

Use `INDEX_FILTER` to select layers: `INDEX_FILTER="10-15"` = only layer 10~15.
Use `RANGE_START/END` for cross-op windows: `RANGE_START="qkv_proj#5"` to `RANGE_END="down_proj#5"` = all ops in layer 5.

#### Instrumentation checklist

Before running, verify instrumentation with level 1 (zero-overhead shape logging):

```bash
SGLANG_PRECISION_DEBUG=1 python -m sglang.bench_one_batch \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 1 --input-len 128 --output-len 1 2>&1 | head -50
```

Expected: one `[IN]`/`[OUT]` pair per op per layer. If ops are missing, add more wrappers.

---

## Phase 1: Dump Baseline (Level 3)

Run the baseline workload with tensor dumping. Use `bench_one_batch` for a single
forward pass (no server needed), or run the full server for end-to-end scenarios.

```bash
SGLANG_PRECISION_DEBUG=3 \
SGLANG_PRECISION_TAG=baseline \
SGLANG_PRECISION_DUMP_DIR=/path/to/dumps \
SGLANG_PRECISION_INDEX_FILTER="<10" \
python -m sglang.bench_one_batch \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 1 --input-len 128 --output-len 1 \
    --dtype float32
```

Verify dump structure:

```bash
ls /path/to/dumps/baseline/rank0/
# Expected: sgl_kernel.rmsnorm/  qkv_proj/  attn.decode/  o_proj/  ...
ls /path/to/dumps/baseline/rank0/sgl_kernel.rmsnorm/
# Expected: call_000000/  call_000001/  ... call_000031/
```

Dump layout:
```
/path/to/dumps/baseline/rank0/
  {op_name}/call_{idx:06d}/
    inputs.pt        ← named tensor dict
    outputs.pt       ← return value(s)
    inputs_after.pt  ← tensor state after execution (for inplace ops)
    meta.json        ← shapes, dtypes, stats, timestamp
```

---

## Phase 2: Align & Compare (Level 4) — One-Pass Root Cause Detection

Run the target workload with level 4, pointing `ALIGN_DIR` to the baseline dumps.
This **replaces Phase 1b + Phase 2 + Phase 3** of the old workflow — no need to
dump target separately or run the compare CLI.

```bash
SGLANG_PRECISION_DEBUG=4 \
SGLANG_PRECISION_ALIGN_DIR=/path/to/dumps/baseline \
SGLANG_PRECISION_ALIGN_THRESHOLD=1e-3 \
SGLANG_PRECISION_INDEX_FILTER="<10" \
python -m sglang.bench_one_batch \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 1 --input-len 128 --output-len 1 \
    --dtype bfloat16 2>&1 | tee align_report.txt
```

### Reading the output

For each op call, the decorator prints three comparison sections:

```
[ALIGN] sgl_kernel.rmsnorm #0 — replacing inputs from .../baseline/rank0/sgl_kernel.rmsnorm/call_000000
  [ALIGN PASS] #0 input_diff/x:       rel_diff=0          ← actual vs baseline inputs
  [ALIGN PASS] #0 input_diff/weight:   rel_diff=0
  [ALIGN FAIL] #0 output_aligned/return: rel_diff=3.2e-3   ← target(baseline_inputs) vs baseline_outputs
  [ALIGN PASS] #0 inplace_aligned/x:   rel_diff=1e-8       ← post-exec state vs baseline
```

| Section | Compares | Purpose |
|---------|----------|---------|
| `input_diff` | actual inputs vs baseline inputs | Shows upstream divergence |
| `output_aligned` | target(baseline_inputs) vs baseline_outputs | **Root cause judgment** |
| `inplace_aligned` | post-exec tensors vs baseline `inputs_after.pt` | For inplace ops |

### Root cause judgment

- **`output_aligned` FAIL** → This op itself produces different results with identical inputs.
  **Root cause confirmed.** Investigate: different kernel? wrong dtype? numerical instability?
- **`output_aligned` PASS** → This op works correctly; divergence comes from upstream.

### Why level 4 is sufficient

Level 4 replaces each op's inputs with baseline data before executing. This means:
- Each op is tested independently — upstream errors don't propagate
- `input_diff` shows what the natural input divergence would be (for reference)
- `output_aligned` is the definitive per-op verdict

Extract all root causes in one command:

```bash
grep "ALIGN FAIL.*output_aligned" align_report.txt
```

### Behavior note: input replacement chain

```
Op0: inject baseline inputs → execute → aligned output → passed to Op1
Op1: inject baseline inputs → execute → aligned output → passed to Op2
...
```

Each op gets baseline inputs regardless of what upstream produced. The `input_diff`
section still reports the actual vs baseline input difference (before replacement),
so you can see the natural error propagation if needed.

---

## Phase 3: Narrow Down (Optional)

If too many ops show as FAIL, or you want to focus on specific layers:

### Filter by op name

```bash
SGLANG_PRECISION_OP_FILTER="sgl_kernel.rmsnorm,qkv_proj" \
SGLANG_PRECISION_DEBUG=4 ...
```

### Filter by layer (call index)

```bash
# Only check layers 10-15
SGLANG_PRECISION_INDEX_FILTER="10-15" \
SGLANG_PRECISION_DEBUG=4 ...
```

### Filter by layer range (cross-op window)

```bash
# All ops in layer 5 only
SGLANG_PRECISION_RANGE_START="sgl_kernel.rmsnorm#5" \
SGLANG_PRECISION_RANGE_END="down_proj#5" \
SGLANG_PRECISION_DEBUG=4 ...
```

The `#N` matches the per-op call counter (same as `INDEX_FILTER`).
Filter order: rank → op_name → shape → (increment counter) → range → index_filter.

### Filter by input shape

```bash
# Only capture when dim0 >= 1024 (long sequences)
SGLANG_PRECISION_SHAPE_FILTER="dim0>=1024" \
SGLANG_PRECISION_DEBUG=4 ...
```

### Re-dump with finer granularity

```bash
# Dump more invocations of a specific suspect op
SGLANG_PRECISION_DEBUG=3 \
SGLANG_PRECISION_TAG=baseline_detailed \
SGLANG_PRECISION_DUMP_DIR=/path/to/dumps \
SGLANG_PRECISION_OP_FILTER="sgl_kernel.rmsnorm" \
SGLANG_PRECISION_INDEX_FILTER="0-100" \
python -m sglang.bench_one_batch ...
```

---

## Phase 4: Root Cause Analysis

### Quick stats check (level 2, no dump)

```bash
SGLANG_PRECISION_DEBUG=2 \
SGLANG_PRECISION_OP_FILTER="the_divergent_op" \
SGLANG_PRECISION_INDEX_FILTER="<5" \
python -m sglang.bench_one_batch ...
```

Look for: `nan>0`, `inf>0`, very large `max`, very small `min`.

### Common root causes

| Symptom | Likely Cause |
|---------|-------------|
| Inputs match, outputs differ by ~1e-3 | Different CUDA kernel (e.g., FlashAttention v2 vs v3) |
| Inputs match, outputs differ by ~1e-1 | Wrong dtype (FP16 vs BF16) or wrong eps |
| Inputs match, outputs completely different | Wrong kernel selected, or index/offset bug |
| First op diverges | Weight loading or initialization difference |
| Divergence grows layer-by-layer | Numerical accumulation; check residual connections |
| Only specific layers diverge | Shape-dependent code path (warp vs block kernel) |

### Inplace ops

For inplace ops (e.g., `fused_add_rmsnorm`), check `inplace_aligned` instead of
`output_aligned`. The decorator captures tensor state after execution in `inputs_after.pt`.

### Compound/fused ops

If the root cause op is a fused op:
1. Temporarily replace the fused op with its unfused equivalent
2. Re-instrument the sub-operations with `@precision_debug`
3. Re-run the bisection on sub-operations
4. Recursively narrow down to the atomic divergent operation

---

## Optional: Offline Comparison (dump both sides)

For cases where you want to see **natural error propagation** (without input replacement),
or when level 4 cannot be used, fall back to the 3-step dump+compare workflow:

```bash
# Step 1: Dump baseline (level 3)
SGLANG_PRECISION_DEBUG=3 SGLANG_PRECISION_TAG=baseline ...

# Step 2: Dump target (level 3)
SGLANG_PRECISION_DEBUG=3 SGLANG_PRECISION_TAG=target ...

# Step 3: Compare offline
python -m sglang.kernel_precision_debug compare \
    --baseline /path/to/dumps/baseline \
    --target /path/to/dumps/target \
    --rank 0 --diff-threshold 1e-3
```

### Reading compare output

```
[PASS] call_000000/inputs.pt/x: shape=[1,128,4096] rel_diff=0
[FAIL] call_000000/outputs.pt/return: shape=[1,128,4096] rel_diff=0.015
```

Key pattern: **inputs PASS + outputs FAIL** = root cause candidate (but needs level 4 to confirm).

### Per-side dim0 slicing

When baseline and target have different batch/seq lengths:

```bash
python -m sglang.kernel_precision_debug compare \
    --baseline .../baseline --target .../target \
    --baseline-slice "0:4" --target-slice "0:4"

# Or pick specific dim0 indices
    --baseline-index "0,5,10" --target-index "0,5,10"
```

---

## Multi-Rank Comparison

For distributed inference (TP/PP):

```bash
# Dump all ranks
SGLANG_PRECISION_RANK_FILTER="" \
SGLANG_PRECISION_DEBUG=3 ...

# Level 4 align per rank
SGLANG_PRECISION_RANK_FILTER="0" \
SGLANG_PRECISION_DEBUG=4 ...

# Or compare offline per rank
for RANK in 0 1 2 3; do
  python -m sglang.kernel_precision_debug compare \
      --baseline .../baseline --target .../target --rank $RANK
done
```

---

## Typical Scenarios

| Scenario | Baseline | Target | Focus |
|----------|----------|--------|-------|
| FP32 vs BF16 | `--dtype float32` | `--dtype bfloat16` | Which kernel is most precision-sensitive |
| FlashInfer vs Triton | `SGLANG_ATTENTION_BACKEND=flashinfer` | `=triton` | Attention output diff |
| New vs old kernel | Old code branch | New code branch | Regression detection |
| Cross-GPU | H100 dump | H20 dump | Architecture-induced drift |
| Quantization | FP16 weights | INT8/FP8 quantized | Per-layer quantization loss |
| MoE routing | Reference TopK | Fused TopK | Router precision affects expert selection |
| DeepEP vs normal | `forward_normal()` | `forward_deepep()` | A2A dispatch precision |

### Model-specific notes

- **Qwen**: Dense model, standard flow. Focus on rmsnorm + linear precision.
- **DeepSeek V2**: MoE + MLA attention. Key areas: `moe.gate` routing precision
  (affects expert selection), `kv_a_proj`/`kv_b_proj` compressed KV precision,
  `moe.experts` FusedMoE GEMM. MoE layers and dense layers have separate counters.
- **Bailing MoE**: MoE + standard MHA. Key areas: `moe.gate` (uses `F.linear`),
  `moe.experts` (FusedMoE or DeepEP), optional QK norm precision.

---

## Tips

1. **Start with level 1** to see which ops are called and in what order — no GPU sync overhead.

2. **Use level 2** to spot NaN/Inf without generating dump files.

3. **Prefer the 2-step flow** (dump baseline + level 4) over the 3-step flow. Level 4
   gives per-op root cause judgment directly, without dumping target separately.

4. **Default `INDEX_FILTER=<10`** captures the first 10 invocations per op (0-based: indices
   0–9). For a 32-layer model, set `INDEX_FILTER="<32"` or `""` (all) to cover every layer.

5. **Default `RANK_FILTER=""`** captures all ranks. Set to `"0"` for rank 0 only.

6. **For inplace ops**, check `inplace_aligned` — it captures tensor state after the op
   modifies its inputs.

7. **Torch compile**: The decorator automatically skips during `torch.compile` tracing.

8. **CUDA graph**: Disable CUDA graph for the debug run to ensure tensors are accessible.

9. **`align_transform` callback**: For custom preprocessing of baseline tensors (dtype
   conversion, reshape, slice) in level 4:
   ```python
   def my_transform(tensors, category):
       # category: "inputs", "outputs", or "inputs_after"
       return {k: v.to(torch.bfloat16) for k, v in tensors.items()}

   @precision_debug(align_transform=my_transform)
   def my_kernel(...): ...
   ```

---

## Comparison CLI Full Reference

```bash
python -m sglang.kernel_precision_debug compare \
    --baseline <path>              # Baseline dump directory (required)
    --target <path>                # Target dump directory (required)
    --rank <int>                   # Rank to compare (default: 0)
    --op-filter <patterns>         # fnmatch op filter (default: all)
    --diff-threshold <float>       # PASS/FAIL threshold (default: 1e-3)
    --slice <start:stop>           # Dim0 slice for both sides
    --baseline-slice <start:stop>  # Dim0 slice for baseline only
    --target-slice <start:stop>    # Dim0 slice for target only
    --index <i,j,k>                # Dim0 indices for both sides
    --baseline-index <i,j,k>       # Dim0 indices for baseline only
    --target-index <i,j,k>         # Dim0 indices for target only
```
