# Kernel Precision Debug API

`kernel_precision_debug.py` -- 全自包含的 kernel 精度调试装饰器，仅依赖 torch + stdlib。
通过环境变量控制所有行为，装饰后零代码修改。

---

## 目录

1. [快速上手](#1-快速上手)
2. [装饰器参数](#2-装饰器参数-precision_debug)
3. [环境变量](#3-环境变量)
4. [Debug Level 详解](#4-debug-level-详解)
5. [输入处理机制](#5-输入处理机制)
6. [过滤器管线](#6-过滤器管线)
7. [Level 4 对齐模式](#7-level-4-对齐模式)
8. [比较 CLI](#8-比较-cli)
9. [Dump 文件结构](#9-dump-文件结构)
10. [跨框架精度对齐](#10-跨框架精度对齐)
11. [完整示例](#11-完整示例)
12. [FAQ](#12-faq)

---

## 1. 快速上手

```python
from sglang.kernel_precision_debug import precision_debug

# 最简用法 — 直接装饰
@precision_debug
def my_kernel(x, weight):
    return x @ weight

# 指定 op 名
@precision_debug(op_name="matmul")
def my_kernel(x, weight):
    return x @ weight
```

```bash
# 查看 shape（无 GPU 同步开销）
SGLANG_PRECISION_DEBUG=1 python my_script.py

# 查看统计信息（有 GPU 同步）
SGLANG_PRECISION_DEBUG=2 python my_script.py

# Dump tensor 到磁盘
SGLANG_PRECISION_DEBUG=3 SGLANG_PRECISION_TAG=baseline python my_script.py

# 对齐模式（与 baseline 逐 op 对比）
SGLANG_PRECISION_DEBUG=4 SGLANG_PRECISION_ALIGN_DIR=./dumps/baseline python my_script.py
```

`SGLANG_PRECISION_DEBUG=0`（默认）时装饰器直接返回原函数，**零开销**。

---

## 2. 装饰器参数 `precision_debug`

```python
@precision_debug(
    func=None,           # 被装饰函数（直接装饰时自动传入）
    *,
    op_name=None,        # str | None
    align_transform=None,# AlignTransform | None
    input_filter=None,   # InputFilter | None
    input_names=None,    # InputNames (list | dict | None)
)
```

### 2.1 `op_name: str | None`

自定义操作名称。默认从 `func.__qualname__` 推断。

```python
@precision_debug(op_name="sgl_kernel.rmsnorm")
def traced_rmsnorm(x, weight, eps):
    return rmsnorm(x, weight, eps)
```

日志输出: `[IN] sgl_kernel.rmsnorm #0  x:shape=[4,128],...`

### 2.2 `align_transform: AlignTransform | None`

Level 4 对齐模式下，对 baseline tensor 做自定义变换（在替换/比较之前调用）。

**类型签名:**
```python
AlignTransform = Callable[
    [dict[str, torch.Tensor], str],  # (tensors, category)
    dict[str, torch.Tensor]          # transformed tensors
]
# category: "inputs" | "outputs" | "inputs_after"
```

**示例 — baseline 是 fp32，target 是 bf16，需要 dtype 对齐:**
```python
def cast_to_bf16(tensors, category):
    return {k: v.to(torch.bfloat16) for k, v in tensors.items()}

@precision_debug(op_name="rmsnorm", align_transform=cast_to_bf16)
def rmsnorm(x, weight, eps=1e-5): ...
```

### 2.3 `input_filter: InputFilter | None`

从自定义类对象中提取 tensor 的回调。仅当参数不是 Tensor/dict/tuple/list/标量 时触发。

**类型签名:**
```python
InputFilter = Callable[
    [str, Any],                           # (param_name, value)
    Optional[Dict[str, torch.Tensor]]     # {dump_name: tensor} or None
]
```

**示例 — 从 ForwardBatch 提取 tensor:**
```python
class ForwardBatch:
    hidden_states: torch.Tensor   # [seq, hidden]
    positions: torch.Tensor       # [seq]
    temperature: float            # 非 tensor

def batch_filter(name, value):
    if isinstance(value, ForwardBatch):
        return {
            f"{name}.hidden_states": value.hidden_states,
            f"{name}.positions": value.positions,
        }
    return None  # 跳过其他类型

@precision_debug(op_name="attn", input_filter=batch_filter)
def attention(batch, weight):
    return batch.hidden_states @ weight
```

Dump 文件中的 key: `batch.hidden_states`, `batch.positions`, `weight`

**触发规则:**

| 参数类型 | 默认行为 | input_filter |
|---------|---------|-------------|
| `torch.Tensor` | 直接 dump，key=参数名 | 不触发 |
| `dict` | 递归展开，dot-path 命名 | 不触发 |
| `tuple` / `list` | 递归展开，索引命名 | 不触发 |
| `int/float/str/bool/None` | 存 metadata，不 dump | 不触发 |
| **自定义类** | **跳过 + [WARN]** | **触发** |

### 2.4 `input_names: InputNames`

输入名称过滤 + 跨框架名称映射。

**类型签名:**
```python
InputNames = Union[List[str], Dict[str, str], None]
```

#### 形式 1: `list` — 白名单过滤

只 dump/log 指定的参数，其余跳过并输出 `[SKIP]`。

```python
@precision_debug(op_name="linear", input_names=["x", "weight"])
def linear(x, weight, bias, eps=1e-5):
    return F.linear(x, weight, bias)
```

效果: 只 dump `x` 和 `weight`，`bias` 和 `eps` 被跳过。

#### 形式 2: `dict` — 白名单 + 重命名

key = 原始参数名，value = dump 名称。用于跨框架对齐。

```python
# SGLang 的参数名是 hidden_states, weight
@precision_debug(op_name="rmsnorm", input_names=["hidden_states", "weight"])
def sglang_rmsnorm(hidden_states, weight, eps=1e-5): ...

# vLLM 的参数名是 x, w — 映射到相同 dump key
@precision_debug(op_name="rmsnorm", input_names={"x": "hidden_states", "w": "weight"})
def vllm_rmsnorm(x, w, residual=None, eps=1e-6): ...
```

两者 dump 的 `inputs.pt` 都包含 key `hidden_states` 和 `weight` → 可直接 compare。

#### 形式 3: `None`（默认）

处理全部参数，无过滤。

#### 与 input_filter 组合使用

`input_names` 先过滤，`input_filter` 后提取:

```python
def batch_filter(name, value):
    if isinstance(value, ForwardBatch):
        return {f"{name}.hidden": value.hidden}
    return None

@precision_debug(
    op_name="attn",
    input_names=["batch", "weight"],  # 只处理这两个参数
    input_filter=batch_filter,         # batch 是自定义类，用 filter 提取
)
def attention(batch, weight, config):  # config 被 input_names 过滤掉
    ...
```

---

## 3. 环境变量

所有行为通过环境变量控制，**import 时一次性读取并缓存**（子进程需设置后再 import）。

| 变量 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `SGLANG_PRECISION_DEBUG` | int | `0` | Debug level: 0=off, 1=shapes, 2=+stats, 3=+dump, 4=align |
| `SGLANG_PRECISION_DUMP_DIR` | str | `sglang_precision_dumps` | Dump 目录。支持 `%i`=PID, `%r`=rank |
| `SGLANG_PRECISION_TAG` | str | `""` | 子目录标签: `"baseline"`, `"target"` 等 |
| `SGLANG_PRECISION_OP_FILTER` | str | `""` | fnmatch 模式过滤 op 名，逗号分隔。空=全部 |
| `SGLANG_PRECISION_INDEX_FILTER` | str | `<10` | 调用索引过滤（0-based）: `5-10`, `5,10,15`, `>=100`, `<10` |
| `SGLANG_PRECISION_SHAPE_FILTER` | str | `""` | tensor shape 条件: `dim0>=1024,dim-1<256` |
| `SGLANG_PRECISION_RANK_FILTER` | str | `""` | Rank 过滤: `0`, `0,1`, `0-3`。空=全部 |
| `SGLANG_PRECISION_LOG_DEST` | str | `stderr` | 日志输出: `stdout`, `stderr`, 或文件路径 |
| `SGLANG_PRECISION_RANGE_START` | str | `""` | 范围起始锚点: `"attention"` 或 `"attention#3"` |
| `SGLANG_PRECISION_RANGE_END` | str | `""` | 范围结束锚点（包含）。与 START 必须成对 |
| `SGLANG_PRECISION_ALIGN_DIR` | str | `""` | Level 4: baseline dump 目录路径 |
| `SGLANG_PRECISION_ALIGN_THRESHOLD` | float | `1e-3` | Level 4: rel_diff 的 PASS/FAIL 阈值 |

---

## 4. Debug Level 详解

### Level 0: Off（默认）

装饰器返回原函数，**零运行时开销**。

### Level 1: 形状日志

```
[IN]  rmsnorm #0  x:shape=[16,128],dtype=torch.float32  weight:shape=[128],dtype=torch.float32  eps=1e-05
[OUT] rmsnorm #0  return:shape=[16,128],dtype=torch.float32
```

- 无 GPU 同步
- 适合验证 instrumentation 是否正确

### Level 2: + 统计信息

```
[IN]  rmsnorm #0  x:shape=[16,128],dtype=torch.float32
  [STATS IN] rmsnorm #0  x: min=-2.35 max=3.12 mean=0.0012 nan=0 inf=0
[OUT] rmsnorm #0  return:shape=[16,128],dtype=torch.float32
  [STATS OUT] rmsnorm #0  return: min=-1.82 max=2.45 mean=0.0008 nan=0 inf=0
```

- **有 GPU 同步**（需要读取 tensor 值）
- 快速发现 NaN/Inf
- 额外输出 `[WARN]`/`[SKIP]` 警告:
  - `[WARN]`: 自定义类对象无 `input_filter`，被跳过
  - `[SKIP]`: 被 `input_names` 白名单排除的参数
  - `[WARN]`: 没有提取到任何 tensor

### Level 3: + Dump Tensor

将 tensor 保存到磁盘:
```
{DUMP_DIR}/{TAG}/rank{N}/{op_name}/call_{idx:06d}/
    inputs.pt        ← dict[str, Tensor]
    outputs.pt       ← dict[str, Tensor]
    inputs_after.pt  ← dict[str, Tensor]（inplace 修改后的状态）
    meta.json        ← 元信息（shape/dtype/stats/timestamp）
```

### Level 4: 对齐模式

从 `ALIGN_DIR` 加载 baseline 输入，替换当前输入后执行，比较输出。
详见 [Level 4 对齐模式](#7-level-4-对齐模式)。

---

## 5. 输入处理机制

装饰器自动处理各种输入类型:

### 5.1 Top-level Tensor

```python
@precision_debug(op_name="linear")
def linear(x, weight, bias):  # 三个 Tensor
    ...
```

Dump key: `x`, `weight`, `bias`

### 5.2 嵌套 dict

```python
@precision_debug(op_name="op")
def my_op(data):
    # data = {"hidden": tensor, "mask": tensor}
    ...
```

Dump key: `data.hidden`, `data.mask`（dot-path 命名）

### 5.3 嵌套 tuple / list

```python
@precision_debug(op_name="op")
def my_op(tensors):
    # tensors = (tensor_a, tensor_b)
    ...
```

Dump key: `tensors.0`, `tensors.1`（索引命名）

### 5.4 深层嵌套

```python
config = {
    "layers": [
        {"weight": tensor, "bias": tensor},
    ],
}
```

Dump key: `config.layers.0.weight`, `config.layers.0.bias`

递归深度上限: `max_depth=3`，防止无限递归。

### 5.5 自定义类（需要 `input_filter`）

```python
class ForwardBatch:
    hidden: torch.Tensor
    temperature: float

# 不提供 input_filter → 被跳过，level>=2 输出 [WARN]
# 提供 input_filter  → 按回调返回的 dict 提取
```

### 5.6 self / cls 自动跳过

`nn.Module` 方法的 `self` 和类方法的 `cls` 自动被识别并跳过:

```python
class MyModule(nn.Module):
    @precision_debug(op_name="my_forward")
    def forward(self, x, weight):  # self 被跳过
        ...
```

Dump 只包含 `x` 和 `weight`。

---

## 6. 过滤器管线

每次调用 wrapper 时，过滤器按以下顺序执行:

```
  rank_filter → op_filter → shape_filter → [counter++] → range_filter → index_filter
       ↓            ↓            ↓                              ↓            ↓
    跳过整个调用   跳过整个调用  跳过整个调用                  跳过该调用    跳过该调用
```

**关键**: counter 在 shape_filter 之后、range/index 之前递增。即使被 range/index 跳过，counter 也已经 +1。

### 6.1 Rank Filter (`RANK_FILTER`)

```bash
SGLANG_PRECISION_RANK_FILTER="0"      # 只捕获 rank 0
SGLANG_PRECISION_RANK_FILTER="0,1"    # 捕获 rank 0 和 1
SGLANG_PRECISION_RANK_FILTER="0-3"    # 捕获 rank 0~3
SGLANG_PRECISION_RANK_FILTER=""       # 全部（默认）
```

### 6.2 Op Filter (`OP_FILTER`)

```bash
SGLANG_PRECISION_OP_FILTER="rmsnorm"            # 精确匹配
SGLANG_PRECISION_OP_FILTER="*attn*"             # fnmatch 通配符
SGLANG_PRECISION_OP_FILTER="rmsnorm,*proj*"     # 多模式，逗号分隔
```

### 6.3 Shape Filter (`SHAPE_FILTER`)

基于 tensor 参数的 shape 维度过滤。**任一** tensor 参数满足即通过。

```bash
SGLANG_PRECISION_SHAPE_FILTER="dim0>=1024"          # 第 0 维 >= 1024
SGLANG_PRECISION_SHAPE_FILTER="dim-1<256"            # 最后一维 < 256
SGLANG_PRECISION_SHAPE_FILTER="dim0>=1024,dim-1<256" # 同时满足
```

### 6.4 Index Filter (`INDEX_FILTER`)

按 per-op 调用索引（0-based）过滤:

```bash
SGLANG_PRECISION_INDEX_FILTER="<10"        # 默认: 前 10 次调用 (0~9)
SGLANG_PRECISION_INDEX_FILTER="5-10"       # 第 5~10 次
SGLANG_PRECISION_INDEX_FILTER="5,10,15"    # 第 5/10/15 次
SGLANG_PRECISION_INDEX_FILTER=">=100"      # 第 100 次起
SGLANG_PRECISION_INDEX_FILTER=""           # 全部
```

### 6.5 Range Filter (`RANGE_START` / `RANGE_END`)

跨 op 的窗口过滤。从 START 匹配到 END（含）:

```bash
# 从 linear#1 到 linear#2 之间的所有 op
SGLANG_PRECISION_RANGE_START="linear#1"
SGLANG_PRECISION_RANGE_END="linear#2"
```

捕获: `linear #1`, `gelu #1`, `layernorm #2`, `linear #2`
排除: `layernorm #0`, `linear #0`, `gelu #0`, `gelu #2` 及之后

`#N` 中的 N 使用 per-op call counter（与 INDEX_FILTER 相同的计数器）。

---

## 7. Level 4 对齐模式

### 7.1 工作流程

**两步即可定位根因:**

```bash
# Step 1: Dump baseline（fp32 参考）
SGLANG_PRECISION_DEBUG=3 \
SGLANG_PRECISION_TAG=baseline \
SGLANG_PRECISION_DUMP_DIR=./dumps \
python run_model.py --dtype float32

# Step 2: 对齐模式运行 target（bf16）
SGLANG_PRECISION_DEBUG=4 \
SGLANG_PRECISION_ALIGN_DIR=./dumps/baseline \
SGLANG_PRECISION_ALIGN_THRESHOLD=1e-3 \
python run_model.py --dtype bfloat16
```

### 7.2 每个 op 调用的对齐过程

```
1. 加载 baseline inputs.pt
2. (可选) align_transform(baseline_inputs, "inputs")
3. 输出 input_diff: 实际输入 vs baseline 输入
4. 替换: 将实际输入替换为 baseline 输入
5. 执行 f(baseline_inputs)
6. 加载 baseline outputs.pt
7. 输出 output_aligned: 执行结果 vs baseline 输出  ← 核心判定
8. 加载 baseline inputs_after.pt
9. 输出 inplace_aligned: inplace 状态 vs baseline  ← inplace op 用
```

### 7.3 输出格式

```
[ALIGN] rmsnorm #0 — replacing inputs from .../call_000000
  [ALIGN PASS] #0 input_diff/x:           rel_diff=0
  [ALIGN PASS] #0 input_diff/weight:      rel_diff=0
  [ALIGN FAIL] #0 output_aligned/return:   rel_diff=3.2e-3   ← 根因!
```

### 7.4 判定规则

| 结果 | 含义 |
|------|------|
| `output_aligned PASS` | 该 op 没问题，divergence 来自上游 |
| `output_aligned FAIL` | **该 op 就是根因** — 相同输入产生不同输出 |

### 7.5 警告信息

| 标签 | 含义 |
|------|------|
| `[WARN] baseline keys not found in actual inputs (not replaced)` | baseline 有某些 key 但当前输入没有 |
| `[WARN] actual input keys not in baseline (not compared)` | 当前输入有某些 key 但 baseline 没有 |
| `[WARN] no tensors extracted from inputs` | 没有提取到任何 tensor |
| `[SKIP] arg 'bias' not in input_names` | 被 input_names 白名单排除 |
| `[WARN] arg 'batch' is ForwardBatch, skipped` | 自定义类无 input_filter |

### 7.6 相对差异公式

```
rel_diff = 1 - 2*(x·y) / (x² + y²)
```

来自 DeepGEMM。当 x == y 时 rel_diff = 0，完全不相关时 rel_diff = 1。

---

## 8. 比较 CLI

用于离线比较两次 dump（不使用 level 4 时的替代方案）。

```bash
python -m sglang.kernel_precision_debug compare \
    --baseline ./dumps/baseline \
    --target ./dumps/target \
    [options]
```

### 参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--baseline` | (必填) | Baseline dump 目录 |
| `--target` | (必填) | Target dump 目录 |
| `--rank` | `0` | 比较的 rank |
| `--op-filter` | `""` | fnmatch 模式过滤 |
| `--diff-threshold` | `1e-3` | PASS/FAIL 阈值 |
| `--slice` | `""` | 两侧 dim0 切片: `"5:10"` |
| `--baseline-slice` | `""` | 仅 baseline dim0 切片 |
| `--target-slice` | `""` | 仅 target dim0 切片 |
| `--index` | `""` | 两侧 dim0 索引: `"0,3,7"` |
| `--baseline-index` | `""` | 仅 baseline dim0 索引 |
| `--target-index` | `""` | 仅 target dim0 索引 |

### 跨框架 dim0 不一致处理

当 baseline 和 target 的 batch/seq 长度不同:

```bash
# 各取前 4 个 token
python -m sglang.kernel_precision_debug compare \
    --baseline ./baseline --target ./target \
    --baseline-slice "0:4" --target-slice "0:4"

# 选择特定 dim0 索引
    --baseline-index "0,5,10" --target-index "0,5,10"
```

单侧指定时，另一侧应用相同值。两侧仍不匹配时自动截断到 min(dim0)。

### 输出格式

```
=== Op: rmsnorm ===
[PASS] call_000000/inputs.pt/x:      shape=[16,128] rel_diff=0.000
[FAIL] call_000000/outputs.pt/return: shape=[16,128] rel_diff=0.015

Summary: 24 checked, 22 PASS, 2 FAIL
```

---

## 9. Dump 文件结构

```
{DUMP_DIR}/
  {TAG}/                     ← "baseline" 或 "target"
    rank{N}/                 ← rank 编号
      {op_name}/             ← 操作名（特殊字符替换为 _）
        call_{idx:06d}/      ← 调用索引（6 位补零）
          inputs.pt          ← torch.save({name: tensor, ...})
          outputs.pt         ← torch.save({"return": tensor} 或 {"return_0": t0, ...})
          inputs_after.pt    ← torch.save({name: tensor, ...}) inplace 后状态
          meta.json          ← 元信息
```

### meta.json 示例

```json
{
  "op_name": "sgl_kernel.rmsnorm",
  "call_index": 5,
  "rank": 0,
  "pid": 12345,
  "timestamp": "2026-04-12T10:30:00",
  "inputs": {
    "x": {"shape": [16, 128], "dtype": "torch.float32", "min": -2.35, "max": 3.12, "mean": 0.001, "nan_count": 0, "inf_count": 0},
    "weight": {"shape": [128], "dtype": "torch.float32", ...},
    "eps": 1e-05
  },
  "outputs": {
    "return": {"shape": [16, 128], ...}
  }
}
```

### inputs.pt 中的 key 命名规则

| 输入类型 | Key 格式 | 示例 |
|---------|---------|------|
| 直接 Tensor | 参数名 | `x`, `weight` |
| dict 嵌套 | `param.key` | `data.hidden`, `data.mask` |
| tuple/list | `param.index` | `tensors.0`, `tensors.1` |
| 深层嵌套 | `param.key.index.key` | `config.layers.0.weight` |
| input_filter | 回调返回的 key | `batch.hidden_states` |
| input_names 映射 | 映射后的 dump 名 | `hidden_states`（原名 `x`） |

---

## 10. 跨框架精度对齐

### 场景

SGLang vs vLLM 的同一模型，参数名不同但语义相同。

### 做法

```python
# === SGLang 端（baseline）===
@precision_debug(
    op_name="rmsnorm",
    input_names=["hidden_states", "weight"],  # list: 白名单，原名 dump
)
def sglang_rmsnorm(hidden_states, weight, eps=1e-5):
    return rmsnorm(hidden_states, weight, eps)

# === vLLM 端（target）===
@precision_debug(
    op_name="rmsnorm",
    input_names={"x": "hidden_states", "w": "weight"},  # dict: 映射到相同 key
)
def vllm_rmsnorm(x, w, residual=None, eps=1e-6):
    return rms_norm(x, w, eps)
```

两端 dump 的 `inputs.pt` 都包含 `hidden_states` 和 `weight` → 可直接比较。

### Level 4 对齐

```bash
# Step 1: SGLang dump baseline
SGLANG_PRECISION_DEBUG=3 SGLANG_PRECISION_TAG=baseline ... sglang_run.py

# Step 2: vLLM 用 level 4 对齐
SGLANG_PRECISION_DEBUG=4 SGLANG_PRECISION_ALIGN_DIR=./dumps/baseline ... vllm_run.py
```

替换时自动 reverse-map: baseline 的 `hidden_states` key → vLLM 的参数 `x` 位置。

---

## 11. 完整示例

### 11.1 基础: 多层模型精度调试

```python
import torch
import torch.nn.functional as F
from sglang.kernel_precision_debug import precision_debug

@precision_debug(op_name="layernorm")
def layernorm(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, x.shape[-1:], weight, bias, eps)

@precision_debug(op_name="linear")
def linear(x, weight, bias):
    return F.linear(x, weight, bias)

@precision_debug(op_name="gelu")
def gelu(x):
    return F.gelu(x)

# 4 层: [LayerNorm → Linear → GELU] × 4
for i in range(4):
    x = layernorm(x, ln_w, ln_b)   # layernorm #0, #1, #2, #3
    x = linear(x, w, b)            # linear #0, #1, #2, #3
    x = gelu(x)                    # gelu #0, #1, #2, #3
```

### 11.2 nn.Module 方法装饰（Pattern B）

```python
class DeepseekV2MoE(nn.Module):
    @precision_debug(op_name="moe.forward_normal")
    def forward_normal(self, hidden_states, topk_ids, topk_weights):
        # self 自动跳过，只 dump hidden_states, topk_ids, topk_weights
        ...
```

### 11.3 嵌套输入 + input_filter

```python
class ModelInput:
    def __init__(self, hidden, positions):
        self.hidden = hidden
        self.positions = positions

def extract_model_input(name, value):
    if isinstance(value, ModelInput):
        return {
            f"{name}.hidden": value.hidden,
            f"{name}.positions": value.positions,
        }
    return None

@precision_debug(op_name="layer", input_filter=extract_model_input)
def layer_forward(model_input, weight, config):
    # model_input → via input_filter → dump model_input.hidden, model_input.positions
    # weight → Tensor → dump weight
    # config → dict → 递归展开 → dump config.scale (if tensor)
    ...
```

### 11.4 input_names 跨框架对齐

```python
# Framework A
@precision_debug(op_name="attn", input_names=["q", "k", "v"])
def fw_a_attn(q, k, v, mask=None): ...

# Framework B — 参数名不同
@precision_debug(op_name="attn", input_names={"query": "q", "key": "k", "value": "v"})
def fw_b_attn(query, key, value, attention_mask=None, dropout=0.0): ...
```

### 11.5 Inplace 操作

```python
@precision_debug(op_name="fused_add_rmsnorm")
def fused_add_rmsnorm(x, residual, weight, eps):
    sgl_kernel.fused_add_rmsnorm(x, residual, weight, eps)
    return x  # inplace 修改 x 和 residual

# 装饰器自动 dump:
#   inputs.pt       ← x, residual (修改前)
#   inputs_after.pt ← x, residual (修改后)
#   outputs.pt      ← return 值
```

### 11.6 完整 debug 会话

```bash
# 1. 验证 instrumentation
SGLANG_PRECISION_DEBUG=1 python -m sglang.bench_one_batch \
    --model-path model --batch-size 1 --input-len 128 --output-len 1 2>&1 | head -50

# 2. 快速检查 NaN/Inf
SGLANG_PRECISION_DEBUG=2 SGLANG_PRECISION_INDEX_FILTER="<3" python -m sglang.bench_one_batch ...

# 3. Dump baseline (fp32)
SGLANG_PRECISION_DEBUG=3 SGLANG_PRECISION_TAG=baseline \
    SGLANG_PRECISION_DUMP_DIR=./dumps \
    python -m sglang.bench_one_batch ... --dtype float32

# 4. 对齐模式找根因 (bf16)
SGLANG_PRECISION_DEBUG=4 \
    SGLANG_PRECISION_ALIGN_DIR=./dumps/baseline \
    SGLANG_PRECISION_ALIGN_THRESHOLD=1e-3 \
    python -m sglang.bench_one_batch ... --dtype bfloat16 2>&1 | tee report.txt

# 5. 提取根因
grep "ALIGN FAIL.*output_aligned" report.txt
```

---

## 12. FAQ

### Q: level=0 时有性能开销吗？
A: 没有。`_LEVEL == 0` 时装饰器直接返回原函数。

### Q: 支持 torch.compile 吗？
A: wrapper 检测到 `torch.compiler.is_compiling()` 时自动跳过，不影响编译。

### Q: CUDA Graph 下能用吗？
A: 需要关闭 CUDA Graph，因为 dump/stats 需要访问 tensor 值。

### Q: 多进程/分布式怎么用？
A: `RANK_FILTER` 控制哪些 rank 捕获。Dump 目录自动按 `rank{N}` 分开。

### Q: counter 什么时候递增？
A: 在 shape_filter 之后、range/index 之前。即使被 range/index 跳过，counter 也 +1。

### Q: 环境变量何时读取？
A: import 时一次性读取并缓存。子进程需要在 import 前设置好环境变量。

### Q: input_names 的 dict 映射在 level 4 替换时怎么工作？
A: 自动 reverse-map。baseline key 是 dump 名，通过 `{dump_name: orig_name}` 反查定位原始参数位置进行替换。
