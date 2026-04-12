# Trace Tool 使用指南

PyTorch Profiler Trace 分析工具，用于截取、对比、合并和分析 `.pt.trace.json` / `.json.gz` 格式的 Chrome trace 文件。

## 目录

- [支持格式](#支持格式)
- [命令总览](#命令总览)
- [trim - 截取 trace](#trim---截取-trace)
- [diff - 对比两个 trace](#diff---对比两个-trace)
- [merge - 合并两个 trace](#merge---合并两个-trace)
- [blocks - CPU 调用层次分析](#blocks---cpu-调用层次分析)
- [典型工作流](#典型工作流)
- [索引与命名约定](#索引与命名约定)
- [故障处理](#故障处理)

## 支持格式

| 格式 | 说明 |
|------|------|
| `.json` | Chrome trace JSON |
| `.json.gz` | gzip 压缩的 Chrome trace JSON |
| `.pt.trace.json` | PyTorch profiler 导出格式 |

输入输出均按文件后缀自动判断格式。

## 命令总览

```bash
CLI=".claude/skills/trace-tool/cli.py"

python3 $CLI trim   INPUT -o OUTPUT [OPTIONS]     # 截取
python3 $CLI diff   TRACE_A TRACE_B [OPTIONS]     # 对比
python3 $CLI merge  TRACE_A TRACE_B -o OUT [OPT]  # 合并可视化
python3 $CLI blocks INPUT [OPTIONS]               # 调用层次分析
```

---

## trim - 截取 trace

从大 trace 文件中按条件截取子集，输出可在 `chrome://tracing` 或 Perfetto 中直接打开。

### 基本用法

```bash
python3 $CLI trim INPUT -o OUTPUT [OPTIONS]
```

### 全部参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `INPUT` | 位置参数 | 输入 trace 文件 |
| `-o, --output` | TEXT (必填) | 输出文件路径 |
| `-n, --name` | TEXT (可多次) | 按算子名称关键词过滤（多个为 OR 关系） |
| `--time-start` | FLOAT | 起始时间（归一化 us，以第一个算子为 0） |
| `--time-end` | FLOAT | 结束时间（归一化 us） |
| `--step` | INT | 按 ProfilerStep 编号截取 |
| `--from-nth` | INT | 从第 N 个 complete 事件开始（0-based） |
| `--to-nth` | INT | 到第 N 个 complete 事件结束（0-based，含） |
| `--from-block` | TEXT | 从第 N 次出现的指定算子开始（NAME 或 NAME:N，0-based） |
| `--to-block` | TEXT | 到第 N 次出现的指定算子结束（NAME 或 NAME:N，0-based，含） |
| `--block-range` | TEXT | 算子范围简写 NAME:START:END（0-based，END 不含） |
| `--top` | INT | 只保留耗时 top-N 的算子类型 |
| `--tid` | INT (可多次) | 按线程 ID 过滤 |

### 时间归一化

- 输出 trace 以**第一个 GPU kernel** 的 ts 作为 t=0
- `--time-start` / `--time-end` 基于**第一个算子**的 ts 归一化（非 GPU kernel）

### 过滤行为

- 所有过滤条件可组合使用，取交集
- 只有 metadata 事件（ph=M，如线程名、进程名）无条件保留
- flow 事件（ph=s/f）和 counter 事件（ph=C）同样受时间过滤

### 示例

```bash
# 截取 step 2
python3 $CLI trim trace.json -o step2.json --step 2

# 截取第 100-200 个算子（0-based）
python3 $CLI trim trace.json -o range.json --from-nth 100 --to-nth 200

# 截取第 2 个到第 5 个 aten::mm（0-based: index 1 到 4，含首尾）
python3 $CLI trim trace.json -o out.json --from-block "aten::mm:1" --to-block "aten::mm:4"

# 同上，使用 --block-range 简写（END 不含，所以写 5）
python3 $CLI trim trace.json -o out.json --block-range "aten::mm:1:5"

# 截取 forward 的第 2 和第 3 次出现（index 1 和 2，含中间事件）
python3 $CLI trim trace.json -o out.json --block-range "forward:1:3"

# 按名称过滤（OR 逻辑）
python3 $CLI trim trace.json -o out.json -n gemm -n attention

# 名称 + 时间范围组合
python3 $CLI trim trace.json -o out.json -n gemm --time-start 1000000 --time-end 2000000

# 只保留耗时最长的 20 种算子，输出 gzip
python3 $CLI trim trace.json.gz -o top20.json.gz --top 20

# 按线程 ID 过滤
python3 $CLI trim trace.json -o out.json --tid 316659
```

### --from-block / --to-block vs --block-range

两者等价，`--block-range` 是简写形式：

```bash
# 以下两种写法等价
--from-block "forward:1" --to-block "forward:4"
--block-range "forward:1:5"     # END 不含，所以 5 = 4+1
```

**注意**：算子名可能包含冒号（如 `aten::mm`），解析时取最后两个 `:` 分隔的部分作为 START:END，前面的全部作为名称。

---

## diff - 对比两个 trace

多维度分析两个 trace 的差异。

### 基本用法

```bash
python3 $CLI diff TRACE_A TRACE_B [OPTIONS]
```

### 全部参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `TRACE_A` | 位置参数 | 第一个 trace 文件 |
| `TRACE_B` | 位置参数 | 第二个 trace 文件 |
| `--csv` | TEXT | 导出对比报告到 CSV 文件 |
| `--max-seq-diff` | INT | 最大序列 diff 行数（默认 100） |
| `--gpu-only` | FLAG | 只对比 GPU/计算相关事件 |
| `--block-name` | TEXT | 用 CPU 事件名作为重复 block 边界（如 `run_batch`、`forward`） |
| `--blocks` | TEXT | 分析指定 block 范围（0-based，如 `10:20`、`100:`、`:50`） |

### 分析维度

1. **GPU Kernel 差异** - 只在 A/B 中出现的 kernel、耗时差异、替换检测
2. **GPU Kernel 调用序列** - 基于重复 block 检测的序列 diff，标注 CPU 调用栈
3. **调用路径级对比** - 重建调用树，按路径（如 `forward > attn > gemm`）聚合对比耗时
4. **子算子实现差异** - 同一父路径下子算子组成不同时高亮
5. **算子级耗时对比** - 按 total duration diff 排序
6. **整体统计** - GPU/CPU 总时间、算子数

### 示例

```bash
# 基本对比
python3 $CLI diff trace_a.json trace_b.json

# 只看 GPU 相关事件
python3 $CLI diff a.json b.json --gpu-only

# 指定 block 边界进行序列对比
python3 $CLI diff a.json b.json --block-name forward

# 只看第 10-20 个 block
python3 $CLI diff a.json b.json --block-name run_batch --blocks 10:20

# 导出 CSV
python3 $CLI diff a.json b.json --csv report.csv
```

### diff 输出示例

```
=== GPU Kernel Diff ===
Only in A (3):
  flash_attn_fwd          count=128  total=45.2ms
Only in B (2):
  flash_attn_v2_fwd       count=128  total=42.1ms

=== Call Path Diff (by |duration diff|) ===
Call Path                    | A avg(us) | B avg(us) | Diff
forward > attn > flash_attn |  1180.2   |     -     | REMOVED
forward > attn > flash_v2   |     -     |  1234.5   | NEW
forward > mlp > gemm        |   856.3   |   912.1   | +6.5%

=== Overall Stats ===
                 Trace A     Trace B     Diff
Total GPU:       45.2 ms     43.8 ms    -1.4 ms (-3.1%)
Total CPU:       52.1 ms     51.3 ms    -0.8 ms (-1.5%)
```

---

## merge - 合并两个 trace

将两个 trace 合并为一个文件，在 chrome://tracing 或 Perfetto 中可视化查看差异。

### 基本用法

```bash
python3 $CLI merge TRACE_A TRACE_B -o OUTPUT [OPTIONS]
```

### 全部参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `TRACE_A` | 位置参数 | 第一个 trace 文件 |
| `TRACE_B` | 位置参数 | 第二个 trace 文件 |
| `-o, --output` | TEXT (必填) | 输出合并文件 |
| `--threshold` | FLOAT | 耗时差异阈值百分比（默认 10%） |
| `--no-flow` | FLAG | 禁用 A/B 之间的 flow 箭头 |

### 可视化特性

- **进程分组**：A:GPU / A:CPU / B:GPU / B:CPU 各为独立进程，上下对照
- **颜色标注**：
  - 默认蓝色 - 两边都有且耗时接近
  - 红色标签 `[ONLY_A]` / `[ONLY_B]` - 只在一边出现
  - 黄色标签 `[SLOW +xx%]` / `[FAST -xx%]` - 耗时差异超过阈值
- **配对对比**：第 N 次出现 vs 第 N 次出现（非平均值对比）
- **Flow 箭头**：连接 A/B 中匹配的 GPU kernel
- **详情面板**：点击事件可看到 `diff_pct`、`other_dur`、`diff_status` 等

### 示例

```bash
# 基本合并
python3 $CLI merge a.json b.json -o merged.json.gz

# 自定义差异阈值 20%
python3 $CLI merge a.json b.json -o merged.json.gz --threshold 20

# 禁用 flow 箭头（减少视觉干扰）
python3 $CLI merge a.json b.json -o merged.json.gz --no-flow
```

---

## blocks - CPU 调用层次分析

分析 trace 的 CPU 事件层次结构，用于理解调用关系、发现重复 block 模式。

### 基本用法

```bash
python3 $CLI blocks INPUT [OPTIONS]
```

### 全部参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `INPUT` | 位置参数 | 输入 trace 文件 |
| `--min-dur` | FLOAT | 最小事件持续时间 us（默认 100） |
| `--max-depth` | INT | 最大调用深度（默认 6） |
| `--top` | INT | 每层最多显示条数（默认 15） |
| `-n, --name` | TEXT (可多次) | 按名称关键词过滤（OR 逻辑） |
| `--expand` | TEXT | 展开指定 block：NAME:IDX（0-based） |
| `--max-per-name` | INT | 展开模式下同名子节点最多显示数（默认 3） |

### 功能说明

1. **层次视图**：按调用嵌套深度分层显示，从最外层（长时间运行）到内层
2. **统计信息**：每个算子显示 count、avg/min/max duration、调用路径
3. **模式检测**：自动检测重复 block 模式，建议 `--block-name` 参数
4. **展开模式**：深入某个 block 实例，查看完整调用树和 GPU kernel 详情

### 展开模式特性

- **模式分组**：名称只差尾部数字的子节点（如 `DecoderLayer_0` ~ `DecoderLayer_77`）自动归为同类，只显示前 N 个 + 汇总行
- **GPU kernel 聚合**：按 kernel 名称聚合，显示 count / avg / total / range

### 示例

```bash
# 基本层次视图
python3 $CLI blocks trace.json

# 只看 > 1ms 的事件，深度限制 4
python3 $CLI blocks trace.json --min-dur 1000 --max-depth 4

# 按名称过滤
python3 $CLI blocks trace.json -n forward -n backward

# 展开第 1 个 run_batch（0-based），深度 10
python3 $CLI blocks trace.json --expand run_batch:0 --max-depth 10

# 展开第 5 个 CompiledFxGraph（0-based: index 4）
python3 $CLI blocks trace.json --expand CompiledFxGraph:4

# 展开 dispatch_event_loop 的第 2 个实例，同名子节点最多显示 2 个
python3 $CLI blocks trace.json --expand dispatch_event_loop:1 --max-depth 4 --max-per-name 2
```

### blocks 输出示例

```
=== Level 0: Outermost intervals ===
  dispatch_event_loop  count=1  dur=5234.5ms  path: dispatch_event_loop

=== Level 1 ===
  run_batch            count=5  avg=1046.9ms  path: dispatch > run_batch
  process_batch_result count=5  avg=293.3ms   path: dispatch > process_batch

Suggested --block-name: run_batch (5 repeats, avg 1046.9ms)
```

---

## 典型工作流

### 1. 分析单个 trace

```bash
# 先看整体结构
python3 $CLI blocks trace.json

# 展开感兴趣的 block
python3 $CLI blocks trace.json --expand forward:0 --max-depth 6

# 截取某个迭代
python3 $CLI trim trace.json -o iter2.json --block-range "forward:2:3"

# 只保留 top-20 算子
python3 $CLI trim trace.json -o top.json --top 20
```

### 2. 对比两个 trace（优化前后）

```bash
# 先各自截取同一个迭代
python3 $CLI trim before.json -o a.json --block-range "forward:2:3"
python3 $CLI trim after.json  -o b.json --block-range "forward:2:3"

# 文本对比
python3 $CLI diff a.json b.json --block-name forward

# 可视化对比（在 chrome://tracing 打开）
python3 $CLI merge a.json b.json -o merged.json.gz
```

### 3. 大文件处理

```bash
# 先截取 step
python3 $CLI trim huge.json.gz -o step2.json.gz --step 2

# 再从 step 中截取特定 block
python3 $CLI trim step2.json.gz -o target.json --block-range "forward:1:3"
```

---

## 索引与命名约定

### 索引

所有命令的索引**统一从 0 开始**：

| 参数 | 索引含义 |
|------|---------|
| `--from-nth 0` | 第 1 个 complete 事件 |
| `--from-block "forward:0"` | forward 的第 1 次出现 |
| `--block-range "forward:1:3"` | forward 第 2 到第 3 次（END 不含） |
| `--blocks 0:10` | 第 1 到第 10 个 block（diff 命令） |
| `--expand "forward:0"` | 展开 forward 的第 1 个实例 |

### 名称匹配

- 所有名称参数使用**子串匹配**（不是精确匹配）
- `--from-block "forward"` 会匹配 `model_runner.py(2907): forward`
- 名称中可以包含冒号（如 `aten::mm`），解析器自动处理

### 冒号语法

统一使用冒号分隔名称和索引：

```
NAME:INDEX     →  --from-block "aten::mm:2"   （第 3 个 aten::mm）
NAME:START:END →  --block-range "aten::mm:1:4" （第 2 到第 4 个）
NAME:IDX       →  --expand "forward:0"         （展开第 1 个）
START:END      →  --blocks "10:20"             （第 11 到第 20 个 block）
```

---

## 故障处理

> **如果本工具的任何命令失败，必须先考虑修复本工具，而不是绕过它！**

## 依赖

- Python 3.8+
- click（CLI 框架）
- 仅使用标准库（json, gzip, csv, collections, difflib, re）
