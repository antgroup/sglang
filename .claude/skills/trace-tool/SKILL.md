# Trace Tool Skill

PyTorch Profiler Trace 截取、对比、合并与分析工具。用于处理 `.pt.trace.json` / `.json.gz` 文件。

详细使用文档见 [GUIDE.md](GUIDE.md)。

## 快速开始

```bash
CLI=".claude/skills/trace-tool/cli.py"

# 截取 step 2
python3 $CLI trim trace.json -o trimmed.json --step 2

# 截取第 10 到第 50 个算子（0-based）
python3 $CLI trim trace.json -o trimmed.json --from-nth 10 --to-nth 50

# 截取 forward 第 2 到第 3 次出现（0-based，END 不含）
python3 $CLI trim trace.json -o trimmed.json --block-range "forward:1:3"

# 截取第 2 个 aten::mm 到第 5 个 aten::mm（0-based: index 1 到 4）
python3 $CLI trim trace.json -o out.json --from-block "aten::mm:1" --to-block "aten::mm:4"

# 按算子名称过滤
python3 $CLI trim trace.json -o out.json -n gemm -n attention

# 对比两个 trace
python3 $CLI diff trace_a.json trace_b.json.gz

# 对比并导出 CSV
python3 $CLI diff trace_a.json trace_b.json --csv report.csv

# 合并两个 trace 可视化对比
python3 $CLI merge a.json b.json -o merged.json.gz

# 分析 CPU 调用层次
python3 $CLI blocks trace.json

# 展开特定 block
python3 $CLI blocks trace.json --expand forward:0 --max-depth 6
```

## 命令与参数速查

### trim - 截取 trace

```bash
python3 $CLI trim INPUT -o OUTPUT [OPTIONS]
```

| 选项 | 说明 | 示例 |
|------|------|------|
| `-o, --output` | 输出文件（必填） | `-o out.json` |
| `-n, --name` | 按名称过滤（可多次，OR） | `-n gemm -n attention` |
| `--time-start` | 起始时间（归一化 us） | `--time-start 1000000` |
| `--time-end` | 结束时间（归一化 us） | `--time-end 2000000` |
| `--step` | 按 ProfilerStep 编号截取 | `--step 2` |
| `--from-nth` | 从第 N 个算子开始（0-based） | `--from-nth 10` |
| `--to-nth` | 到第 N 个算子结束（0-based，含） | `--to-nth 50` |
| `--from-block` | 从第 N 次出现的算子开始（NAME:N） | `--from-block "aten::mm:2"` |
| `--to-block` | 到第 N 次出现的算子结束（NAME:N） | `--to-block "aten::mm:4"` |
| `--block-range` | 算子范围 NAME:START:END（END 不含） | `--block-range "forward:1:3"` |
| `--top` | 只保留耗时 top-N 算子类型 | `--top 20` |
| `--tid` | 按线程 ID 过滤（可多次） | `--tid 316659` |

### diff - 对比两个 trace

```bash
python3 $CLI diff TRACE_A TRACE_B [OPTIONS]
```

| 选项 | 说明 | 示例 |
|------|------|------|
| `--csv` | 导出 CSV | `--csv report.csv` |
| `--max-seq-diff` | 最大序列 diff 行数（默认 100） | `--max-seq-diff 200` |
| `--gpu-only` | 只对比 GPU 相关事件 | `--gpu-only` |
| `--block-name` | 指定重复 block 边界 | `--block-name forward` |
| `--blocks` | 分析 block 范围（0-based） | `--blocks 10:20` |

### merge - 合并可视化

```bash
python3 $CLI merge TRACE_A TRACE_B -o OUTPUT [OPTIONS]
```

| 选项 | 说明 | 示例 |
|------|------|------|
| `-o, --output` | 输出文件（必填） | `-o merged.json.gz` |
| `--threshold` | 耗时差异阈值 %（默认 10） | `--threshold 20` |
| `--no-flow` | 禁用 flow 箭头 | `--no-flow` |

### blocks - 调用层次分析

```bash
python3 $CLI blocks INPUT [OPTIONS]
```

| 选项 | 说明 | 示例 |
|------|------|------|
| `--min-dur` | 最小事件时长 us（默认 100） | `--min-dur 1000` |
| `--max-depth` | 最大深度（默认 6） | `--max-depth 10` |
| `--top` | 每层最多条数（默认 15） | `--top 30` |
| `-n, --name` | 按名称过滤（可多次，OR） | `-n forward` |
| `--expand` | 展开 block：NAME:IDX（0-based） | `--expand forward:0` |
| `--max-per-name` | 展开时同名子节点最多显示数（默认 3） | `--max-per-name 2` |

## 索引约定

- **所有索引从 0 开始**
- **NAME:N** — 冒号分隔名称和索引（如 `aten::mm:2` = 第 3 个 `aten::mm`）
- **NAME:START:END** — END 不含（同 Python slice）
- **名称匹配** — 子串匹配，非精确匹配

## 故障处理原则

> **如果本 skill 的任何命令失败，必须先考虑修复本工具，而不是绕过它！**

## 依赖

- Python 3.8+
- click
