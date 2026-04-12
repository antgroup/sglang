"""Blocks subcommand - show CPU interval hierarchy for block analysis."""

import sys
from collections import defaultdict
from pathlib import Path

skill_dir = Path(__file__).parent
if str(skill_dir) not in sys.path:
    sys.path.insert(0, str(skill_dir))

import click

from call_tree import CallNode, build_call_trees
from trace_io import get_events, get_first_ts, is_complete_event, load_trace

_GPU_CATS = {"kernel", "gpu_user_annotation", "gpu_memcpy"}


def _fmt_dur(us):
    """Format duration in human-readable form."""
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f}s"
    if us >= 1000:
        return f"{us / 1000:.1f}ms"
    return f"{us:.0f}us"


def _find_cpu_thread(events):
    """Find the main CPU thread (pid, tid) with the most non-GPU complete events."""
    counts = defaultdict(int)
    for e in events:
        if (e.get("ph") == "X" and e.get("cat", "") not in _GPU_CATS
                and "ts" in e and "dur" in e):
            counts[(e.get("pid", 0), e.get("tid", 0))] += 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _depth_of(node):
    """Compute depth of a node (0 = root)."""
    d = 0
    n = node.parent
    while n is not None:
        d += 1
        n = n.parent
    return d


def _collect_levels(roots, max_depth=8):
    """Walk the call tree and group events by depth level.

    Returns dict: depth -> list of CallNode
    """
    levels = defaultdict(list)

    def _walk(node, depth):
        levels[depth].append(node)
        if depth < max_depth:
            for child in node.children:
                _walk(child, depth + 1)

    for root in roots:
        _walk(root, 0)
    return levels


def _aggregate_level(nodes):
    """Aggregate nodes at the same level by name.

    Returns list of dicts sorted by total_dur desc:
        [{name, count, total_dur, avg_dur, min_dur, max_dur, example_path}]
    """
    by_name = defaultdict(lambda: {
        "count": 0, "total_dur": 0, "min_dur": float("inf"),
        "max_dur": 0, "example_node": None,
    })
    for node in nodes:
        rec = by_name[node.name]
        rec["count"] += 1
        rec["total_dur"] += node.dur
        rec["min_dur"] = min(rec["min_dur"], node.dur)
        rec["max_dur"] = max(rec["max_dur"], node.dur)
        if rec["example_node"] is None:
            rec["example_node"] = node

    result = []
    for name, rec in by_name.items():
        path = rec["example_node"].call_path() if rec["example_node"] else name
        result.append({
            "name": name,
            "count": rec["count"],
            "total_dur": rec["total_dur"],
            "avg_dur": rec["total_dur"] / rec["count"],
            "min_dur": rec["min_dur"],
            "max_dur": rec["max_dur"],
            "call_path": path,
        })
    result.sort(key=lambda x: -x["total_dur"])
    return result


def _count_gpu_in_range(events, ts_start, ts_end):
    """Count GPU kernel events within a time range."""
    count = 0
    for e in events:
        if (e.get("ph") == "X" and e.get("cat", "") in _GPU_CATS
                and "ts" in e):
            if e["ts"] >= ts_start and e["ts"] < ts_end:
                count += 1
    return count


def _iter_all(nodes):
    """Iterate all nodes in a tree (depth-first)."""
    for n in nodes:
        yield n
        yield from _iter_all(n.children)


def _build_gpu_ts_list(events):
    """Build a sorted list of GPU kernel start timestamps for fast lookup."""
    gpu_ts = sorted(
        e["ts"] for e in events
        if e.get("ph") == "X" and e.get("cat", "") in _GPU_CATS and "ts" in e
    )
    return gpu_ts


def _has_gpu_in_range(gpu_ts_list, ts_start, ts_end):
    """Check if any GPU kernel starts within [ts_start, ts_end) using bisect."""
    import bisect
    idx = bisect.bisect_left(gpu_ts_list, ts_start)
    return idx < len(gpu_ts_list) and gpu_ts_list[idx] < ts_end


def _prune_no_gpu(nodes, gpu_ts_list):
    """Remove call tree nodes whose time range contains no GPU kernels.

    Bottom-up: a node is kept if it directly covers a GPU kernel,
    or if any of its children are kept.
    """
    result = []
    for n in nodes:
        n.children = _prune_no_gpu(n.children, gpu_ts_list)
        if n.children or _has_gpu_in_range(gpu_ts_list, n.ts, n.ts + n.dur):
            result.append(n)
    return result


def _normalize_name(name):
    """Strip trailing numeric suffixes to group similar names.

    e.g. 'nn.Module: DeepseekV2DecoderLayer_42' -> 'nn.Module: DeepseekV2DecoderLayer_*'
         'aten::mm' -> 'aten::mm'
    """
    import re
    return re.sub(r'_\d+$', '_*', name)


def _print_subtree(node, events, gpu_ts_list, base_ts, depth=0, max_depth=10,
                   max_per_name=3):
    """Recursively print a call tree node and its children with GPU kernel info.

    max_per_name: max times to print children with the same pattern at each level.
    Names are grouped by pattern (trailing numbers stripped), so
    DeepseekV2DecoderLayer_0 .. _77 count as the same pattern.
    """
    indent = "    " + "  " * depth
    ts_norm = node.ts - base_ts
    has_gpu = _has_gpu_in_range(gpu_ts_list, node.ts, node.ts + node.dur)
    gpu_tag = " [GPU]" if has_gpu and not node.children else ""

    name = node.name
    if len(name) > 100:
        name = name[:97] + "..."
    click.echo(
        f"{indent}{name}  "
        f"dur={_fmt_dur(node.dur)}  t={_fmt_dur(ts_norm)}{gpu_tag}"
    )

    if depth < max_depth:
        # Group children by normalized pattern to limit repetition
        pattern_counts = defaultdict(int)
        pattern_stats = defaultdict(lambda: {
            "count": 0, "total_dur": 0, "min_dur": float("inf"),
            "max_dur": 0,
        })
        for child in node.children:
            pat = _normalize_name(child.name)
            pattern_stats[pat]["count"] += 1
            pattern_stats[pat]["total_dur"] += child.dur
            pattern_stats[pat]["min_dur"] = min(pattern_stats[pat]["min_dur"], child.dur)
            pattern_stats[pat]["max_dur"] = max(pattern_stats[pat]["max_dur"], child.dur)

        for child in node.children:
            pat = _normalize_name(child.name)
            pattern_counts[pat] += 1
            if pattern_counts[pat] <= max_per_name:
                _print_subtree(child, events, gpu_ts_list, base_ts,
                               depth + 1, max_depth, max_per_name)
            elif pattern_counts[pat] == max_per_name + 1:
                # Print summary for remaining same-pattern siblings
                st = pattern_stats[pat]
                skipped = st["count"] - max_per_name
                avg_dur = st["total_dur"] / st["count"]
                child_indent = "    " + "  " * (depth + 1)
                pat_display = pat if len(pat) <= 60 else pat[:57] + "..."
                click.echo(
                    f"{child_indent}... 同类 \"{pat_display}\" "
                    f"还有 {skipped} 个 "
                    f"(avg={_fmt_dur(avg_dur)}, "
                    f"range=[{_fmt_dur(st['min_dur'])}, {_fmt_dur(st['max_dur'])}])"
                )

    # At leaf level, show GPU kernels launched within this node's time range
    if not node.children and has_gpu:
        import bisect
        lo = bisect.bisect_left(gpu_ts_list, node.ts)
        hi = bisect.bisect_left(gpu_ts_list, node.ts + node.dur)
        n_kernels = hi - lo
        if n_kernels > 0:
            click.echo(f"{indent}  └─ {n_kernels} GPU kernels")


def _do_expand(trace, block_name, block_idx, min_dur=10, max_depth=10,
               max_per_name=3):
    """Expand the block_idx-th (0-based) occurrence of block_name.

    Shows the full call subtree under that CPU interval, including
    GPU kernels at leaf nodes.
    """
    events = get_events(trace)
    base_ts = get_first_ts(events)

    cpu_thread = _find_cpu_thread(events)
    if cpu_thread is None:
        click.echo("  未找到 CPU 线程事件")
        return

    gpu_ts_list = _build_gpu_ts_list(events)

    # Build call tree for the main CPU thread
    trees = build_call_trees(events)
    roots = trees.get(cpu_thread, [])
    if not roots:
        click.echo("  未构建到 CPU 调用树")
        return

    # Find all nodes matching block_name (DFS, sorted by ts)
    matches = []

    def _find(nodes):
        for n in nodes:
            if block_name.lower() in n.name.lower():
                matches.append(n)
            _find(n.children)

    _find(roots)
    matches.sort(key=lambda n: n.ts)

    if not matches:
        click.echo(f"  未找到匹配 \"{block_name}\" 的 CPU 事件")
        return

    click.echo(f"  匹配 \"{block_name}\" 的 CPU 事件: {len(matches)} 个")

    if block_idx < 0 or block_idx >= len(matches):
        click.echo(f"  索引 {block_idx} 超出范围 [0, {len(matches) - 1}]")
        # Show a summary of all matches
        click.echo(f"\n  可用索引:")
        for i, m in enumerate(matches[:20]):
            ts_norm = m.ts - base_ts
            n_kids = len(m.children)
            click.echo(
                f"    #{i:<4d} t={_fmt_dur(ts_norm):<12s} "
                f"dur={_fmt_dur(m.dur):<12s} "
                f"children={n_kids}"
            )
        if len(matches) > 20:
            click.echo(f"    ... 共 {len(matches)} 个")
        return

    target = matches[block_idx]
    ts_norm = target.ts - base_ts
    click.echo(
        f"\n  展开: \"{block_name}\" #{block_idx}/{len(matches)}"
    )
    click.echo(
        f"  时间: t={_fmt_dur(ts_norm)}, dur={_fmt_dur(target.dur)}"
    )

    # Count GPU kernels in this range
    import bisect
    lo = bisect.bisect_left(gpu_ts_list, target.ts)
    hi = bisect.bisect_left(gpu_ts_list, target.ts + target.dur)
    click.echo(f"  GPU kernels: {hi - lo}")

    # Filter children by min_dur and prune no-gpu
    def _filter_and_prune(nodes):
        result = []
        for n in nodes:
            if n.dur >= min_dur:
                n.children = _filter_and_prune(n.children)
                if n.children or _has_gpu_in_range(gpu_ts_list, n.ts, n.ts + n.dur):
                    result.append(n)
        return result

    target.children = _filter_and_prune(target.children)

    click.echo()
    _print_subtree(target, events, gpu_ts_list, base_ts,
                   depth=0, max_depth=max_depth, max_per_name=max_per_name)

    # GPU kernel summary by name (aggregated)
    gpu_events = [
        e for e in events
        if e.get("ph") == "X" and e.get("cat", "") in _GPU_CATS
        and e["ts"] >= target.ts and e["ts"] < target.ts + target.dur
        and e.get("dur", 0) >= min_dur
    ]
    if gpu_events:
        # Aggregate by name
        gpu_by_name = defaultdict(lambda: {
            "count": 0, "total_dur": 0,
            "min_dur": float("inf"), "max_dur": 0,
        })
        for e in gpu_events:
            rec = gpu_by_name[e["name"]]
            rec["count"] += 1
            dur = e.get("dur", 0)
            rec["total_dur"] += dur
            rec["min_dur"] = min(rec["min_dur"], dur)
            rec["max_dur"] = max(rec["max_dur"], dur)

        gpu_agg = sorted(gpu_by_name.items(), key=lambda x: -x[1]["total_dur"])
        total_gpu_dur = sum(r["total_dur"] for _, r in gpu_agg)

        click.echo(
            f"\n  [GPU kernels] {len(gpu_events)} 个, "
            f"{len(gpu_agg)} 种 (dur >= {_fmt_dur(min_dur)}, "
            f"总耗时 {_fmt_dur(total_gpu_dur)})"
        )
        for i, (name, rec) in enumerate(gpu_agg[:30]):
            display = name if len(name) <= 90 else name[:87] + "..."
            avg = rec["total_dur"] / rec["count"]
            dur_range = ""
            if rec["count"] > 1:
                dur_range = (
                    f"  range=[{_fmt_dur(rec['min_dur'])}, "
                    f"{_fmt_dur(rec['max_dur'])}]"
                )
            click.echo(
                f"    {display}\n"
                f"      x{rec['count']:<5d}  "
                f"avg={_fmt_dur(avg):<10s}  "
                f"total={_fmt_dur(rec['total_dur']):<10s}"
                f"{dur_range}"
            )
        if len(gpu_agg) > 30:
            click.echo(f"    ... 还有 {len(gpu_agg) - 30} 种")
    click.echo()


def do_blocks(trace, min_dur=100, max_depth=6, top_n=15, name_filter=None):
    """Analyze CPU interval hierarchy in a trace.

    Shows levels from outermost (long-running) to inner (short) intervals,
    with count, name, duration stats, and call path context.
    Only shows CPU intervals that contain GPU kernel activity.
    """
    events = get_events(trace)
    base_ts = get_first_ts(events)

    cpu_thread = _find_cpu_thread(events)
    if cpu_thread is None:
        click.echo("  未找到 CPU 线程事件")
        return

    cpu_pid, cpu_tid = cpu_thread
    click.echo(f"  CPU 主线程: pid={cpu_pid}, tid={cpu_tid}")

    # Build sorted GPU timestamp list for fast range queries
    gpu_ts_list = _build_gpu_ts_list(events)
    click.echo(f"  GPU kernel 总数: {len(gpu_ts_list)}")

    # Build call tree for the main CPU thread
    trees = build_call_trees(events)
    roots = trees.get(cpu_thread, [])
    if not roots:
        click.echo("  未构建到 CPU 调用树")
        return

    click.echo(f"  CPU 顶层事件数: {len(roots)}")

    # Filter by min_dur — prune nodes below threshold
    def _filter_tree(nodes, min_d):
        result = []
        for n in nodes:
            if n.dur >= min_d:
                n.children = _filter_tree(n.children, min_d)
                result.append(n)
        return result

    filtered_roots = _filter_tree(roots, min_dur)

    # Prune CPU intervals that contain no GPU kernels
    n_before = sum(1 for _ in _iter_all(filtered_roots))
    filtered_roots = _prune_no_gpu(filtered_roots, gpu_ts_list)
    n_after = sum(1 for _ in _iter_all(filtered_roots))
    click.echo(f"  GPU 过滤: {n_before} → {n_after} 个 CPU 区间 (去除无 GPU kernel 的区间)")

    # Collect levels
    levels = _collect_levels(filtered_roots, max_depth=max_depth)

    if not levels:
        click.echo(f"  无满足 min_dur={_fmt_dur(min_dur)} 的 CPU 事件")
        return

    total_cpu_events = sum(len(v) for v in levels.values())
    click.echo(
        f"  满足条件的 CPU 事件: {total_cpu_events} "
        f"(min_dur={_fmt_dur(min_dur)}, max_depth={max_depth})\n"
    )

    # Print each level
    for depth in sorted(levels.keys()):
        nodes = levels[depth]
        agg = _aggregate_level(nodes)

        # Apply name filter
        if name_filter:
            agg = [a for a in agg
                   if any(kw.lower() in a["name"].lower() for kw in name_filter)]

        if not agg:
            continue

        indent = "  " * (depth + 1)
        total_dur = sum(a["total_dur"] for a in agg)
        click.echo(
            f"  {'─' * 70}"
        )
        click.echo(
            f"  层级 {depth}  |  {len(nodes)} 个事件, "
            f"{len(agg)} 种名称, 总耗时 {_fmt_dur(total_dur)}"
        )

        for i, a in enumerate(agg[:top_n]):
            dur_range = ""
            if a["count"] > 1:
                dur_range = (
                    f"  range=[{_fmt_dur(a['min_dur'])}, {_fmt_dur(a['max_dur'])}]"
                )

            name_display = a["name"]
            if len(name_display) > 90:
                name_display = name_display[:87] + "..."

            click.echo(
                f"{indent}{name_display}"
            )
            click.echo(
                f"{indent}  x{a['count']:<5d}  "
                f"avg={_fmt_dur(a['avg_dur']):<10s}  "
                f"total={_fmt_dur(a['total_dur']):<10s}"
                f"{dur_range}"
            )
            # Show call path if it differs from name (i.e., has parent context)
            if " > " in a["call_path"]:
                path = a["call_path"]
                if len(path) > 100:
                    # Show last 100 chars with ellipsis
                    path = "..." + path[-97:]
                click.echo(f"{indent}  路径: {path}")

        if len(agg) > top_n:
            click.echo(f"{indent}... 还有 {len(agg) - top_n} 种 (已省略)")
        click.echo()

    # === Repeating pattern detection ===
    click.echo(f"  {'═' * 70}")
    click.echo("  [重复模式检测]")
    click.echo()

    # For each level, check if names form a repeating pattern
    for depth in sorted(levels.keys()):
        nodes = levels[depth]
        if len(nodes) < 4:
            continue

        names = [n.name for n in nodes]
        # Find most-common names
        name_counts = defaultdict(int)
        for n in names:
            name_counts[n] += 1

        # Report any name that repeats >= 3 times and >= 30% of this level
        candidates = sorted(
            [(n, c) for n, c in name_counts.items()
             if c >= 3 and c >= len(nodes) * 0.3],
            key=lambda x: -x[1],
        )
        if not candidates:
            continue

        for top_name, top_count in candidates:
            # Good candidate for --block-name
            # Compute duration stats for this repeating name
            durs = [n.dur for n in nodes if n.name == top_name]
            avg_d = sum(durs) / len(durs)
            min_d = min(durs)
            max_d = max(durs)

            # Check for duration variation
            spread = (max_d - min_d) / avg_d * 100 if avg_d > 0 else 0

            click.echo(
                f"  层级 {depth}: \"{top_name}\" "
                f"重复 {top_count} 次"
            )
            click.echo(
                f"    avg={_fmt_dur(avg_d)}  "
                f"range=[{_fmt_dur(min_d)}, {_fmt_dur(max_d)}]  "
                f"spread={spread:.1f}%"
            )

            # Extract a keyword from the name for --block-name suggestion
            # Use the shortest unique substring that identifies this name
            suggest_name = top_name
            # If name is long, try to extract a meaningful keyword
            for part in top_name.replace("#", " ").replace("(", " ").split():
                if len(part) >= 6 and not part.startswith(("/", ".", "<")):
                    suggest_name = part
                    break

            click.echo(
                f"    → 建议: --block-name \"{suggest_name}\""
            )
            click.echo()

    # === Top-level interval list ===
    # Show the first few top-level events as a timeline
    click.echo(f"  {'═' * 70}")
    click.echo("  [顶层事件时间线] (前 30 个)")
    click.echo()
    for i, root in enumerate(filtered_roots[:30]):
        ts_norm = root.ts - base_ts
        n_children = len(root.children)
        name = root.name
        if len(name) > 80:
            name = name[:77] + "..."
        click.echo(
            f"    #{i:<4d} t={_fmt_dur(ts_norm):<12s} "
            f"dur={_fmt_dur(root.dur):<12s} "
            f"children={n_children:<4d} "
            f"{name}"
        )
    if len(filtered_roots) > 30:
        click.echo(f"    ... 共 {len(filtered_roots)} 个顶层事件")
    click.echo()


@click.command("blocks")
@click.argument("input_file", metavar="INPUT")
@click.option("--min-dur", type=float, default=100,
              help="Minimum CPU event duration in us (default: 100)")
@click.option("--max-depth", type=int, default=6,
              help="Maximum call tree depth to show (default: 6)")
@click.option("--top", type=int, default=15,
              help="Max entries per level (default: 15)")
@click.option("-n", "--name", multiple=True,
              help="Filter by name keyword (repeatable, OR logic)")
@click.option("--expand", type=str, default=None,
              help="Expand a specific block: NAME:IDX (0-based, e.g. 'CompiledFxGraph:4')")
@click.option("--max-per-name", type=int, default=3,
              help="Max same-name siblings to print per level in expand mode (default: 3)")
def blocks(input_file, min_dur, max_depth, top, name, expand, max_per_name):
    """Show CPU interval hierarchy for block analysis.

    Displays CPU events organized by nesting depth, from outermost
    (long-running) to inner intervals. Shows count, duration stats,
    call path, and detects repeating patterns suitable for --block-name.

    Use --expand to drill into a specific block occurrence and see its
    full call tree with GPU kernel details.

    \b
    Examples:
      # Basic hierarchy view
      trace-tool blocks trace.json

      # Only show events > 1ms, up to depth 4
      trace-tool blocks trace.json --min-dur 1000 --max-depth 4

      # Filter by name
      trace-tool blocks trace.json -n forward -n backward

      # Expand the 5th CompiledFxGraph block (0-based)
      trace-tool blocks trace.json --expand CompiledFxGraph:4

      # Expand the 1st run_batch block (index 0) with deeper depth
      trace-tool blocks trace.json --expand run_batch:0 --max-depth 10

      # Then use the suggested --block-name in diff:
      trace-tool diff a.json b.json --block-name CompiledFxGraph
    """
    click.echo(f"Loading {input_file} ...")
    trace = load_trace(input_file)
    click.echo()

    if expand:
        # Parse NAME:IDX
        if ":" not in expand:
            raise click.ClickException(
                f"--expand 格式应为 NAME:IDX (如 'CompiledFxGraph:5'), 收到: '{expand}'"
            )
        parts = expand.rsplit(":", 1)
        block_name = parts[0]
        try:
            block_idx = int(parts[1])
        except ValueError:
            raise click.ClickException(
                f"--expand 索引必须是整数, 收到: '{parts[1]}'"
            )
        click.echo(f"=== 展开 CPU 区间: \"{block_name}\" #{block_idx} ===")
        _do_expand(trace, block_name, block_idx, min_dur=min_dur,
                   max_depth=max_depth, max_per_name=max_per_name)
    else:
        click.echo("=== CPU 区间层级分析 ===")
        do_blocks(trace, min_dur=min_dur, max_depth=max_depth, top_n=top,
                  name_filter=list(name) if name else None)
