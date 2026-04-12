"""Diff subcommand - compare two trace files."""

import csv
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

skill_dir = Path(__file__).parent
if str(skill_dir) not in sys.path:
    sys.path.insert(0, str(skill_dir))

import click

from trace_io import get_events, get_first_ts, is_complete_event, get_op_sequence, load_trace
from call_tree import build_call_trees, collect_all_paths

# Categories that contain GPU-related operations in PyTorch profiler traces
_GPU_CATS = {"cpu_op", "kernel", "cuda_runtime", "user_annotation", "ac2g", "Trace"}
# Infrastructure patterns to exclude (these are never GPU-related)
_INFRA_PATTERNS = (
    "threading.py", "multiprocessing/", "tqdm/", "watchdog",
    "_bootstrap", "spawn_main", "subproc_pool", "_read_thread",
    "_recv_msg", "<string>(1)", "_monitor.py",
    "<built-in method acquire of _thread.lock",
)


def is_gpu_related(ev):
    """Check if an event is GPU/compute-related (not framework infrastructure)."""
    cat = ev.get("cat", "")
    name = ev.get("name", "")
    # Keep events with GPU-related categories
    if cat in _GPU_CATS:
        return True
    # Keep aten:: operators regardless of category
    if name.startswith("aten::"):
        return True
    # Keep CUDA runtime calls
    if name.startswith("cuda") or "CUDA" in name:
        return True
    # Keep known compute patterns
    if any(k in name for k in ("kernel", "gemm", "nccl", "gloo:", "nsa", "fused_experts", "all_gather", "all_reduce", "reduce_scatter")):
        return True
    # Exclude known infrastructure
    if any(p in name for p in _INFRA_PATTERNS):
        return False
    # If no cat, it's likely python_function - keep if it looks model-related
    if not cat or cat == "python_function":
        if any(k in name for k in ("forward", "run_batch", "model_runner", "sglang/srt/layers/", "sglang/srt/models/", "torch/nn/", "Pregraph", "CompiledFxGraph")):
            return True
        return False
    return True


def filter_gpu_events(events):
    """Filter events to only GPU/compute-related ones."""
    return [ev for ev in events if is_gpu_related(ev)]


def fmt_time(us):
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f} s"
    elif us >= 1_000:
        return f"{us / 1_000:.2f} ms"
    return f"{us:.1f} us"


def fmt_diff(a, b):
    d = b - a
    pct = (d / a * 100) if a != 0 else 0
    return f"{fmt_time(d)} ({pct:+.1f}%)"


def normalize_events(events, base_ts):
    """Return new event dicts with normalized timestamps."""
    result = []
    for ev in events:
        e = dict(ev)
        if "ts" in e:
            e["ts"] = e["ts"] - base_ts
        result.append(e)
    return result


def aggregate_paths(trees):
    """Aggregate call paths across all threads.

    Returns dict: call_path -> {total_dur, count, first_ts}
    """
    agg = defaultdict(lambda: {"total_dur": 0, "count": 0, "first_ts": float("inf")})
    for key, roots in trees.items():
        for path, dur, ts, _ in collect_all_paths(roots):
            rec = agg[path]
            rec["total_dur"] += dur
            rec["count"] += 1
            rec["first_ts"] = min(rec["first_ts"], ts)
    return agg


def get_children_map(trees):
    """Map parent_path -> {child_name: count}."""
    parent_children = defaultdict(lambda: defaultdict(int))
    for key, roots in trees.items():
        for path, dur, ts, child_names in collect_all_paths(roots):
            if child_names:
                for cn in child_names:
                    parent_children[path][cn] += 1
    return parent_children


def compute_stats(events):
    total_dur = 0
    gpu_dur = 0
    cpu_dur = 0
    unique_ops = set()
    for ev in events:
        if is_complete_event(ev):
            dur = ev["dur"]
            total_dur += dur
            unique_ops.add(ev["name"])
            cat = ev.get("cat", "")
            if any(k in cat.lower() for k in ("kernel", "cuda", "gpu")):
                gpu_dur += dur
            else:
                cpu_dur += dur
    return {"total": total_dur, "gpu": gpu_dur, "cpu": cpu_dur, "unique_ops": len(unique_ops)}


def do_diff(trace_a, trace_b, max_seq_diff=100, csv_path=None, gpu_only=False,
            block_name=None, blocks_range=None):
    """Compare two traces and print analysis report."""
    events_a = get_events(trace_a)
    events_b = get_events(trace_b)

    base_a = get_first_ts(events_a)
    base_b = get_first_ts(events_b)

    click.echo("=== 时间归一化 ===")
    click.echo(f"Trace A: first event offset = {base_a:.1f} us -> normalized to 0")
    click.echo(f"Trace B: first event offset = {base_b:.1f} us -> normalized to 0")
    click.echo()

    events_a_norm = normalize_events(events_a, base_a)
    events_b_norm = normalize_events(events_b, base_b)

    if gpu_only:
        n_before_a = len(events_a_norm)
        n_before_b = len(events_b_norm)
        events_a_norm = filter_gpu_events(events_a_norm)
        events_b_norm = filter_gpu_events(events_b_norm)
        click.echo(f"GPU-only filter: A {n_before_a} -> {len(events_a_norm)}, B {n_before_b} -> {len(events_b_norm)}")
        click.echo()

    csv_rows = []

    n_complete_a = sum(1 for e in events_a_norm if is_complete_event(e))
    n_complete_b = sum(1 for e in events_b_norm if is_complete_event(e))
    click.echo(f"Trace A: {n_complete_a} complete events, Trace B: {n_complete_b} complete events")
    click.echo()

    # --- 1. GPU Kernel Diff ---
    click.echo("[1/3] Computing GPU kernel diff ...")
    csv_rows += _print_gpu_kernel_diff(events_a_norm, events_b_norm)

    # Parse blocks_range (e.g. "100:200", "100:", ":50")
    block_start = None
    block_end = None
    if blocks_range:
        parts = blocks_range.split(":")
        if len(parts) == 2:
            if parts[0].strip():
                block_start = int(parts[0].strip())
            if parts[1].strip():
                block_end = int(parts[1].strip())
        else:
            raise click.ClickException(
                f"Invalid --blocks format '{blocks_range}', expected START:END (e.g. 100:200, 100:, :50)"
            )

    # --- 2. GPU Kernel Call Sequence ---
    click.echo("[2/3] Computing GPU kernel call sequence ...")
    _print_gpu_kernel_call_sequence(events_a_norm, events_b_norm,
                                    block_name=block_name,
                                    block_start=block_start,
                                    block_end=block_end)

    # --- 3. Overall Stats ---
    _print_overall_stats(events_a_norm, events_b_norm)

    # CSV export
    if csv_path:
        fieldnames = [
            "section", "call_path", "a_avg_us", "b_avg_us",
            "diff_us", "diff_pct", "count_a", "count_b", "a_ts_us", "b_ts_us",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        click.echo(f"CSV report saved to: {csv_path}")


def _print_sequence_diff(events_a_norm, events_b_norm, max_show):
    seq_a = get_op_sequence(events_a_norm)
    seq_b = get_op_sequence(events_b_norm)

    names_a = [s[0] for s in seq_a]
    names_b = [s[0] for s in seq_b]

    click.echo(f"=== Operator Sequence Diff (main thread, {len(seq_a)} vs {len(seq_b)} ops) ===")

    # For large traces (>50k ops), use frequency-based summary instead of O(n^2) SequenceMatcher
    LARGE_THRESHOLD = 50000
    if len(names_a) > LARGE_THRESHOLD or len(names_b) > LARGE_THRESHOLD:
        click.echo(f"  (Large trace detected, using frequency-based comparison)")
        _print_sequence_diff_large(seq_a, seq_b)
        return

    matcher = SequenceMatcher(None, names_a, names_b)
    diff_lines = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                na, ts_a, dur_a = seq_a[k]
                nb, ts_b, dur_b = seq_b[k - i1 + j1]
                diff_lines.append(("SAME", na, ts_a, dur_a, ts_b, dur_b))
        elif tag == "delete":
            for k in range(i1, i2):
                diff_lines.append(("DEL", seq_a[k][0], seq_a[k][1], seq_a[k][2], None, None))
        elif tag == "insert":
            for k in range(j1, j2):
                diff_lines.append(("ADD", seq_b[k][0], None, None, seq_b[k][1], seq_b[k][2]))
        elif tag == "replace":
            for k in range(i1, i2):
                diff_lines.append(("DEL", seq_a[k][0], seq_a[k][1], seq_a[k][2], None, None))
            for k in range(j1, j2):
                diff_lines.append(("ADD", seq_b[k][0], None, None, seq_b[k][1], seq_b[k][2]))

    diff_only = [l for l in diff_lines if l[0] != "SAME"]
    if not diff_only:
        click.echo("  No sequence differences found.")
    else:
        click.echo(f"  ({len(diff_only)} differences found, showing with context)")
        context = 2
        diff_indices = set()
        for i, line in enumerate(diff_lines):
            if line[0] != "SAME":
                for j in range(max(0, i - context), min(len(diff_lines), i + context + 1)):
                    diff_indices.add(j)

        last_printed = -2
        shown = 0
        for i in sorted(diff_indices):
            if i > last_printed + 1 and last_printed >= 0:
                click.echo("  ...")
            line = diff_lines[i]
            tag, name = line[0], line[1]
            ts_a_s = f"A:{line[2]:.1f}us" if line[2] is not None else ""
            ts_b_s = f"B:{line[4]:.1f}us" if line[4] is not None else ""
            dur_a_s = f"dur={line[3]:.1f}us" if line[3] is not None else ""
            dur_b_s = f"dur={line[5]:.1f}us" if line[5] is not None else ""

            if tag == "SAME":
                click.echo(f"  [SAME]  {name:<50s} {ts_a_s:>16s} {ts_b_s:>16s}")
            elif tag == "DEL":
                click.echo(f"  [DEL ]  {name:<50s} {ts_a_s:>16s} {dur_a_s:>16s}  <- only in A")
            elif tag == "ADD":
                click.echo(f"  [ADD ]  {name:<50s} {ts_b_s:>16s} {dur_b_s:>16s}  <- only in B")

            last_printed = i
            shown += 1
            if shown >= max_show:
                remaining = len(diff_indices) - shown
                if remaining > 0:
                    click.echo(f"  ... ({remaining} more lines)")
                break
    click.echo()


def _print_sequence_diff_large(seq_a, seq_b):
    """Frequency-based sequence comparison for large traces (O(n) instead of O(n^2))."""
    from collections import Counter

    freq_a = Counter(s[0] for s in seq_a)
    freq_b = Counter(s[0] for s in seq_b)

    all_ops = sorted(set(freq_a.keys()) | set(freq_b.keys()))

    only_in_a = [(op, freq_a[op]) for op in all_ops if op not in freq_b]
    only_in_b = [(op, freq_b[op]) for op in all_ops if op not in freq_a]
    count_diff = [
        (op, freq_a[op], freq_b[op], freq_b[op] - freq_a[op])
        for op in all_ops
        if op in freq_a and op in freq_b and freq_a[op] != freq_b[op]
    ]
    count_diff.sort(key=lambda x: -abs(x[3]))

    if only_in_a:
        click.echo(f"\n  Operators only in A ({len(only_in_a)} types):")
        for op, cnt in sorted(only_in_a, key=lambda x: -x[1])[:30]:
            # Find first occurrence timestamp
            first_ts = next((s[1] for s in seq_a if s[0] == op), None)
            ts_str = f"first @ {first_ts:.1f}us" if first_ts is not None else ""
            click.echo(f"    {op:<55s} {cnt:>8d}x  {ts_str}")
        if len(only_in_a) > 30:
            click.echo(f"    ... ({len(only_in_a) - 30} more)")

    if only_in_b:
        click.echo(f"\n  Operators only in B ({len(only_in_b)} types):")
        for op, cnt in sorted(only_in_b, key=lambda x: -x[1])[:30]:
            first_ts = next((s[1] for s in seq_b if s[0] == op), None)
            ts_str = f"first @ {first_ts:.1f}us" if first_ts is not None else ""
            click.echo(f"    {op:<55s} {cnt:>8d}x  {ts_str}")
        if len(only_in_b) > 30:
            click.echo(f"    ... ({len(only_in_b) - 30} more)")

    if count_diff:
        click.echo(f"\n  Operators with different counts ({len(count_diff)} types, by |diff|):")
        hdr = f"    {'Operator':<55s} {'#A':>8s} {'#B':>8s} {'Diff':>8s}"
        click.echo(hdr)
        click.echo("    " + "-" * (len(hdr) - 4))
        for op, ca, cb, d in count_diff[:30]:
            click.echo(f"    {op:<55s} {ca:>8d} {cb:>8d} {d:>+8d}")
        if len(count_diff) > 30:
            click.echo(f"    ... ({len(count_diff) - 30} more)")

    if not only_in_a and not only_in_b and not count_diff:
        click.echo("  Operator frequencies are identical.")

    click.echo()


def _print_call_path_diff(trees_a, trees_b):
    agg_a = aggregate_paths(trees_a)
    agg_b = aggregate_paths(trees_b)
    all_paths = sorted(set(agg_a.keys()) | set(agg_b.keys()))

    click.echo("=== 调用路径对比 (按耗时差异排序) ===")

    path_diffs = []
    for path in all_paths:
        ra = agg_a.get(path)
        rb = agg_b.get(path)
        avg_a = ra["total_dur"] / ra["count"] if ra else None
        avg_b = rb["total_dur"] / rb["count"] if rb else None
        count_a = ra["count"] if ra else 0
        count_b = rb["count"] if rb else 0
        ts_a = ra["first_ts"] if ra else None
        ts_b = rb["first_ts"] if rb else None

        if avg_a is not None and avg_b is not None:
            diff_val = avg_b - avg_a
            diff_pct = (diff_val / avg_a * 100) if avg_a != 0 else 0
            sort_key = abs(diff_val)
        elif avg_a is not None:
            diff_val = diff_pct = None
            sort_key = avg_a
        else:
            diff_val = diff_pct = None
            sort_key = avg_b if avg_b else 0

        path_diffs.append((sort_key, path, avg_a, avg_b, diff_val, diff_pct, count_a, count_b, ts_a, ts_b))

    path_diffs.sort(key=lambda x: -x[0])

    hdr = f"{'Call Path':<60s} | {'A avg(us)':>10s} | {'B avg(us)':>10s} | {'Diff':>10s} | {'Diff%':>7s} | {'#A':>4s} | {'#B':>4s} | {'A ts(us)':>10s} | {'B ts(us)':>10s}"
    click.echo(hdr)
    click.echo("-" * len(hdr))

    csv_rows = []
    max_paths = 50
    for i, (_, path, avg_a, avg_b, diff_val, diff_pct, ca, cb, ts_a, ts_b) in enumerate(path_diffs):
        if i >= max_paths:
            click.echo(f"  ... ({len(path_diffs) - max_paths} more paths)")
            break

        path_str = path if len(path) <= 58 else "..." + path[-(58 - 3):]
        a_str = f"{avg_a:.1f}" if avg_a is not None else "-"
        b_str = f"{avg_b:.1f}" if avg_b is not None else "-"

        if diff_val is not None:
            d_str = f"{diff_val:+.1f}"
            p_str = f"{diff_pct:+.1f}%"
        elif avg_a is not None:
            d_str, p_str = "REMOVED", ""
        else:
            d_str, p_str = "NEW", ""

        ts_a_str = f"{ts_a:.1f}" if ts_a is not None else "-"
        ts_b_str = f"{ts_b:.1f}" if ts_b is not None else "-"

        click.echo(
            f"{path_str:<60s} | {a_str:>10s} | {b_str:>10s} | {d_str:>10s} | {p_str:>7s}"
            f" | {ca:>4d} | {cb:>4d} | {ts_a_str:>10s} | {ts_b_str:>10s}"
        )

        csv_rows.append({
            "section": "call_path_diff", "call_path": path,
            "a_avg_us": f"{avg_a:.1f}" if avg_a else "",
            "b_avg_us": f"{avg_b:.1f}" if avg_b else "",
            "diff_us": f"{diff_val:.1f}" if diff_val is not None else "",
            "diff_pct": f"{diff_pct:.1f}" if diff_pct is not None else "",
            "count_a": ca, "count_b": cb,
            "a_ts_us": f"{ts_a:.1f}" if ts_a is not None else "",
            "b_ts_us": f"{ts_b:.1f}" if ts_b is not None else "",
        })

    click.echo()
    return csv_rows


def _print_impl_diff(trees_a, trees_b):
    pc_a = get_children_map(trees_a)
    pc_b = get_children_map(trees_b)
    all_parents = sorted(set(pc_a.keys()) | set(pc_b.keys()))

    impl_diffs = [
        (p, pc_a.get(p, {}), pc_b.get(p, {}))
        for p in all_parents
        if set(pc_a.get(p, {}).keys()) != set(pc_b.get(p, {}).keys())
    ]

    click.echo("=== 实现差异 (相同父节点, 不同子节点) ===")
    csv_rows = []

    if not impl_diffs:
        click.echo("  No implementation differences found.")
    else:
        for parent, ca, cb in impl_diffs[:30]:
            parent_str = parent if len(parent) <= 80 else "..." + parent[-(80 - 3):]
            click.echo(f"  Parent: {parent_str}")
            a_kids = ", ".join(f"{n} ({c}x)" for n, c in sorted(ca.items()))
            b_kids = ", ".join(f"{n} ({c}x)" for n, c in sorted(cb.items()))
            click.echo(f"    A: {a_kids}")
            click.echo(f"    B: {b_kids}")

            only_a = set(ca.keys()) - set(cb.keys())
            only_b = set(cb.keys()) - set(ca.keys())
            if only_a:
                click.echo(f"    Removed in B: {', '.join(sorted(only_a))}")
            if only_b:
                click.echo(f"    Added in B:   {', '.join(sorted(only_b))}")
            click.echo()

            csv_rows.append({
                "section": "impl_diff", "call_path": parent,
                "a_avg_us": a_kids, "b_avg_us": b_kids,
                "diff_us": "", "diff_pct": "",
                "count_a": "", "count_b": "",
                "a_ts_us": "", "b_ts_us": "",
            })

    click.echo()
    return csv_rows


def _print_op_duration_diff(events_a, events_b):
    def agg_by_name(events):
        agg = defaultdict(lambda: {"total_dur": 0, "count": 0})
        for ev in events:
            if is_complete_event(ev):
                rec = agg[ev["name"]]
                rec["total_dur"] += ev["dur"]
                rec["count"] += 1
        return agg

    by_name_a = agg_by_name(events_a)
    by_name_b = agg_by_name(events_b)
    all_names = sorted(set(by_name_a.keys()) | set(by_name_b.keys()))

    click.echo("=== 算子耗时对比 (按总耗时差异排序) ===")

    op_diffs = []
    for name in all_names:
        ra = by_name_a.get(name)
        rb = by_name_b.get(name)
        total_a = ra["total_dur"] if ra else 0
        total_b = rb["total_dur"] if rb else 0
        count_a = ra["count"] if ra else 0
        count_b = rb["count"] if rb else 0
        avg_a = total_a / count_a if count_a else 0
        avg_b = total_b / count_b if count_b else 0
        diff_total = total_b - total_a
        op_diffs.append((abs(diff_total), name, avg_a, avg_b, total_a, total_b, diff_total, count_a, count_b))

    op_diffs.sort(key=lambda x: -x[0])

    hdr = (
        f"{'Operator':<50s} | {'A avg':>10s} | {'B avg':>10s}"
        f" | {'A total':>12s} | {'B total':>12s} | {'Diff total':>12s} | {'#A':>5s} | {'#B':>5s}"
    )
    click.echo(hdr)
    click.echo("-" * len(hdr))

    csv_rows = []
    for i, (_, name, avg_a, avg_b, tot_a, tot_b, diff_t, ca, cb) in enumerate(op_diffs):
        if i >= 50:
            click.echo(f"  ... ({len(op_diffs) - 50} more operators)")
            break
        n_str = name if len(name) <= 48 else "..." + name[-(48 - 3):]
        click.echo(
            f"{n_str:<50s} | {avg_a:>10.1f} | {avg_b:>10.1f}"
            f" | {tot_a:>12.1f} | {tot_b:>12.1f} | {diff_t:>+12.1f} | {ca:>5d} | {cb:>5d}"
        )
        csv_rows.append({
            "section": "op_duration_diff", "call_path": name,
            "a_avg_us": f"{avg_a:.1f}", "b_avg_us": f"{avg_b:.1f}",
            "diff_us": f"{diff_t:.1f}",
            "diff_pct": f"{(diff_t / tot_a * 100):.1f}" if tot_a else "",
            "count_a": ca, "count_b": cb,
            "a_ts_us": f"{tot_a:.1f}", "b_ts_us": f"{tot_b:.1f}",
        })

    click.echo()
    return csv_rows


_GPU_KERNEL_CATS = {"kernel", "gpu_user_annotation", "gpu_memcpy"}


def _gpu_kernel_stats(events):
    """Aggregate GPU kernel events by name -> {count, total_dur}."""
    stats = defaultdict(lambda: {"count": 0, "total_dur": 0.0})
    for e in events:
        if e.get("ph") == "X" and e.get("cat", "") in _GPU_KERNEL_CATS:
            s = stats[e["name"]]
            s["count"] += 1
            s["total_dur"] += e.get("dur", 0)
    return stats


def _print_gpu_kernel_diff(events_a, events_b):
    """Print GPU kernel diff summary: only-in-A, only-in-B, both sorted by |total diff|."""
    sa = _gpu_kernel_stats(events_a)
    sb = _gpu_kernel_stats(events_b)
    all_names = set(sa) | set(sb)

    if not all_names:
        click.echo("=== GPU Kernel 耗时对比 ===")
        click.echo("  No GPU kernel events found.")
        click.echo()
        return []

    only_a = sorted(
        [(n, sa[n]) for n in all_names if n not in sb],
        key=lambda x: -x[1]["total_dur"],
    )
    only_b = sorted(
        [(n, sb[n]) for n in all_names if n not in sa],
        key=lambda x: -x[1]["total_dur"],
    )
    both = []
    for n in all_names:
        if n in sa and n in sb:
            diff = sb[n]["total_dur"] - sa[n]["total_dur"]
            both.append((n, sa[n], sb[n], diff))
    both.sort(key=lambda x: -abs(x[3]))

    # Totals
    total_a = sum(s["total_dur"] for s in sa.values())
    total_b = sum(s["total_dur"] for s in sb.values())
    count_a = sum(s["count"] for s in sa.values())
    count_b = sum(s["count"] for s in sb.values())

    click.echo("=== GPU Kernel 耗时对比 ===")
    click.echo(
        f"  A: {len(sa)} unique kernels, {count_a} events, "
        f"total {fmt_time(total_a)}"
    )
    click.echo(
        f"  B: {len(sb)} unique kernels, {count_b} events, "
        f"total {fmt_time(total_b)}"
    )
    if total_a > 0:
        click.echo(
            f"  GPU kernel time diff: {fmt_diff(total_a, total_b)}"
        )
    click.echo()

    csv_rows = []

    # Only in A
    if only_a:
        click.echo(f"  --- Only in A ({len(only_a)} kernels) ---")
        click.echo(f"  {'Kernel':<80s} {'#':>6s} {'Total(us)':>12s} {'Avg(us)':>10s}")
        click.echo("  " + "-" * 112)
        for n, s in only_a[:20]:
            avg = s["total_dur"] / s["count"]
            ns = n[:80] if len(n) <= 80 else n[:77] + "..."
            click.echo(f"  {ns:<80s} {s['count']:>6d} {s['total_dur']:>12.1f} {avg:>10.1f}")
            csv_rows.append({
                "section": "gpu_kernel_only_A", "call_path": n,
                "a_avg_us": f"{avg:.1f}", "b_avg_us": "",
                "diff_us": f"{-s['total_dur']:.1f}", "diff_pct": "",
                "count_a": s["count"], "count_b": 0,
                "a_ts_us": f"{s['total_dur']:.1f}", "b_ts_us": "",
            })
        if len(only_a) > 20:
            click.echo(f"  ... ({len(only_a) - 20} more)")
        click.echo()

    # Only in B
    if only_b:
        click.echo(f"  --- Only in B ({len(only_b)} kernels) ---")
        click.echo(f"  {'Kernel':<80s} {'#':>6s} {'Total(us)':>12s} {'Avg(us)':>10s}")
        click.echo("  " + "-" * 112)
        for n, s in only_b[:20]:
            avg = s["total_dur"] / s["count"]
            ns = n[:80] if len(n) <= 80 else n[:77] + "..."
            click.echo(f"  {ns:<80s} {s['count']:>6d} {s['total_dur']:>12.1f} {avg:>10.1f}")
            csv_rows.append({
                "section": "gpu_kernel_only_B", "call_path": n,
                "a_avg_us": "", "b_avg_us": f"{avg:.1f}",
                "diff_us": f"{s['total_dur']:.1f}", "diff_pct": "",
                "count_a": 0, "count_b": s["count"],
                "a_ts_us": "", "b_ts_us": f"{s['total_dur']:.1f}",
            })
        if len(only_b) > 20:
            click.echo(f"  ... ({len(only_b) - 20} more)")
        click.echo()

    # Both - top by |total diff|
    if both:
        click.echo(f"  --- Both traces ({len(both)} kernels, by |total dur diff|) ---")
        hdr = (
            f"  {'Kernel':<70s} {'#A':>5s} {'#B':>5s}"
            f" {'A avg':>10s} {'B avg':>10s}"
            f" {'A total':>12s} {'B total':>12s}"
            f" {'Diff total':>12s} {'Diff%':>8s}"
        )
        click.echo(hdr)
        click.echo("  " + "-" * (len(hdr) - 2))
        for i, (n, a, b, d) in enumerate(both):
            if i >= 30:
                click.echo(f"  ... ({len(both) - 30} more)")
                break
            a_avg = a["total_dur"] / a["count"]
            b_avg = b["total_dur"] / b["count"]
            pct = d / a["total_dur"] * 100 if a["total_dur"] else (
                100.0 if d > 0 else -100.0
            )
            ns = n[:70] if len(n) <= 70 else n[:67] + "..."
            click.echo(
                f"  {ns:<70s} {a['count']:>5d} {b['count']:>5d}"
                f" {a_avg:>10.1f} {b_avg:>10.1f}"
                f" {a['total_dur']:>12.1f} {b['total_dur']:>12.1f}"
                f" {d:>+12.1f} {pct:>+7.1f}%"
            )
            csv_rows.append({
                "section": "gpu_kernel_both", "call_path": n,
                "a_avg_us": f"{a_avg:.1f}", "b_avg_us": f"{b_avg:.1f}",
                "diff_us": f"{d:.1f}", "diff_pct": f"{pct:.1f}",
                "count_a": a["count"], "count_b": b["count"],
                "a_ts_us": f"{a['total_dur']:.1f}",
                "b_ts_us": f"{b['total_dur']:.1f}",
            })
        click.echo()

    # Kernel replacement detection: similar count, only-in-A paired with only-in-B
    if only_a and only_b:
        replacements = []
        used_b = set()
        for na, sa_s in only_a:
            for j, (nb, sb_s) in enumerate(only_b):
                if j in used_b:
                    continue
                if sa_s["count"] == sb_s["count"]:
                    replacements.append((na, nb, sa_s, sb_s))
                    used_b.add(j)
                    break
        if replacements:
            click.echo("  --- Possible kernel replacements (same count, A->B) ---")
            for na, nb, sa_s, sb_s in replacements:
                a_avg = sa_s["total_dur"] / sa_s["count"]
                b_avg = sb_s["total_dur"] / sb_s["count"]
                diff = sb_s["total_dur"] - sa_s["total_dur"]
                pct = diff / sa_s["total_dur"] * 100 if sa_s["total_dur"] else 0
                na_s = na[:60] if len(na) <= 60 else na[:57] + "..."
                nb_s = nb[:60] if len(nb) <= 60 else nb[:57] + "..."
                click.echo(f"    A: {na_s}")
                click.echo(f"    B: {nb_s}")
                click.echo(
                    f"       {sa_s['count']}x, "
                    f"A avg {a_avg:.1f}us -> B avg {b_avg:.1f}us, "
                    f"total diff {diff:+.1f}us ({pct:+.1f}%)"
                )
                click.echo()

    click.echo()
    return csv_rows


def _collect_gpu_kernels(events, min_dur=0):
    """Collect GPU kernel events sorted by timestamp, optionally filtered by min duration."""
    kernels = []
    for e in events:
        if e.get("ph") == "X" and e.get("cat", "") in _GPU_KERNEL_CATS:
            dur = e.get("dur", 0)
            if dur >= min_dur:
                kernels.append((e["ts"], e["name"], dur))
    kernels.sort()
    return kernels


def _compress_sequence(seq):
    """Run-length encode consecutive same-name kernels.

    Returns list of (name, count, avg_dur).
    """
    if not seq:
        return []
    compressed = []
    cur_name = seq[0][0]
    cur_durs = [seq[0][1]]
    for name, dur in seq[1:]:
        if name == cur_name:
            cur_durs.append(dur)
        else:
            compressed.append((cur_name, len(cur_durs), sum(cur_durs) / len(cur_durs)))
            cur_name = name
            cur_durs = [dur]
    compressed.append((cur_name, len(cur_durs), sum(cur_durs) / len(cur_durs)))
    return compressed


def _detect_repeating_pattern(names, min_period=2):
    """Detect the minimum repeating pattern in a name sequence.

    Generic algorithm — works on any sequence of string names (CPU events,
    GPU kernels, etc.) with no hardcoded name assumptions.

    Returns (n_repeats, start_idx, period) or (0, 0, 0) if not found.
    """
    from collections import Counter

    n = len(names)
    if n < 6:
        return 0, 0, 0

    counts = Counter(names)
    best = (0, 0, 0)  # (n_repeats, start, period)

    for marker_name, marker_count in counts.most_common(20):
        if marker_count < 3:
            break
        positions = [i for i, nm in enumerate(names) if nm == marker_name]
        if len(positions) < 3:
            continue

        # Most common gap between consecutive markers = likely period
        gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        gap_counts = Counter(gaps)
        period, gap_freq = gap_counts.most_common(1)[0]
        if period < min_period or gap_freq < marker_count // 2:
            continue

        # Try multiple start positions (first block often differs)
        for si in range(min(5, len(positions))):
            start = positions[si]
            if start + period > n:
                break
            block_names = names[start:start + period]

            n_repeats = 0
            for r in range((n - start) // period):
                chunk = names[start + r * period:start + (r + 1) * period]
                if chunk == block_names:
                    n_repeats += 1
                else:
                    break

            if n_repeats > best[0]:
                best = (n_repeats, start, period)

    return best


def _is_low_level_cpu(name):
    """Check if a CPU event name is a low-level torch/framework internal.

    These are too granular for block detection — they appear in every layer
    and drown out the structural pattern.
    """
    return (
        name.startswith(("torch/", "aten::", "c10d::", "c10::", "<built-in"))
        or "/torch/" in name
        or name.startswith(("triton/", "deep_gemm/"))
        or name.startswith("record_param_comms")
    )


def _find_cpu_blocks(events, min_cpu_dur=50, block_name=None):
    """Find repeating CPU event blocks and their time ranges.

    Generic — no hardcoded event names. Tries pattern detection on CPU events
    grouped by category, then falls back to duration-threshold filtering.
    Filters out low-level torch/framework internals for cleaner detection.

    If block_name is given (substring), only CPU events whose name contains
    that substring are used for block detection.

    Returns dict with:
        blocks: list of (start_ts, end_ts) per repeating block
        preamble: (start_ts, end_ts) before first block, or None
        epilogue: (start_ts, end_ts) after last block, or None
        n_repeats: number of repeating blocks
        period: number of CPU events per block
    Returns None if no repeating pattern found.
    """
    # Get all non-GPU complete events, excluding low-level torch internals
    all_cpu = [
        e for e in events
        if e.get("ph") == "X"
        and e.get("cat", "") not in _GPU_KERNEL_CATS
        and e.get("dur", 0) > min_cpu_dur
        and not _is_low_level_cpu(e.get("name", ""))
    ]

    def _try_detect(cpu_events):
        """Try pattern detection on a sorted list of CPU events.
        Returns (n_repeats, result_dict) or (0, None).
        """
        if len(cpu_events) < 6:
            return 0, None
        names = [e["name"] for e in cpu_events]
        n_repeats, start_idx, period = _detect_repeating_pattern(names)
        if n_repeats < 2:
            return 0, None
        return n_repeats, _build_blocks_result(cpu_events, n_repeats, start_idx, period)

    # If user specified block_name, use only matching events
    if block_name:
        matched = sorted(
            [e for e in events
             if e.get("ph") == "X"
             and block_name in e.get("name", "")
             and e.get("dur", 0) > min_cpu_dur],
            key=lambda e: e["ts"],
        )
        if len(matched) < 2:
            return None

        # Try to detect repeating pattern among the matched events
        names = [e["name"] for e in matched]
        n_repeats, start_idx, period = _detect_repeating_pattern(names, min_period=1)

        if n_repeats >= 2 and period >= 1:
            # Build blocks: each block = 'period' consecutive matched events
            # Block time range: from first event's ts to next block's ts (or last event end)
            blocks = []
            for r in range(n_repeats):
                bi = start_idx + r * period
                next_bi = start_idx + (r + 1) * period
                block_start = matched[bi]["ts"]
                block_end = (matched[next_bi]["ts"] if next_bi < len(matched)
                             else matched[bi + period - 1]["ts"] + matched[bi + period - 1].get("dur", 0))
                blocks.append((block_start, block_end))

            first_ts = min(e["ts"] for e in events if "ts" in e)
            last_ts = max(e["ts"] + e.get("dur", 0) for e in events
                          if e.get("ph") == "X" and "ts" in e)
            preamble = (first_ts, blocks[0][0]) if blocks[0][0] > first_ts else None
            epilogue = (blocks[-1][1], last_ts) if blocks[-1][1] < last_ts else None

            return {
                "blocks": blocks,
                "preamble": preamble,
                "epilogue": epilogue,
                "n_repeats": n_repeats,
                "period": period,
            }

        # Fallback: each matched event = one block boundary
        blocks = []
        for i in range(len(matched)):
            bs = matched[i]["ts"]
            be = (matched[i + 1]["ts"] if i + 1 < len(matched)
                  else matched[i]["ts"] + matched[i].get("dur", 0))
            blocks.append((bs, be))

        first_ts = min(e["ts"] for e in events if "ts" in e)
        last_ts = max(e["ts"] + e.get("dur", 0) for e in events
                      if e.get("ph") == "X" and "ts" in e)
        preamble = (first_ts, blocks[0][0]) if blocks[0][0] > first_ts else None
        epilogue = (blocks[-1][1], last_ts) if blocks[-1][1] < last_ts else None

        return {
            "blocks": blocks,
            "preamble": preamble,
            "epilogue": epilogue,
            "n_repeats": len(blocks),
            "period": 1,
        }

    best_repeats = 0
    best_result = None

    # Strategy 1: try each category separately
    by_cat = defaultdict(list)
    for e in all_cpu:
        by_cat[e.get("cat", "")].append(e)

    for cat in by_cat:
        subset = sorted(by_cat[cat], key=lambda e: e["ts"])
        n_reps, result = _try_detect(subset)
        # Prefer the pattern with the MOST repeats (= minimum repeating unit)
        if n_reps > best_repeats:
            best_repeats = n_reps
            best_result = result

    if best_result:
        return best_result

    # Strategy 2: try with increasing duration thresholds on all events
    for min_d in [1000, 5000, 10000]:
        subset = sorted(
            [e for e in all_cpu if e.get("dur", 0) > min_d],
            key=lambda e: e["ts"],
        )
        n_reps, result = _try_detect(subset)
        if n_reps > best_repeats:
            best_repeats = n_reps
            best_result = result

    return best_result


def _build_blocks_result(cpu_events, n_repeats, start_idx, period):
    """Build the blocks result dict from pattern detection output."""
    blocks = []
    for r in range(n_repeats):
        bi = start_idx + r * period
        ei = start_idx + (r + 1) * period
        block_start = cpu_events[bi]["ts"]
        block_end = (cpu_events[ei]["ts"] if ei < len(cpu_events)
                     else cpu_events[-1]["ts"] + cpu_events[-1].get("dur", 0))
        blocks.append((block_start, block_end))

    first_ts = cpu_events[0]["ts"]
    last_ts = cpu_events[-1]["ts"] + cpu_events[-1].get("dur", 0)
    preamble = (first_ts, blocks[0][0]) if start_idx > 0 else None
    epi_start_idx = start_idx + n_repeats * period
    epilogue = (blocks[-1][1], last_ts) if epi_start_idx < len(cpu_events) else None

    return {
        "blocks": blocks,
        "preamble": preamble,
        "epilogue": epilogue,
        "n_repeats": n_repeats,
        "period": period,
    }


def _build_ac2g_map(events):
    """Build ac2g (async CPU to GPU) mapping: (gpu_pid, gpu_tid, gpu_ts) -> cpu_launch_ts."""
    flow_s = {}
    flow_f = {}
    for e in events:
        if e.get("cat") == "ac2g":
            fid = e.get("id")
            if e.get("ph") == "s":
                flow_s[fid] = e
            elif e.get("ph") == "f":
                flow_f[fid] = e

    gpu_to_cpu_ts = {}
    for fid in set(flow_s) & set(flow_f):
        s_ev = flow_s[fid]
        f_ev = flow_f[fid]
        gpu_key = (f_ev.get("pid"), f_ev.get("tid"), f_ev["ts"])
        gpu_to_cpu_ts[gpu_key] = s_ev["ts"]

    return gpu_to_cpu_ts


def _assign_gpu_to_blocks(events, blocks_info, min_dur=0):
    """Assign GPU kernels to CPU-detected blocks using ac2g flow events.

    Falls back to ordering-based assignment if ac2g coverage is low.

    Returns (preamble_kernels, block_kernels_list, epilogue_kernels)
    where each is a list of (name, dur) tuples.
    """
    import bisect

    blocks = blocks_info["blocks"]
    preamble_range = blocks_info["preamble"]
    epilogue_range = blocks_info["epilogue"]

    # Collect GPU kernel events (full objects for ac2g lookup)
    gpu_events = sorted(
        [e for e in events
         if e.get("ph") == "X" and e.get("cat", "") in _GPU_KERNEL_CATS
         and e.get("dur", 0) >= min_dur],
        key=lambda e: e["ts"],
    )

    if not gpu_events:
        return [], [[] for _ in blocks], []

    # Try ac2g mapping
    ac2g_map = _build_ac2g_map(events)
    mapped_count = 0
    for gk in gpu_events:
        key = (gk.get("pid"), gk.get("tid"), gk["ts"])
        if key in ac2g_map:
            gk["_cpu_ts"] = ac2g_map[key]
            mapped_count += 1
        else:
            gk["_cpu_ts"] = None

    use_ac2g = mapped_count > len(gpu_events) * 0.5

    # Block CPU time boundaries for bisect
    block_starts = [bs for bs, _ in blocks]

    preamble = []
    block_kernels = [[] for _ in blocks]
    epilogue = []

    for gk in gpu_events:
        name = gk["name"]
        dur = gk.get("dur", 0)

        if use_ac2g and gk["_cpu_ts"] is not None:
            # Use CPU launch time to find block
            cpu_ts = gk["_cpu_ts"]
            idx = bisect.bisect_right(block_starts, cpu_ts) - 1
            if 0 <= idx < len(blocks):
                bs, be = blocks[idx]
                if bs <= cpu_ts < be:
                    block_kernels[idx].append((name, dur))
                    continue
            # Outside blocks — preamble or epilogue
            if preamble_range and cpu_ts < blocks[0][0]:
                preamble.append((name, dur))
            elif epilogue_range and cpu_ts >= blocks[-1][1]:
                epilogue.append((name, dur))
        else:
            # Fallback: use GPU timestamp vs block time ranges
            ts = gk["ts"]
            idx = bisect.bisect_right(block_starts, ts) - 1
            if 0 <= idx < len(blocks):
                bs, be = blocks[idx]
                if bs <= ts < be:
                    block_kernels[idx].append((name, dur))
                    continue
            if preamble_range and ts < blocks[0][0]:
                preamble.append((name, dur))
            elif epilogue_range and ts >= blocks[-1][1]:
                epilogue.append((name, dur))

    return preamble, block_kernels, epilogue


def _build_cpu_stack_for_block(events, block_range):
    """Build CPU call stack for GPU kernels within a block's time range.

    Uses ac2g to find the CPU launch point of each GPU kernel, then walks up
    the call tree to build the full CPU call path.

    Returns list of (gpu_name, gpu_dur, cpu_path) sorted by GPU timestamp.
    cpu_path is a list of (name, dur) from outermost to innermost CPU frame.
    """
    import bisect

    bs, be = block_range

    # Build ac2g: gpu_key -> cpu event info
    ac2g_map = _build_ac2g_map(events)
    # Reverse: cpu_ts -> set of (gpu_pid, gpu_tid, gpu_ts)
    cpu_ts_to_gpu = defaultdict(list)
    for gpu_key, cpu_ts in ac2g_map.items():
        cpu_ts_to_gpu[cpu_ts].append(gpu_key)

    # Collect CPU events on the main CPU thread (the one with cpu_op)
    cpu_events = sorted(
        [e for e in events
         if e.get("ph") == "X" and e.get("cat", "") not in _GPU_KERNEL_CATS],
        key=lambda e: (e["ts"], -e.get("dur", 0)),
    )

    # Build call tree for the block's time range (only CPU events overlapping the block)
    block_cpu = [e for e in cpu_events if e["ts"] < be and e["ts"] + e.get("dur", 0) > bs]
    trees = build_call_trees(block_cpu)

    # Build a lookup: cpu_ts -> call path (list of ancestor names from root)
    # For each leaf-ish node, if its ts matches a cpu launch point, record path
    cpu_ts_path = {}

    def _walk_tree(node, path):
        cur_path = path + [(node.name, node.dur)]
        cpu_ts_path[node.ts] = cur_path
        for child in node.children:
            _walk_tree(child, cur_path)

    for key, roots in trees.items():
        for root in roots:
            _walk_tree(root, [])

    # Collect GPU kernels in this block
    gpu_events_in_block = sorted(
        [e for e in events
         if e.get("ph") == "X" and e.get("cat", "") in _GPU_KERNEL_CATS
         and e.get("dur", 0) > 0],
        key=lambda e: e["ts"],
    )

    results = []
    for gk in gpu_events_in_block:
        gpu_key = (gk.get("pid"), gk.get("tid"), gk["ts"])
        cpu_ts = ac2g_map.get(gpu_key)
        if cpu_ts is None or not (bs <= cpu_ts < be):
            # Use GPU ts fallback for block membership
            if not (bs <= gk["ts"] < be):
                continue
            results.append((gk["name"], gk.get("dur", 0), []))
            continue

        # Find the best matching CPU path: the deepest node whose ts <= cpu_ts
        # Walk backward from cpu_ts to find containing node
        best_path = []
        for ts_key in sorted(cpu_ts_path.keys()):
            if ts_key <= cpu_ts:
                candidate = cpu_ts_path[ts_key]
                # Check containment: last frame must contain cpu_ts
                last_name, last_dur = candidate[-1]
                if ts_key + last_dur >= cpu_ts:
                    if len(candidate) > len(best_path):
                        best_path = candidate
            elif ts_key > cpu_ts:
                break

        results.append((gk["name"], gk.get("dur", 0), best_path))

    return results


def _print_compressed(items, indent, max_lines=60):
    """Print a compressed kernel list."""
    compressed = _compress_sequence(items)
    shown = 0
    for name, count, avg_dur in compressed:
        if shown >= max_lines:
            click.echo(f"{indent}... ({len(compressed) - shown} more)")
            break
        ns = name[:120] if len(name) <= 120 else name[:117] + "..."
        if count > 1:
            click.echo(f"{indent}{ns}  x{count:<4d} avg {avg_dur:>10.1f} us")
        else:
            click.echo(f"{indent}{ns}  {avg_dur:>10.1f} us")
        shown += 1


def _group_blocks(block_kernels):
    """Group blocks by kernel name signature.

    Returns dict: signature_tuple -> list of block indices.
    """
    groups = defaultdict(list)
    for i, bk in enumerate(block_kernels):
        sig = tuple(name for name, _ in bk)
        groups[sig].append(i)
    return groups


def _find_joint_representative(bk_a, bk_b):
    """Find a block index that is representative (most common pattern) in BOTH traces.

    Picks the same layer index so A and B are comparing the same layer.
    Falls back to A's best if no joint index found.

    Returns (rep_idx, groups_a, groups_b).
    """
    groups_a = _group_blocks(bk_a)
    groups_b = _group_blocks(bk_b)

    # Find the most common signature for each trace
    best_sig_a = max(groups_a, key=lambda s: len(groups_a[s]))
    best_sig_b = max(groups_b, key=lambda s: len(groups_b[s]))

    set_a = set(groups_a[best_sig_a])
    set_b = set(groups_b[best_sig_b])

    # Prefer an index that is in BOTH best groups
    joint = set_a & set_b
    if joint:
        return min(joint), groups_a, groups_b

    # Fallback: index in A's best that is also in any B group with >= half max size
    half_b = len(groups_b[best_sig_b]) // 2
    big_b = set()
    for sig, idxs in groups_b.items():
        if len(idxs) >= max(half_b, 2):
            big_b.update(idxs)
    joint2 = set_a & big_b
    if joint2:
        return min(joint2), groups_a, groups_b

    # Last resort: just use A's first representative
    return min(set_a), groups_a, groups_b


def _build_cpu_lookup(cpu_stack):
    """Map gpu_name -> simplified CPU call path string."""
    by_gpu = defaultdict(list)
    for gpu_name, gpu_dur, path in cpu_stack:
        if path:
            filtered = [
                name for name, _ in path
                if not _is_low_level_cpu(name)
                and name not in ("ProfilerStep*", "")
            ]
            if filtered:
                by_gpu[gpu_name].append(" > ".join(filtered[-3:]))
    from collections import Counter
    result = {}
    for gpu_name, paths in by_gpu.items():
        mc = Counter(paths).most_common(1)
        if mc:
            result[gpu_name] = mc[0][0]
    return result


# ANSI color helpers
_RED = "\033[31m"
_RESET = "\033[0m"


def _fmt_kernel(name, count, dur, width=120):
    ns = name[:width] if len(name) <= width else name[:width - 3] + "..."
    if count > 1:
        return f"{ns}  x{count:<3d} avg {dur:>9.1f} us"
    return f"{ns}  {dur:>9.1f} us"


def _fmt_diff(dur_a, dur_b):
    if dur_a > 0 and dur_b > 0:
        pct = (dur_b - dur_a) / dur_a * 100
        if abs(pct) < 1:
            return ""
        return f"{'↑' if pct > 0 else '↓'}{abs(pct):.0f}%"
    return ""


def _print_block_ab_comparison(block_a, block_b, cpu_stack_a, cpu_stack_b, indent="    "):
    """Print A vs B comparison for one block with CPU call stack.

    block_a/b: list of (name, dur)
    cpu_stack_a/b: list of (gpu_name, gpu_dur, cpu_path) from _build_cpu_stack_for_block
    Differences are highlighted in red (ONLY A/B, different CPU stacks).
    """
    comp_a = _compress_sequence(block_a)
    comp_b = _compress_sequence(block_b)

    cpu_lookup_a = _build_cpu_lookup(cpu_stack_a) if cpu_stack_a else {}
    cpu_lookup_b = _build_cpu_lookup(cpu_stack_b) if cpu_stack_b else {}

    names_a = [n for n, _, _ in comp_a]
    names_b = [n for n, _, _ in comp_b]
    sm = SequenceMatcher(None, names_a, names_b)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for ai, bi in zip(range(i1, i2), range(j1, j2)):
                na, ca, da = comp_a[ai]
                nb, cb, db = comp_b[bi]
                diff = _fmt_diff(da, db)
                cpu_a = cpu_lookup_a.get(na, "")
                cpu_b = cpu_lookup_b.get(nb, "")

                click.echo(f"{indent}  A: {_fmt_kernel(na, ca, da)}")
                if diff or db != da:
                    click.echo(f"{indent}  B: {_fmt_kernel(nb, cb, db)}  {diff}")
                else:
                    click.echo(f"{indent}  B: (same)")
                if cpu_a and cpu_b and cpu_a != cpu_b:
                    click.echo(f"{indent}     {_RED}cpu A: {cpu_a}{_RESET}")
                    click.echo(f"{indent}     {_RED}cpu B: {cpu_b}{_RESET}")
                elif cpu_a:
                    click.echo(f"{indent}     cpu: {cpu_a}")
                click.echo()

        elif tag == "replace":
            for ai in range(i1, i2):
                na, ca, da = comp_a[ai]
                cpu_a = cpu_lookup_a.get(na, "")
                click.echo(f"{indent}  {_RED}A: {_fmt_kernel(na, ca, da)}  [ONLY A]{_RESET}")
                if cpu_a:
                    click.echo(f"{indent}     {_RED}cpu: {cpu_a}{_RESET}")
            for bi in range(j1, j2):
                nb, cb, db = comp_b[bi]
                cpu_b = cpu_lookup_b.get(nb, "")
                click.echo(f"{indent}  {_RED}B: {_fmt_kernel(nb, cb, db)}  [ONLY B]{_RESET}")
                if cpu_b:
                    click.echo(f"{indent}     {_RED}cpu: {cpu_b}{_RESET}")
            click.echo()

        elif tag == "delete":
            for ai in range(i1, i2):
                na, ca, da = comp_a[ai]
                cpu_a = cpu_lookup_a.get(na, "")
                click.echo(f"{indent}  {_RED}A: {_fmt_kernel(na, ca, da)}  [ONLY A]{_RESET}")
                if cpu_a:
                    click.echo(f"{indent}     {_RED}cpu: {cpu_a}{_RESET}")
            click.echo()

        elif tag == "insert":
            for bi in range(j1, j2):
                nb, cb, db = comp_b[bi]
                cpu_b = cpu_lookup_b.get(nb, "")
                click.echo(f"{indent}  {_RED}B: {_fmt_kernel(nb, cb, db)}  [ONLY B]{_RESET}")
                if cpu_b:
                    click.echo(f"{indent}     {_RED}cpu: {cpu_b}{_RESET}")
            click.echo()


def _print_outlier_diff(outlier_block, rep_block, indent="      "):
    """Print kernel-level diff between an outlier block and the representative.

    Only shows the differing kernels (added/removed), not the full block.
    """
    comp_out = _compress_sequence(outlier_block)
    comp_rep = _compress_sequence(rep_block)
    names_out = [n for n, _, _ in comp_out]
    names_rep = [n for n, _, _ in comp_rep]
    sm = SequenceMatcher(None, names_rep, names_out)

    diffs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            for ai in range(i1, i2):
                n, c, d = comp_rep[ai]
                diffs.append(f"{indent}  {_RED}- {_fmt_kernel(n, c, d)}{_RESET}")
            for bi in range(j1, j2):
                n, c, d = comp_out[bi]
                diffs.append(f"{indent}  {_RED}+ {_fmt_kernel(n, c, d)}{_RESET}")
        elif tag == "delete":
            for ai in range(i1, i2):
                n, c, d = comp_rep[ai]
                diffs.append(f"{indent}  {_RED}- {_fmt_kernel(n, c, d)}{_RESET}")
        elif tag == "insert":
            for bi in range(j1, j2):
                n, c, d = comp_out[bi]
                diffs.append(f"{indent}  {_RED}+ {_fmt_kernel(n, c, d)}{_RESET}")

    return diffs


def _block_gpu_total_dur(block_kernels):
    """Sum GPU kernel durations for a block."""
    return sum(dur for _, dur in block_kernels)


def _print_gpu_kernel_call_sequence(events_a, events_b, block_name=None,
                                     block_start=None, block_end=None):
    """Show GPU kernel call sequence: one representative block with A vs B comparison.

    Detects repeating CPU blocks, picks the most common block pattern,
    shows A vs B kernel-by-kernel comparison with CPU call stack.
    Other blocks that differ are summarized.

    Default: explore up to MAX_EXPLORE_BLOCKS, detail-analyze first
    MAX_ANALYZE_BLOCKS. Remaining blocks are compared by total GPU duration
    against the analyzed average.

    block_start/block_end: 0-based block range (from --blocks).
    """
    MAX_EXPLORE_BLOCKS = 100
    MAX_ANALYZE_BLOCKS = 10
    MIN_DUR_US = 10.0
    DUR_DIFF_THRESHOLD = 0.20  # 20%
    click.echo(f"=== GPU Kernel 调用序列 (kernels > {MIN_DUR_US:.0f}us) ===")
    if block_name:
        click.echo(f"  (按 *{block_name}* 划分重复块)")

    # Detect blocks for both A and B
    blocks_a = _find_cpu_blocks(events_a, block_name=block_name)
    blocks_b = _find_cpu_blocks(events_b, block_name=block_name)

    if blocks_a is None and blocks_b is None:
        click.echo("\n  两个 trace 均未检测到重复 CPU 模式，回退到平铺列表。\n")
        for label, events in [("A", events_a), ("B", events_b)]:
            sig_gpu = _collect_gpu_kernels(events, min_dur=MIN_DUR_US)
            if sig_gpu:
                click.echo(f"  {label}: {len(sig_gpu)} significant kernels")
                _print_compressed([(n, d) for _, n, d in sig_gpu], "    ", max_lines=40)
        click.echo()
        return

    # Get full block assignments
    preamble_a_full, bk_a_full, epilogue_a_full = (
        _assign_gpu_to_blocks(events_a, blocks_a, min_dur=MIN_DUR_US)
        if blocks_a else ([], [], [])
    )
    preamble_b_full, bk_b_full, epilogue_b_full = (
        _assign_gpu_to_blocks(events_b, blocks_b, min_dur=MIN_DUR_US)
        if blocks_b else ([], [], [])
    )

    n_a_full = blocks_a["n_repeats"] if blocks_a else 0
    n_b_full = blocks_b["n_repeats"] if blocks_b else 0
    period_a = blocks_a["period"] if blocks_a else 0
    period_b = blocks_b["period"] if blocks_b else 0

    click.echo(
        f"\n  A: {n_a_full} 个重复块 (周期={period_a}), "
        f"B: {n_b_full} 个重复块 (周期={period_b})"
    )

    # Apply --blocks range (0-based, end exclusive)
    total_blocks = max(n_a_full, n_b_full)
    range_lo = block_start if block_start is not None else 0
    range_hi = block_end if block_end is not None else total_blocks
    range_lo = max(0, min(range_lo, total_blocks))
    range_hi = max(range_lo, min(range_hi, total_blocks))

    if block_start is not None or block_end is not None:
        click.echo(
            f"  指定区间: 块 #{range_lo} ~ #{range_hi} "
            f"(共 {range_hi - range_lo} 个块)"
        )

    # Cap explore range
    explore_hi = min(range_hi, range_lo + MAX_EXPLORE_BLOCKS)
    if explore_hi < range_hi:
        click.echo(
            f"  探索上限: 前 {MAX_EXPLORE_BLOCKS} 个块 "
            f"(#{range_lo} ~ #{explore_hi}), "
            f"剩余 {range_hi - explore_hi} 个块跳过"
        )

    # Slice to explore range
    bk_a_explore = bk_a_full[range_lo:explore_hi] if bk_a_full else []
    bk_b_explore = bk_b_full[range_lo:explore_hi] if bk_b_full else []
    blocks_a_ranges = blocks_a["blocks"][range_lo:explore_hi] if blocks_a else []
    blocks_b_ranges = blocks_b["blocks"][range_lo:explore_hi] if blocks_b else []

    # Detail-analyze first MAX_ANALYZE_BLOCKS
    analyze_n = min(MAX_ANALYZE_BLOCKS, max(len(bk_a_explore), len(bk_b_explore)))
    bk_a = bk_a_explore[:analyze_n]
    bk_b = bk_b_explore[:analyze_n]
    n_a = len(bk_a)
    n_b = len(bk_b)

    n_explore = max(len(bk_a_explore), len(bk_b_explore))
    if n_explore > analyze_n:
        click.echo(
            f"  详细分析前 {analyze_n} 个块, "
            f"剩余 {n_explore - analyze_n} 个块仅做耗时对比"
        )

    # Use preamble/epilogue only when not using custom range
    preamble_a = preamble_a_full if block_start is None else []
    preamble_b = preamble_b_full if block_start is None else []
    epilogue_a = epilogue_a_full if block_end is None else []
    epilogue_b = epilogue_b_full if block_end is None else []

    # --- GPU block duration analysis ---
    # 1) Compare A vs B for each of the first analyze_n blocks
    def _block_dur(block_ranges, idx):
        if idx < len(block_ranges):
            return block_ranges[idx][1] - block_ranges[idx][0]
        return 0

    ab_diff_blocks = []
    n_common_analyze = min(n_a, n_b)
    for i in range(n_common_analyze):
        dur_a = _block_dur(blocks_a_ranges, i)
        dur_b = _block_dur(blocks_b_ranges, i)
        if dur_a > 0 and dur_b > 0:
            pct = (dur_a - dur_b) / dur_b * 100
            if abs(pct) >= DUR_DIFF_THRESHOLD * 100:
                ab_diff_blocks.append((range_lo + i, dur_a, dur_b, pct))

    if ab_diff_blocks:
        click.echo(
            f"\n  {_RED}[前 {analyze_n} 块 A/B 耗时差异 >= {DUR_DIFF_THRESHOLD:.0%}]{_RESET}"
        )
        for block_num, dur_a, dur_b, pct in ab_diff_blocks[:10]:
            tag = "A 慢" if pct > 0 else "B 慢"
            click.echo(
                f"    {_RED}块 #{block_num}: "
                f"A={dur_a / 1000:.1f}ms  B={dur_b / 1000:.1f}ms  "
                f"({pct:+.1f}% {tag}){_RESET}"
            )
        if len(ab_diff_blocks) > 10:
            click.echo(f"    ... 共 {len(ab_diff_blocks)} 个块有显著差异")

    # 2) Compute average GPU duration of first analyze_n blocks per trace,
    #    then flag remaining explore blocks that deviate >20%
    def _avg_gpu_dur(bk_list):
        durs = [_block_gpu_total_dur(b) for b in bk_list if b]
        return sum(durs) / len(durs) if durs else 0

    avg_a = _avg_gpu_dur(bk_a)
    avg_b = _avg_gpu_dur(bk_b)

    if n_explore > analyze_n:
        tail_outliers_a = []
        tail_outliers_b = []
        for i in range(analyze_n, n_explore):
            if i < len(bk_a_explore) and bk_a_explore[i] and avg_a > 0:
                d = _block_gpu_total_dur(bk_a_explore[i])
                pct = (d - avg_a) / avg_a * 100
                if abs(pct) >= DUR_DIFF_THRESHOLD * 100:
                    tail_outliers_a.append((range_lo + i, d, pct))
            if i < len(bk_b_explore) and bk_b_explore[i] and avg_b > 0:
                d = _block_gpu_total_dur(bk_b_explore[i])
                pct = (d - avg_b) / avg_b * 100
                if abs(pct) >= DUR_DIFF_THRESHOLD * 100:
                    tail_outliers_b.append((range_lo + i, d, pct))

        if tail_outliers_a or tail_outliers_b:
            click.echo(
                f"\n  {_RED}[后续块 GPU 耗时偏离前 {analyze_n} 块均值 >= {DUR_DIFF_THRESHOLD:.0%}]{_RESET}"
            )
            for label, outliers, avg in [("A", tail_outliers_a, avg_a),
                                          ("B", tail_outliers_b, avg_b)]:
                if not outliers:
                    continue
                click.echo(
                    f"    {label} 均值={avg / 1000:.1f}ms, "
                    f"{len(outliers)} 个块偏离:"
                )
                for block_num, d, pct in outliers[:8]:
                    click.echo(
                        f"      {_RED}块 #{block_num}: "
                        f"{d / 1000:.1f}ms ({pct:+.1f}%){_RESET}"
                    )
                if len(outliers) > 8:
                    click.echo(f"      ... 共 {len(outliers)} 个")

    # Find representative block — same index for A and B (comparing same layer)
    rep_idx = 0
    groups_a = _group_blocks(bk_a) if bk_a else {}
    groups_b = _group_blocks(bk_b) if bk_b else {}

    if bk_a and bk_b:
        rep_idx, groups_a, groups_b = _find_joint_representative(bk_a, bk_b)
    elif bk_a:
        best_sig = max(groups_a, key=lambda s: len(groups_a[s]))
        rep_idx = groups_a[best_sig][0]
    elif bk_b:
        best_sig = max(groups_b, key=lambda s: len(groups_b[s]))
        rep_idx = groups_b[best_sig][0]

    rep_a = bk_a[rep_idx] if bk_a and rep_idx < len(bk_a) else []
    rep_b = bk_b[rep_idx] if bk_b and rep_idx < len(bk_b) else []

    rep_sig_a = tuple(name for name, _ in rep_a)
    rep_sig_b = tuple(name for name, _ in rep_b)
    match_a = len(groups_a.get(rep_sig_a, [])) if groups_a else 0
    match_b = len(groups_b.get(rep_sig_b, [])) if groups_b else 0

    # Display block number in full-range terms
    rep_full_idx = range_lo + rep_idx
    click.echo(
        f"\n  代表块: #{rep_full_idx} "
        f"(A: {match_a}/{n_a} 相同, B: {match_b}/{n_b} 相同)"
    )

    # Build CPU call stacks for the representative blocks
    click.echo("  构建 CPU 调用栈 ...")
    cpu_stack_a = (
        _build_cpu_stack_for_block(events_a, blocks_a_ranges[rep_idx])
        if blocks_a_ranges and rep_idx < len(blocks_a_ranges) else []
    )
    cpu_stack_b = (
        _build_cpu_stack_for_block(events_b, blocks_b_ranges[rep_idx])
        if blocks_b_ranges and rep_idx < len(blocks_b_ranges) else []
    )

    # Preamble
    if preamble_a or preamble_b:
        click.echo(f"\n  [前导区]")
        if preamble_a:
            click.echo(f"    A: {len(preamble_a)} 个 kernel")
            _print_compressed(preamble_a, "      ", max_lines=10)
        if preamble_b:
            click.echo(f"    B: {len(preamble_b)} 个 kernel")
            _print_compressed(preamble_b, "      ", max_lines=10)

    # Representative block comparison
    click.echo(
        f"\n  [代表块 #{rep_full_idx + 1}]"
        f" (A: {len(rep_a)} vs B: {len(rep_b)} 个显著 kernel)"
    )
    _print_block_ab_comparison(rep_a, rep_b, cpu_stack_a, cpu_stack_b)

    # Outlier blocks summary - show kernel-level diffs (within analyzed range only)
    outlier_a = {sig: idxs for sig, idxs in groups_a.items() if sig != rep_sig_a}
    outlier_b = {sig: idxs for sig, idxs in groups_b.items() if sig != rep_sig_b}

    if outlier_a or outlier_b:
        click.echo(f"\n  [差异块]")

    def _show_outliers(outlier_groups, block_kernels, rep_block, label):
        n_outlier = sum(len(v) for v in outlier_groups.values())
        click.echo(
            f"    {label}: {n_outlier} 个块与代表块不同 "
            f"({len(outlier_groups)} 种模式)"
        )
        for sig, idxs in sorted(outlier_groups.items(), key=lambda x: -len(x[1])):
            # Display block numbers in full-range terms
            full_idxs = [range_lo + i for i in idxs]
            block_list = ", ".join(f"#{i + 1}" for i in full_idxs[:6])
            if len(full_idxs) > 6:
                block_list += f" ... (#{full_idxs[-1] + 1})"
            diff_lines = _print_outlier_diff(block_kernels[idxs[0]], rep_block)
            if len(diff_lines) <= 10:
                click.echo(f"      [{block_list}] ({len(idxs)} 个块):")
                for line in diff_lines:
                    click.echo(line)
            else:
                # Too many diffs, just summarize
                names_rep = [n for n, _, _ in _compress_sequence(rep_block)]
                names_out = [n for n, _, _ in _compress_sequence(block_kernels[idxs[0]])]
                added = set(names_out) - set(names_rep)
                removed = set(names_rep) - set(names_out)
                diff_parts = []
                if added:
                    diff_parts.append(f"+{len(added)} kernels")
                if removed:
                    diff_parts.append(f"-{len(removed)} kernels")
                if not diff_parts:
                    diff_parts.append(f"顺序/数量不同")
                click.echo(
                    f"      [{block_list}] ({len(idxs)} 个块): "
                    f"{', '.join(diff_parts)}"
                )

    if outlier_a:
        _show_outliers(outlier_a, bk_a, rep_a, "A")
    if outlier_b:
        _show_outliers(outlier_b, bk_b, rep_b, "B")

    # Epilogue
    if epilogue_a or epilogue_b:
        click.echo(f"\n  [尾部区]")
        if epilogue_a:
            click.echo(f"    A: {len(epilogue_a)} 个 kernel")
            _print_compressed(epilogue_a, "      ", max_lines=10)
        if epilogue_b:
            click.echo(f"    B: {len(epilogue_b)} 个 kernel")
            _print_compressed(epilogue_b, "      ", max_lines=10)

    click.echo()


def _print_overall_stats(events_a, events_b):
    stats_a = compute_stats(events_a)
    stats_b = compute_stats(events_b)

    click.echo("=== 总体统计 ===")
    click.echo(f"{'':>20s} {'Trace A':>15s} {'Trace B':>15s} {'Diff':>20s}")
    click.echo(
        f"{'Total time:':>20s} {fmt_time(stats_a['total']):>15s}"
        f" {fmt_time(stats_b['total']):>15s} {fmt_diff(stats_a['total'], stats_b['total']):>20s}"
    )
    click.echo(
        f"{'GPU kernel time:':>20s} {fmt_time(stats_a['gpu']):>15s}"
        f" {fmt_time(stats_b['gpu']):>15s} {fmt_diff(stats_a['gpu'], stats_b['gpu']):>20s}"
    )
    click.echo(
        f"{'CPU time:':>20s} {fmt_time(stats_a['cpu']):>15s}"
        f" {fmt_time(stats_b['cpu']):>15s} {fmt_diff(stats_a['cpu'], stats_b['cpu']):>20s}"
    )
    click.echo(
        f"{'Unique operators:':>20s} {stats_a['unique_ops']:>15d}"
        f" {stats_b['unique_ops']:>15d} {stats_b['unique_ops'] - stats_a['unique_ops']:>+20d}"
    )
    click.echo()


@click.command("diff")
@click.argument("trace_a")
@click.argument("trace_b")
@click.option("--csv", "csv_path", help="Export diff report to CSV file")
@click.option("--max-seq-diff", type=int, default=100, help="Max sequence diff lines to show (default: 100)")
@click.option("--gpu-only", is_flag=True, default=False, help="Only compare GPU/compute-related events (filter out infrastructure)")
@click.option("--block-name", type=str, default=None,
              help="CPU event name substring to use as repeating block boundary (e.g. 'run_batch', 'forward')")
@click.option("--blocks", "blocks_range", type=str, default=None,
              help="Block range to analyze (0-based, e.g. '10:20', '100:', ':50')")
def diff(trace_a, trace_b, csv_path, max_seq_diff, gpu_only, block_name, blocks_range):
    """Compare two trace files.

    Analyzes GPU kernel diff, call sequence, and overall stats between
    two PyTorch profiler traces. All timestamps are normalized
    (first event = 0) for easy comparison.

    \b
    Examples:
      trace-tool diff trace_a.json trace_b.json
      trace-tool diff a.json b.json --block-name CompiledFxGraph
      trace-tool diff a.json b.json --block-name CompiledFxGraph --blocks 10:20
      trace-tool diff a.json b.json --block-name run_batch
      trace-tool diff a.json b.json.gz --gpu-only --csv report.csv
    """
    click.echo(f"Loading trace A: {trace_a} ...")
    ta = load_trace(trace_a)
    click.echo(f"Loading trace B: {trace_b} ...")
    tb = load_trace(trace_b)
    click.echo()
    do_diff(ta, tb, max_seq_diff=max_seq_diff, csv_path=csv_path,
            gpu_only=gpu_only, block_name=block_name, blocks_range=blocks_range)
