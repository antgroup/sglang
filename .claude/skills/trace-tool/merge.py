"""Merge subcommand - merge two traces with visual diff annotations."""

import sys
from collections import defaultdict
from pathlib import Path

skill_dir = Path(__file__).parent
if str(skill_dir) not in sys.path:
    sys.path.insert(0, str(skill_dir))

import click

from trace_io import get_events, is_complete_event, load_trace, save_trace

# GPU event categories
_GPU_CATS = {"kernel", "gpu_user_annotation", "gpu_memcpy"}

# Flow ID ranges to avoid collision with existing ac2g flows
_FLOW_ID_OFFSET = 1_000_000
_DIFF_FLOW_ID_BASE = 5_000_000


def _first_gpu_kernel_ts(events):
    """Find the timestamp of the first GPU kernel event."""
    return min(
        (e["ts"] for e in events
         if e.get("ph") == "X" and e.get("cat", "") in _GPU_CATS),
        default=None,
    )


def _data_pids(events):
    """Find pids that have actual complete (X) events."""
    pids = set()
    for e in events:
        if e.get("ph") == "X":
            pids.add(e.get("pid"))
    return pids


def _classify_pids(events, dpids):
    """Classify pids into GPU and CPU based on event categories."""
    pid_cats = defaultdict(set)
    for e in events:
        if e.get("ph") == "X" and e.get("pid") in dpids:
            pid_cats[e["pid"]].add(e.get("cat", ""))
    gpu_pid = cpu_pid = None
    for pid, cats in pid_cats.items():
        if "kernel" in cats:
            gpu_pid = pid
        elif "cpu_op" in cats:
            cpu_pid = pid
    return gpu_pid, cpu_pid


def _build_pid_map(data_pids, gpu_pid, cpu_pid, gpu_new, cpu_new, start):
    """Build pid mapping: original -> new pid."""
    pid_map = {gpu_pid: gpu_new, cpu_pid: cpu_new}
    next_pid = start
    for pid in sorted(data_pids - {gpu_pid, cpu_pid}, key=str):
        pid_map[pid] = next_pid
        next_pid += 1
    return pid_map, next_pid


def _normalize_events(events, offset):
    """Shift all timestamps by offset."""
    result = []
    for ev in events:
        e = dict(ev)
        if "ts" in e:
            e["ts"] = e["ts"] - offset
        result.append(e)
    return result


def _build_paired_dur_map(events_a, events_b):
    """Build per-event paired diff: Nth occurrence of (name,cat) in A vs Nth in B.

    Returns two dicts: a_map[(name,cat,idx)] -> b_dur, b_map[(name,cat,idx)] -> a_dur.
    """
    def group_durs(events):
        groups = defaultdict(list)
        for ev in events:
            if ev.get("ph") == "X":
                key = (ev["name"], ev.get("cat", ""))
                groups[key].append(ev.get("dur", 0))
        return groups

    ga = group_durs(events_a)
    gb = group_durs(events_b)

    a_map = {}
    b_map = {}
    for key in set(ga) | set(gb):
        durs_a = ga.get(key, [])
        durs_b = gb.get(key, [])
        for i in range(max(len(durs_a), len(durs_b))):
            da = durs_a[i] if i < len(durs_a) else None
            db = durs_b[i] if i < len(durs_b) else None
            if da is not None:
                a_map[(key[0], key[1], i)] = db
            if db is not None:
                b_map[(key[0], key[1], i)] = da
    return a_map, b_map


def _annotate_event(e, label, pair_map, occurrence, threshold_pct):
    """Annotate a complete event with paired diff info."""
    other_label = "B" if label == "A" else "A"
    name = e.get("name", "")
    cat = e.get("cat", "")
    dur = e.get("dur", 0)
    args = dict(e.get("args", {}))
    args["original_name"] = name

    key = (name, cat)
    idx = occurrence[key]
    occurrence[key] += 1
    other_dur = pair_map.get((name, cat, idx))

    if other_dur is None:
        args["diff_status"] = f"only_in_{label}"
        if cat in _GPU_CATS:
            e["name"] = f"[ONLY_{label}] {name}"
    else:
        args[f"{other_label}_dur_us"] = round(other_dur, 1)
        if other_dur > 0 and dur > 0:
            diff_pct = (dur - other_dur) / other_dur * 100
        elif other_dur == 0 and dur == 0:
            diff_pct = 0
        else:
            diff_pct = 100 if dur > other_dur else -100

        args["diff_pct"] = f"{diff_pct:+.1f}%"

        if abs(diff_pct) > threshold_pct:
            args["diff_status"] = "slower" if diff_pct > 0 else "faster"
            if cat in _GPU_CATS:
                tag = f"[SLOW {diff_pct:+.0f}%]" if diff_pct > 0 else f"[FAST {diff_pct:+.0f}%]"
                e["name"] = f"{tag} {name}"
        else:
            args["diff_status"] = "matched"

    e["args"] = args


def _process_trace(events, pid_map, time_offset, label,
                   pair_map, threshold_pct, flow_id_offset=0):
    """Remap pids, shift timestamps, annotate diffs. Skip empty-pid and metadata events."""
    occurrence = defaultdict(int)
    result = []
    for e in events:
        pid = e.get("pid")
        if pid not in pid_map:
            continue
        ne = dict(e)
        ne["pid"] = pid_map[ne["pid"]]
        if "ts" in ne:
            ne["ts"] = ne["ts"] - time_offset
        # Offset flow ids for B to avoid collision with A
        if flow_id_offset and ne.get("ph") in ("s", "f", "t") and "id" in ne:
            fid = ne["id"]
            ne["id"] = fid + flow_id_offset if isinstance(fid, int) else f"b_{fid}"
        # Skip original metadata (we generate our own)
        if ne.get("ph") == "M":
            continue
        if ne.get("ph") == "X":
            _annotate_event(ne, label, pair_map, occurrence, threshold_pct)
        result.append(ne)
    return result


def _index_gpu_by_name(events, gpu_pid):
    """Group GPU events by original_name, sorted by timestamp."""
    by_name = defaultdict(list)
    for ev in events:
        if ev.get("ph") != "X" or ev.get("pid") != gpu_pid:
            continue
        if ev.get("cat", "") not in _GPU_CATS:
            continue
        orig = ev.get("args", {}).get("original_name", ev.get("name", ""))
        by_name[orig].append(ev)
    for name in by_name:
        by_name[name].sort(key=lambda e: e["ts"])
    return by_name


def _build_flow_events(out_a, out_b, a_gpu_pid, b_gpu_pid):
    """Create flow arrows linking matching GPU kernels between A and B.

    Format matches ac2g conventions:
    - "s" always earlier in time than "f"
    - "s" has no "bp", "f" has "bp":"e"
    - IDs start from _DIFF_FLOW_ID_BASE to avoid collision
    """
    idx_a = _index_gpu_by_name(out_a, a_gpu_pid)
    idx_b = _index_gpu_by_name(out_b, b_gpu_pid)

    flows = []
    flow_id = _DIFF_FLOW_ID_BASE

    for name in sorted(set(idx_a) & set(idx_b)):
        evts_a = idx_a[name]
        evts_b = idx_b[name]
        n = min(len(evts_a), len(evts_b))
        for i in range(n):
            ea = evts_a[i]
            eb = evts_b[i]
            ts_a = ea["ts"] + ea.get("dur", 0) / 2
            ts_b = eb["ts"] + eb.get("dur", 0) / 2
            # "s" must be earlier in time
            if ts_a <= ts_b:
                s_pid, s_tid, s_ts = a_gpu_pid, ea["tid"], ts_a
                f_pid, f_tid, f_ts = b_gpu_pid, eb["tid"], ts_b
            else:
                s_pid, s_tid, s_ts = b_gpu_pid, eb["tid"], ts_b
                f_pid, f_tid, f_ts = a_gpu_pid, ea["tid"], ts_a
            flows.append({
                "ph": "s", "id": flow_id, "cat": "diff_flow",
                "name": name, "pid": s_pid, "tid": s_tid, "ts": s_ts,
            })
            flows.append({
                "ph": "f", "id": flow_id, "cat": "diff_flow",
                "name": name, "pid": f_pid, "tid": f_tid, "ts": f_ts,
                "bp": "e",
            })
            flow_id += 1

    return flows, flow_id - _DIFF_FLOW_ID_BASE


def _get_thread_names(events, target_pid):
    names = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name" and e.get("pid") == target_pid:
            names[e["tid"]] = e.get("args", {}).get("name", "")
    return names


def _add_metadata(pid, process_name, sort_index, thread_names=None):
    meta = [
        {"ph": "M", "pid": pid, "tid": 0, "name": "process_name",
         "args": {"name": process_name}},
        {"ph": "M", "pid": pid, "tid": 0, "name": "process_sort_index",
         "args": {"sort_index": sort_index}},
    ]
    if thread_names:
        for tid, tname in thread_names.items():
            meta.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_name",
                         "args": {"name": tname}})
    return meta


def do_merge(trace_a, trace_b, output_path, threshold_pct=10.0, flow=True):
    """Merge two traces into one file with diff annotations.

    - Pids remapped to clean values (A: GPU=1, CPU=2; B: GPU=3, CPU=4)
    - Empty device pids (1-15) dropped
    - Aligned by first GPU kernel at t=0
    - Paired (Nth vs Nth) diff comparison
    - GPU event names prefixed with [SLOW]/[FAST]/[ONLY] for significant diffs
    - CPU event names unchanged, diff info in args only
    - Flow arrows between A:GPU and B:GPU matching kernels
    """
    events_a = get_events(trace_a)
    events_b = get_events(trace_b)

    # Classify pids
    a_dpids = _data_pids(events_a)
    b_dpids = _data_pids(events_b)
    a_gpu_pid, a_cpu_pid = _classify_pids(events_a, a_dpids)
    b_gpu_pid, b_cpu_pid = _classify_pids(events_b, b_dpids)
    click.echo(f"A: GPU pid={a_gpu_pid}, CPU pid={a_cpu_pid}")
    click.echo(f"B: GPU pid={b_gpu_pid}, CPU pid={b_cpu_pid}")

    # New pid assignments
    A_GPU, A_CPU, B_GPU, B_CPU = 1, 2, 3, 4
    a_pid_map, next_pid = _build_pid_map(a_dpids, a_gpu_pid, a_cpu_pid, A_GPU, A_CPU, 5)
    b_pid_map, _ = _build_pid_map(b_dpids, b_gpu_pid, b_cpu_pid, B_GPU, B_CPU, next_pid)

    # Align by first GPU kernel
    gts_a = _first_gpu_kernel_ts(events_a)
    gts_b = _first_gpu_kernel_ts(events_b)
    if gts_a is not None and gts_b is not None:
        click.echo(f"Aligning by first GPU kernel: A={gts_a:.0f}us, B={gts_b:.0f}us")
    else:
        gts_a = gts_a or min((e["ts"] for e in events_a if "ts" in e), default=0)
        gts_b = gts_b or min((e["ts"] for e in events_b if "ts" in e), default=0)
        click.echo("No GPU kernels found, aligning by first event")

    # Build paired diff map
    click.echo("Building paired diff map ...")
    a_pair_map, b_pair_map = _build_paired_dur_map(events_a, events_b)

    # Process traces
    click.echo("Processing A ...")
    out_a = _process_trace(events_a, a_pid_map, gts_a, "A", a_pair_map, threshold_pct)
    click.echo("Processing B ...")
    out_b = _process_trace(events_b, b_pid_map, gts_b, "B", b_pair_map, threshold_pct,
                           flow_id_offset=_FLOW_ID_OFFSET)

    # Metadata
    meta = []
    meta += _add_metadata(A_GPU, "A: GPU", 0, _get_thread_names(events_a, a_gpu_pid))
    meta += _add_metadata(A_CPU, "A: CPU", 1, _get_thread_names(events_a, a_cpu_pid))
    meta += _add_metadata(B_GPU, "B: GPU", 2, _get_thread_names(events_b, b_gpu_pid))
    meta += _add_metadata(B_CPU, "B: CPU", 3, _get_thread_names(events_b, b_cpu_pid))
    for orig_pid, new_pid in a_pid_map.items():
        if new_pid not in (A_GPU, A_CPU):
            meta += _add_metadata(new_pid, f"A: {orig_pid}", new_pid)
    for orig_pid, new_pid in b_pid_map.items():
        if new_pid not in (B_GPU, B_CPU):
            meta += _add_metadata(new_pid, f"B: {orig_pid}", new_pid)

    merged_events = meta + out_a + out_b

    # Flow arrows
    if flow:
        click.echo("Building diff flow arrows ...")
        flow_events, n_arrows = _build_flow_events(out_a, out_b, A_GPU, B_GPU)
        click.echo(f"  {n_arrows} arrow pairs (A:GPU <-> B:GPU)")
        merged_events += flow_events

    # Save
    merged_trace = {"traceEvents": merged_events}
    click.echo(f"Total events: {len(merged_events)}")
    save_trace(merged_trace, output_path)
    click.echo(f"Saved to: {output_path}")

    # Summary
    gpu_status = defaultdict(int)
    for e in merged_events:
        if e.get("ph") == "X" and e.get("cat", "") in _GPU_CATS:
            s = e.get("args", {}).get("diff_status", "")
            if s:
                gpu_status[s] += 1
    click.echo("\n=== GPU Diff Summary ===")
    for s, c in sorted(gpu_status.items(), key=lambda x: -x[1]):
        click.echo(f"  {s}: {c}")
    click.echo()


@click.command("merge")
@click.argument("trace_a")
@click.argument("trace_b")
@click.option("-o", "--output", required=True, help="Output merged trace file (.json or .json.gz)")
@click.option("--threshold", type=float, default=10.0,
              help="Duration diff threshold %% for highlighting (default: 10)")
@click.option("--no-flow", is_flag=True, default=False,
              help="Disable flow arrows between matching GPU kernels")
def merge(trace_a, trace_b, output, threshold, no_flow):
    """Merge two traces into one file with visual diff annotations.

    Creates a combined trace viewable in Perfetto / chrome://tracing where:
    - A: GPU, A: CPU, B: GPU, B: CPU as separate process groups
    - GPU events prefixed with [SLOW +xx%] / [FAST -xx%] / [ONLY_A] / [ONLY_B]
    - Flow arrows connect matching GPU kernels between A and B
    - Click any event to see diff details (diff_pct, other_dur, etc.)
    - Paired comparison: Nth occurrence vs Nth occurrence (not vs average)
    - Aligned by first GPU kernel at t=0
    - Pre-existing flow events (ac2g) preserved within each trace

    \b
    Examples:
      trace-tool merge a.json b.json -o merged.json.gz
      trace-tool merge a.json b.json -o merged.json.gz --threshold 20
      trace-tool merge a.json b.json -o merged.json.gz --no-flow
    """
    click.echo(f"Loading Trace A: {trace_a} ...")
    ta = load_trace(trace_a)
    click.echo(f"Loading Trace B: {trace_b} ...")
    tb = load_trace(trace_b)
    click.echo()
    do_merge(ta, tb, output, threshold_pct=threshold, flow=not no_flow)
