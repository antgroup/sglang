"""Trace file I/O and event helpers."""

import gzip
import json
from collections import defaultdict


def load_trace(path):
    """Load a Chrome trace JSON file (.json or .json.gz)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_trace(data, path):
    """Save a Chrome trace JSON file (.json or .json.gz)."""
    text = json.dumps(data, ensure_ascii=False)
    if path.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def get_events(trace):
    """Extract traceEvents list from trace data."""
    if isinstance(trace, list):
        return trace
    return trace.get("traceEvents", [])


def is_complete_event(ev):
    """Check if event is a complete duration event (ph=X) with dur."""
    return ev.get("ph") == "X" and "dur" in ev and "ts" in ev


def get_first_ts(events):
    """Get the timestamp of the first complete event."""
    for ev in sorted(events, key=lambda e: e.get("ts", float("inf"))):
        if is_complete_event(ev):
            return ev["ts"]
    return 0


def sorted_complete_events(events):
    """Return complete events sorted by (tid, ts)."""
    return sorted(
        [ev for ev in events if is_complete_event(ev)],
        key=lambda e: (str(e.get("tid", 0)), e["ts"]),
    )


def find_step_boundaries(events):
    """Find step boundaries from ProfilerStep events.

    Returns dict: step_number -> (start_ts, end_ts)
    """
    steps = {}
    for ev in events:
        name = ev.get("name", "")
        if "ProfilerStep" in name and is_complete_event(ev):
            parts = name.split("#")
            step_num = (
                int(parts[-1])
                if len(parts) > 1 and parts[-1].isdigit()
                else len(steps)
            )
            steps[step_num] = (ev["ts"], ev["ts"] + ev["dur"])
    return steps


def get_op_sequence(events):
    """Get the ordered sequence of operator names from complete events on main thread.

    Returns list of (name, normalized_ts, dur)
    """
    complete = sorted_complete_events(events)
    if not complete:
        return []

    tid_counts = defaultdict(int)
    for ev in complete:
        tid_counts[ev.get("tid", 0)] += 1
    main_tid = max(tid_counts, key=tid_counts.get)

    return [
        (ev["name"], ev["ts"], ev.get("dur", 0))
        for ev in complete
        if ev.get("tid", 0) == main_tid
    ]
