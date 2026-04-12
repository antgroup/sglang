"""Trim subcommand - filter/extract a subset of trace events."""

import sys
from collections import defaultdict
from pathlib import Path

# Add skill directory to path
skill_dir = Path(__file__).parent
if str(skill_dir) not in sys.path:
    sys.path.insert(0, str(skill_dir))

import click

from trace_io import (
    get_events,
    get_first_ts,
    is_complete_event,
    find_step_boundaries,
    load_trace,
    save_trace,
    sorted_complete_events,
)

_GPU_CATS = {"kernel", "gpu_user_annotation", "gpu_memcpy"}


def find_nth_block_ts(sorted_evts, block_name, n):
    """Find the timestamp of the n-th occurrence (0-based) of a named block."""
    count = 0
    for ev in sorted_evts:
        if block_name in ev.get("name", ""):
            if count == n:
                return ev["ts"]
            count += 1
    raise click.ClickException(
        f"Block '{block_name}' occurrence #{n} not found (only {count} found)"
    )


def _first_gpu_kernel_ts(events):
    """Find the timestamp of the first GPU kernel event."""
    return min(
        (e["ts"] for e in events
         if e.get("ph") == "X" and e.get("cat", "") in _GPU_CATS),
        default=None,
    )


def _normalize_to_gpu(events):
    """Shift all timestamps so the first GPU kernel is at t=0.

    Returns (shifted_events, offset) or (events, 0) if no GPU kernels found.
    """
    gpu_ts = _first_gpu_kernel_ts(events)
    if gpu_ts is None:
        return events, 0
    result = []
    for ev in events:
        e = dict(ev)
        if "ts" in e:
            e["ts"] = e["ts"] - gpu_ts
        result.append(e)
    return result, gpu_ts


def do_trim(trace, name, time_start, time_end, step, from_nth, to_nth,
            from_block, to_block, block_range, top, tid):
    """Filter trace events based on trim arguments.

    Returns (result_trace, summary_text).
    """
    events = get_events(trace)
    base_ts = get_first_ts(events)
    complete_sorted = sorted_complete_events(events)

    # Determine absolute time range
    ts_start = None
    ts_end = None

    # --step
    if step is not None:
        step_bounds = find_step_boundaries(events)
        if step not in step_bounds:
            available = sorted(step_bounds.keys())
            raise click.ClickException(
                f"Step {step} not found. Available steps: {available}"
            )
        ts_start, ts_end = step_bounds[step]

    # --time-start / --time-end (normalized us -> absolute)
    if time_start is not None:
        ts = base_ts + time_start
        ts_start = ts if ts_start is None else max(ts_start, ts)
    if time_end is not None:
        ts = base_ts + time_end
        ts_end = ts if ts_end is None else min(ts_end, ts)

    # --from-nth / --to-nth (0-based)
    if from_nth is not None:
        if not (0 <= from_nth < len(complete_sorted)):
            raise click.ClickException(
                f"--from-nth {from_nth} out of range (total {len(complete_sorted)} events)"
            )
        ts = complete_sorted[from_nth]["ts"]
        ts_start = ts if ts_start is None else max(ts_start, ts)
    if to_nth is not None:
        if not (0 <= to_nth < len(complete_sorted)):
            raise click.ClickException(
                f"--to-nth {to_nth} out of range (total {len(complete_sorted)} events)"
            )
        ev = complete_sorted[to_nth]
        ts = ev["ts"] + ev["dur"]
        ts_end = ts if ts_end is None else min(ts_end, ts)

    # --block-range NAME:START:END (0-based, END exclusive)
    # Converts to from_block / to_block internally
    if block_range:
        parts = block_range.split(":")
        if len(parts) < 2:
            raise click.ClickException(
                f"Invalid --block-range format '{block_range}', "
                f"expected NAME:START:END or NAME:START: or NAME::END"
            )
        # Last two parts are start:end, everything before is the name
        # (name may contain colons like "aten::mm")
        end_part = parts[-1]
        start_part = parts[-2]
        blk_name_parts = parts[:-2]
        if not blk_name_parts:
            raise click.ClickException(
                f"Invalid --block-range format '{block_range}', "
                f"expected NAME:START:END (e.g. 'aten::mm:1:4')"
            )
        blk_name = ":".join(blk_name_parts)
        if start_part.strip():
            from_block = from_block or f"{blk_name}:{start_part.strip()}"
        if end_part.strip():
            # END is exclusive in block_range, but to_block is inclusive, so subtract 1
            end_idx = int(end_part.strip()) - 1
            if end_idx < 0:
                raise click.ClickException(
                    f"Invalid --block-range END '{end_part.strip()}', must be >= 1"
                )
            to_block = to_block or f"{blk_name}:{end_idx}"

    # --from-block / --to-block (NAME or NAME:N, 0-based)

    def _parse_block_spec(spec):
        """Parse 'NAME' or 'NAME:N' into (name, index)."""
        if ":" in spec:
            parts = spec.rsplit(":", 1)
            return parts[0], int(parts[1])
        return spec, 0

    if from_block or to_block:
        from_name, from_n = _parse_block_spec(from_block) if from_block else (None, 0)
        to_name, to_n = _parse_block_spec(to_block) if to_block else (None, None)

        if from_block:
            ts = find_nth_block_ts(complete_sorted, from_name, from_n)
            ts_start = ts if ts_start is None else max(ts_start, ts)

        if to_block:
            count = 0
            ts = None
            for ev in complete_sorted:
                if to_name in ev.get("name", ""):
                    if count == to_n:
                        ts = ev["ts"] + ev["dur"]
                        break
                    count += 1
            if ts is None:
                raise click.ClickException(
                    f"Block '{to_name}' occurrence #{to_n} not found (only {count} found)"
                )
            ts_end = ts if ts_end is None else min(ts_end, ts)

    # Apply filters
    name_keywords = name if name else None
    tid_filter = set(tid) if tid else None

    filtered = []
    for ev in events:
        ph = ev.get("ph", "")

        # Always keep metadata events (but NOT flow/counter - they need filtering too)
        if ph == "M":
            filtered.append(ev)
            continue

        # Time range filter (strict: event must start within range)
        if ts_start is not None and "ts" in ev:
            if ev["ts"] < ts_start:
                continue
        if ts_end is not None and "ts" in ev:
            if ev["ts"] > ts_end:
                continue

        # Name filter
        if name_keywords:
            ev_name = ev.get("name", "")
            if not any(kw.lower() in ev_name.lower() for kw in name_keywords):
                if ph != "X":
                    filtered.append(ev)
                    continue
                continue

        # TID filter
        if tid_filter and ev.get("tid") not in tid_filter:
            continue

        filtered.append(ev)

    # --top: keep only top-N operator types by total duration
    if top:
        dur_by_name = defaultdict(float)
        for ev in filtered:
            if is_complete_event(ev):
                dur_by_name[ev["name"]] += ev["dur"]
        top_names = set(
            n for n, _ in sorted(dur_by_name.items(), key=lambda x: -x[1])[:top]
        )
        filtered = [
            ev
            for ev in filtered
            if not is_complete_event(ev) or ev.get("name", "") in top_names
        ]

    # Normalize: align first GPU kernel to t=0
    filtered, gpu_offset = _normalize_to_gpu(filtered)

    # Build output trace
    if isinstance(trace, dict):
        result = {k: (filtered if k == "traceEvents" else v) for k, v in trace.items()}
    else:
        result = filtered

    # Summary
    orig_count = sum(1 for e in events if is_complete_event(e))
    filt_count = sum(1 for e in filtered if is_complete_event(e))
    lines = [
        f"Original: {orig_count} complete events",
        f"Filtered: {filt_count} complete events",
    ]
    if gpu_offset:
        lines.append(f"Aligned first GPU kernel to t=0 (offset: {gpu_offset:.0f} us)")
    if ts_start is not None or ts_end is not None:
        t0 = (ts_start - base_ts) if ts_start else 0
        t1 = (ts_end - base_ts) if ts_end else "end"
        lines.append(f"Time range (normalized): {t0} us - {t1} us")

    return result, "\n".join(lines)


@click.command("trim")
@click.argument("input_file", metavar="INPUT")
@click.option("-o", "--output", required=True, help="Output trace file (.json or .json.gz)")
@click.option("-n", "--name", multiple=True, help="Filter by operator name keyword (repeatable, OR logic)")
@click.option("--time-start", type=float, help="Start time in normalized us")
@click.option("--time-end", type=float, help="End time in normalized us")
@click.option("--step", type=int, help="Filter by ProfilerStep number")
@click.option("--from-nth", type=int, help="Start from the N-th complete event (0-based)")
@click.option("--to-nth", type=int, help="End at the N-th complete event (0-based, inclusive)")
@click.option("--from-block", type=str, default=None,
              help="Start from N-th occurrence of NAME (NAME or NAME:N, 0-based)")
@click.option("--to-block", type=str, default=None,
              help="End at N-th occurrence of NAME (NAME or NAME:N, 0-based)")
@click.option("--block-range", type=str, default=None,
              help="Block range NAME:START:END (0-based, END exclusive). E.g. 'forward:1:3'")
@click.option("--top", type=int, help="Keep only top-N operator types by total duration")
@click.option("--tid", multiple=True, type=int, help="Filter by thread ID (repeatable)")
def trim(input_file, output, name, time_start, time_end, step, from_nth,
         to_nth, from_block, to_block, block_range, top, tid):
    """Trim/filter a trace file.

    Extract a subset of operators from a large trace file.
    Output can be viewed in chrome://tracing.

    \b
    Examples:
      # Extract step 2
      trace-tool trim trace.json -o trimmed.json --step 2

      # Extract operators 10-50 (0-based)
      trace-tool trim trace.json -o trimmed.json --from-nth 10 --to-nth 50

      # Extract between 2nd and 5th aten::mm (0-based: index 1 to 4)
      trace-tool trim trace.json -o out.json --from-block "aten::mm:1" --to-block "aten::mm:4"

      # Same as above using --block-range (END exclusive)
      trace-tool trim trace.json -o out.json --block-range "aten::mm:1:5"

      # Extract 2 iterations of forward
      trace-tool trim trace.json -o out.json --block-range "forward:1:3"

      # Filter by name keywords
      trace-tool trim trace.json -o out.json -n gemm -n attention
    """
    click.echo(f"Loading {input_file} ...")
    trace = load_trace(input_file)

    result, summary = do_trim(
        trace, name=list(name), time_start=time_start, time_end=time_end,
        step=step, from_nth=from_nth, to_nth=to_nth,
        from_block=from_block, to_block=to_block, block_range=block_range,
        top=top, tid=list(tid),
    )
    click.echo(summary)

    click.echo(f"Saving to {output} ...")
    save_trace(result, output)
    click.echo("Done.")
