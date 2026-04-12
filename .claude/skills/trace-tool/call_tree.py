"""Call tree reconstruction from Chrome trace events."""

from collections import defaultdict

from trace_io import is_complete_event


class CallNode:
    """A node in the call tree reconstructed from nested trace events."""

    __slots__ = ("name", "ts", "dur", "tid", "pid", "children", "parent", "event")

    def __init__(self, event):
        self.name = event.get("name", "")
        self.ts = event["ts"]
        self.dur = event.get("dur", 0)
        self.tid = event.get("tid", 0)
        self.pid = event.get("pid", 0)
        self.children = []
        self.parent = None
        self.event = event

    @property
    def end(self):
        return self.ts + self.dur

    def call_path(self):
        """Build call path string from root to this node."""
        parts = []
        node = self
        while node is not None:
            parts.append(node.name)
            node = node.parent
        parts.reverse()
        return " > ".join(parts)


def build_call_trees(events):
    """Build call trees from complete events grouped by (pid, tid).

    For each thread, events are sorted by ts and nested by containment:
    a parent event's [ts, ts+dur] range fully contains the child's range.

    Returns dict: (pid, tid) -> list of root CallNodes
    """
    by_thread = defaultdict(list)
    for ev in events:
        if is_complete_event(ev):
            by_thread[(ev.get("pid", 0), ev.get("tid", 0))].append(ev)

    trees = {}
    for key, evts in by_thread.items():
        evts.sort(key=lambda e: (e["ts"], -e["dur"]))
        roots = []
        stack = []

        for ev in evts:
            node = CallNode(ev)
            while stack and stack[-1].end <= node.ts:
                stack.pop()

            if stack and stack[-1].ts <= node.ts and node.end <= stack[-1].end:
                node.parent = stack[-1]
                stack[-1].children.append(node)
            else:
                roots.append(node)

            stack.append(node)

        trees[key] = roots
    return trees


def collect_all_paths(roots, max_depth=10):
    """Collect all node call paths with duration, timestamp, and children names.

    Args:
        roots: list of root CallNodes
        max_depth: maximum call tree depth to traverse (default 10)

    Returns list of (call_path, duration, ts, children_names)
    """
    results = []

    def _walk(node, path_prefix, depth):
        path = f"{path_prefix} > {node.name}" if path_prefix else node.name
        child_names = [c.name for c in node.children]
        results.append((path, node.dur, node.ts, child_names))
        if depth < max_depth:
            for child in node.children:
                _walk(child, path, depth + 1)

    for root in roots:
        _walk(root, "", 0)
    return results
