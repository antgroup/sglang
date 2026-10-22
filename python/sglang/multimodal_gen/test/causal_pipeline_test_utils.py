# SPDX-License-Identifier: Apache-2.0

import queue
import threading


class _Links:
    def __init__(self, ranks):
        self.queues = {
            (src, dst): queue.Queue(maxsize=1)
            for src in range(ranks)
            for dst in range(ranks)
            if src != dst
        }
        self.barrier = threading.Barrier(ranks, timeout=5)
        self.errors = [None] * ranks


class _Transport:
    def __init__(self, links, rank):
        self.links, self.rank = links, rank

    def receive(self):
        return self.recv_tensor_dict(src=self.rank - 1)

    def send(self, payload):
        self.send_tensor_dict(payload, dst=self.rank + 1)

    def recv_tensor_dict(self, src):
        return self.links.queues[src, self.rank].get(timeout=5)

    def send_tensor_dict(self, payload, dst):
        self.links.queues[self.rank, dst].put(payload, timeout=5)

    def finish(self, error):
        self.links.errors[self.rank] = error
        self.links.barrier.wait()
        first = next((item for item in self.links.errors if item), None)
        self.links.barrier.wait()
        if first:
            raise RuntimeError(first)


def _parallel(functions):
    results, errors = [None] * len(functions), [None] * len(functions)

    def invoke(rank, function):
        try:
            results[rank] = function()
        except Exception as exc:
            errors[rank] = exc

    threads = [
        threading.Thread(target=invoke, args=(i, fn), daemon=True)
        for i, fn in enumerate(functions)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=7)
    if any(thread.is_alive() for thread in threads):
        raise AssertionError("Pipeline did not drain")
    return results, errors
