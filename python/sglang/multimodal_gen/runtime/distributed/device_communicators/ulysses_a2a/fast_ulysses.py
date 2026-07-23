"""Optional fast-ulysses adapter.

The currently reviewed upstream binding does not expose a stream-ordered
cross-rank slot reclaim primitive. Reusing its symmetric output tags without
that primitive can overwrite a previous generation while a peer is still
copying or consuming it. The adapter therefore fails closed until a compatible
binding is installed.
"""

from __future__ import annotations

import importlib.util

from .base import UlyssesA2AUnsupportedError


class FastUlyssesA2ABackend:
    name = "fast_ulysses"

    def __init__(self, *args, **kwargs) -> None:
        if importlib.util.find_spec("fast_ulysses") is None:
            raise UlyssesA2AUnsupportedError(
                "fast_ulysses is not installed; forced backend cannot start"
            )
        raise UlyssesA2AUnsupportedError(
            "installed fast_ulysses binding has no verified stream-ordered "
            "cross-rank reclaim adapter; backend remains fail-closed"
        )

    def close(self) -> None:
        return
