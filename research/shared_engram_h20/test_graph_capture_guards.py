import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_remote_engram():
    path = ROOT / "python/sglang/srt/layers/remote_engram.py"
    spec = importlib.util.spec_from_file_location("remote_engram_port_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_lookup_does_not_use_prefetch_representatives():
    remote = _load_remote_engram()
    stale_map = object()
    context = SimpleNamespace(representatives=lambda _layer: stale_map)

    with patch.object(remote.torch.cuda, "is_current_stream_capturing", return_value=True):
        assert remote._lookup_representatives(context, 1) is None

    with patch.object(remote.torch.cuda, "is_current_stream_capturing", return_value=False):
        assert remote._lookup_representatives(context, 1) is stale_map


def test_qp_prefetch_capture_lookup_fails_closed():
    remote = _load_remote_engram()

    remote._reject_prefetch_graph_capture(True, False)
    with pytest.raises(RuntimeError, match="unsupported with QP prefetch"):
        remote._reject_prefetch_graph_capture(True, True)
