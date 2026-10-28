"""Unit tests for model-specific server warmup inputs and the startup warmups."""

import asyncio
import base64
import os
import random
import struct
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import warmup as warmup_module
from sglang.srt.entrypoints.http_server import (
    KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64,
    KIMI_VLM_WARMUP_PNG_PICTURE_BASE64,
    MINIMUM_PNG_PICTURE_BASE64,
    _get_vlm_warmup_image_base64,
)
from sglang.srt.entrypoints.warmup import (
    execute_warmups,
    execute_warmups_once,
    prefill_shapes,
    serving_shapes,
    startup_warmup_names,
    warmup_launch_key,
)
from sglang.srt.runtime_context import get_context, get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestVlmWarmupImage(CustomTestCase):
    def test_kimi_k2_uses_representative_vision_image(self):
        image_base64 = _get_vlm_warmup_image_base64(
            {"architectures": ["KimiK25ForConditionalGeneration"]}
        )
        self.assertEqual(image_base64, KIMI_VLM_WARMUP_PNG_PICTURE_BASE64)

        png = base64.b64decode(KIMI_VLM_WARMUP_PNG_PICTURE_BASE64)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (512, 512))

    def test_kimi_k3_uses_native_patch_grid_image(self):
        for model_info in (
            {"architectures": ["KimiK3ForConditionalGeneration"]},
            {"architectures": None, "model_type": "kimi_k3"},
        ):
            with self.subTest(model_info=model_info):
                self.assertEqual(
                    _get_vlm_warmup_image_base64(model_info),
                    KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64,
                )

        png = base64.b64decode(KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (448, 448))

    def test_other_vlms_keep_minimal_startup_image(self):
        self.assertEqual(
            _get_vlm_warmup_image_base64(
                {"architectures": ["Qwen3VLForConditionalGeneration"]}
            ),
            MINIMUM_PNG_PICTURE_BASE64,
        )
        self.assertEqual(
            _get_vlm_warmup_image_base64({"architectures": None}),
            MINIMUM_PNG_PICTURE_BASE64,
        )


class _RecordingTokenizerManager:
    """Stands in for TokenizerManager: records every warmup request and how many overlap."""

    def __init__(self):
        self.context_len = 1 << 20
        self.reqs = []
        self.in_flight = self.max_in_flight = 0

    async def generate_request(self, req, request):
        self.reqs.append(req)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0)
        yield {}
        # A PD prefill server yields before the request is done; completion comes later.
        await asyncio.sleep(0)
        self.in_flight -= 1


def _run_warmup(warmup_fn, mode, *, startup=True, max_running=64, dp_size=1):
    tm = _RecordingTokenizerManager()
    schedule = SimpleNamespace(
        chunked_prefill_size=16384, max_running_requests=max_running
    )
    with ExitStack() as stack:
        for name, value in (
            ("dsv41_sm90_on", lambda subsystem: startup),
            ("get_schedule", lambda: schedule),
            ("get_parallel", lambda: SimpleNamespace(dp_size=dp_size)),
        ):
            stack.enter_context(patch.object(warmup_module, name, value))
        asyncio.run(warmup_fn(mode, tm))
    return tm


class TestStartupWarmupNames(CustomTestCase):
    def test_explicit_serving_shapes_cannot_bypass_startup_selection(self):
        """Opt-out --warmups serving_shapes must submit no generation requests."""
        for enabled in (False, True):
            tm = _RecordingTokenizerManager()
            with (
                self.subTest(enabled=enabled),
                get_context().override_server_args(
                    chunked_prefill_size=16384, max_running_requests=1
                ),
                get_flags().dsv41_sm90.override(
                    subsystems=frozenset({"startup"}) if enabled else frozenset()
                ),
            ):
                asyncio.run(execute_warmups("prefill", ["serving_shapes"], tm))
                self.assertEqual(bool(tm.reqs), enabled)

    def test_role_sets_and_explicit_override(self):
        self.assertEqual(
            startup_warmup_names(explicit=None, disaggregation_mode="null"),
            ["prefill_shapes", "serving_shapes"],
        )
        self.assertEqual(
            startup_warmup_names(explicit=None, disaggregation_mode="prefill"),
            ["prefill_shapes", "serving_shapes"],
        )
        self.assertEqual(
            startup_warmup_names(explicit=None, disaggregation_mode="decode"),
            ["serving_shapes"],
        )
        self.assertEqual(
            startup_warmup_names(explicit="voice_chat", disaggregation_mode="null"),
            ["voice_chat"],
        )
        for off in ("", "none"):
            self.assertEqual(
                startup_warmup_names(explicit=off, disaggregation_mode="null"), []
            )


class _FlushRecorder:
    """Tokenizer-manager stand-in for the once-per-launch tests: counts flush_cache calls."""

    def __init__(self):
        self.flushes = []

    async def flush_cache(self, timeout_s=None):
        self.flushes.append(timeout_s)
        return SimpleNamespace(success=True, message="")


class TestWarmupOncePerLaunch(CustomTestCase):
    def test_explicit_warmup_failure_is_fatal(self):
        """Explicit warmup failures must abort each worker before later warmups."""
        ran = []

        async def boom(mode, tm):
            raise RuntimeError("warmup exploded")

        async def ok(mode, tm):
            ran.append("ok")

        with patch.dict(warmup_module._warmup_registry, _test_boom=boom, _test_ok=ok):
            for launch_key in (None, "explicit-warmup"):
                tm = _FlushRecorder()
                with (
                    self.subTest(launch_key=launch_key),
                    self.assertRaisesRegex(RuntimeError, "warmup exploded"),
                ):
                    asyncio.run(
                        execute_warmups_once(
                            "null",
                            ["_test_boom", "_test_ok"],
                            tm,
                            launch_key=launch_key,
                            explicit=True,
                        )
                    )
                self.assertEqual(tm.flushes, [])
        self.assertEqual(ran, [])

    def test_tokenizer_workers_share_one_warmup_run(self):
        """Every tokenizer worker runs the lifespan; on a PD prefill server with 4 workers,
        overlapping fake-bootstrap streams crashed P. One runs; the rest wait."""
        runs, done = [], []

        async def slow(mode, tm):
            runs.append(mode)
            await asyncio.sleep(0.3)
            done.append("warmup")

        tm = _FlushRecorder()
        launch_key = f"{os.getpid()}-{random.randrange(1 << 30)}"

        async def worker(i):
            await execute_warmups_once(
                "prefill", ["_test_slow"], tm, launch_key=launch_key
            )
            done.append(i)

        async def main():
            await asyncio.gather(*(worker(i) for i in range(4)))

        warmup_module._warmup_registry["_test_slow"] = slow
        try:
            asyncio.run(main())
        finally:
            del warmup_module._warmup_registry["_test_slow"]
        self.assertEqual(runs, ["prefill"])
        self.assertEqual(done[0], "warmup")
        # Exactly one flush (the leader's), after its warmups, waiting for idle.
        self.assertEqual(len(tm.flushes), 1)
        self.assertGreater(tm.flushes[0], 0)

    def test_a_reused_launcher_pid_still_warms_up(self):
        # Done markers outlive an engine restart in the same container; a later launch
        # whose launcher reuses the pid must not take the old marker as its own.
        runs = []

        async def count(mode, tm):
            runs.append(mode)

        warmup_module._warmup_registry["_test_count"] = count
        pid = (os.getpid() << 16) + random.randrange(1 << 16)
        try:
            for start in (1000.0, 2000.0):
                with patch.object(warmup_module.psutil, "Process") as proc:
                    proc.return_value.create_time.return_value = start
                    key = warmup_launch_key(pid)
                asyncio.run(
                    execute_warmups_once(
                        "null", ["_test_count"], _FlushRecorder(), launch_key=key
                    )
                )
        finally:
            del warmup_module._warmup_registry["_test_count"]
        self.assertEqual(runs, ["null", "null"])

    def test_failed_warmup_is_logged_and_the_rest_still_run(self):
        ran = []

        async def boom(mode, tm):
            raise RuntimeError("warmup exploded")

        async def ok(mode, tm):
            ran.append("ok")

        tm = _FlushRecorder()
        warmup_module._warmup_registry.update(_test_boom=boom, _test_ok=ok)
        try:
            with self.assertLogs(warmup_module.logger, "ERROR") as logs:
                asyncio.run(
                    execute_warmups_once(
                        "null", ["_test_boom", "_test_ok"], tm, launch_key=None
                    )
                )
        finally:
            del warmup_module._warmup_registry["_test_boom"]
            del warmup_module._warmup_registry["_test_ok"]
        self.assertEqual(ran, ["ok"])
        self.assertIn("_test_boom", "\n".join(logs.output))
        self.assertEqual(len(tm.flushes), 1)


class TestWarmupShapes(CustomTestCase):
    def test_pd_prefill_warmups_run_one_single_chunk_request_at_a_time(self):
        """A PD prefill batches overlapping fake-bootstrap requests, and upstream cannot
        prefill one over several chunks; each warmup must finish before the next and
        fit one chunk (16384). Off the switch prefill_shapes keeps its upstream sizes."""
        for fn in (prefill_shapes, serving_shapes):
            with self.subTest(fn=fn.__name__):
                tm = _run_warmup(fn, "prefill", max_running=128)
                lens = [len(r.input_ids) for r in tm.reqs]
                self.assertEqual(tm.max_in_flight, 1)
                self.assertEqual(max(lens), 16384)
                self.assertTrue(
                    all(r.bootstrap_host == FAKE_BOOTSTRAP_HOST for r in tm.reqs)
                )
                if fn is serving_shapes:
                    self.assertTrue(any(n % 16 for n in lens[1:]))
        upstream = _run_warmup(prefill_shapes, "prefill", startup=False)
        self.assertEqual(max(len(r.input_ids) for r in upstream.reqs), 32768)

    def test_mix_runs_long_prompt_and_full_waves(self):
        tm = _run_warmup(serving_shapes, "null", max_running=64)
        lens = [len(r.input_ids) for r in tm.reqs]
        self.assertEqual(lens[0], 1)
        self.assertIn(262144, lens)
        # Real prompts are not 16-aligned; Triton specializes on it.
        self.assertTrue(any(n % 16 for n in lens if n > 1))
        # Concurrent 1K waves 1, 2, 4, ..., 64 after the six single prompts.
        self.assertEqual(len(lens), 6 + 127)
        self.assertEqual(tm.max_in_flight, 64)
        self.assertTrue(all(r.bootstrap_host is None for r in tm.reqs))

    def test_decode_waves_cover_every_dp_rank(self):
        tm = _run_warmup(serving_shapes, "decode", max_running=128, dp_size=8)
        self.assertEqual({r.routed_dp_rank for r in tm.reqs}, set(range(8)))
        self.assertTrue(all(r.bootstrap_host == FAKE_BOOTSTRAP_HOST for r in tm.reqs))
        # 128 / 8 = 16 per rank: waves 1, 2, 4, 8, 16 on each of the 8 ranks.
        self.assertEqual(len(tm.reqs), 8 * (1 + 2 + 4 + 8 + 16))


if __name__ == "__main__":
    unittest.main()
