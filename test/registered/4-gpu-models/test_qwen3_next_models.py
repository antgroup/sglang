import shutil
import tempfile
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_decode_cache_hit_helper,
    test_input_output_logprobs_match_prefill_cache_hit_helper,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=350, suite="stage-c-test-4-gpu-h100")

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
QWEN3_NEXT_8B_MODEL = "/sgl-workspace/mnt/Qwen3.5-9B"
ACC_THRESHOLDS = {
    QWEN3_NEXT_MODEL: {"kl_div": 0.0025, "gsm8k": 0.7},
    QWEN3_NEXT_8B_MODEL: {"kl_div": 0.0025, "gsm8k": 0.7},
}


def flush_cache(base_url, retries: int = 60, retry_interval: float = 1.0) -> bool:
    """Flush device cache to force remote storage access."""
    import time

    for _ in range(retries):
        try:
            response = requests.post(f"{base_url}/flush_cache", timeout=10)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            return False
        time.sleep(retry_interval)
    return False


def send_request_helper(base_url: str, text: str):
    response = requests.post(
        base_url + "/generate",
        json={
            "text": text,
            "sampling_params": {
                "max_new_tokens": 1,
            },
        },
    )
    return response.json()


class TestQwen3Next(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_NEXT_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--chunked-prefill-size",
                "2048",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                "128",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(
            metrics["accuracy"], ACC_THRESHOLDS[self.model]["gsm8k"]
        )

    def test_input_output_logprobs_match_prefill_cache_hit(self):
        test_input_output_logprobs_match_prefill_cache_hit_helper(
            self.base_url,
            ACC_THRESHOLDS,
            self.model,
            max_samples=32,
            max_new_tokens=512,
        )

    def test_input_output_logprobs_match_decode_cache_hit(self):
        test_input_output_logprobs_match_decode_cache_hit_helper(
            self.base_url,
            ACC_THRESHOLDS,
            self.model,
            max_samples=32,
            max_new_tokens=512,
        )

    def test_prefix_cache_branching(self):
        print("running test_prefix_cache_branching")
        requests.get(self.base_url + "/flush_cache")
        branching_pos = 257
        text_prefix = "hi" * branching_pos
        suffix_list = ["this" * 256, "here" * 256, "that" * 256]
        cache_hit_list = [False, False, True]

        # First request only prefill the entire sequence
        # Second request won't have cache hit, but will cache the branching point
        # Third request will have cache hit on the branching point
        for i, (suffix, cache_hit) in enumerate(
            zip(suffix_list, cache_hit_list, strict=True)
        ):
            result = send_request_helper(self.base_url, text_prefix + suffix)
            cached_tokens = result["meta_info"]["cached_tokens"]
            if cache_hit:
                expected_cached_tokens = branching_pos // 64 * 64
                assert (
                    cached_tokens == expected_cached_tokens
                ), f"{i=}, {cache_hit=}, {cached_tokens=} is not equal to {expected_cached_tokens=}, {branching_pos=}"
            else:
                assert (
                    cached_tokens == 0
                ), f"{i=}, {cache_hit=}, {cached_tokens=} is not 0"
        print("test_prefix_cache_branching passed")


class TestQwen3NextWithHiCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_NEXT_8B_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.storage_dir = tempfile.mkdtemp(prefix="qwen3-next-hicache-")
        env = {
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.storage_dir,
        }
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=env,
            other_args=[
                "--tp",
                "4",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--page-size",
                "64",
                "--max-total-tokens",
                "131072",
                "--max-mamba-cache-size",
                "600",
                "--hicache-mem-layout",
                "page_first_direct",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "2",
                "--hicache-size",
                "0",
                "--hicache-write-policy",
                "write_through",
                "--hicache-storage-backend",
                "file",
                "--hicache-storage-prefetch-policy",
                "wait_complete",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        shutil.rmtree(cls.storage_dir, ignore_errors=True)

    def _run_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=100,
            max_new_tokens=512,
            parallel=10,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        return run_eval(args)

    def test_gsm8k(self):
        first_metrics = self._run_gsm8k()
        print(f"first_metrics={first_metrics}")
        self.assertGreaterEqual(
            first_metrics["accuracy"], ACC_THRESHOLDS[self.model]["gsm8k"]
        )

        print(f"flush cache")
        assert flush_cache(self.base_url)

        second_metrics = self._run_gsm8k()
        print(f"second_metrics={second_metrics}")
        self.assertGreaterEqual(
            second_metrics["accuracy"], ACC_THRESHOLDS[self.model]["gsm8k"]
        )
        self.assertLessEqual(
            abs(second_metrics["accuracy"] - first_metrics["accuracy"]),
            0.05,
            f"HiCache prefetch accuracy drift too large: "
            f"first={first_metrics['accuracy']}, second={second_metrics['accuracy']}",
        )


if __name__ == "__main__":
    unittest.main()
