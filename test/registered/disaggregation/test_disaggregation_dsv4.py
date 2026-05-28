import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
)

register_cuda_ci(est_time=1200, stage="base-c", runner_config="dsv4-8-gpu-h200")

DSV4_FLASH_MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"

DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

DSV4_FLASH_ENV = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
}

DSV4_FLASH_HISPARSE_ENV = {
    **DSV4_FLASH_ENV,
    "SGLANG_OPT_USE_COMPRESSOR_V2": "false",
}

HISPARSE_CONFIG = '{"top_k":128,"device_buffer_size":512,"host_to_device_ratio":2}'

_EAGLE_SPEC_ARGS = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "1",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "2",
]


class TestDisaggregationDSV4(PDDisaggregationServerBase, GSM8KMixin):

    gsm8k_accuracy_thres = 0.93

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.model = try_cached_model(DSV4_FLASH_MODEL)

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            4,
            "--dp",
            4,
            "--enable-dp-attention",
            "--moe-a2a-backend",
            "deepep",
            "--deepep-config",
            DEEPEP_CONFIG,
            "--cuda-graph-max-bs",
            "128",
            "--max-running-requests",
            "128",
            *_EAGLE_SPEC_ARGS,
            "--watchdog-timeout",
            "900",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=DSV4_FLASH_ENV,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            4,
            "--dp",
            4,
            "--enable-dp-attention",
            "--base-gpu-id",
            4,
            "--moe-a2a-backend",
            "deepep",
            "--deepep-config",
            DEEPEP_CONFIG,
            "--cuda-graph-max-bs",
            "128",
            "--max-running-requests",
            "128",
            *_EAGLE_SPEC_ARGS,
            "--watchdog-timeout",
            "900",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=DSV4_FLASH_ENV,
        )


class TestDisaggregationDSV4HiSparse(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.model = try_cached_model(DSV4_FLASH_MODEL)

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "4",
            "--page-size",
            "256",
            "--chunked-prefill-size",
            "8192",
            "--max-running-requests",
            "4",
            "--mem-fraction-static",
            "0.75",
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
            "--skip-server-warmup",
            "--watchdog-timeout",
            "900",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=DSV4_FLASH_HISPARSE_ENV,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "4",
            "--base-gpu-id",
            "4",
            "--disable-radix-cache",
            "--page-size",
            "256",
            "--chunked-prefill-size",
            "8192",
            "--max-running-requests",
            "4",
            "--mem-fraction-static",
            "0.75",
            "--cuda-graph-bs",
            "1",
            "2",
            "4",
            "--cuda-graph-max-bs",
            "4",
            "--enable-hisparse",
            "--hisparse-config",
            HISPARSE_CONFIG,
            "--watchdog-timeout",
            "900",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=DSV4_FLASH_HISPARSE_ENV,
        )

    def test_long_prompt_generate(self):
        prompt_len = 4096
        max_new_tokens = 8
        input_ids = [100 + (i % 1000) for i in range(prompt_len)]

        response = requests.post(
            self.lb_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=600,
        )

        self.assertEqual(response.status_code, 200, response.text)
        output = response.json()
        self.assertIn("output_ids", output)
        self.assertEqual(len(output["output_ids"]), max_new_tokens)
        self.assertEqual(output["meta_info"]["prompt_tokens"], prompt_len)

    def test_gsm8k_24_shot(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
            num_shots=24,
        )
        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")

        self.assertGreater(metrics["score"], 0.90)


if __name__ == "__main__":
    unittest.main()
