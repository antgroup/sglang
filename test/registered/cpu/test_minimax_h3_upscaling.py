# SPDX-License-Identifier: Apache-2.0
"""H3 post-generation super-resolution admission and delivery contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import materialize_output_sample
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    video_adapter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMiniMaxH3Upscaling(CustomTestCase):
    def test_http_and_offline_accept_upscaling(self):
        adapter = video_adapter.MiniMaxH3VideoModelAdapter()
        for scale in (1, 2, 4):
            with self.subTest(scale=scale):
                options = dict(
                    enable_upscaling=True,
                    upscaling_scale=scale,
                    upscaling_model_path="/models/realesrgan.pth",
                )
                request = VideoGenerationsRequest(
                    prompt="A landscape", task="t2va", **options
                )
                lowered = adapter.lower_video_request_kwargs(request, options)
                for key, value in options.items():
                    self.assertEqual(lowered[key], value)
                params = MiniMaxH3SamplingParams(
                    task="t2va", save_output=True, output_path="outputs", **options
                )
                adapter.validate_sampling_params(params)
                self.assertEqual(params.fps, 24)

    def test_invalid_scale_and_interpolation_are_rejected(self):
        for scale in (0, -1, 1.5, True, None):
            with (
                self.subTest(scale=scale),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                MiniMaxH3SamplingParams(enable_upscaling=True, upscaling_scale=scale)
        with self.assertRaisesRegex(ValueError, "enable_frame_interpolation"):
            MiniMaxH3SamplingParams(
                enable_upscaling=True, enable_frame_interpolation=True
            )

    def test_delivery_dimensions_follow_scale_without_changing_generation_shape(self):
        adapter = video_adapter.MiniMaxH3VideoModelAdapter()
        shape = dict(width=896, height=512, frame_count=120, fps=24)
        for enabled, scale, expected in (
            (False, 4, (896, 512)),
            (True, 2, (1792, 1024)),
            (True, 4, (3584, 2048)),
        ):
            params = MiniMaxH3SamplingParams(
                enable_upscaling=enabled, upscaling_scale=scale
            )
            batch = SimpleNamespace(sampling_params=params, num_outputs_per_prompt=1)
            with (
                self.subTest(enabled=enabled, scale=scale),
                patch.object(adapter, "_resolved_shape", return_value=shape),
                patch.object(
                    video_adapter, "_probe_minimax_h3_output_fields", return_value={}
                ) as probe,
            ):
                fields = adapter.project_queued_job_fields(batch)
                self.assertEqual(fields["size"], f"{expected[0]}x{expected[1]}")
                self.assertEqual(fields["seconds"], "5")
                adapter.validate_final_outputs_sync(["output.mp4"], batch)
                probe.assert_called_once_with(
                    "output.mp4", expected_frame_count=120, expected_size=expected
                )
                self.assertEqual((shape["width"], shape["height"]), (896, 512))

    def test_upscaling_preserves_audio_fps_and_frame_count(self):
        video = torch.zeros(3, 5, 8, 12)
        audio = np.zeros((1000, 2), dtype=np.float32)

        def upscale(frames, *, model_path, scale):
            self.assertEqual(model_path, "/models/realesrgan.pth")
            return [
                frame.repeat(scale, axis=0).repeat(scale, axis=1) for frame in frames
            ]

        with patch(
            "sglang.multimodal_gen.runtime.postprocess.upscale_frames",
            side_effect=upscale,
        ) as sr:
            result = materialize_output_sample(
                (video, audio),
                DataType.VIDEO,
                fps=24,
                enable_upscaling=True,
                upscaling_model_path="/models/realesrgan.pth",
                upscaling_scale=2,
            )
        sr.assert_called_once()
        self.assertIs(result.audio, audio)
        self.assertEqual(result.fps, 24)
        self.assertEqual(len(result.frames), 5)
        self.assertEqual(result.frames[0].shape, (16, 24, 3))


if __name__ == "__main__":
    unittest.main()
