# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.pdd import (
    project_pdd_head,
    shard_pdd_heads,
    validate_pdd_schedule,
)
from sglang.multimodal_gen.tools import build_minimax_h3_pdd_weights as build
from sglang.multimodal_gen.tools import fuse_minimax_h3_pdd_heads as fusion
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMiniMaxH3PDD(CustomTestCase):
    def heads(self, steps=8):
        return {
            f"{name}.{kind}": torch.randn(shape)
            for name, width in (("video_out", 6), ("audio_out", 4))
            for kind, shape in (("weight", (steps, width, 5)), ("bias", (steps, width)))
        }

    def test_tp_shards_reconstruct_projection(self):
        heads = self.heads()
        x = torch.randn(7, 5)
        for tp in (1, 2):
            shards = [
                shard_pdd_heads(
                    heads,
                    video_width=6 // tp,
                    audio_width=4 // tp,
                    hidden_size=5,
                    tp_size=tp,
                    tp_rank=rank,
                )
                for rank in range(tp)
            ]
            for name in ("video_out", "audio_out"):
                for step in (0, 7):
                    actual = torch.cat(
                        [
                            torch.nn.functional.linear(
                                x,
                                shard[f"{name}.weight"][step],
                                shard[f"{name}.bias"][step],
                            )
                            for shard in shards
                        ],
                        dim=-1,
                    )
                    expected = torch.nn.functional.linear(
                        x, heads[f"{name}.weight"][step], heads[f"{name}.bias"][step]
                    )
                    torch.testing.assert_close(actual, expected)

    def test_projection_keeps_sidecar_on_cpu(self):
        heads = self.heads()
        originals = dict(heads)
        x = torch.randn(7, 5, dtype=torch.float64)
        for name in ("video_out", "audio_out"):
            for step in (0, 7):
                actual = project_pdd_head(heads, x, name, step)
                expected = torch.nn.functional.linear(
                    x,
                    heads[f"{name}.weight"][step].double(),
                    heads[f"{name}.bias"][step].double(),
                )
                torch.testing.assert_close(actual, expected)
                # Meta exercises a device transfer even on CPU-only CI hosts.
                project_pdd_head(heads, x.to("meta"), name, step)
        for key, original in originals.items():
            self.assertIs(heads[key], original)
            self.assertEqual(heads[key].device.type, "cpu")
            self.assertEqual(heads[key].dtype, torch.float32)
        for step in (-1, 8):
            with self.assertRaises(ValueError):
                project_pdd_head(heads, x, "video_out", step)

    def test_invalid_head_banks(self):
        for problem in ("missing", "width", "steps", "nan"):
            heads = self.heads()
            if problem == "missing":
                heads.pop("audio_out.bias")
            elif problem == "width":
                heads["video_out.weight"] = torch.zeros(8, 5, 5)
            elif problem == "steps":
                heads["audio_out.weight"] = torch.zeros(7, 4, 5)
            else:
                heads["video_out.bias"][0, 0] = float("nan")
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                shard_pdd_heads(
                    heads,
                    video_width=3,
                    audio_width=2,
                    hidden_size=5,
                    tp_size=2,
                    tp_rank=0,
                )

    def test_fusion_matches_euler_substeps(self):
        torch.manual_seed(0)
        weight = torch.randn(32, 6, 5).bfloat16()
        bias = torch.randn(32, 6).bfloat16()
        h = torch.randn(7, 5)
        for shift in (3.0, 12.0):
            sigma = fusion.sigma_grid(33, shift)
            fw, fb = fusion.fuse(weight, sigma, 4), fusion.fuse(bias, sigma, 4)
            self.assertEqual(fw.dtype, torch.float32)
            for block in range(8):
                start = block * 4
                x = torch.randn(7, 6)
                expected = x.clone()
                for j in range(start, start + 4):
                    expected -= float(
                        sigma[j] - sigma[j + 1]
                    ) * torch.nn.functional.linear(
                        h, weight[j].float(), bias[j].float()
                    )
                actual = x - float(
                    sigma[start] - sigma[start + 4]
                ) * torch.nn.functional.linear(h, fw[block], fb[block])
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_invalid_fusion_config(self):
        for shift in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fusion.sigma_grid(33, shift)
        for block in (0, 3, 64):
            with self.assertRaises(ValueError):
                fusion.fuse(torch.ones(32, 2), fusion.sigma_grid(33, 3), block)
        with self.assertRaises(ValueError):
            fusion.fuse(torch.ones(32, 2), fusion.sigma_grid(33, 3).flip(0), 4)

    def test_schedule_validation(self):
        sigmas = {
            name: fusion.sigma_grid(9, shift).float().tolist()
            for name, shift in (("video", 12), ("audio", 3))
        }
        metadata = {
            f"{name}_sigmas": json.dumps(values) for name, values in sigmas.items()
        }
        validate_pdd_schedule(8, metadata, sigmas)
        for count in (5, 10):
            wrong = {name: fusion.sigma_grid(count, 3).tolist() for name in sigmas}
            with self.assertRaises(ValueError):
                validate_pdd_schedule(8, metadata, wrong)
        wrong = dict(sigmas, video=fusion.sigma_grid(9, 6).tolist())
        with self.assertRaises(ValueError):
            validate_pdd_schedule(8, metadata, wrong)
        validate_pdd_schedule(
            8,
            metadata,
            {name: values[:3] for name, values in sigmas.items()},
            warmup=True,
        )

    def test_conversion_keeps_heads_outside_transformer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base"
            base.mkdir()
            save_file(
                {
                    "blocks.0.mlp.fc2.weight": torch.zeros(3, 2),
                    "blocks.0.mlp.fc1.weight": torch.zeros(4, 2),
                    "token_refiner.blocks.0.mlp.fc1.weight": torch.zeros(4, 2),
                },
                str(base / "model.safetensors"),
            )
            heads = self.heads(32)
            lora = {
                key.replace("video_out", "proj_out").replace(
                    "audio_out", "audio_proj_out"
                ): value
                for key, value in heads.items()
            }
            lora.update(
                {
                    "transformer_blocks.0.ff.net.2.lora_down": torch.ones(1, 2),
                    "transformer_blocks.0.ff.net.2.lora_up": torch.ones(3, 1),
                }
            )
            for prefix in ("transformer_blocks.0", "token_refiner.refiner_blocks.0"):
                lora[f"{prefix}.ff.net.0.proj.lora_down"] = torch.ones(1, 2)
                lora[f"{prefix}.ff.net.0.proj.lora_up"] = torch.tensor(
                    [[1.0], [2.0], [10.0], [20.0]]
                )
            adapter = root / "adapter.safetensors"
            save_file(lora, str(adapter), metadata={"lora_rank": "1"})
            out = root / "out"
            args = ["build", str(base), str(adapter), str(out)]
            with patch("sys.argv", args):
                self.assertEqual(build.main(), 0)
            self.assertEqual(
                [p.name for p in (out / "transformer").glob("*.safetensors")],
                ["model.safetensors"],
            )
            with safe_open(
                str(out / "transformer/model.safetensors"), framework="pt"
            ) as f:
                torch.testing.assert_close(
                    f.get_tensor("blocks.0.mlp.fc2.weight"), torch.ones(3, 2)
                )
                for prefix in ("blocks.0", "token_refiner.blocks.0"):
                    expected = torch.tensor(
                        [[10.0, 10.0], [20.0, 20.0], [1.0, 1.0], [2.0, 2.0]]
                    )
                    torch.testing.assert_close(
                        f.get_tensor(f"{prefix}.mlp.fc1.weight"), expected
                    )
            with patch("sys.argv", ["fuse", str(out)]):
                self.assertEqual(fusion.main(), 0)
            with safe_open(
                str(out / "pdd_fused_heads.safetensors"), framework="pt"
            ) as f:
                self.assertEqual(f.get_tensor("video_out.weight").shape[0], 8)
                self.assertEqual(len(json.loads(f.metadata()["video_sigmas"])), 9)
            with patch("sys.argv", args), self.assertRaises(SystemExit):
                build.main()


if __name__ == "__main__":
    unittest.main(verbosity=3)
