# SPDX-License-Identifier: Apache-2.0
import copy
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_SIGMAS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.pdd import (
    PDD_CONFIG_METADATA_KEY,
    PDDConfig,
    fuse_configured_heads,
    legacy_pdd_config,
    load_pdd_adapter,
    pdd_config_from_metadata,
    project_pdd_head,
    shard_pdd_heads,
    validate_pdd_schedule,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)
from sglang.multimodal_gen.tools import build_minimax_h3_pdd_weights as build
from sglang.multimodal_gen.tools import fuse_minimax_h3_pdd_heads as fusion
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMiniMaxH3PDD(CustomTestCase):
    def custom_config(self):
        return {
            "format_version": 1,
            "prediction_type": "velocity",
            "update_rule": "euler_eta0",
            "shared_backbone_features": True,
            "modalities": {
                name: {
                    "interval_sigmas": sigma,
                    "groups": [[0, 1], [1, 3], [3, 4]],
                    "head_keys": {
                        "weight": f"custom.{name}.w",
                        "bias": f"custom.{name}.b",
                    },
                }
                for name, sigma in (
                    ("video", [1.0, 0.91, 0.6, 0.2, 0.0]),
                    ("audio", [1.0, 0.8, 0.7, 0.4, 0.0]),
                )
            },
        }

    def custom_heads(self, config):
        return {
            key: torch.randn(shape)
            for name, spec in config.modalities.items()
            for key, shape in (
                (
                    spec.weight_key,
                    (len(spec.interval_sigmas) - 1, 6 if name == "video" else 4, 5),
                ),
                (
                    spec.bias_key,
                    (len(spec.interval_sigmas) - 1, 6 if name == "video" else 4),
                ),
            )
        }

    def test_config_roundtrip_and_invalid_protocols(self):
        data = self.custom_config()
        config = PDDConfig.from_dict(data)
        self.assertEqual(config.to_dict(), data)
        self.assertEqual(config.nfe, 3)
        for key, value in (
            ("format_version", 2),
            ("format_version", True),
            ("prediction_type", "epsilon"),
            ("update_rule", "ancestral"),
            ("shared_backbone_features", False),
        ):
            invalid = copy.deepcopy(data)
            invalid[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                PDDConfig.from_dict(invalid)
        for field in data:
            invalid = copy.deepcopy(data)
            del invalid[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                PDDConfig.from_dict(invalid)
        for field, value in (
            ("groups", [[0, 1], [2, 4]]),
            ("groups", [[0, 2], [1, 4]]),
            ("groups", [[0, 3]]),
            ("groups", [[0, True], [1, 4]]),
            ("interval_sigmas", [1.0, 0.6, 0.7, 0.2, 0.0]),
            ("interval_sigmas", [1.0, 0.9, float("nan"), 0.2, 0.0]),
        ):
            invalid = copy.deepcopy(data)
            invalid["modalities"]["video"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                PDDConfig.from_dict(invalid)
        invalid = copy.deepcopy(data)
        invalid["modalities"]["audio"]["groups"] = [[0, 4]]
        with self.assertRaises(ValueError):
            PDDConfig.from_dict(invalid)

    def test_irregular_fusion_matches_sequential_euler(self):
        config = PDDConfig.from_dict(self.custom_config())
        heads = self.custom_heads(config)
        fused = fuse_configured_heads(heads, config)
        h = torch.randn(7, 5)
        for name, spec in config.modalities.items():
            for step, (start, stop) in enumerate(spec.groups):
                x = torch.randn(7, heads[spec.bias_key].shape[1])
                expected = x.clone()
                for j in range(start, stop):
                    expected -= (
                        spec.interval_sigmas[j] - spec.interval_sigmas[j + 1]
                    ) * torch.nn.functional.linear(
                        h, heads[spec.weight_key][j], heads[spec.bias_key][j]
                    )
                actual = x - (
                    spec.interval_sigmas[start] - spec.interval_sigmas[stop]
                ) * project_pdd_head(fused, h, f"{name}_out", step)
                torch.testing.assert_close(actual, expected)
        missing = dict(heads)
        missing.pop(config.modalities["video"].bias_key)
        with self.assertRaises(ValueError):
            fuse_configured_heads(missing, config)

    def test_versioned_schedule_roundtrip(self):
        config = PDDConfig.from_dict(self.custom_config()).with_canonical_keys()
        metadata = {PDD_CONFIG_METADATA_KEY: json.dumps(config.to_dict())}
        loaded = pdd_config_from_metadata(metadata)
        self.assertEqual(loaded, config)
        sigmas = loaded.request_sigmas(4)
        self.assertEqual(sigmas["video"], [1.0, 0.91, 0.2, 0.0])
        validate_pdd_schedule(3, metadata, sigmas)
        validate_pdd_schedule(
            3, metadata, loaded.request_sigmas(1, warmup=True), warmup=True
        )
        with self.assertRaises(ValueError):
            loaded.request_sigmas(5)
        with self.assertRaises(ValueError):
            validate_pdd_schedule(
                3, metadata, {name: [1.0, 0.8, 0.2, 0.0] for name in sigmas}
            )

    def test_timestep_stage_uses_configured_boundaries(self):
        config = PDDConfig.from_dict(self.custom_config())
        stage = SimpleNamespace(pdd_config=config, sigma_shift_scales=None)
        plan = SimpleNamespace(flow_shift=None, audio_flow_shift=None)
        for count, warmup in ((4, False), (1, True)):
            batch = SimpleNamespace(
                extra={}, num_inference_steps=count, is_warmup=warmup
            )
            MiniMaxH3TimestepPreparationStage._generate_sigmas_from_plan(
                stage, batch, plan
            )
            self.assertEqual(
                batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY],
                config.request_sigmas(count, warmup=warmup),
            )
        plan.flow_shift = 12.0
        with self.assertRaises(ValueError):
            MiniMaxH3TimestepPreparationStage._generate_sigmas_from_plan(
                stage,
                SimpleNamespace(extra={}, num_inference_steps=4, is_warmup=False),
                plan,
            )

    def test_online_adapter_loads_original_file(self):
        for versioned in (False, True):
            config = (
                PDDConfig.from_dict(self.custom_config())
                if versioned
                else legacy_pdd_config(32, 4)
            )
            tensors = self.custom_heads(config)
            tensors["transformer_blocks.0.attn.to_q.lora_down"] = torch.randn(2, 5)
            metadata = {"lora_alpha": "64.0"}
            if versioned:
                metadata[PDD_CONFIG_METADATA_KEY] = json.dumps(config.to_dict())
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "adapter.safetensors"
                save_file(tensors, str(path), metadata=metadata)
                loaded, heads, alpha = load_pdd_adapter(str(path))
                self.assertEqual(alpha, 64)
                self.assertEqual(loaded.to_dict(), config.to_dict())
                expected = fuse_configured_heads(tensors, config)
                for key in expected:
                    torch.testing.assert_close(heads[key], expected[key])
                    self.assertEqual(heads[key].device.type, "cpu")
                self.assertEqual(
                    [p.name for p in Path(temp).iterdir()], ["adapter.safetensors"]
                )

    def test_online_adapter_ordinary_and_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "adapter.safetensors"
            save_file({"layer.lora_A": torch.zeros(2, 5)}, str(path))
            self.assertIsNone(load_pdd_adapter(str(path)))
            save_file({"proj_out.weight": torch.zeros(32, 6, 5)}, str(path))
            with self.assertRaises(ValueError):
                load_pdd_adapter(str(path))

    def test_lora_wrapper_does_not_accept_prequantized_input(self):
        from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
            _accepts_mxfp8_input,
        )

        wrapper = torch.nn.Module()
        wrapper.base_layer = torch.nn.Linear(5, 6)
        self.assertFalse(_accepts_mxfp8_input(wrapper))
        native = torch.nn.Linear(5, 6)
        native.quant_method = SimpleNamespace(accepts_mxfp8_input=lambda layer: True)
        self.assertTrue(_accepts_mxfp8_input(native))

    def test_pdd_startup_requires_unfiltered_backbone(self):
        from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
            MiniMaxH3Pipeline,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
            LoRAPipeline,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
            pdd,
        )
        from sglang.multimodal_gen.runtime.utils import hf_diffusers_utils

        config = PDDConfig.from_dict(self.custom_config())
        heads = fuse_configured_heads(self.custom_heads(config), config)
        for targets in (None, [], ["attn"]):
            args = SimpleNamespace(
                lora_path="adapter.safetensors",
                lora_weight_name=None,
                lora_scale=1.0,
                lora_target_modules=targets,
                lora_alpha=None,
            )
            final_layer = SimpleNamespace(_pdd_heads=None, install_pdd_heads=Mock())
            model = SimpleNamespace(final_layer=final_layer)
            pipeline = SimpleNamespace(get_module=Mock(return_value=model))
            with (
                self.subTest(targets=targets),
                patch.object(
                    hf_diffusers_utils,
                    "maybe_download_lora",
                    return_value="adapter.safetensors",
                ),
                patch.object(pdd, "load_pdd_adapter", return_value=(config, heads, 64)),
            ):
                if targets is None:
                    self.assertTrue(
                        LoRAPipeline.is_target_layer(args, "blocks.0.attn.qkv_proj")
                    )
                    MiniMaxH3Pipeline._prepare_startup_pdd(pipeline, args, None)
                    final_layer.install_pdd_heads.assert_called_once_with(heads, config)
                    self.assertEqual(pipeline._pdd_config, config.with_canonical_keys())
                else:
                    if not targets:
                        self.assertFalse(
                            LoRAPipeline.is_target_layer(args, "blocks.0.attn.qkv_proj")
                        )
                    with self.assertRaisesRegex(ValueError, "complete backbone"):
                        MiniMaxH3Pipeline._prepare_startup_pdd(pipeline, args, None)
                    pipeline.get_module.assert_not_called()
                    final_layer.install_pdd_heads.assert_not_called()

    def metadata_test_model(self, *, h3=False, config=None):
        from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
            MiniMaxH3DiTModel,
        )

        model = torch.nn.Linear(5, 6)
        if h3:
            model.validate_lora_metadata = MethodType(
                MiniMaxH3DiTModel.validate_lora_metadata, model
            )
            model.prepare_lora_state_dict = MethodType(
                MiniMaxH3DiTModel.prepare_lora_state_dict, model
            )
            model.final_layer = SimpleNamespace(_pdd_heads=None)
        if config is not None:
            model._pdd_adapter_config = config
            model._pdd_adapter_head_keys = {
                key
                for spec in config.modalities.values()
                for key in (spec.weight_key, spec.bias_key)
            }
            model.final_layer._pdd_heads = {"installed": torch.empty(0)}
        return model

    def metadata_test_pipeline(self, path, model):
        arch = SimpleNamespace(
            param_names_mapping={r"^(.*)$": r"\1"},
            lora_param_names_mapping={r"^(.*)$": r"\1"},
        )
        return SimpleNamespace(
            server_args=SimpleNamespace(
                lora_path=str(path),
                lora_weight_name=None,
                pipeline_config=SimpleNamespace(
                    dit_config=SimpleNamespace(arch_config=arch)
                ),
            ),
            modules={"transformer": model},
            device=torch.device("cpu"),
            lora_adapters=defaultdict(dict),
            loaded_adapter_paths={},
            loaded_adapter_alphas={},
        )

    def load_metadata_test_adapter(self, state, path):
        from sglang.multimodal_gen.runtime.pipelines_core.lora import (
            pipeline as lora_pipeline,
        )

        with (
            patch.object(lora_pipeline, "maybe_download_lora", return_value=str(path)),
            patch.object(lora_pipeline.dist, "is_initialized", return_value=False),
        ):
            lora_pipeline.LoRAPipeline.load_lora_adapter(
                state, str(path), "test", rank=0
            )

    def test_metadata_hook_preserves_ordinary_lora_and_old_hook_signature(self):
        from sglang.multimodal_gen.runtime.pipelines_core.lora import (
            pipeline as lora_pipeline,
        )

        tensors = {
            "layer.lora_A.weight": torch.randn(2, 5),
            "layer.lora_B.weight": torch.randn(6, 2),
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ordinary.safetensors"
            save_file(tensors, str(path), metadata={"description": "ordinary adapter"})
            for kind in ("no_hook", "old_single_argument_hook", "h3"):
                model = self.metadata_test_model(h3=kind == "h3")
                if kind == "old_single_argument_hook":
                    model.prepare_lora_state_dict = Mock(
                        side_effect=lambda state: state
                    )
                state = self.metadata_test_pipeline(path, model)
                with (
                    self.subTest(kind=kind),
                    patch.object(
                        lora_pipeline, "safe_open", wraps=safe_open
                    ) as header_reader,
                ):
                    self.load_metadata_test_adapter(state, path)
                    if kind != "h3":
                        header_reader.assert_not_called()
                for key, tensor in tensors.items():
                    torch.testing.assert_close(
                        state.lora_adapters["test"][key.removesuffix(".weight")], tensor
                    )
                if kind == "old_single_argument_hook":
                    self.assertEqual(
                        len(model.prepare_lora_state_dict.call_args.args), 1
                    )
                    self.assertEqual(model.prepare_lora_state_dict.call_args.kwargs, {})

    def test_custom_pdd_runtime_rejected_before_loading_tensors(self):
        from sglang.multimodal_gen.runtime.pipelines_core.lora import (
            pipeline as lora_pipeline,
        )

        config = PDDConfig.from_dict(self.custom_config())
        tensors = self.custom_heads(config)
        tensors.update(
            {
                "layer.lora_A.weight": torch.randn(2, 5),
                "layer.lora_B.weight": torch.randn(6, 2),
            }
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "custom_pdd.safetensors"
            save_file(
                tensors,
                str(path),
                metadata={PDD_CONFIG_METADATA_KEY: json.dumps(config.to_dict())},
            )
            model = self.metadata_test_model(h3=True)
            original_weight = model.weight.detach().clone()
            state = self.metadata_test_pipeline(path, model)
            existing = {"sentinel": torch.ones(1)}
            state.lora_adapters["test"] = existing
            with (
                patch.object(lora_pipeline, "load_file") as tensor_reader,
                self.assertRaisesRegex(ValueError, "server startup"),
            ):
                self.load_metadata_test_adapter(state, path)
            tensor_reader.assert_not_called()
            self.assertIs(state.lora_adapters["test"], existing)
            torch.testing.assert_close(model.weight, original_weight)

    def test_versioned_startup_metadata_must_match_installed_heads(self):
        config = PDDConfig.from_dict(self.custom_config())
        tensors = self.custom_heads(config)
        a, b = torch.randn(2, 5), torch.randn(6, 2)
        tensors.update({"layer.lora_A.weight": a, "layer.lora_B.weight": b})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "custom_pdd.safetensors"
            save_file(
                tensors,
                str(path),
                metadata={PDD_CONFIG_METADATA_KEY: json.dumps(config.to_dict())},
            )
            model = self.metadata_test_model(h3=True, config=config)
            state = self.metadata_test_pipeline(path, model)
            self.load_metadata_test_adapter(state, path)
            self.assertEqual(
                set(state.lora_adapters["test"]), {"layer.lora_A", "layer.lora_B"}
            )
            torch.testing.assert_close(state.lora_adapters["test"]["layer.lora_A"], a)
            changed = config.to_dict()
            changed["modalities"]["video"]["interval_sigmas"][1] = 0.92
            model._pdd_adapter_config = PDDConfig.from_dict(changed)
            with self.assertRaisesRegex(ValueError, "differs"):
                self.load_metadata_test_adapter(state, path)
            model._pdd_adapter_config = config
            model.final_layer._pdd_heads = None
            with self.assertRaisesRegex(ValueError, "server startup"):
                self.load_metadata_test_adapter(state, path)

    def test_legacy_pdd_runtime_is_still_rejected(self):
        config = legacy_pdd_config(32, 4)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy_pdd.safetensors"
            save_file(self.custom_heads(config), str(path))
            state = self.metadata_test_pipeline(path, self.metadata_test_model(h3=True))
            with self.assertRaisesRegex(ValueError, "server startup"):
                self.load_metadata_test_adapter(state, path)

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
