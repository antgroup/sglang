# SPDX-License-Identifier: Apache-2.0
"""High-value task, partition, and public request admission contracts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    LAYERWISE_OFFLOAD,
    RESIDENT,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3PartitionAdmissionStage,
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_resolve_plan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    partition_for_task,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)

TARGET = {
    "short_edge": 768,
    "aspect_ratio": "16:9",
    "duration_seconds": 5.0,
}


@pytest.mark.parametrize(
    ("task", "conditions", "partition", "visual", "audio", "chains"),
    [
        ("t2va", [], "fl2va", [], [], []),
        (
            "fl2va",
            [
                {
                    "type": "image",
                    "uri": "file:///first.png",
                    "role": "keyframe",
                    "frame_index": 0,
                },
                {
                    "type": "image",
                    "uri": "file:///last.png",
                    "role": "keyframe",
                    "frame_index": -1,
                },
            ],
            "fl2va",
            [0, 1],
            [],
            ["image.target_canvas", "image.target_canvas"],
        ),
        (
            "ref2va",
            [
                {
                    "type": "image",
                    "uri": "file:///image.png",
                    "role": "reference",
                },
                {
                    "type": "video",
                    "uri": "file:///video.mp4",
                    "role": "reference",
                    "start_time_seconds": 12.5,
                },
                {
                    "type": "audio",
                    "uri": "file:///audio.wav",
                    "role": "reference",
                },
                {
                    "type": "video_audio",
                    "uri": "file:///av.mp4",
                    "role": "reference",
                },
            ],
            "ref2va",
            [0, 1, 3],
            [1, 2, 3],
            [
                "image.reference_preserve",
                "video.reference_preserve",
                "audio",
                "video_audio.reference_preserve",
            ],
        ),
    ],
)
def test_public_tasks_resolve_to_exact_partition_and_encoder_plan(
    task, conditions, partition, visual, audio, chains
):
    canonical = minimax_h3_validate_canonical_request(
        task=task,
        prompt="contract",
        conditions=conditions,
        target=TARGET,
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)

    assert partition_for_task(task) == partition
    assert plan.task == task
    assert plan.encoders["visual"] == visual
    assert plan.encoders["audio"] == audio
    assert [material.material_chain for material in plan.materials] == chains
    if task == "ref2va":
        assert plan.materials[1].start_time_seconds == 12.5
    assert plan.shape["frame_count"] == 124
    assert plan.shape["video_latent_t"] == 37


@pytest.mark.parametrize(
    ("partition", "tasks"),
    [("fl2va", ["t2va", "fl2va"]), ("ref2va", ["ref2va"])],
)
def test_loaded_weight_partition_admits_only_its_declared_tasks(partition, tasks):
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(
        {
            "_minimax_h3": {
                "schema_version": 1,
                "partition": partition,
                "tasks": tasks,
                "task_aliases": {},
                "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
            }
        }
    )

    assert [metadata.canonical_task(task) for task in tasks] == tasks
    rejected = "ref2va" if partition == "fl2va" else "t2va"
    with pytest.raises(ValueError):
        metadata.canonical_task(rejected)


def test_duration_admission_accepts_released_4_to_15_second_range():
    for duration in (4.0, 15.0):
        target = {**TARGET, "duration_seconds": duration}
        canonical = minimax_h3_validate_canonical_request(
            task="t2va",
            prompt="duration contract",
            conditions=[],
            target=target,
            seed=0,
        )
        assert canonical["target"]["duration_seconds"] == duration

    for duration in (3.9, 15.1):
        target = {**TARGET, "duration_seconds": duration}
        with pytest.raises(ValueError, match=r"\[4, 15\]"):
            minimax_h3_validate_canonical_request(
                task="t2va",
                prompt="duration contract",
                conditions=[],
                target=target,
                seed=0,
            )


def test_video_adapter_lowers_only_native_fields_and_rejects_cfg():
    request = VideoGenerationsRequest(
        prompt="contract",
        task="t2va",
        conditions=[],
        target=TARGET,
        flow_shift=8.0,
        audio_flow_shift=2.0,
        quality="high",
        imgvid_cond_noise_aug_for_inference=0.75,
        audio_cond_noise_aug_for_inference=0.5,
    )
    generic = {
        "prompt": request.prompt,
        "seed": request.seed,
        "flow_shift": request.flow_shift,
    }

    lowered = MiniMaxH3SamplingParams.lower_video_request_kwargs(request, generic)
    assert lowered == {
        "prompt": "contract",
        "seed": request.seed,
        "task": "t2va",
        "conditions": [],
        "target": TARGET,
        "flow_shift": 8.0,
        "audio_flow_shift": 2.0,
        "quality": "high",
        "imgvid_cond_noise_aug_for_inference": 0.75,
        "audio_cond_noise_aug_for_inference": 0.5,
    }

    with pytest.raises(ValueError):
        MiniMaxH3SamplingParams.lower_video_request_kwargs(
            request, {**generic, "guidance_scale": 7.5}
        )


@pytest.mark.parametrize("bad_quality", ["ultra", "draft", "", 1])
def test_video_adapter_rejects_invalid_quality(bad_quality):
    request = VideoGenerationsRequest(
        prompt="contract",
        task="t2va",
        conditions=[],
        target=TARGET,
        quality=bad_quality,
    )
    with pytest.raises(ValueError, match="quality must be one of"):
        MiniMaxH3SamplingParams.lower_video_request_kwargs(
            request, {"prompt": request.prompt, "seed": request.seed}
        )


class _Capability:
    def __init__(self, major, minor):
        self.major = major
        self.minor = minor

    def as_version_str(self):
        return f"{self.major}.{self.minor}"


def _quality_server_args(**overrides):
    values = {
        "enable_breakable_cuda_graph": False,
        "enable_torch_compile": False,
        "use_fsdp_inference": False,
        "is_dit_layerwise_offload_selected": False,
        "ring_degree": 1,
        "pipeline_config": MiniMaxH3PipelineConfig(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _high_quality_cache_dit_stage():
    stage = MiniMaxH3DenoisingStage.__new__(MiniMaxH3DenoisingStage)
    stage.server_args = _quality_server_args()
    stage._cache_dit_enabled = False
    stage._minimax_h3_quality = "lossless"
    stage._minimax_h3_subblock_enabled = False
    return stage


def _high_quality_cache_dit_batch(enable_cache_dit=None, quality="high"):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(
            quality=quality,
            _explicit_fields={"quality"},
            enable_cache_dit=enable_cache_dit,
            cache_dit_params=None,
            attention_backend_override=None,
        )
    )


@pytest.mark.parametrize(
    ("server_flag", "message"),
    [
        ("enable_breakable_cuda_graph", "breakable CUDA graphs"),
        ("use_fsdp_inference", "FSDP-managed transformer"),
        ("is_dit_layerwise_offload_selected", "DiT layerwise offload"),
    ],
)
def test_high_quality_rejects_incompatible_cache_dit_modes(server_flag, message):
    server_args = _quality_server_args(**{server_flag: True})

    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        pytest.raises(ValueError, match=message),
    ):
        server_args.pipeline_config.validate_quality_deployment(server_args)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((9, 0), True),
        ((10, 0), True),
        ((8, 0), False),
        ((10, 3), False),
        ((12, 0), False),
    ],
)
def test_high_quality_subblock_support_is_capability_based(capability, expected):
    stage = _high_quality_cache_dit_stage()

    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(
            current_platform,
            "get_device_capability",
            return_value=_Capability(*capability),
        ),
    ):
        supported = stage._supports_high_quality_subblock()

    assert supported is expected


def test_high_quality_subblock_switches_at_batch_boundaries():
    calls = []
    transformer = SimpleNamespace(
        set_high_quality_subblock=lambda enabled, schedule=None: calls.append(
            (enabled, schedule)
        )
    )
    stage = _high_quality_cache_dit_stage()
    stage.transformer = transformer
    stage._attention_backend_active_override = None

    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(
            current_platform,
            "get_device_capability",
            return_value=_Capability(9, 0),
        ),
    ):
        stage._maybe_override_attention_backend(_high_quality_cache_dit_batch())
        stage._maybe_override_attention_backend(
            _high_quality_cache_dit_batch(quality="lossless")
        )

    assert calls[0][0] is True
    assert calls[0][1].sparsity == 0.75
    assert calls[0][1].n_k == calls[0][1].n_q == 4
    assert calls[0][1].skip_first_steps == 10
    assert calls[1] == (False, None)


def test_quality_admission_keeps_support_checks_without_gpu_model_allowlist():
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(
        {
            "_minimax_h3": {
                "schema_version": 1,
                "partition": "fl2va",
                "tasks": ["t2va", "fl2va"],
                "task_aliases": {},
                "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
            }
        }
    )
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="quality",
        conditions=[],
        target=TARGET,
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(
            task="t2va", quality="high", enable_cache_dit=None
        ),
        num_inference_steps=50,
        is_warmup=False,
    )
    stage = MiniMaxH3PartitionAdmissionStage(metadata)
    server_args = _quality_server_args()

    server_args.enable_breakable_cuda_graph = True
    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
            "minimax_h3.release_metadata.minimax_h3_plan_from_batch"
        ) as plan_from_batch,
        pytest.raises(ValueError, match="breakable CUDA graphs"),
    ):
        stage.forward(batch, server_args)
    plan_from_batch.assert_not_called()

    server_args.enable_breakable_cuda_graph = False
    batch.sampling_params.enable_cache_dit = False
    with (
        patch.object(current_platform, "is_cuda", return_value=False),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
            "minimax_h3.release_metadata.minimax_h3_plan_from_batch",
            return_value=plan,
        ),
    ):
        assert stage.forward(batch, server_args) is batch

    batch.sampling_params.enable_cache_dit = None
    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(current_platform, "get_device_name") as get_device_name,
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
            "minimax_h3.release_metadata.minimax_h3_plan_from_batch",
            return_value=plan,
        ),
    ):
        assert stage.forward(batch, server_args) is batch
        batch.num_inference_steps = 40
        with pytest.raises(ValueError, match="validated only"):
            stage.forward(batch, server_args)
    get_device_name.assert_not_called()

    batch.sampling_params.quality = "lossless"
    batch.num_inference_steps = 50
    assert stage.forward(batch, server_args) is batch

    batch.sampling_params.quality = "ultra"
    with pytest.raises(ValueError, match="quality must be one of"):
        stage.forward(batch, server_args)


def test_validate_server_args_requires_packed_varlen_backend():
    config = SimpleNamespace(
        vae_config=SimpleNamespace(resolved_parallel_decode_mode=lambda: None),
        dit_config=SimpleNamespace(arch_config=SimpleNamespace(attention_head_dim=128)),
        _server_arg_value=MiniMaxH3PipelineConfig._server_arg_value,
    )
    server_args = SimpleNamespace(
        component_attention_backends={}, attention_backend="sage_attn"
    )
    with patch(
        "sglang.multimodal_gen.configs.pipeline_configs.minimax_h3.get_attn_backend"
    ) as get_attn_backend:
        MiniMaxH3PipelineConfig.validate_server_args(config, server_args)
    get_attn_backend.assert_called_once_with(
        128,
        torch.bfloat16,
        selected_attention_backend=AttentionBackendEnum.SAGE_ATTN,
        attention_requirements=AttentionRequirements(packed_varlen=True),
    )
    with patch(
        "sglang.multimodal_gen.configs.pipeline_configs.minimax_h3.get_attn_backend",
        side_effect=ValueError("does not implement packed varlen attention"),
    ):
        with pytest.raises(ValueError, match="does not implement packed varlen"):
            MiniMaxH3PipelineConfig.validate_server_args(config, server_args)


def test_mps_admission_requires_layerwise_residency_for_every_h3_component():
    config = SimpleNamespace(
        vae_config=SimpleNamespace(resolved_parallel_decode_mode=lambda: None),
        dit_config=SimpleNamespace(arch_config=SimpleNamespace(attention_head_dim=128)),
        _server_arg_value=MiniMaxH3PipelineConfig._server_arg_value,
    )
    modes = {
        "transformer": LAYERWISE_OFFLOAD,
        "text_encoder": LAYERWISE_OFFLOAD,
        "video_vae": LAYERWISE_OFFLOAD,
        "audio_vae": LAYERWISE_OFFLOAD,
    }
    server_args = SimpleNamespace(
        component_attention_backends={},
        attention_backend=None,
        enable_torch_compile=False,
        residency_mode=modes.get,
    )

    with patch.object(current_platform, "is_mps", return_value=True):
        MiniMaxH3PipelineConfig.validate_server_args(config, server_args)

        modes["audio_vae"] = RESIDENT
        with pytest.raises(ValueError, match="audio_vae"):
            MiniMaxH3PipelineConfig.validate_server_args(config, server_args)
