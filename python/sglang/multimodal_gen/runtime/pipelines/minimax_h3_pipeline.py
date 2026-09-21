# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    MiniMaxH3AudioEncodingStage,
    MiniMaxH3DecodingStage,
    MiniMaxH3DenoisingStage,
    MiniMaxH3LatentPreparationStage,
    MiniMaxH3TextEncodingStage,
    MiniMaxH3TimestepPreparationStage,
    MiniMaxH3VisualEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3PartitionAdmissionStage,
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_TASK_PARTITIONS,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class MiniMaxH3Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MiniMaxH3Pipeline"
    default_model_subfolder = "FL2VA"
    is_video_pipeline = True
    pipeline_config_cls = MiniMaxH3PipelineConfig
    sampling_params_cls = MiniMaxH3SamplingParams
    _required_config_modules = [
        "processor",
        "text_encoder",
        "tokenizer",
        "video_vae",
        "audio_vae",
        # scheduler intentionally absent: model_index carries scheduler=null;
        # per-modality sigma schedules are generated in TimestepPreparation
        # from the task profile, and the loop scheduler math lives in
        # scheduling_minimax_h3_euler_ancestral (stages accept scheduler=None).
        "transformer",
    ]

    def __init__(self, *args, **kwargs):
        # TODO: Enable this check on ROCm after adding ffmpeg to the AMD Docker
        # image and CI dependency installer.
        if not current_platform.is_rocm():
            missing_media_tools = [
                executable
                for executable in ("ffmpeg", "ffprobe")
                if shutil.which(executable) is None
            ]
            if missing_media_tools:
                raise RuntimeError(
                    "MiniMax H3 requires ffmpeg and ffprobe for media processing "
                    "and validated output delivery; missing executables: "
                    f"{', '.join(missing_media_tools)}. Install the ffmpeg system "
                    "package before starting SGLang."
                )
        super().__init__(*args, **kwargs)
        self._pdd_startup_complete = True

    def _check_pdd_runtime_mutation(self) -> None:
        if (
            getattr(self, "_pdd_startup_complete", False)
            and getattr(self, "_pdd_config", None) is not None
        ):
            raise ValueError(
                "PDD is loaded at startup; restart with the desired --lora-path to change or disable it"
            )

    def set_lora(self, *args: Any, **kwargs: Any) -> None:
        self._check_pdd_runtime_mutation()
        return super().set_lora(*args, **kwargs)

    def deactivate_lora_weights(self, *args: Any, **kwargs: Any) -> None:
        self._check_pdd_runtime_mutation()
        return super().deactivate_lora_weights(*args, **kwargs)

    def merge_lora_weights(self, *args: Any, **kwargs: Any) -> None:
        self._check_pdd_runtime_mutation()
        return super().merge_lora_weights(*args, **kwargs)

    def unmerge_lora_weights(self, *args: Any, **kwargs: Any) -> None:
        self._check_pdd_runtime_mutation()
        return super().unmerge_lora_weights(*args, **kwargs)

    def _prepare_startup_pdd(
        self, server_args: ServerArgs, scales: Mapping[str, float] | None
    ) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.pdd import (
            load_pdd_adapter,
        )

        self._pdd_config = None
        path = getattr(server_args, "lora_path", None)
        if not path:
            return
        from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
            maybe_download_lora,
        )

        path = maybe_download_lora(
            path, weight_name=getattr(server_args, "lora_weight_name", None)
        )
        bank = load_pdd_adapter(
            path,
            video_shift=float((scales or {}).get("video", 12.0)),
            audio_shift=float((scales or {}).get("audio", 3.0)),
        )
        if bank is None:
            return
        config, heads, alpha = bank
        if server_args.lora_scale != 1.0 or server_args.lora_target_modules is not None:
            raise ValueError(
                "PDD requires the complete backbone adapter at --lora-scale 1"
            )
        model = self.get_module("transformer")
        for _ in range(4):
            if hasattr(model, "final_layer"):
                break
            model = getattr(model, "_orig_mod", None) or getattr(model, "module", None)
            if model is None:
                raise ValueError("Cannot find H3 model to install PDD heads")
        if getattr(model.final_layer, "_pdd_heads", None) is not None:
            raise ValueError("Do not combine --lora-path PDD with offline PDD heads")
        model.final_layer.install_pdd_heads(heads, config)
        model._pdd_adapter_config = config
        model._pdd_adapter_head_keys = {
            key
            for spec in config.modalities.values()
            for key in (spec.weight_key, spec.bias_key)
        }
        if alpha is not None:
            if server_args.lora_alpha is not None and server_args.lora_alpha != alpha:
                raise ValueError(
                    "PDD LoRA alpha override disagrees with the checkpoint"
                )
            server_args.lora_alpha = alpha
        self._pdd_config = config.with_canonical_keys()

    @staticmethod
    def model_subfolder_for_variant(variant: str) -> str:
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError("MiniMax H3 model variant must be a non-empty string")
        normalized = variant.strip().lower()
        subfolders = {
            "fl2va": "FL2VA",
            "ref2va": "Ref2VA",
            "hybrid": "Ref2VA",
        }
        try:
            return subfolders[normalized]
        except KeyError as exc:
            raise ValueError(
                f"unsupported MiniMax H3 model variant {variant!r}; "
                f"supported: {sorted(subfolders)!r}"
            ) from exc

    def _load_config(self):
        model_variant = self.server_args.model_variant
        if model_variant is not None:
            semantic_subfolder = self.model_subfolder_for_variant(model_variant)
            if model_variant.strip().lower() == "hybrid" and not (
                self.server_args.component_weights_paths.get("transformer")
                or self.server_args.transformer_weights_path
            ):
                raise ValueError(
                    "MiniMax H3 --model-variant hybrid requires explicit merged "
                    "weights via --component-weights-paths.transformer"
                )
            explicit_subfolder = self.server_args.model_subfolder
            if (
                explicit_subfolder is not None
                and explicit_subfolder.strip().lower() != semantic_subfolder.lower()
            ):
                raise ValueError(
                    "MiniMax H3 --model-variant and --model-subfolder select "
                    f"different weight partitions: variant={model_variant!r} maps to "
                    f"{semantic_subfolder!r}, model_subfolder="
                    f"{explicit_subfolder!r}"
                )
            self.server_args.model_subfolder = semantic_subfolder
        model_index = super()._load_config()
        self.release_metadata = MiniMaxH3ReleaseMetadata.from_model_index(model_index)
        if (
            model_variant is not None
            and self.release_metadata.partition != semantic_subfolder.lower()
        ):
            raise ValueError(
                "MiniMax H3 loaded checkpoint partition does not match "
                f"--model-variant {model_variant!r}"
            )
        if model_variant is not None and model_variant.strip().lower() == "hybrid":
            # merged checkpoints share the native graph across all three tasks
            # keep the base partition contract strict unless explicitly selected
            self.release_metadata = replace(
                self.release_metadata,
                partition="hybrid",
                tasks=tuple(MINIMAX_H3_TASK_PARTITIONS),
            )
        return model_index

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "MiniMaxH3Pipeline only supports monolithic deployment; "
                f"disaggregation role {role.value!r} is not supported"
            )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        # Per-model sigma override from model_index.json; contract tests
        # construct the pipeline without model_path, hence the guard.
        release_metadata = getattr(self, "release_metadata", None)
        sigma_shift_scales = (
            release_metadata.sigma_shift_scales
            if release_metadata is not None
            else None
        )
        self._prepare_startup_pdd(server_args, sigma_shift_scales)
        self.add_stage(InputValidationStage())
        if release_metadata is not None:
            self.add_stage(MiniMaxH3PartitionAdmissionStage(release_metadata))
        self.add_stage(
            MiniMaxH3TextEncodingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
            )
        )
        self.add_stage(
            MiniMaxH3VisualEncodingStage(
                video_vae=self.get_module("video_vae"),
                vae_arch_config=server_args.pipeline_config.vae_config.arch_config,
            )
        )
        self.add_stage(
            MiniMaxH3AudioEncodingStage(
                audio_vae=self.get_module("audio_vae"),
                vae_arch_config=server_args.pipeline_config.audio_vae_config.arch_config,
            )
        )
        self.add_stage(MiniMaxH3LatentPreparationStage())
        self.add_stage(
            MiniMaxH3TimestepPreparationStage(
                sigma_shift_scales=sigma_shift_scales,
                pdd_config=self._pdd_config,
            )
        )
        self.add_stage(
            MiniMaxH3DenoisingStage(
                transformer=self.get_module("transformer"),
                pipeline=self,
            )
        )
        self.add_stage(
            MiniMaxH3DecodingStage(
                video_vae=self.get_module("video_vae"),
                audio_vae=self.get_module("audio_vae"),
            )
        )


class FastH3Pipeline(MiniMaxH3Pipeline):
    """FastH3: 4-step DMD2-distilled MiniMax-H3 (t2va only).

    The flat single-partition repo is materialized into the base-H3 layout by
    the bundled model overlay (see model_overlays/), so every stage, loader,
    and admission path below is exactly the MiniMax-H3 one. There is no
    FL2VA/Ref2VA partition layout to default into.
    """

    pipeline_name = "FastH3Pipeline"
    default_model_subfolder = None


EntryClass = [MiniMaxH3Pipeline, FastH3Pipeline]
