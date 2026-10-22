# SPDX-License-Identifier: Apache-2.0
"""Validation shared by the launcher and LingBot's causal pipeline."""

from sglang.multimodal_gen.runtime.realtime.causal_pipeline import (
    CausalPipelinePlacement,
    CausalPipelinePlan,
)


def placement_from_args(args) -> CausalPipelinePlacement:
    return CausalPipelinePlacement(
        args.causal_pipeline_parallel_size,
        args.causal_vae_placement,
        args.causal_vae_rank,
    )


def configure_causal_pipeline_parallelism(args) -> None:
    placement = placement_from_args(args)
    if args.num_gpus != placement.world_size:
        raise ValueError(
            f"--num-gpus must be {placement.world_size}: "
            f"{placement.denoising_ranks} denoising ranks with "
            f"{placement.vae_placement} VAE placement"
        )
    for name in (
        "tp_size",
        "sp_degree",
        "ulysses_degree",
        "ring_degree",
        "kv_gather_degree",
    ):
        if getattr(args, name) not in (None, 1):
            raise ValueError(f"Causal pipeline does not yet compose with {name}")
        setattr(args, name, 1)
    if (
        args.dp_size != 1
        or args.enable_cfg_parallel
        or args.cfg_parallel_degree not in (None, 1)
    ):
        raise ValueError("Causal pipeline requires DP=1 and CFG parallelism disabled")
    args.enable_cfg_parallel = False
    args.cfg_parallel_degree = 1
    if args.encoder_parallel not in ("auto", "replicate"):
        raise ValueError("Causal pipeline requires replicated condition encoders")
    args.encoder_parallel = "replicate"


def validate_causal_pipeline_args(args) -> None:
    if args.causal_pipeline_mode == "off":
        if (
            args.causal_pipeline_parallel_size != 1
            or args.causal_vae_placement != "shared"
            or args.causal_vae_rank is not None
        ):
            raise ValueError(
                "Causal pipeline placement requires --causal-pipeline-mode"
            )
        return
    if not args.pipeline_config.supports_causal_pipeline():
        raise ValueError("Causal pipeline currently supports only LingBot causal DMD")
    from sglang.multimodal_gen import envs
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if not current_platform.is_cuda():
        raise ValueError("Experimental causal pipeline currently requires NVIDIA CUDA")
    if envs.SGLANG_CACHE_DIT_ENABLED:
        raise ValueError("Causal pipeline does not support cache-dit")
    placement_from_args(args)
    CausalPipelinePlan(
        len(args.pipeline_config.dmd_denoising_steps),
        args.pipeline_config.dit_config.num_layers,
        args.causal_pipeline_parallel_size,
        args.causal_pipeline_mode,
    )
    if not 1 <= args.causal_pipeline_window <= 32:
        raise ValueError("--causal-pipeline-window must be between 1 and 32")
    if args.nnodes != 1 or args.disagg_role.value != "monolithic":
        raise ValueError(
            "Causal pipeline initially requires a single-node monolithic launch"
        )
    if args.performance_mode != "manual":
        raise ValueError(
            "Experimental causal pipeline requires --performance-mode manual"
        )
    if args.batching_max_size != 1:
        raise ValueError(
            "Causal pipeline uses chunk windows, not dynamic request batching"
        )
    if args.lora_path is not None:
        raise ValueError(
            "Merge any LoRA into the checkpoint before using the causal pipeline"
        )
    if (
        args.use_fsdp_inference
        or args.has_layerwise_offload_components()
        or args.enable_torch_compile
        or args.enable_breakable_cuda_graph
        or args.comfyui_mode
        or args.kv_cache_quant_config.enabled
    ):
        raise ValueError(
            "Causal pipeline requires eager execution without FSDP, layerwise offload, "
            "ComfyUI mode or KV quantization"
        )
    if args.pipeline_config.interactive_kv_window_enable:
        raise ValueError("Causal pipeline requires a fixed KV window")
