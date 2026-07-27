# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
    LingBotWorldCausalDMDConfig,
    LingBotWorldLegacyCompatCausalDMDConfig,
    LingBotWorldLegacyCompatI2VConfig,
)
from sglang.multimodal_gen.configs.sample.lingbot_world import (
    LingBotWorldSamplingParams,
)
from sglang.multimodal_gen.registry import _get_config_info


def test_legacy_lingbot_configs_preserve_pre_mainline_defaults():
    i2v_config = LingBotWorldLegacyCompatI2VConfig()
    causal_config = LingBotWorldLegacyCompatCausalDMDConfig()

    assert i2v_config.flow_shift == 5.0
    assert causal_config.flow_shift == 5.0
    assert causal_config.lazy_vae_encode_black_frames == 60
    assert causal_config.interactive_kv_window_enable is True
    assert causal_config.kv_cache_reset_enable is True
    assert causal_config.kv_cache_reset_max_window_latent_frames == 88
    assert causal_config.kv_cache_reset_keep_prev_chunks == 1
    assert causal_config.kv_cache_reset_rope_gap_latent_frames == -1


def test_lingbot_sampling_params_keep_legacy_extras():
    params = LingBotWorldSamplingParams(actions=[["w"]], chunk_size=3)

    extra = params.build_request_extra()

    assert extra["actions"] == [["w"]]
    assert extra["chunk_size"] == 3


def test_local_lingbot_deployment_resolves_legacy_compat_config():
    _get_config_info.cache_clear()

    local_info = _get_config_info("/home/admin/lingbot-world-fast-diffusers")
    main_info = _get_config_info("IPostYellow/lingbot-world-fast-diffusers")

    assert local_info is not None
    assert local_info.pipeline_config_cls is LingBotWorldLegacyCompatCausalDMDConfig
    assert main_info is not None
    assert main_info.pipeline_config_cls is LingBotWorldCausalDMDConfig


def test_lingbot_condition_mask_preserves_batch_dimension():
    config = LingBotWorldLegacyCompatCausalDMDConfig()
    vae_arch = config.vae_config.arch_config
    latent = torch.zeros(2, 16, 3, 2, 2)
    batch = SimpleNamespace(
        height=2 * int(vae_arch.spatial_compression_ratio),
        width=2 * int(vae_arch.spatial_compression_ratio),
        num_frames=2 * int(vae_arch.temporal_compression_ratio) + 1,
    )

    condition = config.postprocess_image_latent(latent, batch)

    assert condition.shape[0] == 2
    assert condition.shape[2:] == latent.shape[2:]
