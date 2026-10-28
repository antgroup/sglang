"""DeepSeek-V4.1 EPD: encode-server preprocessing and language-side prompt rebuild."""

import asyncio
import base64
import concurrent.futures
import io
import json
import os
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from sglang.srt.arg_groups.pd_disaggregation_hook import _dsv41_epd_requested
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE, Modality
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4Model
from sglang.srt.multimodal.deepseek_v41_image_processing import (
    image_token_types,
    patchify_image,
)
from sglang.srt.multimodal.deepseek_v41_mm_inputs import (
    build_epd_mm_output,
    encoder_image_meta,
    image_preprocess_pool,
    iter_vision_weights,
    patch_grid_num_tokens,
    prepare_encoder_images,
    vision_weight_allow_patterns,
)
from sglang.srt.multimodal.encoder_preprocessing import EncoderMediaProcessorConfig
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.runtime_context import get_context, get_flags, get_model, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# The DeepSeek-V4.1-Flash-1017 checkpoint's vision_config, flattened as
# DeepseekV41Config stores it.
HF_CONFIG = SimpleNamespace(
    vision_patch_size=14,
    vision_downsample_ratio=3,
    vision_max_n_token=1024,
    vision_min_pixels=295936,
    vision_max_wh_ratio=None,
    image_token_id=129264,
)
IMAGE_TOKEN = HF_CONFIG.image_token_id

# Below min_pixels, 1080p, a token-budget resize, extreme aspect ratios, sizes
# that are not a multiple of patch_size * downsample_ratio, and the testset's.
IMAGE_SIZES = [
    (224, 224),
    (1920, 1080),
    (4000, 3000),
    (3000, 200),
    (200, 3000),
    (517, 389),
    (160, 120),
    (2400, 360),
    (360, 1440),
    (300, 1200),
    (2048, 1536),
    (1280, 720),
    (1, 1),
    (42, 4096),
    (4096, 42),
    (543, 543),
]


def _image(width, height, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def _prepare(images):
    return prepare_encoder_images(
        images, hf_config=HF_CONFIG, backend="cpu", cpu_patchify=patchify_image
    )


class TestEncoderPreprocessing(CustomTestCase):
    def test_grid_alone_gives_every_span_length(self):
        """P and a metadata-only decode rebuild prompts from the grid without the
        image; the count must equal the local processor's span for that image."""
        images = [_image(w, h, seed) for seed, (w, h) in enumerate(IMAGE_SIZES)]
        output = _prepare(images)
        grids = output["image_grid_thw"].tolist()
        for image, item, (t, n_vit_h, n_vit_w), count in zip(
            images, output.mm_items, grids, output.token_counts
        ):
            self.assertEqual(t, 1)
            self.assertEqual(
                count,
                patch_grid_num_tokens(
                    n_vit_h, n_vit_w, HF_CONFIG.vision_downsample_ratio
                ),
            )
            # The local processor expands one placeholder by these token types.
            _, local_h, local_w, n_llm_h, n_llm_w = patchify_image(image, HF_CONFIG)
            self.assertEqual((n_vit_h, n_vit_w), (local_h, local_w))
            self.assertEqual(count, len(image_token_types(n_llm_h, n_llm_w)))
            self.assertLessEqual(count, HF_CONFIG.vision_max_n_token)
            self.assertEqual(item.feature.shape[0], n_vit_h * n_vit_w)
            self.assertEqual((item.n_vit_h, item.n_vit_w), (n_vit_h, n_vit_w))

    def test_pooled_preprocessing_keeps_image_order(self):
        """The encoder preprocesses a batch's images concurrently; items, grids
        and counts must stay in prompt order, on one torch thread each."""
        images = [_image(w, h, seed) for seed, (w, h) in enumerate(IMAGE_SIZES)]
        pool = image_preprocess_pool(4)
        self.addCleanup(pool.shutdown)
        pooled = prepare_encoder_images(
            images,
            hf_config=HF_CONFIG,
            backend="cpu",
            cpu_patchify=patchify_image,
            executor=pool,
        )
        serial = _prepare(images)
        self.assertEqual(pooled.token_counts, serial.token_counts)
        self.assertTrue(torch.equal(pooled["image_grid_thw"], serial["image_grid_thw"]))
        for a, b in zip(pooled.mm_items, serial.mm_items):
            self.assertTrue(
                torch.equal(torch.as_tensor(a.feature), torch.as_tensor(b.feature))
            )
        self.assertEqual(pool.submit(torch.get_num_threads).result(), 1)

    def _preprocessor(self, model_preprocessor, executor):
        preprocessor = object.__new__(EncoderPreprocessor)
        preprocessor.model_type = "deepseek_v41"
        preprocessor.image_processor = None
        preprocessor.use_image_processor_gpu = False
        preprocessor.vision_config = {"image": {}}
        preprocessor.encoder_media_processor_config = EncoderMediaProcessorConfig()
        preprocessor._model_preprocessor = model_preprocessor
        preprocessor.io_executor = executor
        preprocessor.preproc_executor = executor
        return preprocessor

    @staticmethod
    def _urls(sizes):
        urls = []
        for width, height in sizes:
            buffer = io.BytesIO()
            _image(width, height).save(buffer, format="PNG")
            urls.append(
                "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
            )
        return urls

    def test_encode_server_reports_model_token_counts(self):
        """The encode server must slice V4.1 spans by the model's row counts."""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        preprocessor = self._preprocessor(
            lambda images, modality, config: _prepare(images), executor
        )
        try:
            result = asyncio.run(
                preprocessor.process_mm_items(
                    self._urls(IMAGE_SIZES[:3]), Modality.IMAGE
                )
            )
        finally:
            executor.shutdown()

        expected = [
            patch_grid_num_tokens(h, w, HF_CONFIG.vision_downsample_ratio)
            for _, h, w in result.grid_thw.tolist()
        ]
        self.assertEqual(result.token_counts, expected)
        # The generic prod(grid) / merge**2 rule would size every span wrongly.
        generic = [
            preprocessor.get_num_tokens(g, Modality.IMAGE) for g in result.grid_thw
        ]
        self.assertTrue(all(a != b for a, b in zip(generic, expected)))


class TestHeaderMeta(CustomTestCase):
    def test_header_meta_matches_encode_preprocessing(self):
        """/preprocess_meta answers decode from the image header; its grids and
        token counts must equal what /encode computes from the decoded image."""
        rng = random.Random(0)
        sizes = [
            (1920, 1080),
            (1080, 1920),
            (160, 120),
            (2400, 360),
            (300, 1200),
            (14, 14),
            (5000, 40),
            (40, 5000),
            (2048, 1536),
            (543, 544),
        ]
        sizes += [(rng.randint(8, 3000), rng.randint(8, 3000)) for _ in range(30)]
        for i, size in enumerate(sizes):
            fmt, mode = [("JPEG", "RGB"), ("PNG", "RGB"), ("PNG", "L"), ("PNG", "P")][
                i % 4
            ]
            buffer = io.BytesIO()
            Image.new(mode, size, color=(30, 60, 90) if mode == "RGB" else 128).save(
                buffer, fmt
            )
            data = buffer.getvalue()
            # As the encode server loads it: decode, then convert to RGB.
            with Image.open(io.BytesIO(data)) as decoded:
                full = _prepare([decoded.convert("RGB")])
            meta = encoder_image_meta([data], hf_config=HF_CONFIG)
            self.assertEqual(
                meta["image_grid_thw"].tolist(),
                full["image_grid_thw"].tolist(),
                (size, fmt, mode),
            )
            self.assertEqual(meta.token_counts, full.token_counts, (size, fmt, mode))


class TestLanguageSideRebuild(CustomTestCase):
    GRIDS = torch.tensor([[1, 40, 70], [1, 22, 22]])

    def _counts(self):
        return [
            patch_grid_num_tokens(h, w, HF_CONFIG.vision_downsample_ratio)
            for _, h, w in self.GRIDS.tolist()
        ]

    def _build(self, ids, embeddings, grids=None):
        return build_epd_mm_output(
            ids,
            embeddings=embeddings,
            img_grid_thw=self.GRIDS if grids is None else grids,
            image_token_id=IMAGE_TOKEN,
            downsample_ratio=HF_CONFIG.vision_downsample_ratio,
        )

    def test_spans_land_on_their_placeholders_in_order(self):
        ids = [0, 11, IMAGE_TOKEN, 12, 13, IMAGE_TOKEN, 14]
        counts = self._counts()
        rows = torch.arange(sum(counts), dtype=torch.float32)[:, None].repeat(1, 4)
        embeddings = {Modality.IMAGE: rows}
        output = self._build(ids, embeddings)

        self.assertEqual(len(output.input_ids), len(ids) - 2 + sum(counts))
        consumed = 0
        for item, count in zip(output.mm_items, counts):
            ((start, end),) = item.offsets
            self.assertEqual(end - start + 1, count)
            self.assertEqual(set(output.input_ids[start : end + 1]), {IMAGE_TOKEN})
            self.assertEqual(int(item.precomputed_embeddings[0, 0]), consumed)
            consumed += count
        text = [t for t in output.input_ids if t != IMAGE_TOKEN]
        self.assertEqual(text, [0, 11, 12, 13, 14])
        # The receiver's own layout check must accept the rebuild.
        BaseMultimodalProcessor._validate_precomputed_embedding_layout(
            output, embeddings
        )

    def test_metadata_only_rebuild_matches_embedding_rebuild(self):
        """An EPD decode worker gets no rows yet must match P's token layout."""
        ids = [5, IMAGE_TOKEN, IMAGE_TOKEN, 6]
        rows = torch.zeros(sum(self._counts()), 4)
        with_rows = self._build(ids, {Modality.IMAGE: rows})
        meta_only = self._build(ids, None)
        self.assertEqual(meta_only.input_ids, with_rows.input_ids)
        self.assertEqual(
            [item.offsets for item in meta_only.mm_items],
            [item.offsets for item in with_rows.mm_items],
        )
        self.assertTrue(
            all(item.precomputed_embeddings is None for item in meta_only.mm_items)
        )

    def test_placeholder_and_grid_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "placeholders and images must match"):
            self._build([1, IMAGE_TOKEN, 2], None)
        with self.assertRaisesRegex(ValueError, "image grids"):
            build_epd_mm_output(
                [1, IMAGE_TOKEN],
                embeddings=None,
                img_grid_thw=None,
                image_token_id=IMAGE_TOKEN,
                downsample_ratio=3,
            )


class TestEncoderOnlyWeights(CustomTestCase):
    def setUp(self):
        # _prepare_weights reads the published config; the defaults skip checksums.
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def test_checkpoint_vision_names_map_to_modules(self):
        # Tensor names from the DeepSeek-V4.1-Flash-1017 safetensors headers.
        checkpoint = {
            "vision.blocks.0.attn.wqkv.weight": "vision.blocks.0.attn.qkv_proj.weight",
            "vision.blocks.0.attn.wqkv.bias": "vision.blocks.0.attn.qkv_proj.bias",
            "vision.blocks.0.attn.wo.weight": "vision.blocks.0.attn.proj.weight",
            "vision.blocks.0.attn.wo.bias": "vision.blocks.0.attn.proj.bias",
            "vision.blocks.0.mlp.w1.weight": "vision.blocks.0.mlp.w1.weight",
            "vision.blocks.0.norm1.weight": "vision.blocks.0.norm1.weight",
            "vision.patch_embed.proj.weight": "vision.patch_embed.proj.weight",
            "vision.norm.weight": "vision.norm.weight",
            "aligner.w1.weight": "aligner.w1.weight",
            "image_newline": "image_newline",
        }
        language = ["embed.weight", "layers.0.attn.wq_a.weight", "mtp.0.e_proj.weight"]
        stream = [(name, torch.zeros(1)) for name in [*language, *checkpoint]]
        mapped = [name for name, _ in iter_vision_weights(stream)]
        self.assertEqual(mapped, list(checkpoint.values()))
        # The language model's loader applies the same mapping.
        for name, module_name in checkpoint.items():
            self.assertEqual(
                DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(name),
                module_name,
            )

    def _write_index(self, root, weight_map):
        with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": weight_map}, f)

    def _loaded_shards(self, root):
        loader = DefaultModelLoader(LoadConfig(load_format=LoadFormat.SAFETENSORS))
        _, files, use_safetensors = loader._prepare_weights(
            root,
            None,
            False,
            allow_patterns_overrides=vision_weight_allow_patterns(root),
        )
        self.assertTrue(use_safetensors)
        return sorted(os.path.basename(f) for f in files)

    def test_encoder_only_loader_reads_only_vision_shards(self):
        """An encode server reads the vision shards, not the 476 GB language model."""
        shards = [f"model-{i:05d}-of-00048.safetensors" for i in range(1, 49)]
        weight_map = {f"layers.{i}.ffn.w1.weight": s for i, s in enumerate(shards)}
        weight_map.update(
            {
                "vision.blocks.0.attn.wqkv.weight": shards[0],
                "aligner.w1.weight": shards[0],
                "image_start": shards[1],
            }
        )
        with tempfile.TemporaryDirectory() as root:
            for shard in shards:
                open(os.path.join(root, shard), "w").close()
            # No index: every shard is read.
            self.assertIsNone(vision_weight_allow_patterns(root))
            self._write_index(root, weight_map)
            self.assertEqual(self._loaded_shards(root), shards[:2])

            # Shards 12 and 21: the glob may over-match but never misses one.
            weight_map["vision.norm.weight"] = shards[11]
            weight_map["image_end"] = shards[20]
            self._write_index(root, weight_map)
            loaded = self._loaded_shards(root)
            for index in (0, 1, 11, 20):
                self.assertIn(shards[index], loaded)
            self.assertLess(len(loaded), len(shards))


class TestEpdRoles(CustomTestCase):
    """Which V4.1 modules each EPD role builds, and that EPD needs the switch."""

    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def _config(self, **roles):
        config = DeepSeekV4Config(
            architectures=["DeepseekV4ForCausalLM"],
            model_type="deepseek_v41",
            vision_n_layers=1,
            hidden_size=4,
            tie_word_embeddings=True,
            quantization_config={},
            rope_scaling={},
            compress_ratios=[],
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            engram_layer_ids=[],
            engram_num_embeddings=[],
        )
        for name, value in roles.items():
            setattr(config, name, value)
        return config

    def _build(self, config, *, epd_on=True, cp=1):
        backbone = nn.Module()
        backbone.embed_tokens = nn.Identity()
        backbone.start_layer = 0
        backbone.end_layer = 0
        with (
            # The whole topology, groups included: an earlier test in the
            # process may have built real (world size 1) groups.
            get_parallel().override(
                tp_size=4,
                tp_rank=0,
                attn_tp_size=4 // cp,
                attn_tp_rank=0,
                attn_cp_size=cp,
                attn_cp_rank=0,
                attn_dp_size=1,
                attn_dp_rank=0,
                moe_tp_size=4,
                moe_ep_size=1,
                moe_dp_size=1,
                pp_size=1,
                tp_group=SimpleNamespace(world_size=4),
                attn_tp_group=SimpleNamespace(world_size=4 // cp),
                attn_cp_group=SimpleNamespace(world_size=cp),
                moe_ep_group=SimpleNamespace(world_size=1),
                pp_group=SimpleNamespace(world_size=1, is_last_rank=True),
            ),
            get_flags().dsv41_sm90.override(
                subsystems=frozenset({"epd"} if epd_on else ())
            ),
            get_model().override(
                shared_experts=SharedExpertsConfig(fusion_disabled=True),
                model_path="/nonexistent",
            ),
            patch.object(deepseek_v4, "ViT", return_value=nn.Identity()),
            patch.object(deepseek_v4, "Aligner", return_value=nn.Identity()),
            patch.object(deepseek_v4, "DeepseekV4Model", return_value=backbone),
            patch.object(deepseek_v4, "LogitsProcessor", return_value=nn.Identity()),
            patch.object(deepseek_v4, "get_attn_tp_context"),
            # On CUDA the real check loads the published model config from disk.
            patch.object(deepseek_v4, "is_dsa_enable_prefill_cp", return_value=cp > 1),
        ):
            return DeepseekV4ForCausalLM(config)

    def test_language_only_takes_images_without_a_tower(self):
        """A CP prefill must build without the tower and still embed encoder
        spans, remap their hash ids and keep the visual routing bias."""
        model = self._build(self._config(language_only=True), cp=4)
        self.assertIsNone(model.vision)
        self.assertTrue(model.takes_image_inputs)
        self.assertFalse(
            any(n.startswith("image_") for n, _ in model.named_parameters())
        )
        # The same CP layout still rejects a tower.
        with self.assertRaisesRegex(ValueError, "V4.1 vision"):
            self._build(self._config(), cp=4)

    def test_cp_prefill_embeds_encoder_spans_before_sharding(self):
        """The CP runner embeds token ids itself when the model gives no shard:
        an image batch would reach the layers as pad-hash embeddings."""
        model = self._build(self._config(language_only=True), cp=4)
        model.cp_embed_scatter = False
        spans = torch.arange(16.0).view(8, 2)
        forward_mode = SimpleNamespace(
            is_decode=lambda: False, is_target_verify=lambda: False
        )
        batch = SimpleNamespace(forward_mode=forward_mode, mm_inputs=[object()])
        ids = torch.zeros(8, dtype=torch.long)
        with (
            patch.object(model, "_prepare_mm_embeddings", return_value=spans),
            patch.object(
                deepseek_v4,
                "cp_shard_hidden_states",
                side_effect=lambda full, _batch: full[::4],
            ),
        ):
            shard = model.cp_shard_input_embeds(ids, batch)
        self.assertTrue(torch.equal(shard, spans[::4]))
        # A text batch keeps the runner's own embedding.
        batch.mm_inputs = [None]
        self.assertIsNone(model.cp_shard_input_embeds(ids, batch))
        self.assertEqual(model.model.cp_image_token_id, IMAGE_TOKEN)

    def test_cp_body_routes_image_tokens_by_image_token_id(self):
        """Under CP the body gets the scheduler's ids; Engram and the visual
        MoE routing must see image_token_id where the pad hashes are."""
        ids = torch.tensor([5, MM_PAD_SHIFT_VALUE + 7, 9])
        expected = torch.tensor([5, IMAGE_TOKEN, 9])
        for token_id, cp_active, want in (
            (IMAGE_TOKEN, True, expected),
            (IMAGE_TOKEN, False, ids),
            (None, True, ids),
        ):
            with self.subTest(token_id=token_id, cp_active=cp_active):
                body = SimpleNamespace(cp_image_token_id=token_id)
                with patch.object(deepseek_v4, "is_cp_active", return_value=cp_active):
                    out = DeepseekV4Model.cp_vocab_ids(body, ids, None)
                self.assertTrue(torch.equal(out, want))

    def test_encoder_only_builds_the_tower_alone(self):
        model = self._build(self._config(encoder_only=True))
        self.assertIsInstance(model.vision, nn.Module)
        self.assertEqual(model.image_start.shape, (4,))
        self.assertFalse(hasattr(model, "model"))

    def test_epd_roles_need_the_switch(self):
        """With the SM90 set empty (switch off, another GPU) V4.1 EPD stays rejected."""
        for role in ("encoder_only", "language_only"):
            with self.subTest(role=role):
                with self.assertRaisesRegex(ValueError, "SGLANG_OPT_DSV41_SM90"):
                    self._build(self._config(**{role: True}), epd_on=False)
        # Non-EPD serving does not depend on the switch.
        self.assertIsInstance(
            self._build(self._config(), epd_on=False).vision, nn.Module
        )

    def test_whitelist_admits_v41_only_with_the_switch(self):
        v41, v4 = (
            SimpleNamespace(model_type=t) for t in ("deepseek_v41", "deepseek_v4")
        )
        with envs.SGLANG_OPT_DSV41_SM90.override(False):
            self.assertFalse(_dsv41_epd_requested(v41))
        with envs.SGLANG_OPT_DSV41_SM90.override(True):
            self.assertTrue(_dsv41_epd_requested(v41))
            self.assertFalse(_dsv41_epd_requested(v4))
            with envs.SGLANG_OPT_DSV41_SM90_DISABLE.override("epd"):
                self.assertFalse(_dsv41_epd_requested(v41))


if __name__ == "__main__":
    unittest.main()
