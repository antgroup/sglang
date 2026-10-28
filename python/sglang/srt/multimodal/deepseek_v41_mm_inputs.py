"""DeepSeek-V4.1 image inputs shared by the local processor and EPD.

An encode server (``--encoder-only``) and a language-only server must reproduce
the local processor's items and prompt layout. Each image travels as its ViT
patch grid ``(1, n_vit_h, n_vit_w)``: the encoder returns one span of
``num_image_tokens`` rows per image, and the language side derives the same row
count from the grid alone, with or without the embedding attached.
"""

import functools
import glob
import io
import json
import logging
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, Optional

import msgspec
import torch
from PIL import Image

from sglang.srt.environ import envs
from sglang.srt.layers.dsv41_sm90 import dsv41_sm90_on
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.deepseek_v41_image_processing import (
    GPU_PLAN_KEY,
    num_image_tokens,
    patchify_image,
    patchify_image_rust,
    plan_image_grid,
    prepare_image_gpu,
)
from sglang.srt.multimodal.encoder_preprocessing import EncoderPreprocessOutput
from sglang.srt.runtime_context import get_mm
from sglang.srt.rust_extensions import load_rust_extension

logger = logging.getLogger(__name__)

# Checkpoint tensors of the vision tower, the aligner and the image marker rows.
VISION_WEIGHT_PREFIXES = ("vision.", "aligner.", "image_")


class ImageFeatures(msgspec.Struct, frozen=True, kw_only=True):
    """One image as the V4.1 ViT consumes it."""

    # bf16 [n_vit_h * n_vit_w, 3, p, p] patches; uint8 [3, H, W] pixels when
    # gpu_plan is set (the model resizes and patchifies on its device).
    feature: Any
    n_vit_h: int
    n_vit_w: int
    num_tokens: int
    gpu_plan: Optional[dict] = None

    def model_specific_data(self) -> dict:
        metadata = {"n_vit_h": self.n_vit_h, "n_vit_w": self.n_vit_w}
        if self.gpu_plan is not None:
            metadata[GPU_PLAN_KEY] = self.gpu_plan
        return metadata


def resolve_image_backend() -> tuple[str, Callable]:
    """Return (backend, cpu_patchify); every server that preprocesses must agree."""
    mm = get_mm()
    if mm.image_processor_backend == "pil" or mm.disable_fast_image_processor:
        return "cpu", patchify_image
    if envs.SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU.get():
        return "gpu", patchify_image
    if mm.image_processor_backend != "auto":
        return "cpu", patchify_image
    # A load failure falls back to PIL; later image errors are not retried.
    try:
        extension = load_rust_extension("sglang.srt.rust_extensions._multimodal")
        resize_patchify = extension.dsv41.resize_patchify
    except (ImportError, OSError, RuntimeError, AttributeError) as error:
        fallback_logger = (
            logger
            if dsv41_sm90_on("epd")
            else logging.getLogger("sglang.srt.multimodal.processors.deepseek_v41")
        )
        fallback_logger.warning(
            "V4.1 Rust image processor unavailable; using PIL: %s", error
        )
        return "cpu", patchify_image
    return "rust", functools.partial(
        patchify_image_rust, resize_patchify=resize_patchify
    )


def load_image_features(
    image, *, hf_config, backend: str, cpu_patchify: Callable
) -> ImageFeatures:
    if backend == "gpu":
        pixels, plan, n_llm_h, n_llm_w = prepare_image_gpu(image, hf_config)
        return ImageFeatures(
            feature=pixels,
            n_vit_h=plan["height"] // plan["patch_size"],
            n_vit_w=plan["width"] // plan["patch_size"],
            num_tokens=num_image_tokens(n_llm_h, n_llm_w),
            gpu_plan=plan,
        )
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = cpu_patchify(image, hf_config)
    return ImageFeatures(
        feature=patches,
        n_vit_h=n_vit_h,
        n_vit_w=n_vit_w,
        num_tokens=num_image_tokens(n_llm_h, n_llm_w),
    )


def patch_grid_num_tokens(n_vit_h: int, n_vit_w: int, downsample_ratio: int) -> int:
    """Rows of one image span; the aligner pads a partial cell to a full one."""
    return num_image_tokens(
        -(-n_vit_h // downsample_ratio), -(-n_vit_w // downsample_ratio)
    )


def expand_image_placeholders(
    ids: Sequence[int], *, image_token_id: int, num_tokens: Sequence[int]
) -> tuple[list[int], list[tuple[int, int]]]:
    """Expand each placeholder to its span; returns the ids and inclusive offsets."""
    if list(ids).count(image_token_id) != len(num_tokens):
        raise ValueError("Image placeholders and images must match")
    tokens, offsets = [], []
    counts = iter(num_tokens)
    for token in ids:
        if token != image_token_id:
            tokens.append(token)
            continue
        count = next(counts)
        offsets.append((len(tokens), len(tokens) + count - 1))
        tokens.extend([image_token_id] * count)
    return tokens, offsets


def build_epd_mm_output(
    ids: Sequence[int],
    *,
    embeddings: Optional[dict],
    img_grid_thw,
    image_token_id: int,
    downsample_ratio: int,
) -> MultimodalProcessorOutput:
    """Rebuild a prompt from encoder patch grids.

    ``embeddings`` maps modality to the encoder's concatenated spans. None is
    the EPD decode worker's metadata-only rebuild: it needs the expanded ids,
    never the image rows.
    """
    if img_grid_thw is None:
        raise ValueError("DeepSeek-V4.1 EPD inputs need the encoder's image grids")
    grids = torch.as_tensor(img_grid_thw).reshape(-1, 3).tolist()
    tokens, offsets = expand_image_placeholders(
        ids,
        image_token_id=image_token_id,
        num_tokens=[patch_grid_num_tokens(h, w, downsample_ratio) for _, h, w in grids],
    )
    image_embeddings = None if embeddings is None else embeddings[Modality.IMAGE]
    items, consumed = [], 0
    for start, end in offsets:
        rows = end - start + 1
        items.append(
            MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=[(start, end)],
                precomputed_embeddings=(
                    None
                    if image_embeddings is None
                    else image_embeddings[consumed : consumed + rows]
                ),
            )
        )
        consumed += rows
    return MultimodalProcessorOutput(
        input_ids=tokens, mm_items=items, im_token_id=image_token_id
    )


def image_preprocess_pool(max_workers: int) -> ThreadPoolExecutor:
    """Threads that preprocess one image each, with one torch intra-op thread.

    PIL decode/resize release the GIL, so images scale across threads. Left at
    its default, torch's intra-op pool spins every core for each image: on an
    H20 node a 4x1080p request cost 28 core-s instead of 0.25 core-s.
    """
    return ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="dsv41-image",
        initializer=torch.set_num_threads,
        initargs=(1,),
    )


def prepare_encoder_images(
    images: Iterable,
    *,
    hf_config,
    backend: str,
    cpu_patchify: Callable,
    executor: Optional[Executor] = None,
) -> EncoderPreprocessOutput:
    """Encode-server preprocessing: one item per image, as the local processor builds.

    With ``executor`` the images are preprocessed concurrently; order is kept.
    """
    load = functools.partial(
        load_image_features,
        hf_config=hf_config,
        backend=backend,
        cpu_patchify=cpu_patchify,
    )
    items, grids, token_counts = [], [], []
    for features in (map if executor is None else executor.map)(load, images):
        items.append(
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=features.feature,
                model_specific_data=features.model_specific_data(),
            )
        )
        grids.append([1, features.n_vit_h, features.n_vit_w])
        token_counts.append(features.num_tokens)
    return EncoderPreprocessOutput(
        {
            "pixel_values": [item.feature for item in items],
            "image_grid_thw": torch.tensor(grids, dtype=torch.int64),
        },
        mm_items=items,
        token_counts=token_counts,
    )


def encoder_image_meta(
    payloads: Sequence[bytes], *, hf_config
) -> EncoderPreprocessOutput:
    """Grids and token counts of encoded images from their headers alone.

    prepare_encoder_images plans every backend's grid from the decoded RGB
    image's size, and decoding (Image.open + convert("RGB"), no EXIF
    transpose) keeps the header's size, so this matches it without decoding.
    """
    p = hf_config.vision_patch_size
    items, grids, token_counts = [], [], []
    for data in payloads:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
        n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(
            width, height, hf_config
        )
        n_vit_h, n_vit_w = best_height // p, best_width // p
        # Featureless: metadata consumers never read pixels.
        items.append(
            MultimodalDataItem(
                modality=Modality.IMAGE,
                model_specific_data={"n_vit_h": n_vit_h, "n_vit_w": n_vit_w},
            )
        )
        grids.append([1, n_vit_h, n_vit_w])
        token_counts.append(num_image_tokens(n_llm_h, n_llm_w))
    return EncoderPreprocessOutput(
        {"image_grid_thw": torch.tensor(grids, dtype=torch.int64)},
        mm_items=items,
        token_counts=token_counts,
    )


def remap_vision_weight_name(name: str) -> str:
    if name.startswith("vision."):
        return name.replace(".attn.wqkv.", ".attn.qkv_proj.").replace(
            ".attn.wo.", ".attn.proj."
        )
    return name


def iter_vision_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Vision checkpoint tensors under their module names; others are dropped."""
    for name, tensor in weights:
        if name.startswith(VISION_WEIGHT_PREFIXES):
            yield remap_vision_weight_name(name), tensor


def vision_weight_allow_patterns(model_path: str) -> Optional[list[str]]:
    """DefaultModelLoader allow patterns for the shards that hold vision tensors.

    The loader stops at the first pattern that matches, so this is a single
    glob; it may also match a few other shards (one character class per
    position), whose tensors are dropped by name. None means read every shard.
    """
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return None
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    files = sorted(
        {
            file
            for name, file in weight_map.items()
            if name.startswith(VISION_WEIGHT_PREFIXES)
        }
    )
    if not files or len({len(file) for file in files}) != 1:
        return None
    pattern = []
    for chars in zip(*files):
        distinct = sorted(set(chars))
        if len(distinct) == 1:
            pattern.append(glob.escape(distinct[0]))
        elif all(char.isalnum() for char in distinct):
            pattern.append("[" + "".join(distinct) + "]")
        else:
            return None
    return ["".join(pattern)]
