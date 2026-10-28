"""Preprocess-only multimodal metadata served by the EPD encoder.

A decode worker in EPD never embeds multimodal tokens; it only needs the
expanded input_ids and M-RoPE positions, which follow from the processor
grids. `/preprocess_meta` runs the CPU preprocessing (no encoder forward) and
returns per-part grids, token counts, item hashes and video/image aux data in
the same shape the embedding path carries, so the receiver rebuilds
`mm_inputs` through the same `get_mm_data` call with no embedding attached.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import msgspec
import torch

from sglang.srt.disaggregation.encoder.preprocessor import (
    EncoderPreprocessor,
    EncoderPreprocessResult,
    _build_mm_aux_data,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.cache import parse_content_hash, snapshot_media

logger = logging.getLogger(__name__)

# Aux attrs that carry one entry per grid and can be sliced per media item.
# Anything else present (MiMo audio-in-video fields) bypasses the cache.
_PER_GRID_AUX_ATTRS = ("video_timestamps", "second_per_grid_ts", "original_image_sizes")


class MMItemMeta(msgspec.Struct, frozen=True, kw_only=True):
    """Preprocess result of one media item (possibly several grids)."""

    grid_thw: List[List[int]]
    token_counts: List[int]
    hashes: List[int]
    aux: Dict[str, list]


class MMMetaCache:
    """LRU of MMItemMeta keyed by media identity; single event loop, no lock."""

    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self._entries: OrderedDict[str, MMItemMeta] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> Optional[MMItemMeta]:
        meta = self._entries.get(key)
        if meta is not None:
            self._entries.move_to_end(key)
        return meta

    def put(self, key: str, meta: MMItemMeta) -> None:
        if self.capacity == 0:
            return
        self._entries[key] = meta
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)


def _media_url(mm_item: Any) -> Any:
    return mm_item.get("url") if isinstance(mm_item, dict) else mm_item


def _inline_payload(url: Any) -> Optional[bytes]:
    """The bytes of media carried inline, or None for a reference (http, file,
    path) whose content can change behind the same string."""
    if isinstance(url, (bytes, bytearray)):
        return bytes(url)
    if isinstance(url, str) and url.startswith("data:"):
        return url.encode("utf-8", "surrogatepass")
    return None


def mm_meta_key(mm_item: Any, modality: Modality) -> Optional[str]:
    """Content identity of a media item plus every per-item preprocess option:
    the client's content hash, else a digest of the inline payload. None for a
    reference without a content hash; such an item is neither cached nor
    hashed by name."""
    options: Dict[str, Any] = {}
    content_hash = None
    url = _media_url(mm_item)
    if isinstance(mm_item, dict):
        content_hash = mm_item.get("content_hash")
        options = {
            k: v
            for k, v in mm_item.items()
            if k not in ("url", "content_hash", "_request_image_scope")
        }
    if not isinstance(url, (str, bytes, bytearray)):
        return None
    try:
        options_json = json.dumps(options, sort_keys=True, default=str)
    except TypeError:
        return None
    if content_hash is not None:
        try:
            identity = parse_content_hash(content_hash)
        except ValueError:
            return None
    else:
        payload = _inline_payload(url)
        if payload is None:
            return None
        identity = "raw:" + hashlib.sha256(payload).hexdigest()
    return f"{modality.name}|{identity}|{options_json}"


def _is_unhashed_image_reference(mm_item: Any) -> bool:
    url = _media_url(mm_item)
    has_content_hash = (
        isinstance(mm_item, dict) and mm_item.get("content_hash") is not None
    )
    return (
        isinstance(url, str) and _inline_payload(url) is None and not has_content_hash
    )


async def _snapshot_image_references(
    preprocessor: EncoderPreprocessor, leaves: Sequence[Any]
) -> List[Any]:
    """Fetch referenced images once, so their key and their preprocessing both
    come from the same bytes rather than from whatever the URL serves later."""
    loop = asyncio.get_running_loop()
    pending = [i for i, leaf in enumerate(leaves) if _is_unhashed_image_reference(leaf)]

    def snapshot(leaf):
        scope = leaf.get("_request_image_scope") if isinstance(leaf, dict) else None
        if scope is not None:
            return preprocessor.snapshot_request_image(_media_url(leaf), scope)
        return snapshot_media(_media_url(leaf))

    snapshots = await asyncio.gather(
        *(
            loop.run_in_executor(preprocessor.io_executor, snapshot, leaves[i])
            for i in pending
        )
    )
    resolved = list(leaves)
    for i, snapshot in zip(pending, snapshots):
        options = (
            {k: v for k, v in leaves[i].items() if k != "_request_image_scope"}
            if isinstance(leaves[i], dict)
            else {}
        )
        resolved[i] = {
            **options,
            "url": snapshot.data,
            "content_hash": snapshot.content_digest,
        }
    return resolved


def _meta_keys(leaves: Sequence[Any], modality: Modality) -> List[Optional[str]]:
    return [mm_meta_key(leaf, modality) for leaf in leaves]


def _grid_hash(key: Optional[str], grid_idx: int) -> int:
    if key is None:
        return uuid.uuid4().int & ((1 << 64) - 1)
    digest = hashlib.sha256(f"{key}#{grid_idx}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


def _grid_entries(grid_thw: Any) -> List[List[int]]:
    """One list per grid entry; audio grids are scalars, so wrap them."""
    entries = _to_jsonable(grid_thw)
    return [e if isinstance(e, list) else [e] for e in entries]


def _split_aux_per_leaf(
    aux: Dict[str, Any], leaf_ranges: Sequence[Tuple[int, int]]
) -> Optional[List[Dict[str, list]]]:
    """Slice part-level aux into per-leaf pieces; None when a field cannot be
    attributed to single grids (the request then bypasses the cache)."""
    present = {k: v for k, v in aux.items() if v is not None}
    if any(k not in _PER_GRID_AUX_ATTRS for k in present):
        return None
    num_grids = leaf_ranges[-1][1] if leaf_ranges else 0
    per_leaf: List[Dict[str, list]] = [{} for _ in leaf_ranges]
    for key, value in present.items():
        values = _to_jsonable(value)
        if not isinstance(values, list) or len(values) != num_grids:
            return None
        for i, (start, end) in enumerate(leaf_ranges):
            per_leaf[i][key] = values[start:end]
    return per_leaf


def _leaf_ranges(grid_counts: Sequence[int]) -> List[Tuple[int, int]]:
    ranges, offset = [], 0
    for count in grid_counts:
        ranges.append((offset, offset + count))
        offset += count
    return ranges


def _metas_from_result(
    *,
    preprocessor: EncoderPreprocessor,
    model_type: str,
    modality: Modality,
    leaves: Sequence[Any],
    keys: Sequence[Optional[str]],
    result: EncoderPreprocessResult,
    aux: Optional[Dict[str, Any]] = None,
) -> Optional[List[MMItemMeta]]:
    """Per-leaf metas for one processed batch; None if aux is not sliceable."""
    grid_counts = preprocessor._grid_count_per_leaf(list(leaves), modality)
    ranges = _leaf_ranges(grid_counts)
    grids = _grid_entries(result.grid_thw)
    if ranges and ranges[-1][1] != len(grids):
        raise ValueError(
            f"Preprocess produced {len(grids)} grids for {sum(grid_counts)} "
            f"expected leaf grids ({modality.name})"
        )
    if aux is None:
        aux = _build_mm_aux_data(result.mm_inputs, model_type)
    per_leaf_aux = _split_aux_per_leaf(aux, ranges)
    if per_leaf_aux is None:
        return None
    metas = []
    for key, (start, end), leaf_aux in zip(keys, ranges, per_leaf_aux):
        metas.append(
            MMItemMeta(
                grid_thw=grids[start:end],
                token_counts=[int(c) for c in result.token_counts[start:end]],
                hashes=[_grid_hash(key, i) for i in range(end - start)],
                aux=leaf_aux,
            )
        )
    return metas


def _merge_metas(metas: Sequence[MMItemMeta]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "grid_thw": [],
        "token_counts": [],
        "hashes": [],
        "aux": {},
    }
    for meta in metas:
        merged["grid_thw"].extend(meta.grid_thw)
        merged["token_counts"].extend(meta.token_counts)
        merged["hashes"].extend(meta.hashes)
        for key, values in meta.aux.items():
            merged["aux"].setdefault(key, []).extend(values)
    return merged


def batch_meta_entries(
    *,
    preprocessor: EncoderPreprocessor,
    model_type: str,
    modality: Modality,
    leaves: Sequence[Any],
    result: EncoderPreprocessResult,
    aux: Dict[str, Any],
) -> List[Tuple[str, MMItemMeta]]:
    """Cacheable (key, meta) pairs of an /encode batch that already
    preprocessed; pure CPU work, safe to run off the event loop."""
    keys = _meta_keys(leaves, modality)
    if not any(keys):
        return []
    try:
        metas = _metas_from_result(
            preprocessor=preprocessor,
            model_type=model_type,
            modality=modality,
            leaves=leaves,
            keys=keys,
            result=result,
            aux=aux,
        )
    except ValueError as e:
        logger.warning("Skip caching preprocess meta: %s", e)
        return []
    if metas is None:
        return []
    return [(key, meta) for key, meta in zip(keys, metas) if key is not None]


def batch_leaves(
    preprocessor: EncoderPreprocessor, requests: Sequence[dict]
) -> List[Any]:
    return [
        leaf
        for req in requests
        for leaf in preprocessor._flatten_nested_items(req["mm_items"])
    ]


def store_batch_meta(
    *,
    cache: MMMetaCache,
    preprocessor: EncoderPreprocessor,
    model_type: str,
    modality: Modality,
    requests: Sequence[dict],
    result: EncoderPreprocessResult,
    aux: Dict[str, Any],
) -> None:
    """Populate the cache from an /encode batch that already preprocessed."""
    if cache.capacity == 0:
        return
    for key, meta in batch_meta_entries(
        preprocessor=preprocessor,
        model_type=model_type,
        modality=modality,
        leaves=batch_leaves(preprocessor, requests),
        result=result,
        aux=aux,
    ):
        cache.put(key, meta)


async def compute_mm_meta(
    *,
    preprocessor: EncoderPreprocessor,
    cache: MMMetaCache,
    model_type: str,
    embedding_dim: int,
    request: dict,
) -> dict:
    """Preprocess (or recall) every media item of one request part."""
    modality = Modality.from_str(request["modality"])
    leaves = preprocessor._flatten_nested_items(request["mm_items"])
    if modality == Modality.IMAGE:
        leaves = await _snapshot_image_references(preprocessor, leaves)
    # Inline payloads can be megabytes; hash them off the event loop.
    keys = await asyncio.to_thread(_meta_keys, leaves, modality)
    metas: List[Optional[MMItemMeta]] = [
        cache.get(key) if key is not None else None for key in keys
    ]
    missing = [i for i, meta in enumerate(metas) if meta is None]
    if missing:
        miss_leaves = [leaves[i] for i in missing]
        result = await preprocessor.process_mm_items_meta(miss_leaves, modality)
        miss_metas = _metas_from_result(
            preprocessor=preprocessor,
            model_type=model_type,
            modality=modality,
            leaves=miss_leaves,
            keys=[keys[i] for i in missing],
            result=result,
        )
        if miss_metas is None:
            # Aux cannot be attributed per item: answer from the batch as one
            # uncached unit (the batch is exactly this request's misses).
            if len(missing) != len(leaves):
                result = await preprocessor.process_mm_items_meta(leaves, modality)
            return _response(
                request, modality, embedding_dim, _uncached(result, keys, model_type)
            )
        for i, meta in zip(missing, miss_metas):
            metas[i] = meta
            if keys[i] is not None:
                cache.put(keys[i], meta)
    merged = _merge_metas([meta for meta in metas if meta is not None])
    return _response(request, modality, embedding_dim, merged)


def _uncached(
    result: EncoderPreprocessResult, keys: Sequence[Optional[str]], model_type: str
) -> Dict[str, Any]:
    grids = _grid_entries(result.grid_thw)
    aux = {
        k: _to_jsonable(v)
        for k, v in _build_mm_aux_data(result.mm_inputs, model_type).items()
        if v is not None
    }
    key = keys[0] if len(keys) == 1 else None
    return {
        "grid_thw": grids,
        "token_counts": [int(c) for c in result.token_counts],
        "hashes": [_grid_hash(key, i) for i in range(len(grids))],
        "aux": aux,
    }


def _response(
    request: dict, modality: Modality, embedding_dim: int, merged: Dict[str, Any]
) -> dict:
    hashes = request.get("hashes")
    if hashes is not None and len(hashes) == len(merged["grid_thw"]):
        merged = {**merged, "hashes": [int(h) for h in hashes]}
    return {
        "req_id": request["req_id"],
        "num_parts": request["num_parts"],
        "part_idx": request["part_idx"],
        "modality": modality.name,
        "embedding_dim": int(embedding_dim),
        **merged,
    }
