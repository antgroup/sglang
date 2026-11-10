# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unified L3 objects name global cache contents independently of their writer.

The live typed-component planner uses explicit global layer and head/channel
coordinates. Known head shards default to one global head per group, and layer
records default to one global layer per object. Replicated latent/index/state
components have no head coordinate. TP/PP/CP repartitioning therefore preserves
the same objects whenever the deployment owns complete cells of that grid.

Explicit fleet-wide head/layer partitions can reduce object counts. Partition
boundaries and the byte schema are namespace identity; incompatible sparse
layers, packing schemas, ownership or PP boundaries fail instead of silently
ignoring the requested grid. Unknown legacy shard descriptors retain topology
identity and miss safely after resharding.

The older ``plan_unified_kv`` API below remains available for canonical dense
KV and keeps its documented rank-local defaults when partitions are omitted.
All planners distinguish logical dtypes even when their physical bytes share
the same uint8 storage type.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, replace

import msgspec

from sglang.srt.mem_cache.pool_host.page_unified import (
    PAGE_UNIFIED_OBJECT_LAYOUT,
    PageStorageComponent,
)

logger = logging.getLogger(__name__)

# The digest covers the encoded struct, so any schema change changes every key.
# Bump on any field change.
_SCHEMA_VERSION = 1


class KVCacheNamespace(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Immutable identity of one shared L3 KV keyspace.

    Everything that must match for two deployments' KV bytes to be
    interchangeable. Field names and order are part of the encoding: append
    only, and bump ``schema_version``.
    """

    schema_version: int = _SCHEMA_VERSION
    model_id: str
    # Logical dtype, not the storage view: fp8 variants all store as uint8
    # and must not share a keyspace.
    dtype: str
    page_size: int
    # MLA-family pools: KV is replicated across attn-TP ranks, so there is no
    # head axis (total_kv_heads and head_group are 0).
    rank_replicated: bool
    total_kv_heads: int
    # Layers per chunk; 0 = this rank's own range (same-split sharing only).
    # A stage must START on a multiple of it; only the last PP stage may end
    # short, forming the model's trailing remainder chunk.
    layer_partition: int = 0
    head_group: int
    # Byte order of the stored objects. Layouts serialize a page in different
    # orders at EQUAL sizes, so omitting this would let two deployments
    # exchange byte-permuted KV under identical keys.
    object_layout: str


def namespace_digest(namespace: KVCacheNamespace) -> str:
    """Digest of the namespace encoding, used as the key prefix.

    msgpack encoding of a Struct is deterministic given the class definition,
    which is why the schema versions the encoding.
    """
    encoded = msgspec.msgpack.encode(namespace)
    return f"ukv{_SCHEMA_VERSION}-{hashlib.sha256(encoded).hexdigest()[:16]}"


def derive_namespace(
    *,
    model_id: str,
    dtype: str,
    page_size: int,
    rank_replicated: bool,
    total_kv_heads: int,
    head_group: int,
    layer_partition: int = 0,
    object_layout: str = PAGE_UNIFIED_OBJECT_LAYOUT,
) -> KVCacheNamespace:
    """Derive the namespace from deployment facts plus the fleet agreements."""
    namespace = KVCacheNamespace(
        model_id=model_id,
        dtype=dtype,
        page_size=page_size,
        rank_replicated=rank_replicated,
        total_kv_heads=0 if rank_replicated else total_kv_heads,
        head_group=0 if rank_replicated else head_group,
        layer_partition=layer_partition,
        object_layout=object_layout,
    )
    _validate_grid(namespace)
    return namespace


def _validate_grid(namespace: KVCacheNamespace) -> None:
    if namespace.layer_partition < 0:
        raise ValueError(f"layer_partition must be non-negative: {namespace}")
    if namespace.page_size <= 0:
        raise ValueError(f"page_size must be positive: {namespace}")
    if not namespace.model_id:
        raise ValueError(
            "the unified key scheme requires a non-empty model_id: an empty id "
            "would merge different models into one keyspace."
        )
    if namespace.rank_replicated:
        if namespace.head_group != 0 or namespace.total_kv_heads != 0:
            raise ValueError(
                f"rank_replicated namespaces have no head axis; set "
                f"total_kv_heads=0 and head_group=0: {namespace}"
            )
        return
    if namespace.head_group <= 0 or namespace.total_kv_heads <= 0:
        raise ValueError(
            f"sharded-KV namespaces need positive total_kv_heads/head_group: "
            f"{namespace}"
        )
    if namespace.total_kv_heads % namespace.head_group != 0:
        raise ValueError(
            f"head_group={namespace.head_group} must divide "
            f"total_kv_heads={namespace.total_kv_heads}."
        )


def resolve_head_group_num(local_kv_heads: int) -> int:
    """How many head groups this rank's page_unified pool is cut into.

    Host pools allocate before storage attaches, so construction and the live
    typed planner use the same fleet-wide head partition. Unified storage
    defaults to one global KV head per group; a rank owns as many groups as
    it has local heads. Other key schemes retain one local allocation group.
    """
    from sglang.srt.runtime_context import get_memory

    memory = get_memory()
    if memory.hicache_storage_key_scheme != "unified":
        return 1
    head_group = memory.hicache_storage_head_partition
    if head_group is None:
        # One global KV head per object is the topology-independent minimum.
        # A larger explicit fleet-wide unit may reduce the object count.
        return local_kv_heads
    if local_kv_heads % head_group != 0:
        raise ValueError(
            f"--hicache-storage-head-partition {head_group} must divide this rank's "
            f"{local_kv_heads} kv heads."
        )
    return local_kv_heads // head_group


def normalize_dtype(dtype: object) -> str:
    """``torch.bfloat16`` -> ``"bfloat16"``."""
    return str(dtype).removeprefix("torch.")


class UnifiedKVPlan(msgspec.Struct, frozen=True, kw_only=True):
    """One rank's unified-key objects and where their bytes sit in a page.

    ``suffixes`` is layer-major / head-minor, and ``layer_ranges`` x
    ``head_ranges`` is the same cross product in the same order: entry
    ``i * len(head_ranges) + j`` of ``suffixes`` names
    ``(layer_ranges[i], head_ranges[j])``. Everything downstream relies on that
    pairing, so the two are built together here rather than re-derived.

    The ranges are LOCAL to this rank's pool, so they index the host pool
    directly; the suffixes carry the absolute, model-global coordinates.
    """

    namespace: KVCacheNamespace
    suffixes: list[str]
    layer_ranges: list[tuple[int, int]]
    # Head groups, or a single whole-pool range for rank-replicated pools.
    head_ranges: list[tuple[int, int]]

    @property
    def head_group_num(self) -> int:
        return len(self.head_ranges)


def plan_unified_kv(
    *,
    model_id: str,
    dtype: str,
    page_size: int,
    rank_replicated: bool,
    local_kv_heads: int,
    attn_tp_rank: int,
    attn_tp_size: int,
    attn_cp_size: int,
    start_layer: int,
    end_layer: int,
    is_final_stage: bool,
    head_group: int | None = None,
    layer_partition: int | None = None,
) -> UnifiedKVPlan:
    """Derive the namespace and this rank's chunk plan from deployment facts.

    Raises rather than degrading: a rank whose shard does not tile the declared
    grid would write objects a reader cannot reassemble, and the symptom would
    be silently wrong KV rather than a miss.
    """
    # Prefill CP replicates full KV pages across its ranks. Its size does not
    # change object identity; the controller elects CP rank 0 to write each
    # attention-TP shard. DCP's token slices are rejected during configuration.
    if attn_cp_size < 1:
        raise ValueError(f"attn_cp_size must be positive: {attn_cp_size}")
    if not 0 <= start_layer < end_layer:
        raise ValueError(f"invalid layer range [{start_layer}, {end_layer}).")

    total_kv_heads = 0 if rank_replicated else local_kv_heads * attn_tp_size
    resolved_head_group = _resolve_head_group(
        rank_replicated=rank_replicated,
        local_kv_heads=local_kv_heads,
        attn_tp_size=attn_tp_size,
        total_kv_heads=total_kv_heads,
        head_group=head_group,
    )
    namespace = derive_namespace(
        model_id=model_id,
        dtype=dtype,
        page_size=page_size,
        rank_replicated=rank_replicated,
        total_kv_heads=total_kv_heads,
        head_group=resolved_head_group,
        layer_partition=layer_partition or 0,
    )

    layer_coords, layer_ranges = _layer_chunks(
        namespace.layer_partition, start_layer, end_layer, is_final_stage
    )
    digest = namespace_digest(namespace)
    if rank_replicated:
        return UnifiedKVPlan(
            namespace=namespace,
            suffixes=[f"{digest}_{coord}" for coord in layer_coords],
            layer_ranges=layer_ranges,
            head_ranges=[(0, 1)],
        )

    if local_kv_heads % resolved_head_group != 0:
        raise ValueError(
            f"this rank's {local_kv_heads} kv heads do not tile "
            f"head_group={resolved_head_group}."
        )
    groups_per_rank = local_kv_heads // resolved_head_group
    first_head_index = attn_tp_rank * groups_per_rank
    return UnifiedKVPlan(
        namespace=namespace,
        # Layer-major / head-minor, the order the page block packs them in.
        suffixes=[
            f"{digest}_{coord}_H{first_head_index + i}"
            for coord in layer_coords
            for i in range(groups_per_rank)
        ],
        layer_ranges=layer_ranges,
        head_ranges=[
            (i * resolved_head_group, (i + 1) * resolved_head_group)
            for i in range(groups_per_rank)
        ],
    )


def _resolve_head_group(
    *,
    rank_replicated: bool,
    local_kv_heads: int,
    attn_tp_size: int,
    total_kv_heads: int,
    head_group: int | None,
) -> int:
    if rank_replicated:
        return 0
    if head_group is None:
        if local_kv_heads == 1 and attn_tp_size > 1:
            # Ambiguous: 1 head/rank could equally mean kv-head replication,
            # where two ranks hold the same head and would race on one key.
            # An explicit head_group is the operator attesting it is sharding.
            raise NotImplementedError(
                "the unified key scheme cannot derive a namespace at 1 kv head "
                "per rank; set --hicache-storage-head-partition, or use "
                "--hicache-storage-key-scheme rank-suffix."
            )
        if local_kv_heads != total_kv_heads:
            # Correct but unshareable: another attn-TP size derives a different
            # digest and misses everything. Nothing else reports this -- the
            # symptom is a 0% hit rate against a populated store.
            logger.warning(
                "unified key scheme: --hicache-storage-head-partition is unset, so "
                "this namespace is keyed to this rank's %d of %d kv heads and "
                "can only share objects with deployments at attn-TP %d. Set it "
                "to total_kv_heads / lcm of the fleet's attn-TP sizes to share "
                "across TP sizes; a rank owning several groups simply owns "
                "several chunks.",
                local_kv_heads,
                total_kv_heads,
                attn_tp_size,
            )
        return local_kv_heads
    if head_group <= 0:
        raise ValueError(f"head_group must be positive: {head_group}")
    if local_kv_heads % head_group != 0 or head_group > local_kv_heads:
        raise ValueError(
            f"head_group={head_group} must divide this rank's {local_kv_heads} "
            f"kv heads. As a fleet grid it must divide every member's local "
            f"kv-head count, so pick total_kv_heads / lcm(the fleet's attn-TP "
            f"sizes) -- here total_kv_heads={total_kv_heads}."
        )
    return head_group


def _layer_chunks(
    layer_partition: int, start_layer: int, end_layer: int, is_final_stage: bool
) -> tuple[list[str], list[tuple[int, int]]]:
    """Absolute layer coordinates and their pool-local ranges.

    Coordinates are ABSOLUTE, so any PP split -- uneven stages included --
    yields collision-free names, and a differing split misses rather than
    colliding.
    """
    if not layer_partition:
        return [f"L{start_layer}-{end_layer}"], [(0, end_layer - start_layer)]
    if start_layer % layer_partition != 0:
        raise ValueError(
            f"this rank's layer range [{start_layer}, {end_layer}) does not "
            f"start on a multiple of layer_partition={layer_partition}; stages "
            f"whose boundaries do not align to the layer unit cannot share a "
            f"partitioned namespace (drop layer_partition for per-stage ranges)."
        )
    if end_layer % layer_partition != 0 and not is_final_stage:
        raise ValueError(
            f"this rank's layer range [{start_layer}, {end_layer}) ends off the "
            f"layer_partition={layer_partition} grid; only the FINAL pipeline "
            f"stage may end short (the model's trailing remainder forms the "
            f"short last chunk)."
        )
    bounds = [
        (a, min(a + layer_partition, end_layer))
        for a in range(start_layer, end_layer, layer_partition)
    ]
    return (
        [f"L{a}-{b}" for a, b in bounds],
        [(a - start_layer, b - start_layer) for a, b in bounds],
    )


@dataclass(frozen=True)
class UnifiedPoolStoragePlan:
    suffixes: tuple[str, ...]
    chunk_spans: tuple[tuple[int, int], ...]
    is_storage_writer: bool = True
    writer_mask: tuple[bool, ...] = ()

    @property
    def owned_suffixes(self) -> tuple[str, ...]:
        return tuple(
            suffix for suffix, owned in zip(self.suffixes, self.chunk_writers) if owned
        )

    @property
    def chunk_writers(self) -> tuple[bool, ...]:
        return self.writer_mask or (self.is_storage_writer,) * len(self.suffixes)

    @property
    def page_bytes(self) -> int:
        return sum(length for _, length in self.chunk_spans)

    def validate(self) -> None:
        if not self.suffixes and not self.chunk_spans:
            return  # A layer-sharded rank may own no sidecar records.
        if not self.suffixes or len(set(self.suffixes)) != len(self.suffixes):
            raise ValueError("Unified pool keys must be nonempty and unique.")
        if len(self.suffixes) != len(self.chunk_spans):
            raise ValueError("Every unified pool key requires a byte span.")
        if self.writer_mask and len(self.writer_mask) != len(self.suffixes):
            raise ValueError("Every unified pool key requires writer ownership.")
        end = 0
        for offset, length in sorted(self.chunk_spans):
            if type(offset) is not int or type(length) is not int:
                raise ValueError("Unified pool spans require integer byte coordinates.")
            if offset != end or length <= 0:
                raise ValueError("Unified pool spans must cover a page without gaps.")
            end += length


def host_page_components(host_pool) -> tuple[PageStorageComponent, ...]:
    components = getattr(host_pool, "storage_component_metadata", None)
    if components is not None:
        return tuple(components)
    if (
        getattr(host_pool, "kv_buffer", object()) is None
        and host_pool.get_dummy_flat_data_page().numel() == 0
    ):
        return ()  # Logical anchors carry indices; their side pools carry bytes.
    layout = getattr(host_pool, "page_unified_layout", None)
    if layout is None:
        raise ValueError(
            f"{type(host_pool).__name__} does not describe its page_unified bytes."
        )
    # Canonical KV in a sparse-layer or packed-draft group still has contiguous
    # layer records inside each head group. The generic format names those
    # records independently; the dense KV path retains its canonical grid.
    return tuple(
        PageStorageComponent(
            name=f"kv-head-group-{group}",
            shape=(
                layout.layer_num,
                layout.components,
                layout.page_size,
                layout.heads_per_group,
                layout.head_dim,
            ),
            dtype=str(host_pool.dtype),
            byte_offset=group * layout.head_group_bytes,
            byte_length=layout.head_group_bytes,
        )
        for group in range(layout.head_group_num)
    )


@dataclass(frozen=True)
class _PoolComponentRecord:
    identity: tuple
    coordinate: str
    byte_offset: int
    byte_length: int
    is_writer: bool
    shard_coordinate: str | None
    has_layer_axis: bool


def _partition_component_layers(
    records: list[_PoolComponentRecord],
    partition: int,
    *,
    is_final_stage: bool,
    stage_end_layer: int | None,
) -> list[_PoolComponentRecord]:
    """Coalesce complete global layer cells without crossing byte/schema gaps."""
    groups = {}
    for index, record in enumerate(records):
        coordinate = record.coordinate
        if not (
            record.has_layer_axis
            and coordinate.startswith("L")
            and coordinate[1:].isdigit()
        ):
            continue  # Drafts and slot-wide state retain independent objects.
        layer = int(coordinate[1:])
        if stage_end_layer is not None and layer >= stage_end_layer:
            raise ValueError("Unified layer coordinates exceed the model stage end")
        grid_start = layer // partition * partition
        key = (
            record.identity,
            record.shard_coordinate,
            record.is_writer,
            record.byte_length,
            grid_start,
        )
        groups.setdefault(key, []).append((layer, index, record))
    replacements, consumed = {}, set()
    for entries in groups.values():
        entries.sort(key=lambda entry: entry[0])
        first_layer = entries[0][0]
        start = first_layer // partition * partition
        end = start + partition
        if is_final_stage and stage_end_layer is not None:
            end = min(end, stage_end_layer)
        if [layer for layer, _, _ in entries] != list(range(start, end)):
            raise ValueError(
                f"layer_partition={partition} requires a complete global layer grid "
                f"[{start}, {end}) with the same schema, shard and writer ownership; "
                "sparse layers and PP/CP boundaries cannot form partial objects. "
                "Only the final model-stage remainder may end short."
            )
        for (_, _, left), (_, _, right) in zip(entries, entries[1:]):
            if left.byte_offset + left.byte_length != right.byte_offset:
                raise ValueError(
                    f"layer_partition={partition} requires byte-adjacent layers "
                    "in ascending global layer order"
                )
        positions = [index for _, index, _ in entries]
        first = min(positions)
        replacements[first] = replace(
            entries[0][2],
            coordinate=f"L{start}-{end}",
            byte_length=sum(record.byte_length for _, _, record in entries),
        )
        consumed.update(index for index in positions if index != first)
    return [
        replacements.get(index, record)
        for index, record in enumerate(records)
        if index not in consumed
    ]


def plan_pool_components(
    *,
    model_id: str,
    pool_name: str,
    object_layout: str,
    page_size: int,
    components: tuple[PageStorageComponent, ...],
    layer_ids: tuple[str, ...],
    tp_rank: int,
    tp_size: int,
    is_storage_writer: bool = True,
    logical_dtype: str = "",
    attn_tp_rank: int | None = None,
    attn_tp_size: int | None = None,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
    layer_partition: int | None = None,
    is_final_stage: bool = True,
    stage_end_layer: int | None = None,
) -> UnifiedPoolStoragePlan:
    """Pair each global layer/shard object with one contiguous host byte span.

    Explicit shard coordinates describe content independently of TP/PP/CP.
    Replicated components use an empty shard coordinate and elect a writer.
    Only legacy descriptors without ownership metadata retain topology keys,
    so an unknown shard can never masquerade as a global state/head range.
    """
    if not model_id or page_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError("Invalid unified component namespace.")
    if layer_partition is not None and (
        type(layer_partition) is not int or layer_partition < 1
    ):
        raise ValueError("layer_partition must be a positive integer")
    partition = layer_partition or 1
    attn_tp_rank = tp_rank if attn_tp_rank is None else attn_tp_rank
    attn_tp_size = tp_size if attn_tp_size is None else attn_tp_size
    if not (0 <= attn_tp_rank < attn_tp_size and 0 <= attn_cp_rank < attn_cp_size):
        raise ValueError("Invalid unified component attention topology.")
    records = []
    for component in components:
        if component.byte_length == 0:
            continue
        if component.layer_axis not in (None, 0):
            raise ValueError("Stored layer components must be layer-major.")
        if component.layer_axis == 0:
            count = component.shape[0]
            if count <= 0 or component.byte_length % count:
                raise ValueError("Component bytes must split into complete layers.")
            start = component.layer_offset
            coords = layer_ids[start : start + count]
            if start < 0 or len(coords) != count:
                raise ValueError(
                    "Storage component is missing model layer coordinates."
                )
            shape = component.shape[1:]
        else:
            count = 1
            # Slot-wide side state may depend on the complete local layer set.
            coords = (
                component.layer_coordinate
                or "S"
                + hashlib.sha256(msgspec.msgpack.encode(layer_ids)).hexdigest()[:16],
            )
            shape = component.shape
        explicit_shard = component.shard_coordinate is not None
        identity = (
            "page-components-v2" if explicit_shard else "page-components-v1",
            model_id,
            str(pool_name),
            object_layout,
            page_size,
            component.name,
            component.dtype,
            logical_dtype,
            shape,
        )
        if not explicit_shard:
            identity += (
                tp_size,
                tp_rank,
                attn_tp_size,
                attn_tp_rank,
                attn_cp_size,
                attn_cp_rank,
            )
        if partition > 1:
            identity += ("layer_partition", partition)
        length = component.byte_length // count
        for index, coordinate in enumerate(coords):
            records.append(
                _PoolComponentRecord(
                    identity,
                    coordinate,
                    component.byte_offset + index * length,
                    length,
                    is_storage_writer and component.is_storage_writer,
                    component.shard_coordinate,
                    component.layer_axis == 0,
                )
            )
    if partition > 1:
        records = _partition_component_layers(
            records,
            partition,
            is_final_stage=is_final_stage,
            stage_end_layer=stage_end_layer,
        )
    suffixes, spans, writers = [], [], []
    for record in records:
        digest = hashlib.sha256(msgspec.msgpack.encode(record.identity)).hexdigest()[
            :24
        ]
        shard = f"_{record.shard_coordinate}" if record.shard_coordinate else ""
        version = 2 if record.shard_coordinate is not None else 1
        suffixes.append(f"upc{version}-{digest}_{record.coordinate}{shard}")
        spans.append((record.byte_offset, record.byte_length))
        writers.append(record.is_writer)
    plan = UnifiedPoolStoragePlan(
        tuple(suffixes),
        tuple(spans),
        any(writers) if suffixes else is_storage_writer,
        tuple(writers),
    )
    plan.validate()
    return plan
