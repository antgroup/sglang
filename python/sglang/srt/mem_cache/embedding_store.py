# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import abc
import hashlib
import importlib
import json
import logging
from typing import Any, Dict, List, Mapping, Optional

import torch

from sglang.srt.layers.dsv41_sm90 import dsv41_sm90_on

logger = logging.getLogger(__name__)


class EmbeddingStore(abc.ABC):
    """Abstract base class for multimodal embedding storage backends.

    Stores pre-computed vision/audio embeddings by content hash so they
    can be shared across nodes without re-running the encoder.
    """

    @abc.abstractmethod
    def batch_get(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        raise NotImplementedError

    @abc.abstractmethod
    def batch_put(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        raise NotImplementedError

    @abc.abstractmethod
    def batch_get_into_multi_buffers(
        self,
        hashes: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ) -> List[bool]:
        raise NotImplementedError

    @abc.abstractmethod
    def batch_put_from_multi_buffers(
        self,
        hashes: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ) -> List[bool]:
        raise NotImplementedError

    @abc.abstractmethod
    def batch_is_exist(self, hashes: List[str]) -> List[bool]:
        raise NotImplementedError

    def register_buffer(self, tensor: torch.Tensor) -> None:
        pass

    def bind_model(self, fingerprint: bytes) -> None:
        """Scope entries to the model that produces them; see model_fingerprint.

        A no-op by default: backends that keep the plain key do not scope entries.
        """

    def get_key(self, mm_hash: str) -> str:
        return f"emb_{mm_hash}"


def model_fingerprint(
    *,
    model_path: str,
    revision: Optional[str],
    dtype: torch.dtype,
    dims_by_modality: Mapping[Any, int],
) -> bytes:
    """16-byte identity of what an embedding of a given content hash means:
    the content hash covers the preprocessed input, not the encoder weights,
    dtype or per-modality width that turned it into the stored bytes."""
    payload = {
        "model_path": model_path,
        "revision": revision,
        "dtype": str(dtype),
        "dims": {str(k): int(v) for k, v in sorted(dims_by_modality.items(), key=str)},
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.blake2b(encoded, digest_size=16).digest()


class EmbeddingStoreFactory:
    """Factory for creating embedding store backend instances."""

    _registry: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _load_backend_class(
        module_path: str, class_name: str, backend_name: str
    ) -> type:
        try:
            module = importlib.import_module(module_path)
            backend_class = getattr(module, class_name)
            if not issubclass(backend_class, EmbeddingStore):
                raise TypeError(
                    f"Backend class {class_name} must inherit from EmbeddingStore"
                )
            return backend_class
        except ImportError as e:
            raise ImportError(
                f"Failed to import embedding store backend '{backend_name}' "
                f"from '{module_path}': {e}"
            ) from e
        except AttributeError as e:
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_path}': {e}"
            ) from e

    @classmethod
    def register_backend(cls, name: str, module_path: str, class_name: str) -> None:
        if name in cls._registry:
            logger.warning(
                f"Embedding store backend '{name}' is already registered, overwriting"
            )

        def loader() -> type:
            return cls._load_backend_class(module_path, class_name, name)

        cls._registry[name] = {
            "loader": loader,
            "module_path": module_path,
            "class_name": class_name,
        }

    @classmethod
    def create_backend(cls, backend_name: str, **kwargs) -> EmbeddingStore:
        if (
            backend_name == "shm"
            and backend_name not in cls._registry
            and dsv41_sm90_on("epd")
        ):
            cls.register_backend(
                "shm",
                "sglang.srt.mem_cache.storage.shm.shm_embedding_store",
                "ShmEmbeddingStore",
            )
        if backend_name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown embedding store backend '{backend_name}'. "
                f"Registered backends: {available}."
            )

        entry = cls._registry[backend_name]
        backend_class = entry["loader"]()
        logger.info(
            f"Creating embedding store backend '{backend_name}' "
            f"({entry['module_path']}.{entry['class_name']})"
        )
        return backend_class(**kwargs)


EmbeddingStoreFactory.register_backend(
    "mooncake",
    "sglang.srt.mem_cache.storage.mooncake_store.mooncake_embedding_store",
    "MooncakeEmbeddingStore",
)
