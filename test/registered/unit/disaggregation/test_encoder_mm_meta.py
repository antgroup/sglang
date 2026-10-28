"""Metadata-only EPD path: a language-only decode worker rebuilds mm_inputs
from encoder preprocess metadata and never receives an embedding."""

import asyncio
import concurrent.futures
import threading
import time
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
from fastapi import HTTPException
from requests.exceptions import HTTPError

import sglang.srt.disaggregation.encoder.mm_meta as mm_meta
import sglang.srt.disaggregation.encoder.receiver as receiver_module
from sglang.srt.disaggregation.encoder.mm_meta import (
    MMItemMeta,
    MMMetaCache,
    compute_mm_meta,
    mm_meta_key,
    store_batch_meta,
)
from sglang.srt.disaggregation.encoder.preprocessor import (
    EncoderPreprocessor,
    EncoderPreprocessResult,
)
from sglang.srt.disaggregation.encoder.receiver import (
    MMReceiverBase,
    MMReceiverGrpc,
    MMReceiverHTTP,
    _request_image_scope,
    build_mm_inputs_from_meta,
    epd_decode_uses_meta_only,
)
from sglang.srt.disaggregation.encoder.request_image_cache import RequestImageCache
from sglang.srt.disaggregation.encoder.server import BadRequestError, MMEncoder, MMError
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.cache.identity import MediaSnapshot
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


_HASH_A = "sha256:" + "ab" * 32
_DATA_A = "data:image/png;base64,QUFB"
_DATA_B = "data:image/png;base64,QkJC"
_DATA_C = "data:image/png;base64,Q0ND"


def _make_preprocessor(model_type="qwen3_vl", grid_of=None):
    preprocessor = EncoderPreprocessor.__new__(EncoderPreprocessor)
    preprocessor.model_type = model_type
    preprocessor.image_processor = SimpleNamespace(merge_size=2)
    preprocessor.io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    preprocessor.calls = []

    async def process_mm_items(mm_items, modality):
        preprocessor.calls.append(list(mm_items))
        grids = torch.tensor(
            [grid_of(item) if grid_of else [1, 4, 4] for item in mm_items]
        )
        return EncoderPreprocessResult(
            mm_inputs={"image_grid_thw": grids},
            grid_thw=grids,
            token_counts=[int(g.prod()) // 4 for g in grids],
        )

    preprocessor.process_mm_items = process_mm_items
    return preprocessor


def _compute(preprocessor, cache, request):
    return asyncio.run(
        compute_mm_meta(
            preprocessor=preprocessor,
            cache=cache,
            model_type="qwen3_vl",
            embedding_dim=8,
            request=request,
        )
    )


def _meta_request(mm_items, req_id="r_0", part_idx=0, num_parts=1):
    return {
        "req_id": req_id,
        "num_parts": num_parts,
        "part_idx": part_idx,
        "modality": "IMAGE",
        "mm_items": mm_items,
    }


class TestRequestImageSnapshots(CustomTestCase):
    def test_pd_identity_is_not_the_transport_or_client_request_id(self):
        """Two PD consumers share one attempt, but retry rooms and tenants do not."""
        p = SimpleNamespace(
            bootstrap_host="p",
            bootstrap_port=8998,
            bootstrap_room=1,
            cache_salt="tenant",
            rid="p-rid",
        )
        d = SimpleNamespace(**{**vars(p), "rid": "d-rid"})
        self.assertEqual(_request_image_scope(p), _request_image_scope(d))
        for field, value in (
            ("bootstrap_host", "other-p"),
            ("bootstrap_port", 9000),
            ("bootstrap_room", 2),
            ("cache_salt", "other-tenant"),
        ):
            with self.subTest(field=field):
                other = SimpleNamespace(**{**vars(p), field: value})
                self.assertNotEqual(
                    _request_image_scope(p), _request_image_scope(other)
                )
        no_pd = SimpleNamespace(
            **{**vars(p), "bootstrap_host": None, "bootstrap_port": None}
        )
        self.assertIsNone(_request_image_scope(no_pd))

    def test_scope_travels_only_with_the_switch(self):
        """Switch off, the /encode wire items stay upstream's: no scope key."""
        obj = GenerateReqInput(
            rid="r",
            text="hi",
            image_data=["http://image/x", _DATA_A],
            bootstrap_host="p",
            bootstrap_port=8998,
            bootstrap_room=1,
        )
        receiver = MMReceiverHTTP.__new__(MMReceiverHTTP)
        for epd_on in (False, True):
            receiver.dsv41_epd = epd_on
            with self.subTest(epd_on=epd_on):
                url_item, data_item = receiver._extract_url_data(obj)
                self.assertEqual("_request_image_scope" in url_item, epd_on)
                # Inline bytes are their own identity and need no download share.
                self.assertNotIn("_request_image_scope", data_item)

    def test_encode_and_meta_share_snapshot_in_both_arrival_orders(self):
        """P /encode and D metadata previously fetched the same URL twice.
        Force the first download to block until the other consumer has joined.
        """
        for first in ("encode", "meta"):
            with self.subTest(first=first):
                self._check_arrival_order(first)

    def _check_arrival_order(self, first):
        import sglang.srt.disaggregation.encoder.preprocessor as preproc_module

        preprocessor = _make_preprocessor(grid_of=lambda item: [1, 4, 4])
        self.addCleanup(preprocessor.io_executor.shutdown)
        preprocessor.request_image_cache = RequestImageCache(1024)
        preprocessor.use_image_processor_gpu = False
        preprocessor.encoder_media_processor_config = SimpleNamespace(
            preserve_media_metadata=False
        )
        item = {"url": "http://image/x", "_request_image_scope": "a" * 64}
        entered, release, joined = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def download(url):
            entered.set()
            self.assertTrue(release.wait(5))
            return MediaSnapshot(b"pixels", _HASH_A, 6, "bytes")

        class JoinFuture(concurrent.futures.Future):
            def result(self, timeout=None):
                joined.set()
                return super().result(timeout)

        async def run():
            loop = asyncio.get_running_loop()

            async def encode():
                return await loop.run_in_executor(
                    preprocessor.io_executor,
                    preprocessor._load_single_item,
                    item,
                    Modality.IMAGE,
                )

            async def meta():
                return await compute_mm_meta(
                    preprocessor=preprocessor,
                    cache=MMMetaCache(8),
                    model_type="qwen3_vl",
                    embedding_dim=8,
                    request=_meta_request([item]),
                )

            a = asyncio.create_task(encode() if first == "encode" else meta())
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                b = asyncio.create_task(meta() if first == "encode" else encode())
                self.assertTrue(await asyncio.to_thread(joined.wait, 5))
            finally:
                release.set()
            return await asyncio.gather(a, b)

        with (
            patch.object(
                preproc_module, "snapshot_media", side_effect=download
            ) as fetch,
            patch.object(
                preproc_module,
                "load_image",
                return_value=(SimpleNamespace(mode="RGB"), None),
            ),
            patch(
                "sglang.srt.disaggregation.encoder.request_image_cache.Future",
                JoinFuture,
            ),
        ):
            results = asyncio.run(run())
        self.assertEqual(fetch.call_count, 1)
        meta = results[1 if first == "encode" else 0]
        self.assertEqual(meta["token_counts"], [4])
        # Internal transport metadata must not become a preprocess option.
        self.assertNotIn("_request_image_scope", preprocessor.calls[0][0])

    def test_shared_snapshot_still_validates_each_consumers_content_hash(self):
        import sglang.srt.disaggregation.encoder.preprocessor as preproc_module

        preprocessor = _make_preprocessor()
        self.addCleanup(preprocessor.io_executor.shutdown)
        preprocessor.request_image_cache = RequestImageCache(1024)
        preprocessor.use_image_processor_gpu = False
        preprocessor.encoder_media_processor_config = SimpleNamespace(
            preserve_media_metadata=False
        )
        item = {
            "url": "http://image/x",
            "_request_image_scope": "a" * 64,
            "content_hash": _HASH_A,
        }
        with (
            patch.object(
                preproc_module,
                "snapshot_media",
                return_value=MediaSnapshot(b"pixels", _HASH_A, 6, "bytes"),
            ) as fetch,
            patch.object(
                preproc_module,
                "load_image",
                return_value=(SimpleNamespace(mode="RGB"), None),
            ) as decode,
        ):
            preprocessor._load_single_item(item, Modality.IMAGE)
            with self.assertRaisesRegex(BadRequestError, "content hash mismatch"):
                preprocessor._load_single_item(
                    {**item, "content_hash": "sha256:" + "cd" * 32}, Modality.IMAGE
                )
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(decode.call_count, 1)

    def test_new_attempt_and_unscoped_requests_refetch_mutable_url(self):
        cache = RequestImageCache(1024)
        loader = Mock(
            side_effect=[
                MediaSnapshot(b"old", _HASH_A, 3, "bytes"),
                MediaSnapshot(b"new", _HASH_A, 3, "bytes"),
            ]
        )
        old = cache.get("a" * 64, "http://image/x", loader)
        self.assertIs(cache.get("a" * 64, "http://image/x", loader), old)
        self.assertEqual(cache.get("b" * 64, "http://image/x", loader).data, b"new")
        for scope in (None, "client-rid"):
            with self.subTest(scope=scope):
                fetch = Mock(return_value=old)
                cache.get(scope, "http://image/x", fetch)
                cache.get(scope, "http://image/x", fetch)
                self.assertEqual(fetch.call_count, 2)

    def test_cancelled_consumer_does_not_cancel_shared_download(self):
        cache = RequestImageCache(1024)
        started, release = threading.Event(), threading.Event()
        snapshot = MediaSnapshot(b"ok", _HASH_A, 2, "bytes")

        def loader(url):
            started.set()
            self.assertTrue(release.wait(5))
            return snapshot

        async def run():
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                loop = asyncio.get_running_loop()
                a = loop.run_in_executor(
                    executor, cache.get, "a" * 64, "http://x", fetch
                )
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 5))
                    a.cancel()
                    b = loop.run_in_executor(
                        executor, cache.get, "a" * 64, "http://x", fetch
                    )
                finally:
                    release.set()
                self.assertIs(await b, snapshot)

        fetch = Mock(side_effect=loader)
        asyncio.run(run())
        self.assertEqual(fetch.call_count, 1)

    def test_failure_is_not_cached_and_limits_do_not_retain_oversized_bytes(self):
        cache = RequestImageCache(3, capacity=1)
        snapshot = MediaSnapshot(b"1234", _HASH_A, 4, "bytes")
        loader = Mock(side_effect=[OSError("broken"), snapshot, snapshot])
        with self.assertRaises(OSError):
            cache.get("a" * 64, "http://x", loader)
        self.assertIs(cache.get("a" * 64, "http://x", loader), snapshot)
        cache.get("a" * 64, "http://x", loader)
        self.assertEqual(loader.call_count, 3)
        self.assertEqual(cache._bytes, 0)
        self.assertEqual(len(cache._entries), 0)

    def test_expiry_byte_eviction_and_disabled_cache_refetch(self):
        snapshot = MediaSnapshot(b"12", _HASH_A, 2, "bytes")
        cache = RequestImageCache(3, ttl=5)
        fetch = Mock(return_value=snapshot)
        with patch(
            "sglang.srt.disaggregation.encoder.request_image_cache.time.monotonic",
            return_value=0,
        ):
            cache.get("a" * 64, "http://x", fetch)
            cache.get("a" * 64, "http://y", fetch)  # 4 bytes exceeds the budget
            cache.get("a" * 64, "http://x", fetch)
        self.assertEqual(fetch.call_count, 3)
        self.assertLessEqual(cache._bytes, 3)
        with patch(
            "sglang.srt.disaggregation.encoder.request_image_cache.time.monotonic",
            return_value=6,
        ):
            cache.get("a" * 64, "http://x", fetch)
        self.assertEqual(fetch.call_count, 4)
        disabled = RequestImageCache(0)
        disabled.get("a" * 64, "http://x", fetch)
        disabled.get("a" * 64, "http://x", fetch)
        self.assertEqual(fetch.call_count, 6)


class TestMMMetaKey(CustomTestCase):
    def test_content_hash_wins_over_payload(self):
        a = mm_meta_key({"url": _DATA_A, "content_hash": _HASH_A}, Modality.IMAGE)
        b = mm_meta_key({"url": _DATA_B, "content_hash": _HASH_A}, Modality.IMAGE)
        self.assertIsNotNone(a)
        self.assertEqual(a, b)

    def test_options_and_modality_are_part_of_the_key(self):
        base = mm_meta_key(_DATA_A, Modality.IMAGE)
        self.assertNotEqual(base, mm_meta_key(_DATA_A, Modality.VIDEO))
        self.assertNotEqual(
            base,
            mm_meta_key({"url": _DATA_A, "detail": "low"}, Modality.IMAGE),
        )
        self.assertEqual(base, mm_meta_key({"url": _DATA_A}, Modality.IMAGE))

    def test_unhashable_item_is_not_cacheable(self):
        self.assertIsNone(mm_meta_key({"url": None}, Modality.IMAGE))

    def test_a_reference_is_not_an_identity(self):
        """The bytes behind an http, file or path reference can change, so the
        name alone must never key the cache or derive a pad hash."""
        for url in ("http://x/a.png", "file:///x/a.png", "/x/a.png"):
            with self.subTest(url=url):
                self.assertIsNone(mm_meta_key(url, Modality.IMAGE))
                self.assertIsNone(mm_meta_key({"url": url}, Modality.VIDEO))


class TestMMMetaCache(CustomTestCase):
    def test_lru_eviction(self):
        cache = MMMetaCache(2)
        meta = MMItemMeta(grid_thw=[[1, 2, 2]], token_counts=[1], hashes=[1], aux={})
        cache.put("a", meta)
        cache.put("b", meta)
        cache.get("a")
        cache.put("c", meta)
        self.assertIsNone(cache.get("b"))
        self.assertIsNotNone(cache.get("a"))
        self.assertIsNotNone(cache.get("c"))

    def test_zero_capacity_never_stores(self):
        cache = MMMetaCache(0)
        cache.put("a", MMItemMeta(grid_thw=[], token_counts=[], hashes=[], aux={}))
        self.assertEqual(len(cache), 0)


class TestComputeMMMeta(CustomTestCase):
    def test_misses_are_preprocessed_once_then_served_from_cache(self):
        preprocessor = _make_preprocessor()
        cache = MMMetaCache(16)
        items = [_DATA_A, _DATA_B]

        first = _compute(preprocessor, cache, _meta_request(items))
        self.assertEqual(first["grid_thw"], [[1, 4, 4], [1, 4, 4]])
        self.assertEqual(first["token_counts"], [4, 4])
        self.assertEqual(len(first["hashes"]), 2)
        self.assertEqual(first["embedding_dim"], 8)
        self.assertEqual(preprocessor.calls, [items])

        # Second request: only the new item is preprocessed, order is kept.
        second = _compute(preprocessor, cache, _meta_request([_DATA_C, _DATA_A]))
        self.assertEqual(preprocessor.calls[-1], [_DATA_C])
        self.assertEqual(second["hashes"][1], first["hashes"][0])
        self.assertNotEqual(second["hashes"][0], first["hashes"][0])

    def test_user_hashes_override_computed_ones(self):
        result = _compute(
            _make_preprocessor(),
            MMMetaCache(0),
            {**_meta_request([_DATA_A]), "hashes": [42]},
        )
        self.assertEqual(result["hashes"], [42])

    def test_image_behind_an_unchanged_url_is_keyed_by_its_bytes(self):
        """An object rewritten behind the same URL must not be answered with the
        previous grid or pad hash; the bytes fetched for the key are also the
        bytes preprocessed."""
        grids = {b"small": [1, 32, 32], b"large": [1, 48, 64]}
        preprocessor = _make_preprocessor(grid_of=lambda item: grids[item["url"]])
        cache = MMMetaCache(16)
        served = [b"small"]

        def snapshot(url):
            self.assertEqual(url, "http://oss/x/latest.jpg")
            data = served[0]
            return MediaSnapshot(data, "sha256:" + data.hex().ljust(64, "0"), 1, "b")

        request = _meta_request(["http://oss/x/latest.jpg"])
        with patch.object(mm_meta, "snapshot_media", side_effect=snapshot):
            first = _compute(preprocessor, cache, request)
            served[0] = b"large"
            second = _compute(preprocessor, cache, request)

        self.assertEqual(first["token_counts"], [256])
        self.assertEqual(second["token_counts"], [768])
        self.assertNotEqual(first["hashes"], second["hashes"])
        self.assertEqual(
            [call[0]["url"] for call in preprocessor.calls], [b"small", b"large"]
        )

    def test_video_reference_is_neither_cached_nor_hashed_by_name(self):
        preprocessor = _make_preprocessor()
        cache = MMMetaCache(16)
        request = {**_meta_request(["http://x/a.mp4"]), "modality": "VIDEO"}

        first = _compute(preprocessor, cache, request)
        second = _compute(preprocessor, cache, request)

        self.assertEqual(len(preprocessor.calls), 2)
        self.assertEqual(len(cache), 0)
        self.assertNotEqual(first["hashes"], second["hashes"])

    def test_encode_batch_populates_cache(self):
        preprocessor = _make_preprocessor()
        cache = MMMetaCache(16)
        grids = torch.tensor([[1, 4, 4], [1, 2, 2], [1, 2, 2]])
        result = EncoderPreprocessResult(
            mm_inputs={"image_grid_thw": grids}, grid_thw=grids, token_counts=[4, 1, 1]
        )
        store_batch_meta(
            cache=cache,
            preprocessor=preprocessor,
            model_type="qwen3_vl",
            modality=Modality.IMAGE,
            requests=[
                _meta_request([_DATA_A], req_id="p"),
                _meta_request([_DATA_B, "http://x/c.png"], req_id="q"),
            ],
            result=result,
            aux={"video_timestamps": None},
        )
        b = cache.get(mm_meta_key(_DATA_B, Modality.IMAGE))
        self.assertEqual(b.grid_thw, [[1, 2, 2]])
        self.assertEqual(b.token_counts, [1])
        self.assertEqual(len(cache), 2)
        self.assertEqual(preprocessor.calls, [])


class _StubProcessor:
    """Mimics get_mm_data: one item per grid, in prompt order."""

    def get_mm_data(self, prompt, embeddings, **kwargs):
        assert embeddings is None
        img_grid_thw = kwargs["img_grid_thw"]
        items = [
            MultimodalDataItem(modality=Modality.IMAGE, offsets=[(i, i)])
            for i in range(len(img_grid_thw))
        ]
        self.last_kwargs = kwargs
        return MultimodalProcessorOutput(input_ids=[1] * len(items), mm_items=items)


class TestBuildMMInputsFromMeta(CustomTestCase):
    def _metas(self):
        return [
            {
                "req_id": "r_1",
                "num_parts": 2,
                "part_idx": 1,
                "modality": "IMAGE",
                "grid_thw": [[1, 2, 2]],
                "token_counts": [1],
                "hashes": [22],
                "embedding_dim": 8,
                "aux": {},
            },
            {
                "req_id": "r_0",
                "num_parts": 2,
                "part_idx": 0,
                "modality": "IMAGE",
                "grid_thw": [[1, 4, 4]],
                "token_counts": [4],
                "hashes": [11],
                "embedding_dim": 8,
                "aux": {},
            },
        ]

    def test_parts_are_ordered_and_hashes_bound_to_items(self):
        processor = _StubProcessor()
        output = build_mm_inputs_from_meta(
            self._metas(), "r", processor, [1, 2, 3], "qwen3_vl"
        )
        torch.testing.assert_close(
            processor.last_kwargs["img_grid_thw"], torch.tensor([[1, 4, 4], [1, 2, 2]])
        )
        self.assertEqual([item.hash for item in output.mm_items], [11, 22])
        self.assertTrue(all(item.pad_value is not None for item in output.mm_items))
        self.assertTrue(
            all(item.precomputed_embeddings is None for item in output.mm_items)
        )

    def test_hash_count_mismatch_rejects_the_rebuild(self):
        metas = self._metas()
        metas[0]["hashes"] = []
        self.assertIsNone(
            build_mm_inputs_from_meta(metas, "r", _StubProcessor(), [1], "qwen3_vl")
        )

    def test_fetch_falls_back_to_none_on_encoder_failure(self):
        receiver = SimpleNamespace(
            encode_urls=["http://enc"],
            model_type="qwen3_vl",
            _extract_url_data=lambda obj: [
                {"url": "http://x/a.png", "modality": Modality.IMAGE}
            ],
            _assign_items_by_modality=lambda mm_data, n: {Modality.IMAGE: [1]},
            preprocess_meta=AsyncMock(return_value=None),
        )
        result = asyncio.run(
            MMReceiverBase.fetch_mm_meta_inputs(
                receiver,
                request_obj=SimpleNamespace(rid="r"),
                mm_processor=_StubProcessor(),
                prompt=[1],
            )
        )
        self.assertIsNone(result)
        receiver.preprocess_meta.assert_awaited_once()

    def test_fetch_is_bounded_by_the_meta_timeout(self):
        """A stalled encoder must not hold a decode request for the embedding
        receive timeout; the metadata wait gives up after its own bound."""

        async def stalled(*args, **kwargs):
            await asyncio.sleep(1.0)
            return None

        receiver = SimpleNamespace(
            encode_urls=["http://enc"],
            model_type="qwen3_vl",
            _extract_url_data=lambda obj: [
                {"url": "http://x/a.png", "modality": Modality.IMAGE}
            ],
            _assign_items_by_modality=lambda mm_data, n: {Modality.IMAGE: [1]},
            preprocess_meta=stalled,
        )
        start = time.monotonic()
        with patch.object(receiver_module, "_META_TIMEOUT_S", 0.05):
            result = asyncio.run(
                MMReceiverBase.fetch_mm_meta_inputs(
                    receiver,
                    request_obj=SimpleNamespace(rid="r"),
                    mm_processor=_StubProcessor(),
                    prompt=[1],
                )
            )
        self.assertIsNone(result)
        self.assertLess(time.monotonic() - start, 0.5)

    def test_fetch_falls_back_when_the_processor_needs_embeddings(self):
        """A processor whose get_mm_data indexes the embedding must send the
        request to a remote encode instead of failing it."""

        class EmbeddingOnlyProcessor:
            def get_mm_data(self, prompt, embeddings, **kwargs):
                return embeddings[Modality.IMAGE]

        receiver = SimpleNamespace(
            encode_urls=["http://enc"],
            model_type="qwen3_vl",
            _extract_url_data=lambda obj: [
                {"url": "http://x/a.png", "modality": Modality.IMAGE}
            ],
            _assign_items_by_modality=lambda mm_data, n: {Modality.IMAGE: [1]},
            preprocess_meta=AsyncMock(return_value=self._metas()),
        )
        result = asyncio.run(
            MMReceiverBase.fetch_mm_meta_inputs(
                receiver,
                request_obj=SimpleNamespace(rid="r"),
                mm_processor=EmbeddingOnlyProcessor(),
                prompt=[1],
            )
        )
        self.assertIsNone(result)


class TestEncoderPreprocessMeta(CustomTestCase):
    def test_bad_media_metadata_is_a_client_error(self):
        """A broken URL used to become a 500 and trigger a duplicate encode.
        Unexpected encoder failures must still retain their server semantics."""
        encoder = MMEncoder.__new__(MMEncoder)
        encoder.preprocessor = _make_preprocessor()
        self.addCleanup(encoder.preprocessor.io_executor.shutdown)
        encoder.mm_meta_cache = MMMetaCache(16)
        encoder.model_type = "qwen3_vl"
        encoder._embedding_dims = {Modality.IMAGE: 8}
        for error in (
            HTTPError("404 Client Error: Not Found"),
            OSError("cannot identify image file"),
            ValueError("invalid image options"),
            RuntimeError("unexpected processor failure"),
            MMError("metadata unsupported", code=HTTPStatus.NOT_IMPLEMENTED),
        ):
            with self.subTest(error=type(error).__name__):
                with patch(
                    "sglang.srt.disaggregation.encoder.server.compute_mm_meta",
                    new=AsyncMock(side_effect=error),
                ):
                    if isinstance(error, (OSError, ValueError)):
                        with self.assertRaises(BadRequestError) as raised:
                            asyncio.run(
                                encoder.preprocess_meta(_meta_request([_DATA_A]))
                            )
                        self.assertEqual(raised.exception.code, HTTPStatus.BAD_REQUEST)
                        self.assertIn(str(error), str(raised.exception))
                    else:
                        with self.assertRaises(type(error)) as raised:
                            asyncio.run(
                                encoder.preprocess_meta(_meta_request([_DATA_A]))
                            )
                        self.assertIs(raised.exception, error)

    def test_receiver_propagates_bad_media_but_keeps_metadata_fallbacks(self):
        """Exercise both HTTP parsing and fetch: client errors must not turn
        into None, while legacy 404/501 and transient failures still do."""
        receiver = MMReceiverHTTP.__new__(MMReceiverHTTP)
        receiver.model_type = "qwen3_vl"
        receiver.encode_urls = ["http://enc"]
        receiver._extract_url_data = lambda obj: [
            {"url": _DATA_A, "modality": Modality.IMAGE}
        ]
        receiver._assign_items_by_modality = lambda data, n: {Modality.IMAGE: [1]}
        for status in (400, 422, 404, 501, 500, 503, 504):
            with self.subTest(status=status):
                response = SimpleNamespace(
                    status=status,
                    json=AsyncMock(
                        return_value={"message": "bad media or unavailable"}
                    ),
                )
                session = AsyncMock()
                session.post = AsyncMock(return_value=response)
                with patch(
                    "sglang.srt.disaggregation.encoder.receiver.aiohttp.ClientSession"
                ) as client:
                    client.return_value.__aenter__ = AsyncMock(return_value=session)
                    client.return_value.__aexit__ = AsyncMock(return_value=False)
                    fetch = receiver.fetch_mm_meta_inputs(
                        request_obj=SimpleNamespace(rid="r"),
                        mm_processor=_StubProcessor(),
                        prompt=[1],
                    )
                    if status in (400, 422):
                        with self.assertRaises(HTTPException) as raised:
                            asyncio.run(fetch)
                        self.assertEqual(raised.exception.status_code, status)
                        self.assertIn("bad media", raised.exception.detail)
                    else:
                        self.assertIsNone(asyncio.run(fetch))
                    session.post.assert_awaited_once()

    def test_encode_batch_meta_is_keyed_off_the_event_loop(self):
        """Hashing a batch's inline payloads must not stall rank 0 before the
        forward, and must survive the pipeline dropping mm_items afterwards."""
        encoder = MMEncoder.__new__(MMEncoder)
        encoder.rank = 0
        encoder.model_type = "qwen3_vl"
        encoder.preprocessor = _make_preprocessor()
        encoder.mm_meta_cache = MMMetaCache(16)
        encoder.background_tasks = set()
        grids = torch.tensor([[1, 4, 4]])
        ctx = SimpleNamespace(
            is_health_check=False,
            modality=Modality.IMAGE,
            preprocess_result=EncoderPreprocessResult(
                mm_inputs={"image_grid_thw": grids}, grid_thw=grids, token_counts=[4]
            ),
            aux_data={},
        )
        requests = [_meta_request([_DATA_A])]
        key_threads = []
        real_key = mm_meta.mm_meta_key

        def recording_key(item, modality):
            key_threads.append(threading.current_thread())
            return real_key(item, modality)

        async def run():
            encoder._cache_preprocess_meta(ctx, requests)
            requests[0].pop("mm_items")
            await asyncio.gather(*encoder.background_tasks)

        with patch.object(mm_meta, "mm_meta_key", side_effect=recording_key):
            asyncio.run(run())

        self.assertTrue(key_threads)
        self.assertNotIn(threading.main_thread(), key_threads)
        self.assertIsNotNone(
            encoder.mm_meta_cache.get(real_key(_DATA_A, Modality.IMAGE))
        )

    def test_tp_sharded_video_decode_is_refused(self):
        """Only rank 0 serves /preprocess_meta, so a TP-collective video decode
        there would wait on peers forever; the decode side must fall back."""
        encoder = MMEncoder.__new__(MMEncoder)
        encoder.server_args = SimpleNamespace(mm_enable_dp_encoder=True)
        with (
            patch(
                "sglang.srt.disaggregation.encoder.server.get_parallel",
                return_value=SimpleNamespace(attn_tp_size=2),
            ),
            self.assertRaises(MMError) as ctx,
        ):
            asyncio.run(
                encoder.preprocess_meta(
                    {**_meta_request(["http://x/a.mp4"]), "modality": "VIDEO"}
                )
            )
        self.assertEqual(ctx.exception.code, HTTPStatus.NOT_IMPLEMENTED)


class TestModeSelection(CustomTestCase):
    def test_only_language_only_decode_with_scheduler_backends(self):
        http = MMReceiverHTTP.__new__(MMReceiverHTTP)
        http.dsv41_epd = True

        def disagg(mode, backend, language_only=True):
            return SimpleNamespace(
                language_only=language_only,
                disaggregation_mode=mode,
                encoder_transfer_backend=backend,
            )

        self.assertTrue(
            epd_decode_uses_meta_only(disagg("decode", "zmq_to_scheduler"), http)
        )
        self.assertTrue(epd_decode_uses_meta_only(disagg("decode", "mooncake"), http))
        self.assertFalse(
            epd_decode_uses_meta_only(disagg("prefill", "zmq_to_scheduler"), http)
        )
        self.assertFalse(
            epd_decode_uses_meta_only(disagg("decode", "zmq_to_tokenizer"), http)
        )
        self.assertFalse(
            epd_decode_uses_meta_only(disagg("decode", "zmq_to_scheduler", False), http)
        )
        # Switch off: the upstream decode keeps the embedding path.
        http.dsv41_epd = False
        self.assertFalse(
            epd_decode_uses_meta_only(disagg("decode", "zmq_to_scheduler"), http)
        )

    def test_grpc_receiver_keeps_the_embedding_path(self):
        """gRPC encoders have no /preprocess_meta; a meta-only decode behind
        them would preprocess every request locally."""
        grpc = MMReceiverGrpc.__new__(MMReceiverGrpc)
        grpc.dsv41_epd = True
        decode = SimpleNamespace(
            language_only=True,
            disaggregation_mode="decode",
            encoder_transfer_backend="zmq_to_scheduler",
        )
        self.assertFalse(epd_decode_uses_meta_only(decode, grpc))


class _TokenizerStub:
    """The TokenizerManager state the EPD decode dispatch reads."""

    def __init__(self, meta_inputs=None):
        self.epd_decode_meta_only = True
        self.enable_trace = False
        self.rid_to_state = {}
        self.encoder_dispatch_ready = {}
        self.mm_processor = object()
        self.dispatch_ready = threading.Event()
        self.mm_receiver = SimpleNamespace(
            fetch_mm_meta_inputs=AsyncMock(return_value=meta_inputs),
            send_encode_request=Mock(return_value=self.dispatch_ready),
        )
        self._schedule_encoder_dispatch_error = Mock()
        for name in (
            "_epd_should_dispatch",
            "_epd_fetches_meta",
            "_fetch_epd_decode_mm_inputs",
            "_dispatch_epd_encode",
            "_handle_epd_disaggregation_encode_request",
        ):
            setattr(self, name, getattr(TokenizerManager, name).__get__(self))


def _decode_disagg():
    return patch(
        "sglang.srt.managers.tokenizer_manager.get_disagg",
        return_value=SimpleNamespace(
            encoder_transfer_backend="mooncake",
            enable_adaptive_dispatch_to_encoder=False,
        ),
    )


class TestDecodeMetaFallback(CustomTestCase):
    def test_meta_failure_dispatches_the_encode_instead_of_local_preprocessing(self):
        """When the decode cannot rebuild inputs from encoder metadata, its
        layout must still come from the encoder prefill used, not from
        tokenizer-local preprocessing that may expand differently."""
        tm = _TokenizerStub(meta_inputs=None)
        obj = GenerateReqInput(rid="r", text="hi", image_data=[_DATA_A])
        with _decode_disagg():
            tm._handle_epd_disaggregation_encode_request(obj)
            self.assertFalse(obj.need_wait_for_mm_inputs)
            result = asyncio.run(tm._fetch_epd_decode_mm_inputs(obj, [1]))

        self.assertIsNone(result)
        self.assertTrue(obj.need_wait_for_mm_inputs)
        tm.mm_receiver.send_encode_request.assert_called_once()
        self.assertIs(tm.encoder_dispatch_ready["r"], tm.dispatch_ready)

    def test_video_on_a_meta_decode_dispatches_the_encode_up_front(self):
        tm = _TokenizerStub()
        obj = GenerateReqInput(rid="v", text="hi", video_data=["http://x/a.mp4"])
        with _decode_disagg():
            tm._handle_epd_disaggregation_encode_request(obj)

        self.assertTrue(obj.need_wait_for_mm_inputs)
        tm.mm_receiver.send_encode_request.assert_called_once()
        self.assertFalse(tm._epd_fetches_meta(obj))

    def test_meta_success_dispatches_nothing(self):
        rebuilt = object()
        tm = _TokenizerStub(meta_inputs=rebuilt)
        obj = GenerateReqInput(rid="i", text="hi", image_data=[_DATA_A])
        with _decode_disagg():
            tm._handle_epd_disaggregation_encode_request(obj)
            result = asyncio.run(tm._fetch_epd_decode_mm_inputs(obj, [1]))

        self.assertIs(result, rebuilt)
        self.assertFalse(obj.need_wait_for_mm_inputs)
        tm.mm_receiver.send_encode_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
