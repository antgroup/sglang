import torch

from sglang.multimodal_gen.runtime.models.dits.lingbot_world import (
    _lingbot_advance_ring_kv,
    _lingbot_copy_to_ring,
    _lingbot_gather_ring_kv,
    _lingbot_kv_attention_ranges,
    _lingbot_linearize_ring_tail,
    _lingbot_ring_kv_state,
)


def _cache(cache_tokens: int):
    return {
        "k": torch.zeros(1, cache_tokens, 1, 1),
        "v": torch.zeros(1, cache_tokens, 1, 1),
        "global_end_index": torch.tensor([0], dtype=torch.long),
        "local_end_index": torch.tensor([0], dtype=torch.long),
        "global_end_index_int": 0,
        "local_end_index_int": 0,
        "tail_start_index_int": 0,
        "tail_global_start_index_int": 0,
        "tail_span_tokens_int": 0,
    }


def _write(cache, values, current_start: int, sink_tokens: int):
    cache_tokens = cache["k"].shape[1]
    tail_capacity = cache_tokens - sink_tokens
    global_end, _, tail_start, tail_global_start, tail_span = _lingbot_ring_kv_state(
        cache, sink_tokens
    )
    state = _lingbot_advance_ring_kv(
        global_end=global_end,
        tail_start=tail_start,
        tail_global_start=tail_global_start,
        tail_span=tail_span,
        current_end=current_start + values.shape[1],
        sink_tokens=sink_tokens,
        tail_capacity=tail_capacity,
    )
    global_end, local_end, tail_start, tail_global_start, tail_span = state
    for name, offset in (("k", 0), ("v", 1000)):
        _lingbot_copy_to_ring(
            cache[name],
            values + offset,
            current_start=current_start,
            sink_tokens=sink_tokens,
            tail_capacity=tail_capacity,
            tail_start=tail_start,
            tail_global_start=tail_global_start,
        )
    cache.update(
        global_end_index_int=global_end,
        local_end_index_int=local_end,
        tail_start_index_int=tail_start,
        tail_global_start_index_int=tail_global_start,
        tail_span_tokens_int=tail_span,
    )
    cache["global_end_index"].fill_(global_end)
    cache["local_end_index"].fill_(local_end)
    return state


def test_ring_cache_wrap_and_repeated_chunk_overwrite_preserve_logical_order():
    sink_tokens = 3
    cache = _cache(cache_tokens=10)
    expected = {}

    for current_start in range(0, 16, 2):
        for version in (0, 100):
            values = (
                torch.arange(current_start, current_start + 2, dtype=torch.float32)
                + version
            ).view(1, 2, 1, 1)
            state = _write(cache, values, current_start, sink_tokens)
            expected.update(
                {
                    token: float(token + version)
                    for token in range(current_start, current_start + 2)
                }
            )

            global_end, local_end, tail_start, tail_global_start, _ = state
            current_local_start = (
                current_start
                if current_start < sink_tokens
                else sink_tokens + current_start - tail_global_start
            )
            ranges = _lingbot_kv_attention_ranges(
                visible_local_end=local_end,
                current_local_start=current_local_start,
                sink_tokens=sink_tokens,
                valid_local_start=0,
                sample_tokens=3,
                local_attn_size=-1,
                max_attention_size=100,
            )
            key, value = _lingbot_gather_ring_kv(
                cache,
                ranges,
                sink_tokens=sink_tokens,
                tail_start=tail_start,
            )
            logical_tokens = []
            for start, length in ranges:
                for local_idx in range(start, start + length):
                    logical_tokens.append(
                        local_idx
                        if local_idx < sink_tokens
                        else tail_global_start + local_idx - sink_tokens
                    )
            expected_key = torch.tensor([expected[token] for token in logical_tokens])
            assert global_end == current_start + 2
            torch.testing.assert_close(key.flatten(), expected_key, atol=0, rtol=0)
            torch.testing.assert_close(
                value.flatten(), expected_key + 1000, atol=0, rtol=0
            )


def test_ring_cache_reset_gap_is_excluded_but_sink_is_preserved():
    sink_tokens = 3
    cache = _cache(cache_tokens=10)
    _write(
        cache,
        torch.arange(3, dtype=torch.float32).view(1, 3, 1, 1),
        0,
        sink_tokens,
    )
    replay_start = 5
    state = _write(
        cache,
        torch.tensor([50.0, 60.0]).view(1, 2, 1, 1),
        replay_start,
        sink_tokens,
    )
    _, local_end, tail_start, tail_global_start, _ = state
    current_local_start = sink_tokens + replay_start - tail_global_start
    ranges = _lingbot_kv_attention_ranges(
        visible_local_end=local_end,
        current_local_start=current_local_start,
        sink_tokens=sink_tokens,
        valid_local_start=replay_start,
        sample_tokens=3,
        local_attn_size=-1,
        max_attention_size=100,
    )
    key, _ = _lingbot_gather_ring_kv(
        cache,
        ranges,
        sink_tokens=sink_tokens,
        tail_start=tail_start,
    )

    assert ranges == [(0, 3), (5, 2)]
    assert key.flatten().tolist() == [0.0, 1.0, 2.0, 50.0, 60.0]


def test_full_window_linearizes_wrapped_tail_for_repeated_attention():
    sink_tokens = 3
    cache = _cache(cache_tokens=10)
    for current_start in range(0, 12, 2):
        values = torch.arange(
            current_start, current_start + 2, dtype=torch.float32
        ).view(1, 2, 1, 1)
        state = _write(cache, values, current_start, sink_tokens)

    _, local_end, tail_start, tail_global_start, _ = state
    assert tail_start != 0
    _lingbot_linearize_ring_tail(
        cache,
        sink_tokens=sink_tokens,
        visible_local_end=local_end,
        tail_start=tail_start,
    )
    key, value = _lingbot_gather_ring_kv(
        cache,
        [(0, local_end)],
        sink_tokens=sink_tokens,
        tail_start=0,
    )

    expected = torch.tensor([0, 1, 2, 5, 6, 7, 8, 9, 10, 11], dtype=torch.float32)
    assert cache["tail_start_index_int"] == 0
    assert key.data_ptr() == cache["k"].data_ptr()
    torch.testing.assert_close(key.flatten(), expected, atol=0, rtol=0)
    torch.testing.assert_close(value.flatten(), expected + 1000, atol=0, rtol=0)
    assert tail_global_start == 5
