from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import tempfile
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import psutil
import tqdm

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.layers.dsv41_sm90 import dsv41_sm90_on
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.runtime_context import get_parallel, get_schedule

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}


def warmup(name: str):
    def decorator(fn):
        _warmup_registry[name] = fn
        return fn

    return decorator


# DeepSeek-V4.1 SM90 fast startup: what each role runs before readiness when
# --warmups is not given, keyed by --disaggregation-mode.
_STARTUP_WARMUPS: Dict[str, Tuple[str, ...]] = {
    "null": ("prefill_shapes", "serving_shapes"),
    "prefill": ("prefill_shapes", "serving_shapes"),
    "decode": ("serving_shapes",),
}

# The long serving_shapes prompt fills every chunk and exercises deep prefixes.
_LONG_PREFILL_TOKENS = 262144


def startup_warmup_names(
    *, explicit: Optional[str], disaggregation_mode: str
) -> List[str]:
    """--warmups when given (empty or "none" runs nothing), else the role's set."""
    if explicit is not None:
        return [name for name in explicit.split(",") if name and name != "none"]
    return list(_STARTUP_WARMUPS.get(disaggregation_mode, ()))


def warmup_launch_key(pid: int) -> str:
    """Names one launch: the pid plus its start time, so a later launch in the same
    container that reuses the pid never finds this launch's done marker."""
    return f"{pid}-{int(psutil.Process(pid).create_time() * 1000)}"


def one_chunk_lengths(
    lengths: List[int], *, disaggregation_mode: str, chunked_prefill_size: int
) -> List[int]:
    """PD prefill warmups are fake-bootstrap requests, which skip the radix insert;
    upstream loops and leaks pages when such a request spans several chunks (it
    lacks the prefix advance and the out-of-window SWA free in mem_cache/common.py),
    so on a prefill server every warmup prompt fits one chunk."""
    if disaggregation_mode != "prefill" or chunked_prefill_size <= 0:
        return lengths
    return sorted({min(n, chunked_prefill_size) for n in lengths})


async def execute_warmups_once(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
    *,
    launch_key: Optional[str],
    explicit: bool = False,
) -> None:
    # Explicit warmups retain upstream's per-worker execution and fail-fast.
    # Only automatic warmups share a best-effort run across tokenizer workers.
    if explicit:
        await execute_warmups(disaggregation_mode, warmup_names, tokenizer_manager)
        await _flush_after_warmups(tokenizer_manager)
        return
    if launch_key is None:
        await _run_warmups_logging_failures(
            disaggregation_mode, warmup_names, tokenizer_manager
        )
        await _flush_after_warmups(tokenizer_manager)
        return
    lock_path = os.path.join(
        tempfile.gettempdir(), f"sglang-startup-warmup-{launch_key}.lock"
    )
    done_path = lock_path + ".done"
    with open(lock_path, "a") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            while not os.path.exists(done_path):
                await asyncio.sleep(0.5)
            return
        try:
            if not os.path.exists(done_path):
                await _run_warmups_logging_failures(
                    disaggregation_mode, warmup_names, tokenizer_manager
                )
                await _flush_after_warmups(tokenizer_manager)
        finally:
            open(done_path, "w").close()
            fcntl.flock(lock_file, fcntl.LOCK_UN)


async def _flush_after_warmups(tokenizer_manager: TokenizerManager) -> None:
    """Flush once idle: drop the warmup prompts from the prefix cache and hand the
    blocks the long warmup prompts left in the caching allocator back to the device
    (H20 TP8 PD prefill: ~4 GiB less free device memory without it)."""
    try:
        result = await tokenizer_manager.flush_cache(timeout_s=60)
    except Exception:
        logger.exception("Startup warmup: flush_cache failed.")
        return
    if not result.success:
        logger.warning(f"Startup warmup: flush_cache did not run: {result.message}")


async def _run_warmups_logging_failures(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
) -> None:
    for name in warmup_names:
        try:
            await execute_warmups(disaggregation_mode, [name], tokenizer_manager)
        except Exception:
            logger.exception(
                f"Startup warmup {name!r} failed; serving without it (its kernels "
                "compile on first use)."
            )


async def execute_warmups(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
):
    warmups = _warmup_registry
    if dsv41_sm90_on("startup"):
        warmups = {**warmups, "serving_shapes": serving_shapes}
    for warmup_name in warmup_names:
        if warmup_name not in warmups:
            logger.warning(f"Could not find custom warmup {warmup_name}")
            continue
        logger.info(f"Running warmup {warmup_name}")
        await warmups[warmup_name](disaggregation_mode, tokenizer_manager)


@warmup("whisper_autodetect")
async def whisper_autodetect(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Pre-compile the xgrammar FSM for both Whisper auto-detect regexes.

    The first request that uses each structured-generation regex incurs a
    ~15-20s compilation cost. xgrammar caches compiled grammars by the
    exact regex string, so we warm both the notimestamps and timestamps
    variants here — otherwise the first ``language=None +
    timestamp_granularities`` request would still pay the full spike.
    """
    # A short silent audio encoded as base64 WAV (0.1s, 16kHz, mono) —
    # soundfile produces the WAV header + PCM data from a list of floats.
    import base64
    import io

    import soundfile as sf

    from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (
        FUSED_AUTODETECT_FLAG,
        WHISPER_AUTODETECT_REGEX,
        WHISPER_AUTODETECT_TS_REGEX,
    )

    sr, dur = 16000, 0.1
    n = int(sr * dur)
    buf = io.BytesIO()
    sf.write(buf, [0.0] * n, sr, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

    for variant_name, regex in (
        ("notimestamps", WHISPER_AUTODETECT_REGEX),
        ("timestamps", WHISPER_AUTODETECT_TS_REGEX),
    ):
        logger.info(
            "Compiling Whisper auto-detect regex FSM (%s, one-time, ~15-20s)...",
            variant_name,
        )
        req = GenerateReqInput(
            text="",
            audio_data=audio_data_uri,
            sampling_params={
                "max_new_tokens": 4,
                "temperature": 0,
                "regex": regex,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                FUSED_AUTODETECT_FLAG: True,
            },
            modalities=["audio"],
        )
        # PD prefill servers assert req.bootstrap_room is not None in the
        # default follow_bootstrap_room scheduler; the fake values match
        # what the voice_chat warmup uses for the same reason.
        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        # Drain the generator so the FSM is fully installed and any
        # downstream exception surfaces instead of being swallowed after
        # the first yield.
        async for _ in tokenizer_manager.generate_request(req, None):
            pass
    logger.info("Whisper auto-detect regex FSMs compiled.")


@warmup("voice_chat")
async def voice_chat(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    # this warms up the fused_moe triton kernels and caches them
    # if we don't do this we break real time inference for voice chat
    for i in tqdm.trange(1, 512):
        size = i * 4
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 30,
                "temperature": 0.8,
                "stop_token_ids": [1],
                "min_p": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


@warmup("prefill_shapes")
async def prefill_shapes(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Warmup Triton kernels across a wide range of prefill seq_lens (up to 32K).

    Uses power-of-2 sizes plus intermediate points to cover the shape space
    that fused_moe, attention extend, and other Triton kernels may encounter.
    """
    page_size = 64
    sizes = set()
    base = 64
    while base <= 32768:
        sizes.add(base)
        mid = base * 3 // 2
        mid = (mid + page_size - 1) // page_size * page_size
        if mid <= 32768:
            sizes.add(mid)
        base *= 2
    sizes = sorted(sizes)
    fast_startup = dsv41_sm90_on("startup")
    if fast_startup:
        sizes = one_chunk_lengths(
            sizes,
            disaggregation_mode=disaggregation_mode,
            chunked_prefill_size=get_schedule().chunked_prefill_size,
        )

    for size in tqdm.tqdm(sizes, desc="Warmup prefill shapes (up to 32K)"):
        generate_req_input = GenerateReqInput(
            input_ids=(np.random.randint(2**16, size=[size])).tolist(),
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        if not fast_startup:
            await tokenizer_manager.generate_request(
                generate_req_input, None
            ).__anext__()
            continue
        # Drain: on a PD prefill server the first yield can precede completion, and
        # overlapping fake-bootstrap requests must not be batched together.
        async for _ in tokenizer_manager.generate_request(generate_req_input, None):
            pass


async def serving_shapes(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Compile the shape-specialized kernels real traffic hits, before readiness.

    Mix: the /health shape (1 token), odd short prompts, the long prompt aligned and
    with a 14-token tail, then concurrent 1K-token waves at every power-of-two batch
    size up to --max-running-requests. PD prefill: the prompts cut to one chunk, one
    request at a time. PD decode: fake-bootstrap decode waves on every DP rank.
    """
    max_running = get_schedule().max_running_requests or 64
    rng = np.random.default_rng(0)
    room = 1 << 40
    in_flight = max_in_flight = 0

    def make(
        num_tokens: int, max_new_tokens: int, dp_rank: Optional[int] = None
    ) -> GenerateReqInput:
        nonlocal room
        req = GenerateReqInput(
            input_ids=rng.integers(2**16, size=num_tokens).tolist(),
            sampling_params={
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            routed_dp_rank=dp_rank,
        )
        if disaggregation_mode != "null":
            room += 1
            req.bootstrap_room = room
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        return req

    async def run_all(reqs: List[GenerateReqInput]) -> None:
        async def run(req: GenerateReqInput) -> None:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                async for _ in tokenizer_manager.generate_request(req, None):
                    pass
            finally:
                in_flight -= 1

        await asyncio.gather(*(run(req) for req in reqs))

    tic = time.perf_counter()
    if disaggregation_mode == "decode":
        dp_size = get_parallel().dp_size
        per_rank = max(1, max_running // dp_size)
        bs = 1
        while bs <= per_rank:
            await run_all(
                [make(64, 8, dp_rank=r) for r in range(dp_size) for _ in range(bs)]
            )
            bs *= 2
        logger.info(
            f"serving_shapes warmup (decode): waves up to {bs // 2}/rank x "
            f"{dp_size} ranks, {time.perf_counter() - tic:.1f} s"
        )
        return

    long_tokens = min(_LONG_PREFILL_TOKENS, tokenizer_manager.context_len - 32)
    # Triton specializes integer arguments and pointers on divisibility by 16, and real
    # prompts are not 16-aligned: also warm odd lengths and a 14-token final chunk.
    lengths = [1, 17, 4099, 16397]
    if long_tokens > 0:
        lengths += [long_tokens, long_tokens + 14]
    lengths = one_chunk_lengths(
        lengths,
        disaggregation_mode=disaggregation_mode,
        chunked_prefill_size=get_schedule().chunked_prefill_size,
    )
    for num_tokens in lengths:
        await run_all([make(num_tokens, 1)])
    if disaggregation_mode == "null":
        bs = 1
        while bs <= max_running:
            await run_all([make(1024, 8) for _ in range(bs)])
            bs *= 2
    # A PD prefill server batches concurrent fake-bootstrap requests; keep one in flight.
    assert disaggregation_mode != "prefill" or max_in_flight == 1, max_in_flight
    logger.info(
        f"serving_shapes warmup ({disaggregation_mode}): prompts {lengths}, "
        f"max {max_in_flight} in flight, {time.perf_counter() - tic:.1f} s"
    )
