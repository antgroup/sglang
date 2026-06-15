# SPDX-License-Identifier: Apache-2.0
"""
LingBot-World realtime text stages.

The reference lingbot_fast_server initializes prompt embeddings once per session.
Cache text encoder outputs across realtime chunks so control sampling stays closer
to the actual denoising step.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core.realtime_session import (
    BaseRealtimeState,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _normalize_prompt_value(
    value: str | list[str] | None,
) -> str | tuple[str, ...] | None:
    if isinstance(value, list):
        return tuple(value)
    return value


def _copy_tensor_list(
    value: list[torch.Tensor] | None,
) -> list[torch.Tensor] | None:
    if value is None:
        return None
    return list(value)


@dataclass
class _TextEncodingOutputs:
    prompt_embeds: list[torch.Tensor] | None = None
    pooled_embeds: list[torch.Tensor] | None = None
    prompt_attention_mask: list[torch.Tensor] | None = None
    negative_prompt_embeds: list[torch.Tensor] | None = None
    neg_pooled_embeds: list[torch.Tensor] | None = None
    negative_attention_mask: list[torch.Tensor] | None = None


class LingBotWorldRealtimeTextState(BaseRealtimeState):
    def __init__(self):
        super().__init__()
        self.base_outputs: dict[tuple[Any, ...], _TextEncodingOutputs] = {}
        self.event_outputs: dict[
            tuple[Any, ...], dict[tuple[str, ...], _TextEncodingOutputs]
        ] = {}
        self.prompt_variant_keys: tuple[tuple[Any, ...], ...] | None = None
        self.last_effective_cache_key: tuple[Any, ...] | None = None

    def clear_text_cache(self):
        self.base_outputs.clear()
        self.event_outputs.clear()
        self.prompt_variant_keys = None
        self.last_effective_cache_key = None

    def dispose(self):
        super().dispose()
        self.clear_text_cache()


class LingBotWorldRealtimeTextEncodingStage(TextEncodingStage):
    def _make_cache_key(self, batch: Req) -> tuple[Any, ...]:
        return self._make_cache_key_for_prompt(batch, batch.prompt)

    def _make_cache_key_for_prompt(
        self, batch: Req, prompt: str | list[str]
    ) -> tuple[Any, ...]:
        return (
            _normalize_prompt_value(prompt),
            bool(batch.do_classifier_free_guidance),
            (
                _normalize_prompt_value(batch.negative_prompt)
                if batch.do_classifier_free_guidance
                else None
            ),
        )

    def _restore_outputs(self, batch: Req, outputs: _TextEncodingOutputs) -> Req:
        batch.prompt_embeds = _copy_tensor_list(outputs.prompt_embeds) or []
        batch.pooled_embeds = _copy_tensor_list(outputs.pooled_embeds) or []
        batch.prompt_attention_mask = _copy_tensor_list(outputs.prompt_attention_mask)
        batch.negative_prompt_embeds = _copy_tensor_list(outputs.negative_prompt_embeds)
        batch.neg_pooled_embeds = _copy_tensor_list(outputs.neg_pooled_embeds) or []
        batch.negative_attention_mask = _copy_tensor_list(
            outputs.negative_attention_mask
        )
        return batch

    def _normalize_prompt_variants(self, batch: Req) -> list[str | list[str]]:
        prompts: list[str | list[str]] = []
        raw_variants = batch.extra.get("movement_prompt_variants")
        if isinstance(raw_variants, list):
            for raw_prompt in raw_variants:
                if isinstance(raw_prompt, str) and raw_prompt not in prompts:
                    prompts.append(raw_prompt)
        if batch.prompt not in prompts:
            prompts.insert(0, batch.prompt)
        return prompts

    def _ensure_base_outputs(
        self,
        batch: Req,
        server_args: ServerArgs,
        state: LingBotWorldRealtimeTextState,
        prompt: str | list[str],
    ) -> tuple[tuple[Any, ...], _TextEncodingOutputs]:
        cache_key = self._make_cache_key_for_prompt(batch, prompt)
        outputs = state.base_outputs.get(cache_key)
        if outputs is None:
            outputs = self._encode_outputs(batch, server_args, prompt)
            state.base_outputs[cache_key] = outputs
        return cache_key, outputs

    def _encode_outputs(
        self, batch: Req, server_args: ServerArgs, prompt_text: str | list[str]
    ) -> _TextEncodingOutputs:
        all_indices: list[int] = list(range(len(self.text_encoders)))
        prompt_embeds, prompt_masks, pooled_embeds = self.encode_text(
            prompt_text,
            server_args,
            encoder_index=all_indices,
            return_attention_mask=True,
        )

        negative_prompt_embeds = []
        negative_attention_mask = None
        neg_pooled_embeds = []
        if batch.do_classifier_free_guidance:
            assert isinstance(batch.negative_prompt, str)
            negative_prompt_embeds, negative_attention_mask, neg_pooled_embeds = (
                self.encode_text(
                    batch.negative_prompt,
                    server_args,
                    encoder_index=all_indices,
                    return_attention_mask=True,
                )
            )

        return _TextEncodingOutputs(
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
            prompt_attention_mask=prompt_masks,
            negative_prompt_embeds=negative_prompt_embeds,
            neg_pooled_embeds=neg_pooled_embeds,
            negative_attention_mask=negative_attention_mask,
        )

    def _normalize_prompt_events(self, batch: Req) -> dict[str, str]:
        events = batch.extra.get("prompt_events")
        if not isinstance(events, dict):
            return {}
        return {
            str(event_id): str(prompt)
            for event_id, prompt in events.items()
            if event_id is not None and prompt is not None and str(event_id)
        }

    def _build_event_prompt(
        self,
        base_prompt: str | list[str],
        event_prompt: str,
        mode: str,
    ) -> str | list[str]:
        if mode == "append":
            if isinstance(base_prompt, list):
                return [f"{prompt}\n{event_prompt}" for prompt in base_prompt]
            return f"{base_prompt}\n{event_prompt}"
        if isinstance(base_prompt, list):
            return [event_prompt for _ in base_prompt]
        return event_prompt

    def _normalize_active_event_ids(
        self, batch: Req, events: dict[str, str]
    ) -> tuple[str, ...]:
        raw_event_ids = batch.extra.get("prompt_event_ids")
        if raw_event_ids is None:
            raw_event_ids = batch.extra.get("prompt_event_id")
        if raw_event_ids is None:
            return ()
        if isinstance(raw_event_ids, (list, tuple)):
            values = raw_event_ids
        else:
            values = [raw_event_ids]

        active_event_ids: list[str] = []
        seen_event_ids: set[str] = set()
        for raw_event_id in values:
            if raw_event_id is None:
                continue
            event_id = str(raw_event_id).strip()
            if not event_id or event_id not in events or event_id in seen_event_ids:
                continue
            active_event_ids.append(event_id)
            seen_event_ids.add(event_id)
        return tuple(active_event_ids)

    def _build_joined_event_prompt(
        self, events: dict[str, str], event_ids: tuple[str, ...]
    ) -> str:
        return "\n".join(events[event_id] for event_id in event_ids)

    def _iter_event_id_combinations(
        self, events: dict[str, str]
    ) -> tuple[tuple[str, ...], ...]:
        event_ids = tuple(events.keys())
        return tuple(
            event_id_combo
            for combo_size in range(1, len(event_ids) + 1)
            for event_id_combo in combinations(event_ids, combo_size)
        )

    def _make_event_config_key(
        self,
        base_cache_key: tuple[Any, ...],
        events: dict[str, str],
        event_mode: str,
    ) -> tuple[Any, ...] | None:
        if not events:
            return None
        if event_mode == "append":
            return (
                event_mode,
                base_cache_key,
                tuple(events.items()),
            )
        return (
            event_mode,
            base_cache_key[1:],
            tuple(events.items()),
        )

    def _ensure_event_outputs(
        self,
        batch: Req,
        server_args: ServerArgs,
        state: LingBotWorldRealtimeTextState,
        *,
        base_cache_key: tuple[Any, ...],
        base_prompt: str | list[str],
        events: dict[str, str],
        event_mode: str,
    ) -> tuple[Any, ...] | None:
        event_config_key = self._make_event_config_key(
            base_cache_key, events, event_mode
        )
        if event_config_key is None or event_config_key in state.event_outputs:
            return event_config_key

        event_outputs: dict[tuple[str, ...], _TextEncodingOutputs] = {}
        for active_event_ids in self._iter_event_id_combinations(events):
            event_prompt = self._build_joined_event_prompt(events, active_event_ids)
            prompt_text = self._build_event_prompt(
                base_prompt, event_prompt, event_mode
            )
            event_outputs[active_event_ids] = self._encode_outputs(
                batch, server_args, prompt_text
            )
        state.event_outputs[event_config_key] = event_outputs
        return event_config_key

    def _log_effective_prompt(
        self,
        batch: Req,
        *,
        active_event_ids: tuple[str, ...],
        events: dict[str, str],
        event_mode: str,
    ) -> None:
        if os.environ.get("SGLANG_LOG_FULL_PROMPT", "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return

        prompt: str | list[str] = batch.prompt
        if active_event_ids:
            event_prompt = self._build_joined_event_prompt(events, active_event_ids)
            prompt = self._build_event_prompt(batch.prompt, event_prompt, event_mode)

        logger.info(
            "LingBot effective prompt: session_id=%s block_idx=%s "
            "event_mode=%s event_ids=%s prompt=%r",
            batch.extra.get("realtime_session_id"),
            batch.block_idx,
            event_mode,
            list(active_event_ids),
            prompt,
        )

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        if batch.session is None:
            return super().forward(batch, server_args)

        state = batch.session.get_or_create_state(LingBotWorldRealtimeTextState)
        assert isinstance(state, LingBotWorldRealtimeTextState)

        base_cache_key = self._make_cache_key(batch)
        events = self._normalize_prompt_events(batch)
        event_mode = str(batch.extra.get("prompt_event_mode") or "overwrite")
        if event_mode not in ("overwrite", "append"):
            event_mode = "overwrite"

        prompt_variants = self._normalize_prompt_variants(batch)
        prompt_variant_keys = tuple(
            self._make_cache_key_for_prompt(batch, prompt) for prompt in prompt_variants
        )
        if (
            state.prompt_variant_keys is not None
            and state.prompt_variant_keys != prompt_variant_keys
        ):
            state.clear_text_cache()
        state.prompt_variant_keys = prompt_variant_keys

        base_outputs = None
        for prompt in prompt_variants:
            prompt_cache_key, prompt_outputs = self._ensure_base_outputs(
                batch, server_args, state, prompt
            )
            self._ensure_event_outputs(
                batch,
                server_args,
                state,
                base_cache_key=prompt_cache_key,
                base_prompt=prompt,
                events=events,
                event_mode=event_mode,
            )
            if prompt_cache_key == base_cache_key:
                base_outputs = prompt_outputs

        if base_outputs is None:
            _, base_outputs = self._ensure_base_outputs(
                batch, server_args, state, batch.prompt
            )

        event_config_key = self._make_event_config_key(
            base_cache_key, events, event_mode
        )

        active_event_ids = self._normalize_active_event_ids(batch, events)
        effective_outputs = base_outputs
        effective_cache_key: tuple[Any, ...] = ("base", base_cache_key)
        if active_event_ids and event_config_key is not None:
            cached_event = state.event_outputs.get(event_config_key, {}).get(
                active_event_ids
            )
            if cached_event is None:
                batch.extra.pop("prompt_event_id", None)
                batch.extra.pop("prompt_event_ids", None)
                active_event_ids = ()
                cached_event = None

            if cached_event is not None:
                effective_outputs = cached_event
                effective_cache_key = ("events", event_config_key, active_event_ids)
                batch.extra["active_prompt_event_ids"] = list(active_event_ids)
                if len(active_event_ids) == 1:
                    batch.extra["active_prompt_event_id"] = active_event_ids[0]

        self._log_effective_prompt(
            batch,
            active_event_ids=active_event_ids,
            events=events,
            event_mode=event_mode,
        )
        self._restore_outputs(batch, effective_outputs)
        batch.update_prompt_embeds = (
            state.last_effective_cache_key != effective_cache_key
        )
        state.last_effective_cache_key = effective_cache_key
        return batch
