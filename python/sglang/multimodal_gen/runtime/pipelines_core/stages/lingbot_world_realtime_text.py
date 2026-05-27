# SPDX-License-Identifier: Apache-2.0
"""
LingBot-World realtime text stages.

The reference lingbot_fast_server initializes prompt embeddings once per session.
Cache text encoder outputs across realtime chunks so control sampling stays closer
to the actual denoising step.
"""

from __future__ import annotations

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
        self.cache_key: tuple[Any, ...] | None = None
        self.prompt_embeds: list[torch.Tensor] | None = None
        self.pooled_embeds: list[torch.Tensor] | None = None
        self.prompt_attention_mask: list[torch.Tensor] | None = None
        self.negative_prompt_embeds: list[torch.Tensor] | None = None
        self.neg_pooled_embeds: list[torch.Tensor] | None = None
        self.negative_attention_mask: list[torch.Tensor] | None = None
        self.event_config_key: tuple[Any, ...] | None = None
        self.event_outputs: dict[tuple[str, ...], _TextEncodingOutputs] = {}
        self.last_effective_cache_key: tuple[Any, ...] | None = None

    def clear_text_cache(self):
        self.cache_key = None
        self.prompt_embeds = None
        self.pooled_embeds = None
        self.prompt_attention_mask = None
        self.negative_prompt_embeds = None
        self.neg_pooled_embeds = None
        self.negative_attention_mask = None
        self.event_config_key = None
        self.event_outputs.clear()
        self.last_effective_cache_key = None

    def dispose(self):
        super().dispose()
        self.clear_text_cache()


class LingBotWorldRealtimeTextEncodingStage(TextEncodingStage):
    def _make_cache_key(self, batch: Req) -> tuple[Any, ...]:
        return (
            _normalize_prompt_value(batch.prompt),
            bool(batch.do_classifier_free_guidance),
            (
                _normalize_prompt_value(batch.negative_prompt)
                if batch.do_classifier_free_guidance
                else None
            ),
        )

    def _restore_cached_outputs(
        self, batch: Req, state: LingBotWorldRealtimeTextState
    ) -> Req:
        return self._restore_outputs(batch, self._outputs_from_state(state))

    def _store_outputs(self, batch: Req, state: LingBotWorldRealtimeTextState) -> None:
        state.prompt_embeds = _copy_tensor_list(batch.prompt_embeds)
        state.pooled_embeds = _copy_tensor_list(batch.pooled_embeds)
        state.prompt_attention_mask = _copy_tensor_list(batch.prompt_attention_mask)
        state.negative_prompt_embeds = _copy_tensor_list(batch.negative_prompt_embeds)
        state.neg_pooled_embeds = _copy_tensor_list(batch.neg_pooled_embeds)
        state.negative_attention_mask = _copy_tensor_list(batch.negative_attention_mask)

    def _outputs_from_state(
        self, state: LingBotWorldRealtimeTextState
    ) -> _TextEncodingOutputs:
        return _TextEncodingOutputs(
            prompt_embeds=state.prompt_embeds,
            pooled_embeds=state.pooled_embeds,
            prompt_attention_mask=state.prompt_attention_mask,
            negative_prompt_embeds=state.negative_prompt_embeds,
            neg_pooled_embeds=state.neg_pooled_embeds,
            negative_attention_mask=state.negative_attention_mask,
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
        return (
            base_cache_key,
            event_mode,
            tuple(events.items()),
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
        event_config_key = self._make_event_config_key(
            base_cache_key, events, event_mode
        )

        if state.cache_key != base_cache_key or state.prompt_embeds is None:
            state.clear_text_cache()
            base_outputs = self._encode_outputs(batch, server_args, batch.prompt)
            self._restore_outputs(batch, base_outputs)
            state.cache_key = base_cache_key
            self._store_outputs(batch, state)
        elif state.event_config_key != event_config_key:
            state.event_outputs.clear()

        if event_config_key is not None and state.event_config_key != event_config_key:
            for active_event_ids in self._iter_event_id_combinations(events):
                event_prompt = self._build_joined_event_prompt(events, active_event_ids)
                prompt_text = self._build_event_prompt(
                    batch.prompt, event_prompt, event_mode
                )
                state.event_outputs[active_event_ids] = self._encode_outputs(
                    batch, server_args, prompt_text
                )
            state.event_config_key = event_config_key
        elif event_config_key is None:
            state.event_config_key = None
            state.event_outputs.clear()

        active_event_ids = self._normalize_active_event_ids(batch, events)
        effective_outputs = self._outputs_from_state(state)
        effective_cache_key: tuple[Any, ...] = ("base", base_cache_key)
        if active_event_ids and event_config_key is not None:
            cached_event = state.event_outputs.get(active_event_ids)
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

        self._restore_outputs(batch, effective_outputs)
        batch.update_prompt_embeds = (
            state.last_effective_cache_key != effective_cache_key
        )
        state.last_effective_cache_key = effective_cache_key
        return batch
