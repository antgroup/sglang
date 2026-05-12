# SPDX-License-Identifier: Apache-2.0
"""Small model-loading helpers for the vendored FlashVSR Tiny Long modules."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager

import torch


@contextmanager
def init_weights_on_device(
    device: torch.device = torch.device("meta"), include_buffers: bool = False
):
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs
            )

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer


def _state_dict_keys_to_single_str(state_dict, with_shape: bool = True) -> str:
    keys = []
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, torch.Tensor):
            if with_shape:
                shape = "_".join(map(str, list(value.shape)))
                keys.append(f"{key}:{shape}")
            keys.append(key)
        elif isinstance(value, dict):
            nested = _state_dict_keys_to_single_str(value, with_shape=with_shape)
            keys.append(f"{key}|{nested}")
    keys.sort()
    return ",".join(keys)


def hash_state_dict_keys(state_dict, with_shape: bool = True) -> str:
    keys_str = _state_dict_keys_to_single_str(state_dict, with_shape=with_shape).encode(
        "UTF-8"
    )
    return hashlib.md5(keys_str).hexdigest()
