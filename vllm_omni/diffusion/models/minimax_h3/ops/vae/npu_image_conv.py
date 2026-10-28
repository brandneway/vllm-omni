# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in causal T=1 Conv3D reduction for the H3 NPU image encoder."""

from __future__ import annotations

import ast
import hashlib
import inspect
import os
import textwrap
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGE_ENCODE = ContextVar("h3_npu_image_conv", default=False)
_ENV = "VLLM_OMNI_H3_NPU_IMAGE_T1_CONV"
_CACHE_ENV = "VLLM_OMNI_H3_NPU_IMAGE_T1_WEIGHT_CACHE"
# AST fingerprints of the checkpoint's BaseConv3d / SpatialParallelConv3d.
# Unknown remote implementations retain their own padding and forward methods.
_CONTRACT = (
    "9cee7416a7c5c67358b79e93776ba313f02bf8088c02550c7119fc37f228bef0",
    "21a7f96b4dbfdb169c3ddbc748dd64c1dc37a38cc739e49b475f86e0797efe73",
)


def _fingerprint(cls: type) -> str:
    def canonical(node):
        if isinstance(node, ast.AST):
            return (
                type(node).__name__,
                tuple(
                    (key, canonical(value)) for key, value in ast.iter_fields(node) if value is not None and value != []
                ),
            )
        if isinstance(node, list):
            return tuple(canonical(value) for value in node)
        return node

    source = textwrap.dedent(inspect.getsource(cls))
    return hashlib.sha256(repr(canonical(ast.parse(source))).encode()).hexdigest()


def _supported_module(module: nn.Module) -> bool:
    if type(module) is nn.Conv3d:
        return module.kernel_size[0] == 1 and module.padding == (0, 0, 0) and module.padding_mode == "zeros"
    cls = type(module)
    if cls.__name__ != "SpatialParallelConv3d" or cls.__bases__[0].__name__ != "BaseConv3d":
        return False
    try:
        return (_fingerprint(cls.__bases__[0]), _fingerprint(cls)) == _CONTRACT
    except (OSError, TypeError, SyntaxError):
        return False


@contextmanager
def image_encode_context() -> Iterator[None]:
    """The video and decoder paths never enter this scope."""
    token = _IMAGE_ENCODE.set(True)
    try:
        yield
    finally:
        _IMAGE_ENCODE.reset(token)


def _eligible(module: nn.Conv3d, x: torch.Tensor) -> bool:
    return (
        _IMAGE_ENCODE.get()
        and not torch.compiler.is_compiling()
        and not module.training
        and not torch.is_grad_enabled()
        and x.device.type == "npu"
        and x.dtype == torch.float32
        and x.ndim == 5
        and x.shape[2] == 1
        and module.weight.device == x.device
        and module.weight.dtype == x.dtype
        and (module.bias is None or (module.bias.device == x.device and module.bias.dtype == x.dtype))
        and module.groups == 1
        and module.dilation == (1, 1, 1)
        and not getattr(module, "spatial_parallel", False)
        and (
            (
                type(module) is nn.Conv3d
                and module.kernel_size[0] == 1
                and module.padding == (0, 0, 0)
                and module.padding_mode == "zeros"
            )
            or (
                module.causal
                and module.pad_mode_t == "constant"
                and module.kernel_size[0] in (1, 3)
                and module.padding[0] == module.kernel_size[0] // 2
                and module.pad_mode in ("constant", "reflect", "replicate")
            )
        )
    )


def _clear_weight(module: nn.Module, *args) -> None:
    module._omni_image_conv_weight = None


def clear_npu_image_conv_cache(model: nn.Module) -> None:
    """Release derived encoder weights before H3 residency storage is rebound.

    Also call before edits through .data or external aliases, which can bypass
    PyTorch's version counter. No parameter or state_dict entry is added.
    """
    for part in ("encoder", "quant_conv"):
        root = getattr(model, part, None)
        if isinstance(root, nn.Module):
            for module in root.modules():
                if getattr(module, "_omni_image_conv_cache_enabled", False):
                    _clear_weight(module)


def _apply(self: nn.Module, fn: Callable, recurse: bool = True):
    _clear_weight(self)
    return self._omni_image_conv_apply(fn, recurse=recurse)


def _weight_key(weight: torch.Tensor) -> tuple | None:
    try:
        version = weight._version
    except RuntimeError:
        # Inference tensors cannot report mutations: use the live slice.
        return None
    if weight.device.type == "cpu":
        stream = None
    else:
        backend = torch.get_device_module(weight.device)
        current = backend.current_stream(weight.device)
        stream = getattr(current, "npu_stream", None)
        if stream is None:
            return None
    return (
        version,
        weight.data_ptr(),
        weight.untyped_storage()._cdata,
        weight.device,
        weight.dtype,
        tuple(weight.shape),
        tuple(weight.stride()),
        stream,
    )


def _weight(module: nn.Conv3d) -> torch.Tensor:
    source = module.weight
    if not getattr(module, "_omni_image_conv_cache_enabled", False) or source.shape[2] == 1:
        return source[:, :, -1, :, :].contiguous().unsqueeze(2)
    key = _weight_key(source)
    cached = module._omni_image_conv_weight
    if key is not None and cached is not None and cached[0] is source and cached[1] == key:
        return cached[2]
    _clear_weight(module)
    packed = source[:, :, -1, :, :].contiguous().unsqueeze(2)
    if key is not None:
        module._omni_image_conv_weight = (source, key, packed)
    return packed


def _forward(self: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    if not _eligible(self, x):
        if getattr(self, "_omni_image_conv_cache_enabled", False):
            _clear_weight(self)
        return self._omni_image_conv_forward(x)
    # Keep the Conv3D backend and its current HF32 policy. For causal T=1,
    # preceding temporal slices multiply padded zeros. The optional cache
    # reuses only this exact contiguous slice within one residency interval.
    ph, pw = self.padding[1:]
    if ph or pw:
        x = F.pad(x.squeeze(2), (pw, pw, ph, ph), mode=self.pad_mode).unsqueeze(2)
    weight = _weight(self)
    return F.conv3d(x, weight, self.bias, self.stride, 0, self.dilation, self.groups)


def install_npu_image_conv(model: nn.Module, *, device: torch.device) -> int:
    """Install once on encoder/quant_conv only; default OFF on every device."""
    if device.type != "npu" or os.environ.get(_ENV, "0") != "1":
        return 0
    installed = 0
    for part in ("encoder", "quant_conv"):
        root = getattr(model, part, None)
        if not isinstance(root, nn.Module):
            continue
        for module in root.modules():
            if not isinstance(module, nn.Conv3d) or not _supported_module(module):
                continue
            if not hasattr(module, "_omni_image_conv_forward"):
                if "forward" in module.__dict__:
                    continue
                module._omni_image_conv_forward = module.forward
                module.forward = MethodType(_forward, module)
            if os.environ.get(_CACHE_ENV, "0") == "1" and not getattr(module, "_omni_image_conv_cache_enabled", False):
                module._omni_image_conv_cache_enabled = True
                module._omni_image_conv_weight = None
                module._omni_image_conv_apply = module._apply
                module._apply = MethodType(_apply, module)
                module.register_load_state_dict_pre_hook(_clear_weight)
            installed += 1
    return installed
