# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in GroupNorm/SiLU fusion for the H3 NPU image encoder."""

from __future__ import annotations

import ast
import hashlib
import inspect
import os
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import FunctionType, MethodType

import torch
import torch.nn as nn

_IMAGE_ENCODE = ContextVar("h3_npu_image_norm", default=False)
_ENV = "VLLM_OMNI_H3_NPU_IMAGE_GN_SILU"
_NORM_SILU_CONTRACT = "d416a2d66d3cddb5fb29f40de73def681c60d4facbf2968686292f8f3afb5020"
_NORM_CONTRACTS = {
    "SpatialParallelGroupNorm": "f67585e91435818be8c5726949818ccbaf84de472411ef5e03271004be038c8b",
    "TemporalIsolatedSpatialParallelGroupNorm": "e1090051af31bb803a16da813ce75634af2243a675d5b4eda7ef0deb3078c347",
}


def _fingerprint(target) -> str:
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

    source = textwrap.dedent(inspect.getsource(target))
    return hashlib.sha256(repr(canonical(ast.parse(source))).encode()).hexdigest()


def _supported_norm(norm: nn.Module) -> bool:
    if type(norm) is nn.GroupNorm:
        return True
    expected = _NORM_CONTRACTS.get(type(norm).__name__)
    if expected is None:
        return False
    try:
        actual = _fingerprint(type(norm))
        if actual != expected:
            return False
        # The temporal-isolated subclass delegates to this parent at T=1.
        if type(norm).__name__ == "TemporalIsolatedSpatialParallelGroupNorm":
            parent = type(norm).__bases__[0]
            return _fingerprint(parent) == _NORM_CONTRACTS.get(parent.__name__)
        return True
    except (OSError, TypeError, SyntaxError):
        return False


@contextmanager
def image_encode_context() -> Iterator[None]:
    token = _IMAGE_ENCODE.set(True)
    try:
        yield
    finally:
        _IMAGE_ENCODE.reset(token)


def _eligible(x: torch.Tensor, norm: nn.Module, cond: torch.Tensor | None) -> bool:
    return (
        _IMAGE_ENCODE.get()
        and not torch.compiler.is_compiling()
        and cond is None
        and not norm.training
        and not torch.is_grad_enabled()
        and x.device.type == "npu"
        and x.dtype == torch.float32
        and x.ndim == 5
        and x.shape[2] == 1
        and x.is_contiguous()
        and getattr(norm, "_omni_image_norm_supported", False)
        and not norm._forward_hooks
        and not norm._forward_pre_hooks
        and "forward" not in norm.__dict__
        and not nn.modules.module._global_forward_hooks
        and not nn.modules.module._global_forward_pre_hooks
        and not getattr(norm, "spatial_parallel", False)
        and norm.weight is not None
        and norm.bias is not None
        and norm.weight.device == x.device
        and norm.bias.device == x.device
        and norm.weight.dtype == x.dtype
        and norm.bias.dtype == x.dtype
        and norm.eps > 0
        and x.shape[1] == norm.num_channels
        and x.numel() > 0
    )


def install_npu_image_norm(model: nn.Module, *, device: torch.device) -> int:
    """Bind private forward namespaces; never mutate the remote-code globals."""
    encoder = getattr(model, "encoder", None)
    if device.type != "npu" or os.environ.get(_ENV, "0") != "1" or not isinstance(encoder, nn.Module):
        return 0
    import torch_npu

    fused = getattr(torch_npu, "npu_group_norm_silu", None)
    if not callable(fused):
        return 0
    for norm in encoder.modules():
        if isinstance(norm, nn.GroupNorm):
            norm._omni_image_norm_supported = _supported_norm(norm)
    installed = 0
    for module in encoder.modules():
        if getattr(module, "_omni_image_norm_installed", False):
            installed += 1
            continue
        # These are the two official CNN call sites. Preserve their complete
        # forward body, including checkpointing, residual and downsample order.
        if type(module).__name__ not in ("ResnetBlock3D", "EncoderFCN3D"):
            continue
        forward = type(module).forward
        if not isinstance(forward, FunctionType) or forward.__closure__ or "forward" in module.__dict__:
            continue
        original = forward.__globals__.get("norm_silu")
        if not isinstance(original, FunctionType):
            continue
        try:
            signature = _fingerprint(original)
        except (OSError, TypeError, SyntaxError):
            continue
        if signature != _NORM_SILU_CONTRACT:
            continue

        def norm_silu(
            x: torch.Tensor, norm: nn.Module, cond: torch.Tensor | None = None, _original: FunctionType = original
        ) -> torch.Tensor:
            if _eligible(x, norm, cond):
                return fused(x, norm.weight, norm.bias, norm.num_groups, norm.eps)[0]
            return _original(x, norm, cond)

        namespace = dict(forward.__globals__, norm_silu=norm_silu)
        private_forward = FunctionType(forward.__code__, namespace, forward.__name__, forward.__defaults__)
        private_forward.__kwdefaults__ = forward.__kwdefaults__
        module.forward = MethodType(private_forward, module)
        module._omni_image_norm_installed = True
        installed += 1
    return installed
