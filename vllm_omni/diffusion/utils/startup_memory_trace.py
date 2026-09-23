# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stage-wise accelerator memory tracing for diffusion model startup.

Opt-in via ``VLLM_OMNI_DIFFUSION_STARTUP_MEM_TRACE=1``. Every line reports
allocated/reserved/free/peak for the loader device so a startup OOM can be
attributed to the construction window (``after_construct``), the per-layer
online-quantization stream (``online_quant_layers``), the weight-load tail
(``after_load_empty_cache`` / ``after_load_finalize``), or the startup
profile run (``before_profile_run`` / ``after_profile_run``).

Disabled by default: with the env unset, ``trace_startup_memory`` is a
no-op that never touches a memory API.
"""

import os
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_TRACE_ENV = "VLLM_OMNI_DIFFUSION_STARTUP_MEM_TRACE"
_TRACE_LAYERS_ENV = "VLLM_OMNI_DIFFUSION_STARTUP_MEM_TRACE_LAYERS"
_SKIP_POST_LOAD_EMPTY_CACHE_ENV = "VLLM_OMNI_DIFFUSION_SKIP_POST_LOAD_EMPTY_CACHE"
_STAGGER_ENV = "VLLM_OMNI_DIFFUSION_STAGGER_COMPONENT_LOAD"

_DEFAULT_LAYER_INTERVAL = 10
_GiB = 1024 * 1024 * 1024

_online_quant_layer_count = 0


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def startup_mem_trace_enabled() -> bool:
    return _env_flag(_TRACE_ENV)


def skip_post_load_empty_cache() -> bool:
    """Escape hatch for the loader's post-load ``empty_cache``."""
    return _env_flag(_SKIP_POST_LOAD_EMPTY_CACHE_ENV)


def stagger_component_load_enabled() -> bool:
    """Whether encoders/VAEs are staged to the host during the weight load."""
    return _env_flag(_STAGGER_ENV)


def trace_layer_interval() -> int:
    raw = os.environ.get(_TRACE_LAYERS_ENV, "").strip()
    if not raw:
        return _DEFAULT_LAYER_INTERVAL
    try:
        interval = int(raw)
    except ValueError:
        return _DEFAULT_LAYER_INTERVAL
    return interval if interval > 0 else _DEFAULT_LAYER_INTERVAL


def _gib(num_bytes: int | float) -> float:
    return float(num_bytes) / _GiB


def _memory_module(device: torch.device | None):
    """Pick the memory-stat module for the device.

    ``torch.accelerator`` memory stats assert on NPU (its allocator is not a
    torch ``DeviceAllocator``, hard-failing under expandable_segments), so
    NPU reads go through ``torch.npu``; CUDA and others keep the generic API.
    """
    from vllm_omni.platforms import current_omni_platform

    device_type = getattr(device, "type", None) or getattr(current_omni_platform, "device_type", None)
    if device_type == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "memory_allocated"):
        return torch.npu
    return torch.accelerator


def _accelerator_memory(device: torch.device | None) -> tuple[int, int, int]:
    """Return (allocated, reserved, peak-allocated) in bytes.

    ``max_memory_allocated`` postdates the other two on some accelerators;
    a missing API degrades the peak reading to 0 instead of failing the run.
    """
    module = _memory_module(device)
    allocated = int(module.memory_allocated(device))
    reserved = int(module.memory_reserved(device))
    max_allocated = getattr(module, "max_memory_allocated", None)
    peak = int(max_allocated(device)) if callable(max_allocated) else 0
    return allocated, reserved, peak


def trace_startup_memory(
    stage: str,
    *,
    extra: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> None:
    """Log one stage snapshot; no-op unless the trace env is enabled."""
    if not startup_mem_trace_enabled():
        return
    from vllm_omni.platforms import current_omni_platform

    allocated, reserved, peak = _accelerator_memory(device)
    free, _total = current_omni_platform.get_device_memory(device)
    suffix = ""
    if extra:
        suffix = " " + " ".join(f"{key}={value}" for key, value in extra.items())
    logger.info(
        "[startup-mem] stage=%s rank=%s alloc=%.2fGiB reserved=%.2fGiB free=%.2fGiB peak=%.2fGiB%s",
        stage,
        os.environ.get("LOCAL_RANK", "0"),
        _gib(allocated),
        _gib(reserved),
        _gib(free),
        _gib(peak),
        suffix,
    )


def note_online_quant_layer(device: torch.device | None = None) -> None:
    """Count one finished online-quantized layer; trace every N layers."""
    global _online_quant_layer_count
    if not startup_mem_trace_enabled():
        return
    _online_quant_layer_count += 1
    interval = trace_layer_interval()
    if _online_quant_layer_count % interval:
        return
    trace_startup_memory(
        "online_quant_layers",
        extra={"layers": _online_quant_layer_count},
        device=device,
    )


def reset_online_quant_layer_count_for_tests() -> None:
    global _online_quant_layer_count
    _online_quant_layer_count = 0
