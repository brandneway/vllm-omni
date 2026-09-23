# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2024 xDiT team.
# Adapted from
# https://github.com/xdit-project/xDiT/blob/main/xfuser/envs.py
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    MASTER_ADDR: str = ""
    MASTER_PORT: int | None = None
    CUDA_HOME: str | None = None
    LOCAL_RANK: int = 0
    VLLM_OMNI_DIFFUSION_SKIP_POST_LOAD_EMPTY_CACHE: str | None = None
    VLLM_OMNI_MINIMAX_H3_STAGED_COMPONENTS: str | None = None
    VLLM_OMNI_DIFFUSION_STAGGER_COMPONENT_LOAD: str | None = None
    VLLM_OMNI_DIFFUSION_STARTUP_MEM_TRACE: str | None = None
    VLLM_OMNI_DIFFUSION_STARTUP_MEM_TRACE_LAYERS: str | None = None
    VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS: str | None = None

environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Runtime Env Vars ==================
    # used in distributed environment to determine the master address
    "MASTER_ADDR": lambda: os.getenv("MASTER_ADDR", ""),
    # used in distributed environment to manually set the communication port
    "MASTER_PORT": lambda: int(os.getenv("MASTER_PORT", "0")) if "MASTER_PORT" in os.environ else None,
    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME": lambda: os.environ.get("CUDA_HOME", None),
    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK": lambda: int(os.environ.get("LOCAL_RANK", "0")),
    # Minimum rotary-table span (B*S positions) at which consumers of the shared
    # fused_qk_norm_rope op take the fused path instead of their eager chain;
    # "0" = always fuse; unset = each consumer's own measured default. Raw
    # string or None; validated by fused_qk_norm_rope_min_tokens().
    "VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS": lambda: os.environ.get("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", None),
    # Escape hatch keeping the caching-allocator pages freed by the online
    # quantization stream (reusable bf16 workspaces) after model load; set "1"
    # to skip the post-load empty_cache on the loader device.
    "VLLM_OMNI_DIFFUSION_SKIP_POST_LOAD_EMPTY_CACHE": lambda: os.environ.get(
        "VLLM_OMNI_DIFFUSION_SKIP_POST_LOAD_EMPTY_CACHE", None
    ),
    # Set "1" to stage the pipeline's encoders/VAEs on the host while the DiT
    # online-quantization weight stream runs, restoring them once the load
    # finishes (startup-load peak reduction; final residency unchanged).
    "VLLM_OMNI_DIFFUSION_STAGGER_COMPONENT_LOAD": lambda: os.environ.get(
        "VLLM_OMNI_DIFFUSION_STAGGER_COMPONENT_LOAD", None
    ),
    # Set "1" to log stage-wise accelerator memory snapshots during diffusion
    # model startup (construction, per-layer online quantization, load tail,
    # startup profile run).
    "VLLM_OMNI_DIFFUSION_STARTUP_MEM_TRACE": lambda: os.environ.get("VLLM_OMNI_DIFFUSION_STARTUP_MEM_TRACE", None),
    # Every N finished online-quantized layers, emit one startup-mem line
    # (default 10 when the trace is enabled).
    "VLLM_OMNI_DIFFUSION_STARTUP_MEM_TRACE_LAYERS": lambda: os.environ.get(
        "VLLM_OMNI_DIFFUSION_STARTUP_MEM_TRACE_LAYERS", None
    ),
    # Comma-separated MiniMax-H3 components ("te", "vae") that wait in host
    # memory and only reach the device inside the pipeline phase using them;
    # empty keeps every component device-resident.
    "VLLM_OMNI_MINIMAX_H3_STAGED_COMPONENTS": lambda: os.environ.get("VLLM_OMNI_MINIMAX_H3_STAGED_COMPONENTS", None),
}


class PackagesEnvChecker:
    """Singleton class for checking package availability."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        packages_info = {}
        packages_info["has_flash_attn"] = self._check_flash_attn()
        self.packages_info = packages_info

    def _check_flash_attn(self) -> bool:
        """Check if flash attention is available and compatible."""
        platform = current_omni_platform

        if platform.get_device_count() == 0:
            return False

        return platform.has_flash_attn_package()

    def get_packages_info(self) -> dict:
        """Get the packages info dictionary."""
        return self.packages_info


PACKAGES_CHECKER = PackagesEnvChecker()


def __getattr__(name):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
