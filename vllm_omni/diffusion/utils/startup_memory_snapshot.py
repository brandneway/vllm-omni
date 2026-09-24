# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in memory snapshots for diffusion model startup.

Two collectors run side by side, each behind its own environment variable, so a
startup whose residency is split between the accelerator and the host (DLO pins
whole weight shards on the host and leaves only a double buffer on device) can
be attributed from both ends:

* ``VLLM_OMNI_DIFFUSION_STARTUP_MEM_SNAPSHOT`` -- accelerator pickles written by
  the platform memory module's ``_dump_snapshot`` (``torch.npu.memory`` on NPU).
  These are the artifacts the NPU snapshot analyzer consumes.
* ``VLLM_OMNI_DIFFUSION_STARTUP_MEM_SNAPSHOT_HOST`` -- host accounting as JSON:
  ``/proc`` RSS/PSS totals, a census of every live CPU tensor storage, and a
  per-component host/device residency split. ``_dump_snapshot`` cannot see host
  weight shards, so this is the only view that covers them.

Both accept either a truthy flag (capture every stage in
:data:`_DEFAULT_STAGES`) or a comma-separated stage list (capture only those).
The stage list is the disk guard: one accelerator pickle runs ~100MiB, so a
rank-per-stage capture of every stage on eight ranks is several GiB.

``VLLM_OMNI_DIFFUSION_STARTUP_MEM_SNAPSHOT_DIR`` picks the output directory for
both collectors (default ``startup_mem_snapshots`` under the working directory).

Disabled by default: with the env unset, :func:`capture_startup_snapshot` never
touches a memory API or the filesystem. Capture is bounded to startup -- the
recorder is stopped at the end of the model load, so serving pays nothing.
"""

import gc
import inspect
import itertools
import json
import os
import time
import warnings
from collections import Counter
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEVICE_ENV = "VLLM_OMNI_DIFFUSION_STARTUP_MEM_SNAPSHOT"
_HOST_ENV = "VLLM_OMNI_DIFFUSION_STARTUP_MEM_SNAPSHOT_HOST"
_DIR_ENV = "VLLM_OMNI_DIFFUSION_STARTUP_MEM_SNAPSHOT_DIR"

_DEFAULT_DIR = "startup_mem_snapshots"
_TRUE_VALUES = ("1", "true", "yes", "on")

# Recording every allocation with a Python stack has a real cost, so the
# history is capped rather than left to grow with the online-quantization stream.
_MAX_ENTRIES = 100000

# Stages a bare "1" captures. `online_quant_layers` is deliberately absent: the
# tracer fires it every N finished layers, and a ~100MiB pickle per firing would
# bury the disks. Naming it explicitly still works.
_DEFAULT_STAGES = frozenset(
    {
        "after_construct",
        "after_component_stage_off",
        "after_load_empty_cache",
        "after_component_restore",
        "after_load_finalize",
        "after_load",
        "after_offload_enable",
        "startup_complete",
    }
)

_STATUS_KEYS = ("VmRSS", "VmHWM", "VmPeak", "VmSwap")
_SMAPS_KEYS = (
    "Pss",
    "Private_Dirty",
    "Private_Clean",
    "Shared_Clean",
    "Anonymous",
    "AnonHugePages",
)

_MiB = 1024 * 1024

_device_state: dict[str, Any] = {"enabled": False, "module": None, "backend": None, "seq": 0}
_host_seq = 0


def _selected_stages(env_name: str) -> frozenset[str]:
    """Stages this collector should capture; empty when it is switched off."""
    raw = os.environ.get(env_name)
    if raw is None:
        return frozenset()
    value = raw.strip()
    if not value:
        return frozenset()
    if value.lower() in _TRUE_VALUES:
        return _DEFAULT_STAGES
    return frozenset(part.strip() for part in value.split(",") if part.strip())


def _output_dir() -> str:
    return os.environ.get(_DIR_ENV, "").strip() or _DEFAULT_DIR


def _rank_tag(device: torch.device | None) -> str:
    """Prefer the device's own index (remapped ranks differ from LOCAL_RANK)."""
    index = getattr(device, "index", None)
    if isinstance(index, int):
        return str(index)
    return os.environ.get("LOCAL_RANK", "0")


def _output_path(device: torch.device | None, seq: int, stage: str, suffix: str) -> str:
    directory = _output_dir()
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"rank{_rank_tag(device)}_{seq:02d}_{stage}{suffix}")


def _accepts_keyword(func: Any, name: str) -> bool:
    """Whether ``func`` takes ``name``.

    Platform builds differ: the NPU wrapper drops ``clear_history``, which the
    CUDA path gained in torch 2.7. Passing it blind aborts worker startup, so a
    signature we cannot read means "do not pass it".
    """
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    if name in parameters:
        return True
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def _memory_history_module() -> tuple[str | None, Any | None]:
    """Resolve the accelerator memory module that carries history/snapshot APIs."""
    backends = (
        ("NPU", getattr(torch, "npu", None)),
        ("CUDA", getattr(torch, "cuda", None)),
    )
    for backend_name, device_module in backends:
        if device_module is None:
            continue
        is_available = getattr(device_module, "is_available", None)
        if callable(is_available) and not is_available():
            continue
        memory_module = getattr(device_module, "memory", None)
        if memory_module is not None:
            return backend_name, memory_module
    return None, None


def enable_startup_device_snapshot(device: torch.device | None = None) -> None:
    """Start recording accelerator memory history; no-op unless enabled by env.

    Must run before the first allocation worth attributing -- the model runner
    and the weight load both allocate after device init.
    """
    stages = _selected_stages(_DEVICE_ENV)
    if not stages:
        return
    if _device_state["enabled"]:
        return

    try:
        backend_name, memory_module = _memory_history_module()
        if memory_module is None:
            logger.warning("[startup-mem-snapshot] no accelerator memory module found; device snapshots disabled")
            return

        record_memory_history = getattr(memory_module, "_record_memory_history", None)
        if record_memory_history is None:
            logger.warning(
                "[startup-mem-snapshot] %s memory history is not supported; device snapshots disabled",
                backend_name,
            )
            return

        kwargs: dict[str, Any] = {
            "enabled": "all",
            "context": "all",
            "stacks": "python",
            "max_entries": _MAX_ENTRIES,
        }
        if _accepts_keyword(record_memory_history, "clear_history"):
            kwargs["clear_history"] = True
        record_memory_history(**kwargs)

        _device_state.update({"enabled": True, "module": memory_module, "backend": backend_name, "seq": 0})
        logger.info(
            "[startup-mem-snapshot] %s memory history enabled; stages=%s dir=%s",
            backend_name,
            ",".join(sorted(stages)),
            _output_dir(),
        )
    except Exception as exc:
        logger.warning("[startup-mem-snapshot] failed to enable %s memory history: %s", device, exc)


def disable_startup_device_snapshot() -> None:
    """Stop recording; called once the startup window closes."""
    if not _device_state["enabled"]:
        return
    memory_module = _device_state["module"]
    try:
        if memory_module is not None:
            record_memory_history = getattr(memory_module, "_record_memory_history", None)
            if record_memory_history is not None:
                record_memory_history(enabled=None)
        logger.info("[startup-mem-snapshot] %s memory history disabled", _device_state["backend"])
    except Exception as exc:
        logger.warning("[startup-mem-snapshot] failed to disable memory history: %s", exc)
    finally:
        _device_state.update({"enabled": False, "module": None, "backend": None})


def _dump_device_snapshot(stage: str, device: torch.device | None) -> None:
    if not _device_state["enabled"]:
        return
    memory_module = _device_state["module"]
    dump_snapshot = getattr(memory_module, "_dump_snapshot", None)
    if dump_snapshot is None:
        logger.warning(
            "[startup-mem-snapshot] %s does not support _dump_snapshot; skipping",
            _device_state["backend"],
        )
        return

    _device_state["seq"] += 1
    path = _output_path(device, _device_state["seq"], stage, ".pickle")
    dump_snapshot(path)
    logger.info(
        "[startup-mem-snapshot] device stage=%s rank=%s size=%.1fMiB path=%s",
        stage,
        _rank_tag(device),
        os.path.getsize(path) / _MiB,
        path,
    )


def _read_proc(path: str) -> str:
    try:
        with open(path) as handle:
            return handle.read()
    except OSError:
        return ""


def _first_int(value: str) -> int:
    parts = value.split()
    if not parts:
        return 0
    try:
        return int(parts[0])
    except ValueError:
        return 0


def _proc_memory_kib() -> dict[str, int]:
    """Process memory from ``/proc``; empty on platforms without it."""
    report: dict[str, int] = {}
    for line in _read_proc("/proc/self/status").splitlines():
        key, separator, value = line.partition(":")
        if separator and key in _STATUS_KEYS:
            report[key] = _first_int(value)
    for line in _read_proc("/proc/self/smaps_rollup").splitlines():
        key, separator, value = line.partition(":")
        if separator and key in _SMAPS_KEYS:
            report[f"Smaps{key}"] = _first_int(value)
    return report


def _cpu_tensor_census() -> dict[str, Any]:
    """Count live CPU tensor storages held by the process.

    Storages are counted once by data pointer: DLO keeps a flat pinned shard per
    dtype and hands out views of it, so counting tensors would multiply the same
    bytes by the number of views.
    """
    total_bytes = 0
    used: set[int] = set()
    by_dtype: Counter[str] = Counter()
    by_dtype_count: Counter[str] = Counter()
    counts = 0
    tensor_type = torch.Tensor
    # Sweeping every object in the process trips deprecation shims on unrelated
    # classes (torch's own ``_reduce_op`` enum among them). A read-only census
    # must not turn those into startup log noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            if not isinstance(obj, tensor_type):
                continue
            try:
                if obj.device.type != "cpu":
                    continue
                storage = obj.untyped_storage()
                pointer = storage.data_ptr()
                if pointer in used:
                    continue
                used.add(pointer)
                nbytes = storage.nbytes()
            except Exception:
                continue
            total_bytes += nbytes
            counts += 1
            by_dtype[str(obj.dtype)] += nbytes
            by_dtype_count[str(obj.dtype)] += 1
    return {
        "total_bytes": total_bytes,
        "storages": counts,
        "by_dtype_bytes": dict(by_dtype),
        "by_dtype_storages": dict(by_dtype_count),
    }


def _component_residency(model: Any) -> dict[str, dict[str, int]]:
    """Host/device weight bytes per top-level pipeline component.

    Only module parameters and buffers are visible here. DLO moves its shards
    out of the modules and into the offload hook, so a component that reads as
    device-only after the offload handoff means its weights live on the host
    through the hook -- the census above is what sizes them.
    """
    if not isinstance(model, torch.nn.Module):
        return {}
    report: dict[str, dict[str, int]] = {}
    for name, component in model.named_children():
        host_bytes = 0
        device_bytes = 0
        for tensor in itertools.chain(component.parameters(), component.buffers()):
            try:
                nbytes = tensor.numel() * tensor.element_size()
                if tensor.device.type == "cpu":
                    host_bytes += nbytes
                else:
                    device_bytes += nbytes
            except Exception:
                continue
        report[name] = {"host_bytes": host_bytes, "device_bytes": device_bytes}
    return report


def _dump_host_snapshot(stage: str, device: torch.device | None, model: Any) -> None:
    global _host_seq
    _host_seq += 1
    proc_kib = _proc_memory_kib()
    cpu_tensors = _cpu_tensor_census()
    payload: dict[str, Any] = {
        "stage": stage,
        "rank": _rank_tag(device),
        "pid": os.getpid(),
        "time": time.time(),
        "proc_kib": proc_kib,
        "cpu_tensors": cpu_tensors,
        "components": _component_residency(model),
    }
    path = _output_path(device, _host_seq, stage, ".json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    logger.info(
        "[startup-mem-snapshot] host stage=%s rank=%s rss=%.2fGiB cpu_tensors=%.2fGiB path=%s",
        stage,
        _rank_tag(device),
        proc_kib.get("VmRSS", 0) / _MiB,
        cpu_tensors["total_bytes"] / (1024 * _MiB),
        path,
    )


def capture_startup_snapshot(stage: str, *, device: torch.device | None = None, model: Any = None) -> None:
    """Run every enabled startup collector for ``stage``.

    Failures are logged and swallowed: memory diagnostics must never be the
    reason a service fails to come up.
    """
    try:
        if stage in _selected_stages(_DEVICE_ENV):
            _dump_device_snapshot(stage, device)
        if stage in _selected_stages(_HOST_ENV):
            _dump_host_snapshot(stage, device, model)
    except Exception as exc:
        logger.warning("[startup-mem-snapshot] stage=%s capture failed: %s", stage, exc)


def reset_startup_snapshot_state_for_tests() -> None:
    global _host_seq
    _host_seq = 0
    _device_state.update({"enabled": False, "module": None, "backend": None, "seq": 0})
