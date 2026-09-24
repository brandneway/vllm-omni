# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the opt-in startup memory snapshot collectors."""

import json
import logging
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.utils.startup_memory_snapshot as snapshot_module
from vllm_omni.diffusion.utils.startup_memory_snapshot import (
    capture_startup_snapshot,
    disable_startup_device_snapshot,
    enable_startup_device_snapshot,
    reset_startup_snapshot_state_for_tests,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_DEVICE_ENV = "VLLM_OMNI_DIFFUSION_STARTUP_MEM_SNAPSHOT"
_HOST_ENV = "VLLM_OMNI_DIFFUSION_STARTUP_MEM_SNAPSHOT_HOST"
_DIR_ENV = "VLLM_OMNI_DIFFUSION_STARTUP_MEM_SNAPSHOT_DIR"

_MiB = 1024 * 1024


class _FakeMemory:
    """Common state for the ``torch.npu.memory`` stand-ins below."""

    def __init__(self, *, dump_raises: bool = False):
        self.calls: list[dict] = []
        self.dumped: list[str] = []
        self._dump_raises = dump_raises

    def _dump_snapshot(self, path: str) -> None:
        if self._dump_raises:
            raise RuntimeError("dump failed")
        with open(path, "wb") as handle:
            handle.write(b"x" * 16)
        self.dumped.append(path)


class NoClearHistoryMemory(_FakeMemory):
    """The NPU build: ``clear_history`` does not exist, passing it aborts startup."""

    def _record_memory_history(self, *, enabled=None, context=None, stacks=None, max_entries=None) -> None:
        self.calls.append({"enabled": enabled, "context": context, "stacks": stacks, "max_entries": max_entries})


class WithClearHistoryMemory(_FakeMemory):
    def _record_memory_history(
        self, *, enabled=None, context=None, stacks=None, max_entries=None, clear_history=None
    ) -> None:
        self.calls.append({"enabled": enabled, "clear_history": clear_history})


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    reset_startup_snapshot_state_for_tests()
    yield
    reset_startup_snapshot_state_for_tests()


@pytest.fixture
def snapshot_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(_DIR_ENV, str(tmp_path))
    return tmp_path


def _use_memory_module(monkeypatch, memory_module):
    monkeypatch.setattr(snapshot_module, "_memory_history_module", lambda: ("NPU", memory_module))


def test_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv(_DEVICE_ENV, raising=False)
    monkeypatch.delenv(_HOST_ENV, raising=False)
    monkeypatch.setenv(_DIR_ENV, str(tmp_path))
    memory = NoClearHistoryMemory()
    _use_memory_module(monkeypatch, memory)

    enable_startup_device_snapshot(None)
    capture_startup_snapshot("after_construct")

    assert memory.calls == []
    assert list(tmp_path.iterdir()) == []


def test_device_toggle_dumps_only_device_artifacts(monkeypatch, snapshot_dir):
    monkeypatch.setenv(_DEVICE_ENV, "1")
    monkeypatch.delenv(_HOST_ENV, raising=False)
    _use_memory_module(monkeypatch, NoClearHistoryMemory())

    enable_startup_device_snapshot(None)
    capture_startup_snapshot("after_construct")

    assert [p.suffix for p in snapshot_dir.iterdir()] == [".pickle"]


def test_host_toggle_runs_without_the_device_toggle(monkeypatch, snapshot_dir):
    monkeypatch.delenv(_DEVICE_ENV, raising=False)
    monkeypatch.setenv(_HOST_ENV, "1")
    memory = NoClearHistoryMemory()
    _use_memory_module(monkeypatch, memory)

    capture_startup_snapshot("after_construct")

    assert [p.suffix for p in snapshot_dir.iterdir()] == [".json"]
    assert memory.calls == []


def test_stage_list_limits_capture(monkeypatch, snapshot_dir):
    monkeypatch.setenv(_DEVICE_ENV, "after_construct,after_load")
    _use_memory_module(monkeypatch, NoClearHistoryMemory())

    enable_startup_device_snapshot(None)
    capture_startup_snapshot("after_construct")
    capture_startup_snapshot("after_load_empty_cache")
    capture_startup_snapshot("after_load")

    assert [p.name for p in sorted(snapshot_dir.iterdir())] == [
        "rank0_01_after_construct.pickle",
        "rank0_02_after_load.pickle",
    ]


def test_high_frequency_stage_needs_explicit_opt_in(monkeypatch, snapshot_dir):
    """A bare "1" must not dump a ~100MiB pickle every N quantized layers."""
    monkeypatch.setenv(_DEVICE_ENV, "1")
    _use_memory_module(monkeypatch, NoClearHistoryMemory())

    enable_startup_device_snapshot(None)
    capture_startup_snapshot("online_quant_layers")

    assert list(snapshot_dir.iterdir()) == []

    monkeypatch.setenv(_DEVICE_ENV, "online_quant_layers")
    capture_startup_snapshot("online_quant_layers")

    assert [p.suffix for p in snapshot_dir.iterdir()] == [".pickle"]


def test_backend_without_clear_history_is_tolerated(monkeypatch, snapshot_dir):
    monkeypatch.setenv(_DEVICE_ENV, "1")
    memory = NoClearHistoryMemory()
    _use_memory_module(monkeypatch, memory)

    enable_startup_device_snapshot(None)

    assert len(memory.calls) == 1
    assert memory.calls[0]["enabled"] == "all"
    assert memory.calls[0]["max_entries"] == snapshot_module._MAX_ENTRIES


def test_backend_with_clear_history_receives_it(monkeypatch, snapshot_dir):
    monkeypatch.setenv(_DEVICE_ENV, "1")
    memory = WithClearHistoryMemory()
    _use_memory_module(monkeypatch, memory)

    enable_startup_device_snapshot(None)

    assert memory.calls[0]["clear_history"] is True


def test_memory_history_module_prefers_npu(monkeypatch):
    npu_memory = NoClearHistoryMemory()
    cuda_memory = NoClearHistoryMemory()
    monkeypatch.setattr(
        snapshot_module,
        "torch",
        SimpleNamespace(
            npu=SimpleNamespace(is_available=lambda: True, memory=npu_memory),
            cuda=SimpleNamespace(is_available=lambda: True, memory=cuda_memory),
        ),
    )

    assert snapshot_module._memory_history_module() == ("NPU", npu_memory)


def test_memory_history_module_skips_unavailable_backend(monkeypatch):
    cuda_memory = NoClearHistoryMemory()
    monkeypatch.setattr(
        snapshot_module,
        "torch",
        SimpleNamespace(
            npu=SimpleNamespace(is_available=lambda: False, memory=NoClearHistoryMemory()),
            cuda=SimpleNamespace(is_available=lambda: True, memory=cuda_memory),
        ),
    )

    assert snapshot_module._memory_history_module() == ("CUDA", cuda_memory)


def test_memory_history_module_without_any_backend(monkeypatch):
    monkeypatch.setattr(snapshot_module, "torch", SimpleNamespace())
    assert snapshot_module._memory_history_module() == (None, None)


def test_dump_failure_is_swallowed(monkeypatch, snapshot_dir, caplog):
    """Memory diagnostics must never be what keeps a service from starting."""
    monkeypatch.setenv(_DEVICE_ENV, "1")
    _use_memory_module(monkeypatch, NoClearHistoryMemory(dump_raises=True))

    enable_startup_device_snapshot(None)

    with caplog.at_level(logging.WARNING, logger=snapshot_module.__name__):
        capture_startup_snapshot("after_construct")

    assert any("[startup-mem-snapshot]" in record.getMessage() for record in caplog.records)


def test_enable_failure_is_swallowed(monkeypatch, snapshot_dir, caplog):
    monkeypatch.setenv(_DEVICE_ENV, "1")

    def _boom():
        raise RuntimeError("no accelerator")

    monkeypatch.setattr(snapshot_module, "_memory_history_module", _boom)

    with caplog.at_level(logging.WARNING, logger=snapshot_module.__name__):
        enable_startup_device_snapshot(None)

    assert any("memory history" in record.getMessage() for record in caplog.records)


def test_disable_stops_recording(monkeypatch, snapshot_dir):
    monkeypatch.setenv(_DEVICE_ENV, "1")
    memory = NoClearHistoryMemory()
    _use_memory_module(monkeypatch, memory)

    enable_startup_device_snapshot(None)
    disable_startup_device_snapshot()
    capture_startup_snapshot("startup_complete")

    assert list(snapshot_dir.iterdir()) == []
    assert memory.calls[-1]["enabled"] is None


def test_enable_is_idempotent(monkeypatch, snapshot_dir):
    monkeypatch.setenv(_DEVICE_ENV, "1")
    memory = NoClearHistoryMemory()
    _use_memory_module(monkeypatch, memory)

    enable_startup_device_snapshot(None)
    enable_startup_device_snapshot(None)

    assert len(memory.calls) == 1


def test_host_report_carries_proc_census_and_component_split(monkeypatch, snapshot_dir):
    monkeypatch.delenv(_DEVICE_ENV, raising=False)
    monkeypatch.setenv(_HOST_ENV, "1")

    model = torch.nn.Module()
    model.add_module("transformer", torch.nn.Linear(4, 4))
    model.add_module("text_encoder", torch.nn.Module())
    model.text_encoder.register_buffer("held", torch.zeros(2 * _MiB // 4))

    capture_startup_snapshot("after_offload_enable", model=model)

    files = list(snapshot_dir.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["stage"] == "after_offload_enable"
    assert payload["rank"] == "0"
    assert isinstance(payload["proc_kib"], dict)
    assert payload["cpu_tensors"]["total_bytes"] > 0
    assert payload["components"]["text_encoder"] == {"host_bytes": 2 * _MiB, "device_bytes": 0}
    assert payload["components"]["transformer"]["host_bytes"] > 0


def test_host_report_without_a_model_still_writes(monkeypatch, snapshot_dir):
    monkeypatch.setenv(_HOST_ENV, "after_load")

    capture_startup_snapshot("after_load")

    payload = json.loads(next(snapshot_dir.glob("*.json")).read_text())
    assert payload["components"] == {}


def test_cpu_tensor_census_counts_each_storage_once():
    """DLO hands out views of one pinned shard; counting tensors would inflate it."""
    shard = torch.zeros(64)
    views = [shard[:32], shard[16:]]

    census = snapshot_module._cpu_tensor_census()

    assert census["total_bytes"] >= shard.numel() * shard.element_size()
    assert census["storages"] >= 1
    assert census["by_dtype_bytes"]["torch.float32"] >= shard.numel() * shard.element_size()
    assert views


def test_component_residency_ignores_non_modules():
    assert snapshot_module._component_residency(None) == {}
    assert snapshot_module._component_residency(object()) == {}


def test_trace_startup_memory_routes_to_the_snapshot_hook(monkeypatch, snapshot_dir):
    """The trace call sites are the snapshot call sites; keep them wired."""
    from vllm_omni.diffusion.utils import startup_memory_trace

    monkeypatch.setenv(_DEVICE_ENV, "1")
    _use_memory_module(monkeypatch, NoClearHistoryMemory())

    enable_startup_device_snapshot(None)
    startup_memory_trace.trace_startup_memory("after_construct")

    assert [p.suffix for p in snapshot_dir.iterdir()] == [".pickle"]


def test_trace_startup_memory_can_opt_out_of_snapshots(monkeypatch, snapshot_dir):
    from vllm_omni.diffusion.utils import startup_memory_trace

    monkeypatch.setenv(_DEVICE_ENV, "1")
    _use_memory_module(monkeypatch, NoClearHistoryMemory())

    enable_startup_device_snapshot(None)
    startup_memory_trace.trace_startup_memory("online_quant_layers", snapshot=False)

    assert list(snapshot_dir.iterdir()) == []


def test_stage_filter_ignores_blank_values(monkeypatch):
    monkeypatch.setenv(_DEVICE_ENV, "   ")
    assert snapshot_module._selected_stages(_DEVICE_ENV) == frozenset()

    monkeypatch.setenv(_DEVICE_ENV, "on")
    assert snapshot_module._selected_stages(_DEVICE_ENV) == snapshot_module._DEFAULT_STAGES

    monkeypatch.setenv(_DEVICE_ENV, " after_construct , ")
    assert snapshot_module._selected_stages(_DEVICE_ENV) == frozenset({"after_construct"})
