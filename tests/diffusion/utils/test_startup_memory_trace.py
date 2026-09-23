# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the opt-in startup memory tracer."""

import logging
from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.utils.startup_memory_trace as trace_module

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_TRACE_ENV = "VLLM_OMNI_DIFFUSION_STARTUP_MEM_TRACE"
_TRACE_LAYERS_ENV = "VLLM_OMNI_DIFFUSION_STARTUP_MEM_TRACE_LAYERS"

_GiB = 1024 * 1024 * 1024


def _fake_torch(recorded: list):
    """A stand-in for the torch module exposing only the accelerator memory APIs."""
    accelerator = SimpleNamespace(
        memory_allocated=lambda device: (recorded.append(("alloc", device)), 3 * _GiB)[1],
        memory_reserved=lambda device: (recorded.append(("reserved", device)), 5 * _GiB)[1],
        max_memory_allocated=lambda device: (recorded.append(("peak", device)), 4 * _GiB)[1],
    )
    return SimpleNamespace(accelerator=accelerator)


@pytest.fixture
def fake_platform(monkeypatch):
    stub = SimpleNamespace(get_device_memory=lambda device: (2 * _GiB, 60 * _GiB))
    monkeypatch.setattr("vllm_omni.platforms.current_omni_platform", stub)
    return stub


def test_trace_disabled_by_default(monkeypatch, fake_platform):
    monkeypatch.delenv(_TRACE_ENV, raising=False)
    recorded: list = []
    monkeypatch.setattr(trace_module, "torch", _fake_torch(recorded))

    trace_module.trace_startup_memory("after_construct", device="cpu")

    assert recorded == []


def test_trace_logs_allocated_reserved_free_peak(monkeypatch, fake_platform, caplog):
    monkeypatch.setenv(_TRACE_ENV, "1")
    recorded: list = []
    monkeypatch.setattr(trace_module, "torch", _fake_torch(recorded))

    with caplog.at_level(logging.INFO, logger=trace_module.__name__):
        trace_module.trace_startup_memory("after_construct", device="cpu", extra={"note": "x"})

    assert ("alloc", "cpu") in recorded and ("reserved", "cpu") in recorded and ("peak", "cpu") in recorded
    line = next(r.getMessage() for r in caplog.records if "[startup-mem]" in r.getMessage())
    assert "stage=after_construct" in line
    assert "alloc=3.00GiB" in line
    assert "reserved=5.00GiB" in line
    assert "free=2.00GiB" in line
    assert "peak=4.00GiB" in line
    assert "note=x" in line


def test_layer_note_traces_every_n_layers(monkeypatch, fake_platform, caplog):
    monkeypatch.setenv(_TRACE_ENV, "1")
    monkeypatch.setenv(_TRACE_LAYERS_ENV, "2")
    monkeypatch.setattr(trace_module, "torch", _fake_torch([]))
    trace_module.reset_online_quant_layer_count_for_tests()

    with caplog.at_level(logging.INFO, logger=trace_module.__name__):
        trace_module.note_online_quant_layer(device="cpu")
        trace_module.note_online_quant_layer(device="cpu")
        trace_module.note_online_quant_layer(device="cpu")

    layers = [r.getMessage() for r in caplog.records if "stage=online_quant_layers" in r.getMessage()]
    assert len(layers) == 1
    assert "layers=2" in layers[0]
    trace_module.reset_online_quant_layer_count_for_tests()


def test_layer_note_skips_queries_when_disabled(monkeypatch, fake_platform):
    monkeypatch.delenv(_TRACE_ENV, raising=False)
    recorded: list = []
    monkeypatch.setattr(trace_module, "torch", _fake_torch(recorded))

    trace_module.note_online_quant_layer(device="cpu")

    assert recorded == []


def test_escape_hatch_and_stagger_flags(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_DIFFUSION_SKIP_POST_LOAD_EMPTY_CACHE", raising=False)
    monkeypatch.delenv("VLLM_OMNI_DIFFUSION_STAGGER_COMPONENT_LOAD", raising=False)
    assert trace_module.skip_post_load_empty_cache() is False
    assert trace_module.stagger_component_load_enabled() is False

    monkeypatch.setenv("VLLM_OMNI_DIFFUSION_SKIP_POST_LOAD_EMPTY_CACHE", "1")
    monkeypatch.setenv("VLLM_OMNI_DIFFUSION_STAGGER_COMPONENT_LOAD", "true")
    assert trace_module.skip_post_load_empty_cache() is True
    assert trace_module.stagger_component_load_enabled() is True


def test_layer_interval_falls_back_to_default_on_bad_value(monkeypatch):
    monkeypatch.setenv(_TRACE_LAYERS_ENV, "not-a-number")
    assert trace_module.trace_layer_interval() == 10
    monkeypatch.setenv(_TRACE_LAYERS_ENV, "0")
    assert trace_module.trace_layer_interval() == 10
    monkeypatch.setenv(_TRACE_LAYERS_ENV, "25")
    assert trace_module.trace_layer_interval() == 25
