# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the MiniMax-H3 staged-component residency switch."""

import pytest
import torch.nn as nn

from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    _STAGED_COMPONENTS_ENV,
    MiniMaxH3Pipeline,
    _resolve_staged_components,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_resolve_staged_components_parses_and_normalizes(monkeypatch):
    monkeypatch.delenv(_STAGED_COMPONENTS_ENV, raising=False)
    assert _resolve_staged_components() == frozenset()

    monkeypatch.setenv(_STAGED_COMPONENTS_ENV, " VAE, te ")
    assert _resolve_staged_components() == frozenset({"vae", "te"})


def test_resolve_staged_components_rejects_unknown(monkeypatch):
    monkeypatch.setenv(_STAGED_COMPONENTS_ENV, "vae,dit")
    with pytest.raises(ValueError, match="dit"):
        _resolve_staged_components()


def _bare_pipeline(staged_objects, od_config=None) -> MiniMaxH3Pipeline:
    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.od_config = od_config
    pipeline._staged_component_objects = staged_objects
    return pipeline


def test_manual_component_offload_drives_staged_objects(monkeypatch):
    class _Part(nn.Module):
        pass

    text_encoder = nn.Module()
    video_vae = nn.Module()
    part = _Part()
    audio_vae = nn.Module()
    dit = nn.Module()
    pipeline = _bare_pipeline([video_vae, part, audio_vae, text_encoder])

    assert pipeline._uses_manual_component_offload(text_encoder)
    assert pipeline._uses_manual_component_offload(video_vae)
    assert pipeline._uses_manual_component_offload(part)
    assert pipeline._uses_manual_component_offload(audio_vae)
    assert not pipeline._uses_manual_component_offload(dit)


def test_manual_component_offload_untouched_without_switch(monkeypatch):
    monkeypatch.delenv(_STAGED_COMPONENTS_ENV, raising=False)
    text_encoder = nn.Module()
    pipeline = _bare_pipeline([], od_config=None)

    assert not pipeline._uses_manual_component_offload(text_encoder)


def test_bare_pipeline_offload_identity_not_equality():
    """Membership must be identity-based: an unrelated module with the same
    shape must never be staged."""
    pipeline = _bare_pipeline([nn.Linear(4, 4)])
    assert not pipeline._uses_manual_component_offload(nn.Linear(4, 4))
