# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the MiniMax-H3 staged-component residency switch."""

import pytest
import torch
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


# ---------------------------------------------------------------------------
# VAE dtype switch
# ---------------------------------------------------------------------------


def test_resolve_vae_dtype_defaults_to_fp32(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.vae import _VAE_DTYPE_ENV, _resolve_vae_dtype

    monkeypatch.delenv(_VAE_DTYPE_ENV, raising=False)
    assert _resolve_vae_dtype() is torch.float32

    monkeypatch.setenv(_VAE_DTYPE_ENV, "  ")
    assert _resolve_vae_dtype() is torch.float32


@pytest.mark.parametrize(
    ("value", "expected"),
    [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16), ("BF16", torch.bfloat16)],
)
def test_resolve_vae_dtype_accepts_known_values(monkeypatch, value, expected):
    from vllm_omni.diffusion.models.minimax_h3.vae import _VAE_DTYPE_ENV, _resolve_vae_dtype

    monkeypatch.setenv(_VAE_DTYPE_ENV, value)
    assert _resolve_vae_dtype() is expected


def test_resolve_vae_dtype_rejects_unknown(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.vae import _VAE_DTYPE_ENV, _resolve_vae_dtype

    monkeypatch.setenv(_VAE_DTYPE_ENV, "int8")
    with pytest.raises(ValueError, match="int8"):
        _resolve_vae_dtype()


# ---------------------------------------------------------------------------
# Text-encoder stager switch
# ---------------------------------------------------------------------------


def test_te_stager_flag_reads_env(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.encoder import _TE_STAGER_ENV, _te_stager_enabled

    monkeypatch.delenv(_TE_STAGER_ENV, raising=False)
    assert _te_stager_enabled() is False
    for value in ("1", "true", "YES", " on "):
        monkeypatch.setenv(_TE_STAGER_ENV, value)
        assert _te_stager_enabled() is True
    monkeypatch.setenv(_TE_STAGER_ENV, "0")
    assert _te_stager_enabled() is False


def test_te_stager_created_once_and_cached(monkeypatch):
    """The whole-encoder master is built lazily and reused across phases."""
    from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder

    encoder = object.__new__(MiniMaxH3Qwen3VLEncoder)
    nn.Module.__init__(encoder)
    encoder.device_target = torch.device("meta")
    encoder.vision = nn.Linear(2, 2)
    encoder.text_model = nn.Linear(2, 2)
    built: list = []

    class _SpyStager:
        def __init__(self, modules, device, **kwargs):
            built.append((list(modules), device))

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.encoder.PinnedModuleStager",
        _SpyStager,
    )

    first = encoder._staged_component_stager()
    second = encoder._staged_component_stager()

    assert first is second
    assert len(built) == 1
    assert built[0][0] == [encoder.vision, encoder.text_model]
