# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the optional TAEH3 lightweight video decoder."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as pipeline_module
import vllm_omni.diffusion.models.minimax_h3.taeh3 as taeh3_module
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    _TAEH3_ENABLE_ENV,
    MiniMaxH3Pipeline,
    _resolve_taeh3_enabled,
)
from vllm_omni.diffusion.models.minimax_h3.taeh3 import TAEH3Decoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _fake_parallel_decoder(_decoder, value):
    """Stand in for the sequential pass: 4x time upscale, 8x space upscale."""
    batch, frames, channels, height, width = value.shape
    assert channels == 24
    return torch.zeros(batch, frames * 4, 12, height * 8, width * 8)


def test_taeh3_layout_matches_pinned_topology():
    decoder = TAEH3Decoder()

    assert decoder.latent_channels == 24
    assert decoder.patch_size == 2
    assert decoder.temporal_upscale == 4
    # Pinned weight geometry of madebyollin/taehv taeh3.pth: 24ch in, 3*2^2 out.
    assert decoder.decoder[1].weight.shape == (256, 24, 3, 3)
    assert decoder.decoder[22].weight.shape == (12, 64, 3, 3)
    assert len(taeh3_module.TAEH3_CHECKPOINT_SHA256) == 64


def test_taeh3_decode_video_full_resolution_contract(monkeypatch):
    monkeypatch.setattr(taeh3_module, "_apply_decoder_parallel", _fake_parallel_decoder)
    decoder = TAEH3Decoder()

    video = decoder.decode_video(torch.zeros(1, 24, 2, 2, 2))

    # latent_t=2 rebuilds the 5-frame H3 chunk; 16x space keeps the full canvas.
    assert video.shape == (1, 3, 5, 32, 32)
    assert float(video.min()) >= 0.0
    assert float(video.max()) <= 1.0


def test_taeh3_decode_video_rebuilds_17n5_frame_chunks(monkeypatch):
    monkeypatch.setattr(taeh3_module, "_apply_decoder_parallel", _fake_parallel_decoder)
    decoder = TAEH3Decoder()

    # 124 frames == 17*7+5 come from latent_t == 5*7+2 == 37.
    video = decoder.decode_video(torch.zeros(1, 24, 37, 4, 4))

    assert video.shape[2] == 124


class _SpyVideoVAE:
    def __init__(self):
        self.decode_calls = 0

    def decode_latent(self, latent):
        self.decode_calls += 1
        return torch.ones(1, 3, 5, 4, 4)


class _SpyAudioVAE:
    def __init__(self):
        self.decode_calls = 0

    def decode_latent(self, latent):
        self.decode_calls += 1
        return torch.ones(1, 2, 5)


def _bare_pipeline(**attrs) -> MiniMaxH3Pipeline:
    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline._component_on_device = lambda component: nullcontext()
    for name, value in attrs.items():
        setattr(pipeline, name, value)
    return pipeline


def test_decode_uses_taeh3_and_skips_full_vae():
    class FakeTAEH3:
        def decode_video(self, latent):
            assert latent.shape == (1, 24, 2, 2, 2)
            return torch.ones(1, 3, 5, 4, 4)

    video_vae = _SpyVideoVAE()
    audio_vae = _SpyAudioVAE()
    pipeline = _bare_pipeline(taeh3_decoder=FakeTAEH3(), video_vae=video_vae, audio_vae=audio_vae)

    video, audio = pipeline.decode(
        torch.zeros(1, 24, 2, 2, 2),
        torch.zeros(1, 2, 2, 2),
        height=4,
        width=4,
    )

    assert video_vae.decode_calls == 0
    assert audio_vae.decode_calls == 1
    assert video.shape == (1, 3, 5, 4, 4)
    assert audio.shape == (1, 2, 5)


def test_decode_falls_back_to_full_vae(monkeypatch):
    monkeypatch.setattr(
        pipeline_module,
        "current_omni_platform",
        SimpleNamespace(create_autocast_context=lambda **kwargs: nullcontext()),
    )
    video_vae = _SpyVideoVAE()
    audio_vae = _SpyAudioVAE()
    pipeline = _bare_pipeline(
        taeh3_decoder=None,
        video_vae=video_vae,
        audio_vae=audio_vae,
        device=SimpleNamespace(type="cpu"),
    )

    pipeline.decode(torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 2, 2, 2), height=4, width=4)

    assert video_vae.decode_calls == 1
    assert audio_vae.decode_calls == 1


def test_decode_raises_when_taeh3_geometry_mismatches():
    class BadTAEH3:
        def decode_video(self, latent):
            return torch.ones(1, 3, 5, 2, 2)

    pipeline = _bare_pipeline(taeh3_decoder=BadTAEH3(), video_vae=_SpyVideoVAE(), audio_vae=_SpyAudioVAE())

    with pytest.raises(ValueError, match="TAEH3 decoded"):
        pipeline.decode(torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 2, 2, 2), height=4, width=4)


@pytest.mark.parametrize(
    ("config", "env", "expected"),
    [
        ({}, None, False),
        ({}, "1", True),
        ({}, "true", True),
        ({}, "0", False),
        ({}, "off", False),
        ({"taeh3_decoder": True}, None, True),
        ({"taeh3_decoder": False}, None, False),
        # An explicit config value suppresses the env escape hatch.
        ({"taeh3_decoder": False}, "1", False),
    ],
)
def test_resolve_taeh3_enabled_precedence(monkeypatch, config, env, expected):
    monkeypatch.delenv(_TAEH3_ENABLE_ENV, raising=False)
    if env is not None:
        monkeypatch.setenv(_TAEH3_ENABLE_ENV, env)

    assert _resolve_taeh3_enabled(config) is expected
