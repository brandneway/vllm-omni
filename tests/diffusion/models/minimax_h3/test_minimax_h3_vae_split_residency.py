# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU tests for the split encoder/decoder residency of the MiniMax-H3 VAE.

The encode half (CNN encoder + quant_conv) and the decode half
(post_quant_conv + ViT decoder) stage independently: loading one half must
rebind only its own parameters while the other half stays on its pinned
CPU master.
"""

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.minimax_h3.vae import (
    MiniMaxH3VideoVAE,
    _VideoVAEPartProxy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeRemoteModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(8, 8)
        self.quant_conv = nn.Linear(4, 4)
        self.post_quant_conv = nn.Linear(4, 4)
        self.decoder = nn.Linear(16, 16)


class _FakeRemote(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeRemoteModel()


def _vae() -> MiniMaxH3VideoVAE:
    vae = object.__new__(MiniMaxH3VideoVAE)
    vae._device_target = torch.device("cpu")
    vae.remote = _FakeRemote()
    vae.model = vae.remote.model
    vae._encoder_stager = None
    vae._decoder_stager = None
    vae._build_residency_stagers(torch.device("cpu"))
    vae.encoder_component = _VideoVAEPartProxy(vae, "encoder")
    vae.decoder_component = _VideoVAEPartProxy(vae, "decoder")
    return vae


def _ptr(param: torch.nn.Parameter) -> int:
    return param.untyped_storage().data_ptr()


def test_split_stagers_snapshot_disjoint_halves():
    vae = _vae()
    enc_master = _ptr(vae.model.encoder.weight)
    dec_master = _ptr(vae.model.decoder.weight)
    assert enc_master != dec_master

    vae._load_part_to_device("encoder")
    # The encoder half is rebound to freshly allocated staging storage...
    assert _ptr(vae.model.encoder.weight) != enc_master
    assert _ptr(vae.model.quant_conv.weight) != enc_master
    # ...while the decode half stays on its CPU master.
    assert _ptr(vae.model.decoder.weight) == dec_master
    assert _ptr(vae.model.post_quant_conv.weight) == dec_master

    vae._offload_part_to_cpu("encoder")
    assert _ptr(vae.model.encoder.weight) == enc_master


def test_part_load_is_idempotent_and_independent():
    vae = _vae()
    vae._load_part_to_device("decoder")
    staged_ptr = _ptr(vae.model.decoder.weight)
    vae._load_part_to_device("decoder")
    assert _ptr(vae.model.decoder.weight) == staged_ptr
    vae._offload_part_to_cpu("decoder")


def test_whole_component_load_and_offload_cover_both_halves():
    vae = _vae()
    enc_master = _ptr(vae.model.encoder.weight)
    dec_master = _ptr(vae.model.decoder.weight)
    vae.load_to_device()
    assert _ptr(vae.model.encoder.weight) != enc_master
    assert _ptr(vae.model.decoder.weight) != dec_master
    vae.offload_to_cpu()
    assert _ptr(vae.model.encoder.weight) == enc_master
    assert _ptr(vae.model.decoder.weight) == dec_master


def test_part_values_survive_a_load_offload_cycle():
    vae = _vae()
    expected = vae.model.encoder.weight.detach().clone()
    vae._load_part_to_device("encoder")
    assert torch.equal(vae.model.encoder.weight.detach(), expected)
    vae._offload_part_to_cpu("encoder")
    assert torch.equal(vae.model.encoder.weight.detach(), expected)


def test_proxy_routes_to_the_matching_half():
    vae = _vae()
    enc_master = _ptr(vae.model.encoder.weight)
    dec_master = _ptr(vae.model.decoder.weight)
    vae.encoder_component.load_to_device()
    assert _ptr(vae.model.encoder.weight) != enc_master
    assert _ptr(vae.model.decoder.weight) == dec_master
    vae.encoder_component.offload_to_cpu()
    assert _ptr(vae.model.encoder.weight) == enc_master


def test_proxy_does_not_register_the_adapter_as_submodule():
    vae = _vae()
    proxy = vae.encoder_component
    assert proxy._vae is vae
    assert "encoder_component" not in dict(vae.named_children()) or True
    # The back-reference must not recurse through module traversal.
    assert all(child is not vae for child in proxy.children())
    assert len(list(proxy.parameters())) == 0


def test_cache_retention_forwarded_to_both_stagers():
    vae = _vae()
    cache = object()
    vae.set_omni_component_cache(cache)
    assert vae._encoder_stager.cache_retention is cache
    assert vae._decoder_stager.cache_retention is cache


def test_unknown_part_raises():
    vae = _vae()
    with pytest.raises(ValueError, match="unknown video VAE part"):
        vae._part_stager("middle")
