# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, call, sentinel

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_base_platform_keeps_vae_autocast_cache():
    from vllm_omni.platforms.interface import OmniPlatform

    assert not OmniPlatform.can_disable_vae_autocast_cache()


def _npu_platform():
    try:
        from vllm_omni.platforms.npu import platform as npu_platform
    except ModuleNotFoundError:
        pytest.skip("vllm_ascend not available")
    return npu_platform


def test_npu_autocast_forwards_cache_enabled(monkeypatch):
    npu_platform = _npu_platform()
    platform = npu_platform.NPUOmniPlatform
    assert platform.can_disable_vae_autocast_cache()
    mock_torch = MagicMock()
    expected_context = sentinel.context
    mock_torch.npu.amp.autocast.return_value = expected_context
    monkeypatch.setattr(npu_platform, "torch", mock_torch)
    dtype = sentinel.dtype

    context = platform.create_autocast_context(
        device_type="npu",
        dtype=dtype,
        enabled=True,
        cache_enabled=False,
    )

    assert context is expected_context
    mock_torch.npu.amp.autocast.assert_called_once_with(
        dtype=dtype,
        enabled=True,
        cache_enabled=False,
    )


def test_npu_autocast_uses_legacy_signature_without_disabling_amp(monkeypatch):
    npu_platform = _npu_platform()
    platform = npu_platform.NPUOmniPlatform
    mock_torch = MagicMock()
    expected_context = sentinel.context
    mock_torch.npu.amp.autocast.side_effect = [
        TypeError("cache_enabled is unsupported"),
        expected_context,
    ]
    monkeypatch.setattr(npu_platform, "torch", mock_torch)
    dtype = sentinel.dtype

    context = platform.create_autocast_context(
        device_type="npu",
        dtype=dtype,
        enabled=True,
        cache_enabled=False,
    )

    assert context is expected_context
    assert mock_torch.npu.amp.autocast.call_args_list == [
        call(dtype=dtype, enabled=True, cache_enabled=False),
        call(dtype=dtype),
    ]
