# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from einops import rearrange

from vllm_omni.diffusion.models.minimax_h3.vae_processor_fastpath import (
    _revert_tensor_fastpath,
    apply_vae_processor_fastpath,
)

try:  # noqa: SIM105 - availability depends on the deployment platform
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None


def _available_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch_npu is not None:
        try:
            if torch.npu.is_available():
                devices.append(torch.device("npu"))
        except Exception:
            pass
    return devices


def _checkpoint_convert(numpy_array: Any, device: Any = None) -> torch.Tensor:
    """Faithful copy of the checkpoint processor's host-side staging."""
    if isinstance(numpy_array, list):
        numpy_array = np.stack(numpy_array, axis=0)
    numpy_array = numpy_array.astype(np.float32)
    tensor = torch.from_numpy(numpy_array)
    tensor = tensor.permute(0, 3, 1, 2)
    tensor = tensor / 255.0
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _checkpoint_revert(processor: Any, tensor: torch.Tensor) -> torch.Tensor:
    """Faithful copy of the checkpoint processor's post-decode revert."""
    B, T = None, None
    if processor.use_3d_conv:
        tensor = tensor.unsqueeze(2) if tensor.ndim == 4 else tensor
        B, _, T, _, _ = tensor.shape
        tensor = rearrange(tensor, "b c t h w -> (b t) c h w")
    tensor_rev = processor.transform_rev(tensor).clamp(0, 1)
    if B is not None:
        tensor_rev = rearrange(tensor_rev, "(b t) c h w -> b c t h w", b=B, t=T)
    return tensor_rev.contiguous()


class _StubProcessor:
    """Minimal stand-in matching the checkpoint processor's member shape."""

    def __init__(self, *, use_3d_conv: bool, transform_rev: Any) -> None:
        self.use_3d_conv = use_3d_conv
        self.transform_rev = transform_rev

    @staticmethod
    def convert_numpy_to_tensor(numpy_array: Any, device: Any = None) -> torch.Tensor:
        return _checkpoint_convert(numpy_array, device)

    def revert_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return _checkpoint_revert(self, tensor)


def _fresh_transform(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 0.5 + 0.25


def test_apply_patches_processor_members() -> None:
    stub = _StubProcessor(use_3d_conv=True, transform_rev=_fresh_transform)
    assert apply_vae_processor_fastpath(stub) is True
    # The staticmethod is shadowed by the fast closure and the instance
    # method by a bound fast path; both must stay callable.
    assert stub.convert_numpy_to_tensor is not _StubProcessor.convert_numpy_to_tensor
    assert stub.revert_tensor.__func__ is _revert_tensor_fastpath


def test_apply_skips_processor_with_unexpected_shape() -> None:
    class _Bare:
        pass

    bare = _Bare()
    assert apply_vae_processor_fastpath(bare) is False
    # Remotes without a processor (mocks in contract tests) pass None.
    assert apply_vae_processor_fastpath(None) is False

    class _NoStaticmethod(_StubProcessor):
        convert_numpy_to_tensor = None  # type: ignore[assignment]

    stub = _NoStaticmethod(use_3d_conv=True, transform_rev=_fresh_transform)
    assert apply_vae_processor_fastpath(stub) is False
    # The untouched class method keeps serving checkpoint behavior.
    assert stub.revert_tensor.__func__ is _StubProcessor.revert_tensor


@pytest.mark.parametrize("device", _available_devices())
def test_convert_uint8_matches_checkpoint_path(device: torch.device) -> None:
    stub = _StubProcessor(use_3d_conv=True, transform_rev=_fresh_transform)
    assert apply_vae_processor_fastpath(stub) is True

    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(5, 7, 6, 3), dtype=np.uint8)

    fast = stub.convert_numpy_to_tensor(frames, device)
    expected = _checkpoint_convert(frames, device)

    assert fast.device.type == device.type
    assert fast.dtype is torch.float32
    assert fast.is_contiguous()
    assert torch.equal(fast, expected)


@pytest.mark.parametrize("device", _available_devices())
def test_convert_list_of_uint8_frames_matches(device: torch.device) -> None:
    stub = _StubProcessor(use_3d_conv=True, transform_rev=_fresh_transform)
    assert apply_vae_processor_fastpath(stub) is True

    rng = np.random.default_rng(1)
    # The checkpoint API stacks single-frame arrays, mirroring decord output.
    frames = [rng.integers(0, 256, size=(5, 4, 3), dtype=np.uint8) for _ in range(4)]

    fast = stub.convert_numpy_to_tensor(frames, device)
    expected = _checkpoint_convert(frames, device)
    assert torch.equal(fast, expected)


@pytest.mark.parametrize(
    "frames",
    [
        np.zeros((5, 7, 6, 3), dtype=np.float32),
        np.zeros((5, 7, 6, 3), dtype=np.uint16),
        np.zeros((5, 7, 6, 3), dtype=np.uint8).transpose(1, 0, 2, 3),
    ],
    ids=["float32", "uint16", "non-contiguous"],
)
def test_convert_falls_back_to_checkpoint_for_unsupported_input(frames: np.ndarray) -> None:
    calls: list[int] = []

    class _Counting(_StubProcessor):
        @staticmethod
        def convert_numpy_to_tensor(numpy_array: Any, device: Any = None) -> torch.Tensor:
            calls.append(1)
            return _checkpoint_convert(numpy_array, device)

    stub = _Counting(use_3d_conv=True, transform_rev=_fresh_transform)
    assert apply_vae_processor_fastpath(stub) is True

    readonly = np.zeros((5, 7, 6, 3), dtype=np.uint8)
    readonly.setflags(write=False)

    assert torch.equal(stub.convert_numpy_to_tensor(frames), _checkpoint_convert(frames))
    assert torch.equal(stub.convert_numpy_to_tensor(readonly), _checkpoint_convert(readonly))
    # device=None always uses the checkpoint path.
    assert torch.equal(
        stub.convert_numpy_to_tensor(np.zeros((5, 7, 6, 3), dtype=np.uint8)),
        _checkpoint_convert(np.zeros((5, 7, 6, 3), dtype=np.uint8)),
    )
    assert len(calls) == 3


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize(
    ("use_3d_conv", "shape"),
    [
        (True, (2, 3, 4, 5, 6)),
        (True, (8, 3, 5, 6)),  # 4D input takes the unsqueeze(2) path
        (False, (7, 3, 5, 6)),
    ],
    ids=["3d-conv-5d", "3d-conv-4d", "no-3d-conv"],
)
def test_revert_tensor_matches_checkpoint_path(
    device: torch.device, use_3d_conv: bool, shape: tuple[int, ...]
) -> None:
    stub = _StubProcessor(use_3d_conv=use_3d_conv, transform_rev=_fresh_transform)
    assert apply_vae_processor_fastpath(stub) is True

    tensor = torch.randn(*shape, dtype=torch.float32, device=device) * 3.0
    original = tensor.clone()

    fast = stub.revert_tensor(tensor)
    expected = _checkpoint_revert(
        _StubProcessor(use_3d_conv=use_3d_conv, transform_rev=_fresh_transform),
        tensor,
    )

    assert fast.device.type == device.type
    assert fast.is_contiguous()
    assert torch.equal(fast, expected)
    assert torch.equal(tensor, original)


@pytest.mark.parametrize("device", _available_devices())
def test_revert_identity_transform_keeps_input_unmutated(device: torch.device) -> None:
    """A transform that returns its input must not be clamped in place."""
    stub = _StubProcessor(use_3d_conv=True, transform_rev=lambda tensor: tensor)
    assert apply_vae_processor_fastpath(stub) is True

    tensor = torch.rand(2, 3, 2, 5, 6, dtype=torch.float32, device=device) * 3.0
    original = tensor.clone()

    fast = stub.revert_tensor(tensor)
    expected = _checkpoint_revert(
        _StubProcessor(use_3d_conv=True, transform_rev=lambda value: value), tensor
    )
    assert torch.equal(tensor, original)
    assert float(fast.min()) >= 0.0 and float(fast.max()) <= 1.0
    assert torch.equal(fast, expected)


def test_revert_matches_with_torchvision_denormalize() -> None:
    torchvision = pytest.importorskip("torchvision")
    from torchvision.transforms import Normalize

    inv_mean = (-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225)
    inv_std = (1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225)
    transform_rev = Normalize(inv_mean, inv_std)

    stub = _StubProcessor(use_3d_conv=True, transform_rev=transform_rev)
    assert apply_vae_processor_fastpath(stub) is True

    tensor = torch.randn(2, 3, 4, 5, 6) * 2.0
    assert torch.equal(
        stub.revert_tensor(tensor),
        _checkpoint_revert(_StubProcessor(use_3d_conv=True, transform_rev=transform_rev), tensor),
    )
