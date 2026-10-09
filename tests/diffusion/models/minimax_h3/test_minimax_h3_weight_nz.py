# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the MiniMax-H3 DiT ND -> FRACTAL_NZ weight opt-in.

The cast itself needs an Ascend device, so ``torch_npu`` is replaced by a
recording stub and the input-weight predicate is the only thing swapped: every
gate the real code applies to a module is still the real check.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

import vllm_omni.diffusion.models.minimax_h3.npu.weight_nz as weight_nz
from vllm_omni.diffusion.models.minimax_h3.npu.weight_nz import (
    WEIGHT_NZ_ENV,
    install_minimax_h3_dit_weight_nz,
)
from vllm_omni.diffusion.offloader.config import OffloadStrategy, ResolvedOffload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_ACL_FORMAT_ND = 2
_ACL_FORMAT_FRACTAL_NZ = 29


class _FakeTorchNpu:
    """Records casts and remembers each storage's format, keyed by data_ptr.

    ``npu_format_cast`` returns a fresh tensor, mirroring the real op, so the
    post-cast readback in ``install_minimax_h3_dit_weight_nz`` is exercised
    rather than assumed.
    """

    def __init__(self) -> None:
        self.cast_formats: list[int] = []
        self._formats: dict[int, int] = {}
        self.dropped = False

    def npu_format_cast(self, weight: torch.Tensor, fmt: int) -> torch.Tensor:
        self.cast_formats.append(fmt)
        if self.dropped:
            # Mirrors a device where ``allow_internal_format`` is off: the op
            # returns an ND tensor instead of raising.
            return weight.clone()
        out = weight.clone()
        self._formats[out.untyped_storage().data_ptr()] = fmt
        return out

    def get_npu_format(self, weight: torch.Tensor) -> int:
        return self._formats.get(weight.untyped_storage().data_ptr(), _ACL_FORMAT_ND)


class _StubLinear(LinearBase):
    """A vLLM linear with just enough state for the eligibility checks."""

    def __init__(self, dtype: torch.dtype, quant_method: object | None = None, shape=(8, 4)) -> None:
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.zeros(*shape, dtype=dtype))
        self.quant_method = UnquantizedLinearMethod() if quant_method is None else quant_method


class _QuantMethodStub:
    """Anything that is not ``UnquantizedLinearMethod`` owns its own layout."""


@pytest.fixture()
def fake_torch_npu(monkeypatch: pytest.MonkeyPatch) -> _FakeTorchNpu:
    fake = _FakeTorchNpu()
    monkeypatch.setitem(sys.modules, "torch_npu", fake)
    return fake


@pytest.fixture()
def on_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend every weight is device-resident; CPU tensors cannot be NZ."""
    monkeypatch.setattr(weight_nz, "_is_npu_weight", lambda _weight: True)


def _model_with(*linears: nn.Module) -> nn.Module:
    model = nn.Module()
    for index, linear in enumerate(linears):
        model.add_module(f"block{index}", linear)
    return model


def _offload_config(strategy: str) -> SimpleNamespace:
    """An od_config carrying a resolved offload policy of the given strategy."""
    resolved = ResolvedOffload(
        strategy=OffloadStrategy(strategy),
        public=None,
        components=frozenset(),
        transfers={},
        pin_memory=False,
        resident_layers=0,
    )
    return SimpleNamespace(_resolved_diffusion_offload=resolved, diffusion_offload_config=None)


def test_env_disabled_is_a_noop(monkeypatch: pytest.MonkeyPatch, fake_torch_npu: _FakeTorchNpu, on_device: None):
    monkeypatch.delenv(WEIGHT_NZ_ENV, raising=False)
    model = _model_with(_StubLinear(torch.bfloat16))

    assert install_minimax_h3_dit_weight_nz(model) == 0
    assert fake_torch_npu.cast_formats == []


@pytest.mark.parametrize("env_value", ["1", "true", "yes", "on", "TRUE", " on "])
def test_env_parsing_accepts_truthy_spellings(
    monkeypatch: pytest.MonkeyPatch, fake_torch_npu: _FakeTorchNpu, on_device: None, env_value: str
):
    monkeypatch.setenv(WEIGHT_NZ_ENV, env_value)

    assert install_minimax_h3_dit_weight_nz(_model_with(_StubLinear(torch.bfloat16))) == 1


@pytest.mark.parametrize("env_value", ["0", "false", "no", "off", "", "2"])
def test_env_parsing_rejects_everything_else(
    monkeypatch: pytest.MonkeyPatch, fake_torch_npu: _FakeTorchNpu, on_device: None, env_value: str
):
    monkeypatch.setenv(WEIGHT_NZ_ENV, env_value)

    assert install_minimax_h3_dit_weight_nz(_model_with(_StubLinear(torch.bfloat16))) == 0


def test_bf16_weights_are_cast_to_nz(monkeypatch: pytest.MonkeyPatch, fake_torch_npu: _FakeTorchNpu, on_device: None):
    monkeypatch.setenv(WEIGHT_NZ_ENV, "1")
    first = _StubLinear(torch.bfloat16)
    second = _StubLinear(torch.float16)

    assert install_minimax_h3_dit_weight_nz(_model_with(first, second)) == 2
    assert fake_torch_npu.cast_formats == [_ACL_FORMAT_FRACTAL_NZ, _ACL_FORMAT_FRACTAL_NZ]
    assert fake_torch_npu.get_npu_format(first.weight) == _ACL_FORMAT_FRACTAL_NZ


def test_second_call_is_idempotent(monkeypatch: pytest.MonkeyPatch, fake_torch_npu: _FakeTorchNpu, on_device: None):
    monkeypatch.setenv(WEIGHT_NZ_ENV, "1")
    model = _model_with(_StubLinear(torch.bfloat16))

    assert install_minimax_h3_dit_weight_nz(model) == 1
    assert install_minimax_h3_dit_weight_nz(model) == 0
    assert len(fake_torch_npu.cast_formats) == 1


def test_host_weight_is_skipped(monkeypatch: pytest.MonkeyPatch, fake_torch_npu: _FakeTorchNpu):
    """CPU tensors are left alone even with the opt-in on."""
    monkeypatch.setenv(WEIGHT_NZ_ENV, "1")

    assert install_minimax_h3_dit_weight_nz(_model_with(_StubLinear(torch.bfloat16))) == 0
    assert fake_torch_npu.cast_formats == []


def test_fp32_weight_is_skipped(monkeypatch: pytest.MonkeyPatch, fake_torch_npu: _FakeTorchNpu, on_device: None):
    """The fp32 residents (patch projections, time embedder, final layer)."""
    monkeypatch.setenv(WEIGHT_NZ_ENV, "1")

    assert install_minimax_h3_dit_weight_nz(_model_with(_StubLinear(torch.float32))) == 0
    assert fake_torch_npu.cast_formats == []


def test_quantized_linear_is_skipped(monkeypatch: pytest.MonkeyPatch, fake_torch_npu: _FakeTorchNpu, on_device: None):
    monkeypatch.setenv(WEIGHT_NZ_ENV, "1")
    model = _model_with(_StubLinear(torch.bfloat16, quant_method=_QuantMethodStub()))

    assert install_minimax_h3_dit_weight_nz(model) == 0
    assert fake_torch_npu.cast_formats == []


def test_plain_module_is_skipped(monkeypatch: pytest.MonkeyPatch, fake_torch_npu: _FakeTorchNpu, on_device: None):
    monkeypatch.setenv(WEIGHT_NZ_ENV, "1")
    plain = nn.Module()
    plain.weight = nn.Parameter(torch.zeros(8, 4, dtype=torch.bfloat16))

    assert install_minimax_h3_dit_weight_nz(_model_with(plain)) == 0
    assert fake_torch_npu.cast_formats == []


def test_one_dimensional_weight_is_skipped(
    monkeypatch: pytest.MonkeyPatch, fake_torch_npu: _FakeTorchNpu, on_device: None
):
    """Norm scales and other vectors that borrow the linear class."""
    monkeypatch.setenv(WEIGHT_NZ_ENV, "1")

    assert install_minimax_h3_dit_weight_nz(_model_with(_StubLinear(torch.bfloat16, shape=(4,)))) == 0
    assert fake_torch_npu.cast_formats == []


def test_silently_dropped_cast_is_reported(
    monkeypatch: pytest.MonkeyPatch, fake_torch_npu: _FakeTorchNpu, on_device: None
):
    """allow_internal_format off: the op returns ND, so nothing counts as cast."""
    monkeypatch.setenv(WEIGHT_NZ_ENV, "1")
    fake_torch_npu.dropped = True

    assert install_minimax_h3_dit_weight_nz(_model_with(_StubLinear(torch.bfloat16))) == 0


def test_cast_failure_keeps_the_weight_and_continues(
    monkeypatch: pytest.MonkeyPatch, fake_torch_npu: _FakeTorchNpu, on_device: None
):
    monkeypatch.setenv(WEIGHT_NZ_ENV, "1")
    good = _StubLinear(torch.bfloat16)
    bad = _StubLinear(torch.bfloat16)
    real_cast = fake_torch_npu.npu_format_cast

    def fail_on_bad(weight: torch.Tensor, fmt: int) -> torch.Tensor:
        if weight.untyped_storage().data_ptr() == bad.weight.untyped_storage().data_ptr():
            raise RuntimeError("unsupported shape")
        return real_cast(weight, fmt)

    monkeypatch.setattr(fake_torch_npu, "npu_format_cast", fail_on_bad)

    assert install_minimax_h3_dit_weight_nz(_model_with(bad, good)) == 1
    assert fake_torch_npu.get_npu_format(good.weight) == _ACL_FORMAT_FRACTAL_NZ
    assert fake_torch_npu.get_npu_format(bad.weight) == _ACL_FORMAT_ND


@pytest.mark.parametrize("strategy", ["distributed_layer_wise", "layer_wise", "model_level"])
def test_any_offload_backend_disables_the_cast(
    monkeypatch: pytest.MonkeyPatch, fake_torch_npu: _FakeTorchNpu, on_device: None, strategy: str
):
    """Offload backends relocate the parameters, so the NZ layout cannot survive."""
    monkeypatch.setenv(WEIGHT_NZ_ENV, "1")
    model = _model_with(_StubLinear(torch.bfloat16))

    assert install_minimax_h3_dit_weight_nz(model, od_config=_offload_config(strategy)) == 0
    assert fake_torch_npu.cast_formats == []


def test_no_offload_config_still_casts(monkeypatch: pytest.MonkeyPatch, fake_torch_npu: _FakeTorchNpu, on_device: None):
    monkeypatch.setenv(WEIGHT_NZ_ENV, "1")

    assert install_minimax_h3_dit_weight_nz(_model_with(_StubLinear(torch.bfloat16)), od_config=None) == 1
