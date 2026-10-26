# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Selection logic for the env-gated MindIE-SD Eagle fused operators."""

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import eagle_fused

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


class _FakeTensor:
    """Minimal stand-in carrying the metadata the compatibility checks read."""

    def __init__(self, shape, dtype=torch.bfloat16, device="npu", contiguous=True):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device
        self._contiguous = contiguous

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self._contiguous


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("1", True),
        ("TRUE", True),
        ("yes", True),
        (" on ", True),
    ],
)
def test_env_switches_parse_truthy_spellings(monkeypatch, value, expected):
    for name in (eagle_fused.SPLIT_NORM_ROPE_ENV, eagle_fused.INDEXED_GATE_NORM_MODULATE_ENV):
        monkeypatch.delenv(name, raising=False)
        if value is not None:
            monkeypatch.setenv(name, value)
        assert eagle_fused._env_enabled(name) is expected

    enabled = eagle_fused.split_norm_rope_enabled()
    assert enabled is expected
    assert eagle_fused.indexed_gate_norm_modulate_enabled() is expected
    # The availability helpers stay off whenever the switch is off.
    if not expected:
        assert not eagle_fused.split_norm_rope_available()
        assert not eagle_fused.indexed_gate_norm_modulate_available()


@pytest.mark.cpu
def test_rope_table_split_is_cached_per_table():
    eagle_fused._ROPE_SPLIT_CACHE = None
    rotary_dim = 96
    table = torch.randn(7, rotary_dim, dtype=torch.bfloat16)

    cos, sin = eagle_fused._split_rope_table(table, rotary_dim)
    assert cos.shape == (7, rotary_dim // 2)
    assert sin.shape == (7, rotary_dim // 2)
    assert cos.is_contiguous() and sin.is_contiguous()
    assert torch.equal(cos, table[:, : rotary_dim // 2])
    assert torch.equal(sin, table[:, rotary_dim // 2 :])

    assert eagle_fused._split_rope_table(table, rotary_dim)[0] is cos
    other = table.clone()
    assert eagle_fused._split_rope_table(other, rotary_dim)[0] is not cos
    eagle_fused._ROPE_SPLIT_CACHE = None


def _split_rope_inputs(**overrides):
    num_heads, num_kv_heads, head_dim, rotary_dim = 56, 56, 128, 96
    sequence = 11
    inputs = {
        "qkv": _FakeTensor((sequence, (num_heads + 2 * num_kv_heads) * head_dim)),
        "q_weight": _FakeTensor((head_dim,)),
        "k_weight": _FakeTensor((head_dim,)),
        "rope_table": _FakeTensor((sequence, rotary_dim)),
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
    }
    inputs.update(overrides)
    return inputs


@pytest.mark.cpu
def test_split_norm_rope_accepts_the_h3_geometry():
    assert eagle_fused.split_norm_rope_compatible(**_split_rope_inputs())


@pytest.mark.cpu
@pytest.mark.parametrize(
    "overrides",
    [
        {"qkv": _FakeTensor((11, 168 * 128), device="cpu")},
        {"qkv": _FakeTensor((11, 168 * 128), dtype=torch.float32)},
        {"qkv": _FakeTensor((11, 168 * 128), contiguous=False)},
        {"head_dim": 96, "qkv": _FakeTensor((11, 168 * 96))},
        {"rotary_dim": 100},
        {"rope_table": _FakeTensor((11, 96), dtype=torch.float32)},
        {"rope_table": _FakeTensor((10, 96))},
        {"q_weight": None},
        {"k_weight": _FakeTensor((128,), contiguous=False)},
        {"num_heads": 60},
    ],
)
def test_split_norm_rope_rejects_unsupported_shapes(overrides):
    assert not eagle_fused.split_norm_rope_compatible(**_split_rope_inputs(**overrides))


def _indexed_gate_inputs(**overrides):
    rows, hidden, conditions = 13, 5376, 6
    inputs = {
        "residual": _FakeTensor((rows, hidden)),
        "gate": _FakeTensor((conditions, hidden)),
        "branch": _FakeTensor((rows, hidden)),
        "shift": _FakeTensor((conditions, hidden)),
        "scale": _FakeTensor((conditions, hidden)),
        "indices": _FakeTensor((rows,), dtype=torch.int64),
        "weight": _FakeTensor((hidden,)),
    }
    inputs.update(overrides)
    return inputs


@pytest.mark.cpu
def test_indexed_gate_norm_modulate_accepts_the_h3_geometry():
    assert eagle_fused.indexed_gate_norm_modulate_compatible(**_indexed_gate_inputs())


@pytest.mark.cpu
@pytest.mark.parametrize(
    "overrides",
    [
        {"residual": _FakeTensor((13, 5376), device="cpu")},
        {"residual": _FakeTensor((13, 5376), dtype=torch.float32)},
        {"residual": _FakeTensor((13, 7169))},
        {"branch": _FakeTensor((12, 5376))},
        {"gate": _FakeTensor((6, 5375))},
        {"shift": _FakeTensor((5, 5376))},
        {"indices": _FakeTensor((13,), dtype=torch.float32)},
        {"indices": _FakeTensor((12,), dtype=torch.int64)},
        {"weight": None},
        {"scale": _FakeTensor((6, 5376), device="cpu")},
    ],
)
def test_indexed_gate_norm_modulate_rejects_unsupported_shapes(overrides):
    assert not eagle_fused.indexed_gate_norm_modulate_compatible(**_indexed_gate_inputs(**overrides))


@pytest.mark.cpu
def test_availability_requires_both_switch_and_operator(monkeypatch):
    monkeypatch.setattr(eagle_fused, "_mindiesd_op", lambda name: None)
    monkeypatch.setenv(eagle_fused.SPLIT_NORM_ROPE_ENV, "1")
    monkeypatch.setenv(eagle_fused.INDEXED_GATE_NORM_MODULATE_ENV, "1")
    assert not eagle_fused.split_norm_rope_available()
    assert not eagle_fused.indexed_gate_norm_modulate_available()

    monkeypatch.setattr(eagle_fused, "_mindiesd_op", lambda name: object())
    assert eagle_fused.split_norm_rope_available()
    assert eagle_fused.indexed_gate_norm_modulate_available()

    monkeypatch.delenv(eagle_fused.SPLIT_NORM_ROPE_ENV)
    assert not eagle_fused.split_norm_rope_available()
    assert eagle_fused.indexed_gate_norm_modulate_available()
