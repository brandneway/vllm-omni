# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NPU FP8 KV quantization helpers.

These tests load ``kv_quant_npu`` from its source file via ``importlib`` so
the test module itself does not ``import vllm_omni`` (which would pull
``patch`` → ``aenum``, vLLM, etc.).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _repo_root() -> Path:
    """Resolve checkout root (parent of ``vllm_omni/``), not ``tests/``."""
    here = Path(__file__).resolve()
    marker = Path("vllm_omni") / "platforms" / "npu" / "quant" / "kv_quant_npu.py"
    for parent in here.parents:
        if (parent / marker).is_file():
            return parent
    msg = f"could not locate repo root (no {marker}) starting from {here}"
    raise FileNotFoundError(msg)


def _load_kv_quant_npu() -> ModuleType:
    path = _repo_root() / "vllm_omni" / "platforms" / "npu" / "quant" / "kv_quant_npu.py"
    if not path.is_file():
        msg = f"kv_quant_npu source not found: {path}"
        raise FileNotFoundError(msg)
    name = "vllm_omni_test_kv_quant_npu_standalone"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"cannot load import spec for {path}"
        raise RuntimeError(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kv_quant_npu = _load_kv_quant_npu()


def _npu_smoke_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return bool(hasattr(torch, "npu") and torch.npu.is_available())


npu_smoke = pytest.mark.skipif(not _npu_smoke_available(), reason="NPU device or torch_npu not available.")


def test_is_quantized_kv_cache() -> None:
    assert kv_quant_npu.is_quantized_kv_cache("fp8")
    assert not kv_quant_npu.is_quantized_kv_cache(None)
    assert not kv_quant_npu.is_quantized_kv_cache("int8")


class TestKVQuantNPUUnit:
    @pytest.fixture(autouse=True)
    def clear_rot_cache(self):
        kv_quant_npu._ROT_MATRIXS.clear()

    def test_get_rot_matrix_caches_by_device_dtype_and_head_dim(self) -> None:
        calls = {"count": 0}

        class FakeQuaRotMode:
            HADAMARD = "hadamard"

        def fake_create_rot(mode, head_dim, seed):
            calls["count"] += 1
            assert mode == FakeQuaRotMode.HADAMARD
            assert seed == 425500
            return torch.eye(head_dim, dtype=torch.float32)

        device = torch.device("cpu")
        rot_1 = kv_quant_npu._get_rot_matrix(device, torch.float16, 8, FakeQuaRotMode, fake_create_rot)
        rot_2 = kv_quant_npu._get_rot_matrix(device, torch.float16, 8, FakeQuaRotMode, fake_create_rot)
        rot_3 = kv_quant_npu._get_rot_matrix(device, torch.bfloat16, 8, FakeQuaRotMode, fake_create_rot)
        rot_4 = kv_quant_npu._get_rot_matrix(device, torch.float16, 16, FakeQuaRotMode, fake_create_rot)

        assert calls["count"] == 3
        assert rot_1 is rot_2
        assert rot_3.dtype == torch.bfloat16
        assert rot_4.shape == (16, 16)

    @pytest.fixture
    def fake_quant_ops(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        captured: dict[str, Any] = {
            "fa_calls": [],
            "varlen_fa_calls": [],
            "npu_kwargs": None,
            "out_shape": None,
        }

        class FakeTorchNPU:
            float8_e4m3fn = "fp8_marker"

        def fake_fia_v2(q, k, v, **kwargs):
            del q, k, v
            captured["npu_kwargs"] = kwargs
            out_shape = captured["out_shape"]
            return (torch.ones(out_shape, dtype=torch.float32),)

        def fake_fa_block_quant_preprocess(x, block_size, dst_type, layout):
            captured["fa_calls"].append(
                {
                    "block_size": block_size,
                    "layout": layout,
                    "dst_type": dst_type,
                    "shape": tuple(x.shape),
                }
            )
            scale = torch.full((1,), float(block_size), dtype=torch.float32)
            return x, scale

        def fake_fa_block_quant_preprocess_varlen(x, cu_seq_lens, block_size, dst_type):
            captured["varlen_fa_calls"].append(
                {
                    "block_size": block_size,
                    "cu_seq_lens": list(cu_seq_lens),
                    "shape": tuple(x.shape),
                }
            )
            scale = torch.full((1,), float(block_size), dtype=torch.float32)
            return x, scale

        fake_qua_rot_mode = SimpleNamespace(HADAMARD="hadamard")

        def fake_create_rot(mode, head_dim, seed):
            assert mode == "hadamard"
            assert seed == 425500
            return torch.eye(head_dim, dtype=torch.float32)

        monkeypatch.setattr(
            kv_quant_npu,
            "_load_quant_ops",
            lambda: (
                FakeTorchNPU,
                fake_fia_v2,
                fake_fa_block_quant_preprocess,
                fake_fa_block_quant_preprocess_varlen,
                fake_qua_rot_mode,
                fake_create_rot,
            ),
        )

        return captured

    @staticmethod
    def _make_qkv(shape: tuple[int, int, int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = torch.randn(*shape, dtype=torch.float32)
        key = torch.randn(*shape, dtype=torch.float32)
        value = torch.randn(*shape, dtype=torch.float32)
        return query, key, value

    @pytest.mark.parametrize(
        "layout,input_shape,out_shape,softmax_scale,expected_scale",
        [
            ("BNSD", (2, 3, 4, 8), (2, 3, 6, 8), None, 1.0 / math.sqrt(8)),
            ("BSND", (2, 4, 3, 8), (2, 6, 3, 8), 0.125, 0.125),
        ],
    )
    def test_fp8_rotate_quant_fa_layouts_scale_and_crop(
        self,
        fake_quant_ops: dict[str, Any],
        layout: str,
        input_shape: tuple[int, int, int, int],
        out_shape: tuple[int, int, int, int],
        softmax_scale: float | None,
        expected_scale: float,
    ) -> None:
        query, key, value = self._make_qkv(input_shape)
        fake_quant_ops["out_shape"] = out_shape

        out = kv_quant_npu.fp8_rotate_quant_fa(query, key, value, layout=layout, softmax_scale=softmax_scale)

        assert out.shape == query.shape
        assert out.dtype == query.dtype
        assert fake_quant_ops["npu_kwargs"]["input_layout"] == layout
        # BNSD: shape[1]==heads, BSND: shape[2]==heads.
        expected_heads = input_shape[1] if layout == "BNSD" else input_shape[2]
        assert fake_quant_ops["npu_kwargs"]["num_query_heads"] == expected_heads
        assert fake_quant_ops["npu_kwargs"]["softmax_scale"] == pytest.approx(expected_scale)
        assert [call["block_size"] for call in fake_quant_ops["fa_calls"]] == [128, 256, 256]

    def test_fp8_rotate_quant_fa_invalid_layout_raises(self, fake_quant_ops) -> None:
        query = torch.randn(1, 2, 3, 4, dtype=torch.float32)
        key = torch.randn(1, 2, 3, 4, dtype=torch.float32)
        value = torch.randn(1, 2, 3, 4, dtype=torch.float32)
        fake_quant_ops["out_shape"] = (1, 2, 3, 4)

        with pytest.raises(ValueError, match="unsupported layout"):
            kv_quant_npu.fp8_rotate_quant_fa(query, key, value, layout="INVALID")

    def test_fp8_rotate_quant_fa_varlen_ntd_contract(self, fake_quant_ops) -> None:
        total_len, num_heads, head_dim = 8, 2, 4
        query = torch.randn(total_len, num_heads, head_dim, dtype=torch.float32)
        key = torch.randn(total_len, num_heads, head_dim, dtype=torch.float32)
        value = torch.randn(total_len, num_heads, head_dim, dtype=torch.float32)
        cu = [5, 8]
        # The op hands back NTD; the wrapper must normalize to the caller's TND.
        fake_quant_ops["out_shape"] = (num_heads, total_len, head_dim)

        out = kv_quant_npu.fp8_rotate_quant_fa_varlen(
            query, key, value, cu, cu, softmax_scale=0.125
        )

        assert out.shape == query.shape
        kwargs = fake_quant_ops["npu_kwargs"]
        assert kwargs["input_layout"] == "NTD_TND"
        assert kwargs["num_query_heads"] == num_heads
        assert kwargs["num_key_value_heads"] == num_heads
        assert kwargs["actual_seq_qlen"] == cu
        assert kwargs["actual_seq_kvlen"] == cu
        assert kwargs["sparse_mode"] == 0
        assert kwargs["query_quant_mode"] == 7
        assert kwargs["key_quant_mode"] == 7
        assert kwargs["value_quant_mode"] == 7
        assert kwargs["softmax_scale"] == 0.125
        # Varlen quantization runs on the NTD view and never crosses docs.
        assert [call["shape"] for call in fake_quant_ops["varlen_fa_calls"]] == [
            (num_heads, total_len, head_dim),
            (num_heads, total_len, head_dim),
            (num_heads, total_len, head_dim),
        ]
        assert [call["cu_seq_lens"] for call in fake_quant_ops["varlen_fa_calls"]] == [cu, cu, cu]
        assert [call["block_size"] for call in fake_quant_ops["varlen_fa_calls"]] == [128, 256, 256]

    def test_fp8_rotate_quant_fa_varlen_gqa_kv_heads(self, fake_quant_ops) -> None:
        total_len, num_heads, num_kv_heads, head_dim = 8, 4, 2, 4
        query = torch.randn(total_len, num_heads, head_dim, dtype=torch.float32)
        key = torch.randn(total_len, num_kv_heads, head_dim, dtype=torch.float32)
        value = torch.randn(total_len, num_kv_heads, head_dim, dtype=torch.float32)
        fake_quant_ops["out_shape"] = (num_heads, total_len, head_dim)

        out = kv_quant_npu.fp8_rotate_quant_fa_varlen(query, key, value, [8], [8])

        assert out.shape == query.shape
        assert fake_quant_ops["npu_kwargs"]["num_query_heads"] == num_heads
        assert fake_quant_ops["npu_kwargs"]["num_key_value_heads"] == num_kv_heads

    def test_fp8_rotate_quant_fa_varlen_strips_leading_zero_cu(self, fake_quant_ops) -> None:
        """Regression: callers pass cu_seqlens with a leading zero; the op
        contract (actual_seq = cu[1:]) must receive the stripped ends."""
        total_len, num_heads, head_dim = 26, 2, 4
        query = torch.randn(total_len, num_heads, head_dim, dtype=torch.float32)
        key = torch.randn(total_len, num_heads, head_dim, dtype=torch.float32)
        value = torch.randn(total_len, num_heads, head_dim, dtype=torch.float32)
        fake_quant_ops["out_shape"] = (num_heads, total_len, head_dim)

        out = kv_quant_npu.fp8_rotate_quant_fa_varlen(query, key, value, [0, total_len], [0, total_len])

        assert out.shape == query.shape
        assert fake_quant_ops["npu_kwargs"]["actual_seq_qlen"] == [total_len]
        assert fake_quant_ops["npu_kwargs"]["actual_seq_kvlen"] == [total_len]
        assert [call["cu_seq_lens"] for call in fake_quant_ops["varlen_fa_calls"]] == [
            [total_len],
            [total_len],
            [total_len],
        ]

    def test_fp8_rotate_quant_fa_varlen_invalid_dim_raises(self, fake_quant_ops) -> None:
        query = torch.randn(1, 2, 3, 4, dtype=torch.float32)
        key = torch.randn(1, 2, 3, 4, dtype=torch.float32)
        value = torch.randn(1, 2, 3, 4, dtype=torch.float32)

        with pytest.raises(ValueError, match="expected packed TND 3D tensors"):
            kv_quant_npu.fp8_rotate_quant_fa_varlen(query, key, value, [4], [4])


@npu_smoke
class TestKVQuantNPUSmoke:
    """Smoke tests using real torch_npu/mindiesd stack, only on NPU."""

    def test_fp8_rotate_quant_fa_real_npu_shape_contract(self):
        try:
            kv_quant_npu._load_quant_ops.cache_clear()
            kv_quant_npu._load_quant_ops()
        except ImportError:
            pytest.skip("NPU quant dependencies are not fully installed.")

        query = torch.randn(1, 2, 4, 64, dtype=torch.float16, device="npu")
        key = torch.randn(1, 2, 4, 64, dtype=torch.float16, device="npu")
        value = torch.randn(1, 2, 4, 64, dtype=torch.float16, device="npu")

        out = kv_quant_npu.fp8_rotate_quant_fa(query, key, value, layout="BNSD")
        assert out.shape == query.shape
        assert out.dtype == query.dtype

    def test_fp8_rotate_quant_fa_varlen_real_npu_shape_contract(self):
        try:
            kv_quant_npu._load_quant_ops.cache_clear()
            kv_quant_npu._load_quant_ops()
        except ImportError:
            pytest.skip("NPU quant dependencies are not fully installed.")

        query = torch.randn(8, 2, 64, dtype=torch.float16, device="npu")
        key = torch.randn(8, 2, 64, dtype=torch.float16, device="npu")
        value = torch.randn(8, 2, 64, dtype=torch.float16, device="npu")

        out = kv_quant_npu.fp8_rotate_quant_fa_varlen(query, key, value, [5, 8], [5, 8])
        assert out.shape == query.shape
        assert out.dtype == query.dtype
