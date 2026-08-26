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


def test_fp8_kv_slice_enabled_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINDIESD_FP8_KV_SLICE", raising=False)
    assert not kv_quant_npu.fp8_kv_slice_enabled()
    monkeypatch.setenv("MINDIESD_FP8_KV_SLICE", "1")
    assert kv_quant_npu.fp8_kv_slice_enabled()
    monkeypatch.setenv("MINDIESD_FP8_KV_SLICE", "true")
    assert kv_quant_npu.fp8_kv_slice_enabled()
    monkeypatch.setenv("MINDIESD_FP8_KV_SLICE", "0")
    assert not kv_quant_npu.fp8_kv_slice_enabled()


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
            "fia_calls": [],
            "npu_kwargs": None,
            "out_shape": None,
        }

        class FakeTorchNPU:
            float8_e4m3fn = "fp8_marker"

        def fake_fia_v2(q, k, v, **kwargs):
            captured["fia_calls"].append(
                {
                    "q_shape": tuple(q.shape),
                    "k_shape": tuple(k.shape),
                    "v_shape": tuple(v.shape),
                    "kwargs": kwargs,
                }
            )
            captured["npu_kwargs"] = kwargs
            out_dtype = kwargs.get("out_dtype") or torch.float32
            out_shape = captured["out_shape"]
            if out_shape is not None:
                return (torch.ones(out_shape, dtype=out_dtype),)
            # Default: echo q (FIA output has q's shape), so chunked-output
            # reassembly can be verified exactly.
            return (q.to(out_dtype),)

        def fake_frequency_regulator(freq: int):
            del freq
            return torch.empty((1,), dtype=torch.int64)

        def fake_fa_block_quant_preprocess(x, block_size, dst_type, layout):
            captured["fa_calls"].append(
                {
                    "block_size": block_size,
                    "layout": layout,
                    "dst_type": dst_type,
                    "shape": tuple(x.shape),
                }
            )
            # Mirror the real kernel contract: BSND inputs are transposed
            # before quantization, so the returned tensor is ALWAYS
            # BNSD-logical [B, N, S, D] and the scale is per (head, row-block):
            # [B, N, ceil(S / block_size), ceil(D / 128)].
            if layout == "BSND":
                x = x.transpose(1, 2)
            x = x.contiguous()
            b, n, s, _d = x.shape
            blocks = -(-s // block_size)
            scale = torch.ones(b, n, blocks, 1, dtype=torch.float32)
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
            aligned = []
            acc = 0
            prev = 0
            for end in cu_seq_lens:
                acc += (end - prev + block_size - 1) // block_size * block_size
                prev = end
                aligned.append(acc)
            return x, scale, aligned

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
                fake_frequency_regulator,
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
        # Docs pad to 128 (q) / 256 (kv) multiples: q aligned total 256, kv 512.
        # The op hands back [N, T, D]; the wrapper transposes and gathers.
        fake_quant_ops["out_shape"] = (num_heads, 256, head_dim)

        out = kv_quant_npu.fp8_rotate_quant_fa_varlen(
            query, key, value, cu, cu, softmax_scale=0.125
        )

        assert out.shape == query.shape
        kwargs = fake_quant_ops["npu_kwargs"]
        assert kwargs["input_layout"] == "NTD_TND"
        assert kwargs["num_query_heads"] == num_heads
        assert kwargs["num_key_value_heads"] == num_heads
        assert kwargs["actual_seq_qlen"] == [128, 256]
        assert kwargs["actual_seq_kvlen"] == [256, 512]
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
        fake_quant_ops["out_shape"] = (num_heads, 128, head_dim)

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
        fake_quant_ops["out_shape"] = (num_heads, 128, head_dim)

        out = kv_quant_npu.fp8_rotate_quant_fa_varlen(query, key, value, [0, total_len], [0, total_len])

        assert out.shape == query.shape
        assert fake_quant_ops["npu_kwargs"]["actual_seq_qlen"] == [128]
        assert fake_quant_ops["npu_kwargs"]["actual_seq_kvlen"] == [256]
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

    @pytest.mark.parametrize(
        "layout,input_shape,fia_out_shape,expected_quant_shapes",
        [
            # Quant always returns BNSD-logical tensors, so the FIA output is
            # BNSD too; the wrapper transposes back to the caller's layout.
            ("BSND", (1, 8, 2, 4), (1, 2, 8, 4), [(1, 8, 2, 4), (1, 5, 2, 4), (1, 5, 2, 4)]),
            ("BNSD", (1, 2, 8, 4), (1, 2, 8, 4), [(1, 2, 8, 4), (1, 2, 5, 4), (1, 2, 5, 4)]),
        ],
    )
    def test_fp8_rotate_quant_kv_slice_dense_contract(
        self,
        fake_quant_ops: dict[str, Any],
        layout: str,
        input_shape: tuple[int, int, int, int],
        fia_out_shape: tuple[int, int, int, int],
        expected_quant_shapes: list[tuple[int, int, int, int]],
    ) -> None:
        query, key, value = self._make_qkv(input_shape)
        fake_quant_ops["out_shape"] = fia_out_shape

        out = kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 5, layout=layout, softmax_scale=0.125)

        assert out.shape == query.shape
        # Q is quantized at full length; K/V are sliced to kv_len BEFORE quant.
        assert [call["shape"] for call in fake_quant_ops["fa_calls"]] == expected_quant_shapes
        assert [call["layout"] for call in fake_quant_ops["fa_calls"]] == [layout] * 3
        assert [call["block_size"] for call in fake_quant_ops["fa_calls"]] == [128, 256, 256]
        assert fake_quant_ops["varlen_fa_calls"] == []
        # One FIA call by default (MINDIESD_FP8_FIA_QCHUNK unset).
        assert len(fake_quant_ops["fia_calls"]) == 1
        kwargs = fake_quant_ops["npu_kwargs"]
        # Quant output is always BNSD-logical, so the FIA dispatch is BNSD
        # regardless of the caller-facing layout.
        assert kwargs["input_layout"] == "BNSD"
        # Dense dispatch: the FIA varlen feature stays off.
        assert "actual_seq_qlen" not in kwargs
        assert "actual_seq_kvlen" not in kwargs
        assert kwargs["query_quant_mode"] == 7
        assert kwargs["key_quant_mode"] == 7
        assert kwargs["value_quant_mode"] == 7
        assert kwargs["softmax_scale"] == 0.125
        expected_heads = input_shape[1] if layout == "BNSD" else input_shape[2]
        assert kwargs["num_query_heads"] == expected_heads
        assert kwargs["num_key_value_heads"] == expected_heads

    def test_fp8_rotate_quant_kv_slice_full_length_kv_is_noop_slice(self, fake_quant_ops) -> None:
        query, key, value = self._make_qkv((1, 8, 2, 4))
        fake_quant_ops["out_shape"] = (1, 2, 8, 4)  # FIA output is BNSD-logical

        out = kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 8, layout="BSND")

        assert out.shape == query.shape
        assert [call["shape"] for call in fake_quant_ops["fa_calls"]] == [(1, 8, 2, 4)] * 3

    @pytest.mark.parametrize("kv_len", [0, -1, 9, 5.0])
    def test_fp8_rotate_quant_kv_slice_invalid_kv_len_raises(self, fake_quant_ops, kv_len) -> None:
        query, key, value = self._make_qkv((1, 8, 2, 4))

        with pytest.raises(ValueError, match="kv_len"):
            kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, kv_len, layout="BSND")

    def test_fp8_rotate_quant_kv_slice_invalid_layout_raises(self, fake_quant_ops) -> None:
        query, key, value = self._make_qkv((1, 8, 2, 4))

        with pytest.raises(ValueError, match="unsupported layout"):
            kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 5, layout="INVALID")

    def test_fia_q_chunk_count_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MINDIESD_FP8_FIA_QCHUNK", raising=False)
        assert kv_quant_npu._fia_q_chunk_count() == 1
        monkeypatch.setenv("MINDIESD_FP8_FIA_QCHUNK", "4")
        assert kv_quant_npu._fia_q_chunk_count() == 4
        # Non-positive values clamp to the single-call behavior.
        monkeypatch.setenv("MINDIESD_FP8_FIA_QCHUNK", "0")
        assert kv_quant_npu._fia_q_chunk_count() == 1
        monkeypatch.setenv("MINDIESD_FP8_FIA_QCHUNK", "abc")
        with pytest.raises(ValueError, match="MINDIESD_FP8_FIA_QCHUNK"):
            kv_quant_npu._fia_q_chunk_count()

    @pytest.mark.parametrize(
        "seq_len,n_chunks,expected",
        [
            (100, 1, [(0, 100)]),  # chunking disabled
            (128, 8, [(0, 128)]),  # seq fits in one Q block: no split
            (512, 2, [(0, 256), (256, 512)]),
            (256, 2, [(0, 128), (128, 256)]),
            # More chunks requested than 128-row blocks: fewer, aligned chunks.
            (384, 4, [(0, 128), (128, 256), (256, 384)]),
            # Ragged tail: boundaries stay block-aligned, last chunk is short.
            (300, 2, [(0, 256), (256, 300)]),
        ],
    )
    def test_q_chunk_bounds_aligned_to_q_block(self, seq_len, n_chunks, expected) -> None:
        assert kv_quant_npu._q_chunk_bounds(seq_len, n_chunks) == expected

    def test_fp8_rotate_quant_kv_slice_qchunk_splits_fia_and_reassembles(
        self,
        fake_quant_ops: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MINDIESD_FP8_FIA_QCHUNK", "2")
        # Shrink the Q block so the tiny fake seq (8 rows) actually splits.
        monkeypatch.setattr(kv_quant_npu, "_Q_BLOCK_SIZE", 4)
        query, key, value = self._make_qkv((1, 8, 2, 4))  # BSND: S=8 -> 2 chunks of 4

        out = kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 5, layout="BSND")

        assert out.shape == query.shape
        calls = fake_quant_ops["fia_calls"]
        assert len(calls) == 2
        # Each call gets a BNSD-logical 4-row Q chunk and the full (kv_len-sliced) K/V.
        assert [c["q_shape"] for c in calls] == [(1, 2, 4, 4), (1, 2, 4, 4)]
        assert [c["k_shape"] for c in calls] == [(1, 2, 5, 4), (1, 2, 5, 4)]
        assert [c["v_shape"] for c in calls] == [(1, 2, 5, 4), (1, 2, 5, 4)]
        assert [c["kwargs"]["input_layout"] for c in calls] == ["BNSD", "BNSD"]
        assert "actual_seq_qlen" not in calls[0]["kwargs"]
        # Per-chunk Q dequant scales are the matching block slices (1 block per chunk).
        assert [tuple(c["kwargs"]["dequant_scale_query"].shape) for c in calls] == [(1, 2, 1, 1), (1, 2, 1, 1)]
        # K/V (and Q) are quantized exactly once, not per chunk.
        assert len(fake_quant_ops["fa_calls"]) == 3
        # The fake rotation is identity and fake FIA echoes its Q chunk, so the
        # reassembled output must equal the input query exactly (chunk order kept).
        assert torch.equal(out, query)

    def test_fia_chunk_a2a_enabled_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MINDIESD_FP8_CHUNK_A2A", raising=False)
        assert not kv_quant_npu._fia_chunk_a2a_enabled()
        monkeypatch.setenv("MINDIESD_FP8_CHUNK_A2A", "1")
        assert kv_quant_npu._fia_chunk_a2a_enabled()
        monkeypatch.setenv("MINDIESD_FP8_CHUNK_A2A", "true")
        assert kv_quant_npu._fia_chunk_a2a_enabled()
        monkeypatch.setenv("MINDIESD_FP8_CHUNK_A2A", "0")
        assert not kv_quant_npu._fia_chunk_a2a_enabled()

    @pytest.mark.parametrize(
        "total_seq,world,n_chunks,expected",
        [
            # chunk = ceil(4096/1024)*128 = 512 -> 8 aligned chunks.
            (4096, 4, 8, [(0, 512), (512, 1024), (1024, 1536), (1536, 2048),
                          (2048, 2560), (2560, 3072), (3072, 3584), (3584, 4096)]),
            # chunk = ceil(4096/896)*128 = 640 -> 7 chunks, ragged tail; chunks
            # straddle shard boundaries (shard=1024), handled by the gather.
            (4096, 4, 7, [(0, 640), (640, 1280), (1280, 1920), (1920, 2560),
                          (2560, 3200), (3200, 3840), (3840, 4096)]),
            # shard need NOT be 128-aligned (shard=1000): straddling chunks.
            (4000, 4, 8, [(0, 512), (512, 1024), (1024, 1536), (1536, 2048),
                          (2048, 2560), (2560, 3072), (3072, 3584), (3584, 4000)]),
            # total not divisible by world -> infeasible.
            (4100, 4, 8, None),
            # chunking disabled / single rank.
            (4096, 4, 1, None),
            (4096, 1, 8, None),
        ],
    )
    def test_scheme_b_chunk_bounds(self, total_seq, world, n_chunks, expected) -> None:
        assert kv_quant_npu._scheme_b_chunk_bounds(total_seq, world, n_chunks) == expected

    def test_fp8_rotate_quant_kv_slice_chunk_callback_mode(self, fake_quant_ops) -> None:
        query, key, value = self._make_qkv((1, 8, 2, 4))  # BSND
        received = []

        out = kv_quant_npu.fp8_rotate_quant_kv_slice(
            query,
            key,
            value,
            5,
            layout="BSND",
            q_chunk_bounds=[(0, 4), (4, 8)],
            chunk_callback=lambda chunk, idx: received.append((idx, chunk)),
        )

        # Callback mode consumes the output per chunk and returns None.
        assert out is None
        assert len(received) == 2
        # Chunks arrive in order, in the caller-facing (BSND) layout, with exact
        # values (fake rotation is identity, fake FIA echoes the q chunk).
        assert [idx for idx, _ in received] == [0, 1]
        assert torch.equal(received[0][1], query[:, 0:4])
        assert torch.equal(received[1][1], query[:, 4:8])
        assert [c["q_shape"] for c in fake_quant_ops["fia_calls"]] == [(1, 2, 4, 4), (1, 2, 4, 4)]
        # K/V are quantized exactly once, not per chunk.
        assert len(fake_quant_ops["fa_calls"]) == 3

    def test_fp8_rotate_quant_kv_slice_qchunk_ragged_tail(
        self,
        fake_quant_ops: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MINDIESD_FP8_FIA_QCHUNK", "2")
        monkeypatch.setattr(kv_quant_npu, "_Q_BLOCK_SIZE", 4)
        query, key, value = self._make_qkv((1, 10, 2, 4))  # S=10 -> chunks (0,8), (8,10)

        out = kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 10, layout="BSND")

        assert out.shape == query.shape
        calls = fake_quant_ops["fia_calls"]
        assert [c["q_shape"] for c in calls] == [(1, 2, 8, 4), (1, 2, 2, 4)]
        # Tail chunk (2 rows) gets the ceil(2/4)=1 trailing scale block;
        # first chunk gets its 2 blocks.
        assert [tuple(c["kwargs"]["dequant_scale_query"].shape) for c in calls] == [(1, 2, 2, 1), (1, 2, 1, 1)]
        assert torch.equal(out, query)


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

    def test_fp8_rotate_quant_kv_slice_real_npu_shape_contract(self):
        try:
            kv_quant_npu._load_quant_ops.cache_clear()
            kv_quant_npu._load_quant_ops()
        except ImportError:
            pytest.skip("NPU quant dependencies are not fully installed.")

        query = torch.randn(1, 8, 2, 64, dtype=torch.float16, device="npu")
        key = torch.randn(1, 8, 2, 64, dtype=torch.float16, device="npu")
        value = torch.randn(1, 8, 2, 64, dtype=torch.float16, device="npu")

        out = kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 5, layout="BSND")
        assert out.shape == query.shape
        assert out.dtype == query.dtype

    def test_fp8_rotate_quant_kv_slice_qchunk_matches_single_call(self, monkeypatch):
        """Chunked FIA (MINDIESD_FP8_FIA_QCHUNK>1) must match the single wide call.

        Chunk boundaries align to the 128-row Q quant block, so quantization is
        identical; only FIA-internal tiling may differ, hence allclose rather
        than bitwise equality.
        """
        try:
            kv_quant_npu._load_quant_ops.cache_clear()
            kv_quant_npu._load_quant_ops()
        except ImportError:
            pytest.skip("NPU quant dependencies are not fully installed.")

        torch.manual_seed(0)
        # S=640 -> 3 chunks (256, 256, 128) with QCHUNK=3; kv_len exercises the
        # prefix slice and the 256-row K/V block's ragged tail.
        query = torch.randn(1, 640, 2, 128, dtype=torch.bfloat16, device="npu")
        key = torch.randn(1, 640, 2, 128, dtype=torch.bfloat16, device="npu")
        value = torch.randn(1, 640, 2, 128, dtype=torch.bfloat16, device="npu")

        monkeypatch.setenv("MINDIESD_FP8_FIA_QCHUNK", "1")
        ref = kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 384, layout="BSND")
        monkeypatch.setenv("MINDIESD_FP8_FIA_QCHUNK", "3")
        out = kv_quant_npu.fp8_rotate_quant_kv_slice(query, key, value, 384, layout="BSND")

        assert out.shape == ref.shape
        assert out.dtype == ref.dtype
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
