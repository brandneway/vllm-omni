# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass, replace
from types import ModuleType
from typing import TypeAlias
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.platforms.npu.usp import AscendUSPExecutor

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@dataclass
class _ParallelConfigStub:
    ulysses_degree: int = 2
    ring_degree: int = 2
    allgather_degree: int = 1
    ulysses_mode: str = "strict"


@dataclass
class _SPGroupsStub:
    ulysses_group: object
    ring_group: object
    ring_rank: int = 0


USPErrorType: TypeAlias = type[BaseException]


def _parallel_config(**overrides):
    return replace(_ParallelConfigStub(), **overrides)


def _usp_module(usp_attention, usp_error: USPErrorType = RuntimeError) -> ModuleType:
    module = ModuleType("mindiesd.layers.usp")
    setattr(module, "usp_attention", usp_attention)
    setattr(module, "USPError", usp_error)
    # Mirror the real MindIE-SD hierarchy: USPNotSupported is the capability
    # signal eligible for fallback; shape/topology errors are contract bugs and
    # propagate.
    setattr(module, "USPNotSupported", type("USPNotSupported", (usp_error,), {}))
    setattr(module, "USPShapeError", type("USPShapeError", (usp_error,), {}))
    return module


def _executor(ring_rank: int = 0, **overrides):
    config = _parallel_config(**overrides)
    groups = _SPGroupsStub(
        ulysses_group=object(),
        ring_group=object(),
        ring_rank=ring_rank,
    )
    return AscendUSPExecutor(
        sp_group=groups,
        ulysses_degree=config.ulysses_degree,
        ring_degree=config.ring_degree,
        allgather_degree=config.allgather_degree,
        ulysses_mode=config.ulysses_mode,
    )


def test_parallel_config_exposes_technical_usp_switch():
    config = DiffusionParallelConfig(
        ulysses_degree=2,
        enable_usp=True,
    )

    assert config.sequence_parallel_size == 2
    assert config.enable_usp is True


def test_executor_maps_vllm_owned_state_to_explicit_mindie_contract(monkeypatch):
    executor = _executor()
    usp_attention = Mock(return_value=torch.full((1, 3, 4, 8), 7.0))
    module = _usp_module(usp_attention)
    monkeypatch.setattr(executor, "_load_usp_module", lambda: module)

    query = torch.randn(1, 3, 4, 8)
    key = torch.randn(1, 3, 4, 8)
    value = torch.randn(1, 3, 4, 8)
    metadata = AttentionMetadata()

    output = executor.try_forward(
        query,
        key,
        value,
        attn_metadata=metadata,
        backend_name="FLASH_ATTN",
        causal=False,
        softmax_scale=8**-0.5,
        scatter_dim=2,
        gather_dim=1,
    )

    assert output is usp_attention.return_value
    usp_attention.assert_called_once_with(
        query,
        key,
        value,
        ulysses_group=executor.sp_group.ulysses_group,
        kv_gather_group=executor.sp_group.ring_group,
    )


def test_executor_maps_pure_ring_to_kv_gather(monkeypatch):
    executor = _executor(ulysses_degree=1, ring_degree=2)
    usp_attention = Mock(return_value=torch.zeros(1, 3, 4, 8))
    monkeypatch.setattr(
        executor,
        "_load_usp_module",
        lambda: _usp_module(usp_attention),
    )
    query = torch.randn(1, 3, 4, 8)

    executor.try_forward(
        query,
        query,
        query,
        attn_metadata=None,
        backend_name="FLASH_ATTN",
        causal=False,
        softmax_scale=8**-0.5,
        scatter_dim=2,
        gather_dim=1,
    )

    kwargs = usp_attention.call_args.kwargs
    assert kwargs["ulysses_group"] is None
    assert kwargs["kv_gather_group"] is executor.sp_group.ring_group


def test_attention_delegates_before_native_sequence_parallel_collectives():
    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)
    strategy = Mock()
    layer._get_active_parallel_strategy = Mock(return_value=strategy)
    layer._no_parallel_strategy = Mock()
    layer._active_paged_kv_adapter = Mock(return_value=None)
    layer._scheduler_paged_kv = False
    layer.paged_kv_cache_role = None
    layer._kv_cache_dtype = None
    layer._disable_kv_quant = False
    layer._kv_cache_skip_steps = None
    layer._kv_cache_skip_layers = None
    layer.attn_backend = Mock(get_name=Mock(return_value="FLASH_ATTN"))
    layer.causal = False
    layer.softmax_scale = 8**-0.5
    layer.scatter_idx = 2
    layer.gather_idx = 1
    expected = torch.zeros(1, 3, 4, 8)
    layer._usp_executor = Mock(try_forward=Mock(return_value=expected))
    query = torch.randn(1, 3, 4, 8)

    output = layer._forward_impl(query, query, query)

    assert output is expected
    strategy.pre_attention.assert_not_called()
    strategy.post_attention.assert_not_called()


def test_attention_does_not_delegate_outside_sp_sharded_region():
    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)
    layer._no_parallel_strategy = Mock()
    layer._get_active_parallel_strategy = Mock(return_value=layer._no_parallel_strategy)
    layer._active_paged_kv_adapter = Mock(return_value=None)
    layer._scheduler_paged_kv = False
    layer.paged_kv_cache_role = None
    layer._usp_executor = Mock()
    layer.use_ring = False
    layer._with_kv_cache_dtype = Mock(side_effect=lambda metadata: metadata)
    layer._run_local_attention = Mock(return_value=torch.zeros(1, 3, 4, 8))
    layer._no_parallel_strategy.pre_attention.return_value = (
        torch.zeros(1, 3, 4, 8),
        torch.zeros(1, 3, 4, 8),
        torch.zeros(1, 3, 4, 8),
        None,
        object(),
    )
    layer._no_parallel_strategy.post_attention.side_effect = lambda output, _ctx: output
    query = torch.randn(1, 3, 4, 8)

    layer._forward_impl(query, query, query)

    layer._usp_executor.try_forward.assert_not_called()


@pytest.mark.parametrize(
    ("executor_overrides", "call_overrides", "metadata"),
    [
        ({}, {"backend_name": "TORCH_SDPA"}, None),
        ({}, {"causal": True}, None),
        ({}, {"softmax_scale": 0.25}, None),
        ({"ulysses_mode": "advanced_uaa"}, {}, None),
        ({"ulysses_degree": 1, "ring_degree": 1, "allgather_degree": 2}, {}, None),
        ({}, {}, AttentionMetadata(joint_query=torch.zeros(1, 1, 4, 8))),
        ({}, {}, AttentionMetadata(attn_mask=torch.ones(1, 3, dtype=torch.bool))),
        ({}, {}, AttentionMetadata(full_attn_spans=[[(0, 1)]])),
        ({}, {}, AttentionMetadata(extra={"kv_cache_dtype": "fp8"})),
    ],
)
def test_executor_skips_semantics_not_covered_by_mindie(
    monkeypatch,
    executor_overrides,
    call_overrides,
    metadata,
):
    executor = _executor(**executor_overrides)
    usp_attention = Mock(return_value=torch.zeros(1, 3, 4, 8))
    monkeypatch.setattr(
        executor,
        "_load_usp_module",
        lambda: _usp_module(usp_attention),
    )
    query = torch.randn(1, 3, 4, 8)
    call = {
        "attn_metadata": metadata,
        "backend_name": "FLASH_ATTN",
        "causal": False,
        "softmax_scale": 8**-0.5,
        "scatter_dim": 2,
        "gather_dim": 1,
    }
    call.update(call_overrides)

    assert executor.try_forward(query, query, query, **call) is None
    usp_attention.assert_not_called()


def test_executor_falls_back_only_for_structured_mindie_errors(monkeypatch):
    class USPError(RuntimeError):
        pass

    executor = _executor()
    module = _usp_module(Mock(), USPError)
    usp_attention = Mock(side_effect=module.USPNotSupported("unsupported shape"))
    monkeypatch.setattr(
        executor,
        "_load_usp_module",
        lambda: module,
    )
    module.usp_attention = usp_attention
    query = torch.randn(1, 3, 4, 8)

    assert (
        executor.try_forward(
            query,
            query,
            query,
            attn_metadata=None,
            backend_name="FLASH_ATTN",
            causal=False,
            softmax_scale=8**-0.5,
            scatter_dim=2,
            gather_dim=1,
        )
        is None
    )

    # Shape/topology contract violations are bugs, not capability signals.
    usp_attention.side_effect = module.USPShapeError("broken geometry")
    with pytest.raises(module.USPShapeError, match="broken geometry"):
        executor.try_forward(
            query,
            query,
            query,
            attn_metadata=None,
            backend_name="FLASH_ATTN",
            causal=False,
            softmax_scale=8**-0.5,
            scatter_dim=2,
            gather_dim=1,
        )

    usp_attention.side_effect = ValueError("programming error")
    with pytest.raises(ValueError, match="programming error"):
        executor.try_forward(
            query,
            query,
            query,
            attn_metadata=None,
            backend_name="FLASH_ATTN",
            causal=False,
            softmax_scale=8**-0.5,
            scatter_dim=2,
            gather_dim=1,
        )


# --- USP + RainFusion (KV-AllGather sparse) extension -------------------------

# Geometry: 8 ranks (usp4 x cp2), packed S=1224 rows = 100 prefix + 8x128 video
# + 100 pad; the CP boundary at S/2=612 lands exactly on frame 4.
_TXT, _T, _H, _W = 100, 8, 16, 8
_FRAME_ROWS = _H * _W  # 128
_LOCAL_ROWS = 153  # 1224 / 8
_USED = _TXT + _T * _FRAME_ROWS  # 1124


def _sparse_plan():
    return {
        "sparse": "rf_v3",
        "sparsity": 0.8,
        "sparse_precision": "mix",
        "txt_len_kv": _TXT,
        "latent_shape_kv": [_T, _H, _W],
        "kv_used_len": _USED,
    }


def _h3_metadata():
    return AttentionMetadata(
        extra={
            # Single-request H3 packing is [0, used, packed_total]: one real
            # document plus a padding-tail document.
            "cu_seqlens_q": torch.tensor([0, _USED, _USED + 100], dtype=torch.int32),
            "cu_seqlens_k": torch.tensor([0, _USED, _USED + 100], dtype=torch.int32),
            "max_seqlen_q": _USED,
            "max_seqlen_k": _USED,
            "valid_kv_length": _USED,
            "num_requests": 1,
            "npu_attn_varlen": True,
        }
    )


def _try(executor, sparse_plan=None, metadata=None):
    return executor.try_forward(
        torch.randn(1, _LOCAL_ROWS, 4, 8),
        torch.randn(1, _LOCAL_ROWS, 4, 8),
        torch.randn(1, _LOCAL_ROWS, 4, 8),
        attn_metadata=metadata,
        backend_name="RAINFUSION_ATTN",
        causal=False,
        softmax_scale=8**-0.5,
        scatter_dim=2,
        gather_dim=1,
        sparse_plan=sparse_plan,
    )


def test_rainfusion_backend_is_eligible_and_dense_consumes_kv_used_len(monkeypatch):
    executor = _executor(ulysses_degree=4, ring_degree=2)
    usp_attention = Mock(return_value=torch.zeros(1, _LOCAL_ROWS, 4, 8))
    monkeypatch.setattr(executor, "_load_usp_module", lambda: _usp_module(usp_attention))

    out = _try(executor, metadata=_h3_metadata())

    assert out is usp_attention.return_value
    kwargs = usp_attention.call_args.kwargs
    assert kwargs["kv_used_len"] == _USED
    assert "sparse" not in kwargs
    # H3 packed extras are consumed, not rejected.
    assert kwargs["ulysses_group"] is executor.sp_group.ulysses_group
    assert kwargs["kv_gather_group"] is executor.sp_group.ring_group


def test_sparse_plan_first_segment_geometry(monkeypatch):
    executor = _executor(ring_rank=0, ulysses_degree=4, ring_degree=2)
    usp_attention = Mock(return_value=torch.zeros(1, _LOCAL_ROWS, 4, 8))
    monkeypatch.setattr(executor, "_load_usp_module", lambda: _usp_module(usp_attention))

    _try(executor, sparse_plan=_sparse_plan(), metadata=_h3_metadata())

    kwargs = usp_attention.call_args.kwargs
    assert kwargs["sparse"] == "rf_v3"
    assert kwargs["sparsity"] == 0.8
    assert kwargs["sparse_precision"] == "mix"
    assert kwargs["q_row_offset"] == 0
    assert kwargs["txt_len_q"] == _TXT
    assert kwargs["txt_len_kv"] == _TXT
    assert "latent_shape_q" not in kwargs  # derived MindIE-side
    assert kwargs["latent_shape_kv"] == [_T, _H, _W]
    assert kwargs["kv_used_len"] == _USED


def test_sparse_plan_second_segment_geometry(monkeypatch):
    executor = _executor(ring_rank=1, ulysses_degree=4, ring_degree=2)
    usp_attention = Mock(return_value=torch.zeros(1, _LOCAL_ROWS, 4, 8))
    monkeypatch.setattr(executor, "_load_usp_module", lambda: _usp_module(usp_attention))

    _try(executor, sparse_plan=_sparse_plan(), metadata=_h3_metadata())

    kwargs = usp_attention.call_args.kwargs
    assert kwargs["q_row_offset"] == 612
    assert kwargs["txt_len_q"] == 0
    assert kwargs["txt_len_kv"] == _TXT


def test_sparse_plan_mid_frame_boundary_passes_through(monkeypatch):
    executor = _executor(ring_rank=0, ulysses_degree=4, ring_degree=2)
    usp_attention = Mock(return_value=torch.zeros(1, 154, 4, 8))
    monkeypatch.setattr(executor, "_load_usp_module", lambda: _usp_module(usp_attention))

    # 154 local rows -> S=1232 -> boundary at 616, which cuts frame 4 mid-way.
    # MindIE splits the partial-frame rows to dense FA, so this must NOT raise.
    out = executor.try_forward(
        torch.randn(1, 154, 4, 8),
        torch.randn(1, 154, 4, 8),
        torch.randn(1, 154, 4, 8),
        attn_metadata=_h3_metadata(),
        backend_name="RAINFUSION_ATTN",
        causal=False,
        softmax_scale=8**-0.5,
        scatter_dim=2,
        gather_dim=1,
        sparse_plan=_sparse_plan(),
    )

    assert out is usp_attention.return_value
    kwargs = usp_attention.call_args.kwargs
    assert kwargs["q_row_offset"] == 0
    assert kwargs["kv_used_len"] == _USED


def test_sparse_plan_multi_request_packing_raises(monkeypatch):
    executor = _executor(ulysses_degree=4, ring_degree=2)
    usp_attention = Mock(return_value=torch.zeros(1, _LOCAL_ROWS, 4, 8))
    monkeypatch.setattr(executor, "_load_usp_module", lambda: _usp_module(usp_attention))

    metadata = AttentionMetadata(
        extra={
            "cu_seqlens_q": torch.tensor([0, 1000, 2000], dtype=torch.int32),
            "valid_kv_length": 2000,
            "num_requests": 2,
        }
    )
    with pytest.raises(ValueError, match="single-request"):
        _try(executor, sparse_plan=_sparse_plan(), metadata=metadata)
    usp_attention.assert_not_called()


def test_single_request_three_entry_cu_seqlens_is_not_multi_request(monkeypatch):
    # Regression: H3 single-request packing carries a padding-tail document, so
    # cu_seqlens has 3 entries. The multi-request gate must read num_requests,
    # not the cu_seqlens length.
    executor = _executor(ulysses_degree=4, ring_degree=2)
    usp_attention = Mock(return_value=torch.zeros(1, _LOCAL_ROWS, 4, 8))
    monkeypatch.setattr(executor, "_load_usp_module", lambda: _usp_module(usp_attention))

    out = _try(executor, metadata=_h3_metadata())

    assert out is usp_attention.return_value
    assert usp_attention.call_args.kwargs["kv_used_len"] == _USED


def test_kv_used_len_falls_back_to_max_seqlen_q(monkeypatch):
    executor = _executor(ulysses_degree=4, ring_degree=2)
    usp_attention = Mock(return_value=torch.zeros(1, _LOCAL_ROWS, 4, 8))
    monkeypatch.setattr(executor, "_load_usp_module", lambda: _usp_module(usp_attention))

    metadata = AttentionMetadata(extra={"max_seqlen_q": 777})
    _try(executor, metadata=metadata)

    assert usp_attention.call_args.kwargs["kv_used_len"] == 777


def test_pure_ulysses_sparse_plan_uses_full_grid(monkeypatch):
    # ring_degree=1: no KV gather, the whole sequence is local after the A2A.
    executor = _executor(ring_rank=0, ulysses_degree=8, ring_degree=1)
    usp_attention = Mock(return_value=torch.zeros(1, _LOCAL_ROWS, 4, 8))
    monkeypatch.setattr(executor, "_load_usp_module", lambda: _usp_module(usp_attention))

    _try(executor, sparse_plan=_sparse_plan(), metadata=_h3_metadata())

    kwargs = usp_attention.call_args.kwargs
    assert kwargs["kv_gather_group"] is None
    assert kwargs["q_row_offset"] == 0
    assert kwargs["txt_len_q"] == _TXT
