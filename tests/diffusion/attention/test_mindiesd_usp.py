# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Contract tests for the Ascend USP executor backed by mindiesd.parallel.

The executor delegates only sparse-eligible RAINFUSION_ATTN calls to
``mindiesd.parallel.distributed_sparse_attention`` (the quantized EagleQBSA
chain, matching precision="mix" on A5). Everything else must decline so the
native Ulysses path takes over.
"""

from dataclasses import dataclass, replace
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.platforms.npu import usp as usp_module
from vllm_omni.platforms.npu.usp import AscendUSPExecutor

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@dataclass
class _ParallelConfigStub:
    ulysses_degree: int = 4
    ring_degree: int = 1
    allgather_degree: int = 1
    ulysses_mode: str = "strict"


@dataclass
class _SPGroupsStub:
    ulysses_group: object
    ring_group: object
    ring_rank: int = 0


class _FakeParallelState:
    """Stand-in for mindiesd.parallel.ParallelState (construction is counted)."""

    instances: list["_FakeParallelState"] = []

    def __init__(self, *, group=None):
        self.group = group
        self.stream = None
        self.streams_made = 0
        _FakeParallelState.instances.append(self)

    def make_stream(self, device):
        self.streams_made += 1
        self.stream = object()


@pytest.fixture(autouse=True)
def _reset_process_state():
    usp_module._PROCESS_STATE = None
    _FakeParallelState.instances.clear()
    yield
    usp_module._PROCESS_STATE = None
    _FakeParallelState.instances.clear()


def _parallel_config(**overrides):
    return replace(_ParallelConfigStub(), **overrides)


def _parallel_module(distributed_sparse_attention) -> ModuleType:
    module = ModuleType("mindiesd.parallel")
    setattr(module, "distributed_sparse_attention", distributed_sparse_attention)
    setattr(module, "ParallelState", _FakeParallelState)
    return module


def _executor(**overrides):
    config = _parallel_config(**overrides)
    groups = _SPGroupsStub(
        ulysses_group=object(),
        ring_group=object(),
    )
    return AscendUSPExecutor(
        sp_group=groups,
        ulysses_degree=config.ulysses_degree,
        ring_degree=config.ring_degree,
        allgather_degree=config.allgather_degree,
        ulysses_mode=config.ulysses_mode,
    )


# Geometry: 8 ranks, packed S=1224 rows = 100 prefix + 8x128 video + 100 pad.
_TXT, _T, _H, _W = 100, 8, 16, 8
_LOCAL_ROWS = 153  # 1224 / 8
_USED = _TXT + _T * _H * _W  # 1124


def _sparse_plan():
    return {
        "spans": [{"start": _TXT, "latent_shape": [_T, _H, _W]}],
        "used_len": _USED,
        "sparsity": 0.8,
    }


def _h3_metadata():
    return AttentionMetadata(
        extra={
            # Single-request H3 packing is [0, used, packed_total]: one real
            # document plus a padding-tail document.
            "cu_seqlens_q": torch.tensor([0, _USED, _USED + 100], dtype=torch.int32),
            "max_seqlen_q": _USED,
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


def test_parallel_config_exposes_technical_usp_switch():
    config = DiffusionParallelConfig(
        ulysses_degree=2,
        enable_usp=True,
    )

    assert config.sequence_parallel_size == 2
    assert config.enable_usp is True


def test_executor_maps_omni_state_to_mindiesd_parallel_contract(monkeypatch):
    executor = _executor(ulysses_degree=8)
    dsa = Mock(return_value=torch.full((1, _LOCAL_ROWS, 4, 8), 7.0))
    monkeypatch.setattr(executor, "_load_parallel_module", lambda: _parallel_module(dsa))

    output = _try(executor, sparse_plan=_sparse_plan(), metadata=_h3_metadata())

    assert output is dsa.return_value
    q, k, v, spans = dsa.call_args.args
    assert q.shape == k.shape == v.shape == (1, _LOCAL_ROWS, 4, 8)
    assert spans == [{"start": _TXT, "latent_shape": [_T, _H, _W]}]
    kwargs = dsa.call_args.kwargs
    assert kwargs["group"] is executor.sp_group.ulysses_group
    assert kwargs["scale"] == pytest.approx(8**-0.5)
    assert kwargs["sparsity"] == 0.8
    assert kwargs["used_len"] == _USED
    # The process-wide ParallelState is passed so the chain reuses one stream.
    assert kwargs["state"] is _FakeParallelState.instances[0]


def test_process_state_is_shared_and_stream_made_once(monkeypatch):
    executor_a = _executor(ulysses_degree=8)
    executor_b = _executor(ulysses_degree=8)
    dsa = Mock(return_value=torch.zeros(1, _LOCAL_ROWS, 4, 8))
    module = _parallel_module(dsa)
    monkeypatch.setattr(executor_a, "_load_parallel_module", lambda: module)
    monkeypatch.setattr(executor_b, "_load_parallel_module", lambda: module)

    _try(executor_a, sparse_plan=_sparse_plan(), metadata=_h3_metadata())
    _try(executor_b, sparse_plan=_sparse_plan(), metadata=_h3_metadata())

    # Two layers, one ParallelState, one stream: the chain's one-issuer
    # discipline depends on it.
    assert len(_FakeParallelState.instances) == 1
    assert _FakeParallelState.instances[0].streams_made == 1
    assert dsa.call_args_list[0].kwargs["state"] is dsa.call_args_list[1].kwargs["state"]


def test_dense_forward_without_sparse_plan_declines(monkeypatch):
    executor = _executor()
    dsa = Mock(return_value=torch.zeros(1, _LOCAL_ROWS, 4, 8))
    monkeypatch.setattr(executor, "_load_parallel_module", lambda: _parallel_module(dsa))

    assert _try(executor, sparse_plan=None, metadata=_h3_metadata()) is None
    dsa.assert_not_called()


def test_missing_parallel_module_declines():
    executor = _executor()
    executor._parallel_module = None
    executor._load_attempted = True  # pretend the import already failed

    assert _try(executor, sparse_plan=_sparse_plan(), metadata=_h3_metadata()) is None


def test_loader_rejects_module_without_parallel_chain(monkeypatch):
    executor = _executor()
    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: ModuleType("mindiesd.parallel"),  # no distributed_sparse_attention
    )

    assert executor._load_parallel_module() is None


@pytest.mark.parametrize(
    ("executor_overrides", "call_overrides", "metadata", "sparse_plan"),
    [
        # Dense backend: the parallel chain has no dense arm.
        ({}, {"backend_name": "FLASH_ATTN"}, None, _sparse_plan()),
        # Pure Ulysses only: no ring / allgather composition, and SP must be on.
        ({"ring_degree": 2}, {}, None, _sparse_plan()),
        ({"allgather_degree": 2}, {}, None, _sparse_plan()),
        ({"ulysses_degree": 1, "ring_degree": 1}, {}, None, _sparse_plan()),
        ({}, {"causal": True}, None, _sparse_plan()),
        ({}, {"scatter_dim": 1}, None, _sparse_plan()),
        ({}, {}, AttentionMetadata(joint_query=torch.zeros(1, 1, 4, 8)), _sparse_plan()),
        ({}, {}, AttentionMetadata(attn_mask=torch.ones(1, 3, dtype=torch.bool)), _sparse_plan()),
        ({}, {}, AttentionMetadata(full_attn_spans=[[(0, 1)]]), _sparse_plan()),
        ({}, {}, AttentionMetadata(extra={"kv_cache_dtype": "fp8"}), _sparse_plan()),
        # Multi-request step-mode batches cannot be expressed by one span set.
        ({}, {}, AttentionMetadata(extra={"num_requests": 2}), _sparse_plan()),
    ],
)
def test_executor_declines_what_the_chain_cannot_express(
    monkeypatch,
    executor_overrides,
    call_overrides,
    metadata,
    sparse_plan,
):
    executor = _executor(**executor_overrides)
    dsa = Mock(return_value=torch.zeros(1, _LOCAL_ROWS, 4, 8))
    monkeypatch.setattr(executor, "_load_parallel_module", lambda: _parallel_module(dsa))
    call = {
        "attn_metadata": metadata,
        "backend_name": "RAINFUSION_ATTN",
        "causal": False,
        "softmax_scale": 8**-0.5,
        "scatter_dim": 2,
        "gather_dim": 1,
        "sparse_plan": sparse_plan,
    }
    call.update(call_overrides)

    assert (
        executor.try_forward(
            torch.randn(1, _LOCAL_ROWS, 4, 8),
            torch.randn(1, _LOCAL_ROWS, 4, 8),
            torch.randn(1, _LOCAL_ROWS, 4, 8),
            **call,
        )
        is None
    )
    dsa.assert_not_called()


def test_mindiesd_errors_propagate(monkeypatch):
    """The parallel chain has no structured capability exception: errors are real."""
    executor = _executor(ulysses_degree=8)
    dsa = Mock(side_effect=RuntimeError("stale state"))
    monkeypatch.setattr(executor, "_load_parallel_module", lambda: _parallel_module(dsa))

    with pytest.raises(RuntimeError, match="stale state"):
        _try(executor, sparse_plan=_sparse_plan(), metadata=_h3_metadata())


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
    layer.attn_backend = Mock(get_name=Mock(return_value="RAINFUSION_ATTN"))
    # No sparse-plan resolver on this backend (spec=[] keeps getattr at None).
    layer.attention = Mock(spec=[])
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
