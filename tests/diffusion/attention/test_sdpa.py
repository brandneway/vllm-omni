# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import vllm_omni.diffusion.attention.backends.sdpa as sdpa_backend
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import (
    SDPAImpl,
    _resolve_valid_kv_prefix,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _clear_verified_prefix_cache():
    sdpa_backend._VERIFIED_PREFIX.clear()
    yield
    sdpa_backend._VERIFIED_PREFIX.clear()


def test_sdpa_expands_kv_when_native_gqa_kernel_is_unavailable(monkeypatch):
    calls = []

    def fake_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return query

    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", lambda *args: False)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=8, softmax_scale=0.5)
    output = impl.forward_cuda(
        torch.randn(1, 3, 4, 8),
        torch.randn(1, 3, 2, 8),
        torch.randn(1, 3, 2, 8),
    )

    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert query_shape == key_shape == value_shape == (1, 4, 3, 8)
    assert kwargs["enable_gqa"] is False
    assert output.shape == (1, 3, 4, 8)


def test_sdpa_keeps_compressed_kv_when_native_gqa_kernel_is_available(monkeypatch):
    calls = []

    def fake_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return query

    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", lambda *args: True)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=8, softmax_scale=0.5)
    output = impl.forward_cuda(
        torch.randn(1, 3, 4, 8),
        torch.randn(1, 3, 2, 8),
        torch.randn(1, 3, 2, 8),
    )

    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert query_shape == (1, 4, 3, 8)
    assert key_shape == value_shape == (1, 2, 3, 8)
    assert kwargs["enable_gqa"] is True
    assert output.shape == (1, 3, 4, 8)


# ---------------------------------------------------------------------------
# MiniMax-H3 packed prefix slicing (the quadratic full_qk mask OOM).
# ---------------------------------------------------------------------------

HEADS = 4
HEAD_DIM = 32
SCALE = 1.0 / (HEAD_DIM**0.5)


def _h3_extra(total: int, used: int) -> dict:
    """The packed metadata MiniMax-H3 puts in AttentionMetadata.extra."""
    cu = torch.tensor([0, used, total], dtype=torch.int32)
    return {
        "cu_seqlens_q": cu,
        "cu_seqlens_k": cu.clone(),
        "max_seqlen_q": used,
        "max_seqlen_k": used,
    }


def _h3_metadata(total: int, used: int, *, extra: dict | None = None, attn_mask="default", **overrides):
    cu = torch.tensor([0, used, total], dtype=torch.int32)
    if extra is None:
        extra = _h3_extra(total, used)
    mask = (torch.arange(total)[None] < used) if attn_mask == "default" else attn_mask
    metadata = AttentionMetadata(attn_mask=mask, extra=extra)
    for name, value in overrides.items():
        setattr(metadata, name, value)
    return metadata


def _packed_tensors(total: int, *, dtype=torch.float32, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(1, total, HEADS, HEAD_DIM, dtype=dtype, generator=generator) for _ in range(3)
    )


def _masked_reference(query, key, value, used: int, scale: float):
    """What the old full_qk path computes: [B, Skv] mask expanded to [B, 1, Sq, Skv]."""
    batch, seq_len = query.shape[0], query.shape[1]
    mask = (torch.arange(seq_len, device=query.device)[None] < used).to(torch.bool)
    mask = mask.unsqueeze(1).expand(batch, seq_len, seq_len).unsqueeze(1).contiguous()
    q, k, v = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return out.permute(0, 2, 1, 3)


@pytest.mark.parametrize("total,used", [(128, 100), (512, 331), (1024, 700)])
def test_npu_prefix_slice_matches_masked_reference(total, used):
    """Truncating K/V must reproduce the masked result. This is the whole point."""
    query, key, value = _packed_tensors(total)
    impl = SDPAImpl(num_heads=HEADS, num_kv_heads=HEADS, head_size=HEAD_DIM, softmax_scale=SCALE)

    got = impl.forward_npu(query, key, value, _h3_metadata(total, used))
    want = _masked_reference(query, key, value, used, SCALE)

    assert got.shape == query.shape
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_npu_prefix_slice_matches_masked_reference_bf16():
    total, used = 512, 331
    query, key, value = _packed_tensors(total, dtype=torch.bfloat16)
    impl = SDPAImpl(num_heads=HEADS, num_kv_heads=HEADS, head_size=HEAD_DIM, softmax_scale=SCALE)

    got = impl.forward_npu(query, key, value, _h3_metadata(total, used))
    want = _masked_reference(query, key, value, used, SCALE)

    torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)


def test_npu_prefix_slice_preserves_padding_row_outputs():
    """Unlike a varlen/TND path, padding *query* rows keep the same values.

    The H3 mask does not depend on the query row, so truncated K/V and masked
    K/V agree on every row -- including rows past ``used``.
    """
    total, used = 320, 220
    query, key, value = _packed_tensors(total)
    impl = SDPAImpl(num_heads=HEADS, num_kv_heads=HEADS, head_size=HEAD_DIM, softmax_scale=SCALE)

    got = impl.forward_npu(query, key, value, _h3_metadata(total, used))
    want = _masked_reference(query, key, value, used, SCALE)

    torch.testing.assert_close(got[:, used:], want[:, used:], atol=1e-5, rtol=1e-5)


def test_npu_prefix_slice_skips_mask_materialization(monkeypatch):
    """The 38 GiB allocation is gone: no reshape helper, no mask into SDPA."""
    total, used = 256, 200
    reshape_calls = []
    sdpa_calls = []

    monkeypatch.setattr(
        sdpa_backend,
        "_maybe_reshape_attn_mask",
        lambda *args, **kwargs: reshape_calls.append(args),
    )

    def fake_sdpa(query, key, value, **kwargs):
        sdpa_calls.append((query.shape, key.shape, value.shape, kwargs))
        return query

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    query, key, value = _packed_tensors(total)
    impl = SDPAImpl(num_heads=HEADS, num_kv_heads=HEADS, head_size=HEAD_DIM, softmax_scale=SCALE)
    impl.forward_npu(query, key, value, _h3_metadata(total, used))

    assert reshape_calls == []
    query_shape, key_shape, value_shape, kwargs = sdpa_calls[0]
    assert query_shape == (1, HEADS, total, HEAD_DIM)  # Q untouched
    assert key_shape[2] == used  # permuted to BNSD: sequence is dim 2
    assert value_shape[2] == used
    assert kwargs["attn_mask"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "causal",
        "missing_cu_seqlens_q",
        "missing_cu_seqlens_k",
        "missing_max_seqlen_q",
        "missing_max_seqlen_k",
        "max_seqlen_is_tensor",
        "cu_has_four_docs",
        "cu_has_one_doc",
        "cu_shape_mismatch",
        "max_seqlens_differ",
        "query_seq_mismatch",
        "batch_two",
        "no_mask",
        "four_d_attn_mask",
        "float_attn_mask",
        "no_padding",
        "used_zero",
        "padding_longer_than_real",
        "no_metadata",
    ],
)
def test_resolve_valid_kv_prefix_rejects_non_contract(mutation):
    total, used = 512, 331
    impl_causal = mutation == "causal"
    query, key, value = _packed_tensors(total)

    if mutation == "no_metadata":
        metadata = None
    elif mutation == "missing_cu_seqlens_q":
        metadata = _h3_metadata(total, used, extra={k: v for k, v in _h3_extra(total, used).items() if k != "cu_seqlens_q"})
    elif mutation == "missing_cu_seqlens_k":
        metadata = _h3_metadata(total, used, extra={k: v for k, v in _h3_extra(total, used).items() if k != "cu_seqlens_k"})
    elif mutation == "missing_max_seqlen_q":
        metadata = _h3_metadata(total, used, extra={k: v for k, v in _h3_extra(total, used).items() if k != "max_seqlen_q"})
    elif mutation == "missing_max_seqlen_k":
        metadata = _h3_metadata(total, used, extra={k: v for k, v in _h3_extra(total, used).items() if k != "max_seqlen_k"})
    elif mutation == "max_seqlen_is_tensor":
        extra = _h3_extra(total, used)
        extra["max_seqlen_k"] = torch.tensor(used, dtype=torch.int32)
        metadata = _h3_metadata(total, used, extra=extra)
    elif mutation == "cu_has_four_docs":
        extra = _h3_extra(total, used)
        extra["cu_seqlens_q"] = torch.tensor([0, 10, used, total], dtype=torch.int32)
        extra["cu_seqlens_k"] = extra["cu_seqlens_q"].clone()
        metadata = _h3_metadata(total, used, extra=extra)
    elif mutation == "cu_has_one_doc":
        extra = _h3_extra(total, used)
        extra["cu_seqlens_q"] = torch.tensor([total], dtype=torch.int32)
        extra["cu_seqlens_k"] = extra["cu_seqlens_q"].clone()
        metadata = _h3_metadata(total, used, extra=extra)
    elif mutation == "cu_shape_mismatch":
        extra = _h3_extra(total, used)
        extra["cu_seqlens_k"] = torch.tensor([0, total], dtype=torch.int32)
        metadata = _h3_metadata(total, used, extra=extra)
    elif mutation == "max_seqlens_differ":
        extra = _h3_extra(total, used)
        extra["max_seqlen_q"] = used - 1
        metadata = _h3_metadata(total, used, extra=extra)
    elif mutation == "query_seq_mismatch":
        metadata = _h3_metadata(total, used)
        query = query[:, : total - 8]
    elif mutation == "batch_two":
        metadata = _h3_metadata(total, used)
        query = query.repeat(2, 1, 1, 1)
        key = key.repeat(2, 1, 1, 1)
        value = value.repeat(2, 1, 1, 1)
    elif mutation == "no_mask":
        metadata = _h3_metadata(total, used, attn_mask=None)
    elif mutation == "four_d_attn_mask":
        metadata = _h3_metadata(total, used, attn_mask=torch.zeros(1, 1, total, total, dtype=torch.bool))
    elif mutation == "float_attn_mask":
        metadata = _h3_metadata(total, used, attn_mask=torch.zeros(1, total))
    elif mutation == "no_padding":
        metadata = _h3_metadata(total, total)
    elif mutation == "used_zero":
        metadata = _h3_metadata(total, 0)
    elif mutation == "padding_longer_than_real":
        # cu = [0, 30, 64]: the aligned tail is longer than the real document.
        query, key, value = _packed_tensors(64)
        metadata = _h3_metadata(64, 30)
    else:
        metadata = _h3_metadata(total, used)

    assert (
        _resolve_valid_kv_prefix(query, key, value, metadata, causal=impl_causal) is None
    ), f"contract unexpectedly accepted for mutation={mutation!r}"


def test_resolve_valid_kv_prefix_returns_used_for_h3_metadata():
    total, used = 512, 331
    query, key, value = _packed_tensors(total)
    metadata = _h3_metadata(total, used)

    assert _resolve_valid_kv_prefix(query, key, value, metadata, causal=False) == used


def test_resolve_valid_kv_prefix_does_not_sync_in_steady_state(monkeypatch):
    """After the one-shot mask check, the resolver must stay host-side only."""
    total, used = 512, 331
    query, key, value = _packed_tensors(total)
    metadata = _h3_metadata(total, used)

    assert _resolve_valid_kv_prefix(query, key, value, metadata, causal=False) == used

    def _no_sync(*args, **kwargs):
        raise AssertionError("resolver must not read device data on the steady-state path")

    for name in ("item", "cpu", "tolist", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, _no_sync)

    assert _resolve_valid_kv_prefix(query, key, value, metadata, causal=False) == used


def test_npu_falls_back_to_full_mask_when_contract_fails(monkeypatch):
    """The revocation case: no contract, no slicing, byte-for-byte old behaviour."""
    total, used = 256, 200
    sdpa_calls = []

    def fake_sdpa(query, key, value, **kwargs):
        sdpa_calls.append((query.shape, key.shape, value.shape, kwargs))
        return query

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    query, key, value = _packed_tensors(total)
    # padding_longer_than_real: the guard must reject and keep the mask path.
    metadata = _h3_metadata(total, 100)
    impl = SDPAImpl(num_heads=HEADS, num_kv_heads=HEADS, head_size=HEAD_DIM, softmax_scale=SCALE)
    impl.forward_npu(query, key, value, metadata)

    _, key_shape, _, kwargs = sdpa_calls[0]
    assert key_shape[2] == total  # K not sliced
    mask = kwargs["attn_mask"]
    assert mask is not None
    assert mask.shape == (1, 1, total, total)
    assert mask.is_contiguous()


def test_npu_prefix_slice_can_be_disabled_by_env(monkeypatch):
    total, used = 256, 200
    sdpa_calls = []

    def fake_sdpa(query, key, value, **kwargs):
        sdpa_calls.append((query.shape, key.shape, kwargs))
        return query

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    monkeypatch.setenv(sdpa_backend._PREFIX_SLICE_ENV, "0")

    query, key, value = _packed_tensors(total)
    impl = SDPAImpl(num_heads=HEADS, num_kv_heads=HEADS, head_size=HEAD_DIM, softmax_scale=SCALE)
    impl.forward_npu(query, key, value, _h3_metadata(total, used))

    _, key_shape, kwargs = sdpa_calls[0]
    assert key_shape[2] == total
    assert kwargs["attn_mask"] is not None
