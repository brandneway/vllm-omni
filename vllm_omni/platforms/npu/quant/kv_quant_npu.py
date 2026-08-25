# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 quantization utilities for diffusion attention tensors.

Provides per-tensor dynamic quantization of Q/K/V tensors to
float8_e4m3fn format. Designed for diffusion models where Q/K/V are
computed fresh each forward pass (no persistent KV cache).

Two entry points dispatch to the MindIE-SD FIA operator:
``fp8_rotate_quant_fa`` for dense batched layouts (BNSD/BSND) and
``fp8_rotate_quant_fa_varlen`` for packed varlen TND tensors with
cu_seqlens document boundaries. Setting ``MINDIESD_SET_FREQ_MANUAL`` to
a truthy value caps the AI-core frequency around the varlen FIA call
via mindiesd ``frequency_regulator``.
"""

from __future__ import annotations

import math
import os
import threading
from functools import lru_cache

import torch

# Hadamard rotation matrix for QuaRot-style preprocessing
# keyed by (device, dtype, head_dim) to avoid matmul dtype mismatch.
_ROT_MATRIXS: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}
_ROT_MATRIX_LOCK = threading.Lock()

_FP8_KV_LABELS = frozenset({"fp8"})

# AI-core frequency caps (MHz) applied around the varlen FIA call when
# MINDIESD_SET_FREQ_MANUAL is truthy: one wide-head_num FIA call is a burst
# of instantaneous compute that can trip NPU power management into
# downclocking the chip. Capping the frequency before the call keeps the
# burst inside the power envelope; the cap is lifted right after.
_FREQ_CAP_BEFORE_VARLEN_FIA = 1400
_FREQ_RESTORE_AFTER_VARLEN_FIA = 1650


def is_quantized_kv_cache(kv_cache_dtype: str | None) -> bool:
    """True if config requests FP8-style KV / QKV quantization for the NPU FA path."""
    return kv_cache_dtype in _FP8_KV_LABELS


@lru_cache(maxsize=1)
def _varlen_freq_caps() -> tuple[int | None, int | None]:
    """(before, after) AI-core frequency caps for the varlen FIA call.

    Returns (None, None) unless MINDIESD_SET_FREQ_MANUAL is truthy
    (``1``/``true``/``yes``/``on``). The env var is read once per process.
    """
    enabled = os.environ.get("MINDIESD_SET_FREQ_MANUAL", "false").strip().lower()
    if enabled in ("1", "true", "yes", "on"):
        return _FREQ_CAP_BEFORE_VARLEN_FIA, _FREQ_RESTORE_AFTER_VARLEN_FIA
    return None, None


@lru_cache(maxsize=1)
def _load_quant_ops():
    try:
        import torch_npu
        from mindiesd import frequency_regulator
        from mindiesd.layers.flash_attn.fused_infer_attention_score import fused_infer_attention_score_v2
        from mindiesd.layers.quant.block_quant import (
            fa_block_quant_preprocess,
            fa_block_quant_preprocess_varlen,
        )
        from msmodelslim.processor.quarot.common.quarot_utils import QuaRotMode, create_rot
    except ImportError as e:
        raise ImportError(
            "fp8_rotate_quant_fa requires torch_npu, MindIE-SD (mindiesd), and MSModelSlim. "
            "See https://gitcode.com/Ascend/MindIE-SD and https://gitcode.com/Ascend/msmodelslim"
        ) from e
    return (
        torch_npu,
        fused_infer_attention_score_v2,
        fa_block_quant_preprocess,
        fa_block_quant_preprocess_varlen,
        QuaRotMode,
        create_rot,
        frequency_regulator,
    )


@lru_cache(maxsize=1)
def _freq_stream_mode() -> bool:
    """Truthy MINDIESD_SET_FREQ_STREAM dispatches the op on a dedicated stream."""
    enabled = os.environ.get("MINDIESD_SET_FREQ_STREAM", "false").strip().lower()
    return enabled in ("1", "true", "yes", "on")


# Dedicated stream and reused events for the opt-in stream dispatch mode
# (created lazily on first use).
_FREQ_STREAM: torch.npu.Stream | None = None
_FREQ_COMPUTE_DONE: torch.npu.Event | None = None  # compute stream → freq stream
_FREQ_REG_DONE: torch.npu.Event | None = None  # freq stream → compute stream


def _get_freq_stream() -> tuple[torch.npu.Stream, torch.npu.Event, torch.npu.Event]:
    global _FREQ_STREAM, _FREQ_COMPUTE_DONE, _FREQ_REG_DONE
    if _FREQ_STREAM is None:
        _FREQ_STREAM = torch.npu.Stream()
        _FREQ_COMPUTE_DONE = torch.npu.Event()
        _FREQ_REG_DONE = torch.npu.Event()
    return _FREQ_STREAM, _FREQ_COMPUTE_DONE, _FREQ_REG_DONE


def _freq_cap_before_fia(frequency_regulator, freq: int) -> None:
    """Cap the AI-core frequency before the FIA call.

    Default dispatch is a plain call on the compute stream: the op-side
    lifetime fix keeps the async executor's storage alive, and the kernel
    serializes with the surrounding compute work. The op enqueues its kernel
    from a background AICPU thread, so exact enqueue position is not
    guaranteed — acceptable for a power knob.

    With MINDIESD_SET_FREQ_STREAM truthy, dispatch on a dedicated stream
    instead (host never blocks; ordering enforced device-side with events),
    which lets the frequency kernel run concurrently with compute-stream
    work.
    """
    if not _freq_stream_mode():
        frequency_regulator(freq)
        return
    stream, compute_done, reg_done = _get_freq_stream()
    cur = torch.npu.current_stream()
    compute_done.record(cur)  # engage the cap only after current compute work
    stream.wait_event(compute_done)
    with torch.npu.stream(stream):
        frequency_regulator(freq)
    reg_done.record(stream)
    cur.wait_event(reg_done)  # the following FIA waits until the cap is set


def _freq_restore_after_fia(frequency_regulator, freq: int) -> None:
    """Restore the AI-core frequency after the FIA call."""
    if not _freq_stream_mode():
        frequency_regulator(freq)
        return
    stream, compute_done, _ = _get_freq_stream()
    cur = torch.npu.current_stream()
    compute_done.record(cur)  # restore only once the FIA work has completed
    stream.wait_event(compute_done)
    with torch.npu.stream(stream):
        frequency_regulator(freq)


def _get_rot_matrix(
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
    qua_rot_mode,
    create_rot,
) -> torch.Tensor:
    key = (device, dtype, head_dim)
    with _ROT_MATRIX_LOCK:
        rot = _ROT_MATRIXS.get(key)
        if rot is None:
            rot = create_rot(qua_rot_mode.HADAMARD, head_dim, seed=425500).to(device=device, dtype=dtype)
            _ROT_MATRIXS[key] = rot
    return rot


def fp8_rotate_quant_fa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layout: str = "BNSD",
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run NPU fused attention with dynamic FP8 Q/K/V and optional QuaRot preprocess.

    Args:
        query: Query tensor in ``layout`` order (default BNSD: batch, heads, seq, dim).
        key: Key tensor in ``layout`` order (default BNSD: batch, heads, seq, dim).
        value: Value tensor in ``layout`` order (default BNSD: batch, heads, seq, dim).
        layout: ``BNSD`` or ``BSND`` for ``npu_fused_infer_attention_score_v2``.
        softmax_scale: If None, uses ``1 / sqrt(head_dim)``.

    Returns:
        Attention output in the same layout as inputs.
    """
    torch_npu, fia_v2, fa_block_quant_preprocess, _varlen_quant, qua_rot_mode, create_rot, _frequency_regulator = (
        _load_quant_ops()
    )

    out_dtype = query.dtype
    device = query.device

    if layout == "BNSD":
        _, n, s, d = query.shape
    elif layout == "BSND":
        _, s, n, d = query.shape
    else:
        raise ValueError(f"fp8_rotate_quant_fa: unsupported layout {layout!r}, expected BNSD or BSND")

    rot = _get_rot_matrix(device, query.dtype, d, qua_rot_mode, create_rot)
    q_f = torch.matmul(query, rot)
    k_f = torch.matmul(key, rot)

    q, q_scale = fa_block_quant_preprocess(q_f, block_size=128, dst_type=torch_npu.float8_e4m3fn, layout=layout)
    k, k_scale = fa_block_quant_preprocess(k_f, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout=layout)
    v, v_scale = fa_block_quant_preprocess(value, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout=layout)

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)

    out = fia_v2(
        q,
        k,
        v,
        input_layout=layout,
        num_query_heads=n,
        softmax_scale=scale,
        pre_tokens=2147483647,  # INT32_MAX: no left-context truncation.
        next_tokens=2147483647,  # INT32_MAX: no right-context truncation.
        query_quant_mode=7,  # NPU mode id for block FP8 dequant path.
        key_quant_mode=7,  # Same quant mode as query branch.
        value_quant_mode=7,  # Same quant mode as key/query branches.
        dequant_scale_query=q_scale,
        dequant_scale_key=k_scale,
        dequant_scale_value=v_scale,
        out_dtype=out_dtype,
    )[0]

    if out.shape[2] != s:
        if layout == "BNSD":
            out = out[:, :, :s, :]
        elif layout == "BSND":
            out = out[:, :s, :, :]

    return out


def fp8_rotate_quant_fa_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: list[int],
    cu_seqlens_k: list[int],
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run packed varlen NPU fused attention with dynamic FP8 Q/K/V.

    Inputs are packed TND ``(total_tokens, num_heads, head_dim)`` tensors.
    ``cu_seqlens_q``/``cu_seqlens_k`` are host lists of cumulative document
    ends (``[doc1_end, doc2_end, ...]``); a cu_seqlens-style leading zero is
    accepted and stripped. Documents attend fully within themselves and
    never across boundaries (non-causal varlen semantics, matching mindiesd
    ``attention_forward_varlen``).

    The TND tensors are viewed as NTD and dispatched with the combined
    ``NTD_TND`` layout (NTD query packing, token-major TND output) — the
    only varlen form the MindIE-SD FIA operator's FP8 per-block path accepts
    — and quantized with :func:`fa_block_quant_preprocess_varlen` so
    quantization blocks never cross document boundaries.

    When MINDIESD_SET_FREQ_MANUAL is truthy, the AI-core frequency is
    capped to 1400 MHz before the FIA call and restored to 1650 MHz after
    via mindiesd ``frequency_regulator`` (dispatched on the compute stream;
    set MINDIESD_SET_FREQ_STREAM for a dedicated-stream dispatch), to keep
    the FIA compute burst inside the NPU power envelope.

    Returns the attention output in the same TND layout as the inputs.
    """
    (
        torch_npu,
        fia_v2,
        _dense_quant,
        fa_block_quant_preprocess_varlen,
        qua_rot_mode,
        create_rot,
        frequency_regulator,
    ) = _load_quant_ops()

    if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
        raise ValueError(
            f"fp8_rotate_quant_fa_varlen: expected packed TND 3D tensors, got "
            f"{query.dim()}D/{key.dim()}D/{value.dim()}D."
        )
    # Normalize to per-document cumulative ends: the op contract is
    # actual_seq = cu_seqlens[1:], so strip a leading zero when present.
    ends_q = list(cu_seqlens_q)
    ends_k = list(cu_seqlens_k)
    if ends_q and ends_q[0] == 0:
        ends_q = ends_q[1:]
    if ends_k and ends_k[0] == 0:
        ends_k = ends_k[1:]
    total_len, num_heads, head_dim = query.shape
    num_kv_heads = key.shape[1]
    out_dtype = query.dtype
    device = query.device

    rot = _get_rot_matrix(device, query.dtype, head_dim, qua_rot_mode, create_rot)
    q_f = torch.matmul(query, rot)
    k_f = torch.matmul(key, rot)

    # TND -> NTD view: [N, T, D] is the FIA varlen layout and matches the
    # [N, S, D] expectation of the block-quant kernel.
    q_ntd = q_f.transpose(0, 1)
    k_ntd = k_f.transpose(0, 1)
    v_ntd = value.transpose(0, 1)

    q, q_scale, q_aligned_ends = fa_block_quant_preprocess_varlen(
        q_ntd, ends_q, block_size=128, dst_type=torch_npu.float8_e4m3fn
    )
    k, k_scale, kv_aligned_ends = fa_block_quant_preprocess_varlen(
        k_ntd, ends_k, block_size=256, dst_type=torch_npu.float8_e4m3fn
    )
    v, v_scale, _ = fa_block_quant_preprocess_varlen(
        v_ntd, ends_k, block_size=256, dst_type=torch_npu.float8_e4m3fn
    )

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)

    freq_cap, freq_restore = _varlen_freq_caps()
    if freq_cap is not None:
        _freq_cap_before_fia(frequency_regulator, freq_cap)
    out = fia_v2(
        q,
        k,
        v,
        input_layout="NTD_TND",  # NTD query packing, token-major TND output
        num_query_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        softmax_scale=scale,
        pre_tokens=2147483647,  # INT32_MAX: full attention within each document.
        next_tokens=2147483647,
        sparse_mode=0,
        actual_seq_qlen=list(q_aligned_ends),
        actual_seq_kvlen=list(kv_aligned_ends),
        query_quant_mode=7,
        key_quant_mode=7,
        value_quant_mode=7,
        dequant_scale_query=q_scale,
        dequant_scale_key=k_scale,
        dequant_scale_value=v_scale,
        out_dtype=out_dtype,
    )[0]
    if freq_restore is not None:
        _freq_restore_after_fia(frequency_regulator, freq_restore)

    # The op may hand back either [N, T, D] or a token-major [T, N, D];
    # normalize to token-major first.
    if out.shape[0] == num_heads and out.shape[0] != out.shape[1]:
        out = out.transpose(0, 1)
    # Documents were padded to block multiples inside the packing; gather the
    # real rows of each document back into the original packed layout.
    real_starts = [0, *ends_q[:-1]]
    aligned_starts = [0, *q_aligned_ends[:-1]]
    out = torch.cat(
        [
            out[a : a + (e - s)]
            for (s, e), a in zip(zip(real_starts, ends_q), aligned_starts)
        ],
        dim=0,
    )
    return out.contiguous()
