# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 quantization utilities for diffusion attention tensors.

Provides per-tensor dynamic quantization of Q/K/V tensors to
float8_e4m3fn format. Designed for diffusion models where Q/K/V are
computed fresh each forward pass (no persistent KV cache).

Three entry points dispatch to the MindIE-SD FIA operator:
``fp8_rotate_quant_fa`` for dense batched layouts (BNSD/BSND),
``fp8_rotate_quant_fa_varlen`` for packed varlen TND tensors with
cu_seqlens document boundaries, and ``fp8_rotate_quant_kv_slice`` for the
packed [real, pad] layout with K/V sliced to the valid prefix so a plain
dense BNSD/BSND FIA call (no varlen feature) suffices. Setting
``MINDIESD_SET_FREQ_MANUAL`` to a truthy value caps the AI-core frequency
around the wide FIA calls via mindiesd ``frequency_regulator``.

Setting ``MINDIESD_FP8_FIA_QCHUNK`` to an integer N > 1 splits the dense
kv-slice FIA call into up to N smaller calls along the query sequence axis
(K/V kept whole), replacing one wide attention burst with N narrower ones
to stay inside the NPU power envelope. Chunk boundaries align to the Q
block-quant row block (128 rows), so per-chunk quantization is identical to
the full-length quantization. Only the dense kv-slice path is chunked; the
varlen path warns and keeps a single call.
"""

from __future__ import annotations

import math
import os
import threading
import warnings
from functools import lru_cache

import torch

# Hadamard rotation matrix for QuaRot-style preprocessing
# keyed by (device, dtype, head_dim) to avoid matmul dtype mismatch.
_ROT_MATRIXS: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}
_ROT_MATRIX_LOCK = threading.Lock()

_FP8_KV_LABELS = frozenset({"fp8"})

# Block-quant row-block sizes for the FIA per-block FP8 path. Q chunk
# boundaries (MINDIESD_FP8_FIA_QCHUNK) must align to _Q_BLOCK_SIZE so chunked
# quantization stays block-identical to full-length quantization.
_Q_BLOCK_SIZE = 128
_KV_BLOCK_SIZE = 256

# AI-core frequency caps (MHz) applied around wide FIA calls when
# MINDIESD_SET_FREQ_MANUAL is truthy: one wide-head_num FIA call is a burst
# of instantaneous compute that can trip NPU power management into
# downclocking the chip. Capping the frequency before the call keeps the
# burst inside the power envelope; the cap is lifted right after.
_FREQ_CAP_BEFORE_VARLEN_FIA = 1400
_FREQ_RESTORE_AFTER_VARLEN_FIA = 1650


def is_quantized_kv_cache(kv_cache_dtype: str | None) -> bool:
    """True if config requests FP8-style KV / QKV quantization for the NPU FA path."""
    return kv_cache_dtype in _FP8_KV_LABELS


def fp8_kv_slice_enabled() -> bool:
    """Truthy ``MINDIESD_FP8_KV_SLICE`` (``1``/``true``/``yes``/``on``) routes
    packed FP8 attention through :func:`fp8_rotate_quant_kv_slice` — K/V sliced
    to the valid prefix, then dense BNSD/BSND FIA — instead of the TND varlen
    FIA path. Read per call (no cache) so a changed environment takes effect
    without a process restart, matching the ``MINDIE_SD_FA_TYPE`` dispatch."""
    enabled = os.environ.get("MINDIESD_FP8_KV_SLICE", "false").strip().lower()
    return enabled in ("1", "true", "yes", "on")


def _fia_q_chunk_count() -> int:
    """Number of query-sequence chunks for the dense kv-slice FIA call.

    ``MINDIESD_FP8_FIA_QCHUNK`` N > 1 splits the single wide FIA call into up
    to N smaller ones (K/V kept whole), shortening each compute burst so the
    NPU power management is less likely to downclock. 1 (default) keeps the
    single wide call. Read per call (no cache) so a changed environment takes
    effect without a process restart, matching ``fp8_kv_slice_enabled``.
    """
    raw = os.environ.get("MINDIESD_FP8_FIA_QCHUNK", "1").strip()
    try:
        n = int(raw)
    except ValueError as e:
        raise ValueError(f"MINDIESD_FP8_FIA_QCHUNK must be an integer, got {raw!r}") from e
    return max(1, n)


def _q_chunk_bounds(seq_len: int, n_chunks: int) -> list[tuple[int, int]]:
    """``[start, end)`` query-row chunk boundaries covering ``[0, seq_len)``.

    Boundaries align to the Q block-quant row block (``_Q_BLOCK_SIZE``) so
    per-chunk quantization blocks and dequant scales are identical to the
    full-length quantization; only the last chunk may be ragged. Fewer than
    ``n_chunks`` chunks are returned when there are not enough row blocks.
    """
    if n_chunks <= 1 or seq_len <= _Q_BLOCK_SIZE:
        return [(0, seq_len)]
    chunk = -(-seq_len // (n_chunks * _Q_BLOCK_SIZE)) * _Q_BLOCK_SIZE
    bounds = []
    start = 0
    while start < seq_len:
        end = min(seq_len, start + chunk)
        bounds.append((start, end))
        start = end
    return bounds


def _freq_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as e:
        raise ValueError(f"{name} must be an integer frequency in MHz, got {raw!r}") from e


@lru_cache(maxsize=1)
def _fia_freq_caps() -> tuple[int | None, int | None]:
    """(before, after) AI-core frequency caps for wide FIA calls.

    Returns (None, None) unless MINDIESD_SET_FREQ_MANUAL is truthy
    (``1``/``true``/``yes``/``on``). Override the default 1400/1650 MHz via
    MINDIESD_SET_FREQ_CAP / MINDIESD_SET_FREQ_RESTORE. Env vars are read
    once per process.
    """
    enabled = os.environ.get("MINDIESD_SET_FREQ_MANUAL", "false").strip().lower()
    if enabled not in ("1", "true", "yes", "on"):
        return None, None
    cap = _freq_env_int("MINDIESD_SET_FREQ_CAP", _FREQ_CAP_BEFORE_VARLEN_FIA)
    restore = _freq_env_int("MINDIESD_SET_FREQ_RESTORE", _FREQ_RESTORE_AFTER_VARLEN_FIA)
    return cap, restore


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


@lru_cache(maxsize=1)
def _freq_sync_mode() -> bool:
    """Truthy MINDIESD_SET_FREQ_SYNC blocks on the current stream after each call."""
    enabled = os.environ.get("MINDIESD_SET_FREQ_SYNC", "false").strip().lower()
    return enabled in ("1", "true", "yes", "on")


# Dedicated stream, its reusable stream context, and reused events for the
# opt-in stream dispatch mode — all created lazily exactly once per process;
# per-call work is only the context enter/exit and event record/wait.
_FREQ_STREAM: torch.npu.Stream | None = None
_FREQ_STREAM_CTX = None  # torch.npu.stream(_FREQ_STREAM), reusable
_FREQ_COMPUTE_DONE: torch.npu.Event | None = None  # compute stream → freq stream
_FREQ_REG_DONE: torch.npu.Event | None = None  # freq stream → compute stream


def _get_freq_stream() -> tuple[torch.npu.Stream, object, torch.npu.Event, torch.npu.Event]:
    global _FREQ_STREAM, _FREQ_STREAM_CTX, _FREQ_COMPUTE_DONE, _FREQ_REG_DONE
    if _FREQ_STREAM is None:
        _FREQ_STREAM = torch.npu.Stream()
        _FREQ_STREAM_CTX = torch.npu.stream(_FREQ_STREAM)
        _FREQ_COMPUTE_DONE = torch.npu.Event()
        _FREQ_REG_DONE = torch.npu.Event()
    return _FREQ_STREAM, _FREQ_STREAM_CTX, _FREQ_COMPUTE_DONE, _FREQ_REG_DONE


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
    work. With MINDIESD_SET_FREQ_SYNC truthy, block on the current stream
    after each call (diagnostic; approximates ASCEND_LAUNCH_BLOCKING for
    this op only).
    """
    if not _freq_stream_mode():
        frequency_regulator(freq)
        if _freq_sync_mode():
            torch.npu.current_stream().synchronize()
        return
    _, freq_ctx, compute_done, reg_done = _get_freq_stream()
    cur = torch.npu.current_stream()
    compute_done.record(cur)  # engage the cap only after current compute work
    _FREQ_STREAM.wait_event(compute_done)
    with freq_ctx:
        frequency_regulator(freq)
    reg_done.record(_FREQ_STREAM)
    cur.wait_event(reg_done)  # the following FIA waits until the cap is set
    if _freq_sync_mode():
        cur.synchronize()


def _freq_restore_after_fia(frequency_regulator, freq: int) -> None:
    """Restore the AI-core frequency after the FIA call."""
    if not _freq_stream_mode():
        frequency_regulator(freq)
        if _freq_sync_mode():
            torch.npu.current_stream().synchronize()
        return
    _, freq_ctx, compute_done, _ = _get_freq_stream()
    cur = torch.npu.current_stream()
    compute_done.record(cur)  # restore only once the FIA work has completed
    _FREQ_STREAM.wait_event(compute_done)
    with freq_ctx:
        frequency_regulator(freq)
    if _freq_sync_mode():
        cur.synchronize()


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
    if _fia_q_chunk_count() > 1:
        warnings.warn(
            "MINDIESD_FP8_FIA_QCHUNK>1 is only implemented for the dense kv-slice "
            "FP8 path; the varlen path keeps a single FIA call.",
            stacklevel=2,
        )
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

    freq_cap, freq_restore = _fia_freq_caps()
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


def fp8_rotate_quant_kv_slice(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_len: int,
    *,
    layout: str = "BSND",
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run dense NPU fused attention with dynamic FP8 Q/K/V after slicing K/V
    to the valid prefix.

    Alternative to :func:`fp8_rotate_quant_fa_varlen` for the packed
    [real, pad] two-document layout: the padding document is a strict suffix,
    so dropping it from K/V is identical to masking it out, and the FIA
    operator's varlen feature (``actual_seq_*`` / ``NTD_TND``) stays off —
    the call is a plain dense ``BNSD``/``BSND`` FIA whose query is longer
    than its K/V. K/V are sliced (zero-copy views) BEFORE rotation and
    quantization, so padding rows are neither attended nor quantized. Query
    keeps its full length; outputs on padding rows are never consumed
    downstream (same contract as the unquantized prefix-K/V-slice path).

    Query padding rows are quantized together with the real rows; per-block
    scales (128-row blocks) can therefore mix both in the boundary block.
    This is exact when the packing pads with zeros (zeros never raise the
    block absmax) — the MiniMax-H3 packing this path is built for.

    When ``MINDIESD_FP8_FIA_QCHUNK`` > 1 the FIA call runs as several
    query-row chunks (see :func:`_q_chunk_bounds`) over the same quantized
    K/V, shortening each compute burst for power management. Quantization is
    done once up front; chunk boundaries are Q-block-aligned so per-chunk
    scales are exact slices of the full-length scales.

    Args:
        query: Query tensor in ``layout`` order; its seq length may exceed
            ``kv_len``.
        key: Key tensor in ``layout`` order; sliced to ``kv_len`` on the seq
            axis before quantization.
        value: Value tensor in ``layout`` order; sliced to ``kv_len`` on the
            seq axis before quantization.
        kv_len: Valid K/V prefix length (real document length of the packed
            row).
        layout: Caller-facing tensor layout, ``BNSD`` or ``BSND``. The FIA
            operator itself is always fed BNSD (the quant kernel's output
            layout) and its output is transposed back for ``BSND`` callers.
        softmax_scale: If None, uses ``1 / sqrt(head_dim)``.

    Returns:
        Attention output in the same layout as the inputs, at the query's
        full sequence length.
    """
    (
        torch_npu,
        fia_v2,
        fa_block_quant_preprocess,
        _varlen_quant,
        qua_rot_mode,
        create_rot,
        frequency_regulator,
    ) = _load_quant_ops()

    if layout == "BNSD":
        _, num_heads, seq_len, head_dim = query.shape
        kv_seq_dim = 2
        num_kv_heads = key.shape[1]
    elif layout == "BSND":
        _, seq_len, num_heads, head_dim = query.shape
        kv_seq_dim = 1
        num_kv_heads = key.shape[2]
    else:
        raise ValueError(f"fp8_rotate_quant_kv_slice: unsupported layout {layout!r}, expected BNSD or BSND")

    kv_total = key.shape[kv_seq_dim]
    if not isinstance(kv_len, int) or not 0 < kv_len <= kv_total:
        raise ValueError(f"fp8_rotate_quant_kv_slice: kv_len must be an int in (0, {kv_total}], got {kv_len!r}")

    out_dtype = query.dtype
    device = query.device

    rot = _get_rot_matrix(device, query.dtype, head_dim, qua_rot_mode, create_rot)
    q_f = torch.matmul(query, rot)
    # Slice K/V to the valid prefix (zero-copy views) before rotation and
    # quantization: pad rows are neither attended nor quantized.
    key = key.narrow(kv_seq_dim, 0, kv_len)
    value = value.narrow(kv_seq_dim, 0, kv_len)
    k_f = torch.matmul(key, rot)

    # fa_block_quant_preprocess always returns BNSD-logical tensors (BSND
    # inputs are transposed before the quant kernel), so the FIA call is
    # always dispatched with input_layout="BNSD" and the output is transposed
    # back to the caller's layout below.
    q, q_scale = fa_block_quant_preprocess(
        q_f, block_size=_Q_BLOCK_SIZE, dst_type=torch_npu.float8_e4m3fn, layout=layout
    )
    k, k_scale = fa_block_quant_preprocess(
        k_f, block_size=_KV_BLOCK_SIZE, dst_type=torch_npu.float8_e4m3fn, layout=layout
    )
    v, v_scale = fa_block_quant_preprocess(
        value, block_size=_KV_BLOCK_SIZE, dst_type=torch_npu.float8_e4m3fn, layout=layout
    )

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)

    freq_cap, freq_restore = _fia_freq_caps()
    if freq_cap is not None:
        _freq_cap_before_fia(frequency_regulator, freq_cap)
    # Chunk the FIA call along the query rows (MINDIESD_FP8_FIA_QCHUNK): one
    # wide attention burst becomes several narrower ones. q may be block-padded
    # beyond seq_len by the quant kernel; chunks cover the quantized rows and
    # each chunk's output keeps only its real rows (pad-row outputs are dropped
    # exactly like the single-call trim below).
    bounds = _q_chunk_bounds(q.shape[2], _fia_q_chunk_count())
    out_parts = []
    for row0, row1 in bounds:
        real_rows = min(row1, seq_len) - row0
        if real_rows <= 0:
            break  # remaining chunks cover only quant padding rows
        out_c = fia_v2(
            q[:, :, row0:row1, :].contiguous(),
            k,
            v,
            input_layout="BNSD",
            num_query_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            softmax_scale=scale,
            pre_tokens=2147483647,  # INT32_MAX: no left-context truncation.
            next_tokens=2147483647,  # INT32_MAX: no right-context truncation.
            query_quant_mode=7,  # NPU mode id for block FP8 dequant path.
            key_quant_mode=7,  # Same quant mode as query branch.
            value_quant_mode=7,  # Same quant mode as key/query branches.
            # Per-chunk Q scale slice: block-aligned boundaries make this the
            # exact block range of the full-length quantization.
            dequant_scale_query=q_scale[:, :, row0 // _Q_BLOCK_SIZE : -(-row1 // _Q_BLOCK_SIZE), :].contiguous(),
            dequant_scale_key=k_scale,
            dequant_scale_value=v_scale,
            out_dtype=out_dtype,
        )[0]
        # The op may hand back a padded seq axis; keep this chunk's real rows.
        if out_c.shape[2] != real_rows:
            out_c = out_c[:, :, :real_rows, :]
        out_parts.append(out_c)
    out = torch.cat(out_parts, dim=2) if len(out_parts) > 1 else out_parts[0]
    if freq_restore is not None:
        _freq_restore_after_fia(frequency_regulator, freq_restore)

    if layout == "BSND":
        out = out.transpose(1, 2)
    return out
