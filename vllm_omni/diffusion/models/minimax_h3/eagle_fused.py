# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Env-gated MindIE-SD Eagle fused operators for the MiniMax-H3 DiT.

Two AscendC operators are wired here behind independent switches:

* ``eagle_split_norm_rope_concat`` fuses the packed-QKV split, the Q/K RMSNorm
  and the Q/K RoPE of every attention layer into one call.
* ``eagle_indexed_gate_norm_modulate`` fuses the gated residual update, the
  second RMSNorm and the AdaLN affine transform of every DiT block.

Both stay off unless their environment variable is truthy, so a deployment can
A/B them without rebuilding, and both require the matching MindIE-SD build;
when the operator is missing the caller keeps its existing chain.
"""

from __future__ import annotations

import os
from importlib.util import find_spec

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Set "1" to replace the attention layer's split + Q/K RMSNorm + RoPE chain
# with MindIE-SD EagleSplitNormRopeConcat.
SPLIT_NORM_ROPE_ENV = "VLLM_OMNI_MINIMAX_H3_EAGLE_SPLIT_NORM_ROPE"
# Set "1" to replace the block's gated residual + RMSNorm + AdaLN chain with
# MindIE-SD EagleIndexedGateNormModulate.
INDEXED_GATE_NORM_MODULATE_ENV = "VLLM_OMNI_MINIMAX_H3_EAGLE_INDEXED_GATE_NORM_MODULATE"

_TRUTHY = frozenset(("1", "true", "yes", "on"))

# EagleSplitNormRopeConcat contract (see MindIE-SD PR #792).
_SPLIT_NORM_ROPE_HEAD_DIM = 128
_SPLIT_NORM_ROPE_ROTARY_DIMS = (32, 64, 96, 128)
_SPLIT_NORM_ROPE_NORM_TYPE = 3  # weighted RMSNorm
_SPLIT_NORM_ROPE_ROPE_TYPE = 1  # half-split (NeoX) rotation used by H3

# EagleIndexedGateNormModulate contract (see MindIE-SD PR #790).
_INDEXED_GATE_NORM_TYPE = 0  # weighted RMSNorm
_INDEXED_GATE_MAX_HIDDEN = 7168

_FLOAT_DTYPES = (torch.bfloat16, torch.float16)


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def split_norm_rope_enabled() -> bool:
    return _env_enabled(SPLIT_NORM_ROPE_ENV)


def indexed_gate_norm_modulate_enabled() -> bool:
    return _env_enabled(INDEXED_GATE_NORM_MODULATE_ENV)


def _mindiesd_op(name: str):
    """Return the registered MindIE-SD custom op, or None when unavailable.

    Importing ``mindiesd`` loads the operator library and registers the fake
    implementations the custom ops need under ``torch.compile``/fake tensors.
    """
    if find_spec("mindiesd") is None:
        return None
    try:
        import mindiesd  # noqa: F401

        return getattr(torch.ops.mindiesd, name, None)
    except Exception:
        return None


def split_norm_rope_available() -> bool:
    """Whether the switch is on and the operator is actually installed."""
    if not split_norm_rope_enabled():
        return False
    if _mindiesd_op("eagle_split_norm_rope_concat") is None:
        logger.warning_once(
            "%s is set but MindIE-SD eagle_split_norm_rope_concat is not registered; "
            "keeping the existing split + fused_qk_norm_rope chain.",
            SPLIT_NORM_ROPE_ENV,
        )
        return False
    logger.info_once("MiniMax-H3 attention uses MindIE-SD EagleSplitNormRopeConcat.")
    return True


def indexed_gate_norm_modulate_available() -> bool:
    """Whether the switch is on and the operator is actually installed."""
    if not indexed_gate_norm_modulate_enabled():
        return False
    if _mindiesd_op("eagle_indexed_gate_norm_modulate") is None:
        logger.warning_once(
            "%s is set but MindIE-SD eagle_indexed_gate_norm_modulate is not registered; "
            "keeping the Triton indexed-modulation kernel.",
            INDEXED_GATE_NORM_MODULATE_ENV,
        )
        return False
    logger.info_once("MiniMax-H3 DiT blocks use MindIE-SD EagleIndexedGateNormModulate.")
    return True


# The packed RoPE table stores [cos half, sin half] side by side, and the fused
# operator wants each half contiguous. The split is a copy, so cache it for the
# table object: every block in a forward step shares one prepared table.
_ROPE_SPLIT_CACHE: tuple[torch.Tensor, int, torch.Tensor, torch.Tensor] | None = None


def _split_rope_table(rope_table: torch.Tensor, rotary_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    global _ROPE_SPLIT_CACHE
    cached = _ROPE_SPLIT_CACHE
    if cached is not None and cached[0] is rope_table and cached[1] == rotary_dim:
        return cached[2], cached[3]
    half = rotary_dim // 2
    cos = rope_table[..., :half].contiguous()
    sin = rope_table[..., half:].contiguous()
    _ROPE_SPLIT_CACHE = (rope_table, rotary_dim, cos, sin)
    return cos, sin


def split_norm_rope_compatible(
    qkv: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    rope_table: torch.Tensor | None,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> bool:
    """Check the per-call geometry against the operator's fixed contract."""
    if qkv.device.type != "npu" or not qkv.is_contiguous():
        return False
    if qkv.dtype not in _FLOAT_DTYPES or qkv.dim() != 2:
        return False
    if head_dim != _SPLIT_NORM_ROPE_HEAD_DIM or rotary_dim not in _SPLIT_NORM_ROPE_ROTARY_DIMS:
        return False
    if num_kv_heads <= 0 or num_heads <= 0 or num_heads % num_kv_heads:
        return False
    total_heads = num_heads + 2 * num_kv_heads
    if qkv.shape[-1] != total_heads * head_dim:
        return False
    if rope_table is None or rope_table.device != qkv.device or rope_table.dtype != qkv.dtype:
        return False
    if rope_table.dim() != 2 or rope_table.shape[-1] != rotary_dim or rope_table.shape[0] != qkv.shape[0]:
        return False
    for weight in (q_weight, k_weight):
        if (
            weight is None
            or weight.device != qkv.device
            or weight.dtype != qkv.dtype
            or not weight.is_contiguous()
            or tuple(weight.shape) != (head_dim,)
        ):
            return False
    return True


def split_norm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed QKV, then RMSNorm + RoPE Q/K, returning SND q/k/v."""
    logger.info_once("EagleSplitNormRopeConcat: fused attention call engaged.")
    packed = qkv.view(qkv.shape[0], num_heads + 2 * num_kv_heads, head_dim)
    cos, sin = _split_rope_table(rope_table, rotary_dim)
    query, key, value = torch.ops.mindiesd.eagle_split_norm_rope_concat(
        packed,
        q_weight,
        k_weight,
        cos,
        sin,
        [num_heads, num_kv_heads, num_kv_heads],
        _SPLIT_NORM_ROPE_NORM_TYPE,
        _SPLIT_NORM_ROPE_ROPE_TYPE,
        rotary_dim,
        float(eps),
        False,
    )
    return query, key, value


def indexed_gate_norm_modulate_compatible(
    residual: torch.Tensor,
    gate: torch.Tensor | None,
    branch: torch.Tensor | None,
    shift: torch.Tensor | None,
    scale: torch.Tensor | None,
    indices: torch.Tensor | None,
    weight: torch.Tensor | None,
) -> bool:
    """Check the per-call geometry against the operator's fixed contract."""
    if residual.device.type != "npu" or residual.dim() != 2:
        return False
    if residual.dtype not in _FLOAT_DTYPES:
        return False
    hidden = residual.shape[-1]
    if hidden < 1 or hidden > _INDEXED_GATE_MAX_HIDDEN:
        return False
    for tensor in (gate, branch, shift, scale, weight):
        if tensor is None or tensor.device != residual.device or tensor.dtype != residual.dtype:
            return False
    if branch.shape != residual.shape:
        return False
    if gate.dim() != 2 or gate.shape[1] != hidden:
        return False
    if shift.shape != gate.shape or scale.shape != gate.shape:
        return False
    if weight.dim() != 1 or weight.shape[0] != hidden:
        return False
    if indices is None or indices.device != residual.device:
        return False
    if indices.dim() != 1 or indices.shape[0] != residual.shape[0]:
        return False
    if indices.dtype not in (torch.int32, torch.int64):
        return False
    return True


def indexed_gate_norm_modulate(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(residual + gate[indices] * branch, RMSNorm -> AdaLN)``."""
    logger.info_once("EagleIndexedGateNormModulate: fused DiT block call engaged.")
    return torch.ops.mindiesd.eagle_indexed_gate_norm_modulate(
        residual,
        gate,
        branch,
        shift,
        scale,
        indices,
        weight=weight,
        bias=None,
        norm_type=_INDEXED_GATE_NORM_TYPE,
        eps=float(eps),
        output_dtype=None,
    )


__all__ = [
    "INDEXED_GATE_NORM_MODULATE_ENV",
    "SPLIT_NORM_ROPE_ENV",
    "indexed_gate_norm_modulate",
    "indexed_gate_norm_modulate_available",
    "indexed_gate_norm_modulate_compatible",
    "indexed_gate_norm_modulate_enabled",
    "split_norm_rope",
    "split_norm_rope_available",
    "split_norm_rope_compatible",
    "split_norm_rope_enabled",
]
