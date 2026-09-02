# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU patches for the MiniMax H3 Qwen3-VL text encoder.

Isolation variant for PR #6040 only: the shared patch module was reduced to
the native-GQA SDPA patch. The RoPE (#6061) and SwiGLU (#6167) patches are
NOT re-added on this branch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_SDPA_PATCHED = False


def _scaled_dot_product_attention_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Run causal SDPA with compressed K/V heads through NPU native GQA."""
    num_heads = query.shape[1]
    num_key_value_heads = key.shape[1]
    num_value_heads = value.shape[1]
    if num_key_value_heads != num_value_heads:
        raise ValueError(
            "GQA requires key and value to have the same number of heads, "
            f"got k_heads={num_key_value_heads} and v_heads={num_value_heads}."
        )
    if num_key_value_heads == 0:
        raise ValueError("GQA requires at least one KV head.")
    if num_heads % num_key_value_heads != 0:
        raise ValueError(
            "GQA requires query heads to be a multiple of KV heads, "
            f"got q_heads={num_heads} and kv_heads={num_key_value_heads}."
        )
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=True,
        enable_gqa=num_heads != num_key_value_heads,
    )


def apply_minimax_h3_qwen3vl_sdpa_patch() -> None:
    """Route MiniMax H3 Qwen3-VL text attention to NPU native GQA."""
    global _SDPA_PATCHED
    if _SDPA_PATCHED:
        return

    from vllm_omni.diffusion.models.minimax_h3 import encoder

    encoder._scaled_dot_product_attention = _scaled_dot_product_attention_npu
    _SDPA_PATCHED = True
    logger.debug("Applied NPU SDPA patch for MiniMax H3 Qwen3-VL text encoder")
