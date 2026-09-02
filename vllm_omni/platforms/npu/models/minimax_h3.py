# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU patches for the MiniMax H3 Qwen3-VL text encoder.

Isolation variant for PR #6167 only: the shared patch module was reduced to
the fused SwiGLU patch. The RoPE (#6061) and native-GQA SDPA (#6040) patches
are NOT re-added on this branch.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torch_npu
from vllm.logger import init_logger

logger = init_logger(__name__)

_SWIGLU_PATCHED = False


def npu_swiglu_from_packed(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply fused SwiGLU to a packed gate/up projection tensor."""
    return torch_npu.npu_swiglu(gate_up, dim=-1)


def _forward_minimax_h3_qwen3vl_text_mlp_npu(self: Any, x: torch.Tensor) -> torch.Tensor:
    """Run the Qwen3-VL MLP with one packed GEMM and fused SwiGLU."""
    gate_up = F.linear(x, self.gate_up_proj.weight)
    return self.down_proj(npu_swiglu_from_packed(gate_up))


def apply_minimax_h3_qwen3vl_swiglu_patch() -> None:
    """Route MiniMax H3 Qwen3-VL text MLP to the Ascend fused SwiGLU path."""
    global _SWIGLU_PATCHED
    if _SWIGLU_PATCHED:
        return

    from vllm_omni.diffusion.models.minimax_h3 import encoder

    encoder.MiniMaxH3Qwen3VLTextMLP.forward = _forward_minimax_h3_qwen3vl_text_mlp_npu  # type: ignore[method-assign]
    _SWIGLU_PATCHED = True
    logger.debug("Applied NPU fused SwiGLU patch for MiniMax H3 Qwen3-VL text encoder")
