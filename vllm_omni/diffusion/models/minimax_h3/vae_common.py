# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Small helpers shared by the MiniMax-H3 VAE adapter and its decode paths.

Kept out of ``vae.py`` so the chunked-decode module can use them without
importing the adapter (which imports the chunked-decode module).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def match_param_dtype(module: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """Align an entry tensor with ``module``'s own parameter precision.

    Reduced-precision VAE residency (``VLLM_OMNI_MINIMAX_H3_VAE_DTYPE``)
    narrows the checkpoint's FP32 weights. The decode path runs under an FP16
    autocast written for the FP32 contract, and that autocast does not cover
    every operator in the remote model, so a FP32 entry tensor can meet a
    narrowed bias and fail on dtype mismatch. Matching on the way in keeps
    every consumer consistent regardless of autocast coverage.
    """
    parameter = next(module.parameters(), None)
    if parameter is None or tensor.dtype is parameter.dtype:
        return tensor
    return tensor.to(parameter.dtype)
