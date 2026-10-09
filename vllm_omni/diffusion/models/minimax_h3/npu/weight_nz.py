# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pre-convert MiniMax-H3 DiT linear weights from ND to Ascend FRACTAL_NZ.

Setting::

    VLLM_OMNI_MINIMAX_H3_WEIGHT_NZ=1

casts every eligible DiT linear weight into the ``ACL_FORMAT_FRACTAL_NZ``
layout once, right after the checkpoint has been loaded and the DiT is on the
device, so each denoising step's matmul reads an NZ weight instead of paying a
transdata on the weight side every call.

``npu_format_cast`` only rewrites the storage layout; the values are untouched,
so the matmul result is bit-identical to the ND path (measured maxdiff ``0.0``
for bf16 on Ascend910). The win is shape dependent -- largest on the wide-M
GEMMs the DiT issues, slightly negative on small-M calls -- so the switch is
opt-in and off by default.

Only the DiT is touched: the text encoder and the VAEs keep their ND weights.

Two prerequisites and one hazard are handled here:

* ``torch.npu.config.allow_internal_format`` must be ``True``. ``NPUPlatform``
  sets it for every NPU deployment (``platforms/npu/platform.py``); with it
  ``False``, ``npu_format_cast`` silently returns an ND tensor, so the cast is
  verified afterwards rather than assumed.
* Quantized linears are skipped: their quant method owns the weight layout, and
  mxfp4 already casts its own weights to NZ.
* Offloaded DiTs are skipped entirely. Every offload backend relocates the
  parameters after this hook -- the layerwise ones flatten them to pinned host
  memory, the sequential one moves whole modules -- and an internal-format
  tensor crossing that boundary transdata's back to ND, while the storage
  descriptors only record logical strides and so would not notice.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

logger = init_logger(__name__)

WEIGHT_NZ_ENV = "VLLM_OMNI_MINIMAX_H3_WEIGHT_NZ"

# ACL_FORMAT_FRACTAL_NZ, the format id ``torch_npu.npu_format_cast`` takes.
_ACL_FORMAT_FRACTAL_NZ = 29

# FRACTAL_NZ is a 16x16 fractal layout; fp32 has no NZ matmul path at all.
_NZ_DTYPES = (torch.bfloat16, torch.float16)


def _weight_nz_env_enabled() -> bool:
    return os.environ.get(WEIGHT_NZ_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _npu_format(tensor: torch.Tensor) -> int | None:
    """Read a tensor's ACL format id, or ``None`` when it cannot be read."""
    import torch_npu

    try:
        return int(torch_npu.get_npu_format(tensor))
    except Exception:
        return None


def _is_npu_weight(weight: torch.Tensor) -> bool:
    """NZ is an Ascend storage layout, so only device-resident weights qualify.

    A weight left on the host belongs to an offload path, which the caller has
    already rejected; this is the second line of defence.
    """
    return weight.device.type == "npu"


def _is_eligible_linear(module: nn.Module) -> bool:
    """A vLLM linear whose weight layout no quant method owns.

    ``quant_config is None`` resolves to ``AscendUnquantizedLinearMethod`` on
    NPU, a subclass of ``UnquantizedLinearMethod``, so a single ``isinstance``
    accepts both the plain and the Ascend unquantized case while rejecting every
    quantized one (fp8/int8/mxfp8 build their own storage).
    """
    if not isinstance(module, LinearBase):
        return False
    return isinstance(getattr(module, "quant_method", None), UnquantizedLinearMethod)


def install_minimax_h3_dit_weight_nz(model: nn.Module, od_config: Any | None = None) -> int:
    """Cast eligible DiT linear weights ND -> FRACTAL_NZ.

    Returns the number of weights actually left in the NZ layout, which is ``0``
    when the env opt-in is off, when an offload backend owns the parameters, or
    when nothing qualifies. Safe to call more than once: an already-NZ weight is
    left alone.
    """
    if not _weight_nz_env_enabled():
        return 0

    # Imported here rather than at module scope: this hook runs once per DiT at
    # load time, and the offload package pulls in the model loader.
    from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload_strategy
    from vllm_omni.diffusion.offloader.tensor_utils import is_dtensor

    if resolve_offload_strategy(od_config) is not OffloadStrategy.NONE:
        logger.info(
            "MiniMax-H3 DiT weight NZ skipped (%s=%s): an offload backend owns the parameters",
            WEIGHT_NZ_ENV,
            os.environ.get(WEIGHT_NZ_ENV),
        )
        return 0

    import torch_npu

    cast = 0
    skipped = 0
    for name, module in model.named_modules():
        if not _is_eligible_linear(module):
            continue
        weight = getattr(module, "weight", None)
        # DTensor shards (HSDP/FSDP2) are not plain 2-D parameters, and a 1-D
        # flat shard has no meaningful NZ layout.
        if weight is None or weight.dim() != 2 or is_dtensor(weight):
            skipped += 1
            continue
        if weight.dtype not in _NZ_DTYPES:
            # The fp32 residents (patch projections, time embedder, final layer)
            # stay ND; NZ does not support fp32.
            skipped += 1
            continue
        if not _is_npu_weight(weight):
            skipped += 1
            continue
        if _npu_format(weight) == _ACL_FORMAT_FRACTAL_NZ:
            skipped += 1
            continue
        try:
            module.weight.data = torch_npu.npu_format_cast(weight.data.contiguous(), _ACL_FORMAT_FRACTAL_NZ)
        except Exception as exc:
            logger.warning("MiniMax-H3 DiT weight NZ: could not transpose %s, keeping ND: %s", name, exc)
            skipped += 1
            continue
        if _npu_format(module.weight) != _ACL_FORMAT_FRACTAL_NZ:
            # ``allow_internal_format`` is off, so the cast was silently dropped.
            logger.warning(
                "MiniMax-H3 DiT weight NZ: %s stayed ND; torch.npu.config.allow_internal_format is not enabled",
                name,
            )
            skipped += 1
            continue
        cast += 1

    logger.info(
        "MiniMax-H3 DiT weight NZ installed (%s=%s): %d linear weights now FRACTAL_NZ, %d left ND",
        WEIGHT_NZ_ENV,
        os.environ.get(WEIGHT_NZ_ENV),
        cast,
        skipped,
    )
    return cast


__all__ = [
    "WEIGHT_NZ_ENV",
    "install_minimax_h3_dit_weight_nz",
]
