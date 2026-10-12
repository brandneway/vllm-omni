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
for both bf16 and int8 on Ascend910).

Two regimes, and they behave very differently:

* **Quantized (int8)** -- uniformly better. The four DiT linears measured
  0.834 / 0.865 / 0.826 / 0.861 against the real ``npu_quant_matmul`` call, all
  bit-identical, with no shape regressing.
* **Dense bf16** -- shape dependent, and it nets out flat: 0.764x on out_proj
  and 0.902x on qkv, but 1.023x on fc1 and 1.144x on fc2.

The switch is opt-in and off by default.

Only the DiT is touched: the text encoder and the VAEs keep their ND weights.

Two prerequisites and one hazard are handled here:

* ``torch.npu.config.allow_internal_format`` must be ``True``. ``NPUPlatform``
  sets it for every NPU deployment (``platforms/npu/platform.py``); with it
  ``False``, ``npu_format_cast`` silently returns an ND tensor, so the cast is
  verified afterwards rather than assumed.
* The load-time sweep only takes unquantized linears: a quantized layer's
  method owns its weight layout, and mxfp4 casts its own weights to NZ. The
  offload path has no such guard and instead keys off the dtype set, which is
  why int8 reaches it there and not here.
* Offloaded DiTs cannot be cast here. Every offload backend relocates the
  parameters after this hook -- the layerwise ones flatten them to pinned host
  memory, the sequential one moves whole modules -- and an internal-format
  tensor crossing that boundary transdata's back to ND, while the storage
  descriptors only record logical strides and so would not notice. The one
  layer that can still work is the layerwise upload path, which re-points the
  parameters at a freshly filled device buffer every block: it calls
  ``maybe_cast_weight_to_nz`` there instead, at the price of one ND->NZ
  conversion per weight per upload. ``weight_nz_enabled`` is the shared
  opt-in, and ``nz_cast_takes_effect`` lets that path fail loudly rather than
  silently measure an unchanged baseline.
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

# FRACTAL_NZ is a fractal layout (16x16 blocks; 16 elements per block row for
# bf16/fp16, 32 for int8). fp32 has no NZ matmul path at all.
#
# int8 is included because that is where the win actually is: measured on
# Ascend910 with the real npu_quant_matmul call (dynamic-quant activations,
# per-channel weight scale, bf16 output), the four DiT linears come out
# bit-identical and 13-17% faster on an NZ weight -- 0.834 / 0.865 / 0.826 /
# 0.861 -- uniformly, with no shape regressing. The bf16 dense path, by
# contrast, swings between 0.76x and 1.14x and nets out flat.
#
# fp8 is deliberately NOT here yet: its online methods pack their own layout
# (mxfp4 casts its weights itself), and no fp8 kernel has been checked against
# an NZ weight on this stack.
_NZ_DTYPES = (torch.bfloat16, torch.float16, torch.int8)


def _weight_nz_env_enabled() -> bool:
    return os.environ.get(WEIGHT_NZ_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def weight_nz_enabled() -> bool:
    """Public form of the opt-in, for callers outside this module."""
    return _weight_nz_env_enabled()


def maybe_cast_weight_to_nz(weight: torch.Tensor) -> torch.Tensor | None:
    """Cast one weight to FRACTAL_NZ, or return ``None`` when it should stay ND.

    The eligibility filter is deliberately shape/dtype-only: the layerwise
    offload path re-points parameters from a flattened device buffer and has no
    module handle, so it cannot consult ``quant_method``. The dtype set carries
    that decision instead -- see ``_NZ_DTYPES`` -- and the 1-D bias/scale
    tensors that travel in the same buffer fall out on the rank check.

    Quantized weights need no special handling here: ``process_weights_after_loading``
    has already put them in the layout the kernel wants (int8, ``(K, N)``), so
    the NZ cast applies to that tensor exactly as it does to a dense one.

    No readback check here: this runs once per weight per offload upload, and
    the caller verifies the cast takes effect once at setup instead.
    """
    if weight.dim() != 2 or weight.dtype not in _NZ_DTYPES or not _is_npu_weight(weight):
        return None
    import torch_npu

    try:
        # Allocate the NZ destination and copy into it, rather than
        # ``npu_format_cast(weight.contiguous(), 29)``. The cast silently
        # returns an ND tensor when the input does not own its storage, which
        # is exactly the layerwise offload case (the weight is an as_strided
        # view over a shared device buffer), and ``.contiguous()`` does not
        # help: an already-contiguous view is returned unchanged and is still
        # non-owning. Measured on Ascend910 -- the cast then costs a full
        # copy + allocation and produces ND, so the matmul sees no NZ at all.
        out = torch_npu.empty_with_format(
            tuple(weight.shape),
            dtype=weight.dtype,
            device=weight.device,
            acl_format=_ACL_FORMAT_FRACTAL_NZ,
        )
        out.copy_(weight)
        return out
    except Exception as exc:
        logger.warning("MiniMax-H3 weight NZ: could not transpose %s, keeping ND: %s", tuple(weight.shape), exc)
        return None


def nz_cast_takes_effect() -> bool:
    """Probe once whether ``npu_format_cast`` actually produces NZ on this box.

    With ``torch.npu.config.allow_internal_format`` off the cast silently
    returns an ND tensor. Every caller of the offload-path cast checks this at
    setup so a no-op cannot be mistaken for "NZ measured, no gain".
    """
    import torch_npu

    try:
        probe = torch.zeros(16, 16, dtype=torch.bfloat16, device="npu")
        return _npu_format(torch_npu.npu_format_cast(probe, _ACL_FORMAT_FRACTAL_NZ)) == _ACL_FORMAT_FRACTAL_NZ
    except Exception:
        return False


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
            "MiniMax-H3 DiT weight NZ deferred to the offload path (%s=%s): "
            "an offload backend owns the parameters and re-casts at upload time",
            WEIGHT_NZ_ENV,
            os.environ.get(WEIGHT_NZ_ENV),
        )
        return 0

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
        cast_weight = maybe_cast_weight_to_nz(weight.data)
        if cast_weight is None:
            skipped += 1
            continue
        module.weight.data = cast_weight
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
    "maybe_cast_weight_to_nz",
    "nz_cast_takes_effect",
    "weight_nz_enabled",
]
