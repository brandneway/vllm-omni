# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drop-in fast paths for the MiniMax-H3 video VAE's checkpoint processor.

The video VAE ships as trust_remote_code inside the checkpoint directory, so
its pixel staging code stays outside this repository.  Two of its hot paths
are host-side glue that the CPU runs serially between device stages:

* ``convert_numpy_to_tensor`` upcasts every frame to FP32, permutes and
  scales by 1/255 on the host, and only then copies to the device -- a
  four-times-heavier H2D than the raw uint8 bytes need.
* ``revert_tensor`` clamps out-of-place right after decode, allocating one
  more decoded-video-sized tensor.

``apply_vae_processor_fastpath`` binds replacements on the loaded processor
instance: uint8 frames cross to the device first and normalize there, and
the clamp reuses the denormalize output instead of allocating again.  Both
keep the numeric contract of the checkpoint implementation (uint8->FP32 is
exact and the FP32 division rounds identically on either side of the copy)
and fall back to the checkpoint functions for anything but the plain
contiguous-uint8 case, so a future processor change degrades to today's
behavior instead of breaking.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _revert_tensor_fastpath(self: Any, tensor: torch.Tensor) -> torch.Tensor:
    """``revert_tensor`` with the clamp folded into the denormalize output.

    Mirrors the checkpoint implementation, including the ``(b t) c h w``
    fold around the per-frame transform, but clamps in place.  torchvision's
    ``Normalize`` clones before its in-place arithmetic, so the denormalize
    result is always a fresh tensor; the identity check keeps the function
    safe even for a transform that hands the input straight back.
    """
    batch = frames = None
    if self.use_3d_conv:
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(2)
        batch, _, frames, _, _ = tensor.shape
        tensor = tensor.permute(0, 2, 1, 3, 4).flatten(0, 1)
    tensor_rev = self.transform_rev(tensor)
    if tensor_rev is not tensor:
        tensor_rev.clamp_(0, 1)
    else:
        tensor_rev = tensor_rev.clamp(0, 1)
    if batch is not None:
        tensor_rev = tensor_rev.reshape(batch, frames, *tensor_rev.shape[1:]).permute(0, 2, 1, 3, 4)
    return tensor_rev.contiguous()


def apply_vae_processor_fastpath(processor: Any) -> bool:
    """Bind the fast paths onto a loaded video VAE processor instance.

    Returns ``True`` when the instance was patched.  The checkpoint keeps
    working unpatched when its processor does not match the expected shape,
    which is what happens if a future checkpoint revision renames or drops
    these members.
    """
    if processor is None:
        # A remote component without a processor (mocks, non-VAE remotes)
        # simply stays unpatched.
        return False
    cls = type(processor)
    if not all(hasattr(processor, name) for name in ("use_3d_conv", "transform_rev")):
        logger.warning_once(
            "MiniMax-H3 video VAE processor lacks the expected members; "
            "skipping the device-side pixel staging fast paths."
        )
        return False
    original_convert = getattr(cls, "convert_numpy_to_tensor", None)
    if original_convert is None or getattr(cls, "revert_tensor", None) is None:
        logger.warning_once(
            "MiniMax-H3 video VAE processor lacks convert_numpy_to_tensor/revert_tensor; "
            "skipping the device-side pixel staging fast paths."
        )
        return False

    def convert_numpy_to_tensor(numpy_array: Any, device: Any = None) -> torch.Tensor:
        frames = np.stack(numpy_array, axis=0) if isinstance(numpy_array, list) else numpy_array
        if (
            device is None
            or frames.dtype != np.uint8
            or frames.ndim != 4
            or not frames.flags.c_contiguous
            or not frames.flags.writeable
        ):
            return original_convert(numpy_array, device=device)
        # DMA the raw uint8 frames (a quarter of the FP32 payload) and do the
        # cast, permute and 1/255 scaling on the accelerator.  The output
        # layout matches the checkpoint path: contiguous NCHW FP32.
        tensor = torch.from_numpy(frames).to(device=device)
        return tensor.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32).div_(255.0)

    processor.convert_numpy_to_tensor = convert_numpy_to_tensor
    processor.revert_tensor = types.MethodType(_revert_tensor_fastpath, processor)
    return True
