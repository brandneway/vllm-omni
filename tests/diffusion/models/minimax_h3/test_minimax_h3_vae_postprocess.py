# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _DummyVideoVAEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.processor = SimpleNamespace(transform_rev=self._channelwise_transform)

    @staticmethod
    def _channelwise_transform(tensor: torch.Tensor) -> torch.Tensor:
        leading_dims = (1,) * (tensor.ndim - 3)
        scale = tensor.new_tensor([0.25, 0.5, 0.75]).view(*leading_dims, 3, 1, 1)
        bias = tensor.new_tensor([0.1, 0.2, 0.3]).view(*leading_dims, 3, 1, 1)
        return tensor * scale + bias


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (1, 3, 2, 4, 5)])
def test_inplace_revert_matches_reference(shape):
    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

    vae = object.__new__(MiniMaxH3VideoVAE)
    nn.Module.__init__(vae)
    vae.model = _DummyVideoVAEModel()

    decoded = torch.linspace(-2, 2, int(np.prod(shape)), dtype=torch.float32).reshape(shape)
    if decoded.ndim == 5:
        batch, channels, frames, height, width = decoded.shape
        flattened = decoded.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        expected = vae.model.processor.transform_rev(flattened).clamp(0, 1)
        expected = expected.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
    else:
        expected = vae.model.processor.transform_rev(decoded).clamp(0, 1)

    torch.testing.assert_close(vae._inplace_revert(decoded), expected)
