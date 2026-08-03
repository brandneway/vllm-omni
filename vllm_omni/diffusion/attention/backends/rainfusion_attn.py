# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

logger = init_logger(__name__)

# The rf_v2 kernel only implements a 128-token block.
_BLOCK_SIZE = 128

# Below this many video blocks, the pooling and gather that block selection adds
# cost more than the QK work it removes, so stay dense.
_MIN_VIDEO_BLOCKS = 32

# vLLM-Omni diffusion attention always hands the impl [B, S, N, D].
_INPUT_LAYOUT = "BSND"

# Packed video geometry the model must publish in AttentionMetadata.extra. These
# are plain ints so resolving a plan never forces a device-to-host sync.
#   "rainfusion_prefix_len": rows before the video segment (text + cond + audio),
#     which rf_v2 keeps dense.
#   "rainfusion_latent_grid": (t, h, w) of the video segment, used to restore
#     spatiotemporal locality before block selection.
#   "max_seqlen_q": length of packed document 0.
#
# Publishing these keys asserts that the sequence is laid out as
# [prefix | t*h*w video rows | right padding] and that any attn_mask only masks
# that trailing padding, which this backend reproduces by slicing to
# prefix_len + t*h*w. A model with a richer mask must not publish them.
_REQUIRED_EXTRA = ("max_seqlen_q", "rainfusion_prefix_len", "rainfusion_latent_grid")

_WRONG_PLATFORM = (
    "RAINFUSION_ATTN runs the MindIE-SD rf_v2 kernel and is available on Ascend NPU only. "
    "Select FLASH_ATTN or TORCH_SDPA on this platform."
)


def _prefix_pad_rows(video_len: int, prefix_len: int) -> int:
    """Rows to prepend so rf_v2 treats a whole video segment plus a whole prefix.

    rf_v2 reorders the sequence to [video | prefix] and lets the kernel tile it
    into ceil((video + prefix) / 128) blocks, but it builds the mask from the
    two segments pooled apart, i.e. ceil(video / 128) + ceil(prefix / 128).
    Write video = 128a + r and prefix = 128b + s. Two things go wrong once
    r != 0, and padding the prefix has to answer both:

    1. The counts disagree when s is also nonzero and r + s <= 128. The surplus
       block is a prefix column, which is force-selected for every query, so it
       indexes past the end of the sequence and corrupts every row.
    2. Kernel block a straddles the seam: it holds the r leftover video rows
       *and* the 128 - r rows that follow them. Pooling assigns that block to
       the video, so block selection may drop it -- while every block pooled as
       prefix is forced on. Any real prefix row landing there loses the dense
       treatment the prefix is supposed to get, and it is the *front* of the
       prefix that lands there, which for MiniMax-H3 is the text embedding.

    Padding at least 128 - r rows fills the seam block with pad instead, so the
    only rows that can be dropped are zeros. The loop then walks up to the first
    pad that also satisfies (1). Both conditions together cost under 255 rows.

    Padding the video instead would fix the grouping outright, but latent_shape
    must satisfy t*h*w == video length, so the pad would have to be a whole
    frame or column plane -- thousands of rows, and measurably worse, because
    padded keys cannot be masked off and so take a share of the softmax mass
    (see _forward_sparse_npu).
    """
    video_rem = video_len % _BLOCK_SIZE
    if video_rem == 0:
        # The video already ends on a block boundary, so the pooled blocks and
        # the kernel's tiles are the same rows and the prefix starts a tile.
        return 0
    min_pad = _BLOCK_SIZE - video_rem
    for pad in range(min_pad, min_pad + _BLOCK_SIZE):
        prefix_rem = (prefix_len + pad) % _BLOCK_SIZE
        if prefix_rem == 0 or video_rem + prefix_rem > _BLOCK_SIZE:
            return pad
    raise AssertionError(  # unreachable: pad sweeps every residue mod 128
        f"no prefix pad reconciles video_len={video_len} prefix_len={prefix_len}"
    )


def _try_extract_layer_index(prefix: str) -> int | None:
    if not prefix:
        return None
    try:
        return extract_layer_index(prefix)
    except (AssertionError, ValueError):
        return None


@dataclass(frozen=True)
class RainFusionConfig:
    """Resolved RainFusion controls for one attention layer.

    ``sparsity`` is the nominal fraction of key blocks dropped per query block.
    The realized sparsity is lower because rf_v2 always keeps the prefix rows and
    the first-frame blocks. ``start_step`` and ``skip_layers`` are the accuracy
    knobs: early denoise steps and specific DiT blocks stay dense.
    """

    sparsity: float = 0.0
    start_step: int = 0
    inner_precise: int = 0
    skip_layers: frozenset[int] = frozenset()

    @classmethod
    def from_backend_kwargs(cls, backend_kwargs: dict | None) -> RainFusionConfig:
        bk = backend_kwargs or {}
        return cls(
            sparsity=float(bk.get("sparsity", 0.0)),
            start_step=int(bk.get("start_step", 0)),
            inner_precise=int(bk.get("inner_precise", 0)),
            skip_layers=frozenset(bk.get("skip_layers") or ()),
        )

    @property
    def enabled(self) -> bool:
        return self.sparsity > 0.0


@dataclass(frozen=True)
class RainFusionPlan:
    """Per-forward geometry handed to the rf_v2 kernel."""

    prefix_len: int
    used_len: int
    latent_shape: list[int]
    prefix_pad: int = 0


class RainFusionAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 96, 128, 192, 256]

    @staticmethod
    def get_name() -> str:
        return "RAINFUSION_ATTN"

    @staticmethod
    def get_impl_cls() -> type[RainFusionAttentionImpl]:
        return RainFusionAttentionImpl


class RainFusionAttentionImpl(AttentionImpl):
    """Block-sparse video attention via MindIE-SD RainFusion (rf_v2) on Ascend NPU.

    Sparsity applies only to the video segment of a packed multimodal sequence.
    Every other case — warmup denoise steps, exempt layers, sequences without
    published video geometry, video segments too short to pay for block
    selection — delegates to FlashAttention, so a model can select this backend
    unconditionally.

    Resolution is unconstrained. Geometries that would desynchronize rf_v2's
    block mask from the kernel's tiling are brought back into step by padding
    the prefix; see ``_prefix_pad_rows``.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.qkv_layout = qkv_layout

        self.rainfusion = RainFusionConfig.from_backend_kwargs(backend_kwargs)
        self.layer_idx = _try_extract_layer_index(prefix)

        if self.rainfusion.enabled:
            self._validate_parallel_config()
            if causal:
                raise ValueError(
                    "RAINFUSION_ATTN does not support causal attention: rf_v2 selects key "
                    "blocks by pooled relevance and cannot express a causal mask. Select "
                    "FLASH_ATTN for causal roles."
                )

        self.dense_fallback = FlashAttentionBackend.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            qkv_layout=qkv_layout,
        )

    def _validate_parallel_config(self) -> None:
        config = get_current_diffusion_config_or_none()
        parallel_config = getattr(config, "parallel_config", None)
        ring_degree = getattr(parallel_config, "ring_degree", 1)
        if ring_degree > 1:
            # Ring gives each rank a slice of the sequence, so block selection
            # would score only local keys and the layer bypasses the backend
            # entirely (see Attention._run_ring_attention).
            raise ValueError(
                "RAINFUSION_ATTN is not compatible with ring sequence parallelism "
                f"(ring_degree={ring_degree}): rf_v2 needs the whole key sequence to rank "
                "blocks. Use Ulysses SP (ring_degree=1) instead."
            )

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        # ROCm and MUSA route through forward_cuda by default, so this covers them too.
        raise NotImplementedError(_WRONG_PLATFORM)

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(_WRONG_PLATFORM)

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        plan = self._resolve_plan(attn_metadata)
        if plan is None:
            return self.dense_fallback.forward_npu(query, key, value, attn_metadata)
        return self._forward_sparse_npu(query, key, value, plan)

    def _resolve_plan(self, attn_metadata: AttentionMetadata | None) -> RainFusionPlan | None:
        """Return the rf_v2 geometry, or None when this forward must stay dense."""
        rf = self.rainfusion
        if not rf.enabled:
            return None
        if self.layer_idx is not None and self.layer_idx in rf.skip_layers:
            return None
        if is_forward_context_available():
            step_idx = get_forward_context().denoise_step_idx
            if step_idx is not None and step_idx < rf.start_step:
                return None

        if attn_metadata is None:
            return None

        extra = attn_metadata.extra
        missing = [key for key in _REQUIRED_EXTRA if key not in extra]
        if missing:
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: attention metadata is missing %s. The model must "
                "publish the packed video geometry (see _REQUIRED_EXTRA in rainfusion_attn.py).",
                # warning_once memoizes on the args, so they must be hashable.
                ", ".join(missing),
            )
            return None

        prefix_len = int(extra["rainfusion_prefix_len"])
        latent_shape = [int(dim) for dim in extra["rainfusion_latent_grid"]]
        # rf_v2 splits the sequence as [prefix | t*h*w video rows]. Document 0 of
        # the packed sequence holds those rows; anything past it is alignment
        # padding that rf_v2 must not see.
        video_len = math.prod(latent_shape)
        used_len = prefix_len + video_len

        if used_len != int(extra["max_seqlen_q"]):
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: prefix (%d) plus latent grid %s does not fill "
                "packed document 0 (%d rows). rf_v2 requires the video segment to be its tail.",
                prefix_len,
                tuple(latent_shape),
                int(extra["max_seqlen_q"]),
            )
            return None
        if video_len < _MIN_VIDEO_BLOCKS * _BLOCK_SIZE:
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: %d video rows is under the %d-row "
                "(%d block) threshold where sparse selection pays off.",
                video_len,
                _MIN_VIDEO_BLOCKS * _BLOCK_SIZE,
                _MIN_VIDEO_BLOCKS,
            )
            return None

        prefix_pad = _prefix_pad_rows(video_len, prefix_len)
        logger.info_once(
            "RAINFUSION_ATTN active: sparsity=%.2f, start_step=%d, exempt_layers=%d, "
            "latent_grid=%s, prefix_rows=%d, video_rows=%d, prefix_pad=%d. Realized sparsity is "
            "lower than nominal because prefix and first-frame blocks are always kept.",
            rf.sparsity,
            rf.start_step,
            len(rf.skip_layers),
            tuple(latent_shape),
            prefix_len,
            video_len,
            prefix_pad,
        )
        return RainFusionPlan(
            prefix_len=prefix_len,
            used_len=used_len,
            latent_shape=latent_shape,
            prefix_pad=prefix_pad,
        )

    def _forward_sparse_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        plan: RainFusionPlan,
    ) -> torch.Tensor:
        try:
            from mindiesd import sparse_attention
        except ImportError:
            raise ImportError(
                "RAINFUSION_ATTN requires MindIE-SD. Please install MindIE-SD to enable "
                "RainFusion sparse attention on Ascend NPU. For installation details, see "
                "https://gitcode.com/Ascend/MindIE-SD "
                "Otherwise, use FlashAttention by setting DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN"
            )

        used = plan.used_len
        pad = plan.prefix_pad
        q, k, v = (tensor[:, :used] for tensor in (query, key, value))
        if pad:
            # The pad goes in front of the prefix so the video rows stay
            # contiguous and keep their (t, h, w) order. rf_v2 ignores attn_mask
            # and reads actual_seq_lengths off the tensor, so these rows are
            # attended: zeroing the values keeps them out of the numerator, and
            # at a couple hundred rows against tens of thousands the share of
            # the softmax denominator they take stays negligible. Measured on
            # Ascend 910 at sparsity=0, where the mask is fully populated and
            # the kernel must reproduce dense attention: padded grids land at
            # 0.06-0.16% mean relative error against 0.06% for grids that need
            # no pad, and 4-5% for the unpadded mismatch.
            head = query.new_zeros((query.shape[0], pad, *query.shape[2:]))
            q, k, v = (torch.cat((head, tensor), dim=1) for tensor in (q, k, v))
        # Ulysses has already gathered the full sequence onto this rank and split
        # the heads, so read the head count off the tensor rather than num_heads.
        out = sparse_attention(
            q,
            k,
            v,
            scale=self.softmax_scale,
            head_num=query.shape[-2],
            input_layout=_INPUT_LAYOUT,
            inner_precise=self.rainfusion.inner_precise,
            sparse_type="rf_v2",
            txt_len=plan.prefix_len + pad,
            block_size=_BLOCK_SIZE,
            latent_shape_q=plan.latent_shape,
            latent_shape_k=plan.latent_shape,
            sparsity=self.rainfusion.sparsity,
        )
        if pad:
            out = out[:, pad:].contiguous()
        if used == query.shape[1]:
            return out
        padded = torch.zeros_like(query)
        padded[:, :used] = out
        return padded
