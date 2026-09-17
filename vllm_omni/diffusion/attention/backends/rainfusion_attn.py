# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import functools
import inspect
import math
import os
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

# rf_v2's ``input_layout`` describes the caller's tensors, and everything below
# slices the sequence on dim 1. vLLM-Omni diffusion attention hands the impl
# [B, S, N, D], so that is the only layout this backend accepts.
_INPUT_LAYOUT = "BSND"

# rf_v2 precision mode: 0 = high precision, 1 = high performance. Kept at the
# precise setting because the sparsity knob is the intended perf lever, and on
# A5 devices the kernel is routed to rf_v3, which overrides this anyway.
_INNER_PRECISE = 0

# --- estimate-mask (ada_bsa stage 1) experiment --------------------------------
#
# VLLM_OMNI_RAINFUSION_MASK_ALGO=estimate swaps the mask algorithm only: the
# RainFusion pooling mask (spatial rearrange + avgpool scoring + inverse
# rearrange) is replaced by the A5-native sparse_block_estimate operator,
# while the execution kernel stays eagle_quant_block_sparse_attention with the
# same EagleQBSA quantization. This mode performs no spatial rearrangement at
# all — sparse_block_estimate scores 128-token blocks of the packed sequence
# in place, so single-span and multi-span plans share one code path.
_ESTIMATE_MASK_ALGO = "estimate"

# sparse_block_estimate A5 kernel contract: Q/K downsampled by this stride.
_ESTIMATE_STRIDE = 8

# A5 rf_v3 convention for the EagleQBSA kernel (950 requires inner_precise=4).
_ESTIMATE_INNER_PRECISE = 4

# The A5 kernel accepts at most 2048 KV128 blocks (256K tokens).
_ESTIMATE_MAX_KV_TOKENS = 2048 * _BLOCK_SIZE

# The A5 kernel only implements head_dim 128.
_ESTIMATE_HEAD_DIM = 128

_WRONG_PLATFORM = (
    "RAINFUSION_ATTN runs the MindIE-SD rf_v2 kernel and is available on Ascend NPU only. "
    "Select FLASH_ATTN or TORCH_SDPA on this platform."
)

_MISSING_MINDIESD = (
    "RAINFUSION_ATTN requires MindIE-SD. Please install MindIE-SD to enable RainFusion sparse "
    "attention on Ascend NPU. For installation details, see https://gitcode.com/Ascend/MindIE-SD "
    "Otherwise, use FlashAttention by setting DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN"
)

_INCOMPATIBLE_MINDIESD = (
    "RAINFUSION_ATTN requires a MindIE-SD build whose sparse_attention supports the video_spans "
    "argument. Please upgrade MindIE-SD or select FLASH_ATTN."
)


# Whether the installed mindiesd ``sparse_attention`` accepts ``precision=``.
# Releases without it accept the kwarg through ``**kwargs`` but silently ignore
# it, so a requested mix/fp8 mode would silently run the BF16 path. Cached
# because forwards run per layer per denoise step.
@functools.cache
def _mindiesd_supports_precision() -> bool:
    try:
        from inspect import signature

        from mindiesd import sparse_attention

        return "precision" in signature(sparse_attention).parameters
    except Exception:
        return False


def _try_extract_layer_index(prefix: str) -> int | None:
    if not prefix:
        return None
    try:
        return extract_layer_index(prefix)
    except (AssertionError, ValueError):
        return None


def _supports_video_spans(sparse_attention: Any) -> bool:
    try:
        return "video_spans" in inspect.signature(sparse_attention).parameters
    except (TypeError, ValueError):
        return False


@functools.cache
def _estimate_mask_requested() -> bool:
    return os.environ.get("VLLM_OMNI_RAINFUSION_MASK_ALGO", "").lower() == _ESTIMATE_MASK_ALGO


@functools.cache
def _estimate_cdf_threshold() -> float:
    try:
        return float(os.environ.get("VLLM_OMNI_RAINFUSION_ESTIMATE_CDF", "1.0"))
    except ValueError:
        return 1.0


@functools.cache
def _mindiesd_has_estimate_mask() -> bool:
    try:
        from mindiesd.layers.flash_attn.sparse_flash_attn_ada_bsa import get_estimate_mask  # noqa: F401

        return True
    except Exception:
        return False


@functools.cache
def _estimate_protections_enabled() -> bool:
    """Replicate the rf_v2 mask contract (dense context + first frame) on the
    estimate mask. On by default because prompt adherence relies on text key
    blocks being unconditionally visible to every query row; set
    VLLM_OMNI_RAINFUSION_ESTIMATE_PROTECT=0 to A/B the raw estimate mask."""
    return os.environ.get("VLLM_OMNI_RAINFUSION_ESTIMATE_PROTECT", "1") != "0"


def _estimate_protected_token_ranges(plan: RainFusionPlan) -> list[tuple[int, int]]:
    """Token ranges the rf_v2 mask contract keeps dense, in packed order.

    Mirrors get_blockwise_mask / get_multi_span_blockwise_mask: non-video
    context rows and columns are always kept, and so is each clip's first
    frame (t*h*w latent grid, first h*w rows).
    """
    if plan.video_spans is not None:
        ranges: list[tuple[int, int]] = []
        prev_end = 0
        for span in sorted(plan.video_spans, key=lambda item: int(item["start"])):
            start = int(span["start"])
            t, h, w = (int(dim) for dim in span["latent_shape"])
            if start > prev_end:
                ranges.append((prev_end, start))
            ranges.append((start, start + h * w))
            prev_end = start + t * h * w
        if plan.used_len > prev_end:
            ranges.append((prev_end, plan.used_len))
        return ranges
    assert plan.prefix_len is not None and plan.latent_shape is not None
    _, h, w = (int(dim) for dim in plan.latent_shape)
    return [(0, int(plan.prefix_len) + h * w)]


def _apply_estimate_protections(
    mask: torch.Tensor, token_ranges: list[tuple[int, int]]
) -> torch.Tensor:
    """Force the protected block rows/columns to 1 on the trimmed estimate mask.

    Token ranges are widened to whole 128-token blocks (a range straddling a
    block boundary protects the whole block), which slightly over-protects —
    the same trade the rf_v2 boundary-dense layout makes by construction.
    """
    n_blocks = mask.shape[-1]
    block_ranges = sorted(
        (start // _BLOCK_SIZE, -(-end // _BLOCK_SIZE))
        for start, end in token_ranges
        if end > start
    )
    merged: list[list[int]] = []
    for lo, hi in block_ranges:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    for lo, hi in merged:
        lo, hi = max(lo, 0), min(hi, n_blocks)
        if hi > lo:
            # Dense queries: these rows attend to every KV block.
            mask[:, :, lo:hi, :] = 1
            # Visible keys: every query row attends to these KV blocks.
            mask[..., lo:hi] = 1
    return mask


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
    end_step: int = 0
    precision: str = "bf16"
    skip_layers: frozenset[int] = frozenset()

    @classmethod
    def from_backend_kwargs(cls, backend_kwargs: dict | None) -> RainFusionConfig:
        bk = backend_kwargs or {}
        return cls(
            sparsity=float(bk.get("sparsity", 0.0)),
            start_step=int(bk.get("start_step", 0)),
            end_step=int(bk.get("end_step", 0)),
            precision=str(bk.get("precision", "bf16")),
            skip_layers=frozenset(bk.get("skip_layers") or ()),
        )

    @property
    def enabled(self) -> bool:
        return self.sparsity > 0.0


@dataclass(frozen=True)
class RainFusionPlan:
    """Per-forward geometry handed to the rf_v2 kernel."""

    used_len: int
    prefix_len: int | None = None
    latent_shape: list[int] | None = None
    video_spans: list[dict[str, object]] | None = None


class RainFusionAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_platforms: tuple[str, ...] = ("npu",)

    @classmethod
    def validate_available(cls) -> None:
        from importlib.util import find_spec

        if find_spec("mindiesd") is None:
            raise ValueError(_MISSING_MINDIESD)
        try:
            from mindiesd import sparse_attention
        except ImportError as exc:
            raise ValueError(_MISSING_MINDIESD) from exc
        if not _supports_video_spans(sparse_attention):
            raise ValueError(_INCOMPATIBLE_MINDIESD)

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

    Sparsity applies only to the video segment of a packed multimodal sequence,
    whose extent the model publishes as ``AttentionMetadata.video_layout``. Every
    other case — warmup denoise steps, exempt layers, a layer that does not declare
    ``qkv_layout="BSND"``, sequences without a published video segment, video
    segments too short to pay for block selection — delegates to FlashAttention,
    so a model can select this backend unconditionally. MindIE-SD handles an
    irregular video tail internally, retaining it outside the sparse blocks.
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
            if qkv_layout is not None and qkv_layout.upper() != _INPUT_LAYOUT:
                raise ValueError(
                    f"RAINFUSION_ATTN needs {_INPUT_LAYOUT} tensors to locate the video segment along "
                    f"the sequence axis, but this layer declares qkv_layout={qkv_layout!r}. Select "
                    "FLASH_ATTN for this role."
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
            total_steps = get_forward_context().total_denoise_steps
            if step_idx is not None and step_idx < rf.start_step:
                return None
            # Tail fallback: keep the last ``end_step`` denoise steps dense.
            if (
                rf.end_step > 0
                and step_idx is not None
                and total_steps is not None
                and step_idx >= total_steps - rf.end_step
            ):
                return None
        if self.qkv_layout is None:
            # The sparse path reads the sequence off dim 1, which the tensors alone
            # do not establish, and the dense fallback resolves an absent layout its
            # own way. Sparsifying on an assumption would put the two paths on
            # different axes, so an undeclared layout stays dense.
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: this layer does not declare qkv_layout, and rf_v2 "
                "needs %s to locate the video segment along the sequence axis. Set qkv_layout=%r on "
                "the Attention layer to enable sparsity.",
                _INPUT_LAYOUT,
                _INPUT_LAYOUT,
            )
            return None

        if attn_metadata is None:
            return None

        layout = attn_metadata.video_layout
        if layout is None:
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: this attention role carries no video segment. The "
                "model must publish AttentionMetadata.video_layout for the sequence to be sparsified."
            )
            return None
        max_seqlen_q = attn_metadata.extra.get("max_seqlen_q")
        if max_seqlen_q is None:
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: attention metadata is missing max_seqlen_q, so the "
                "video segment cannot be confirmed to be the tail of packed document 0."
            )
            return None

        if layout.video_spans:
            return self._resolve_multi_span_plan(layout, max_seqlen_q)

        if layout.prefix_len is None or layout.latent_grid is None:
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: video layout has neither a legacy video tail nor multi-video spans."
            )
            return None
        prefix_len = int(layout.prefix_len)
        latent_shape = [int(dim) for dim in layout.latent_grid]
        # rf_v2 splits the sequence as [prefix | t*h*w video rows]. Document 0 of
        # the packed sequence holds those rows; anything past it is alignment
        # padding that rf_v2 must not see.
        video_len = math.prod(latent_shape)
        used_len = prefix_len + video_len

        if used_len != int(max_seqlen_q):
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: prefix (%d) plus latent grid %s does not fill "
                "packed document 0 (%d rows). rf_v2 requires the video segment to be its tail.",
                prefix_len,
                tuple(latent_shape),
                int(max_seqlen_q),
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
        logger.info_once(
            "RAINFUSION_ATTN active: sparsity=%.2f, start_step=%d, exempt_layers=%d, "
            "latent_grid=%s, prefix_rows=%d, video_rows=%d. Realized sparsity is lower than nominal "
            "because prefix and first-frame blocks are always kept.",
            rf.sparsity,
            rf.start_step,
            len(rf.skip_layers),
            tuple(latent_shape),
            prefix_len,
            video_len,
        )
        return RainFusionPlan(
            used_len=used_len,
            prefix_len=prefix_len,
            latent_shape=latent_shape,
        )

    def _resolve_multi_span_plan(self, layout, max_seqlen_q: int) -> RainFusionPlan | None:
        """Validate Ref2VA's non-contiguous video grids before sparse dispatch."""
        used_len = layout.used_len
        if used_len is None or int(used_len) != int(max_seqlen_q):
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: multi-video layout used_len=%r does not match packed document 0 (%d).",
                used_len,
                int(max_seqlen_q),
            )
            return None

        spans: list[dict[str, object]] = []
        span_summaries: list[str] = []
        previous_end = 0
        target_count = 0
        video_seqlen = 0
        boundary_dense_seqlen = 0
        previous_video_length: int | None = None
        for span in sorted(layout.video_spans, key=lambda item: item.start):
            grid = tuple(int(dim) for dim in span.latent_grid)
            length = math.prod(grid)
            start = int(span.start)
            if (
                len(grid) != 3
                or any(dim <= 0 for dim in grid)
                or start < previous_end
                or start + length > int(used_len)
            ):
                logger.warning_once(
                    "RAINFUSION_ATTN staying dense: invalid multi-video span start=%d grid=%s used_len=%d.",
                    start,
                    grid,
                    int(used_len),
                )
                return None
            role = span.role
            if role not in ("reference", "target"):
                logger.warning_once("RAINFUSION_ATTN staying dense: unsupported multi-video span role %r.", role)
                return None
            if role == "target":
                target_count += 1
            if previous_video_length is not None:
                # rf_v2 works on fixed 128-token blocks. The preceding clip
                # needs these dense rows to complete its tail block before
                # this clip begins, otherwise one sparse block would cross a
                # clip boundary.
                boundary_dense_seqlen += (-previous_video_length) % _BLOCK_SIZE
            spans.append({"start": start, "latent_shape": list(grid)})
            span_summaries.append(f"role={role}, start={start}, seqlen={length}, latent_shape={grid}")
            previous_end = start + length
            video_seqlen += length
            previous_video_length = length

        if target_count != 1:
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: Ref2VA layout must contain exactly one target video span, got %d.",
                target_count,
            )
            return None
        if video_seqlen < _MIN_VIDEO_BLOCKS * _BLOCK_SIZE:
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: multi-video seqlen=%d is under the sparse threshold seqlen=%d.",
                video_seqlen,
                _MIN_VIDEO_BLOCKS * _BLOCK_SIZE,
            )
            return None
        dense_context_seqlen = int(used_len) - video_seqlen
        if boundary_dense_seqlen > dense_context_seqlen:
            logger.warning_once(
                "RAINFUSION_ATTN staying dense: multi-video spans need dense_context_seqlen=%d "
                "to isolate clip block boundaries, but this layout has only %d.",
                boundary_dense_seqlen,
                dense_context_seqlen,
            )
            return None
        logger.info_once(
            "RAINFUSION_ATTN multi-video active: sparsity=%.2f, spans=[%s], "
            "video_seqlen=%d, dense_context_seqlen=%d, valid_packed_seqlen=%d.",
            self.rainfusion.sparsity,
            "; ".join(span_summaries),
            video_seqlen,
            dense_context_seqlen,
            int(used_len),
        )
        return RainFusionPlan(used_len=int(used_len), video_spans=spans)

    def _estimate_mask_eligible(
        self, query: torch.Tensor, key: torch.Tensor, plan: RainFusionPlan
    ) -> bool:
        """Gate the estimate-mask experiment on the A5 kernel's hard contract.

        sparse_block_estimate only implements BNSD fp16/bf16 with head_dim 128,
        GQA-aligned heads and at most 2048 KV128 blocks, and refuses varlen and
        causal inputs. Anything the model hands us outside that contract keeps
        the proven rf_v2 path rather than failing the forward.
        """
        if not _mindiesd_has_estimate_mask():
            logger.warning_once(
                "estimate mask algo requested but mindiesd lacks "
                "sparse_flash_attn_ada_bsa.get_estimate_mask; staying on the rf_v2 path."
            )
            return False
        if query.shape[-1] != _ESTIMATE_HEAD_DIM:
            logger.warning_once(
                "estimate mask algo staying on the rf_v2 path: head_dim=%d but the A5 "
                "sparse_block_estimate kernel only implements %d.",
                query.shape[-1],
                _ESTIMATE_HEAD_DIM,
            )
            return False
        if query.shape[-2] % key.shape[-2] != 0:
            logger.warning_once(
                "estimate mask algo staying on the rf_v2 path: q_heads=%d is not divisible "
                "by kv_heads=%d as the GQA flattening in sparse_block_estimate requires.",
                query.shape[-2],
                key.shape[-2],
            )
            return False
        if plan.used_len > _ESTIMATE_MAX_KV_TOKENS:
            logger.warning_once(
                "estimate mask algo staying on the rf_v2 path: used_len=%d exceeds the A5 "
                "sparse_block_estimate capacity of %d KV128 blocks.",
                plan.used_len,
                _ESTIMATE_MAX_KV_TOKENS,
            )
            return False
        return True

    def _forward_sparse_estimate_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        plan: RainFusionPlan,
    ) -> torch.Tensor:
        """ada_bsa feasibility path: estimate mask + EagleQBSA kernel.

        Replaces the whole RainFusion mask algorithm — no spatial rearrangement
        and no inverse rearrangement. sparse_block_estimate scores 128-token
        blocks of the packed sequence in place (stride-8 downsampled QK with
        online-softmax block scoring, sink/recent blocks always kept), the mask
        is trimmed to the eagle layout (the estimate output pads its KV-block
        columns to a multiple of 32; the eagle kernel derives its row stride
        from the sequence length instead), and the exact rf_v3 'mix' execution
        path runs unchanged: EagleQBSA quantization into
        eagle_quant_block_sparse_attention.
        """
        import torch_npu

        from mindiesd.layers.flash_attn.sparse_flash_attn_ada_bsa import get_estimate_mask
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import _eagle_qbsa_quant_qkv

        used = plan.used_len
        batch = query.shape[0]
        num_kv_heads = key.shape[-2]
        # The estimate kernel and the eagle kernel both speak BNSD; the caller's
        # BSND tensors are converted once and reused for scoring and quantization.
        q = query[:, :used].transpose(1, 2).contiguous()
        k = key[:, :used].transpose(1, 2).contiguous()
        v = value[:, :used].transpose(1, 2).contiguous()

        logger.info_once(
            "estimate mask algo active: sparsity=%.2f, cdf_threshold=%.3f, used_len=%d, "
            "q_heads=%d, kv_heads=%d, rf_contract_protections=%s. Blocks are packed-sequence "
            "128-token strips; sink and recent blocks are always kept.",
            self.rainfusion.sparsity,
            _estimate_cdf_threshold(),
            used,
            query.shape[-2],
            num_kv_heads,
            _estimate_protections_enabled(),
        )

        smask, _sct = get_estimate_mask(
            q,
            k,
            v,
            scale=self.softmax_scale,
            head_num=query.shape[-2],
            input_layout="BNSD",
            keep_sink=True,
            keep_recent=True,
            sparsity=self.rainfusion.sparsity,
            cdf_threshold=_estimate_cdf_threshold(),
            sparse_size=_BLOCK_SIZE,
            stride=_ESTIMATE_STRIDE,
        )
        kv_blocks = -(-used // _BLOCK_SIZE)
        mask = smask[..., :kv_blocks].contiguous()
        if _estimate_protections_enabled():
            # The pooling mask unconditionally keeps non-video context (rows
            # and columns) and each clip's first frame; sparse_block_estimate
            # only guarantees the sink and recent blocks per row, which lets
            # text key blocks lose the CDF competition on some rows and
            # throttles prompt conditioning.
            mask = _apply_estimate_protections(mask, _estimate_protected_token_ranges(plan))

        q_q, k_q, v_q, q_scales, k_scales, v_scales = _eagle_qbsa_quant_qkv(
            q, k, v, block_size_q=64, layout="BNSD"
        )
        seq_lens = [used] * batch
        out, _ = torch.ops.mindiesd.eagle_quant_block_sparse_attention(
            query=q_q,
            key=k_q,
            value=v_q.view(torch.int8),
            block_sparse_mask=mask,
            block_shape=[_BLOCK_SIZE, _BLOCK_SIZE],
            q_input_layout="BNSD",
            kv_input_layout="BNSD",
            num_key_value_heads=num_kv_heads,
            scale_value=self.softmax_scale,
            inner_precise=_ESTIMATE_INNER_PRECISE,
            softmax_lse_flag=0,
            actual_seq_lengths=seq_lens,
            actual_seq_lengths_kv=seq_lens,
            query_scale=q_scales,
            key_scale=k_scales,
            value_scale=v_scales,
            query_dtype=torch.int8,
            key_dtype=torch.int8,
            value_dtype=torch_npu.float8_e4m3fn,
            output_dtype=torch.bfloat16,
        )
        out = out.transpose(1, 2)
        if used == query.shape[1]:
            return out
        padded = torch.zeros_like(query)
        padded[:, :used] = out
        return padded

    def _forward_sparse_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        plan: RainFusionPlan,
    ) -> torch.Tensor:
        if _estimate_mask_requested() and self._estimate_mask_eligible(query, key, plan):
            return self._forward_sparse_estimate_npu(query, key, value, plan)
        try:
            from mindiesd import sparse_attention
        except ImportError:
            raise ImportError(_MISSING_MINDIESD)
        if self.rainfusion.precision != "bf16" and not _mindiesd_supports_precision():
            raise RuntimeError(
                f"block_sparse.precision={self.rainfusion.precision!r} requires MindIE-SD "
                "with sparse_attention(precision=...) support; the installed mindiesd "
                "silently ignores it and would run the BF16 path. Install a compatible "
                "MindIE-SD release or use precision='bf16'."
            )

        used = plan.used_len
        q, k, v = (tensor[:, :used] for tensor in (query, key, value))
        # Ulysses has already gathered the full sequence onto this rank and split
        # the heads, so read the head count off the tensor rather than num_heads.
        common_kwargs: dict[str, object] = {
            "scale": self.softmax_scale,
            "head_num": query.shape[-2],
            "input_layout": _INPUT_LAYOUT,
            "inner_precise": _INNER_PRECISE,
            "block_size": _BLOCK_SIZE,
            "sparsity": self.rainfusion.sparsity,
            "precision": self.rainfusion.precision,
        }
        if plan.video_spans is not None:
            out = sparse_attention(
                q,
                k,
                v,
                sparse_type="rf_v2",
                video_spans=plan.video_spans,
                **common_kwargs,
            )
        else:
            assert plan.prefix_len is not None and plan.latent_shape is not None
            common_kwargs.update(
                sparse_type="rf_v2",
                txt_len=plan.prefix_len,
                latent_shape_q=plan.latent_shape,
                latent_shape_k=plan.latent_shape,
            )
            out = sparse_attention(q, k, v, **common_kwargs)
        if used == query.shape[1]:
            return out
        padded = torch.zeros_like(query)
        padded[:, :used] = out
        return padded
