# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Literal

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.utils.attn_runtime_selector import can_sdpa_use_fused_gqa

logger = init_logger(__name__)


SDPAMaskMode = Literal["broadcast_k", "full_qk"]

_PACKED_PREFIX_KEYS = ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k")

# Escape hatch for hardware A/B: set to "0" to force the original full_qk mask
# path even when the packed prefix contract holds.
_PREFIX_SLICE_ENV = "OMNI_DIFFUSION_SDPA_KV_PREFIX_SLICE"

# (total_k, used) pairs whose 2D mask has already been verified to be exactly
# ``arange(total_k) < used``. See _resolve_valid_kv_prefix.
_VERIFIED_PREFIX: set[tuple[int, int]] = set()
_VERIFIED_PREFIX_MAX = 8


def _resolve_valid_kv_prefix(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: AttentionMetadata | None,
    *,
    causal: bool,
) -> int | None:
    """Length of the K/V prefix that can be dropped to avoid a quadratic mask.

    MiniMax-H3 packs a real document followed by an alignment-padding document
    (``cu_seqlens == [0, used, total]``) and masks the padding by building the
    query-independent 2D prefix mask ``arange(total) < used``. The NPU SDPA
    path expands that into a contiguous ``[B, 1, Sq, Skv]`` bool mask, which at
    ``Skv ~ 202k`` is a 38 GiB allocation -- quadratic in the *padded* length
    and therefore unavoidable whenever any padding exists.

    Truncating K/V to ``[:used]`` reproduces the masked result exactly: the
    excluded positions contribute softmax weight 0, so they can simply be
    dropped. Query keeps its full length, so the output shape and row order are
    unchanged and no re-assembly is needed. Note this is *stronger* than the
    varlen contract -- because the mask does not depend on the query row,
    padding-row outputs stay identical too.

    Every check below reads host-side shape/type metadata only, so no device
    sync is introduced. Returns ``None`` when the contract does not hold, which
    means "fall back to materializing the mask" -- the pre-existing behaviour.
    """
    # Load-bearing: with a causal mask, dropping the K suffix would change the
    # result for query rows that sit before it.
    if causal:
        return None
    if attn_metadata is None:
        return None

    # Strategies that break the "every query row sees the same K prefix"
    # premise, notably AllGather-KV (which slices Q to a shard and rewrites
    # attn_mask while leaving ``extra`` untouched), and the joint-* variants.
    if (
        attn_metadata.query_ranges is not None
        or attn_metadata.joint_query is not None
        or attn_metadata.joint_key is not None
        or attn_metadata.joint_attn_mask is not None
        or attn_metadata.full_attn_spans is not None
    ):
        return None
    if attn_metadata.attn_mask is None:
        return None
    if attn_metadata.attn_mask.dtype != torch.bool:
        return None

    extra = attn_metadata.extra
    if any(name not in extra for name in _PACKED_PREFIX_KEYS):
        return None
    cu_q, cu_k = extra["cu_seqlens_q"], extra["cu_seqlens_k"]
    used_q, used_k = extra["max_seqlen_q"], extra["max_seqlen_k"]
    # Python ints, not tensors: reading their value never syncs.
    if type(used_q) is not int or type(used_k) is not int or used_q != used_k:
        return None
    if getattr(cu_q, "shape", None) != (3,) or getattr(cu_k, "shape", None) != (3,):
        # Exactly two documents: [real, pad].
        return None

    if query.shape[0] != 1 or key.shape[0] != 1:
        return None
    total = key.shape[1]
    if query.shape[1] != total or value.shape != key.shape:
        return None

    attn_mask = attn_metadata.attn_mask
    if attn_mask.ndim != 2 or attn_mask.shape != (1, total):
        return None

    used = used_k
    if not 0 < used < total:
        return None
    if used < total - used:
        # [real, pad] convention: the real document is the longer one. This is
        # what distinguishes cu=[0, used, total] from cu=[0, pad, total]
        # without reading the device tensor.
        return None

    # The structural checks above cannot prove the mask *threshold* equals
    # ``used`` without touching device data. Verify once per shape: a missed
    # detection only OOMs loudly, while a false positive silently produces
    # wrong numbers, so the asymmetry is worth one sync per request.
    token = (total, used)
    if token not in _VERIFIED_PREFIX:
        expected = torch.arange(total, device=attn_mask.device)[None] < used
        if not bool(torch.equal(attn_mask.reshape(-1), expected.reshape(-1))):
            return None
        if len(_VERIFIED_PREFIX) >= _VERIFIED_PREFIX_MAX:
            _VERIFIED_PREFIX.clear()
        _VERIFIED_PREFIX.add(token)

    logger.info_once("SDPA packed prefix slice active (used=%d, total=%d)", used, total)
    return used


def _maybe_reshape_attn_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    mask_mode: SDPAMaskMode = "broadcast_k",
):
    """
    Reshape Attention Mask
    2D [batch_size, seq_len_k] ->
      - broadcast_k: [batch_size, 1, 1, seq_len_k]
      - full_qk: [batch_size, 1, seq_len_q, seq_len_k]
    """
    # Reshape Attention Mask
    # 2D [batch_size, seq_len_k] mask only.
    if (
        attn_mask is not None
        and attn_mask.ndim == 2
        and attn_mask.shape[0] == query.shape[0]
        and attn_mask.shape[1] == key.shape[1]
    ):
        B, Sq, Skv = attn_mask.shape[0], query.shape[1], key.shape[1]
        attn_mask = attn_mask.to(torch.bool)
        if mask_mode == "full_qk":
            # NPU path requires explicit [B, 1, Q, K] mask layout.
            attn_mask = attn_mask.unsqueeze(1).expand(B, Sq, Skv).unsqueeze(1).contiguous()
        elif mask_mode == "broadcast_k":
            # CUDA-like backends prefer [B, 1, 1, K] and rely on SDPA broadcast.
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        else:
            raise ValueError(f"Unsupported SDPA mask mode: {mask_mode}")
    return attn_mask


class SDPABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [x for x in range(1024)]  # todo

    @staticmethod
    def get_name() -> str:
        return "SDPA"

    @staticmethod
    def get_impl_cls() -> type["SDPAImpl"]:
        return SDPAImpl


class SDPAImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        if backend_kwargs:
            logger.warning("SDPAImpl ignoring backend_kwargs: %s", list(backend_kwargs.keys()))

    def _forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        mask_mode: SDPAMaskMode = "broadcast_k",
    ) -> torch.Tensor:
        # Normalize mask before permuting q/k/v.
        # _maybe_reshape_attn_mask expects sequence length on dim=1.
        attention_mask = None
        if attn_metadata:
            if mask_mode == "full_qk" and os.environ.get(_PREFIX_SLICE_ENV, "1") != "0":
                # full_qk materializes [B, 1, Sq, Skv]; for the packed
                # [real, pad] layout that is quadratic in the padded length.
                # Drop the fully masked K/V suffix instead -- equivalent, and
                # it leaves Sq (hence the output shape and row order) alone, so
                # nothing has to be re-assembled.
                used = _resolve_valid_kv_prefix(query, key, value, attn_metadata, causal=self.causal)
                if used is not None:
                    key = key[:, :used]
                    value = value[:, :used]
                    # Rebind the local only. Parallel strategies mutate the
                    # caller's AttentionMetadata in place during pre_attention.
                    attn_metadata = None
            if attn_metadata is not None:
                attention_mask = _maybe_reshape_attn_mask(query, key, attn_metadata.attn_mask, mask_mode=mask_mode)

        enable_gqa = query.shape[2] != key.shape[2]
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        # Only the PyTorch SDPA backend needs this dispatch check. If SDPA
        # cannot select a fused GQA kernel for the runtime shape/mask, expand
        # K/V locally so it can use the better-supported equal-head path.
        if enable_gqa and not can_sdpa_use_fused_gqa(query, key, value, attention_mask, self.causal):
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError(
                    "GQA requires query heads to be a multiple of KV heads, "
                    f"got q_heads={query.shape[1]} and kv_heads={key.shape[1]}."
                )
            repeat_num = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat_num, dim=1)
            value = value.repeat_interleave(repeat_num, dim=1)
            enable_gqa = False
            logger.debug(
                "CUDA SDPA cannot use a fused native-GQA kernel for this shape; expanding K/V heads before SDPA."
            )
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=self.causal,
            scale=self.softmax_scale,
            enable_gqa=enable_gqa,
        )
        out = output.permute(0, 2, 1, 3)
        return out

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata, mask_mode="broadcast_k")

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata, mask_mode="broadcast_k")

    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata, mask_mode="broadcast_k")

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata, mask_mode="full_qk")
