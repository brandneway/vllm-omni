# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ascend implementation of unified sequence-parallel attention."""

from __future__ import annotations

import importlib
import math
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

logger = init_logger(__name__)


class SequenceParallelGroups(Protocol):
    """Process groups owned by vLLM-Omni's SP coordinator."""

    ulysses_group: object
    ring_group: object
    ring_rank: int


class AscendUSPExecutor:
    """Execute supported Ascend SP attention through MindIE-SD.

    Unsupported calls return ``None`` so the portable attention layer can use
    its existing Ulysses/Ring implementation without duplicating collectives.
    """

    def __init__(
        self,
        *,
        sp_group: SequenceParallelGroups,
        ulysses_degree: int,
        ring_degree: int,
        allgather_degree: int,
        ulysses_mode: str,
    ) -> None:
        self.sp_group = sp_group
        self.ulysses_degree = ulysses_degree
        self.ring_degree = ring_degree
        self.allgather_degree = allgather_degree
        self.ulysses_mode = ulysses_mode
        self._usp_module: ModuleType | None = None
        self._load_attempted = False

    def _load_usp_module(self) -> ModuleType | None:
        if self._load_attempted:
            return self._usp_module
        self._load_attempted = True
        try:
            module = importlib.import_module("mindiesd.layers.usp")
        except ImportError as exc:
            logger.warning_once(
                "Ascend USP is enabled but mindiesd.layers.usp is unavailable; "
                "using vLLM-Omni native sequence-parallel attention: %s",
                exc,
            )
            return None

        required = ("usp_attention",)
        error_types = ("USPError", "USPNotSupported")
        if not all(callable(getattr(module, name, None)) for name in required) or not all(
            isinstance(getattr(module, name, None), type) for name in error_types
        ):
            logger.warning_once(
                "Ascend USP is enabled but the installed MindIE-SD API is incompatible; "
                "using vLLM-Omni native sequence-parallel attention."
            )
            return None
        self._usp_module = module
        logger.info_once("Using the Ascend unified sequence-parallel attention executor.")
        return module

    def _groups(self) -> tuple[object | None, object | None]:
        ulysses_group = self.sp_group.ulysses_group if self.ulysses_degree > 1 else None
        kv_gather_group = self.sp_group.ring_group if self.ring_degree > 1 else None
        return ulysses_group, kv_gather_group

    def _supports_call(
        self,
        query: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        *,
        backend_name: str,
        causal: bool,
        softmax_scale: float,
        scatter_dim: int,
        gather_dim: int,
    ) -> bool:
        if backend_name not in ("FLASH_ATTN", "RAINFUSION_ATTN"):
            return False
        if self.allgather_degree > 1 or self.ulysses_degree * self.ring_degree == 1:
            return False
        if self.ulysses_mode != "strict" or causal:
            return False
        if scatter_dim != 2 or gather_dim != 1 or query.ndim != 4:
            return False
        if not math.isclose(float(softmax_scale), query.shape[-1] ** -0.5, rel_tol=1e-6, abs_tol=1e-8):
            return False
        if attn_metadata is None:
            return True
        if any(
            tensor is not None
            for tensor in (
                attn_metadata.joint_query,
                attn_metadata.joint_key,
                attn_metadata.joint_value,
                attn_metadata.joint_attn_mask,
            )
        ):
            return False
        if attn_metadata.attn_mask is not None:
            return False
        if attn_metadata.full_attn_spans is not None or attn_metadata.query_ranges is not None:
            return False
        # packed_padding / video_layout and the packed-varlen extras are
        # consumed by the executor (kv_used_len derivation, sparse plan), not
        # passed through blindly.
        unsupported_extra = {
            "gate_compress",
            "kv_cache_dtype",
            "seq_lens",
        }
        return not unsupported_extra.intersection(attn_metadata.extra)

    def _resolve_kv_used_len(self, attn_metadata: AttentionMetadata | None) -> int | None:
        """Derive the valid (non-padding) prefix length of the packed sequence."""
        if attn_metadata is None:
            return None
        extra = attn_metadata.extra
        if not extra:
            return None
        cu_seqlens = extra.get("cu_seqlens_q")
        if cu_seqlens is not None and hasattr(cu_seqlens, "shape") and cu_seqlens.shape[0] > 2:
            raise ValueError(
                "Ascend USP execution supports single-request packed sequences in v1: "
                f"cu_seqlens_q has {cu_seqlens.shape[0]} entries (multi-request step-mode batching). "
                "Run with enable_usp=False for batched requests."
            )
        used = extra.get("valid_kv_length")
        if isinstance(used, int):
            return used
        max_seqlen_q = extra.get("max_seqlen_q")
        if isinstance(max_seqlen_q, int):
            return max_seqlen_q
        return None

    def _derive_sparse_geometry(
        self,
        query: torch.Tensor,
        sparse_plan: dict,
        kv_used_len: int,
    ) -> dict:
        """Map this rank's post-A2A Q segment to explicit MindIE sparse kwargs.

        After the Ulysses A2A, each rank holds a contiguous ``S / ring_degree``
        row segment of the packed sequence (the kv-gather group index selects
        which one). All computations are host-side. Segment boundaries may cut
        a video frame mid-way; MindIE-SD splits those rows to dense FA.
        """
        sp_size = self.ulysses_degree * self.ring_degree
        s_global = query.shape[1] * sp_size
        if s_global % self.ring_degree != 0:
            raise ValueError(f"packed length {s_global} is not divisible by ring_degree={self.ring_degree}.")
        segment_rows = s_global // self.ring_degree
        half_index = int(getattr(self.sp_group, "ring_rank")) if self.ring_degree > 1 else 0
        q_row_offset = half_index * segment_rows

        prefix_len = int(sparse_plan["txt_len_kv"])
        if q_row_offset == 0 and segment_rows < prefix_len:
            raise ValueError(
                f"the first CP segment ({segment_rows} rows) does not cover the prefix "
                f"({prefix_len} rows); lower ring_degree or use a longer sequence."
            )
        # Whole/partial video-frame splitting happens MindIE-side: a mid-frame
        # segment boundary is legal and only routes those rows through dense FA.
        return {
            "sparse": sparse_plan["sparse"],
            "sparsity": sparse_plan["sparsity"],
            "sparse_precision": sparse_plan["sparse_precision"],
            "txt_len_q": prefix_len if q_row_offset == 0 else 0,
            "txt_len_kv": prefix_len,
            "latent_shape_kv": [int(x) for x in sparse_plan["latent_shape_kv"]],
            "q_row_offset": q_row_offset,
            "kv_used_len": int(kv_used_len),
        }

    def try_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_metadata: AttentionMetadata | None,
        backend_name: str,
        causal: bool,
        softmax_scale: float,
        scatter_dim: int,
        gather_dim: int,
        sparse_plan: dict | None = None,
    ) -> torch.Tensor | None:
        """Run the Ascend USP executor when compatible, else request fallback."""
        if not self._supports_call(
            query,
            attn_metadata,
            backend_name=backend_name,
            causal=causal,
            softmax_scale=softmax_scale,
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
        ):
            return None

        module = self._load_usp_module()
        if module is None:
            return None

        ulysses_group, kv_gather_group = self._groups()

        kv_used_len = self._resolve_kv_used_len(attn_metadata)
        sparse_kwargs: dict = {}
        if sparse_plan is not None:
            if kv_used_len is None:
                raise ValueError(
                    "USP sparse execution needs the packed used length (valid_kv_length/max_seqlen_q) "
                    "in attention metadata extras."
                )
            sparse_kwargs = self._derive_sparse_geometry(query, sparse_plan, kv_used_len)
        elif kv_used_len is not None:
            sparse_kwargs = {"kv_used_len": kv_used_len}

        try:
            return module.usp_attention(
                query,
                key,
                value,
                ulysses_group=ulysses_group,
                kv_gather_group=kv_gather_group,
                **sparse_kwargs,
            )
        except module.USPNotSupported as exc:
            logger.warning_once(
                "Ascend USP rejected the current attention contract; "
                "using vLLM-Omni native sequence-parallel attention: %s",
                exc,
            )
            return None


__all__ = ["AscendUSPExecutor"]
