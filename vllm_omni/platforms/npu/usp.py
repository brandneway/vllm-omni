# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ascend unified sequence-parallel attention executor backed by MindIE-SD.

The executor delegates sparse-eligible RAINFUSION_ATTN calls to
``mindiesd.parallel.distributed_sparse_attention`` — the multi-rank form of the
quantized block-sparse chain (Q/K per-block INT8, V per-channel FP8, the
EagleQBSA device operator). Dense calls and every other backend return
``None`` so the portable attention layer falls back to vLLM-Omni's native
Ulysses/Ring implementation without duplicating collectives.

Scope is deliberately narrow: pure Ulysses topologies only (a single process
group — the MindIE-SD chain has no KV-gather group concept), single-request
packed sequences, and the ``mix`` sparse precision (the only contract the
parallel chain implements).
"""

from __future__ import annotations

import importlib
import threading
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

logger = init_logger(__name__)

# One ParallelState per process: it owns the single long-lived collective
# stream and the layer-invariant plan/engine caches. Creating one per layer
# would spawn one collective-issuing side stream per layer — the exact
# double-issuer hazard the MindIE-SD chain documents (stale a2a receives).
_PROCESS_STATE = None
_PROCESS_STATE_LOCK = threading.Lock()


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
        self._parallel_module: ModuleType | None = None
        self._load_attempted = False

    def _load_parallel_module(self) -> ModuleType | None:
        if self._load_attempted:
            return self._parallel_module
        self._load_attempted = True
        try:
            module = importlib.import_module("mindiesd.parallel")
        except ImportError as exc:
            logger.warning_once(
                "Ascend USP is enabled but mindiesd.parallel is unavailable; "
                "using vLLM-Omni native sequence-parallel attention: %s",
                exc,
            )
            return None

        if not callable(getattr(module, "distributed_sparse_attention", None)) or not isinstance(
            getattr(module, "ParallelState", None), type
        ):
            logger.warning_once(
                "Ascend USP is enabled but the installed MindIE-SD lacks the "
                "mindiesd.parallel sparse chain; using vLLM-Omni native "
                "sequence-parallel attention."
            )
            return None
        self._parallel_module = module
        logger.info_once("Using the Ascend unified sequence-parallel attention executor (mindiesd.parallel).")
        return module

    def _process_state(self, module: ModuleType, group: object, device: torch.device):
        """Return the one process-wide ParallelState, creating it on first use."""
        global _PROCESS_STATE
        with _PROCESS_STATE_LOCK:
            if _PROCESS_STATE is None:
                state = module.ParallelState(group=group)
                # One long-lived collective stream for the whole process; the
                # chain's overlap pipeline is built on it.
                state.make_stream(device)
                _PROCESS_STATE = state
            return _PROCESS_STATE

    def _supports_call(
        self,
        query: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        *,
        backend_name: str,
        causal: bool,
        scatter_dim: int,
        gather_dim: int,
    ) -> bool:
        # The MindIE-SD parallel chain is the quantized block-sparse path only;
        # dense calls and all other backends stay on the native path.
        if backend_name != "RAINFUSION_ATTN":
            return False
        # Pure Ulysses only: the chain takes a single process group and has no
        # KV-gather group to compose with.
        if self.ulysses_degree <= 1 or self.ring_degree > 1 or self.allgather_degree > 1:
            return False
        if causal:
            return False
        if scatter_dim != 2 or gather_dim != 1 or query.ndim != 4:
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
        # The chain accepts single-request packed sequences only: the spans
        # geometry describes one document, and padding is excluded by used_len.
        num_requests = attn_metadata.extra.get("num_requests", 1)
        if num_requests != 1:
            return False
        # packed_padding / video_layout and the packed-varlen extras are
        # consumed by the sparse plan resolver, not passed through blindly.
        unsupported_extra = {
            "gate_compress",
            "kv_cache_dtype",
            "seq_lens",
        }
        return not unsupported_extra.intersection(attn_metadata.extra)

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
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
        ):
            logger.warning_once(
                "Ascend USP executor declined this attention call (backend=%s); "
                "falling back to the native sequence-parallel path.",
                backend_name,
            )
            return None

        # A forward without a sparse plan is dense by definition; the parallel
        # chain has no dense arm, so the native path takes it.
        if sparse_plan is None:
            return None

        module = self._load_parallel_module()
        if module is None:
            return None

        state = self._process_state(module, self.sp_group.ulysses_group, query.device)

        return module.distributed_sparse_attention(
            query,
            key,
            value,
            sparse_plan["spans"],
            group=self.sp_group.ulysses_group,
            scale=float(softmax_scale),
            sparsity=float(sparse_plan["sparsity"]),
            used_len=int(sparse_plan["used_len"]),
            state=state,
        )


__all__ = ["AscendUSPExecutor"]
