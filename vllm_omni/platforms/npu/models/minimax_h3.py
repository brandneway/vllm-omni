# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""NPU patches for the MiniMax H3 Qwen3-VL text encoder."""

from __future__ import annotations

import os

import torch
from vllm.logger import init_logger

from vllm_omni.platforms.npu.layers.rotary_embedding import (
    npu_rotary_mul_with_bsnd_fallback,
)

logger = init_logger(__name__)

_PATCHED = False

# Accuracy-triage dump for the fused RoPE path (all env-gated, off by
# default; unset VLLM_OMNI_ROPE_DUMP_DIR leaves this branch behaviorally
# identical to the plain single-variable isolation branch):
#   VLLM_OMNI_ROPE_DUMP_DIR        directory for the .pt dumps ("" = off)
#   VLLM_OMNI_ROPE_DUMP_MAX_CALLS  stop after N dumped calls (default 4)
#   VLLM_OMNI_ROPE_DUMP_SKIP_CALLS skip the first N calls, e.g. warmup (default 0)
#   VLLM_OMNI_ROPE_DUMP_RANK       rank that dumps (default "0"; "all" = every rank)
_ROPE_DUMP_DIR = os.environ.get("VLLM_OMNI_ROPE_DUMP_DIR", "")
_ROPE_DUMP_MAX_CALLS = int(os.environ.get("VLLM_OMNI_ROPE_DUMP_MAX_CALLS", "4"))
_ROPE_DUMP_SKIP_CALLS = int(os.environ.get("VLLM_OMNI_ROPE_DUMP_SKIP_CALLS", "0"))
_ROPE_DUMP_RANK = os.environ.get("VLLM_OMNI_ROPE_DUMP_RANK", "0")
_rope_dump_call_count = 0


def _rope_dump_rank_tag() -> str | None:
    """Rank tag when this process should dump, else None."""
    rank = "0"
    for var in ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK"):
        if var in os.environ:
            rank = os.environ[var]
            break
    if _ROPE_DUMP_RANK == "all" or rank == _ROPE_DUMP_RANK:
        return rank
    return None


def _rotate_half_ref(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _dump_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item()


def _maybe_dump_rope_call(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
) -> None:
    """Dump one real call of the fused RoPE path with reference outputs.

    For each dumped call the .pt payload carries the real inputs, the
    npu_rotary_mul outputs, the bf16 eager outputs (exactly the math the
    unpatched encoder runs, i.e. what fused-off executes), and an fp32
    golden over the same inputs. The server log gets the pairwise error
    stats; compare "kernel-vs-eager" against "eager-vs-fp32" (the bf16
    arithmetic floor): a kernel error far above that floor indicts the
    fused op, not the integration.
    """
    global _rope_dump_call_count
    if not _ROPE_DUMP_DIR:
        return
    tag = _rope_dump_rank_tag()
    if tag is None:
        return
    idx = _rope_dump_call_count
    _rope_dump_call_count += 1
    if idx < _ROPE_DUMP_SKIP_CALLS or idx - _ROPE_DUMP_SKIP_CALLS >= _ROPE_DUMP_MAX_CALLS:
        return
    try:
        cos_b, sin_b = cos.unsqueeze(1), sin.unsqueeze(1)
        # bf16 eager reference: the unpatched encoder's exact expression.
        q_ref = (q * cos_b) + (_rotate_half_ref(q) * sin_b)
        k_ref = (k * cos_b) + (_rotate_half_ref(k) * sin_b)
        # fp32 golden over the same (already bf16) inputs: the arithmetic ceiling.
        cos_f, sin_f = cos_b.float(), sin_b.float()
        q_gold = (q.float() * cos_f) + (_rotate_half_ref(q.float()) * sin_f)
        k_gold = (k.float() * cos_f) + (_rotate_half_ref(k.float()) * sin_f)

        k_ve_max, k_ve_mean = _dump_stats(q_out, q_ref)
        e_g_max, e_g_mean = _dump_stats(q_ref, q_gold)
        k_g_max, _ = _dump_stats(q_out, q_gold)
        kk_ve_max, kk_ve_mean = _dump_stats(k_out, k_ref)
        logger.info(
            "[rope-dump] call#%d rank=%s q%s %s cos%s | "
            "kernel-vs-eager max=%.3e mean=%.3e | eager-vs-fp32(floor) max=%.3e mean=%.3e | "
            "kernel-vs-fp32 max=%.3e | k: kernel-vs-eager max=%.3e mean=%.3e | q_amax=%.3f",
            idx,
            tag,
            tuple(q.shape),
            str(q.dtype).removeprefix("torch."),
            tuple(cos.shape),
            k_ve_max,
            k_ve_mean,
            e_g_max,
            e_g_mean,
            k_g_max,
            kk_ve_max,
            kk_ve_mean,
            q.float().abs().max().item(),
        )

        os.makedirs(_ROPE_DUMP_DIR, exist_ok=True)
        payload = {
            "meta": {
                "call_index": idx,
                "rank": tag,
                "q_shape": list(q.shape),
                "q_dtype": str(q.dtype),
                "k_shape": list(k.shape),
                "cos_shape": list(cos.shape),
                "cos_dtype": str(cos.dtype),
                "stats": {
                    "q_kernel_vs_eager": {"max": k_ve_max, "mean": k_ve_mean},
                    "q_eager_vs_fp32_floor": {"max": e_g_max, "mean": e_g_mean},
                    "q_kernel_vs_fp32": {"max": k_g_max},
                    "k_kernel_vs_eager": {"max": kk_ve_max, "mean": kk_ve_mean},
                },
            },
            "q": q.detach().cpu(),
            "k": k.detach().cpu(),
            "cos": cos.detach().cpu(),
            "sin": sin.detach().cpu(),
            "q_out_kernel": q_out.detach().cpu(),
            "k_out_kernel": k_out.detach().cpu(),
            "q_out_eager": q_ref.detach().cpu(),
            "k_out_eager": k_ref.detach().cpu(),
            "q_out_fp32": q_gold.detach().cpu(),
            "k_out_fp32": k_gold.detach().cpu(),
        }
        path = os.path.join(_ROPE_DUMP_DIR, f"rope_r{tag}_call{idx:04d}.pt")
        torch.save(payload, path)
        logger.info("[rope-dump] saved %s", path)
    except Exception:
        logger.exception("[rope-dump] dump failed (ignored, serving continues)")


def _apply_rotary_pos_emb_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3-VL text RoPE with fused BNSD rotary multiplication."""
    q_out = npu_rotary_mul_with_bsnd_fallback(q, cos, sin, unsqueeze_dim=1)
    k_out = npu_rotary_mul_with_bsnd_fallback(k, cos, sin, unsqueeze_dim=1)
    _maybe_dump_rope_call(q, k, cos, sin, q_out, k_out)
    return q_out, k_out


def apply_minimax_h3_qwen3vl_patch() -> None:
    """Route MiniMax H3 Qwen3-VL text RoPE to the Ascend fused operator."""
    global _PATCHED
    if _PATCHED:
        return

    from vllm_omni.diffusion.models.minimax_h3 import encoder

    encoder._apply_rotary_pos_emb = _apply_rotary_pos_emb_npu
    _PATCHED = True
    logger.debug("Applied NPU fused RoPE patch for MiniMax H3 Qwen3-VL text encoder")
