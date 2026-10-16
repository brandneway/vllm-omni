# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the VLLM-Omni project
"""Is splitting the head axis neutral for MindIE-SD's sparse attention?

The Ulysses head-split overlap path ("2a") hands the backend one head group at a
time instead of the full head axis. That is only sound if the sparse kernel's
result is per-head separable: the block mask is selected from pooled Q/K, and if
that selection is computed across heads rather than within each head, a group of
8 heads would select a different set of blocks than the same 8 heads padded out
to 56 -- and the split would silently change the output.

This runs one tensor through the kernel whole and again as ``--groups`` head
slices, and reports whether the concatenation matches bit for bit. It is a
single-card, no-model check: a couple of minutes instead of a full A/B.

    python tools/sparse_attn_head_split_check.py --only rf_v2 --layout spans
"""

from __future__ import annotations

import argparse
import sys

import torch

try:
    from mindiesd import sparse_attention
except Exception as exc:  # noqa: BLE001
    print(f"[FATAL] cannot import mindiesd.sparse_attention: {type(exc).__name__}: {exc}")
    sys.exit(2)


def _parse_latent(text: str) -> list[int]:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"latent shape must be T,H,W; got {text!r}")
    return [int(p) for p in parts]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="rf_v2", choices=("rf_v2", "rf_v3"))
    parser.add_argument("--layout", default="spans", choices=("spans", "latent"))
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--groups", type=int, default=7)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--sparsity", type=float, default=0.8)
    parser.add_argument("--txt-len", type=int, default=128)
    parser.add_argument("--latent", type=_parse_latent, default=[4, 24, 42])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _build(args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    span_tokens = args.latent[0] * args.latent[1] * args.latent[2]
    if args.layout == "spans":
        spans = [
            {"start": args.txt_len, "latent_shape": list(args.latent)},
            {"start": args.txt_len + span_tokens, "latent_shape": list(args.latent)},
        ]
        seq = args.txt_len + 2 * span_tokens
    else:
        spans = []
        seq = args.txt_len + span_tokens
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    q = torch.randn(1, seq, args.heads, args.dim, dtype=dtype, device="npu")
    k = torch.randn(1, seq, args.heads, args.dim, dtype=dtype, device="npu")
    v = torch.randn(1, seq, args.heads, args.dim, dtype=dtype, device="npu")
    return q, k, v, spans


def _call(args, q, k, v, spans, heads: int) -> torch.Tensor:
    common = {
        "head_num": heads,
        "input_layout": "BSND",
        "sparsity": args.sparsity,
        "block_size": 128,
        "precision": "bf16",
    }
    if args.layout == "spans":
        return sparse_attention(q, k, v, sparse_type=args.only, video_spans=spans, **common)
    return sparse_attention(
        q,
        k,
        v,
        sparse_type=args.only,
        txt_len=args.txt_len,
        latent_shape_q=list(args.latent),
        latent_shape_k=list(args.latent),
        **common,
    )


def _split(args, q, k, v, spans) -> torch.Tensor:
    """Same work, one head group at a time, concatenated back along heads."""
    step = args.heads // args.groups
    pieces = []
    for index in range(args.groups):
        lo, hi = index * step, (index + 1) * step
        pieces.append(_call(args, q[:, :, lo:hi].contiguous(), k[:, :, lo:hi].contiguous(),
                            v[:, :, lo:hi].contiguous(), spans, step))
    return torch.cat(pieces, dim=2)


def main() -> int:
    args = _parse_args()
    if not hasattr(torch, "npu"):
        print("[SKIP] torch.npu unavailable; run inside the NPU container.")
        return 0
    torch.npu.set_device(args.device)

    if args.heads % args.groups:
        print(f"[FATAL] heads={args.heads} is not divisible by groups={args.groups}")
        return 2

    q, k, v, spans = _build(args)
    label = f"{args.only}/{args.layout} heads={args.heads} groups={args.groups} seq={q.shape[1]}"
    print(f"[run] {label}")

    whole = _call(args, q, k, v, spans, args.heads)
    torch.npu.synchronize()
    split = _split(args, q, k, v, spans)
    torch.npu.synchronize()

    if whole.shape != split.shape:
        print(f"[FAIL] shape mismatch: whole={tuple(whole.shape)} split={tuple(split.shape)}")
        return 1

    diff = (whole.float() - split.float()).abs()
    max_abs = float(diff.max())
    exact = bool(torch.equal(whole, split))
    # Only meaningful once the two paths disagree.
    ref_scale = float(whole.float().abs().max()) or 1.0
    print(
        f"[result] bit-exact={exact} max_abs_diff={max_abs:.6e} "
        f"(reference scale {ref_scale:.3e}, relative {max_abs / ref_scale:.3e})"
    )
    # A single head group is the trivial case: it must reproduce the whole call.
    if args.groups == 1 and not exact:
        print("[FAIL] groups=1 must equal the unsplit call; the harness itself is wrong")
        return 1
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
