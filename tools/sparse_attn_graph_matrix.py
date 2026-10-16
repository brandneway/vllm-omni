#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Which A2/A3 sparse-attention operators can be captured by an NPUGraph?

Graph-capturing H3's head-split region failed on-device with two errors that
name the operator, not the Python around it:

    aclnnNonzeroV2 ...
    Not_Supported(EE1016): Synchronizing a stream failed.
      Reason: Stream (stream_id=92) during the capture stage is not supported.

The Python layer of every sparse path is clean -- no host synchronize, no
``.item()``/``.cpu()``/``.tolist()``, no ``nonzero``/``unique``, no boolean-mask
indexing -- so the blocker has to be inside the aclnn operator each path ends
in. ``rf_v2`` ends in ``rain_fusion_attention`` and is the one we measured
failing. ``rf_v3`` and ``ada_bsa`` end in *different* operators
(``bsa_sparse_attention_v3`` / ``ada_block_sparse_attention``), so the rf_v2
verdict does not carry over.

This script settles that per operator, with no model and a single process:

    ASCEND_RT_VISIBLE_DEVICES=4 python3 tools/sparse_attn_graph_matrix.py --only rf_v2

Run one operator per invocation. A capture that the runtime refuses can leave
the stream unusable, so serialising several operators in one process would
report the first failure and lie about the rest; the shell loop in
``--help`` keeps them isolated.

Shapes follow H3 (56 heads, head_dim 128, BSND, bf16, sparsity 0.8) at a
fraction of the packed length; the operator's own block structure needs the
sequence to cover `latent_shape` exactly.
"""

from __future__ import annotations

import argparse
import traceback

import torch

try:
    from mindiesd import sparse_attention
except Exception as exc:  # noqa: BLE001 - the verdict is the script's output
    print(f"[FATAL] cannot import mindiesd.sparse_attention: {type(exc).__name__}: {exc}")
    raise SystemExit(2)


def _parse_latent(text: str) -> list[int]:
    dims = [int(part) for part in text.split(",")]
    if len(dims) != 3:
        raise argparse.ArgumentTypeError(f"latent shape must be T,H,W; got {text!r}")
    return dims


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Isolate the operators so one bad capture does not mask the rest:\n"
            "  for t in rf_v2 rf_v3 ada_bsa; do\n"
            "    ASCEND_RT_VISIBLE_DEVICES=4 python3 tools/sparse_attn_graph_matrix.py --only $t\n"
            "  done\n"
        ),
    )
    parser.add_argument("--only", required=True, choices=("rf_v2", "rf_v3", "ada_bsa"))
    parser.add_argument("--layout", default="spans", choices=("spans", "latent"), help="rf_v2/rf_v3 geometry form")
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--sparsity", type=float, default=0.8)
    parser.add_argument("--txt-len", type=int, default=128)
    parser.add_argument("--latent", type=_parse_latent, default=[4, 24, 42], help="T,H,W per video span")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="capture only the multi-span rearrange/pooling + mask, without the attention op. "
        "Localises whether the offending op is in MindIE-SD's Python or below it.",
    )
    parser.add_argument(
        "--patch-permutation",
        action="store_true",
        help="replace the boolean-mask gather in _multi_span_permutation with its closed form, "
        "then capture. Proves the fix works and that it is bit-exact.",
    )
    return parser.parse_args()


def _multi_span_preprocess(args, q: torch.Tensor, spans: list[dict]):
    """The video_spans preprocessing, called directly so it can be captured alone."""
    from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import (
        do_multi_span_tensor_rearrange_pooling,
        get_multi_span_blockwise_mask,
    )

    key = torch.randn_like(q)
    value = torch.randn_like(q)
    _, _, _, qkv_pool, inverse, dense_blocks, first_frame_blocks = do_multi_span_tensor_rearrange_pooling(
        q, key, value, spans, 128, "BSND"
    )
    select_idx, select_num_idx = get_multi_span_blockwise_mask(
        qkv_pool, args.sparsity, args.dim**-0.5, dense_blocks, first_frame_blocks, "BSND"
    )
    return select_idx, select_num_idx, inverse


def _patched_multi_span_permutation(module):
    """The same permutation, built without boolean-mask indexing.

    ``_multi_span_permutation`` derives the dense-context rows with
    ``all_indices[~is_video]``. On Ascend that lowers to ``aclnnNonzeroV2``,
    which is data-dependent in shape and therefore refused by stream capture --
    and it is the only capture-hostile op anywhere in the sparse paths.

    The rows it selects are exactly the complement of the ``video_spans``
    ranges, and ``video_spans`` is Python ints, so the same tensor is available
    in closed form. Same values, same ascending order, so this is a bit-exact
    replacement that happens to be capturable.
    """
    plan = module._multi_span_plan
    span_indices = module._span_rearranged_indices

    def _multi_span_permutation(spans, sequence_len, pool_size, input_layout, device):
        segments, dense_blocks, first_frame_blocks = plan(spans, sequence_len, pool_size)

        gaps = []
        cursor = 0
        for start, _shape, length in spans:
            if start > cursor:
                gaps.append((cursor, start))
            cursor = start + length
        if cursor < sequence_len:
            gaps.append((cursor, sequence_len))
        dense_indices = (
            torch.cat([torch.arange(lo, hi, device=device, dtype=torch.long) for lo, hi in gaps])
            if gaps
            else torch.empty(0, device=device, dtype=torch.long)
        )

        dense_cursor = 0
        parts = []
        for segment in segments:
            if segment[0] == "clip":
                parts.append(span_indices(segment[1], segment[2], input_layout, device))
            else:
                parts.append(dense_indices[dense_cursor : dense_cursor + segment[1]])
                dense_cursor += segment[1]

        permutation = torch.cat(parts)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(sequence_len, device=device, dtype=torch.long)
        return permutation, inverse, dense_blocks, first_frame_blocks

    return _multi_span_permutation


def _patched_span_blockwise_mask(module):
    """Same mask, with the host-to-device index build hoisted out of the call.

    ``get_multi_span_blockwise_mask`` builds ``torch.tensor(retained, device=...)``
    from a Python list on every call. That is a host-to-device copy, which stream
    capture refuses (``copy_between_host_and_device_opapi``, 107030). The list is
    a function of the spans, so the tensor is cached per index set: the copy
    happens once, and the captured region only sees an existing device tensor.
    """
    import math

    from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import get_mask_index

    cache: dict[tuple[int, ...], torch.Tensor] = {}

    def get_multi_span_blockwise_mask(
        qkv_pool, sparsity, scale, dense_blocks, first_frame_blocks, input_layout, return_binary=False
    ):
        query_pool, key_pool, _ = torch.chunk(qkv_pool, 3, dim=0)
        if input_layout == "BSND":
            scores = torch.einsum("blnd,bsnd->bnls", query_pool, key_pool) * scale
        else:
            scores = torch.einsum("bnld,bnsd->bnls", query_pool, key_pool) * scale
        score_matrix = torch.nn.functional.softmax(scores, dim=-1)
        keep_len = max(1, math.ceil(score_matrix.shape[-1] * (1 - sparsity)))
        threshold = torch.topk(score_matrix, k=keep_len, dim=-1).values[..., -1:]
        mask = score_matrix >= threshold
        retained = sorted(dense_blocks | first_frame_blocks)
        if retained:
            key = tuple(retained)
            retained_idx = cache.get(key)
            if retained_idx is None or retained_idx.device != mask.device:
                retained_idx = torch.tensor(retained, device=mask.device, dtype=torch.long)
                cache[key] = retained_idx
            mask[:, :, retained_idx, :] = True
            mask[:, :, :, retained_idx] = True
        if return_binary:
            return mask.to(torch.int8)
        select_idx = get_mask_index(mask)[0].transpose(0, 1)
        select_num_idx = mask[0].transpose(0, 1).sum(dim=-1)
        return select_idx, select_num_idx

    return get_multi_span_blockwise_mask


def _check_patch_is_bit_exact(args, q: torch.Tensor, spans: list[dict]) -> bool:
    """Original vs patched permutation: the precision claim, checked directly."""
    import mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 as rf_v2

    seq = q.shape[1]
    normalized = rf_v2._normalize_video_spans(spans, seq)
    original = rf_v2._multi_span_permutation(normalized, seq, 128, "BSND", q.device)
    patched = _patched_multi_span_permutation(rf_v2)(normalized, seq, 128, "BSND", q.device)
    same_perm = bool(torch.equal(original[0], patched[0]))
    same_inv = bool(torch.equal(original[1], patched[1]))
    print(f"[patch] permutation identical: {same_perm}; inverse identical: {same_inv}")
    return same_perm and same_inv


def _build_inputs(args) -> tuple[torch.Tensor, list[dict]]:
    """Return (q, spans) with q shaped (1, seq, heads, dim) in BSND.

    The two geometry forms need different sequence lengths: ``spans`` packs two
    videos (the Ref2VA shape the H3 deployment actually uses), ``latent`` is the
    single-video form whose video extent must equal ``t*h*w``.
    """
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
    q = torch.randn(1, seq, args.heads, args.dim, dtype=dtype, device="npu")
    return q, spans


def _call(args, q: torch.Tensor, spans: list[dict], key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    common = {
        "head_num": args.heads,
        "input_layout": "BSND",
        "sparsity": args.sparsity,
        "block_size": 128,
        "precision": "bf16",
    }
    if args.only == "ada_bsa":
        return sparse_attention(q, key, value, sparse_type="ada_bsa", keep_sink=True, keep_recent=True, **common)
    if args.layout == "spans":
        return sparse_attention(q, key, value, sparse_type=args.only, video_spans=spans, **common)
    return sparse_attention(
        q,
        key,
        value,
        sparse_type=args.only,
        txt_len=args.txt_len,
        latent_shape_q=list(args.latent),
        latent_shape_k=list(args.latent),
        **common,
    )


def main() -> int:
    args = _parse_args()
    if not hasattr(torch, "npu"):
        print("[SKIP] torch.npu unavailable; run inside the NPU container.")
        return 0
    torch.npu.set_device(args.device)

    q, spans = _build_inputs(args)
    key = torch.randn_like(q)
    value = torch.randn_like(q)
    label = f"{args.only}/{args.layout}" if args.only != "ada_bsa" else "ada_bsa"
    if args.patch_permutation:
        import mindiesd.layers.flash_attn.sparse_flash_attn as sf
        import mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 as rf_v2

        _check_patch_is_bit_exact(args, q, spans)
        rf_v2._multi_span_permutation = _patched_multi_span_permutation(rf_v2)
        # ``sparse_attention`` binds this name with `from ... import`, so patching
        # the definition module alone would leave the caller on the original.
        patched_mask = _patched_span_blockwise_mask(rf_v2)
        rf_v2.get_multi_span_blockwise_mask = patched_mask
        sf.get_multi_span_blockwise_mask = patched_mask
        label += "+patched"
    if args.preprocess_only:
        label += "+preprocess-only"

        def call():
            return _multi_span_preprocess(args, q, spans)

    else:

        def call():
            return _call(args, q, spans, key, value)

    try:
        eager = call()
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] EAGER FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(f"[{label}] eager ok")

    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    try:
        torch.npu.synchronize()
        with torch.inference_mode(), torch.npu.graph(graph, pool=pool):
            captured = call()
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001 - the verdict is the point
        print(f"[{label}] CAPTURE FAILED: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        return 1
    print(f"[{label}] CAPTURE OK")

    try:
        graph.replay()
        torch.npu.synchronize()
        drift = (captured[0].float() - eager[0].float()).abs().max().item()
        print(f"[{label}] replay-vs-eager max abs diff = {drift:.3e}")
    except Exception as exc:  # noqa: BLE001
        print(f"[{label}] REPLAY FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
