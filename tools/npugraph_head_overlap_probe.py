#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Probe whether an Ascend NPUGraph can capture the head-split Ulysses pipeline.

``Attention._forward_impl_head_overlap`` splits the QKV all-to-all into one
collective per head group, runs them back to back on a dedicated stream, and
runs each group's attention on the compute stream behind that group's
completion event. Whether that region can be captured is the open question for
the graph-mode follow-up: a capture that contains cross-stream HCCL collectives
is the risky part, and it is cheaper to answer here than by wiring a mode into
the model.

This script is standalone on purpose. It reproduces the *shape* of the pipeline
(Ulysses head-scatter all-to-all, a per-group elementwise stand-in for the
attention, the split reverse all-to-all, reassembly) on random tensors, so a
verdict does not depend on the DiT, the checkpoint, or the sparse kernel.

Run inside the NPU container, one process per rank:

    torchrun --nproc_per_node=8 tools/npugraph_head_overlap_probe.py

It reports, in order:

1. whether the split pipeline reproduces the unsplit one (arithmetic check --
   this is also the invariant ``plan_head_overlap`` is built on);
2. whether ``torch.npu.graph`` accepts the multi-stream region;
3. whether a replayed graph matches the eager pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heads", type=int, default=56, help="total attention heads (H3 uses 56)")
    parser.add_argument("--seq", type=int, default=4096, help="local sequence length per rank")
    parser.add_argument("--dim", type=int, default=128, help="head dim")
    parser.add_argument("--groups", type=int, default=0, help="head groups; 0 derives the plainest valid split")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--skip-capture", action="store_true", help="only run the arithmetic check")
    return parser.parse_args()


def _valid_group_counts(heads: int, world: int) -> list[int]:
    return [g for g in range(2, heads + 1) if heads % g == 0 and (heads // g) % world == 0]


def _ulysses_a2a(x: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
    """Head-scatter / sequence-gather all-to-all, equal shards (B, S, H, D).

    Mirrors ``_ulysses_all_to_all_any_qkv`` for the equal-length, unpadded case
    the probe assumes.
    """
    world = dist.get_world_size(pg)
    if world == 1:
        return x
    bsz, s_local, heads, dim = x.shape
    heads_local = heads // world
    x_t = x.reshape(bsz, s_local, world, heads_local, dim).permute(2, 1, 0, 3, 4).contiguous()
    x_t = x_t.flatten(0, 1)
    out = torch.empty((world * s_local, bsz, heads_local, dim), device=x.device, dtype=x.dtype)
    dist.all_to_all_single(out, x_t, group=pg)
    return out.permute(1, 0, 2, 3).contiguous()


def _ulysses_a2a_reverse(x: torch.Tensor, pg: dist.ProcessGroup, s_local: int) -> torch.Tensor:
    """Inverse of :func:`_ulysses_a2a`, (B, S_global, H_local, D) -> (B, S_local, H, D)."""
    world = dist.get_world_size(pg)
    if world == 1:
        return x
    bsz, _, heads_local, dim = x.shape
    x_t = x.permute(1, 0, 2, 3).contiguous()
    out = torch.empty((world * s_local, bsz, heads_local, dim), device=x.device, dtype=x.dtype)
    dist.all_to_all_single(out, x_t, group=pg)
    out = out.reshape(world, s_local, bsz, heads_local, dim).permute(2, 1, 0, 3, 4).contiguous()
    return out.reshape(bsz, s_local, world * heads_local, dim)


def _attention_stand_in(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Head-independent elementwise stand-in; only the pipeline shape matters."""
    return (q.to(torch.float32) * 0.5 + k.to(torch.float32) * 2.0 + v.to(torch.float32)).to(q.dtype)


def _unsplit_forward(query, key, value, pg, s_local):
    q = _ulysses_a2a(query, pg)
    k = _ulysses_a2a(key, pg)
    v = _ulysses_a2a(value, pg)
    out = _attention_stand_in(q, k, v)
    return _ulysses_a2a_reverse(out, pg, s_local)


def _split_forward(query, key, value, pg, s_local, groups, comm, forward_done, computed):
    """The pipelined shape: per-group all-to-all on ``comm``, attention behind it."""
    step = query.shape[2] // groups
    compute = torch.npu.current_stream()

    comm.wait_stream(compute)
    prepared = []
    with torch.npu.stream(comm):
        for index in range(groups):
            lo = index * step
            hi = lo + step
            prepared.append(
                (
                    _ulysses_a2a(query[:, :, lo:hi].contiguous(), pg),
                    _ulysses_a2a(key[:, :, lo:hi].contiguous(), pg),
                    _ulysses_a2a(value[:, :, lo:hi].contiguous(), pg),
                )
            )
            forward_done[index].record(comm)

    pieces = []
    for index, (q, k, v) in enumerate(prepared):
        compute.wait_event(forward_done[index])
        part = _attention_stand_in(q, k, v)
        computed[index].record(compute)
        with torch.npu.stream(comm):
            comm.wait_event(computed[index])
            pieces.append(_ulysses_a2a_reverse(part, pg, s_local))
    torch.npu.current_stream().wait_stream(comm)
    return torch.cat(pieces, dim=2)


def main() -> int:
    args = _parse_args()
    if not hasattr(torch, "npu"):
        print("[SKIP] torch.npu is unavailable; run this inside the NPU container.")
        return 0

    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.npu.set_device(rank)
    dtype = getattr(torch, args.dtype)

    valid = _valid_group_counts(args.heads, world)
    groups = args.groups or (valid[-1] if valid else 0)
    if groups < 2:
        if rank == 0:
            print(f"[SKIP] heads={args.heads} world={world} admits no head split (valid={valid})")
        dist.destroy_process_group()
        return 0

    torch.manual_seed(1234 + rank)
    query = torch.randn(1, args.seq, args.heads, args.dim, dtype=dtype, device="npu")
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    reference = _unsplit_forward(query, key, value, dist.group.WORLD, args.seq)

    comm = torch.npu.Stream()
    forward_done = [torch.npu.Event() for _ in range(groups)]
    computed = [torch.npu.Event() for _ in range(groups)]
    eager = _split_forward(query, key, value, dist.group.WORLD, args.seq, groups, comm, forward_done, computed)
    torch.npu.synchronize()

    drift = (eager.float() - reference.float()).abs().max().item()
    if rank == 0:
        print(f"world={world} heads={args.heads} groups={groups} valid_splits={valid}")
        print(f"[1] split-vs-unsplit max abs diff = {drift:.3e}")

    if args.skip_capture:
        dist.destroy_process_group()
        return 0

    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    try:
        torch.npu.synchronize()
        with torch.npu.graph(graph, pool=pool):
            captured = _split_forward(
                query, key, value, dist.group.WORLD, args.seq, groups, comm, forward_done, computed
            )
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001 - the verdict is the point of the probe
        if rank == 0:
            print(f"[2] NPUGraph capture FAILED: {type(exc).__name__}: {exc}")
            print("    -> the head-overlap region is not capturable as written.")
        dist.destroy_process_group()
        return 1

    if rank == 0:
        print("[2] NPUGraph capture OK")

    graph.replay()
    torch.npu.synchronize()
    replay_drift = (captured.float() - eager.float()).abs().max().item()
    if rank == 0:
        print(f"[3] replay-vs-eager max abs diff = {replay_drift:.3e}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    sys.exit(main())
