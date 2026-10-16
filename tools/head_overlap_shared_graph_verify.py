#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Verify the premise behind sharing ONE head-overlap graph across every layer.

``HeadOverlapGraphCache`` today keeps a graph per attention layer, and that is
what ran the A3 box out of HBM: a captured region retains its static inputs,
its outputs and every intermediate allocated inside it, and those were
full-packed-sequence sized (348 MiB per Q/K/V). One graph per layer times 50
layers is ~175 GiB against ~20 GiB of headroom.

Sharing one graph is only sound if the region is layer-independent. That claim
rests on two facts that do not need a model to check:

* **the region holds no weights.**  It runs after the projections: head-scatter
  all-to-all -> sparse attention -> reverse all-to-all.  The RainFusion path
  reads only shape-derived Python scalars from the metadata
  (``used_len``/``video_spans``/``latent_shape``/``max_seqlen_q``), never a
  tensor and never a parameter, so nothing in the captured op sequence is bound
  to a particular block.
* **every block gets the same shapes and the same metadata values.**  The DiT
  block loop hands the identical ``cu_seqlens``/``max_seqlen``/``num_requests``
  to all blocks, and the deployment sets no ``skip_layers``, so all blocks are
  in the same dense/sparse phase.

What that leaves to check is the *mechanism*: a graph captured from one set of
data must produce the right answer when a different set is copied into its
static inputs. This script checks exactly that, and nothing else.

Run inside the NPU container, eight processes, no torchrun (the container has no
``/etc/hosts`` entry for its own hostname, so c10d rendezvous hangs):

    for i in 0 1 2 3 4 5 6 7; do
      RANK=$i WORLD_SIZE=8 LOCAL_RANK=$i MASTER_ADDR=127.0.0.1 MASTER_PORT=29531 \\
        PYTHONUNBUFFERED=1 python3 tools/head_overlap_shared_graph_verify.py > r$i.log 2>&1 &
    done; wait

Expected: every data set reports 0.000e+00.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npugraph_head_overlap_probe import (  # noqa: E402
    _attention_stand_in,
    _ulysses_a2a,
    _ulysses_a2a_reverse,
    _valid_group_counts,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--seq", type=int, default=4096, help="local sequence length per rank")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--groups", type=int, default=0)
    parser.add_argument("--datasets", type=int, default=4, help="how many distinct input sets to replay through")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def _region(query, key, value, pg, s_local, groups, comm, forward_done, computed):
    """The ping-pong, as a pure function of Q/K/V: one graph can serve any caller."""
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


def _eager(query, key, value, pg, s_local, groups):
    comm = torch.npu.Stream()
    forward_done = [torch.npu.Event() for _ in range(groups)]
    computed = [torch.npu.Event() for _ in range(groups)]
    return _region(query, key, value, pg, s_local, groups, comm, forward_done, computed)


def main() -> int:
    args = _parse_args()
    if not hasattr(torch, "npu"):
        print("[SKIP] torch.npu unavailable; run inside the NPU container.")
        return 0

    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.npu.set_device(rank)
    dtype = getattr(torch, args.dtype)
    pg = dist.group.WORLD

    valid = _valid_group_counts(args.heads, world)
    groups = args.groups or (valid[-1] if valid else 0)
    if groups < 2:
        if rank == 0:
            print(f"[SKIP] heads={args.heads} world={world} admits no head split (valid={valid})")
        dist.destroy_process_group()
        return 0

    shape = (1, args.seq, args.heads, args.dim)
    torch.manual_seed(99 + rank)
    datasets = []
    for index in range(args.datasets):
        gen = torch.Generator(device="cpu").manual_seed(1000 + index)
        datasets.append(
            tuple(
                torch.randn(shape, dtype=torch.float32, generator=gen).to(dtype=dtype, device="npu") for _ in range(3)
            )
        )

    eager = [_eager(q, k, v, pg, args.seq, groups) for (q, k, v) in datasets]
    torch.npu.synchronize()

    # Capture once, from the first data set, on buffers copied from it.
    comm = torch.npu.Stream()
    forward_done = [torch.npu.Event() for _ in range(groups)]
    computed = [torch.npu.Event() for _ in range(groups)]
    static_in = [value.detach().clone() for value in datasets[0]]
    for static, current in zip(static_in, datasets[0], strict=True):
        static.copy_(current)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    pool = torch.npu.graph_pool_handle()
    try:
        with torch.inference_mode(), torch.npu.graph(graph, pool=pool):
            static_out = _region(
                static_in[0], static_in[1], static_in[2], pg, args.seq, groups, comm, forward_done, computed
            )
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001 - the verdict is the probe's output
        if rank == 0:
            print(f"[FAIL] capture failed: {type(exc).__name__}: {exc}")
        dist.destroy_process_group()
        return 1

    if rank == 0:
        print(f"world={world} heads={args.heads} groups={groups} shape={shape} datasets={args.datasets}")
        print("[capture] one graph captured from data set 0")

    # The point of the check: feed *different* data through the same graph.
    worst = 0.0
    for index in range(args.datasets):
        for static, current in zip(static_in, datasets[index], strict=True):
            static.copy_(current)
        graph.replay()
        torch.npu.synchronize()
        drift = (static_out.float() - eager[index].float()).abs().max().item()
        worst = max(worst, drift)
        if rank == 0:
            print(f"[replay] data set {index}: max abs diff vs eager = {drift:.3e}")

    ok = worst == 0.0
    if rank == 0:
        print(f"[verdict] {'PASS' if ok else 'FAIL'} — worst drift {worst:.3e} over {args.datasets} data sets")
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
