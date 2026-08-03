# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Geometry contract RAINFUSION_ATTN enforces before handing a forward to rf_v2.

The grids here are the ones MiniMax-H3 FL2VA actually produces for an 8.7s clip:
1344x768 gives a 62x24x42 latent grid whose 62496 video rows straddle the 128-row
block, and 1280x768 gives 62x24x40 whose 59520 rows land on it. Neither is
rejected -- rf_v2 only breaks when its mask and the kernel's tiling disagree on
the block count, which the resolver repairs by padding the prefix.
"""

import pytest

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.rainfusion_attn import (
    _BLOCK_SIZE,
    RainFusionAttentionImpl,
    _prefix_pad_rows,
)

PREFIX_ROWS = 710  # 14 text rows + 696 audio rows
ALIGNED_GRID = (62, 24, 40)  # 1280x768 -> 59520 video rows, 465 blocks
MISALIGNED_GRID = (62, 24, 42)  # 1344x768 -> 62496 video rows, 488 blocks + 32 rows


def make_impl(**backend_kwargs):
    return RainFusionAttentionImpl(
        num_heads=8,
        head_size=128,
        softmax_scale=128**-0.5,
        prefix="transformer_blocks.0.attn",
        backend_kwargs={"sparsity": 0.8, **backend_kwargs},
    )


def make_metadata(grid, prefix_len=PREFIX_ROWS, max_seqlen_q=None):
    video_rows = grid[0] * grid[1] * grid[2]
    return AttentionMetadata(
        extra={
            "max_seqlen_q": prefix_len + video_rows if max_seqlen_q is None else max_seqlen_q,
            "rainfusion_prefix_len": prefix_len,
            "rainfusion_latent_grid": grid,
        }
    )


def blocks(rows):
    return -(-rows // _BLOCK_SIZE)


def test_block_aligned_video_segment_runs_sparse_without_padding():
    plan = make_impl()._resolve_plan(make_metadata(ALIGNED_GRID))

    assert plan is not None
    assert plan.prefix_len == PREFIX_ROWS
    assert plan.used_len == PREFIX_ROWS + 59520
    assert plan.latent_shape == list(ALIGNED_GRID)
    assert plan.prefix_pad == 0


def test_misaligned_video_segment_runs_sparse_on_a_padded_prefix():
    plan = make_impl()._resolve_plan(make_metadata(MISALIGNED_GRID))

    assert plan is not None
    # 62496 video rows leave 32 over, so the kernel block at the seam holds those
    # 32 plus the next 96 rows, and pooling files that block under the video --
    # where selection may drop it. 96 rows of pad keep real prefix out of it, and
    # 155 is the first pad at or above 96 that also keeps the block counts equal.
    assert plan.prefix_pad == 155
    # The pad is invisible to the rest of the plan; it exists only inside the
    # rf_v2 call, which slices it back off.
    assert plan.used_len == PREFIX_ROWS + 62496


def test_padding_is_spent_only_on_a_misaligned_video_segment():
    # A video segment that ends on a block boundary needs nothing: the pooled
    # blocks are the kernel's tiles, and the prefix starts a fresh tile.
    for prefix_len in (1, 127, _BLOCK_SIZE, 710, 900):
        plan = make_impl()._resolve_plan(make_metadata(ALIGNED_GRID, prefix_len=prefix_len))
        assert plan is not None, prefix_len
        assert plan.prefix_pad == 0, prefix_len


@pytest.mark.parametrize("grid", [ALIGNED_GRID, MISALIGNED_GRID, (62, 27, 36), (62, 20, 50)])
def test_pad_clears_the_seam_block_and_keeps_the_block_counts_equal(grid):
    video = grid[0] * grid[1] * grid[2]
    video_rem = video % _BLOCK_SIZE

    def ok(pad):
        padded = prefix_len + pad
        counts_agree = blocks(video) + blocks(padded) == blocks(video + padded)
        # rf_v2 lays the sequence out as [video | pad | prefix], so the kernel
        # block straddling the seam covers the leftover video rows plus the next
        # _BLOCK_SIZE - video_rem. Real prefix must start past it.
        seam_clear = video_rem == 0 or pad >= _BLOCK_SIZE - video_rem
        return counts_agree and seam_clear

    for prefix_len in range(1, 2 * _BLOCK_SIZE + 1):
        pad = _prefix_pad_rows(video, prefix_len)
        assert ok(pad), (prefix_len, pad)
        assert pad < 2 * _BLOCK_SIZE, (prefix_len, pad)
        assert all(not ok(smaller) for smaller in range(pad)), (prefix_len, pad)
        if video_rem == 0:
            assert pad == 0, (prefix_len, pad)


def test_sparsity_zero_never_resolves_a_plan():
    assert make_impl(sparsity=0.0)._resolve_plan(make_metadata(ALIGNED_GRID)) is None


def test_missing_video_geometry_falls_back_to_dense():
    assert make_impl()._resolve_plan(AttentionMetadata(extra={"max_seqlen_q": 60230})) is None


def test_video_segment_must_be_the_tail_of_packed_document_zero():
    metadata = make_metadata(ALIGNED_GRID, max_seqlen_q=PREFIX_ROWS + 59520 + 128)

    assert make_impl()._resolve_plan(metadata) is None


@pytest.mark.parametrize("grid", [(4, 24, 40), (1, 24, 40)])
def test_short_video_stays_dense(grid):
    assert make_impl()._resolve_plan(make_metadata(grid)) is None
