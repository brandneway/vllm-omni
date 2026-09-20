#!/usr/bin/env python3
"""对照测试：验证 NPU piecewise 注意力路由的数值正确性。

causal 段与 full 段都走 torch_npu.npu_fusion_attention：causal 段用
sparse_mode=2/3 + 2048x2048 压缩因果 mask，full 段用 sparse_mode=0 且
atten_mask=None（无 mask，全局双向）。

两种运行模式：

1. 本地假机模式（默认）：无 NPU 环境，注入 fake torch_npu，用纯 PyTorch 严格
   实现该算子的 sparse_mode 语义（0=全注意力 / 2=leftUpCausal /
   3=rightDownCausal，mask 约定 True=丢弃），校验路由选择与数值。
2. 真机模式（TEST_REAL_NPU=1）：在 NPU 上用真实 torch_npu 算子，与稠密 4D
   block-causal mask 参考对比。

通过标准：所有数值 allclose。
"""

import os
import sys
import types
from unittest.mock import MagicMock

import torch

torch.manual_seed(0)

NEG_INF = float("-inf")
USE_REAL_NPU = os.environ.get("TEST_REAL_NPU") == "1"


# ---- fake torch_npu 算子（仅本地假机模式使用） ----


def _torch_ref_attn(query, key, value, scale, keep):
    """(B, Sq, H, D) BSND 参考实现，keep 为 (Sq, Skv) bool。"""
    q = query.transpose(1, 2).float()
    k = key.transpose(1, 2).float()
    v = value.transpose(1, 2).float()
    scores = q @ k.transpose(-1, -2) * scale
    scores = scores.masked_fill(~keep.to(scores.device), NEG_INF)
    return (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).to(query.dtype)


def _sparse_keep(Sq, Skv, sparse_mode):
    qi = torch.arange(Sq).view(-1, 1)
    ki = torch.arange(Skv).view(1, -1)
    if sparse_mode == 0:
        return torch.ones(Sq, Skv, dtype=torch.bool)
    if sparse_mode == 2:  # leftUpCausal：左上角对齐
        return ki <= qi
    if sparse_mode == 3:  # rightDownCausal：右下角对齐
        return ki <= qi + (Skv - Sq)
    raise AssertionError(f"fake 算子未实现 sparse_mode={sparse_mode}")


def _check_compressed_causal_mask(atten_mask):
    """sparse_mode 2/3 的固定 2048x2048 压缩掩码图案（True = masked out）。"""
    assert atten_mask.shape == (2048, 2048), f"压缩 mask 形状错误: {atten_mask.shape}"
    assert atten_mask.dtype == torch.bool
    expected = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1)
    assert torch.equal(atten_mask.cpu(), expected), "causal mask 图案不符合 Ascend 约定"


def fake_npu_fusion_attention(
    query, key, value, head_num, input_layout, pse=None, padding_mask=None,
    atten_mask=None, scale=1.0, keep_prob=1.0, pre_tockens=2147483647,
    next_tockens=2147483647, sparse_mode=0, **kwargs,
):
    """纯 PyTorch 实现 npu_fusion_attention 语义（本测试用到的子集）。"""
    assert input_layout == "BSND"
    assert query.shape[2] == head_num and key.shape[2] == head_num  # MHA
    assert keep_prob == 1.0
    if sparse_mode in (2, 3):
        assert atten_mask is not None, "sparse_mode 2/3 必须传 atten_mask"
        _check_compressed_causal_mask(atten_mask)
    else:
        assert atten_mask is None, "sparse_mode=0 不应构造 mask"
    keep = _sparse_keep(query.shape[1], key.shape[1], sparse_mode)
    out = _torch_ref_attn(query, key, value, scale, keep)
    # 真实接口返回 (out, softmax_max, softmax_sum, softmax_out, seed, offset, numels)
    return out, torch.zeros(1), torch.zeros(1), torch.zeros(1), 0, 0, 0


# ---- 导入目标模块 ----

if USE_REAL_NPU:
    import torch_npu  # noqa: F401

    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
    from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import piecewise_attn
    DEVICE = "npu:0"
else:
    fake_torch_npu = types.ModuleType("torch_npu")
    fake_torch_npu.npu_fusion_attention = fake_npu_fusion_attention
    sys.modules["torch_npu"] = fake_torch_npu

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "0.10.0"
    fake_vllm.__version_tuple__ = (0, 10, 0)
    fake_vllm_logger = types.ModuleType("vllm.logger")
    fake_vllm_logger.init_logger = lambda name: MagicMock()
    fake_vllm.logger = fake_vllm_logger
    sys.modules["vllm"] = fake_vllm
    sys.modules["vllm.logger"] = fake_vllm_logger

    # 用带 __path__ 的空壳父包替换 vllm_omni 包树，跳过其重量级 __init__；
    # 目标子模块仍从真实源码加载。
    REPO = os.path.dirname(os.path.abspath(__file__))

    def _stub_pkg(name, rel_path):
        mod = types.ModuleType(name)
        mod.__path__ = [os.path.join(REPO, rel_path)]
        sys.modules[name] = mod
        return mod

    _stub_pkg("vllm_omni", "vllm_omni")
    _stub_pkg("vllm_omni.diffusion", "vllm_omni/diffusion")
    _stub_pkg("vllm_omni.diffusion.attention", "vllm_omni/diffusion/attention")
    _stub_pkg("vllm_omni.diffusion.attention.backends", "vllm_omni/diffusion/attention/backends")
    _stub_pkg("vllm_omni.diffusion.attention.backends.utils", "vllm_omni/diffusion/attention/backends/utils")

    fake_platforms = types.ModuleType("vllm_omni.platforms")
    fake_platforms.current_omni_platform = MagicMock()
    sys.modules["vllm_omni.platforms"] = fake_platforms

    fake_config = types.ModuleType("vllm_omni.diffusion.config")
    fake_config.get_current_diffusion_config_or_none = lambda: None
    sys.modules["vllm_omni.diffusion.config"] = fake_config

    sys.path.insert(0, ".")
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
    from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import piecewise_attn
    DEVICE = "cpu"

DTYPE = torch.bfloat16
# NPU 上 bf16 算子误差大于 fake fp32 参考，容差放宽
ATOL = 5e-2 if USE_REAL_NPU else 2e-2


def make_impl():
    impl = FlashAttentionImpl.__new__(FlashAttentionImpl)
    impl.num_heads = 2
    impl.causal = False
    impl.softmax_scale = 0.5
    impl.qkv_layout = None
    impl.is_cross_attn = False
    impl.fa_deterministic = False
    impl._npu_causal_masks = {}
    return impl


def dense_block_causal_ref(q, k, v, image_ids, scale):
    """稠密参考：(q_idx >= kv_idx) or same_image_block。"""
    qf, kf, vf = (x.transpose(1, 2).float() for x in (q, k, v))
    scores = qf @ kf.transpose(-1, -2) * scale
    S = q.shape[1]
    qi = torch.arange(S, device=q.device).view(-1, 1)
    ki = torch.arange(S, device=q.device).view(1, -1)
    same_block = (image_ids.view(-1, 1) == image_ids.view(1, -1)) & (image_ids.view(-1, 1) >= 0)
    keep = (qi >= ki) | same_block
    scores = scores.masked_fill(~keep, NEG_INF)
    return (torch.softmax(scores, dim=-1) @ vf).transpose(1, 2).to(q.dtype)


def causal_ref(q, kv, scale):
    """bottom-right 因果参考：Q[:, i] 可见 K[:, :Skv-Sq+i+1]。"""
    qf, kf = q.float().transpose(1, 2), kv.float().transpose(1, 2)
    scores = qf @ kf.transpose(-1, -2) * scale
    Sq, Skv = q.shape[1], kv.shape[1]
    keep = torch.tril(torch.ones(Sq, Skv, dtype=torch.bool, device=q.device), diagonal=Skv - Sq)
    return (torch.softmax(scores.masked_fill(~keep, NEG_INF), -1) @ kf).transpose(1, 2)


def test_causal_segment_sparse_mode_selection():
    """causal 段：Sq==Skv → sparse_mode=2；Sq<Skv → 3；数值对齐 bottom-right 因果。"""
    impl = make_impl()
    B, H, D = 1, 2, 8

    # Sq == Skv（首段 text）：leftUpCausal
    q = torch.randn(B, 5, H, D, dtype=DTYPE, device=DEVICE)
    out = impl._npu_piecewise_attn_func(q, q, q, causal=True, softmax_scale=0.5)
    ref = causal_ref(q, q, 0.5)
    assert torch.allclose(out.float(), ref, atol=ATOL), "Sq==Skv causal 数值不符"

    # Sq < Skv（中段 text，Q[:, 3:5] 对 K[:, :5]）：bottom-right 因果
    kv = torch.randn(B, 5, H, D, dtype=DTYPE, device=DEVICE)
    qq = torch.randn(B, 2, H, D, dtype=DTYPE, device=DEVICE)
    out = impl._npu_piecewise_attn_func(qq, kv, kv, causal=True, softmax_scale=0.5)
    ref = causal_ref(qq, kv, 0.5)
    assert torch.allclose(out.float(), ref, atol=ATOL), "Sq<Skv bottom-right causal 数值不符"
    print("✅ test_causal_segment_sparse_mode_selection 通过")


def test_full_segment_no_mask():
    """非 causal 段（图像块）：sparse_mode=0 无 mask，等价于截断 KV 的全注意力。"""
    impl = make_impl()
    B, H, D = 1, 2, 8
    kv = torch.randn(B, 6, H, D, dtype=DTYPE, device=DEVICE)
    q = torch.randn(B, 3, H, D, dtype=DTYPE, device=DEVICE)
    out = impl._npu_piecewise_attn_func(q, kv, kv, causal=False, softmax_scale=0.5)
    qf = q.float().transpose(1, 2)
    kf = kv.float().transpose(1, 2)
    ref = (torch.softmax(qf @ kf.transpose(-1, -2) * 0.5, -1) @ kf).transpose(1, 2)
    assert torch.allclose(out.float(), ref, atol=ATOL)
    print("✅ test_full_segment_no_mask 通过")


def test_piecewise_matches_dense_block_causal():
    """完整分段：模拟 [text*3 | 图A*2 | text*2 | 图B*2 | 目标图*3]，对比稠密 block-causal。"""
    impl = make_impl()
    B, H, D = 1, 2, 8
    # 布局：text(0-2), 图A(3-4), text(5-6), 图B(7-8), 目标图(9-11)
    image_ids = torch.tensor([-1, -1, -1, 0, 0, -1, -1, 1, 1, 2, 2, 2], device=DEVICE)
    spans = [(3, 5), (7, 9), (9, 12)]
    S = image_ids.shape[0]
    q = torch.randn(B, S, H, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(B, S, H, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B, S, H, D, dtype=DTYPE, device=DEVICE)

    out = piecewise_attn(q, k, v, [spans], 0.5, impl._npu_piecewise_attn_func)
    ref = dense_block_causal_ref(q, k, v, image_ids, 0.5)
    assert out.shape == q.shape
    assert torch.allclose(out.float(), ref.float(), atol=ATOL), \
        f"piecewise 与稠密参考不符，max diff={(out.float() - ref.float()).abs().max()}"
    print("✅ test_piecewise_matches_dense_block_causal 通过")


if __name__ == "__main__":
    mode = "真机 NPU" if USE_REAL_NPU else "本地 fake torch_npu"
    print(f"运行模式：{mode} (device={DEVICE})")
    test_causal_segment_sparse_mode_selection()
    test_full_segment_no_mask()
    test_piecewise_matches_dense_block_causal()
    print("\n全部通过：NPU piecewise 路由与稠密 block-causal 参考数值一致")
