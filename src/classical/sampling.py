# -*- coding: utf-8 -*-
"""分块压缩感知（Block-based Compressive Sensing）采样算子。

全项目统一约定
--------------
图像张量   (B, 1, H, W)，H、W 均为 33 的倍数（预处理阶段保证）
图块张量   (B, P, N)，P = (H/33) * (W/33) 为块数，N = 33*33 = 1089
测量值     (B, P, M)，M = round(rate * N)，rate 为采样率
测量矩阵   Phi: (M, N)，高斯随机矩阵按行归一化（每行 L2 范数为 1）

数学形式
--------
    对每个 33x33 块 x ∈ R^1089，测量 y = Phi x ∈ R^M
    采样率 = M / 1089

说明：行归一化（每行单位范数）保证测量值 y 各分量与原始信号处于同一
量级，这是 ReconNet 等文献的常用做法，便于网络训练时数值稳定。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

BLOCK = 33
BLOCK_DIM = BLOCK * BLOCK  # 1089


def make_measurement_matrix(m: int, n: int = BLOCK_DIM, seed: int = 0) -> torch.Tensor:
    """生成行归一化的高斯随机测量矩阵，形状 (M, N)。"""
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(m, n, generator=g)
    phi = phi / phi.norm(dim=1, keepdim=True)
    return phi


def rate_to_m(rate: float, n: int = BLOCK_DIM) -> int:
    """采样率换算为测量维数 M。例如 rate=0.25 → M=272。"""
    return int(round(rate * n))


def image_to_blocks(img: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """整图 → 不重叠图块序列。(B,1,H,W) → (B, P, block*block)"""
    b, c, h, w = img.shape
    assert c == 1, "本项目处理灰度图像"
    assert h % block == 0 and w % block == 0, f"H,W 必须是 {block} 的倍数"
    hb, wb = h // block, w // block
    x = img[:, 0]                                            # (B,H,W)
    x = x.view(b, hb, block, wb, block)
    x = x.permute(0, 1, 3, 2, 4).contiguous()                # (B,hb,wb,blk,blk)
    return x.view(b, hb * wb, block * block)


def blocks_to_image(blocks: torch.Tensor, h: int, w: int, block: int = BLOCK) -> torch.Tensor:
    """图块序列 → 整图。(B, P, block*block) → (B,1,H,W)"""
    b, p, n = blocks.shape
    assert n == block * block
    hb, wb = h // block, w // block
    assert hb * wb == p, f"块数不匹配：{p} != {hb}*{wb}"
    x = blocks.view(b, hb, wb, block, block)
    x = x.permute(0, 1, 3, 2, 4).contiguous()                # (B,hb,blk,wb,blk)
    return x.view(b, 1, h, w)


class BlockSampler(nn.Module):
    """分块压缩采样器：整图 → 逐块线性测量。

    固定 Φ（同一数据集用同一个矩阵，种子可复现），保证训练、测试与
    传统算法对比时使用完全相同的采样条件。
    """

    def __init__(self, rate: float, seed: int = 0, device=None, dtype=torch.float32):
        super().__init__()
        self.rate = rate
        self.m = rate_to_m(rate)
        phi = make_measurement_matrix(self.m, BLOCK_DIM, seed)
        self.register_buffer("phi", phi.to(device=device, dtype=dtype))

    @property
    def phi_transpose(self) -> torch.Tensor:
        """Φ^T，形状 (N, M)，供稀疏域字典构造使用。"""
        return self.phi.t()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) → (B, P, M)"""
        blocks = image_to_blocks(img)                        # (B,P,N)
        return blocks @ self.phi.t()                         # (B,P,M)

    def extra_repr(self) -> str:
        return f"rate={self.rate}, M={self.m}, N={BLOCK_DIM}"


if __name__ == "__main__":
    # 自检：分块 → 采样的维度与数值合理性
    torch.manual_seed(0)
    img = torch.rand(2, 1, 66, 99)          # 2x3 = 6 块
    for r in (0.10, 0.25, 0.50):
        sampler = BlockSampler(r)
        y = sampler(img)
        assert y.shape == (2, 6, sampler.m), y.shape
        print(f"rate={r:.2f}  M={sampler.m:3d}  测量值形状={tuple(y.shape)}  "
              f"y 标准差={y.std():.3f}  x 标准差={img.std():.3f}")
    # 分块-还原往返一致性
    blocks = image_to_blocks(img)
    back = blocks_to_image(blocks, 66, 99)
    print("分块往返误差:", (back - img).abs().max().item())
