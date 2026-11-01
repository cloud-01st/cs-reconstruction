# -*- coding: utf-8 -*-
"""ReconNet 复现（Kulkarni et al., CVPR 2016）。

论文结构（非迭代重建网络的首批代表工作）：
    全连接层:  M → 1089          （测量值映射回图像块维度，reshape 成 33×33）
    卷积 11×11: 1 → 64 通道, ReLU
    卷积 1×1:  64 → 32 通道, ReLU
    卷积 7×7:  32 → 1 通道        （线性输出）

输入: (B, M) 测量值（测量矩阵行归一化，与 classical 部分同一口径）
输出: (B, 1, 33, 33) 重建块（未裁剪，评估时 clamp 到 [0,1]）

说明：本实现不加入论文之外的组件（不加 BN / Dropout / 残差），
保证与文献可比；训练配置见 src/train.py 的默认参数。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..classical.sampling import BLOCK, BLOCK_DIM


class ReconNet(nn.Module):
    samples_internally = False  # 前向接受测量值 (B,M)，采样在外部用固定 Φ 完成

    def __init__(self, m: int, base_channels: int = 64):
        super().__init__()
        self.m = m
        self.fc = nn.Linear(m, BLOCK_DIM)
        self.conv = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=11, padding=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=7, padding=3),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """(B, M) → (B, 1, 33, 33)"""
        x = self.fc(y).view(-1, 1, BLOCK, BLOCK)
        return self.conv(x)

    def extra_repr(self) -> str:
        return f"M={self.m}"


if __name__ == "__main__":
    # 自检：输出形状 + 参数量
    for rate in (0.10, 0.25, 0.50):
        m = int(round(rate * BLOCK_DIM))
        net = ReconNet(m)
        y = torch.randn(4, m)
        out = net(y)
        n_param = sum(p.numel() for p in net.parameters())
        print(f"rate={rate:.2f}  M={m:3d}  输出={tuple(out.shape)}  参数量={n_param/1e3:.1f}K")
