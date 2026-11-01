# -*- coding: utf-8 -*-
"""自研重建网络：三级级联结构 + 多尺度残差 + 通道注意力。

结构总览（三级级联，端到端联合训练）
--------------------------------------
    ┌─ 第一级  采样网络：Conv2d(1, M, kernel_size=33)
    │    每个 33×33 卷积核即一个"可学习测量模式"，输出 M 个测量值。
    │    初始化为高斯随机（与经典 CS 的 Φ 同口径），再通过训练自适应优化。
    │    等价于学习一个测量矩阵 Φ_learn (M×1089)，但以卷积形式实现。
    │
    ├─ 第二级  初始重建网络：Linear(M, 1089) → reshape 33×33
    │    将测量值线性映射回图像块空间，作为深度重建的初值。
    │
    └─ 第三级  深度重建网络：
          Conv(1→base) → N × 多尺度残差块（每 2 块插入通道注意力）→ Conv(base→1)
          以残差方式输出修正量：x_hat = x_init + correction

设计要点（论文与答辩的说明口径）
--------------------------------
- **可学习采样**：与固定随机 Φ 相比，采样端可感知重建任务需求，
  与重建网络联合优化（代价：削弱了 RIP 理论保证，需在论文中讨论）。
- **多尺度残差**：三个不同扩张率（1/2/3）的并行分支，在不增加参数量的
  前提下覆盖不同感受野——小感受野修细节（块效应），大感受野修结构。
- **通道注意力**：SE 风格，让网络自适应加权不同特征通道。
- **残差学习**：网络只学"对初始重建的修正量"，与重建任务本质一致。

接口约定：`samples_internally = True`
    前向直接接受图像块 (B,1,33,33)，内部完成采样→重建全流程；
    与 ReconNet（外部固定 Φ 采样）通过类属性区分，训练/评估脚本据此分支。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..classical.sampling import BLOCK, BLOCK_DIM, make_measurement_matrix


class MultiScaleResBlock(nn.Module):
    """多尺度残差块：扩张率 1/2/3 的三分支并行卷积 + 1×1 融合 + 残差。"""

    def __init__(self, channels: int):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1, dilation=1),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.Conv2d(channels, channels, 3, padding=3, dilation=3),
        ])
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [self.act(b(x)) for b in self.branches]
        return x + self.fuse(torch.cat(outs, dim=1))


class PlainResBlock(nn.Module):
    """普通残差块（消融对照用）：两个 3×3 卷积分支，参数量与多尺度块同量级。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return x + out


class ChannelAttention(nn.Module):
    """通道注意力（SE 风格）：全局平均池化 → 两层 MLP → 通道加权。"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(2, 3))                       # (B, C)
        return x * self.mlp(w)[..., None, None]


class OursNet(nn.Module):
    """三级级联自研重建网络（可学习采样 + 初始重建 + 深度重建）。"""

    samples_internally = True  # 前向直接接受图像块（内部含采样）

    def __init__(self, m: int, base_channels: int = 64,
                 n_blocks: int = 8, attn_every: int = 2,
                 learnable_sampling: bool = True,
                 use_multiscale: bool = True,
                 use_attention: bool = True):
        """消融开关（默认全开 = 完整模型）：
        learnable_sampling=False → 采样端用固定随机 Φ（seed=0），仅训练重建部分
        use_multiscale=False      → 多尺度残差块替换为普通残差块
        use_attention=False       → 去掉通道注意力
        """
        super().__init__()
        self.m = m
        self.base_channels = base_channels
        self.learnable_sampling = learnable_sampling
        self.use_multiscale = use_multiscale
        self.use_attention = use_attention

        # ---- 第一级：采样网络 ----
        if learnable_sampling:
            # 33×33 卷积核 = 可学习测量模式；初始化为高斯随机（与经典 CS 同口径）
            self.sampler = nn.Conv2d(1, m, kernel_size=BLOCK, bias=True)
            nn.init.normal_(self.sampler.weight, std=1.0 / math.sqrt(BLOCK_DIM))
            nn.init.zeros_(self.sampler.bias)
        else:
            # 消融对照：固定随机测量矩阵（不可学习）
            self.sampler = None
            self.register_buffer("fixed_phi",
                                 make_measurement_matrix(m, BLOCK_DIM, seed=0))

        # ---- 第二级：初始重建 ----
        self.init_recon = nn.Linear(m, BLOCK_DIM)

        # ---- 第三级：深度重建 ----
        self.head = nn.Conv2d(1, base_channels, 3, padding=1)
        block_cls = MultiScaleResBlock if use_multiscale else PlainResBlock
        blocks: list[nn.Module] = []
        for i in range(n_blocks):
            blocks.append(block_cls(base_channels))
            if use_attention and (i + 1) % attn_every == 0:
                blocks.append(ChannelAttention(base_channels))
        self.body = nn.Sequential(*blocks)
        self.tail = nn.Conv2d(base_channels, 1, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    # ---- 接口 ----
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """测量：(B,1,33,33) → (B,M)

        可学习采样：33×33 卷积；固定采样（消融对照）：行归一化高斯 Φ。
        """
        if self.sampler is not None:
            return self.sampler(x).flatten(1)
        return x.flatten(1) @ self.fixed_phi.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """端到端重建：(B,1,33,33) → (B,1,33,33)"""
        y = self.sample(x)                                   # 第一级：采样
        x_init = self.init_recon(y).view(-1, 1, BLOCK, BLOCK)  # 第二级：初始重建
        feat = self.act(self.head(x_init))                   # 第三级：深度重建
        feat = self.body(feat)
        return x_init + self.tail(feat)                      # 残差输出

    def extra_repr(self) -> str:
        return f"M={self.m}, base={self.base_channels}"


if __name__ == "__main__":
    # 自检：形状、参数量、前向可导
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for rate in (0.10, 0.25, 0.50):
        m = int(round(rate * BLOCK_DIM))
        net = OursNet(m).to(dev)
        x = torch.rand(4, 1, BLOCK, BLOCK, device=dev)
        out = net(x)
        n_param = sum(p.numel() for p in net.parameters())
        print(f"rate={rate:.2f}  M={m:3d}  输出={tuple(out.shape)}  参数量={n_param/1e6:.2f}M")
    # 反传检查
    loss = out.mean()
    loss.backward()
    print("反向传播 OK")
