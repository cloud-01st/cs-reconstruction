# -*- coding: utf-8 -*-
"""图像重建质量评价指标：PSNR 与 SSIM。

约定（全项目统一）：
    图像张量取值范围 [0, 1]，形状 (B, 1, H, W) 或 (B, C, H, W)
    PSNR 计算式：10 * log10(MAX^2 / MSE)，MAX = 1.0（即在 [0,1] 域上计算）
    SSIM 使用 11x11 高斯窗（sigma=1.5），与 Wang 等 2004 原始论文一致

注意：不同论文报告 PSNR 时可能出现 [0,1] 域与 [0,255] 域两种口径，
两者数值一样（因为 MAX^2 / MSE 中分子分母同比例缩放），无需换算。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """峰值信噪比（dB）。pred 与 target 均假设在 [0, 1] 范围。"""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(max_val ** 2 / mse)).item()


def _gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 1) -> torch.Tensor:
    """构造二维高斯窗，形状 (C, 1, size, size)，归一化到总和为 1。"""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]
    return g2d.expand(channels, 1, size, size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """结构相似性（均值 SSIM）。pred 与 target 均假设在 [0, 1] 范围。

    实现说明：采用 valid 卷积（不做边界填充），这与 scikit-image 在
    gaussian_weights=True 时的口径一致；无偏/有偏协方差估计的差异在此
    设置下可忽略（小数第 4 位量级）。
    """
    assert pred.shape == target.shape, "pred 与 target 形状必须一致"
    c = pred.shape[1]
    window = _gaussian_window(11, 1.5, c).to(dtype=pred.dtype, device=pred.device)

    mu1 = F.conv2d(pred, window, groups=c)
    mu2 = F.conv2d(target, window, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, groups=c) - mu12

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


def evaluate(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """一次性返回 PSNR 与 SSIM（便于实验中批量调用）。"""
    return {"psnr": psnr(pred, target), "ssim": ssim(pred, target)}


if __name__ == "__main__":
    # 自检：相同图像 PSNR 应为 inf、SSIM 应为 1；加噪后应明显下降
    torch.manual_seed(0)
    x = torch.rand(1, 1, 64, 64)
    noise = torch.randn_like(x) * 0.05
    y = (x + noise).clamp(0, 1)
    print("相同图像:", evaluate(x, x))
    print("加噪图像:", {k: round(v, 4) for k, v in evaluate(x, y).items()})
