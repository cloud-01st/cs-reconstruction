# -*- coding: utf-8 -*-
"""正交匹配追踪（OMP）批处理实现 —— 传统重建算法对照组之一。

在 DCT 稀疏域上逐块重建：
    求解  min ||s||_0  s.t.  A s = y,   A = Phi @ Psi.T  （字典，M×N）
    重建  x_hat = Psi.T @ s_hat

批量实现：所有块并行推进（每轮为每块各选一个最大相关原子），
在小批量上可直接 GPU 加速。
"""
from __future__ import annotations

import torch

from .sampling import BLOCK_DIM, BlockSampler, blocks_to_image
from .sparse import from_sparse, get_dct_basis


def omp_batch(A: torch.Tensor, y: torch.Tensor, sparsity: int) -> torch.Tensor:
    """批量 OMP 求解 y = A s。

    参数
    ----
    A: (M, N) 字典（列称为原子）
    y: (P, M) P 个测量向量
    sparsity: 迭代次数（允许的最大非零原子数）

    返回
    ----
    s: (P, N) 稀疏系数
    """
    assert A.dim() == 2 and y.dim() == 2
    P, M = y.shape
    N = A.shape[1]
    device, dtype = y.device, y.dtype
    at = A.t()                                   # (N, M)

    residual = y.clone()                         # (P, M)
    idx_all: torch.Tensor | None = None          # (P, k) 每块的支撑集
    sol: torch.Tensor | None = None              # (P, k, 1)

    for k in range(sparsity):
        # 各原子与残差的相关性绝对值（注意先取 abs 再屏蔽：
        # 若对原始 corr 置 -inf，abs(-inf)=+inf 会让已选原子反而成为最大值）
        corr = (residual @ A).abs()              # (P, N)
        if idx_all is not None:
            # 已入选原子置 -inf（abs 域最小值），避免重复选取
            corr.scatter_(1, idx_all, float("-inf"))
        idx = corr.argmax(dim=1, keepdim=True)                # (P, 1)
        idx_all = idx if idx_all is None else torch.cat([idx_all, idx], dim=1)

        a_sel = at[idx_all]                      # (P, k+1, M)
        try:
            sol = torch.linalg.lstsq(a_sel.transpose(1, 2), y.unsqueeze(-1)).solution
        except torch._C._LinAlgError:            # 数值退化时退回最小范数解
            sol = torch.linalg.pinv(a_sel.transpose(1, 2)) @ y.unsqueeze(-1)
        approx = (a_sel * sol).sum(dim=1)                    # (P,k+1,M)*(P,k+1,1)→(P,M)
        residual = y - approx

    s = torch.zeros(P, N, dtype=dtype, device=device)
    s.scatter_(1, idx_all, sol.squeeze(-1))
    return s


def omp_reconstruct(
    img: torch.Tensor,
    rate: float,
    sparsity: int | None = None,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """端到端：整图 → 采样 → OMP 重建 → 整图。

    img: (B,1,H,W)，返回同形状重建图。
    sparsity 默认取 max(15, M//8)：DCT 系数能量集中，稀疏度过大反而
    引入拟合误差（实测 Set11 上 M//8 附近为最优点）。
    """
    sampler = BlockSampler(rate, seed=seed, device=device)
    img = img.to(device or sampler.phi.device)
    y = sampler(img)                                          # (B,P,M)
    b, p, m = y.shape
    y2 = y.reshape(b * p, m)

    sparsity = sparsity or max(15, sampler.m // 8)
    psi = get_dct_basis(device=y2.device, dtype=y2.dtype)
    a = sampler.phi @ psi.t()                                 # (M, N) 字典
    s = omp_batch(a, y2, sparsity)                            # (B*P, N)
    x_hat = from_sparse(s, psi).reshape(b, p, BLOCK_DIM)
    _, _, h, w = img.shape
    return blocks_to_image(x_hat, h, w)


if __name__ == "__main__":
    # 自检：随机块上验证 OMP 能恢复稀疏信号
    torch.manual_seed(0)
    n, m, p = 256, 64, 4
    a = torch.randn(m, n)
    a = a / a.norm(dim=1, keepdim=True)
    s_true = torch.zeros(p, n)
    for i in range(p):
        pos = torch.randperm(n)[:8]
        s_true[i, pos] = torch.randn(8)
    y = s_true @ a.t()
    s_hat = omp_batch(a, y, sparsity=8)
    err = (s_hat - s_true).abs().max().item()
    print(f"OMP 恢复误差（已知稀疏度）: {err:.2e}")
