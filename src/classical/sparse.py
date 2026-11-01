# -*- coding: utf-8 -*-
"""DCT 稀疏基：33x33 图像块的正交变换。

约定
----
Psi: (N, N) 正交矩阵，N = 1089
    正变换 s = Psi @ x   （图像块 → DCT 系数，能量集中于低频，近似稀疏）
    逆变换 x = Psi.T @ s （DCT 系数 → 图像块）

2D-DCT 与矩阵形式的对应关系（推导）：
    X_dct = D X D^T
    vec(D X D^T) = (D ⊗ D) vec(X)     ← 利用 vec(AXB) = (B^T ⊗ A) vec(X)
    因此 Psi = D ⊗ D（Kronecker 积），D 为 33 点正交 DCT-II 矩阵
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.fft import dct

from .sampling import BLOCK

_BASIS_CACHE: dict = {}


def get_dct_basis(n: int = BLOCK, device=None, dtype=torch.float32) -> torch.Tensor:
    """返回 (n*n, n*n) 的 2D-DCT 正交基矩阵 Psi = D ⊗ D（带缓存）。"""
    key = (n, str(device), str(dtype))
    if key not in _BASIS_CACHE:
        d = dct(np.eye(n), axis=0, norm="ortho")      # 列向量为 1D DCT 基
        psi = np.kron(d, d)                            # (n^2, n^2)
        _BASIS_CACHE[key] = torch.as_tensor(psi, dtype=dtype, device=device)
    return _BASIS_CACHE[key]


def to_sparse(blocks: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
    """图像块 → 稀疏系数。(B,P,N) @ (N,N)^T → (B,P,N)"""
    psi = basis if basis is not None else get_dct_basis(device=blocks.device, dtype=blocks.dtype)
    return blocks @ psi.t()


def from_sparse(coeffs: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
    """稀疏系数 → 图像块。(B,P,N) @ (N,N) → (B,P,N)"""
    psi = basis if basis is not None else get_dct_basis(device=coeffs.device, dtype=coeffs.dtype)
    return coeffs @ psi


if __name__ == "__main__":
    # 自检：往返一致性 + 正交性
    psi = get_dct_basis()
    eye_err = (psi.t() @ psi - torch.eye(psi.shape[0])).abs().max().item()
    print(f"正交性误差: {eye_err:.2e}")

    x = torch.rand(2, 3, 1089)
    s = to_sparse(x)
    back = from_sparse(s)
    print(f"正逆变换往返误差: {(back - x).abs().max().item():.2e}")

    # 稀疏性验证：常数块的 DCT 系数应只有直流分量
    # （float32 下非直流分量存在 ~1e-6 量级浮点残差，属正常精度极限）
    const_block = torch.ones(1, 1, 1089)
    cs = to_sparse(const_block)
    nz = (cs.abs() > 1e-4).sum().item()
    print(f"常数块的显著 DCT 系数个数: {nz}（应为 1，即直流分量）")
    print(f"最大非直流系数绝对值: {cs[..., 1:].abs().max().item():.2e}")
