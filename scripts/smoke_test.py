# -*- coding: utf-8 -*-
"""阶段 0 冒烟测试：验证「环境 → 采样 → 重建 → 指标」全链路可跑通。

用法
----
    python scripts/smoke_test.py          # 环境检查 + 合成图重建
    python scripts/smoke_test.py --real   # 额外用 Set11 真实图像验证（需先下载数据）

本脚本不训练任何模型，只验证基础设施。全部通过即代表阶段 0 完成。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

SEP = "=" * 62


def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def check_env() -> str:
    section("1. 环境检查")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"NumPy       : {np.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA 可用   : {cuda}")
    device = "cpu"
    if cuda:
        device = "cuda"
        props = torch.cuda.get_device_properties(0)
        print(f"GPU         : {props.name}")
        print(f"显存        : {props.total_memory / 1024**3:.1f} GB")
        print(f"CUDA 版本   : {torch.version.cuda}")
        print(f"cuDNN       : {torch.backends.cudnn.version()}")
    else:
        print("!! 未检测到可用 GPU —— 训练前必须解决！")
    return device


def check_tensor(device: str) -> None:
    section("2. 33x33 张量运算（阶段 0 验收标准）")
    x = torch.randn(1, 1, 33, 33, device=device)
    weight = torch.randn(8, 1, 3, 3, device=device) * 0.1
    out = torch.nn.functional.conv2d(x, weight, padding=1)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"输入 {tuple(x.shape)} → 卷积输出 {tuple(out.shape)}  [{device}]")
    # 矩阵乘 + 自动求导快速验证
    a = torch.randn(256, 1089, device=device, requires_grad=True)
    b = torch.randn(1089, 272, device=device)
    c = (a @ b).sum()
    c.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"矩阵乘 (256x1089)@(1089x272) + 反向传播 : OK  [{device}]")


def make_synthetic() -> torch.Tensor:
    """合成测试图 264x264（33 的 8 倍），含平滑块与纹理块。"""
    x = np.zeros((264, 264), dtype=np.float32)
    yy, xx = np.mgrid[0:264, 0:264]
    x += 0.5 + 0.3 * np.sin(xx / 9.0) * np.cos(yy / 11.0)
    x[:, 132:] += 0.2 * np.sin(xx[:, 132:] * 0.9)     # 高频纹理区
    x = np.clip(x, 0, 1)
    return torch.from_numpy(x)[None, None]


def run_reconstruction(img: torch.Tensor, name: str, device: str, quick: bool = False) -> None:
    from src.classical.ista import ista_reconstruct
    from src.classical.omp import omp_reconstruct
    from src.metrics import evaluate

    print(f"\n--- 重建「{name}」({img.shape[2]}x{img.shape[3]}) ---")
    rates = [0.25] if quick else [0.10, 0.25, 0.50]
    for rate in rates:
        # OMP
        t0 = time.time()
        recon = omp_reconstruct(img, rate=rate, device=device)
        t1 = time.time()
        m = evaluate(recon.cpu(), img)
        print(f"  OMP  rate={rate:.2f}  PSNR={m['psnr']:6.2f} dB  "
              f"SSIM={m['ssim']:.4f}  用时 {t1-t0:.1f}s")

        # ISTA（合成图上迭代少一些即可验证链路）
        t0 = time.time()
        recon = ista_reconstruct(img, rate=rate, lam=0.01,
                                 iters=100 if quick else 300, device=device)
        t1 = time.time()
        m = evaluate(recon.cpu(), img)
        print(f"  ISTA rate={rate:.2f}  PSNR={m['psnr']:6.2f} dB  "
              f"SSIM={m['ssim']:.4f}  用时 {t1-t0:.1f}s")


def run_real(set11_dir: Path, device: str) -> None:
    section("4. Set11 真实图像验证")
    from src.classical.ista import ista_reconstruct
    from src.classical.omp import omp_reconstruct
    from src.data.dataset import load_set11
    from src.data.prepare import pad_to_multiple
    from src.metrics import evaluate

    items = load_set11(set11_dir, size=256)
    print(f"加载 {len(items)} 张测试图：{', '.join(n for n, _ in items)}")
    name, img_np = items[0]                          # 用第一张验证
    img_pad = pad_to_multiple(img_np)                # 256 → 264
    img = torch.from_numpy(img_pad)[None, None]
    ph, pw = img.shape[2] - 256, img.shape[3] - 256  # 记录 pad 尺寸

    for rate in (0.10, 0.25):
        recon = omp_reconstruct(img, rate=rate, device=device)
        recon = recon[:, :, :256, :256]              # 裁回原尺寸再算指标
        m = evaluate(recon.cpu(), torch.from_numpy(img_np)[None, None])
        print(f"  OMP  rate={rate:.2f}  PSNR={m['psnr']:6.2f} dB  SSIM={m['ssim']:.4f}  "
              f"（{name}, 256x256 口径, pad {ph}x{pw}）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="用 Set11 真实图像验证")
    ap.add_argument("--quick", action="store_true", help="快速模式（单采样率、少迭代）")
    args = ap.parse_args()

    print(SEP)
    print("阶段 0 冒烟测试 —— 基于深度学习的图像压缩感知重建")
    print(SEP)

    device = check_env()
    check_tensor(device)

    section("3. 采样-重建链路（合成图）")
    run_reconstruction(make_synthetic(), "合成图", device, quick=args.quick)

    set11_dir = ROOT / "data" / "Set11"
    if args.real or set11_dir.exists():
        if set11_dir.exists():
            run_real(set11_dir, device)
        else:
            print(f"\n[跳过] 未找到 Set11：{set11_dir}")
            print("       运行 python scripts/download_data.py --set11 后重试 --real")

    section("冒烟测试完成")
    print("若上方各项均无报错 —— 环境与代码链路已就绪，阶段 0 完成 ✓")
    print("下一步：python scripts/download_data.py --all 准备数据，"
          "随后进入阶段 1（基线复现）")


if __name__ == "__main__":
    main()
