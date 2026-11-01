# -*- coding: utf-8 -*-
"""深度学习模型训练脚本（阶段 2：ReconNet；阶段 3 可复用）。

用法
----
    # 快速验证（5 epoch，看 loss 是否下降）
    python -m src.train --model reconnet --rate 0.25 --epochs 5

    # 正式训练（三档采样率，各自一个模型）
    python -m src.train --model reconnet --rate 0.10 --epochs 100
    python -m src.train --model reconnet --rate 0.25 --epochs 100
    python -m src.train --model reconnet --rate 0.50 --epochs 100

设计说明
--------
- 训练数据：BSDS500 训练块 npy（37800 块），在线采样（与测试同一测量矩阵口径：
  Phi 由 seed=0 生成、行归一化，保证训练/测试/传统算法对比公平）。
- 不使用测试集做任何模型选择；按固定 epoch 训练，保存最终模型。
- 每 batch 数据在 GPU 上完成采样与增强（数据集小，无需 DataLoader 多进程）。

产物
----
    checkpoints/<model>_r<rate>.pth        模型权重 + 配置
    results/train_<model>_r<rate>_log.csv  逐 epoch 训练日志
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.classical.sampling import BLOCK_DIM, make_measurement_matrix  # noqa: E402
from src.models import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="训练深度学习 CS 重建模型")
    ap.add_argument("--model", default="reconnet", help="模型名（reconnet / ours）")
    ap.add_argument("--rate", type=float, required=True, help="采样率（M/1089）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/processed/bsds300_train.npy")
    ap.add_argument("--init-from", default=None,
                    help="从此 checkpoint 加载权重继续训练（用于训练异常后的微调修复）")
    ap.add_argument("--amp", action="store_true", default=True, help="混合精度（默认开）")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    # ---- 消融开关（仅对 ours 模型生效）----
    ap.add_argument("--no-learnable-sampling", action="store_true",
                    help="消融：采样端改用固定随机 Φ")
    ap.add_argument("--no-multiscale", action="store_true",
                    help="消融：多尺度残差块替换为普通残差块")
    ap.add_argument("--no-attention", action="store_true",
                    help="消融：去掉通道注意力")
    return ap.parse_args()


@torch.no_grad()
def sample_blocks(x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """(B,1,33,33) 块 → (B,M) 测量值（与 BlockSampler 同一口径）。"""
    return x.flatten(1) @ phi.t()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = int(round(args.rate * BLOCK_DIM))

    # ---- 模型变体命名与结构参数（消融实验用）----
    model_kwargs: dict = {}
    variant = ""
    if args.model == "ours":
        model_kwargs = {
            "learnable_sampling": not args.no_learnable_sampling,
            "use_multiscale": not args.no_multiscale,
            "use_attention": not args.no_attention,
        }
        if args.no_learnable_sampling:
            variant += "_noLS"
        if args.no_multiscale:
            variant += "_noMS"
        if args.no_attention:
            variant += "_noAttn"
    run_name = f"{args.model}{variant}"

    print(f"[配置] 模型={run_name} 采样率={args.rate} (M={m}) "
          f"epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
          f"AMP={args.amp} 设备={device}")

    # ---- 数据（常驻内存）----
    data_np = np.load(args.data)                       # (P, 33, 33) uint8
    p_total = data_np.shape[0]
    print(f"[数据] {args.data} 共 {p_total} 块")

    # ---- 模型与优化器 ----
    model = build_model(args.model, m, **model_kwargs).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[模型] 参数量 {n_param/1e3:.1f}K")
    if args.init_from:
        _ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(_ckpt["model"])
        print(f"[续训] 已加载权重：{args.init_from}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")

    # ---- 采样矩阵（seed=0，与评估/传统方法一致）----
    phi = make_measurement_matrix(m, BLOCK_DIM, seed=0).to(device)

    # ---- 输出路径（日志防覆盖：已存在则加时间戳后缀）----
    ckpt_path = ROOT / "checkpoints" / f"{run_name}_r{int(args.rate*100):02d}.pth"
    ckpt_path.parent.mkdir(exist_ok=True)
    log_path = ROOT / "results" / f"train_{run_name}_r{int(args.rate*100):02d}_log.csv"
    log_path.parent.mkdir(exist_ok=True)
    if log_path.exists():
        stamp = time.strftime("%m%d_%H%M")
        log_path = log_path.with_name(
            f"train_{run_name}_r{int(args.rate*100):02d}_{stamp}_log.csv")
        print(f"[日志] 原日志已存在，本次写入新文件：{log_path.name}")

    # ---- 训练循环（日志实时落盘，可随时 tail 查看进度）----
    log_file = open(log_path, "w", newline="", encoding="utf-8-sig")
    log_writer = csv.DictWriter(log_file, fieldnames=["epoch", "loss", "epoch_time_s"])
    log_writer.writeheader()
    log_file.flush()
    log_rows = []
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(p_total)
        running, n_seen = 0.0, 0
        t0 = time.time()
        for bt in range(0, p_total, args.batch_size):
            idx = perm[bt:bt + args.batch_size]
            xb = torch.from_numpy(data_np[idx.numpy()].astype(np.float32) / 255.0)
            xb = xb.unsqueeze(1).to(device)            # (B,1,33,33)
            # 数据增强：随机翻转（不插值，无伪影）
            if torch.rand(1).item() < 0.5:
                xb = torch.flip(xb, dims=[3])
            if torch.rand(1).item() < 0.5:
                xb = torch.flip(xb, dims=[2])

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device == "cuda"):
                if getattr(model, "samples_internally", False):
                    # 可学习采样模型（ours）：前向直接接受图像块
                    x_hat = model(xb)
                else:
                    # 固定 Φ 采样模型（reconnet）：外部采样后送入测量值
                    y = sample_blocks(xb, phi)
                    x_hat = model(y)
                loss = F.mse_loss(x_hat, xb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        avg_loss = running / n_seen
        dt = time.time() - t0
        row = {"epoch": epoch, "loss": round(avg_loss, 6), "epoch_time_s": round(dt, 1)}
        log_rows.append(row)
        log_writer.writerow(row)
        log_file.flush()
        if epoch % 5 == 0 or epoch == 1:
            el = time.time() - t_start
            eta = el / epoch * (args.epochs - epoch)
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.6f}  "
                  f"{dt:5.1f}s/epoch  已用{el/60:.1f}min  预计剩余{eta/60:.1f}min")

    # ---- 保存 ----
    torch.save({
        "model": model.state_dict(),
        "config": {
            "model": args.model, "run_name": run_name, "rate": args.rate, "m": m,
            "model_kwargs": model_kwargs,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "seed": args.seed, "amp": args.amp,
            "data": args.data, "phi_seed": 0, "n_blocks": p_total,
        },
    }, ckpt_path)
    print(f"[保存] {ckpt_path}")

    log_file.close()
    print(f"[日志] {log_path}")
    print(f"\n训练完成：{args.epochs} epoch，总用时 {(time.time()-t_start)/60:.1f} 分钟，"
          f"最终 loss={log_rows[-1]['loss']:.6f}")


if __name__ == "__main__":
    main()
