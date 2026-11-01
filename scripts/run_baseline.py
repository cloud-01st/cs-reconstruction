# -*- coding: utf-8 -*-
"""阶段 1：传统基线实验（Set11 × 三档采样率 × OMP / ISTA-FISTA）。

产出（results/ 目录）
---------------------
    baseline_set11.csv    逐图逐档原始结果（机器可读，含耗时）
    baseline_set11.md     论文用对比表（PSNR / SSIM，含平均）
    baseline_visual.png   重建效果对比图（3 张代表图 × 7 列）
    baseline_config.json  本次实验配置快照（保证可复现）

用法
----
    python scripts/run_baseline.py            # 全量（11 张 × 3 档 × 2 算法，约 10 分钟）
    python scripts/run_baseline.py --quick    # 快速验证（3 张图、仅 0.25，约 1 分钟）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.classical.ista import ista_reconstruct  # noqa: E402
from src.classical.omp import omp_reconstruct  # noqa: E402
from src.data.dataset import load_set11  # noqa: E402
from src.data.prepare import pad_to_multiple  # noqa: E402
from src.metrics import evaluate  # noqa: E402

# ---- 实验配置（会写入 baseline_config.json，改这里即改实验口径）----
RATES = [0.10, 0.25, 0.50]
ALGOS = ["OMP", "ISTA"]
OMP_SPARSITY = None          # None = 自动 max(15, M//8)（经预实验确定）
ISTA_LAM = 0.01
ISTA_ITERS = 1000
ISTA_FISTA = True
TEST_SIZE = 256
VIS_IMAGES = ["barbara", "lena256", "cameraman"]   # 效果图用图


def reconstruct(algo: str, img: torch.Tensor, rate: float, device: str) -> torch.Tensor:
    if algo == "OMP":
        return omp_reconstruct(img, rate=rate, sparsity=OMP_SPARSITY, device=device)
    return ista_reconstruct(img, rate=rate, lam=ISTA_LAM, iters=ISTA_ITERS,
                            fista=ISTA_FISTA, device=device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="快速验证（3 张图、仅 0.25）")
    ap.add_argument("--rates", nargs="+", type=float, default=None)
    ap.add_argument("--algorithms", nargs="+", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rates = [0.25] if args.quick else (args.rates or RATES)
    algos = args.algorithms or ALGOS

    items = load_set11(ROOT / "data" / "Set11", size=TEST_SIZE)
    if args.quick:
        items = items[:3]
    print(f"设备: {device} | 图像 {len(items)} 张 | 采样率 {rates} | 算法 {algos}\n")

    # Warmup（避免 CUDA 首次调用初始化时间混入计时）
    if device == "cuda":
        _ = torch.randn(256, 1089, device=device) @ torch.randn(1089, 272, device=device)
        torch.cuda.synchronize()

    results: list[dict] = []
    visuals: dict[tuple, np.ndarray] = {}

    for name, img_np in items:
        img = torch.from_numpy(pad_to_multiple(img_np))[None, None]      # 264×264
        target = torch.from_numpy(img_np)[None, None]                    # 256×256
        for rate in rates:
            for algo in algos:
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                rec = reconstruct(algo, img, rate, device)
                if device == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t0
                rec = rec[:, :, :TEST_SIZE, :TEST_SIZE].cpu()
                m = evaluate(rec, target)
                results.append({
                    "algorithm": algo, "rate": rate, "image": name,
                    "psnr": round(m["psnr"], 2), "ssim": round(m["ssim"], 4),
                    "time_s": round(dt, 1),
                })
                visuals[(name, rate, algo)] = rec[0, 0].numpy()
                print(f"  {algo:4s} rate={rate:.2f}  {name:12s}  "
                      f"PSNR={m['psnr']:6.2f} dB  SSIM={m['ssim']:.4f}  {dt:6.1f}s")

    # ---------- 保存 CSV ----------
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "baseline_set11.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["algorithm", "rate", "image", "psnr", "ssim", "time_s"])
        w.writeheader()
        w.writerows(results)
    print(f"\n[CSV] {csv_path}")

    # ---------- 汇总 Markdown 表（PSNR 与 SSIM）----------
    image_names = [n for n, _ in items]
    lines = ["# 基线实验结果（Set11, {}×{}, 无噪声）\n".format(TEST_SIZE, TEST_SIZE)]
    for metric, unit in (("psnr", "dB"), ("ssim", "")):
        lines.append(f"\n## {'PSNR（'+unit+'）' if unit else 'SSIM'}\n")
        header = "| 方法 | 采样率 | " + " | ".join(image_names) + " | 平均 |"
        lines.append(header)
        lines.append("|" + "---|" * (len(image_names) + 3))
        fmt = "{:.2f}" if metric == "psnr" else "{:.4f}"
        for algo in algos:
            for rate in rates:
                row_res = [r for r in results
                           if r["algorithm"] == algo and abs(r["rate"] - rate) < 1e-9]
                vals = {r["image"]: r[metric] for r in row_res}
                avg = np.mean([vals[n] for n in image_names]) if row_res else 0
                cells = " | ".join(fmt.format(vals.get(n, float("nan"))) for n in image_names)
                lines.append(f"| {algo} | {int(rate*100)}% | {cells} | **{fmt.format(avg)}** |")
    lines.append("\n> 口径：256×256 灰度、resize 统一尺寸、33×33 不重叠块、"
                 "edge padding 至 264 后裁回 256 计指标；PSNR 于 [0,1] 域计算。\n")
    md = "\n".join(lines)
    md_path = out_dir / "baseline_set11.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"[MD ] {md_path}")

    # ---------- 配置快照 ----------
    cfg = {
        "dataset": "Set11", "test_size": TEST_SIZE, "rates": rates, "algorithms": algos,
        "omp_sparsity": OMP_SPARSITY, "ista_lam": ISTA_LAM, "ista_iters": ISTA_ITERS,
        "ista_fista": ISTA_FISTA, "device": device,
        "images": image_names,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    cfg_path = out_dir / "baseline_config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 重建效果对比图 ----------
    make_visual(items, rates, algos, visuals, results, out_dir / "baseline_visual.png")

    print("\n阶段 1 基线实验完成 ✓")


def make_visual(items, rates, algos, visuals, results, out_path: Path) -> None:
    """3 张代表图 × （原图 + 算法×采样率）网格对比图，标题带 PSNR。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    psnr_map = {(r["algorithm"], round(r["rate"], 4), r["image"]): r["psnr"] for r in results}

    vis_names = [n for n in VIS_IMAGES if any(k[0] == n for k in visuals)]
    if not vis_names:
        vis_names = sorted({k[0] for k in visuals})[:3]
    cols = 1 + len(algos) * len(rates)
    n_rows = len(vis_names)
    fig, axes = plt.subplots(n_rows, cols, figsize=(2.1 * cols, 2.5 * n_rows), dpi=150)
    if n_rows == 1:
        axes = axes[None, :]

    orig = {n: v for n, v in items}
    for r, name in enumerate(vis_names):
        axes[r, 0].imshow(orig[name], cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title("原图", fontsize=9)
        axes[r, 0].set_ylabel(name, fontsize=9)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        c = 1
        for algo in algos:
            for rate in rates:
                ax = axes[r, c]
                ax.imshow(visuals[(name, rate, algo)], cmap="gray", vmin=0, vmax=1)
                p = psnr_map.get((algo, round(rate, 4), name))
                title = f"{algo} {int(rate*100)}%" + (f"\n{p:.2f} dB" if p is not None else "")
                ax.set_title(title, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                c += 1

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIS] {out_path}")


if __name__ == "__main__":
    main()
