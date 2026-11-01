# -*- coding: utf-8 -*-
"""统一方法对比：汇总表 + 重建对比图（论文核心图表）。

读取 results/ 下所有评估数据：
    baseline_set11.csv        传统方法（OMP / ISTA）
    deep_<model>_set11.csv    深度学习模型（自动发现，可多个）

输出：
    results/comparison_set11.md     全部方法 × 三档采样率的 PSNR/SSIM 总表
    results/comparison_visual.png   指定采样率下「原图 vs 各方法」重建对比

用法：
    python scripts/make_comparison.py
    python scripts/make_comparison.py --visual-rate 0.25 --images barbara cameraman Monarch
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.classical.ista import ista_reconstruct  # noqa: E402
from src.classical.omp import omp_reconstruct  # noqa: E402
from src.classical.sampling import (  # noqa: E402
    BLOCK, BLOCK_DIM, blocks_to_image, image_to_blocks, make_measurement_matrix,
)
from src.data.dataset import load_set11  # noqa: E402
from src.data.prepare import pad_to_multiple  # noqa: E402
from src.models import build_model  # noqa: E402

TEST_SIZE = 256
RATES = [0.10, 0.25, 0.50]


def load_all_results() -> dict:
    """读取所有评估 CSV → {method: {rate: {image: (psnr, ssim)}}}"""
    data: dict = {}
    files = [ROOT / "results" / "baseline_set11.csv"]
    files += sorted((ROOT / "results").glob("deep_*_set11.csv"))
    for f in files:
        if not f.exists():
            continue
        with open(f, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                method = row["algorithm"]
                rate = float(row["rate"])
                data.setdefault(method, {}).setdefault(rate, {})[row["image"]] = (
                    float(row["psnr"]), float(row["ssim"]))
    return data


@torch.no_grad()
def recon_deep(model_name: str, img_np: np.ndarray, rate: float, device: str) -> np.ndarray:
    ckpt_path = ROOT / "checkpoints" / f"{model_name}_r{int(rate*100):02d}.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], cfg["m"], **cfg.get("model_kwargs", {})).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img_pad = pad_to_multiple(img_np)
    x = torch.from_numpy(img_pad)[None, None].to(device)
    blocks = image_to_blocks(x)
    if getattr(model, "samples_internally", False):
        out = model(blocks.view(-1, 1, BLOCK, BLOCK))
    else:
        y = blocks @ make_measurement_matrix(cfg["m"], BLOCK_DIM, 0).to(device).t()
        out = model(y.reshape(-1, cfg["m"]))
    rec_blocks = out.view(1, -1, BLOCK_DIM).clamp(0, 1)
    return blocks_to_image(rec_blocks, x.shape[2], x.shape[3])[0, 0, :TEST_SIZE, :TEST_SIZE].cpu().numpy()


def recon_any(method: str, img_np: np.ndarray, rate: float, device: str) -> np.ndarray:
    if method == "OMP":
        img = torch.from_numpy(pad_to_multiple(img_np))[None, None]
        return omp_reconstruct(img, rate=rate, device=device)[0, 0, :TEST_SIZE, :TEST_SIZE].cpu().numpy()
    if method == "ISTA":
        img = torch.from_numpy(pad_to_multiple(img_np))[None, None]
        return ista_reconstruct(img, rate=rate, lam=0.01, iters=1000,
                                device=device)[0, 0, :TEST_SIZE, :TEST_SIZE].cpu().numpy()
    return recon_deep(method, img_np, rate, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visual-rate", type=float, default=0.25)
    ap.add_argument("--images", nargs="+", default=["barbara", "cameraman", "Monarch"])
    args = ap.parse_args()

    data = load_all_results()
    if not data:
        print("results/ 下没有任何评估数据，先跑 run_baseline.py / eval_model.py")
        return
    methods = list(data.keys())
    print(f"发现方法: {methods}")

    # ---------- 汇总表 ----------
    lines = ["# 方法对比总表（Set11, 256×256, 无噪声, 平均 PSNR / SSIM）\n",
             "| 方法 | " + " | ".join(f"{int(r*100)}%" for r in RATES) + " |",
             "|" + "---|" * (len(RATES) + 1)]
    for method in methods:
        cells = []
        for rate in RATES:
            d = data[method].get(rate, {})
            if d:
                p = np.mean([v[0] for v in d.values()])
                s = np.mean([v[1] for v in d.values()])
                cells.append(f"**{p:.2f}** / {s:.4f}")
            else:
                cells.append("—")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.append("\n> 口径见 README 第 4 节；传统方法超参见 results/baseline_config.json。")
    md_path = ROOT / "results" / "comparison_set11.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[MD ] {md_path}")

    # ---------- 对比图 ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = dict(load_set11(ROOT / "data" / "Set11", size=TEST_SIZE))
    vis_names = [n for n in args.images if n in items]
    rate = args.visual_rate

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    cols = 1 + len(methods)
    fig, axes = plt.subplots(len(vis_names), cols,
                             figsize=(2.15 * cols, 2.45 * len(vis_names)), dpi=150)
    if len(vis_names) == 1:
        axes = axes[None, :]
    for r_i, name in enumerate(vis_names):
        axes[r_i, 0].imshow(items[name], cmap="gray", vmin=0, vmax=1)
        axes[r_i, 0].set_title("原图", fontsize=9)
        axes[r_i, 0].set_ylabel(name, fontsize=9)
        axes[r_i, 0].set_xticks([]); axes[r_i, 0].set_yticks([])
        for c_i, method in enumerate(methods, start=1):
            ax = axes[r_i, c_i]
            try:
                rec = recon_any(method, items[name], rate, device)
                ax.imshow(rec, cmap="gray", vmin=0, vmax=1)
                p = data.get(method, {}).get(rate, {}).get(name, (None,))[0]
                ax.set_title(f"{method}" + (f"  {p:.2f} dB" if p else ""), fontsize=8)
            except Exception as e:  # noqa: BLE001
                ax.set_title(f"{method} 不可用", fontsize=8)
                print(f"  [跳过] {method}@{rate} {name}: {e}")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"重建效果对比 @ {int(rate*100)}% 采样率（PSNR 见各图标题）", fontsize=11)
    fig.tight_layout()
    out_png = ROOT / "results" / "comparison_visual.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIS] {out_png}")


if __name__ == "__main__":
    main()
