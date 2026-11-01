# -*- coding: utf-8 -*-
"""消融实验报告生成器（@25% 采样率）

读取完整模型与三个消融变体在 Set11 上的评估结果，生成：
  results/ablation_set11.md    消融对比表（PSNR / SSIM / ΔPSNR）
  results/ablation_set11.png   柱状图（含模块贡献标注）

数据源（由 run_ablation_variant.sh 的评估环节产出）：
  results/deep_ours_set11.csv          完整模型
  results/deep_ours_noLS_set11.csv     无可学习采样
  results/deep_ours_noMS_set11.csv     无多尺度残差
  results/deep_ours_noAttn_set11.csv   无通道注意力

用法:
  python scripts/make_ablation_report.py            # 正式输出
  python scripts/make_ablation_report.py --test     # 演练（输出到 _ablation_TEST.*）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RATE = 0.25

# (csv 文件名, 显示名, 说明)
VARIANTS = [
    ("deep_ours_set11.csv",        "完整模型",     "三级级联 + 可学习采样 + 多尺度残差 + 注意力"),
    ("deep_ours_noLS_set11.csv",   "无可学习采样", "采样端改回固定随机 Φ"),
    ("deep_ours_noMS_set11.csv",   "无多尺度残差", "多尺度残差块替换为普通残差块"),
    ("deep_ours_noAttn_set11.csv", "无通道注意力", "去掉通道注意力模块"),
]


def load_csv(path: Path) -> dict | None:
    """返回 {psnr: float, ssim: float, n: int}；文件缺失返回 None。"""
    if not path.exists():
        return None
    import csv
    psnrs, ssims = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if abs(float(row["rate"]) - RATE) < 1e-9:
                psnrs.append(float(row["psnr"]))
                ssims.append(float(row["ssim"]))
    if not psnrs:
        return None
    return {"psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)), "n": len(psnrs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="演练模式：输出到 _ablation_TEST.*")
    args = ap.parse_args()

    data, missing = [], []
    for fname, label, desc in VARIANTS:
        d = load_csv(RESULTS / fname)
        if d is None:
            missing.append(fname)
        else:
            data.append({"label": label, "desc": desc, **d})

    print(f"=== 消融实验数据检查（@ {int(RATE*100)}%）===")
    for fname, label, _ in VARIANTS:
        mark = "✓" if not any(fname in m for m in missing) else "✗ 缺失"
        print(f"  {mark}  {label:8s}  <- {fname}")
    if missing:
        print(f"\n[等待] 还有 {len(missing)} 个变体未完成评估：{missing}")

    complete = len(data) == len(VARIANTS)
    if args.test or not complete:
        md_path = RESULTS / "_ablation_TEST.md"
        png_path = RESULTS / "_ablation_TEST.png"
        if complete:
            print("[注] 数据已齐全，但处于 --test 模式，输出到 _ablation_TEST.*")
        else:
            print("[注] 数据未齐，演练输出到 _ablation_TEST.*（正式文件不会被污染）")
    else:
        md_path = RESULTS / "ablation_set11.md"
        png_path = RESULTS / "ablation_set11.png"

    # ---- 表格（以完整模型为基准计 Δ）----
    baseline = data[0]["psnr"] if data else 0.0
    lines = [f"# 消融实验对比表（Set11, {int(RATE*100)}% 采样率）", ""]
    lines.append("| 模型 | PSNR (dB) | SSIM | ΔPSNR | 模块贡献 |")
    lines.append("|---|---|---|---|---|")
    for i, d in enumerate(data):
        delta = d["psnr"] - baseline
        if i == 0:
            lines.append(f"| **{d['label']}** | **{d['psnr']:.2f}** | **{d['ssim']:.4f}** | — | 基准 |")
        else:
            contrib = f"**+{abs(delta):.2f} dB**" if delta < 0 else f"{delta:+.2f} dB（未提升）"
            lines.append(f"| {d['label']} | {d['psnr']:.2f} | {d['ssim']:.4f} | {delta:+.2f} | {contrib} |")
    lines.append("")
    lines.append("> ΔPSNR = 该变体 − 完整模型（负值表示该模块被移除后的性能损失，即模块贡献）。")
    lines.append("> 三个变体与完整模型使用完全相同的训练配方（100 epoch + 10 epoch 精调，lr=1e-3 → 2e-4）。")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[MD ] {md_path}")
    print("\n".join(lines[2:8]))

    # ---- 柱状图 ----
    if data:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
        plt.rcParams["axes.unicode_minus"] = False

        labels = [d["label"] for d in data]
        vals = [d["psnr"] for d in data]
        colors = ["#1E5AA8"] + ["#9FB6D4"] * (len(data) - 1)

        fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=160)
        bars = ax.bar(range(len(data)), vals, color=colors, width=0.55,
                      edgecolor="white", linewidth=0.8)
        for i, (b, d) in enumerate(zip(bars, data)):
            y = d["psnr"]
            ax.text(b.get_x() + b.get_width() / 2, y + 0.12, f"{y:.2f}",
                    ha="center", va="bottom", fontsize=11.5,
                    color="#1A1A1A", fontweight="bold" if i == 0 else "normal")
            if i > 0:
                delta = y - baseline
                ax.text(b.get_x() + b.get_width() / 2, y - 0.55,
                        f"Δ {delta:+.2f}", ha="center", va="top", fontsize=10.5,
                        color="#C00000" if delta < 0 else "#808080")
        # 基准虚线
        ax.axhline(baseline, color="#1E5AA8", ls="--", lw=1, alpha=0.45)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(labels, fontsize=11.5)
        lo = min(vals) - 1.2
        hi = max(vals) + 0.9
        ax.set_ylim(lo, hi)
        ax.set_ylabel("PSNR (dB)", fontsize=12)
        ax.set_title(f"消融实验：各模块对重建质量的贡献（Set11 平均，{int(RATE*100)}% 采样率）",
                     fontsize=13, pad=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#E8E8E8", lw=0.7)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[VIS] {png_path}")

    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
