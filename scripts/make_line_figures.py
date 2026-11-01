# -*- coding: utf-8 -*-
"""生成论文第 5 章用的折线图（静态 PNG，供 Word 插图使用）。

数据来源与演示系统折线图完全一致：results/ 下已跑完的 Set11 评估结果，
不重新推理、不引入任何新数字。

产出（results/figures/）：
    fig5_rate_curve.png     各方法 PSNR / SSIM 随采样率变化曲线（双子图）
    fig5_per_image.png      25% 采样率下 11 张测试图的逐图像 PSNR 对比曲线

用法：.venv/Scripts/python.exe scripts/make_line_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.plotting import (  # noqa: E402
    COLORS, IMAGE_ORDER, label_of, load_result_tables, mean_metric,
)

OUT = ROOT / "results" / "figures"
RATES = [0.10, 0.25, 0.50]
XS = [10, 25, 50]
METHODS = ["ours", "reconnet", "ISTA", "OMP"]
MARKERS = {"ours": "o", "reconnet": "s", "ISTA": "^", "OMP": "D"}

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11


def style_ax(ax):
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=10)


def fig_rate_curve(data: dict) -> Path:
    """(a) PSNR (b) SSIM —— 横轴采样率，四条折线，两子图共用底部图例。"""
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), dpi=400)
    handles, labels = [], []
    for ax, metric, tag in ((axes[0], "psnr", "（a）PSNR"),
                            (axes[1], "ssim", "（b）SSIM")):
        for m in METHODS:
            ys = [mean_metric(data, m, r, metric) for r in RATES]
            if any(v is None for v in ys):
                continue
            ln, = ax.plot(XS, ys, marker=MARKERS[m], markersize=5.5, linewidth=1.7,
                          color=COLORS[m], markerfacecolor="white",
                          markeredgewidth=1.4, label=label_of(m))
            if ax is axes[0]:
                handles.append(ln)
                labels.append(label_of(m))
        style_ax(ax)
        ax.set_xticks(XS)
        ax.set_xlabel("采样率（%）", fontsize=11)
        ax.set_ylabel("平均 PSNR（dB）" if metric == "psnr" else "平均 SSIM",
                      fontsize=11)
        ax.set_title(tag, fontsize=11)
    fig.legend(handles, labels, fontsize=9.5, frameon=True, edgecolor="#BBBBBB",
               ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    path = OUT / "fig5_rate_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_per_image(data: dict, rate: float = 0.25) -> Path:
    """25% 采样率下 11 张测试图的逐图像 PSNR 折线。"""
    imgs = [i for i in IMAGE_ORDER if all(i in (data.get(m, {}).get(rate) or {})
                                          for m in METHODS)]
    fig, ax = plt.subplots(figsize=(7.8, 3.3), dpi=400)
    for m in METHODS:
        d = data[m][rate]
        ys = [d[i]["psnr"] for i in imgs]
        ax.plot(range(len(imgs)), ys, marker=MARKERS[m], markersize=5,
                linewidth=1.6, color=COLORS[m], markerfacecolor="white",
                markeredgewidth=1.2, label=label_of(m))
    style_ax(ax)
    ax.set_xticks(range(len(imgs)))
    ax.set_xticklabels(imgs, rotation=20, ha="right", fontsize=9.5)
    ax.set_xlabel("测试图像（Set11）", fontsize=11)
    ax.set_ylabel("PSNR（dB）", fontsize=11)
    ax.legend(fontsize=9.5, frameon=True, edgecolor="#BBBBBB", ncol=2,
              loc="lower left")
    fig.tight_layout()
    path = OUT / "fig5_per_image.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_result_tables()
    if not data:
        print("results/ 下没有评估数据")
        return
    p1 = fig_rate_curve(data)
    p2 = fig_per_image(data)
    for p in (p1, p2):
        print(f"[FIG] {p}")
    # 打印曲线上的数值（供核对正文引用）
    print("\n采样率曲线数值：")
    for m in METHODS:
        print(f"  {m:9s} PSNR={[round(mean_metric(data, m, r, 'psnr'), 2) for r in RATES]}"
              f"  SSIM={[round(mean_metric(data, m, r, 'ssim'), 4) for r in RATES]}")
    print("\n逐图像 PSNR（25%）领先 OMP 的幅度：")
    for i in IMAGE_ORDER:
        d = {m: data[m][0.25][i]["psnr"] for m in METHODS}
        print(f"  {i:12s} ours={d['ours']:.2f}  reconnet={d['reconnet']:.2f}"
              f"  ISTA={d['ISTA']:.2f}  OMP={d['OMP']:.2f}  Δ(OMP)={d['ours']-d['OMP']:.2f}")


if __name__ == "__main__":
    main()
