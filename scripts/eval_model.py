# -*- coding: utf-8 -*-
"""评估训练好的深度学习模型（Set11 全量）。

用法
----
    python scripts/eval_model.py --model reconnet                       # 三档全评
    python scripts/eval_model.py --model reconnet --rates 0.25          # 单档

产物（results/）
    deep_<model>_set11.csv   逐图逐档数据
    deep_<model>_set11.md    汇总表（PSNR / SSIM）
    deep_<model>_visual.png  重建效果网格（图 × 采样率）
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

from src.classical.sampling import (  # noqa: E402
    BLOCK, BLOCK_DIM, blocks_to_image, image_to_blocks, make_measurement_matrix,
)
from src.data.dataset import load_set11  # noqa: E402
from src.data.prepare import pad_to_multiple  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.models import build_model  # noqa: E402

TEST_SIZE = 256
VIS_IMAGES = ["barbara", "lena256", "cameraman"]


def load_model(model_name: str, rate: float, device: str):
    """model_name 可带消融后缀，如 ours / ours_noMS / ours_noAttn / ours_noLS。"""
    ckpt_path = ROOT / "checkpoints" / f"{model_name}_r{int(rate*100):02d}.pth"
    assert ckpt_path.exists(), f"未找到模型：{ckpt_path}（先训练）"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], cfg["m"], **cfg.get("model_kwargs", {})).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def reconstruct(model, phi: torch.Tensor, img_np: np.ndarray, device: str) -> np.ndarray:
    """整图重建（与 classical 同一口径：pad 264 → 采样 → 重建 → 裁 256）。"""
    img_pad = pad_to_multiple(img_np)
    x = torch.from_numpy(img_pad)[None, None].to(device)
    h, w = x.shape[2], x.shape[3]
    blocks = image_to_blocks(x)                       # (1,P,1089)
    if getattr(model, "samples_internally", False):
        out = model(blocks.view(-1, 1, BLOCK, BLOCK))  # 可学习采样模型（ours）
    else:
        y = blocks @ phi.t()                          # 固定 Φ 采样
        out = model(y.reshape(-1, y.shape[-1]))       # (P,1,33,33)
    rec_blocks = out.view(1, -1, BLOCK_DIM).clamp(0, 1)
    rec = blocks_to_image(rec_blocks, h, w)[:, :, :TEST_SIZE, :TEST_SIZE]
    return rec[0, 0].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="reconnet")
    ap.add_argument("--rates", nargs="+", type=float, default=[0.10, 0.25, 0.50])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = load_set11(ROOT / "data" / "Set11", size=TEST_SIZE)
    print(f"设备={device} | 模型={args.model} | 图像 {len(items)} 张\n")

    results, visuals = [], {}
    for rate in args.rates:
        model, cfg = load_model(args.model, rate, device)
        phi = make_measurement_matrix(cfg["m"], BLOCK_DIM, seed=0).to(device)
        print(f"--- {args.model} rate={rate:.2f} (M={cfg['m']}, "
              f"训练 {cfg['epochs']} epochs) ---")
        for name, img_np in items:
            t0 = time.time()
            rec = reconstruct(model, phi, img_np, device)
            dt = time.time() - t0
            m = evaluate(torch.from_numpy(rec)[None, None],
                         torch.from_numpy(img_np)[None, None])
            results.append({"algorithm": args.model, "rate": rate, "image": name,
                            "psnr": round(m["psnr"], 2), "ssim": round(m["ssim"], 4),
                            "time_s": round(dt, 2)})
            visuals[(name, rate)] = rec
            print(f"  {name:12s} PSNR={m['psnr']:6.2f} dB  SSIM={m['ssim']:.4f}  {dt:5.2f}s")

    # ---- 输出 CSV ----
    out_dir = ROOT / "results"
    csv_path = out_dir / f"deep_{args.model}_set11.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["algorithm", "rate", "image", "psnr", "ssim", "time_s"])
        w.writeheader()
        w.writerows(results)
    print(f"\n[CSV] {csv_path}")

    # ---- 汇总 Markdown ----
    image_names = [n for n, _ in items]
    lines = [f"# {args.model} 评估结果（Set11, {TEST_SIZE}×{TEST_SIZE}）\n"]
    for metric in ("psnr", "ssim"):
        fmt = "{:.2f}" if metric == "psnr" else "{:.4f}"
        lines.append(f"\n## {'PSNR（dB）' if metric == 'psnr' else 'SSIM'}\n")
        lines.append("| 采样率 | " + " | ".join(image_names) + " | 平均 |")
        lines.append("|" + "---|" * (len(image_names) + 2))
        for rate in args.rates:
            vals = {r["image"]: r[metric] for r in results if abs(r["rate"] - rate) < 1e-9}
            avg = np.mean([vals[n] for n in image_names])
            cells = " | ".join(fmt.format(vals[n]) for n in image_names)
            lines.append(f"| {int(rate*100)}% | {cells} | **{fmt.format(avg)}** |")
    md_path = out_dir / f"deep_{args.model}_set11.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[MD ] {md_path}")

    # ---- 可视化 ----
    make_visual(items, args.rates, visuals, out_dir / f"deep_{args.model}_visual.png")
    print("\n评估完成 ✓")


def make_visual(items, rates, visuals, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    vis_names = [n for n in VIS_IMAGES if any(k[0] == n for k in visuals)]
    orig = {n: v for n, v in items}
    cols = 1 + len(rates)
    fig, axes = plt.subplots(len(vis_names), cols,
                             figsize=(2.1 * cols, 2.4 * len(vis_names)), dpi=150)
    if len(vis_names) == 1:
        axes = axes[None, :]
    for r, name in enumerate(vis_names):
        axes[r, 0].imshow(orig[name], cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title("原图", fontsize=9)
        axes[r, 0].set_ylabel(name, fontsize=9)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for c, rate in enumerate(rates, start=1):
            ax = axes[r, c]
            ax.imshow(visuals[(name, rate)], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{int(rate*100)}% 采样", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIS] {out_path}")


if __name__ == "__main__":
    main()
