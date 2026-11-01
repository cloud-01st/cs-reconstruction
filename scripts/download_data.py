# -*- coding: utf-8 -*-
"""数据集下载脚本。

用法
----
    python scripts/download_data.py --all        # BSDS500 + Set11
    python scripts/download_data.py --bsds500    # 仅训练集
    python scripts/download_data.py --set11      # 仅测试集
    python scripts/download_data.py --check      # 只检查数据是否就绪

数据源与国内网络说明（2026-10 实测）
------------------------------------
BSDS500（训练集，300 张 = train 200 + val 100）:
    默认经 jsDelivr CDN 并行下载（见 fetch_bsds500_cdn.py），约 4 分钟
    伯克利官方 tgz 国内直连约 8 KB/s（仅 --official 时使用）
Set11（测试集，11 张经典灰度图）:
    ISTA-Net 官方仓库（CVPR 2018）内容，同样经 jsDelivr CDN 加速
    （raw.githubusercontent.com 国内直连不可达）
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.prepare import build_bsds_train_npy  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SET11_DIR = ROOT / "data" / "Set11"

# ---- BSDS500 ----
BSDS_OFFICIAL_URL = ("https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/"
                     "grouping/BSR/BSR_bsds500.tgz")

# ---- Set11（多镜像，按顺序尝试）----
SET11_FILES = [
    "Monarch.tif", "Parrots.tif", "barbara.tif", "boats.tif", "cameraman.tif",
    "fingerprint.tif", "flinstones.tif", "foreman.tif", "house.tif",
    "lena256.tif", "peppers256.tif",
]
SET11_MIRRORS = [
    "https://cdn.jsdelivr.net/gh/jianzhangcs/ISTA-Net-PyTorch@master/data/Set11/",
    "https://raw.githubusercontent.com/jianzhangcs/ISTA-Net-PyTorch/master/data/Set11/",
]


def download(urls: str | list[str], dst: Path, min_bytes: int = 1024) -> None:
    """下载文件（已存在且大小合理则跳过）。urls 为镜像列表，依次尝试。"""
    if dst.exists() and dst.stat().st_size >= min_bytes:
        print(f"  已存在，跳过：{dst.name} ({dst.stat().st_size/1024:.0f} KB)")
        return
    if isinstance(urls, str):
        urls = [urls]
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in urls:
        try:
            print(f"  下载：{url}")
            tmp = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dst)
            print(f"  完成：{dst.name} ({dst.stat().st_size/1024:.0f} KB)")
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"    失败（{type(e).__name__}），尝试下一个镜像…")
    raise RuntimeError(f"全部镜像均失败：{dst.name}") from last_err


def _build_train_npy(images_dir: Path) -> None:
    out = PROCESSED / "bsds300_train.npy"
    build_bsds_train_npy([images_dir / "train", images_dir / "val"], out)


def setup_bsds500(official: bool = False) -> None:
    """准备 BSDS500 训练集。

    默认走 jsDelivr CDN 并行下载（2026-10 实测最稳定，约 4 分钟；
    GitHub git clone 实测不稳定、伯克利官网约 8 KB/s 不可用）。
    official=True 时走伯克利官方 tgz。
    """
    print("[1/2] BSDS500（训练集）")
    if official:
        tgz = RAW / "BSR_bsds500.tgz"
        download(BSDS_OFFICIAL_URL, tgz, min_bytes=10_000_000)
        if not (RAW / "BSR" / "BSDS500" / "images").exists():
            print("  解压中…")
            with tarfile.open(tgz, "r:gz") as tar:
                tar.extractall(RAW)
        _build_train_npy(RAW / "BSR" / "BSDS500" / "images")
        return

    # jsDelivr 并行下载（独立脚本，已验证）
    script = Path(__file__).resolve().parent / "fetch_bsds500_cdn.py"
    subprocess.run([sys.executable, str(script), "--workers", "8"], check=True)
    images = RAW / "BSDS500_src" / "BSDS500" / "data" / "images"
    _build_train_npy(images)


def setup_set11() -> None:
    """下载 Set11 的 11 张测试图（经 jsDelivr CDN）。"""
    print("[2/2] Set11（测试集）")
    for name in SET11_FILES:
        download([m + name for m in SET11_MIRRORS], SET11_DIR / name, min_bytes=50_000)


def check() -> bool:
    ok = True
    npy = PROCESSED / "bsds300_train.npy"
    if npy.exists():
        import numpy as np
        n = np.load(npy, mmap_mode="r").shape[0]
        print(f"  ✓ 训练块：{npy}（{n} 块）")
    else:
        print(f"  ✗ 训练块缺失：{npy}")
        ok = False
    n_set11 = len(list(SET11_DIR.glob("*.tif"))) if SET11_DIR.exists() else 0
    if n_set11 >= 11:
        print(f"  ✓ Set11：{SET11_DIR}（{n_set11} 张）")
    else:
        print(f"  ✗ Set11 缺失或不完整：{SET11_DIR}（{n_set11}/11）")
        ok = False
    print("数据就绪 ✓" if ok else "数据未就绪，请运行：python scripts/download_data.py --all")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="下载毕设所需数据集")
    ap.add_argument("--all", action="store_true", help="下载全部")
    ap.add_argument("--bsds500", action="store_true")
    ap.add_argument("--set11", action="store_true")
    ap.add_argument("--official", action="store_true",
                    help="BSDS500 走伯克利官方 tgz（国内很慢，默认用 GitHub 镜像）")
    ap.add_argument("--check", action="store_true", help="仅检查")
    args = ap.parse_args()

    if args.check:
        check()
        return
    if args.all or args.bsds500:
        setup_bsds500(official=args.official)
    if args.all or args.set11:
        setup_set11()
    if not (args.all or args.bsds500 or args.set11):
        ap.print_help()
        return
    check()


if __name__ == "__main__":
    main()
