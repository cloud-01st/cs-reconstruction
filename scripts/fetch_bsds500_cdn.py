# -*- coding: utf-8 -*-
"""从 jsDelivr CDN 并行下载 BSDS500 训练图片（train 200 + val 100 = 300 张）。

背景（2026-10 实测国内网络）
---------------------------
伯克利官网直连：约 8 KB/s（不可用）
GitHub git clone：不稳定，实测多次 invalid index-pack output 中断
jsDelivr CDN：稳定可达，多线程并行后速度可用  ← 本脚本采用

用法
----
    python scripts/fetch_bsds500_cdn.py            # 下载全部
    python scripts/fetch_bsds500_cdn.py --workers 12

输出
----
    data/raw/BSDS500_src/BSDS500/data/images/{train,val}/*.jpg
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "BSDS500_src" / "BSDS500" / "data" / "images"

API = "https://api.github.com/repos/BIDS/BSDS500/contents/BSDS500/data/images/{split}"
MIRRORS = [
    "https://cdn.jsdelivr.net/gh/BIDS/BSDS500@master/BSDS500/data/images",
    "https://fastly.jsdelivr.net/gh/BIDS/BSDS500@master/BSDS500/data/images",
]
UA = {"User-Agent": "Mozilla/5.0 (bsds-fetch)"}
_ssl_ctx = ssl.create_default_context()


def list_split(split: str) -> list[str]:
    """从 GitHub API 获取某划分的文件名列表（仅 jpg）。"""
    req = Request(API.format(split=split), headers=UA)
    with urlopen(req, timeout=30, context=_ssl_ctx) as r:
        data = json.load(r)
    return [x["name"] for x in data if x.get("type") == "file"
            and x["name"].lower().endswith(".jpg")]


def fetch_one(split: str, name: str, retries: int = 4) -> bool:
    """下载单个文件；已存在且大小合理则跳过。节点轮换 + 重试。"""
    dst = DEST / split / name
    if dst.exists() and dst.stat().st_size > 1024:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        url = f"{MIRRORS[i % len(MIRRORS)]}/{split}/{name}"
        try:
            req = Request(url, headers=UA)
            with urlopen(req, timeout=90, context=_ssl_ctx) as r:
                data = r.read()
            if len(data) > 1024:
                dst.write_bytes(data)
                return True
        except (URLError, TimeoutError, OSError):
            time.sleep(0.8 * (i + 1))
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks: list[tuple[str, str]] = []
    for split in ("train", "val"):
        names = list_split(split)
        print(f"[列表] {split}: {len(names)} 张")
        tasks += [(split, n) for n in names]

    todo = [(s, n) for s, n in tasks
            if not ((DEST / s / n).exists() and (DEST / s / n).stat().st_size > 1024)]
    print(f"[下载] 共 {len(tasks)} 张，其中待下载 {len(todo)} 张，并行 {args.workers} 线程")

    ok, fail = 0, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, s, n): (s, n) for s, n in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            s, n = futs[fu]
            if fu.result():
                ok += 1
            else:
                fail.append(f"{s}/{n}")
            if i % 30 == 0 or i == len(futs):
                el = time.time() - t0
                print(f"  进度 {i}/{len(futs)}  成功 {ok}  失败 {len(fail)}  "
                      f"用时 {el:.0f}s（{i/el:.1f} 张/秒）")

    n_train = len(list((DEST / "train").glob("*.jpg")))
    n_val = len(list((DEST / "val").glob("*.jpg")))
    print(f"[完成] train {n_train}/200  val {n_val}/100  失败 {len(fail)}")
    if fail:
        print("[失败列表]", fail[:20])
        sys.exit(1)
    print("BSDS500 图片就绪 ✓")


if __name__ == "__main__":
    main()
