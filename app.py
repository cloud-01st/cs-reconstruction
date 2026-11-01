# -*- coding: utf-8 -*-
"""图像压缩感知重建演示系统（毕业设计 4.3）

功能
----
  1. 图像上传 / 使用内置示例图（Set11）
  2. 采样率选择：10% / 25% / 50%，或三档全对比
  3. 重建结果与评价指标展示（PSNR / SSIM / 耗时）
  4. 传统方法对照（OMP / ISTA）—— 加强版功能
  5. 可学习测量核可视化（展示采样网络学到了什么）—— 加强版功能
  6. 模型对比折线图（指标–采样率 / 逐图像 / 消融 / 训练收敛 / 本图实测）
     —— 加强版功能，见 src/plotting.py

运行
----
    cd D:/毕业设计/code
    .venv/Scripts/streamlit.exe run app.py

作者：周靖凌（南昌职业大学 · 信息技术学院）
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.classical.ista import ista_reconstruct  # noqa: E402
from src.classical.omp import omp_reconstruct  # noqa: E402
from src.classical.sampling import (  # noqa: E402
    BLOCK, BLOCK_DIM, blocks_to_image, image_to_blocks,
)
from src.data.prepare import pad_to_multiple  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.models import build_model  # noqa: E402
from src.plotting import (  # noqa: E402
    fig_ablation, fig_live_vs_rate, fig_metric_per_image, fig_metric_vs_rate,
    fig_train_curves, load_result_tables, load_train_logs, pretty_log_name,
)

MAX_SIDE = 256
RATES = {"10%": 0.10, "25%": 0.25, "50%": 0.50}
LABELS = {"ours": "本方法（OursNet）", "omp": "OMP（传统）", "ista": "ISTA（传统）"}
SET11_DIR = ROOT / "data" / "Set11"

st.set_page_config(page_title="图像压缩感知重建演示系统", layout="wide",
                   page_icon="🧩")


# ================================================================ 模型与推理
@st.cache_resource(show_spinner=False)
def load_ours(rate: float, device: str):
    """加载指定采样率的 OursNet（缓存，只加载一次）。"""
    ckpt_path = ROOT / "checkpoints" / f"ours_r{int(rate * 100):02d}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"缺少模型权重：{ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], cfg["m"], **cfg.get("model_kwargs", {})).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def run_ours(model, img_np: np.ndarray, device: str):
    """本方法重建（与评估脚本同一口径）。"""
    t0 = time.time()
    pad = pad_to_multiple(img_np)
    x = torch.from_numpy(pad)[None, None].to(device)
    h, w = x.shape[2], x.shape[3]
    blocks = image_to_blocks(x)                              # (1,P,1089)
    out = model(blocks.view(-1, 1, BLOCK, BLOCK))            # 可学习采样 + 重建
    rec_blocks = out.view(1, -1, BLOCK_DIM).clamp(0, 1)
    rec = blocks_to_image(rec_blocks, h, w)
    rec = rec[0, 0, :img_np.shape[0], :img_np.shape[1]].cpu().numpy()
    return rec, time.time() - t0


@torch.no_grad()
def run_classical(kind: str, img_np: np.ndarray, rate: float, device: str):
    """传统方法重建（OMP / ISTA）。"""
    t0 = time.time()
    pad = pad_to_multiple(img_np)
    x = torch.from_numpy(pad)[None, None].to(device)
    if kind == "omp":
        rec = omp_reconstruct(x, rate=rate, device=device)
    else:
        rec = ista_reconstruct(x, rate=rate, device=device)
    rec = rec[0, 0, :img_np.shape[0], :img_np.shape[1]].cpu().numpy()
    return rec, time.time() - t0


def get_result(img_np: np.ndarray, rate: float, method: str, device: str):
    """带缓存的统一入口：返回 (rec, psnr, ssim, dt)。"""
    img_key = hashlib.md5(img_np.tobytes()).hexdigest()[:12]
    ck = (img_key, round(rate, 4), method)
    cache = st.session_state.setdefault("_cache", {})
    if ck not in cache:
        if method == "ours":
            model = load_ours(rate, device)
            rec, dt = run_ours(model, img_np, device)
        else:
            rec, dt = run_classical(method, img_np, rate, device)
        m = evaluate(torch.from_numpy(rec)[None, None],
                     torch.from_numpy(img_np)[None, None])
        cache[ck] = (rec, m["psnr"], m["ssim"], dt)
    return cache[ck]


def preprocess(pil_img: Image.Image) -> np.ndarray:
    """转灰度 + 限制最大边 ≤ 256（保持比例）+ 归一化到 [0,1]。"""
    img = pil_img.convert("L")
    w, h = img.size
    scale = MAX_SIDE / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


# ================================================================ 附加功能
@st.cache_resource(show_spinner=False)
def get_kernel_grid(rate: float, device: str):
    """可视化采样网络学习到的测量核（取 16 个，拼 4×4）。"""
    model = load_ours(rate, device)
    with torch.no_grad():
        w = model.sampler.weight.detach().cpu()[:, 0]        # (M,33,33)
        n = w.shape[0]
        idx = torch.linspace(0, n - 1, 16).long()
        ks = w[idx]
        ks = (ks - ks.min()) / (ks.max() - ks.min() + 1e-8)
        pad = 2
        rows = []
        for r in range(4):
            cols = []
            for c in range(4):
                cols.append(ks[r * 4 + c].numpy())
                cols.append(np.zeros((BLOCK, pad), dtype=np.float32))
            row = np.concatenate(cols[:-1], axis=1)
            rows.append(row)
            rows.append(np.zeros((pad, row.shape[1]), dtype=np.float32))
        grid = np.concatenate(rows[:-1], axis=0)
    return grid


def measure_count(rate: float) -> int:
    return int(round(rate * BLOCK_DIM))


# ================================================================ 页面
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

st.title("图像压缩感知重建演示系统")
st.caption("基于深度学习的图像压缩感知重建算法实现 ｜ 周靖凌 ｜ 南昌职业大学 信息技术学院")

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("⚙️ 控制面板")

    rate_choice = st.radio("采样率（测量值占像素比）",
                           ["25%", "10%", "50%", "三档全对比"], index=0)

    show_cmp = st.checkbox("显示传统方法对照（OMP / ISTA）", value=True,
                           help="不勾选则只显示本方法（更快）")

    st.divider()
    source = st.radio("图像来源", ["使用示例图", "上传图像"], index=0)

    img_np = None
    img_title = ""
    if source == "使用示例图":
        samples = sorted(p.name for p in SET11_DIR.glob("*.tif"))
        if samples:
            pick = st.selectbox("选择 Set11 测试图", samples,
                                index=samples.index("cameraman.tif")
                                if "cameraman.tif" in samples else 0)
            img_np = preprocess(Image.open(SET11_DIR / pick))
            img_title = pick.replace(".tif", "")
        else:
            st.warning("未找到示例图（data/Set11 为空）")
    else:
        up = st.file_uploader("上传图像（jpg / png / bmp / tif）",
                              type=["jpg", "jpeg", "png", "bmp", "tif",
                                    "tiff", "webp"])
        if up is not None:
            img_np = preprocess(Image.open(up))
            img_title = up.name

    st.divider()
    st.caption(f"计算设备：**{gpu_name}**")
    st.caption("图像自动转为灰度，最大边缩放至 256 像素。")

# ---------- 主区域 ----------
if img_np is None:
    st.info("👈 请在左侧选择示例图或上传一张图像")
    st.stop()

h, w = img_np.shape
padded = pad_to_multiple(img_np)
n_blocks = (padded.shape[0] // BLOCK) * (padded.shape[1] // BLOCK)

rates_to_show = list(RATES.values()) if rate_choice == "三档全对比" \
    else [RATES[rate_choice]]
methods = ["ours", "omp", "ista"] if show_cmp else ["ours"]

st.markdown(f"**当前图像：** `{img_title}` ｜ 尺寸 **{w}×{h}** ｜ "
            f"分块 {BLOCK}×{BLOCK}，共 **{n_blocks}** 块 ｜ "
            f"每块 1089 像素 → 压缩为 M 个测量值")

col_a, col_b = st.columns([1.4, 1])
with col_a:
    st.image(img_np, caption="原图（灰度）", width=260)

progress = st.progress(0.0, text="正在重建…")
total_jobs = len(rates_to_show) * len(methods)
done = 0
table_rows = []
for rate in rates_to_show:
    pct = int(rate * 100)
    for mkey in methods:
        r, psnr, ssim, dt = get_result(img_np, rate, mkey, device)
        done += 1
        progress.progress(done / total_jobs,
                          text=f"正在重建… {done}/{total_jobs}")
        table_rows.append({"采样率": f"{pct}%", "方法": LABELS[mkey],
                           "PSNR (dB)": round(psnr, 2),
                           "SSIM": round(ssim, 4),
                           "耗时 (秒)": round(dt, 2)})
progress.empty()

# ---------- 结果网格 ----------
st.divider()
for rate in rates_to_show:
    pct = int(rate * 100)
    m_val = measure_count(rate)
    st.subheader(f"采样率 {pct}%（每块 {m_val} / 1089 个测量值）")
    cols = st.columns(1 + len(methods))
    with cols[0]:
        st.image(img_np, clamp=True)
        st.caption("原图")
    for i, mkey in enumerate(methods):
        rec, psnr, ssim, dt = get_result(img_np, rate, mkey, device)
        with cols[i + 1]:
            st.image(np.clip(rec, 0, 1), clamp=True)
            st.caption(f"{LABELS[mkey]}　PSNR **{psnr:.2f}** dB ／ "
                       f"SSIM **{ssim:.4f}**（{dt:.2f}s）")

# ---------- 指标汇总表 ----------
st.divider()
st.subheader("评价指标汇总")
import pandas as pd  # noqa: E402

df = pd.DataFrame(table_rows)
st.dataframe(df, hide_index=True, width="stretch")

# 速度对比结论（单档模式时显示）
if len(rates_to_show) == 1 and show_cmp:
    row_ours = next(r for r in table_rows if r["方法"].startswith("本方法"))
    row_omp = next(r for r in table_rows if r["方法"].startswith("OMP"))
    speedup = row_omp["耗时 (秒)"] / max(row_ours["耗时 (秒)"], 1e-6)
    st.success(
        f"本方法在该采样率下：PSNR 领先 OMP **{row_ours['PSNR (dB)'] - row_omp['PSNR (dB)']:+.2f} dB**，"
        f"重建速度约为 OMP 的 **{speedup:.0f} 倍**（{row_ours['耗时 (秒)']}s vs {row_omp['耗时 (秒)']}s）"
    )

# ---------- 模型对比曲线（折线图） ----------
st.divider()
st.subheader("📈 模型对比曲线（折线图）")
st.caption(
    "把 Set11 全量评估结果（11 张测试图逐图平均，口径见 README 第 4 节）画成折线，"
    "直接看出各方法之间的数据差距；鼠标悬停可读具体数值，点击图例可单独显示/隐藏某条曲线。"
)


@st.cache_data(show_spinner=False)
def _load_tables():
    return load_result_tables()


@st.cache_data(show_spinner=False)
def _load_logs():
    return load_train_logs()


res_data = _load_tables()
train_logs = _load_logs()
cur_rate = None if rate_choice == "三档全对比" else RATES[rate_choice]   # 当前档位（虚线参照）
mark_values = st.checkbox("在折线点上标注数值", value=False, key="curve_values")

# 当前图像在三档采样率下的实测结果（复用上方已有缓存，切档时可即时看差距）
live_data: dict = {}
with st.spinner("正在汇总当前图像的实测曲线…"):
    for _mkey in methods:
        for _rate in RATES.values():
            _rec, _p, _s, _dt = get_result(img_np, _rate, _mkey, device)
            live_data.setdefault(_mkey, {})[_rate] = {"psnr": _p, "ssim": _s}

tab_rate, tab_img, tab_ab, tab_tr, tab_live = st.tabs(
    ["① 指标–采样率", "② 逐图像对比", "③ 消融实验", "④ 训练收敛", "⑤ 本图实测"])

# --- ① 指标–采样率 ---
with tab_rate:
    if res_data:
        st.plotly_chart(
            fig_metric_vs_rate(res_data, "psnr", focus_rate=cur_rate,
                               show_values=mark_values),
            theme=None, width="stretch")
        st.plotly_chart(
            fig_metric_vs_rate(res_data, "ssim", focus_rate=cur_rate,
                               show_values=mark_values),
            theme=None, width="stretch")
        st.caption(
            "横轴为采样率，纵轴为 Set11 全量 11 张图的平均指标；灰色虚线标出当前选中的档位。"
            "曲线上方间距越大，说明方法间差距越明显——本方法在三档上均高于其余方法。"
        )
    else:
        st.info("results/ 下未找到评估数据，请先运行 scripts/run_baseline.py 与 scripts/eval_model.py")

# --- ② 逐图像对比 ---
with tab_img:
    if res_data:
        c1, c2 = st.columns([3, 2])
        with c1:
            rate_pick = st.radio(
                "采样率档位", [0.10, 0.25, 0.50], horizontal=True,
                index=[0.10, 0.25, 0.50].index(cur_rate) if cur_rate else 1,
                format_func=lambda v: f"{int(v * 100)}%", key="curve_img_rate")
        with c2:
            metric_pick = st.radio("对比指标", ["PSNR", "SSIM"], horizontal=True,
                                   key="curve_img_metric")
        st.plotly_chart(
            fig_metric_per_image(res_data, rate_pick, metric_pick.lower(),
                                 highlight_image=img_title if source == "使用示例图" else None,
                                 show_values=mark_values),
            theme=None, width="stretch")
        st.caption(
            "横轴为 Set11 的 11 张测试图（纹理复杂度差异大），纵轴为指标；"
            "灰色虚线标出演示系统当前图像在同档采样率下的位置。"
        )

# --- ③ 消融实验 ---
with tab_ab:
    if res_data:
        st.plotly_chart(fig_ablation(res_data, 0.25, "psnr"),
                        theme=None, width="stretch")
        st.plotly_chart(fig_ablation(res_data, 0.25, "ssim"),
                        theme=None, width="stretch")
        st.caption(
            "25% 采样率下依次移除各个模块后的性能落差。灰色虚线为对比方法在同一档采样率下的水平；"
            "「无可学习采样」一步掉 3 dB 以上，说明可学习采样是本网络贡献最大的模块。"
        )

# --- ④ 训练收敛 ---
with tab_tr:
    if train_logs:
        avail = list(train_logs)
        default = [n for n in ("ours_r10", "ours_r25", "ours_r50", "reconnet_r25")
                   if n in avail]
        picked = st.multiselect("选择要对比的训练曲线", avail, default=default,
                                format_func=pretty_log_name, key="curve_train_pick")
        drop_anomaly = st.checkbox("隐藏训练异常点（loss 突增的发散点）",
                                   value=True, key="curve_train_clip")
        f_tr = fig_train_curves(train_logs, picked, clip_outliers=drop_anomaly)
        st.plotly_chart(f_tr, theme=None, width="stretch")
        n_drop = (f_tr.layout.meta or {}).get("dropped", 0)
        st.caption(
            "纵轴为训练损失 MSE（对数轴），越低越好。"
            + (f"已隐藏 {n_drop} 个训练异常点（loss 超过中位数 20 倍，属训练发散段，"
               "与训练日志中的异常记录一致）。" if n_drop else "")
        )
    else:
        st.info("results/ 下未找到训练日志（train_*_log.csv）")

# --- ⑤ 本图实测 ---
with tab_live:
    st.caption(
        f"当前图像 `{img_title}` 在 10% / 25% / 50% 三档采样率下的实测结果"
        "（与本页上方重建结果同源，非离线数据）。"
    )
    st.plotly_chart(fig_live_vs_rate(live_data, "psnr", focus_rate=cur_rate),
                    theme=None, width="stretch")
    st.plotly_chart(fig_live_vs_rate(live_data, "ssim", focus_rate=cur_rate),
                    theme=None, width="stretch")
    _r = cur_rate or 0.25
    _p_ours = (live_data.get("ours", {}).get(_r) or {}).get("psnr")
    _p_base = max(((live_data[m].get(_r) or {}).get("psnr", -1e9)
                   for m in ("omp", "ista") if m in live_data), default=None)
    if _p_ours is not None and _p_base is not None and _p_base > -1e8:
        st.success(
            f"当前图像 @ {int(_r * 100)}% 采样率：本方法 PSNR 比最好的传统方法高 "
            f"**{_p_ours - _p_base:+.2f} dB**（{_p_ours:.2f} vs {_p_base:.2f} dB）"
        )

# ---------- 可学习测量核 ----------
with st.expander("🔍 查看「可学习采样」—— 网络自己学到的测量方式"):
    st.caption(
        "本方法的采样网络用 M 个 33×33 卷积核实现测量（代替传统固定的随机矩阵）。"
        "下图随机展示其中 16 个测量核，是网络为'保留对重建最有价值的信息'而自己学出来的模式。"
    )
    try:
        grid = get_kernel_grid(rates_to_show[0], device)
        st.image(grid, caption=f"测量核示例（{int(rates_to_show[0]*100)}% 档）",
                 width=420, clamp=True)
    except Exception as e:  # noqa: BLE001
        st.warning(f"测量核可视化不可用：{e}")

st.divider()
st.caption("毕业设计演示系统 ｜ 算法：三级级联网络（可学习采样 + 初始重建 + 多尺度残差/注意力）"
           " ｜ 传统对照：OMP、ISTA(FISTA) ｜ 对比曲线：src/plotting.py（Plotly 交互折线图）")
