# -*- coding: utf-8 -*-
"""模型对比折线图（演示系统增强功能）

把 results/ 目录下已有的评估产物画成折线图，用于在演示系统里直观展示
各方法之间的"数据差距"。本模块只读数据、不做任何推理，因此不会拖慢演示。

数据来源（均已存在，脚本自动发现）：
    results/baseline_set11.csv       传统方法 OMP / ISTA（逐图逐档）
    results/deep_<model>_set11.csv   深度学习模型（ours / reconnet / 消融变体）
    results/train_*_log.csv          训练日志（epoch–loss）

对外接口：
    load_result_tables()      评估数据 → {method: {rate: {image: (psnr, ssim)}}}
    load_train_logs()         训练日志 → {name: {"epoch": [...], "loss": [...]}}
    fig_metric_vs_rate()      指标–采样率折线（多方法横向对比）
    fig_metric_per_image()    逐图像折线（同档采样率下的逐图差异）
    fig_ablation()            消融折线（各模块贡献落差）
    fig_train_curves()        训练收敛折线
    fig_live_vs_rate()        演示系统当前图像的实测折线

配色与图例风格对齐经典论文插图（tab10 系），并支持在图中用灰色虚线
标出"当前档位 / 当前图像"作为参照。
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import plotly.graph_objects as go

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# ----------------------------------------------------------------- 方法元信息
LABELS = {
    "ours": "本方法（OursNet）",
    "reconnet": "ReconNet",
    "OMP": "OMP（传统）",
    "ISTA": "ISTA（传统）",
    "ours_noMS": "无多尺度残差",
    "ours_noLS": "无可学习采样",
    "ours_noAttn": "无通道注意力",
}
COLORS = {
    "ours": "#D62728",        # 红：本方法，视觉上最突出
    "reconnet": "#FF7F0E",    # 橙
    "ISTA": "#1F77B4",        # 蓝
    "OMP": "#2CA02C",         # 绿
    "ours_noMS": "#9467BD",   # 紫
    "ours_noAttn": "#17BECF",  # 青
    "ours_noLS": "#8C564B",   # 棕
}
ORDER = ["ours", "reconnet", "ISTA", "OMP", "ours_noMS", "ours_noAttn", "ours_noLS"]
FALLBACK = ["#7F7F7F", "#BCBD22", "#E377C2", "#17BECF", "#8C564B"]
IMAGE_ORDER = ["barbara", "boats", "cameraman", "fingerprint", "flinstones",
               "foreman", "house", "lena256", "Monarch", "Parrots", "peppers256"]
METRIC_META = {"psnr": ("PSNR (dB)", "psnr"), "ssim": ("SSIM", "ssim")}

# 消融曲线的展示顺序（按性能从高到低，折线呈"下坡"更直观）
ABLATION_ORDER = ["ours", "ours_noMS", "ours_noAttn", "ours_noLS"]


def label_of(method: str) -> str:
    return LABELS.get(method, method)


def _color_of(method: str, idx: int = 0) -> str:
    return COLORS.get(method, FALLBACK[idx % len(FALLBACK)])


def _sorted_methods(methods) -> list[str]:
    known = [m for m in ORDER if m in methods]
    rest = sorted(m for m in methods if m not in ORDER)
    return known + rest


# =============================================================== 读取评估数据
def load_result_tables(results_dir: Path | str = RESULTS_DIR) -> dict:
    """读取 results/ 下全部评估 CSV。

    返回 {method: {rate: {image: {"psnr": float, "ssim": float, "time_s": float}}}}
    """
    results_dir = Path(results_dir)
    data: dict = {}
    files = [results_dir / "baseline_set11.csv"]
    files += sorted(results_dir.glob("deep_*_set11.csv"))
    for f in files:
        if not f.exists():
            continue
        with open(f, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                method = (row.get("algorithm") or "").strip()
                if not method:
                    continue
                rate = float(row["rate"])
                data.setdefault(method, {}).setdefault(rate, {})[row["image"]] = {
                    "psnr": float(row["psnr"]),
                    "ssim": float(row["ssim"]),
                    "time_s": float(row.get("time_s") or 0.0),
                }
    return data


def mean_metric(data: dict, method: str, rate: float, metric: str = "psnr"):
    """某方法在某档采样率下的逐图平均指标；无数据返回 None。"""
    d = data.get(method, {}).get(rate) or {}
    if not d:
        return None
    return sum(v[metric] for v in d.values()) / len(d)


def available_rates(data: dict) -> list[float]:
    rates = sorted({r for m in data.values() for r in m})
    return rates


def n_images(data: dict, method: str, rate: float) -> int:
    return len(data.get(method, {}).get(rate) or {})


# =============================================================== 读取训练日志
def load_train_logs(results_dir: Path | str = RESULTS_DIR) -> dict:
    """读取训练日志 → {name: {"epoch": [...], "loss": [...], "time": [...]}}

    文件名形如 train_<name>_log.csv；带时间戳（_1013_0435）的是精调阶段日志，
    与主日志重复，自动跳过。
    """
    results_dir = Path(results_dir)
    logs: dict = {}
    for p in sorted(results_dir.glob("train_*_log.csv")):
        name = p.stem[len("train_"):-len("_log")]
        if re.search(r"_\d{4}_\d{4}$", name):
            continue
        ep, loss, ts = [], [], []
        with open(p, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                try:
                    ep.append(int(float(row["epoch"])))
                    loss.append(float(row["loss"]))
                    ts.append(float(row.get("epoch_time_s") or 0.0))
                except (KeyError, ValueError):
                    continue
        if ep:
            logs[name] = {"epoch": ep, "loss": loss, "time": ts}
    return logs


def _pretty_log_name(name: str) -> str:
    m = re.match(r"(ours_noMS|ours_noLS|ours_noAttn|ours|reconnet)_r(\d+)$", name)
    if m:
        base, rate = m.group(1), m.group(2)
        return f"{label_of(base)} @ {rate}%"
    return name


# 对外别名（界面里用于下拉框显示）
pretty_log_name = _pretty_log_name


# =============================================================== 绘图通用设置
def _common_layout(fig: go.Figure, xlab: str, ylab: str, title: str,
                   focus_x=None, focus_text: str = "", categorical: bool = False):
    """统一的坐标轴/图例/参考虚线样式。"""
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15)),
        xaxis_title=xlab,
        yaxis_title=ylab,
        template="plotly_white",
        font=dict(family="Microsoft YaHei, SimHei, Arial, sans-serif", size=13),
        margin=dict(l=70, r=30, t=60, b=60),
        hovermode="x unified",
        legend=dict(borderwidth=1, bordercolor="#CCCCCC", bgcolor="rgba(255,255,255,0.9)",
                    x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EAEAEA", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EAEAEA", zeroline=False)
    if focus_x is not None:
        try:
            fig.add_vline(x=focus_x, line_dash="dash", line_color="#888888",
                          line_width=1.4,
                          annotation_text=focus_text or None,
                          annotation_position="top",
                          annotation_font=dict(size=12, color="#555555"))
        except Exception:  # noqa: BLE001  分类轴极端情况下 add_vline 可能失败
            pass
    return fig


def _add_metric_labels(fig: go.Figure, x, y, decimals: int = 2, dy: float = 0.0):
    """在折线点旁标注数值（可选）。"""
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="text",
        text=[f"{v:.{decimals}f}" for v in y],
        textposition="top center",
        textfont=dict(size=11, color="#444444"),
        showlegend=False, hoverinfo="skip",
    ))


# =============================================================== 图 1：指标–采样率
def fig_metric_vs_rate(data: dict, metric: str = "psnr", methods=None,
                       rates=None, focus_rate: float | None = 0.25,
                       show_values: bool = False,
                       min_points: int = 2) -> go.Figure:
    """指标–采样率折线：横轴采样率，纵轴平均 PSNR/SSIM，一条线一个方法。

    min_points：少于该点数的序列不画（消融变体只在 25% 档有数据，
    放进采样率曲线里会变成孤立的单点，故默认过滤掉）。
    """
    ylab = f"平均 {METRIC_META[metric][0]}"
    methods = _sorted_methods(methods or list(data.keys()))
    rates = rates or available_rates(data)
    xs = [round(r * 100) for r in rates]

    series = []
    for m in methods:
        ys, use_x = [], []
        for r, x in zip(rates, xs):
            v = mean_metric(data, m, r, metric)
            if v is not None:
                ys.append(v)
                use_x.append(x)
        if len(ys) >= min_points:
            series.append((m, use_x, ys))

    fig = go.Figure()
    for i, (m, use_x, ys) in enumerate(series):
        fig.add_trace(go.Scatter(
            x=use_x, y=ys, name=label_of(m), mode="lines+markers",
            line=dict(color=_color_of(m, i), width=2.4),
            marker=dict(size=9, symbol="circle",
                        line=dict(width=1.2, color="white")),
            hovertemplate=(f"<b>{label_of(m)}</b><br>采样率 %{{x}}%<br>"
                           f"{ylab} %{{y:.4f}}<extra></extra>"
                           if metric == "ssim" else
                           f"<b>{label_of(m)}</b><br>采样率 %{{x}}%<br>"
                           f"{ylab} %{{y:.2f}}<extra></extra>"),
        ))
        if show_values:
            _add_metric_labels(fig, use_x, ys, decimals=4 if metric == "ssim" else 2)

    fig.update_xaxes(tickmode="array", tickvals=xs, ticksuffix="%")
    pad = (max(xs) - min(xs)) * 0.08
    fig.update_xaxes(range=[min(xs) - pad, max(xs) + pad])
    if metric == "psnr" and series:
        all_v = [v for _, _, ys in series for v in ys]
        lo, hi = min(all_v), max(all_v)
        span = max(hi - lo, 1.0)
        fig.update_yaxes(range=[lo - span * 0.35, hi + span * 0.25])
    fig.layout.meta = {"methods": [m for m, _, _ in series]}
    return _common_layout(
        fig, "采样率（测量值 / 像素）", ylab,
        f"各方法 {ylab} 随采样率变化（Set11 逐图平均）",
        focus_x=focus_rate * 100 if focus_rate else None,
        focus_text=f"当前 {int(focus_rate*100)}%" if focus_rate else "")


# =============================================================== 图 2：逐图像折线
def fig_metric_per_image(data: dict, rate: float = 0.25, metric: str = "psnr",
                         methods=None, highlight_image: str | None = None,
                         show_values: bool = False) -> go.Figure:
    """逐图像折线：横轴测试图，纵轴指标，一条线一个方法（同一档采样率）。"""
    ylab = METRIC_META[metric][0]
    methods = _sorted_methods(methods or list(data.keys()))
    # 只保留在本档下确有数据的图，顺序按 IMAGE_ORDER
    imgs = IMAGE_ORDER[:]
    seen = {img for m in methods for img in (data.get(m, {}).get(rate) or {})}
    extra = sorted(seen - set(imgs))
    imgs = [i for i in imgs if i in seen] + extra

    fig = go.Figure()
    for i, m in enumerate(methods):
        d = data.get(m, {}).get(rate) or {}
        if not d:
            continue
        use_x = [img for img in imgs if img in d]
        ys = [d[img][metric] for img in use_x]
        fig.add_trace(go.Scatter(
            x=use_x, y=ys, name=label_of(m), mode="lines+markers",
            line=dict(color=_color_of(m, i), width=2.0),
            marker=dict(size=8, line=dict(width=1.2, color="white")),
            hovertemplate=(f"<b>{label_of(m)}</b><br>%{{x}}<br>"
                           f"{ylab} %{{y:.4f}}<extra></extra>"
                           if metric == "ssim" else
                           f"<b>{label_of(m)}</b><br>%{{x}}<br>"
                           f"{ylab} %{{y:.2f}}<extra></extra>"),
        ))
        if show_values:
            _add_metric_labels(fig, use_x, ys, decimals=3 if metric == "ssim" else 2)

    hl = highlight_image if highlight_image in imgs else None
    return _common_layout(
        fig, "测试图像（Set11）", ylab,
        f"逐图像 {ylab} 对比（采样率 {int(rate*100)}%）",
        focus_x=hl, focus_text="演示系统当前图像" if hl else "")


# =============================================================== 图 3：消融
def fig_ablation(data: dict, rate: float = 0.25, metric: str = "psnr",
                 reference: tuple = ("reconnet", "ISTA", "OMP")) -> go.Figure:
    """消融折线：横轴为模型变体（按性能降序），纵轴指标。

    同时用灰色虚线画出对比方法在本档采样率下的水平，直观看出模块贡献。
    """
    ylab = METRIC_META[metric][0]
    variants = [m for m in ABLATION_ORDER if mean_metric(data, m, rate, metric) is not None]
    xs = [label_of(m) for m in variants]
    ys = [mean_metric(data, m, rate, metric) for m in variants]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, name="自研网络消融", mode="lines+markers",
        line=dict(color=COLORS["ours"], width=2.6, shape="linear"),
        marker=dict(size=12, symbol="diamond", color=COLORS["ours"],
                    line=dict(width=1.4, color="white")),
        hovertemplate=f"<b>%{{x}}</b><br>{ylab} %{{y:.4f}}<extra></extra>",
    ))
    if metric == "psnr":
        _add_metric_labels(fig, xs, ys, decimals=2)
    # 完整模型作为水平基准
    if variants and variants[0] == "ours":
        fig.add_hline(y=ys[0], line_dash="dot", line_color="#D62728",
                      line_width=1.2, opacity=0.55)
    # 对比方法参考线
    for k, m in enumerate(reference):
        v = mean_metric(data, m, rate, metric)
        if v is None:
            continue
        fig.add_hline(y=v, line_dash="dash", line_color=_color_of(m, k + 3),
                      line_width=1.6,
                      annotation_text=f"{label_of(m)} {v:.2f}" if metric == "psnr"
                      else f"{label_of(m)} {v:.4f}",
                      annotation_position="right",
                      annotation_font=dict(size=11, color=_color_of(m, k + 3)))

    return _common_layout(
        fig, "模型变体（自上而下依次移除模块）", ylab,
        f"消融实验：各模块移除后的性能落差（采样率 {int(rate*100)}%）")


# =============================================================== 图 4：训练收敛
def fig_train_curves(logs: dict, names=None, log_y: bool = True,
                     clip_outliers: bool = True) -> go.Figure:
    """训练收敛折线：横轴 epoch，纵轴 loss（默认对数轴）。

    日志里存在个别训练发散点（loss 突然放大几个数量级），画在对数轴上会
    把整条曲线压成一条直线，故默认剔除：loss > 中位数 × 20 视为异常点，
    剔除数量记在 fig.layout.meta["dropped"] 里，界面上如实说明。
    """
    names = names or list(logs.keys())
    names = [n for n in names if n in logs]
    fig = go.Figure()
    dropped = 0
    for n in names:
        rec = logs[n]
        ep, loss = list(rec["epoch"]), list(rec["loss"])
        if clip_outliers and loss:
            med = sorted(loss)[len(loss) // 2]
            keep = [(e, v) for e, v in zip(ep, loss)
                    if med <= 0 or v <= med * 20]
            dropped += len(ep) - len(keep)
            ep, loss = [e for e, _ in keep], [v for _, v in keep]
        fig.add_trace(go.Scatter(
            x=ep, y=loss, name=_pretty_log_name(n),
            mode="lines", line=dict(width=2.0),
            hovertemplate=(f"<b>{_pretty_log_name(n)}</b><br>"
                           "epoch %{x}<br>loss %{y:.6f}<extra></extra>"),
        ))
    fig.update_yaxes(type="log" if log_y else "linear")
    fig.layout.meta = {"dropped": dropped}
    return _common_layout(fig, "训练轮次（epoch）", "训练损失 MSE（对数轴）",
                          "训练收敛曲线对比：loss 随 epoch 下降")


# =============================================================== 图 5：本图实测
def fig_live_vs_rate(live: dict, metric: str = "psnr", rates=None,
                     focus_rate: float | None = None) -> go.Figure:
    """演示系统当前图像的实测折线。

    live = {method: {rate: {"psnr": float, "ssim": float}}}
    """
    ylab = METRIC_META[metric][0]
    rates = rates or sorted({r for m in live.values() for r in m})
    xs = [round(r * 100) for r in rates]
    methods = [m for m in ORDER if m in live] + [m for m in live if m not in ORDER]

    fig = go.Figure()
    for i, m in enumerate(methods):
        ys = [(live[m].get(r) or {}).get(metric) for r in rates]
        pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pairs:
            continue
        use_x, use_y = zip(*pairs)
        fig.add_trace(go.Scatter(
            x=list(use_x), y=list(use_y), name=label_of(m), mode="lines+markers",
            line=dict(color=_color_of(m, i), width=2.6),
            marker=dict(size=11, line=dict(width=1.4, color="white")),
            hovertemplate=(f"<b>{label_of(m)}</b><br>采样率 %{{x}}%<br>"
                           f"{ylab} %{{y:.4f}}<extra></extra>"
                           if metric == "ssim" else
                           f"<b>{label_of(m)}</b><br>采样率 %{{x}}%<br>"
                           f"{ylab} %{{y:.2f}}<extra></extra>"),
        ))
        _add_metric_labels(fig, list(use_x), list(use_y),
                           decimals=4 if metric == "ssim" else 2)

    fig.update_xaxes(tickmode="array", tickvals=xs, ticksuffix="%")
    return _common_layout(
        fig, "采样率（测量值 / 像素）", ylab,
        "当前图像实测：各方法指标随采样率变化",
        focus_x=focus_rate * 100 if focus_rate else None,
        focus_text="当前选中档位" if focus_rate else "")
