"""
plot_run_report.py — 整理 output/metrics.csv + run_meta.json

标注策略（综合：含义 / 美观 / 可读 / 后续 LLM 也适用）：
  1) 默认不「逢点必标」
  2) 平台期（数值几乎不变）：只标该平台的代表点（段首，必要时段尾）
  3) 变化期：标起点、终点、全局 max/min、明显转折
  4) 仅 fig2 在「起点多条线数值挤在一起」时，对起点 label 拉短引线错开
  5) legend loc='best'；dpi=300；无方框（引线仅用于 fig2 起点特例）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "avg_opinion": "#6469BD",
    "polarization": "#D56959",
    "emotional_contagion": "#C73AD6",
    "message_count": "#52BE80",
    "negative_emotion": "#F5B041",
    "distortion_level": "#5DADE2",
    "cross_group_forward": "#4898FF",
    "cross_group_spread": "#7A5CFF",
    "mute_count": "#B45309",
    "group_heat_max": "#CA8A04",
    "group_negative_max": "#B91C1C",
    "upward_forward": "#1D4ED8",
    "lateral_forward": "#F59E0B",
    "downward_forward": "#94A3B8",
    "recovery_time": "#8A7067",
    "step_seconds": "#D56959",
    "init": "#5DADE2",
    "run": "#D56959",
    "total": "#6469BD",
}

DPI = 300
LW = 1.6
MS = 3.5

# 指标含义 → 标注偏好（在通用算法上微调）
METRIC_HINTS = {
    # 观点均值：关注起终与是否偏移
    "avg_opinion": {"prefer": ("start", "end", "max", "min", "turns"), "max_labels": 6},
    # 极化：常较平稳；变了才多标
    "polarization": {"prefer": ("start", "end", "max", "min", "turns"), "max_labels": 5},
    # 负面占比：关注越过阈值附近的变化
    "negative_emotion": {"prefer": ("start", "end", "max", "min", "turns"), "max_labels": 6},
    # 失真：关注尖峰
    "distortion_level": {"prefer": ("start", "end", "max", "turns"), "max_labels": 6},
    # 消息量：峰值与起终最重要
    "message_count": {"prefer": ("start", "end", "max", "min", "turns"), "max_labels": 7},
    # 跨群累计：常单调，起终 + 明显跳变
    "cross_group_forward": {"prefer": ("start", "end", "max", "turns"), "max_labels": 6},
    "cross_group_spread": {"prefer": ("start", "end", "max", "turns"), "max_labels": 6},
    "mute_count": {"prefer": ("start", "end", "max", "turns"), "max_labels": 5},
    "group_heat_max": {"prefer": ("start", "end", "max", "turns"), "max_labels": 6},
    "group_negative_max": {"prefer": ("start", "end", "max", "turns"), "max_labels": 6},
    # 恢复时长：常为 0 平台；一旦跳变必须标出
    "recovery_time": {"prefer": ("start", "end", "max", "turns"), "max_labels": 4},
    # 情绪传染：波动小，标极值与起终即可
    "emotional_contagion": {"prefer": ("start", "end", "max", "min"), "max_labels": 4},
    # 每步耗时：瓶颈=max 最关键
    "step_seconds": {"prefer": ("start", "end", "max", "min", "turns"), "max_labels": 6},
}


def _setup_cn_font() -> None:
    from matplotlib import font_manager as fm

    for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]:
        try:
            path = fm.findfont(name, fallback_to_default=False)
            if path and "DejaVu" not in path:
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                return
        except Exception:
            continue


def load_inputs(metrics_path: Path, meta_path: Path):
    df = pd.read_csv(metrics_path)
    if "tick" not in df.columns:
        df = df.reset_index(drop=True)
        df.insert(0, "tick", np.arange(len(df)))

    for col in ("avg_opinion", "polarization", "emotional_contagion"):
        if col in df.columns and df[col].abs().max() > 2.0:
            df[col] = df[col].astype(float) / 32767.0

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return df, meta


def _rel_eps(y: np.ndarray) -> float:
    """相对平坦判定阈值：随数据量级自适应（LLM 变剧烈时自动变严/变松）。"""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) == 0:
        return 1e-9
    span = float(np.nanmax(y) - np.nanmin(y))
    scale = max(abs(float(np.nanmean(y))), abs(float(np.nanmax(y))), 1e-9)
    # 几乎水平：span 极小 → 大 eps；有明显变化 → eps 为 span 的一小部分
    if span < 1e-12:
        return max(1e-6, 0.002 * scale)
    return max(1e-9, 0.02 * span, 0.001 * scale)


def _plateau_reps(y: np.ndarray) -> list[int]:
    """每个平台段只取段首（段尾若与段首不同 index 再取段尾）。"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return []
    eps = _rel_eps(y)
    reps: list[int] = []
    i = 0
    while i < n:
        if np.isnan(y[i]):
            i += 1
            continue
        j = i + 1
        while j < n and (np.isnan(y[j]) or abs(y[j] - y[i]) <= eps):
            if np.isnan(y[j]):
                break
            j += 1
        # 平台 [i, j)
        reps.append(i)
        if j - 1 > i:
            reps.append(j - 1)
        i = j if j > i else i + 1
    # 去重保序
    out = []
    for k in reps:
        if k not in out:
            out.append(k)
    return out


def _local_turns(y: np.ndarray) -> list[int]:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return []
    eps = _rel_eps(y)
    turns = []
    for i in range(1, n - 1):
        if np.isnan(y[i]) or np.isnan(y[i - 1]) or np.isnan(y[i + 1]):
            continue
        left = y[i] - y[i - 1]
        right = y[i + 1] - y[i]
        # 方向改变且幅度足够
        if left * right < 0 and (abs(left) > eps or abs(right) > eps):
            turns.append(i)
        # 尖峰：明显高于两侧
        elif y[i] >= y[i - 1] + eps and y[i] >= y[i + 1] + eps:
            turns.append(i)
        elif y[i] <= y[i - 1] - eps and y[i] <= y[i + 1] - eps:
            turns.append(i)
    return turns


def select_label_indices(y: np.ndarray, metric: str) -> list[int]:
    """
    根据序列形态选标注点：
      - 全平台（几乎不变）：只标 1～2 个代表点
      - 有变化：start/end + max/min + 转折（受 max_labels 限制）
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    valid = np.where(~np.isnan(y))[0]
    if len(valid) == 0:
        return []

    hint = METRIC_HINTS.get(metric, {"prefer": ("start", "end", "max", "min", "turns"), "max_labels": 6})
    prefer = set(hint.get("prefer", ()))
    max_labels = int(hint.get("max_labels", 6))

    span = float(np.nanmax(y) - np.nanmin(y))
    eps = _rel_eps(y)
    nearly_flat = span <= max(3 * eps, 1e-9)

    chosen: list[int] = []

    def add(i: int) -> None:
        if i not in chosen and 0 <= i < n and not np.isnan(y[i]):
            chosen.append(i)

    if nearly_flat:
        # 水平线：标段首；若很长也标段尾（避免 LLM 后来突然变化时你还以为整段都标满）
        add(int(valid[0]))
        if len(valid) > 1:
            add(int(valid[-1]))
        return chosen[:2]

    # 非平坦
    if "start" in prefer:
        add(int(valid[0]))
    if "end" in prefer:
        add(int(valid[-1]))
    if "max" in prefer:
        add(int(valid[np.nanargmax(y[valid])]))
    if "min" in prefer:
        add(int(valid[np.nanargmin(y[valid])]))

    if "turns" in prefer:
        # 转折按偏离均值排序，优先大拐弯
        mean = float(np.nanmean(y))
        turns = sorted(_local_turns(y), key=lambda i: abs(y[i] - mean), reverse=True)
        for i in turns:
            add(i)
            if len(chosen) >= max_labels:
                break

    # 平台代表：补上「进入新平台」的点（变化后的稳定段）
    for i in _plateau_reps(y):
        add(i)
        if len(chosen) >= max_labels + 2:
            break

    # 最终裁剪：优先保留 start/end/max/min
    if len(chosen) > max_labels:
        must = []
        if "start" in prefer:
            must.append(int(valid[0]))
        if "end" in prefer:
            must.append(int(valid[-1]))
        if "max" in prefer:
            must.append(int(valid[np.nanargmax(y[valid])]))
        if "min" in prefer:
            must.append(int(valid[np.nanargmin(y[valid])]))
        rest = [i for i in chosen if i not in must]
        chosen = []
        for i in must + rest:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= max_labels:
                break

    return chosen


def _annotate(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    indices: list[int],
    color: str,
    *,
    side: str = "above",
    fmt: str = "{:.2f}",
    fontsize: int = 8,
    leader: dict[int, tuple[float, float]] | None = None,
) -> None:
    """
    leader: {index: (dx, dy)} 仅对这些点拉短引线（用于 fig2 起点错开）。
    其它点：无框无线，贴在 marker 上/下。
    """
    for i in indices:
        yi = float(y[i])
        if np.isnan(yi):
            continue
        if leader and i in leader:
            dx, dy = leader[i]
            ax.annotate(
                fmt.format(yi),
                xy=(x[i], yi),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="bottom" if dy >= 0 else "top",
                fontsize=fontsize,
                color=color,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7, shrinkA=0, shrinkB=2),
            )
        else:
            dy = 7 if side == "above" else -7
            ax.annotate(
                fmt.format(yi),
                xy=(x[i], yi),
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                va="bottom" if side == "above" else "top",
                fontsize=fontsize,
                color=color,
                fontweight="bold",
            )


def _pad_ylim(ax, ys: list[np.ndarray], label_side: str = "both", frac: float = 0.14) -> None:
    vals = []
    for y in ys:
        y = np.asarray(y, dtype=float)
        vals.extend(y[~np.isnan(y)].tolist())
    if not vals:
        return
    lo, hi = float(min(vals)), float(max(vals))
    if lo == hi:
        pad = max(abs(lo) * 0.12, 0.05)
        ax.set_ylim(lo - pad, hi + pad)
        return
    span = hi - lo
    pad_lo = span * frac if label_side in ("both", "below") else span * 0.05
    pad_hi = span * frac if label_side in ("both", "above") else span * 0.05
    ax.set_ylim(lo - pad_lo, hi + pad_hi)


def fig1_bounded_trends(df: pd.DataFrame, out: Path, meta: dict) -> None:
    tick = df["tick"].to_numpy()
    series = [
        ("avg_opinion", "-", "o", "above", "{:.2f}"),
        ("polarization", "-", "s", "above", "{:.2f}"),
        ("negative_emotion", "--", "^", "below", "{:.2f}"),
        ("distortion_level", ":", "D", "above", "{:.2f}"),
    ]
    series = [s for s in series if s[0] in df.columns]
    if not series:
        return

    fig, ax = plt.subplots(figsize=(12.5, 6.8), dpi=DPI)
    ys = []
    for name, ls, mk, side, fmt in series:
        y = df[name].to_numpy(dtype=float)
        ys.append(y)
        ax.plot(
            tick, y, color=COLORS[name], linestyle=ls, marker=mk,
            markersize=MS, linewidth=LW, alpha=0.92, label=name,
            markevery=max(1, len(tick) // 20),
        )
        idxs = select_label_indices(y, name)
        _annotate(ax, tick, y, idxs, COLORS[name], side=side, fmt=fmt)

    _pad_ylim(ax, ys, label_side="both", frac=0.14)
    ax.axhline(0.0, color="#888", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.axhline(0.5, color="#E74C3C", linestyle="--", linewidth=1.1, alpha=0.75)
    ax.set_title(
        f"Bounded Metrics over Ticks | n={meta.get('n_agents','?')}, "
        f"steps={meta.get('n_steps', len(df))}, llm={meta.get('llm_provider','?')}",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.set_xlabel("tick", fontsize=11, fontweight="bold")
    ax.set_ylabel("value", fontsize=11, fontweight="bold")
    ax.grid(axis="both", linestyle="dotted", color="black", alpha=0.30)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "fig1_bounded_trends.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig2_count_trends(df: pd.DataFrame, out: Path, meta: dict) -> None:
    """计数类；智能标注。仅起点拥挤时对起点拉短引线错开。"""
    tick = df["tick"].to_numpy()
    series = [
        ("message_count", "-", "o", "{:.0f}"),
        ("cross_group_forward", "--", "s", "{:.0f}"),
        ("cross_group_spread", ":", "D", "{:.0f}"),
        ("recovery_time", ":", "^", "{:.1f}"),
    ]
    series = [s for s in series if s[0] in df.columns]
    if not series:
        return

    start_vals = []
    for name, _, _, _ in series:
        v = float(df[name].iloc[0])
        if not np.isnan(v):
            start_vals.append(v)
    start_crowded = False
    if len(start_vals) >= 2:
        start_crowded = (max(start_vals) - min(start_vals)) <= max(0.5, 0.05 * (max(abs(v) for v in start_vals) + 1))

    # 起点引线偏移：左/上/右错开，避免粘连
    start_leaders = {
        0: (0, 14),
        1: (-16, 22),
        2: (16, 22),
    }

    fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=DPI)
    ys = []
    for slot, (name, ls, mk, fmt) in enumerate(series):
        y = df[name].to_numpy(dtype=float)
        ys.append(y)
        ax.plot(
            tick, y, color=COLORS[name], linestyle=ls, marker=mk,
            markersize=MS, linewidth=LW, alpha=0.92, label=name,
            markevery=max(1, len(tick) // 20),
        )
        idxs = select_label_indices(y, name)
        leader = None
        if start_crowded and 0 in idxs:
            leader = {0: start_leaders.get(slot, (0, 14))}
        _annotate(
            ax, tick, y, idxs, COLORS[name],
            side="above", fmt=fmt, leader=leader,
        )

    _pad_ylim(ax, ys, label_side="above", frac=0.16)
    ax.set_title(
        f"Count / Recovery Metrics over Ticks | n={meta.get('n_agents','?')}",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.set_xlabel("tick", fontsize=11, fontweight="bold")
    ax.set_ylabel("count", fontsize=11, fontweight="bold")
    ax.grid(axis="both", linestyle="dotted", color="black", alpha=0.30)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "fig2_count_trends.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig3_runtime_bars(meta: dict, out: Path) -> None:
    """单次 run 分段耗时：柱顶只标该柱秒数；sec/step 单独放角落，不叠在 total 上。"""
    init_s = float(meta.get("init_seconds", 0.0))
    run_s = float(meta.get("run_seconds", 0.0))
    total = float(meta.get("total_seconds", init_s + run_s))
    sps = float(meta.get("seconds_per_step", 0.0))

    labels = ["init", "run_loop", "total"]
    values = [init_s, run_s, total]
    colors = [COLORS["init"], COLORS["run"], COLORS["total"]]

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=DPI)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, alpha=0.88, edgecolor="#333",
                  linewidth=1.0, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("seconds", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Runtime Breakdown | n={meta.get('n_agents','?')}, "
        f"steps={meta.get('n_steps','?')}, seed={meta.get('random_seed','?')}",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.50)
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2, b.get_height(),
            f"{v:.3f}s", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    # sec/step 是派生量，不是 total 柱的第二行数据 → 角落说明
    ax.text(
        0.02, 0.98,
        f"avg sec/step = {sps:.4f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, fontweight="bold", color="#333",
    )
    fig.tight_layout()
    fig.savefig(out / "fig3_runtime_bars.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig4_metric_panels(df: pd.DataFrame, out: Path, meta: dict) -> None:
    metrics = [
        m for m in [
            "avg_opinion", "polarization", "message_count",
            "negative_emotion", "distortion_level", "cross_group_forward", "cross_group_spread",
            "emotional_contagion", "recovery_time",
        ]
        if m in df.columns
    ]
    if not metrics:
        return

    tick = df["tick"].to_numpy()
    n = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 3.8 * nrows), dpi=DPI)
    axes = np.atleast_1d(axes).ravel()

    for i, m in enumerate(metrics):
        ax = axes[i]
        y = df[m].to_numpy(dtype=float)
        ax.plot(tick, y, color=COLORS.get(m, "#333"), linewidth=1.35,
                marker="o", markersize=2.6, alpha=0.95, markevery=max(1, len(tick) // 15))
        ax.set_title(m, fontsize=11, fontweight="bold")
        ax.set_xlabel("tick", fontsize=9)
        ax.grid(axis="both", linestyle="dotted", color="black", alpha=0.28)
        if m == "avg_opinion":
            ax.axhline(0.0, color="#666", linestyle=":", linewidth=1.0, alpha=0.85)
        if m in ("negative_emotion", "distortion_level"):
            ax.axhline(0.5, color="#E74C3C", linestyle="--", linewidth=1.0, alpha=0.8)

        # 分面更小：再收紧一点标签数
        idxs = select_label_indices(y, m)[:4]
        fmt = "{:.3f}" if m in ("emotional_contagion", "step_seconds") else "{:.2f}"
        if m in ("message_count", "cross_group_forward", "cross_group_spread"):
            fmt = "{:.0f}"
        _annotate(ax, tick, y, idxs, COLORS.get(m, "#333"), side="above", fmt=fmt, fontsize=7)
        _pad_ylim(ax, [y], label_side="above", frac=0.16)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Per-Metric Panels | n={meta.get('n_agents','?')} · steps={meta.get('n_steps','?')}",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / "fig4_metric_panels.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig5_contagion_optional(df: pd.DataFrame, out: Path, meta: dict) -> None:
    if "emotional_contagion" not in df.columns:
        return
    tick = df["tick"].to_numpy()
    y = df["emotional_contagion"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=DPI)
    ax.plot(
        tick, y, color=COLORS["emotional_contagion"], linewidth=1.5,
        marker="o", markersize=3.0, alpha=0.95, label="emotional_contagion",
        markevery=max(1, len(tick) // 15),
    )
    mean_v = float(np.nanmean(y))
    ax.axhline(mean_v, color="#E74C3C", linestyle="--", linewidth=1.2, alpha=0.85,
               label=f"mean={mean_v:.4f}")
    idxs = select_label_indices(y, "emotional_contagion")
    _annotate(ax, tick, y, idxs, COLORS["emotional_contagion"], side="above", fmt="{:.4f}")
    _pad_ylim(ax, [y], label_side="above", frac=0.14)
    ax.set_title("Emotional Contagion over Ticks", fontsize=13, fontweight="bold")
    ax.set_xlabel("tick", fontsize=11, fontweight="bold")
    ax.set_ylabel("mean |Δarousal|", fontsize=11, fontweight="bold")
    ax.grid(axis="both", linestyle="dotted", color="black", alpha=0.30)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "fig5_emotional_contagion.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig7_by_agent_type(df: pd.DataFrame, out: Path, meta: dict) -> None:
    """老师要求：按角色分图，不要只画全网平均。"""
    role_cols = [
        ("avg_opinion_ORDINARY", "ORDINARY", "#6469BD"),
        ("avg_opinion_ACTIVE", "ACTIVE", "#D56959"),
        ("avg_opinion_RATIONAL", "RATIONAL", "#52BE80"),
        ("avg_opinion_CONTROLLER", "CONTROLLER", "#F5B041"),
    ]
    present = [(c, lab, col) for c, lab, col in role_cols if c in df.columns]
    if len(present) < 2:
        print("skip fig7_by_agent_type: missing avg_opinion_<ROLE> columns (请用新版 run_sim 重跑)")
        return

    tick = df["tick"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), dpi=DPI, sharex=True)

    ax = axes[0]
    for c, lab, col in present:
        y = df[c].to_numpy(dtype=float)
        ax.plot(tick, y, color=col, linewidth=LW, marker="o", markersize=MS,
                markevery=max(1, len(tick) // 12), label=lab, alpha=0.92)
    if "avg_opinion" in df.columns:
        ax.plot(tick, df["avg_opinion"].to_numpy(dtype=float), color="#333333",
                linewidth=1.2, linestyle="--", label="ALL(mean)", alpha=0.75)
    ax.axhline(0.0, color="#888", linestyle=":", linewidth=1.0)
    ax.set_ylabel("avg opinion", fontsize=11, fontweight="bold")
    ax.set_title("By AgentType · Opinion", fontsize=12, fontweight="bold")
    ax.grid(axis="both", linestyle="dotted", color="black", alpha=0.30)
    ax.legend(loc="best", fontsize=8, ncol=3, framealpha=0.9)

    ax2 = axes[1]
    neg_cols = [
        ("neg_emotion_ORDINARY", "ORDINARY", "#6469BD"),
        ("neg_emotion_ACTIVE", "ACTIVE", "#D56959"),
        ("neg_emotion_RATIONAL", "RATIONAL", "#52BE80"),
        ("neg_emotion_CONTROLLER", "CONTROLLER", "#F5B041"),
    ]
    neg_present = [(c, lab, col) for c, lab, col in neg_cols if c in df.columns]
    if neg_present:
        for c, lab, col in neg_present:
            y = df[c].to_numpy(dtype=float)
            ax2.plot(tick, y, color=col, linewidth=LW, marker="o", markersize=MS,
                     markevery=max(1, len(tick) // 12), label=lab, alpha=0.92)
    elif "negative_emotion" in df.columns:
        ax2.plot(tick, df["negative_emotion"].to_numpy(dtype=float),
                 color="#D56959", linewidth=LW, label="negative_emotion")
    ax2.set_xlabel("tick", fontsize=11, fontweight="bold")
    ax2.set_ylabel("negative emotion", fontsize=11, fontweight="bold")
    ax2.set_title("By AgentType · Negative Emotion", fontsize=12, fontweight="bold")
    ax2.grid(axis="both", linestyle="dotted", color="black", alpha=0.30)
    ax2.legend(loc="best", fontsize=8, ncol=3, framealpha=0.9)

    fig.suptitle(
        f"Role Facets | n={meta.get('n_agents','?')} seed={meta.get('random_seed','?')}",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out / "fig7_by_agent_type.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig8_by_group_type(df: pd.DataFrame, out: Path, meta: dict) -> None:
    """老师要求：按群类型分图（宿舍/班级/专业/校园）。"""
    heat_cols = [
        ("group_heat_DORM", "DORM", "#8A7067"),
        ("group_heat_CLASS", "CLASS", "#4898FF"),
        ("group_heat_MAJOR", "MAJOR", "#7A5CFF"),
        ("group_heat_CAMPUS", "CAMPUS", "#D56959"),
    ]
    neg_cols = [
        ("group_neg_DORM", "DORM", "#8A7067"),
        ("group_neg_CLASS", "CLASS", "#4898FF"),
        ("group_neg_MAJOR", "MAJOR", "#7A5CFF"),
        ("group_neg_CAMPUS", "CAMPUS", "#D56959"),
    ]
    heat_present = [(c, lab, col) for c, lab, col in heat_cols if c in df.columns]
    neg_present = [(c, lab, col) for c, lab, col in neg_cols if c in df.columns]
    if len(heat_present) < 2 and len(neg_present) < 2:
        print("skip fig8_by_group_type: missing group_heat_/group_neg_ columns")
        return

    tick = df["tick"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), dpi=DPI, sharex=True)

    ax = axes[0]
    for c, lab, col in heat_present:
        ax.plot(tick, df[c].to_numpy(dtype=float), color=col, linewidth=LW,
                marker="o", markersize=MS, markevery=max(1, len(tick) // 12),
                label=lab, alpha=0.92)
    ax.set_ylabel("group heat H", fontsize=11, fontweight="bold")
    ax.set_title("By GroupType · Heat", fontsize=12, fontweight="bold")
    ax.grid(axis="both", linestyle="dotted", color="black", alpha=0.30)
    ax.legend(loc="best", fontsize=8, ncol=4, framealpha=0.9)

    ax2 = axes[1]
    for c, lab, col in neg_present:
        ax2.plot(tick, df[c].to_numpy(dtype=float), color=col, linewidth=LW,
                 marker="o", markersize=MS, markevery=max(1, len(tick) // 12),
                 label=lab, alpha=0.92)
    ax2.set_xlabel("tick", fontsize=11, fontweight="bold")
    ax2.set_ylabel("group negative", fontsize=11, fontweight="bold")
    ax2.set_title("By GroupType · Negative", fontsize=12, fontweight="bold")
    ax2.grid(axis="both", linestyle="dotted", color="black", alpha=0.30)
    ax2.legend(loc="best", fontsize=8, ncol=4, framealpha=0.9)

    # 若有 wall_hour / time_slot，在标题提示仿真时钟
    clock_note = ""
    if "wall_hour" in df.columns:
        clock_note = f" | hour {df['wall_hour'].iloc[0]:.0f}→{df['wall_hour'].iloc[-1]:.0f}"
    fig.suptitle(
        f"Group Facets{clock_note} | n={meta.get('n_agents','?')}",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out / "fig8_by_group_type.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig6_step_time_line(df: pd.DataFrame, out: Path, meta: dict) -> None:
    if "step_seconds" not in df.columns:
        return
    tick = df["tick"].to_numpy()
    y = df["step_seconds"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=DPI)
    ax.plot(
        tick, y, color=COLORS["step_seconds"], linewidth=1.5,
        marker="o", markersize=3.0, label="step_seconds",
        markevery=max(1, len(tick) // 15),
    )
    mean_v = float(np.nanmean(y))
    ax.axhline(mean_v, color="#6469BD", linestyle="--", linewidth=1.2, alpha=0.9,
               label=f"mean={mean_v:.4f}s")
    idxs = select_label_indices(y, "step_seconds")
    _annotate(ax, tick, y, idxs, COLORS["step_seconds"], side="above", fmt="{:.3f}")
    _pad_ylim(ax, [y], label_side="above", frac=0.14)
    ax.set_title(
        f"Per-Tick Runtime | n={meta.get('n_agents','?')}, llm={meta.get('llm_provider','?')}",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.set_xlabel("tick", fontsize=11, fontweight="bold")
    ax.set_ylabel("seconds / tick", fontsize=11, fontweight="bold")
    ax.grid(axis="both", linestyle="dotted", color="black", alpha=0.30)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "fig6_step_seconds.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="output/metrics.csv")
    parser.add_argument("--meta", default="output/run_meta.json")
    parser.add_argument("--out", default="output/figures")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    metrics_path = here / args.metrics
    meta_path = here / args.meta
    out = here / args.out
    out.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"找不到 {metrics_path}，请先运行 run_sim.py")

    _setup_cn_font()
    df, meta = load_inputs(metrics_path, meta_path)

    fig1_bounded_trends(df, out, meta)
    fig2_count_trends(df, out, meta)
    fig3_runtime_bars(meta, out)
    fig4_metric_panels(df, out, meta)
    fig5_contagion_optional(df, out, meta)
    fig6_step_time_line(df, out, meta)
    fig7_by_agent_type(df, out, meta)
    fig8_by_group_type(df, out, meta)

    print("Saved figures to:", out)
    for p in sorted(out.glob("fig*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
