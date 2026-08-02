#!/usr/bin/env python3
"""E 模块报告生成器：分角色/分群/实验后端对照的可视化与汇总。

真实数据用法：
    python generate_e_report.py --input metrics.csv --output-dir e_report

支持的最低字段：
    tick, avg_opinion, polarization, negative_emotion,
    emotional_contagion, message_count, distortion_level

分面字段：
    agent_type / AgentType / role
    group_type / GroupType / group
    experiment / experiment_label / backend

可选字段：治理干预用 governance_tick 或 intervention_tick。
如果输入来自 save_simulation_results 的 16-bit 编码，可加 --decode-16bit。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

for _font in ["PingFang SC", "STHeiti", "Heiti SC", "Microsoft YaHei", "Noto Sans CJK SC"]:
    if _font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

METRICS = [
    "avg_opinion",
    "polarization",
    "negative_emotion",
    "emotional_contagion",
    "message_count",
    "distortion_level",
]
GROUP_ALIASES = {
    "agent_type": ["agent_type", "AgentType", "role", "agent_role"],
    "group_type": ["group_type", "GroupType", "group", "group_name"],
    "experiment": ["experiment", "experiment_label", "backend", "mode"],
}
M11 = {"avg_opinion"}
M01 = {"polarization", "negative_emotion", "emotional_contagion", "distortion_level"}

COLORS = {
    "Mock": "#4C78A8",
    "DeepSeek": "#E45756",
    "unknown": "#777777",
}


def _find_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    for name in aliases:
        if name in df.columns:
            return name
    return None


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    raise ValueError(f"只支持 CSV/JSON 输入，不支持: {path.suffix}")


def decode_16bit(df: pd.DataFrame) -> pd.DataFrame:
    """还原 utils.save_simulation_results 的 16-bit 指标编码。"""
    out = df.copy()
    for col in M11 & set(out.columns):
        out[col] = out[col] / 32767.0
    for col in M01 & set(out.columns):
        out[col] = out[col] / 65535.0
    return out


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for target, aliases in GROUP_ALIASES.items():
        source = _find_column(out, aliases)
        if source and source != target:
            rename[source] = target
    out = out.rename(columns=rename)
    if "experiment" not in out.columns:
        out["experiment"] = "unknown"
    for col in ("agent_type", "group_type"):
        if col not in out.columns:
            out[col] = "all"
        out[col] = out[col].fillna("unknown").astype(str)
    out["experiment"] = out["experiment"].fillna("unknown").astype(str)
    out["tick"] = pd.to_numeric(out["tick"], errors="coerce")
    out = out.dropna(subset=["tick"]).copy()
    return out


def validate(df: pd.DataFrame) -> None:
    required = {"tick", *METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"输入缺少 E 模块必要字段: {missing}")


def _style(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_facets(df: pd.DataFrame, facet: str, output: Path) -> None:
    """每个 facet 一个子图；颜色表示 Mock/DeepSeek，输出六项核心指标。"""
    values = list(dict.fromkeys(df[facet].astype(str)))
    ncols = 2
    nrows = max(1, (len(values) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.2 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, value in zip(axes, values):
        subset = df[df[facet].astype(str) == value]
        for experiment, exp_df in subset.groupby("experiment", sort=False):
            color = COLORS.get(str(experiment), None)
            for metric in ["avg_opinion", "polarization", "negative_emotion"]:
                grouped = exp_df.groupby("tick", as_index=False)[metric].mean().sort_values("tick")
                ax.plot(grouped["tick"], grouped[metric], label=f"{experiment} · {metric}",
                        color=color, alpha=0.55 if metric != "avg_opinion" else 1.0,
                        linewidth=2.0 if metric == "avg_opinion" else 1.2,
                        linestyle="-" if metric == "avg_opinion" else "--")
        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.set_title(f"{facet} = {value}")
        _style(ax, "指标值")
        ax.legend(fontsize=8, ncol=2)
    for ax in axes[len(values):]:
        ax.axis("off")
    fig.suptitle(f"E 模块：按 {facet} 的观点/极化/负面情绪趋势", fontsize=16, fontweight="bold")
    fig.supxlabel("tick")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_backend_comparison(df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for ax, metric in zip(axes.ravel(), ["avg_opinion", "polarization", "negative_emotion", "emotional_contagion"]):
        for experiment, exp_df in df.groupby("experiment", sort=False):
            grouped = exp_df.groupby("tick", as_index=False)[metric].mean().sort_values("tick")
            ax.plot(grouped["tick"], grouped[metric], label=str(experiment),
                    color=COLORS.get(str(experiment)), linewidth=2)
        ax.set_title(metric)
        _style(ax, "均值")
        ax.legend()
    fig.suptitle("E 模块：Mock / DeepSeek 全网对照", fontsize=16, fontweight="bold")
    fig.supxlabel("tick")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["experiment", "agent_type", "group_type"]
    summary = df.groupby(group_cols, dropna=False)[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns]
    return summary.round(6)


def make_intervention_summary(df: pd.DataFrame) -> pd.DataFrame:
    tick_col = _find_column(df, ["governance_tick", "intervention_tick"])
    if not tick_col:
        return pd.DataFrame(columns=["experiment", "agent_type", "group_type", "window", *METRICS])
    rows = []
    for keys, subset in df.groupby(["experiment", "agent_type", "group_type"], dropna=False):
        intervention = pd.to_numeric(subset[tick_col], errors="coerce").dropna()
        if intervention.empty:
            continue
        t0 = int(intervention.min())
        for window, part in [("before", subset[subset.tick < t0]), ("after", subset[subset.tick >= t0])]:
            row = dict(zip(["experiment", "agent_type", "group_type"], keys))
            row["window"] = window
            row.update(part[METRICS].mean().to_dict())
            rows.append(row)
    return pd.DataFrame(rows).round(6)


def generate_report(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    paths = [
        ("01_by_agent_type.png", "agent_type"),
        ("02_by_group_type.png", "group_type"),
    ]
    for filename, facet in paths:
        target = output_dir / filename
        plot_facets(df, facet, target)
        outputs.append(target)
    target = output_dir / "03_mock_vs_deepseek.png"
    plot_backend_comparison(df, target)
    outputs.append(target)
    summary = make_summary(df)
    summary_path = output_dir / "summary_by_agent_and_group.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    outputs.append(summary_path)
    intervention = make_intervention_summary(df)
    intervention_path = output_dir / "intervention_before_after.csv"
    intervention.to_csv(intervention_path, index=False, encoding="utf-8-sig")
    outputs.append(intervention_path)
    return outputs


def demo_data() -> pd.DataFrame:
    """生成验收用示例数据；输出会明确标记 DEMO，不代表真实实验。"""
    rows = []
    roles = ["ORDINARY", "ACTIVE", "RATIONAL", "CONTROLLER"]
    groups = ["DORM", "CLASS", "MAJOR", "CAMPUS"]
    for backend, shift in [("Mock", 0.0), ("DeepSeek", 0.12)]:
        for role_i, role in enumerate(roles):
            for group_i, group in enumerate(groups):
                for tick in range(30):
                    rows.append({
                        "tick": tick,
                        "agent_type": role,
                        "group_type": group,
                        "experiment": backend,
                        "avg_opinion": -0.1 + shift + 0.02 * role_i + 0.01 * group_i + 0.004 * tick,
                        "polarization": 0.5 - 0.002 * tick + 0.01 * role_i,
                        "negative_emotion": 0.45 + (0.04 if backend == "DeepSeek" else 0) - 0.003 * tick,
                        "emotional_contagion": 0.02 + (0.01 if backend == "DeepSeek" else 0) * (tick > 8),
                        "message_count": 10 + 2 * tick + role_i,
                        "distortion_level": 0.05 + (0.02 if backend == "DeepSeek" else 0),
                        "governance_tick": 15,
                    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 E 模块分面图和汇总表")
    parser.add_argument("--input", type=Path, help="真实 CSV/JSON 指标文件")
    parser.add_argument("--output-dir", type=Path, default=Path("e_report"))
    parser.add_argument("--decode-16bit", action="store_true")
    parser.add_argument("--demo", action="store_true", help="生成 DEMO 示例图")
    args = parser.parse_args()
    if not args.input and not args.demo:
        parser.error("请提供 --input metrics.csv，或使用 --demo 生成验收示例")
    df = demo_data() if args.demo else load_table(args.input)
    if args.decode_16bit:
        df = decode_16bit(df)
    df = normalise_columns(df)
    validate(df)
    outputs = generate_report(df, args.output_dir)
    print(f"rows={len(df)}; output={args.output_dir.resolve()}")
    for path in outputs:
        print(path.resolve())


if __name__ == "__main__":
    main()
