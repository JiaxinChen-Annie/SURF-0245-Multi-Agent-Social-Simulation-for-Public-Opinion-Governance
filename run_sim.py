"""
run_sim.py — 仿真主入口 + W2 自测脚本
---------------------------------------
用法：
    python3 run_sim.py                         # 使用默认 config.yaml（100 agents）
    python3 run_sim.py --config config_1000.yaml  # 压力测试（1000 agents）
    python3 run_sim.py --no-plot               # 不显示弹窗（服务器环境）

自测验收标准（接口表要求）：
    ✓ 50 轮仿真运行 < 30 秒
    ✓ 1000 Agent 无调度冲突
    ✓ DataCollector 每步指标格式 100% 与 E 对齐（avg_opinion / polarization / emotional_contagion）
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")   # 无图形界面环境时强制 Agg 后端
import matplotlib.font_manager as _fm

# 自动检测 CJK 字体（有则使用，无则回退 ASCII 标签）
_CJK_FONT = next(
    (f.name for f in _fm.fontManager.ttflist
     if any(k in f.name for k in ["Noto Sans CJK", "WenQuanYi", "SimHei", "Microsoft YaHei"])),
    None,
)
if _CJK_FONT:
    matplotlib.rcParams["font.family"] = _CJK_FONT

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import yaml

# ─── 日志 ──────────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
_LOG = logging.getLogger("run_sim")

# ─── 项目目录加入 sys.path ─────────────────────────────────────────────── #
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from types_def import SimConfig
from opinion_model import OpinionModel


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
# 辅助：从 YAML 读取 SimConfig（对应 E 模块 load_config）                   #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def load_config(path: str) -> SimConfig:
    """从 YAML/JSON 文件读取仿真配置，返回 SimConfig 实例。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return SimConfig(
        n_agents         = raw.get("n_agents",          100),
        agent_type_ratio = raw.get("agent_type_ratio",  {}),
        network_type     = raw.get("network_type",      "barabasi_albert"),
        network_params   = raw.get("network_params",    {"m": 3}),
        n_steps          = raw.get("n_steps",           50),
        hawkes_params    = raw.get("hawkes_params",     {"mu":0.1,"alpha":0.5,"beta":1.0}),
        llm_config       = raw.get("llm_config",        {}),
        random_seed      = raw.get("random_seed",       42),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
# 主仿真流程                                                                 #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def run(config_path: str, save_plot: bool = True) -> None:
    banner = "=" * 60
    print(f"\n{banner}")
    print("  多智能体舆情仿真 · W2 自测  (模块二 OpinionModel)")
    print(f"{banner}")

    # ① 加载配置
    _LOG.info(f"读取配置: {config_path}")
    config = load_config(config_path)
    print(f"\n[配置]")
    print(f"  Agent 数量  : {config.n_agents}")
    print(f"  网络类型    : {config.network_type}")
    print(f"  运行步数    : {config.n_steps}")
    print(f"  随机种子    : {config.random_seed}")

    # ② 初始化模型
    _LOG.info("初始化 OpinionModel ...")
    t0 = time.perf_counter()
    model = OpinionModel(config)
    t_init = time.perf_counter() - t0
    print(f"\n[初始化] 耗时 {t_init:.3f}s")

    # ③ 运行仿真
    print(f"\n[运行] 开始 {config.n_steps} 步仿真 ...")
    t_start = time.perf_counter()

    for step_i in range(config.n_steps):
        model.step()
        if (step_i + 1) % 10 == 0:
            df_tmp = model.datacollector.get_model_vars_dataframe()
            last   = df_tmp.iloc[-1]
            print(
                f"  step {step_i+1:3d} | "
                f"avg_opinion={last['avg_opinion']:+.3f} | "
                f"polarization={last['polarization']:.3f} | "
                f"emotional_contagion={last['emotional_contagion']:.4f}"
            )

    t_run = time.perf_counter() - t_start

    # ④ 收集结果
    df = model.datacollector.get_model_vars_dataframe()

    # ━━ 验收检查 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    print(f"\n{'─'*60}")
    print("  验收结果")
    print(f"{'─'*60}")

    # 耗时
    time_ok = t_run < 30.0
    print(f"  ① 运行耗时: {t_run:.2f}s {'✅ < 30s' if time_ok else '❌ 超过 30s'}")

    # 行数
    rows_ok = len(df) == config.n_steps
    print(f"  ② DataCollector 行数: {len(df)} {'✅ 与步数一致' if rows_ok else '❌ 行数不符'}")

    # 列名
    expected_cols = {"avg_opinion", "polarization", "emotional_contagion"}
    cols_ok = expected_cols.issubset(set(df.columns))
    print(f"  ③ 指标列名: {list(df.columns)} {'✅ 含全部必需列' if cols_ok else '❌ 缺失列'}")

    # 无 NaN
    nan_count = df.isnull().sum().sum()
    nan_ok = nan_count == 0
    print(f"  ④ 数据完整性: NaN={nan_count} {'✅' if nan_ok else '❌'}")

    # 观点演化
    op_start = df["avg_opinion"].iloc[0]
    op_end   = df["avg_opinion"].iloc[-1]
    op_moved = abs(op_end - op_start) > 0.005
    print(f"  ⑤ 观点演化: {op_start:+.3f} → {op_end:+.3f} "
          f"{'✅ 观点有移动' if op_moved else '⚠ 变化极小（可能正常）'}")

    # 极化检查
    pol_mean = df["polarization"].mean()
    print(f"  ⑥ 平均极化度: {pol_mean:.3f} (期望 > 0.1)")

    all_pass = time_ok and rows_ok and cols_ok and nan_ok
    verdict  = "✅ 全部通过" if all_pass else "❌ 存在未通过项"
    print(f"\n  综合评分: {verdict}")
    print(f"{'─'*60}")

    # ━━ 摘要统计 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    print(f"\n[指标摘要]")
    print(df.describe().round(4).to_string())

    # ━━ 保存 CSV ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    out_dir = os.path.join(_HERE, "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")
    df.to_csv(csv_path, index_label="step")
    print(f"\n[输出] 指标 CSV → {csv_path}")

    # ━━ 绘图 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    if save_plot:
        _plot_trends(df, config, out_dir)

    print(f"\n{banner}")
    print("  完成！")
    print(f"{banner}\n")


def _plot_trends(df, config: SimConfig, out_dir: str) -> None:
    """绘制三张指标演化曲线图并保存。"""
    steps  = np.arange(len(df))
    fig    = plt.figure(figsize=(12, 9))
    fig.suptitle(
        f"多智能体舆情仿真 · W2 指标演化\n"
        f"n={config.n_agents} · {config.network_type} · seed={config.random_seed}",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 1, hspace=0.45)

    # ── subplot 1：avg_opinion ────────────────────────────────────────── #
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(steps, df["avg_opinion"], color="#2563EB", linewidth=1.8, label="avg_opinion")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.fill_between(steps, df["avg_opinion"], 0,
                     where=(df["avg_opinion"] > 0), alpha=0.12, color="#22C55E")
    ax1.fill_between(steps, df["avg_opinion"], 0,
                     where=(df["avg_opinion"] <= 0), alpha=0.12, color="#EF4444")
    ax1.set_ylabel("平均观点值", fontsize=10)
    ax1.set_title("① 全网平均观点值（+1=正面支持，-1=负面反对）", fontsize=10)
    ax1.set_ylim(-1.05, 1.05)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── subplot 2：polarization ───────────────────────────────────────── #
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(steps, df["polarization"], color="#DC2626", linewidth=1.8, label="polarization")
    ax2.set_ylabel("极化程度（标准差）", fontsize=10)
    ax2.set_title("② 观点极化程度（越高=意见越分裂）", fontsize=10)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── subplot 3：emotional_contagion ────────────────────────────────── #
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(steps, df["emotional_contagion"], color="#7C3AED",
             linewidth=1.8, label="emotional_contagion")
    ax3.set_ylabel("情绪传播速度", fontsize=10)
    ax3.set_xlabel("时间步（Tick）", fontsize=10)
    ax3.set_title("③ 情绪传播速度（单步唤醒度变化均值）", fontsize=10)
    ax3.set_ylim(bottom=0)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, len(steps) - 1)

    img_path = os.path.join(out_dir, "simulation_trends.png")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[输出] 趋势图 → {img_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
# 入口                                                                       #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多智能体舆情仿真 · W2 自测")
    parser.add_argument(
        "--config", default="config.yaml",
        help="YAML 配置文件路径（默认：config.yaml）",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="跳过绘图（CI 环境使用）",
    )
    args = parser.parse_args()

    config_path = os.path.join(_HERE, args.config)
    if not os.path.exists(config_path):
        print(f"ERROR: 配置文件不存在: {config_path}", file=sys.stderr)
        sys.exit(1)

    run(config_path, save_plot=not args.no_plot)
