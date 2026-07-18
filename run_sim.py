"""
run_sim.py — 仿真主入口（W4 版本）
---------------------------------------
用法：
    python3 run_sim.py                              # 使用默认 config.yaml（100 agents）
    python3 run_sim.py --config config_1000.yaml   # 压力测试（1000 agents）
    python3 run_sim.py --no-plot                   # 不显示弹窗（服务器环境）

场景：大学校园多群舆情扩散（DORM/CLASS/MAJOR/CAMPUS）
      角色：ORDINARY/ACTIVE/RATIONAL/CONTROLLER

自测验收标准：
    ✓ 50 轮仿真运行 < 30 秒
    ✓ 1000 Agent 无调度冲突
    ✓ DataCollector 指标含接口表§5 全部字段
    ✓ 热度演化公式正确（CAMPUS 衰减最快）
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
_LOG = logging.getLogger("run_sim")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from types_def import SimConfig
from opinion_model import OpinionModel


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
# load_config（对应 E 模块 load_config，#32）                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def load_config(path: str) -> SimConfig:
    """从 YAML/JSON 文件读取仿真配置，返回 SimConfig 实例。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return SimConfig(
        n_agents         = raw.get("n_agents",         100),
        agent_type_ratio = raw.get("agent_type_ratio", {}),
        group_type_ratio = raw.get("group_type_ratio", {}),
        network_type     = raw.get("network_type",     "barabasi_albert"),
        network_params   = raw.get("network_params",   {"m": 3}),
        n_steps          = raw.get("n_steps",          50),
        hawkes_params    = raw.get("hawkes_params",    {"mu": 0.1, "alpha": 0.5, "beta": 1.0}),
        llm_config       = raw.get("llm_config",       {}),
        random_seed      = raw.get("random_seed",      42),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
# 主仿真流程                                                                #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def run(config_path: str, save_plot: bool = True) -> None:
    banner = "=" * 65
    print(f"\n{banner}")
    print("  多智能体舆情仿真 · W4 自测  (模块二 OpinionModel v2)")
    print(f"  场景：大学校园多群舆情扩散")
    print(f"{banner}")

    _LOG.info(f"读取配置: {config_path}")
    config = load_config(config_path)

    print(f"\n[配置]")
    print(f"  Agent 数量  : {config.n_agents}")
    print(f"  AgentType   : {config.agent_type_ratio}")
    print(f"  GroupType   : {config.group_type_ratio}")
    print(f"  网络类型    : {config.network_type}")
    print(f"  运行步数    : {config.n_steps}")
    print(f"  随机种子    : {config.random_seed}")

    _LOG.info("初始化 OpinionModel ...")
    t0 = time.perf_counter()
    model = OpinionModel(config)
    t_init = time.perf_counter() - t0
    print(f"\n[初始化] 耗时 {t_init:.3f}s")

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
                f"msg_count={last['message_count']:5.0f} | "
                f"cross_fwd={last['cross_group_forward']:4.0f}"
            )

    t_run = time.perf_counter() - t_start

    df = model.datacollector.get_model_vars_dataframe()

    # ── 验收检查 ──────────────────────────────────────────────────────── #
    print(f"\n{'─'*65}")
    print("  验收结果")
    print(f"{'─'*65}")

    time_ok = t_run < 30.0
    print(f"  ① 运行耗时: {t_run:.2f}s {'✅ < 30s' if time_ok else '❌ 超过 30s'}")

    rows_ok = len(df) == config.n_steps
    print(f"  ② DataCollector 行数: {len(df)} {'✅' if rows_ok else '❌ 行数不符'}")

    required_cols = {
        "avg_opinion", "polarization", "emotional_contagion",
        "message_count", "negative_emotion", "distortion_level",
        "cross_group_forward", "intervention_tick", "recovery_time",
    }
    cols_ok = required_cols.issubset(set(df.columns))
    missing = required_cols - set(df.columns)
    print(f"  ③ 指标列名: {'✅ 含全部必需列' if cols_ok else f'❌ 缺失: {missing}'}")

    nan_count = df.isnull().sum().sum()
    nan_ok = nan_count == 0
    print(f"  ④ 数据完整性: NaN={nan_count} {'✅' if nan_ok else '❌'}")

    op_start = df["avg_opinion"].iloc[0]
    op_end   = df["avg_opinion"].iloc[-1]
    print(f"  ⑤ 观点演化: {op_start:+.3f} → {op_end:+.3f}")

    print(f"  ⑥ 跨群转发: 累计 {int(df['cross_group_forward'].iloc[-1])} 次")
    int_tick = df["intervention_tick"].iloc[-1]
    print(f"  ⑦ 最早干预时刻: {int_tick} tick")

    all_pass = time_ok and rows_ok and cols_ok and nan_ok
    print(f"\n  综合评分: {'✅ 全部通过' if all_pass else '❌ 存在未通过项'}")
    print(f"{'─'*65}")

    print(f"\n[指标摘要]")
    print(df.describe().round(4).to_string())

    out_dir  = os.path.join(_HERE, "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")
    df.to_csv(csv_path, index_label="step")
    print(f"\n[输出] 指标 CSV → {csv_path}")

    if save_plot:
        _plot_trends(df, config, out_dir)

    print(f"\n{banner}")
    print("  完成！")
    print(f"{banner}\n")


def _plot_trends(df, config: SimConfig, out_dir: str) -> None:
    """绘制 6 张指标演化曲线图并保存。"""
    steps = np.arange(len(df))
    fig   = plt.figure(figsize=(14, 14))
    fig.suptitle(
        f"多智能体舆情仿真 · W4 指标演化\n"
        f"场景：大学校园多群舆情 | n={config.n_agents} · {config.network_type} · seed={config.random_seed}",
        fontsize=12, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.35)

    def _sub(row, col, y_data, color, ylabel, title, ylim=None):
        ax = fig.add_subplot(gs[row, col])
        ax.plot(steps, y_data, color=color, linewidth=1.6)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlim(0, len(steps) - 1)
        ax.grid(True, alpha=0.3)
        return ax

    _sub(0, 0, df["avg_opinion"],         "#2563EB", "平均观点值",     "① 全网平均观点",      (-1.05, 1.05))
    _sub(0, 1, df["polarization"],        "#DC2626", "极化程度(std)",  "② 观点极化程度")
    _sub(1, 0, df["emotional_contagion"], "#7C3AED", "情绪传播速度",   "③ 情绪传播速度")
    _sub(1, 1, df["message_count"],       "#16A34A", "消息总量",       "④ 信息流消息量")
    _sub(2, 0, df["negative_emotion"],    "#EA580C", "负面情绪比例",   "⑤ 负面情绪指数",      (0, 1.05))
    _sub(2, 1, df["cross_group_forward"], "#0891B2", "跨群转发次数",   "⑥ 跨群转发累计")

    img_path = os.path.join(out_dir, "simulation_trends_w4.png")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[输出] 趋势图 → {img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多智能体舆情仿真 · W4 自测")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    config_path = os.path.join(_HERE, args.config)
    if not os.path.exists(config_path):
        print(f"ERROR: 配置文件不存在: {config_path}", file=sys.stderr)
        sys.exit(1)

    run(config_path, save_plot=not args.no_plot)
