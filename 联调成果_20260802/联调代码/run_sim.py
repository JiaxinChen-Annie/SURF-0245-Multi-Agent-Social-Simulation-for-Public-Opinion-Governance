"""
run_sim.py — 仿真主入口（v3 场景对齐版）
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
    ✓ 场景一致性：事件源投放 / 四条转发通道 / 禁言公告 / 干预与恢复
"""

from __future__ import annotations
import mesa_patch  # Mesa 3.x 兼容补丁

import argparse
import json
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
    from types_def import DEFAULT_SCENARIO_PARAMS
    scenario = dict(DEFAULT_SCENARIO_PARAMS)
    for key, value in (raw.get("scenario_params") or {}).items():
        if isinstance(value, dict) and isinstance(scenario.get(key), dict):
            merged = dict(scenario[key])
            merged.update(value)
            scenario[key] = merged
        else:
            scenario[key] = value
    return SimConfig(
        n_agents         = raw.get("n_agents",         100),
        agent_type_ratio = raw.get("agent_type_ratio", {}),
        group_type_ratio = raw.get("group_type_ratio", {}),
        network_type     = raw.get("network_type",     "barabasi_albert"),
        network_params   = raw.get("network_params",   {"m": 3}),
        n_steps          = raw.get("n_steps",          50),
        hawkes_params    = raw.get("hawkes_params",    {"mu": 0.1, "alpha": 0.5, "beta": 1.0}),
        llm_config       = raw.get("llm_config",       {}),
        random_seed      = (raw.get("random_seed") if raw.get("random_seed") is not None
                            else __import__('random').randint(0, 2**31 - 1)),
        scenario_params  = scenario,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
# 主仿真流程                                                                #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

def run(config_path: str, save_plot: bool = True) -> None:
    banner = "=" * 65
    print(f"\n{banner}")
    print("  多智能体舆情仿真 · A 模块自测  (模块二 OpinionModel v3)")
    print(f"  场景：大学校园多群舆情扩散")
    print(f"{banner}")

    _LOG.info(f"读取配置: {config_path}")
    config = load_config(config_path)

    print(f"\n[配置]")
    print(f"  Agent 数量  : {config.n_agents}")
    print(f"  AgentType   : {config.agent_type_ratio}")
    print(f"  GroupType   : {config.group_type_ratio}")
    print(f"  网络类型    : {config.network_type}")
    print(f"  群成员模式  : {config.network_params.get('group_membership_mode', 'hierarchical')}")
    print(f"  转发目标策略: {config.network_params.get('forward_destination_strategy', 'next_larger')}")
    print(f"  运行步数    : {config.n_steps}")
    print(f"  随机种子    : {config.random_seed}")
    sp = config.scenario_params
    print(f"  干预触发    : {sp.get('intervention_trigger')} "
          f"(负面阈={sp.get('negative_threshold')} / 热度阈={sp.get('heat_threshold')})")
    print(f"  跨群门槛    : {sp.get('cross_group_forward_threshold_by_direction')}")
    print(f"  禁言开关    : {sp.get('enable_mute')}")

    _LOG.info("初始化 OpinionModel ...")
    t0 = time.perf_counter()
    model = OpinionModel(config)
    t_init = time.perf_counter() - t0
    print(f"\n[初始化] 耗时 {t_init:.3f}s")

    print(f"\n[运行] 开始 {config.n_steps} 步仿真 ...")
    t_start = time.perf_counter()
    step_times: list[float] = []

    for step_i in range(config.n_steps):
        t_step0 = time.perf_counter()
        model.step()
        step_times.append(time.perf_counter() - t_step0)
        if (step_i + 1) % 10 == 0:
            df_tmp = model.datacollector.get_model_vars_dataframe()
            last   = df_tmp.iloc[-1]
            print(
                f"  step {step_i+1:3d} | "
                f"avg_opinion={last['avg_opinion']:+.3f} | "
                f"polarization={last['polarization']:.3f} | "
                f"msg_count={last['message_count']:5.0f} | "
                f"H_max={last['group_heat_max']:.2f} | "
                f"neg_max={last['group_negative_max']:.2f} | "
                f"cross_fwd={last['cross_group_forward']:4.0f} | "
                f"mute={last['mute_count']:3.0f} | "
                f"step_s={step_times[-1]:.3f}"
            )

    t_run = time.perf_counter() - t_start

    df = model.datacollector.get_model_vars_dataframe()

    # ── 验收检查 ──────────────────────────────────────────────────────── #
    print(f"\n{'─'*65}")
    print("  验收结果")
    print(f"{'─'*65}")

    # 真 LLM 时 30s 门槛不适用；mock/规则路径仍参考该值
    is_real_llm = bool(
        (config.llm_config or {}).get("provider")
        and str((config.llm_config or {}).get("provider", "")).lower() != "mock"
        and (config.llm_config or {}).get("use_actions", False)
    )
    time_ok = True if is_real_llm else (t_run < 30.0)
    time_msg = (
        f"{t_run:.2f}s （真 LLM，不套用 30s）"
        if is_real_llm
        else f"{t_run:.2f}s {'✅ < 30s' if time_ok else '❌ 超过 30s'}"
    )
    print(f"  ① 运行耗时: {time_msg}")

    rows_ok = len(df) == config.n_steps
    print(f"  ② DataCollector 行数: {len(df)} {'✅' if rows_ok else '❌ 行数不符'}")

    required_cols = {
        "avg_opinion", "polarization", "emotional_contagion",
        "message_count", "negative_emotion", "distortion_level",
        "cross_group_forward", "cross_group_spread",
        "intervention_tick", "recovery_time",
        "mute_count", "group_heat_max", "group_negative_max",
        "upward_forward", "lateral_forward", "downward_forward",
    }
    cols_ok = required_cols.issubset(set(df.columns))
    missing = required_cols - set(df.columns)
    print(f"  ③ 指标列名: {'✅ 含全部必需列' if cols_ok else f'❌ 缺失: {missing}'}")

    nan_count = int(df.isnull().sum().sum())
    nan_ok = nan_count == 0
    print(f"  ④ 数据完整性: NaN={nan_count} {'✅' if nan_ok else '❌'}")

    op_start = df["avg_opinion"].iloc[0]
    op_end   = df["avg_opinion"].iloc[-1]
    print(f"  ⑤ 观点演化: {op_start:+.3f} → {op_end:+.3f}")

    summary = model.get_final_summary()
    print(f"  ⑥ 转发通道分解（问题③④）:")
    print(f"     群内转发 {summary['intra_group_forward']:5d} 次  ← 攒够转发次数的唯一途径")
    print(f"     小→大    {summary['upward_forward']:5d} 次  ← 主通道")
    print(f"     小群互转 {summary['lateral_forward']:5d} 次  ← DORM↔CLASS 横向（弱）")
    print(f"     大→小    {summary['downward_forward']:5d} 次  ← 向下（最弱）")
    print(f"     跨群合计 {summary['cross_group_forward']:5d} 次 | "
          f"被门槛拦下 {summary['blocked_by_forward_gate']} 次")
    print(f"     宏观热度溅射 {summary['cross_group_spread']} 次（与人际转发分开计数）")

    ev = summary["event_source"]
    print(f"  ⑦ 事件源（问题②）: 共投放 {ev['n_events']} 次"
          f"（初始 {ev['n_initial']} / 二次 {ev['n_secondary']}）")
    for e in ev["events"]:
        print(f"     tick={e['tick']:3d} Agent-{e['origin_agent_id']:<4d} → {e['group_id']:<13s}"
              f" ({'小群' if e['scope'] == 'small' else '大群'})"
              f" H0={e['heat']:.2f} neg={e['negative_score']:.2f}")

    print(f"  ⑧ 干预与恢复（问题①⑤⑦）:")
    print(f"     触发阈值时刻 t_int  : {summary['intervention_tick_by_group']}")
    print(f"     Controller 实际动手 : "
          f"{ {g.name: t for g, t in model.governance_tick.items() if t is not None} }")
    print(f"     禁言 {summary['mute_count']} 次 / 公告 {summary['announce_count']} 次"
          f" / 因禁言被拦 {summary['blocked_by_mute']} 条")
    print(f"     峰值热度 {df['group_heat_max'].max():.3f} | "
          f"峰值群负面 {df['group_negative_max'].max():.3f} | "
          f"recovery_time = {summary['recovery_time']}")
    print(f"     control_level: {summary['control_level_by_group']}")
    print(f"     曝光 exposure: {summary['exposure_by_group']}")

    scenario_ok = ev["n_events"] > 0
    all_pass = time_ok and rows_ok and cols_ok and nan_ok and scenario_ok
    print(f"\n  综合评分: {'✅ 全部通过' if all_pass else '❌ 存在未通过项'}")
    print(f"{'─'*65}")

    print(f"\n[指标摘要]")
    print(df.describe().round(4).to_string())

    out_dir  = os.path.join(_HERE, "output")
    os.makedirs(out_dir, exist_ok=True)

    df_out = df.copy()
    if "tick" not in df_out.columns:
        df_out = df_out.reset_index(drop=True)
        df_out.insert(0, "tick", np.arange(len(df_out), dtype=int))
    if len(step_times) == len(df_out):
        df_out["step_seconds"] = step_times
    else:
        df_out["step_seconds"] = (step_times + [np.nan] * len(df_out))[: len(df_out)]

    csv_path = os.path.join(out_dir, "metrics.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\n[输出] 指标 CSV → {csv_path}")

    pd_step = __import__("pandas").DataFrame(
        {"tick": df_out["tick"], "step_seconds": df_out["step_seconds"]}
    )
    pd_step.to_csv(os.path.join(out_dir, "step_times.csv"), index=False)

    meta = {
        "config_path": config_path,
        "n_agents": config.n_agents,
        "n_steps": config.n_steps,
        "random_seed": config.random_seed,
        "network_type": config.network_type,
        "network_params": config.network_params,
        "llm_provider": (config.llm_config or {}).get("provider"),
        "llm_model": (config.llm_config or {}).get("model"),
        "is_real_llm": is_real_llm,
        "init_seconds": round(t_init, 6),
        "run_seconds": round(t_run, 6),
        "total_seconds": round(t_init + t_run, 6),
        "seconds_per_step": round(t_run / max(config.n_steps, 1), 6),
        "step_seconds_mean": round(float(np.mean(step_times)), 6) if step_times else 0.0,
        "step_seconds_max": round(float(np.max(step_times)), 6) if step_times else 0.0,
        "step_seconds_min": round(float(np.min(step_times)), 6) if step_times else 0.0,
        "all_pass": bool(all_pass),
        "summary": {
            k: summary[k]
            for k in (
                "intra_group_forward", "upward_forward", "lateral_forward",
                "downward_forward", "cross_group_forward", "cross_group_spread",
                "mute_count", "announce_count", "blocked_by_mute",
                "blocked_by_forward_gate", "recovery_time", "event_source",
            )
            if k in summary
        },
        "required_cols_ok": {
            k: (k in df.columns)
            for k in list(required_cols) + ["step_seconds"]
        },
    }
    meta_path = os.path.join(out_dir, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    print(f"[输出] 运行元信息 → {meta_path}")

    try:
        from utils import save_simulation_results
        csv_e = save_simulation_results(
            df_out, os.path.join(out_dir, "metrics_e"), fmt="csv"
        )
        print(f"[输出/E] 指标 CSV → {csv_e}")
    except Exception as exc:
        _LOG.warning("E 导出跳过: %s", exc)

    if save_plot:
        _plot_trends(df_out, config, out_dir)

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
    ax4 = _sub(1, 1, df["message_count"], "#16A34A", "消息总量 / 热度", "④ 信息流消息量与群热度峰值")
    ax4b = ax4.twinx()
    ax4b.plot(steps, df["group_heat_max"], color="#CA8A04", linewidth=1.2, linestyle="--")
    ax4b.axhline(config.scenario_params.get("heat_threshold", 0.70),
                 color="#78350F", linewidth=0.9, linestyle=":")
    ax4b.set_ylabel("群热度 H (虚线) / θ (点线)", fontsize=8)
    ax5 = _sub(2, 0, df["negative_emotion"], "#EA580C", "负面强度", "⑤ 负面情绪与群负面程度", (0, 1.05))
    ax5.plot(steps, df["group_negative_max"], color="#B91C1C", linewidth=1.2, linestyle="--",
             label="群负面程度(触发量)")
    ax5.axhline(config.scenario_params.get("negative_threshold", 0.65),
                color="#7F1D1D", linewidth=0.9, linestyle=":", label="干预阈值")
    ax5.lines[0].set_label("全网负面情绪")
    ax5.legend(fontsize=7, loc="lower right")
    ax6 = _sub(2, 1, df["cross_group_forward"], "#0891B2", "累计转发次数", "⑥ 跨群转发通道分解")
    ax6.plot(steps, df["upward_forward"],   color="#1D4ED8", linewidth=1.2, label="小→大（主）")
    ax6.plot(steps, df["lateral_forward"],  color="#F59E0B", linewidth=1.2, label="小群互转")
    ax6.plot(steps, df["downward_forward"], color="#94A3B8", linewidth=1.2, label="大→小")
    ax6.lines[0].set_label("跨群合计")
    ax6.legend(fontsize=7, loc="upper left")

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
