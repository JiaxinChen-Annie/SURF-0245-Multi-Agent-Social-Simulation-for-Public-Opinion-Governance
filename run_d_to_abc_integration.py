from __future__ import annotations

import argparse
import json
from pathlib import Path

from abc_model_adapter import ABCOpinionModelAdapter
from strategy_eval import InterventionType, StrategyEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D-to-ABC integration test.")
    parser.add_argument("--steps", type=int, default=50, help="simulation ticks")
    parser.add_argument("--agents", type=int, default=50, help="agent count")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--out",
        type=str,
        default="d_to_abc_results",
        help="output directory for integration results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_model = ABCOpinionModelAdapter(agent_count=args.agents, seed=args.seed)
    evaluator = StrategyEvaluator(model=base_model, steps=args.steps, seed=args.seed)

    baseline = evaluator.set_baseline()
    event_df = evaluator.apply_intervention(
        InterventionType.EVENT_INJECTION,
        {
            "name": "event_injection_public_info",
            "message_strength": 0.35,
            "emotion_calm": 0.10,
            "seed": args.seed,
        },
    )
    node_df = evaluator.apply_intervention(
        InterventionType.NODE_CONTROL,
        {
            "name": "node_control_leaders",
            "control_ratio": 0.08,
            "seed": args.seed,
        },
    )
    platform_df = evaluator.apply_intervention(
        InterventionType.PLATFORM_PARAM,
        {
            "name": "platform_param_downrank",
            "downrank_factor": 0.18,
            "reshare_friction": 0.12,
            "seed": args.seed,
        },
    )

    results = evaluator.evaluate()
    out_dir = evaluator.export_results(Path(args.out))
    report_path = out_dir / "d_to_abc_integration_report.txt"

    lines = _build_report_lines(
        args=args,
        baseline=baseline,
        event_df=event_df,
        node_df=node_df,
        platform_df=platform_df,
        results=results,
        out_dir=out_dir,
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def _build_report_lines(
    args: argparse.Namespace,
    baseline,
    event_df,
    node_df,
    platform_df,
    results,
    out_dir: Path,
) -> list[str]:
    first = baseline.iloc[0]
    last = baseline.iloc[-1]
    result_json = json.dumps(
        {
            name: {
                "type": result.intervention_type,
                "overall_control_score": result.summary["overall_control_score"],
                "speed_drop": result.summary["speed_drop"],
                "polarization_drop": result.summary["polarization_drop"],
                "leader_centrality_drop": result.summary["leader_centrality_drop"],
            }
            for name, result in results.items()
        },
        ensure_ascii=False,
        indent=2,
    )

    lines = [
        "=" * 80,
        "📥 D 模块接入 ABC 联调测试",
        "=" * 80,
        "",
        "▶ 输入给 D 模块（来自 A/B/C 汇总后的 DataCollector metrics）:",
        f"  - agents          : {args.agents}",
        f"  - steps           : {args.steps}",
        f"  - seed            : {args.seed}",
        f"  - baseline rows   : {len(baseline)}",
        f"  - metric columns  : {', '.join(baseline.columns[:13])}",
        "",
        "  Baseline 首尾指标:",
        (
            "  [tick 0] "
            f"avg_opinion={first['avg_opinion']:.3f}, "
            f"polarization={first['polarization']:.3f}, "
            f"propagation={first['propagation_speed']:.3f}, "
            f"participation={first['participation_rate']:.3f}"
        ),
        (
            f"  [tick {int(last['tick'])}] "
            f"avg_opinion={last['avg_opinion']:.3f}, "
            f"polarization={last['polarization']:.3f}, "
            f"propagation={last['propagation_speed']:.3f}, "
            f"participation={last['participation_rate']:.3f}"
        ),
        "",
        "▶ D 模块评估输出（三类干预与 baseline 对照）:",
    ]

    for idx, (name, result) in enumerate(results.items(), start=1):
        summary = result.summary
        lines.append(
            f"  [{idx}] {name}: "
            f"type={result.intervention_type}, "
            f"score={summary['overall_control_score']:.4f}, "
            f"speed_drop={summary['speed_drop']:.4f}, "
            f"polarization_drop={summary['polarization_drop']:.4f}, "
            f"leader_drop={summary['leader_centrality_drop']:.4f}"
        )

    lines.extend(
        [
            "",
            "▶ 干预数据行数检查:",
            f"  - event_injection_public_info : {len(event_df)} rows",
            f"  - node_control_leaders        : {len(node_df)} rows",
            f"  - platform_param_downrank     : {len(platform_df)} rows",
            "",
            "▶ JSON 摘要:",
            result_json,
            "",
            "=" * 80,
            "✅ 联调验证:",
            "  A/B/C → D: DataCollector metrics 已生成 ✅",
            "  D: baseline 已建立 ✅",
            "  D: EVENT_INJECTION / NODE_CONTROL / PLATFORM_PARAM 三类干预已评估 ✅",
            "  D: CSV / JSON / TXT 结果已导出 ✅",
            f"  输出目录: {out_dir.resolve()}",
            "=" * 80,
        ]
    )
    return lines


if __name__ == "__main__":
    main()
