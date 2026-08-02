from __future__ import annotations

import argparse
import json
from pathlib import Path

from abc_model_adapter import ABCOpinionModelAdapter
from strategy_eval import InterventionType, StrategyEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D-to-ABC governance evaluation.")
    parser.add_argument("--steps", type=int, default=50, help="simulation ticks")
    parser.add_argument("--agents", type=int, default=50, help="agent count")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--governance-tick", type=int, default=10, help="real official governance tick")
    parser.add_argument("--eval-window", type=int, default=10, help="pre/post ticks used for evaluation")
    parser.add_argument("--rolling-window", type=int, default=5, help="rolling window for negative/contagion")
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
    evaluator = StrategyEvaluator(
        model=base_model,
        steps=args.steps,
        seed=args.seed,
        eval_window=args.eval_window,
        rolling_window=args.rolling_window,
    )
    time_profile = _default_time_profile(args.steps)

    baseline = evaluator.set_baseline()
    event_df = evaluator.apply_intervention(
        InterventionType.EVENT_INJECTION,
        {
            "name": "event_injection_public_info",
            "message_strength": 0.35,
            "emotion_calm": 0.10,
            "governance_tick": args.governance_tick,
            "time_profile": time_profile,
            "seed": args.seed,
        },
    )
    node_df = evaluator.apply_intervention(
        InterventionType.NODE_CONTROL,
        {
            "name": "node_control_leaders",
            "control_ratio": 0.08,
            "governance_tick": args.governance_tick,
            "time_profile": time_profile,
            "seed": args.seed,
        },
    )
    platform_df = evaluator.apply_intervention(
        InterventionType.PLATFORM_PARAM,
        {
            "name": "platform_param_downrank",
            "downrank_factor": 0.18,
            "reshare_friction": 0.12,
            "governance_tick": args.governance_tick,
            "time_profile": time_profile,
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
        time_profile=time_profile,
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
    time_profile: list[dict],
) -> list[str]:
    first = baseline.iloc[0]
    last = baseline.iloc[-1]
    result_json = json.dumps(
        {
            name: {
                "type": result.intervention_type,
                "governance_tick": result.summary["governance_tick"],
                "overall_control_score": result.summary["overall_control_score"],
                "speed_drop": result.summary["speed_drop"],
                "polarization_drop": result.summary["polarization_drop"],
                "negative_drop": result.summary["negative_drop"],
                "contagion_drop": result.summary["contagion_drop"],
                "leader_centrality_drop": result.summary["leader_centrality_drop"],
            }
            for name, result in results.items()
        },
        ensure_ascii=False,
        indent=2,
    )

    lines = [
        "=" * 80,
        "D module -> ABC governance evaluation",
        "=" * 80,
        "",
        "Input from A/B/C DataCollector metrics:",
        f"  - agents          : {args.agents}",
        f"  - steps           : {args.steps}",
        f"  - seed            : {args.seed}",
        f"  - governance_tick : {args.governance_tick}",
        f"  - eval_window     : {args.eval_window}",
        f"  - rolling_window  : {args.rolling_window}",
        f"  - time_profile    : {time_profile}",
        f"  - baseline rows   : {len(baseline)}",
        f"  - metric columns  : {', '.join(baseline.columns[:13])}",
        "",
        "Baseline head/tail:",
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
        "D evaluation output:",
    ]

    for idx, (name, result) in enumerate(results.items(), start=1):
        summary = result.summary
        window = result.governance_window
        lines.append(
            f"  [{idx}] {name}: "
            f"type={result.intervention_type}, "
            f"governance_tick={summary['governance_tick']:.0f}, "
            f"score={summary['overall_control_score']:.4f}, "
            f"speed_drop={summary['speed_drop']:.4f}, "
            f"polarization_drop={summary['polarization_drop']:.4f}, "
            f"negative_drop={summary['negative_drop']:.4f}, "
            f"contagion_drop={summary['contagion_drop']:.4f}, "
            f"leader_drop={summary['leader_centrality_drop']:.4f}"
        )
        lines.append(
            "      windows: "
            f"pre={window['pre_window']}, post={window['post_window']}, "
            f"rolling_post={window['rolling_post_mean']}"
        )
        lines.append("      GroupType report: " + _format_segment_line(result.group_report))
        lines.append("      AgentType report: " + _format_segment_line(result.agent_report))

    lines.extend(
        [
            "",
            "Intervention data rows:",
            f"  - event_injection_public_info : {len(event_df)} rows",
            f"  - node_control_leaders        : {len(node_df)} rows",
            f"  - platform_param_downrank     : {len(platform_df)} rows",
            "",
            "JSON brief:",
            result_json,
            "",
            "=" * 80,
            "Validation:",
            "  A/B/C -> D DataCollector metrics generated.",
            "  D uses governance_tick as the real official intervention tick.",
            "  D records pre/post evaluation windows and rolling negative/contagion windows.",
            "  D exports GroupType and AgentType reports, not only global averages.",
            "  D supports time-period strategy intensity through time_profile.",
            f"  Output directory: {out_dir.resolve()}",
            "=" * 80,
        ]
    )
    return lines


def _default_time_profile(steps: int) -> list[dict[str, float | int | str]]:
    class_end = max(1, int(steps * 0.45))
    return [
        {"label": "class_time", "start": 0, "end": class_end, "multiplier": 0.85},
        {"label": "evening_time", "start": class_end, "end": int(steps), "multiplier": 1.15},
    ]


def _format_segment_line(report: list[dict]) -> str:
    parts: list[str] = []
    for item in report:
        metrics = item.get("post_window_mean", {})
        delta = item.get("delta_vs_baseline", {})
        parts.append(
            f"{item.get('segment')}("
            f"neg={metrics.get('negative_emotion', 0):.3f}, "
            f"contagion={metrics.get('emotional_contagion', 0):.3f}, "
            f"delta_neg={delta.get('negative_emotion', 0):+.3f})"
        )
    return "; ".join(parts)


if __name__ == "__main__":
    main()
