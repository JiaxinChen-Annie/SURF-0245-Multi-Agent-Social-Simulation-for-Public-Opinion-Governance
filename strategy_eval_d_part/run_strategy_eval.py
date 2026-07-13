from __future__ import annotations

import argparse
from pathlib import Path

from strategy_eval import InterventionType, MockOpinionModel, StrategyEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone D module demo.")
    parser.add_argument("--steps", type=int, default=50, help="simulation ticks")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--agents", type=int, default=1000, help="mock agent count")
    parser.add_argument(
        "--out",
        type=str,
        default="strategy_eval_results",
        help="output directory for csv/json results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = MockOpinionModel(agent_count=args.agents, seed=args.seed)
    evaluator = StrategyEvaluator(model=model, steps=args.steps, seed=args.seed)

    evaluator.set_baseline()
    evaluator.apply_intervention(
        InterventionType.EVENT_INJECTION,
        {
            "name": "event_injection_public_info",
            "message_strength": 0.35,
            "emotion_calm": 0.10,
            "seed": args.seed,
        },
    )
    evaluator.apply_intervention(
        InterventionType.NODE_CONTROL,
        {
            "name": "node_control_leaders",
            "control_ratio": 0.08,
            "seed": args.seed,
        },
    )
    evaluator.apply_intervention(
        InterventionType.PLATFORM_PARAM,
        {
            "name": "platform_param_downrank",
            "downrank_factor": 0.18,
            "reshare_friction": 0.12,
            "seed": args.seed,
        },
    )

    results = evaluator.evaluate()
    output_dir = evaluator.export_results(Path(args.out))

    print(f"Saved results to: {output_dir.resolve()}")
    print("\nD module evaluation summary:")
    for name, result in results.items():
        s = result.summary
        print(
            f"- {name}: score={s['overall_control_score']:.4f}, "
            f"speed_drop={s['speed_drop']:.4f}, "
            f"polarization_drop={s['polarization_drop']:.4f}, "
            f"leader_drop={s['leader_centrality_drop']:.4f}, "
            f"opinion_guidance={s['opinion_guidance']:.4f}"
        )


if __name__ == "__main__":
    main()
