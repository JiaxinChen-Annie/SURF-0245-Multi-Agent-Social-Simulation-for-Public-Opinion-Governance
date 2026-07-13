from strategy_eval import InterventionType, MockOpinionModel, StrategyEvaluator


def test_strategy_evaluator_standalone() -> None:
    evaluator = StrategyEvaluator(MockOpinionModel(agent_count=200, seed=7), steps=30, seed=7)
    baseline = evaluator.set_baseline()
    assert len(baseline) == 30

    run = evaluator.apply_intervention(
        InterventionType.NODE_CONTROL,
        {"name": "node_control_test", "control_ratio": 0.1, "seed": 7},
    )
    assert len(run) == 30

    results = evaluator.evaluate()
    assert "node_control_test" in results
    assert results["node_control_test"].behavior.dimension == "behavior"
    assert "overall_control_score" in results["node_control_test"].summary


if __name__ == "__main__":
    test_strategy_evaluator_standalone()
    print("test passed")
