from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
import json

import numpy as np
import pandas as pd


class InterventionType(IntEnum):
    """Official intervention categories frozen in the Week 1 interface table."""

    EVENT_INJECTION = 0
    NODE_CONTROL = 1
    PLATFORM_PARAM = 2


@dataclass
class DimensionResult:
    dimension: str
    metrics: Dict[str, float]
    delta_vs_baseline: Dict[str, float]
    interpretation: str


@dataclass
class EvaluationResult:
    intervention_name: str
    intervention_type: str
    behavior: DimensionResult
    content: DimensionResult
    topology: DimensionResult
    summary: Dict[str, float]


class OpinionModelLike(Protocol):
    """Minimum interface D needs from A's OpinionModel.

    Real OpinionModel only needs to provide clone_with_intervention(...) and run(...).
    The standalone MockOpinionModel below implements the same protocol.
    """

    def clone_with_intervention(
        self,
        intervention_type: Optional[InterventionType],
        params: Optional[Dict[str, Any]],
        seed: int,
    ) -> "OpinionModelLike":
        ...

    def run(self, steps: int) -> pd.DataFrame:
        ...


class MockOpinionModel:
    """Standalone simulator for D module development before A/B/C are ready.

    This is not the final project model. It produces stable, seed-controlled
    time series with the same columns StrategyEvaluator expects from OpinionModel.
    """

    def __init__(
        self,
        agent_count: int = 1000,
        seed: int = 42,
        intervention_type: Optional[InterventionType] = None,
        intervention_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.agent_count = int(agent_count)
        self.seed = int(seed)
        self.intervention_type = intervention_type
        self.intervention_params = intervention_params or {}
        self.random = np.random.default_rng(self.seed)

    def clone_with_intervention(
        self,
        intervention_type: Optional[InterventionType],
        params: Optional[Dict[str, Any]],
        seed: int,
    ) -> "MockOpinionModel":
        return MockOpinionModel(
            agent_count=self.agent_count,
            seed=seed,
            intervention_type=intervention_type,
            intervention_params=params or {},
        )

    def run(self, steps: int) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        avg_opinion = -0.18 + self.random.normal(0, 0.015)
        polarization = 0.42 + self.random.normal(0, 0.01)
        emotion = 0.34 + self.random.normal(0, 0.01)

        for tick in range(int(steps)):
            shock = float(np.exp(-tick / 18.0))
            noise = lambda scale: float(self.random.normal(0, scale))

            participation_rate = 0.25 + 0.15 * shock + noise(0.015)
            propagation_speed = 0.52 + 0.22 * shock + noise(0.02)
            interaction_density = 0.18 + 0.10 * shock + noise(0.01)
            topic_shift = 0.12 + 0.22 * shock + noise(0.012)
            network_density = 0.10 + 0.03 * shock + noise(0.004)
            modularity = 0.38 + 0.08 * shock + noise(0.01)
            leader_centrality = 0.55 + 0.14 * shock + noise(0.015)

            avg_opinion += 0.012 * shock + noise(0.015)
            polarization += 0.006 * shock + noise(0.009)
            emotion += 0.010 * shock + noise(0.01)

            if self.intervention_type == InterventionType.EVENT_INJECTION:
                strength = float(self.intervention_params.get("message_strength", 0.35))
                calm = float(self.intervention_params.get("emotion_calm", 0.08))
                avg_opinion += 0.018 * strength
                emotion -= calm * 0.020
                topic_shift += 0.10 * strength
                participation_rate += 0.03 * strength

            elif self.intervention_type == InterventionType.NODE_CONTROL:
                control_ratio = float(self.intervention_params.get("control_ratio", 0.08))
                leader_centrality -= 0.75 * control_ratio
                propagation_speed -= 0.55 * control_ratio
                polarization -= 0.35 * control_ratio
                modularity -= 0.20 * control_ratio

            elif self.intervention_type == InterventionType.PLATFORM_PARAM:
                downrank = float(self.intervention_params.get("downrank_factor", 0.18))
                friction = float(self.intervention_params.get("reshare_friction", 0.12))
                propagation_speed -= 0.45 * downrank
                interaction_density -= 0.35 * friction
                emotion -= 0.22 * downrank
                participation_rate -= 0.12 * friction

            avg_opinion = _clip(avg_opinion, -1.0, 1.0)
            polarization = _clip(polarization, 0.0, 1.0)
            emotion = _clip(emotion, 0.0, 1.0)

            rows.append(
                {
                    "tick": float(tick),
                    "avg_opinion": avg_opinion,
                    "polarization": polarization,
                    "emotional_contagion": emotion,
                    "participation_rate": _clip(participation_rate, 0.0, 1.0),
                    "propagation_speed": _clip(propagation_speed, 0.0, 1.0),
                    "interaction_density": _clip(interaction_density, 0.0, 1.0),
                    "sentiment_mean": avg_opinion,
                    "sentiment_variance": polarization,
                    "topic_shift": _clip(topic_shift, 0.0, 1.0),
                    "network_density": _clip(network_density, 0.0, 1.0),
                    "modularity": _clip(modularity, 0.0, 1.0),
                    "leader_centrality": _clip(leader_centrality, 0.0, 1.0),
                }
            )

        return pd.DataFrame(rows)


class StrategyEvaluator:
    """D module: baseline, official interventions, and three-dimension evaluation."""

    def __init__(
        self,
        model: OpinionModelLike,
        steps: int = 50,
        seed: int = 42,
        tail_window: int = 10,
    ) -> None:
        self.model = model
        self.steps = int(steps)
        self.seed = int(seed)
        self.tail_window = int(tail_window)
        self.baseline: Optional[pd.DataFrame] = None
        self.intervention_runs: Dict[str, pd.DataFrame] = {}
        self.intervention_meta: Dict[str, Tuple[InterventionType, Dict[str, Any]]] = {}

    def set_baseline(self) -> pd.DataFrame:
        """Run one no-intervention simulation and cache it as baseline."""

        baseline_model = self.model.clone_with_intervention(None, {}, self.seed)
        self.baseline = baseline_model.run(self.steps)
        self._validate_metrics(self.baseline, "baseline")
        return self.baseline

    def apply_intervention(
        self,
        intervention_type: InterventionType,
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        """Run one intervention scenario.

        Args:
            intervention_type: EVENT_INJECTION / NODE_CONTROL / PLATFORM_PARAM.
            params: Strategy parameters, for example {"message_strength": 0.35}.

        Returns:
            Time-series metrics DataFrame for this intervention.
        """

        if self.baseline is None:
            self.set_baseline()

        intervention_type = InterventionType(intervention_type)
        name = _intervention_name(intervention_type, params)
        run_seed = int(params.get("seed", self.seed))
        model = self.model.clone_with_intervention(intervention_type, params, run_seed)
        result = model.run(self.steps)
        self._validate_metrics(result, name)

        self.intervention_runs[name] = result
        self.intervention_meta[name] = (intervention_type, dict(params))
        return result

    def evaluate(self) -> Dict[str, EvaluationResult]:
        """Evaluate all applied interventions against baseline."""

        if self.baseline is None:
            raise RuntimeError("set_baseline() must be called before evaluate().")
        if not self.intervention_runs:
            raise RuntimeError("apply_intervention() must be called before evaluate().")

        results: Dict[str, EvaluationResult] = {}
        for name, run in self.intervention_runs.items():
            intervention_type, _ = self.intervention_meta[name]
            behavior = self._analyze_behavior(run)
            content = self._analyze_content(run)
            topology = self._analyze_topology(run)
            summary = self._build_summary(behavior, content, topology)
            results[name] = EvaluationResult(
                intervention_name=name,
                intervention_type=intervention_type.name,
                behavior=behavior,
                content=content,
                topology=topology,
                summary=summary,
            )
        return results

    def _analyze_behavior(self, run: pd.DataFrame) -> DimensionResult:
        baseline_metrics = self._tail_mean(
            self.baseline,
            ["participation_rate", "propagation_speed", "interaction_density"],
        )
        run_metrics = self._tail_mean(
            run,
            ["participation_rate", "propagation_speed", "interaction_density"],
        )
        delta = _delta(run_metrics, baseline_metrics)
        return DimensionResult(
            dimension="behavior",
            metrics=run_metrics,
            delta_vs_baseline=delta,
            interpretation=(
                "传播速度下降表示扩散被抑制；参与率上升可能表示信息公开后讨论增加。"
            ),
        )

    def _analyze_content(self, run: pd.DataFrame) -> DimensionResult:
        baseline_metrics = self._tail_mean(
            self.baseline,
            ["sentiment_mean", "sentiment_variance", "topic_shift", "polarization"],
        )
        run_metrics = self._tail_mean(
            run,
            ["sentiment_mean", "sentiment_variance", "topic_shift", "polarization"],
        )
        delta = _delta(run_metrics, baseline_metrics)
        return DimensionResult(
            dimension="content",
            metrics=run_metrics,
            delta_vs_baseline=delta,
            interpretation=(
                "情感方差/极化度下降表示内容分歧收敛；topic_shift表示议题迁移程度。"
            ),
        )

    def _analyze_topology(self, run: pd.DataFrame) -> DimensionResult:
        baseline_metrics = self._tail_mean(
            self.baseline,
            ["network_density", "modularity", "leader_centrality"],
        )
        run_metrics = self._tail_mean(
            run,
            ["network_density", "modularity", "leader_centrality"],
        )
        delta = _delta(run_metrics, baseline_metrics)
        return DimensionResult(
            dimension="topology",
            metrics=run_metrics,
            delta_vs_baseline=delta,
            interpretation=(
                "意见领袖中心性下降表示关键节点放大效应被削弱；模块度下降表示圈层隔离减弱。"
            ),
        )

    def export_results(self, output_dir: Union[str, Path]) -> Path:
        """Export baseline/intervention CSV and evaluation summary JSON."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.baseline is not None:
            self.baseline.to_csv(output_path / "baseline.csv", index=False)
        for name, frame in self.intervention_runs.items():
            safe_name = name.replace(" ", "_").replace("/", "_")
            frame.to_csv(output_path / f"{safe_name}.csv", index=False)

        evaluated = self.evaluate()
        summary = {name: _dataclass_to_dict(result) for name, result in evaluated.items()}
        with (output_path / "evaluation_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return output_path

    def _tail_mean(self, frame: Optional[pd.DataFrame], columns: List[str]) -> Dict[str, float]:
        if frame is None:
            raise RuntimeError("baseline is not initialized.")
        tail = frame.tail(max(1, self.tail_window))
        return {col: round(float(tail[col].mean()), 6) for col in columns}

    def _validate_metrics(self, frame: pd.DataFrame, name: str) -> None:
        required = {
            "tick",
            "participation_rate",
            "propagation_speed",
            "interaction_density",
            "sentiment_mean",
            "sentiment_variance",
            "topic_shift",
            "polarization",
            "network_density",
            "modularity",
            "leader_centrality",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} metrics missing columns: {missing}")
        if len(frame) != self.steps:
            raise ValueError(f"{name} expected {self.steps} rows, got {len(frame)}")

    def _build_summary(
        self,
        behavior: DimensionResult,
        content: DimensionResult,
        topology: DimensionResult,
    ) -> Dict[str, float]:
        speed_drop = -behavior.delta_vs_baseline["propagation_speed"]
        polarization_drop = -content.delta_vs_baseline["polarization"]
        leader_drop = -topology.delta_vs_baseline["leader_centrality"]
        opinion_guidance = content.delta_vs_baseline["sentiment_mean"]
        overall_score = (
            0.30 * speed_drop
            + 0.30 * polarization_drop
            + 0.20 * leader_drop
            + 0.20 * opinion_guidance
        )
        return {
            "speed_drop": round(speed_drop, 6),
            "polarization_drop": round(polarization_drop, 6),
            "leader_centrality_drop": round(leader_drop, 6),
            "opinion_guidance": round(opinion_guidance, 6),
            "overall_control_score": round(overall_score, 6),
        }


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _delta(current: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    return {key: round(current[key] - baseline[key], 6) for key in current}


def _intervention_name(intervention_type: InterventionType, params: Dict[str, Any]) -> str:
    label = params.get("name")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return intervention_type.name.lower()


def _dataclass_to_dict(value: Any) -> Dict[str, Any]:
    return asdict(value)
