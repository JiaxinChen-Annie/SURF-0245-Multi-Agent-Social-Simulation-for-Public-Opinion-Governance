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
    governance_window: Dict[str, Any]
    group_report: List[Dict[str, Any]]
    agent_report: List[Dict[str, Any]]
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
        governance_tick = _governance_tick(self.intervention_params, int(steps))

        for tick in range(int(steps)):
            shock = float(np.exp(-tick / 18.0))
            noise = lambda scale: float(self.random.normal(0, scale))
            active = tick >= governance_tick
            elapsed = max(0, tick - governance_tick)
            effect_curve = (1.0 - float(np.exp(-(elapsed + 1.0) / 8.0))) if active else 0.0
            time_multiplier = _time_profile_multiplier(tick, self.intervention_params)

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

            if active and self.intervention_type == InterventionType.EVENT_INJECTION:
                strength = float(self.intervention_params.get("message_strength", 0.35))
                calm = float(self.intervention_params.get("emotion_calm", 0.08))
                avg_opinion += 0.018 * strength * effect_curve * time_multiplier
                emotion -= calm * 0.020 * effect_curve * time_multiplier
                topic_shift += 0.10 * strength * effect_curve * time_multiplier
                participation_rate += 0.03 * strength * effect_curve * time_multiplier

            elif active and self.intervention_type == InterventionType.NODE_CONTROL:
                control_ratio = float(self.intervention_params.get("control_ratio", 0.08))
                leader_centrality -= 0.75 * control_ratio * effect_curve * time_multiplier
                propagation_speed -= 0.55 * control_ratio * effect_curve * time_multiplier
                polarization -= 0.35 * control_ratio * effect_curve * time_multiplier
                modularity -= 0.20 * control_ratio * effect_curve * time_multiplier

            elif active and self.intervention_type == InterventionType.PLATFORM_PARAM:
                downrank = float(self.intervention_params.get("downrank_factor", 0.18))
                friction = float(self.intervention_params.get("reshare_friction", 0.12))
                propagation_speed -= 0.45 * downrank * effect_curve * time_multiplier
                interaction_density -= 0.35 * friction * effect_curve * time_multiplier
                emotion -= 0.22 * downrank * effect_curve * time_multiplier
                participation_rate -= 0.12 * friction * effect_curve * time_multiplier

            avg_opinion = _clip(avg_opinion, -1.0, 1.0)
            polarization = _clip(polarization, 0.0, 1.0)
            emotion = _clip(emotion, 0.0, 1.0)
            negative_emotion = _clip(0.25 + 0.85 * emotion + noise(0.01), 0.0, 1.0)

            row = {
                    "tick": float(tick),
                    "avg_opinion": avg_opinion,
                    "polarization": polarization,
                    "emotional_contagion": emotion,
                    "negative_emotion": negative_emotion,
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
            row.update(_segment_columns_from_global(row))
            rows.append(row)

        return pd.DataFrame(rows)


class StrategyEvaluator:
    """D module: baseline, official interventions, and three-dimension evaluation."""

    def __init__(
        self,
        model: OpinionModelLike,
        steps: int = 50,
        seed: int = 42,
        tail_window: int = 10,
        eval_window: int = 10,
        rolling_window: int = 5,
    ) -> None:
        self.model = model
        self.steps = int(steps)
        self.seed = int(seed)
        self.tail_window = int(tail_window)
        self.eval_window = int(eval_window)
        self.rolling_window = int(rolling_window)
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
        params = self._normalize_intervention_params(params)
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
            intervention_type, params = self.intervention_meta[name]
            governance_tick = _governance_tick(params, self.steps)
            behavior = self._analyze_behavior(run, governance_tick)
            content = self._analyze_content(run, governance_tick)
            topology = self._analyze_topology(run, governance_tick)
            governance_window = self._analyze_governance_window(run, governance_tick)
            group_report = self._build_segment_report(run, "group", governance_tick)
            agent_report = self._build_segment_report(run, "agent", governance_tick)
            summary = self._build_summary(behavior, content, topology, governance_window)
            results[name] = EvaluationResult(
                intervention_name=name,
                intervention_type=intervention_type.name,
                behavior=behavior,
                content=content,
                topology=topology,
                governance_window=governance_window,
                group_report=group_report,
                agent_report=agent_report,
                summary=summary,
            )
        return results

    def _analyze_behavior(self, run: pd.DataFrame, governance_tick: int) -> DimensionResult:
        baseline_metrics = self._window_mean(
            self.baseline,
            ["participation_rate", "propagation_speed", "interaction_density"],
            *self._post_window_bounds(governance_tick),
        )
        run_metrics = self._window_mean(
            run,
            ["participation_rate", "propagation_speed", "interaction_density"],
            *self._post_window_bounds(governance_tick),
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

    def _analyze_content(self, run: pd.DataFrame, governance_tick: int) -> DimensionResult:
        baseline_metrics = self._window_mean(
            self.baseline,
            ["sentiment_mean", "sentiment_variance", "topic_shift", "polarization"],
            *self._post_window_bounds(governance_tick),
        )
        run_metrics = self._window_mean(
            run,
            ["sentiment_mean", "sentiment_variance", "topic_shift", "polarization"],
            *self._post_window_bounds(governance_tick),
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

    def _analyze_topology(self, run: pd.DataFrame, governance_tick: int) -> DimensionResult:
        baseline_metrics = self._window_mean(
            self.baseline,
            ["network_density", "modularity", "leader_centrality"],
            *self._post_window_bounds(governance_tick),
        )
        run_metrics = self._window_mean(
            run,
            ["network_density", "modularity", "leader_centrality"],
            *self._post_window_bounds(governance_tick),
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
        return {col: round(float(_series_or_zero(tail, col).mean()), 6) for col in columns}

    def _window_mean(
        self,
        frame: Optional[pd.DataFrame],
        columns: List[str],
        start: int,
        end: int,
    ) -> Dict[str, float]:
        if frame is None:
            raise RuntimeError("baseline is not initialized.")
        window = frame.iloc[max(0, start): max(0, end)]
        if window.empty:
            window = frame.tail(1)
        return {col: round(float(_series_or_zero(window, col).mean()), 6) for col in columns}

    def _normalize_intervention_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(params or {})
        if "governance_tick" not in normalized:
            normalized["governance_tick"] = int(
                normalized.get("intervention_tick", min(max(1, self.steps // 3), max(0, self.steps - 1)))
            )
        normalized.setdefault("eval_window", self.eval_window)
        normalized.setdefault("rolling_window", self.rolling_window)
        return normalized

    def _pre_window_bounds(self, governance_tick: int) -> Tuple[int, int]:
        end = max(0, min(int(governance_tick), self.steps))
        start = max(0, end - max(1, self.eval_window))
        return start, end

    def _post_window_bounds(self, governance_tick: int) -> Tuple[int, int]:
        start = max(0, min(int(governance_tick), max(0, self.steps - 1)))
        end = min(self.steps, start + max(1, self.eval_window))
        return start, end

    def _analyze_governance_window(self, run: pd.DataFrame, governance_tick: int) -> Dict[str, Any]:
        pre_start, pre_end = self._pre_window_bounds(governance_tick)
        post_start, post_end = self._post_window_bounds(governance_tick)
        negative_column = "negative_emotion" if "negative_emotion" in run.columns else "emotional_contagion"
        metrics = ["emotional_contagion", negative_column]
        pre = self._window_mean(run, metrics, pre_start, pre_end)
        post = self._window_mean(run, metrics, post_start, post_end)
        baseline_post = self._window_mean(self.baseline, metrics, post_start, post_end)

        negative_roll = _rolling_window_mean(run, negative_column, self.rolling_window, post_start, post_end)
        contagion_roll = _rolling_window_mean(run, "emotional_contagion", self.rolling_window, post_start, post_end)
        return {
            "governance_tick": int(governance_tick),
            "eval_window": int(self.eval_window),
            "rolling_window": int(self.rolling_window),
            "pre_window": {"start": int(pre_start), "end": int(pre_end)},
            "post_window": {"start": int(post_start), "end": int(post_end)},
            "pre_mean": pre,
            "post_mean": post,
            "delta_post_vs_pre": _delta(post, pre),
            "delta_post_vs_baseline": _delta(post, baseline_post),
            "rolling_post_mean": {
                negative_column: round(negative_roll, 6),
                "emotional_contagion": round(contagion_roll, 6),
            },
        }

    def _build_segment_report(
        self,
        run: pd.DataFrame,
        segment_kind: str,
        governance_tick: int,
    ) -> List[Dict[str, Any]]:
        if self.baseline is None:
            raise RuntimeError("baseline is not initialized.")
        names = _segment_names(segment_kind)
        post_start, post_end = self._post_window_bounds(governance_tick)
        report: List[Dict[str, Any]] = []
        for name in names:
            metrics = self._segment_metrics(run, segment_kind, name, post_start, post_end)
            baseline_metrics = self._segment_metrics(self.baseline, segment_kind, name, post_start, post_end)
            report.append(
                {
                    "segment_type": "GroupType" if segment_kind == "group" else "AgentType",
                    "segment": name,
                    "post_window_mean": metrics,
                    "delta_vs_baseline": _delta(metrics, baseline_metrics),
                }
            )
        return report

    def _segment_metrics(
        self,
        frame: pd.DataFrame,
        segment_kind: str,
        segment: str,
        start: int,
        end: int,
    ) -> Dict[str, float]:
        prefix = f"{segment_kind}_{segment}"
        columns = {
            "negative_emotion": _first_existing_column(frame, [f"{prefix}_negative_emotion", f"{segment}_negative_emotion"]),
            "emotional_contagion": _first_existing_column(frame, [f"{prefix}_emotional_contagion", f"{segment}_emotional_contagion"]),
            "avg_opinion": _first_existing_column(frame, [f"{prefix}_avg_opinion", f"{segment}_avg_opinion"]),
            "polarization": _first_existing_column(frame, [f"{prefix}_polarization", f"{segment}_polarization"]),
        }
        factors = _segment_factors(segment_kind, segment)
        window = frame.iloc[max(0, start): max(0, end)]
        if window.empty:
            window = frame.tail(1)
        return {
            "negative_emotion": round(float(_series_or_zero(window, columns["negative_emotion"], "negative_emotion").mean() * factors["negative_emotion"]), 6),
            "emotional_contagion": round(float(_series_or_zero(window, columns["emotional_contagion"], "emotional_contagion").mean() * factors["emotional_contagion"]), 6),
            "avg_opinion": round(float(_series_or_zero(window, columns["avg_opinion"], "avg_opinion").mean() * factors["avg_opinion"]), 6),
            "polarization": round(float(_series_or_zero(window, columns["polarization"], "polarization").mean() * factors["polarization"]), 6),
        }

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
        governance_window: Dict[str, Any],
    ) -> Dict[str, float]:
        speed_drop = -behavior.delta_vs_baseline["propagation_speed"]
        polarization_drop = -content.delta_vs_baseline["polarization"]
        leader_drop = -topology.delta_vs_baseline["leader_centrality"]
        opinion_guidance = content.delta_vs_baseline["sentiment_mean"]
        negative_drop = -governance_window["delta_post_vs_baseline"].get("negative_emotion", 0.0)
        contagion_drop = -governance_window["delta_post_vs_baseline"].get("emotional_contagion", 0.0)
        overall_score = (
            0.22 * speed_drop
            + 0.22 * polarization_drop
            + 0.16 * leader_drop
            + 0.16 * opinion_guidance
            + 0.14 * negative_drop
            + 0.10 * contagion_drop
        )
        return {
            "governance_tick": float(governance_window["governance_tick"]),
            "speed_drop": round(speed_drop, 6),
            "polarization_drop": round(polarization_drop, 6),
            "leader_centrality_drop": round(leader_drop, 6),
            "opinion_guidance": round(opinion_guidance, 6),
            "negative_drop": round(negative_drop, 6),
            "contagion_drop": round(contagion_drop, 6),
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


def _governance_tick(params: Dict[str, Any], steps: int) -> int:
    raw = params.get("governance_tick", params.get("intervention_tick", 0))
    try:
        tick = int(raw)
    except (TypeError, ValueError):
        tick = 0
    return max(0, min(tick, max(0, int(steps) - 1)))


def _series_or_zero(frame: pd.DataFrame, column: Optional[str], fallback: Optional[str] = None) -> pd.Series:
    if column and column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    if fallback and fallback in frame.columns:
        return pd.to_numeric(frame[fallback], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)


def _rolling_window_mean(frame: pd.DataFrame, column: str, window: int, start: int, end: int) -> float:
    series = _series_or_zero(frame, column).rolling(max(1, int(window)), min_periods=1).mean()
    segment = series.iloc[max(0, start): max(0, end)]
    if segment.empty:
        segment = series.tail(1)
    return float(segment.mean()) if not segment.empty else 0.0


def _first_existing_column(frame: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _segment_names(segment_kind: str) -> List[str]:
    if segment_kind == "group":
        return ["DORM", "CLASS", "MAJOR", "CAMPUS"]
    if segment_kind == "agent":
        return ["ORDINARY", "ACTIVE", "RATIONAL", "CONTROLLER"]
    raise ValueError(f"unknown segment kind: {segment_kind}")


def _segment_factors(segment_kind: str, segment: str) -> Dict[str, float]:
    group_factors = {
        "DORM": {"negative_emotion": 1.12, "emotional_contagion": 1.08, "avg_opinion": 1.00, "polarization": 1.05},
        "CLASS": {"negative_emotion": 1.02, "emotional_contagion": 1.00, "avg_opinion": 1.00, "polarization": 1.00},
        "MAJOR": {"negative_emotion": 0.96, "emotional_contagion": 0.96, "avg_opinion": 1.00, "polarization": 0.98},
        "CAMPUS": {"negative_emotion": 0.86, "emotional_contagion": 0.90, "avg_opinion": 1.00, "polarization": 0.92},
    }
    agent_factors = {
        "ORDINARY": {"negative_emotion": 1.00, "emotional_contagion": 1.00, "avg_opinion": 1.00, "polarization": 1.00},
        "ACTIVE": {"negative_emotion": 1.12, "emotional_contagion": 1.15, "avg_opinion": 1.05, "polarization": 1.08},
        "RATIONAL": {"negative_emotion": 0.82, "emotional_contagion": 0.86, "avg_opinion": 0.92, "polarization": 0.78},
        "CONTROLLER": {"negative_emotion": 0.72, "emotional_contagion": 0.76, "avg_opinion": 0.75, "polarization": 0.68},
    }
    source = group_factors if segment_kind == "group" else agent_factors
    return source.get(segment, {"negative_emotion": 1.0, "emotional_contagion": 1.0, "avg_opinion": 1.0, "polarization": 1.0})


def _segment_columns_from_global(row: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for kind in ["group", "agent"]:
        for name in _segment_names(kind):
            factors = _segment_factors(kind, name)
            prefix = f"{kind}_{name}"
            for metric in ["negative_emotion", "emotional_contagion", "avg_opinion", "polarization"]:
                value = float(row.get(metric, 0.0)) * factors[metric]
                if metric in {"negative_emotion", "emotional_contagion", "polarization"}:
                    value = _clip(value, 0.0, 1.0)
                else:
                    value = _clip(value, -1.0, 1.0)
                out[f"{prefix}_{metric}"] = value
    return out


def _time_profile_multiplier(tick: int, params: Dict[str, Any]) -> float:
    profile = params.get("time_profile")
    if isinstance(profile, list):
        for item in profile:
            if not isinstance(item, dict):
                continue
            start = int(item.get("start", 0))
            end = int(item.get("end", start))
            if start <= int(tick) < end:
                return float(item.get("multiplier", 1.0))
    return 1.0
