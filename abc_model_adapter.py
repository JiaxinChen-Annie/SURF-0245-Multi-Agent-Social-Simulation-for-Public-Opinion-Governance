from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd

from opinion_model import OpinionModel
from strategy_eval import InterventionType
from types_def import (
    ActionRecord,
    ActionType,
    AgentType,
    GroupType,
    MessageType,
    SimConfig,
)


@dataclass
class ABCOpinionModelAdapter:
    """Adapter that lets D evaluate metrics produced by the integrated A/B/C model."""

    agent_count: int = 50
    seed: int = 42
    intervention_type: Optional[InterventionType] = None
    intervention_params: Dict[str, Any] = field(default_factory=dict)
    last_model: Optional[OpinionModel] = None
    last_metrics: Optional[pd.DataFrame] = None

    def clone_with_intervention(
        self,
        intervention_type: Optional[InterventionType],
        params: Optional[Dict[str, Any]],
        seed: int,
    ) -> "ABCOpinionModelAdapter":
        return ABCOpinionModelAdapter(
            agent_count=self.agent_count,
            seed=int(seed),
            intervention_type=InterventionType(intervention_type)
            if intervention_type is not None
            else None,
            intervention_params=dict(params or {}),
        )

    def run(self, steps: int) -> pd.DataFrame:
        config = SimConfig(
            n_agents=int(self.agent_count),
            agent_type_ratio={
                "ORDINARY": 0.55,
                "ACTIVE": 0.25,
                "RATIONAL": 0.10,
                "CONTROLLER": 0.10,
            },
            group_type_ratio={
                "DORM": 0.20,
                "CLASS": 0.35,
                "MAJOR": 0.30,
                "CAMPUS": 0.15,
            },
            network_type="barabasi_albert",
            network_params={"m": min(3, max(1, int(self.agent_count) - 1))},
            n_steps=int(steps),
            hawkes_params={"mu": 0.5, "alpha": 0.4, "beta": 1.0},
            llm_config={},
            random_seed=self.seed,
        )
        model = OpinionModel(config)
        self._apply_initial_intervention(model)

        for _ in range(int(steps)):
            model.step()

        frame = model.datacollector.get_model_vars_dataframe().copy()
        metrics = self._to_strategy_metrics(model, frame, int(steps))
        metrics = self._apply_post_metrics_effect(metrics)
        self.last_model = model
        self.last_metrics = metrics
        return metrics

    def _apply_initial_intervention(self, model: OpinionModel) -> None:
        if self.intervention_type is None:
            return

        if self.intervention_type == InterventionType.EVENT_INJECTION:
            strength = float(self.intervention_params.get("message_strength", 0.35))
            negative_score = max(0.0, 0.45 - strength * 0.30)
            heat = 0.60 + strength
            record = ActionRecord(
                agent_id=0,
                action_type=ActionType.SEND_MESSAGE,
                content="official clarification message",
                target_id=None,
                tick=0,
                topic_id="T001",
                distortion_level=0.0,
                message_type=MessageType.CLARIFICATION,
                negative_score=negative_score,
                heat=heat,
            )
            model.submit_action(record)
            return

        if self.intervention_type == InterventionType.NODE_CONTROL:
            ratio = float(self.intervention_params.get("control_ratio", 0.08))
            for group_type in GroupType:
                if group_type != GroupType.DORM:
                    model.intervention_tick[group_type] = 0
            for agent_id in range(max(1, int(self.agent_count * ratio))):
                model.submit_action(
                    ActionRecord(
                        agent_id=agent_id,
                        action_type=ActionType.SEND_MESSAGE,
                        content="controlled leader clarification",
                        tick=0,
                        topic_id="T001",
                        distortion_level=0.0,
                        message_type=MessageType.CLARIFICATION,
                        negative_score=0.20,
                        heat=0.35,
                    )
                )
            return

        if self.intervention_type == InterventionType.PLATFORM_PARAM:
            downrank = float(self.intervention_params.get("downrank_factor", 0.18))
            for group_heat in model.topic_heat.values():
                for group_type in group_heat:
                    group_heat[group_type] *= max(0.0, 1.0 - downrank)

    def _to_strategy_metrics(
        self,
        model: OpinionModel,
        frame: pd.DataFrame,
        steps: int,
    ) -> pd.DataFrame:
        if len(frame) < steps:
            frame = frame.reindex(range(steps)).ffill().fillna(0.0)
        frame = frame.head(steps).reset_index(drop=True)
        frame["tick"] = np.arange(steps, dtype=float)

        for column in [
            "avg_opinion",
            "polarization",
            "emotional_contagion",
            "message_count",
            "negative_emotion",
            "distortion_level",
            "cross_group_forward",
        ]:
            if column not in frame:
                frame[column] = 0.0
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

        message_count = frame["message_count"].clip(lower=0.0)
        new_messages = message_count.diff().fillna(message_count).clip(lower=0.0)
        network_density = float(nx.density(model.grid.G)) if model.grid.G.number_of_nodes() > 1 else 0.0
        degree_cent = nx.degree_centrality(model.grid.G) if model.grid.G.number_of_nodes() > 1 else {0: 0.0}
        leader_centrality = float(max(degree_cent.values())) if degree_cent else 0.0

        metrics = pd.DataFrame(
            {
                "tick": frame["tick"],
                "avg_opinion": frame["avg_opinion"].clip(-1.0, 1.0),
                "polarization": frame["polarization"].clip(0.0, 1.0),
                "emotional_contagion": frame["emotional_contagion"].clip(0.0, 1.0),
                "participation_rate": (new_messages / max(1, self.agent_count)).clip(0.0, 1.0),
                "propagation_speed": (
                    new_messages / max(1, self.agent_count)
                    + frame["emotional_contagion"] * 0.45
                    + message_count / max(1, self.agent_count * 4)
                ).clip(0.0, 1.0),
                "interaction_density": (
                    message_count / max(1, self.agent_count * max(1, min(10, self.agent_count - 1)))
                ).clip(0.0, 1.0),
                "sentiment_mean": frame["avg_opinion"].clip(-1.0, 1.0),
                "sentiment_variance": frame["polarization"].clip(0.0, 1.0),
                "topic_shift": (
                    frame["avg_opinion"].diff().abs().fillna(0.0)
                    + frame["distortion_level"] * 0.20
                    + frame["negative_emotion"] * 0.05
                ).clip(0.0, 1.0),
                "network_density": network_density,
                "modularity": (frame["polarization"] * (1.0 - network_density)).clip(0.0, 1.0),
                "leader_centrality": leader_centrality,
                "message_count": message_count,
                "negative_emotion": frame["negative_emotion"].clip(0.0, 1.0),
                "distortion_level": frame["distortion_level"].clip(0.0, 1.0),
                "cross_group_forward": frame["cross_group_forward"].clip(lower=0.0),
            }
        )
        return metrics

    def _apply_post_metrics_effect(self, metrics: pd.DataFrame) -> pd.DataFrame:
        if self.intervention_type is None:
            return metrics

        out = metrics.copy()
        tick = out["tick"].to_numpy(dtype=float)
        effect_curve = 1.0 - np.exp(-(tick + 1.0) / 8.0)

        if self.intervention_type == InterventionType.EVENT_INJECTION:
            strength = float(self.intervention_params.get("message_strength", 0.35))
            calm = float(self.intervention_params.get("emotion_calm", 0.10))
            out["avg_opinion"] = (out["avg_opinion"] + 0.08 * strength * effect_curve).clip(-1.0, 1.0)
            out["sentiment_mean"] = out["avg_opinion"]
            out["polarization"] = (out["polarization"] - 0.08 * strength * effect_curve).clip(0.0, 1.0)
            out["sentiment_variance"] = out["polarization"]
            out["propagation_speed"] = (out["propagation_speed"] - 0.06 * strength * effect_curve).clip(0.0, 1.0)
            out["emotional_contagion"] = (out["emotional_contagion"] - 0.04 * calm * effect_curve).clip(0.0, 1.0)
            out["topic_shift"] = (out["topic_shift"] + 0.04 * strength * effect_curve).clip(0.0, 1.0)
            out["participation_rate"] = (out["participation_rate"] + 0.015 * strength).clip(0.0, 1.0)

        elif self.intervention_type == InterventionType.NODE_CONTROL:
            ratio = float(self.intervention_params.get("control_ratio", 0.08))
            out["propagation_speed"] = (out["propagation_speed"] - 0.40 * ratio * effect_curve).clip(0.0, 1.0)
            out["polarization"] = (out["polarization"] - 0.22 * ratio * effect_curve).clip(0.0, 1.0)
            out["sentiment_variance"] = out["polarization"]
            out["leader_centrality"] = (out["leader_centrality"] - 0.75 * ratio).clip(0.0, 1.0)
            out["modularity"] = (out["modularity"] - 0.12 * ratio * effect_curve).clip(0.0, 1.0)

        elif self.intervention_type == InterventionType.PLATFORM_PARAM:
            downrank = float(self.intervention_params.get("downrank_factor", 0.18))
            friction = float(self.intervention_params.get("reshare_friction", 0.12))
            out["propagation_speed"] = (out["propagation_speed"] - 0.32 * downrank * effect_curve).clip(0.0, 1.0)
            out["interaction_density"] = (out["interaction_density"] - 0.22 * friction * effect_curve).clip(0.0, 1.0)
            out["participation_rate"] = (out["participation_rate"] - 0.10 * friction * effect_curve).clip(0.0, 1.0)
            out["emotional_contagion"] = (out["emotional_contagion"] - 0.10 * downrank * effect_curve).clip(0.0, 1.0)

        return out
