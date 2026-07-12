"""
多智能体舆情仿真 — 自定义类型定义
数据契约，所有模块共享
"""

from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any


class AgentType(IntEnum):
    """智能体类型"""
    PUBLIC = 0           # 普通网民
    OPINION_LEADER = 1   # 意见领袖
    MEDIA = 2            # 媒体
    OFFICIAL = 3         # 官方


class ActionType(IntEnum):
    """动作类型"""
    POST = 0        # 发帖
    COMMENT = 1     # 评论
    REPOST = 2      # 转发
    LIKE = 3        # 点赞
    SILENT = 4      # 沉默


class InterventionType:
    """干预类型 (使用 int 常量)"""
    EVENT_INJECTION = 0       # 事件注入
    NODE_CONTROL = 1          # 节点控制
    PLATFORM_PARAM = 2        # 平台参数调整


class Personality:
    """大五人格: 各维度 ∈ [0, 1]"""
    def __init__(
        self,
        openness: float = 0.5,
        conscientiousness: float = 0.5,
        extraversion: float = 0.5,
        agreeableness: float = 0.5,
        neuroticism: float = 0.5,
    ):
        self.openness = self._clamp01(openness)
        self.conscientiousness = self._clamp01(conscientiousness)
        self.extraversion = self._clamp01(extraversion)
        self.agreeableness = self._clamp01(agreeableness)
        self.neuroticism = self._clamp01(neuroticism)

    @staticmethod
    def _clamp01(v: float) -> float:
        return max(0.0, min(1.0, v))

    def to_dict(self) -> Dict[str, float]:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "Personality":
        return cls(**{k: float(v) for k, v in d.items()})


class EmotionState:
    """情绪状态"""
    def __init__(self, valence: float = 0.0, arousal: float = 0.5):
        self.valence = max(-1.0, min(1.0, valence))   # ∈ [-1, 1]
        self.arousal = max(0.0, min(1.0, arousal))     # ∈ [0, 1]

    def to_dict(self) -> Dict[str, float]:
        return {"valence": self.valence, "arousal": self.arousal}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "EmotionState":
        return cls(valence=float(d.get("valence", 0.0)),
                   arousal=float(d.get("arousal", 0.5)))


class IdentityBelief:
    """身份信念"""
    def __init__(self, agent_type: AgentType = AgentType.PUBLIC,
                 role_desc: str = "", stance_prior: float = 0.0):
        self.agent_type = agent_type
        self.role_desc = role_desc
        self.stance_prior = max(-1.0, min(1.0, stance_prior))

    def to_dict(self) -> Dict:
        return {"agent_type": int(self.agent_type), "role_desc": self.role_desc,
                "stance_prior": self.stance_prior}

    @classmethod
    def from_dict(cls, d: Dict) -> "IdentityBelief":
        return cls(agent_type=AgentType(d.get("agent_type", 0)),
                   role_desc=str(d.get("role_desc", "")),
                   stance_prior=float(d.get("stance_prior", 0.0)))


class PsychologyBelief:
    """心理信念"""
    def __init__(self, personality: Optional[Personality] = None,
                 risk_aversion: float = 0.5):
        self.personality = personality or Personality()
        self.risk_aversion = max(0.0, min(1.0, risk_aversion))

    def to_dict(self) -> Dict:
        return {"personality": self.personality.to_dict(),
                "risk_aversion": self.risk_aversion}

    @classmethod
    def from_dict(cls, d: Dict) -> "PsychologyBelief":
        return cls(
            personality=Personality.from_dict(d.get("personality", {})),
            risk_aversion=float(d.get("risk_aversion", 0.5)),
        )


class OpinionBelief:
    """观点信念"""
    def __init__(self, event_id: str = "", opinion_value: float = 0.0,
                 confidence: float = 0.5):
        self.event_id = event_id
        self.opinion_value = max(-1.0, min(1.0, opinion_value))
        self.confidence = max(0.0, min(1.0, confidence))

    def to_dict(self) -> Dict:
        return {"event_id": self.event_id,
                "opinion_value": self.opinion_value,
                "confidence": self.confidence}

    @classmethod
    def from_dict(cls, d: Dict) -> "OpinionBelief":
        return cls(event_id=str(d.get("event_id", "")),
                   opinion_value=float(d.get("opinion_value", 0.0)),
                   confidence=float(d.get("confidence", 0.5)))


class BeliefSystem:
    """信念系统"""
    def __init__(self, identity: Optional[IdentityBelief] = None,
                 psychology: Optional[PsychologyBelief] = None,
                 opinions: Optional[Dict[str, OpinionBelief]] = None,
                 emotion: Optional[EmotionState] = None):
        self.identity = identity or IdentityBelief()
        self.psychology = psychology or PsychologyBelief()
        self.opinions = opinions or {}
        self.emotion = emotion or EmotionState()

    def to_dict(self) -> Dict:
        return {"identity": self.identity.to_dict(),
                "psychology": self.psychology.to_dict(),
                "opinions": {k: v.to_dict() for k, v in self.opinions.items()},
                "emotion": self.emotion.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict) -> "BeliefSystem":
        return cls(
            identity=IdentityBelief.from_dict(d.get("identity", {})),
            psychology=PsychologyBelief.from_dict(d.get("psychology", {})),
            opinions={k: OpinionBelief.from_dict(v)
                      for k, v in d.get("opinions", {}).items()},
            emotion=EmotionState.from_dict(d.get("emotion", {})),
        )


class SocialInfo:
    """社交信息"""
    def __init__(self, source_id: int = 0, action_type: ActionType = ActionType.POST,
                 content: str = "", event_id: str = "", opinion_value: float = 0.0,
                 timestamp: int = 0):
        self.source_id = source_id
        self.action_type = action_type
        self.content = content
        self.event_id = event_id
        self.opinion_value = max(-1.0, min(1.0, opinion_value))
        self.timestamp = timestamp

    def to_dict(self) -> Dict:
        return {"source_id": self.source_id,
                "action_type": int(self.action_type),
                "content": self.content, "event_id": self.event_id,
                "opinion_value": self.opinion_value,
                "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, d: Dict) -> "SocialInfo":
        return cls(source_id=int(d.get("source_id", 0)),
                   action_type=ActionType(d.get("action_type", 0)),
                   content=str(d.get("content", "")),
                   event_id=str(d.get("event_id", "")),
                   opinion_value=float(d.get("opinion_value", 0.0)),
                   timestamp=int(d.get("timestamp", 0)))


class Perception:
    """感知信息"""
    def __init__(self, neighbors: Optional[List[SocialInfo]] = None,
                 trending_topics: Optional[List[Tuple[str, float]]] = None,
                 mentions: Optional[List[SocialInfo]] = None,
                 tick: int = 0):
        self.neighbors = neighbors or []
        self.trending_topics = trending_topics or []
        self.mentions = mentions or []
        self.tick = tick


class MemoryRecord:
    """记忆记录"""
    def __init__(self, tick: int = 0, info: Optional[SocialInfo] = None,
                 relevance: float = 0.5):
        self.tick = tick
        self.info = info or SocialInfo()
        self.relevance = max(0.0, min(1.0, relevance))

    def to_dict(self) -> Dict:
        return {"tick": self.tick, "info": self.info.to_dict(),
                "relevance": self.relevance}


class ActionRecord:
    """动作记录 — B (SocialAgent) 输出, A (OpinionModel) 消费"""
    def __init__(self, agent_id: int = 0,
                 action_type: ActionType = ActionType.POST,
                 content: str = "", event_id: str = "",
                 opinion_value: float = 0.0,
                 target_id: Optional[int] = None,
                 tick: int = 0):
        self.agent_id = agent_id
        self.action_type = action_type
        self.content = content
        self.event_id = event_id
        self.opinion_value = max(-1.0, min(1.0, opinion_value))
        self.target_id = target_id
        self.tick = tick

    def to_dict(self) -> Dict:
        return {"agent_id": self.agent_id,
                "action_type": int(self.action_type),
                "content": self.content,
                "event_id": self.event_id,
                "opinion_value": self.opinion_value,
                "target_id": self.target_id,
                "tick": self.tick}

    @classmethod
    def from_dict(cls, d: Dict) -> "ActionRecord":
        return cls(agent_id=int(d["agent_id"]),
                   action_type=ActionType(d["action_type"]),
                   content=str(d.get("content", "")),
                   event_id=str(d.get("event_id", "")),
                   opinion_value=float(d.get("opinion_value", 0.0)),
                   target_id=d.get("target_id"),
                   tick=int(d["tick"]))


class SimConfig:
    """仿真实验配置 — load_config 返回类型"""
    def __init__(self, n_agents: int = 50,
                 agent_type_ratio: Optional[Dict[str, float]] = None,
                 network_type: str = "small_world",
                 network_params: Optional[Dict] = None,
                 n_steps: int = 100,
                 hawkes_params: Optional[Dict] = None,
                 llm_config: Optional[Dict] = None,
                 random_seed: int = 42):
        self.n_agents = n_agents
        self.agent_type_ratio = agent_type_ratio or {
            "PUBLIC": 0.7, "OPINION_LEADER": 0.15,
            "MEDIA": 0.1, "OFFICIAL": 0.05,
        }
        self.network_type = network_type
        self.network_params = network_params or {}
        self.n_steps = n_steps
        self.hawkes_params = hawkes_params or {}
        self.llm_config = llm_config or {}
        self.random_seed = random_seed

    def to_dict(self) -> Dict:
        return {"n_agents": self.n_agents,
                "agent_type_ratio": self.agent_type_ratio,
                "network_type": self.network_type,
                "network_params": self.network_params,
                "n_steps": self.n_steps,
                "hawkes_params": self.hawkes_params,
                "llm_config": self.llm_config,
                "random_seed": self.random_seed}

    @classmethod
    def from_dict(cls, d: Dict) -> "SimConfig":
        return cls(
            n_agents=int(d.get("n_agents", 50)),
            agent_type_ratio=d.get("agent_type_ratio"),
            network_type=str(d.get("network_type", "small_world")),
            network_params=d.get("network_params"),
            n_steps=int(d.get("n_steps", 100)),
            hawkes_params=d.get("hawkes_params"),
            llm_config=d.get("llm_config"),
            random_seed=int(d.get("random_seed", 42)),
        )


class DimensionResult:
    """单维度评估结果"""
    def __init__(self, metrics: Optional[Dict[str, float]] = None):
        self.metrics = metrics or {}

    def to_dict(self) -> Dict:
        return {"metrics": self.metrics}


class EvaluationResult:
    """完整评估结果 — D (StrategyEvaluator) 输出"""
    def __init__(self, behavior: Optional[DimensionResult] = None,
                 content: Optional[DimensionResult] = None,
                 topology: Optional[DimensionResult] = None,
                 delta_vs_baseline: Optional[Dict[str, float]] = None):
        self.behavior = behavior or DimensionResult()
        self.content = content or DimensionResult()
        self.topology = topology or DimensionResult()
        self.delta_vs_baseline = delta_vs_baseline or {}

    def to_dict(self) -> Dict:
        return {"behavior": self.behavior.to_dict(),
                "content": self.content.to_dict(),
                "topology": self.topology.to_dict(),
                "delta_vs_baseline": self.delta_vs_baseline}


# LLMClient 协议: 所有 LLM 客户端统一暴露 chat(prompt: str) -> str
# 由 setup_llm_client 工厂函数创建
