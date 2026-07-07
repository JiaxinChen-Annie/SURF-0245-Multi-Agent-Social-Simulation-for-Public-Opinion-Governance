"""
types_def.py — 全项目共享数据类型
------------------------------------
依据《全函数接口表》Sheet「类型与约定说明」定义。
所有模块（A/B/C/D/E）均从此处 import，保证类型统一。

数值约定：
  opinion_value / valence / stance_prior : float ∈ [-1.0, +1.0]
  arousal / confidence / 人格 / risk_aversion : float ∈ [0.0, 1.0]
  tick / agent_id : int ≥ 0
  event_id : str，形如 "E001"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 枚举类型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentType(IntEnum):
    """智能体类型。"""
    PUBLIC          = 0   # 普通网民
    OPINION_LEADER  = 1   # 意见领袖
    MEDIA           = 2   # 媒体
    OFFICIAL        = 3   # 官方


class ActionType(IntEnum):
    """行动类型。"""
    POST    = 0   # 发帖
    COMMENT = 1   # 评论
    REPOST  = 2   # 转发
    LIKE    = 3   # 点赞
    SILENT  = 4   # 沉默


class InterventionType(IntEnum):
    """干预类型（D 模块使用）。"""
    EVENT_INJECTION = 0   # 事件注入
    NODE_CONTROL    = 1   # 节点控制
    PLATFORM_PARAM  = 2   # 平台参数调整


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 信念系统
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Personality:
    """五因素人格模型，每维 [0, 1]。"""
    openness:          float = 0.5
    conscientiousness: float = 0.5
    extraversion:      float = 0.5
    agreeableness:     float = 0.5
    neuroticism:       float = 0.5


@dataclass
class EmotionState:
    """情绪状态：效价 [-1, 1]，唤醒度 [0, 1]。"""
    valence: float = 0.0    # -1=负面, 0=中性, +1=正面
    arousal: float = 0.5    # 0=平静, 1=激动


@dataclass
class IdentityBelief:
    """身份信念。"""
    agent_type:   AgentType = AgentType.PUBLIC
    role_desc:    str = ""
    stance_prior: float = 0.0    # 初始立场倾向 [-1, 1]


@dataclass
class PsychologyBelief:
    """心理信念：人格 + 风险规避系数。"""
    personality:    Personality = field(default_factory=Personality)
    risk_aversion:  float = 0.5    # [0, 1]


@dataclass
class OpinionBelief:
    """对某事件的观点信念。"""
    event_id:       str   = "E001"
    opinion_value:  float = 0.0    # [-1, 1]
    confidence:     float = 0.5    # [0, 1]


@dataclass
class BeliefSystem:
    """四层信念系统：身份 / 心理 / 事件观点 / 情绪。"""
    identity:   IdentityBelief  = field(default_factory=IdentityBelief)
    psychology: PsychologyBelief = field(default_factory=PsychologyBelief)
    opinions:   Dict[str, OpinionBelief] = field(default_factory=dict)
    emotion:    EmotionState    = field(default_factory=EmotionState)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 感知 / 记忆 / 行动
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SocialInfo:
    """社交信息单元（邻居发出的一条消息）。"""
    source_id:    int        = 0
    action_type:  ActionType = ActionType.POST
    content:      str        = ""
    event_id:     str        = "E001"
    opinion_value: float     = 0.0    # [-1, 1]
    timestamp:    int        = 0      # tick


@dataclass
class Perception:
    """智能体单步感知结果。"""
    neighbor_actions:  List[SocialInfo]         = field(default_factory=list)
    trending_topics:   List[Tuple[str, float]]  = field(default_factory=list)
    mentions:          List[SocialInfo]         = field(default_factory=list)
    tick:              int                      = 0


@dataclass
class MemoryRecord:
    """短期记忆条目。"""
    tick:       int        = 0
    info:       SocialInfo = field(default_factory=SocialInfo)
    relevance:  float      = 0.5    # [0, 1]


@dataclass
class ActionRecord:
    """行动记录（写入环境信息流缓存）。"""
    agent_id:    int                = 0
    action_type: ActionType         = ActionType.POST
    content:     str                = ""
    event_id:    str                = "E001"
    opinion_value: float            = 0.0
    target_id:   Optional[int]      = None
    tick:        int                = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BDI 推理中间结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Desire:
    """欲望：目标类型 + 优先级。"""
    goal_type: str   = "get_info"   # seek_identity | vent_emotion | get_info | persuade
    priority:  float = 0.5          # [0, 1]
    event_id:  str   = "E001"


@dataclass
class Intention:
    """意图：具体行动计划。"""
    action_type:  ActionType      = ActionType.SILENT
    event_id:     str             = "E001"
    content_plan: str             = ""
    target_id:    Optional[int]   = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 配置 / 评估结果
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SimConfig:
    """仿真全局配置（由 load_config 从 YAML 读取）。"""
    n_agents:          int                   = 100
    agent_type_ratio:  Dict[str, float]      = field(default_factory=lambda: {
        "PUBLIC":          0.70,
        "OPINION_LEADER":  0.10,
        "MEDIA":           0.10,
        "OFFICIAL":        0.10,
    })
    network_type:      str                   = "barabasi_albert"
    network_params:    Dict[str, Any]        = field(default_factory=lambda: {"m": 3})
    n_steps:           int                   = 50
    hawkes_params:     Dict[str, Any]        = field(default_factory=lambda: {
        "mu": 0.1, "alpha": 0.5, "beta": 1.0
    })
    llm_config:        Dict[str, Any]        = field(default_factory=dict)
    random_seed:       int                   = 42


@dataclass
class DimensionResult:
    """三维评估中的单维结果（D 模块使用）。"""
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """三维评估完整结果（D 模块使用）。"""
    behavior:         DimensionResult = field(default_factory=DimensionResult)
    content:          DimensionResult = field(default_factory=DimensionResult)
    topology:         DimensionResult = field(default_factory=DimensionResult)
    delta_vs_baseline: Dict[str, float] = field(default_factory=dict)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLMClient 协议（C 模块实现，此处仅类型注释用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LLMClient:
    """约定暴露统一方法 chat(prompt: str) -> str（C 模块负责实现）。"""

    def chat(self, prompt: str) -> str:     # pragma: no cover
        raise NotImplementedError
