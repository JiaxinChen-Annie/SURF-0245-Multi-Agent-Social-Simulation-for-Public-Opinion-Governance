"""
types_def.py — 全项目共享数据类型
------------------------------------
依据《全函数接口表 20260718_v2》Sheet「类型与约定说明」定义。
所有模块（A/B/C/D/E）均从此处 import，保证类型统一。

【v2 主要变更（相对旧版 20260705）】
  - AgentType: PUBLIC/OPINION_LEADER/MEDIA/OFFICIAL → ORDINARY/ACTIVE/RATIONAL/CONTROLLER
  - ActionType: POST/COMMENT/REPOST/LIKE/SILENT → SEND_MESSAGE/REPLY/FORWARD/SILENT（移除LIKE）
  - 新增 GroupType: DORM/CLASS/MAJOR/CAMPUS
  - 新增 MessageType 字符串常量类
  - SocialInfo 新增 source_nickname/message_type/is_mention/topic_id/
              distortion_level/original_content/negative_score/heat
  - Perception 新增 group_id/group_type/beta/topic_heat/topic_negative
              neighbor_actions → recent_messages 重命名
  - ActionRecord 新增 distortion_level/message_type/negative_score/heat
  - 全项目 event_id → topic_id
  - IdentityBelief 新增 group_type/nickname
  - OpinionBelief: event_id → topic_id
  - Desire: event_id→topic_id，新增 target_id
  - Intention: event_id→topic_id
  - SimConfig.agent_type_ratio 键对应新 AgentType 名称

数值约定：
  opinion_value / valence / stance_prior : float ∈ [-1.0, +1.0]
  arousal / confidence / 人格 5 维 / risk_aversion : float ∈ [0.0, 1.0]
  heat : float ∈ [0.0, +∞)
  negative_score / distortion_level : float ∈ [0.0, 1.0]
  tick / agent_id : int ≥ 0
  topic_id : str，形如 "T001"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 枚举类型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentType(IntEnum):
    """
    智能体类型（v2 已更新）。
    旧: PUBLIC/OPINION_LEADER/MEDIA/OFFICIAL
    新: ORDINARY/ACTIVE/RATIONAL/CONTROLLER
    """
    ORDINARY    = 0   # 普通群员（多数沉默，偶尔附和/提问）
    ACTIVE      = 1   # 活跃讨论者（乐于转发、分享信息）
    RATIONAL    = 2   # 理性讨论者（质疑信息真实性、查证来源、分析逻辑）
    CONTROLLER  = 3   # 管理者（提醒、澄清、禁言、公告）


class GroupType(IntEnum):
    """
    群类型（v2 新增）。
    与 GROUP_BETA 干预衰减系数联动。
    DORM 群 t_dorm^int = +∞，不触发干预。
    """
    DORM   = 0   # 宿舍群（管理强度很弱，低曝光）
    CLASS  = 1   # 班级群（管理强度中弱）
    MAJOR  = 2   # 专业群（管理强度中）
    CAMPUS = 3   # 校园群（管理强度强，高曝光，干预效果最强）


class ActionType(IntEnum):
    """
    行动类型（v2 已更新，移除 LIKE）。
    旧: POST/COMMENT/REPOST/LIKE/SILENT
    新: SEND_MESSAGE/REPLY/FORWARD/SILENT
    """
    SEND_MESSAGE = 0   # 发新消息
    REPLY        = 1   # 回复他人消息
    FORWARD      = 2   # 转发消息
    SILENT       = 3   # 沉默（不提交 ActionRecord，不写 memory）


class MessageType:
    """
    消息内容语义类型（v2 新增，替代旧 ActionType 中的内容语义）。
    使用字符串常量而非 IntEnum，方便序列化。
    """
    ORIGINAL      = "original"       # 原始信息
    FORWARD       = "forward"        # 直接转发
    PARAPHRASE    = "paraphrase"     # 转述（distortion_level 中等）
    EXAGGERATE    = "exaggerate"     # 夸大（distortion_level 高）
    CLARIFICATION = "clarification"  # 澄清（降低 negative_score）


class InterventionType(IntEnum):
    """干预类型（D 模块使用）。"""
    EVENT_INJECTION = 0   # 事件注入
    NODE_CONTROL    = 1   # 节点控制
    PLATFORM_PARAM  = 2   # 平台参数调整


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B 模块热度衰减模型参数常量（SocialAgent 类变量，此处同步定义供 A 模块引用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALPHA: float = 0.15   # 自然衰减率 α，所有群共用

THETA: float = 0.7    # 干预触发阈值 θ：H(t) ≥ θ 时 Controller 触发干预，同时满足跨群扩散条件

# 群类型分级干预系数 βₖ，约束 β₁ < β₂ < β₃ < β₄
GROUP_BETA: Dict[GroupType, float] = {
    GroupType.DORM:   0.05,   # β₁ 私密性强，Controller 几乎无法介入
    GroupType.CLASS:  0.12,   # β₂ 班委/班主任可干预，覆盖范围有限
    GroupType.MAJOR:  0.20,   # β₃ 辅导员/专业负责人权威较高
    GroupType.CAMPUS: 0.30,   # β₄ 覆盖全校，管理层级最高
}

# 启动时断言 β 单调递增
assert (GROUP_BETA[GroupType.DORM] < GROUP_BETA[GroupType.CLASS]
        < GROUP_BETA[GroupType.MAJOR] < GROUP_BETA[GroupType.CAMPUS]), \
    "GROUP_BETA 须满足 β_DORM < β_CLASS < β_MAJOR < β_CAMPUS"


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
    """
    身份信念（v2 新增 group_type / nickname）。
    """
    agent_type:   AgentType  = AgentType.ORDINARY
    group_type:   GroupType  = GroupType.CLASS    # v2 新增
    nickname:     str        = ""                 # v2 新增：群内昵称
    role_desc:    str        = ""
    stance_prior: float      = 0.0               # [-1, 1]


@dataclass
class PsychologyBelief:
    """心理信念：人格 + 风险规避系数。"""
    personality:   Personality = field(default_factory=Personality)
    risk_aversion: float       = 0.5    # [0, 1]


@dataclass
class OpinionBelief:
    """
    对某话题的观点信念（v2：event_id → topic_id）。
    """
    topic_id:      str   = "T001"   # 原 event_id，形如 "T001"
    opinion_value: float = 0.0      # [-1, 1]
    confidence:    float = 0.5      # [0, 1]


@dataclass
class BeliefSystem:
    """四层信念系统：身份 / 心理 / 话题观点 / 情绪。"""
    identity:   IdentityBelief   = field(default_factory=IdentityBelief)
    psychology: PsychologyBelief = field(default_factory=PsychologyBelief)
    opinions:   Dict[str, OpinionBelief] = field(default_factory=dict)  # 键为 topic_id
    emotion:    EmotionState     = field(default_factory=EmotionState)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 感知 / 记忆 / 行动
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SocialInfo:
    """
    社交信息单元（v2 全面更新）。
    原结构缺少 message_type/distortion_level 等关键字段，已补全。
    """
    source_id:        int        = 0
    source_nickname:  str        = ""                    # v2 新增
    content:          str        = ""
    message_type:     str        = MessageType.ORIGINAL  # v2 新增，替代旧 action_type 的内容语义
    timestamp:        int        = 0                     # tick
    is_mention:       bool       = False                 # v2 新增
    topic_id:         str        = "T001"               # v2 新增（原 event_id）
    distortion_level: float      = 0.0                  # v2 新增，[0,1]
    original_content: str        = ""                   # v2 新增：原始消息内容
    negative_score:   float      = 0.0                  # v2 新增，[0,1]
    heat:             float      = 0.0                  # v2 新增，[0,+∞)


@dataclass
class Perception:
    """
    智能体单步感知结果（v2 全面更新）。
    neighbor_actions → recent_messages 重命名；新增 group_id/group_type/beta/topic_heat/topic_negative。
    """
    group_id:        str                       = ""                   # v2 新增
    group_type:      GroupType                 = GroupType.CLASS      # v2 新增
    beta:            float                     = 0.12                 # v2 新增：GROUP_BETA[group_type]
    recent_messages: List[SocialInfo]          = field(default_factory=list)  # 原 neighbor_actions
    mentions:        List[SocialInfo]          = field(default_factory=list)
    tick:            int                       = 0
    topic_heat:      Dict[str, float]          = field(default_factory=dict)  # v2 新增
    topic_negative:  Dict[str, float]          = field(default_factory=dict)  # v2 新增


@dataclass
class MemoryRecord:
    """短期记忆条目。"""
    tick:      int        = 0
    info:      SocialInfo = field(default_factory=SocialInfo)
    relevance: float      = 0.5    # [0, 1]


@dataclass
class ActionRecord:
    """
    行动记录（v2 全面更新：新增 4 字段；event_id → topic_id）。
    """
    agent_id:         int             = 0
    action_type:      ActionType      = ActionType.SEND_MESSAGE
    content:          str             = ""
    target_id:        Optional[int]   = None
    tick:             int             = 0
    topic_id:         str             = "T001"           # 原 event_id
    distortion_level: float           = 0.0              # v2 新增，[0,1]
    message_type:     str             = MessageType.ORIGINAL  # v2 新增
    negative_score:   float           = 0.0              # v2 新增，[0,1]
    heat:             float           = 0.0              # v2 新增：本次消息热度贡献


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BDI 推理中间结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Desire:
    """
    欲望（v2 更新：goal_type 枚举变更；event_id→topic_id；新增 target_id）。
    goal_type: reply / discuss / share / clarify / intervene / silent
    """
    goal_type: str           = "discuss"
    priority:  float         = 0.5          # [0, 1]
    topic_id:  str           = "T001"       # 原 event_id
    target_id: Optional[int] = None         # v2 新增：reply 欲望时指向目标 agent


@dataclass
class Intention:
    """
    意图（v2 更新：event_id→topic_id）。
    """
    action_type:  ActionType      = ActionType.SILENT
    content_plan: str             = ""
    topic_id:     str             = "T001"   # 原 event_id
    target_id:    Optional[int]   = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 配置 / 评估结果
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SimConfig:
    """
    仿真全局配置（由 load_config 从 YAML 读取）。
    agent_type_ratio 键对应新 AgentType 名称：ORDINARY/ACTIVE/RATIONAL/CONTROLLER。
    新增 group_type_ratio 配置各群比例。
    """
    n_agents:          int              = 100
    agent_type_ratio:  Dict[str, float] = field(default_factory=lambda: {
        "ORDINARY":    0.70,
        "ACTIVE":      0.10,
        "RATIONAL":    0.10,
        "CONTROLLER":  0.10,
    })
    group_type_ratio:  Dict[str, float] = field(default_factory=lambda: {
        "DORM":   0.25,
        "CLASS":  0.35,
        "MAJOR":  0.25,
        "CAMPUS": 0.15,
    })
    network_type:      str              = "barabasi_albert"
    network_params:    Dict[str, Any]   = field(default_factory=lambda: {"m": 3})
    n_steps:           int              = 50
    hawkes_params:     Dict[str, Any]   = field(default_factory=lambda: {
        "mu": 0.1, "alpha": 0.5, "beta": 1.0
    })
    llm_config:        Dict[str, Any]   = field(default_factory=dict)
    random_seed:       int              = 42


@dataclass
class DimensionResult:
    """三维评估中的单维结果（D 模块使用）。"""
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """三维评估完整结果（D 模块使用）。"""
    behavior:          DimensionResult = field(default_factory=DimensionResult)
    content:           DimensionResult = field(default_factory=DimensionResult)
    topology:          DimensionResult = field(default_factory=DimensionResult)
    delta_vs_baseline: Dict[str, float] = field(default_factory=dict)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLMClient 协议（C/E 模块实现，此处仅类型注释用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LLMClient:
    """约定暴露统一方法 chat(prompt: str) -> str（C/E 模块负责实现）。"""

    def chat(self, prompt: str) -> str:     # pragma: no cover
        raise NotImplementedError
