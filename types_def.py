"""
types_def.py — 全项目共享数据类型
------------------------------------
依据《全函数接口表 20260718_v2》Sheet「类型与约定说明」+《场景设定_20260718_v2》定义。
所有模块（A/B/C/D/E）均从此处 import，保证类型统一。

【v3 变更（场景对齐版，A 模块牵头冻结）】
  ① 干预触发：新增 negative_threshold / heat_threshold 双阈值语义，
     默认触发条件改为《场景设定》原文的「负面程度超过阈值 → controller 干预」。
  ② 事件发起：新增 EventRecord，事件源由独立模块 event_source.py 投放。
  ③ 跨群条件：ActionRecord / SocialInfo 新增 forward_count（消息谱系被转发次数），
     跨群通道需 forward_count ≥ 阈值才打开。
  ④ 小群互转：新增 GROUP_BAND（小群 {DORM,CLASS} / 大群 {MAJOR,CAMPUS}）与
     ForwardDirection，支持 upward / lateral(小群互转) / downward 三种方向。
  ⑤ 禁言：ActionType 新增 MUTE=4 / ANNOUNCE=5，对应 Controller 的「禁言 / 公告」。
  ⑥ 曝光：新增 GROUP_EXPOSURE，取自《场景设定》§1 群类型表的「曝光」列。
  ⑦ 降温公式：新增 GROUP_CONTROL_LEVEL，把《场景设定》§3
     「实际降温 = 基础降温 × control_level(>1)」与 §4.2 指数公式统一，
     二者代数等价：exp(-α·control_level_k) ≡ exp(-α)·exp(-β_k)，β_k = α·(control_level_k − 1)。
  ⑧ 人格细项：PsychologyBelief 新增 trust / confirmation_bias，
     不再把 trust/bias 混在「人格 + stance」里。

【v2 变更（相对旧版 20260705）】
  - AgentType: PUBLIC/OPINION_LEADER/MEDIA/OFFICIAL → ORDINARY/ACTIVE/RATIONAL/CONTROLLER
  - ActionType: POST/COMMENT/REPOST/LIKE/SILENT → SEND_MESSAGE/REPLY/FORWARD/SILENT（移除LIKE）
  - 新增 GroupType: DORM/CLASS/MAJOR/CAMPUS
  - 新增 MessageType 字符串常量类
  - 全项目 event_id → topic_id

数值约定：
  opinion_value / valence / stance_prior : float ∈ [-1.0, +1.0]
  arousal / confidence / 人格 5 维 / risk_aversion / trust / confirmation_bias : float ∈ [0.0, 1.0]
  heat : float ∈ [0.0, HEAT_CAP]
  negative_score / distortion_level / exposure : float ∈ [0.0, 1.0]
  tick / agent_id / forward_count : int ≥ 0
  topic_id : str，形如 "T001"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 枚举类型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentType(IntEnum):
    """智能体类型（《场景设定》§2）。"""
    ORDINARY    = 0   # 普通群员（多数沉默，偶尔附和/提问）
    ACTIVE      = 1   # 活跃讨论者（乐于转发、分享信息）
    RATIONAL    = 2   # 理性讨论者（质疑信息真实性、查证来源、分析逻辑）
    CONTROLLER  = 3   # 管理者（提醒、澄清、禁言、公告）


class GroupType(IntEnum):
    """
    群类型（《场景设定》§1）。
    与 GROUP_CONTROL_LEVEL / GROUP_BETA / GROUP_EXPOSURE 三张表联动。
    DORM 群 t_dorm^int = +∞，不触发干预。
    """
    DORM   = 0   # 宿舍群（管理很弱，曝光低）
    CLASS  = 1   # 班级群（管理中弱，曝光中）
    MAJOR  = 2   # 专业群（管理中，  曝光中高）
    CAMPUS = 3   # 校园群（管理强，  曝光高，干预效果最强）


class ActionType(IntEnum):
    """
    行动类型。

    【v3】新增 MUTE / ANNOUNCE，落实《场景设定》§2 Controller 的
    「提醒、澄清、禁言、公告」四种手段：
        - 提醒 / 澄清 → SEND_MESSAGE + MessageType.CLARIFICATION
        - 禁言       → MUTE      （对某 agent 在某群内禁言 N tick）
        - 公告       → ANNOUNCE  （全群公告，强制降负面 / 降失真）
    """
    SEND_MESSAGE = 0   # 发新消息
    REPLY        = 1   # 回复他人消息
    FORWARD      = 2   # 转发消息（群内转发亦计入 forward_count）
    SILENT       = 3   # 沉默（不提交 ActionRecord，不写 memory）
    MUTE         = 4   # v3 新增：禁言（仅 CONTROLLER，且仅在 control_level>1 的群）
    ANNOUNCE     = 5   # v3 新增：公告（仅 CONTROLLER）


#: 只有 CONTROLLER 可以发起的治理动作
CONTROLLER_ONLY_ACTIONS = frozenset({ActionType.MUTE, ActionType.ANNOUNCE})


class MessageType:
    """消息内容语义类型（《场景设定》§3，共 5 类，禁止扩展）。"""
    ORIGINAL      = "original"       # 原始信息
    FORWARD       = "forward"        # 直接转发
    PARAPHRASE    = "paraphrase"     # 转述（distortion_level 中等）
    EXAGGERATE    = "exaggerate"     # 夸大（distortion_level 高）
    CLARIFICATION = "clarification"  # 澄清 / 提醒 / 公告（降低 negative_score）


class ForwardDirection:
    """
    v3 新增：转发方向语义（对应流程图的三条通道）。

    UPWARD   小群 → 更大群（DORM→MAJOR、CLASS→CAMPUS、MAJOR→CAMPUS…）主通道
    LATERAL  小群互转 —— 即流程图中唯一那条「DORM–CLASS 横向」箭头
    DOWNWARD 大群 → 更小群（最弱通道）
    INTRA    群内转发（不跨群，但累加 forward_count，是打开跨群通道的前置）
    """
    UPWARD   = "upward"
    LATERAL  = "lateral"
    DOWNWARD = "downward"
    INTRA    = "intra"


class InterventionType(IntEnum):
    """干预类型（D 模块使用）。"""
    EVENT_INJECTION = 0   # 事件注入
    NODE_CONTROL    = 1   # 节点控制
    PLATFORM_PARAM  = 2   # 平台参数调整


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 热度演化模型常量（《场景设定》§4.2）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALPHA: float = 0.15   # 自然衰减率 α，所有群共用（《场景设定》式(1)）

THETA: float = 0.7    # 热度阈值 θ：H(t) ≥ θ 视为「热度高」

#: 负面程度阈值（《场景设定》§3「负面程度超过阈值 → controller 开始干预」）
NEGATIVE_THETA: float = 0.65

HEAT_CAP: float = 3.0  # 热度上限，避免消息量线性堆叠导致 H 失去「阈值」语义

# ── 群类型分级管理强度 control_level（《场景设定》§3）─────────────────────── #
#
#   《场景设定》§3 用自然语言表述：实际降温 = 基础降温 × control_level (>1)
#   《场景设定》§4.2 用指数形式表述：H_k(t+1) = H_k(t)·e^(-α)·e^(-β_k·𝟙[t≥t_k^int])
#
#   两者代数等价，本项目以 control_level 为**唯一可调参数**，β_k 由它派生：
#
#       e^(-α · control_level_k) ≡ e^(-α) · e^(-β_k)
#       ⟺ α · control_level_k = α + β_k
#       ⟺ β_k = α · (control_level_k − 1)
#
#   于是「基础降温 α」乘上「管理强度 control_level_k(>1)」就是实际降温指数，
#   与 §4.2 的公式逐位相同，不再是两套语义。
#
GROUP_CONTROL_LEVEL: Dict[GroupType, float] = {
    GroupType.DORM:   1.0 + 0.05 / ALPHA,   # ≈1.333  私密性强，Controller 几乎无法介入
    GroupType.CLASS:  1.0 + 0.12 / ALPHA,   # =1.800  班委/班主任可干预，覆盖有限
    GroupType.MAJOR:  1.0 + 0.20 / ALPHA,   # ≈2.333  辅导员/专业负责人权威较高
    GroupType.CAMPUS: 1.0 + 0.30 / ALPHA,   # =3.000  覆盖全校，管理层级最高
}

#: 干预衰减率 β_k —— 由 control_level 派生，保持与 §4.2 表格一致（0.05/0.12/0.20/0.30）
GROUP_BETA: Dict[GroupType, float] = {
    group_type: ALPHA * (control_level - 1.0)
    for group_type, control_level in GROUP_CONTROL_LEVEL.items()
}

# ── 群类型曝光度（《场景设定》§1 群类型表「曝光」列）──────────────────────── #
#
#   低 / 中 / 中高 / 高 → 归一化到 [0,1]。曝光影响三件事（A 模块 opinion_model 使用）：
#     1. 消息进入该群后的热度增益（高曝光 → 同一条消息掀起更大热度）
#     2. 宏观热度跨群扩散的目标选择概率（高曝光群更容易被外部舆情渗透）
#     3. 群成员单 tick 能感知到的消息条数上限（高曝光 → 信息流更长）
#
GROUP_EXPOSURE: Dict[GroupType, float] = {
    GroupType.DORM:   0.20,   # 低
    GroupType.CLASS:  0.50,   # 中
    GroupType.MAJOR:  0.75,   # 中高
    GroupType.CAMPUS: 1.00,   # 高
}

# ── 群带（band）：用于区分「小群互转（横向）」与「小→大（向上）」────────── #
GROUP_BAND: Dict[GroupType, str] = {
    GroupType.DORM:   "small",
    GroupType.CLASS:  "small",
    GroupType.MAJOR:  "large",
    GroupType.CAMPUS: "large",
}

SMALL_GROUPS = tuple(g for g, band in GROUP_BAND.items() if band == "small")
LARGE_GROUPS = tuple(g for g, band in GROUP_BAND.items() if band == "large")

#: 流程图中唯一被画成「横向」的一条边：宿舍群 ↔ 班级群。
#: 其余群对一律按规模判 upward / downward —— 专业群→校园群仍然是「小→大」，
#: 不能因为都属于「大群」这一档就被误判成横向。
LATERAL_PAIRS = frozenset({
    (GroupType.DORM, GroupType.CLASS),
    (GroupType.CLASS, GroupType.DORM),
})

# 启动时断言：管理强度 / β / 曝光 三条约束
assert all(cl > 1.0 for cl in GROUP_CONTROL_LEVEL.values()), \
    "control_level 必须 > 1（《场景设定》§3）"
assert (GROUP_CONTROL_LEVEL[GroupType.DORM] < GROUP_CONTROL_LEVEL[GroupType.CLASS]
        < GROUP_CONTROL_LEVEL[GroupType.MAJOR] < GROUP_CONTROL_LEVEL[GroupType.CAMPUS]), \
    "control_level 须满足 DORM < CLASS < MAJOR < CAMPUS"
assert (GROUP_BETA[GroupType.DORM] < GROUP_BETA[GroupType.CLASS]
        < GROUP_BETA[GroupType.MAJOR] < GROUP_BETA[GroupType.CAMPUS]), \
    "GROUP_BETA 须满足 β_DORM < β_CLASS < β_MAJOR < β_CAMPUS"
assert (GROUP_EXPOSURE[GroupType.DORM] < GROUP_EXPOSURE[GroupType.CLASS]
        < GROUP_EXPOSURE[GroupType.MAJOR] < GROUP_EXPOSURE[GroupType.CAMPUS]), \
    "GROUP_EXPOSURE 须满足 曝光 DORM < CLASS < MAJOR < CAMPUS"


def heat_decay(
    current_heat: float,
    group_type: GroupType,
    intervened: bool,
) -> float:
    """
    《场景设定》§3 + §4.2 的统一降温实现（全项目唯一权威公式）。

        未干预：H(t+1) = H(t) · exp(−α)                      「基础降温」
        已干预：H(t+1) = H(t) · exp(−α · control_level_k)     「基础降温 × control_level」
                        ≡ H(t) · exp(−α) · exp(−β_k)          （与 §4.2 式(2) 完全等价）

    Parameters
    ----------
    current_heat : H_k(t)
    group_type   : 群类型 k
    intervened   : 𝟙[t ≥ t_k^int]，即该群是否已触发 Controller 干预

    Returns
    -------
    float ≥ 0.0
    """
    import math
    control_level = GROUP_CONTROL_LEVEL[group_type] if intervened else 1.0
    return max(0.0, current_heat * math.exp(-ALPHA * control_level))


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
    身份信念。

    ``group_type`` / ``primary_group_id`` 保留「主群」语义以兼容旧调用；
    ``group_ids`` 才是 Agent 的真实群成员关系。一个 Agent 可以同时属于
    DORM / CLASS / MAJOR / CAMPUS 多个层级群。
    """
    agent_type:   AgentType  = AgentType.ORDINARY
    group_type:   GroupType  = GroupType.CLASS    # 兼容字段：主群类型
    primary_group_id: str    = "GROUP_CLASS"
    group_ids:    List[str]  = field(default_factory=lambda: ["GROUP_CLASS"])
    nickname:     str        = ""                 # 群内昵称
    role_desc:    str        = ""
    stance_prior: float      = 0.0               # [-1, 1]


@dataclass
class PsychologyBelief:
    """
    心理信念：人格 + 风险规避 + 【v3 新增】信任度 / 确认偏误。

    trust             对「他人发布的信息」的基础信任度。低 trust 的 agent
                      更容易折价高 distortion 消息，也更不愿意转发。
    confirmation_bias 确认偏误强度。高 bias 的 agent 只吸收与自身立场同向的
                      信息，对反向信息几乎不更新观点（放大极化）。
    """
    personality:       Personality = field(default_factory=Personality)
    risk_aversion:     float       = 0.5    # [0, 1]
    trust:             float       = 0.5    # [0, 1]  v3 新增
    confirmation_bias: float       = 0.5    # [0, 1]  v3 新增


@dataclass
class OpinionBelief:
    """对某话题的观点信念。"""
    topic_id:      str   = "T001"
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
    """社交信息单元。"""
    source_id:        int        = 0
    source_nickname:  str        = ""
    content:          str        = ""
    message_type:     str        = MessageType.ORIGINAL
    timestamp:        int        = 0                     # tick
    is_mention:       bool       = False
    topic_id:         str        = "T001"
    distortion_level: float      = 0.0                  # [0,1]
    original_content: str        = ""
    negative_score:   float      = 0.0                  # [0,1]
    heat:             float      = 0.0                  # [0,HEAT_CAP]
    message_id:       str        = ""                   # 环境分配的唯一消息 ID
    group_id:         str        = ""                   # 当前消息实际所在/投递到的群
    source_message_id: Optional[str] = None              # FORWARD 的源消息 ID
    source_group_id:  str        = ""                   # FORWARD 从哪个群取出；原创=group_id
    forward_count:    int        = 0                    # v3 新增：该消息谱系已被转发次数
    root_message_id:  str        = ""                   # v3 新增：谱系根消息 ID


@dataclass
class Perception:
    """智能体单步感知结果。"""
    group_id:        str                       = ""                   # 兼容字段：主群 ID
    group_ids:       List[str]                 = field(default_factory=list)  # 真实成员群
    group_type:      GroupType                 = GroupType.CLASS      # 兼容字段：主群类型
    beta:            float                     = 0.12                 # 主群 β_k
    control_level:   float                     = 1.8                  # v3 新增：主群管理强度
    exposure:        float                     = 0.5                  # v3 新增：主群曝光
    group_exposure:  Dict[str, float]          = field(default_factory=dict)  # v3 新增
    recent_messages: List[SocialInfo]          = field(default_factory=list)
    mentions:        List[SocialInfo]          = field(default_factory=list)
    tick:            int                       = 0
    topic_heat:      Dict[str, float]          = field(default_factory=dict)
    topic_negative:  Dict[str, float]          = field(default_factory=dict)
    topic_heat_by_group: Dict[str, Dict[str, float]] = field(default_factory=dict)
    topic_negative_by_group: Dict[str, Dict[str, float]] = field(default_factory=dict)
    intervened_groups: List[str]               = field(default_factory=list)  # v3 新增
    muted_groups:      List[str]               = field(default_factory=list)  # v3 新增：本人被禁言的群


@dataclass
class MemoryRecord:
    """短期记忆条目。"""
    tick:      int        = 0
    info:      SocialInfo = field(default_factory=SocialInfo)
    relevance: float      = 0.5    # [0, 1]


@dataclass
class ActionRecord:
    """行动记录。"""
    agent_id:         int             = 0
    action_type:      ActionType      = ActionType.SEND_MESSAGE
    content:          str             = ""
    target_id:        Optional[int]   = None
    tick:             int             = 0
    topic_id:         str             = "T001"
    distortion_level: float           = 0.0              # [0,1]
    message_type:     str             = MessageType.ORIGINAL
    negative_score:   float           = 0.0              # [0,1]
    heat:             float           = 0.0              # 本次消息热度贡献
    message_id:       str             = ""               # 环境写入时分配
    group_id:         str             = ""               # 消息真正投递到的目标群
    source_message_id: Optional[str]  = None              # FORWARD 的源消息
    source_group_id:  str             = ""               # FORWARD 来源群；原创=group_id
    forward_count:    int             = 0                # v3 新增：谱系被转发次数
    root_message_id:  str             = ""               # v3 新增：谱系根消息 ID
    forward_direction: str            = ""               # v3 新增：ForwardDirection
    mute_duration:    int             = 0                # v3 新增：MUTE 动作的禁言时长（tick）


@dataclass
class EventRecord:
    """
    v3 新增：事件源投放记录（event_source.py 产出，D/E 模块可直接读取）。
    对应流程图的「事件源随机发到小群 / 大群」节点。
    """
    event_seq:      int   = 0
    tick:           int   = 0
    origin_agent_id: int  = 0
    group_id:       str   = ""
    group_type:     GroupType = GroupType.DORM
    scope:          str   = "small"       # small | large
    heat:           float = 0.0
    negative_score: float = 0.0
    topic_id:       str   = "T001"
    content:        str   = ""
    is_initial:     bool  = True
    message_id:     str   = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BDI 推理中间结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Desire:
    """
    欲望。
    goal_type: reply / discuss / share / clarify / intervene / mute / announce / silent
    """
    goal_type: str           = "discuss"
    priority:  float         = 0.5          # [0, 1]
    topic_id:  str           = "T001"
    target_id: Optional[int] = None         # reply/mute 欲望时指向目标 agent
    source_message_id: Optional[str] = None
    source_group_id: str = ""
    destination_group_id: str = ""


@dataclass
class Intention:
    """意图。"""
    action_type:  ActionType      = ActionType.SILENT
    content_plan: str             = ""
    topic_id:     str             = "T001"
    target_id:    Optional[int]   = None
    source_message_id: Optional[str] = None
    source_group_id: str = ""
    destination_group_id: str = ""
    message_type: str = ""              # LLM 可显式指定；空串表示由规则推导
    mute_duration: int = 0              # v3 新增


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 配置 / 评估结果
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#: 场景参数的全项目默认值。A 模块 OpinionModel 与 E 模块 load_config 共用此表，
#: 保证「YAML 未写的键」在任何入口下都取到同一个默认值。
DEFAULT_SCENARIO_PARAMS: Dict[str, Any] = {
    # ── ① 干预触发（《场景设定》§3「负面程度超过阈值 → controller 开始干预」）──
    "intervention_trigger": "negative",   # negative | heat | heat_or_negative | heat_and_negative
    "negative_threshold": NEGATIVE_THETA,
    "heat_threshold": THETA,
    "intervention_requires_controller": True,
    "intervention_min_messages": 4,        # 群内至少要有这么多条存活消息才可能触发
                                           # （避免开局 1 条消息就把整群判定为「负面超阈」）

    # ── ② 事件发起（流程图「事件源随机发到小群/大群」）────────────────────
    "inject_initial_event": True,
    "initial_event_group_scope": "small_or_large",  # small | large | small_or_large | any
    "initial_event_heat": 0.80,
    "initial_event_negative": 0.72,
    "initial_event_content": "校园事件相关消息开始在群聊中传播",
    "secondary_event_probability": 0.05,   # 每 tick 追加事件的概率（0 = 只投一次）
    "secondary_event_max": 2,
    "secondary_event_min_interval": 8,

    # ── ③ 跨群条件（流程图「转发次数 ≥ 阈 → 进入跨群通道」）────────────────
    "cross_group_forward_threshold": 2,
    "cross_group_heat_bypass": True,        # 群热度 ≥ θ 时允许绕过转发次数门槛
    "macro_spread_requires_threshold": True,
    # 门槛按方向分级：往大群捅需要「这事已经传开了」，小群互转是熟人圈层的
    # 顺手一转，最容易；大群往小群倒灌最难。三条通道于是有了强弱之分，
    # 而不是被同一个阈值一刀切成「只有小→大」。
    #   upward   : 往更大的群捅，需要「这事已经被转过至少一次」；
    #   lateral  : DORM↔CLASS 小群互转。流程图里这条横向箭头直接连在两个小群
    #              之间，不挂在「转发次数 ≥ 阈」分支下 —— 熟人圈层顺手一转，
    #              门槛为 0；
    #   downward : 大群往小群倒灌最难。
    "cross_group_forward_threshold_by_direction": {
        "upward":   1,
        "lateral":  0,
        "downward": 3,
    },

    # ── ④ 转发方向权重（流程图 DORM–CLASS 横向 = lateral）──────────────────
    "allow_downward_forward": True,
    "forward_direction_weights": {
        "upward":   1.00,   # 小群 → 更大群（主通道）
        "lateral":  0.55,   # 小群互转 / 大群互转（横向，弱于主通道但真实存在）
        "downward": 0.15,   # 大群 → 小群（最弱）
    },
    # 选择转发目标时曝光的影响力度。取 1.0 会让宿舍群被校园群彻底压死，
    # 「横向」通道有名无实；0.5 保留高曝光群的吸引力又不至于赢者通吃。
    "forward_exposure_weight": 0.50,
    "intra_group_forward_weight": 0.60,     # 群内转发（累加 forward_count 的前置）

    # ── ⑤ 禁言 / 公告（《场景设定》§2 Controller：提醒、澄清、禁言、公告）──
    "enable_mute": True,
    "mute_duration_ticks": 3,
    "mute_negative_trigger": 0.68,          # 群负面 ≥ 该值时 Controller 优先禁言
    "mute_max_per_tick": 2,
    "governance_cooldown_ticks": 3,   # 同一 Controller 对同一群的治理动作最小间隔
    "announce_negative_relief": 0.18,       # 公告对群负面的即时压降比例
    "announce_distortion_relief": 0.30,

    # ── ⑥ 曝光（《场景设定》§1 群类型表「曝光」列）─────────────────────────
    "exposure_heat_weight": 0.60,           # 热度增益受曝光影响的强度
    "exposure_spread_weight": 1.00,         # 宏观扩散目标选择受曝光影响的强度
    "exposure_perception_weight": 0.60,     # 可感知消息条数受曝光影响的强度

    # ── 传播失真 / 干预后恢复（《场景设定》§3）────────────────────────────
    "forward_distortion_probability": 0.70,
    "forward_distortion_increment_min": 0.03,
    "forward_distortion_increment_max": 0.15,
    "intervention_negative_decay": 0.75,
    "intervention_distortion_decay": 0.80,

    # ── 热度标度（保证 θ 有阈值语义，H 不会被消息量线性顶穿）───────────────
    "base_heat_gain": 0.038,
    "heat_cap": HEAT_CAP,
    "message_expire_ticks": 10,
    "negative_smoothing": 0.50,
}


@dataclass
class SimConfig:
    """仿真全局配置（由 load_config 从 YAML 读取）。"""
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
    scenario_params:   Dict[str, Any]   = field(
        default_factory=lambda: dict(DEFAULT_SCENARIO_PARAMS)
    )


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
# LLMClient 协议（C 模块实现，此处仅类型注释用）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LLMClient:
    """约定暴露统一方法 chat(prompt: str) -> str（C 模块负责实现）。"""

    def chat(self, prompt: str) -> str:     # pragma: no cover
        raise NotImplementedError
