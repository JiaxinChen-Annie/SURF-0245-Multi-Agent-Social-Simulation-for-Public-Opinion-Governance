"""
social_agent.py
B 模块：智能体大脑（SocialAgent）- 微信舆论传播管理版
基于最终场景设定文档 v3.0

角色：ORDINARY / ACTIVE / RATIONAL / CONTROLLER
群类型：DORM / CLASS / MAJOR / CAMPUS
消息类型：original / forward / paraphrase / exaggerate / clarification

传播机制：指数衰减模型（热传导理论，Wei 2022）
         H_k(t+1) = H_k(t) · e^(-α) · e^(-β_k · 𝟙[t ≥ t_k^int])

【模块职责声明】
- A 模块（OpinionModel）：维护群状态（topic_heat / topic_negative / intervention_tick / cross_group_forward），
  执行 H(t) 热度演化（公式 1、2），处理跨群扩散，记录 intervention_tick
- B 模块（SocialAgent）：感知群状态 → BDI 推理 → 生成 ActionRecord；
  提供 calc_heat_decay() 工具函数供 A 模块调用（或本地测试）
- C 模块（Hawkes + LLM）：Hawkes 用于 Agent 激活采样，独立于 H(t)；
  LLM 工具链负责 Prompt 构建与解析

【DORM 群规则】
- 可以有 Controller 角色（agent_type=CONTROLLER）
- 但 Controller 不触发官方干预（t_dorm^int = +∞）
- 热度只走自然衰减：H(t+1) = H(t)·e^(-α)
- 原因：宿舍群为私密小群，几乎无正式管理

【v3.1 修复】
- 修复 ActionRecord.heat 字段未正确计算的问题（2026-07-18）
- heat 现在按接口表 #9 公式计算：heat = extraversion × (1 + topic_heat_now × 0.1)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from enum import IntEnum
from dataclasses import dataclass, field
import logging
import math

# ======================== 类型定义 ========================

# ---------- 角色枚举（4类） ----------
class AgentType(IntEnum):
    ORDINARY = 0      # 普通群员（多数沉默，偶尔附和/提问）
    ACTIVE = 1        # 活跃讨论者（乐于转发、分享信息）
    RATIONAL = 2      # 理性讨论者（质疑、查证、分析逻辑）
    CONTROLLER = 3    # 管理者（提醒、澄清、禁言、公告）


# ---------- 群类型枚举（4类） ----------
class GroupType(IntEnum):
    DORM = 0          # 宿舍群（管理强度很弱，低曝光；t_dorm^int = +∞，不触发干预）
    CLASS = 1         # 班级群（管理强度中弱）
    MAJOR = 2         # 专业群（管理强度中）
    CAMPUS = 3        # 校园群（管理强度强，高曝光）


# ---------- 群类型分级干预系数 βₖ ----------
# 参照 Wang et al. (2024)：不同规模与类型的群，Controller 管控能力存在显著差异
# 约束关系：β₁ < β₂ < β₃ < β₄
# 取值逻辑：干预衰减率 ≈ 管理强度 × 0.1（经验缩放因子，待真实数据校准）
GROUP_BETA = {
    GroupType.DORM: 0.05,      # β₁：私密性强，Controller 几乎无法介入
    GroupType.CLASS: 0.12,     # β₂：班委/班主任可干预，但覆盖范围有限
    GroupType.MAJOR: 0.20,     # β₃：辅导员/专业负责人权威较高
    GroupType.CAMPUS: 0.30,    # β₄：覆盖全校，管理层级最高
}

# 约束检查：β₁ < β₂ < β₃ < β₄
assert GROUP_BETA[GroupType.DORM] < GROUP_BETA[GroupType.CLASS] < \
       GROUP_BETA[GroupType.MAJOR] < GROUP_BETA[GroupType.CAMPUS], \
       "βₖ 必须满足 β₁ < β₂ < β₃ < β₄"


# ---------- 行动类型枚举 ----------
class ActionType(IntEnum):
    SEND_MESSAGE = 0  # 发新消息
    REPLY = 1         # 回复他人消息
    FORWARD = 2       # 转发消息
    SILENT = 3        # 沉默


# ---------- 消息类型（5类，与 ActionType 独立） ----------
class MessageType:
    ORIGINAL = "original"            # 原始信息
    FORWARD = "forward"              # 直接转发
    PARAPHRASE = "paraphrase"        # 转述
    EXAGGERATE = "exaggerate"        # 夸大
    CLARIFICATION = "clarification"  # 澄清


# ---------- 数据类定义 ----------

@dataclass
class Personality:
    """五因素人格"""
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5


@dataclass
class EmotionState:
    """情绪状态"""
    valence: float = 0.0          # 情绪效价 [-1, 1]
    arousal: float = 0.5          # 唤醒度 [0, 1]


@dataclass
class IdentityBelief:
    """身份信念"""
    agent_type: AgentType
    group_type: GroupType
    nickname: str = ""
    role_desc: str = ""
    stance_prior: float = 0.0


@dataclass
class PsychologyBelief:
    """心理信念"""
    personality: Personality = field(default_factory=Personality)
    risk_aversion: float = 0.5


@dataclass
class OpinionBelief:
    """观点信念（按话题组织）"""
    topic_id: str
    opinion_value: float = 0.0    # [-1, 1]
    confidence: float = 0.5       # [0, 1]


@dataclass
class BeliefSystem:
    """四层信念系统"""
    identity: IdentityBelief
    psychology: PsychologyBelief
    opinions: Dict[str, OpinionBelief] = field(default_factory=dict)
    emotion: EmotionState = field(default_factory=EmotionState)


@dataclass
class SocialInfo:
    """微信消息信息"""
    source_id: int
    source_nickname: str
    content: str
    message_type: str = MessageType.ORIGINAL
    timestamp: int = 0
    is_mention: bool = False
    topic_id: str = ""
    distortion_level: float = 0.0           # [0, 1]
    original_content: str = ""              # 原始内容
    negative_score: float = 0.0             # 负面程度 [0, 1]
    heat: float = 0.0                       # 消息热度 H(t)，由 A 模块管理


@dataclass
class Perception:
    """感知对象（仅读取，不持有权威状态）"""
    group_id: str = ""
    group_type: GroupType = GroupType.CLASS
    beta: float = 0.0                       # 当前群的干预系数 βₖ
    recent_messages: List[SocialInfo] = field(default_factory=list)
    mentions: List[SocialInfo] = field(default_factory=list)
    tick: int = 0
    topic_heat: Dict[str, float] = field(default_factory=dict)       # A 模块维护的权威热度
    topic_negative: Dict[str, float] = field(default_factory=dict)   # A 模块维护的负面程度


@dataclass
class MemoryRecord:
    """记忆记录"""
    tick: int
    info: SocialInfo
    relevance: float = 0.5


@dataclass
class Desire:
    """欲望"""
    goal_type: str                # reply / discuss / share / clarify / intervene / silent
    priority: float
    topic_id: str
    target_id: Optional[int] = None


@dataclass
class Intention:
    """意图"""
    action_type: ActionType
    content_plan: str
    topic_id: str
    target_id: Optional[int] = None


@dataclass
class ActionRecord:
    """行动记录（ActionType + MessageType 双体系）"""
    agent_id: int
    action_type: ActionType       # SEND_MESSAGE / REPLY / FORWARD / SILENT
    content: str
    target_id: Optional[int] = None
    tick: int = 0
    topic_id: str = ""
    distortion_level: float = 0.0
    message_type: str = MessageType.ORIGINAL   # 5类消息类型，独立于 ActionType
    negative_score: float = 0.0
    heat: float = 0.0                         # 由 B 模块按接口表 #9 计算


# ======================== SocialAgent 类实现 ========================

class SocialAgent:
    """
    微信场景智能体大脑

    热度演化公式（Wei, 2022 热传导理论）：
        H_k(t+1) = H_k(t) · e^(-α) · e^(-β_k · 𝟙[t ≥ t_k^int])

    参数说明：
        α  : 自然衰减率（统一常数）
        βₖ : 群 k 的干预衰减率（GROUP_BETA 表）
        θ  : 干预触发阈值（THETA）
        t_k^int : 群 k 的干预触发时间步（由 A 模块维护）
    """

    # ---------- 模型参数 ----------
    ALPHA: float = 0.15           # 自然衰减率 α（Wei, 2022 热传导理论）
    THETA: float = 0.7            # 干预触发阈值 θ（Sun et al., 2025）

    # 注意：GROUP_BETA 在类外部定义，通过 self.beta 引用

    def __init__(
        self,
        unique_id: int,
        model: Any,
        agent_type: AgentType,
        group_type: GroupType,
        init_config: Dict[str, Any],
    ):
        self.unique_id = unique_id
        self.model = model
        self.agent_type = agent_type
        self.group_type = group_type

        # ---- 读取当前群的干预系数 βₖ ----
        self.beta = GROUP_BETA.get(group_type, 0.0)

        # ---- 信念系统 ----
        self.beliefs = BeliefSystem(
            identity=IdentityBelief(
                agent_type=agent_type,
                group_type=group_type,
                nickname=init_config.get("nickname", f"用户{unique_id}"),
                role_desc=init_config.get("role_desc", ""),
                stance_prior=init_config.get("stance_prior", 0.0),
            ),
            psychology=PsychologyBelief(),
            opinions={},
            emotion=EmotionState(),
        )

        # ---- 短期记忆 ----
        self.memory: List[MemoryRecord] = []
        self.max_memory = init_config.get("max_memory", 20)

        # ---- 待提交行动 ----
        self.pending_action: Optional[ActionRecord] = None

        # ---- 缓存最近感知（仅缓存，不持有权威状态） ----
        self._last_perception: Optional[Perception] = None

        # ---- 干预记录（仅供 A 模块参考，权威状态由 A 维护） ----
        self.intervention_tick: Optional[int] = None
        self.last_intervention_topic: Optional[str] = None

        # ---- H₀ 仅记录值（权威状态由 A 模块管理） ----
        self.initial_heat: float = init_config.get("initial_heat", 0.0)

        # ---- 初始化 ----
        self._init_psychology()

        for topic_id, val in init_config.get("initial_opinions", {}).items():
            self.beliefs.opinions[topic_id] = OpinionBelief(
                topic_id=topic_id,
                opinion_value=val,
                confidence=init_config.get("initial_confidence", 0.5)
            )

        self.logger = logging.getLogger(f"Agent_{unique_id}")

    # ---------- 函数1：心理学初始化 ----------
    def _init_psychology(self) -> None:
        rng = self.model.random
        self.beliefs.psychology = PsychologyBelief(
            personality=Personality(
                openness=rng.random(),
                conscientiousness=rng.random(),
                extraversion=rng.random(),
                agreeableness=rng.random(),
                neuroticism=rng.random(),
            ),
            risk_aversion=rng.random(),
        )

    # ---------- 函数2：感知（仅读取 A 的权威状态） ----------
    def _perceive(self) -> Perception:
        """从 A 模块读取群聊上下文及当前热度（只读）"""
        group_id = "unknown"
        group_type = self.group_type
        beta = self.beta

        if hasattr(self.model, "get_agent_group"):
            group_id = self.model.get_agent_group(self.unique_id)

        if hasattr(self.model, "get_group_type"):
            group_type = self.model.get_group_type(group_id)
            beta = GROUP_BETA.get(group_type, 0.0)

        recent_msgs = []
        if hasattr(self.model, "get_group_messages"):
            recent_msgs = self.model.get_group_messages(group_id, limit=20)

        mentions = [m for m in recent_msgs if m.is_mention and m.source_id != self.unique_id]

        # ---- 从 A 模块读取权威状态 ----
        topic_heat = {}
        topic_negative = {}
        if hasattr(self.model, "get_topic_heat"):
            topic_heat = self.model.get_topic_heat(group_id)
        if hasattr(self.model, "get_topic_negative"):
            topic_negative = self.model.get_topic_negative(group_id)

        tick = 0
        if hasattr(self.model, "schedule") and hasattr(self.model.schedule, "time"):
            tick = self.model.schedule.time

        perception = Perception(
            group_id=group_id,
            group_type=group_type,
            beta=beta,
            recent_messages=recent_msgs,
            mentions=mentions,
            tick=tick,
            topic_heat=topic_heat,
            topic_negative=topic_negative,
        )

        self._last_perception = perception
        return perception

    # ---------- 函数3：记忆检索 ----------
    def _retrieve_memory(self, perception: Perception) -> List[MemoryRecord]:
        if not self.memory:
            return []
        sorted_mem = sorted(self.memory, key=lambda x: x.tick, reverse=True)
        return sorted_mem[:5]

    # ---------- 函数4：更新信念 ----------
    def _update_beliefs(self, perception: Perception, memories: List[MemoryRecord]) -> None:
        """调用 LLM 更新信念，失败时保持原状"""
        env_info = {
            "group_type": perception.group_type.name,
            "beta": perception.beta,
            "role": self.agent_type.name,
            "nickname": self.beliefs.identity.nickname,
            "recent_messages": [
                f"[{m.message_type}] {m.source_nickname}: {m.content[:50]}"
                for m in perception.recent_messages[-5:]
            ],
            "mentions": [
                f"{m.source_nickname}: {m.content[:50]}"
                for m in perception.mentions
            ],
            "topic_heat": perception.topic_heat,
            "topic_negative": perception.topic_negative,
        }

        try:
            prompt = self.model.llm_utils.build_prompt(
                belief=self.beliefs,
                memory=memories,
                env_info=env_info,
            )
            client = self.model.llm_utils.get_client()
            raw_response = client.chat(prompt)
            parsed = self.model.llm_utils.parse_llm_response(raw_response)

            if "opinion_updates" in parsed:
                for topic_id, val in parsed["opinion_updates"].items():
                    new_val = max(-1.0, min(1.0, val))
                    if topic_id in self.beliefs.opinions:
                        self.beliefs.opinions[topic_id].opinion_value = new_val
                    else:
                        self.beliefs.opinions[topic_id] = OpinionBelief(
                            topic_id=topic_id, opinion_value=new_val
                        )

            if "emotion_delta" in parsed:
                delta = parsed["emotion_delta"]
                new_valence = self.beliefs.emotion.valence + delta.get("valence", 0.0)
                self.beliefs.emotion.valence = max(-1.0, min(1.0, new_valence))
                new_arousal = self.beliefs.emotion.arousal + delta.get("arousal", 0.0)
                self.beliefs.emotion.arousal = max(0.0, min(1.0, new_arousal))

        except Exception as e:
            self.logger.warning(f"LLM update failed: {e}")

    # ---------- 函数5：推断欲望 ----------
    def _infer_desires(self) -> List[Desire]:
        """基于信念生成欲望"""
        desires = []

        # ---- Controller 干预逻辑 ----
        # DORM 群不触发干预（t_dorm^int = +∞）
        if self.agent_type == AgentType.CONTROLLER and self._last_perception:
            if self.group_type != GroupType.DORM:
                for topic_id, heat in self._last_perception.topic_heat.items():
                    if heat >= self.THETA:
                        desires.append(Desire(
                            goal_type="intervene",
                            priority=0.95,
                            topic_id=topic_id
                        ))

        # 被 @ → 回复
        if self._last_perception and self._last_perception.mentions:
            desires.append(Desire(
                goal_type="reply",
                priority=0.9,
                topic_id=self._last_perception.mentions[0].topic_id or "mentioned",
                target_id=self._last_perception.mentions[0].source_id
            ))

        # 情绪负面 → 讨论
        if self.beliefs.emotion.valence < -0.3:
            topic_id = list(self.beliefs.opinions.keys())[0] if self.beliefs.opinions else "general"
            desires.append(Desire(
                goal_type="discuss",
                priority=0.7,
                topic_id=topic_id
            ))

        # 活跃讨论者 → 分享
        if self.agent_type == AgentType.ACTIVE:
            desires.append(Desire(
                goal_type="share",
                priority=0.6,
                topic_id="general"
            ))

        # 理性讨论者 → 澄清
        if self.agent_type == AgentType.RATIONAL:
            desires.append(Desire(
                goal_type="clarify",
                priority=0.8,
                topic_id="fact_check"
            ))

        if not desires:
            desires.append(Desire(
                goal_type="silent",
                priority=0.1,
                topic_id=""
            ))

        desires.sort(key=lambda d: d.priority, reverse=True)
        return desires

    # ---------- 函数6：规划意图 ----------
    def _plan_intentions(self, desires: List[Desire]) -> Intention:
        """将欲望转化为具体意图"""
        if not desires:
            return Intention(ActionType.SILENT, "", "")

        top = desires[0]
        goal = top.goal_type

        mapping = {
            "reply": (ActionType.REPLY, "回复内容"),
            "discuss": (ActionType.SEND_MESSAGE, "表达观点"),
            "share": (ActionType.FORWARD, "转发分享"),
            "clarify": (ActionType.SEND_MESSAGE, "查证澄清"),
            "intervene": (ActionType.SEND_MESSAGE, "【官方提醒】请理性讨论，请勿传播不实信息"),
        }
        action, content = mapping.get(goal, (ActionType.SILENT, ""))

        return Intention(
            action_type=action,
            content_plan=content,
            topic_id=top.topic_id,
            target_id=top.target_id,
        )

    # ---------- 函数7：执行行动 ----------
    def _execute_action(self, intention: Intention) -> ActionRecord:
        tick = 0
        if hasattr(self.model, "schedule") and hasattr(self.model.schedule, "time"):
            tick = self.model.schedule.time

        if intention.action_type == ActionType.SILENT:
            return ActionRecord(
                agent_id=self.unique_id,
                action_type=ActionType.SILENT,
                content="",
                tick=tick,
                topic_id=intention.topic_id,
                distortion_level=0.0,
                message_type=MessageType.ORIGINAL,
                negative_score=0.0,
                heat=0.0,
            )

        # ---- 计算 distortion 和 message_type ----
        distortion = 0.0
        message_type = MessageType.ORIGINAL
        negative_score = 0.0

        # 获取当前话题热度和负面程度
        current_negative = 0.0
        current_topic_heat = 0.0
        if self._last_perception:
            current_negative = self._last_perception.topic_negative.get(intention.topic_id, 0.0)
            current_topic_heat = self._last_perception.topic_heat.get(intention.topic_id, 0.0)

        # 干预消息自动标记为 CLARIFICATION
        if "【官方提醒】" in intention.content_plan:
            message_type = MessageType.CLARIFICATION
            negative_score = max(0.0, current_negative - 0.3)
            distortion = 0.0

        elif intention.action_type == ActionType.FORWARD:
            message_type = MessageType.FORWARD
            if self.agent_type == AgentType.ORDINARY:
                distortion = min(1.0, self.model.random.random() * 0.6 + 0.1)
                if distortion > 0.5:
                    message_type = MessageType.EXAGGERATE
                elif distortion > 0.3:
                    message_type = MessageType.PARAPHRASE
            elif self.agent_type == AgentType.ACTIVE:
                distortion = self.model.random.random() * 0.4
                if distortion > 0.4:
                    message_type = MessageType.PARAPHRASE
            elif self.agent_type == AgentType.RATIONAL:
                distortion = self.model.random.random() * 0.1
                message_type = MessageType.CLARIFICATION
            else:
                distortion = self.model.random.random() * 0.05

            negative_score = min(1.0, current_negative + self.model.random.random() * 0.1)

        elif intention.action_type == ActionType.SEND_MESSAGE:
            if "查证澄清" in intention.content_plan or self.agent_type == AgentType.RATIONAL:
                message_type = MessageType.CLARIFICATION
                negative_score = max(0.0, current_negative - 0.2)
            else:
                message_type = MessageType.ORIGINAL
                negative_score = min(1.0, current_negative + 0.05)

        elif intention.action_type == ActionType.REPLY:
            message_type = MessageType.ORIGINAL
            negative_score = max(0.0, current_negative - 0.1)

        # ---- 计算本次消息的热度贡献（接口表 #9） ----
        # heat = extraversion × (1 + topic_heat_now × 0.1)
        extraversion = self.beliefs.psychology.personality.extraversion
        heat_contribution = extraversion * (1.0 + current_topic_heat * 0.1)
        heat_contribution = max(0.0, heat_contribution)

        # ---- Controller 干预记录 ----
        if "【官方提醒】" in intention.content_plan:
            if self.group_type != GroupType.DORM:
                self.intervention_tick = tick
                self.last_intervention_topic = intention.topic_id

        record = ActionRecord(
            agent_id=self.unique_id,
            action_type=intention.action_type,
            content=intention.content_plan,
            target_id=intention.target_id,
            tick=tick,
            topic_id=intention.topic_id,
            distortion_level=distortion,
            message_type=message_type,
            negative_score=negative_score,
            heat=heat_contribution,
        )

        # ---- 写入记忆 ----
        info = SocialInfo(
            source_id=self.unique_id,
            source_nickname=self.beliefs.identity.nickname,
            content=intention.content_plan,
            message_type=message_type,
            timestamp=record.tick,
            topic_id=intention.topic_id,
            distortion_level=distortion,
            original_content=intention.content_plan,
            negative_score=negative_score,
            heat=heat_contribution,
        )
        self.memory.append(MemoryRecord(tick=record.tick, info=info, relevance=0.5))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

        return record

    # ---------- 函数8：更新情绪 ----------
    def _update_emotion(self, env_feedback: Dict[str, float]) -> None:
        """情绪更新：独立于 βₖ，纯自然心理恢复"""
        base_valence = 0.0
        base_arousal = 0.5
        emotion_decay = 0.05

        if "valence_delta" in env_feedback:
            new_val = self.beliefs.emotion.valence + env_feedback["valence_delta"]
        else:
            new_val = self.beliefs.emotion.valence * (1 - emotion_decay) + base_valence * emotion_decay

        if "arousal_delta" in env_feedback:
            new_ar = self.beliefs.emotion.arousal + env_feedback["arousal_delta"]
        else:
            new_ar = self.beliefs.emotion.arousal * (1 - emotion_decay) + base_arousal * emotion_decay

        self.beliefs.emotion.valence = max(-1.0, min(1.0, new_val))
        self.beliefs.emotion.arousal = max(0.0, min(1.0, new_ar))

    # ---------- 函数9：热度衰减计算（工具函数，供 A 模块调用） ----------
    def calc_heat_decay(
        self,
        current_heat: float,
        elapsed_steps: int,
        intervention_tick: Optional[int] = None
    ) -> float:
        """
        计算经过指数衰减后的热度 H(t)（纯工具函数，不存储状态）

        公式：H_k(t+1) = H_k(t) · e^(-α) · e^(-β_k · 𝟙[t ≥ t_k^int])

        【注意】此函数为工具函数，返回计算值供 A 模块使用。
        topic_heat 的权威状态由 A 模块维护，B 模块不存储热度。
        """
        # 1. 自然衰减：e^(-α)
        natural_decay = math.exp(-self.ALPHA)

        # 2. 干预衰减：e^(-β_k · 𝟙[t ≥ t_k^int])
        # DORM 群不触发干预
        if self.group_type == GroupType.DORM:
            intervention_decay = 1.0
        elif intervention_tick is not None and elapsed_steps >= intervention_tick:
            intervention_decay = math.exp(-self.beta)
        else:
            intervention_decay = 1.0

        # 3. 综合衰减
        new_heat = current_heat * natural_decay * intervention_decay
        return max(0.0, new_heat)

    # ---------- 函数10：主循环 ----------
    def step(self) -> None:
        """主入口，串联完整 BDI 推理链"""
        try:
            perception = self._perceive()
            memories = self._retrieve_memory(perception)
            self._update_beliefs(perception, memories)
            desires = self._infer_desires()
            intention = self._plan_intentions(desires)
            action_record = self._execute_action(intention)
            self._update_emotion({})

            if action_record and action_record.action_type != ActionType.SILENT:
                self.pending_action = action_record
                if hasattr(self.model, "submit_action"):
                    self.model.submit_action(action_record)

        except Exception as e:
            self.logger.error(f"Agent {self.unique_id} step failed: {e}")
            self.pending_action = None