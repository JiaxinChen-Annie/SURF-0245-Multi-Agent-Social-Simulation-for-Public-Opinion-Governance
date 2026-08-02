"""
social_agent.py
B 模块：智能体大脑（SocialAgent）- 微信舆论传播管理版
v3.2 - 记忆系统升级 + 感知/记忆分离 + 分角色行为差异

更新内容（2026-08-02）：
1. 记忆系统：短期明细 + 长期摘要（压缩），不再仅靠 [-5:] 切片
2. Perception 与 Memory 分离：感知数据需筛选后才写入记忆
3. 分角色差异保证：各 AgentType 在信念更新/转发/干预上差异可量化
4. belief 迁移预留（虚假信息累积 → trust↓，钩子已埋）
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from enum import IntEnum
from dataclasses import dataclass, field
import logging
import math
import time
from collections import deque

# ======================== 类型定义 ========================

class AgentType(IntEnum):
    ORDINARY = 0      # 普通群员
    ACTIVE = 1        # 活跃讨论者
    RATIONAL = 2      # 理性讨论者
    CONTROLLER = 3    # 管理者


class GroupType(IntEnum):
    DORM = 0
    CLASS = 1
    MAJOR = 2
    CAMPUS = 3


GROUP_BETA = {
    GroupType.DORM: 0.05,
    GroupType.CLASS: 0.12,
    GroupType.MAJOR: 0.20,
    GroupType.CAMPUS: 0.30,
}

assert GROUP_BETA[GroupType.DORM] < GROUP_BETA[GroupType.CLASS] < \
       GROUP_BETA[GroupType.MAJOR] < GROUP_BETA[GroupType.CAMPUS], \
       "βₖ 必须满足 β₁ < β₂ < β₃ < β₄"


class ActionType(IntEnum):
    SEND_MESSAGE = 0
    REPLY = 1
    FORWARD = 2
    SILENT = 3


class MessageType:
    ORIGINAL = "original"
    FORWARD = "forward"
    PARAPHRASE = "paraphrase"
    EXAGGERATE = "exaggerate"
    CLARIFICATION = "clarification"


# ---------- 记忆系统数据结构（v3.2 新增） ----------

@dataclass
class MemoryEntry:
    """单条记忆条目（带重要性标记）"""
    tick: int
    content: str
    source_id: int = -1
    topic_id: str = ""
    importance: float = 0.5          # 0~1，决定是否固化为长期记忆
    is_high_impact: bool = False     # 高影响事件标记（用于固化）
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LongTermSummary:
    """长期记忆摘要（压缩形式）"""
    tick_range: Tuple[int, int]      # (start_tick, end_tick)
    summary: str                      # 文本摘要
    key_topics: List[str] = field(default_factory=list)
    key_events: List[str] = field(default_factory=list)
    avg_opinion_shift: float = 0.0   # 该时段观点平均偏移


@dataclass
class MemorySystem:
    """Agent 的记忆系统（v3.2 升级）"""
    # 短期记忆：明细列表（有限容量）
    short_term: deque = field(default_factory=lambda: deque(maxlen=20))
    
    # 长期摘要：压缩后的历史摘要列表
    long_term_summaries: List[LongTermSummary] = field(default_factory=list)
    
    # 最近 N 步的高影响事件（用于快速检索）
    high_impact_events: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # 总步数计数（用于摘要触发）
    total_steps: int = 0
    
    # 摘要触发间隔（每 N 步生成一次摘要）
    summary_interval: int = 10


# ---------- 原有数据类（保持兼容） ----------

@dataclass
class Personality:
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5


@dataclass
class EmotionState:
    valence: float = 0.0
    arousal: float = 0.5


@dataclass
class IdentityBelief:
    agent_type: AgentType
    group_type: GroupType
    nickname: str = ""
    role_desc: str = ""
    stance_prior: float = 0.0


@dataclass
class PsychologyBelief:
    personality: Personality = field(default_factory=Personality)
    risk_aversion: float = 0.5
    # v3.2 新增：信任度（可因虚假信息累积而下降）
    trust_in_info: float = 0.8       # 0~1，初始高信任


@dataclass
class OpinionBelief:
    topic_id: str
    opinion_value: float = 0.0
    confidence: float = 0.5


@dataclass
class BeliefSystem:
    identity: IdentityBelief
    psychology: PsychologyBelief
    opinions: Dict[str, OpinionBelief] = field(default_factory=dict)
    emotion: EmotionState = field(default_factory=EmotionState)


@dataclass
class SocialInfo:
    source_id: int
    source_nickname: str
    content: str
    message_type: str = MessageType.ORIGINAL
    timestamp: int = 0
    is_mention: bool = False
    topic_id: str = ""
    distortion_level: float = 0.0
    original_content: str = ""
    negative_score: float = 0.0
    heat: float = 0.0


@dataclass
class Perception:
    group_id: str = ""
    group_type: GroupType = GroupType.CLASS
    beta: float = 0.0
    recent_messages: List[SocialInfo] = field(default_factory=list)
    mentions: List[SocialInfo] = field(default_factory=list)
    tick: int = 0
    topic_heat: Dict[str, float] = field(default_factory=dict)
    topic_negative: Dict[str, float] = field(default_factory=dict)
    # v3.2 新增：时间/场所上下文
    time_of_day: str = "day"          # "morning" / "afternoon" / "evening" / "night"
    place: str = "general"            # "dorm" / "classroom" / "cafeteria" / "general"


@dataclass
class Desire:
    goal_type: str
    priority: float
    topic_id: str
    target_id: Optional[int] = None


@dataclass
class Intention:
    action_type: ActionType
    content_plan: str
    topic_id: str
    target_id: Optional[int] = None


@dataclass
class ActionRecord:
    agent_id: int
    action_type: ActionType
    content: str
    target_id: Optional[int] = None
    tick: int = 0
    topic_id: str = ""
    distortion_level: float = 0.0
    message_type: str = MessageType.ORIGINAL
    negative_score: float = 0.0
    heat: float = 0.0
    # v3.2 新增：行为角色标识（方便 E 按角色分表）
    agent_type: str = ""
    group_type: str = ""


# ======================== SocialAgent 类实现 ========================

class SocialAgent:
    """
    v3.2 更新：
    - 记忆系统：短期明细 + 长期摘要
    - Perception 与 Memory 分离
    - 分角色行为差异保证
    """

    ALPHA: float = 0.15
    THETA: float = 0.7

    # 记忆系统参数
    SHORT_TERM_MAX: int = 20          # 短期记忆最大条数
    SUMMARY_INTERVAL: int = 10        # 每 N 步生成一次摘要
    HIGH_IMPACT_THRESHOLD: float = 0.7  # 重要性 > 此值固化到长期

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
        self.beta = GROUP_BETA.get(group_type, 0.0)

        # ---- v3.2 记忆系统（升级） ----
        self.memory = MemorySystem(
            short_term=deque(maxlen=self.SHORT_TERM_MAX),
            summary_interval=self.SUMMARY_INTERVAL,
        )

        # ---- 信念系统（新增 trust_in_info） ----
        self.beliefs = BeliefSystem(
            identity=IdentityBelief(
                agent_type=agent_type,
                group_type=group_type,
                nickname=init_config.get("nickname", f"用户{unique_id}"),
                role_desc=init_config.get("role_desc", ""),
                stance_prior=init_config.get("stance_prior", 0.0),
            ),
            psychology=PsychologyBelief(
                trust_in_info=init_config.get("initial_trust", 0.8),
            ),
            opinions={},
            emotion=EmotionState(),
        )

        # ---- 其他状态 ----
        self.pending_action: Optional[ActionRecord] = None
        self._last_perception: Optional[Perception] = None
        self.intervention_tick: Optional[int] = None
        self.last_intervention_topic: Optional[str] = None
        self.initial_heat: float = init_config.get("initial_heat", 0.0)

        self._init_psychology()

        for topic_id, val in init_config.get("initial_opinions", {}).items():
            self.beliefs.opinions[topic_id] = OpinionBelief(
                topic_id=topic_id, opinion_value=val
            )

        self.logger = logging.getLogger(f"Agent_{unique_id}")

    # ---------- 函数1：心理学初始化 ----------
    def _init_psychology(self) -> None:
        rng = self.model.random
        self.beliefs.psychology.personality = Personality(
            openness=rng.random(),
            conscientiousness=rng.random(),
            extraversion=rng.random(),
            agreeableness=rng.random(),
            neuroticism=rng.random(),
        )
        self.beliefs.psychology.risk_aversion = rng.random()
        # trust_in_info 已在 __init__ 中设置，这里保持

    # ---------- 函数2：感知 ----------
    def _perceive(self) -> Perception:
        """从 A 模块读取感知数据（不自动写入记忆）"""
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

        topic_heat = {}
        topic_negative = {}
        if hasattr(self.model, "get_topic_heat"):
            topic_heat = self.model.get_topic_heat(group_id)
        if hasattr(self.model, "get_topic_negative"):
            topic_negative = self.model.get_topic_negative(group_id)

        tick = 0
        if hasattr(self.model, "schedule") and hasattr(self.model.schedule, "time"):
            tick = self.model.schedule.time

        # ---- v3.2 新增：时间/场所上下文 ----
        time_of_day = "day"
        place = "general"
        if hasattr(self.model, "get_time_of_day"):
            time_of_day = self.model.get_time_of_day(tick)
        if hasattr(self.model, "get_place"):
            place = self.model.get_place(group_id)

        perception = Perception(
            group_id=group_id,
            group_type=group_type,
            beta=beta,
            recent_messages=recent_msgs,
            mentions=mentions,
            tick=tick,
            topic_heat=topic_heat,
            topic_negative=topic_negative,
            time_of_day=time_of_day,
            place=place,
        )

        self._last_perception = perception
        return perception

    # ---------- 函数3：记忆检索（v3.2 升级） ----------
    def _retrieve_memory(self, perception: Perception) -> List[MemoryEntry]:
        """
        检索相关记忆：优先返回短期记忆中最近的 + 长期摘要中的关键事件
        不再仅靠 [-5:] 切片
        """
        retrieved = []

        # 1. 短期记忆：返回最近 N 条（最多 10 条）
        short_term_list = list(self.memory.short_term)
        retrieved.extend(short_term_list[-10:])

        # 2. 高影响事件：返回最近的高重要性事件
        high_impact_list = list(self.memory.high_impact_events)
        retrieved.extend(high_impact_list[-3:])

        # 3. 长期摘要：如果话题匹配，返回相关摘要
        if perception.topic_heat:
            active_topics = list(perception.topic_heat.keys())
            for summary in self.memory.long_term_summaries[-3:]:  # 最近 3 个摘要
                if any(t in summary.key_topics for t in active_topics):
                    # 把摘要当作一条记忆返回（用特殊格式）
                    retrieved.append(MemoryEntry(
                        tick=summary.tick_range[1],
                        content=f"[摘要] {summary.summary}",
                        topic_id=active_topics[0] if active_topics else "",
                        importance=0.3,
                    ))

        # 去重（按 tick + content 简单去重）
        seen = set()
        unique = []
        for entry in retrieved:
            key = (entry.tick, entry.content[:20])
            if key not in seen:
                seen.add(key)
                unique.append(entry)

        return unique[:15]  # 最多返回 15 条

    # ---------- 函数4：写入记忆（v3.2 新增分离方法） ----------
    def _write_memory(self, entry: MemoryEntry) -> None:
        """将筛选后的事件写入记忆系统"""
        # 更新总步数
        self.memory.total_steps += 1

        # 写入短期记忆
        self.memory.short_term.append(entry)

        # 高影响事件 → 固化到长期
        if entry.importance >= self.HIGH_IMPACT_THRESHOLD or entry.is_high_impact:
            self.memory.high_impact_events.append(entry)

        # 触发摘要压缩
        if self.memory.total_steps % self.memory.summary_interval == 0 and len(self.memory.short_term) > 5:
            self._compress_memory()

    def _compress_memory(self) -> None:
        """压缩短期记忆为长期摘要（v3.2 新增）"""
        if len(self.memory.short_term) < 5:
            return

        # 取最近一个区间内的记忆
        entries = list(self.memory.short_term)[-self.memory.summary_interval:]
        if not entries:
            return

        # 提取关键信息
        key_topics = list(set(e.topic_id for e in entries if e.topic_id))
        key_events = [e.content[:30] for e in entries if e.importance > 0.5][:3]

        # 计算观点平均偏移（如果有 opinion 数据）
        avg_shift = 0.0
        # 这里简化：可以从 entries 的 metadata 中读取

        start_tick = entries[0].tick if entries else 0
        end_tick = entries[-1].tick if entries else 0

        # 生成摘要文本（简化版，实际可用 LLM 生成）
        summary_text = f"第{start_tick}-{end_tick}步: 共{len(entries)}条事件"

        summary = LongTermSummary(
            tick_range=(start_tick, end_tick),
            summary=summary_text,
            key_topics=key_topics[:3],
            key_events=key_events,
            avg_opinion_shift=avg_shift,
        )

        self.memory.long_term_summaries.append(summary)

        # 限制摘要数量（保留最近 10 个）
        if len(self.memory.long_term_summaries) > 10:
            self.memory.long_term_summaries = self.memory.long_term_summaries[-10:]

    # ---------- 函数5：更新信念（v3.2 新增角色差异 + trust 迁移预留） ----------
    def _update_beliefs(self, perception: Perception, memories: List[MemoryEntry]) -> None:
        """调用 LLM 更新信念，LLM 失败时保持原状"""
        # ---- 构造 env_info（精简版，不超过 8k token） ----
        env_info = {
            "group_type": perception.group_type.name,
            "beta": perception.beta,
            "role": self.agent_type.name,
            "nickname": self.beliefs.identity.nickname,
            "time_of_day": perception.time_of_day,
            "place": perception.place,
            # 只保留最近 5 条消息（而非全量）
            "recent_messages": [
                f"[{m.message_type}] {m.source_nickname}: {m.content[:80]}"
                for m in perception.recent_messages[-5:]
            ],
            "mentions": [
                f"{m.source_nickname}: {m.content[:80]}"
                for m in perception.mentions
            ],
            "topic_heat": perception.topic_heat,
            "topic_negative": perception.topic_negative,
            # 记忆摘要（而非完整记忆）
            "memory_summary": [m.content[:50] for m in memories[-3:]],
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

            # ---- 分角色差异保证 ----
            # 不同角色对 LLM 输出的信任度不同
            trust_factor = self.beliefs.psychology.trust_in_info

            if "opinion_updates" in parsed:
                for topic_id, val in parsed["opinion_updates"].items():
                    # RATIONAL 角色对更新更保守（乘以较低系数）
                    if self.agent_type == AgentType.RATIONAL:
                        val = val * 0.7 + 0.3 * self.beliefs.opinions.get(topic_id, OpinionBelief(topic_id,0)).opinion_value
                    # ORDINARY 角色更易受虚假信息影响（trust_factor 衰减预留）
                    elif self.agent_type == AgentType.ORDINARY:
                        val = val * trust_factor + (1 - trust_factor) * 0.0

                    new_val = max(-1.0, min(1.0, val))
                    if topic_id in self.beliefs.opinions:
                        self.beliefs.opinions[topic_id].opinion_value = new_val
                    else:
                        self.beliefs.opinions[topic_id] = OpinionBelief(
                            topic_id=topic_id, opinion_value=new_val
                        )

            if "emotion_delta" in parsed:
                delta = parsed["emotion_delta"]
                # CONTROLLER 情绪变化更平缓
                if self.agent_type == AgentType.CONTROLLER:
                    delta["valence"] = delta.get("valence", 0.0) * 0.5
                new_valence = self.beliefs.emotion.valence + delta.get("valence", 0.0)
                self.beliefs.emotion.valence = max(-1.0, min(1.0, new_valence))
                new_arousal = self.beliefs.emotion.arousal + delta.get("arousal", 0.0)
                self.beliefs.emotion.arousal = max(0.0, min(1.0, new_arousal))

            # ---- v3.2 预留：虚假信息累积导致 trust 下降 ----
            # 如果本消息 distortion_level > 0.5 且 agent_type == ORDINARY，trust 微降
            if self.agent_type == AgentType.ORDINARY and self._last_perception:
                high_distortion_msgs = [
                    m for m in self._last_perception.recent_messages
                    if m.distortion_level > 0.5
                ]
                if high_distortion_msgs:
                    # 每次接触高失真消息，trust 下降 0.01（预留，暂不启用）
                    # self.beliefs.psychology.trust_in_info = max(0.1, self.beliefs.psychology.trust_in_info - 0.01)
                    pass

        except Exception as e:
            self.logger.warning(f"LLM update failed: {e}")

    # ---------- 函数6：推断欲望（分角色差异保证） ----------
    def _infer_desires(self) -> List[Desire]:
        """基于信念生成欲望，各角色行为差异可度量"""
        desires = []

        # ---- Controller 干预 ----
        if self.agent_type == AgentType.CONTROLLER and self._last_perception:
            if self.group_type != GroupType.DORM:
                for topic_id, heat in self._last_perception.topic_heat.items():
                    if heat >= self.THETA:
                        desires.append(Desire(
                            goal_type="intervene",
                            priority=0.95,
                            topic_id=topic_id
                        ))

        # ---- 被 @ → 回复 ----
        if self._last_perception and self._last_perception.mentions:
            desires.append(Desire(
                goal_type="reply",
                priority=0.9,
                topic_id=self._last_perception.mentions[0].topic_id or "mentioned",
                target_id=self._last_perception.mentions[0].source_id
            ))

        # ---- 情绪负面 → 讨论 ----
        if self.beliefs.emotion.valence < -0.3:
            topic_id = list(self.beliefs.opinions.keys())[0] if self.beliefs.opinions else "general"
            desires.append(Desire(
                goal_type="discuss",
                priority=0.7,
                topic_id=topic_id
            ))

        # ---- 分角色：各 AgentType 差异化欲望 ----
        if self.agent_type == AgentType.ACTIVE:
            desires.append(Desire(
                goal_type="share",
                priority=0.6,
                topic_id="general"
            ))

        elif self.agent_type == AgentType.RATIONAL:
            desires.append(Desire(
                goal_type="clarify",
                priority=0.8,
                topic_id="fact_check"
            ))

        elif self.agent_type == AgentType.ORDINARY:
            # 普通群员：有低概率参与讨论，但优先级低
            if self.beliefs.emotion.valence > 0.3:
                desires.append(Desire(
                    goal_type="discuss",
                    priority=0.3,
                    topic_id="general"
                ))

        # ---- 沉默兜底 ----
        if not desires:
            desires.append(Desire(
                goal_type="silent",
                priority=0.1,
                topic_id=""
            ))

        desires.sort(key=lambda d: d.priority, reverse=True)
        return desires

    # ---------- 函数7：规划意图 ----------
    def _plan_intentions(self, desires: List[Desire]) -> Intention:
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

    # ---------- 函数8：执行行动 ----------
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
                agent_type=self.agent_type.name,
                group_type=self.group_type.name,
            )

        distortion = 0.0
        message_type = MessageType.ORIGINAL
        negative_score = 0.0

        current_negative = 0.0
        current_topic_heat = 0.0
        if self._last_perception:
            current_negative = self._last_perception.topic_negative.get(intention.topic_id, 0.0)
            current_topic_heat = self._last_perception.topic_heat.get(intention.topic_id, 0.0)

        # 干预消息
        if "【官方提醒】" in intention.content_plan:
            message_type = MessageType.CLARIFICATION
            negative_score = max(0.0, current_negative - 0.3)
            distortion = 0.0

        elif intention.action_type == ActionType.FORWARD:
            message_type = MessageType.FORWARD
            # ---- 分角色：转发行为差异（确保 E 可度量） ----
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
            else:  # CONTROLLER
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

        # ---- 计算 heat ----
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
            agent_type=self.agent_type.name,   # v3.2：便于 E 按角色分表
            group_type=self.group_type.name,   # v3.2：便于 E 按群分表
        )

        # ---- v3.2：感知与记忆分离 ----
        # 只有重要事件才写入记忆
        importance = 0.3
        is_high_impact = False

        # 高失真消息 → 高重要性
        if distortion > 0.5:
            importance = 0.7
            is_high_impact = True

        # 干预消息 → 高重要性
        if "【官方提醒】" in intention.content_plan:
            importance = 0.8
            is_high_impact = True

        # 非 SILENT 且内容非空 → 写入记忆
        if intention.action_type != ActionType.SILENT and intention.content_plan:
            entry = MemoryEntry(
                tick=record.tick,
                content=intention.content_plan,
                source_id=self.unique_id,
                topic_id=intention.topic_id,
                importance=importance,
                is_high_impact=is_high_impact,
                metadata={
                    "action_type": intention.action_type.name,
                    "message_type": message_type,
                    "distortion_level": distortion,
                    "negative_score": negative_score,
                }
            )
            self._write_memory(entry)

        return record

    # ---------- 函数9：情绪更新 ----------
    def _update_emotion(self, env_feedback: Dict[str, float]) -> None:
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

    # ---------- 函数10：热度衰减计算（工具函数） ----------
    def calc_heat_decay(
        self,
        current_heat: float,
        elapsed_steps: int,
        intervention_tick: Optional[int] = None
    ) -> float:
        natural_decay = math.exp(-self.ALPHA)

        if self.group_type == GroupType.DORM:
            intervention_decay = 1.0
        elif intervention_tick is not None and elapsed_steps >= intervention_tick:
            intervention_decay = math.exp(-self.beta)
        else:
            intervention_decay = 1.0

        new_heat = current_heat * natural_decay * intervention_decay
        return max(0.0, new_heat)

    # ---------- 函数11：主循环 ----------
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

    # ---------- v3.2 新增：获取记忆系统状态（供 E 采集） ----------
    def get_memory_stats(self) -> Dict[str, Any]:
        """返回记忆系统统计信息，供 E 模块采集"""
        return {
            "short_term_count": len(self.memory.short_term),
            "long_term_summary_count": len(self.memory.long_term_summaries),
            "high_impact_count": len(self.memory.high_impact_events),
            "total_steps": self.memory.total_steps,
            "trust_in_info": self.beliefs.psychology.trust_in_info,
        }