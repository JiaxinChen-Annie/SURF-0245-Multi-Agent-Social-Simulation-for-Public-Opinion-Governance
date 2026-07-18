"""
social_agent.py — 智能体层（接口表 #1–11）
-----------------------------------------
负责人：B
本文件为 A 在 W4 阶段使用的「规则存根」，接口已对齐接口表 v2：
  - AgentType: ORDINARY/ACTIVE/RATIONAL/CONTROLLER
  - GroupType: DORM/CLASS/MAJOR/CAMPUS
  - ActionType: SEND_MESSAGE/REPLY/FORWARD/SILENT（移除 LIKE）
  - MessageType: ORIGINAL/FORWARD/PARAPHRASE/EXAGGERATE/CLARIFICATION
  - 所有 event_id → topic_id
  - ActionRecord / SocialInfo / Perception / Desire / Intention 均使用 v2 字段
  - __init__ 新增 group_type 参数（A 模块实例化时传入）
  - 新增 calc_heat_decay (#11，A 模块在 _update_environment 中调用)

B 同学 S2 阶段用真实 LLM 版本替换 _update_beliefs 即可。
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from mesa import Agent
except ImportError:
    from mesa_compat import Agent

from types_def import (
    ActionRecord, ActionType, AgentType, GroupType, MessageType,
    BeliefSystem, Desire, EmotionState,
    IdentityBelief, Intention, MemoryRecord, OpinionBelief, Perception,
    PsychologyBelief, Personality, SocialInfo,
    ALPHA, THETA, GROUP_BETA,
)

_LOG = logging.getLogger("SocialAgent")

# ─── 规则超参（按新 AgentType 定义）────────────────────────────────────────── #
_CONFIDENCE_BOUND: Dict[AgentType, float] = {
    AgentType.ORDINARY:   0.50,   # 普通群员：中等宽容度
    AgentType.ACTIVE:     0.65,   # 活跃者：广泛接纳信息
    AgentType.RATIONAL:   0.30,   # 理性者：立场较固定，质疑来源
    AgentType.CONTROLLER: 0.20,   # 管理者：立场最固定
}
_LEARNING_RATE: Dict[AgentType, float] = {
    AgentType.ORDINARY:   0.12,
    AgentType.ACTIVE:     0.10,
    AgentType.RATIONAL:   0.03,
    AgentType.CONTROLLER: 0.01,
}
_MEMORY_CAPACITY = 20   # 短期记忆最大条数


class SocialAgent(Agent):
    """
    单个社交智能体（W4 规则存根，接口对齐 v2）。
    场景：大学校园多群舆情扩散。
    """

    # B 模块类变量（接口表§二）
    ALPHA: float = ALPHA   # 自然衰减率
    GROUP_BETA   = GROUP_BETA  # {GroupType: β_k}

    # ------------------------------------------------------------------ #
    #  #1  __init__                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        unique_id: int,
        model,
        agent_type: AgentType,
        group_type: GroupType,        # v2 新增参数，A 模块必须传入
        init_config: Dict[str, Any],
    ) -> None:
        super().__init__(unique_id, model)
        self.unique_id = unique_id

        # βₖ（由 group_type 决定，B 模块内部读取 GROUP_BETA）
        self.beta: float = GROUP_BETA[group_type]

        # intervention_tick：A 模块为权威，此处仅供参考记录
        self.intervention_tick: Optional[int] = None

        # 初始热度 H₀
        self.initial_heat: float = float(init_config.get("initial_heat", 0.5))

        topic_id    = init_config.get("topic_id", "T001")
        stance_prior = float(init_config.get("stance_prior", 0.0))
        nickname    = init_config.get("nickname", f"{agent_type.name[:3]}-{unique_id}")

        # 初始化信念系统
        self.beliefs = BeliefSystem(
            identity=IdentityBelief(
                agent_type=agent_type,
                group_type=group_type,          # v2 新增
                nickname=nickname,              # v2 新增
                role_desc=f"{agent_type.name}-{group_type.name}-{unique_id}",
                stance_prior=stance_prior,
            ),
            opinions={
                topic_id: OpinionBelief(
                    topic_id=topic_id,
                    opinion_value=stance_prior,
                    confidence=model.random.uniform(0.3, 0.9),
                )
            },
        )

        self.memory: deque = deque(maxlen=_MEMORY_CAPACITY)
        self.pending_action: Optional[ActionRecord] = None
        self._last_perception: Optional[Perception] = None

        self._init_psychology()

    # ------------------------------------------------------------------ #
    #  #2  _init_psychology                                                #
    # ------------------------------------------------------------------ #
    def _init_psychology(self) -> None:
        """生成五因素人格 + 风险规避系数，写入 self.beliefs.psychology。"""
        rng = self.model.random
        self.beliefs.psychology = PsychologyBelief(
            personality=Personality(
                openness=rng.uniform(0.2, 0.8),
                conscientiousness=rng.uniform(0.2, 0.8),
                extraversion=rng.uniform(0.1, 0.9),
                agreeableness=rng.uniform(0.2, 0.8),
                neuroticism=rng.uniform(0.1, 0.7),
            ),
            risk_aversion=rng.uniform(0.1, 0.9),
        )
        stance = self.beliefs.identity.stance_prior
        neur   = self.beliefs.psychology.personality.neuroticism
        self.beliefs.emotion = EmotionState(
            valence=float(np.clip(stance * 0.3, -1.0, 1.0)),
            arousal=float(np.clip(0.4 + neur * 0.3, 0.0, 1.0)),
        )

    # ------------------------------------------------------------------ #
    #  #3  step                                                            #
    # ------------------------------------------------------------------ #
    def step(self) -> None:
        """串联完整 BDI 链；子步异常须降级，不崩溃。"""
        self.pending_action = None
        try:
            perception = self._perceive()
        except Exception as e:
            _LOG.debug(f"Agent-{self.unique_id} _perceive 异常: {e}")
            return

        try:
            memories = self._retrieve_memory(perception)
        except Exception:
            memories = []

        try:
            self._update_beliefs(perception, memories)
        except Exception as e:
            _LOG.debug(f"Agent-{self.unique_id} _update_beliefs 异常: {e}")

        try:
            desires   = self._infer_desires()
            intention = self._plan_intentions(desires)
            env_fb    = self._execute_action(intention)
            self._update_emotion(env_fb)
        except Exception as e:
            _LOG.debug(f"Agent-{self.unique_id} BDI 后半段异常: {e}")

    # ------------------------------------------------------------------ #
    #  #4  _perceive                                                       #
    # ------------------------------------------------------------------ #
    def _perceive(self) -> Perception:
        """
        读取社交信息流（v2 接口：group_id/group_type/beta/recent_messages/topic_heat/topic_negative）。
        调用 A 模块提供的接口方法。
        """
        tick       = int(self.model.schedule.time)
        group_type = self.beliefs.identity.group_type
        group_id   = self.model.get_agent_group(self.unique_id) or f"GROUP_{group_type.name}"
        beta       = self.beta

        # 从 A 模块获取同群近期消息（最多 20 条）
        raw_records = self.model.get_group_messages(self.unique_id, limit=20)
        recent_messages: List[SocialInfo] = []
        mentions: List[SocialInfo] = []

        for record in raw_records:
            if record.agent_id == self.unique_id:
                continue  # 不感知自己的消息
            src_agent = self.model._get_agent_by_id(record.agent_id)
            src_nick  = (src_agent.beliefs.identity.nickname
                         if src_agent and hasattr(src_agent, "beliefs") else "")
            si = SocialInfo(
                source_id=record.agent_id,
                source_nickname=src_nick,
                content=record.content,
                message_type=record.message_type,
                timestamp=record.tick,
                is_mention=(record.target_id == self.unique_id),
                topic_id=record.topic_id,
                distortion_level=record.distortion_level,
                original_content=record.content,
                negative_score=record.negative_score,
                heat=record.heat,
            )
            recent_messages.append(si)
            if si.is_mention:
                mentions.append(si)

        topic_heat     = self.model.get_topic_heat()
        topic_negative = self.model.get_topic_negative()

        perception = Perception(
            group_id=group_id,
            group_type=group_type,
            beta=beta,
            recent_messages=recent_messages,
            mentions=mentions,
            tick=tick,
            topic_heat=topic_heat,
            topic_negative=topic_negative,
        )
        self._last_perception = perception
        return perception

    # ------------------------------------------------------------------ #
    #  #5  _retrieve_memory                                                #
    # ------------------------------------------------------------------ #
    def _retrieve_memory(self, perception: Perception) -> List[MemoryRecord]:
        """从短期记忆缓冲区检索最近 5 条。"""
        return list(self.memory)[-5:]

    # ------------------------------------------------------------------ #
    #  #6  _update_beliefs  （S1 规则存根；S2 替换为 LLM）                  #
    # ------------------------------------------------------------------ #
    def _update_beliefs(
        self,
        perception: Perception,
        memories: List[MemoryRecord],
    ) -> None:
        """
        有界置信度模型（Deffuant-Weisbuch）存根。
        LLM 失败或 S1 阶段均使用此规则。
        opinion_value ∈ [-1, 1]，emotion clip 处理。
        """
        agent_type = self.beliefs.identity.agent_type
        eps = _CONFIDENCE_BOUND[agent_type]
        mu  = _LEARNING_RATE[agent_type]

        topic_id = self._get_primary_topic_id()
        if topic_id not in self.beliefs.opinions:
            self.beliefs.opinions[topic_id] = OpinionBelief(topic_id=topic_id)

        current_op = self.beliefs.opinions[topic_id].opinion_value

        # RATIONAL 角色：对高 distortion_level 消息打折
        def _effective_opinion(si: SocialInfo) -> float:
            if agent_type == AgentType.RATIONAL and si.distortion_level > 0.5:
                return si.negative_score * -1.0  # 质疑负面内容
            # 用 negative_score 辅助估算邻居倾向（负面→负面立场）
            return -si.negative_score if si.negative_score > 0.6 else 0.0

        # 从 recent_messages 取同话题消息的有效观点
        neighbor_ops = []
        for si in perception.recent_messages:
            if si.topic_id == topic_id:
                if abs(_effective_opinion(si) - current_op) < eps:
                    neighbor_ops.append(_effective_opinion(si))

        if neighbor_ops:
            target  = float(np.mean(neighbor_ops))
            new_op  = float(np.clip(current_op + mu * (target - current_op), -1.0, 1.0))
            consistency = (1.0 - float(np.std(neighbor_ops))
                           if len(neighbor_ops) > 1 else 0.5)
            new_conf = float(np.clip(
                self.beliefs.opinions[topic_id].confidence * 0.9 + consistency * 0.1,
                0.0, 1.0,
            ))
            self.beliefs.opinions[topic_id].opinion_value = new_op
            self.beliefs.opinions[topic_id].confidence    = new_conf

        # 把本次感知写入记忆
        for si in perception.recent_messages[:3]:
            self.memory.append(MemoryRecord(
                tick=perception.tick,
                info=si,
                relevance=float(1.0 - abs(si.negative_score - 0.5)),
            ))

    # ------------------------------------------------------------------ #
    #  #7  _infer_desires                                                  #
    # ------------------------------------------------------------------ #
    def _infer_desires(self) -> List[Desire]:
        """
        基于信念生成欲望列表（v2：goal_type = reply/discuss/share/clarify/intervene/silent）。
        CONTROLLER 且 group_type ≠ DORM 时可生成 intervene 欲望。
        DORM 群 Controller 不生成 intervene（t_dorm^int = +∞）。
        热度 H(t) ≥ θ 时 Controller 触发干预欲望。
        """
        e          = self.beliefs.emotion
        agent_type = self.beliefs.identity.agent_type
        group_type = self.beliefs.identity.group_type
        topic_id   = self._get_primary_topic_id()
        desires: List[Desire] = []

        # 当前话题热度（从感知缓存读取）
        current_heat = 0.0
        if self._last_perception:
            current_heat = self._last_perception.topic_heat.get(topic_id, 0.0)

        # CONTROLLER：H(t) ≥ θ 且非 DORM → 干预欲望
        if (agent_type == AgentType.CONTROLLER
                and group_type != GroupType.DORM
                and current_heat >= THETA):
            desires.append(Desire("intervene", priority=0.9, topic_id=topic_id))

        # RATIONAL：高失真内容存在 → 澄清欲望
        if agent_type == AgentType.RATIONAL and self._last_perception:
            avg_distortion = float(np.mean([
                si.distortion_level for si in self._last_perception.recent_messages
            ])) if self._last_perception.recent_messages else 0.0
            if avg_distortion > 0.5:
                desires.append(Desire("clarify", priority=0.75, topic_id=topic_id))

        # 高唤醒情绪欲望
        if e.arousal > 0.65:
            if e.valence < -0.2:
                desires.append(Desire("discuss", priority=0.7, topic_id=topic_id))
            else:
                desires.append(Desire("share", priority=0.65, topic_id=topic_id))

        # ACTIVE：偏好转发
        if agent_type == AgentType.ACTIVE:
            desires.append(Desire("share", priority=0.6, topic_id=topic_id))

        # 背景欲望（所有角色）
        desires.append(Desire("discuss",  priority=0.35, topic_id=topic_id))
        desires.append(Desire("reply",    priority=0.25, topic_id=topic_id))

        desires.sort(key=lambda d: d.priority, reverse=True)
        return desires

    # ------------------------------------------------------------------ #
    #  #8  _plan_intentions                                                #
    # ------------------------------------------------------------------ #
    def _plan_intentions(self, desires: List[Desire]) -> Intention:
        """
        将欲望转化为具体行动意图（v2：ActionType 已更新）。
        goal_type → ActionType 映射：
          reply→REPLY, discuss→SEND_MESSAGE, share→FORWARD,
          clarify→SEND_MESSAGE, intervene→SEND_MESSAGE(message_type=clarification),
          silent→SILENT
        """
        if not desires:
            return Intention(action_type=ActionType.SILENT, topic_id=self._get_primary_topic_id())

        primary  = desires[0]
        extra    = self.beliefs.psychology.personality.extraversion
        arousal  = self.beliefs.emotion.arousal
        risk_av  = self.beliefs.psychology.risk_aversion

        # 是否行动
        p_act = extra * arousal * (1.0 - risk_av * 0.5)
        if self.model.random.random() > p_act:
            return Intention(action_type=ActionType.SILENT, topic_id=primary.topic_id)

        action_map = {
            "reply":    ActionType.REPLY,
            "discuss":  ActionType.SEND_MESSAGE,
            "share":    ActionType.FORWARD,
            "clarify":  ActionType.SEND_MESSAGE,
            "intervene":ActionType.SEND_MESSAGE,
        }
        action_type = action_map.get(primary.goal_type, ActionType.SILENT)

        op_val  = self._get_primary_opinion_value()
        stance  = "支持" if op_val > 0.1 else ("反对" if op_val < -0.1 else "观望")
        role    = self.beliefs.identity.agent_type.name
        group   = self.beliefs.identity.group_type.name
        content = (
            f"[{group}/{role}] Agent-{self.unique_id}({self.beliefs.identity.nickname}) "
            f"对话题{primary.topic_id}表示{stance}"
        )

        return Intention(
            action_type=action_type,
            content_plan=content,
            topic_id=primary.topic_id,
            target_id=primary.target_id,
        )

    # ------------------------------------------------------------------ #
    #  #9  _execute_action                                                 #
    # ------------------------------------------------------------------ #
    def _execute_action(self, intention: Intention) -> Dict[str, float]:
        """
        执行意图，构造 v2 ActionRecord（新增 distortion_level/message_type/negative_score/heat）。
        SILENT：不写 memory，不调 submit_action。
        CONTROLLER 且非 DORM：写 self.intervention_tick（参考值；权威由 A 维护）。
        """
        self.pending_action = None

        if intention.action_type == ActionType.SILENT:
            return {}

        tick       = int(self.model.schedule.time)
        agent_type = self.beliefs.identity.agent_type
        group_type = self.beliefs.identity.group_type
        op_val     = self._get_primary_opinion_value()

        # 确定 message_type
        if intention.action_type == ActionType.FORWARD:
            if agent_type == AgentType.ACTIVE:
                msg_type = MessageType.FORWARD
            else:
                msg_type = MessageType.PARAPHRASE
        elif agent_type == AgentType.CONTROLLER:
            msg_type = MessageType.CLARIFICATION
        elif agent_type == AgentType.RATIONAL:
            msg_type = MessageType.ORIGINAL
        else:
            # ORDINARY / ACTIVE 发原创时，根据情绪决定是否夸大
            if self.beliefs.emotion.arousal > 0.75 and self.beliefs.emotion.valence < -0.3:
                msg_type = MessageType.EXAGGERATE
            else:
                msg_type = MessageType.ORIGINAL

        # 计算 distortion_level
        distortion_map = {
            MessageType.ORIGINAL:      0.0,
            MessageType.FORWARD:       0.05,
            MessageType.PARAPHRASE:    0.25,
            MessageType.EXAGGERATE:    0.70,
            MessageType.CLARIFICATION: 0.0,
        }
        distortion_level = float(distortion_map.get(msg_type, 0.0))
        # RATIONAL 角色总是低失真
        if agent_type == AgentType.RATIONAL:
            distortion_level = min(distortion_level, 0.1)

        # negative_score：与情绪效价负相关
        negative_score = float(np.clip((1.0 - self.beliefs.emotion.valence) / 2.0, 0.0, 1.0))
        if msg_type == MessageType.CLARIFICATION:
            negative_score = max(0.0, negative_score - 0.3)

        # heat 贡献：基于 extraversion 和当前热度
        topic_heat_now = 0.0
        if self._last_perception:
            topic_heat_now = self._last_perception.topic_heat.get(intention.topic_id, 0.0)
        heat = float(np.clip(
            self.beliefs.psychology.personality.extraversion * (1 + topic_heat_now * 0.1),
            0.0, 2.0,
        ))

        record = ActionRecord(
            agent_id=self.unique_id,
            action_type=intention.action_type,
            content=intention.content_plan,
            target_id=intention.target_id,
            tick=tick,
            topic_id=intention.topic_id,
            distortion_level=distortion_level,
            message_type=msg_type,
            negative_score=negative_score,
            heat=heat,
        )

        # 写 memory
        self.memory.append(MemoryRecord(
            tick=tick,
            info=SocialInfo(
                source_id=self.unique_id,
                source_nickname=self.beliefs.identity.nickname,
                content=record.content,
                message_type=msg_type,
                timestamp=tick,
                topic_id=intention.topic_id,
                distortion_level=distortion_level,
                negative_score=negative_score,
                heat=heat,
            ),
            relevance=1.0,
        ))

        self.pending_action = record
        self.model.submit_action(record)

        # CONTROLLER 非 DORM：记录参考性 intervention_tick
        if (agent_type == AgentType.CONTROLLER
                and group_type != GroupType.DORM
                and self.intervention_tick is None):
            self.intervention_tick = tick

        return {
            "submitted":      1.0,
            "opinion_value":  op_val,
            "negative_score": negative_score,
            "heat":           heat,
        }

    # ------------------------------------------------------------------ #
    #  #10  _update_emotion                                                #
    # ------------------------------------------------------------------ #
    def _update_emotion(self, env_feedback: Dict[str, float]) -> None:
        """
        据环境反馈更新 arousal / valence。
        情绪衰减独立于 βₖ（纯自然心理恢复，emotion_decay=0.05）。
        """
        e             = self.beliefs.emotion
        emotion_decay = 0.05

        if env_feedback.get("submitted"):
            op_val        = env_feedback.get("opinion_value", 0.0)
            negative_score = env_feedback.get("negative_score", 0.0)
            # 发消息后情绪向观点方向强化，负面内容强化负效价
            valence_delta = 0.15 * op_val - 0.1 * negative_score
            e.valence = float(np.clip(e.valence + valence_delta, -1.0, 1.0))
            # 发帖后唤醒度稍降（宣泄）
            e.arousal = float(np.clip(e.arousal - 0.05, 0.0, 1.0))
        else:
            # 无提交：以 emotion_decay=0.05 向基线（valence=0, arousal=0.5）指数回归
            e.valence = float(np.clip(
                e.valence + emotion_decay * (0.0 - e.valence), -1.0, 1.0
            ))
            e.arousal = float(np.clip(
                e.arousal + emotion_decay * (0.5 - e.arousal), 0.0, 1.0
            ))

    # ------------------------------------------------------------------ #
    #  #11  calc_heat_decay【新增，接口表 v2】                              #
    # ------------------------------------------------------------------ #
    def calc_heat_decay(
        self,
        current_heat: float,
        elapsed_steps: int,
        intervention_tick: Optional[int] = None,
    ) -> float:
        """
        热度衰减纯计算函数（无副作用），供 A 模块 _update_environment 调用。

        公式：H_k(t+1) = H_k(t) · e^(-α) · e^(-β_k · 𝟙[t ≥ t_k^int])

        Parameters
        ----------
        current_heat       : 当前热度 H_k(t)
        elapsed_steps      : 已经过的步数 t
        intervention_tick  : 该群首次干预时刻 t_k^int（None 表示未触发）

        Returns
        -------
        float ≥ 0.0
        """
        group_type = self.beliefs.identity.group_type
        natural_decay = math.exp(-self.ALPHA)

        if group_type == GroupType.DORM:
            # DORM：t_dorm^int = +∞，干预衰减固定为 1.0
            intervention_decay = 1.0
        elif intervention_tick is not None and elapsed_steps >= intervention_tick:
            intervention_decay = math.exp(-self.beta)
        else:
            intervention_decay = 1.0

        return max(0.0, current_heat * natural_decay * intervention_decay)

    # ------------------------------------------------------------------ #
    #  内部辅助                                                            #
    # ------------------------------------------------------------------ #
    def _get_primary_opinion_value(self) -> float:
        """返回第一个话题的观点值，默认 0.0。"""
        if self.beliefs.opinions:
            return list(self.beliefs.opinions.values())[0].opinion_value
        return 0.0

    def _get_primary_topic_id(self) -> str:
        """返回第一个话题 ID，默认 'T001'。"""
        if self.beliefs.opinions:
            return list(self.beliefs.opinions.keys())[0]
        return "T001"
