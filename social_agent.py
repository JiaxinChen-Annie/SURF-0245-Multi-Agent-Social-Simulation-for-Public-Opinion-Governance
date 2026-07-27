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

【W4 LLM 接入变更】
  - _update_beliefs 改为双路径：优先走 LLM（C 模块），失败时自动降级规则存根
  - LLM 路径：从 self.model._llm_client 取客户端，调用 C 模块
    build_prompt / chat / parse_llm_response，将返回的
    opinion_updates 和 emotion_delta 写入 self.beliefs
  - 降级路径：原 Deffuant-Weisbuch 有界置信度规则，逻辑不变
  - _llm_client 为 None（config.llm_config 为空）时仅走规则存根

【转发行为修复】
  - 所有角色按差异化概率生成 share，不再只允许 ACTIVE 转发
  - share 必须绑定真实 source_message_id/source_group_id 与 destination_group_id
  - target_id 仅保留源消息作者语义，不再承担群路由
  - LLM 返回的 action_type/target/消息与群路由直接转换为 Intention 并优先执行
  - 转发在目标群创建新 ActionRecord，供目标群成员下一 tick 感知

真实 LLM 与规则降级共用同一套 Intention/ActionRecord 路由校验。
"""

from __future__ import annotations
import mesa_patch  # Mesa 3.x 兼容补丁

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

# 各角色在“看见可转发消息”后的基础转发概率。
# ACTIVE 仍最积极，但其他角色也不再被硬性排除。
_SHARE_PROBABILITY: Dict[AgentType, float] = {
    AgentType.ORDINARY:   0.05,
    AgentType.ACTIVE:     0.30,
    AgentType.RATIONAL:   0.10,
    AgentType.CONTROLLER: 0.05,
}

_SHARE_PRIORITY: Dict[AgentType, float] = {
    AgentType.ORDINARY:   0.55,
    AgentType.ACTIVE:     0.75,
    AgentType.RATIONAL:   0.60,
    AgentType.CONTROLLER: 0.50,
}

_MEMORY_CAPACITY = 20   # 短期记忆最大条数

# ─── C 模块懒加载（避免无 llm_utils 时整体崩溃）───────────────────────────── #
_llm_utils_module = None

def _get_llm_utils():
    """
    懒加载 C 模块 llm_utils。
    首次调用时尝试 import，成功后缓存；失败则返回 None，后续不重试。
    """
    global _llm_utils_module
    if _llm_utils_module is None:
        try:
            import llm_utils as _m
            _llm_utils_module = _m
        except ImportError:
            _llm_utils_module = False   # 标记为"已尝试但不可用"
    return _llm_utils_module if _llm_utils_module else None


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
        # Mesa 3.x: Agent.__init__(self, model)，unique_id 已移除
        # Mesa 2.x / mesa_compat: Agent.__init__(self, unique_id, model)
        try:
            super().__init__(unique_id, model)
        except TypeError:
            super().__init__(model)
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
        primary_group_id = str(
            init_config.get("primary_group_id", f"GROUP_{group_type.name}")
        )
        group_ids = list(init_config.get("group_ids", [primary_group_id]))
        if primary_group_id not in group_ids:
            group_ids.insert(0, primary_group_id)

        # 初始化信念系统
        self.beliefs = BeliefSystem(
            identity=IdentityBelief(
                agent_type=agent_type,
                group_type=group_type,          # 兼容字段：主群类型
                primary_group_id=primary_group_id,
                group_ids=group_ids,
                nickname=nickname,
                role_desc=(
                    f"{agent_type.name}-{group_type.name}-{unique_id}"
                    f" memberships={','.join(group_ids)}"
                ),
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
        self._llm_intention: Optional[Intention] = None

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
        self._llm_intention = None
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
            if (
                self._llm_intention is not None
                and bool(getattr(self.model, "llm_action_enabled", True))
            ):
                # LLM 的 action_type / target / 路由字段直接驱动本轮执行。
                intention = self._llm_intention
            else:
                desires = self._infer_desires()
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
        group_ids  = self.model.get_agent_groups(self.unique_id) or [group_id]
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
                message_id=record.message_id,
                group_id=record.group_id,
                source_message_id=record.source_message_id,
                source_group_id=record.source_group_id,
            )
            recent_messages.append(si)
            if si.is_mention:
                mentions.append(si)

        topic_heat     = self.model.get_topic_heat()
        topic_negative = self.model.get_topic_negative()
        topic_heat_by_group = self.model.get_topic_heat_by_group(self.unique_id)
        topic_negative_by_group = self.model.get_topic_negative_by_group(self.unique_id)

        perception = Perception(
            group_id=group_id,
            group_ids=group_ids,
            group_type=group_type,
            beta=beta,
            recent_messages=recent_messages,
            mentions=mentions,
            tick=tick,
            topic_heat=topic_heat,
            topic_negative=topic_negative,
            topic_heat_by_group=topic_heat_by_group,
            topic_negative_by_group=topic_negative_by_group,
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
    #  #6  _update_beliefs  （W4：LLM 优先，规则存根降级）                  #
    # ------------------------------------------------------------------ #
    def _update_beliefs(
        self,
        perception: Perception,
        memories: List[MemoryRecord],
    ) -> None:
        """
        W4 双路径信念更新：
          1) LLM 路径（优先）：调用 C 模块工具链，将 opinion_updates /
             emotion_delta 写入 self.beliefs。
          2) 规则降级路径：有界置信度模型（Deffuant-Weisbuch），
             LLM 不可用或调用失败时自动切换。

        LLM 路径触发条件（同时满足）：
          - C 模块 llm_utils 可以 import
          - self.model._llm_client 不为 None（A 模块在 __init__ 中挂载）

        opinion_value ∈ [-1, 1]，emotion clip 处理。
        """

        # ══════════════════════════════════════════════════════════════ #
        #  路径 1：LLM（C 模块 llm_utils）                               #
        # ══════════════════════════════════════════════════════════════ #
        llm_mod    = _get_llm_utils()
        llm_client = getattr(self.model, "_llm_client", None)

        if llm_mod is not None and llm_client is not None:
            try:
                # ── 构造 env_info（对齐 C 模块 build_prompt 期望的字段）──
                env_info = {
                    "group_type": self.beliefs.identity.group_type.name,
                    "primary_group_id": self.beliefs.identity.primary_group_id,
                    "group_ids": list(self.beliefs.identity.group_ids),
                    "beta":       self.beta,
                    "group_beta": {
                        group_id: GROUP_BETA[self.model.get_group_type_by_id(group_id)]
                        for group_id in self.beliefs.identity.group_ids
                        if self.model.get_group_type_by_id(group_id) is not None
                    },
                    "role":       self.beliefs.identity.agent_type.name,
                    "nickname":   self.beliefs.identity.nickname,
                    # 给 LLM 稳定的消息 ID 与群路由字段，禁止凭空造来源。
                    "recent_messages": [
                        {
                            "message_id": si.message_id,
                            "group_id": si.group_id,
                            "source_id": si.source_id,
                            "source_nickname": si.source_nickname,
                            "message_type": (
                                si.message_type.name
                                if hasattr(si.message_type, "name")
                                else si.message_type
                            ),
                            "content": si.content,
                            "topic_id": si.topic_id,
                            "distortion_level": si.distortion_level,
                            "negative_score": si.negative_score,
                        }
                        for si in perception.recent_messages[:5]
                    ],
                    "mentions": [
                        {
                            "message_id": si.message_id,
                            "group_id": si.group_id,
                            "source_id": si.source_id,
                            "content": si.content,
                        }
                        for si in perception.mentions[:3]
                    ],
                    "topic_heat":     perception.topic_heat,
                    "topic_negative": perception.topic_negative,
                    "topic_heat_by_group": perception.topic_heat_by_group,
                    "topic_negative_by_group": perception.topic_negative_by_group,
                }

                # ── 调用 C 模块三个接口 ────────────────────────────────
                prompt = llm_mod.build_prompt(self.beliefs, list(memories), env_info)
                raw    = llm_client.chat(prompt)
                result = llm_mod.parse_llm_response(raw)

                # ── 写回 opinion_updates ───────────────────────────────
                for tid, delta in result.get("opinion_updates", {}).items():
                    if tid not in self.beliefs.opinions:
                        self.beliefs.opinions[tid] = OpinionBelief(topic_id=tid)
                    old_val = self.beliefs.opinions[tid].opinion_value
                    self.beliefs.opinions[tid].opinion_value = float(
                        np.clip(old_val + delta, -1.0, 1.0)
                    )

                # ── 写回 emotion_delta ────────────────────────────────
                ed = result.get("emotion_delta", {})
                self.beliefs.emotion.valence = float(np.clip(
                    self.beliefs.emotion.valence + ed.get("valence", 0.0),
                    -1.0, 1.0,
                ))
                self.beliefs.emotion.arousal = float(np.clip(
                    self.beliefs.emotion.arousal + ed.get("arousal", 0.0),
                    0.0, 1.0,
                ))

                # LLM 的行动字段不再丢弃：转换成可执行 Intention，step() 会优先执行。
                if bool(getattr(self.model, "llm_action_enabled", True)):
                    self._llm_intention = self._intention_from_llm_result(
                        result,
                        perception,
                    )

                # ── 写入记忆（与规则路径逻辑一致）───────────────────────
                for si in perception.recent_messages[:3]:
                    self.memory.append(MemoryRecord(
                        tick=perception.tick,
                        info=si,
                        relevance=float(1.0 - abs(si.negative_score - 0.5)),
                    ))

                _LOG.debug(
                    f"Agent-{self.unique_id} LLM 路径成功 | "
                    f"action={result.get('action_type_name', '?')} | "
                    f"opinion_updates={result.get('opinion_updates', {})}"
                )
                return   # LLM 信念与行动意图均已处理，跳过规则存根

            except Exception as e:
                # LLM 调用失败：打日志后自动降级，不向上抛出
                _LOG.debug(
                    f"Agent-{self.unique_id} LLM 路径失败，降级规则存根: {e}"
                )

        # ══════════════════════════════════════════════════════════════ #
        #  路径 2：规则降级（Deffuant-Weisbuch 有界置信度模型）            #
        #  LLM 不可用 / 调用失败 / llm_config 为空时执行此路径            #
        # ══════════════════════════════════════════════════════════════ #
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
        基于信念与多群感知生成欲望。

        share/reply/clarify 都绑定真实消息与真实群；share 还必须给出另一个
        Agent 已加入的 ``destination_group_id``，因此不会再生成 target=None
        或“只改计数、不投递消息”的伪跨群转发。
        """
        emotion = self.beliefs.emotion
        agent_type = self.beliefs.identity.agent_type
        topic_id = self._get_primary_topic_id()
        primary_group_id = self._get_primary_group_id()
        desires: List[Desire] = []

        hottest_group_id = primary_group_id
        hottest_heat = 0.0
        if self._last_perception is not None:
            for group_id, topic_map in self._last_perception.topic_heat_by_group.items():
                heat = float(topic_map.get(topic_id, 0.0))
                if heat > hottest_heat:
                    hottest_heat = heat
                    hottest_group_id = group_id

        # CONTROLLER 在自己加入的非 DORM 群中，对真正达到阈值的群执行干预。
        hottest_type = self.model.get_group_type_by_id(hottest_group_id)
        if (
            agent_type == AgentType.CONTROLLER
            and hottest_type is not None
            and hottest_type != GroupType.DORM
            and hottest_heat >= THETA
        ):
            desires.append(Desire(
                "intervene",
                priority=0.9,
                topic_id=topic_id,
                destination_group_id=hottest_group_id,
            ))

        # RATIONAL 对一条具体高失真消息在其所在群内澄清。
        if agent_type == AgentType.RATIONAL and self._last_perception:
            distorted = max(
                self._last_perception.recent_messages,
                key=lambda si: si.distortion_level,
                default=None,
            )
            if distorted is not None and distorted.distortion_level > 0.5:
                desires.append(Desire(
                    "clarify",
                    priority=0.75,
                    topic_id=distorted.topic_id or topic_id,
                    target_id=distorted.source_id,
                    source_message_id=distorted.message_id,
                    source_group_id=distorted.group_id,
                    destination_group_id=distorted.group_id,
                ))

        if emotion.arousal > 0.65 and emotion.valence < -0.2:
            desires.append(Desire(
                "discuss",
                priority=0.7,
                topic_id=topic_id,
                destination_group_id=hottest_group_id,
            ))

        # 真实跨群转发：源消息来自当前可见流，目标群是发送者的另一成员群。
        forward_source = self._select_forward_source()
        if forward_source is not None:
            destination_group_id = self.model.select_forward_destination(
                self.unique_id,
                forward_source.group_id,
            )
            share_priority: Optional[float] = None
            if destination_group_id is not None:
                if self.model.random.random() < _SHARE_PROBABILITY[agent_type]:
                    share_priority = _SHARE_PRIORITY[agent_type]
                if emotion.arousal > 0.65 and emotion.valence >= -0.2:
                    share_priority = max(share_priority or 0.0, 0.65)

            if share_priority is not None and destination_group_id is not None:
                desires.append(Desire(
                    "share",
                    priority=share_priority,
                    topic_id=forward_source.topic_id or topic_id,
                    target_id=forward_source.source_id,
                    source_message_id=forward_source.message_id,
                    source_group_id=forward_source.group_id,
                    destination_group_id=destination_group_id,
                ))

        # 回复也绑定具体消息，并留在消息所在群。
        reply_source = None
        if self._last_perception and self._last_perception.recent_messages:
            reply_source = self._last_perception.recent_messages[0]

        desires.append(Desire(
            "discuss",
            priority=0.35,
            topic_id=topic_id,
            destination_group_id=primary_group_id,
        ))
        if reply_source is not None:
            desires.append(Desire(
                "reply",
                priority=0.25,
                topic_id=reply_source.topic_id or topic_id,
                target_id=reply_source.source_id,
                source_message_id=reply_source.message_id,
                source_group_id=reply_source.group_id,
                destination_group_id=reply_source.group_id,
            ))

        desires.sort(key=lambda desire: desire.priority, reverse=True)
        return desires

    # ------------------------------------------------------------------ #
    #  #8  _plan_intentions                                                #
    # ------------------------------------------------------------------ #
    def _plan_intentions(self, desires: List[Desire]) -> Intention:
        """把规则欲望转换成带完整消息路由的可执行意图。"""
        if not desires:
            return Intention(
                action_type=ActionType.SILENT,
                topic_id=self._get_primary_topic_id(),
                destination_group_id=self._get_primary_group_id(),
            )

        primary = desires[0]
        extraversion = self.beliefs.psychology.personality.extraversion
        arousal = self.beliefs.emotion.arousal
        risk_aversion = self.beliefs.psychology.risk_aversion

        p_act = extraversion * arousal * (1.0 - risk_aversion * 0.5)
        if self.model.random.random() > p_act:
            return Intention(
                action_type=ActionType.SILENT,
                topic_id=primary.topic_id,
                destination_group_id=(
                    primary.destination_group_id or self._get_primary_group_id()
                ),
            )

        action_map = {
            "reply": ActionType.REPLY,
            "discuss": ActionType.SEND_MESSAGE,
            "share": ActionType.FORWARD,
            "clarify": ActionType.SEND_MESSAGE,
            "intervene": ActionType.SEND_MESSAGE,
        }
        action_type = action_map.get(primary.goal_type, ActionType.SILENT)
        destination_group_id = (
            primary.destination_group_id or self._get_primary_group_id()
        )
        destination_type = self.model.get_group_type_by_id(destination_group_id)
        group_label = destination_type.name if destination_type is not None else destination_group_id

        opinion_value = self._get_primary_opinion_value()
        stance = (
            "支持" if opinion_value > 0.1
            else "反对" if opinion_value < -0.1
            else "观望"
        )
        role = self.beliefs.identity.agent_type.name

        source_info = self._find_perceived_message(primary.source_message_id)
        if action_type == ActionType.FORWARD:
            source_label = (
                source_info.source_nickname
                if source_info is not None and source_info.source_nickname
                else f"Agent-{primary.target_id}"
            )
            source_text = (
                source_info.content.strip()
                if source_info is not None and source_info.content.strip()
                else f"关于话题{primary.topic_id}的消息"
            )
            if len(source_text) > 100:
                source_text = source_text[:97] + "..."
            content = (
                f"[{group_label}/{role}] {self.beliefs.identity.nickname} "
                f"从{primary.source_group_id}转发自{source_label}：{source_text}"
            )
        elif action_type == ActionType.REPLY and source_info is not None:
            content = (
                f"[{group_label}/{role}] {self.beliefs.identity.nickname} "
                f"回复{source_info.source_nickname or source_info.source_id}："
                f"对话题{primary.topic_id}表示{stance}"
            )
        else:
            content = (
                f"[{group_label}/{role}] Agent-{self.unique_id}"
                f"({self.beliefs.identity.nickname}) 对话题{primary.topic_id}表示{stance}"
            )

        return Intention(
            action_type=action_type,
            content_plan=content,
            topic_id=primary.topic_id,
            target_id=primary.target_id,
            source_message_id=primary.source_message_id,
            source_group_id=primary.source_group_id,
            destination_group_id=destination_group_id,
        )

    # ------------------------------------------------------------------ #
    #  #9  _execute_action                                                 #
    # ------------------------------------------------------------------ #
    def _execute_action(self, intention: Intention) -> Dict[str, float]:
        """执行带明确来源群与目标群的意图，并交由 A 模块校验投递。"""
        self.pending_action = None
        if intention.action_type == ActionType.SILENT:
            return {}

        tick = int(self.model.schedule.time)
        agent_type = self.beliefs.identity.agent_type
        destination_group_id = (
            intention.destination_group_id or self._get_primary_group_id()
        )
        destination_group_type = self.model.get_group_type_by_id(destination_group_id)
        if destination_group_type is None:
            _LOG.debug(
                "Agent-%s 目标群无效，行动降级 SILENT: %r",
                self.unique_id,
                destination_group_id,
            )
            return {}

        opinion_value = self._get_primary_opinion_value()

        requested_message_type = str(getattr(intention, "message_type", "") or "").lower()
        legal_message_types = {
            MessageType.ORIGINAL,
            MessageType.FORWARD,
            MessageType.PARAPHRASE,
            MessageType.EXAGGERATE,
            MessageType.CLARIFICATION,
        }
        if requested_message_type in legal_message_types:
            message_type = requested_message_type
        elif intention.action_type == ActionType.FORWARD:
            message_type = (
                MessageType.FORWARD
                if agent_type == AgentType.ACTIVE
                else MessageType.PARAPHRASE
            )
        elif agent_type == AgentType.CONTROLLER:
            message_type = MessageType.CLARIFICATION
        elif agent_type == AgentType.RATIONAL:
            message_type = MessageType.ORIGINAL
        elif (
            self.beliefs.emotion.arousal > 0.75
            and self.beliefs.emotion.valence < -0.3
        ):
            message_type = MessageType.EXAGGERATE
        else:
            message_type = MessageType.ORIGINAL

        # FORWARD 的内容语义只能是直接转发或转述；Controller 的非转发干预保持澄清。
        if intention.action_type == ActionType.FORWARD and message_type not in {
            MessageType.FORWARD,
            MessageType.PARAPHRASE,
        }:
            message_type = MessageType.FORWARD
        if (
            agent_type == AgentType.CONTROLLER
            and intention.action_type != ActionType.FORWARD
        ):
            message_type = MessageType.CLARIFICATION

        distortion_map = {
            MessageType.ORIGINAL: 0.0,
            MessageType.FORWARD: 0.05,
            MessageType.PARAPHRASE: 0.25,
            MessageType.EXAGGERATE: 0.70,
            MessageType.CLARIFICATION: 0.0,
        }
        distortion_level = float(distortion_map.get(message_type, 0.0))
        if agent_type == AgentType.RATIONAL:
            distortion_level = min(distortion_level, 0.1)

        negative_score = float(np.clip(
            (1.0 - self.beliefs.emotion.valence) / 2.0,
            0.0,
            1.0,
        ))
        if message_type == MessageType.CLARIFICATION:
            negative_score = max(0.0, negative_score - 0.3)

        topic_heat_now = 0.0
        if self._last_perception:
            topic_heat_now = self._last_perception.topic_heat_by_group.get(
                destination_group_id,
                {},
            ).get(intention.topic_id, 0.0)
        heat = float(np.clip(
            self.beliefs.psychology.personality.extraversion
            * (1 + topic_heat_now * 0.1),
            0.0,
            2.0,
        ))

        content = intention.content_plan.strip()
        if not content:
            content = (
                f"[{destination_group_type.name}/{agent_type.name}] "
                f"{self.beliefs.identity.nickname} 关于{intention.topic_id}的消息"
            )

        record = ActionRecord(
            agent_id=self.unique_id,
            action_type=intention.action_type,
            content=content,
            target_id=intention.target_id,
            tick=tick,
            topic_id=intention.topic_id,
            distortion_level=distortion_level,
            message_type=message_type,
            negative_score=negative_score,
            heat=heat,
            group_id=destination_group_id,
            source_message_id=intention.source_message_id,
            source_group_id=intention.source_group_id,
        )

        accepted = bool(self.model.submit_action(record))
        if not accepted:
            return {}

        self.pending_action = record
        self.memory.append(MemoryRecord(
            tick=tick,
            info=SocialInfo(
                source_id=self.unique_id,
                source_nickname=self.beliefs.identity.nickname,
                content=record.content,
                message_type=message_type,
                timestamp=tick,
                topic_id=intention.topic_id,
                distortion_level=distortion_level,
                negative_score=negative_score,
                heat=heat,
                message_id=record.message_id,
                group_id=record.group_id,
                source_message_id=record.source_message_id,
                source_group_id=record.source_group_id,
            ),
            relevance=1.0,
        ))

        if (
            agent_type == AgentType.CONTROLLER
            and destination_group_type != GroupType.DORM
            and self.intervention_tick is None
        ):
            self.intervention_tick = tick

        return {
            "submitted": 1.0,
            "opinion_value": opinion_value,
            "negative_score": negative_score,
            "heat": heat,
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
    def _select_forward_source(self) -> Optional[SocialInfo]:
        """选择一条可被投递到另一个成员群的真实可见消息。"""
        if self._last_perception is None:
            return None

        candidates = [
            social_info
            for social_info in self._last_perception.recent_messages
            if social_info.source_id != self.unique_id
            and bool(social_info.content.strip())
            and bool(social_info.message_id)
            and bool(social_info.group_id)
            and bool(self.model.get_forward_destination_candidates(
                self.unique_id,
                social_info.group_id,
            ))
        ]
        if not candidates:
            return None
        return self.model.random.choice(candidates[:5])

    def _find_perceived_message(self, message_id: Optional[str]) -> Optional[SocialInfo]:
        if not message_id or self._last_perception is None:
            return None
        return next(
            (
                social_info
                for social_info in self._last_perception.recent_messages
                if social_info.message_id == message_id
            ),
            None,
        )

    def _get_primary_group_id(self) -> str:
        identity = self.beliefs.identity
        return (
            getattr(identity, "primary_group_id", "")
            or self.model.get_agent_group(self.unique_id)
            or f"GROUP_{identity.group_type.name}"
        )

    def _intention_from_llm_result(
        self,
        result: Dict[str, Any],
        perception: Perception,
    ) -> Optional[Intention]:
        """校验 LLM 行动 JSON，并转换成可执行意图；无效幻觉路由返回 None。"""
        raw_action_name = result.get("action_type_name")
        if not raw_action_name:
            raw_action = result.get("action_type", ActionType.SILENT)
            raw_action_name = getattr(raw_action, "name", str(raw_action))
        action_name = str(raw_action_name).upper().split(".")[-1]
        action_map = {
            "SEND_MESSAGE": ActionType.SEND_MESSAGE,
            "REPLY": ActionType.REPLY,
            "FORWARD": ActionType.FORWARD,
            "SILENT": ActionType.SILENT,
        }
        action_type = action_map.get(action_name, ActionType.SILENT)
        topic_id = str(result.get("topic_id") or self._get_primary_topic_id())
        primary_group_id = self._get_primary_group_id()

        if action_type == ActionType.SILENT:
            return Intention(
                action_type=ActionType.SILENT,
                topic_id=topic_id,
                destination_group_id=primary_group_id,
            )

        member_group_ids = set(perception.group_ids or [primary_group_id])
        destination_group_id = str(result.get("destination_group_id") or "").strip()
        source_message_id_raw = result.get("source_message_id")
        source_message_id = (
            str(source_message_id_raw).strip()
            if source_message_id_raw not in (None, "")
            else None
        )
        target_id = result.get("target_id")
        try:
            target_id = int(target_id) if target_id is not None else None
        except (TypeError, ValueError):
            target_id = None

        source_info = self._find_perceived_message(source_message_id)
        if source_info is None and target_id is not None:
            source_info = next(
                (
                    social_info
                    for social_info in perception.recent_messages
                    if social_info.source_id == target_id
                ),
                None,
            )

        source_group_id = str(result.get("source_group_id") or "").strip()
        if source_info is not None:
            source_message_id = source_info.message_id
            source_group_id = source_info.group_id
            target_id = source_info.source_id
            if not result.get("topic_id"):
                topic_id = source_info.topic_id or topic_id

        if action_type == ActionType.FORWARD:
            if source_info is None:
                _LOG.debug(
                    "Agent-%s 丢弃 LLM FORWARD：source_message_id/target_id 未命中可见消息",
                    self.unique_id,
                )
                return None
            valid_destinations = self.model.get_forward_destination_candidates(
                self.unique_id,
                source_info.group_id,
            )
            if destination_group_id not in valid_destinations:
                destination_group_id = self.model.select_forward_destination(
                    self.unique_id,
                    source_info.group_id,
                ) or ""
            if not destination_group_id:
                return None
        elif action_type == ActionType.REPLY:
            if source_info is None:
                _LOG.debug(
                    "Agent-%s 丢弃 LLM REPLY：未命中可见源消息",
                    self.unique_id,
                )
                return None
            # 回复必须留在被回复消息所在群。
            destination_group_id = source_info.group_id
        elif destination_group_id not in member_group_ids:
            destination_group_id = primary_group_id

        content = str(result.get("content") or "").strip()
        if not content and action_type == ActionType.FORWARD and source_info is not None:
            content = (
                f"[{destination_group_id}] {self.beliefs.identity.nickname} "
                f"从{source_info.group_id}转发：{source_info.content}"
            )
        elif not content:
            content = (
                f"[{destination_group_id}] {self.beliefs.identity.nickname} "
                f"关于{topic_id}的消息"
            )

        return Intention(
            action_type=action_type,
            content_plan=content,
            topic_id=topic_id,
            target_id=target_id,
            source_message_id=source_message_id,
            source_group_id=source_group_id,
            destination_group_id=destination_group_id or primary_group_id,
            message_type=str(result.get("message_type") or ""),
        )

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
