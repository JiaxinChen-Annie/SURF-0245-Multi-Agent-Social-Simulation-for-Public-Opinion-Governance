"""
social_agent.py — 智能体层（接口表 #1–11）
-----------------------------------------
负责人：B
本文件是 A 在 W4/W5 阶段使用的「规则存根」，接口已对齐接口表 v3。

【v3 变更（A 模块为对齐《场景设定》而改动的部分，B 接手时请保留语义）】
  ⑤ 禁言：CONTROLLER 新增 mute / announce 两种欲望与意图，
     真正提交 ActionType.MUTE / ActionType.ANNOUNCE 给环境层执行；
     被禁言的 Agent 在该群内不再生成任何行动。
  ③④ 转发：
     - 新增群内转发（intra），是攒够 forward_count、打开跨群通道的唯一途径；
     - 转发目标由 model.select_forward_destination(..., forward_count=...) 决定，
       方向权重 upward / lateral(小群互转) / downward 由 A 模块配置。
  ⑥ 曝光：Perception 新增 exposure / group_exposure，转发意愿与曝光正相关。
  ⑦ 降温：calc_heat_decay 改写为「基础降温 × control_level」形式，
     与 types_def.heat_decay 数值完全一致。
  ⑧ 人格：新增 trust（信任度）与 confirmation_bias（确认偏误），
     不再把它们混在人格五因素 + stance 里：
       - trust 低      → 折价高 distortion 消息，转发意愿下降；
       - bias 高       → 只吸收与自身立场同向的信息，放大极化。

【W4 LLM 接入】
  - _update_beliefs 双路径：优先走 LLM（C 模块），失败自动降级规则存根；
  - LLM 返回的 action_type / target / 群路由直接转成 Intention 并优先执行。

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
    BeliefSystem, Desire, EmotionState, ForwardDirection,
    IdentityBelief, Intention, MemoryRecord, OpinionBelief, Perception,
    PsychologyBelief, Personality, SocialInfo,
    ALPHA, THETA, NEGATIVE_THETA,
    GROUP_BETA, GROUP_CONTROL_LEVEL, GROUP_EXPOSURE,
    heat_decay as _formula_heat_decay,
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

# v3：群内转发概率（问题③）。群内接力是「攒够 forward_count → 打开跨群通道」
# 的唯一途径，因此单独给一套更高的概率；ACTIVE 依旧最积极。
_INTRA_SHARE_PROBABILITY: Dict[AgentType, float] = {
    AgentType.ORDINARY:   0.12,
    AgentType.ACTIVE:     0.45,
    AgentType.RATIONAL:   0.15,
    AgentType.CONTROLLER: 0.04,
}

# v3：各角色的信任度 / 确认偏误取值区间（问题⑧）
_TRUST_RANGE: Dict[AgentType, tuple] = {
    AgentType.ORDINARY:   (0.45, 0.85),   # 普通群员最容易轻信
    AgentType.ACTIVE:     (0.40, 0.80),
    AgentType.RATIONAL:   (0.10, 0.40),   # 理性讨论者天然低信任、爱查证
    AgentType.CONTROLLER: (0.30, 0.60),
}
_BIAS_RANGE: Dict[AgentType, tuple] = {
    AgentType.ORDINARY:   (0.30, 0.70),
    AgentType.ACTIVE:     (0.40, 0.85),   # 活跃者更容易只看同温层
    AgentType.RATIONAL:   (0.05, 0.30),   # 理性讨论者确认偏误最低
    AgentType.CONTROLLER: (0.20, 0.50),
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

        # βₖ 与 control_level（问题⑦：二者由 α 联系，β_k = α·(control_level_k − 1)）
        self.beta: float = GROUP_BETA[group_type]
        self.control_level: float = GROUP_CONTROL_LEVEL[group_type]
        # 主群曝光度（问题⑥，《场景设定》§1 群类型表）
        self.exposure: float = GROUP_EXPOSURE[group_type]

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

        # v3：Controller 已在哪些群公告过（用于「先公告、后禁言」的升级阶梯）
        self._announced_groups: set = set()
        # v3：对每个群最近一次治理动作的 tick（冷却期）
        self._last_governance_tick: Dict[str, int] = {}

        self.memory: deque = deque(maxlen=_MEMORY_CAPACITY)
        self.pending_action: Optional[ActionRecord] = None
        self._last_perception: Optional[Perception] = None
        self._llm_intention: Optional[Intention] = None

        self._init_psychology()

    # ------------------------------------------------------------------ #
    #  #2  _init_psychology                                                #
    # ------------------------------------------------------------------ #
    def _init_psychology(self) -> None:
        """
        生成五因素人格 + 风险规避系数 + 【v3】信任度 / 确认偏误。

        问题⑧：trust 与 confirmation_bias 过去被隐含在「人格 + stance」里，
        现在单独建模，并按角色给出不同取值区间：
          RATIONAL   低 trust、低 bias  → 质疑来源、愿意接受反向证据
          ACTIVE     中 trust、高 bias  → 转得多，但只转同温层
          ORDINARY   高 trust、中 bias  → 容易被高失真消息带走
          CONTROLLER 中 trust、中低 bias
        """
        rng = self.model.random
        agent_type = self.beliefs.identity.agent_type
        trust_lo, trust_hi = _TRUST_RANGE[agent_type]
        bias_lo, bias_hi = _BIAS_RANGE[agent_type]
        self.beliefs.psychology = PsychologyBelief(
            personality=Personality(
                openness=rng.uniform(0.2, 0.8),
                conscientiousness=rng.uniform(0.2, 0.8),
                extraversion=rng.uniform(0.1, 0.9),
                agreeableness=rng.uniform(0.2, 0.8),
                neuroticism=rng.uniform(0.1, 0.7),
            ),
            risk_aversion=rng.uniform(0.1, 0.9),
            trust=rng.uniform(trust_lo, trust_hi),
            confirmation_bias=rng.uniform(bias_lo, bias_hi),
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
            # 环境情绪传染：与 LLM / 规则两条信念路径都正交，始终执行。
            self._absorb_emotional_contagion(perception)
        except Exception as e:
            _LOG.debug(f"Agent-{self.unique_id} 情绪传染异常: {e}")

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
                forward_count=record.forward_count,
                root_message_id=record.root_message_id,
            )
            recent_messages.append(si)
            if si.is_mention:
                mentions.append(si)

        topic_heat     = self.model.get_topic_heat()
        topic_negative = self.model.get_topic_negative()
        topic_heat_by_group = self.model.get_topic_heat_by_group(self.unique_id)
        topic_negative_by_group = self.model.get_topic_negative_by_group(self.unique_id)

        # ── 仿真时钟上下文（A 模块 tick→时段常识，问题 place/time_of_day）──
        clock_ctx = {}
        if hasattr(self.model, "get_clock_context"):
            try:
                clock_ctx = self.model.get_clock_context(group_type=group_type)
            except Exception:
                clock_ctx = {}

        perception = Perception(
            group_id=group_id,
            group_ids=group_ids,
            group_type=group_type,
            beta=beta,
            control_level=self.control_level,
            exposure=self.exposure,
            group_exposure={
                gid: self.model.get_group_exposure(gid) for gid in group_ids
            },
            recent_messages=recent_messages,
            mentions=mentions,
            tick=tick,
            topic_heat=topic_heat,
            topic_negative=topic_negative,
            topic_heat_by_group=topic_heat_by_group,
            topic_negative_by_group=topic_negative_by_group,
            intervened_groups=self.model.get_intervened_groups(self.unique_id),
            muted_groups=self.model.get_muted_groups(self.unique_id),
            # 时钟字段
            sim_time_seconds=int(clock_ctx.get("sim_time_seconds", 0)),
            wall_hour=float(clock_ctx.get("wall_hour", 9.0)),
            time_slot=str(clock_ctx.get("time_slot", "")),
            place=str(clock_ctx.get("place", "")),
            activity=str(clock_ctx.get("activity", "")),
            group_activity_multiplier=float(
                clock_ctx.get("group_activity_multiplier", 1.0)),
            topic_suitability=dict(clock_ctx.get("topic_suitability", {})),
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
                    # 仿真时钟 / 时段常识（place + time_of_day）
                    "time_slot":        perception.time_slot,
                    "wall_hour":        perception.wall_hour,
                    "place":            perception.place,
                    "activity":         perception.activity,
                    "group_activity":   perception.group_activity_multiplier,
                    "topic_suitability": perception.topic_suitability,
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
        psychology = self.beliefs.psychology
        trust = float(np.clip(psychology.trust, 0.0, 1.0))
        bias = float(np.clip(psychology.confirmation_bias, 0.0, 1.0))

        # 【问题⑧】信任度与确认偏误单独参与推理：
        #   - 置信区间 eps 随 trust 放大（信任的人接纳更宽的观点区间）；
        #   - 学习率 mu 随 trust 放大、随 bias 缩小（偏见强 → 几乎不被说服）。
        eps = float(np.clip(_CONFIDENCE_BOUND[agent_type] * (0.6 + 0.8 * trust), 0.05, 1.5))
        mu = float(np.clip(_LEARNING_RATE[agent_type] * (0.5 + trust) * (1.0 - 0.7 * bias),
                           0.0, 1.0))

        topic_id = self._get_primary_topic_id()
        if topic_id not in self.beliefs.opinions:
            self.beliefs.opinions[topic_id] = OpinionBelief(topic_id=topic_id)

        current_op = self.beliefs.opinions[topic_id].opinion_value

        def _effective_opinion(si: SocialInfo) -> float:
            """把一条消息折算成「它在表达什么立场」。"""
            if si.message_type == MessageType.CLARIFICATION:
                # 澄清 / 公告：把观点往中性拉
                return 0.0
            raw = -si.negative_score if si.negative_score > 0.6 else 0.0
            if agent_type == AgentType.RATIONAL and si.distortion_level > 0.5:
                raw = si.negative_score * -1.0      # 质疑高失真的负面内容
            # 低信任者对高失真消息整体打折（trust 越低折得越狠）
            credibility = 1.0 - si.distortion_level * (1.0 - trust)
            return raw * float(np.clip(credibility, 0.0, 1.0))

        # 从 recent_messages 取同话题消息的有效观点
        neighbor_ops: List[float] = []
        weights: List[float] = []
        for si in perception.recent_messages:
            if si.topic_id != topic_id:
                continue
            value = _effective_opinion(si)
            if abs(value - current_op) >= eps:
                continue
            # 确认偏误：与自身立场同向的信息权重更高，反向的被压低
            aligned = 1.0 if value * current_op >= 0 else -1.0
            weight = 1.0 + bias * aligned
            if weight <= 0.0:
                continue
            neighbor_ops.append(value)
            weights.append(weight)

        if neighbor_ops:
            target = float(np.average(neighbor_ops, weights=weights))
            new_op = float(np.clip(current_op + mu * (target - current_op), -1.0, 1.0))
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
    #  情绪传染（v3 新增，支撑 negative_emotion / emotional_contagion 指标）
    # ------------------------------------------------------------------ #
    def _absorb_emotional_contagion(self, perception: Perception) -> None:
        """
        把「群里正在流传什么」转化为个体情绪。

        没有这一步，agent 情绪只会向基线回归，群负面程度永远停在 0.5 附近，
        《场景设定》「负面程度超过阈值 → controller 干预」就永远触发不了。

        易感度由人格与信任度共同决定（问题⑧）：
            神经质高 → 更容易被带动；
            信任度高 → 更容易照单全收；
            失真高的消息对低信任者影响被打折。
        """
        messages = perception.recent_messages
        if not messages:
            return

        psychology = self.beliefs.psychology
        trust = float(np.clip(psychology.trust, 0.0, 1.0))
        neuroticism = float(np.clip(psychology.personality.neuroticism, 0.0, 1.0))

        w_sum = 0.0
        neg_sum = 0.0
        for si in messages:
            credibility = float(np.clip(1.0 - si.distortion_level * (1.0 - trust), 0.05, 1.0))
            weight = credibility * (1.0 + float(np.clip(si.heat, 0.0, 2.0)))
            if si.message_type == MessageType.CLARIFICATION:
                weight *= 1.5          # 澄清 / 公告的降温作用更明显
            w_sum += weight
            neg_sum += weight * float(np.clip(si.negative_score, 0.0, 1.0))
        if w_sum <= 1e-12:
            return

        perceived_negative = neg_sum / w_sum
        target_valence = 1.0 - 2.0 * perceived_negative          # neg=1 → -1, neg=0 → +1
        susceptibility = 0.10 + 0.20 * neuroticism + 0.10 * trust

        emotion = self.beliefs.emotion
        delta_valence = susceptibility * (target_valence - emotion.valence)
        if delta_valence < 0:
            delta_valence *= 1.20   # 负性偏向：坏消息比好消息更"传得动"
        emotion.valence = float(np.clip(emotion.valence + delta_valence, -1.0, 1.0))
        target_arousal = float(np.clip(0.35 + 0.65 * perceived_negative, 0.0, 1.0))
        emotion.arousal = float(np.clip(
            emotion.arousal + susceptibility * (target_arousal - emotion.arousal), 0.0, 1.0))

    # ------------------------------------------------------------------ #
    #  #7  _infer_desires                                                  #
    # ------------------------------------------------------------------ #
    def _infer_desires(self) -> List[Desire]:
        """
        基于信念与多群感知生成欲望。

        v3 关键改动：
          ⑤ CONTROLLER 在已触发干预的群里会真正生成 mute / announce 欲望，
            而不是只发一条澄清；
          ③ 转发欲望分成「群内接力（intra）」与「跨群（cross）」两条，
            跨群通道只有在源消息转发次数达阈时才由 A 模块开放；
          ⑥ 转发意愿与目标群曝光正相关；
          ⑧ 低 trust 的 agent 不愿意转发高失真消息。
        """
        emotion = self.beliefs.emotion
        psychology = self.beliefs.psychology
        agent_type = self.beliefs.identity.agent_type
        topic_id = self._get_primary_topic_id()
        primary_group_id = self._get_primary_group_id()
        perception = self._last_perception
        desires: List[Desire] = []

        muted_groups = set(perception.muted_groups) if perception else set()
        negative_threshold = float(getattr(self.model, "negative_threshold", NEGATIVE_THETA))
        heat_threshold = float(getattr(self.model, "heat_threshold", THETA))

        # ── 找出「风险最高」的群：热度或负面任一更突出 ────────────────
        hottest_group_id = primary_group_id
        hottest_heat = 0.0
        hottest_negative = 0.0
        if perception is not None:
            best_score = -1.0
            for group_id, topic_map in perception.topic_heat_by_group.items():
                if group_id in muted_groups:
                    continue
                heat = float(topic_map.get(topic_id, 0.0))
                negative = float(
                    perception.topic_negative_by_group.get(group_id, {}).get(topic_id, 0.0)
                )
                score = max(heat / max(heat_threshold, 1e-9),
                            negative / max(negative_threshold, 1e-9))
                if score > best_score:
                    best_score = score
                    hottest_heat = heat
                    hottest_negative = negative
                    hottest_group_id = group_id

        hottest_type = self.model.get_group_type_by_id(hottest_group_id)
        trigger_mode = getattr(self.model, "intervention_trigger", "negative")
        heat_hit = hottest_heat >= heat_threshold
        negative_hit = hottest_negative >= negative_threshold
        if trigger_mode == "heat":
            trigger_hit = heat_hit
        elif trigger_mode == "negative":
            trigger_hit = negative_hit
        elif trigger_mode == "heat_and_negative":
            trigger_hit = heat_hit and negative_hit
        else:
            trigger_hit = heat_hit or negative_hit

        # ── CONTROLLER：提醒 / 澄清 / 禁言 / 公告（问题⑤）─────────────
        #    只要该群仍处于「已触发干预」状态就持续治理（一次公告压不住是常态），
        #    但对同一个群有冷却期，避免 10 个 Controller 每 tick 刷屏。
        flagged = bool(perception and hottest_group_id in perception.intervened_groups)
        cooldown = int(getattr(self.model, "governance_cooldown_ticks", 3))
        tick_now = perception.tick if perception else 0
        cooled = (tick_now - self._last_governance_tick.get(hottest_group_id, -10 ** 9)) >= cooldown

        if (agent_type == AgentType.CONTROLLER
                and hottest_type is not None
                and hottest_type != GroupType.DORM        # DORM 群介入不了
                and hottest_group_id not in muted_groups
                and flagged            # 以 A 模块的 intervention_tick 为唯一权威
                and cooled):

            worst = self._select_mute_target(hottest_group_id)
            mute_enabled = bool(getattr(self.model, "enable_mute", True))
            mute_trigger = float(getattr(self.model, "mute_negative_trigger", 0.68))
            already_announced = hottest_group_id in self._announced_groups

            # 升级阶梯（《场景设定》§2「提醒、澄清、禁言、公告」）：
            #   第一次触发 → 公告 / 澄清；
            #   已经公告过但负面仍未压下去 → 禁言最恶劣的传播者。
            if mute_enabled and worst is not None and already_announced:
                desires.append(Desire(
                    "mute",
                    priority=0.98,
                    topic_id=topic_id,
                    target_id=worst.source_id,
                    source_message_id=worst.message_id,
                    source_group_id=worst.group_id,
                    destination_group_id=hottest_group_id,
                ))
            else:
                desires.append(Desire(
                    "announce" if hottest_negative >= negative_threshold else "intervene",
                    priority=0.92,
                    topic_id=topic_id,
                    destination_group_id=hottest_group_id,
                ))

        # ── RATIONAL：针对一条具体高失真消息澄清（低 trust 更敏感）────
        if agent_type == AgentType.RATIONAL and perception is not None:
            distorted = max(
                (si for si in perception.recent_messages
                 if si.group_id not in muted_groups),
                key=lambda si: si.distortion_level,
                default=None,
            )
            scrutiny_threshold = 0.30 + 0.40 * float(np.clip(psychology.trust, 0.0, 1.0))
            if distorted is not None and distorted.distortion_level > scrutiny_threshold:
                desires.append(Desire(
                    "clarify",
                    priority=0.75,
                    topic_id=distorted.topic_id or topic_id,
                    target_id=distorted.source_id,
                    source_message_id=distorted.message_id,
                    source_group_id=distorted.group_id,
                    destination_group_id=distorted.group_id,
                ))

        # 情绪触发讨论：晚间自由时段（time_activity≈1.0）更容易被情绪推动发言；
        # 上课时段（time_activity≈0.2）则需要更高的情绪唤醒才愿意"插嘴"
        arousal_threshold_discuss = float(np.clip(0.40 + 0.30 * (1.0 - time_activity), 0.40, 0.70))
        if (emotion.arousal > arousal_threshold_discuss
                and emotion.valence < -0.2
                and hottest_group_id not in muted_groups):
            desires.append(Desire(
                "discuss",
                priority=0.7,
                topic_id=topic_id,
                destination_group_id=hottest_group_id,
            ))

        # ── 转发（问题③④⑥⑧ + 时段活跃度）───────────────────────────
        # 时段活跃度乘数：上课时间（0.15–0.85）/ 晚间自由（1.00）
        time_activity = float(
            perception.group_activity_multiplier if perception else 1.0)

        forward_source = self._select_forward_source()
        if forward_source is not None:
            destination_group_id = self.model.select_forward_destination(
                self.unique_id,
                forward_source.group_id,
                forward_count=forward_source.forward_count,
            )
            if destination_group_id is not None and destination_group_id not in muted_groups:
                source_type = self.model.get_group_type_by_id(forward_source.group_id)
                dest_type = self.model.get_group_type_by_id(destination_group_id)
                is_intra = (source_type == dest_type)

                base_p = (_INTRA_SHARE_PROBABILITY[agent_type] if is_intra
                          else _SHARE_PROBABILITY[agent_type])
                # ⑥ 曝光：目标群曝光越高，越值得转过去
                exposure = self.model.get_group_exposure(destination_group_id)
                base_p *= (0.55 + 0.45 * exposure)
                # ⑧ 信任：低信任者不愿意扩散高失真消息
                trust = float(np.clip(psychology.trust, 0.0, 1.0))
                base_p *= float(np.clip(1.0 - forward_source.distortion_level * (1.0 - trust),
                                        0.05, 1.0))
                # 时段活跃度：上课期间转发概率随活跃度乘数下降
                base_p *= float(np.clip(time_activity, 0.05, 1.0))

                share_priority: Optional[float] = None
                if self.model.random.random() < base_p:
                    share_priority = _SHARE_PRIORITY[agent_type]
                if emotion.arousal > 0.65 and emotion.valence >= -0.2:
                    share_priority = max(share_priority or 0.0, 0.65)

                if share_priority is not None:
                    desires.append(Desire(
                        "share",
                        priority=share_priority * (0.85 if is_intra else 1.0),
                        topic_id=forward_source.topic_id or topic_id,
                        target_id=forward_source.source_id,
                        source_message_id=forward_source.message_id,
                        source_group_id=forward_source.group_id,
                        destination_group_id=destination_group_id,
                    ))

        # ── 兜底：讨论 / 回复 ────────────────────────────────────────
        fallback_group_id = primary_group_id
        if fallback_group_id in muted_groups:
            available = [gid for gid in (perception.group_ids if perception else [])
                         if gid not in muted_groups]
            fallback_group_id = available[0] if available else ""
        if fallback_group_id:
            desires.append(Desire(
                "discuss",
                priority=0.35,
                topic_id=topic_id,
                destination_group_id=fallback_group_id,
            ))

        reply_source = None
        if perception is not None:
            reply_source = next(
                (si for si in perception.recent_messages if si.group_id not in muted_groups),
                None,
            )
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

    def _select_mute_target(self, group_id: str) -> Optional[SocialInfo]:
        """
        挑选禁言对象（问题⑤）：该群中「失真 × 负面」最高、且不是 Controller
        自己的消息作者。返回该消息，None 表示没有值得禁言的对象。
        """
        if self._last_perception is None:
            return None
        candidates = [
            si for si in self._last_perception.recent_messages
            if si.group_id == group_id
            and si.source_id != self.unique_id
            and si.message_type != MessageType.CLARIFICATION
        ]
        if not candidates:
            return None
        worst = max(candidates,
                    key=lambda si: si.distortion_level * 0.5 + si.negative_score * 0.5)
        if worst.distortion_level * 0.5 + worst.negative_score * 0.5 < 0.35:
            return None
        return worst

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
        # 治理动作（禁言 / 公告 / 干预澄清）是职责，不受外向性与唤醒度的概率门限约束
        is_governance = primary.goal_type in {"mute", "announce", "intervene"}
        if not is_governance and self.model.random.random() > p_act:
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
            "intervene": ActionType.ANNOUNCE,       # 提醒 / 澄清（走公告通道）
            "mute": ActionType.MUTE,                # v3 禁言
            "announce": ActionType.ANNOUNCE,        # v3 公告
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
        if action_type == ActionType.MUTE:
            content = (
                f"[{group_label}/管理员] {self.beliefs.identity.nickname} 已对 "
                f"Agent-{primary.target_id} 执行禁言，并提醒群内理性发言、勿传未经证实信息"
            )
            return Intention(
                action_type=ActionType.MUTE,
                content_plan=content,
                topic_id=primary.topic_id,
                target_id=primary.target_id,
                source_message_id=primary.source_message_id,
                source_group_id=primary.source_group_id,
                destination_group_id=destination_group_id,
                message_type=MessageType.CLARIFICATION,
                mute_duration=int(getattr(self.model, "mute_duration_ticks", 3)),
            )
        if action_type == ActionType.ANNOUNCE:
            content = (
                f"[{group_label}/管理员] {self.beliefs.identity.nickname} 发布公告："
                f"关于{primary.topic_id}的情况正在核实，请以官方通报为准，勿信勿传"
            )
            return Intention(
                action_type=ActionType.ANNOUNCE,
                content_plan=content,
                topic_id=primary.topic_id,
                destination_group_id=destination_group_id,
                message_type=MessageType.CLARIFICATION,
            )
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

        # 【问题⑤】被禁言者在该群内无法发声
        if self.model.is_muted(self.unique_id, destination_group_id):
            _LOG.debug("Agent-%s 在 %s 被禁言，本轮沉默", self.unique_id, destination_group_id)
            return {}

        opinion_value = self._get_primary_opinion_value()

        # 【问题⑤】治理动作走专用分支：不产生热度，直接交环境层执行
        if intention.action_type in (ActionType.MUTE, ActionType.ANNOUNCE):
            record = ActionRecord(
                agent_id=self.unique_id,
                action_type=intention.action_type,
                content=intention.content_plan,
                target_id=intention.target_id,
                tick=tick,
                topic_id=intention.topic_id,
                distortion_level=0.0,
                message_type=MessageType.CLARIFICATION,
                negative_score=0.0,
                heat=0.0,
                group_id=destination_group_id,
                mute_duration=int(getattr(intention, "mute_duration", 0)),
            )
            if not self.model.submit_action(record):
                return {}
            self.pending_action = record
            self._last_governance_tick[destination_group_id] = tick
            self._announced_groups.add(destination_group_id)
            if self.intervention_tick is None:
                self.intervention_tick = tick
            return {
                "submitted": 1.0,
                "opinion_value": opinion_value,
                "negative_score": 0.0,
                "heat": 0.0,
                "governance": 1.0,
            }

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
            self.beliefs.emotion.arousal > 0.65
            and self.beliefs.emotion.valence < -0.15
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
        # 消息语义类型直接影响负面程度（《场景设定》§3 五类消息）：
        # 夸大 > 转述 > 直接转发 ≈ 原创 > 澄清
        _NEGATIVE_BY_MESSAGE_TYPE = {
            MessageType.EXAGGERATE:    +0.18,
            MessageType.PARAPHRASE:    +0.06,
            MessageType.FORWARD:       +0.02,
            MessageType.ORIGINAL:       0.00,
            MessageType.CLARIFICATION: -0.35,
        }
        negative_score = float(np.clip(
            negative_score + _NEGATIVE_BY_MESSAGE_TYPE.get(message_type, 0.0), 0.0, 1.0))

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

        if (agent_type == AgentType.CONTROLLER
                and destination_group_type != GroupType.DORM):
            pass
            if self.intervention_tick is None:
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

        【问题⑦：两处表述统一】
        《场景设定》§3 文字版：实际降温 = 基础降温 × control_level (>1)
        《场景设定》§4.2 公式版：H_k(t+1) = H_k(t)·e^(−α)·e^(−β_k·𝟙[t ≥ t_k^int])

        本实现直接采用文字版形式，并保证与公式版逐位等价：

            未干预：H(t+1) = H(t) · exp(−α)                    （control_level = 1）
            已干预：H(t+1) = H(t) · exp(−α · control_level_k)
                          ≡ H(t) · exp(−α) · exp(−β_k)
            因为 β_k = α · (control_level_k − 1)。

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
        # DORM：t_dorm^int = +∞，永远只有基础降温
        intervened = (
            group_type != GroupType.DORM
            and intervention_tick is not None
            and elapsed_steps >= intervention_tick
        )
        return _formula_heat_decay(current_heat, group_type, intervened)

    # ------------------------------------------------------------------ #
    #  内部辅助                                                            #
    # ------------------------------------------------------------------ #
    def _select_forward_source(self) -> Optional[SocialInfo]:
        """选择一条可被投递到另一个成员群的真实可见消息。"""
        if self._last_perception is None:
            return None

        muted = set(self._last_perception.muted_groups)
        candidates = [
            social_info
            for social_info in self._last_perception.recent_messages
            if social_info.source_id != self.unique_id
            and social_info.message_type != MessageType.CLARIFICATION
            and bool(social_info.content.strip())
            and bool(social_info.message_id)
            and bool(social_info.group_id)
            and social_info.group_id not in muted
            # 【问题③】把源消息的转发次数传给 A 模块：未达门槛时只剩群内通道，
            # 达到门槛后跨群通道才会出现在候选里。
            and bool(self.model.get_forward_destination_candidates(
                self.unique_id,
                social_info.group_id,
                forward_count=social_info.forward_count,
            ))
        ]
        if not candidates:
            return None
        # 在全部可见消息中抽样，而不是只看最新 5 条 —— 否则大群消息量压倒性
        # 领先，小群消息永远轮不到被转发，小群互转就永远是 0。
        return self.model.random.choice(candidates)

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
            "MUTE": ActionType.MUTE,          # v3：仅 CONTROLLER 合法
            "ANNOUNCE": ActionType.ANNOUNCE,  # v3：仅 CONTROLLER 合法
        }
        action_type = action_map.get(action_name, ActionType.SILENT)
        # 越权的治理动作直接降级为普通发言，交由 A 模块最终裁决
        if (action_type in (ActionType.MUTE, ActionType.ANNOUNCE)
                and self.beliefs.identity.agent_type != AgentType.CONTROLLER):
            _LOG.debug("Agent-%s 非 CONTROLLER，LLM 的 %s 动作降级为 SEND_MESSAGE",
                       self.unique_id, action_name)
            action_type = ActionType.SEND_MESSAGE
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
                forward_count=source_info.forward_count,
            )
            if destination_group_id not in valid_destinations:
                destination_group_id = self.model.select_forward_destination(
                    self.unique_id,
                    source_info.group_id,
                    forward_count=source_info.forward_count,
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

        mute_duration = 0
        if action_type == ActionType.MUTE:
            if target_id is None:
                _LOG.debug("Agent-%s 丢弃 LLM MUTE：未给出 target_id", self.unique_id)
                return None
            mute_duration = int(getattr(self.model, "mute_duration_ticks", 3))

        return Intention(
            action_type=action_type,
            content_plan=content,
            topic_id=topic_id,
            target_id=target_id,
            source_message_id=source_message_id,
            source_group_id=source_group_id,
            destination_group_id=destination_group_id or primary_group_id,
            message_type=str(result.get("message_type") or ""),
            mute_duration=mute_duration,
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
