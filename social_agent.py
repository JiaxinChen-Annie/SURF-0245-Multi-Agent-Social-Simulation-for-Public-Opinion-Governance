"""
social_agent.py — 智能体层（函数 1–10）
-----------------------------------------
负责人：B
本文件为 A 在 S1（W2-W3）阶段使用的「规则存根」：
  - 函数名、签名、入参类型与接口表完全一致
  - 观点更新使用有界置信度模型（Deffuant-Weisbuch），无 LLM 调用
  - S2 阶段 B 同学用真实 LLM 版本替换 _update_beliefs 即可，其余不变
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from mesa import Agent
except ImportError:
    from mesa_compat import Agent

from types_def import (
    ActionRecord, ActionType, AgentType, BeliefSystem, Desire, EmotionState,
    IdentityBelief, Intention, MemoryRecord, OpinionBelief, Perception,
    PsychologyBelief, Personality, SocialInfo,
)

_LOG = logging.getLogger("SocialAgent")

# ─── S1 阶段超参 ──────────────────────────────────────────────────────────── #
_CONFIDENCE_BOUND: Dict[AgentType, float] = {
    AgentType.PUBLIC:         0.50,   # 普通用户：中等宽容度
    AgentType.OPINION_LEADER: 0.70,   # 意见领袖：广泛接纳信息
    AgentType.MEDIA:          0.30,   # 媒体：立场较固定
    AgentType.OFFICIAL:       0.25,   # 官方：立场最固定
}
_LEARNING_RATE: Dict[AgentType, float] = {
    AgentType.PUBLIC:         0.12,
    AgentType.OPINION_LEADER: 0.06,   # 意见领袖本身更难被说服
    AgentType.MEDIA:          0.02,
    AgentType.OFFICIAL:       0.01,
}
_MEMORY_CAPACITY = 20     # 短期记忆最大条数


class SocialAgent(Agent):
    """
    单个社交智能体（S1 规则版）。

    四层信念：IdentityBelief / PsychologyBelief / OpinionBelief / EmotionState
    BDI 推理链：_perceive → _retrieve_memory → _update_beliefs
                → _infer_desires → _plan_intentions → _execute_action
                → _update_emotion
    """

    # ------------------------------------------------------------------ #
    #  函数 1  __init__                                                    #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        unique_id: int,
        model,                          # OpinionModel（避免循环导入，不标注类型）
        agent_type: AgentType,
        init_config: Dict[str, Any],
    ) -> None:
        super().__init__(unique_id, model)

        # 初始化信念系统
        self.beliefs = BeliefSystem(
            identity=IdentityBelief(
                agent_type=agent_type,
                role_desc=f"{agent_type.name}-{unique_id}",
                stance_prior=float(init_config.get("stance_prior", 0.0)),
            ),
            opinions={
                init_config.get("event_id", "E001"): OpinionBelief(
                    event_id=init_config.get("event_id", "E001"),
                    opinion_value=float(init_config.get("stance_prior", 0.0)),
                    confidence=model.random.uniform(0.3, 0.9),
                )
            },
        )

        # 短期记忆缓冲（固定容量 deque）
        self.memory: deque = deque(maxlen=_MEMORY_CAPACITY)

        # 待提交行动（由 _execute_action 写入，供 OpinionModel.step 检测）
        self.pending_action: Optional[ActionRecord] = None

        # 初始化心理特质
        self._init_psychology()

    # ------------------------------------------------------------------ #
    #  函数 2  _init_psychology                                            #
    # ------------------------------------------------------------------ #
    def _init_psychology(self) -> None:
        """
        用受 seed 控制的随机源生成五因素人格 + 风险规避系数。
        结果写入 self.beliefs.psychology。
        """
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
        # 初始情绪：根据 stance_prior 和 neuroticism 轻微偏转
        stance = self.beliefs.identity.stance_prior
        neur   = self.beliefs.psychology.personality.neuroticism
        self.beliefs.emotion = EmotionState(
            valence=float(np.clip(stance * 0.3, -1.0, 1.0)),
            arousal=float(np.clip(0.4 + neur * 0.3, 0.0, 1.0)),
        )

    # ------------------------------------------------------------------ #
    #  函数 3  step（Mesa 强制入口）                                        #
    # ------------------------------------------------------------------ #
    def step(self) -> None:
        """
        串联完整 BDI 推理链。子步异常须降级，不允许崩溃传播。
        """
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
            desires    = self._infer_desires()
            intention  = self._plan_intentions(desires)
            env_fb     = self._execute_action(intention)
            self._update_emotion(env_fb)
        except Exception as e:
            _LOG.debug(f"Agent-{self.unique_id} BDI 后半段异常: {e}")

    # ------------------------------------------------------------------ #
    #  函数 4  _perceive                                                   #
    # ------------------------------------------------------------------ #
    def _perceive(self) -> Perception:
        """
        读取社交信息流：邻居动态、话题榜单、@消息。
        只读环境，无副作用。
        """
        tick          = int(self.model.schedule.time)
        neighbor_ids  = set(self.model.grid.get_neighbors(self.pos))
        current_tick  = tick

        # 从信息流缓存中筛选邻居近期行动（最近 5 tick）
        neighbor_actions: List[SocialInfo] = []
        for record in self.model.info_stream_cache:
            if (record.agent_id in neighbor_ids
                    and current_tick - record.tick <= 5):
                neighbor_actions.append(SocialInfo(
                    source_id=record.agent_id,
                    action_type=record.action_type,
                    content=record.content,
                    event_id=record.event_id,
                    opinion_value=record.opinion_value,
                    timestamp=record.tick,
                ))

        return Perception(
            neighbor_actions=neighbor_actions,
            trending_topics=list(self.model.trending[:5]),   # Top-5 话题
            mentions=[],
            tick=tick,
        )

    # ------------------------------------------------------------------ #
    #  函数 5  _retrieve_memory                                            #
    # ------------------------------------------------------------------ #
    def _retrieve_memory(self, perception: Perception) -> List[MemoryRecord]:
        """
        从短期记忆缓冲区检索相关历史交互（返回最近 5 条）。
        """
        return list(self.memory)[-5:]

    # ------------------------------------------------------------------ #
    #  函数 6  _update_beliefs  ← S1：有界置信度规则；S2 替换为 LLM        #
    # ------------------------------------------------------------------ #
    def _update_beliefs(
        self,
        perception: Perception,
        memories: List[MemoryRecord],
    ) -> None:
        """
        有界置信度模型（Deffuant-Weisbuch）：
          - 只接受与自身观点差值 < ε 的邻居影响
          - 向相似邻居均值移动，学习率 μ 因智能体类型而异
          - 媒体/官方基本固守立场（μ 极小）

        LLM 失败时（S2 之后）应回落到本函数逻辑。
        """
        agent_type = self.beliefs.identity.agent_type
        eps = _CONFIDENCE_BOUND[agent_type]
        mu  = _LEARNING_RATE[agent_type]

        event_id = list(self.beliefs.opinions.keys())[0] if self.beliefs.opinions else "E001"
        if event_id not in self.beliefs.opinions:
            self.beliefs.opinions[event_id] = OpinionBelief(event_id=event_id)

        current_op = self.beliefs.opinions[event_id].opinion_value

        # 收集在置信边界内的邻居观点
        similar_opinions = [
            info.opinion_value
            for info in perception.neighbor_actions
            if abs(info.opinion_value - current_op) < eps
        ]

        if similar_opinions:
            target     = float(np.mean(similar_opinions))
            new_op     = current_op + mu * (target - current_op)
            new_op     = float(np.clip(new_op, -1.0, 1.0))
            # 置信度随邻居一致性升高
            consistency = 1.0 - float(np.std(similar_opinions)) if len(similar_opinions) > 1 else 0.5
            new_conf    = float(np.clip(
                self.beliefs.opinions[event_id].confidence * 0.9 + consistency * 0.1,
                0.0, 1.0
            ))
            self.beliefs.opinions[event_id].opinion_value = new_op
            self.beliefs.opinions[event_id].confidence    = new_conf

        # 把本次感知加入记忆
        for info in perception.neighbor_actions[:3]:   # 最多记 3 条
            self.memory.append(MemoryRecord(
                tick=perception.tick,
                info=info,
                relevance=float(1.0 - abs(info.opinion_value - current_op)),
            ))

    # ------------------------------------------------------------------ #
    #  函数 7  _infer_desires                                              #
    # ------------------------------------------------------------------ #
    def _infer_desires(self) -> List[Desire]:
        """
        基于最新信念推断欲望列表。规则：
          - 高唤醒 + 负效价 → 强烈宣泄（vent_emotion）
          - 高唤醒 + 正效价 → 说服他人（persuade）
          - 低唤醒 → 信息获取（get_info）
          - 任何时候都有寻求身份认同的背景欲望（seek_identity）
        """
        e       = self.beliefs.emotion
        op_val  = self._get_primary_opinion_value()
        desires: List[Desire] = []

        event_id = self._get_primary_event_id()

        if e.arousal > 0.65:
            if e.valence < -0.2:
                desires.append(Desire("vent_emotion", priority=0.8,  event_id=event_id))
            else:
                desires.append(Desire("persuade",     priority=0.7,  event_id=event_id))

        desires.append(Desire("seek_identity", priority=0.4, event_id=event_id))
        desires.append(Desire("get_info",       priority=0.3, event_id=event_id))

        # 按优先级排序
        desires.sort(key=lambda d: d.priority, reverse=True)
        return desires

    # ------------------------------------------------------------------ #
    #  函数 8  _plan_intentions                                            #
    # ------------------------------------------------------------------ #
    def _plan_intentions(self, desires: List[Desire]) -> Intention:
        """
        将欲望转化为具体行动意图。
        P(沉默) = 1 - extraversion × arousal，受风险规避系数调节。
        """
        if not desires:
            return Intention(action_type=ActionType.SILENT, event_id="E001")

        primary    = desires[0]
        extra      = self.beliefs.psychology.personality.extraversion
        arousal    = self.beliefs.emotion.arousal
        risk_av    = self.beliefs.psychology.risk_aversion

        # 是否行动
        p_act = extra * arousal * (1.0 - risk_av * 0.5)
        if self.model.random.random() > p_act:
            return Intention(action_type=ActionType.SILENT, event_id=primary.event_id)

        # 根据欲望选择行动类型
        action_map = {
            "vent_emotion":  ActionType.POST,
            "persuade":      ActionType.COMMENT,
            "seek_identity": self.model.random.choice([ActionType.REPOST, ActionType.LIKE]),
            "get_info":      ActionType.LIKE,
        }
        action_type = action_map.get(primary.goal_type, ActionType.SILENT)

        op_val    = self._get_primary_opinion_value()
        stance    = "支持" if op_val > 0.1 else ("反对" if op_val < -0.1 else "观望")
        content   = f"[S1-规则] Agent-{self.unique_id}({self.beliefs.identity.agent_type.name}) 对{primary.event_id}表示{stance}"

        return Intention(
            action_type=action_type,
            event_id=primary.event_id,
            content_plan=content,
            target_id=None,
        )

    # ------------------------------------------------------------------ #
    #  函数 9  _execute_action                                             #
    # ------------------------------------------------------------------ #
    def _execute_action(self, intention: Intention) -> Dict[str, float]:
        """
        执行意图：
          - SILENT → 不提交，不计数
          - 其余  → 构造 ActionRecord，调用 model.submit_action()，
                    触发 HawkesEngine.add_event()
        返回 env_feedback 字典供 _update_emotion 使用。
        """
        self.pending_action = None

        if intention.action_type == ActionType.SILENT:
            return {}

        tick   = int(self.model.schedule.time)
        op_val = self._get_primary_opinion_value()
        record = ActionRecord(
            agent_id=self.unique_id,
            action_type=intention.action_type,
            content=intention.content_plan,
            event_id=intention.event_id,
            opinion_value=op_val,
            target_id=intention.target_id,
            tick=tick,
        )

        self.pending_action = record

        # 写入环境信息流缓存（A 模块的 submit_action）
        self.model.submit_action(record)

        # 触发 Hawkes 计数（OpinionModel.step 统计后批量写入，此处返回标记）
        return {"submitted": 1.0, "opinion_value": op_val}

    # ------------------------------------------------------------------ #
    #  函数 10  _update_emotion                                            #
    # ------------------------------------------------------------------ #
    def _update_emotion(self, env_feedback: Dict[str, float]) -> None:
        """
        据环境反馈更新 arousal / valence（越界 clip；无反馈时向基线衰减）。
        """
        e     = self.beliefs.emotion
        alpha = 0.15   # 情绪更新步长

        if env_feedback.get("submitted"):
            op_val = env_feedback.get("opinion_value", 0.0)
            # 发帖后：情绪向观点方向稍微强化
            e.valence = float(np.clip(e.valence + alpha * op_val, -1.0, 1.0))
            # 发帖后唤醒度稍降（宣泄效果）
            e.arousal = float(np.clip(e.arousal - 0.05, 0.0, 1.0))
        else:
            # 无反馈：向基线衰减
            baseline_valence = self.beliefs.identity.stance_prior * 0.5
            baseline_arousal = 0.40
            e.valence = float(np.clip(
                e.valence + 0.05 * (baseline_valence - e.valence), -1.0, 1.0
            ))
            e.arousal = float(np.clip(
                e.arousal + 0.05 * (baseline_arousal - e.arousal), 0.0, 1.0
            ))

    # ------------------------------------------------------------------ #
    #  内部辅助                                                            #
    # ------------------------------------------------------------------ #
    def _get_primary_opinion_value(self) -> float:
        """返回第一个事件的观点值，默认 0.0。"""
        if self.beliefs.opinions:
            return list(self.beliefs.opinions.values())[0].opinion_value
        return 0.0

    def _get_primary_event_id(self) -> str:
        """返回第一个事件 ID，默认 'E001'。"""
        if self.beliefs.opinions:
            return list(self.beliefs.opinions.keys())[0]
        return "E001"
