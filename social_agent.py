"""
social_agent.py
B 模块：智能体大脑（SocialAgent）
包含 BDI 推理链，支持 LLM 驱动（Week 2 接入）
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
from enum import IntEnum
from dataclasses import dataclass, field
import logging
import json
import random

# 引入 Mesa 基础（假设已安装）
from mesa import Agent

# ======================== 类型定义（与接口表完全一致） ========================

class AgentType(IntEnum):
    PUBLIC = 0
    OPINION_LEADER = 1
    MEDIA = 2
    OFFICIAL = 3

class ActionType(IntEnum):
    POST = 0
    COMMENT = 1
    REPOST = 2
    LIKE = 3
    SILENT = 4

@dataclass
class Personality:
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

@dataclass
class EmotionState:
    valence: float = 0.0      # [-1, 1]
    arousal: float = 0.5      # [0, 1]

@dataclass
class IdentityBelief:
    agent_type: AgentType
    role_desc: str = ""
    stance_prior: float = 0.0   # [-1, 1]

@dataclass
class PsychologyBelief:
    personality: Personality = field(default_factory=Personality)
    risk_aversion: float = 0.5  # [0, 1]

@dataclass
class OpinionBelief:
    event_id: str
    opinion_value: float = 0.0  # [-1, 1]
    confidence: float = 0.5     # [0, 1]

@dataclass
class BeliefSystem:
    identity: IdentityBelief
    psychology: PsychologyBelief
    opinions: Dict[str, OpinionBelief] = field(default_factory=dict)
    emotion: EmotionState = field(default_factory=EmotionState)

@dataclass
class SocialInfo:
    source_id: int
    action_type: ActionType
    content: str
    event_id: str
    opinion_value: float
    timestamp: int  # tick

@dataclass
class Perception:
    neighbor_actions: List[SocialInfo] = field(default_factory=list)
    trending_topics: List[Tuple[str, float]] = field(default_factory=list)  # (topic,热度)
    mentions: List[SocialInfo] = field(default_factory=list)
    tick: int = 0

@dataclass
class MemoryRecord:
    tick: int
    info: SocialInfo
    relevance: float = 0.5

@dataclass
class Desire:
    goal_type: str  # seek_identity, vent_emotion, get_info, persuade
    priority: float
    event_id: str

@dataclass
class Intention:
    action_type: ActionType
    event_id: str
    content_plan: str
    target_id: Optional[int] = None

@dataclass
class ActionRecord:
    agent_id: int
    action_type: ActionType
    content: str
    event_id: str
    opinion_value: float
    target_id: Optional[int] = None
    tick: int = 0


# ======================== SocialAgent 类实现 ========================

class SocialAgent(Agent):
    """
    智能体大脑，BDI 推理全链路。
    依赖 C 模块提供的 LLM 工具（build_prompt, parse_llm_response, setup_llm_client）
    """

    def __init__(
        self,
        unique_id: int,
        model: Any,  # OpinionModel 实例（type hints 可后续导入）
        agent_type: AgentType,
        init_config: Dict[str, Any],
    ):
        self.unique_id = unique_id
        self.model = model
        self.agent_type = agent_type
        
        # 初始化信念系统
        self.beliefs = BeliefSystem(
            identity=IdentityBelief(
                agent_type=agent_type,
                role_desc=init_config.get("role_desc", ""),
                stance_prior=init_config.get("stance_prior", 0.0),
            ),
            psychology=PsychologyBelief(),
            opinions={},
            emotion=EmotionState(),
        )
        # 短期记忆缓冲区（FIFO，限制长度）
        self.memory: List[MemoryRecord] = []
        self.max_memory = init_config.get("max_memory", 20)

        # 待提交行动（由 _execute_action 填充，step 结束后清空）
        self.pending_action: Optional[ActionRecord] = None

        # 调用心理学初始化
        self._init_psychology()

        # 如果有初始事件观点，可以添加
        for eid, val in init_config.get("initial_opinions", {}).items():
            self.beliefs.opinions[eid] = OpinionBelief(event_id=eid, opinion_value=val)

        # 日志
        self.logger = logging.getLogger(f"Agent_{unique_id}")

    # ---------- 函数 2：心理学初始化 ----------
    def _init_psychology(self) -> None:
        """随机生成五因素人格与风险规避系数（使用模型统一的随机源）"""
        rng = self.model.random  # Mesa 的 Random 实例，受 seed 控制
        personality = Personality(
            openness=rng.random(),
            conscientiousness=rng.random(),
            extraversion=rng.random(),
            agreeableness=rng.random(),
            neuroticism=rng.random(),
        )
        risk_aversion = rng.random()
        self.beliefs.psychology = PsychologyBelief(
            personality=personality,
            risk_aversion=risk_aversion,
        )

    # ---------- 函数 3：主循环 ----------
    def step(self) -> None:
        """
        Mesa 强制入口，串联感知→记忆→信念更新→欲望→意图→行动→情绪更新。
        任何子步异常须降级，不崩溃。
        """
        try:
            # 1. 感知
            perception = self._perceive()

            # 2. 检索记忆
            memories = self._retrieve_memory(perception)

            # 3. 更新信念（调用 LLM）
            self._update_beliefs(perception, memories)

            # 4. 推断欲望
            desires = self._infer_desires()

            # 5. 规划意图
            intention = self._plan_intentions(desires)

            # 6. 执行行动
            action_record = self._execute_action(intention)

            # 7. 更新情绪（这里用环境反馈，暂时无反馈则传入空）
            self._update_emotion({})

            # 若行动不是 SILENT，则提交到环境
            if action_record and action_record.action_type != ActionType.SILENT:
                self.pending_action = action_record
                # 调用 A 模块的 submit_action（通过 model 引用）
                self.model.submit_action(action_record)
                # 触发 Hawkes 事件（C 模块）
                self.model.hawkes.add_event(float(action_record.tick))

        except Exception as e:
            self.logger.error(f"Agent {self.unique_id} step failed: {e}")
            # 降级：不清除已有信念，不提交行动
            self.pending_action = None

    # ---------- 函数 4：感知 ----------
    def _perceive(self) -> Perception:
        """从环境读取邻居动态、话题榜、@消息（只读）"""
        # 从 model 获取邻居（假设 model.grid 是网络，提供 get_neighbors 方法）
        # 这里简化：假设 model 有 get_neighbor_actions(self, agent_id) 方法
        neighbor_actions = []
        if hasattr(self.model, "get_neighbor_actions"):
            neighbor_actions = self.model.get_neighbor_actions(self.unique_id)

        # 话题榜
        trending = []
        if hasattr(self.model, "trending"):
            trending = self.model.trending

        # @消息（简化：从信息流中筛选 mentions）
        mentions = []
        if hasattr(self.model, "get_mentions"):
            mentions = self.model.get_mentions(self.unique_id)

        return Perception(
            neighbor_actions=neighbor_actions,
            trending_topics=trending,
            mentions=mentions,
            tick=self.model.schedule.time if hasattr(self.model, "schedule") else 0,
        )

    # ---------- 函数 5：记忆检索 ----------
    def _retrieve_memory(self, perception: Perception) -> List[MemoryRecord]:
        """
        从短期记忆中检索与当前感知相关的记录。
        Week 2 简化版：取最近 N 条，后续可扩展语义相似度。
        """
        if not self.memory:
            return []
        # 简单按时间倒序取最近 5 条（或与 perception.tick 差距小于一定阈值）
        # 这里返回全部（后续可优化）
        sorted_mem = sorted(self.memory, key=lambda x: x.tick, reverse=True)
        # 只取最近 5 条
        return sorted_mem[:5]

    # ---------- 函数 6：更新信念（核心 LLM 接入） ----------
    def _update_beliefs(
        self, perception: Perception, memories: List[MemoryRecord]
    ) -> None:
        """
        调用 C 模块的 LLM 工具链，更新四层信念。
        LLM 失败时保持原信念不变。
        """
        # 构建环境信息（用于 prompt）
        env_info = {
            "tick": perception.tick,
            "trending": [t[0] for t in perception.trending_topics],
            "neighbor_actions_summary": [
                f"agent {a.source_id} {a.action_type.name} on {a.event_id}"
                for a in perception.neighbor_actions[:3]
            ],
        }

        # 调用 C 模块的函数（假设已由 C 实现并全局可用）
        # 实际项目中通过 import 引入，或通过 model 传递
        try:
            # 1) 构造 prompt
            prompt = self.model.llm_utils.build_prompt(
                belief=self.beliefs,
                memory=memories,
                env_info=env_info,
            )
            # 2) 调用 LLM
            client = self.model.llm_utils.get_client()
            raw_response = client.chat(prompt)
            # 3) 解析响应
            parsed = self.model.llm_utils.parse_llm_response(raw_response)
            # 4) 更新信念
            if "opinion_updates" in parsed:
                for eid, val in parsed["opinion_updates"].items():
                    if eid in self.beliefs.opinions:
                        # clip
                        new_val = max(-1.0, min(1.0, val))
                        self.beliefs.opinions[eid].opinion_value = new_val
                    else:
                        # 新增事件观点
                        new_val = max(-1.0, min(1.0, val))
                        self.beliefs.opinions[eid] = OpinionBelief(
                            event_id=eid, opinion_value=new_val
                        )
            if "emotion_delta" in parsed:
                delta = parsed["emotion_delta"]
                # 更新 valence 和 arousal
                new_valence = self.beliefs.emotion.valence + delta.get("valence", 0.0)
                new_valence = max(-1.0, min(1.0, new_valence))
                self.beliefs.emotion.valence = new_valence
                new_arousal = self.beliefs.emotion.arousal + delta.get("arousal", 0.0)
                new_arousal = max(0.0, min(1.0, new_arousal))
                self.beliefs.emotion.arousal = new_arousal
            # 可选：更新身份/心理信念（暂略）
        except Exception as e:
            self.logger.warning(f"LLM update failed: {e}, keeping beliefs unchanged")
            # 保持原信念（符合兜底规则）

    # ---------- 函数 7：推断欲望 ----------
    def _infer_desires(self) -> List[Desire]:
        """
        基于当前信念生成欲望列表，并按优先级排序。
        """
        desires = []
        # 示例规则：情绪低 -> 发泄欲望
        if self.beliefs.emotion.valence < -0.3:
            desires.append(
                Desire(
                    goal_type="vent_emotion",
                    priority=0.8,
                    event_id=list(self.beliefs.opinions.keys())[0]
                    if self.beliefs.opinions
                    else "general",
                )
            )
        # 若有官方事件，尝试获取信息
        if self.agent_type != AgentType.OFFICIAL:
            for eid, opin in self.beliefs.opinions.items():
                if opin.confidence < 0.4:
                    desires.append(
                        Desire(goal_type="get_info", priority=0.6, event_id=eid)
                    )
        # 默认寻求认同（如果是意见领袖或媒体）
        if self.agent_type in (AgentType.OPINION_LEADER, AgentType.MEDIA):
            desires.append(
                Desire(
                    goal_type="seek_identity",
                    priority=0.5,
                    event_id="general",
                )
            )
        # 按优先级排序
        desires.sort(key=lambda d: d.priority, reverse=True)
        return desires

    # ---------- 函数 8：规划意图 ----------
    def _plan_intentions(self, desires: List[Desire]) -> Intention:
        """
        将最高优先级的欲望转化为具体意图。
        """
        if not desires:
            # 默认沉默
            return Intention(
                action_type=ActionType.SILENT,
                event_id="",
                content_plan="",
                target_id=None,
            )

        top_desire = desires[0]
        goal = top_desire.goal_type
        event_id = top_desire.event_id

        # 根据目标类型选择行动类型
        if goal == "vent_emotion":
            action = ActionType.POST
            content = "表达情绪"  # 实际可由 LLM 生成，这里简化
        elif goal == "get_info":
            action = ActionType.COMMENT
            content = "询问信息"
        elif goal == "seek_identity":
            action = ActionType.REPOST
            content = "转发支持"
        else:
            action = ActionType.SILENT
            content = ""

        # 如果 event_id 无效，则沉默
        if event_id not in self.beliefs.opinions and event_id != "general":
            return Intention(ActionType.SILENT, "", "")

        return Intention(
            action_type=action,
            event_id=event_id,
            content_plan=content,
            target_id=None,  # 可根据邻居选择
        )

    # ---------- 函数 9：执行行动 ----------
    def _execute_action(self, intention: Intention) -> ActionRecord:
        """
        执行意图，生成 ActionRecord 并触发 Hawkes 事件（由外部调用）。
        """
        if intention.action_type == ActionType.SILENT:
            return ActionRecord(
                agent_id=self.unique_id,
                action_type=ActionType.SILENT,
                content="",
                event_id="",
                opinion_value=0.0,
                tick=self.model.schedule.time if hasattr(self.model, "schedule") else 0,
            )

        # 获取当前对该事件的观点值
        opinion_val = 0.0
        if intention.event_id in self.beliefs.opinions:
            opinion_val = self.beliefs.opinions[intention.event_id].opinion_value

        record = ActionRecord(
            agent_id=self.unique_id,
            action_type=intention.action_type,
            content=intention.content_plan,
            event_id=intention.event_id,
            opinion_value=opinion_val,
            target_id=intention.target_id,
            tick=self.model.schedule.time if hasattr(self.model, "schedule") else 0,
        )

        # 将此次行动加入短期记忆
        info = SocialInfo(
            source_id=self.unique_id,
            action_type=intention.action_type,
            content=intention.content_plan,
            event_id=intention.event_id,
            opinion_value=opinion_val,
            timestamp=record.tick,
        )
        self.memory.append(MemoryRecord(tick=record.tick, info=info, relevance=0.5))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

        # 注意：实际提交环境和 Hawkes 由 step() 中的 model 调用完成，这里仅返回记录
        return record

    # ---------- 函数 10：更新情绪 ----------
    def _update_emotion(self, env_feedback: Dict[str, float]) -> None:
        """
        根据环境反馈更新情绪，无反馈时向基线衰减。
        """
        # 基线（中性）
        base_valence = 0.0
        base_arousal = 0.5

        # 若存在反馈则应用
        if "valence_delta" in env_feedback:
            new_val = self.beliefs.emotion.valence + env_feedback["valence_delta"]
        else:
            # 向基线衰减（每步衰减 5%）
            new_val = self.beliefs.emotion.valence * 0.95 + base_valence * 0.05

        if "arousal_delta" in env_feedback:
            new_ar = self.beliefs.emotion.arousal + env_feedback["arousal_delta"]
        else:
            new_ar = self.beliefs.emotion.arousal * 0.95 + base_arousal * 0.05

        # clip
        self.beliefs.emotion.valence = max(-1.0, min(1.0, new_val))
        self.beliefs.emotion.arousal = max(0.0, min(1.0, new_ar))