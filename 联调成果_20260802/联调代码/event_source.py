"""
event_source.py — 事件源投放模块（A 模块新增，问题②）
--------------------------------------------------------------------
负责人：A

【为什么单独一个文件】
流程图第一个节点是「事件源 → 随机发到小群 / 大群」，它与 Agent 的自发
发言是两条不同的因果链：

    事件源  ：外生冲击（exogenous shock），不由 Hawkes 强度决定，
              一次性/低频投放，携带指定的 H₀ 与初始负面程度；
    Agent   ：内生扩散（endogenous diffusion），由 Hawkes 采样激活，
              只能转发/回复已经存在的消息。

原实现把事件注入写成 OpinionModel.__init__ 里的一段私有代码，
既没有独立生命周期，也没法在仿真中途追加事件，更没法被 D 模块的
「事件注入型干预」复用。本模块把它抽成一个可被 step() 调用的对象。

【对外接口】
    EventSource(model, params)
        .inject_initial()  -> Optional[EventRecord]   # tick 0 的原始事件
        .step()            -> List[EventRecord]       # 每 tick 调用，可能追加事件
        .events            -> List[EventRecord]       # 全部投放记录（D/E 直接读）

【小群 / 大群的定义】（《场景设定》§1）
    小群 small = {DORM, CLASS}      管理弱、曝光低
    大群 large = {MAJOR, CAMPUS}    管理强、曝光高
    initial_event_group_scope:
        "small"           只投小群
        "large"           只投大群
        "small_or_large"  随机二选一（默认，对应流程图的分叉）
        "any"             四类群等概率
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from types_def import (
    ActionRecord, ActionType, EventRecord, GroupType, MessageType,
    LARGE_GROUPS, SMALL_GROUPS,
)

_LOG = logging.getLogger("EventSource")


def _clip01(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


class EventSource:
    """外生事件投放器。由 OpinionModel 持有并在 step() 开头调用。"""

    def __init__(self, model, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Parameters
        ----------
        model  : OpinionModel 实例（需要 submit_action / group_members / random）
        params : scenario_params 子集，缺省键回落到 types_def.DEFAULT_SCENARIO_PARAMS
        """
        self.model = model
        self.params: Dict[str, Any] = dict(params or {})

        self.events: List[EventRecord] = []
        self._seq: int = 0
        self._last_event_tick: int = -10 ** 9

        self.secondary_probability = _clip01(
            self.params.get("secondary_event_probability", 0.05), 0.05
        )
        self.secondary_max = max(0, int(self.params.get("secondary_event_max", 2)))
        self.secondary_min_interval = max(
            1, int(self.params.get("secondary_event_min_interval", 8))
        )
        self.scope = str(
            self.params.get("initial_event_group_scope", "small_or_large")
        ).strip().lower()
        if self.scope not in {"small", "large", "small_or_large", "any"}:
            _LOG.warning("未知 initial_event_group_scope=%r，回退 small_or_large", self.scope)
            self.scope = "small_or_large"

    # ------------------------------------------------------------------ #
    #  对外：初始事件                                                      #
    # ------------------------------------------------------------------ #
    def inject_initial(self) -> Optional[EventRecord]:
        """投放 tick 0 的 original 事件。返回 EventRecord，失败返回 None。"""
        if not bool(self.params.get("inject_initial_event", True)):
            _LOG.info("inject_initial_event=False，跳过初始事件投放")
            return None
        return self._inject(is_initial=True)

    # ------------------------------------------------------------------ #
    #  对外：每 tick 调用                                                  #
    # ------------------------------------------------------------------ #
    def step(self) -> List[EventRecord]:
        """
        仿真每 tick 调用一次。按概率追加「二次事件」（新爆料 / 后续进展），
        使舆情不是单峰衰减，而是可能出现二次高峰。
        """
        if self.secondary_probability <= 0.0 or self.secondary_max <= 0:
            return []

        n_secondary = sum(1 for e in self.events if not e.is_initial)
        if n_secondary >= self.secondary_max:
            return []

        tick = int(self.model.schedule.time)
        if tick - self._last_event_tick < self.secondary_min_interval:
            return []
        if self.model.random.random() >= self.secondary_probability:
            return []

        record = self._inject(is_initial=False)
        return [record] if record is not None else []

    # ------------------------------------------------------------------ #
    #  内部：一次投放                                                      #
    # ------------------------------------------------------------------ #
    def _candidate_group_types(self) -> List[GroupType]:
        """按 scope 决定本次事件可能落到哪一档群（小群 / 大群）。"""
        if self.scope == "small":
            return list(SMALL_GROUPS)
        if self.scope == "large":
            return list(LARGE_GROUPS)
        if self.scope == "any":
            return list(GroupType)
        # small_or_large：先随机选「小群/大群」这一档，再在档内随机选具体群类型
        band = list(SMALL_GROUPS) if self.model.random.random() < 0.5 else list(LARGE_GROUPS)
        return band

    def _inject(self, is_initial: bool) -> Optional[EventRecord]:
        model = self.model
        agents = list(model.schedule.agents)
        if not agents:
            return None

        candidates = self._candidate_group_types()
        model.random.shuffle(candidates)

        origin_agent = None
        group_id = ""
        group_type = GroupType.DORM
        for candidate in candidates:
            gid = model.group_id_for_type(candidate)
            members = sorted(model.group_members.get(gid, set()))
            if not members:
                continue
            origin_agent = model._get_agent_by_id(model.random.choice(members))
            if origin_agent is None:
                continue
            group_id = gid
            group_type = candidate
            break

        if origin_agent is None:
            # 极端退化（例如 n_agents=1 且群分配异常）：落到任意 agent 的主群
            origin_agent = model.random.choice(agents)
            group_id = model.get_agent_group(origin_agent.unique_id) or "GROUP_CLASS"
            resolved = model.get_group_type_by_id(group_id)
            group_type = resolved if resolved is not None else GroupType.CLASS

        scope_label = "small" if group_type in SMALL_GROUPS else "large"
        heat = max(0.0, float(self.params.get("initial_event_heat", 0.80)))
        negative = _clip01(self.params.get("initial_event_negative", 0.72), 0.72)
        content = str(self.params.get(
            "initial_event_content", "校园事件相关消息开始在群聊中传播"
        ))
        if not is_initial:
            # 二次事件：热度略低、负面略高（后续爆料通常更刺激但覆盖面更窄）
            heat = heat * 0.75
            negative = min(1.0, negative + 0.08)
            content = f"{content}（后续进展 #{len(self.events)}）"

        self._seq += 1
        tick = int(model.schedule.time)

        action = ActionRecord(
            agent_id=origin_agent.unique_id,
            action_type=ActionType.SEND_MESSAGE,
            content=content,
            tick=tick,
            topic_id="T001",
            distortion_level=0.0,
            message_type=MessageType.ORIGINAL,
            negative_score=negative,
            heat=heat,
            group_id=group_id,
            forward_count=0,
        )
        # 事件源是外生冲击：不受禁言约束，且 heat 按 H₀ 绝对值写入目标群，
        # 而不是当成一条普通消息的增量。
        if not model.submit_action(action, bypass_mute=True, absolute_heat=True):
            _LOG.warning("事件投放被拒绝 | agent=%s group=%s", origin_agent.unique_id, group_id)
            return None

        event = EventRecord(
            event_seq=self._seq,
            tick=tick,
            origin_agent_id=origin_agent.unique_id,
            group_id=group_id,
            group_type=group_type,
            scope=scope_label,
            heat=heat,
            negative_score=negative,
            topic_id="T001",
            content=content,
            is_initial=is_initial,
            message_id=action.message_id,
        )
        self.events.append(event)
        self._last_event_tick = tick

        _LOG.info(
            "[tick=%d] 事件源投放 #%d | %s | origin=Agent-%d | group=%s(%s群) "
            "| H0=%.3f negative=%.3f",
            tick, self._seq, "初始事件" if is_initial else "二次事件",
            origin_agent.unique_id, group_id,
            "小" if scope_label == "small" else "大",
            heat, negative,
        )
        return event

    # ------------------------------------------------------------------ #
    #  对外：摘要（供 D / E 使用）                                          #
    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        return {
            "n_events": len(self.events),
            "n_initial": sum(1 for e in self.events if e.is_initial),
            "n_secondary": sum(1 for e in self.events if not e.is_initial),
            "events": [
                {
                    "event_seq": e.event_seq,
                    "tick": e.tick,
                    "origin_agent_id": e.origin_agent_id,
                    "group_id": e.group_id,
                    "group_type": e.group_type.name,
                    "scope": e.scope,
                    "heat": e.heat,
                    "negative_score": e.negative_score,
                    "message_id": e.message_id,
                }
                for e in self.events
            ],
        }
