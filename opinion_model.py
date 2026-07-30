"""
opinion_model.py — 模型 / 环境层（接口表 #12–20）
====================================================================
负责人：A
版本：  v3（场景对齐版）

函数清单（接口表编号）：
    #12  __init__(config)
    #13  _build_social_network() → nx.Graph
    #14  _place_agents()
    #15  step()
    #16  _update_environment()
    #17  _calc_avg_opinion() → float
    #18  _calc_polarization() → float
    #19  _calc_emotional_contagion() → float
    #20  submit_action(record)

════════════════════════════════════════════════════════════════════
【v3 变更：逐条对齐《场景设定_20260718_v2》与流程图】
════════════════════════════════════════════════════════════════════

① 干预触发条件
   设定原文：「负面程度超过阈值 → controller 开始干预」。
   旧实现以热度 H ≥ θ 为主，导致「负面高但不热」不会触发。
   现在：
     - 默认 intervention_trigger = "negative"（负面阈值 0.65）
     - heat / heat_or_negative / heat_and_negative 可切换，热度阈值单列
       为 heat_threshold，不再和 THETA 语义混用；
     - 群负面程度 topic_negative 改为「群内存活消息 negative_score 的
       时间加权均值」，是一个可解释、可跨群比较的量，能真正越过 0.65；
     - 触发判定移到 step() 里 Agent 激活之前（_evaluate_intervention_triggers），
       Controller 当拍即可响应，不再延迟一个 tick。

② 事件发起
   流程图有独立的「事件源 → 随机发到小群 / 大群」节点。
   现在拆出 event_source.EventSource：
     - 初始事件在 __init__ 末尾投放，H₀ / 初始负面由配置给定；
     - EventSource.step() 每 tick 被调用，可按概率追加二次事件；
     - 小群 = {DORM, CLASS}，大群 = {MAJOR, CAMPUS}，不再写死 DORM/CAMPUS；
     - 投放记录为 EventRecord 列表，D / E 可直接读取。

③ 跨群条件
   流程图为「转发次数 ≥ 阈 → 才进入跨群通道」。
   现在：
     - ActionRecord.forward_count 记录消息谱系的转发深度，
       self.lineage_forward_count 记录同一根消息被转发的总次数；
     - 群内 FORWARD 始终允许（这是攒够转发次数的唯一途径）；
     - 跨群 FORWARD 需 max(深度, 谱系次数) ≥ cross_group_forward_threshold，
       或该群热度已 ≥ heat_threshold（cross_group_heat_bypass）；
     - 宏观热度溅射同样受该门槛约束（macro_spread_requires_threshold）。

④ 小群互转
   流程图有 DORM–CLASS 横向箭头。
   现在把转发方向分成四类，各自可配权重：
     upward（小→大） / lateral（同带互转，即小群互转） /
     downward（大→小） / intra（群内）
   DORM↔CLASS 同属 small 带 → lateral；MAJOR↔CAMPUS 同属 large 带 → lateral。
   allow_downward_forward 默认改为 true，但权重仅 0.15（向下弱）。

⑤ 禁言
   《场景设定》§2 Controller = 提醒 / 澄清 / 禁言 / 公告。
   现在 ActionType 新增 MUTE / ANNOUNCE：
     - MUTE 由环境层真正执行：self.muted_until[(agent_id, group_id)]，
       被禁言者在该群内的一切投稿会被 submit_action 拒绝；
     - DORM 群不可禁言（control_level 最低，Controller 介入不了）；
     - MUTE / ANNOUNCE 会即时压降该群的负面程度与缓存消息失真。

⑥ 曝光
   《场景设定》§1 群类型表的「曝光」列 → types_def.GROUP_EXPOSURE。
   曝光在三处生效：热度增益、宏观扩散目标选择、单 tick 可感知消息条数。
   于是各群不再共用同一套跨群概率，差异也不再只靠 β。

⑦ 降温公式表述
   统一为 types_def.heat_decay()：
       H(t+1) = H(t) · exp(−α · control_level_k)
   即《场景设定》§3 的「基础降温 × control_level」，与 §4.2 式(2)
       H(t+1) = H(t) · exp(−α) · exp(−β_k)
   代数等价（β_k = α·(control_level_k − 1)）。control_level 是唯一可调参数。

⑧ 次要
   - A 模块全链路无 event_id 残留，一律 topic_id；
   - trust / confirmation_bias 已在 PsychologyBelief 中单独建模（B 模块使用）。

注意：B 模块的 calc_heat_decay() 是热度衰减的纯计算函数，
      A 模块在 _update_environment 中优先调用它；B 未交付时使用
      types_def.heat_decay() 降级，两者结果必须一致。
"""

from __future__ import annotations
import mesa_patch  # noqa: F401  Mesa 3.x 兼容补丁，必须在 mesa import 之前

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

# Mesa（或兼容垫片）
try:
    from mesa import Model
    from mesa.time import RandomActivation
    from mesa.space import NetworkGrid
    from mesa.datacollection import DataCollector
    _MESA_REAL = True
except ImportError:
    from mesa_compat import Model, RandomActivation, NetworkGrid, DataCollector
    _MESA_REAL = False

from types_def import (
    SimConfig, AgentType, GroupType, ActionRecord, ActionType,
    EmotionState, ForwardDirection, MessageType,
    ALPHA, THETA, HEAT_CAP,
    GROUP_BETA, GROUP_CONTROL_LEVEL, GROUP_EXPOSURE, GROUP_BAND,
    CONTROLLER_ONLY_ACTIONS, DEFAULT_SCENARIO_PARAMS, LATERAL_PAIRS,
    TimeSlot, TIME_SLOT_GROUP_ACTIVITY, TIME_SLOT_TOPIC_SUITABILITY,
    DEFAULT_DAY_SCHEDULE,
    heat_decay as _formula_heat_decay,
)
from event_source import EventSource


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  仿真时钟引擎（A 模块 tick → 真实时段映射）                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

# 各时段对应的「place」与「activity」描述（供 LLM prompt / 规则存根使用）
_TIME_SLOT_CONTEXT: Dict[str, Dict[str, str]] = {
    TimeSlot.MORNING_CLASS: {
        "place":    "教室",
        "activity": "早上上课",
    },
    TimeSlot.LUNCH: {
        "place":    "食堂/宿舍",
        "activity": "午休",
    },
    TimeSlot.AFTERNOON_CLASS: {
        "place":    "教室",
        "activity": "下午上课",
    },
    TimeSlot.EVENING_FREE: {
        "place":    "宿舍/校园",
        "activity": "晚间自由",
    },
    TimeSlot.LATE_NIGHT: {
        "place":    "宿舍",
        "activity": "深夜",
    },
}


class SimClock:
    """
    仿真时钟：把离散的 tick 编号转换成「现实时段」语义。

    ──────────────────────────────────────────────────────────────────────
    设计目标：
      tick 是调度上的整数步，但 Agent 的行为应受「几点了 / 在哪里」的
      常识约束：上课时间不是刷宿舍群八卦的高峰；晚间自由才是舆情扩散
      的主要时段。SimClock 把这套粗粒度时段常识注入 Perception，让
      LLM 路径和规则存根都能读到它。

    使用方式（opinion_model._perceive_clock_context）：
        clock = SimClock(tick_seconds=3600, day_schedule=None)
        ctx = clock.context(tick=7)
        # ctx.time_slot   → "evening_free"
        # ctx.wall_hour   → 16.0  (如果 day_start_hour=9，tick 7 = 9+7 = 16)
        # ctx.place       → "宿舍/校园"
        # ctx.activity    → "晚间自由"
        # ctx.group_activity_multiplier(GroupType.DORM) → 1.00
    ──────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        tick_seconds: int = 3600,
        day_schedule: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.tick_seconds = max(1, int(tick_seconds))
        schedule = day_schedule if day_schedule is not None else DEFAULT_DAY_SCHEDULE
        self.day_start_hour: float = float(schedule.get("day_start_hour", 9))
        # 解析时段列表；slots 的 "hours" 字段为 [start_h, end_h)
        self._slots: List[Dict[str, Any]] = list(schedule.get("slots", []))

    def wall_hour(self, tick: int) -> float:
        """返回 tick 对应的现实小时数（24h 浮点）。"""
        elapsed_seconds = int(tick) * self.tick_seconds
        elapsed_hours = elapsed_seconds / 3600.0
        return (self.day_start_hour + elapsed_hours) % 24.0

    def time_slot(self, tick: int) -> str:
        """
        返回 tick 对应的 TimeSlot 标签。

        按「现实小时」匹配 slots 列表中的区间（支持跨午夜的 hours=[22,33]）。
        无匹配时回退到 LATE_NIGHT。
        """
        hour = self.day_start_hour + (int(tick) * self.tick_seconds / 3600.0)
        # hour 可能超过 24，不 mod（方便跨午夜区间匹配）
        for slot in self._slots:
            h_start, h_end = slot["hours"][0], slot["hours"][1]
            if h_start <= hour % 24 < h_end or (
                h_end > 24 and (hour % 24 >= h_start or hour % 24 < h_end - 24)
            ):
                return slot["name"]
        return TimeSlot.LATE_NIGHT

    def group_activity_multiplier(self, tick: int, group_type: GroupType) -> float:
        """返回当前时段该群类型的活跃度乘数。"""
        slot = self.time_slot(tick)
        table = TIME_SLOT_GROUP_ACTIVITY.get(slot, {})
        return float(table.get(group_type.name, 1.0))

    def hawkes_mu_factor(self, tick: int) -> float:
        """
        根据时段调整 Hawkes 基线强度 μ 的乘数（问题文档 C 节提到
        「若加时段活跃度，可让 μ 随 time_of_day 变化」）。

        使用所有群活跃度的平均值作为全局激活率的调节因子。
        """
        slot = self.time_slot(tick)
        table = TIME_SLOT_GROUP_ACTIVITY.get(slot, {})
        mults = list(table.values())
        return float(sum(mults) / len(mults)) if mults else 1.0

    def topic_suitability(self, tick: int) -> Dict[str, float]:
        """返回当前时段各话题类型的适合性分数。"""
        slot = self.time_slot(tick)
        return dict(TIME_SLOT_TOPIC_SUITABILITY.get(slot, {}))

    def clock_context(self, tick: int, group_type: GroupType) -> Dict[str, Any]:
        """
        返回完整的时钟上下文字典，供 Perception 填充。

        字段含义：
          sim_time_seconds        : 仿真已流逝秒数
          wall_hour               : 对应现实小时（浮点）
          time_slot               : TimeSlot 标签
          place / activity        : 时段常识描述
          group_activity_multiplier : 该群在当前时段的活跃度乘数
          topic_suitability       : 各话题适合性
        """
        slot = self.time_slot(tick)
        ctx_desc = _TIME_SLOT_CONTEXT.get(slot, {"place": "未知", "activity": "未知"})
        return {
            "sim_time_seconds":          int(tick) * self.tick_seconds,
            "wall_hour":                 self.wall_hour(tick),
            "time_slot":                 slot,
            "place":                     ctx_desc["place"],
            "activity":                  ctx_desc["activity"],
            "group_activity_multiplier": self.group_activity_multiplier(tick, group_type),
            "topic_suitability":         self.topic_suitability(tick),
        }

_LOG = logging.getLogger("OpinionModel")

#: 群负面程度的时间加权常数（tick）。越小越只看最近的消息。
_NEGATIVE_RECENCY_TAU = 4.0


def _as_probability(value: Any, default: float) -> float:
    """把配置值安全转换到 [0, 1]；非法值回退到 default。"""
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class OpinionModel(Model):
    """
    多智能体舆情仿真 · 模型 / 环境层（A 模块）。

    场景：大学校园多群舆情扩散（DORM / CLASS / MAJOR / CAMPUS 四类群）。
    角色：ORDINARY / ACTIVE / RATIONAL / CONTROLLER。
    热度演化：H_k(t+1) = H_k(t) · exp(−α · control_level_k)。
    """

    DEFAULT_CROSS_GROUP_SPREAD_PROBABILITY = 0.15
    DEFAULT_GROUP_MEMBERSHIP_MODE = "hierarchical"
    DEFAULT_FORWARD_DESTINATION_STRATEGY = "weighted"

    # ================================================================== #
    #  #12  __init__                                                       #
    # ================================================================== #
    def __init__(self, config: SimConfig) -> None:
        super().__init__()
        self.config = config

        # ── 场景参数：以 types_def 默认表打底，YAML 覆盖 ────────────────
        self.scenario_params: Dict[str, Any] = dict(DEFAULT_SCENARIO_PARAMS)
        for key, value in (config.scenario_params or {}).items():
            if key == "forward_direction_weights" and isinstance(value, dict):
                merged = dict(DEFAULT_SCENARIO_PARAMS["forward_direction_weights"])
                merged.update(value)
                self.scenario_params[key] = merged
            else:
                self.scenario_params[key] = value

        sp = self.scenario_params

        # ── ① 干预触发（问题①）────────────────────────────────────────
        self.negative_threshold = _as_probability(sp.get("negative_threshold"), 0.65)
        self.heat_threshold = max(1e-6, _as_float(sp.get("heat_threshold"), THETA))
        self.intervention_requires_controller = bool(
            sp.get("intervention_requires_controller", True)
        )
        # 最小证据量：群内存活消息不足时不做负面判定。否则开局第一条 original
        # 消息就会把该群的加权均值顶到 0.72，导致「1 条消息触发全群干预」。
        self.intervention_min_messages = max(
            0, _as_int(sp.get("intervention_min_messages"), 4))
        self.intervention_trigger = str(sp.get("intervention_trigger", "negative")).strip().lower()
        if self.intervention_trigger not in {
            "heat", "negative", "heat_or_negative", "heat_and_negative"
        }:
            _LOG.warning("未知 intervention_trigger=%r，回退 negative", self.intervention_trigger)
            self.intervention_trigger = "negative"

        # ── ③ 跨群门槛（问题③）────────────────────────────────────────
        self.cross_group_forward_threshold = max(
            0, _as_int(sp.get("cross_group_forward_threshold"), 2)
        )
        self.cross_group_heat_bypass = bool(sp.get("cross_group_heat_bypass", True))
        _dir_thresholds = dict(
            DEFAULT_SCENARIO_PARAMS["cross_group_forward_threshold_by_direction"])
        _dir_thresholds.update(sp.get("cross_group_forward_threshold_by_direction") or {})
        self.forward_threshold_by_direction: Dict[str, int] = {
            ForwardDirection.UPWARD:   max(0, _as_int(_dir_thresholds.get("upward"), 1)),
            ForwardDirection.LATERAL:  max(0, _as_int(_dir_thresholds.get("lateral"), 0)),
            ForwardDirection.DOWNWARD: max(0, _as_int(_dir_thresholds.get("downward"), 3)),
            ForwardDirection.INTRA:    0,
        }
        self.macro_spread_requires_threshold = bool(
            sp.get("macro_spread_requires_threshold", True)
        )

        # ── ④ 转发方向权重（问题④）────────────────────────────────────
        weights = dict(DEFAULT_SCENARIO_PARAMS["forward_direction_weights"])
        weights.update(sp.get("forward_direction_weights") or {})
        self.forward_direction_weights: Dict[str, float] = {
            ForwardDirection.UPWARD:   max(0.0, _as_float(weights.get("upward"), 1.00)),
            ForwardDirection.LATERAL:  max(0.0, _as_float(weights.get("lateral"), 0.45)),
            ForwardDirection.DOWNWARD: max(0.0, _as_float(weights.get("downward"), 0.15)),
            ForwardDirection.INTRA:    max(0.0, _as_float(
                sp.get("intra_group_forward_weight"), 0.60)),
        }

        # ── ⑤ 禁言（问题⑤）───────────────────────────────────────────
        self.enable_mute = bool(sp.get("enable_mute", True))
        self.mute_duration_ticks = max(1, _as_int(sp.get("mute_duration_ticks"), 3))
        self.mute_negative_trigger = _as_probability(sp.get("mute_negative_trigger"), 0.72)
        self.mute_max_per_tick = max(0, _as_int(sp.get("mute_max_per_tick"), 2))
        self.announce_negative_relief = _as_probability(
            sp.get("announce_negative_relief"), 0.25)
        self.announce_distortion_relief = _as_probability(
            sp.get("announce_distortion_relief"), 0.30)

        # ── ⑥ 曝光（问题⑥）───────────────────────────────────────────
        self.exposure_heat_weight = _as_probability(sp.get("exposure_heat_weight"), 0.60)
        self.exposure_spread_weight = _as_probability(sp.get("exposure_spread_weight"), 1.00)
        self.exposure_perception_weight = _as_probability(
            sp.get("exposure_perception_weight"), 0.60)
        self.forward_exposure_weight = _as_probability(
            sp.get("forward_exposure_weight"), 0.50)

        # ── 仿真时钟（A 模块 tick→真实时段映射）──────────────────────────
        tick_seconds = max(1, _as_int(sp.get("tick_seconds"), 3600))
        day_schedule = sp.get("day_schedule") or None    # None → 使用 DEFAULT_DAY_SCHEDULE
        self.sim_clock = SimClock(tick_seconds=tick_seconds, day_schedule=day_schedule)
        _LOG.info(
            "仿真时钟已初始化 | tick_seconds=%d（每 tick≈%.1f 分钟）| day_start=%s",
            tick_seconds, tick_seconds / 60.0, self.sim_clock.day_start_hour,
        )

        # ── 热度标度与失真恢复 ────────────────────────────────────────
        self.base_heat_gain = max(0.0, _as_float(sp.get("base_heat_gain"), 0.055))
        self.heat_cap = max(0.1, _as_float(sp.get("heat_cap"), HEAT_CAP))
        self.message_expire_ticks = max(1, _as_int(sp.get("message_expire_ticks"), 10))
        self.negative_smoothing = _as_probability(sp.get("negative_smoothing"), 0.50)

        # ── 群成员与转发策略（承载在 network_params）───────────────────
        network_params = config.network_params or {}
        if "cross_group_visibility" in network_params:
            _LOG.warning("cross_group_visibility 已弃用并被忽略；异群信息必须通过真实 FORWARD 投递")
        self.group_membership_mode = str(
            network_params.get("group_membership_mode", self.DEFAULT_GROUP_MEMBERSHIP_MODE)
        ).strip().lower()
        if self.group_membership_mode not in {"hierarchical", "single"}:
            _LOG.warning("未知 group_membership_mode=%r，回退 hierarchical", self.group_membership_mode)
            self.group_membership_mode = self.DEFAULT_GROUP_MEMBERSHIP_MODE

        self.forward_destination_strategy = str(
            network_params.get("forward_destination_strategy",
                               self.DEFAULT_FORWARD_DESTINATION_STRATEGY)
        ).strip().lower()
        self.allow_downward_forward = bool(
            network_params.get("allow_downward_forward",
                               sp.get("allow_downward_forward", True))
        )
        self.cross_group_spread_probability = _as_probability(
            network_params.get("cross_group_spread_probability",
                               self.DEFAULT_CROSS_GROUP_SPREAD_PROBABILITY),
            self.DEFAULT_CROSS_GROUP_SPREAD_PROBABILITY,
        )

        # ① 固定随机种子
        self.random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        # ② 调度器
        self.schedule = RandomActivation(self)

        # ③ 构建社交网络
        G = self._build_social_network()
        self.grid = NetworkGrid(G)

        # ④ 环境状态 —— A 模块权威维护
        self.info_stream_cache: List[ActionRecord] = []

        self.group_type_by_id: Dict[str, GroupType] = {
            self.group_id_for_type(group_type): group_type for group_type in GroupType
        }
        self.group_id_by_type: Dict[GroupType, str] = {
            group_type: group_id for group_id, group_type in self.group_type_by_id.items()
        }
        self.group_members: Dict[str, set] = {gid: set() for gid in self.group_type_by_id}

        # 消息索引与转发谱系（问题③）
        self._message_seq: int = 0
        self._message_by_id: Dict[str, ActionRecord] = {}
        self.lineage_forward_count: Dict[str, int] = {}   # root_message_id -> 累计转发次数
        self.group_forward_count: Dict[str, int] = {gid: 0 for gid in self.group_type_by_id}

        # 热度 / 负面程度：{topic_id: {GroupType: value}}
        self.topic_heat: Dict[str, Dict[GroupType, float]] = {"T001": {g: 0.0 for g in GroupType}}
        self._topic_heat_flat: Dict[str, float] = {"T001": 0.0}
        self.topic_negative: Dict[str, Dict[GroupType, float]] = {"T001": {g: 0.0 for g in GroupType}}
        self._topic_negative_flat: Dict[str, float] = {"T001": 0.0}

        # 各群首次干预时刻 t_k^int（A 为权威）。DORM 恒为 None（等价 +∞）
        #   intervention_tick  ：触发条件首次满足的 tick，对应《场景设定》§4.2.4，
        #                        直接进入热度公式的 𝟙[t ≥ t_k^int]；
        #   governance_tick    ：Controller **真正执行**了澄清/公告/禁言的 tick。
        #                        《场景设定》§3 的「Controller 干预后 → 负面程度和
        #                        失真下降」以它为准 —— 否则只要阈值一到，负面就会
        #                        自动回落，Controller 还没来得及动手就没事了，
        #                        干预效果的消融实验也无从做起。
        self.intervention_tick: Dict[GroupType, Optional[int]] = {g: None for g in GroupType}
        self.governance_tick: Dict[GroupType, Optional[int]] = {g: None for g in GroupType}

        # 禁言状态（问题⑤）：(agent_id, group_id) -> 解禁 tick
        self.muted_until: Dict[Tuple[int, str], int] = {}
        self.mute_count: int = 0
        self.announce_count: int = 0
        self.blocked_by_mute: int = 0

        # 指标计数
        self.cross_group_forward: int = 0     # Agent 真实跨群转发
        self.intra_group_forward: int = 0     # Agent 群内转发（攒转发次数）
        self.cross_group_spread: int = 0      # 宏观热度溅射
        self.lateral_forward: int = 0         # 小群互转 DORM↔CLASS（问题④，流程图横向箭头）
        self.downward_forward: int = 0        # 大群 → 小群
        self.upward_forward: int = 0          # 小群 → 大群（主通道）
        self.blocked_by_forward_gate: int = 0 # 因转发次数未达阈被拦下的跨群尝试

        self._prev_emotion_snapshot: Dict[int, EmotionState] = {}

        # ⑤ Hawkes 引擎（C 模块）
        from hawkes_engine import HawkesEngine
        self.hawkes = HawkesEngine(
            mu=config.hawkes_params.get("mu", 0.1),
            alpha=config.hawkes_params.get("alpha", 0.5),
            beta=config.hawkes_params.get("beta", 1.0),
        )

        # ⑤-b LLM 客户端（C 模块）
        self._llm_client = None
        self.llm_action_enabled = bool((config.llm_config or {}).get("use_actions", True))
        if config.llm_config:
            try:
                from llm_utils import setup_llm_client
                self._llm_client = setup_llm_client(config.llm_config)
                _LOG.info(
                    "LLM 客户端初始化成功 | provider=%s | model=%s",
                    config.llm_config.get("provider", "?"),
                    config.llm_config.get("model", "?"),
                )
            except Exception as exc:
                _LOG.warning("LLM 客户端初始化失败，所有 Agent 将使用规则存根: %s", exc)

        # ⑥ DataCollector（E 模块从此处读取输出）
        self.datacollector = DataCollector(
            model_reporters={
                # 接口表 §5 必需列
                "avg_opinion":          lambda m: m._calc_avg_opinion(),
                "polarization":         lambda m: m._calc_polarization(),
                "emotional_contagion":  lambda m: m._calc_emotional_contagion(),
                "message_count":        lambda m: len(m.info_stream_cache),
                "negative_emotion":     lambda m: m._calc_negative_emotion(),
                "distortion_level":     lambda m: m._calc_avg_distortion(),
                "cross_group_forward":  lambda m: m.cross_group_forward,
                "intervention_tick":    lambda m: m._collect_intervention_tick(),
                "recovery_time":        lambda m: m._calc_recovery_time(),
                # v3 扩展列
                "cross_group_spread":   lambda m: m.cross_group_spread,
                "intra_group_forward":  lambda m: m.intra_group_forward,
                "lateral_forward":      lambda m: m.lateral_forward,
                "upward_forward":       lambda m: m.upward_forward,
                "downward_forward":     lambda m: m.downward_forward,
                "group_negative_max":   lambda m: m._calc_group_negative_max(),
                "group_heat_max":       lambda m: m._calc_group_heat_max(),
                "mute_count":           lambda m: m.mute_count,
                "active_mutes":         lambda m: m._count_active_mutes(),
                "n_intervened_groups":  lambda m: m._count_intervened_groups(),
                # 仿真时钟列（A 模块新增，对应问题①「tick→时钟」）
                "wall_hour":            lambda m: m.get_wall_hour(),
                "time_slot":            lambda m: m.get_time_slot(),
                "time_mu_factor":       lambda m: m.sim_clock.hawkes_mu_factor(
                                            int(m.schedule.time)),
            }
        )

        # ⑦ 热度回落追踪（recovery_time）
        self._heat_exceeded_tick: Optional[int] = None
        self._recovery_time: Optional[int] = None

        # ⑧ A 模块自维护的 agent 查找字典
        self._agent_dict: Dict[int, Any] = {}

        # ⑨ 放置智能体
        self._place_agents()

        # ⑩ 事件源（问题②）—— 独立模块，对应流程图「事件源」节点
        self.event_source = EventSource(self, self.scenario_params)
        self.event_source.inject_initial()

        _LOG.info(
            "OpinionModel v3 初始化完成 | n_agents=%d | network=%s | membership=%s "
            "| 干预触发=%s | 跨群门槛=%d | llm=%s | mesa=%s",
            config.n_agents, config.network_type, self.group_membership_mode,
            self.intervention_trigger, self.cross_group_forward_threshold,
            "已接入" if self._llm_client is not None else "规则存根",
            "系统" if _MESA_REAL else "垫片",
        )

    # ================================================================== #
    #  #13  _build_social_network                                          #
    # ================================================================== #
    def _build_social_network(self) -> nx.Graph:
        """生成社交网络（BA 无标度 / WS 小世界），模拟校园群内好友关系。"""
        n = self.config.n_agents
        ntype = self.config.network_type
        params = self.config.network_params or {}
        seed = self.config.random_seed

        if ntype == "barabasi_albert":
            m = params.get("m", 3)
            if n == 1:
                G = nx.empty_graph(1)
                _LOG.info("BA 网络 n=1，退化为单节点孤立图")
            elif n <= m:
                G = nx.complete_graph(n)
                _LOG.warning("BA 网络要求 m < n，n=%d<=m=%s，退化为完全图", n, m)
            else:
                G = nx.barabasi_albert_graph(n, m, seed=seed)
                _LOG.info("BA 无标度网络 n=%d m=%s | 平均度=%.2f",
                          n, m, 2 * G.number_of_edges() / n)

        elif ntype == "watts_strogatz":
            k = params.get("k", 6)
            p = params.get("p", 0.1)
            if n <= k:
                G = nx.complete_graph(n)
                _LOG.warning("WS 网络 n=%d<=k=%s，退化为完全图", n, k)
            else:
                G = nx.watts_strogatz_graph(n, k, p, seed=seed)
                _LOG.info("WS 小世界网络 n=%d k=%s p=%s", n, k, p)

        else:
            _LOG.warning("未知 network_type='%s'，回退到 BA(m=3)", ntype)
            m_fb = min(3, max(1, n - 1))
            G = nx.barabasi_albert_graph(n, m_fb, seed=seed) if n > 1 else nx.empty_graph(1)

        return G

    # ================================================================== #
    #  #14  _place_agents                                                  #
    # ================================================================== #
    def _place_agents(self) -> None:
        """
        按比例部署四类 Agent。``group_type_ratio`` 决定 Agent 的「主群 / 最小群」。
        默认 ``hierarchical`` 模式下成员关系按校园层级向上包含：

            DORM   → DORM + CLASS + MAJOR + CAMPUS
            CLASS  → CLASS + MAJOR + CAMPUS
            MAJOR  → MAJOR + CAMPUS
            CAMPUS → CAMPUS

        因此只有同时在小群和大群里的人才能做跨群桥接；``single`` 模式保留
        一人一群的旧行为，仅用于对照实验。
        """
        from social_agent import SocialAgent

        n = self.config.n_agents
        ratio = self.config.agent_type_ratio
        nodes = list(self.grid.G.nodes())

        # ── 各 AgentType 数量 ─────────────────────────────────────────
        counts: Dict[AgentType, int] = {}
        remaining = n
        agent_type_list = list(AgentType)
        for i, agent_type in enumerate(agent_type_list):
            if i < len(agent_type_list) - 1:
                c = int(round(n * ratio.get(agent_type.name, 0.0)))
                counts[agent_type] = c
                remaining -= c
            else:
                counts[agent_type] = max(0, remaining)

        # ── 各 GroupType（主群）数量 ──────────────────────────────────
        g_ratio = self.config.group_type_ratio
        group_counts: Dict[GroupType, int] = {}
        g_remaining = n
        group_type_list = list(GroupType)
        for i, gt in enumerate(group_type_list):
            if i < len(group_type_list) - 1:
                c = int(round(n * g_ratio.get(gt.name, 0.0)))
                group_counts[gt] = c
                g_remaining -= c
            else:
                group_counts[gt] = max(0, g_remaining)

        group_assignments: List[GroupType] = []
        for gt, cnt in group_counts.items():
            group_assignments.extend([gt] * cnt)
        self.random.shuffle(group_assignments)

        # ── 逐个实例化并挂载 ──────────────────────────────────────────
        agent_id = 0
        for agent_type, count in counts.items():
            for _ in range(count):
                group_type = group_assignments[agent_id]

                if self.group_membership_mode == "hierarchical":
                    membership_types = [c for c in GroupType if int(c) >= int(group_type)]
                else:
                    membership_types = [group_type]

                group_ids = [self.group_id_for_type(c) for c in membership_types]
                primary_group_id = self.group_id_for_type(group_type)

                init_config: Dict[str, Any] = {
                    "agent_type": agent_type,
                    "group_type": group_type,
                    "primary_group_id": primary_group_id,
                    "group_ids": group_ids,
                    "stance_prior": self.random.uniform(-1.0, 1.0),
                    "topic_id": "T001",
                    "initial_heat": 0.5,
                }
                agent = SocialAgent(
                    unique_id=agent_id,
                    model=self,
                    agent_type=agent_type,
                    group_type=group_type,
                    init_config=init_config,
                )
                self.schedule.add(agent)
                self.grid.place_agent(agent, nodes[agent_id])
                self._agent_dict[agent_id] = agent
                for group_id in group_ids:
                    self.group_members[group_id].add(agent_id)
                agent_id += 1

        _LOG.info("AgentType 分布: %s",
                  " | ".join(f"{t.name}:{c}" for t, c in counts.items()))
        _LOG.info("主群 GroupType 分布: %s",
                  " | ".join(f"{g.name}:{c}" for g, c in group_counts.items()))
        _LOG.info("真实群成员数: %s",
                  " | ".join(f"{gid}:{len(m)}" for gid, m in self.group_members.items()))

    # ================================================================== #
    #  #15  step                                                           #
    # ================================================================== #
    def step(self) -> None:
        """
        单步主循环（v3 时序，与流程图一一对应）：

            ① 事件源投放                 event_source.step()      ← 问题②
            ② 刷新群负面程度             _refresh_group_negative()
            ③ 判定 Controller 干预触发   _evaluate_intervention_triggers()  ← 问题①
            ④ Hawkes 采样 → 激活 Agent（Controller 当拍即可干预）
            ⑤ 环境更新（降温 / 跨群溅射 / 恢复 / 过期 / 解禁）
            ⑥ DataCollector 收集
            ⑦ 时步自增 + Hawkes 回写
        """
        t = float(self.schedule.time)

        # ── ① 事件源（外生冲击，不受 Hawkes 强度约束）─────────────────
        try:
            self.event_source.step()
        except Exception as exc:
            _LOG.warning("[tick=%d] 事件源投放异常，已跳过: %s", int(t), exc)

        # ── ② / ③ 先算负面程度，再判定干预，Controller 本 tick 就能响应 ──
        self._refresh_group_negative()
        self._evaluate_intervention_triggers()

        # ── ④ Hawkes 激活比例（乘以时段活跃度乘数，μ 随 time_of_day 变化）──
        lam = self.hawkes.intensity(t)
        mu = self.config.hawkes_params.get("mu", 0.1)
        # 时段因子：晚间自由≈1.0，上课时段≈0.4；用它调节 Hawkes μ 对激活率的贡献
        time_mu_factor = self.sim_clock.hawkes_mu_factor(int(t))
        effective_mu = max(mu * time_mu_factor, 1e-9)
        activation_rate = float(np.clip(0.30 * lam / effective_mu, 0.10, 0.80))

        all_agents = list(self.schedule.agents)
        n_active = max(1, int(len(all_agents) * activation_rate))
        active_set = self.random.sample(all_agents, k=n_active)

        # Controller 总是被激活：否则「负面超阈却没人干预」的 bug 会以
        # 「本 tick 恰好没抽到 Controller」的形式复现。
        controller_ids = {a.unique_id for a in active_set}
        for agent in all_agents:
            if (getattr(agent, "beliefs", None) is not None
                    and agent.beliefs.identity.agent_type == AgentType.CONTROLLER
                    and agent.unique_id not in controller_ids):
                active_set.append(agent)

        n_events = 0
        for agent in active_set:
            try:
                agent.step()
                if (agent.pending_action is not None
                        and agent.pending_action.action_type != ActionType.SILENT):
                    n_events += 1
            except Exception as exc:
                _LOG.warning("[tick=%d] Agent-%s step 异常，已降级: %s",
                             int(t), agent.unique_id, exc)

        # ── ⑤ 环境更新 ───────────────────────────────────────────────
        self._update_environment()

        # ── ⑥ 数据收集 ───────────────────────────────────────────────
        self.datacollector.collect(self)

        # ── ⑦ 时步自增 + Hawkes 回写 ─────────────────────────────────
        self.schedule.steps += 1
        self.schedule.time += 1
        for _ in range(n_events):
            self.hawkes.add_event(t)

    # ================================================================== #
    #  ①  群负面程度（问题①的核心：让「负面」成为可越阈的量）              #
    # ================================================================== #
    def _refresh_group_negative(self) -> None:
        """
        用「群内存活消息 negative_score 的时间加权均值」刷新 topic_negative。

        旧实现只在 submit_action 里做 EMA(0.2) 累加，导致：
          - 群里没有消息时负面值也不回落；
          - 单条极端负面消息被稀释，永远越不过 0.65，
            于是「负面超阈触发干预」这一条设定形同虚设。

        现在的定义是可解释的：negative(k) ≈ 该群此刻正在流传的内容有多负面。
        """
        current_tick = int(self.schedule.time)
        acc: Dict[Tuple[str, GroupType], Tuple[float, float]] = {}

        for record in self.info_stream_cache:
            group_type = self.group_type_by_id.get(record.group_id)
            if group_type is None:
                continue
            age = max(0, current_tick - record.tick)
            # 权重 = 新鲜度 × 传播力 × 失真度。一条被转了 5 次的夸大消息，
            # 对「这个群现在有多负面」的贡献远大于一条没人理的日常发言。
            weight = (
                math.exp(-age / _NEGATIVE_RECENCY_TAU)
                * (1.0 + 0.8 * min(int(record.forward_count), 6))
                * (1.0 + 0.4 * float(record.distortion_level))
            )
            key = (record.topic_id or "T001", group_type)
            w_sum, wn_sum = acc.get(key, (0.0, 0.0))
            acc[key] = (w_sum + weight, wn_sum + weight * float(record.negative_score))

        smoothing = self.negative_smoothing
        for topic_id, group_map in self.topic_negative.items():
            for group_type in GroupType:
                w_sum, wn_sum = acc.get((topic_id, group_type), (0.0, 0.0))
                instant = (wn_sum / w_sum) if w_sum > 1e-12 else 0.0
                old = group_map[group_type]
                group_map[group_type] = float(np.clip(
                    (1.0 - smoothing) * old + smoothing * instant, 0.0, 1.0
                ))
        self._refresh_flat_views()

    # ================================================================== #
    #  ①  干预触发判定（问题①）                                           #
    # ================================================================== #
    def _evaluate_intervention_triggers(self) -> None:
        """
        按《场景设定》§3 判定各群的 t_k^int。

        默认 intervention_trigger="negative"：负面程度 ≥ negative_threshold 即触发，
        与热度无关 —— 这正是旧实现漏掉的「负面高但不热」场景。

        DORM 群 t_dorm^int = +∞，永不触发（私密性强，Controller 介入不了）。
        """
        pending = [g for g in GroupType
                   if g != GroupType.DORM and self.intervention_tick[g] is None]
        if not pending:
            return   # 所有可干预的群都已触发，无需再扫

        current_tick = int(self.schedule.time)
        controller_groups = self._get_controller_groups()

        # 各 (topic, group) 的存活消息条数 —— 干预判定的最小证据量
        msg_count: Dict[Tuple[str, GroupType], int] = {}
        for record in self.info_stream_cache:
            group_type = self.group_type_by_id.get(record.group_id)
            if group_type is None:
                continue
            key = (record.topic_id or "T001", group_type)
            msg_count[key] = msg_count.get(key, 0) + 1

        for topic_id, heat_map in self.topic_heat.items():
            negative_map = self.topic_negative.setdefault(
                topic_id, {g: 0.0 for g in GroupType}
            )
            for group_type in pending:
                if self.intervention_tick[group_type] is not None:
                    continue
                if self.intervention_requires_controller and group_type not in controller_groups:
                    continue
                if msg_count.get((topic_id, group_type), 0) < self.intervention_min_messages:
                    continue

                heat_hit = heat_map[group_type] >= self.heat_threshold
                negative_hit = negative_map[group_type] >= self.negative_threshold
                if self.intervention_trigger == "heat":
                    hit = heat_hit
                elif self.intervention_trigger == "negative":
                    hit = negative_hit
                elif self.intervention_trigger == "heat_and_negative":
                    hit = heat_hit and negative_hit
                else:
                    hit = heat_hit or negative_hit

                if hit:
                    self.intervention_tick[group_type] = current_tick
                    _LOG.info(
                        "[tick=%d] %s 群触发 Controller 干预 | 触发条件=%s "
                        "| H=%.3f(θ=%.2f) negative=%.3f(阈=%.2f) | control_level=%.2f",
                        current_tick, group_type.name, self.intervention_trigger,
                        heat_map[group_type], self.heat_threshold,
                        negative_map[group_type], self.negative_threshold,
                        GROUP_CONTROL_LEVEL[group_type],
                    )

    # ================================================================== #
    #  #16  _update_environment                                            #
    # ================================================================== #
    def _update_environment(self) -> None:
        """
        每步刷新：降温 → 宏观跨群溅射 → 恢复追踪 → 失真恢复 → 过期 → 解禁。

        降温公式（问题⑦，唯一权威）：
            H_k(t+1) = H_k(t) · exp(−α · control_level_k)
                     ≡ H_k(t) · exp(−α) · exp(−β_k · 𝟙[t ≥ t_k^int])
        """
        current_tick = int(self.schedule.time)

        # 触发判定是幂等的：直接调用 _update_environment() 的测试 / D 模块
        # 也能拿到正确的 t_k^int。
        self._evaluate_intervention_triggers()

        # ── ① 选出每个群用于调用 B.calc_heat_decay 的代表 Agent ────────
        decay_agents: Dict[GroupType, Any] = {}
        for agent in list(self.schedule.agents):
            if not hasattr(agent, "calc_heat_decay"):
                continue
            try:
                decay_agents.setdefault(agent.beliefs.identity.group_type, agent)
            except (AttributeError, TypeError):
                continue

        def _heat_decay(current_heat: float, group_type: GroupType) -> float:
            t_int = self.intervention_tick.get(group_type)
            intervened = (
                group_type != GroupType.DORM
                and t_int is not None
                and current_tick >= t_int
            )
            agent = decay_agents.get(group_type)
            if agent is not None:
                try:
                    return float(agent.calc_heat_decay(
                        current_heat=current_heat,
                        elapsed_steps=current_tick,
                        intervention_tick=t_int,
                    ))
                except Exception as exc:
                    _LOG.debug("%s 群调用 B.calc_heat_decay 失败，改用权威公式: %s",
                               group_type.name, exc)
            return _formula_heat_decay(current_heat, group_type, intervened)

        # ── ② 降温 ───────────────────────────────────────────────────
        for topic_id, group_heat in self.topic_heat.items():
            for group_type in GroupType:
                group_heat[group_type] = max(0.0, _heat_decay(group_heat[group_type], group_type))

            # ── ③ 宏观热度跨群溅射（问题③ + 问题⑥）──────────────────
            #    条件：源群热度 ≥ θ  且  该群转发次数已达跨群门槛；
            #    目标：概率与目标群「曝光」成正比 —— 高曝光群更易被渗透。
            for src_gt in GroupType:
                if group_heat[src_gt] < self.heat_threshold:
                    continue
                if self.macro_spread_requires_threshold and not self._forward_gate_open_for_group(src_gt):
                    continue
                for dst_gt in GroupType:
                    if dst_gt == src_gt:
                        continue
                    p = self.cross_group_spread_probability * self._exposure_factor(
                        dst_gt, self.exposure_spread_weight
                    )
                    if self.random.random() >= p:
                        continue
                    spread_amount = group_heat[src_gt] * 0.10 * self._exposure_factor(
                        dst_gt, self.exposure_spread_weight
                    )
                    group_heat[dst_gt] = min(
                        group_heat[dst_gt] + spread_amount,
                        min(group_heat[src_gt], self.heat_cap),
                    )
                    self.cross_group_spread += 1
                    _LOG.debug("[tick=%d] 热度跨群溅射 %s→%s topic=%s +%.3f",
                               current_tick, src_gt.name, dst_gt.name, topic_id, spread_amount)

        self._refresh_flat_views()

        # ── ④ recovery_time 追踪（《场景设定》§4.2.5）─────────────────
        max_heat = self._calc_group_heat_max()
        if max_heat >= self.heat_threshold and self._heat_exceeded_tick is None:
            self._heat_exceeded_tick = current_tick
            _LOG.info("[tick=%d] 热度首次超阈 max_heat=%.3f ≥ θ=%.2f",
                      current_tick, max_heat, self.heat_threshold)
        elif (max_heat < self.heat_threshold
              and self._heat_exceeded_tick is not None
              and self._recovery_time is None):
            self._recovery_time = current_tick - self._heat_exceeded_tick
            _LOG.info("[tick=%d] 热度全面回落，recovery_time=%d tick",
                      current_tick, self._recovery_time)

        # ── ⑤ 干预后的失真 / 负面恢复（《场景设定》§3 恢复机制）────────
        distortion_decay = _as_probability(
            self.scenario_params.get("intervention_distortion_decay"), 0.80)
        for record in self.info_stream_cache:
            group_type = self.group_type_by_id.get(record.group_id)
            t_gov = self.governance_tick.get(group_type) if group_type else None
            if (group_type is None or group_type == GroupType.DORM
                    or t_gov is None or current_tick < t_gov):
                continue
            # 管理越强（control_level 越大）恢复越快 —— 与降温同一套语义
            relief = math.exp(-ALPHA * (GROUP_CONTROL_LEVEL[group_type] - 1.0))
            record.distortion_level = float(np.clip(
                record.distortion_level * distortion_decay * relief, 0.0, 1.0))
            if record.message_type != MessageType.CLARIFICATION:
                record.negative_score = float(np.clip(
                    record.negative_score * distortion_decay, 0.0, 1.0))

        # ── ⑥ 过期消息淘汰 + 索引重建 ────────────────────────────────
        self.info_stream_cache = [
            r for r in self.info_stream_cache
            if current_tick - r.tick <= self.message_expire_ticks
        ]
        self._message_by_id = {
            r.message_id: r for r in self.info_stream_cache if r.message_id
        }

        # ── ⑦ 解除到期禁言（问题⑤）───────────────────────────────────
        expired = [k for k, until in self.muted_until.items() if current_tick >= until]
        for key in expired:
            self.muted_until.pop(key, None)

    # ================================================================== #
    #  #17 / #18 / #19  DataCollector 回调                                 #
    # ================================================================== #
    def _calc_avg_opinion(self) -> float:
        """全网 Agent 对主话题的平均观点值，范围 [-1, +1]。"""
        vals = self._collect_opinion_values()
        return float(np.mean(vals)) if vals else 0.0

    def _calc_polarization(self) -> float:
        """观点极化程度（标准差），值越大分裂越严重。"""
        vals = self._collect_opinion_values()
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    def _calc_emotional_contagion(self) -> float:
        """情绪传播速度：相邻两 tick 各 Agent arousal 变化量的均值。"""
        curr_snap: Dict[int, EmotionState] = {}
        for agent in list(self.schedule.agents):
            if hasattr(agent, "beliefs"):
                e = agent.beliefs.emotion
                curr_snap[agent.unique_id] = EmotionState(valence=e.valence, arousal=e.arousal)

        if not self._prev_emotion_snapshot:
            self._prev_emotion_snapshot = curr_snap
            return 0.0

        deltas = [
            abs(curr_snap[aid].arousal - self._prev_emotion_snapshot[aid].arousal)
            for aid in curr_snap if aid in self._prev_emotion_snapshot
        ]
        self._prev_emotion_snapshot = curr_snap
        return float(np.mean(deltas)) if deltas else 0.0

    # ================================================================== #
    #  #20  submit_action                                                  #
    # ================================================================== #
    def submit_action(
        self,
        record: ActionRecord,
        bypass_mute: bool = False,
        absolute_heat: bool = False,
    ) -> bool:
        """
        校验并投递一条行动记录。所有 Agent 行动、事件源投放、D 模块干预注入
        都必须走这一个入口。

        Parameters
        ----------
        record        : 待投递的行动记录
        bypass_mute   : True 时跳过禁言检查（事件源 / 官方注入用）
        absolute_heat : True 时把 record.heat 作为该群热度的绝对下限（H₀ 语义），
                        而不是按曝光加权的增量

        校验顺序：
          1. 发送者存在
          2. 目标群存在，且发送者是该群成员
          3. 禁言检查（问题⑤）
          4. 治理动作权限检查（MUTE / ANNOUNCE 仅 CONTROLLER）
          5. FORWARD：源消息必须真实存在且发送者可见
          6. FORWARD 跨群：转发次数必须达到跨群门槛（问题③）
          7. 转发方向必须在允许的方向集合内（问题④）
        """
        sender = self._get_agent_by_id(record.agent_id)
        if sender is None or not hasattr(sender, "beliefs"):
            _LOG.warning("拒绝未知发送者的行动: agent_id=%s", record.agent_id)
            return False

        member_group_ids = set(self.get_agent_groups(record.agent_id))
        destination_group_id = (
            record.group_id or self.get_agent_group(record.agent_id) or ""
        ).strip()
        if destination_group_id not in self.group_type_by_id:
            _LOG.warning("拒绝行动：目标群不存在 | agent=%s group_id=%r",
                         record.agent_id, destination_group_id)
            return False
        if destination_group_id not in member_group_ids:
            _LOG.warning("拒绝行动：发送者不是目标群成员 | agent=%s group_id=%s memberships=%s",
                         record.agent_id, destination_group_id, sorted(member_group_ids))
            return False

        destination_group_type = self.group_type_by_id[destination_group_id]

        # ── ③ 禁言检查（问题⑤）──────────────────────────────────────
        if not bypass_mute and self.is_muted(record.agent_id, destination_group_id):
            self.blocked_by_mute += 1
            _LOG.debug("拒绝行动：Agent-%s 在 %s 被禁言中",
                       record.agent_id, destination_group_id)
            return False

        # ── ④ 治理动作权限 ─────────────────────────────────────────
        sender_type = sender.beliefs.identity.agent_type
        if record.action_type in CONTROLLER_ONLY_ACTIONS:
            if sender_type != AgentType.CONTROLLER:
                _LOG.warning("拒绝治理动作：Agent-%s 不是 CONTROLLER", record.agent_id)
                return False
            if destination_group_type == GroupType.DORM:
                _LOG.debug("拒绝治理动作：DORM 群 control_level 最低，Controller 无法介入")
                return False
            if record.action_type == ActionType.MUTE and not self.enable_mute:
                return False

        # ── ⑤ FORWARD 源消息解析 ────────────────────────────────────
        source_record: Optional[ActionRecord] = None
        if record.action_type == ActionType.FORWARD:
            source_record = self._resolve_forward_source(record, member_group_ids)
            if source_record is None:
                _LOG.debug("拒绝 FORWARD：找不到发送者可见的真实源消息 | agent=%s src=%r",
                           record.agent_id, record.source_message_id)
                return False

            source_group_type = self.group_type_by_id[source_record.group_id]
            direction = self.forward_direction(source_group_type, destination_group_type)

            # ── ⑥ 跨群门槛（问题③）──────────────────────────────
            if direction != ForwardDirection.INTRA:
                if not self.is_cross_group_channel_open(
                        source_record, direction, destination_group_type):
                    self.blocked_by_forward_gate += 1
                    _LOG.debug(
                        "拒绝跨群 FORWARD：转发次数 %d < %s 门槛 %d 且源群未达热度阈 | agent=%s",
                        self.lineage_forward_total(source_record), direction,
                        self.forward_threshold_for(direction, destination_group_type),
                        record.agent_id,
                    )
                    return False
                if (direction == ForwardDirection.DOWNWARD
                        and not self.allow_downward_forward):
                    _LOG.debug("拒绝向下转发：allow_downward_forward=False")
                    return False
                if self.forward_direction_weights.get(direction, 0.0) <= 0.0:
                    return False

            record.source_message_id = source_record.message_id
            record.source_group_id = source_record.group_id
            record.target_id = source_record.agent_id
            record.forward_direction = direction
            record.root_message_id = source_record.root_message_id or source_record.message_id
            record.forward_count = int(source_record.forward_count) + 1
            if not record.topic_id:
                record.topic_id = source_record.topic_id
        else:
            record.source_group_id = destination_group_id
            record.forward_direction = ""
            record.forward_count = 0
            if record.action_type == ActionType.SEND_MESSAGE:
                record.source_message_id = None

        record.group_id = destination_group_id

        # ── 转发失真累积（《场景设定》§3「每转发一次 distortion 可能增加」）──
        if record.action_type == ActionType.FORWARD and source_record is not None:
            p_distort = _as_probability(
                self.scenario_params.get("forward_distortion_probability"), 0.70)
            increment = 0.0
            if self.random.random() < p_distort:
                lo = max(0.0, _as_float(
                    self.scenario_params.get("forward_distortion_increment_min"), 0.03))
                hi = max(lo, _as_float(
                    self.scenario_params.get("forward_distortion_increment_max"), 0.15))
                increment = self.random.uniform(lo, hi)
            record.distortion_level = float(np.clip(
                max(record.distortion_level, source_record.distortion_level) + increment, 0.0, 1.0))
            record.negative_score = float(np.clip(
                max(record.negative_score, source_record.negative_score) + 0.15 * increment,
                0.0, 1.0))

        # ── 分配消息 ID 并入流 ──────────────────────────────────────
        if not record.message_id:
            record.message_id = self._allocate_message_id()
        elif record.message_id in self._message_by_id:
            _LOG.warning("拒绝重复 message_id=%s", record.message_id)
            return False
        if not record.root_message_id:
            record.root_message_id = record.message_id

        topic_id = record.topic_id or "T001"
        record.topic_id = topic_id
        self._ensure_topic(topic_id)

        self.info_stream_cache.append(record)
        self._message_by_id[record.message_id] = record

        # ── 转发计数与方向指标（问题③④）────────────────────────────
        if record.action_type == ActionType.FORWARD and source_record is not None:
            self.lineage_forward_count[record.root_message_id] = (
                self.lineage_forward_count.get(record.root_message_id, 0) + 1
            )
            self.group_forward_count[record.group_id] = (
                self.group_forward_count.get(record.group_id, 0) + 1
            )
            if record.forward_direction == ForwardDirection.INTRA:
                self.intra_group_forward += 1
            else:
                self.cross_group_forward += 1
                if record.forward_direction == ForwardDirection.LATERAL:
                    self.lateral_forward += 1
                elif record.forward_direction == ForwardDirection.DOWNWARD:
                    self.downward_forward += 1
                else:
                    self.upward_forward += 1

        # ── 热度归入消息实际投递群，按曝光加权（问题⑥）──────────────
        self._apply_heat(record, destination_group_type, absolute_heat)

        # ── 治理动作的即时效果（问题⑤）──────────────────────────────
        if record.action_type == ActionType.MUTE:
            self._apply_mute(record, destination_group_id, destination_group_type)
            self._mark_governance(destination_group_type)
        elif record.action_type == ActionType.ANNOUNCE:
            self._apply_announce(record, destination_group_id, destination_group_type)
            self._mark_governance(destination_group_type)
        # 注意：只有 MUTE / ANNOUNCE 这两个显式治理动作才算「Controller 已干预」。
        # Controller 的日常发言 message_type 也是 clarification，但那不是干预，
        # 否则仿真一开始就会进入恢复期，干预效果的对照实验会全部失效。

        self._refresh_flat_views()
        return True

    # ------------------------------------------------------------------ #
    #  热度 / 治理动作的内部实现                                          #
    # ------------------------------------------------------------------ #
    def _apply_heat(
        self,
        record: ActionRecord,
        group_type: GroupType,
        absolute_heat: bool,
    ) -> None:
        """
        把一条消息的热度贡献写入目标群。

        - 事件源投放（absolute_heat=True）：H₀ 语义，直接把该群热度顶到 record.heat；
        - 普通消息：增量 = base_heat_gain × 强度 × 曝光因子 × 负面因子。
          曝光因子来自《场景设定》§1 的「曝光」列 —— 同一条消息在校园群
          掀起的热度显著高于在宿舍群。
        - 治理动作（MUTE / ANNOUNCE / clarification）不制造热度。
        """
        heat_map = self.topic_heat[record.topic_id]
        if absolute_heat:
            heat_map[group_type] = float(np.clip(
                max(heat_map[group_type], record.heat), 0.0, self.heat_cap))
            return

        if (record.action_type in CONTROLLER_ONLY_ACTIONS
                or record.message_type == MessageType.CLARIFICATION):
            return

        intensity = float(np.clip(record.heat if record.heat > 0 else 1.0, 0.0, 2.0))
        exposure_factor = self._exposure_factor(group_type, self.exposure_heat_weight)
        negative_factor = 0.60 + 0.80 * float(np.clip(record.negative_score, 0.0, 1.0))
        gain = self.base_heat_gain * intensity * exposure_factor * negative_factor
        heat_map[group_type] = float(np.clip(
            heat_map[group_type] + gain, 0.0, self.heat_cap))

    def _mark_governance(self, group_type: GroupType) -> None:
        """记录 Controller 首次在该群实际动手的 tick（《场景设定》§3 恢复机制起点）。"""
        if group_type == GroupType.DORM:
            return
        if self.governance_tick.get(group_type) is None:
            self.governance_tick[group_type] = int(self.schedule.time)
            _LOG.info("[tick=%d] %s 群 Controller 首次实际干预，进入恢复阶段",
                      int(self.schedule.time), group_type.name)

    def _apply_mute(
        self,
        record: ActionRecord,
        group_id: str,
        group_type: GroupType,
    ) -> None:
        """
        执行禁言（《场景设定》§2 Controller「禁言」）。

        禁言时长与群管理强度挂钩：control_level 越高的群，Controller 权限越大，
        禁言越久。DORM 群在 submit_action 里已被挡下。
        """
        target_id = record.target_id
        if target_id is None or target_id not in self.group_members.get(group_id, set()):
            _LOG.debug("MUTE 目标无效或不在该群: target=%s group=%s", target_id, group_id)
            return

        duration = record.mute_duration if record.mute_duration > 0 else self.mute_duration_ticks
        duration = max(1, int(round(duration * GROUP_CONTROL_LEVEL[group_type] / 2.0)))
        release_tick = int(self.schedule.time) + duration
        self.muted_until[(int(target_id), group_id)] = release_tick
        self.mute_count += 1
        record.mute_duration = duration

        # 禁言的即时治理效果：被禁言者在该群的存量消息迅速降失真 / 降负面
        for cached in self.info_stream_cache:
            if cached.group_id == group_id and cached.agent_id == target_id:
                cached.distortion_level = float(np.clip(cached.distortion_level * 0.5, 0.0, 1.0))
                cached.negative_score = float(np.clip(cached.negative_score * 0.5, 0.0, 1.0))

        negative_map = self.topic_negative[record.topic_id]
        negative_map[group_type] = float(np.clip(
            negative_map[group_type] * (1.0 - self.announce_negative_relief), 0.0, 1.0))

        _LOG.info("[tick=%d] %s 群 Controller-%s 禁言 Agent-%s %d tick（至 tick=%d）",
                  int(self.schedule.time), group_type.name, record.agent_id,
                  target_id, duration, release_tick)

    def _apply_announce(
        self,
        record: ActionRecord,
        group_id: str,
        group_type: GroupType,
    ) -> None:
        """执行公告（《场景设定》§2 Controller「公告」）：全群压降负面与失真。"""
        self.announce_count += 1
        for cached in self.info_stream_cache:
            if cached.group_id == group_id and cached.message_id != record.message_id:
                cached.distortion_level = float(np.clip(
                    cached.distortion_level * (1.0 - self.announce_distortion_relief), 0.0, 1.0))
                cached.negative_score = float(np.clip(
                    cached.negative_score * (1.0 - self.announce_negative_relief), 0.0, 1.0))
        negative_map = self.topic_negative[record.topic_id]
        negative_map[group_type] = float(np.clip(
            negative_map[group_type] * (1.0 - self.announce_negative_relief), 0.0, 1.0))
        _LOG.info("[tick=%d] %s 群 Controller-%s 发布公告",
                  int(self.schedule.time), group_type.name, record.agent_id)

    # ================================================================== #
    #  转发方向 / 跨群门槛（问题③④）                                      #
    # ================================================================== #
    @staticmethod
    def forward_direction(src: GroupType, dst: GroupType) -> str:
        """
        判定一次转发的方向语义。

            src == dst          → INTRA    群内转发（攒转发次数的唯一途径）
            DORM ↔ CLASS        → LATERAL  小群互转，流程图里那条横向箭头
            群规模变大          → UPWARD   小群 → 大群（主通道，含 MAJOR→CAMPUS）
            群规模变小          → DOWNWARD 大群 → 小群（最弱通道）
        """
        if src == dst:
            return ForwardDirection.INTRA
        if (src, dst) in LATERAL_PAIRS:
            return ForwardDirection.LATERAL
        return ForwardDirection.UPWARD if int(dst) > int(src) else ForwardDirection.DOWNWARD

    def lineage_forward_total(self, source_record: ActionRecord) -> int:
        """
        某条消息「已经被转发了多少次」。

        取两个口径的较大者：
          - source_record.forward_count：这条消息在转发链上的深度；
          - lineage_forward_count[root]：同一根消息在全网被转发的总次数。
        前者刻画链式接力，后者刻画广度扩散，任一达标即认为舆情已具备跨群动能。
        """
        root_id = source_record.root_message_id or source_record.message_id
        return max(
            int(source_record.forward_count),
            int(self.lineage_forward_count.get(root_id, 0)),
        )

    def forward_threshold_for(
        self,
        direction: str,
        destination_group_type: Optional[GroupType] = None,
    ) -> int:
        """该方向打开跨群通道所需的转发次数（问题③④）。"""
        if direction == ForwardDirection.INTRA:
            return 0
        if self.cross_group_forward_threshold <= 0:
            return 0
        return int(self.forward_threshold_by_direction.get(
            direction, self.cross_group_forward_threshold))

    def is_cross_group_channel_open(
        self,
        source_record: ActionRecord,
        direction: str = ForwardDirection.UPWARD,
        destination_group_type: Optional[GroupType] = None,
    ) -> bool:
        """
        流程图的「转发次数 ≥ 阈 → 进入跨群通道」判定，门槛按方向分级。

        额外放行条件（cross_group_heat_bypass）：源消息所在群热度已 ≥ θ 时，
        对应《场景设定》§3 的「消息如果热度高 → 有概率进入其他任意群」，
        此时不再要求转发次数。
        """
        threshold = self.forward_threshold_for(direction, destination_group_type)
        if threshold <= 0:
            return True
        if self.lineage_forward_total(source_record) >= threshold:
            return True
        if self.cross_group_heat_bypass:
            group_type = self.group_type_by_id.get(source_record.group_id)
            if group_type is not None:
                heat = self.topic_heat.get(source_record.topic_id, {}).get(group_type, 0.0)
                if heat >= self.heat_threshold:
                    return True
        return False

    def _forward_gate_open_for_group(self, group_type: GroupType) -> bool:
        """宏观溅射用：该群累计转发次数是否已达跨群门槛。"""
        if self.cross_group_forward_threshold <= 0:
            return True
        group_id = self.group_id_by_type[group_type]
        return self.group_forward_count.get(group_id, 0) >= self.cross_group_forward_threshold

    def get_forward_destination_candidates(
        self,
        agent_id: int,
        source_group_id: str,
        forward_count: Optional[int] = None,
        include_intra: bool = True,
    ) -> List[str]:
        """
        返回该成员可以把源消息投递到的群。

        Parameters
        ----------
        forward_count : 源消息已被转发次数。
                        None  → 不施加跨群门槛（结构性候选，供测试 / D 模块查询）
                        int   → 未达 cross_group_forward_threshold 时只返回群内通道
        include_intra : 是否包含「转发到本群」这一通道（默认包含，
                        它是攒够转发次数、打开跨群通道的唯一途径）
        """
        source_type = self.group_type_by_id.get(source_group_id)
        if source_type is None:
            return []

        member_groups = self.get_agent_groups(agent_id)

        candidates: List[str] = []
        for group_id in member_groups:
            group_type = self.group_type_by_id[group_id]
            direction = self.forward_direction(source_type, group_type)
            if direction == ForwardDirection.INTRA:
                if include_intra and self.forward_direction_weights[ForwardDirection.INTRA] > 0:
                    candidates.append(group_id)
                continue
            if (forward_count is not None
                    and int(forward_count) < self.forward_threshold_for(direction, group_type)):
                continue
            if direction == ForwardDirection.DOWNWARD and not self.allow_downward_forward:
                continue
            if self.forward_direction_weights.get(direction, 0.0) <= 0.0:
                continue
            candidates.append(group_id)

        return sorted(candidates, key=lambda gid: int(self.group_type_by_id[gid]))

    def select_forward_destination(
        self,
        agent_id: int,
        source_group_id: str,
        forward_count: Optional[int] = None,
    ) -> Optional[str]:
        """
        按方向权重 × 目标群曝光度加权抽样一个投递群。

        - strategy="weighted"（默认）：upward 1.0 / lateral 0.45 / downward 0.15 /
          intra 0.60，再乘目标群曝光因子。小群互转因此有真实概率发生（问题④），
          而不是被 next_larger 一刀切掉。
        - strategy="next_larger" / "largest" / "random"：保留旧行为，便于对照实验。
        """
        candidates = self.get_forward_destination_candidates(
            agent_id, source_group_id, forward_count=forward_count)
        if not candidates:
            return None

        if self.forward_destination_strategy == "largest":
            return candidates[-1]
        if self.forward_destination_strategy == "random":
            return self.random.choice(candidates)
        if self.forward_destination_strategy == "next_larger":
            source_type = self.group_type_by_id[source_group_id]
            larger = [c for c in candidates
                      if int(self.group_type_by_id[c]) > int(source_type)]
            return larger[0] if larger else candidates[0]

        # weighted（默认）
        source_type = self.group_type_by_id[source_group_id]
        weights: List[float] = []
        for group_id in candidates:
            group_type = self.group_type_by_id[group_id]
            direction = self.forward_direction(source_type, group_type)
            w = self.forward_direction_weights.get(direction, 0.0)
            w *= self._exposure_factor(group_type, self.forward_exposure_weight)
            weights.append(max(0.0, w))

        total = sum(weights)
        if total <= 0.0:
            return self.random.choice(candidates)
        r = self.random.random() * total
        acc = 0.0
        for group_id, w in zip(candidates, weights):
            acc += w
            if r <= acc:
                return group_id
        return candidates[-1]

    # ================================================================== #
    #  禁言查询（问题⑤）                                                   #
    # ================================================================== #
    def is_muted(self, agent_id: int, group_id: str) -> bool:
        """该 Agent 此刻在该群是否处于禁言状态。"""
        until = self.muted_until.get((int(agent_id), group_id))
        return until is not None and int(self.schedule.time) < until

    def get_muted_groups(self, agent_id: int) -> List[str]:
        """该 Agent 当前被禁言的所有群。"""
        return [
            group_id for group_id in self.get_agent_groups(agent_id)
            if self.is_muted(agent_id, group_id)
        ]

    def _count_active_mutes(self) -> int:
        current_tick = int(self.schedule.time)
        return sum(1 for until in self.muted_until.values() if current_tick < until)

    # ================================================================== #
    #  仿真时钟接口（供 social_agent._perceive 调用）                      #
    # ================================================================== #

    def get_clock_context(self, group_type: Optional[GroupType] = None) -> Dict[str, Any]:
        """
        返回当前 tick 的时钟上下文（time_slot / wall_hour / place / activity /
        group_activity_multiplier / topic_suitability）。

        供 social_agent._perceive() 填充 Perception 的时钟字段，以及 LLM
        prompt builder 读取「现在几点、在哪里、在做什么」的常识背景。

        Parameters
        ----------
        group_type : 目标群类型（影响 group_activity_multiplier）。
                     None 时返回 CAMPUS（全局均值群）的数据。
        """
        tick = int(self.schedule.time)
        gt = group_type if group_type is not None else GroupType.CAMPUS
        return self.sim_clock.clock_context(tick, gt)

    def get_time_slot(self) -> str:
        """返回当前 tick 的时段标签（TimeSlot 字符串）。"""
        return self.sim_clock.time_slot(int(self.schedule.time))

    def get_wall_hour(self) -> float:
        """返回当前 tick 对应的现实小时（24h 浮点）。"""
        return self.sim_clock.wall_hour(int(self.schedule.time))

    # ================================================================== #
    #  B 模块接口适配（供 social_agent._perceive 调用）                    #
    # ================================================================== #
    def get_agent_group(self, agent_id: int) -> Optional[str]:
        """返回 Agent 的主群 ID。"""
        agent = self._get_agent_by_id(agent_id)
        if agent and hasattr(agent, "beliefs"):
            identity = agent.beliefs.identity
            return getattr(identity, "primary_group_id", "") or self.group_id_for_type(
                identity.group_type)
        return None

    def get_agent_groups(self, agent_id: int) -> List[str]:
        """返回 Agent 的全部真实群成员关系。"""
        agent = self._get_agent_by_id(agent_id)
        if not agent or not hasattr(agent, "beliefs"):
            return []
        identity = agent.beliefs.identity
        group_ids = list(getattr(identity, "group_ids", []) or [])
        if not group_ids:
            group_ids = [self.group_id_for_type(identity.group_type)]
        return [gid for gid in group_ids if gid in self.group_type_by_id]

    def get_group_type(self, agent_id: int) -> GroupType:
        """返回 Agent 主群类型。"""
        agent = self._get_agent_by_id(agent_id)
        if agent and hasattr(agent, "beliefs"):
            return agent.beliefs.identity.group_type
        return GroupType.CLASS

    def get_group_type_by_id(self, group_id: str) -> Optional[GroupType]:
        """由真实群 ID 返回 GroupType。"""
        return self.group_type_by_id.get(group_id)

    def get_group_exposure(self, group_id: str) -> float:
        """返回群的曝光度（问题⑥，《场景设定》§1）。"""
        group_type = self.group_type_by_id.get(group_id)
        return GROUP_EXPOSURE[group_type] if group_type is not None else 0.0

    def get_group_control_level(self, group_id: str) -> float:
        """返回群的管理强度 control_level（问题⑦，《场景设定》§3）。"""
        group_type = self.group_type_by_id.get(group_id)
        return GROUP_CONTROL_LEVEL[group_type] if group_type is not None else 1.0

    def get_group_messages(
        self,
        agent_id: int,
        limit: int = 20,
        group_id: Optional[str] = None,
    ) -> List[ActionRecord]:
        """
        返回 Agent 可见的最近消息。

        可见性只由真实群成员关系决定；跨群消息必须由一次真实 FORWARD 在目标群
        中创建新记录才可见。

        【问题⑥】各群可见条数按曝光度分配：校园群（曝光高）信息流更长，
        宿舍群（曝光低）只能看到少量最新消息。
        """
        if limit <= 0:
            return []

        member_group_ids = set(self.get_agent_groups(agent_id))
        if group_id is not None:
            if group_id not in member_group_ids:
                return []
            visible_group_ids = [group_id]
        else:
            visible_group_ids = sorted(member_group_ids,
                                       key=lambda gid: int(self.group_type_by_id[gid]))
        if not visible_group_ids:
            return []

        # 【问题⑥】注意力配额按曝光度分摊，而不是「谁的消息新谁占满」。
        # 校园群（曝光 1.0）拿最多名额，宿舍群（曝光 0.2）也保底 1 条 ——
        # 否则小群消息永远被大群刷掉，小群互转与小→大扩散都无从发生。
        ew = self.exposure_perception_weight
        shares = []
        for gid in visible_group_ids:
            exposure = GROUP_EXPOSURE[self.group_type_by_id[gid]]
            shares.append((1.0 - ew) + ew * exposure)
        total_share = sum(shares) or 1.0
        caps: Dict[str, int] = {}
        for gid, share in zip(visible_group_ids, shares):
            caps[gid] = max(1, int(round(limit * share / total_share)))

        used: Dict[str, int] = {gid: 0 for gid in visible_group_ids}
        result: List[ActionRecord] = []
        for record in reversed(self.info_stream_cache):
            gid = record.group_id
            if gid not in used or used[gid] >= caps[gid]:
                continue
            used[gid] += 1
            result.append(record)
            if len(result) >= limit:
                break
        return result

    def get_topic_heat(self) -> Dict[str, float]:
        """返回各话题平均热度（扁平视图）。"""
        return dict(self._topic_heat_flat)

    def get_topic_negative(self) -> Dict[str, float]:
        """返回各话题平均负面程度（扁平视图）。"""
        return dict(self._topic_negative_flat)

    def get_topic_heat_by_group(self, agent_id: int) -> Dict[str, Dict[str, float]]:
        """返回该 Agent 所在各群的逐话题热度。"""
        result: Dict[str, Dict[str, float]] = {}
        for group_id in self.get_agent_groups(agent_id):
            group_type = self.group_type_by_id[group_id]
            result[group_id] = {
                topic_id: float(group_map.get(group_type, 0.0))
                for topic_id, group_map in self.topic_heat.items()
            }
        return result

    def get_topic_negative_by_group(self, agent_id: int) -> Dict[str, Dict[str, float]]:
        """返回该 Agent 所在各群的逐话题负面值。"""
        result: Dict[str, Dict[str, float]] = {}
        for group_id in self.get_agent_groups(agent_id):
            group_type = self.group_type_by_id[group_id]
            result[group_id] = {
                topic_id: float(group_map.get(group_type, 0.0))
                for topic_id, group_map in self.topic_negative.items()
            }
        return result

    def get_intervened_groups(self, agent_id: Optional[int] = None) -> List[str]:
        """返回已触发 Controller 干预的群 ID（可限定在某 Agent 的成员群内）。"""
        current_tick = int(self.schedule.time)
        scope = (self.get_agent_groups(agent_id) if agent_id is not None
                 else list(self.group_type_by_id))
        out = []
        for group_id in scope:
            group_type = self.group_type_by_id[group_id]
            t_int = self.intervention_tick.get(group_type)
            if t_int is not None and current_tick >= t_int:
                out.append(group_id)
        return out

    # ================================================================== #
    #  额外 DataCollector 指标                                             #
    # ================================================================== #
    def _calc_negative_emotion(self) -> float:
        """
        负面情绪指数 ∈ [0,1]：全网 Agent 负性效价的平均强度 (1 − valence)/2。

        旧实现用「valence < 0 的比例」，在舆情爆发期会迅速饱和到 1.0，
        看不出强度差异，也无法反映干预后的回落，因此改为连续量。
        0.5 = 全网情绪中性，> 0.5 = 整体偏负面。
        """
        agents = list(self.schedule.agents)
        vals = [(1.0 - a.beliefs.emotion.valence) / 2.0
                for a in agents if hasattr(a, "beliefs")]
        return float(np.clip(np.mean(vals), 0.0, 1.0)) if vals else 0.0

    def _calc_avg_distortion(self) -> float:
        """全群平均消息失真程度（近 20 条 ActionRecord 的 distortion_level 均值）。"""
        recent = self.info_stream_cache[-20:] if self.info_stream_cache else []
        return float(np.mean([r.distortion_level for r in recent])) if recent else 0.0

    def _calc_group_negative_max(self) -> float:
        """所有群 / 话题中的最大负面程度 —— 干预触发的直接观测量（问题①）。"""
        vals = [v for gm in self.topic_negative.values() for v in gm.values()]
        return float(max(vals)) if vals else 0.0

    def _calc_group_heat_max(self) -> float:
        """所有群 / 话题中的最大热度。"""
        vals = [v for gm in self.topic_heat.values() for v in gm.values()]
        return float(max(vals)) if vals else 0.0

    def _count_intervened_groups(self) -> int:
        current_tick = int(self.schedule.time)
        return sum(1 for gt, t in self.intervention_tick.items()
                   if gt != GroupType.DORM and t is not None and current_tick >= t)

    def _get_earliest_intervention_tick(self) -> float:
        """最早触发干预的 tick（排除 DORM）。未触发返回 float('inf')。"""
        ticks = [t for gt, t in self.intervention_tick.items()
                 if gt != GroupType.DORM and t is not None]
        return float(min(ticks)) if ticks else float("inf")

    def _collect_intervention_tick(self) -> float:
        """DataCollector 列：未触发用 -1.0 哨兵，避免 inf 污染 describe()。"""
        earliest = self._get_earliest_intervention_tick()
        return -1.0 if math.isinf(earliest) else earliest

    def _calc_recovery_time(self) -> float:
        """恢复耗时；尚未恢复返回 -1.0，避免与「0 tick 即恢复」混淆。"""
        return float(self._recovery_time) if self._recovery_time is not None else -1.0

    def get_final_summary(self) -> Dict[str, Any]:
        """仿真结束后的最终状态汇总（供 D / E 使用）。"""
        earliest = self._get_earliest_intervention_tick()
        return {
            "simulation_tick": int(self.schedule.time),
            "heat_exceeded_tick": self._heat_exceeded_tick,
            "recovery_status": "recovered" if self._recovery_time is not None else "not_recovered",
            "recovery_time": self._recovery_time,
            "earliest_intervention_tick": None if math.isinf(earliest) else int(earliest),
            "intervention_tick_by_group": {
                g.name: t for g, t in self.intervention_tick.items()},
            "control_level_by_group": {
                g.name: round(GROUP_CONTROL_LEVEL[g], 4) for g in GroupType},
            "exposure_by_group": {g.name: GROUP_EXPOSURE[g] for g in GroupType},
            "max_topic_heat": self._calc_group_heat_max(),
            "max_group_negative": self._calc_group_negative_max(),
            "cross_group_forward": int(self.cross_group_forward),
            "intra_group_forward": int(self.intra_group_forward),
            "lateral_forward": int(self.lateral_forward),
            "upward_forward": int(self.upward_forward),
            "downward_forward": int(self.downward_forward),
            "cross_group_spread": int(self.cross_group_spread),
            "blocked_by_forward_gate": int(self.blocked_by_forward_gate),
            "mute_count": int(self.mute_count),
            "announce_count": int(self.announce_count),
            "blocked_by_mute": int(self.blocked_by_mute),
            "active_message_cache_size": len(self.info_stream_cache),
            "event_source": self.event_source.summary(),
        }

    # ================================================================== #
    #  内部辅助                                                            #
    # ================================================================== #
    @staticmethod
    def group_id_for_type(group_type: GroupType) -> str:
        """群类型 → 群频道 ID。"""
        return f"GROUP_{group_type.name}"

    # 兼容旧调用名
    _group_id_for_type = group_id_for_type

    @staticmethod
    def _exposure_factor(group_type: GroupType, weight: float) -> float:
        """曝光因子：weight=0 时退化为 1（关闭曝光效应），weight=1 时等于曝光值。"""
        return (1.0 - weight) + weight * GROUP_EXPOSURE[group_type]

    def _ensure_topic(self, topic_id: str) -> None:
        if topic_id not in self.topic_heat:
            self.topic_heat[topic_id] = {g: 0.0 for g in GroupType}
            self.topic_negative[topic_id] = {g: 0.0 for g in GroupType}
            self._topic_heat_flat[topic_id] = 0.0
            self._topic_negative_flat[topic_id] = 0.0

    def _refresh_flat_views(self) -> None:
        for topic_id, group_map in self.topic_heat.items():
            self._topic_heat_flat[topic_id] = float(np.mean(list(group_map.values())))
        for topic_id, group_map in self.topic_negative.items():
            self._topic_negative_flat[topic_id] = float(np.mean(list(group_map.values())))

    def _allocate_message_id(self) -> str:
        self._message_seq += 1
        return f"M{self._message_seq:08d}"

    def _resolve_forward_source(
        self,
        record: ActionRecord,
        member_group_ids: set,
    ) -> Optional[ActionRecord]:
        """解析 FORWARD 的真实源消息，并验证发送者当下可见。"""
        source: Optional[ActionRecord] = None
        if record.source_message_id:
            source = self._message_by_id.get(record.source_message_id)

        # 兼容旧调用：target_id 曾被当作「原作者」。只允许从发送者真实可见的
        # 缓存中回溯，绝不再用作者主群直接猜来源群。
        if source is None and record.target_id is not None:
            source = next(
                (c for c in reversed(self.info_stream_cache)
                 if c.agent_id == record.target_id and c.group_id in member_group_ids),
                None,
            )

        if source is None or source.group_id not in member_group_ids:
            return None
        return source

    def _collect_opinion_values(self) -> List[float]:
        """从所有 Agent 提取对 T001 话题的观点值列表。"""
        vals: List[float] = []
        for agent in list(self.schedule.agents):
            if hasattr(agent, "beliefs") and agent.beliefs.opinions:
                op = agent.beliefs.opinions.get("T001")
                if op is None:
                    op = list(agent.beliefs.opinions.values())[0]
                vals.append(op.opinion_value)
        return vals

    def _get_agent_by_id(self, agent_id: int):
        """按 unique_id 查找 Agent（O(1)），绕开 Mesa 2/3.x 内部结构差异。"""
        agent = self._agent_dict.get(agent_id) if hasattr(self, "_agent_dict") else None
        if agent is not None:
            return agent
        for a in list(self.schedule.agents):
            if a.unique_id == agent_id:
                return a
        return None

    def _register_agent(self, agent) -> None:
        """将 agent 注册到 A 模块自维护的查找字典与群成员表。"""
        self._agent_dict[agent.unique_id] = agent
        if hasattr(agent, "beliefs"):
            identity = agent.beliefs.identity
            group_ids = list(getattr(identity, "group_ids", []) or [])
            if not group_ids:
                group_ids = [self.group_id_for_type(identity.group_type)]
            for group_id in group_ids:
                if group_id in self.group_members:
                    self.group_members[group_id].add(agent.unique_id)

    def _get_controller_groups(self) -> set:
        """返回拥有至少一个 CONTROLLER 的群类型集合（按 tick 缓存）。"""
        cache_tick = getattr(self, "_controller_cache_tick", None)
        current_tick = int(self.schedule.time)
        if cache_tick == current_tick and getattr(self, "_controller_cache", None) is not None:
            return self._controller_cache
        groups = set()
        for agent in list(self.schedule.agents):
            beliefs = getattr(agent, "beliefs", None)
            if beliefs is None or beliefs.identity.agent_type != AgentType.CONTROLLER:
                continue
            group_ids = list(getattr(beliefs.identity, "group_ids", []) or [])
            if not group_ids:
                group_ids = [self.group_id_for_type(beliefs.identity.group_type)]
            for group_id in group_ids:
                group_type = self.group_type_by_id.get(group_id)
                if group_type is not None:
                    groups.add(group_type)
        self._controller_cache_tick = current_tick
        self._controller_cache = groups
        return groups

    @staticmethod
    def _fallback_heat_decay(
        current_heat: float,
        group_type: GroupType,
        elapsed_steps: int,
        intervention_tick: Optional[int],
    ) -> float:
        """
        兼容保留：B 模块 calc_heat_decay 的本地降级实现。
        内部直接转调 types_def.heat_decay（全项目唯一权威公式）。
        """
        intervened = (
            group_type != GroupType.DORM
            and intervention_tick is not None
            and elapsed_steps >= intervention_tick
        )
        return _formula_heat_decay(current_heat, group_type, intervened)
