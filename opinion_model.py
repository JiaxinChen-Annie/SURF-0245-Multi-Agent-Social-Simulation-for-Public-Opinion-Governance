"""
opinion_model.py — 模型/环境层（接口表 #12–20）
--------------------------------------------------------------
负责人：A
当前阶段：W4

函数清单（接口表编号）：
    #12  __init__(config)
    #13  _build_social_network() → nx.Graph
    #14  _place_agents()
    #15  step()
    #16  _update_environment()        ← W4 重点：H_k 热度公式 + 跨群扩散
    #17  _calc_avg_opinion() → float
    #18  _calc_polarization() → float
    #19  _calc_emotional_contagion() → float
    #20  submit_action(record)        ← W4 重点：新增字段处理

【W4 主要变更】
  - submit_action 新增 topic_heat / topic_negative / cross_group_forward 统计
  - _update_environment 按 H_k(t+1)=H_k(t)·e^(-α)·e^(-β_k·𝟙[t≥t_k^int]) 刷新各群热度
  - 跨群扩散：H(t) ≥ THETA 时有概率向其他群传播
  - intervention_tick 由本模块统一维护（B 模块写的为参考值，A 为权威）
  - 所有 event_id → topic_id，枚举对齐新版

注意：B 模块的 calc_heat_decay() 是热度衰减的纯计算函数，
      A 模块在 _update_environment 中调用它，不自行实现衰减公式。
      若 B 模块尚未交付，使用本文件内置的 _fallback_heat_decay() 降级。
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Any

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
    MessageType, EmotionState, ALPHA, THETA, GROUP_BETA,
)

_LOG = logging.getLogger("OpinionModel")


class OpinionModel(Model):
    """
    多智能体舆情仿真 · 模型/环境层（A 模块）。

    场景：大学校园多群舆情扩散（DORM/CLASS/MAJOR/CAMPUS 四类群）。
    角色：ORDINARY / ACTIVE / RATIONAL / CONTROLLER。
    核心热度演化公式（由 B 模块 calc_heat_decay 提供，A 模块调用）：
        H_k(t+1) = H_k(t) · e^(-α) · e^(-β_k · 𝟙[t ≥ t_k^int])
    """

    # ================================================================== #
    #  #12  __init__                                                       #
    # ================================================================== #
    def __init__(self, config: SimConfig) -> None:
        """
        初始化调度器、社交网络、数据收集器、Hawkes 引擎；
        调用 _build_social_network、_place_agents。

        Parameters
        ----------
        config : SimConfig
        """
        super().__init__()
        self.config = config

        # ① 固定随机种子
        self.random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        # ② 调度器
        self.schedule = RandomActivation(self)

        # ③ 构建社交网络
        G = self._build_social_network()
        self.grid = NetworkGrid(G)

        # ④ 环境状态 —— A 模块权威维护
        #    info_stream_cache: 信息流缓存，供下一 tick 邻居 _perceive 读取
        self.info_stream_cache: List[ActionRecord] = []

        #    topic_heat: {topic_id: {group_type: heat}}  各群各话题热度
        self.topic_heat: Dict[str, Dict[GroupType, float]] = {
            "T001": {g: 0.0 for g in GroupType}
        }
        #    topic_heat 扁平视图（供 B 模块 _perceive 读取）：{topic_id: heat_avg}
        self._topic_heat_flat: Dict[str, float] = {"T001": 0.0}

        #    topic_negative: {topic_id: {group_type: negative_score}}
        self.topic_negative: Dict[str, Dict[GroupType, float]] = {
            "T001": {g: 0.0 for g in GroupType}
        }
        self._topic_negative_flat: Dict[str, float] = {"T001": 0.0}

        #    intervention_tick: {group_type: Optional[int]}  各群首次干预时刻（A 为权威）
        self.intervention_tick: Dict[GroupType, Optional[int]] = {
            GroupType.DORM:   None,   # DORM 永不触发，保持 None（等价 +∞）
            GroupType.CLASS:  None,
            GroupType.MAJOR:  None,
            GroupType.CAMPUS: None,
        }

        #    cross_group_forward: 跨群转发次数统计
        self.cross_group_forward: int = 0

        #    情绪快照（供 _calc_emotional_contagion 使用）
        self._prev_emotion_snapshot: Dict[int, EmotionState] = {}

        # ⑤ Hawkes 引擎（C 模块）
        from hawkes_engine import HawkesEngine
        self.hawkes = HawkesEngine(
            mu    = config.hawkes_params.get("mu",    0.1),
            alpha = config.hawkes_params.get("alpha", 0.5),
            beta  = config.hawkes_params.get("beta",  1.0),
        )

        # ⑥ DataCollector（E 模块从此处读取输出）
        #    指标对齐接口表§5：message_count / avg_opinion / polarization /
        #    negative_emotion / distortion_level / cross_group_forward /
        #    intervention_tick / recovery_time
        self.datacollector = DataCollector(
            model_reporters={
                "avg_opinion":          lambda m: m._calc_avg_opinion(),
                "polarization":         lambda m: m._calc_polarization(),
                "emotional_contagion":  lambda m: m._calc_emotional_contagion(),
                "message_count":        lambda m: len(m.info_stream_cache),
                "negative_emotion":     lambda m: m._calc_negative_emotion(),
                "distortion_level":     lambda m: m._calc_avg_distortion(),
                "cross_group_forward":  lambda m: m.cross_group_forward,
                "intervention_tick":    lambda m: m._get_earliest_intervention_tick(),
                "recovery_time":        lambda m: m._calc_recovery_time(),
            }
        )

        # ⑦ 热度回落追踪（用于 recovery_time 计算）
        self._heat_exceeded_tick: Optional[int] = None   # 热度首次超 THETA 的 tick
        self._recovery_time: Optional[int] = None        # 热度回落后记录

        # ⑧ 放置智能体
        self._place_agents()

        _LOG.info(
            f"OpinionModel 初始化完成 | "
            f"n_agents={config.n_agents} | "
            f"network={config.network_type} | "
            f"mesa={'系统' if _MESA_REAL else '垫片'}"
        )

    # ================================================================== #
    #  #13  _build_social_network                                          #
    # ================================================================== #
    def _build_social_network(self) -> nx.Graph:
        """
        生成社交网络（模拟校园群内关注/好友关系）。

        支持：
          - barabasi_albert：无标度网络（默认），参数 m
          - watts_strogatz：小世界网络，参数 k/p

        Returns
        -------
        nx.Graph
        """
        n      = self.config.n_agents
        ntype  = self.config.network_type
        params = self.config.network_params
        seed   = self.config.random_seed

        if ntype == "barabasi_albert":
            m = params.get("m", 3)
            if n == 1:
                G = nx.empty_graph(1)
                _LOG.info("BA 网络 n=1，退化为单节点孤立图")
            elif n <= m:
                G = nx.complete_graph(n)
                _LOG.warning(f"BA 网络要求 m < n，n={n}<=m={m}，退化为完全图")
            else:
                G = nx.barabasi_albert_graph(n, m, seed=seed)
                _LOG.info(
                    f"BA 无标度网络 n={n} m={m} | "
                    f"平均度={2 * G.number_of_edges() / n:.2f}"
                )

        elif ntype == "watts_strogatz":
            k = params.get("k", 6)
            p = params.get("p", 0.1)
            if n <= k:
                G = nx.complete_graph(n)
                _LOG.warning(f"WS 网络 n={n}<=k={k}，退化为完全图")
            else:
                G = nx.watts_strogatz_graph(n, k, p, seed=seed)
                _LOG.info(f"WS 小世界网络 n={n} k={k} p={p}")

        else:
            _LOG.warning(f"未知 network_type='{ntype}'，回退到 BA(m=3)")
            m_fb = min(3, max(1, n - 1))
            G = nx.barabasi_albert_graph(n, m_fb, seed=seed) if n > 1 else nx.empty_graph(1)

        return G

    # ================================================================== #
    #  #14  _place_agents                                                  #
    # ================================================================== #
    def _place_agents(self) -> None:
        """
        将四类智能体（ORDINARY/ACTIVE/RATIONAL/CONTROLLER）按比例部署到网络节点。
        同时按 group_type_ratio 为每个 agent 分配群类型（GroupType）。
        实例化时须同时传入 agent_type 与 group_type（接口表 #14 要求）。
        GROUP_BETA 由 SocialAgent 内部读取，A 模块无需传入。
        """
        from social_agent import SocialAgent

        n      = self.config.n_agents
        ratio  = self.config.agent_type_ratio
        nodes  = list(self.grid.G.nodes())

        # ── 计算各 AgentType 数量 ──────────────────────────────────────
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

        # ── 计算各 GroupType 数量 ──────────────────────────────────────
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

        # 展开 group_type 列表并打乱（随机分配群）
        group_assignments: List[GroupType] = []
        for gt, cnt in group_counts.items():
            group_assignments.extend([gt] * cnt)
        self.random.shuffle(group_assignments)

        # ── 逐个实例化并挂载 ───────────────────────────────────────────
        agent_id = 0
        for agent_type, count in counts.items():
            for _ in range(count):
                group_type = group_assignments[agent_id]
                init_config: Dict[str, Any] = {
                    "agent_type":   agent_type,
                    "group_type":   group_type,
                    "stance_prior": self.random.uniform(-1.0, 1.0),
                    "topic_id":     "T001",
                    "initial_heat": 0.5,  # H₀ 初始热度
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
                agent_id += 1

        dist_str = " | ".join(f"{t.name}:{c}" for t, c in counts.items())
        gdist_str = " | ".join(f"{g.name}:{c}" for g, c in group_counts.items())
        _LOG.info(f"AgentType 分布: {dist_str}")
        _LOG.info(f"GroupType 分布: {gdist_str}")

    # ================================================================== #
    #  #15  step                                                           #
    # ================================================================== #
    def step(self) -> None:
        """
        单步主循环。

        执行顺序：
        1. Hawkes 采样 → 计算激活比例 [10%, 80%]
        2. 随机采样 n_active 个 Agent，逐个调用 agent.step()
           异常降级：记 warning，不中断整步
        3. _update_environment()
        4. datacollector.collect(self)
        5. schedule.time / schedule.steps 自增
        6. 本 tick 非 SILENT 行动数回写 HawkesEngine
        """
        t = float(self.schedule.time)

        # ── ① Hawkes 激活比例 ─────────────────────────────────────────
        lam = self.hawkes.intensity(t)
        mu  = self.config.hawkes_params.get("mu", 0.1)
        activation_rate = float(np.clip(0.30 * lam / max(mu, 1e-9), 0.10, 0.80))

        all_agents = self.schedule.agents
        n_active   = max(1, int(len(all_agents) * activation_rate))
        active_set = self.random.sample(all_agents, k=n_active)

        # ── ② 激活 Agent ──────────────────────────────────────────────
        n_events = 0
        for agent in active_set:
            try:
                agent.step()
                if (agent.pending_action is not None
                        and agent.pending_action.action_type != ActionType.SILENT):
                    n_events += 1
            except Exception as exc:
                _LOG.warning(
                    f"[tick={int(t)}] Agent-{agent.unique_id} step 异常，已降级: {exc}"
                )

        # ── ③ 环境更新（W4 重点）──────────────────────────────────────
        self._update_environment()

        # ── ④ 数据收集 ────────────────────────────────────────────────
        self.datacollector.collect(self)

        # ── ⑤ 时步自增 ───────────────────────────────────────────────
        self.schedule.steps += 1
        self.schedule.time  += 1

        # ── ⑥ 回写 Hawkes ────────────────────────────────────────────
        for _ in range(n_events):
            self.hawkes.add_event(t)

    # ================================================================== #
    #  #16  _update_environment                                            #
    # ================================================================== #
    def _update_environment(self) -> None:
        """
        W4 核心：每步刷新各群热度、跨群扩散、intervention_tick、淘汰过期信息流。

        热度演化公式（调用 B 模块 calc_heat_decay；B 未就绪时降级到内置）：
            H_k(t+1) = H_k(t) · e^(-α) · e^(-β_k · 𝟙[t ≥ t_k^int])

        跨群扩散：当任一群的 H_k(t) ≥ THETA，以概率 p_spread 向其他群传播。

        intervention_tick：H_k(t) ≥ THETA 且该群存在 CONTROLLER 且未记录时，
                           写入 intervention_tick[group_type] = current_tick。
        """
        current_tick = int(self.schedule.time)
        EXPIRE_TICKS = 10
        SPREAD_PROB  = 0.15   # 跨群扩散概率（每步每个热话题）

        # ── ① 尝试获取 B 模块的 calc_heat_decay ──────────────────────
        # B 模块将 calc_heat_decay 定义在 SocialAgent 上，
        # A 模块通过取第一个 agent 实例来调用（若存在）。
        _b_agent = self.schedule.agents[0] if self.schedule.agents else None
        _has_b_decay = _b_agent is not None and hasattr(_b_agent, "calc_heat_decay")

        def _heat_decay(current_heat: float, group_type: GroupType) -> float:
            """调用 B 模块 calc_heat_decay 或本地降级。"""
            elapsed = current_tick  # elapsed_steps 即当前 tick
            t_int   = self.intervention_tick.get(group_type, None)
            if _has_b_decay:
                return _b_agent.calc_heat_decay(
                    current_heat=current_heat,
                    elapsed_steps=elapsed,
                    intervention_tick=t_int,
                )
            else:
                return self._fallback_heat_decay(current_heat, group_type, elapsed, t_int)

        # ── ② 更新各话题各群热度 ──────────────────────────────────────
        #    同时检测 intervention_tick 触发条件
        controller_groups = self._get_controller_groups()

        for topic_id, group_heat in self.topic_heat.items():
            for group_type in GroupType:
                old_heat = group_heat[group_type]
                new_heat = _heat_decay(old_heat, group_type)

                # 检测 CONTROLLER 干预触发（H(t) ≥ θ，该群有 CONTROLLER，且未记录）
                if (group_type != GroupType.DORM
                        and old_heat >= THETA
                        and group_type in controller_groups
                        and self.intervention_tick[group_type] is None):
                    self.intervention_tick[group_type] = current_tick
                    _LOG.info(
                        f"[tick={current_tick}] {group_type.name} 群触发干预 "
                        f"H={old_heat:.3f} ≥ θ={THETA}"
                    )

                group_heat[group_type] = max(0.0, new_heat)

            # ── ③ 跨群扩散 ────────────────────────────────────────────
            for src_gt in GroupType:
                if group_heat[src_gt] >= THETA:
                    for dst_gt in GroupType:
                        if dst_gt != src_gt:
                            if self.random.random() < SPREAD_PROB:
                                # 热度传播（取较小值叠加，避免无限膨胀）
                                spread_amount = group_heat[src_gt] * 0.1
                                group_heat[dst_gt] = min(
                                    group_heat[dst_gt] + spread_amount,
                                    group_heat[src_gt],
                                )
                                self.cross_group_forward += 1
                                _LOG.debug(
                                    f"[tick={current_tick}] 跨群扩散 "
                                    f"{src_gt.name}→{dst_gt.name} "
                                    f"topic={topic_id} +{spread_amount:.3f}"
                                )

        # ── ④ 刷新扁平视图（供 B 模块 _perceive 读取）────────────────
        for topic_id in self.topic_heat:
            vals = list(self.topic_heat[topic_id].values())
            self._topic_heat_flat[topic_id]    = float(np.mean(vals))
            neg_vals = list(self.topic_negative[topic_id].values())
            self._topic_negative_flat[topic_id] = float(np.mean(neg_vals))

        # ── ⑤ 更新 recovery_time 追踪 ─────────────────────────────────
        avg_heat = float(np.mean([
            v for gd in self.topic_heat.values() for v in gd.values()
        ])) if self.topic_heat else 0.0

        if avg_heat >= THETA and self._heat_exceeded_tick is None:
            self._heat_exceeded_tick = current_tick
        elif avg_heat < THETA and self._heat_exceeded_tick is not None:
            if self._recovery_time is None:
                self._recovery_time = current_tick - self._heat_exceeded_tick
                _LOG.info(
                    f"[tick={current_tick}] 热度回落，recovery_time={self._recovery_time}"
                )

        # ── ⑥ 淘汰过期信息流 ──────────────────────────────────────────
        self.info_stream_cache = [
            r for r in self.info_stream_cache
            if current_tick - r.tick <= EXPIRE_TICKS
        ]

    # ================================================================== #
    #  #17  _calc_avg_opinion                                              #
    # ================================================================== #
    def _calc_avg_opinion(self) -> float:
        """
        全网 Agent 对 T001 话题的平均观点值。
        范围 [-1, +1]，0 代表中立。
        """
        vals = self._collect_opinion_values()
        return float(np.mean(vals)) if vals else 0.0

    # ================================================================== #
    #  #18  _calc_polarization                                             #
    # ================================================================== #
    def _calc_polarization(self) -> float:
        """
        观点极化程度（标准差）。
        范围 [0, ~1]；值越大，观点分裂越严重。
        """
        vals = self._collect_opinion_values()
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    # ================================================================== #
    #  #19  _calc_emotional_contagion                                      #
    # ================================================================== #
    def _calc_emotional_contagion(self) -> float:
        """
        情绪传播速度：相邻两 tick 间各 Agent arousal 变化量的均值。
        第一次调用时无前一快照，返回 0.0。
        范围 [0, 1]。
        """
        agents = self.schedule.agents

        curr_snap: Dict[int, EmotionState] = {}
        for agent in agents:
            if hasattr(agent, "beliefs"):
                e = agent.beliefs.emotion
                curr_snap[agent.unique_id] = EmotionState(
                    valence=e.valence,
                    arousal=e.arousal,
                )

        if not self._prev_emotion_snapshot:
            self._prev_emotion_snapshot = curr_snap
            return 0.0

        deltas = [
            abs(curr_snap[aid].arousal - self._prev_emotion_snapshot[aid].arousal)
            for aid in curr_snap
            if aid in self._prev_emotion_snapshot
        ]
        self._prev_emotion_snapshot = curr_snap

        return float(np.mean(deltas)) if deltas else 0.0

    # ================================================================== #
    #  #20  submit_action（W4 重写）                                        #
    # ================================================================== #
    def submit_action(self, record: ActionRecord) -> None:
        """
        将智能体行动写入环境信息流缓存，供下一 tick 邻居 _perceive 读取。
        同时更新 topic_heat / topic_negative / cross_group_forward 统计。

        W4 新增：
          - 处理 record.heat / record.negative_score / record.message_type
          - 若 source 与 target 属不同群，cross_group_forward 计数 +1
            （此处基于 message_type=FORWARD 且有 target_id 时判断）

        Parameters
        ----------
        record : ActionRecord
        """
        # ① 写入缓存
        self.info_stream_cache.append(record)

        topic_id = record.topic_id

        # ② 确保 topic 存在于热度字典
        if topic_id not in self.topic_heat:
            self.topic_heat[topic_id]     = {g: 0.0 for g in GroupType}
            self.topic_negative[topic_id] = {g: 0.0 for g in GroupType}
            self._topic_heat_flat[topic_id]    = 0.0
            self._topic_negative_flat[topic_id] = 0.0

        # ③ 找到发送方所在群
        src_agent = self._get_agent_by_id(record.agent_id)
        src_group = (src_agent.beliefs.identity.group_type
                     if src_agent and hasattr(src_agent, "beliefs")
                     else GroupType.CLASS)

        # ④ 更新该群 topic_heat（叠加 record.heat）
        heat_delta = record.heat if record.heat > 0 else 0.1  # 最小贡献
        self.topic_heat[topic_id][src_group] = min(
            self.topic_heat[topic_id][src_group] + heat_delta,
            10.0,   # 热度上限，防止无界增长
        )

        # ⑤ 更新该群 topic_negative（指数平滑）
        SMOOTH = 0.2
        old_neg = self.topic_negative[topic_id][src_group]
        self.topic_negative[topic_id][src_group] = float(
            (1 - SMOOTH) * old_neg + SMOOTH * record.negative_score
        )

        # ⑥ 跨群转发判断：FORWARD 类型 + 有 target_id → 检查目标所在群
        if (record.message_type == MessageType.FORWARD
                and record.target_id is not None):
            tgt_agent = self._get_agent_by_id(record.target_id)
            if tgt_agent and hasattr(tgt_agent, "beliefs"):
                tgt_group = tgt_agent.beliefs.identity.group_type
                if tgt_group != src_group:
                    self.cross_group_forward += 1

        # ⑦ 刷新扁平视图
        vals = list(self.topic_heat[topic_id].values())
        self._topic_heat_flat[topic_id] = float(np.mean(vals))
        neg_vals = list(self.topic_negative[topic_id].values())
        self._topic_negative_flat[topic_id] = float(np.mean(neg_vals))

    # ================================================================== #
    #  B 模块接口适配（供 social_agent._perceive 调用）                    #
    # ================================================================== #

    def get_agent_group(self, agent_id: int) -> Optional[str]:
        """返回 agent 所在群 ID（格式：'GROUP_{GroupType.name}'）。"""
        agent = self._get_agent_by_id(agent_id)
        if agent and hasattr(agent, "beliefs"):
            gt = agent.beliefs.identity.group_type
            return f"GROUP_{gt.name}"
        return None

    def get_group_type(self, agent_id: int) -> GroupType:
        """返回 agent 的群类型。"""
        agent = self._get_agent_by_id(agent_id)
        if agent and hasattr(agent, "beliefs"):
            return agent.beliefs.identity.group_type
        return GroupType.CLASS

    def get_group_messages(self, agent_id: int, limit: int = 20) -> List[ActionRecord]:
        """返回与 agent 同群的最近消息（供 B 模块 _perceive 使用）。"""
        src_group = self.get_group_type(agent_id)
        result = []
        for record in reversed(self.info_stream_cache):
            rec_agent = self._get_agent_by_id(record.agent_id)
            if rec_agent and hasattr(rec_agent, "beliefs"):
                if rec_agent.beliefs.identity.group_type == src_group:
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

    # ================================================================== #
    #  额外 DataCollector 指标                                             #
    # ================================================================== #

    def _calc_negative_emotion(self) -> float:
        """负面情绪指数：valence < 0 的 agent 比例。"""
        agents = self.schedule.agents
        if not agents:
            return 0.0
        neg_count = sum(
            1 for a in agents
            if hasattr(a, "beliefs") and a.beliefs.emotion.valence < 0
        )
        return neg_count / len(agents)

    def _calc_avg_distortion(self) -> float:
        """全群平均消息失真程度（近 20 条 ActionRecord 的 distortion_level 均值）。"""
        recent = self.info_stream_cache[-20:] if self.info_stream_cache else []
        if not recent:
            return 0.0
        return float(np.mean([r.distortion_level for r in recent]))

    def _get_earliest_intervention_tick(self) -> float:
        """
        返回最早触发干预的 tick（排除 DORM 和未触发群）。
        若无任何群触发，返回 float('inf')。
        """
        ticks = [
            t for gt, t in self.intervention_tick.items()
            if gt != GroupType.DORM and t is not None
        ]
        return float(min(ticks)) if ticks else float("inf")

    def _calc_recovery_time(self) -> float:
        """返回已记录的 recovery_time；未恢复时返回 0.0。"""
        return float(self._recovery_time) if self._recovery_time is not None else 0.0

    # ================================================================== #
    #  内部辅助                                                            #
    # ================================================================== #

    def _collect_opinion_values(self) -> List[float]:
        """从所有 Agent 提取对 T001 话题的观点值列表。"""
        vals: List[float] = []
        for agent in self.schedule.agents:
            if hasattr(agent, "beliefs") and agent.beliefs.opinions:
                op = agent.beliefs.opinions.get("T001")
                if op is None:
                    op = list(agent.beliefs.opinions.values())[0]
                vals.append(op.opinion_value)
        return vals

    def _get_agent_by_id(self, agent_id: int):
        """按 unique_id 查找 Agent；O(1)（依赖调度器内部字典）。"""
        # RandomActivation 存储在 _agents 字典中
        if hasattr(self.schedule, "_agents"):
            return self.schedule._agents.get(agent_id)
        # 降级：线性扫描
        for a in self.schedule.agents:
            if a.unique_id == agent_id:
                return a
        return None

    def _get_controller_groups(self) -> set:
        """返回拥有至少一个 CONTROLLER agent 的群类型集合。"""
        groups = set()
        for agent in self.schedule.agents:
            if (hasattr(agent, "beliefs")
                    and agent.beliefs.identity.agent_type == AgentType.CONTROLLER):
                groups.add(agent.beliefs.identity.group_type)
        return groups

    @staticmethod
    def _fallback_heat_decay(
        current_heat: float,
        group_type: GroupType,
        elapsed_steps: int,
        intervention_tick: Optional[int],
    ) -> float:
        """
        B 模块 calc_heat_decay 的本地降级实现（B 未交付时使用）。
        公式：H_k(t+1) = H_k(t) · e^(-α) · e^(-β_k · 𝟙[t ≥ t_k^int])
        """
        beta_k = GROUP_BETA[group_type]
        natural_decay = math.exp(-ALPHA)

        if group_type == GroupType.DORM:
            intervention_decay = 1.0
        elif intervention_tick is not None and elapsed_steps >= intervention_tick:
            intervention_decay = math.exp(-beta_k)
        else:
            intervention_decay = 1.0

        return max(0.0, current_heat * natural_decay * intervention_decay)
