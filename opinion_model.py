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

【多群与真实转发修复】
  - IdentityBelief.group_ids 表示真实多群成员关系；group_type 仅保留主群兼容语义
  - 默认采用层级成员关系：DORM→CLASS→MAJOR→CAMPUS
  - ActionRecord.group_id 是真实投递群；FORWARD 用 source_message_id/source_group_id 追踪来源
  - 消息仅对目标群成员可见，不再用随机 cross_group_visibility 泄漏异群消息
  - cross_group_forward 只统计 source_group_id != group_id 的真实 Agent FORWARD
  - 宏观热度扩散单独统计为 cross_group_spread，不再污染转发指标
  - 热度归入消息目标群，衰减按群选择对应 B Agent

【W4 LLM 接入变更】
  - __init__ 新增 ⑤-b：读取 config.llm_config，调用 C 模块 setup_llm_client
    初始化 self._llm_client，挂载到模型实例供所有 SocialAgent 通过
    self.model._llm_client 取用
  - llm_config 为空 {} 时 _llm_client = None，SocialAgent 自动降级规则存根
  - llm_config 含 fallback_mock: true 时，初始化失败也不崩溃

注意：B 模块的 calc_heat_decay() 是热度衰减的纯计算函数，
      A 模块在 _update_environment 中调用它，不自行实现衰减公式。
      若 B 模块尚未交付，使用本文件内置的 _fallback_heat_decay() 降级。
"""

from __future__ import annotations
import mesa_patch  # Mesa 3.x 兼容补丁

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
    EmotionState, ALPHA, THETA, GROUP_BETA,
)

_LOG = logging.getLogger("OpinionModel")


def _as_probability(value: Any, default: float) -> float:
    """把配置值安全转换到 [0, 1]；非法值回退到 default。"""
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return default


class OpinionModel(Model):
    """
    多智能体舆情仿真 · 模型/环境层（A 模块）。

    场景：大学校园多群舆情扩散（DORM/CLASS/MAJOR/CAMPUS 四类群）。
    角色：ORDINARY / ACTIVE / RATIONAL / CONTROLLER。
    核心热度演化公式（由 B 模块 calc_heat_decay 提供，A 模块调用）：
        H_k(t+1) = H_k(t) · e^(-α) · e^(-β_k · 𝟙[t ≥ t_k^int])\n
    LLM 客户端（C 模块）挂载在 self._llm_client，
    供所有 SocialAgent 通过 self.model._llm_client 访问。
    """

    DEFAULT_CROSS_GROUP_SPREAD_PROBABILITY = 0.15
    DEFAULT_GROUP_MEMBERSHIP_MODE = "hierarchical"
    DEFAULT_FORWARD_DESTINATION_STRATEGY = "next_larger"

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

        # 群成员与跨群行为参数。沿用 network_params 承载，避免破坏 SimConfig 接口。
        network_params = config.network_params or {}
        if "cross_group_visibility" in network_params:
            _LOG.warning(
                "cross_group_visibility 已弃用并被忽略；异群信息必须通过真实 FORWARD 投递"
            )
        self.group_membership_mode = str(
            network_params.get("group_membership_mode", self.DEFAULT_GROUP_MEMBERSHIP_MODE)
        ).strip().lower()
        if self.group_membership_mode not in {"hierarchical", "single"}:
            _LOG.warning(
                "未知 group_membership_mode=%r，回退 hierarchical",
                self.group_membership_mode,
            )
            self.group_membership_mode = self.DEFAULT_GROUP_MEMBERSHIP_MODE

        self.forward_destination_strategy = str(
            network_params.get(
                "forward_destination_strategy",
                self.DEFAULT_FORWARD_DESTINATION_STRATEGY,
            )
        ).strip().lower()
        self.allow_downward_forward = bool(
            network_params.get("allow_downward_forward", False)
        )
        self.cross_group_spread_probability = _as_probability(
            network_params.get(
                "cross_group_spread_probability",
                self.DEFAULT_CROSS_GROUP_SPREAD_PROBABILITY,
            ),
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
        #    info_stream_cache: 信息流缓存，供下一 tick 邻居 _perceive 读取
        self.info_stream_cache: List[ActionRecord] = []

        #    群注册表。当前场景每种 GroupType 对应一个真实群频道；Agent 可多群成员。
        self.group_type_by_id: Dict[str, GroupType] = {
            self._group_id_for_type(group_type): group_type
            for group_type in GroupType
        }
        self.group_members: Dict[str, set] = {
            group_id: set() for group_id in self.group_type_by_id
        }

        #    消息索引：FORWARD 必须引用一条真实存在且发送者可见的源消息。
        self._message_seq: int = 0
        self._message_by_id: Dict[str, ActionRecord] = {}

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

        #    cross_group_forward: Agent 实际跨群转发次数（微观行为）
        self.cross_group_forward: int = 0

        #    cross_group_spread: 热度跨群传播次数（宏观环境机制）
        self.cross_group_spread: int = 0

        #    情绪快照（供 _calc_emotional_contagion 使用）
        self._prev_emotion_snapshot: Dict[int, EmotionState] = {}

        # ⑤ Hawkes 引擎（C 模块）
        from hawkes_engine import HawkesEngine
        self.hawkes = HawkesEngine(
            mu    = config.hawkes_params.get("mu",    0.1),
            alpha = config.hawkes_params.get("alpha", 0.5),
            beta  = config.hawkes_params.get("beta",  1.0),
        )

        # ⑤-b LLM 客户端（C 模块）—— W4 新增
        #
        # 读取 config.llm_config，调用 C 模块的 setup_llm_client 工厂函数，
        # 将返回的客户端实例挂载为 self._llm_client。
        #
        # SocialAgent._update_beliefs 通过 self.model._llm_client 取用：
        #   - _llm_client 不为 None  → 走 LLM 路径（C 模块）
        #   - _llm_client 为 None    → 降级为规则存根（Deffuant-Weisbuch）
        #
        # config.llm_config 示例（config.yaml 中填写）：
        #   provider: deepseek
        #   api_key: ""          # 留空则读环境变量 DEEPSEEK_API_KEY
        #   model: deepseek-v4-flash
        #   temperature: 0.3
        #   timeout: 30
        #   max_retries: 2
        #   fallback_mock: true  # 初始化失败时自动降级 MockLLMClient，不崩溃
        #
        # 若 llm_config 为空 {}，则跳过初始化，_llm_client = None。
        self._llm_client = None
        self.llm_action_enabled = bool(config.llm_config.get("use_actions", True))
        if config.llm_config:
            try:
                from llm_utils import setup_llm_client
                self._llm_client = setup_llm_client(config.llm_config)
                _LOG.info(
                    f"LLM 客户端初始化成功 | "
                    f"provider={config.llm_config.get('provider', '?')} | "
                    f"model={config.llm_config.get('model', '?')}"
                )
            except Exception as e:
                _LOG.warning(
                    f"LLM 客户端初始化失败，所有 Agent 将使用规则存根: {e}"
                )
                # _llm_client 保持 None，SocialAgent 自动降级，不影响仿真运行

        # ⑥ DataCollector（E 模块从此处读取输出）
        #    指标对齐接口表§5：message_count / avg_opinion / polarization /
        #    negative_emotion / distortion_level / cross_group_forward /
        #    intervention_tick / recovery_time；另扩展 cross_group_spread，
        #    用于与 Agent 实际转发严格区分。
        self.datacollector = DataCollector(
            model_reporters={
                "avg_opinion":          lambda m: m._calc_avg_opinion(),
                "polarization":         lambda m: m._calc_polarization(),
                "emotional_contagion":  lambda m: m._calc_emotional_contagion(),
                "message_count":        lambda m: len(m.info_stream_cache),
                "negative_emotion":     lambda m: m._calc_negative_emotion(),
                "distortion_level":     lambda m: m._calc_avg_distortion(),
                "cross_group_forward":  lambda m: m.cross_group_forward,
                "cross_group_spread":   lambda m: m.cross_group_spread,
                "intervention_tick":    lambda m: m._get_earliest_intervention_tick(),
                "recovery_time":        lambda m: m._calc_recovery_time(),  # -1 表示尚未恢复
            }
        )

        # ⑦ 热度回落追踪（用于 recovery_time 计算）
        self._heat_exceeded_tick: Optional[int] = None   # 热度首次超 THETA 的 tick
        self._recovery_time: Optional[int] = None        # 热度回落后记录

        # ⑧ A 模块自维护的 agent 查找字典（绕开 Mesa 3.x AgentSet 不支持索引的问题）
        self._agent_dict: Dict[int, Any] = {}

        # ⑨ 放置智能体
        self._place_agents()

        _LOG.info(
            f"OpinionModel 初始化完成 | "
            f"n_agents={config.n_agents} | "
            f"network={config.network_type} | "
            f"membership={self.group_membership_mode} | "
            f"llm={'已接入' if self._llm_client is not None else '规则存根'} | "
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
        ``group_type_ratio`` 决定 Agent 的“主群/最小群”。在默认
        ``group_membership_mode=hierarchical`` 下，成员关系按校园层级向上包含：

        DORM → DORM+CLASS+MAJOR+CAMPUS
        CLASS → CLASS+MAJOR+CAMPUS
        MAJOR → MAJOR+CAMPUS
        CAMPUS → CAMPUS

        这样同一个人可以把自己在小群中看到的真实消息投递到其加入的大群。
        ``single`` 模式保留旧的一人一群行为，仅用于兼容对照实验。
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

                if self.group_membership_mode == "hierarchical":
                    membership_types = [
                        candidate for candidate in GroupType
                        if int(candidate) >= int(group_type)
                    ]
                else:
                    membership_types = [group_type]

                group_ids = [
                    self._group_id_for_type(candidate)
                    for candidate in membership_types
                ]
                primary_group_id = self._group_id_for_type(group_type)

                init_config: Dict[str, Any] = {
                    "agent_type":   agent_type,
                    "group_type":   group_type,
                    "primary_group_id": primary_group_id,
                    "group_ids": group_ids,
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
                self._agent_dict[agent_id] = agent  # 自维护查找字典
                for group_id in group_ids:
                    self.group_members[group_id].add(agent_id)
                agent_id += 1

        dist_str = " | ".join(f"{t.name}:{c}" for t, c in counts.items())
        gdist_str = " | ".join(f"{g.name}:{c}" for g, c in group_counts.items())
        _LOG.info(f"AgentType 分布: {dist_str}")
        _LOG.info(f"主群 GroupType 分布: {gdist_str}")
        membership_str = " | ".join(
            f"{group_id}:{len(members)}"
            for group_id, members in self.group_members.items()
        )
        _LOG.info(f"真实群成员数: {membership_str}")

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

        all_agents = list(self.schedule.agents)  # Mesa 3.x 兼容
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

        # ── ① 尝试获取 B 模块的 calc_heat_decay ──────────────────────
        # B 模块将 calc_heat_decay 定义在 SocialAgent 上，且函数内部使用
        # agent 自身的 group_type / beta。因此必须按群选择对应 Agent，不能
        # 用第一个 Agent 代算所有群；某群无 Agent 时使用 A 的纯函数降级。
        _agents_list = list(self.schedule.agents)  # Mesa 3.x AgentSet 需转 list 才能索引
        _decay_agents: Dict[GroupType, Any] = {}
        for _agent in _agents_list:
            if not hasattr(_agent, "calc_heat_decay"):
                continue
            try:
                _agent_group = _agent.beliefs.identity.group_type
            except (AttributeError, TypeError):
                continue
            _decay_agents.setdefault(_agent_group, _agent)

        def _heat_decay(current_heat: float, group_type: GroupType) -> float:
            """调用 B 模块 calc_heat_decay 或本地降级。"""
            elapsed = current_tick  # elapsed_steps 即当前 tick
            t_int   = self.intervention_tick.get(group_type, None)
            decay_agent = _decay_agents.get(group_type)
            if decay_agent is not None:
                try:
                    return float(decay_agent.calc_heat_decay(
                        current_heat=current_heat,
                        elapsed_steps=elapsed,
                        intervention_tick=t_int,
                    ))
                except Exception as exc:
                    _LOG.debug(
                        f"{group_type.name} 群调用 B.calc_heat_decay 失败，"
                        f"改用 A 降级公式: {exc}"
                    )
            return self._fallback_heat_decay(current_heat, group_type, elapsed, t_int)

        # ── ② 更新各话题各群热度 ──────────────────────────────────────
        #    同时检测 intervention_tick 触发条件
        controller_groups = self._get_controller_groups()

        for topic_id, group_heat in self.topic_heat.items():
            for group_type in GroupType:
                old_heat = group_heat[group_type]

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

                # 触发时先写入 intervention_tick，再计算 H(t+1)，从而严格满足
                # 公式中的 𝟙[t ≥ t_k^int]，避免干预效果晚一 tick 生效。
                new_heat = _heat_decay(old_heat, group_type)
                group_heat[group_type] = max(0.0, new_heat)

            # ── ③ 跨群扩散 ────────────────────────────────────────────
            for src_gt in GroupType:
                if group_heat[src_gt] >= THETA:
                    for dst_gt in GroupType:
                        if dst_gt != src_gt:
                            if self.random.random() < self.cross_group_spread_probability:
                                # 热度传播（取较小值叠加，避免无限膨胀）
                                spread_amount = group_heat[src_gt] * 0.1
                                group_heat[dst_gt] = min(
                                    group_heat[dst_gt] + spread_amount,
                                    group_heat[src_gt],
                                )
                                self.cross_group_spread += 1
                                _LOG.debug(
                                    f"[tick={current_tick}] 热度跨群扩散 "
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
        # 按单群判断：任一群热度超阈 → 记录超阈时刻；所有群都回落 → 记录 recovery_time
        # （用均值会因冷群拉低而永远触发不了，场景上"任一群爆发"才是舆情事件）
        all_heats = [v for gd in self.topic_heat.values() for v in gd.values()]
        max_heat  = float(max(all_heats)) if all_heats else 0.0

        if max_heat >= THETA and self._heat_exceeded_tick is None:
            self._heat_exceeded_tick = current_tick
            _LOG.info(
                f"[tick={current_tick}] 热度首次超阈 max_heat={max_heat:.3f} ≥ θ={THETA}"
            )
        elif max_heat < THETA and self._heat_exceeded_tick is not None:
            if self._recovery_time is None:
                self._recovery_time = current_tick - self._heat_exceeded_tick
                _LOG.info(
                    f"[tick={current_tick}] 热度全面回落，recovery_time={self._recovery_time}"
                )

        # ── ⑥ 淘汰过期信息流 ──────────────────────────────────────────
        self.info_stream_cache = [
            r for r in self.info_stream_cache
            if current_tick - r.tick <= EXPIRE_TICKS
        ]
        self._message_by_id = {
            record.message_id: record
            for record in self.info_stream_cache
            if record.message_id
        }

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
        agents = list(self.schedule.agents)  # Mesa 3.x 兼容

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
    def _register_agent(self, agent) -> None:
        """将 agent 注册到 A 模块自维护的查找字典（Mesa 3.x 兼容）。"""
        self._agent_dict[agent.unique_id] = agent
        if hasattr(agent, "beliefs"):
            identity = agent.beliefs.identity
            group_ids = list(getattr(identity, "group_ids", []) or [])
            if not group_ids:
                group_ids = [self._group_id_for_type(identity.group_type)]
            for group_id in group_ids:
                if group_id in self.group_members:
                    self.group_members[group_id].add(agent.unique_id)

    def submit_action(self, record: ActionRecord) -> bool:
        """
        校验并投递一条行动记录。

        路由语义：
          - ``record.group_id`` 是消息真正进入的信息流群；
          - FORWARD 必须引用 ``source_message_id``（兼容旧调用时可由 target_id
            在发送者可见消息中回溯）；
          - ``source_group_id`` 从真实源消息推导，不能由作者主群猜测；
          - 仅当 ``source_group_id != group_id`` 时计入 ``cross_group_forward``。

        无效路由不会进入信息流，返回 ``False``；成功投递返回 ``True``。
        """
        sender = self._get_agent_by_id(record.agent_id)
        if sender is None or not hasattr(sender, "beliefs"):
            _LOG.warning("拒绝未知发送者的行动: agent_id=%s", record.agent_id)
            return False

        member_group_ids = set(self.get_agent_groups(record.agent_id))
        destination_group_id = (record.group_id or self.get_agent_group(record.agent_id) or "").strip()
        if destination_group_id not in self.group_type_by_id:
            _LOG.warning(
                "拒绝行动：目标群不存在 | agent=%s group_id=%r",
                record.agent_id,
                destination_group_id,
            )
            return False
        if destination_group_id not in member_group_ids:
            _LOG.warning(
                "拒绝行动：发送者不是目标群成员 | agent=%s group_id=%s memberships=%s",
                record.agent_id,
                destination_group_id,
                sorted(member_group_ids),
            )
            return False

        source_record: Optional[ActionRecord] = None
        if record.action_type == ActionType.FORWARD:
            source_record = self._resolve_forward_source(record, member_group_ids)
            if source_record is None:
                _LOG.warning(
                    "拒绝 FORWARD：找不到发送者可见的真实源消息 | "
                    "agent=%s source_message_id=%r target_id=%r",
                    record.agent_id,
                    record.source_message_id,
                    record.target_id,
                )
                return False

            record.source_message_id = source_record.message_id
            record.source_group_id = source_record.group_id
            record.target_id = source_record.agent_id
            if not record.topic_id:
                record.topic_id = source_record.topic_id
        else:
            # 原创/回复消息本身就在目标群中。REPLY 可保留 source_message_id 供追踪，
            # 但不会被当作跨群转发统计。
            record.source_group_id = destination_group_id
            if record.action_type == ActionType.SEND_MESSAGE:
                record.source_message_id = None

        record.group_id = destination_group_id
        if not record.message_id:
            record.message_id = self._allocate_message_id()
        elif record.message_id in self._message_by_id:
            _LOG.warning("拒绝重复 message_id=%s", record.message_id)
            return False

        self.info_stream_cache.append(record)
        self._message_by_id[record.message_id] = record

        topic_id = record.topic_id or "T001"
        record.topic_id = topic_id
        if topic_id not in self.topic_heat:
            self.topic_heat[topic_id] = {g: 0.0 for g in GroupType}
            self.topic_negative[topic_id] = {g: 0.0 for g in GroupType}
            self._topic_heat_flat[topic_id] = 0.0
            self._topic_negative_flat[topic_id] = 0.0

        # 热度/负面值归入消息实际投递群，而不是发送者的主群。
        destination_group_type = self.group_type_by_id[destination_group_id]
        heat_delta = record.heat if record.heat > 0 else 0.1
        self.topic_heat[topic_id][destination_group_type] = min(
            self.topic_heat[topic_id][destination_group_type] + heat_delta,
            10.0,
        )

        smooth = 0.2
        old_neg = self.topic_negative[topic_id][destination_group_type]
        self.topic_negative[topic_id][destination_group_type] = float(
            (1 - smooth) * old_neg + smooth * record.negative_score
        )

        if (
            record.action_type == ActionType.FORWARD
            and source_record is not None
            and source_record.group_id != destination_group_id
        ):
            self.cross_group_forward += 1

        vals = list(self.topic_heat[topic_id].values())
        self._topic_heat_flat[topic_id] = float(np.mean(vals))
        neg_vals = list(self.topic_negative[topic_id].values())
        self._topic_negative_flat[topic_id] = float(np.mean(neg_vals))
        return True

    # ================================================================== #
    #  B 模块接口适配（供 social_agent._perceive 调用）                    #
    # ================================================================== #

    def get_agent_group(self, agent_id: int) -> Optional[str]:
        """返回 Agent 的主群 ID（兼容旧接口）。"""
        agent = self._get_agent_by_id(agent_id)
        if agent and hasattr(agent, "beliefs"):
            identity = agent.beliefs.identity
            primary = getattr(identity, "primary_group_id", "")
            if primary:
                return primary
            return self._group_id_for_type(identity.group_type)
        return None

    def get_agent_groups(self, agent_id: int) -> List[str]:
        """返回 Agent 的全部真实群成员关系。"""
        agent = self._get_agent_by_id(agent_id)
        if not agent or not hasattr(agent, "beliefs"):
            return []
        identity = agent.beliefs.identity
        group_ids = list(getattr(identity, "group_ids", []) or [])
        if not group_ids:
            group_ids = [self._group_id_for_type(identity.group_type)]
        return [group_id for group_id in group_ids if group_id in self.group_type_by_id]

    def get_group_type(self, agent_id: int) -> GroupType:
        """返回 Agent 主群类型（兼容旧接口）。"""
        agent = self._get_agent_by_id(agent_id)
        if agent and hasattr(agent, "beliefs"):
            return agent.beliefs.identity.group_type
        return GroupType.CLASS

    def get_group_type_by_id(self, group_id: str) -> Optional[GroupType]:
        """由真实群 ID 返回 GroupType。"""
        return self.group_type_by_id.get(group_id)

    def get_group_messages(
        self,
        agent_id: int,
        limit: int = 20,
        group_id: Optional[str] = None,
    ) -> List[ActionRecord]:
        """
        返回 Agent 可见的最近消息。

        可见性只由真实群成员关系决定：消息仅对 ``record.group_id`` 的成员可见。
        跨群消息必须由一次 FORWARD 在目标群中创建新记录，不再使用随机“异群泄漏”。
        """
        if limit <= 0:
            return []

        member_group_ids = set(self.get_agent_groups(agent_id))
        if group_id is not None:
            if group_id not in member_group_ids:
                return []
            visible_group_ids = {group_id}
        else:
            visible_group_ids = member_group_ids

        result: List[ActionRecord] = []
        for record in reversed(self.info_stream_cache):
            if record.group_id in visible_group_ids:
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

    # ================================================================== #
    #  额外 DataCollector 指标                                             #
    # ================================================================== #

    def _calc_negative_emotion(self) -> float:
        """负面情绪指数：valence < 0 的 agent 比例。"""
        agents = list(self.schedule.agents)  # Mesa 3.x
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
        """返回恢复耗时；尚未恢复时返回 -1.0，避免与“0 tick 即恢复”混淆。"""
        return float(self._recovery_time) if self._recovery_time is not None else -1.0

    def get_final_summary(self) -> Dict[str, Any]:
        """返回仿真结束后的最终状态，不混入逐 tick DataCollector。

        ``recovery_time`` 使用 ``None`` 表示截至当前仍未恢复；这与
        DataCollector 中用于数值列的 -1.0 哨兵值相互对应。
        """
        all_heats = [v for group_map in self.topic_heat.values() for v in group_map.values()]
        max_heat = float(max(all_heats)) if all_heats else 0.0
        earliest = self._get_earliest_intervention_tick()
        return {
            "simulation_tick": int(self.schedule.time),
            "heat_exceeded_tick": self._heat_exceeded_tick,
            "recovery_status": "recovered" if self._recovery_time is not None else "not_recovered",
            "recovery_time": self._recovery_time,
            "earliest_intervention_tick": None if math.isinf(earliest) else int(earliest),
            "intervention_tick_by_group": {
                group.name: tick for group, tick in self.intervention_tick.items()
            },
            "max_topic_heat": max_heat,
            "cross_group_forward": int(self.cross_group_forward),
            "cross_group_spread": int(self.cross_group_spread),
            "active_message_cache_size": len(self.info_stream_cache),
        }

    # ================================================================== #
    #  内部辅助                                                            #
    # ================================================================== #

    @staticmethod
    def _group_id_for_type(group_type: GroupType) -> str:
        return f"GROUP_{group_type.name}"

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

        # 兼容旧代码：target_id 过去被当作“原作者”。只允许从发送者真实可见
        # 的当前缓存中回溯，绝不再用作者主群直接猜来源群。
        if source is None and record.target_id is not None:
            source = next(
                (
                    candidate
                    for candidate in reversed(self.info_stream_cache)
                    if candidate.agent_id == record.target_id
                    and candidate.group_id in member_group_ids
                ),
                None,
            )

        if source is None or source.group_id not in member_group_ids:
            return None
        return source

    def get_forward_destination_candidates(
        self,
        agent_id: int,
        source_group_id: str,
    ) -> List[str]:
        """返回该成员可把源消息转发到的其他真实群，按层级从小到大排序。"""
        source_type = self.group_type_by_id.get(source_group_id)
        if source_type is None:
            return []

        candidates = [
            group_id
            for group_id in self.get_agent_groups(agent_id)
            if group_id != source_group_id
        ]
        if not self.allow_downward_forward:
            candidates = [
                group_id
                for group_id in candidates
                if int(self.group_type_by_id[group_id]) > int(source_type)
            ]
        return sorted(candidates, key=lambda gid: int(self.group_type_by_id[gid]))

    def select_forward_destination(
        self,
        agent_id: int,
        source_group_id: str,
    ) -> Optional[str]:
        """按配置策略选择一次真实跨群转发的投递群。"""
        candidates = self.get_forward_destination_candidates(agent_id, source_group_id)
        if not candidates:
            return None
        if self.forward_destination_strategy == "largest":
            return candidates[-1]
        if self.forward_destination_strategy == "random":
            return self.random.choice(candidates)
        # 默认 next_larger：模拟宿舍群→班级群→专业群→校园群的逐层扩散。
        return candidates[0]

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
        """
        按 unique_id 查找 Agent（O(1)）。
        使用 A 模块自维护的 _agent_dict，完全绕开 Mesa 2/3.x 内部结构差异。
        """
        # 优先用自维护字典（Mesa 2/3 通用）
        if hasattr(self, '_agent_dict'):
            agent = self._agent_dict.get(agent_id)
            if agent is not None:
                return agent
        # 降级：线性扫描（兜底，理论上不会走到这里）
        for a in list(self.schedule.agents):
            if a.unique_id == agent_id:
                return a
        return None

    def _get_controller_groups(self) -> set:
        """返回拥有至少一个 CONTROLLER agent 的群类型集合。"""
        groups = set()
        for agent in list(self.schedule.agents):
            if (hasattr(agent, "beliefs")
                    and agent.beliefs.identity.agent_type == AgentType.CONTROLLER):
                for group_id in self.get_agent_groups(agent.unique_id):
                    group_type = self.group_type_by_id.get(group_id)
                    if group_type is not None:
                        groups.add(group_type)
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
