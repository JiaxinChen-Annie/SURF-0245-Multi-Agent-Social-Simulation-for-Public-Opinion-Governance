"""
opinion_model.py — 模型/环境层（函数 11–18 + submit_action）
--------------------------------------------------------------
负责人：A
定位：系统骨架与调度中枢，第1周架构牵头

函数清单（接口表 #11–18 + 18+）：
    __init__(config)              初始化骨架
    _build_social_network()       生成社交网络 → nx.Graph
    _place_agents()               部署四类智能体
    step()                        单步主循环
    _update_environment()         更新话题热度 + 清理信息流缓存
    _calc_avg_opinion()           DataCollector 回调：全网平均观点
    _calc_polarization()          DataCollector 回调：极化程度
    _calc_emotional_contagion()   DataCollector 回调：情绪传播速度
    submit_action(record)         将行动写入环境信息流缓存（新增）

上游输入：
    C 提供的 HawkesEngine（hawkes_engine.py）
    B 提供的 SocialAgent（social_agent.py；S1 为规则存根）

下游输出：
    每步 metrics 字典：avg_opinion / polarization / emotional_contagion
    格式与 D(评估) / E(可视化) 约定完全对齐
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any

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
    SimConfig, AgentType, ActionRecord, ActionType, EmotionState,
)

_LOG = logging.getLogger("OpinionModel")


class OpinionModel(Model):
    """
    多智能体舆情仿真 · 模型/环境层。

    架构分层（对应甘特图 A 行）：
        S0/W1 接口冻结  → __init__ 骨架已定义
        S1/W2 搭框架   → 本文件全部函数实现
        S3/W5 全量接入 → step() Hawkes 激活机制在此版本已就绪
        S4/W7 干预注入  → submit_action 支持 D 模块注入
    """

    # ================================================================== #
    #  函数 11  __init__                                                   #
    # ================================================================== #
    def __init__(self, config: SimConfig) -> None:
        """
        初始化总智能体数、调度器、网络空间、数据收集器、霍克斯引擎实例；
        调用 _build_social_network、_place_agents。

        Parameters
        ----------
        config : SimConfig
        """
        super().__init__()
        self.config = config

        # ① 固定随机种子（保证跨 run 可复现）
        self.random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        # ② 调度器
        self.schedule = RandomActivation(self)

        # ③ 构建社交网络 → 包装为 NetworkGrid
        G = self._build_social_network()
        self.grid = NetworkGrid(G)

        # ④ 环境状态
        self.info_stream_cache: List[ActionRecord] = []
        self.trending: List[tuple] = [("E001", 1.0)]          # [(event_id, heat)]
        self._prev_emotion_snapshot: Dict[int, EmotionState] = {}

        # ⑤ 霍克斯引擎（C 模块）
        from hawkes_engine import HawkesEngine
        self.hawkes = HawkesEngine(
            mu    = config.hawkes_params.get("mu",    0.1),
            alpha = config.hawkes_params.get("alpha", 0.5),
            beta  = config.hawkes_params.get("beta",  1.0),
        )

        # ⑥ DataCollector（E 模块从此处读取输出）
        self.datacollector = DataCollector(
            model_reporters={
                "avg_opinion":         lambda m: m._calc_avg_opinion(),
                "polarization":        lambda m: m._calc_polarization(),
                "emotional_contagion": lambda m: m._calc_emotional_contagion(),
            }
        )

        # ⑦ 放置智能体
        self._place_agents()

        _LOG.info(
            f"OpinionModel 初始化完成 | "
            f"n_agents={config.n_agents} | "
            f"network={config.network_type} | "
            f"mesa={'系统' if _MESA_REAL else '垫片'}"
        )

    # ================================================================== #
    #  函数 12  _build_social_network                                      #
    # ================================================================== #
    def _build_social_network(self) -> nx.Graph:
        """
        生成社交网络，模拟平台关注关系。

        支持两种网络类型（network_type 配置项）：
          - barabasi_albert  无标度网络（默认），网络参数：m（新节点连边数）
          - watts_strogatz   小世界网络，网络参数：k（平均度）/ p（重连概率）

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
            # ── 边界处理：BA 算法要求 1 <= m < n ─────────────────────────
            if n == 1:
                # 单节点无法建 BA 图 → 退化为孤立图（冒烟测试场景）
                G = nx.empty_graph(1)
                _LOG.info("BA 网络 n=1，退化为单节点孤立图")
            elif n <= m:
                # 节点数不足以构建 BA 图 → 退化为完全图
                G = nx.complete_graph(n)
                _LOG.warning(
                    f"BA 网络要求 m < n，当前 n={n} <= m={m}，"
                    f"退化为完全图 (edges={G.number_of_edges()})"
                )
            else:
                G = nx.barabasi_albert_graph(n, m, seed=seed)
                _LOG.info(
                    f"BA 无标度网络 n={n} m={m} | "
                    f"平均度={2 * G.number_of_edges() / n:.2f}"
                )

        elif ntype == "watts_strogatz":
            k = params.get("k", 6)
            p = params.get("p", 0.1)
            # WS 要求 k < n，否则退化为完全图
            if n <= k:
                G = nx.complete_graph(n)
                _LOG.warning(f"WS 网络 n={n} <= k={k}，退化为完全图")
            else:
                G = nx.watts_strogatz_graph(n, k, p, seed=seed)
                _LOG.info(f"WS 小世界网络 n={n} k={k} p={p}")

        else:
            _LOG.warning(f"未知 network_type='{ntype}'，回退到 BA(m=3)")
            m_fb = min(3, max(1, n - 1))
            G = nx.barabasi_albert_graph(n, m_fb, seed=seed) if n > 1 else nx.empty_graph(1)

        return G

    # ================================================================== #
    #  函数 13  _place_agents                                              #
    # ================================================================== #
    def _place_agents(self) -> None:
        """
        将四类智能体按比例部署到网络节点。
        类型顺序：PUBLIC → OPINION_LEADER → MEDIA → OFFICIAL
        最后一类兜底，确保 Σcount == n_agents。
        """
        from social_agent import SocialAgent

        n      = self.config.n_agents
        ratio  = self.config.agent_type_ratio
        nodes  = list(self.grid.G.nodes())

        # 计算各类型数量
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

        # 逐个实例化并挂载
        agent_id = 0
        for agent_type, count in counts.items():
            for _ in range(count):
                init_config: Dict[str, Any] = {
                    "agent_type":   agent_type,
                    "stance_prior": self.random.uniform(-1.0, 1.0),
                    "event_id":     "E001",
                }
                agent = SocialAgent(
                    unique_id=agent_id,
                    model=self,
                    agent_type=agent_type,
                    init_config=init_config,
                )
                self.schedule.add(agent)
                self.grid.place_agent(agent, nodes[agent_id])
                agent_id += 1

        # 统计分布用于日志
        dist_str = " | ".join(f"{t.name}:{c}" for t, c in counts.items())
        _LOG.info(f"Agent 分布: {dist_str}")

    # ================================================================== #
    #  函数 14  step                                                       #
    # ================================================================== #
    def step(self) -> None:
        """
        单步主循环（Mesa Model.step() 标准入口）。

        执行顺序：
        1. 读取当前 tick = schedule.time
        2. 霍克斯采样 → 计算激活比例 ∈ [10%, 80%]
        3. 随机采样 n_active 个 Agent，逐个调用 agent.step()
           → 异常降级：记 warning，不中断整步
        4. _update_environment()
        5. datacollector.collect(self)
        6. schedule.time / schedule.steps 自增
        7. 把本 tick 实际提交的非沉默行动数回写 HawkesEngine
        """
        t = float(self.schedule.time)

        # ── ① Hawkes 激活比例 ──────────────────────────────────────────
        lam = self.hawkes.intensity(t)
        mu  = self.config.hawkes_params.get("mu", 0.1)
        # 以基线的 3 倍为参考点，映射到 [0.10, 0.80]
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

        # ── ③ 环境更新 ─────────────────────────────────────────────────
        self._update_environment()

        # ── ④ 数据收集 ─────────────────────────────────────────────────
        self.datacollector.collect(self)

        # ── ⑤ 时步自增 ─────────────────────────────────────────────────
        self.schedule.steps += 1
        self.schedule.time  += 1

        # ── ⑥ 回写 Hawkes（批量注入本 tick 事件）─────────────────────
        for _ in range(n_events):
            self.hawkes.add_event(t)

    # ================================================================== #
    #  函数 15  _update_environment                                        #
    # ================================================================== #
    def _update_environment(self) -> None:
        """
        每步结束后刷新环境状态：
          1. 话题热度按指数衰减（× 0.95/tick）
          2. 本 tick 新增行动注入到话题热度
          3. 淘汰超过 EXPIRE_TICKS=10 tick 的过期信息流
        """
        current_tick  = int(self.schedule.time)
        EXPIRE_TICKS  = 10
        HEAT_DECAY    = 0.95
        HEAT_STEP     = 0.05

        # ① 热度衰减
        self.trending = [
            (eid, heat * HEAT_DECAY)
            for eid, heat in self.trending
        ]

        # ② 本 tick 行动 → 叠加热度
        tick_actions = [r for r in self.info_stream_cache if r.tick == current_tick]
        heat_dict: Dict[str, float] = dict(self.trending)
        for act in tick_actions:
            heat_dict[act.event_id] = heat_dict.get(act.event_id, 0.0) + HEAT_STEP

        # ③ 重建列表，按热度降序
        self.trending = sorted(heat_dict.items(), key=lambda x: x[1], reverse=True)

        # ④ 淘汰过期信息流
        self.info_stream_cache = [
            r for r in self.info_stream_cache
            if current_tick - r.tick <= EXPIRE_TICKS
        ]

    # ================================================================== #
    #  函数 16  _calc_avg_opinion（DataCollector 回调）                     #
    # ================================================================== #
    def _calc_avg_opinion(self) -> float:
        """
        全网 Agent 对 E001 事件的平均观点值。
        范围 [-1, +1]，0 代表中立。

        Returns
        -------
        float
        """
        vals = self._collect_opinion_values()
        return float(np.mean(vals)) if vals else 0.0

    # ================================================================== #
    #  函数 17  _calc_polarization（DataCollector 回调）                    #
    # ================================================================== #
    def _calc_polarization(self) -> float:
        """
        观点极化程度（标准差）。
        范围 [0, ~1]；值越大，观点分裂越严重。

        Returns
        -------
        float
        """
        vals = self._collect_opinion_values()
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    # ================================================================== #
    #  函数 18  _calc_emotional_contagion（DataCollector 回调）             #
    # ================================================================== #
    def _calc_emotional_contagion(self) -> float:
        """
        情绪传播速度：相邻两 tick 间各 Agent arousal 变化量的均值。
        第一次调用时无前一快照，返回 0.0。

        Returns
        -------
        float ∈ [0, 1]
        """
        agents = self.schedule.agents

        # 构建当前快照
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
    #  函数 18+  submit_action（新增）                                      #
    # ================================================================== #
    def submit_action(self, record: ActionRecord) -> None:
        """
        将智能体行动写入环境信息流缓存，供下一 tick 邻居感知。
        D 模块注入官方 Agent 行动时也经由此接口。

        Parameters
        ----------
        record : ActionRecord
        """
        self.info_stream_cache.append(record)

    # ================================================================== #
    #  内部辅助                                                            #
    # ================================================================== #
    def _collect_opinion_values(self) -> List[float]:
        """从所有 Agent 提取对首个事件的观点值列表。"""
        vals: List[float] = []
        for agent in self.schedule.agents:
            if hasattr(agent, "beliefs") and agent.beliefs.opinions:
                op = list(agent.beliefs.opinions.values())[0]
                vals.append(op.opinion_value)
        return vals
