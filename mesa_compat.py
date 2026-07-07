"""
mesa_compat.py — Mesa 兼容层垫片
--------------------------------------
负责人：A（临时垫片，装好 Mesa 后可删除此文件）

用法：opinion_model.py 中已写好自动切换逻辑：
    try:
        from mesa import Model, ...         # 优先用真正的 Mesa
    except ImportError:
        from mesa_compat import Model, ...  # 降级到本垫片

装好 Mesa 后：
    pip install mesa
    # 或 conda install -c conda-forge mesa
    # 然后删掉本文件即可，opinion_model.py 不需要改动任何一行。

覆盖范围（仅 W2 阶段用到的子集）：
    Model            — 模型基类，提供 self.random（受 seed 控制）
    Agent            — 智能体基类，提供 unique_id / model / pos
    RandomActivation — 随机顺序调度器
    NetworkGrid      — 网络格（基于 networkx.Graph）
    DataCollector    — 数据收集器，输出 pandas.DataFrame
"""

from __future__ import annotations

import random as _random
from typing import Any, Callable, Dict, List, Optional

import networkx as nx
import pandas as pd


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Agent                                                                    #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

class Agent:
    """
    智能体基类（对齐 mesa.Agent 最小接口）。

    Attributes
    ----------
    unique_id : int
    model     : Model
    pos       : Optional[int]  — 在 NetworkGrid 中的节点编号
    """

    def __init__(self, unique_id: int, model: "Model") -> None:
        self.unique_id = unique_id
        self.model     = model
        self.pos: Optional[int] = None

    def step(self) -> None:          # pragma: no cover
        """子类需要覆盖此方法。"""
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  RandomActivation（调度器）                                               #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

class RandomActivation:
    """
    随机顺序调度器（对齐 mesa.time.RandomActivation）。

    Attributes
    ----------
    agents : List[Agent]  — 所有已注册 Agent（可安全遍历）
    steps  : int          — 已完成步数
    time   : int          — 当前时刻（与 steps 同步）
    """

    def __init__(self, model: "Model") -> None:
        self.model  = model
        self._agents: Dict[int, Agent] = {}   # {unique_id: agent}
        self.steps  = 0
        self.time   = 0

    @property
    def agents(self) -> List[Agent]:
        """返回当前所有 Agent 的列表（顺序不保证）。"""
        return list(self._agents.values())

    def add(self, agent: Agent) -> None:
        """注册一个 Agent。"""
        self._agents[agent.unique_id] = agent

    def remove(self, agent: Agent) -> None:
        """移除一个 Agent（若不存在则忽略）。"""
        self._agents.pop(agent.unique_id, None)

    def step(self) -> None:
        """
        以随机顺序逐个调用所有 Agent 的 step()。
        注意：OpinionModel.step() 自行管理调度，通常不调用此方法。
        """
        agent_list = list(self._agents.values())
        self.model.random.shuffle(agent_list)
        for agent in agent_list:
            agent.step()
        self.steps += 1
        self.time  += 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  NetworkGrid（网络格）                                                    #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

class NetworkGrid:
    """
    基于 networkx.Graph 的网络格（对齐 mesa.space.NetworkGrid）。

    Attributes
    ----------
    G : nx.Graph  — 底层图（可直接访问）
    """

    def __init__(self, G: nx.Graph) -> None:
        self.G = G
        # 每个节点维护占据的 Agent 列表
        for node in G.nodes():
            if "agent" not in G.nodes[node]:
                G.nodes[node]["agent"] = []

    def place_agent(self, agent: Agent, node: int) -> None:
        """将 Agent 放置到指定节点；更新 agent.pos。"""
        self.G.nodes[node]["agent"].append(agent)
        agent.pos = node

    def move_agent(self, agent: Agent, new_node: int) -> None:
        """将 Agent 从当前节点移动到新节点。"""
        old_node = agent.pos
        if old_node is not None and old_node in self.G.nodes:
            try:
                self.G.nodes[old_node]["agent"].remove(agent)
            except ValueError:
                pass
        self.place_agent(agent, new_node)

    def remove_agent(self, agent: Agent) -> None:
        """从网格中移除 Agent。"""
        node = agent.pos
        if node is not None and node in self.G.nodes:
            try:
                self.G.nodes[node]["agent"].remove(agent)
            except ValueError:
                pass
        agent.pos = None

    def get_neighbors(self, node: int, include_center: bool = False) -> List[int]:
        """
        返回指定节点的邻居节点 ID 列表。

        Parameters
        ----------
        node           : 查询节点
        include_center : 是否包含自身（默认 False）
        """
        if node not in self.G:
            return []
        neighbors = list(self.G.neighbors(node))
        if include_center:
            neighbors.append(node)
        return neighbors

    def get_cell_list_contents(self, cell_list: List[int]) -> List[Agent]:
        """返回给定节点列表中所有 Agent。"""
        agents = []
        for node in cell_list:
            if node in self.G.nodes:
                agents.extend(self.G.nodes[node].get("agent", []))
        return agents

    def get_neighbors_of_type(self, node: int, agent_type: Any) -> List[Agent]:
        """返回邻居节点中特定类型的 Agent。"""
        neighbor_agents = self.get_cell_list_contents(self.get_neighbors(node))
        return [a for a in neighbor_agents if isinstance(a, agent_type)]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  DataCollector                                                            #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

class DataCollector:
    """
    数据收集器（对齐 mesa.datacollection.DataCollector）。

    Parameters
    ----------
    model_reporters : Dict[str, Callable[[Model], Any]]
        格式与真实 Mesa DataCollector 完全相同：
        {"列名": lambda m: m.some_method()}
    """

    def __init__(
        self,
        model_reporters: Optional[Dict[str, Callable]] = None,
        agent_reporters: Optional[Dict[str, Callable]] = None,
    ) -> None:
        self.model_reporters = model_reporters or {}
        self.agent_reporters = agent_reporters or {}
        self._model_data: List[Dict[str, Any]] = []

    def collect(self, model: "Model") -> None:
        """调用所有 model_reporters，记录本步数据。"""
        row: Dict[str, Any] = {}
        for col, fn in self.model_reporters.items():
            try:
                row[col] = fn(model)
            except Exception:
                row[col] = None
        self._model_data.append(row)

    def get_model_vars_dataframe(self) -> pd.DataFrame:
        """返回模型级指标的 DataFrame（每行对应一个 tick）。"""
        return pd.DataFrame(self._model_data)

    def get_agent_vars_dataframe(self) -> pd.DataFrame:
        """返回 Agent 级指标 DataFrame（本垫片简化实现，返回空 DataFrame）。"""
        return pd.DataFrame()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Model                                                                    #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

class Model:
    """
    模型基类（对齐 mesa.Model 最小接口）。

    Attributes
    ----------
    random : random.Random  — 受 seed 控制的随机源，子类通过 self.random.seed() 固定
    """

    def __init__(self) -> None:
        self.random = _random.Random()   # 独立随机源，子类 seed 后可复现

    def step(self) -> None:              # pragma: no cover
        """子类应覆盖此方法，执行单步逻辑。"""
        pass

    def run_model(self, n_steps: int) -> None:
        """便捷方法：连续运行 n 步（子类可覆盖）。"""
        for _ in range(n_steps):
            self.step()
