"""
mesa_compat.py — Mesa 兼容层（支持 Mesa 2.x / 3.x / 无 Mesa 三种环境）
------------------------------------------------------------------------
负责人：A

【核心问题】
  Mesa 3.x 将 schedule.agents 从 List[Agent] 改为 AgentSet，
  AgentSet 不支持：
    - 下标索引 agents[0]
    - random.sample(agents, k=n)
    - len(agents) 在某些情况下
  本文件统一解决上述兼容性问题。

使用方式（opinion_model.py 顶部）：
    try:
        from mesa import Model
        from mesa.time import RandomActivation
        from mesa.space import NetworkGrid
        from mesa.datacollection import DataCollector
        _MESA_REAL = True
    except ImportError:
        from mesa_compat import Model, RandomActivation, NetworkGrid, DataCollector
        _MESA_REAL = False

凡是需要 list 的地方，统一用 model.get_agents() 而非 schedule.agents。
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
    def __init__(self, unique_id: int, model: "Model") -> None:
        self.unique_id = unique_id
        self.model     = model
        self.pos: Optional[int] = None

    def step(self) -> None:
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  RandomActivation — agents 永远返回 list，兼容所有操作                    #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

class RandomActivation:
    def __init__(self, model: "Model") -> None:
        self.model  = model
        self._agents: Dict[int, Agent] = {}
        self.steps  = 0
        self.time   = 0

    @property
    def agents(self) -> List[Agent]:
        """永远返回 list，兼容下标索引、random.sample、len 等操作。"""
        return list(self._agents.values())

    def add(self, agent: Agent) -> None:
        self._agents[agent.unique_id] = agent

    def remove(self, agent: Agent) -> None:
        self._agents.pop(agent.unique_id, None)

    def step(self) -> None:
        agent_list = list(self._agents.values())
        self.model.random.shuffle(agent_list)
        for agent in agent_list:
            agent.step()
        self.steps += 1
        self.time  += 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  NetworkGrid                                                              #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

class NetworkGrid:
    def __init__(self, G: nx.Graph) -> None:
        self.G = G
        for node in G.nodes():
            if "agent" not in G.nodes[node]:
                G.nodes[node]["agent"] = []

    def place_agent(self, agent: Agent, node: int) -> None:
        self.G.nodes[node]["agent"].append(agent)
        agent.pos = node

    def move_agent(self, agent: Agent, new_node: int) -> None:
        old_node = agent.pos
        if old_node is not None and old_node in self.G.nodes:
            try:
                self.G.nodes[old_node]["agent"].remove(agent)
            except ValueError:
                pass
        self.place_agent(agent, new_node)

    def remove_agent(self, agent: Agent) -> None:
        node = agent.pos
        if node is not None and node in self.G.nodes:
            try:
                self.G.nodes[node]["agent"].remove(agent)
            except ValueError:
                pass
        agent.pos = None

    def get_neighbors(self, node: int, include_center: bool = False) -> List[int]:
        if node not in self.G:
            return []
        neighbors = list(self.G.neighbors(node))
        if include_center:
            neighbors.append(node)
        return neighbors

    def get_cell_list_contents(self, cell_list: List[int]) -> List[Agent]:
        agents = []
        for node in cell_list:
            if node in self.G.nodes:
                agents.extend(self.G.nodes[node].get("agent", []))
        return agents

    def get_neighbors_of_type(self, node: int, agent_type: Any) -> List[Agent]:
        neighbor_agents = self.get_cell_list_contents(self.get_neighbors(node))
        return [a for a in neighbor_agents if isinstance(a, agent_type)]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  DataCollector                                                            #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

class DataCollector:
    def __init__(
        self,
        model_reporters: Optional[Dict[str, Callable]] = None,
        agent_reporters: Optional[Dict[str, Callable]] = None,
    ) -> None:
        self.model_reporters = model_reporters or {}
        self.agent_reporters = agent_reporters or {}
        self._model_data: List[Dict[str, Any]] = []

    def collect(self, model: "Model") -> None:
        row: Dict[str, Any] = {}
        for col, fn in self.model_reporters.items():
            try:
                row[col] = fn(model)
            except Exception:
                row[col] = None
        self._model_data.append(row)

    def get_model_vars_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._model_data)

    def get_agent_vars_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
#  Model                                                                    #
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ #

class Model:
    def __init__(self) -> None:
        self.random = _random.Random()

    def step(self) -> None:
        pass

    def run_model(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()
