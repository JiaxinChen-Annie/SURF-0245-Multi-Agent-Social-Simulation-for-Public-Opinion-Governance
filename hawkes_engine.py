"""
hawkes_engine.py — 霍克斯点过程引擎（函数 19–22）
----------------------------------------------------
负责人：C（本文件为 A 在 W2 阶段使用的可运行版本，接口签名与接口表一致）

单变量霍克斯过程公式：
    λ(t) = μ + Σᵢ α · exp(−β · (t − tᵢ)),  tᵢ < t

参数含义：
    μ (mu)    — 基线强度（事件发生的"背景率"），单位：事件/tick
    α (alpha) — 激励系数（一个事件对后续的正向激励强度）
    β (beta)  — 衰减系数（激励随时间衰减的速度）
"""

from __future__ import annotations

import math
import random
from typing import List


class HawkesEngine:
    """
    单变量霍克斯过程。

    函数清单（接口表 #19–22）：
        __init__(mu, alpha, beta)
        intensity(t)
        sample_next_time(current_t)
        add_event(t)
    """

    def __init__(self, mu: float, alpha: float, beta: float) -> None:
        """
        初始化参数并建立空事件历史。

        Parameters
        ----------
        mu    : 基线强度，必须 > 0
        alpha : 激励系数，≥ 0
        beta  : 衰减系数，必须 > 0
        """
        if mu <= 0:
            raise ValueError(f"mu 必须 > 0，当前值: {mu}")
        if beta <= 0:
            raise ValueError(f"beta 必须 > 0，当前值: {beta}")

        self.mu:    float       = mu
        self.alpha: float       = alpha
        self.beta:  float       = beta
        self.history: List[float] = []    # 事件时间戳列表，保持升序

    # ------------------------------------------------------------------ #
    #  函数 20  intensity                                                  #
    # ------------------------------------------------------------------ #
    def intensity(self, t: float) -> float:
        """
        计算 t 时刻的瞬时强度 λ(t)。

        λ(t) = μ + Σᵢ α · exp(−β · (t − tᵢ))，仅对 tᵢ < t 求和。

        Parameters
        ----------
        t : 当前时刻（tick 编号，可为浮点数）

        Returns
        -------
        float ≥ μ
        """
        excitation = sum(
            self.alpha * math.exp(-self.beta * (t - ti))
            for ti in self.history
            if ti < t
        )
        return self.mu + excitation

    # ------------------------------------------------------------------ #
    #  函数 21  sample_next_time                                           #
    # ------------------------------------------------------------------ #
    def sample_next_time(self, current_t: float) -> float:
        """
        用 Ogata 稀疏算法（Thinning）从 t=current_t 起采样下一个事件时刻。

        原理：
          1. 以当前强度 λ(t) 作上界 λ̄
          2. 按泊松过程 Exp(λ̄) 生成候选间隔 Δ
          3. 以概率 λ(t+Δ)/λ̄ 接受（Thinning 步骤）
          4. 若拒绝，更新上界 λ̄ = λ(t+Δ) 并重试

        Parameters
        ----------
        current_t : 采样起始时刻

        Returns
        -------
        float — 下一事件的绝对时刻（> current_t）
        """
        t       = current_t
        lam_bar = self.intensity(t)
        if lam_bar <= 0:
            lam_bar = self.mu      # 保护：避免除以 0

        max_iter = 10_000          # 安全上限，防止极端情况下无限循环
        for _ in range(max_iter):
            u1 = random.random()
            if u1 <= 0:
                u1 = 1e-12
            delta = -math.log(u1) / lam_bar

            t += delta
            lam_t = self.intensity(t)

            u2 = random.random()
            if u2 <= lam_t / max(lam_bar, 1e-12):
                return t           # Thinning 接受

            lam_bar = max(lam_t, self.mu)   # 更新上界（强度单调不增，直到新事件）

        # 极端回退：直接返回基线间隔
        return current_t + (1.0 / self.mu)

    # ------------------------------------------------------------------ #
    #  函数 22  add_event                                                  #
    # ------------------------------------------------------------------ #
    def add_event(self, t: float) -> None:
        """
        将事件时刻 t 追加到历史列表（保持升序）。

        Parameters
        ----------
        t : 事件发生时刻
        """
        self.history.append(t)
        # 保证升序（一般调用时 t 已是递增的，sort 代价极低）
        if len(self.history) > 1 and self.history[-1] < self.history[-2]:
            self.history.sort()
