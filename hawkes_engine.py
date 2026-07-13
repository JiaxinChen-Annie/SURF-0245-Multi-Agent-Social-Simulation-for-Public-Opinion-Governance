"""
Hawkes process engine for module C (functions 19-22).
"""

from __future__ import annotations

import math
import random
from typing import List, Optional


class HawkesEngine:
    """
    Univariate Hawkes process:
        lambda(t) = mu + sum(alpha * exp(-beta * (t - ti)))
    """

    def __init__(
        self,
        mu: float,
        alpha: float,
        beta: float,
        rng: Optional[random.Random] = None,
    ) -> None:
        if mu <= 0:
            raise ValueError(f"mu must be > 0, got {mu}")
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")

        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.history: List[float] = []
        self._rng = rng if rng is not None else random.Random()

    def intensity(self, t: float) -> float:
        """Compute instantaneous intensity lambda(t)."""
        t = float(t)
        if not self.history:
            return self.mu
        excitation = 0.0
        for ti in self.history:
            if ti < t:
                excitation += self.alpha * math.exp(-self.beta * (t - ti))
        return self.mu + excitation

    def sample_next_time(self, current_t: float) -> float:
        """
        Sample next event time via Ogata thinning.
        Returns an absolute timestamp > current_t.
        """
        t = float(current_t)
        lam_bar = max(self.intensity(t), self.mu, 1e-12)

        for _ in range(10_000):
            u1 = max(self._rng.random(), 1e-12)
            delta = -math.log(u1) / lam_bar
            t_candidate = t + delta

            lam_candidate = max(self.intensity(t_candidate), 1e-12)
            u2 = self._rng.random()
            if u2 <= (lam_candidate / lam_bar):
                return t_candidate

            # tighten upper bound for the next thinning step
            t = t_candidate
            lam_bar = max(lam_candidate, self.mu, 1e-12)

        # emergency fallback: approximate as homogeneous Poisson(mu)
        return float(current_t) + (1.0 / self.mu)

    def add_event(self, t: float) -> None:
        """Append event timestamp and keep history sorted ascending."""
        tt = float(t)
        self.history.append(tt)
        if len(self.history) > 1 and self.history[-2] > self.history[-1]:
            self.history.sort()
