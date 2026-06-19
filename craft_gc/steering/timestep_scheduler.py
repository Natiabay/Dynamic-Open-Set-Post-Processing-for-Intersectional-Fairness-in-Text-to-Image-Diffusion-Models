"""Timestep-aware steering schedule beta(t)."""

import numpy as np


class TimestepScheduler:
    def __init__(
        self,
        num_inference_steps: int = 50,
        beta_max: float = 0.8,
        t_star: float = 0.6,
        gamma: float = 10.0,
    ):
        self.T = num_inference_steps
        self.beta_max = beta_max
        self.t_star = t_star
        self.gamma = gamma
        self._schedule = self._compute()

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _compute(self) -> np.ndarray:
        steps = np.arange(self.T, dtype=np.float64)
        normalized = steps / max(self.T - 1, 1)
        return self.beta_max * self._sigmoid(self.gamma * (normalized - self.t_star))

    def get_beta(self, step_index: int) -> float:
        if 0 <= step_index < self.T:
            return float(self._schedule[step_index])
        return 0.0

    def as_list(self) -> list:
        return self._schedule.tolist()
