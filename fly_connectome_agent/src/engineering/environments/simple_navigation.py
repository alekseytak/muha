"""Simple 1D navigation environment for P3."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EnvConfig:
    length: int = 11
    target_position: int = 10
    init_position: int = 0
    max_steps: int = 50
    step_cost: float = -0.1
    boundary_penalty: float = -1.0
    target_reward: float = 10.0


class SimpleNavigation:
    ACTION_SPACE = ("move_left", "move_right", "stay")

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.position = self.config.init_position
        self.step_count = 0
        self.terminated = False
        self._steps_this_episode = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState(0)
        self.position = self.config.init_position
        self.step_count = 0
        self.terminated = False
        self._steps_this_episode = 0
        obs = np.array([self.position], dtype=np.int64)
        info = {"position": self.position, "step": self.step_count}
        return obs, info

    def step(self, action: str) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if action not in self.ACTION_SPACE:
            raise ValueError(f"Invalid action: {action!r}. Valid: {self.ACTION_SPACE}")
        if self.terminated:
            obs = np.array([self.position], dtype=np.int64)
            return obs, 0.0, self.terminated, True, {"reason": "already_terminated"}
        prev_position = self.position
        self._steps_this_episode += 1
        if action == "move_left":
            self.position -= 1
        elif action == "move_right":
            self.position += 1
        reward = self.config.step_cost
        if self.position < 0:
            self.position = 0
            reward += self.config.boundary_penalty
        elif self.position >= self.config.length:
            self.position = self.config.length - 1
            reward += self.config.boundary_penalty
        if self.position == self.config.target_position:
            reward += self.config.target_reward
            self.terminated = True
        truncated = self._steps_this_episode >= self.config.max_steps and not self.terminated
        obs = np.array([self.position], dtype=np.int64)
        info = {"position": self.position, "prev_position": prev_position, "step": self._steps_this_episode}
        return obs, reward, self.terminated, truncated, info
