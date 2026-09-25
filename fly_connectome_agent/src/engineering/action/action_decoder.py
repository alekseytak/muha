"""Action decoder: SNN output spikes → ActionProposal (P3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ActionProposal:
    action_id: str
    action_type: Literal["move_left", "move_right", "stay"]
    score: float
    confidence: float
    source_window_steps: int


class ActionDecoder:
    def __init__(self, n_neurons: int = 2):
        self.n_neurons = n_neurons

    def decode(self, spikes: np.ndarray, window_steps: int = 1) -> ActionProposal:
        spikes = np.asarray(spikes)
        if spikes.shape != (self.n_neurons,):
            raise ValueError(f"spikes must have shape ({self.n_neurons},), got {spikes.shape}")
        binary = (spikes > 0).astype(np.float64)
        left_activity = float(binary[0])
        right_activity = float(binary[1])
        scores = {"move_left": left_activity, "move_right": right_activity, "stay": 0.0}
        max_score = max(scores.values())
        second_score = sorted(scores.values(), reverse=True)[1]
        if abs(left_activity - right_activity) < 1e-10:
            action_type = "stay"
        elif left_activity > right_activity:
            action_type = "move_left"
        else:
            action_type = "move_right"
        if max_score > 0:
            confidence = (max_score - second_score) / max_score
        else:
            confidence = 0.0
        confidence = float(np.clip(confidence, 0.0, 1.0))
        best_score = float(scores[action_type])
        action_id = f"{action_type}:L:{left_activity:.4f}:R:{right_activity:.4f}"
        return ActionProposal(action_id=action_id, action_type=action_type, score=best_score, confidence=confidence, source_window_steps=int(window_steps))
