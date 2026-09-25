"""Трёхфакторная пластичность: eligibility trace x нейромодулятор.

    dw_ij = e_ij * M

Композитный модуляторный сигнал M:
    M = a*task_reward - b*safety - g*energy - d*policy_violation
        + e*coherence + z*advantage

Биология: дофамин (награда), октопамин/серотонин (штраф).
Политика: policy_violation — сильнейший штраф (OS Глаголов).
Когнитивный слой: coherence — из АРФА/ATQEC (KEM 4.2).

Защита от runaway plasticity: клиппинг весов, синаптическая деградация,
гомеостаз частоты, спайковый бюджет (учитывается в energy_penalty).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlasticityParams:
    a_plus: float = 0.010        # усиление pre->post
    a_minus: float = 0.012       # ослабление post->pre
    decay_trace: float = 0.95    # распад eligibility-следа
    mod_clip: float = 2.0        # ограничение модулятора
    dopamine_gain: float = 0.30
    punishment_gain: float = 0.50
    energy_gain: float = 0.10
    violation_gain: float = 1.00
    coherence_gain: float = 0.20
    advantage_gain: float = 0.25
    w_min: float = -0.5
    w_max: float = 1.0
    synaptic_decay: float = 0.9999
    homeostatic_rate_hz: float = 5.0
    homeostatic_gain: float = 0.002


class RewardModulatedSTDP:
    """Пластичность поверх FlyBrain (матрица весов brain.W)."""

    def __init__(self, brain, params: PlasticityParams | None = None):
        self.brain = brain
        self.p = params or PlasticityParams()
        n = brain.n
        self.e_ij = np.zeros((n, n))   # eligibility-следы синапсов
        self.rate_ema = np.zeros(n)    # экспоненциальное среднее частот

    # --- шаг: обновить следы по спайкам ---
    def step(self, spikes: np.ndarray) -> None:
        p = self.p
        s = spikes.astype(float)
        self.e_ij += p.a_plus * np.outer(s, s)    # pre -> post
        self.e_ij -= p.a_minus * np.outer(s, s)   # симметричное ослабление
        self.e_ij *= p.decay_trace
        hz = 1000.0 / self.brain.p.dt_ms
        self.rate_ema = 0.99 * self.rate_ema + 0.01 * s * hz

    # --- композитный модулятор ---
    def modulator(self, *, task_reward: float = 0.0, safety_penalty: float = 0.0,
                  energy_penalty: float = 0.0, policy_violation: bool = False,
                  coherence: float = 0.0, advantage: float = 0.0) -> float:
        p = self.p
        return (p.dopamine_gain * task_reward
                - p.punishment_gain * safety_penalty
                - p.energy_gain * energy_penalty
                - p.violation_gain * float(policy_violation)
                + p.coherence_gain * coherence
                + p.advantage_gain * advantage)

    # --- применение: трёхфакторное обновление весов ---
    def apply(self, M: float) -> None:
        p, W = self.p, self.brain.W
        W += self.e_ij * float(np.clip(M, -p.mod_clip, p.mod_clip))
        W *= p.synaptic_decay
        np.clip(W, p.w_min, p.w_max, out=W)
        self._homeostasis()

    def _homeostasis(self) -> None:
        """Держим среднюю частоту около цели: молчунов поднимаем, болтунов гасим."""
        p = self.p
        delta = p.homeostatic_rate_hz - self.rate_ema
        self.brain.V += p.homeostatic_gain * delta * 100.0

    # --- объяснимость: горячие точки для trace ---
    def top_synapses(self, k: int = 5) -> list[dict]:
        flat = np.abs(self.e_ij).flatten()
        idx = np.argsort(flat)[-k:][::-1]
        n = self.brain.n
        return [{"pre": int(i // n), "post": int(i % n),
                 "eligibility": float(self.e_ij[i // n, i % n])}
                for i in idx]
