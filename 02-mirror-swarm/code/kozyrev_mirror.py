"""Временные зеркала Козырева — рефлексия перед рискованным действием.

Источник: KEM-4.2-SOVEREIGN-EDITION-Technical-Manifesto.pdf §1.4:
  «Решение идёт вперёд по времени. Будущее возвращает искажённый
  сигнал => парадокс.» Дисгармония = рассогласование решения и эха.
  k_paradox > 0.66 -> Quantum Escape (у нас: эскалация Guardian).

Роль в OS Глаголов: зеркало — это ATQEC-слой. Оно «оценивает» (L1)
и «маркирует» (L1) парадоксальность, но НЕ блокирует напрямую:
решение об эскалации принимает Guardian через verb-act.
"""
from __future__ import annotations

import collections

import numpy as np


class KozyrevMirror:
    def __init__(self, delay_us: float = 2.3, buffer_size: int = 256,
                 simulator=None):
        self.delay = delay_us * 1e-6          # асимметрия времени ~2.3 мкс
        self.buffer = collections.deque(maxlen=buffer_size)
        self.simulator = simulator or self._default_sim

    @staticmethod
    def _default_sim(signal: np.ndarray) -> np.ndarray:
        """Простейшая модель последствий: затухание + фазовый сдвиг.

        В реальном проекте сюда подключается симулятор среды
        (MuJoCo fly / 2D-мир) или forward-модель роя.
        """
        rolled = np.roll(signal, 1)
        return 0.9 * rolled - 0.1 * signal

    def backward_reflect(self, consequences: np.ndarray) -> np.ndarray:
        """Обратный сигнал из 'будущего': инверсия порядка + задержка."""
        echo = consequences[::-1].copy()
        shift = max(1, int(self.delay * 1e6))
        return np.roll(echo, shift)

    def reflect_decision(self, decision_signal: np.ndarray) -> dict:
        """Полный цикл зеркала: решение -> последствия -> эхо -> диссонанс."""
        decision_signal = np.asarray(decision_signal, dtype=float)
        self.buffer.append(decision_signal)
        consequences = self.simulator(decision_signal)
        echo = self.backward_reflect(consequences)
        dissonance = float(np.linalg.norm(decision_signal - echo))
        k_paradox = min(1.0, dissonance / 0.5)
        return {
            "echo": echo,
            "dissonance": round(dissonance, 6),
            "paradox_likelihood": round(k_paradox, 4),
            "quantum_escape": k_paradox > 0.66,
            "verb_act_hint": {
                "role": "Analyst",
                "verb": "оценивает",
                "domain": "atqec",
            },
        }

    def temporal_history_dissonance(self) -> float:
        """Диссонанс всей буферизованной истории (для ATQEC-отчёта)."""
        if len(self.buffer) < 2:
            return 0.0
        stack = np.stack(list(self.buffer))
        diffs = np.diff(stack, axis=0)
        return float(np.abs(diffs).mean())


if __name__ == "__main__":
    mirror = KozyrevMirror()
    calm = np.array([0.5, 0.52, 0.49, 0.51, 0.50])
    wild = np.array([0.1, 0.9, -0.8, 0.95, -0.2])
    r1 = mirror.reflect_decision(calm)
    r2 = mirror.reflect_decision(wild)
    print("спокойное решение: paradox =", r1["paradox_likelihood"],
          "| escape =", r1["quantum_escape"])
    print("дикое решение:     paradox =", r2["paradox_likelihood"],
          "| escape =", r2["quantum_escape"])
