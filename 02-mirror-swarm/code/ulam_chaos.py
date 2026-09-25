"""Автоматы Улама — генератор управляемого хаоса (творчество роя).

Источник: KEM-4.2-SOVEREIGN-EDITION-Technical-Manifesto.pdf §1.5:
  «Цель: система не должна быть детерминирована. Иначе она застревает
  в локальных минимумах (День Сурка).»
  Квантовая монета: P(1) = 1/phi ~ 0.618, P(0) ~ 0.382.

Назначение в рое:
  - разрыв эргодичности (Black Swan Solutions);
  - креативно-резонансный механизм («АРФА концепций физического
    консилиума.pdf»);
  - инъекции ТОЛЬКО в exploration-фазе (mirror_policy.yaml).
"""
from __future__ import annotations

import itertools

import numpy as np

PHI = 1.618033988749895


class UlamAutomaton:
    def __init__(self, grid_size: int = 16, rule: str = "majority",
                 seed=None):
        self.rng = np.random.default_rng(seed)
        self.grid = self.rng.integers(0, 2, (grid_size, grid_size))
        self.rule = rule
        self.phi = PHI

    def step(self) -> np.ndarray:
        """Одна итерация эволюции. Генерирует управляемый хаос."""
        n = self.grid.shape[0]
        new = self.grid.copy()
        for i, j in itertools.product(range(n), range(n)):
            i0, i1 = max(0, i - 1), min(n, i + 2)
            j0, j1 = max(0, j - 1), min(n, j + 2)
            neighbors = int(self.grid[i0:i1, j0:j1].sum())
            coin = int(self.rng.choice([0, 1],
                                       p=[1.0 / self.phi,
                                          1.0 - 1.0 / self.phi]))
            if self.rule == "majority":
                new[i, j] = 1 if (neighbors + coin) > 4 else 0
        self.grid = new
        return self.grid

    def perturbation(self, shape) -> np.ndarray:
        """Малое возмущение для исследовательского шума в сигналах роя."""
        self.step()
        need = int(np.prod(shape))
        flat = self.grid.flatten()
        while flat.size < need:
            self.step()
            flat = np.concatenate([flat, self.grid.flatten()])
        return (flat[:need].reshape(shape) - 0.5) * 0.02  # амплитуда ±0.01

    def novelty_score(self) -> float:
        """Насколько текущее состояние далеко от стационарного (0..1)."""
        before = self.grid.copy()
        self.step()
        return float(np.mean(before != self.grid))


if __name__ == "__main__":
    ua = UlamAutomaton(seed=42)
    for ep in range(5):
        noise = ua.perturbation((10,))
        print(f"эпизод {ep}: новизна={ua.novelty_score():.2f} "
              f"возмущение[0:3]={np.round(noise[:3], 4)}")
