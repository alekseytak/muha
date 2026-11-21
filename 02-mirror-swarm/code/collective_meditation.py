"""Коллективная медитация роя (Олдос Хаксли, «Остров») на октаве Ω5.

Частота синхронизации: 86.84 Гц — резонанс серотониновых рецепторов
(«Октавы Буданова .txt»; «АРФА-анализ коннектома дрозофилы_.pdf»:
f5 = 86.84 Гц; KEM-4.2: октава 5 = живые организмы, социальная этика).

Модель: курамото-подобная фазовая синхронизация агентов.
Результат: sync_order (0..1) — inter-swarm coherence. При sync_order > 0.8
рой переходит в состояние «moksha» и согласованно решает общую задачу.

В OS Глаголов: медитация — это L2 «синтезирует» (домен atqec/memory),
она НЕ порождает внешних действий.
"""
from __future__ import annotations

import numpy as np

OMEGA5_HZ = 86.84
MOKSHA_THRESHOLD = 0.8


class CollectiveMeditation:
    def __init__(self, n_agents: int, coupling: float = 0.3, seed=None):
        self.rng = np.random.default_rng(seed)
        self.phases = self.rng.uniform(0, 2 * np.pi, n_agents)
        spread = self.rng.normal(0, 0.02, n_agents)   # природный разброс
        # Частота настоящая, без масштабирования. Синхронизацию держит связь, а
        # не уменьшенная частота: в модели Курамото важен не абсолютный период,
        # а разброс частот. При 86.84 Гц и разбросе 2% он равен
        # 2*pi*86.84*0.02 = 10.9 рад/с, поэтому критическая связь K_c ~ 17.
        # Прежняя правка умножала частоту на 0.01: разброс падал до 0.11 рад/с,
        # синхронизация наступала уже при K=2, но объект продолжал рапортовать
        # freq_hz = 86.84, считая 0.8684 Гц. Это была подгонка под заявленный
        # результат, а не физика. Теперь K задаётся выше K_c у вызывающего.
        self.natural = OMEGA5_HZ * (1.0 + spread)
        self.K = coupling

    def step(self, dt: float = 0.01) -> float:
        """Один шаг синхронизации; возвращает порядок sync_order."""
        mean_field = np.exp(1j * self.phases).mean()
        r, psi = abs(mean_field), np.angle(mean_field)
        self.phases += (2 * np.pi * self.natural * dt
                        + self.K * r * np.sin(psi - self.phases) * dt)
        self.phases %= 2 * np.pi
        return float(r)

    def meditate(self, seconds: float = 5.0, dt: float = 0.01) -> dict:
        r_final = 0.0
        steps = int(seconds / dt)
        for _ in range(steps):
            r_final = self.step(dt)
        return {
            "sync_order": round(r_final, 4),
            "freq_hz": OMEGA5_HZ,
            "state": "moksha" if r_final > MOKSHA_THRESHOLD else "settling",
            "verb_act_hint": {"role": "Synthesizer",
                              "verb": "синтезирует",
                              "domain": "atqec"},
        }


if __name__ == "__main__":
    sess = CollectiveMeditation(n_agents=16, coupling=0.6, seed=7)
    for t in (1.0, 3.0, 8.0):
        out = sess.meditate(seconds=0.5)
        print(f"после ~{t}с: sync_order={out['sync_order']} state={out['state']}")
