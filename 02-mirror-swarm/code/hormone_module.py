"""Гормональный модуль «Mood Organ» (Ф.К. Дик) для агента.

Источник: «Эмоции ии .txt» (документ владельца):
  радость = дофамин↑ серотонин↑ окситоцин↑
  злость  = адреналин↑
  страх   = кортизол↑ норадреналин↑
Шаги из дока:
  1) сбор уровней -> 2) индекс эмоции -> 3) влияние на тон/решения
  -> 4) обратная связь (on_reward / on_punishment).

Резонансные частоты нейромедиаторов («Код на фильтры.txt», csc_filter):
  дофамин 25 Гц, ГАМК 15 Гц, серотонин 35 Гц.
Октавы Буданова («АРФА-анализ коннектома дрозофилы_.pdf»):
  Ω1 = 12.67 Гц (дофаминергические пути),
  Ω5 = 86.84 Гц (серотониновые рецепторы),
  Ω8 = 367.84 Гц (высокочастотная когерентность анаклеарных структур).
"""
from __future__ import annotations

from dataclasses import dataclass, field

BASELINE = {
    "dopamine": 50.0, "serotonin": 50.0, "oxytocin": 50.0,
    "adrenaline": 30.0, "cortisol": 20.0, "noradrenaline": 30.0,
    "octopamine": 30.0, "gaba": 50.0,
}

RESONANCE_HZ = {"dopamine": 25.0, "gaba": 15.0, "serotonin": 35.0}
OCTAVE_HZ = {"dopamine_path": 12.67, "serotonin": 86.84,
             "coherence": 367.84}

EMOTION_STYLES = {
    "радость": {"style": "позитивный", "risk_appetite": +0.10,
                "learning_rate_scale": 1.2},
    "злость":  {"style": "резкий", "risk_appetite": -0.20,
                "learning_rate_scale": 0.9},
    "страх":   {"style": "осторожный-эмпатичный", "risk_appetite": -0.40,
                "learning_rate_scale": 0.5},
    "нейтрально": {"style": "нейтральный", "risk_appetite": 0.0,
                   "learning_rate_scale": 1.0},
}


@dataclass
class HormoneModule:
    state: dict = field(default_factory=lambda: dict(BASELINE))
    decay: float = 0.98

    # --- шаг 4: обратная связь ---
    def on_reward(self, r: float) -> None:
        r = max(0.0, float(r))
        self.state["dopamine"] = min(100.0, self.state["dopamine"] + 8.0 * r)
        self.state["serotonin"] = min(100.0, self.state["serotonin"] + 4.0 * r)
        self.state["oxytocin"] = min(100.0, self.state["oxytocin"] + 2.0 * r)

    def on_punishment(self, s: float) -> None:
        s = max(0.0, float(s))
        self.state["octopamine"] = min(100.0, self.state["octopamine"] + 8.0 * s)
        self.state["cortisol"] = min(100.0, self.state["cortisol"] + 6.0 * s)
        self.state["adrenaline"] = min(100.0, self.state["adrenaline"] + 5.0 * s)
        self.state["noradrenaline"] = min(
            100.0, self.state["noradrenaline"] + 4.0 * s)

    def tick(self) -> None:
        """Возврат к базовой линии (гомеостаз настроения)."""
        for k in self.state:
            base = BASELINE[k]
            self.state[k] = base + (self.state[k] - base) * self.decay

    # --- шаг 2: индекс эмоционального состояния ---
    def emotion(self) -> tuple:
        s = self.state
        joy = (s["dopamine"] + s["serotonin"] + s["oxytocin"]) / 3.0
        anger = s["adrenaline"]
        fear = (s["cortisol"] + s["noradrenaline"]) / 2.0
        candidates = ((joy - 60.0, "радость"),
                      (anger - 55.0, "злость"),
                      (fear - 45.0, "страх"))
        best = max(candidates, key=lambda c: c[0])
        if best[0] <= 0:
            return "нейтрально", 0.0
        return best[1], min(1.0, best[0] / 40.0)

    # --- влияние на познание (Д. Киз: интеллект и эмоции связаны) ---
    def cognitive_ability(self) -> float:
        """Интеллект = дофамин / (кортизол + eps). Стресс сужает мышление."""
        return self.state["dopamine"] / (self.state["cortisol"] + 10.0)

    # --- шаг 3: тон и аппетит к риску ---
    def tone_for_response(self) -> dict:
        name, intensity = self.emotion()
        style = dict(EMOTION_STYLES.get(name, EMOTION_STYLES["нейтрально"]))
        style.update({"emotion": name, "intensity": round(intensity, 3),
                      "cognitive_ability": round(self.cognitive_ability(), 3)})
        return style


if __name__ == "__main__":
    hm = HormoneModule()
    print("старт:", hm.emotion())
    hm.on_reward(2.0)
    print("после награды:", hm.tone_for_response())
    for _ in range(50):
        hm.tick()
    hm.on_punishment(3.0)
    print("после штрафа:", hm.tone_for_response())
