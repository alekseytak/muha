"""Система Учитель -> Ученик -> Омега.

Диалектика (Гегель): тезис (Учитель) <-> антитезис (Ученик)
-> синтез (новый Учитель). Стремление к Омеге (Тейяр де Шарден):
каждое поколение приближает рой к коллективному сверхразуму
через канонические прецеденты Registry.

Promotion path (Паттерн Желязны — прохождение даёт силу):
    shadow -> canary -> limited -> full -> registrar_export

Дистилляция:
    L = l1*L_rl + l2*L_bc + l3*L_spike_kl + l4*L_value
        + l5*L_energy + l6*L_policy_violation + l7*L_sparsity

Учитель НЕ имеет capability WR: он даёт обучающие сигналы,
а в Registry пишет только Registrar.
"""
from __future__ import annotations

from dataclasses import dataclass

MODES = ("shadow", "canary", "limited", "full", "registrar_export")

PROMOTION_GATES = {
    "shadow": 0.80,     # agreement с учителем
    "canary": 0.85,     # успех на малой доле реальных действий
    "limited": 0.90,    # успех в разрешённых доменах
    "full": 0.95,       # полная локальная автономия
}


@dataclass
class Student:
    student_id: str
    mode: str = "shadow"
    success_ema: float = 0.5
    violations: int = 0

    def observe(self, action_proposed, action_executed, success: bool) -> dict:
        """В shadow-режиме ученик только предлагает; исполняет учитель."""
        self.success_ema = 0.95 * self.success_ema + 0.05 * float(success)
        agreement = float(action_proposed == action_executed)
        return {"mode": self.mode,
                "agreement": agreement,
                "success_ema": round(self.success_ema, 3)}

    def register_violation(self) -> None:
        """Нарушение политики отбрасывает ученика в shadow-режим."""
        self.violations += 1
        self.mode = "shadow"

    def eligible_for_promotion(self) -> bool:
        gate = PROMOTION_GATES.get(self.mode, 1.1)
        return self.violations == 0 and self.success_ema >= gate

    def promote(self) -> str:
        if self.eligible_for_promotion():
            i = MODES.index(self.mode)
            self.mode = MODES[min(i + 1, len(MODES) - 1)]
        return self.mode


class Teacher:
    """Учитель: целевые действия, dense reward, объяснения, curriculum."""

    def __init__(self, policy_fn, explain_fn=None):
        self.policy_fn = policy_fn
        self.explain_fn = explain_fn or (lambda ctx: "объяснение недоступно")

    def act(self, state):
        return self.policy_fn(state)

    def curriculum(self, student: Student, tasks: list) -> list:
        """Отбор задач вокруг текущего уровня ученика (+-0.2 сложности)."""
        lo = student.success_ema - 0.2
        hi = student.success_ema + 0.2
        picked = [t for t in tasks if lo <= t.get("difficulty", 0.5) <= hi]
        return picked or tasks[:1]

    def explain_penalty(self, state, action) -> str:
        return self.explain_fn({"state": state, "action": action})


def distillation_loss(*, rl: float, bc: float, spike_kl: float,
                      value_mse: float, energy: float,
                      policy_violation: float, sparsity: float,
                      weights=(1.0, 0.5, 0.3, 0.3, 0.1, 1.0, 0.05)) -> float:
    """Суммарный лосс ученика (все слагаемые >= 0, кроме rl-advantage)."""
    terms = (rl, bc, spike_kl, value_mse, energy, policy_violation, sparsity)
    return sum(w * float(t) for w, t in zip(weights, terms))


if __name__ == "__main__":
    st = Student("FlyStudent-01")
    teacher = Teacher(policy_fn=lambda s: "left")
    tasks = [{"id": 1, "difficulty": 0.4}, {"id": 2, "difficulty": 0.6},
             {"id": 3, "difficulty": 0.9}]
    print("curriculum:", teacher.curriculum(st, tasks))
    for _ in range(60):                       # имитация успешной практики
        st.observe("left", "left", success=True)
    print("режим после обучения:", st.promote(), "| success_ema:",
          round(st.success_ema, 3))
    st.register_violation()
    print("после нарушения:", st.mode)
