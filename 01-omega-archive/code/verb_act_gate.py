"""OS Глаголов · Runtime Gatekeeper (Layer C).

Источники:
  - OS-Glagolov-Charter.md (п.3 verb-act, п.5 роли, п.6 capabilities, п.7 домены)
  - os-glagolov-theory-ru.md (п.9 минимальный контракт)
  - verbs.json (словарь глаголов L0-L5 x домены)

Главное правило: ни один субъект не исполняет действие только потому,
что модель «может» его сгенерировать.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Callable, Optional

# --- словарь глаголов (фрагмент verbs.json; полный — в policy/) ---
VERB_LEVELS = {
    "читает": "L0", "наблюдает": "L0", "считывает": "L0", "ищет": "L0",
    "извлекает": "L1", "сопоставляет": "L1", "классифицирует": "L1",
    "оценивает": "L1", "ранжирует": "L1", "объясняет": "L1",
    "маркирует": "L1", "предсказывает": "L1",
    "фильтрует": "L2", "нормализует": "L2", "синтезирует": "L2",
    "декомпозирует": "L2", "агрегирует": "L2", "маршрутизирует": "L2",
    "кэширует": "L2", "архивирует": "L2",
    "рекомендует": "L3", "приоритизирует": "L3", "выбирает": "L3",
    "отклоняет": "L4", "эскалирует": "L4", "блокирует": "L4",
    "останавливает": "L4", "ограничивает": "L4", "разрешает": "L4",
    "переключает": "L4",
    "сериализует": "L5", "экспортирует": "L5", "хэширует": "L5",
    "аттестует": "L5", "подписывает": "L5", "версирует": "L5",
    "якорит": "L5", "публикует": "L5",
}

ROLE_CAPABILITIES = {
    "Observer":    {"R", "WL"},
    "Analyst":     {"R", "WL"},
    "Synthesizer": {"R", "WL"},
    "Decider":     {"R", "WL", "X", "E"},
    "Guardian":    {"R", "WL", "X", "E"},
    "Registrar":   {"R", "WL", "WE", "WR", "A"},
}

ROLE_VERBS = {
    "Observer":    {"читает", "наблюдает", "считывает", "ищет"},
    "Analyst":     {"извлекает", "сопоставляет", "классифицирует", "оценивает",
                    "объясняет", "предсказывает", "маркирует", "ранжирует"},
    "Synthesizer": {"агрегирует", "синтезирует", "нормализует", "декомпозирует",
                    "фильтрует", "кэширует", "архивирует", "маршрутизирует"},
    "Decider":     {"рекомендует", "приоритизирует", "выбирает",
                    "эскалирует", "отклоняет"},
    "Guardian":    {"блокирует", "останавливает", "ограничивает",
                    "разрешает", "эскалирует", "переключает"},
    "Registrar":   {"сериализует", "экспортирует", "хэширует", "аттестует",
                    "якорит", "версирует", "подписывает", "публикует"},
}

L5_VERBS = {v for v, lvl in VERB_LEVELS.items() if lvl == "L5"}


@dataclass
class PolicyDecision:
    allow: bool
    outcome: str          # allow | deny | escalate | attest
    reason: str = ""
    requires_attestation: bool = False
    requires_escalation: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


class GateKeeper:
    """verb-act -> policy-check -> allow / deny / escalate / attest."""

    def __init__(self, irreversible_verbs: Optional[set] = None):
        self.irreversible_verbs = irreversible_verbs or {
            "останавливает", "переключает", "блокирует"}

    def check(self, act: dict) -> PolicyDecision:
        verb = act.get("verb", "")
        role = act.get("role", "")
        domain = act.get("domain", "")
        ctx = act.get("context", {}) or {}

        # 1. глагол должен быть в словаре os-glagolov/1.0
        if verb not in VERB_LEVELS:
            return PolicyDecision(False, "deny",
                                  f"глагол '{verb}' вне словаря")

        # 2. роль должна иметь право на этот глагол
        if verb not in ROLE_VERBS.get(role, set()):
            return PolicyDecision(False, "deny",
                                  f"роли '{role}' не разрешён глагол '{verb}'")

        # 3. L5 (commit) — только Registrar + аттестация (Charter п.4)
        if verb in L5_VERBS:
            if role != "Registrar":
                return PolicyDecision(False, "deny",
                                      "L5 принадлежит только Registrar (capability WR)")
            return PolicyDecision(True, "attest",
                                  "L5 требует trace и аттестации",
                                  requires_attestation=True)

        # 4. hardware: необратимое + высокий риск без человека -> escalate
        if domain == "hardware" and verb in self.irreversible_verbs:
            if ctx.get("risk_level") == "high" and not ctx.get("human_in_loop"):
                return PolicyDecision(False, "escalate",
                                      "необратимое физическое действие высокого "
                                      "риска без подтверждения",
                                      requires_escalation=True)

        # 5. ATQEC: фильтр не выполняет финальный commit (Charter п.7)
        if domain == "atqec" and verb in {"публикует", "якорит", "подписывает"}:
            return PolicyDecision(False, "deny",
                                  "ATQEC-слой не выполняет финальный commit")

        # 6. memory: перезапись канона — только через отдельную policy
        if domain == "memory" and ctx.get("overwrites_canonical"):
            return PolicyDecision(False, "escalate",
                                  "переопределение канонической памяти "
                                  "требует отдельной policy",
                                  requires_escalation=True)

        # 7. medical / legal: commit-действия требуют человека
        if domain in {"medical", "legal"} and ctx.get("is_commit_action"):
            return PolicyDecision(False, "escalate",
                                  f"{domain}: commit требует внешней авторизации",
                                  requires_escalation=True)

        return PolicyDecision(True, "allow", "policy-check пройден")


def make_trace(act: dict, decision: PolicyDecision) -> str:
    """Хэш следа действия — для истории становления, а не для надзора."""
    payload = json.dumps({"act": act, "decision": decision.as_dict(),
                          "ts": time.time()},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def execute(act: dict, gate: GateKeeper,
            on_action: Optional[Callable] = None) -> dict:
    """Прогнать verb-act через gate; при allow — исполнить on_action(act)."""
    decision = gate.check(act)
    result = {
        "id": str(uuid.uuid4()),
        "policy": decision.as_dict(),
        "trace": make_trace(act, decision),
        "executed": False,
    }
    if decision.allow and on_action is not None:
        result["value"] = on_action(act)
        result["executed"] = True
    return result


if __name__ == "__main__":
    gate = GateKeeper()
    tests = [
        # (verb-act, ожидаемый outcome)
        ({"role": "Observer", "verb": "наблюдает", "domain": "hardware"}, "allow"),
        ({"role": "Observer", "verb": "публикует", "domain": "registry"}, "deny"),
        ({"role": "Decider", "verb": "якорит", "domain": "registry"}, "deny"),
        ({"role": "Registrar", "verb": "якорит", "domain": "registry"}, "attest"),
        ({"role": "Guardian", "verb": "блокирует", "domain": "hardware",
          "context": {"risk_level": "high", "human_in_loop": False}}, "escalate"),
        ({"role": "Decider", "verb": "выбирает", "domain": "hardware",
          "context": {"risk_level": "low"}}, "allow"),
        ({"role": "Analyst", "verb": "фильтрует", "domain": "atqec"}, "deny"),
        ({"role": "Synthesizer", "verb": "фильтрует", "domain": "atqec"}, "allow"),
    ]
    for act, expected in tests:
        got = gate.check(act).outcome
        mark = "OK " if got == expected else "FAIL"
        print(f"[{mark}] {act['role']:<11} {act['verb']:<12} -> {got} (ждём {expected})")
