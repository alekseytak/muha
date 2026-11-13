"""GateKeeper — управление правами (ALLOW/DENY/ESCALATE)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time

from .verb_act import VerbAct, VerbActType, VerbActGrammar


class GateDecision(Enum):
    """Решения GateKeeper."""
    ALLOW = "ALLOW"
    DENY = "DENY"
    ESCALATE = "ESCALATE"


@dataclass
class GateResult:
    """Результат проверки GateKeeper."""
    decision: GateDecision
    reason: str
    required_role: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class GateKeeper:
    """Гражданин-страж: проверяет, можно ли выполнить действие.

    Права ролей берутся из `config.yaml -> gatekeeper.roles`; значения ниже —
    только дефолт, если конфиг не передал ничего (иначе конфиг и код молча
    расходятся, и правки в конфиге не действуют).
    """

    ROLE_HIERARCHY: List[str] = [
        "observer",
        "analyst",
        "decider",
        "guardian",
        "registrar",
    ]

    ROLE_PERMISSIONS: dict[str, set[str]] = {
        "observer": {"read", "list", "think", "select"},
        "analyst": {"read", "write", "log", "think", "select"},
        "decider": {"read", "write", "execute", "log", "think", "recommend",
                    "select", "escalate"},
        "guardian": {"read", "write", "execute", "veto", "escalate", "log",
                     "think", "select", "recommend"},
        "registrar": {"read", "write", "execute", "version", "anchor",
                      "escalate", "log", "think", "select", "recommend"},
    }

    def __init__(
        self,
        current_role: str = "decider",
        roles: Optional[dict] = None,
    ) -> None:
        self.current_role = current_role
        self.decisions: List[GateResult] = []
        self.veto_count = 0
        self.escalation_count = 0
        self.role_permissions = dict(self.ROLE_PERMISSIONS)
        if roles:
            for role, verbs in roles.items():
                if isinstance(verbs, (list, tuple, set)):
                    self.role_permissions[str(role)] = {str(v) for v in verbs}

    def check(self, verb_act: VerbAct, user_role: Optional[str] = None) -> GateResult:
        """Проверяет действие через GateKeeper."""
        role = user_role or self.current_role
        verb = verb_act.verb.value
        domain = VerbActGrammar.domain_for_object(verb_act.object)

        # 1. Проверка грамматики домена
        if not VerbActGrammar.is_allowed(domain, verb_act.verb):
            result = GateResult(
                decision=GateDecision.DENY,
                reason=f"Глагол '{verb}' недопустим в домене '{domain}'",
                required_role=None,
            )
            self._record(result)
            return result

        # 2. Проверка прав роли
        allowed_verbs = self.role_permissions.get(role, set())
        if verb not in allowed_verbs:
            # Пытаемся найти более высокую роль
            higher_role = self._find_higher_role_with_verb(role, verb)
            if higher_role:
                result = GateResult(
                    decision=GateDecision.ESCALATE,
                    reason=f"Требуется роль '{higher_role}' для глагола '{verb}'",
                    required_role=higher_role,
                )
            else:
                result = GateResult(
                    decision=GateDecision.DENY,
                    reason=f"Роль '{role}' не имеет права на глагол '{verb}'",
                    required_role=None,
                )
            self._record(result)
            return result

        # 3. Особые проверки для dangereux действий
        if verb == "execute":
            result = GateResult(
                decision=GateDecision.ALLOW,
                reason=f"Выполнение разрешено для роли '{role}'",
                required_role=role,
            )
        elif verb == "anchor":
            result = GateResult(
                decision=GateDecision.ALLOW,
                reason=f"Якорение разрешено для роли '{role}'",
                required_role=role,
            )
        else:
            result = GateResult(
                decision=GateDecision.ALLOW,
                reason=f"Действие разрешено",
                required_role=role,
            )

        self._record(result)
        return result

    def _find_higher_role_with_verb(self, current_role: str, verb: str) -> Optional[str]:
        """Находит более высокую роль, у которой есть право на глагол."""
        try:
            idx = self.ROLE_HIERARCHY.index(current_role)
        except ValueError:
            return None
        for higher in self.ROLE_HIERARCHY[idx + 1:]:
            if verb in self.role_permissions.get(higher, set()):
                return higher
        return None

    def veto(self, reason: str = "Вето Стража") -> GateResult:
        """Страж накладывает вето."""
        self.veto_count += 1
        result = GateResult(
            decision=GateDecision.DENY,
            reason=f"{reason} (вето #{self.veto_count})",
            required_role=None,
        )
        self._record(result)
        return result

    def escalate(self, reason: str = "Эскалация") -> GateResult:
        """Эскалация на более высокий уровень."""
        self.escalation_count += 1
        result = GateResult(
            decision=GateDecision.ESCALATE,
            reason=f"{reason} (эскалация #{self.escalation_count})",
            required_role=self._next_role(self.current_role),
        )
        self._record(result)
        return result

    def _next_role(self, current: str) -> Optional[str]:
        try:
            idx = self.ROLE_HIERARCHY.index(current)
        except ValueError:
            return None
        if idx + 1 < len(self.ROLE_HIERARCHY):
            return self.ROLE_HIERARCHY[idx + 1]
        return None

    def _record(self, result: GateResult) -> None:
        self.decisions.append(result)
        if len(self.decisions) > 1000:
            self.decisions = self.decisions[-1000:]

    def stats(self) -> dict[str, int]:
        allow = sum(1 for d in self.decisions if d.decision == GateDecision.ALLOW)
        deny = sum(1 for d in self.decisions if d.decision == GateDecision.DENY)
        escalate = sum(1 for d in self.decisions if d.decision == GateDecision.ESCALATE)
        return {"allow": allow, "deny": deny, "escalate": escalate, "total": len(self.decisions)}
