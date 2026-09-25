"""GateKeeper: governance layer for VerbAct (P3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import uuid


_VALID_DOMAINS_P3 = {"simulation"}
_VALID_ACTIONS = {"move_left", "move_right", "stay"}


@dataclass(frozen=True)
class VerbAct:
    event_id: str
    subject_id: str
    role: str
    verb: str
    object_ref: str
    domain: str
    context: dict
    trace_id: str
    proposal_ref: str
    proposed_action: str


@dataclass(frozen=True)
class PolicyDecision:
    status: Literal["ALLOW", "DENY", "ESCALATE"]
    enforced_action: str
    reason: str


class GateKeeper:
    SAFE_FALLBACK = "stay"
    _ALLOWED_VERBS = {"Decider": {"выбирает"}, "Guardian": {"блокирует", "эскалирует", "разрешает"}, "Registrar": {"якорит"}}
    _L5_VERBS = {"якорит", "регистрирует", "записывает"}
    _HIGH_RISK_CONTEXT_KEYS = {"risk_level", "danger", "violation"}
    _HIGH_RISK_VALUES = {"high", "critical", "severe", "violation"}

    def __init__(self):
        pass

    def evaluate(self, verb_act: VerbAct) -> PolicyDecision:
        if not isinstance(verb_act, VerbAct):
            raise ValueError(f"Expected VerbAct, got {type(verb_act)}")

        if verb_act.domain not in _VALID_DOMAINS_P3:
            return PolicyDecision(
                status="DENY",
                enforced_action=self.SAFE_FALLBACK,
                reason=f"invalid domain: {verb_act.domain}",
            )

        if verb_act.proposed_action not in _VALID_ACTIONS:
            return PolicyDecision(
                status="DENY",
                enforced_action=self.SAFE_FALLBACK,
                reason=f"invalid proposed_action: {verb_act.proposed_action}",
            )

        role = verb_act.role
        verb = verb_act.verb
        context = verb_act.context

        if role not in self._ALLOWED_VERBS:
            return PolicyDecision(status="DENY", enforced_action=self.SAFE_FALLBACK, reason=f"unknown role: {role}")
        allowed = self._ALLOWED_VERBS[role]
        if verb not in allowed:
            return PolicyDecision(status="DENY", enforced_action=self.SAFE_FALLBACK, reason=f"verb '{verb}' not allowed for role '{role}'")
        if self._is_high_risk(context):
            return PolicyDecision(status="ESCALATE", enforced_action=self.SAFE_FALLBACK, reason="high-risk context detected")
        if verb in self._L5_VERBS and role != "Registrar":
            return PolicyDecision(status="DENY", enforced_action=self.SAFE_FALLBACK, reason=f"L5 verb '{verb}' blocked for role '{role}'")

        return PolicyDecision(
            status="ALLOW",
            enforced_action=verb_act.proposed_action,
            reason=f"allowed: {role}:{verb}",
        )

    def _is_high_risk(self, context: dict) -> bool:
        for key, val in context.items():
            if key in self._HIGH_RISK_CONTEXT_KEYS:
                if isinstance(val, str) and val.lower() in self._HIGH_RISK_VALUES:
                    return True
        return False
