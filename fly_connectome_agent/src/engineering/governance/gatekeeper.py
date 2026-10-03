"""GateKeeper: governance layer for VerbAct (P3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import uuid


_VALID_DOMAINS_P3 = {"simulation"}
_VALID_ACTIONS = {"move_left", "move_right", "stay"}
ProposedAction = Literal["move_left", "move_right", "stay"]


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
    proposed_action: ProposedAction

    @classmethod
    def from_proposal(
        cls,
        proposal: Any,
        *,
        event_id: str,
        subject_id: str,
        role: str,
        verb: str,
        domain: str,
        trace_id: str,
        context: dict | None = None,
    ) -> "VerbAct":
        """Build a VerbAct straight from an ActionProposal.

        Single sanctioned way to populate proposed_action: it is copied from
        proposal.action_type after a validity check, and proposal_ref /
        object_ref carry the proposal's action_id, so the governance record
        can never drift from what the decoder actually proposed.
        """
        action_type = getattr(proposal, "action_type", None)
        if action_type not in _VALID_ACTIONS:
            raise ValueError(
                f"proposal.action_type must be one of {sorted(_VALID_ACTIONS)}, "
                f"got {action_type!r}"
            )
        ref = proposal.action_id
        return cls(
            event_id=event_id,
            subject_id=subject_id,
            role=role,
            verb=verb,
            object_ref=ref,
            domain=domain,
            context=context if context is not None else {},
            trace_id=trace_id,
            proposal_ref=ref,
            proposed_action=action_type,
        )


@dataclass(frozen=True)
class PolicyDecision:
    status: Literal["ALLOW", "DENY", "ESCALATE"]
    proposed_action: str
    enforced_action: str
    reason: str

    def __post_init__(self) -> None:
        """Structural governance invariant (P3.1).

        ALLOW preserves the proposal; DENY/ESCALATE replace it with the safe
        fallback. A decision that violates this cannot be constructed at all —
        tests and callers can no longer forge ALLOW+stay.
        """
        if self.status not in ("ALLOW", "DENY", "ESCALATE"):
            raise ValueError(f"invalid status: {self.status!r}")
        if self.status == "ALLOW" and self.enforced_action != self.proposed_action:
            raise ValueError(
                f"ALLOW must preserve the proposal: proposed={self.proposed_action!r} "
                f"enforced={self.enforced_action!r}"
            )
        if self.status in ("DENY", "ESCALATE") and self.enforced_action != GateKeeper.SAFE_FALLBACK:
            raise ValueError(
                f"{self.status} must enforce '{GateKeeper.SAFE_FALLBACK}': "
                f"got {self.enforced_action!r}"
            )


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
                proposed_action=verb_act.proposed_action,
                enforced_action=self.SAFE_FALLBACK,
                reason=f"invalid domain: {verb_act.domain}",
            )

        if verb_act.proposed_action not in _VALID_ACTIONS:
            return PolicyDecision(
                status="DENY",
                proposed_action=str(verb_act.proposed_action),
                enforced_action=self.SAFE_FALLBACK,
                reason=f"invalid proposed_action: {verb_act.proposed_action}",
            )

        role = verb_act.role
        verb = verb_act.verb
        context = verb_act.context

        if role not in self._ALLOWED_VERBS:
            return PolicyDecision(status="DENY", proposed_action=verb_act.proposed_action, enforced_action=self.SAFE_FALLBACK, reason=f"unknown role: {role}")
        allowed = self._ALLOWED_VERBS[role]
        if verb not in allowed:
            return PolicyDecision(status="DENY", proposed_action=verb_act.proposed_action, enforced_action=self.SAFE_FALLBACK, reason=f"verb '{verb}' not allowed for role '{role}'")
        if self._is_high_risk(context):
            return PolicyDecision(status="ESCALATE", proposed_action=verb_act.proposed_action, enforced_action=self.SAFE_FALLBACK, reason="high-risk context detected")
        if verb in self._L5_VERBS and role != "Registrar":
            return PolicyDecision(status="DENY", proposed_action=verb_act.proposed_action, enforced_action=self.SAFE_FALLBACK, reason=f"L5 verb '{verb}' blocked for role '{role}'")

        return PolicyDecision(
            status="ALLOW",
            proposed_action=verb_act.proposed_action,
            enforced_action=verb_act.proposed_action,
            reason=f"allowed: {role}:{verb}",
        )

    def _is_high_risk(self, context: dict) -> bool:
        for key, val in context.items():
            if key in self._HIGH_RISK_CONTEXT_KEYS:
                if isinstance(val, str) and val.lower() in self._HIGH_RISK_VALUES:
                    return True
        return False
