"""Tests for GateKeeper governance."""
from __future__ import annotations
import pytest
from fly_connectome_agent.src.engineering.governance.gatekeeper import GateKeeper, VerbAct, PolicyDecision


def _make_va(role="Decider", verb="выбирает", context=None, proposal_ref="prop-001", proposed_action="move_left", domain="simulation"):
    return VerbAct(event_id="evt-001", subject_id="agent-001", role=role, verb=verb, object_ref="obj-001", domain=domain, context=context or {}, trace_id="trace-001", proposal_ref=proposal_ref, proposed_action=proposed_action)


@pytest.fixture
def gk(): return GateKeeper()


class TestDeciderPermissions:
    def test_decider_allows(self, gk):
        assert gk.evaluate(_make_va("Decider", "выбирает")).status == "ALLOW"
    def test_decider_anchor_denied(self, gk):
        d = gk.evaluate(_make_va("Decider", "якорит"))
        assert d.status == "DENY"; assert d.enforced_action == "stay"


class TestForbiddenVerbs:
    def test_unknown_role_denied(self, gk):
        assert gk.evaluate(_make_va("Hacker", "взломать")).status == "DENY"
    def test_forbidden_verb_denied(self, gk):
        assert gk.evaluate(_make_va("Decider", "удалить")).status == "DENY"


class TestHighRisk:
    def test_high_risk_escalates(self, gk):
        d = gk.evaluate(_make_va("Decider", "выбирает", {"risk_level": "high"}))
        assert d.status == "ESCALATE"; assert d.enforced_action == "stay"
    def test_high_value_escalates(self, gk):
        d = gk.evaluate(_make_va("Decider", "выбирает", {"danger": "critical"}))
        assert d.status == "ESCALATE"


class TestSafeFallback:
    def test_deny_stay(self, gk):
        assert gk.evaluate(_make_va("Decider", "якорит")).enforced_action == "stay"
    def test_escalate_stay(self, gk):
        d = gk.evaluate(_make_va("Decider", "выбирает", {"risk_level": "critical"}))
        assert d.enforced_action == "stay"


class TestGuardian:
    def test_guardian_block(self, gk):
        assert gk.evaluate(_make_va("Guardian", "блокирует")).status == "ALLOW"
    def test_guardian_escalate(self, gk):
        assert gk.evaluate(_make_va("Guardian", "эскалирует")).status == "ALLOW"
    def test_guardian_razresayet(self, gk):
        assert gk.evaluate(_make_va("Guardian", "разрешает")).status == "ALLOW"


class TestRegistrar:
    def test_registrar_anchor_allowed(self, gk):
        assert gk.evaluate(_make_va("Registrar", "якорит")).status == "ALLOW"
    def test_decider_anchor_blocked(self, gk):
        assert gk.evaluate(_make_va("Decider", "якорит")).status == "DENY"


class TestDomainValidation:
    def test_invalid_domain_denied(self, gk):
        d = gk.evaluate(_make_va(domain="production"))
        assert d.status == "DENY"; assert d.enforced_action == "stay"
    def test_simulated_domain_allowed(self, gk):
        d = gk.evaluate(_make_va(domain="simulation"))
        assert d.status == "ALLOW"


class TestAllowedActionPassesThrough:
    def test_allow_passes_proposed_move_left(self, gk):
        d = gk.evaluate(_make_va(verb="выбирает", proposed_action="move_left"))
        assert d.status == "ALLOW"; assert d.enforced_action == "move_left"
    def test_allow_passes_proposed_move_right(self, gk):
        d = gk.evaluate(_make_va(verb="выбирает", proposed_action="move_right"))
        assert d.status == "ALLOW"; assert d.enforced_action == "move_right"
    def test_allow_passes_proposed_stay(self, gk):
        d = gk.evaluate(_make_va(verb="выбирает", proposed_action="stay"))
        assert d.status == "ALLOW"; assert d.enforced_action == "stay"
    def test_invalid_proposed_action_denied(self, gk):
        d = gk.evaluate(_make_va(verb="выбирает", proposed_action="jump"))
        assert d.status == "DENY"; assert d.enforced_action == "stay"


class TestTypeSafety:
    def test_non_verbact_raises(self, gk):
        with pytest.raises(ValueError, match="Expected VerbAct"): gk.evaluate({"role": "Decider"})
    def test_decision_frozen(self, gk):
        d = gk.evaluate(_make_va()); 
        with pytest.raises(Exception): d.status = "DENY"
