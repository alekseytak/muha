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


class TestDecisionInvariants:
    """P3.1: an ALLOW that drops the proposal is unconstructable, not just unused."""

    def test_allow_with_swapped_action_cannot_be_built(self):
        with pytest.raises(ValueError, match="ALLOW must preserve"):
            PolicyDecision(status="ALLOW", proposed_action="move_left", enforced_action="stay", reason="forged")

    def test_deny_with_executed_action_cannot_be_built(self):
        with pytest.raises(ValueError, match="DENY must enforce"):
            PolicyDecision(status="DENY", proposed_action="move_left", enforced_action="move_left", reason="forged")

    def test_escalate_with_swapped_action_cannot_be_built(self):
        with pytest.raises(ValueError, match="ESCALATE must enforce"):
            PolicyDecision(status="ESCALATE", proposed_action="move_right", enforced_action="move_right", reason="forged")

    def test_unknown_status_rejected(self):
        with pytest.raises(ValueError, match="invalid status"):
            PolicyDecision(status="MAYBE", proposed_action="stay", enforced_action="stay", reason="")

    def test_every_gatekeeper_decision_carries_the_proposal(self, gk):
        for verb, context in (("выбирает", {}), ("якорит", {}), ("выбирает", {"risk_level": "high"})):
            d = gk.evaluate(_make_va("Decider", verb, context, proposed_action="move_right"))
            assert d.proposed_action == "move_right"


class TestVerbActFromProposal:
    """proposed_action must be derivable only from the decoder's proposal."""

    def test_copies_action_type_and_ref(self):
        from types import SimpleNamespace

        proposal = SimpleNamespace(action_id="prop-77", action_type="move_left", confidence=0.9)
        va = VerbAct.from_proposal(
            proposal,
            event_id="evt-1", subject_id="agent-001", role="Decider", verb="выбирает",
            domain="simulation", trace_id="trace-1",
        )
        assert va.proposed_action == "move_left"
        assert va.proposal_ref == "prop-77" and va.object_ref == "prop-77"

    def test_rejects_unknown_action_type(self):
        from types import SimpleNamespace

        proposal = SimpleNamespace(action_id="prop-78", action_type="fly_away", confidence=0.9)
        with pytest.raises(ValueError, match="action_type"):
            VerbAct.from_proposal(
                proposal,
                event_id="evt-1", subject_id="agent-001", role="Decider", verb="выбирает",
                domain="simulation", trace_id="trace-1",
            )

    def test_proposal_survives_the_governance_round_trip(self, gk):
        from types import SimpleNamespace

        proposal = SimpleNamespace(action_id="prop-79", action_type="move_left", confidence=0.9)
        va = VerbAct.from_proposal(
            proposal,
            event_id="evt-1", subject_id="agent-001", role="Decider", verb="выбирает",
            domain="simulation", trace_id="trace-1",
        )
        decision = gk.evaluate(va)
        assert (decision.status, decision.enforced_action) == ("ALLOW", "move_left")


class TestTypeSafety:
    def test_non_verbact_raises(self, gk):
        with pytest.raises(ValueError, match="Expected VerbAct"): gk.evaluate({"role": "Decider"})
    def test_decision_frozen(self, gk):
        d = gk.evaluate(_make_va()); 
        with pytest.raises(Exception): d.status = "DENY"
