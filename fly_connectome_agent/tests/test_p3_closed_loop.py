"""P3.1 closed-loop integration test.

One real loop, no fabricated decisions: decoder proposal -> VerbAct ->
GateKeeper -> environment step -> provenance record. The governance invariant
(ALLOW executes the proposal, DENY/ESCALATE execute the safe fallback) is
checked on the executed action, and tamper evidence is checked against the
bytes actually written to disk.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from fly_connectome_agent.src.engineering.action.action_decoder import ActionDecoder
from fly_connectome_agent.src.engineering.environments.simple_navigation import SimpleNavigation, EnvConfig
from fly_connectome_agent.src.engineering.governance.gatekeeper import GateKeeper, PolicyDecision, VerbAct
from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog


def _run_one_step(env, decoder, gk, prov, spikes, *, verb="выбирает", context=None, note="evt"):
    """One honest pass through the loop; returns (proposal, decision, observation)."""
    proposal = decoder.decode(spikes)
    verb_act = VerbAct.from_proposal(
        proposal,
        event_id=note, subject_id="agent-001", role="Decider", verb=verb,
        domain="simulation", trace_id="trace-p3", context=context or {},
    )
    decision = gk.evaluate(verb_act)
    obs, reward, terminated, truncated, info = env.step(decision.enforced_action)
    prov.append({
        "note": note,
        "proposal": proposal.action_type,
        "proposal_ref": proposal.action_id,
        "decision": decision.status,
        "enforced_action": decision.enforced_action,
        "position_after": int(info["position"]),
        "reward": float(reward),
    })
    return proposal, decision, obs


@pytest.fixture
def rig(tmp_path):
    env = SimpleNavigation(EnvConfig(length=5, init_position=2, target_position=4, max_steps=10))
    env.reset(seed=42)
    return {
        "env": env,
        "decoder": ActionDecoder(n_neurons=2),
        "gk": GateKeeper(),
        "prov": ProvenanceLog(path=str(tmp_path / "loop.jsonl")),
    }


class TestProposalIsExecuted:
    def test_left_proposal_moves_left(self, rig):
        proposal, decision, obs = _run_one_step(rig["env"], rig["decoder"], rig["gk"], rig["prov"], np.array([True, False]), note="left")
        assert proposal.action_type == "move_left"
        assert decision.status == "ALLOW"
        assert decision.enforced_action == "move_left"
        assert int(obs[0]) == 1  # position 2 -> 1, the proposal really reached the env

    def test_right_proposal_moves_right(self, rig):
        proposal, decision, obs = _run_one_step(rig["env"], rig["decoder"], rig["gk"], rig["prov"], np.array([False, True]), note="right")
        assert (decision.status, decision.enforced_action) == ("ALLOW", "move_right")
        assert int(obs[0]) == 3

    def test_tie_proposes_stay_and_is_executed_as_stay(self, rig):
        proposal, decision, obs = _run_one_step(rig["env"], rig["decoder"], rig["gk"], rig["prov"], np.array([True, True]), note="tie")
        assert (decision.status, decision.enforced_action) == ("ALLOW", "stay")
        assert int(obs[0]) == 2


class TestGovernanceOverrides:
    def test_deny_replaces_move_left_with_stay(self, rig):
        proposal, decision, obs = _run_one_step(
            rig["env"], rig["decoder"], rig["gk"], rig["prov"], np.array([True, False]),
            verb="якорит", note="deny",
        )
        assert proposal.action_type == "move_left"
        assert decision.status == "DENY"
        assert decision.proposed_action == "move_left"  # proposal still recorded
        assert decision.enforced_action == "stay"
        assert int(obs[0]) == 2  # no movement

    def test_escalate_replaces_move_right_with_stay(self, rig):
        proposal, decision, obs = _run_one_step(
            rig["env"], rig["decoder"], rig["gk"], rig["prov"], np.array([False, True]),
            context={"risk_level": "high"}, note="escalate",
        )
        assert (decision.status, decision.proposed_action, decision.enforced_action) == ("ESCALATE", "move_right", "stay")
        assert int(obs[0]) == 2

    def test_decision_invariant_holds_for_every_spike_pattern(self, rig):
        for pattern in ([True, False], [False, True], [False, False], [True, True]):
            _, decision, _ = _run_one_step(rig["env"], rig["decoder"], rig["gk"], rig["prov"], np.array(pattern), note="inv")
            if decision.status == "ALLOW":
                assert decision.enforced_action == decision.proposed_action
            else:
                assert decision.enforced_action == GateKeeper.SAFE_FALLBACK


class TestProvenanceHonesty:
    def test_record_separates_proposal_decision_and_action(self, rig):
        _run_one_step(rig["env"], rig["decoder"], rig["gk"], rig["prov"], np.array([True, False]), verb="якорит", note="one")
        payload = rig["prov"].get_events()[0].payload
        assert payload["proposal"] == "move_left"
        assert payload["decision"] == "DENY"
        assert payload["enforced_action"] == "stay"
        assert payload["position_after"] == 2

    def test_on_disk_tamper_is_detected(self, rig):
        path = rig["prov"].path
        for i, pattern in enumerate(([True, False], [False, True])):
            _run_one_step(rig["env"], rig["decoder"], rig["gk"], rig["prov"], np.array(pattern), note=f"step{i}")
        assert rig["prov"].verify_chain() is True
        with open(path) as f:
            lines = f.readlines()
        entry = json.loads(lines[0])
        entry["payload"]["enforced_action"] = "move_right"  # rewrite history on disk
        with open(path, "w") as f:
            for i, raw in enumerate(lines):
                f.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n" if i == 0 else raw)
        assert ProvenanceLog(path=path).verify_chain() is False

    def test_truncation_is_detected_by_the_writing_instance(self, rig):
        path = rig["prov"].path
        for i, pattern in enumerate(([False, True], [True, False], [False, True])):
            _run_one_step(rig["env"], rig["decoder"], rig["gk"], rig["prov"], np.array(pattern), note=f"step{i}")
        with open(path) as f:
            lines = f.readlines()
        with open(path, "w") as f:
            f.writelines(lines[:2])
        assert rig["prov"].verify_chain() is False


class TestMultiStepRollout:
    def test_rollout_replays_identically_and_chains(self, rig):
        positions = []
        for pattern in ([False, True], [True, False], [False, True]):
            _, _, obs = _run_one_step(rig["env"], rig["decoder"], rig["gk"], rig["prov"], np.array(pattern), note="roll")
            positions.append(int(obs[0]))
        assert positions == [3, 2, 3]
        assert rig["prov"].count == 3
        assert rig["prov"].verify_chain() is True

    def test_forged_allow_is_unconstructable(self, rig):
        """The old test patched a denied decision into ALLOW + stay; the type now refuses it."""
        with pytest.raises(ValueError, match="ALLOW must preserve"):
            PolicyDecision(status="ALLOW", proposed_action="move_left", enforced_action="stay", reason="test")
