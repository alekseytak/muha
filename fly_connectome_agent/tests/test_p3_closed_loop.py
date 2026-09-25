"""P3 closed-loop integration test."""
from __future__ import annotations
import numpy as np, pytest
from fly_connectome_agent.src.engineering.action.action_decoder import ActionDecoder
from fly_connectome_agent.src.engineering.governance.gatekeeper import GateKeeper, VerbAct
from fly_connectome_agent.src.engineering.environments.simple_navigation import SimpleNavigation, EnvConfig
from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog


class TestAllowPath:
    def test_allow_receives_action(self, tmp_path):
        env = SimpleNavigation(EnvConfig(length=5, init_position=1, target_position=4, max_steps=10))
        decoder = ActionDecoder(n_neurons=2); gk = GateKeeper()
        prov = ProvenanceLog(path=str(tmp_path / "p.jsonl"))
        env.reset(seed=42)
        proposal = decoder.decode(np.array([True, False]))
        va = VerbAct(event_id="e1", subject_id="a1", role="Decider", verb="выбирает", object_ref=proposal.action_id, domain="simulation", context={}, trace_id="t1", proposal_ref=proposal.action_id, proposed_action=proposal.action_type)
        decision = gk.evaluate(va)
        assert decision.status == "ALLOW"
        obs, _, _, _, _ = env.step(decision.enforced_action)
        assert decision.enforced_action == "move_left"
        assert obs[0] == 0
        prov.append({"decision": decision.status, "action": decision.enforced_action, "proposal": proposal.action_id})
        assert prov.count == 1; assert prov.verify_chain() is True


class TestDenyPath:
    def test_deny_stay(self, tmp_path):
        env = SimpleNavigation(EnvConfig(length=5, init_position=0, target_position=2, max_steps=10))
        decoder = ActionDecoder(n_neurons=2); gk = GateKeeper()
        prov = ProvenanceLog(path=str(tmp_path / "p.jsonl"))
        env.reset(seed=42)
        proposal = decoder.decode(np.array([True, False]))
        va = VerbAct(event_id="e2", subject_id="a1", role="Decider", verb="якорит", object_ref=proposal.action_id, domain="simulation", context={}, trace_id="t2", proposal_ref=proposal.action_id, proposed_action=proposal.action_type)
        decision = gk.evaluate(va)
        assert decision.status == "DENY"; assert decision.enforced_action == "stay"
        obs, _, _, _, _ = env.step(decision.enforced_action)
        assert obs[0] == 0
        prov.append({"decision": decision.status, "action": decision.enforced_action, "proposal": proposal.action_id})
        assert prov.count == 1; assert prov.verify_chain() is True


class TestEscalatePath:
    def test_escalate_stay(self, tmp_path):
        env = SimpleNavigation(EnvConfig(length=5, init_position=0, target_position=2, max_steps=10))
        decoder = ActionDecoder(n_neurons=2); gk = GateKeeper()
        prov = ProvenanceLog(path=str(tmp_path / "p.jsonl"))
        env.reset(seed=42)
        proposal = decoder.decode(np.array([False, True]))
        va = VerbAct(event_id="e3", subject_id="a1", role="Decider", verb="выбирает", object_ref=proposal.action_id, domain="simulation", context={"risk_level": "high"}, trace_id="t3", proposal_ref=proposal.action_id, proposed_action=proposal.action_type)
        decision = gk.evaluate(va)
        assert decision.status == "ESCALATE"; assert decision.enforced_action == "stay"
        obs, _, _, _, _ = env.step(decision.enforced_action)
        assert obs[0] == 0
        prov.append({"decision": decision.status, "action": decision.enforced_action, "proposal": proposal.action_id})
        assert prov.count == 1; assert prov.verify_chain() is True


class TestDeterministicReplay:
    def test_replay_same_trajectory(self, tmp_path):
        config = EnvConfig(length=10, init_position=2, target_position=5, max_steps=100)
        decoder = ActionDecoder(n_neurons=2); gk = GateKeeper()
        def run(seed):
            env = SimpleNavigation(config); prov = ProvenanceLog(path=str(tmp_path / f"p_{seed}.jsonl"))
            env.reset(seed=seed); positions = []
            for spikes in [np.array([False, True]), np.array([True, False])]:
                proposal = decoder.decode(spikes)
                va = VerbAct(event_id=f"evt-{seed}-{len(positions)}", subject_id="a", role="Decider", verb="выбирает", object_ref=proposal.action_id, domain="simulation", context={}, trace_id="t", proposal_ref=proposal.action_id, proposed_action=proposal.action_type)
                decision = gk.evaluate(va)
                if decision.status != "ALLOW": decision = type(decision)(status="ALLOW", enforced_action="stay", reason="test")
                _, _, terminated, _, info = env.step(decision.enforced_action); positions.append(info["position"])
                prov.append({"step": len(positions), "pos": info["position"]})
                if terminated: break
            assert prov.verify_chain() is True
            return positions
        assert run(123) == run(123)
