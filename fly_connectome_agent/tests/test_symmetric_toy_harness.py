"""P4 harness regression tests (symmetric toy network).

These do NOT re-run the 20-seed measurement — scripts/run_p4_validation.py does
that. What is locked in here is the property that makes that measurement
meaningful: in the `symmetric` regime no direction exists in the wiring, while a
rigged regime (`right_bias`, `anti`) is visible immediately without any
plasticity. If either half of that ever breaks, a P4.1 "learning" result stops
being evidence and we want the suite to say so.

Everything here runs at sigma=0 where the network is deterministic, except the
two tests that need motion (learning and the M≡0 control).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fly_connectome_agent.src.engineering.harness import symmetric_toy as toy
from fly_connectome_agent.src.engineering.harness.symmetric_toy import (
    CMD_LEFT,
    CMD_RIGHT,
    MIXED,
    SENS_LEFT,
    SENS_RIGHT,
    ToyAgent,
    ToySettings,
    build_graph,
    edge_positions,
    init_weights,
    labels_for,
    make_tasks,
    resolve_task,
    run_arm,
    sensory_drive,
)
from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog


def _s(**overrides) -> ToySettings:
    """Small, fast settings; only geometry/noise differ from the calibrated defaults."""
    base = dict(window_steps=10, max_steps=8, track_length=9, cell_distance=3,
                edge_margin=1, episodes=2)
    base.update(overrides)
    return ToySettings(**base)


def _run(agent: ToyAgent, task: str, episodes: int = 2):
    return [agent.run_episode(task, ep) for ep in range(episodes)]


class TestSymmetricRegimeHasNoBuiltInDirection:
    def test_zero_noise_symmetric_always_stays(self):
        # Both command neurons are fed equally by the one driven sensor, so a
        # deterministic symmetric network cannot prefer left or right.
        s = _s(sigma=0.0)
        for task in ("right", "left"):
            agent = ToyAgent(learner_kind="none", init_regime="symmetric", seed=0, settings=s)
            eps = _run(agent, task, episodes=s.episodes)
            moved_left = sum(e.moved_left for e in eps)
            moved_right = sum(e.moved_right for e in eps)
            assert moved_left == 0 and moved_right == 0, (task, moved_left, moved_right)
            assert sum(e.stayed for e in eps) > 0
            assert not any(e.reached for e in eps)

    def test_right_bias_regime_is_visible_without_learning(self):
        # The probe has to fire: a hand-wired right reflex must look like a
        # direction preference even with plasticity switched off.
        s = _s(sigma=0.0)
        agent = ToyAgent(learner_kind="none", init_regime="right_bias", seed=0, settings=s)
        toward = _run(agent, "right", episodes=s.episodes)
        away = _run(agent, "left", episodes=s.episodes)
        assert sum(e.moved_right for e in toward) > 0
        assert sum(e.moved_left for e in toward) == 0
        # On the left task the wired reflex has no support: it can only stay.
        assert sum(e.moved_left for e in away) == 0
        assert sum(e.stayed for e in away) == sum(e.steps for e in away)

    def test_anti_regime_moves_away_from_the_target(self):
        s = _s(sigma=0.0)
        agent = ToyAgent(learner_kind="none", init_regime="anti", seed=0, settings=s)
        eps = _run(agent, "right", episodes=s.episodes)
        assert sum(e.moved_left for e in eps) > 0
        assert sum(e.moved_right for e in eps) == 0
        assert not any(e.reached for e in eps)


class TestInitRegimesAreWhatTheySay:
    """Weight-level semantics of each regime, by slot name.

    The `anti` control once silently wired a right-pointing reflex (direct and
    contralateral slots were swapped); these assertions are what stop that from
    coming back as a "learning result".
    """

    def test_symmetric_favours_no_direction(self):
        w = init_weights("symmetric", np.random.default_rng(0), _s())
        pos = edge_positions(_s().plastic_slots)
        assert w[pos["direct_L"]] == w[pos["contra_L"]] == w[pos["direct_R"]] == w[pos["contra_R"]]

    def test_anti_is_strong_contralateral_weak_direct(self):
        s = _s()
        w = init_weights("anti", np.random.default_rng(0), s)
        pos = edge_positions(s.plastic_slots)
        assert w[pos["direct_L"]] == w[pos["direct_R"]] == s.weak
        assert w[pos["contra_L"]] == w[pos["contra_R"]] == s.strong

    def test_right_bias_is_exactly_one_strong_synapse(self):
        s = _s()
        w = init_weights("right_bias", np.random.default_rng(0), s)
        pos = edge_positions(s.plastic_slots)
        assert w[pos["direct_R"]] == s.strong
        assert [w[pos[k]] for k in ("direct_L", "contra_L", "contra_R")] == [s.weak] * 3

    def test_shuffle_stays_in_the_symmetric_band(self):
        s = _s()
        w = init_weights("shuffle", np.random.default_rng(7), s)
        pos = edge_positions(s.plastic_slots)
        vals = [w[pos[k]] for k in ("direct_L", "contra_L", "direct_R", "contra_R")]
        assert all(0.6 * s.sym_weight <= v <= 1.4 * s.sym_weight for v in vals)
        assert len({round(v, 6) for v in vals}) > 1

    def test_unknown_regime_is_rejected(self):
        with pytest.raises(ValueError, match="unknown init regime"):
            init_weights("whatever", np.random.default_rng(0), _s())


class TestTaskGeometryIsAMirror:
    def test_tasks_are_exact_mirrors(self):
        s = _s()
        tasks = make_tasks(s)
        right, left = tasks["right"], tasks["left"]
        assert right.length == left.length == s.track_length
        assert right.max_steps == left.max_steps
        assert right.target_position - right.init_position == s.cell_distance
        assert left.init_position - left.target_position == s.cell_distance
        assert right.init_position == left.length - 1 - left.init_position

    def test_target_is_not_a_wall(self):
        # If the target hugged the boundary, "wrong way" could be signalled by
        # the boundary penalty instead of by the sensors.
        s = _s()
        tasks = make_tasks(s)
        for cfg in tasks.values():
            assert 0 < cfg.target_position < cfg.length - 1


class TestNetworkContract:
    def test_delays_are_equal_so_no_path_gets_a_head_start(self):
        graph = build_graph(init_weights("symmetric", np.random.default_rng(0), _s()), _s())
        assert {e.delay_ms for e in graph.edges} == {1.0}
        assert len(graph.nodes) == toy.N_NEURONS
        assert len(graph.edges) == len(toy.EDGE_LABELS)

    def test_plastic_class_marks_exactly_the_configured_slots(self):
        s = _s(plastic_slots=(0, 2))
        graph = build_graph(init_weights("symmetric", np.random.default_rng(0), s), s)
        tagged = {i for i, e in enumerate(graph.edges) if e.class_id == "plastic"}
        assert tagged == {0, 2}


class TestSlotHelpers:
    def test_labels_and_positions_align(self):
        s = _s()
        labels = labels_for(s.plastic_slots)
        pos = edge_positions(s.plastic_slots)
        assert labels[pos["direct_L"]] == "S_L→C_L"
        assert labels[pos["contra_L"]] == "S_L→C_R"
        assert labels[pos["direct_R"]] == "S_R→C_R"
        assert labels[pos["contra_R"]] == "S_R→C_L"

    def test_positions_refuse_a_set_without_the_direct_paths(self):
        with pytest.raises(ValueError, match="required for the P4.1 mirror test"):
            edge_positions((0, 1))


class TestSensoryGrounding:
    def test_only_the_sensor_on_the_targets_side_is_driven(self):
        s = _s()
        right = sensory_drive(position=1, target=5, settings=s)
        left = sensory_drive(position=5, target=1, settings=s)
        arrived = sensory_drive(position=3, target=3, settings=s)
        assert right[SENS_RIGHT] == s.i_drive and right[SENS_LEFT] == 0.0
        assert left[SENS_LEFT] == s.i_drive and left[SENS_RIGHT] == 0.0
        assert np.all(arrived == 0.0)
        assert right[CMD_LEFT] == right[CMD_RIGHT] == 0.0


class TestMixedTaskStream:
    """The stream where «learning beats every fixed wiring» is actually testable.

    On one mirror task a randomly asymmetric frozen wiring only has to be lucky
    about its direction. With both mirrors alternating inside one run, no fixed
    wiring can be lucky twice, so the per-mirror split has to stay honest.
    """

    def test_resolve_task_alternates_by_parity(self):
        assert [resolve_task(MIXED, e) for e in range(6)] == [
            "right", "left", "right", "left", "right", "left",
        ]
        # Single-mirror tasks are passed through untouched.
        assert resolve_task("right", 3) == "right" and resolve_task("left", 3) == "left"

    def test_unknown_task_is_rejected(self):
        with pytest.raises(ValueError, match="unknown task"):
            resolve_task("diagonal", 0)

    def test_mixed_stream_plays_each_mirror_half_the_time(self):
        s = _s(sigma=0.0)
        agent = ToyAgent(learner_kind="none", init_regime="symmetric", seed=0, settings=s)
        eps = [agent.run_episode(MIXED, ep) for ep in range(8)]
        assert [e.task for e in eps] == ["right", "left"] * 4

    def test_run_arm_keeps_the_stream_label_but_records_each_mirror(self):
        r = run_arm("no_plasticity", MIXED, seed=0, episodes=4, settings=_s(sigma=0.0))
        assert r.task == MIXED
        assert [e.task for e in r.episodes] == ["right", "left", "right", "left"]

    def test_a_one_direction_wiring_fails_one_half_of_the_stream(self):
        # The property the mixed gate depends on: a hand-wired reflex can carry
        # its own mirror only, so its success splits into 1.0 / 0.0 halves.
        s = _s(sigma=0.0)
        agent = ToyAgent(learner_kind="none", init_regime="right_bias", seed=0, settings=s)
        eps = [agent.run_episode(MIXED, ep) for ep in range(4)]
        right_half = [e.reached for e in eps if e.task == "right"]
        left_half = [e.reached for e in eps if e.task == "left"]
        assert right_half and left_half
        assert all(right_half) and not any(left_half)

    def test_provenance_records_both_the_stream_and_the_mirror_played(self, tmp_path):
        prov = ProvenanceLog(path=str(tmp_path / "mixed_prov.jsonl"))
        s = _s(sigma=1.0)
        agent = ToyAgent(learner_kind="stdp", init_regime="symmetric", seed=5, settings=s)
        for ep in range(2):
            agent.run_episode(MIXED, ep, prov=prov)
        disk = [json.loads(line) for line in Path(prov.path).read_text().splitlines() if line.strip()]
        assert [d["payload"]["task"] for d in disk] == ["right", "left"]
        assert {d["payload"]["task_stream"] for d in disk} == {MIXED}


class TestControlsReallyDoNothing:
    def test_no_plasticity_arm_leaves_weights_untouched(self):
        r = run_arm("no_plasticity", "right", seed=0, episodes=2, settings=_s(sigma=3.0))
        assert np.allclose(r.dw_per_edge, 0.0)
        assert r.mean_abs_dw == 0.0
        assert r.weights_final == pytest.approx(list(r.weights_init))

    def test_m_zero_arm_sees_reward_but_never_learns_from_it(self):
        # Same learner, same spikes, M ≡ 0: this isolates the modulator, so a
        # plastic-vs-m_zero difference cannot be blamed on extra network activity.
        r = run_arm("m_zero", "right", seed=1, episodes=2, settings=_s(sigma=3.0))
        assert r.reward_coupled is False
        assert np.allclose(r.dw_per_edge, 0.0)
        assert r.mean_abs_dw == 0.0

    def test_plastic_arm_moves_at_least_one_weight(self):
        s = _s(sigma=3.0, eta=1.5, episodes=4)
        r = run_arm("plastic", "right", seed=2, episodes=4, settings=s)
        assert np.abs(r.dw_per_edge).sum() > 0.0
        assert r.learner_kind == "stdp" and r.init_regime == "symmetric"


class TestGovernanceInsideTheLoop:
    def test_every_decision_is_governed_and_recorded(self, tmp_path):
        prov = ProvenanceLog(path=str(tmp_path / "p4_prov.jsonl"))
        s = _s(sigma=2.0, episodes=2)
        agent = ToyAgent(learner_kind="stdp", init_regime="symmetric", seed=3, settings=s)
        eps = [agent.run_episode("right", ep, prov=prov) for ep in range(2)]

        steps = sum(e.steps for e in eps)
        assert steps > 0
        assert agent.prov_stats["decisions"] == steps
        assert agent.prov_stats["governance_violations"] == 0
        # One provenance record per episode, and the chain is intact on disk.
        assert prov.count == 2
        assert prov.verify_chain() is True
        # Read the bytes on disk, not the writer's cache: that is what a reviewer sees.
        disk = [json.loads(line) for line in Path(prov.path).read_text().splitlines() if line.strip()]
        assert len(disk) == 2
        last = disk[-1]["payload"]
        assert last["kind"] == "p4_episode"
        assert last["init_regime"] == "symmetric" and last["learner"] == "stdp"
        assert "weights_after" in last and len(last["weights_after"]) == len(s.plastic_slots)


class TestArmSpecs:
    def test_learning_and_baseline_arms_are_separated(self):
        assert toy.LEARNING_ARMS == ("plastic", "weight_shuffled", "direction_shuffled")
        assert toy.BASELINE_ARMS == (
            "no_plasticity", "m_zero", "weight_shuffled_frozen", "direction_shuffled_frozen",
        )
        assert toy.ARMS == toy.LEARNING_ARMS + toy.BASELINE_ARMS
        # ARM_SPECS не может молча обрасти лишним режимом: каждая spec обязана быть
        # задействована в объявленном наборе arms — своего (P4.1) или чужого (P4.2).
        assert set(toy.ARM_SPECS) == set(toy.ARMS) | set(toy.P4_2_ARMS)
        assert {toy.ARM_SPECS[a]["init"] for a in toy.ARMS} == {"symmetric", "shuffle", "anti"}
        # Oracle — единственный режим, добавленный P4.2, и он вне P4.1 воспроизведения.
        assert toy.ARM_SPECS[toy.ORACLE_ARM]["init"] == "oracle"
        assert toy.ORACLE_ARM not in toy.ARMS

    def test_every_baseline_is_frozen_so_its_dw_is_zero_by_construction(self):
        # The Δw control gate only means something if baseline arms cannot move.
        for arm in toy.BASELINE_ARMS:
            spec = toy.ARM_SPECS[arm]
            assert spec["learner"] == "none" or arm == "m_zero", arm
        assert toy.ARM_SPECS["m_zero"]["reward_coupled"] is False
        assert toy.ARM_SPECS["no_plasticity"]["learner"] == "none"

    def test_frozen_shuffled_control_shares_the_learning_arms_init_regime(self):
        # Same wiring, no plasticity: this is the arm that answers "is random
        # asymmetry alone enough to score?".
        assert toy.ARM_SPECS["weight_shuffled"]["init"] == \
            toy.ARM_SPECS["weight_shuffled_frozen"]["init"] == "shuffle"
        assert toy.ARM_SPECS["direction_shuffled"]["init"] == \
            toy.ARM_SPECS["direction_shuffled_frozen"]["init"] == "anti"

    def test_frozen_shuffled_control_never_moves_weights(self):
        r = run_arm("weight_shuffled_frozen", "right", seed=0, episodes=2, settings=_s(sigma=3.0))
        assert np.allclose(r.dw_per_edge, 0.0)
        assert r.init_regime == "shuffle" and r.learner_kind == "none"
