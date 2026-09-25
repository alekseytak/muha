"""Tests for ActionDecoder."""
from __future__ import annotations
import numpy as np
import pytest
from fly_connectome_agent.src.engineering.action.action_decoder import ActionDecoder, ActionProposal


@pytest.fixture
def decoder(): return ActionDecoder(n_neurons=2)


class TestDeterminism:
    def test_same_spikes_same_proposal(self, decoder):
        spikes = np.array([True, False]); p1 = decoder.decode(spikes); p2 = decoder.decode(spikes)
        assert p1 == p2
    def test_deterministic_proposal(self, decoder):
        spikes = np.array([False, True])
        p1 = decoder.decode(spikes, window_steps=5); p2 = decoder.decode(spikes, window_steps=5)
        assert p1.action_id == p2.action_id


class TestWinnerSelection:
    def test_left_wins(self, decoder):
        assert decoder.decode(np.array([1, 0])).action_type == "move_left"
    def test_right_wins(self, decoder):
        assert decoder.decode(np.array([0, 1])).action_type == "move_right"
    def test_left_wins_higher_activity(self, decoder):
        assert decoder.decode(np.array([True, False])).action_type == "move_left"
    def test_right_wins_higher_activity(self, decoder):
        assert decoder.decode(np.array([False, True])).action_type == "move_right"


class TestTieBehavior:
    def test_tie_goes_stay(self, decoder):
        assert decoder.decode(np.array([1, 1])).action_type == "stay"
    def test_tie_zero_goes_stay(self, decoder):
        assert decoder.decode(np.array([0, 0])).action_type == "stay"


class TestConfidence:
    def test_confidence_in_range(self, decoder):
        p = decoder.decode(np.array([1, 0])); assert 0.0 <= p.confidence <= 1.0
    def test_confidence_zero_on_tie(self, decoder):
        assert decoder.decode(np.array([1, 1])).confidence == 0.0
    def test_confidence_one_on_clear_winner(self, decoder):
        assert decoder.decode(np.array([1, 0])).confidence == 1.0


class TestInvalidInput:
    def test_invalid_shape_3d_raises(self, decoder):
        with pytest.raises(ValueError, match="shape"): decoder.decode(np.array([1, 0, 1]))
    def test_invalid_shape_2d_raises(self, decoder):
        with pytest.raises(ValueError, match="shape"): decoder.decode(np.array([[1, 0]]))
    def test_empty_raises(self, decoder):
        with pytest.raises(ValueError, match="shape"): decoder.decode(np.array([]))


class TestProposalFields:
    def test_proposal_frozen(self, decoder):
        p = decoder.decode(np.array([1, 0]))
        with pytest.raises(Exception): p.action_type = "move_right"
    def test_source_window_steps(self, decoder):
        assert decoder.decode(np.array([1, 0]), window_steps=10).source_window_steps == 10
    def test_score_recorded(self, decoder):
        assert decoder.decode(np.array([1, 0])).score == 1.0
