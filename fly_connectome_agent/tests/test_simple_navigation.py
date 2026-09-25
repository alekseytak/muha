"""Tests for SimpleNavigation environment."""
from __future__ import annotations
import numpy as np
import pytest
from fly_connectome_agent.src.engineering.environments.simple_navigation import SimpleNavigation, EnvConfig


class TestResetDeterministic:
    def test_reset_same_seed_same_position(self):
        env1 = SimpleNavigation(); env2 = SimpleNavigation()
        obs1, _ = env1.reset(seed=42); obs2, _ = env2.reset(seed=42)
        assert obs1[0] == obs2[0] == 0

    def test_reset_diff_seed_starts_same(self):
        env1 = SimpleNavigation(); env2 = SimpleNavigation()
        obs1, _ = env1.reset(seed=1); obs2, _ = env2.reset(seed=2)
        assert obs1[0] == 0; assert obs2[0] == 0


class TestBoundaries:
    def test_boundary_left_clamped(self):
        env = SimpleNavigation(EnvConfig(length=11, init_position=2, target_position=10))
        env.reset(seed=0); env.step("move_left"); env.step("move_left")
        obs, _, _, _, info = env.step("move_left")
        assert obs[0] == 0; assert info["position"] == 0

    def test_boundary_right_clamped(self):
        env = SimpleNavigation(EnvConfig(length=5, init_position=3, target_position=4))
        env.reset(seed=0)
        obs, _, terminated, _, _ = env.step("move_right")
        assert obs[0] == 4; assert terminated is True


class TestTargetTermination:
    def test_target_reached_terminates(self):
        env = SimpleNavigation(EnvConfig(length=3, init_position=0, target_position=2))
        env.reset(seed=0); env.step("move_right")
        _, reward, terminated, truncated, _ = env.step("move_right")
        assert terminated is True; assert reward > 0; assert truncated is False


class TestMaxSteps:
    def test_max_steps_truncates(self):
        env = SimpleNavigation(EnvConfig(length=50, init_position=10, target_position=49, max_steps=5))
        env.reset(seed=0); terminated = truncated = False
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step("stay")
            if terminated or truncated: break
        assert truncated is True; assert terminated is False


class TestInvalidAction:
    def test_invalid_action_raises(self):
        env = SimpleNavigation(); env.reset(seed=0)
        with pytest.raises(ValueError, match="Invalid action"): env.step("jump")


class TestStepCost:
    def test_step_cost_applied(self):
        env = SimpleNavigation(EnvConfig(length=50, init_position=10, target_position=49))
        env.reset(seed=0); _, reward, _, _, _ = env.step("stay")
        assert reward == pytest.approx(-0.1)
