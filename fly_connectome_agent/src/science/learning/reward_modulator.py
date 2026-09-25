"""Reward modulator for P2.

Computes modulator M from task reward and penalties:
    M = alpha * R_task - beta * P_safety - gamma * E_energy
        - delta * V_policy + epsilon * C_coherence + zeta * A_advantage

M is clipped to [-M_max, M_max] and non-finite inputs are rejected.
Supports versioned parameter profiles via profile_id.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModulatorParams:
    """Modulator coefficients with version profile.

    M = alpha * R_task - beta * P_safety - gamma * E_energy
        - delta * V_policy + epsilon * C_coherence + zeta * A_advantage

    Attributes:
        alpha: coefficient for task reward
        beta: coefficient for safety penalty
        gamma: coefficient for energy penalty
        delta: coefficient for policy violation
        epsilon: coefficient for coherence
        zeta: coefficient for advantage
        M_max: maximum magnitude of modulator M (applied as clip)
        profile_id: versioned parameter profile identifier
    """
    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0
    delta: float = 0.0
    epsilon: float = 0.0
    zeta: float = 0.0
    M_max: float = 10.0
    profile_id: str = "default_v1"


_VERSIONED_PROFILES: dict[str, ModulatorParams] = {
    "default_v1": ModulatorParams(
        alpha=1.0, beta=0.0, gamma=0.0, delta=0.0,
        epsilon=0.0, zeta=0.0, M_max=10.0, profile_id="default_v1",
    ),
    "no_modulator": ModulatorParams(
        alpha=0.0, beta=0.0, gamma=0.0, delta=0.0,
        epsilon=0.0, zeta=0.0, M_max=10.0, profile_id="no_modulator",
    ),
    "constant_positive_control": ModulatorParams(
        alpha=1.0, beta=0.0, gamma=0.0, delta=0.0,
        epsilon=0.0, zeta=0.0, M_max=10.0, profile_id="constant_positive_control",
    ),
}


@dataclass
class ModulatorDiagnostics:
    raw_M: float
    clipped_M: float


@dataclass
class ModulatorState:
    current_modulator: float = 0.0
    profile_id: str = "default_v1"


def _is_finite(x: float) -> bool:
    if x != x:
        return False
    if x == float("inf") or x == float("-inf"):
        return False
    return True


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


class RewardModulator:
    """Computes modulator M from task reward and penalties.

    For profile_id="constant_positive_control", compute() always
    returns 1.0 (clipped to M_max) regardless of task reward.
    """

    def __init__(
        self,
        params: ModulatorParams | None = None,
        profile_id: str = "default_v1",
    ):
        if profile_id not in _VERSIONED_PROFILES and params is None:
            raise ValueError(
                f"profile_id '{profile_id}' not found in versioned profiles. "
                f"Available: {list(_VERSIONED_PROFILES.keys())}"
            )
        if params is not None:
            self.params = params
        else:
            self.params = _VERSIONED_PROFILES[profile_id]
        self.min_value = -self.params.M_max
        self.max_value = self.params.M_max

    def compute(
        self,
        task_reward: float = 0.0,
        safety_penalty: float = 0.0,
        energy_penalty: float = 0.0,
        policy_violation: float = 0.0,
        coherence: float = 0.0,
        advantage: float = 0.0,
    ) -> float:
        if self.params.profile_id == "constant_positive_control":
            return _clip(1.0, self.min_value, self.max_value)

        for name, val in [
            ("task_reward", task_reward),
            ("safety_penalty", safety_penalty),
            ("energy_penalty", energy_penalty),
            ("policy_violation", policy_violation),
            ("coherence", coherence),
            ("advantage", advantage),
        ]:
            if not _is_finite(val):
                raise ValueError(f"{name} is not finite: {val}")

        M = (
            self.params.alpha * task_reward
            - self.params.beta * safety_penalty
            - self.params.gamma * energy_penalty
            - self.params.delta * policy_violation
            + self.params.epsilon * coherence
            + self.params.zeta * advantage
        )

        if not _is_finite(M):
            raise ValueError(f"modulator M is not finite: {M}")

        M = _clip(M, self.min_value, self.max_value)
        return float(M)

    def compute_with_diagnostics(
        self,
        task_reward: float = 0.0,
        safety_penalty: float = 0.0,
        energy_penalty: float = 0.0,
        policy_violation: float = 0.0,
        coherence: float = 0.0,
        advantage: float = 0.0,
    ) -> tuple[float, ModulatorDiagnostics]:
        if self.params.profile_id == "constant_positive_control":
            raw_M = 1.0
            clipped_M = _clip(raw_M, self.min_value, self.max_value)
            diagnostics = ModulatorDiagnostics(raw_M=raw_M, clipped_M=clipped_M)
            return float(clipped_M), diagnostics

        for name, val in [
            ("task_reward", task_reward),
            ("safety_penalty", safety_penalty),
            ("energy_penalty", energy_penalty),
            ("policy_violation", policy_violation),
            ("coherence", coherence),
            ("advantage", advantage),
        ]:
            if not _is_finite(val):
                raise ValueError(f"{name} is not finite: {val}")

        raw_M = (
            self.params.alpha * task_reward
            - self.params.beta * safety_penalty
            - self.params.gamma * energy_penalty
            - self.params.delta * policy_violation
            + self.params.epsilon * coherence
            + self.params.zeta * advantage
        )

        if not _is_finite(raw_M):
            raise ValueError(f"modulator M is not finite: {raw_M}")

        clipped_M = _clip(raw_M, self.min_value, self.max_value)
        diagnostics = ModulatorDiagnostics(
            raw_M=float(raw_M),
            clipped_M=float(clipped_M),
        )
        return float(clipped_M), diagnostics

    def compute_batch(self, rewards: list[float]) -> float:
        if not rewards:
            return self.compute(task_reward=0.0)
        for r in rewards:
            if not _is_finite(r):
                raise ValueError(f"reward in batch is not finite: {r}")
        avg_reward = sum(rewards) / len(rewards)
        return self.compute(task_reward=avg_reward)

    def reset(self) -> None:
        pass