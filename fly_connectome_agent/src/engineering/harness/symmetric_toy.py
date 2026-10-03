"""Symmetric 5-neuron toy network + governed episode loop (P4 / P4.1).

Why this exists: the P3 demo used a hand-made reflex (strong direct synapse,
shorter direct delay), so "the agent reached the target" said more about how it
was wired than about what it learned. This harness removes every built-in
direction advantage in the `symmetric` regime:

- all four sensory→command synapses start at the same weight;
- all delays are equal, so the direct path has no timing head start;
- the left/right tasks are exact mirrors of each other;
- membrane noise (sigma > 0) is the only symmetry breaker, and it is seeded.

Direction can therefore only come from learning — which is what P4.1 measures.
The other init regimes (`anti`, `right_bias`, `shuffle`) exist to prove the
harness *can* tell a learned direction from a rigged one, and `oracle` is the
opposite extreme: a hand-specified reflex that already knows both answers (P4.2).

The loop is the real P3.1 contour: spikes → ActionDecoder → VerbAct.from_proposal
→ GateKeeper → environment. Whatever the GateKeeper enforces is what the
environment gets, and governance violations are counted, not assumed away.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...science.graph.models import ConnectomeGraph, Edge, Node
from ...science.learning.plasticity_sparse import (
    EdgeSparseSTDP,
    NoPlasticityBaseline,
    STDPParams,
)
from ...science.learning.reward_modulator import ModulatorParams, RewardModulator
from ...science.snn.lif_sparse import LIFParams, SparseLIF
from ..action.action_decoder import ActionDecoder
from ..environments.simple_navigation import EnvConfig, SimpleNavigation
from ..governance.gatekeeper import GateKeeper, VerbAct
from ..logging.provenance_log import ProvenanceLog

# Neuron indices: 0-1 direction sensors, 2-3 command neurons, 4 lateral inhibitor.
SENS_LEFT, SENS_RIGHT, CMD_LEFT, CMD_RIGHT, INH = 0, 1, 2, 3, 4
N_NEURONS = 5

# Third task stream: mirrored targets alternate inside ONE training run. The two
# single-side tasks cannot answer "does learning beat every fixed wiring", because
# a randomly asymmetric wiring is handed a direction for free and only has to be
# lucky about which one. In a mixed stream no fixed wiring can be lucky about
# both halves, so this is where the success criterion is actually testable.
MIXED = "mixed"


def resolve_task(task: str, episode: int) -> str:
    """Which mirror this episode plays.

    Parity is the whole rule: it is direction-neutral, identical for every arm
    and every seed, and with an even episode budget each mirror appears exactly
    half the time.
    """
    if task == MIXED:
        return "right" if episode % 2 == 0 else "left"
    if task not in ("right", "left"):
        raise ValueError(f"unknown task: {task!r}")
    return task


# Edge slots (fixed topology; only weights/labels change per regime). Slots 8/9
# are the reciprocal command→command competition edges; they are inert unless
# ToySettings.command_inh_weight is non-zero (see the P4 run-#1 diagnosis).
EDGE_LABELS = ("S_L→C_L", "S_L→C_R", "S_R→C_R", "S_R→C_L", "S_L→INH", "S_R→INH", "INH→C_L", "INH→C_R",
               "C_L→C_R", "C_R→C_L")
EDGE_ENDPOINTS = (
    (SENS_LEFT, CMD_LEFT), (SENS_LEFT, CMD_RIGHT), (SENS_RIGHT, CMD_RIGHT), (SENS_RIGHT, CMD_LEFT),
    (SENS_LEFT, INH), (SENS_RIGHT, INH), (INH, CMD_LEFT), (INH, CMD_RIGHT),
    (CMD_LEFT, CMD_RIGHT), (CMD_RIGHT, CMD_LEFT),
)
EDGE_NTX = ("glutamate",) * 4 + ("glutamate",) * 2 + ("gaba",) * 2 + ("gaba",) * 2
# Slot ids of the four sensor→command synapses; which of the slots are
# plastic is an explicit experimental parameter (ToySettings.plastic_slots).
SLOT_DIRECT_L, SLOT_CONTRA_L, SLOT_DIRECT_R, SLOT_CONTRA_R = 0, 1, 2, 3


# Mirror left<->right as a permutation of edge slots. Derived from the graph
# endpoints instead of written as a literal: renaming the sides of every edge
# and looking up where it landed keeps the permutation true when slots move.
# test_mirror_perm_is_the_expected_involution pins the derived value.
_MIRROR_RENAME = {SENS_LEFT: SENS_RIGHT, SENS_RIGHT: SENS_LEFT,
                  CMD_LEFT: CMD_RIGHT, CMD_RIGHT: CMD_LEFT, INH: INH}
MIRROR_PERM = [EDGE_ENDPOINTS.index((_MIRROR_RENAME[src], _MIRROR_RENAME[dst]))
               for src, dst in EDGE_ENDPOINTS]


def labels_for(slots) -> tuple[str, ...]:
    return tuple(EDGE_LABELS[i] for i in slots)


def edge_positions(slots) -> dict[str, int]:
    """Position of each direction-relevant synapse inside the dw vector."""
    slots = list(slots)
    missing = [s for s in (SLOT_DIRECT_L, SLOT_CONTRA_L, SLOT_DIRECT_R, SLOT_CONTRA_R) if s not in slots]
    if missing:
        raise ValueError(f"plastic slots {missing} are required for the P4.1 mirror test")
    return {
        "direct_L": slots.index(SLOT_DIRECT_L),
        "contra_L": slots.index(SLOT_CONTRA_L),
        "direct_R": slots.index(SLOT_DIRECT_R),
        "contra_R": slots.index(SLOT_CONTRA_R),
    }


DT_MS = 1.0
W_EXC_MAX = 90.0
W_INH_MAX = 90.0


def make_tasks(s: "ToySettings") -> dict[str, EnvConfig]:
    """Mirror-symmetric tasks: same track, same distance to target, opposite sides.

    Margins keep the walls a few cells away from the target, so a boundary
    penalty is not the only thing that can tell the agent it went the wrong way.
    """
    right_start = s.edge_margin
    left_start = s.track_length - 1 - s.edge_margin
    common = dict(length=s.track_length, max_steps=s.max_steps, step_cost=s.step_cost,
                  target_reward=s.target_reward, boundary_penalty=s.boundary_penalty)
    return {
        "right": EnvConfig(init_position=right_start,
                           target_position=right_start + s.cell_distance, **common),
        "left": EnvConfig(init_position=left_start,
                          target_position=left_start - s.cell_distance, **common),
    }


@dataclass(frozen=True)
class ToySettings:
    """Tunables, kept in one place so a calibration run is reproducible.

    Defaults are the P4 operating point after the run-#1 failure (see
    docs/p4_validation.md): long enough decision window that spike counts are not
    a coin flip, enough noise that behaviour can explore, and a learning rate
    where weights move within 20 episodes without all saturating at the bound.

    Three of these are mechanism fixes, each measured as a variant in
    var/p4_sweep.txt, and each applied identically to every arm and both tasks so
    none of them can inject a direction:
      step_cost=0            a per-step cost makes M negative on ~every step, so
                             R-STDP depresses everything the sensors touch;
      tau_eligibility=80     has to exceed window_steps=60 or the synapses that
                             produced an action forget it before its reward lands;
      command_inh_weight=-40 reciprocal command→command competition, without
                             which the two candidate actions are indistinguishable
                             to the learning rule (both receive identical drive).
    """
    window_steps: int = 60
    episodes: int = 20
    i_drive: float = 35.0
    sigma: float = 3.5
    eta: float = 0.8
    alpha: float = 1.0
    sym_weight: float = 45.0
    inh_weight: float = -45.0
    sensor_inh_weight: float = 40.0
    strong: float = 70.0
    weak: float = 12.0
    # Task geometry (mirrored by make_tasks).
    track_length: int = 17
    cell_distance: int = 6
    edge_margin: int = 2
    max_steps: int = 25
    # Reward shape: with step_cost < 0 the modulator is negative on almost every
    # step, so R-STDP applies global LTD far more often than the rare positive
    # credit for reaching the target. Set it to 0 to let reward track progress.
    step_cost: float = 0.0
    target_reward: float = 10.0
    boundary_penalty: float = -1.0
    # Eligibility decay. It has to be comparable to window_steps, otherwise the
    # synapses that produced a chosen action have already forgotten it by the
    # time the reward for that action arrives.
    tau_eligibility: float = 80.0
    # Reciprocal command→command inhibition (winner-take-all). 0.0 keeps the
    # edges inert. Without competition the two command neurons receive identical
    # drive from the one active sensor, so the learning rule has no way to tell
    # which action the network actually took.
    command_inh_weight: float = -40.0
    # Which edge slots carry plasticity. The lateral-inhibition synapses are
    # left fixed by default: calibration showed that when INH→C is plastic, the
    # dominant effect of the modulator is global disinhibition — both command
    # neurons saturate, their spike counts tie, the decoder answers `stay`, and
    # behaviour degrades even while the direction-specific Δw looks great.
    # Keep that variant explicitly (plastic_slots=(0,1,2,3,6,7)) to report it.
    plastic_slots: tuple = (0, 1, 2, 3)


@dataclass
class EpisodeResult:
    steps: int
    reward: float
    reached: bool
    moved_left: int
    moved_right: int
    stayed: int
    denied: int
    spikes_left: int
    spikes_right: int
    mean_abs_dw: float
    weights_after: np.ndarray = field(default_factory=lambda: np.zeros(0))
    task: str = ""  # the mirror actually played ('right'/'left'), even in a mixed stream


def build_graph(weights: np.ndarray, settings: ToySettings | None = None) -> ConnectomeGraph:
    """Toy connectome; equal delays everywhere (no timing head start)."""
    s = settings or ToySettings()
    nodes = tuple(
        Node(node_id=i, cell_type=ct, region="toy", x=0.0, y=0.0, z=0.0, neurotransmitter=ntx)
        for i, (ct, ntx) in enumerate(
            [
                ("sensory_left", "glutamate"),
                ("sensory_right", "glutamate"),
                ("command_left", "acetylcholine"),
                ("command_right", "acetylcholine"),
                ("lateral_inhibitor", "gaba"),
            ]
        )
    )
    edges = tuple(
        Edge(
            source=src, target=dst, synapse_count=abs(float(w)), weight=float(w),
            delay_ms=1.0, neurotransmitter=ntx,
            class_id="plastic" if slot in s.plastic_slots else "fixed",
        )
        for slot, ((src, dst), w, ntx) in enumerate(zip(EDGE_ENDPOINTS, weights, EDGE_NTX))
    )
    return ConnectomeGraph(nodes=nodes, edges=edges, weights=np.asarray(weights, dtype=np.float64))


def init_weights(regime: str, rng: np.random.Generator, settings: ToySettings | None = None) -> np.ndarray:
    """Initial weights per edge slot.

    symmetric      every sensor→command synapse equal; no direction is favoured.
    shuffle        same value multiset, randomly permuted onto slots: asymmetry
                   exists but is not aligned with left/right.
    anti           contralateral strong / direct weak: the reflex points AWAY
                   from the target, so success must come from learning.
    right_bias     S_R→C_R strong, everything else weak: the scripted-demo
                   pathology, used as a probe, not as a training arm.
    oracle         both direct paths strong, both contralateral weak: a hand-
                   specified reflex that already knows the answer in EITHER
                   mirror. This is the P4.2 upper-bound fixed controller, and it
                   is the symmetric version of `right_bias` — which is exactly
                   why it is written from the named slot constants rather than by
                   hand: `right_bias` is what an "obvious" both-ways reflex
                   quietly turns into if a slot label is mixed up.
    """
    s = settings or ToySettings()
    w = np.zeros(len(EDGE_ENDPOINTS), dtype=np.float64)
    w[4] = w[5] = s.sensor_inh_weight
    w[6] = w[7] = s.inh_weight
    w[8] = w[9] = s.command_inh_weight
    if regime == "symmetric":
        w[0] = w[1] = w[2] = w[3] = s.sym_weight
    elif regime == "shuffle":
        values = np.array([s.sym_weight] * 4) * rng.uniform(0.6, 1.4, size=4)
        values = rng.permutation(values)
        w[0], w[1], w[2], w[3] = values
    elif regime == "anti":
        # direct = S_L→C_L / S_R→C_R, contralateral = S_L→C_R / S_R→C_L.
        # Named by slot on purpose: an earlier version wrote w[0]/w[3] as the
        # direct pair and silently built a right-pointing reflex instead.
        w[SLOT_DIRECT_L] = w[SLOT_DIRECT_R] = s.weak
        w[SLOT_CONTRA_L] = w[SLOT_CONTRA_R] = s.strong
    elif regime == "right_bias":
        w[SLOT_DIRECT_L] = w[SLOT_CONTRA_L] = w[SLOT_CONTRA_R] = s.weak
        w[SLOT_DIRECT_R] = s.strong
    elif regime == "oracle":
        w[SLOT_DIRECT_L] = w[SLOT_DIRECT_R] = s.strong
        w[SLOT_CONTRA_L] = w[SLOT_CONTRA_R] = s.weak
    else:
        raise ValueError(f"unknown init regime: {regime!r}")
    return w


def build_params(settings: ToySettings) -> dict[int, LIFParams]:
    def mk(v_th: float) -> LIFParams:
        return LIFParams(E_L=-70.0, V_th=v_th, V_reset=-70.0, tau_m=10.0, t_refrac=2.0, sigma=settings.sigma)
    return {
        SENS_LEFT: mk(-50.0), SENS_RIGHT: mk(-50.0),
        CMD_LEFT: mk(-52.0), CMD_RIGHT: mk(-52.0),
        INH: mk(-50.0),
    }


def make_learner(kind: str, graph: ConnectomeGraph, settings: ToySettings):
    """'stdp' → EdgeSparseSTDP, 'none' → NoPlasticityBaseline (same interface)."""
    params = STDPParams(
        tau_plus=20.0, tau_minus=20.0, tau_eligibility=settings.tau_eligibility,
        A_plus=0.1, A_minus=0.1, eta=settings.eta,
        w_exc_max=W_EXC_MAX, w_inh_max=W_INH_MAX, M_max=10.0,
    )
    idx = np.array(settings.plastic_slots, dtype=np.int64)
    if kind == "stdp":
        return EdgeSparseSTDP(graph, idx, params=params, dt_ms=DT_MS)
    if kind == "none":
        return NoPlasticityBaseline(graph, idx, params=params, dt_ms=DT_MS)
    raise ValueError(f"unknown learner kind: {kind!r}")


def make_modulator(reward_coupled: bool, settings: ToySettings) -> RewardModulator:
    """reward_coupled=False uses the library's own no_modulator profile (alpha=0)."""
    if reward_coupled:
        return RewardModulator(params=ModulatorParams(alpha=settings.alpha, M_max=10.0, profile_id="p4_reward_v1"))
    return RewardModulator(profile_id="no_modulator")


def sensory_drive(position: int, target: int, settings: ToySettings) -> np.ndarray:
    """Ground the sensors in the environment: stimulate the sensor on the side
    the target lies. This is perception, not a direction reflex — it is applied
    identically to both tasks and to every arm, including the controls."""
    I = np.zeros(N_NEURONS, dtype=np.float64)
    if position < target:
        I[SENS_RIGHT] = settings.i_drive
    elif position > target:
        I[SENS_LEFT] = settings.i_drive
    return I


class ToyAgent:
    """One measured run: a learner + network that persists across episodes."""

    def __init__(
        self,
        *,
        learner_kind: str,
        init_regime: str,
        seed: int,
        reward_coupled: bool = True,
        settings: ToySettings | None = None,
    ):
        self.settings = settings or ToySettings()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.weights_init = init_weights(init_regime, self.rng, self.settings)
        self.graph = build_graph(self.weights_init, self.settings)
        self.learner = make_learner(learner_kind, self.graph, self.settings)
        self.modulator = make_modulator(reward_coupled, self.settings)
        self.decoder = ActionDecoder(n_neurons=2)
        self.gatekeeper = GateKeeper()
        self.net = SparseLIF(
            self.graph, params=build_params(self.settings), dt_ms=DT_MS,
            synapse_tau_ms=5.0, seed=seed,
        )
        self.prov_stats: dict[str, int] = {"decisions": 0, "denied": 0, "governance_violations": 0}
        self.tasks = make_tasks(self.settings)
        self.init_regime = init_regime
        self.learner_kind = learner_kind

    @property
    def plastic_weights(self) -> np.ndarray:
        return np.asarray(self.learner.edge_weights, dtype=np.float64)

    def _carry_weights_to(self, graph: ConnectomeGraph) -> ConnectomeGraph:
        """Rebuild the network on a graph that carries the learned weights."""
        weights = graph.weights.copy()
        weights[np.array(self.settings.plastic_slots, dtype=np.int64)] = self.plastic_weights
        return build_graph(weights, self.settings)

    def run_episode(self, task: str, episode: int, prov: ProvenanceLog | None = None, verbose: bool = False) -> EpisodeResult:
        s = self.settings
        real_task = resolve_task(task, episode)
        config = self.tasks[real_task]
        env = SimpleNavigation(config)
        obs, _ = env.reset(seed=self.seed * 1000 + episode)
        position = int(obs[0])
        # The network runs continuously through the episode: a decision window
        # is a slice of one trajectory, otherwise pre/post timing (and with it
        # eligibility) is thrown away between windows.
        self.graph = self._carry_weights_to(build_graph(self.weights_init, s))
        self.net = SparseLIF(
            self.graph, params=build_params(s), dt_ms=DT_MS, synapse_tau_ms=5.0,
            seed=self.seed * 100 + episode,
        )

        result = EpisodeResult(steps=0, reward=0.0, reached=False, moved_left=0, moved_right=0,
                               stayed=0, denied=0, spikes_left=0, spikes_right=0, mean_abs_dw=0.0)
        dw_abs: list[float] = []

        for step in range(config.max_steps):
            I_ext = sensory_drive(position, config.target_position, s)
            left_spikes = right_spikes = 0
            for _ in range(s.window_steps):
                spikes = self.net.step(I_ext)
                left_spikes += int(spikes[CMD_LEFT])
                right_spikes += int(spikes[CMD_RIGHT])
                # Timing only: eligibility accumulates, weights stay put.
                self.learner.update(spikes, spikes, modulator=0.0)
            result.spikes_left += left_spikes
            result.spikes_right += right_spikes

            flags = np.array([left_spikes > right_spikes, right_spikes > left_spikes])
            proposal = self.decoder.decode(flags)
            verb_act = VerbAct.from_proposal(
                proposal,
                event_id=f"{real_task}-s{self.seed}-ep{episode}-st{step}",
                subject_id="toy-pilot", role="Decider", verb="выбирает",
                domain="simulation", trace_id=f"p4-{real_task}-{self.seed}",
            )
            decision = self.gatekeeper.evaluate(verb_act)
            self.prov_stats["decisions"] += 1
            if decision.status != "ALLOW":
                self.prov_stats["denied"] += 1
                result.denied += 1
            if decision.status == "ALLOW" and decision.enforced_action != proposal.action_type:
                self.prov_stats["governance_violations"] += 1

            _, reward, terminated, truncated, info = env.step(decision.enforced_action)
            position = int(info["position"])
            result.steps = step + 1
            result.reward += reward
            if decision.enforced_action == "move_left":
                result.moved_left += 1
            elif decision.enforced_action == "move_right":
                result.moved_right += 1
            else:
                result.stayed += 1

            # Reward delivery is its own call: learning time is not tied to
            # whether spikes happened to be present at that moment.
            M = self.modulator.compute(task_reward=reward)
            dw = self.learner.apply_modulator(M)
            dw_abs.append(float(np.mean(np.abs(dw))))
            result.mean_abs_dw = float(np.mean(dw_abs)) if dw_abs else 0.0

            if verbose:
                print(
                    f"    {task}/ep{episode} step {step:02d} → pos {position:2d} | "
                    f"spikes L/R {left_spikes:2d}/{right_spikes:2d} | {proposal.action_type:10s} | "
                    f"{decision.status} | r={reward:+.2f} M={M:+.2f}"
                )
            if terminated or truncated:
                break

        result.reached = env.terminated and position == config.target_position
        result.task = real_task
        result.weights_after = self.plastic_weights.copy()

        if prov is not None:
            prov.append({
                "kind": "p4_episode",
                "learner": self.learner_kind,
                "init_regime": self.init_regime,
                "task": real_task,
                "task_stream": task,
                "seed": self.seed,
                "episode": episode,
                "steps": result.steps,
                "reached": result.reached,
                "reward": round(result.reward, 4),
                "moved_left": result.moved_left,
                "moved_right": result.moved_right,
                "stayed": result.stayed,
                "denied": result.denied,
                "spikes_cmd_left": result.spikes_left,
                "spikes_cmd_right": result.spikes_right,
                "mean_abs_dw": round(result.mean_abs_dw, 6),
                "weights_after": [round(float(w), 6) for w in result.weights_after],
                "governance_violations": self.prov_stats["governance_violations"],
            })
        return result

    def dw_per_edge(self) -> np.ndarray:
        """Δw per plastic edge, aligned with labels_for(settings.plastic_slots)."""
        slots = np.array(self.settings.plastic_slots, dtype=np.int64)
        return self.plastic_weights - np.asarray(self.weights_init)[slots]

    def fraction_at_bound(self) -> float:
        w = self.plastic_weights
        labels = np.array(labels_for(self.settings.plastic_slots))
        exc = np.array([l.startswith("S_") for l in labels])
        at = np.zeros(len(w), dtype=bool)
        at[exc] = np.isclose(w[exc], 0.0) | np.isclose(w[exc], W_EXC_MAX)
        at[~exc] = np.isclose(w[~exc], 0.0) | np.isclose(w[~exc], -W_INH_MAX)
        return float(at.mean()) if len(at) else 0.0


@dataclass
class RunResult:
    arm: str
    task: str
    seed: int
    learner_kind: str
    init_regime: str
    reward_coupled: bool
    slots: tuple
    episodes: list[EpisodeResult]
    dw_per_edge: np.ndarray
    weights_init: np.ndarray
    weights_final: np.ndarray
    fraction_at_bound: float
    governance_violations: int

    @property
    def success_rate(self) -> float:
        return float(np.mean([e.reached for e in self.episodes])) if self.episodes else 0.0

    @property
    def steps_to_target(self) -> float:
        reached = [e.steps for e in self.episodes if e.reached]
        return float(np.mean(reached)) if reached else float("nan")

    @property
    def cumulative_reward(self) -> float:
        return float(sum(e.reward for e in self.episodes))

    @property
    def moved_left(self) -> int:
        return int(sum(e.moved_left for e in self.episodes))

    @property
    def stayed(self) -> int:
        return int(sum(e.stayed for e in self.episodes))

    @property
    def moved_right(self) -> int:
        return int(sum(e.moved_right for e in self.episodes))

    @property
    def choice_rate_right(self) -> float:
        total = self.moved_left + self.moved_right
        return self.moved_right / total if total else 0.0

    @property
    def mean_abs_dw(self) -> float:
        vals = [e.mean_abs_dw for e in self.episodes]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def spikes_per_neuron(self) -> tuple[int, int]:
        return int(sum(e.spikes_left for e in self.episodes)), int(sum(e.spikes_right for e in self.episodes))


def run_arm(arm: str, task: str, seed: int, *, episodes: int | None = None,
            settings: ToySettings | None = None, prov: ProvenanceLog | None = None,
            verbose: bool = False) -> RunResult:
    """Execute one (arm, task, seed) cell of the P4 matrix.

    `task` is 'right', 'left' or MIXED (the two mirrors alternating by parity).
    """
    spec = ARM_SPECS[arm]
    s = settings or ToySettings()
    agent = ToyAgent(
        learner_kind=spec["learner"],
        init_regime=spec["init"],
        seed=seed,
        reward_coupled=spec["reward_coupled"],
        settings=s,
    )
    n_episodes = s.episodes if episodes is None else episodes
    results = [agent.run_episode(task, ep, prov=prov, verbose=verbose) for ep in range(n_episodes)]
    return RunResult(
        arm=arm, task=task, seed=seed,
        learner_kind=spec["learner"], init_regime=spec["init"], reward_coupled=spec["reward_coupled"],
        slots=tuple(s.plastic_slots),
        episodes=results,
        dw_per_edge=agent.dw_per_edge(),
        weights_init=np.asarray(agent.weights_init)[np.array(s.plastic_slots, dtype=np.int64)],
        weights_final=agent.plastic_weights.copy(),
        fraction_at_bound=agent.fraction_at_bound(),
        governance_violations=agent.prov_stats["governance_violations"],
    )


# The arms of P4, split by what they can and cannot prove.
#
# LEARNING_ARMS all run R-STDP with reward; they differ in the wiring they start
# from, so they answer "can learning fix a bad or random start" — they are NOT
# controls for the mirror test, because a learning arm develops selectivity too.
#
# BASELINE_ARMS are frozen (Δw ≡ 0 by construction) and share their init regime
# with a learning arm, so each one isolates exactly one confound:
#   no_plasticity          the symmetric wiring alone, with reward flowing
#   m_zero                 timing without reward (M ≡ 0)
#   weight_shuffled_frozen  "is random asymmetry enough to score?"
#   direction_shuffled_frozen an anti-reflex that is never un-learned
LEARNING_ARMS = ("plastic", "weight_shuffled", "direction_shuffled")
BASELINE_ARMS = ("no_plasticity", "m_zero", "weight_shuffled_frozen", "direction_shuffled_frozen")

ARM_SPECS: dict[str, dict[str, object]] = {
    "plastic":                  {"learner": "stdp", "init": "symmetric", "reward_coupled": True},
    "weight_shuffled":          {"learner": "stdp", "init": "shuffle",   "reward_coupled": True},
    "direction_shuffled":       {"learner": "stdp", "init": "anti",      "reward_coupled": True},
    "no_plasticity":            {"learner": "none", "init": "symmetric", "reward_coupled": True},
    "m_zero":                   {"learner": "stdp", "init": "symmetric", "reward_coupled": False},
    "weight_shuffled_frozen":   {"learner": "none", "init": "shuffle",   "reward_coupled": True},
    "direction_shuffled_frozen": {"learner": "none", "init": "anti",     "reward_coupled": True},
    # P4.2 upper bound: fixed, symmetric, non-plastic, reward-blind. Not part of
    # ARMS on purpose — the P4.1 matrix is frozen, and adding an arm to it would
    # silently change what a documented P4.1 command reproduces.
    "oracle_reflex":            {"learner": "none", "init": "oracle",   "reward_coupled": False},
}
ARMS = LEARNING_ARMS + BASELINE_ARMS

# The full arm list of the P4.2 oracle comparison, in the order the protocol
# document reports it. `plastic` is the same R-STDP agent P4.1 scored.
P4_2_ARMS = ("plastic", "no_plasticity", "m_zero",
            "weight_shuffled_frozen", "direction_shuffled_frozen", "oracle_reflex")
ORACLE_ARM = "oracle_reflex"
