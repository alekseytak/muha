"""Контур ЭТАП 4: спайки → глагол → ворота → 2D-коридор → модулятор → STDP.

Роль в архитектуре. `symmetric_toy` доказал механику на 5 нейронах с
симметричной проводкой; здесь тот же контур садится на реальный замороженный
канал FlyWire (входы/выходы из `io_channel` манифеста) и на среду
`HeadingCorridor`. Ничего «под ответ» не подкручено:

- восприятие одинаково во всех руках: борт, на котором цель, зажигает
  соответствующий пул входов; «прямо»/«позади» не зажигают ничего;
- действие — не назначено, а прочитано из спайков: какой пул выходов выиграл
  окно, тот глагол и совершён (`ActionDecoder`);
- глагол проходит `VerbAct` → `GateKeeper`, и в среду попадает ровно то, что
  ворота пропустили; отказ считается, а не замалчивается;
- обучение — R-STDP: внутри окна копится eligibility при M=0, модулятор
  приходит после действия из награды среды.

Две версии правила (P4.3 v1 и v2). В v1 модулятор — один скаляр на все
пластичные рёбра, поэтому обучение может только поднимать общую возбудимость
стимулированного пути. В v2 (рука `pool`) проигравший пул учится в том же
направлении, но медленнее: `M_loser = (1−κ)·M`, `M_winner = M`; остальные
рёбра — `M`. Знак не инвертируется: пилот с `−κM` показал, что при
преобладании отрицательных M инверсия разворачивает чтение (accuracy падает
ниже 0.5), а не обостряет конкуренцию. Выбор берётся из уже прочитанного
глагола, а не из стороны, на которой цель, поэтому подсказка (bearing) в
правило не попадает.

Абляция v3 (`pool_pos`, `pool_neg`) включает ту же поправку только на одном
знаке M: на подкрепляющих или на наказывающих шагах. Разница каждой руки с
v1 показывает, какой канал даёт выигрыш, а какой — вред.

P4.4 (`neuron`, `neuron_shuffled`) уходит от пула к нейрону: третья компонента
становится своей для каждого постсинаптического нейрона и строится по рангу
его доли в массе eligibility окна. Сумма кредита сохраняется (средний фактор
по активным нейронам ровно 1), поэтому правило не отнимает обучение, как
контраст по пулам, а перераспределяет его. `neuron_shuffled` — нулевая рука:
тот же набор факторов переставлен генератором по (seed, эпизод, шаг).

Действие = спайк. Глагол не выбирается декодером «из головы»: он выводится из
того, какой пул реально выстрелил, а `verb_for_action` называет совершённое.

Разделение обучения и симуляции. Веса, которые двигает STDP, нормированы в
[0, 1] (валюта правила, границы не нарушаются). Симуляция идёт на тех же
весах, умноженных на `gain` — физический масштаб проводимости; его изменение
не меняет ни одного пластичного значения и логируется отдельно.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...science.graph.models import ConnectomeGraph
from ...science.learning.plasticity_sparse import (
    EdgeSparseSTDP,
    NoPlasticityBaseline,
    STDPParams,
    select_plastic_edges,
)
from ...science.learning.reward_modulator import ModulatorParams, RewardModulator
from ...science.snn.lif_sparse import LIFParams, SparseLIF
from ..action.action_decoder import ActionDecoder, ActionProposal
from ..environments.heading_corridor import HeadingConfig, HeadingCorridor
from ..governance.gatekeeper import GateKeeper, VerbAct
from ..logging.provenance_log import ProvenanceLog

VERB_BY_ACTION = {
    "move_left": "поворачивает_налево",
    "move_right": "поворачивает_направо",
    "stay": "идёт_прямо",
}

# Верхняя граница: политика по сенсорике, без сети. Нужна, чтобы нулевой
# результат обучения было чем сравнить: среда и ворота обязаны уметь 100%.
ORACLE_ACTION = {"left": "move_left", "right": "move_right",
                 "ahead": "stay", "behind": "move_left"}

ARMS = ("plastic", "pool", "pool_pos", "pool_neg",
        "neuron", "neuron_shuffled",
        "neuron_renorm", "neuron_renorm_shuffled",
        "no_plasticity", "no_modulator", "oracle")

# Абляция v3: одна и та же поправка проигравшего, разведённая по знаку кредита.
# `always` — правило v2 целиком; `positive` — только на подкрепляющих шагах
# (проигравший наказывается наравне, как в v1); `negative` — только на
# наказывающих. Разница между `pool_pos` и v1 показывает, что даёт выигрыш
# победителя, разница `pool_neg` и v1 — откуда берётся вред.
POOL_ARMS = {"pool": "always", "pool_pos": "positive", "pool_neg": "negative"}

# P4.4: кредит на постсинаптический нейрон вместо пула. `rank` — фактор по
# рангу доли нейрона в массе eligibility окна; `shuffled` — та же выборка
# факторов, переставленная между теми же нейронами (нулевая рука: проверяет,
# важна ли привязка к участию, а не сама неоднородность кредита).
# P4.5: `*_renorm` делят факторы на `Σ s_i·E_i / Σ E_i`, поэтому доставленный
# кредит шаг в шаг равен v1, а различие остаётся только в распределении.
NEURON_ARMS = {
    "neuron": "rank",
    "neuron_shuffled": "shuffled",
    "neuron_renorm": "rank_renorm",
    "neuron_renorm_shuffled": "shuffled_renorm",
}


@dataclass(frozen=True)
class Channel:
    """Замороженный канал манифеста, разрезанный по бортам."""
    inputs_left: tuple[int, ...]
    inputs_right: tuple[int, ...]
    outputs_left: tuple[int, ...]
    outputs_right: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.inputs_left or not self.inputs_right:
            raise ValueError("нужны входы с обеих сторон")
        if not self.outputs_left or not self.outputs_right:
            raise ValueError("нужны выходы с обеих сторон")
        sides = [set(self.inputs_left), set(self.inputs_right)]
        if sides[0] & sides[1]:
            raise ValueError("вход не может быть сразу левым и правым")


@dataclass(frozen=True)
class LoopSettings:
    """Рабочая точка контура. Меняется только до подтверждающего прогона."""
    episodes: int = 500
    window_steps: int = 20
    i_drive: float = 20.0
    gain: float = 6.0
    sigma: float = 3.0
    eta: float = 0.02
    tau_eligibility: float = 80.0
    coherence_gain: float = 0.0   # ε: 0.0 → член тождественно выключен
    policy_penalty: float = 1.0   # δ: отказ ворот — отдельный сигнал
    m_max: float = 10.0
    pool_contrast: float = 0.0     # κ: проигравший пул ×(1−κ); 0 → правило v1
    neuron_credit: float = 0.0     # λ: размах пер-нейронного фактора; 0 → v1
    max_episode_steps: int = 40
    width: int = 7
    height: int = 7
    start: tuple[int, int] = (1, 1)
    target: tuple[int, int] = (5, 5)
    heading: int = 0
    shaping_scale: float = 1.0
    seed: int = 7

    def __post_init__(self) -> None:
        for name in ("episodes", "window_steps", "max_episode_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} обязан быть положительным")
        if self.gain <= 0:
            raise ValueError("gain обязан быть положительным")
        if self.eta <= 0:
            raise ValueError("eta обязана быть положительной")
        if not 0.0 <= self.pool_contrast <= 1.0:
            raise ValueError(
                "pool_contrast обязан лежать в [0, 1]: при κ=1 проигравший "
                "пул не учится вовсе, κ=0 возвращает правило v1")
        if not 0.0 <= self.neuron_credit <= 1.0:
            raise ValueError(
                "neuron_credit обязан лежать в [0, 1]: при λ=1 наименее "
                "участвующий нейрон не учится вовсе, λ=0 возвращает v1")


@dataclass
class EpisodeOutcome:
    episode: int
    reached: bool
    steps: int
    reward: float
    spikes: int
    moved_left: int
    moved_right: int
    stayed: int
    bumps: int
    denied: int
    mean_abs_dw: float
    fraction_at_bound: float
    weights_delta: float = 0.0
    graded_decisions: int = 0     # только борт left/right: там ответ определён
    graded_correct: int = 0


@dataclass
class RunSummary:
    arm: str
    seed: int
    episodes: list[EpisodeOutcome]
    governance_violations: int = 0
    nan_seen: bool = False

    @property
    def success_rate(self) -> float:
        return float(np.mean([e.reached for e in self.episodes])) if self.episodes else 0.0

    def success_rate_of(self, first: int, last: int) -> float:
        block = self.episodes[first:last]
        return float(np.mean([e.reached for e in block])) if block else 0.0

    def decision_accuracy_of(self, first: int, last: int) -> float:
        """Доля верных глаголов там, где верный глагол определён (борт).

        Это не критерий ЭТАПА 4, а измеряемая причина: успех растёт тогда и
        только тогда, когда растёт точность чтения борта. Считать её по
        эпизоду целиком нельзя — провал может быть геометрическим.
        """
        block = self.episodes[first:last]
        total = sum(e.graded_decisions for e in block)
        if total == 0:
            return float("nan")
        return sum(e.graded_correct for e in block) / total

    @property
    def total_spikes(self) -> int:
        return int(sum(e.spikes for e in self.episodes))

    @property
    def spikes_per_episode(self) -> float:
        return self.total_spikes / len(self.episodes) if self.episodes else 0.0

    @property
    def mean_steps(self) -> float:
        return float(np.mean([e.steps for e in self.episodes])) if self.episodes else 0.0


def verb_coherence(previous_action: str | None, action_type: str) -> float:
    """1.0 если глагол повторяется; направленно-нейтрально (left↔right равны)."""
    if previous_action is None:
        return 0.0
    return 1.0 if previous_action == action_type else 0.0


class CorridorAgent:
    """Один измеренный прогон: сеть + обучатель + среда, веса живут между эпизодами."""

    def __init__(
        self,
        *,
        graph: ConnectomeGraph,
        channel: Channel,
        settings: LoopSettings,
        arm: str,
        seed: int | None = None,
    ):
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm!r}")
        self.base = graph
        self.channel = channel
        self.settings = settings
        self.arm = arm
        self.seed = settings.seed if seed is None else seed
        self.plastic = select_plastic_edges(graph)
        if len(self.plastic) == 0:
            raise ValueError("в графе нет рёбер с доказанным знаком — обучать нечего")

        self._left_idx = np.asarray(channel.inputs_left, dtype=np.int64)
        self._right_idx = np.asarray(channel.inputs_right, dtype=np.int64)
        self._out_left = np.asarray(channel.outputs_left, dtype=np.int64)
        self._out_right = np.asarray(channel.outputs_right, dtype=np.int64)

        self.env_config = HeadingConfig(
            width=settings.width, height=settings.height, start=settings.start,
            heading=settings.heading, target=settings.target,
            max_steps=settings.max_episode_steps, shaping_scale=settings.shaping_scale,
        )
        self.decoder = ActionDecoder(n_neurons=2)
        self.gatekeeper = GateKeeper()
        self.prov_stats = {"decisions": 0, "denied": 0, "governance_violations": 0}
        self._last_action: str | None = None
        self.learner = self._make_learner()
        self.pool_when = POOL_ARMS.get(arm, "always")
        self.neuron_credit_mode = NEURON_ARMS.get(arm, "none")
        post = self.learner.edge_index[:, 1]
        self._pool_left = np.isin(post, self._out_left)
        self._pool_right = np.isin(post, self._out_right)
        if arm in POOL_ARMS and not (self._pool_left.any() and self._pool_right.any()):
            raise ValueError("контраст по пулам требует пластичных рёбер в обоих пулах")
        self.modulator = self._make_modulator()

    # --- сборка -----------------------------------------------------------
    def _make_learner(self):
        params = STDPParams(
            tau_plus=20.0, tau_minus=20.0,
            tau_eligibility=self.settings.tau_eligibility,
            A_plus=0.1, A_minus=0.1, eta=self.settings.eta,
            w_exc_max=1.0, w_inh_max=1.0, M_max=self.settings.m_max,
        )
        if self.arm == "no_plasticity":
            return NoPlasticityBaseline(self.base, self.plastic, params=params, dt_ms=1.0)
        return EdgeSparseSTDP(self.base, self.plastic, params=params, dt_ms=1.0)

    def _make_modulator(self) -> RewardModulator:
        if self.arm == "no_modulator":
            return RewardModulator(profile_id="no_modulator")
        params = ModulatorParams(
            alpha=1.0, beta=0.0, gamma=0.0, delta=self.settings.policy_penalty,
            epsilon=self.settings.coherence_gain, zeta=0.0,
            M_max=self.settings.m_max,
            profile_id="p4_loop_v2_pool" if self.arm == "pool" else "p4_loop_v1",
        )
        return RewardModulator(params=params)

    def _sim_graph(self) -> ConnectomeGraph:
        """Веса обучения (нормированные) × gain — только для симуляции."""
        w = np.asarray(self.base.weights, dtype=np.float64).copy()
        w[self.plastic] = self.learner.get_weights()
        return self.base.copy_with_edges(self.base.edges, weights=w * self.settings.gain)

    def _stimulus(self, bearing: str) -> np.ndarray:
        """Восприятие: зажечь борт, на котором цель. Ничего — если цели нет сбоку."""
        I = np.zeros(self.base.n_nodes, dtype=np.float64)
        if bearing == "left":
            I[self._left_idx] = self.settings.i_drive
        elif bearing == "right":
            I[self._right_idx] = self.settings.i_drive
        return I

    # --- решение ----------------------------------------------------------
    def _propose(self, bearing: str, left_spikes: int, right_spikes: int,
                 step: int, episode: int) -> ActionProposal:
        if self.arm == "oracle":
            action = ORACLE_ACTION[bearing]
            return ActionProposal(
                action_id=f"{action}:oracle", action_type=action, score=1.0,
                confidence=1.0, source_window_steps=self.settings.window_steps,
            )
        flags = np.array([left_spikes > right_spikes, right_spikes > left_spikes])
        return self.decoder.decode(flags, window_steps=self.settings.window_steps)

    def _edge_modulator(self, modulator: float, action: str) -> np.ndarray | None:
        """Контраст по пулам: проигравший учится в том же направлении, но медленнее.

        `M_winner = M`, `M_loser = (1−κ)·M`, прочие рёбра `M`. Знак кредита
        сохраняется: инверсия знака («−κM») на пилоте развернула чтение.
        Абляция v3 включает ту же поправку только на своём знаке M:
        `pool_pos` — при `M > 0`, `pool_neg` — при `M < 0`; на другом знаке
        возвращается скаляр v1. При `M = 0` поправка тождественна скаляру,
        поэтому тоже возвращается None.

        Кто победил, берётся из уже прочитанного глагола (`action`), а не из
        `bearing`: иначе правило получило бы ответ, который оракул берёт
        напрямую, и сравнение с оракулом потеряло бы смысл. При равенстве
        счётов глагол = `stay`, победителя нет — остаётся скаляр v1.
        """
        if self.arm not in POOL_ARMS or self.settings.pool_contrast <= 0.0:
            return None
        if action == "move_left":
            loser = self._pool_right
        elif action == "move_right":
            loser = self._pool_left
        else:
            return None
        if self.pool_when == "positive" and modulator <= 0.0:
            return None
        if self.pool_when == "negative" and modulator >= 0.0:
            return None
        M = np.full(self.learner.n_plastic, modulator, dtype=np.float64)
        M[loser] *= 1.0 - self.settings.pool_contrast
        return M

    def _neuron_modulator(self, modulator: float, episode: int,
                          step: int) -> np.ndarray | None:
        """Кредит на постсинаптический нейрон вместо пула (P4.4).

        Участие нейрона — его масса eligibility в только что закрытом окне:
        `E_i = сумма |elig| по пластичным рёбрам в i`. Фактор строится по
        рангу, а не по сырой массе: масса тяжелохвостая, а ранг ограничен,
        не требует подобранного порога и имеет среднее ровно 1 по активным
        нейронам:

            `s_i = 1 − λ + 2λ·rank_i/(m − 1)`,

        rank — по `(E_i, id)` по возрастанию, `m` — число нейронов с `E_i > 0`.
        Нейроны без eligibility в окне остаются с `s = 1`: их dw и так равен
        нулю, поэтому фактор там ничего не решает.

        В `neuron_shuffled` та же выборка факторов переставляется между теми же
        нейронами: набор величин и средний кредит совпадают с `neuron`, но
        привязка к участию разорвана. Разница рук и есть вклад привязки.
        Перестановка детерминирована по `(seed, эпизод, шаг)` и не зависит от
        масс.

        Среднее 1 по нейронам не означает равенства доставленного кредита: он
        взвешен массой, и `neuron` даёт `Σ s_i·E_i > Σ E_i` (в v4 — в 1.28–1.36
        раза). Руки `*_renorm` делят факторы на `rho = Σ s_i·E_i / Σ E_i`,
        поэтому `Σ |M|·|elig|` шаг в шаг равен v1, а различие остаётся только
        в распределении. Перенормировка идёт после перестановки, так что обе
        renorm-руки держат одну и ту же амплитуду.

        Возвращает None, когда поправка тождественна скаляру v1: чужая рука,
        λ = 0, `M = 0` или меньше двух активных нейронов.
        """
        if self.arm not in NEURON_ARMS or self.settings.neuron_credit <= 0.0:
            return None
        if modulator == 0.0:
            return None
        lam = self.settings.neuron_credit
        post = self.learner.edge_index[:, 1]
        mass = np.bincount(post, weights=np.abs(self.learner.eligibility),
                           minlength=self.base.n_nodes)
        active = np.flatnonzero(mass > 0.0)
        if active.size < 2:
            return None
        order = np.lexsort((active, mass[active]))
        rank = np.empty(active.size, dtype=np.float64)
        rank[order] = np.arange(active.size, dtype=np.float64)
        factors = np.ones(self.base.n_nodes, dtype=np.float64)
        factors[active] = 1.0 - lam + 2.0 * lam * rank / (active.size - 1)
        mode = NEURON_ARMS[self.arm]
        if mode.startswith("shuffled"):
            rng = np.random.default_rng([self.seed, episode, step])
            factors[active] = factors[active][rng.permutation(active.size)]
        if mode.endswith("_renorm"):
            mass_active = mass[active]
            rho = float(np.dot(factors[active], mass_active) / mass_active.sum())
            factors[active] /= rho
        return float(modulator) * factors[post]

    def _fraction_at_bound(self) -> float:
        w = np.asarray(self.learner.get_weights(), dtype=np.float64)
        if w.size == 0:
            return 0.0
        params = self.learner.params
        exc = getattr(self.learner, "edge_is_exc", None)
        inh = getattr(self.learner, "edge_is_inh", None)
        if exc is None or inh is None:
            return 0.0
        at = np.zeros(w.shape, dtype=bool)
        at[exc] = np.isclose(w[exc], 0.0) | np.isclose(w[exc], params.w_exc_max)
        at[inh] = np.isclose(w[inh], 0.0) | np.isclose(w[inh], -params.w_inh_max)
        return float(at.mean())

    # --- эпизод -----------------------------------------------------------
    def run_episode(self, episode: int, prov: ProvenanceLog | None = None) -> EpisodeOutcome:
        s = self.settings
        env = HeadingCorridor(self.env_config)
        env.reset()

        net = None
        if self.arm != "oracle":
            net = SparseLIF(self._sim_graph(), LIFParams(sigma=s.sigma), dt_ms=1.0,
                            synapse_tau_ms=5.0, seed=self.seed * 1000 + episode)

        outcome = EpisodeOutcome(
            episode=episode, reached=False, steps=0, reward=0.0, spikes=0,
            moved_left=0, moved_right=0, stayed=0, bumps=0, denied=0,
            mean_abs_dw=0.0, fraction_at_bound=0.0,
        )
        dw_abs: list[float] = []
        self._last_action = None
        w_before = np.asarray(self.learner.get_weights(), dtype=np.float64).copy()

        for step in range(s.max_episode_steps):
            bearing = env.bearing()
            left_spikes = right_spikes = 0
            if net is not None:
                I_ext = self._stimulus(bearing)
                for _ in range(s.window_steps):
                    spikes = net.step(I_ext)
                    outcome.spikes += int(np.count_nonzero(spikes))
                    # Только время: eligibility копится, веса стоят.
                    self.learner.update(spikes, spikes, modulator=0.0)
                    left_spikes += int(np.count_nonzero(spikes[self._out_left]))
                    right_spikes += int(np.count_nonzero(spikes[self._out_right]))

            proposal = self._propose(bearing, left_spikes, right_spikes, step, episode)
            verb_act = VerbAct.from_proposal(
                proposal,
                event_id=f"{self.arm}-s{self.seed}-ep{episode}-st{step}",
                subject_id="flywire-corridor", role="Decider",
                verb="выбирает", domain="simulation",
                trace_id=f"p4-{self.arm}-{self.seed}",
            )
            decision = self.gatekeeper.evaluate(verb_act)
            self.prov_stats["decisions"] += 1
            denied = decision.status != "ALLOW"
            if denied:
                self.prov_stats["denied"] += 1
                outcome.denied += 1
            elif decision.enforced_action != proposal.action_type:
                self.prov_stats["governance_violations"] += 1

            _, reward, terminated, truncated, info = env.step(decision.enforced_action)
            outcome.reward += float(reward)
            outcome.steps = step + 1
            if bearing in ("left", "right"):
                outcome.graded_decisions += 1
                expected = "move_left" if bearing == "left" else "move_right"
                outcome.graded_correct += int(decision.enforced_action == expected)
            if info["bumped"]:
                outcome.bumps += 1
            action = decision.enforced_action
            if action == "move_left":
                outcome.moved_left += 1
            elif action == "move_right":
                outcome.moved_right += 1
            else:
                outcome.stayed += 1

            coherence = verb_coherence(self._last_action, action)
            self._last_action = action
            M = self.modulator.compute(
                task_reward=float(reward),
                policy_violation=1.0 if denied else 0.0,
                coherence=coherence,
            )
            M_edge = self._edge_modulator(M, proposal.action_type)
            if M_edge is None:
                M_edge = self._neuron_modulator(M, episode, step)
            dw = self.learner.apply_modulator(M if M_edge is None else M_edge)
            dw_abs.append(float(np.mean(np.abs(dw))) if dw.size else 0.0)

            if terminated or truncated:
                break

        outcome.reached = bool(env.terminated and (env.x, env.y) == self.env_config.target)
        outcome.mean_abs_dw = float(np.mean(dw_abs)) if dw_abs else 0.0
        outcome.fraction_at_bound = self._fraction_at_bound()
        outcome.weights_delta = float(np.mean(np.abs(
            np.asarray(self.learner.get_weights(), dtype=np.float64) - w_before)))

        if prov is not None:
            prov.append({
                "kind": "p4_loop_episode",
                "arm": self.arm,
                "seed": self.seed,
                "episode": episode,
                "reached": outcome.reached,
                "steps": outcome.steps,
                "reward": round(outcome.reward, 4),
                "spikes": outcome.spikes,
                "denied": outcome.denied,
                "bumps": outcome.bumps,
                "mean_abs_dw": round(outcome.mean_abs_dw, 8),
                "fraction_at_bound": round(outcome.fraction_at_bound, 4),
                "graded_decisions": outcome.graded_decisions,
                "graded_correct": outcome.graded_correct,
                "verbs": {VERB_BY_ACTION[a]: n for a, n in (
                    ("move_left", outcome.moved_left),
                    ("move_right", outcome.moved_right),
                    ("stay", outcome.stayed)) if n},
            })
        return outcome

    def run(self, *, episodes: int | None = None,
            prov: ProvenanceLog | None = None) -> RunSummary:
        n = self.settings.episodes if episodes is None else episodes
        outcomes = [self.run_episode(ep, prov=prov) for ep in range(n)]
        nan_seen = any(not np.isfinite(e.reward) or not np.isfinite(e.mean_abs_dw)
                       for e in outcomes)
        return RunSummary(arm=self.arm, seed=self.seed, episodes=outcomes,
                          governance_violations=self.prov_stats["governance_violations"],
                          nan_seen=nan_seen)
