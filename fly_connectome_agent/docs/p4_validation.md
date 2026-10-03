# P4 / P4.1 — symmetric toy validation of reward-modulated learning

Status: **P4.1 passes** (mirror test, 20 seeds), and the **"beats every baseline"
criterion now passes on the mixed stream** — 60 seeds × 80 episodes, significant
against all four baselines on the pre-declared gate, confirmed on 40 fresh seeds
(exact two-sided p = 1.16e-4 vs the lucky frozen wiring: 31 wins / 7 losses / 2
ties over 40 seed pairs). It still does **not** pass on single-task
`success_rate`, and the reason is a property of that metric, not of the rule — see
[Result](#result). Claim class: `simulation_result` only (see
[claim_policy.md](claim_policy.md)) — a 5-neuron toy with scripted sensory grounding
says nothing about *Drosophila*, and the ceiling result stands: a lucky fixed wiring
still solves both mirrors perfectly more often than 80 episodes of learning do.
The real connectome CSV is **not** connected; unblocking it is a user decision, not
a step taken here.

## Why this exists

The P3 demo reached the target with a hand-made reflex: a strong direct synapse
and a shorter direct delay. "The agent reached the target" therefore said more
about how it was wired than about what it learned. P4.1 is the test that
distinguishes the two, and it can only be run on a network that has no direction
in its iron to begin with.

## Protocol

Harness: `fly_connectome_agent/src/engineering/harness/symmetric_toy.py`.
Driver: `scripts/run_p4_validation.py`.

Matrix: **7 conditions × 2 mirrored tasks × 20 seeds × 20 episodes** (280 cells,
5 600 governed episodes), plus **mixed stream** runs where both mirrors alternate
inside one training run: 7 × 20 × 40 episodes at η=0.8 and η=1.5, 5 × 20 × 80
episodes at η=1.5, and a 5 arms × 60 seeds × 80 episode confirmatory run
(300 cells, 24 000 governed episodes).

The conditions split into two classes, and the split is what makes the `Δw` gates
meaningful:

| Class | Arm | Learner | Init | Reward coupled | Question it answers |
|---|---|---|---|---|---|
| learning | `plastic` | `EdgeSparseSTDP` | `symmetric` | yes | does R-STDP learn direction from unbiased wiring |
| learning | `weight_shuffled` | `EdgeSparseSTDP` | `shuffle` | yes | what does learning achieve from a random start |
| learning | `direction_shuffled` | `EdgeSparseSTDP` | `anti` | yes | can learning undo a wrong built-in reflex |
| baseline | `no_plasticity` | `NoPlasticityBaseline` | `symmetric` | yes | is the score just the frozen network |
| baseline | `m_zero` | `EdgeSparseSTDP` | `symmetric` | no (`no_modulator`, α=0) | is it the *reward* and not spike timing |
| baseline | `weight_shuffled_frozen` | `NoPlasticityBaseline` | `shuffle` | yes | is random asymmetry alone enough to score |
| baseline | `direction_shuffled_frozen` | `NoPlasticityBaseline` | `anti` | yes | an anti-reflex that never gets un-learned |

Only the frozen arms can serve as `Δw` controls: a *plastic* shuffled arm develops
selectivity too, so calling it a control for learning would measure the wrong
thing. Every baseline shares its init regime with a learning arm so the
comparison is wiring-for-wiring, and each one isolates exactly one confound.

Symmetry discipline (all of it, so that a direction difference can only come
from learning):

- all four sensory→command synapses start at one value (`sym_weight=45`);
- **every** edge has `delay_ms = 1.0`, so the direct path gets no timing head start;
- the two tasks are exact mirrors on one track (`distance=6`, `edge_margin=2`), so
  the target is never adjacent to a wall;
- sensors are grounded identically for every arm, including controls: the sensor
  on the side the target lies in gets `i_drive`;
- the only symmetry breaker is membrane noise (`sigma=3.5`), seeded per
  (arm, task, seed, episode).

Each step runs the real P3.1 contour — spikes → `ActionDecoder` →
`VerbAct.from_proposal` → `GateKeeper` → `env.step(decision.enforced_action)` —
and governance violations are counted, not assumed away. Timing and reward are
separate calls: `learner.update(spikes, spikes, modulator=0.0)` every sim step,
`learner.apply_modulator(M)` once per environment step.

Metrics per cell: `success_rate`, `steps_to_target`, `cumulative_reward`,
`choice_rate_right`, per-edge `Δw`, `mean_abs_dw`, `fraction_weights_at_bound`,
command-neuron spike counts, first-half vs second-half success.

### Pass criteria

1. **P4.1 mirror**: after training, `Δw(S_R→C_R) > Δw(S_L→C_L)` on task A and
   `Δw(S_L→C_L) > Δw(S_R→C_R)` on task B, winning on a majority of seeds **with
   the paired binomial sign test significant at p < 0.05**.
2. **Selectivity** (mechanism version of the same claim): within the *driven*
   sensor, `Δw(direct) − Δw(contra) > 0`, and no **frozen** arm shows either that
   selectivity or any weight motion at all.
3. **Baselines**: `plastic` beats *every frozen baseline* on `success_rate` per
   task, on a majority of shared seeds (paired sign test, p < 0.05) — not one lucky
   rollout. On the mixed stream the same gate runs on the *worse* half of the
   stream (`success_mirror_min`), which is the version that a lucky fixed wiring
   cannot satisfy.
4. Zero governance violations, intact provenance chain.

A majority of seeds without significance is a coin flip, so `gate_decision` in
`scripts/run_p4_validation.py` requires both and prints `НЕ ДОКАЗАНО` when they
disagree. `--replay var/<csv>` re-scores a saved matrix under changed gates
without re-simulating, so a stricter rule can be applied to the same numbers.

Criterion 1 is intentionally the plan's literal wording. It is weaker than it
looks in this topology: `S_L→C_L` is only eligible when the left sensor fires, so
on task A the comparison partly measures "which sensor was stimulated".
Criterion 2 is the version that cannot be satisfied by a reflex, which is why the
verdict requires both. The `plastic`-vs-`weight_shuffled` comparison (both learn)
is printed as a matter of record and is **not** a gate.

## Detector validation (is the test able to fail?)

A gate is worthless if nothing can trip it. `scripts/run_p4_validation.py --probe`
runs the wiring-only regimes at `sigma=0` with plasticity off:

| init regime | succ(right) | succ(left) | Δ |
|---|---|---|---|
| `symmetric` | 0.19 | 0.19 | +0.00 |
| `right_bias` (one strong `S_R→C_R`) | 1.00 | 0.00 | +1.00 |
| `anti` (strong contralateral) | 0.00 | 0.00 | +0.00 |

So: the symmetric wiring carries no direction, a hand-wired right reflex is
instantly visible, and the anti-reflex never reaches the target. Locked as
`tests/test_symmetric_toy_harness.py` (σ=0 determinism + per-regime weight
semantics).

## What the first complete run measured, and why it failed

The run below (`var/p4_run2.txt`, 20 seeds, operating point `step_cost=-0.1`,
`tau_eligibility=20`, no command competition) returned **НЕ ВЫПОЛНЕН** on both
gates:

- mirror: 11/20 seeds (right task), 10/20 (left task) — a coin flip;
- `plastic` success 0.21 / 0.18 vs `no_plasticity` 0.16 / 0.16;
- `weight_shuffled` (merely *random* asymmetric weights) scored **0.49 / 0.62** —
  better than learning.

That last line is the whole point of having the control: in that regime success
was bought by breaking the tie (so the decoder emits a move instead of `stay`),
not by learning which direction the target was in.

One more honesty note: an earlier pass (`var/p4_run1_stale_anti_bug.txt`) was
killed mid-flight because of a bug in `init_weights` — the `anti` regime wrote its
weak/strong pair into the wrong slots, so `direction_shuffled` was wired as a
*right-pointing* reflex instead of an anti-reflex. The σ=0 probe in the harness
tests found it (the anti network moved toward the wrong wall instead of away from
the target); slots are now addressed by name, and every number quoted here comes
from the corrected regime, where `anti` scores 0.00 on both tasks without
plasticity.

Three causes, each fixed as a variant, each measured in `var/p4_sweep.txt`
(4 seeds × 2 arms × 2 tasks):

| Variant | right succ (plastic / frozen) | left succ | mirror>0 (R / L) | selectivity (R / L) |
|---|---|---|---|---|
| V0 first-run point | 0.21 / 0.21 | 0.12 / 0.09 | 1/4, 2/4 | −0.02 / +0.70 |
| V1 `step_cost=0` | 0.21 / — | — | 1/4 | +0.10 |
| V2 `tau_eligibility=80` | 0.43 / 0.21 | 0.14 / 0.09 | 3/4, 0/4 | +8.67 / +1.34 |
| V3 competition `C↔C=-40` | 0.33 / 0.34 | 0.15 / 0.12 | 1/4, 2/4 | −0.10 / −0.00 |
| V6 all three | **0.45 / 0.34** | **0.19 / 0.12** | **4/4, 3/4** | **+4.94 / +3.09** |
| V7 V6 + `eta=1.5` | 0.62 / 0.34 | 0.20 / 0.12 | 4/4, 2/4 | +19.03 / +7.19 |

1. **Reward shape.** With `step_cost=-0.1` the modulator is negative on nearly
   every step, so R-STDP applies LTD to whatever the sensors touched, ~360 times
   per run, against a handful of `+10` arrivals. Learning was dominated by
   "pay for time", which has no direction in it.
2. **Eligibility vs decision window.** `tau_eligibility=20 ms` with a 60-step
   decision window means the synapses that produced an action have mostly
   forgotten it by the time its reward lands. `tau_eligibility` must be ≥ the
   window; 80 recovered the credit (V2 is the single biggest lever).
3. **No action selection to credit.** With equal weights, equal delays and one
   driven sensor feeding *both* command neurons identically, the two candidate
   actions are physically indistinguishable to the learning rule — there is
   nothing for a third factor to select. Reciprocal command→command inhibition
   (symmetric, equal delay, non-plastic) turns it into a competition, so the
   winning synapse really does have more pre/post coincidence.

All three fixes are direction-neutral by construction: they change the reward
constant, a time constant, and add two *identical* inhibitory edges.

## Second run: learning works, and a control was mis-specified

With the three fixes in place (`var/p4_run3.txt`, 20 seeds, same 5 arms), the
direction claim came out clean for the first time:

- mirror **19/20** seeds on task A (exact p = 4.005e-05) and **18/20** on task B
  (exact p = 4.025e-04); no ties in either direction test;
- within-sensor selectivity **20/20 on both tasks**, mean +6.78 / +7.54;
- behaviour: `plastic` success **0.42 / 0.42** (right/left) vs `no_plasticity` and
  `m_zero` **0.23 / 0.21**, cumulative reward 63.7 / 58.9 vs 12.8 / 6.1,
  paired sign test p=0.0001; still climbing (+0.22 first half → second half);
- `direction_shuffled` shows the rule *does* fight a wrong prior (contralateral
  synapses depressed by −31) but 20 episodes are not enough to reverse it: 0.00;
- the probe still reads: symmetric Δ=0.00, `right_bias` Δ=+1.00, anti 0.00.

Two gates nevertheless reported НЕ ВЫПОЛНЕН, and only one of them was a real
finding:

- **`weight_shuffled` beat `plastic`** (0.53 / 0.73). That is a genuine property of
  this regime, not a measurement error: an unbiased start spends its first
  episodes in ties (`stay`), while random asymmetry commits immediately, and half
  of those random commitments happen to match the task. It also invalidated the
  way the condition had been specified — `weight_shuffled` *learns*, so it is not a
  null control for `Δw`. The frozen `weight_shuffled_frozen` is.
- **the `Δw` control check flagged the plastic shuffled arms as "systemic
  selectivity"** — my gate was wrong, not the learning: any arm with plasticity
  develops direct-over-contra selectivity, which is precisely the mechanism being
  claimed. The gate now applies to frozen arms only.

## Result

Final matrix: 7 arms × 2 mirror tasks × 20 seeds × 20 episodes,
`var/p4_run4.txt`, rows in `var/p4_results.csv`, 5 600 governed episodes.

| arm | task | succ | succ trend | steps | reward | Δw direct | Δw contra | choice→ |
|---|---|---|---|---|---|---|---|---|
| `plastic` | right | **0.42** | +0.22 | 15.1 | 63.7 | **+5.88** | −0.89 | 0.57 |
| `plastic` | left | **0.42** | +0.20 | 15.4 | 58.9 | **+7.02** | −0.51 | 0.44 |
| `weight_shuffled` | right / left | 0.53 / 0.73 | +0.10 | 10.0 / 11.1 | 43.5 / 114.3 | +12.9 / +14.2 | −3.6 / −1.7 | 0.62 / 0.28 |
| `direction_shuffled` | right / left | 0.00 | 0.00 | nan | −388 / −398 | +0.7 | −31.3 | — |
| `no_plasticity` | right / left | 0.23 / 0.21 | +0.03 / −0.06 | 16.4 / 16.8 | 12.8 / 6.1 | 0.000 | 0.000 | 0.51 / 0.50 |
| `m_zero` | right / left | 0.23 / 0.21 | +0.03 / −0.06 | 16.4 / 16.8 | 12.8 / 6.1 | 0.000 | 0.000 | 0.51 / 0.50 |
| `weight_shuffled_frozen` | right / left | 0.50 / 0.64 | +0.04 / +0.02 | 10.9 / 13.4 | **−3.5 / 80.5** | 0.000 | 0.000 | 0.56 / 0.35 |
| `direction_shuffled_frozen` | right / left | 0.00 | 0.00 | nan | −455 | 0.000 | 0.000 | 0.00 / 1.00 |

**Criterion 1 (P4.1 mirror): ВЫПОЛНЕН.** `Δw(S_R→C_R) > Δw(S_L→C_L)` on task A for
**19/20** seeds (exact two-sided binomtest p = 4.005e-05; 19 W / 1 L / 0 T);
`Δw(S_L→C_L) > Δw(S_R→C_R)` on task B for **18/20** (p = 4.025e-04; 18 W / 2 L /
0 T). The direction of the change follows the reward contingency, not the side.

**Criterion 2 (selectivity): ВЫПОЛНЕН.** direct − contra > 0 on **20/20** seeds on
either task (mean +6.78 / +7.54), while all four frozen arms sit at
`|Δw| = 0.0000` and show no selectivity at all. `m_zero` — same learner, same
spikes, `M ≡ 0` — is byte-for-byte identical to `no_plasticity`, so the effect is
the reward, not the timing.

**Criterion 3 (baselines): НЕ ВЫПОЛНЕН on one comparison, and that comparison is
the interesting one.** `plastic` beats `no_plasticity`, `m_zero` and
`direction_shuffled_frozen` on 17–20 of the shared seeds per task (p ≤ 0.0001).
Against `weight_shuffled_frozen` it does not (9/20 right, 6/20 left) — that
statement is about *single-task* `success_rate` and is not re-run at 60 seeds; the
mixed stream below is where the same criterion is decided.

**Criterion 4:** 0 governance violations over 5 600 episodes; provenance chain
intact.

### Why a frozen lucky wiring can out-score learning on one task

`weight_shuffled` draws its four sensor→command weights as
`sym_weight × U(0.6, 1.4)` permuted over slots. Per sensor, the decision depends
on *direct vs contralateral within that sensor*, so a random draw is ~50 % likely
to favour the correct side for a given task — and the same draw is then ~50 %
likely to favour the wrong side for the mirrored task. Averaged over seeds, that
is `0.50 / 0.64`, with mean wiring selectivity `+0.36` (no direction in the
ensemble, a coin flip per seed). Its aggregate `cumulative_reward` is *negative*
on the right task (−3.5): the unlucky half runs into a wall for 25 steps.

Two follow-ups were run to find out whether this defeats the claim or only the
metric.

**Zero-shot mirror transfer** (`var/p4_transfer_probe.py`, `var/p4_tp_{right,left}.txt`,
4 arms × 20 seeds: train 20 episodes on one mirror, freeze, play the other):

| arm | train | transfer | mean min(transfer over both directions) |
|---|---|---|---|
| `plastic` | 0.42 / 0.41 | 0.19 / 0.19 | 0.10 |
| `weight_shuffled` | 0.53 / 0.73 | 0.63 / 0.46 | 0.28 |
| `weight_shuffled_frozen` | 0.50 / 0.64 | 0.63 / 0.48 | 0.28 |
| `no_plasticity` | 0.23 / 0.21 | 0.24 / 0.21 | 0.14 |

Transfer does not rescue the learning claim either: within 20 episodes R-STDP
shapes the *rewarded* direction and leaves the other one near its starting value
(wiring selectivity after training: +5.9 right-trained, −7.0 left-trained), so a
single trained wiring is no better on the mirror than an untrained one. Eval-phase
`Δw` is `0.0000` in every row, which is what makes those numbers a genuine
zero-shot read rather than a second training run.

**Mixed stream** — the condition where "better than every baseline" is actually
falsifiable. Both mirrors alternate inside one training run (`MIXED` in
`symmetric_toy.py`, parity of the episode index, same rule for every arm and
seed). A fixed wiring can only be lucky about one half; a rule that can strengthen
both direct paths is not restricted. The gate is the *worse* half of the stream,
paired per seed (`mixed_verdict` in `scripts/run_p4_validation.py`).

| operating point | arm | succ | right / left | gap | trend | reward | min-half | both ≥0.5 | both ≥0.8 |
|---|---|---|---|---|---|---|---|---|---|
| η=0.8, 40 ep | `plastic` | 0.38 | 0.35 / 0.42 | 0.07 | +0.23 | 101.5 | 0.25 | 3/20 | 0/20 |
| η=0.8, 40 ep | `weight_shuffled_frozen` | **0.58** | 0.49 / 0.66 | 0.17 | +0.01 | 80.5 | 0.33 | 7/20 | 6/20 |
| η=1.5, 40 ep | `plastic` | 0.48 | 0.45 / 0.51 | 0.06 | +0.30 | 146.8 | 0.29 | 3/20 | 0/20 |
| η=1.5, 40 ep | `weight_shuffled_frozen` | **0.58** | 0.49 / 0.66 | 0.17 | +0.01 | 80.5 | 0.33 | 7/20 | 6/20 |
| η=1.5, 80 ep | `plastic` | **0.64** | 0.62 / 0.65 | 0.03 | +0.32 | **440.6** | **0.46** | **10/20** | 1/20 |
| η=1.5, 80 ep | `weight_shuffled_frozen` | 0.57 | 0.49 / 0.65 | 0.16 | −0.01 | 152.6 | 0.33 | 7/20 | 3/20 |
| η=1.5, 80 ep | `no_plasticity` ≡ `m_zero` | 0.21 | 0.21 / 0.22 | 0.01 | +0.01 | 33.0 | 0.18 | 0/20 | 0/20 |
| η=1.5, 80 ep | `direction_shuffled_frozen` | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | −1821.5 | 0.00 | 0/20 | 0/20 |

Sources: `var/p4_mixed.csv` / `var/p4_mixed_analysis.txt` (η=0.8),
`var/p4_mixed_eta15.csv` (η=1.5, 40 ep), `var/p4_mixed80_{a,b}.csv` /
`var/p4_mixed80_analysis.txt` (η=1.5, 80 ep, 0 governance violations, 8 000
provenance records, chain intact).

Three things this table says, separately:

1. **Balance-with-improvement is a learning signature.** `plastic` is the only arm
   that keeps its two mirror halves equal (gap 0.01–0.07) *while* its success
   rises; `weight_shuffled_frozen` is lopsided on every seed set (gap 0.08–0.17)
   and flat (trend −0.01…+0.01). A lucky wiring is good at one half and bad at the
   other, by construction. Balance alone is not the signature — `no_plasticity` is
   balanced too (0.21 / 0.21), it just cannot improve on it.
2. **At 40 episodes learning genuinely loses** (0.38 and 0.48 vs 0.58) — on every
   metric except balance, including `success_rate`. That is reported, not
   explained away. Doubling the budget at η=1.5 flips the comparison: 0.64 vs 0.57,
   reward 440.6 vs 152.6, min-half 0.46 vs 0.33, seeds solving *both* halves ≥0.5
   10/20 vs 7/20 — so the failure at 40 episodes was a budget effect, not a
   ceiling. (The one count where the frozen arm still leads at 80 ep is the top of
   the range: seeds with both halves ≥0.8, 3/20 vs 1/20.)
3. **Against three of the four baselines the gain is significant; against
   `weight_shuffled_frozen` it is not, at n=20.** Paired per-seed, 80 ep:
   `no_plasticity`/`m_zero` 20/0 on success and 16/4 on min-half (p ≤ 0.0118),
   `direction_shuffled_frozen` 20/0 (p = 0.0000). Against `weight_shuffled_frozen`
   on min-half: plastic better on **14/20** seeds, mean 0.46 vs 0.33, sign-test
   **p = 0.1153**, Wilcoxon p = 0.0894 → `gate_decision` prints *НЕ ДОКАЗАНО
   (большинство есть, значимости нет)*, and the run exits 1. This is the one open
   item, and the confirmatory run below settles it.

The paired spread of `min-half` is what sets the sample size, so the confirmation
is pre-declared before looking at the new seeds:

### Confirmatory run — mixed stream, 60 seeds × 80 episodes, η=1.5

Seeds 0–19 are the exploratory set above; **seeds 20–59 are fresh** and the gate is
the same one, unchanged: majority of shared seeds **and** paired binomial sign
test p < 0.05 on `success_mirror_min`, arm `plastic` vs `weight_shuffled_frozen`.

| seed set | arm | succ | right / left | gap | trend | reward | min-half | both ≥0.5 | both ≥0.8 |
|---|---|---|---|---|---|---|---|---|---|
| fresh 20–59 (n=40) | `plastic` | **0.67** | 0.68 / 0.67 | 0.01 | +0.36 | **479.1** | **0.53** | **25/40** | 3/40 |
| fresh 20–59 (n=40) | `weight_shuffled_frozen` | 0.43 | 0.47 / 0.39 | 0.08 | −0.00 | −66.1 | 0.22 | 8/40 | 7/40 |
| fresh 20–59 (n=40) | `no_plasticity` ≡ `m_zero` | 0.21 | 0.21 / 0.21 | 0.00 | +0.01 | 20.2 | 0.17 | 0/40 | 0/40 |
| fresh 20–59 (n=40) | `direction_shuffled_frozen` | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | −1822.1 | 0.00 | 0/40 | 0/40 |
| all 60 | `plastic` | **0.66** | 0.66 / 0.66 | 0.00 | +0.34 | **466.2** | **0.51** | **35/60** | 4/60 |
| all 60 | `weight_shuffled_frozen` | 0.48 | 0.48 / 0.48 | 0.00 | −0.01 | 6.8 | 0.25 | 15/60 | 10/60 |
| all 60 | `no_plasticity` ≡ `m_zero` | 0.21 | 0.21 / 0.21 | 0.01 | +0.01 | 24.4 | 0.18 | 0/60 | 0/60 |
| all 60 | `direction_shuffled_frozen` | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | −1821.9 | 0.00 | 0/60 | 0/60 |

Paired per seed, **fresh seeds only (n=40)** vs **pooled (n=60)**, gate metric
`success_mirror_min` (`var/p4_conf60_report.txt`):

| baseline | fresh seeds (n=40) | pooled (n=60) | gate on fresh seeds |
|---|---|---|---|
| `no_plasticity` | 37/39, p = 0.0000 | 53/59, p = 0.0000 | ОК |
| `m_zero` | 37/39, p = 0.0000 | 53/59, p = 0.0000 | ОК |
| `weight_shuffled_frozen` | **31/38, p = 0.0001** | 45/58, p = 0.0000 | ОК |
| `direction_shuffled_frozen` | 40/40, p = 0.0000 | 60/60, p = 0.0000 | ОК |

Denominators above are `n_informative = wins + losses`; ties leave the
denominator. Ties excluded, fresh seeds: 1 (`no_plasticity`), 1 (`m_zero`),
2 (`weight_shuffled_frozen`), 0 (`direction_shuffled_frozen`). Exact values,
recomputed from the raw per-seed rows by `scripts/recheck_p4_stats.py`:
2.841e-09, 2.841e-09, 1.162e-04, 1.819e-12.

On `success_rate` the fresh set gives 40/40 (p = 0.0000) against `no_plasticity` /
`m_zero`, 40/40 against `direction_shuffled_frozen`, and **28/40, p = 0.0166**
against `weight_shuffled_frozen`; `cumulative_reward` 479.1 vs −66.1 (30/40,
p = 0.0027). Pooled over 60 seeds, `success_rate` vs `weight_shuffled_frozen` is
38/59, p = 0.0363.
→ **Criterion 3 on the mixed stream: ВЫПОЛНЕН against all four baselines.**
The statistical unit is the **seed** (n = 40 fresh, n = 60 pooled), never an
episode and never a log line. The 24 000 governed episodes across the two shards
(0 governance violations, 24 000 provenance records, both chains intact) are
**run integrity**, not replication: 24 000 rows cannot raise n above 40.

Reproducibility cross-check (`var/p4_determinism_check.py`): the earlier standalone
20-seed run and seeds 0–19 of this 60-seed run are two separate processes over the
same cells — 240 compared values, **0 discrepancies**, which is what makes merging
the two shards into one paired comparison legitimate.

What did *not* replicate, and matters:

- the direction replicated, the **magnitude did not**: exploratory n=20 gave
  0.46 vs 0.33 (p = 0.1153), fresh n=40 gives 0.53 vs 0.22. The frozen arm's mean
  over 20 seeds is itself a lottery — the 20-seed set happened to draw a lucky
  ensemble. This is the concrete cost of judging a bimodal baseline on its mean.
- the frozen arm still wins the **top of the range**: seeds solving *both* halves
  ≥ 0.8 are 7/40 fresh (10/60 pooled) vs 3/40 (4/60) for `plastic`. A wiring that
  happens to come out strong on both direct paths gets both mirrors right almost
  every episode and never has to learn anything; R-STDP at 80 episodes averages
  0.66, not 0.9, and 32% of its plastic weights are sitting on a bound.
- neither half alone is significant (p = 0.74 right, 0.09 left) — the signal lives
  in the *worse* half, which is why the gate was declared there and not on
  `success_rate` per task.
- the single-task matrix was never re-run at 60 seeds. Criterion 3 still reads
  **НЕ ВЫПОЛНЕН** on single-task `success_rate` at n=20 (9/20, 6/20 against
  `weight_shuffled_frozen`); the pass above is on the mixed stream, which is the
  condition where the criterion means what the plan intended by it.


## Reproducing

```bash
.venv/bin/python -m pytest fly_connectome_agent/tests -q            # incl. harness gates
./scripts/check.sh                                                  # syntax + tests + schemas
.venv/bin/python scripts/run_p4_validation.py --quick               # ~1 min smoke
.venv/bin/python scripts/run_p4_validation.py --seeds 20 --probe    # mirror matrix, ~20 min
.venv/bin/python scripts/run_p4_validation.py --tasks mixed --seeds 20 --episodes 40 --probe
# the convergence / confirmatory measurements (sharded so one arm set per process):
.venv/bin/python scripts/run_p4_validation.py --tasks mixed --seeds 60 --episodes 80 --eta 1.5 \
    --arms plastic,weight_shuffled_frozen --out var/p4_conf60_a.csv
.venv/bin/python scripts/run_p4_validation.py --tasks mixed --seeds 60 --episodes 80 --eta 1.5 \
    --arms no_plasticity,m_zero,direction_shuffled_frozen --out var/p4_conf60_b.csv
.venv/bin/python var/p4_conf60_report.py var/p4_conf60_a.csv var/p4_conf60_b.csv --min-seed=20
.venv/bin/python var/p4_determinism_check.py          # 240 values, 0 discrepancies
.venv/bin/python scripts/run_p4_validation.py --replay var/p4_mixed.csv   # re-score, no simulation
# the failed operating point, for comparison:
.venv/bin/python scripts/run_p4_validation.py --seeds 8 \
    --step-cost -0.1 --tau-eligibility 20 --command-inh 0
```

`--tasks` takes `right,left[,mixed]`, `--arms` takes a subset — both exist so a
long matrix can be sharded and so the mixed stream can be measured on its own.
`var/p4_conf60_report.py` merges shards, applies the same `gate_decision` the driver
uses, and `--min-seed=N` restricts the verdict to the fresh confirmatory seeds;
`var/p4_mixed_analysis.py` is the wider paired sweep over every metric.
Row-level output: `var/p4_results.csv` (mirror matrix), `var/p4_mixed*.csv` and
`var/p4_conf60_*.csv` (mixed stream, by operating point); per-episode provenance:
`var/p4_provenance.jsonl`, `var/p4_mixed*_prov.jsonl`, `var/p4_conf60_*_prov.jsonl`.

`var/` is gitignored, so the CSVs and reports cited above are local artifacts, not
repo contents — every number in this document is re-derivable from the commands
above, which is the point of writing them out.

## Known limits

- **The strongest non-learning control is not measured.** `weight_shuffled_frozen`
  is a *random* reflex. A reflex that simply strengthens both direct paths (one
  wiring solving both mirrors) would be an oracle control, and it would very likely
  beat 80 episodes of R-STDP. That experiment changes the question from "can the
  rule carry a direction" to "can learning beat knowing the answer", so it is not
  run here — but any claim that learning "wins" has to be read against it.
- One operating point per verdict. η=1.5 / 80 episodes is where learning overtakes
  the lucky wiring; η=0.8 and every 40-episode point are where it does not. All are
  reported, and the pass at η=1.5/80 is the one that needed 60 seeds to become
  significant.
- 5 neurons, 10 edges, 1-D track. It tests whether the *learning rule* can carry
  a direction, not whether a fly brain does.
- `INH→C` synapses are fixed (`plastic_slots=(0,1,2,3)`). When they are plastic,
  the dominant effect of the modulator is global disinhibition: both command
  neurons saturate, their counts tie, the decoder answers `stay`, and behaviour
  degrades while the direction-specific `Δw` looks excellent. Reported, not hidden.
- Terminal reward lands inside the last decision window, so an approach after an
  overshoot credits the opposite sensor. This is genuine credit-assignment noise,
  visible as the residual per-seed scatter in the mirror test.
- `weight_shuffled` is allowed to win on individual seeds: random asymmetry helps
  in a tie-dominated regime. The gate is on the aggregate.
- Sensor grounding is scripted (`sensory_drive`): the environment says which side
  the target is on. Direction *selection* is what is learned here; perception is
  not.
