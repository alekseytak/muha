# Connectome-Informed Drosophila SNN Agent

## Scope

This project uses a *Drosophila melanogaster* connectome as a **structural
graph prior** for computational spiking-network experiments.

It is **not** a model of:
- the full fly brain,
- fly consciousness,
- fly endocrinology,
- quantum processes in neural tissue,
- biologically validated fly behaviour.

Connectome = structural scaffold.
Neuron dynamics, synaptic weights, delays, plasticity parameters,
neuromodulatory variables, and task mappings are **modelling assumptions**
and must be versioned with each experiment.

## What is claimed

- `simulation_result`: measured outcomes of a specific parameterised run.
- `engineering_performance`: behaviour of the governance / provenance /
  action-decoding layers.

## What is NOT claimed

- `biological_plausibility` (without separate experimental validation).
- `reproduced_fly_behaviour`.
- `consciousness`, `emotion`, `quantum cognition`.

## Quick start

```bash
pip install -r requirements.txt
pytest tests/
```

Unit and integration tests: graph loader/validation/subgraph, sparse LIF
(refractory, noise, delays, deterministic replay), sparse reward-modulated STDP
timing and sign constraints, action decoding, governance, provenance, and the
symmetric P4 toy harness gates.

The scientific measurement itself is a script, not a test:

```bash
python ../scripts/run_p4_validation.py --seeds 20 --probe           # mirror matrix
python ../scripts/run_p4_validation.py --tasks mixed --seeds 20 --episodes 40   # both mirrors, one run
python ../scripts/run_p4_validation.py --tasks mixed --seeds 60 --episodes 80 --eta 1.5 \
    --arms plastic,weight_shuffled_frozen --out ../var/p4_conf60_a.csv          # confirmatory
python ../var/p4_conf60_report.py ../var/p4_conf60_a.csv --min-seed=20           # gate on fresh seeds
python ../scripts/run_p4_validation.py --replay ../var/p4_results.csv  # re-score, no simulation
python ../scripts/run_p4_loop.py --out ../var/p4_3_confirm --confirm  # frozen P4.3, refuses drift
python ../scripts/report_p4_3.py ../var/p4_3_confirm/summary.json      # applies H1-H5, G1-G5
```

Protocol, pass criteria, the first (failed) measurement and the mechanism fixes
are in [docs/p4_validation.md](docs/p4_validation.md); the governance contract is
in [docs/p3_action_governance_contract.md](docs/p3_action_governance_contract.md).

## Status

`P3.1 IMPLEMENTED` — `VerbAct.from_proposal` is the only sanctioned constructor,
`PolicyDecision` cannot be built as a lie (ALLOW must preserve the proposal,
DENY/ESCALATE must enforce `stay`), provenance appends under a cross-process file
lock and verifies against the bytes on disk.

`P4 / P4.1 — MIRROR TEST PASSES; BEHAVIOUR CRITERION PASSES ON THE MIXED STREAM`:
on 20 seeds the trained network gives `Δw(S_R→C_R) > Δw(S_L→C_L)` on task A (19/20,
p<0.0001) and the reverse on task B (18/20, p=0.0004), with every frozen arm at
`Δw ≡ 0`, so the direction comes from the reward contingency and not from the wiring.
On behaviour the criterion needed a fair condition first: on a single task a randomly
asymmetric frozen wiring scores 0.50–0.64 by luck alone, so `--tasks mixed` runs both
mirrors inside one training run and scores the *worse* half. At 80 episodes × 60 seeds
`plastic` reaches 0.66 (halves 0.66/0.66, reward 466) against 0.48 for
`weight_shuffled_frozen`, 0.21 for `no_plasticity`/`m_zero` and 0.00 for
`direction_shuffled_frozen`; on the 40 fresh confirmatory seeds the pre-declared gate
gives 31/38 seeds, p=0.0001. Two honest limits: the criterion still fails on
single-task `success_rate` at 20 seeds, and a wiring that is lucky in *both*
directions solves both mirrors perfectly more often than 80 episodes of learning do
(both halves ≥0.8: 10/60 frozen vs 4/60 plastic). See docs/p4_validation.md before
reading any number here as a capability claim. No real connectome data is loaded.

`P4.3 — ACTION IS A SPIKE (pre-registered, frozen channel)`:
on the 1515-node FlyWire role subgraph the verb is not assigned but read from which
output pool won a 100-step window, and the modulator comes from the environment's
reward. 500 episodes × 3 confirmatory seeds (7×7 corridor, 12 decisions): `plastic`
reaches 0.290/0.330/0.390 → 0.750/0.780/0.800 (first vs last 100 episodes) against
0.280/0.270/0.320 → 0.290/0.280/0.320 for the frozen topology, spike budget ratio
0.971 (criterion ≤ 2), zero gate denials, oracle at 1.000. The mechanism endpoint —
accuracy of the verb where the correct verb is defined — grows 0.78 → 0.89–0.90 on all
three plastic seeds; the pre-registered H4 rule still reports NOT PASSED, because two
frozen cells drifted +0.003/+0.011 while the rule demanded no positive delta. That
rule had no effect-size floor and is recorded as mis-specified for v2, not
reinterpreted here. n=3: description, not significance. Protocol and result:
[docs/p4_3_spike_verb_protocol.md](docs/p4_3_spike_verb_protocol.md).

`P4.3 v2 — POOL CONTRAST (pre-registered, negative result)`:
one change to the same loop — the losing output pool learns at `(1−κ)·M` with
κ = 0.5, so the pools differ in learning speed while the sign of the credit is
preserved. 500 episodes × 3 fresh seeds (301/311/323): the treatment loses success
on every seed (0.390/0.220/0.250 → 0.210/0.170/0.090) while the paired `plastic`
control under the v1 scalar gains (0.440/0.270/0.290 → 0.910/0.950/0.690), paired
Δ −0.65/−0.73/−0.56; verb accuracy falls 5.8–6.6 pp where v1 gains 11.0–17.7 pp,
and wall bumps in the last 100 episodes are 34/28/51 against 0/0/2. Spike budget
ratio 1.025 and gates G1–G7 pass, so the run is informative: in this physics the
rule is not neutral but harmful. The rule splits its own effect by the sign of the
reward — on rewarded steps the loser is attenuated (preference toward the winner),
on punished steps it is punished less (preference toward the loser). Measured on
10 episodes of the treatment arm, 26.7% of steps carry `M < 0`; why the positive
majority does not compensate is left to an ablation (contrast only when `M > 0`),
declared as the next pre-registration rather than applied to v2; that ablation is
the v3 entry below. The pilot on a
burned seed had shown the same direction and κ was not retuned after it. Protocol,
pilot and result:
[docs/p4_3_v2_pool_protocol.md](docs/p4_3_v2_pool_protocol.md); run and report
with `scripts/run_p4_loop.py --confirm --kappa 0.5` and `scripts/report_p4_3_v2.py`.

`P4.3 v3 — CREDIT-SIGN ABLATION (pre-registered, negative result)`:
splits the same κ = 0.5 contrast by the sign of the modulator — `pool_pos` applies it
only on rewarded steps (M > 0), `pool_neg` only on punished steps (M < 0), and the
other sign falls back to the v1 scalar. 500 episodes × 3 fresh seeds (401/411/421),
5 arms, 15 cells. `pool_pos` does not learn: 0.310/0.370/0.320 → 0.290/0.130/0.220,
Δ −0.020/−0.240/−0.100, verb accuracy 0.769→0.766, 0.781→0.717, 0.762→0.742,
20–28 wall bumps. `pool_neg` matches the v1 control: Δ +0.520/+0.470/+0.560 against
+0.510/+0.480/+0.590, paired difference −0.01/+0.01/−0.03, accuracy +13.9/+10.4/+15.7
pp, zero bumps. Frozen topology stays flat, oracle 1.000, gates G1–G8 pass, verdict
rc=1 (H1–H4 NOT PASSED). The measured `mean_abs_dw` in the last 100 episodes is
2.1–2.4·10⁻⁴ for `pool_pos` against 4.2–4.8·10⁻⁴ for both other arms, and `stay` is
17/16/14 against 3/6/3: halving the credit on ~45% of edges on most steps removes half
the learning and the policy never forms. So the damage of v2 came from the rewarded
steps, attenuation on punished steps is neutral, and pool contrast is rejected as a
fix in both forms. The productive signal here is the absolute amount of credit on the
stimulated path, not the ratio between pools; the v1 scalar remains the working rule.
The next candidate is per-neuron credit (the v4 entry below), which needs its own
pre-registration. Protocol,
pilot and result:
[docs/p4_3_v3_credit_sign_ablation.md](docs/p4_3_v3_credit_sign_ablation.md); run and
report with `scripts/run_p4_loop.py --confirm --kappa 0.5` and
`scripts/report_p4_3_v3.py`.

`P4.4 v4 — PER-NEURON CREDIT (pre-registered, negative result)`:
replaces the single third factor with one per postsynaptic neuron, built from the
rank of that neuron's eligibility mass in the closed window
(`s_i = 1 − λ + 2λ·rank/(m − 1)`, λ = 0.5, mean exactly 1 over active neurons,
inactive neurons get 1), so credit is redistributed rather than cut.
`neuron_shuffled` is the null: the same set of factors permuted across the same
neurons by `(seed, episode, step)`. 500 episodes × 3 fresh seeds (501/511/521),
5 arms, 15 cells. `neuron` gives Δ +0.63/+0.56/+0.66 with accuracy
+18.5/+14.3/+17.8 pp and `mean_abs_dw` 6.2/5.8/5.9·10⁻⁴; the v1 control gives
+0.64/+0.44/+0.54 at 4.8/4.3/4.4·10⁻⁴; the shuffled null gives +0.58/+0.49/+0.37
at 4.2/4.1/3.7·10⁻⁴. Binding to participation beats the shuffled null on every
seed (+0.05/+0.07/+0.29), so H1, H3 and H4 pass, but the treatment loses to v1 on
seed 501 by one episode of 100 (−0.01) and wins by +0.12 on the other two, so the
pre-registered all-seed criterion H2 fails and the verdict is rc=1. Gates G1–G8
pass, frozen topology stays flat, oracle 1.000. Row 3 of the declared table fires:
the binding matters, the per-neuron form is not a fix, and the v1 scalar remains
the working rule. Caveat found after the run: rank factors correlate with
eligibility by construction, so `neuron` also delivers 1.28–1.36× more total
credit than v1 while the shuffled null preserves only the unweighted mean;
concentration and amplitude are therefore confounded, and the next discriminating
run needs an eta-matched control or a renormalized factor (taken up by the v5
entry below). Protocol, pilot and
result: [docs/p4_4_neuron_credit_protocol.md](docs/p4_4_neuron_credit_protocol.md);
run and report with `scripts/run_p4_loop.py --confirm --neuron-credit 0.5` and
`scripts/report_p4_4_neuron_credit.py`.

`P4.5 v5 — EQUAL AMPLITUDE, DIFFERENT DISTRIBUTION (pre-registered, negative result)`:
removes the confound found after v4. The rank factor is unchanged
(λ = 0.5, κ = 0), but every arm divides it by `rho = Σ s_i·E_i / Σ E_i` over the
active neurons, which makes the credit delivered to the edges equal to the v1
scalar at every step (`Σ |dw|` identical by construction), so the only difference
left is how the same total is distributed. `neuron_renorm_shuffled` permutes the
factors and then renormalizes, so treatment and null are amplitude-matched to each
other and to v1; a random permutation gives `rho` ≈ 1, which is why this null is
nearly the v4 null. Unit tests pin both directions of the confound before any run:
with renormalization `Σ |M|·|elig|` equals v1 to 1e-12, without it the same pattern
gives 28/24 = 1.17×. 500 episodes × 3 fresh seeds (601/611/621), 5 arms, 15 cells,
protocol `p4.5.spike-verb.neuron-credit-amplitude.v1` frozen at sha256
`55b8f747a6773d4b601b9ce83b3b18322d86d7cf582eb8f692aa323c9e1003d7`. Smoke (seed 7,
5 episodes): `mean_abs_dw` 2.89·10⁻⁴ for `neuron_renorm` against 2.93·10⁻⁴ for v1
and 3.53·10⁻⁴ for the un-renormalized `neuron` — the identity holds and the v4
excess reproduces. Pilot on burned seed 182: `neuron_renorm` 0.31 → 0.89
(Δ +0.580), accuracy 0.781 → 0.922, 0.019 at bound; the renormalized null
0.34 → 0.72 (Δ +0.380). Both pass the declared feasibility gates, and neither the
rule nor λ nor the outcome table was changed after the pilot. The verdict turns on
two all-seed signs: A — the treatment beats v1 at equal amplitude, B — it beats the
renormalized null.

Confirmatory result (15 cells, verdict rc=1): `neuron_renorm` gives Δ +0.55/+0.54/+0.53
with accuracy +14.5/+14.8/+13.3 pp, the v1 control +0.46/+0.58/+0.55, the renormalized
null +0.52/+0.53/+0.37. A = +0.09/−0.04/−0.02 — not positive on every seed, so H2 fails;
B = +0.03/+0.01/+0.16, so H3 passes; H1 and H4 pass with C = 1.051/1.023/0.965, which is
the amplitude identity actually holding inside the declared [0.9, 1.1] band (v4 measured
1.28–1.36). Gates G1–G8 pass, frozen topology stays flat (−0.02/−0.08/−0.13), oracle
1.000. Row 3 of the declared table fires: with total credit equalized, the per-neuron
form stops beating the scalar, so the v4 near-miss was amplitude, and the credit-form
line is closed — the v1 scalar remains the working rule and the next step is the timing
of delivery. Caveats recorded after the run: measured `mean_abs_dw` of the null is
slightly below the treatment (3.77–4.29 against 4.48–4.70·10⁻⁴, because trajectories
differ even though the per-step identity holds), so binding is still not fully separated
from delivered credit; and B is one to three episodes of 100 on two of three seeds, so
its sign is stable while its size is not. Result:
[docs/p4_5_neuron_credit_amplitude.md](docs/p4_5_neuron_credit_amplitude.md); run
and report with `scripts/run_p4_loop.py --confirm --neuron-credit 0.5` and
`scripts/report_p4_5_neuron_credit_amplitude.py`.

`P4.6 v6 — CREDIT DELIVERY TIMING (pre-registered; smoke OK, pilot BLOCKED on its
own mechanism veto, confirmatory not run)`:
the closed line was the *form* of the third component, the working rule is the v1
scalar, and every wave so far delivered the modulator in the same instant as the
action. v6 moves one variable only — when credit arrives.
`LoopSettings.credit_delay_steps` holds M in a queue and applies it D steps later;
D is measured in 1 ms simulation steps, the same unit as `tau_eligibility = 80`.
D ≤ window_steps is enforced, which keeps the queue at one item: the tail crosses
into the next episode's first window and exactly one credit of the run is lost
(counted in the cell meta, gate G9). A delay changes two things at once —
amplitude (`elig·exp(−D/τ)`) and address (the trace has picked up the spikes of the
delay interval) — the same confound structure that cost v4 its conclusion, so v6
pre-registers an amplitude-matched internal control instead of an eta-scaled proxy
run: `delay_live` delivers into the live trace, `delay_frozen` into the snapshot
taken at the action moment, decayed by the learner's own `decay**D` constant, and at
equal D the difference between the two arms *is* the price of addressing. Unit tests
pin the contract before any number: D = 0 is bit-identical to `plastic` for both
delay arms, delivery fires exactly D steps later, with no spikes in the interval live
and frozen are bit-identical, and `Σ|dw_frozen| = exp(−D/τ)·Σ|dw_now|` to 1e-12.
Declared levels: D = 0 (reference, with `no_plasticity` and `oracle`), D = 20
(0.7788) and D = 80 = τ (0.3679); the 5-step level from the brief (0.9394) is
excluded on record because at n = 3 it sits under the noise floor. Cell names cannot
encode the delay, so each level is a separate runner invocation into its own
directory (`--credit-delay`, `var/p4_6_d{0,20,80}`) and the manifest carries a
`grids` section that `verify_protocol` selects by delay — it refuses an undeclared
delay, a foreign arm set and a mismatched output directory, while manifests without
`grids` keep the previous behaviour. Working point inherited from p4.5 unchanged
(except `neuron_credit = 0`, i.e. rule v1, asserted against the p4.5 file by a test);
confirmatory seeds 701/711/721, burned 182 for the pilot and 7 for smoke. The verdict
turns on two all-seed signs: A — the D = 80 growth of `delay_live` is below
`plastic`, B — below `delay_frozen`; H4 is a mechanism hypothesis (measured
frozen/plastic last-100 `mean|Δw|` within ±15% of `exp(−1)`), and if it fails neither
A nor B is read as a timing effect. Protocol, frozen manifest (sha256
`d098378ed9966030f660bac9697284c08aac2cf947cf582ff84f9cee4d1ee241`), verdict tool and
22 synthetic fixtures are written before any number:
[docs/p4_6_credit_delay_protocol.md](docs/p4_6_credit_delay_protocol.md); run with
`scripts/run_p4_loop.py --confirm --credit-delay 0|20|80` and report with
`scripts/report_p4_6_credit_delay.py`.

Smoke (burned seed 7, 5 episodes): at D = 0 all three arms are bit-identical (the
three TSVs share one md5) with `deliveries = decisions` and no tail; the frozen arm
tracks the declared amplitude to +0.13% at D = 20 (0.7798 against 0.7788) and +0.25%
at D = 80 (0.3688 against 0.3679), while the live arm at D = 80 keeps 1.025 of the
immediate credit — the trace refills with the spikes of the delay interval, so a delay
here costs address, not amplitude. Pilot (burned seed 182, 500 episodes, two
invocations because `plastic` cannot carry a nonzero delay): feasibility gates pass —
first-block success 0.350 / 0.390 / 0.270 for `plastic` / `delay_live` / `delay_frozen`
all inside [0.2, 0.7], at most 0.022 at bound, both verbs alive at 477 and 449 in the
last 100, zero violations and denials, `deliveries = decisions - 1` in the delay cells.
The fourth, mechanical criterion failed as declared by its own veto: measured
last-100 `mean|Δw|` of `delay_frozen` over `plastic` is 0.2704 against the predicted
0.3679, outside the ±15% band. The confirmatory run was therefore not started and no
fresh seed was burned. Diagnosis recorded before any further number: H4 compares two
*policies that have diverged* (the frozen arm learns far less, Δ +0.160 against
+0.360, and delivers into a different eligibility mass), while the exact identity
pinned by unit tests is within one delivery; a cross-arm ratio is not a constant of
the mechanism at any correctness, so it was the wrong thing to gate on. What the
pilot does hint at, on one burned seed and not as evidence: at D = 80 the live arm
learns *better* than immediate delivery (0.390 → 0.970 against 0.350 → 0.710), which
points at row 4 of the declared table. Next: P4.6 v2 pre-registers a within-cell
mechanism gate (delivered frozen credit against the same credit applied at the action
moment, expected `exp(-D/τ)` to numerical precision), keeps the cross-arm ratio as
diagnostic C, and moves to fresh seeds; the manifest was not edited after the pilot
and the v1 pilot numbers are not reused as evidence.