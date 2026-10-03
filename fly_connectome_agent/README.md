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