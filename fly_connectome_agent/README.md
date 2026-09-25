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

## Quick start (P1)

```bash
pip install -r requirements.txt
pytest tests/
```

This runs P1 unit tests: graph loader/validation/subgraph, sparse LIF
refractory/noise/NaN-free/deterministic replay.

## Status

`P2.1 IMPLEMENTED — PENDING REVIEW` — edge-sparse reward-modulated STDP with
neuron-level traces O(N), edge-level eligibility O(E_plastic), modulator M
clipping with diagnostics, and per-edge-class sign constraints.

P2.1 test suite: 28 tests covering LTP/LTD timing, simultaneous conventions,
sign constraints, sparse memory, NaN-freedom, deterministic replay,
no-plasticity baseline compatibility, modulator scaling/clipping, and
non-finite input rejection.

Excludes GateKeeper runtime, exploratory, and swarm layers until separate P2 review.