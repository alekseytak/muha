# Validation Plan

## 1. Unit tests (engineering)

- LIF: refractory period, `sqrt(dt)` noise scaling, no NaN/overflow.
- STDP timing: pre-before-post → LTP; post-before-pre → LTD;
  simultaneous → declared convention.
- Sparse memory: `e_edge` is `O(E_plastic)`, no dense matrices.
- GateKeeper: ordering (ATQEC L5 deny before general L5 allow),
  deterministic replay at fixed seed.
- Manifest validation: `pytest tests/test_manifest_validation.py`.

## 2. Scientific baselines (P4)

| Baseline | Method | Manifest fields |
|---|---|---|
| `directed_in_out_degree_preserving_rewire` | Maslov–Sneppen edge swaps preserving in- and out-degree separately; no self-loops; no parallel edges unless source graph has them | `baselines[].seed`, `accepted_swaps`, `max_attempts`, `edge_policy` |
| `weight_shuffle` | Permute weights across edges of same sign class | `baselines[].seed`, `shuffle_stratification` |
| `density_matched_random_sparse` | Erdős–Rényi with same edge density | `baselines[].seed`, `density` |
| `no_plasticity` | Same graph, plasticity disabled | — |
| `no_modulator` | `M = 0` (no plasticity drive) | — |
| `constant_positive_modulator_control` | `M = 1` constant | — |

## 3. Metrics (reported per neuron class and per experiment profile)

- Task performance (accuracy / cumulative reward).
- Spike-rate distribution (per class; reported against
  `default_operating_rate_range_hz`).
- CV(ISI) distribution.
- Fano factor (spike-count variance / mean in window).
- E/I balance (ratio of excitatory/inhibitory currents).
- Network synchrony (population event rate).
- Stability (no seizure-like runaway, no all-silent collapse).
- Memory footprint, runtime.

## 4. Spike-rate reporting

`default_operating_rate_range_hz: [1.0, 20.0]` is an
**engineering target**, status `engineering_target_not_biological_claim`.

Reports must include:
- fraction of neurons below / in / above range, per class;
- population mean and median;
- no claim that this range reflects fly biology.

## 5. Repeats

Minimum 5 seeds per condition. 95% confidence intervals reported.

## 6. Acceptance criteria

MVP is accepted when:
- all unit tests pass;
- all 5 baselines run and are compared;
- performance vs baseline is reported with confidence intervals;
- no NaN, no runaway activity, no all-silent state;
- provenance log verifies.