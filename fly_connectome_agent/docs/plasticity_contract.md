# Plasticity Contract

## 1. Rule

Edge-sparse reward-modulated STDP.

## 2. Causal trace ordering

For each timestep:

1. Decay eligibility traces on edges.
2. Compute `dw` using traces from the **previous** timestep and
   current spikes.
3. Apply modulator: `dw_eff = eta * M * dw`.
4. Update edge weights: `w <- clip(w + dw_eff, w_min, w_max)`.
5. Update pre/post traces with current spikes.

Simultaneous spikes (discretisation artefact at `dt = 1 ms`) follow
the convention declared in the manifest:
`simultaneous_spike_convention ∈ {zero, ltd_dominant, ltp_dominant}`.
Default: `zero`.

## 3. Storage

Eligibility traces are stored **only on plastic edges** (`O(E_plastic)`).
No dense `N × N` matrices are permitted at any point.

## 4. Sign constraints

Plasticity may not change the sign of an edge whose biological type
is known (E/I policy from `edge_convention.md`).

Critical invariant: eligibility traces accumulate on plastic edges regardless of the instantaneous modulator value `M`.  When `M = 0` (zero reward signal) the weight update is `dw_eff = 0` and weights remain unchanged for that timestep, but eligibility continues to decay and accumulate — do **not** suppress eligibility updates on plastic edges when `M = 0`; this is required for correct delayed-reward credit assignment.  Any optimisation that zeroes out eligibility when `M = 0` is incorrect and will break temporal credit assignment across reward delays.

Implementation: after `clip(w + dw_eff, w_min, w_max)`, re-apply
sign constraint per edge class.

## 5. Homeostasis (placeholder)

MVP: per-neuron firing-rate homeostasis with target rate declared in
the manifest. Mechanism unspecified at P0; must be selected at P2
with unit tests.

## 6. Modulator

`M = alpha * R_task - beta * P_safety - gamma * E_energy
     - delta * V_policy + epsilon * C_coherence + zeta * A_advantage`

Coefficients are profile parameters recorded in the manifest.
`V_policy` is a hard penalty for GateKeeper violations.

## 7. Reward delivery (`apply_modulator`)

Spike timing and reward delivery are separate operations:

- `update(pre, post, modulator=0.0)` on every simulation timestep accumulates
  eligibility with correct pre/post timing and leaves weights unchanged
  (the `M = 0` invariant from section 4).
- `apply_modulator(M)` delivers a delayed reward in one call:
  `dw = eta * clip(M, -M_max, M_max) * eligibility`, then weights are updated
  with sign constraints. It does **not** decay eligibility, does not touch
  pre/post traces, and does not advance `step_count`. Rewards must not be
  smuggled through a fake `update(silent, silent, M)` call: that conflates
  learning timing with the absence of spikes.
- Non-finite `M` raises `ValueError`; `NoPlasticityBaseline.apply_modulator`
  returns zero `dw` of shape `(E_plastic,)` for interface compatibility.