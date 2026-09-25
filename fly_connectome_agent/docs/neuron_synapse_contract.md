# Neuron & Synapse Contract

## 1. Units

| Quantity | Unit |
|---|---|
| time | ms |
| membrane potential | mV (internal) |
| current | internal units (consistent within a run) |
| weight | internal current units per spike |

## 2. Neuron model (MVP)

Current-based Leaky Integrate-and-Fire:

```text
tau_m * dV/dt = -(V - E_L) + R_m * (I_ext + I_syn) + sigma * sqrt(tau_m) * xi(t)
```

Discretisation: Euler–Maruyama with explicit `sqrt(dt)` noise scaling.

Parameters (per-neuron or per-class):
- `E_L`, `V_th`, `V_reset`, `tau_m`, `R_m`, `sigma`, `t_refrac`.

## 3. Synapse model (MVP)

Current-based exponential synapse:

```text
tau_s * dI_syn/dt = -I_syn + sum_j w_ji * sum_k delta(t - t_j^k - d_ji)
```

Per-synapse: `w`, `d` (delay in ms, quantised to `dt`).

## 4. Time step

Default `dt = 1.0 ms`. Sensitivity analysis at `dt ∈ {0.1, 0.5, 1.0}`
is mandatory (P4). If results change materially, the model is
numerically unstable at `dt = 1.0 ms`.

## 5. RNG

Every random source receives an explicit seed recorded in the manifest:
- membrane noise,
- initial weight jitter,
- baseline graph generation,
- exploration noise (if any).

Deterministic replay at fixed seed is a mandatory unit test.

## 6. Delays

MVP: per-edge integer delay in units of `dt`.
Optional (P2+): delay distributions by neuron class.

## 7. Firing rule (fixed during repository extraction)

A neuron emits a spike in a step if it is supra-threshold **either at the
start of the step or after the integration update**:

```text
spike = (V_before >= V_th or V_after >= V_th) and not refractory
```

Rationale: the leak term pulls `V` toward `E_L`, so with the default
parameters (`E_L = -70`, `V_th = -50`, `tau_m = 20`, `dt = 1`) one step moves
the membrane by about `1.05 mV`. Checking the threshold only after the update
would let a neuron that is *already* above threshold lose its spike by
decaying below it: the leak would cancel a threshold that was already
reached. Checking only before the update would add a step of latency to
neurons driven across the threshold by input inside the step. The rule above
covers both; a driven neuron still fires in the same step it crosses.

The assumption is versioned with the rest of the model and is exercised by
`test_sparse_lif_spike_and_refractory`, `test_sparse_lif_refractory_period`
and `test_sparse_lif_delay_delivery`.
