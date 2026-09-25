# Scientific Scope

## 1. What the connectome provides

- Node identities, region annotations, neurotransmitter hints,
  synapse counts, reconstruction confidence.
- Directed edge list (pre → post).

## 2. What the connectome does NOT provide

- Membrane time constants, resting potentials, threshold distributions.
- Synaptic weights, delays, rise/decay times.
- Neuromodulator dynamics.
- Sensory encoding and motor readout mappings.
- Behavioural function.

All of the above are **modelling assumptions**.

## 3. Prohibited claims at current stage

| Claim | Status |
|---|---|
| "reproduces fly brain" | FORBIDDEN |
| "biologically plausible dynamics" | FORBIDDEN without §4 validation |
| "models fly emotion / consciousness" | FORBIDDEN |
| "quantum effects in neurons" | FORBIDDEN |
| "1–20 Hz is the biological firing rate of fly neurons" | FORBIDDEN |
| "Schumann / Budanov frequencies are fly-brain parameters" | FORBIDDEN |
| "simulates full male CNS" | FORBIDDEN at MVP |

## 4. Path to biological claims

Biological claims require, at minimum:
- comparison against degree-preserving rewired baseline;
- comparison against weight-shuffled baseline;
- firing-rate / CV(ISI) / Fano-factor reporting by neuron class;
- ablation of functionally annotated subgraphs;
- multiple seeds with confidence intervals;
- explicit mapping from connectome annotations to model parameters.

Until all of the above are satisfied, only `simulation_result` and
`engineering_performance` claims are permitted.

## 5. Exploratory modules

Modules in `src/exploratory/` (log-frequency banks, structured
exploration noise, TDA studies) are **hypotheses**, not runtime
components. They may be activated only via an explicit
`experiment_manifest.exploratory_modules_active` entry, with
`biological_interpretation: "NOT_ESTABLISHED"` and mandatory
baseline comparison.