# Edge Convention

## 1. Direction

All edges are directed: `src` = pre-synaptic neuron, `tgt` =
post-synaptic neuron.

Matrix convention: `W[pre, post]` is the contribution of a pre-spike
to the post-synaptic current.

In code: `I_syn_post += W[pre, post] * spike_pre`.

## 2. Weight mapping profiles

Weight mapping from `synapse_count` (and optional `neurotransmitter`)
to initial `W` is a **versioned profile**, not a fixed formula.

Each profile is identified by `profile_id` and fully parameterised.

### Candidate profile: `log_quantile_v1`

```text
raw = synapse_count
x = log1p(raw)
x_norm = (x - quantile_05) / (quantile_95 - quantile_05)
x_clipped = clip(x_norm, 0.0, 1.0)

# Separate profiles per sign class:
excitatory: w = +w_exc_max * x_clipped
inhibitory: w = -w_inh_max * x_clipped
```

where `w_exc_max`, `w_inh_max`, `quantile_05`, `quantile_95` are profile
parameters recorded in the manifest. No single `w_min`/`w_max` —
excitatory and inhibitory ranges are independent.

### E/I / unknown-edge policy

| Neurotransmitter annotation | Weight formula | Range | Plastic? |
|---|---|---|---|
| glutamate / acetylcholine (excitatory) | `w = +w_exc_max * x_clipped` | `w ∈ [0, w_exc_max]` | per subset |
| GABA (inhibitory) | `w = -w_inh_max * x_clipped` | `w ∈ [-w_inh_max, 0]` | per subset |
| unknown | **not mapped by default** | — | per subset |

Sign is **fixed** for edges whose biological type is known; plasticity
may not flip the sign.

### Unknown annotation policy

If a neuron has no neurotransmitter annotation, the manifest must
declare `unknown_annotation_policy`: one of
`exclude`, `separate_class`.
Default: `exclude` (unknown edges are not included in the weight map;
they must be explicitly opted into via `separate_class` with its own
`w_exc_max`/`w_inh_max`).

## 3. Structural vs plastic edges

- `structural_edges`: fixed weights, never modified by plasticity.
- `plastic_edges`: subset on which R-STDP is applied; sign-constrained
  by E/I policy.
- `readout_edges`: fixed, used for motor / action decoding.

The partition must be recorded in the manifest.

## 4. Self-loops and parallel edges

Self-loops: forbidden unless explicitly allowed by
`edge_policy.allow_self_loops` (default: `false`).
Parallel edges: merged by summing `synapse_count` before mapping,
unless `edge_policy.merge_parallel` is `false`.