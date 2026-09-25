# Claim Policy

## 1. Claim classes

| Claim | Evidence required |
|---|---|
| `simulation_result` | Successful run + provenance log + manifest |
| `engineering_performance` | GateKeeper / provenance / action-decoder tests |
| `structural_effect` | Significant difference vs all graph baselines |
| `biological_plausibility` | §2 requirements from `scientific_scope.md` |
| `behavioural_claim` | Out-of-scope at current stage |

## 2. Forbidden inferences

- From `simulation_result` → `biological_plausibility`.
- From `engineering_performance` → `structural_effect`.
- From `exploratory/` module output → any scientific claim without
  baseline comparison and ablation.

## 3. Manifest enforcement

`experiment_manifest.claims_allowed` is a closed enum. Any claim not
listed is forbidden for that run.

## 4. Audit

Each published result must link to:
- `connectome_manifest` (data provenance),
- `experiment_manifest` (run parameters),
- provenance log hash,
- baseline comparison report.