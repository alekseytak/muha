# P4.2b — Sharded Execution Protocol

## Nature

Infrastructure-only: parallel or sequential chunked execution of a frozen
P4.2 experiment. This is NOT a new scientific experiment. No R-STDP, LIF,
arms, metrics, hypotheses, reward, or statistical methods are changed.

## Motivation

P4.2.v1 and P4.2.v2 were aborted by infrastructure limits (O(n) append;
single-process wall-clock exceeding execution-environment timeout). A full
1080-cell run requires approximately 3–4 hours; the hosting environment
cannot sustain a single uninterrupted process for that duration.

P4.2b splits the run into independently-verifiable chunks (shards), each
with its own provenance log and sidecar. A merge tool produces a single
complete result manifest that is the only artifact the scientific gate
accepts.

## Shard identity

Each shard carries (see `schemas/p4_shard_manifest.schema.json`):

```json
{
  "shard_schema_version": "1.0.0",
  "protocol_id": "<parent experiment protocol_id>",
  "manifest_digest": "sha256:...",
  "git_sha": "<exact SHA the shard was run from>",
  "shard_id": "<globally unique>",
  "shard_axis": "arms",
  "seed_subset": [180, 181, ...],
  "arm_subset": ["rstdp_mixed", "no_plasticity", "m_zero"],
  "task_streams": ["left_target", "right_target", "mixed"],
  "episodes_per_seed": 80,
  "partial": true,
  "csv_sha256": "sha256:...",
  "provenance_jsonl_sha256": "sha256:...",
  "provenance_head_sha256": "sha256:...",
  "environment_fingerprint": { ... },
  "started_at_utc": "...",
  "finished_at_utc": "..."
}
```

## Partition rule — v1: arms only

The first version partitions by **arm groups**. No arm is split across
seeds or streams:

```text
Example: 6 arms, 2 shards

Shard A:
  arms = [rstdp_mixed, no_plasticity, m_zero]
  seeds = full confirmatory set
  streams = full stream set
  episodes = 80

Shard B:
  arms = [weight_shuffled_frozen, direction_shuffled_frozen, oracle_reflex]
  seeds = full confirmatory set
  streams = full stream set
  episodes = 80
```

Each shard covers all seeds × all streams for its arm subset.

## Hard requirements

### Exact partition

The union of all shard `arm_subset` values must equal the protocol's full
arm list. No arm may appear in two shards.

### No overlap, no gaps

The merge tool (`check_p4_shard_coverage.py`) refuses if:

- same (arm, seed, stream) cell exists in two shards;
- any required arm, seed, or stream is missing;
- shards use different `manifest_digest`;
- shards use different `git_sha`;
- shards use different `environment_fingerprint`;
- shard CSV hash differs from its recorded `csv_sha256`;
- provenance JSONL or head witness hash mismatch;
- shard is marked `non_confirmatory`;
- shard protocol_id differs from the parent;
- sidecar is missing;
- provenance chain verification fails.

### Merged manifest is the only gate-eligible artifact

The merge tool produces `p4_2_merged_result_manifest.json` with:

```json
{
  "partial": false,
  "coverage_verified": true,
  "merge_status": "COMPLETE"
}
```

The scientific gate (`check_p4_2_oracle_gate.py`) must reject any CSV with
`partial=true` in its sidecar. Only the merged manifest and its associated
merged CSV are eligible for verdict.

### No silent reuse of burned data

The merge tool refuses to incorporate:

- shards from P4.2.v1 or P4.2.v2;
- any protocol whose registry status is not `authorized`;
- any shard whose seed range overlaps previously-burned ranges
  (checked against all known `p4_2_burned_seeds` records).

## Runtime budget

Before launching, `run_p4_shard.py` estimates shard duration:

```
cells = len(arm_subset) × len(seed_subset) × len(task_streams)
estimated_seconds = cells × seconds_per_cell
```

Policy: maximum target = 20 minutes (1200 seconds) per shard.
If estimate exceeds target, the tool prints:

```
P4.2b shard plan requires a new approved seed-axis sharding protocol.
```

and refuses to proceed. It does NOT silently split seeds.

## Tools

| script | role |
|---|---|
| `scripts/run_p4_shard.py` | Execute one shard: validates shard plan against parent manifest, calls the existing runner logic, writes shard CSV + sidecar + provenance |
| `scripts/merge_p4_shards.py` | Merge N shard results into one complete CSV + merged result manifest |
| `scripts/check_p4_shard_coverage.py` | Verify exact coverage: no overlaps, no gaps, all hashes match, all environments agree |

## Prohibited

This protocol does not:

- create new experiment manifests or seed sets;
- authorize any run;
- modify scientific metrics, arms, hypotheses, or statistical tests;
- change the provenance writer;
- touch R-STDP, LIF, GateKeeper, or Soup boundary;
- produce any output file.

## Status

```
P4.2b SHARD INFRASTRUCTURE PREPARED
NO P4.2.v3 PROTOCOL CREATED
NO NEW SEEDS ALLOCATED
NO RUN AUTHORIZED
```
