# P4.2c — Seed-Axis Sharding Protocol

## Nature

Infrastructure-only extension of the P4.2b sharded execution protocol. This
is NOT a new scientific experiment. No R-STDP, LIF, arms, metrics, hypotheses,
reward modulator, GateKeeper, provenance writer, Soup boundary, or statistical
methods are changed. No P4.2.v3 manifest, seed allocation, run authorization,
or output CSV is produced by this document or its tooling.

P4.2c widens `shard_axis` from `["arms"]` to `["arms", "seeds"]` and adds a
strict **merged-gate adapter** so the scientific gate can only ever score a
fully re-verifiable merged result.

## Motivation

P4.2b partitions by arm groups only. Its own runtime planner proved that is not
enough for a full 60-seed confirmatory run:

```text
3 arms × 60 seeds × 3 streams = 540 cells
540 cells × ~12 s/cell ≈ 6 480 s ≫ 1 200 s (20-minute target)
```

Measured from the P4.2.v2 abort (`var/aborted/p4_2_v2_*`): **120 / 1080 cells in
~1 436 s ≈ 12 s/cell** while the journal was still small. The v1/v2 degradation
to 40 s/cell was the O(n) re-read tail; a shard starts its *own* fresh
provenance and never enters that regime, so ~12 s/cell is the honest estimate.

The only dimension that can be cut below the budget is the **seed set**. P4.2c
adds a seed axis so a confirmatory seed set is split into non-overlapping
seed-shards, each starting a fresh small journal.

## Shard identity

Each shard carries the P4.2b identity fields (`schemas/p4_shard_manifest.schema.json`),
with `shard_axis` now one of `arms` or `seeds`.

## Partition rules

Exactly one axis is sliced per merge. A single merge may contain **either**
arm-axis shards **or** seed-axis shards — never a mix (mixed axes do not reduce
to one exact partition).

### Arms axis (P4.2b, retained)

```text
arm_subset partitions the full arm list;
every shard keeps ALL seeds and ALL streams.
```

### Seeds axis (P4.2c)

```text
seed_subset partitions the full confirmatory seed set exactly;
every shard keeps ALL arms and ALL streams and the same episodes_per_seed;
no seed appears in two shards;
the union of seeds equals the full confirmatory set.
```

Illustrative only — NOT a new experiment:

```text
Shard S1: seeds 180–189, all 6 arms, all 3 streams   (180 cells)
Shard S2: seeds 190–199, all 6 arms, all 3 streams   (180 cells)
...
Shard S6: seeds 230–239, all 6 arms, all 3 streams   (180 cells)
```

## Exact partition validation

`check_p4_shard_coverage.py` is mode-aware:

- `arms` mode: arm subsets partition all arms; every shard holds all seeds and streams;
- `seeds` mode: seed subsets partition all seeds; every shard holds all arms and streams;
- mixed axes in one merge are refused immediately.

Coverage/merge also refuse on: seed overlap, seed gap, arm list differing between
seed-shards, stream mismatch, episode mismatch, differing `git_sha`, differing
`manifest_digest`, differing `environment_fingerprint`, invalid
`partial`/`non_confirmatory`, CSV/provenance/head hash mismatch, a provenance
chain that contradicts its witness, duplicate CSV cells, and a total cell count
that is not `arms × seeds × streams`.

## Runtime budget and planner

`run_p4_shard.py` accepts `--axis {arms,seeds}` and estimates:

```text
cells = len(arm_subset) × len(seed_subset) × len(task_streams)
estimated_seconds = cells × seconds_per_cell   (measured default ≈ 12 s/cell)
```

Policy: refuse (exit 2) if `estimated_seconds > 1200`. It never silently changes
the seed subset — a requested seed outside the frozen confirmatory set is an
error, and any seed in the burned range **0–179** (pilot + v1 + v2) is refused.

**Honest finding surfaced by the planner:** the illustrative 10-seed shard is
*still* over the 20-minute budget:

```text
6 arms × 10 seeds × 3 streams = 180 cells → ~2 160 s > 1 200 s   (refused)
6 arms ×  5 seeds × 3 streams =  90 cells → ~1 080 s ≤ 1 200 s   (fits)
```

So seed-axis sharding works, but a 60-seed confirmatory set fits a strict
20-minute-per-shard target only at **≤ 5 seeds per shard** (12 shards), or with a
combined arms×seeds split, or on an environment without a 30-minute timeout.
The reviewer must choose; the tool will not quietly loosen the budget.

## Merged-gate adapter

The scientific gate may render a verdict only on a re-verifiable merged result.
`merge_p4_shards.py` writes `p4_2_merged_result_manifest.json` (with
`merge_axis` and per-shard `source_shards` evidence), then
`check_p4_2_oracle_gate.build_merged_sidecar` produces the conventional final
sidecar for the merged CSV:

```json
{
  "partial": false,
  "non_confirmatory": false,
  "coverage_verified": true,
  "merge_status": "COMPLETE",
  "manifest_digest": "...",
  "git_sha": "...",
  "merged_result_manifest": "…/p4_2_merged_result_manifest.json",
  "merged_result_manifest_sha256": "sha256:...",
  "merged_csv_sha256": "sha256:...",
  "source_shard_count": 6
}
```

`verify_merged_sidecar` re-derives the whole trust chain from disk before a
verdict:

```text
sidecar(merged) → merged result manifest (re-hash == sidecar anchor)
                → merged CSV            (re-hash == sidecar AND manifest)
                → every source shard manifest → CSV/provenance/head re-hash
```

The gate refuses if: an individual shard CSV is passed (its sidecar is
`partial=true`); the merged CSV has no sidecar; the merged result manifest is
missing; the merged manifest SHA does not match the sidecar; `coverage_verified`
≠ true; `merge_status` ≠ COMPLETE; `source_shard_count` disagrees with the
manifest and its `source_shards`; a source shard's on-disk bytes no longer match
its recorded hashes; or the merged CSV bytes were changed.

Because every link is re-hashed, a merged result cannot be forged by editing one
file — the sidecar anchor, the merged manifest, the merged CSV, and all shard
files must agree simultaneously.

## Tools

| script | role |
|---|---|
| `scripts/run_p4_shard.py` | Run one shard on either axis: validate the plan, estimate the budget, refuse burned/out-of-protocol seeds, delegate to the frozen runner, write shard CSV + provenance + shard manifest |
| `scripts/check_p4_shard_coverage.py` | Mode-aware exact-partition check for arms or seeds; refuse mixed axes |
| `scripts/merge_p4_shards.py` | Validate identity/hashes/provenance, re-check burned seeds and aborted/Unauthorized protocols, merge CSVs, write merged manifest + merged sidecar |
| `scripts/check_p4_2_oracle_gate.py` | Scientific gate; `build_merged_sidecar` / `verify_merged_sidecar` enforce the merged-evidence chain |

## Prohibited

This protocol does not: create a P4.2.v3 manifest or seed set; authorize any
run; modify scientific metrics, arms, hypotheses, or statistical tests; change
the provenance writer; touch R-STDP, LIF, GateKeeper, or the Soup boundary;
reopen any aborted archive; or produce any verdict output.

## Status

```
P4.2c SEED-AXIS SHARD INFRASTRUCTURE PREPARED
P4.2.v3 NOT CREATED
NO NEW SEEDS ALLOCATED
NO RUN AUTHORIZED
```
