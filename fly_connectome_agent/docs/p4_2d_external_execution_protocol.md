# P4.2d — External Execution Environment Protocol

## Nature

Infrastructure-only: a repeatable **execution protocol** for running **one full
P4.2.v3 run** in an environment without the 30-minute IDE timeout. This is NOT a
scientific experiment and NOT a run. It does not create a P4.2.v3 manifest, does
not allocate any seed range, does not authorize any run, does not launch Colab,
and does not install anything. No P4.2c shard is run and no combined arms×seeds
split is introduced.

P4.2c proved that a strict 20-minute-per-shard budget cannot fit a 60-seed
confirmatory set for 6 arms × 3 streams at the measured ~12 s/cell. Rather than
slice the science thinner, P4.2d defines how to run the whole thing **once**, in
one environment, reproducibly — and how to collect the evidence so the scientific
gate can consume it afterwards.

## Motivation

P4.2.v1 and P4.2.v2 were aborted by the hosting environment (a single
uninterrupted process cannot survive ~1.5–4 h inside a 30-minute IDE timeout).
Sharding (P4.2b/P4.2c) is the workaround *when the timeout is unavoidable*. When
a non-timeout environment is available, the cleaner path is one full run on one
machine — but only if the environment is verified **before** seeds are burned and
the artifact set is complete **after**. That is what these three tools enforce.

## 1. Environment options

```text
preferred : a local terminal or SSH session with no IDE timeout;
allowed   : Google Colab, but ONLY as a single isolated execution environment
            (one notebook session == one full run == one machine);
prohibited: mixing shards across different machines or environments.
```

The prohibition is enforced by the fingerprint: a run's `execution_mode` and
`host.machine_class` are recorded, and the merged/scoring path may only consume
artifacts that agree on a single environment. Cross-machine mixes are refused —
they cannot reduce to one reproducible fingerprint.

## 2. Required preflight (`scripts/p4_external_preflight.py`)

Before any seed is spent, preflight must be green. It refuses (exit 2) if any
check fails. Checks:

```text
exact git SHA     — HEAD must equal the pinned expected SHA;
clean tree        — `git status --porcelain` empty;
pinned deps       — a `--requirements name==version` lock, installed-matched;
tests             — pytest / check.sh green (--run-tests);
frozen digest     — manifest digest == the in-code frozen anchor;
fingerprint       — all mandatory environment fields present;
empty outputs     — declared bundle paths free, directory has no stray files;
fresh journal     — provenance JSONL and its head witness both ABSENT.
```

The core `evaluate_preflight(inputs)` is a pure function over a snapshot of
facts, so every refusal is testable without touching the real machine. Preflight
runs nothing and writes nothing into the bundle.

## 3. Environment fingerprint (`scripts/p4_environment_fingerprint.py`)

`collect_fingerprint()` produces a record with all mandatory fields (checked by
`missing_fields()`):

```text
git_sha, protocol_id, manifest_digest, python, numpy, scipy, os, cpu_arch,
host, execution_mode, captured_utc
```

`host` is **anonymized**: a `machine_class` (`system/arch`) plus a one-way
`hostname_sha256` used only to compare "same host?" between two records — the raw
hostname never ships in the bundle. `execution_mode ∈ {external-full-run,
colab-isolated-run}`. This is an extension of `collect_p4_2_bundle.py`'s
`environment.json`, not a replacement (that frozen-manifest format is out of
scope); the raw-hostname field of the older fingerprint is deliberately not
reused here.

## 4. Artifact collection (`scripts/p4_collect_external_bundle.py`)

A full external run must leave an 11-artifact bundle. The contract lives in code
(`REQUIRED_ARTIFACTS`), mirroring how `REQUIRED_BUNDLE_ROLES` lives in
`p4_2_protocol.py` rather than in a self-declared file:

| role | file |
|---|---|
| run_csv | `run.csv` |
| csv_sidecar | `run.csv.meta.json` |
| provenance_jsonl | `provenance.jsonl` |
| provenance_head_witness | `provenance.jsonl.head.json` |
| start_metadata | `run.start.json` |
| environment_fingerprint | `environment.json` |
| run_log | `run.log` |
| gate_stdout | `gate.stdout.txt` |
| gate_verdict_json | `gate.verdict.json` |
| per_seed_outcomes | `per_seed_outcomes.csv` |
| file_hashes_manifest | `bundle_sha256.json` (self-excluded) |

The two additions over the frozen 9-role bundle — `start_metadata` and `run_log`
— are what distinguishes "the run died at seed 40" from "the run never started".
The collector refuses (exit 2) if any required artifact is missing (CSV, sidecar,
head witness, …); it sha256-hashes every required file; and it can only mark the
bundle `complete=true` when **all** required hashes are present. `provenance.jsonl`
without its head witness is refused: a single journal file is indistinguishable
from a truncated one.

## 5. No unsafe resume

```text
- if the process dies, archive the partial output as-is (do NOT keep writing);
- do NOT resume on the same seeds;
- do NOT merge a partial run with a later run;
- require a NEW protocol id and a FRESH seed set.
```

Structural support: preflight refuses when the provenance journal or its witness
already exists (`fresh_journal`), and when any declared output path is occupied
(`empty_outputs`). There is no `--resume` in any of these tools. Burned ranges
(0–59 pilot, 60–119 v1, 120–179 v2) stay burned; a dead external run's seeds are
burned too and cannot be re-derived by the same protocol id.

## 6. No scientific execution

This protocol and its tools do not: create a P4.2.v3 manifest; open a new seed
range; authorize a run; run anything on Colab; install Soup or any dependency;
touch R-STDP, LIF, the reward modulator, GateKeeper, the provenance writer, the
Soup boundary, P4 metrics, or statistical methods; or produce a verdict.

## Tools

| script | role |
|---|---|
| `scripts/p4_environment_fingerprint.py` | Collect the external-run environment fingerprint (anonymized host + execution mode) |
| `scripts/p4_external_preflight.py` | Refuse to start unless the environment is exact, clean, pinned, tested, digest-matched, output-empty, and journal-fresh |
| `scripts/p4_collect_external_bundle.py` | Validate the 11-artifact bundle, hash every required file, emit `bundle_sha256.json`, mark COMPLETE only when whole |

## Example external execution sequence (ILLUSTRATIVE — not run here)

```bash
# 0. one machine, no IDE timeout (terminal/SSH; or a single Colab session).
git clone <repo> && cd muha
git checkout <pinned-sha>
python -m venv .venv && .venv/bin/pip install -r requirements.lock   # pinned ==versions

# 1. preflight must be green before any seed is spent.
.venv/bin/python scripts/p4_external_preflight.py \
    --expected-git-sha <pinned-sha> --requirements requirements.lock \
    --execution-mode external-full-run --run-tests            # exit 0 required

# 2. THE run (only ever authorized by a separate review act on a v3 protocol id).

# 3. collect + hash the evidence bundle.
.venv/bin/python scripts/p4_collect_external_bundle.py --bundle-dir var/p4_2_v3
```

## Status

```text
P4.2d EXTERNAL EXECUTION PROTOCOL PREPARED
P4.2.v3 NOT CREATED
NO NEW SEEDS ALLOCATED
NO RUN AUTHORIZED
NO COLAB RUN PERFORMED
```
