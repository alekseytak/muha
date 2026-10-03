# P3 Action Governance Contract

## Scope

Minimal closed-loop contour:

```
SNN spikes → ActionDecoder → VerbAct → GateKeeper → environment → observation/reward → provenance
```

No changes to `science/snn`, `science/learning`, no exploratory modules, no LLM/teacher/student.

## Types

### ActionProposal

```python
@dataclass(frozen=True)
class ActionProposal:
    action_id: str
    action_type: Literal["move_left", "move_right", "stay"]
    score: float
    confidence: float
    source_window_steps: int
```

Rules:
- Deterministic on same spike window.
- Tie (equal best scores) → `stay`.
- `confidence` ∈ [0, 1]; normalized difference best - second.
- Invalid spike shape → `ValueError`.

### VerbAct

```python
@dataclass(frozen=True)
class VerbAct:
    event_id: str
    subject_id: str
    role: str
    verb: str
    object_ref: str
    domain: "simulation"
    context: dict
    trace_id: str
    proposal_ref: str
    proposed_action: Literal["move_left", "move_right", "stay"]
```

Immutable, frozen, JSON-serializable.

Rules (P3.1):
- `VerbAct.from_proposal(proposal, ...)` is the single sanctioned constructor in
  the loop: it copies `proposed_action` from `proposal.action_type` after a
  validity check and carries `proposal.action_id` into `proposal_ref` /
  `object_ref`. The governance record therefore cannot drift from what the
  decoder actually proposed.
- A hand-built `VerbAct` is allowed only in tests; `GateKeeper.evaluate` still
  rejects an unknown `proposed_action` with `DENY`.

### PolicyDecision

```python
@dataclass(frozen=True)
class PolicyDecision:
    status: Literal["ALLOW", "DENY", "ESCALATE"]
    proposed_action: str  # exactly what the decoder proposed
    enforced_action: str  # move_left | move_right | stay
    reason: str
```

Governance invariant, enforced in `__post_init__` — a violating decision cannot
be constructed at all:
- `ALLOW` ⇒ `enforced_action == proposed_action`.
- `DENY` / `ESCALATE` ⇒ `enforced_action == "stay"` (the safe fallback).

GateKeeper rules:
- `Decider` role: verbs `выбирает` → ALLOW (proposal executed as-is).
- `Decider` role: L5 verbs (e.g. `якорит`) → DENY.
- Forbidden verbs, unknown role, invalid domain, invalid `proposed_action` → DENY.
- High-risk context → ESCALATE.
- No SNN weight/state mutation.

Allowed roles:

| Role | Verbs |
|---|---|
| `Decider` | `выбирает` |
| `Guardian` | `блокирует`, `эскалирует`, `разрешает` |
| `Registrar` | `якорит` (only in provenance context) |

### ProvenanceEvent

```python
@dataclass(frozen=True)
class ProvenanceEvent:
    event_id: str
    timestamp_utc: str  # ISO-8601
    previous_hash: str  # sha256 hex or empty
    payload: dict
    payload_hash: str  # sha256 canonical JSON
    entry_hash: str    # sha256 of {previous_hash, payload_hash}
```

Rules:
- Frozen dataclass, JSON-serializable.
- Canonical JSON: `sort_keys=True`, compact `","`, `":"` separators.
- `append()` is **O(1) in the length of the log**. The chain tail comes from a
  compact head witness (`<log>.head.json`: `entry_count`, `last_hash`,
  `log_byte_offset`, plus `version` and a `head_digest` over those fields); only
  bytes written past the witnessed offset are re-read, which is nothing unless
  another process appended in the meantime. The file is never parsed whole per
  append, and events are not retained in memory between appends
  (`get_events()` / `verify_chain()` opt into the full scan).
- **The log and its witness are one provenance artifact.** `<log>.head.json` is not
  a disposable cache and the JSONL is not a standalone record: an archive holding
  only one of the two is an invalid state, not a smaller version of the valid one.
  A result bundle for any confirmatory run must preserve *both* files and carry the
  SHA-256 of *both*. This is not bookkeeping: the witness is what makes a shortened
  log detectable, so a bundle that hashes only the JSONL silently gives up the one
  property the fast writer was allowed to keep.
- Ordering inside `append()`: take `threading.Lock` + `fcntl.flock`, resync from
  the witness, write one canonical line, `fsync` the log, atomically replace the
  witness (`tmp` + `os.replace`), `fsync` the directory. A crash between the two
  durability points leaves the log *ahead* of its witness, and the next append
  reconciles that tail; the witness never runs ahead of the log.
- `append()` raises `ProvenanceIntegrityError` and writes nothing whenever the
  witness and the log disagree: the log is shorter than the witnessed byte offset
  (truncated, or a witness pointing past EOF); the record that actually ends at the
  witnessed offset re-hashes to something other than `last_hash` — this catches a
  witness edited together with a recomputed `head_digest`, and equally a log edited
  under an honest witness; the witness promises a non-empty chain at
  `log_byte_offset = 0`; bytes past the witnessed offset do not parse as JSONL, do
  not continue the chain, or end mid-line; the log is non-empty and has no witness.
- A non-empty log without a witness is never adopted silently.
  `recover_head(reason)` is the only way to continue one: a single full scan, refusal
  if the chain does not close, the witness written, and a
  `provenance_head_recovered` event — reason, number of adopted entries, adopted tail
  hash, log size — appended into the chain itself, outside the lock. The adoption is
  then visible to every later reader of the record it created, instead of being an
  unlogged repair. A confirmatory runner refuses to start on such a log until that
  call has been made deliberately: `scripts/run_p4_2_oracle.py` exits with rc=2
  before the first episode rather than burning a confirmatory budget on a journal the
  writer will not accept (`test_runner_refuses_a_provenance_log_it_cannot_trust`).
  `count` is the one read-only escape hatch: it reports the truth about the bytes
  even when the witness is unusable, and it does not write.
- Appends are serialized by `threading.Lock` **and** `fcntl.flock` (POSIX), so
  two processes writing the same file cannot fork the chain: both resync and
  update the witness under that lock.
- `verify_chain()` is the O(N) audit, for startups, recovery and reporting. It
  re-reads the file from disk (never trusts the memory cache), recomputes payload
  and entry hashes, checks linkage, and cross-checks the witness (count, tail
  hash, byte offset). It also fails when the log shrank below what this instance
  wrote, or when a cached event no longer matches its stored bytes.
- Threat model is *tamper-evident, single-host*: no external anchoring/Registry.
  With a witness present, truncation of a log is detectable even by a fresh
  reader. Delete the witness and the old limitation returns: a shorter, internally
  valid chain is indistinguishable from a truncated one — the writer now refuses to
  append to it, but nothing on the log's side proves what is missing. Rewriting the
  log and recomputing a matching witness is out of scope, and always was: the
  `head_digest` is an inconsistency detector, not a signature.
- No blockchain claims.

### Why the writer was rewritten (P4.2.v1 abort)

The previous `append()` re-read and re-parsed the whole JSONL on every call to find
the chain tail: O(n) per append, therefore O(N²) per run. Median cost of one append,
old versus fixed implementation, measured on the writer that ships now
(`.venv/bin/python scripts/bench_provenance_append.py`). The table is runs 1–2 of
`fly_connectome_agent/docs/benchmarks/provenance_writer_p4_2a.json` — the versioned
record of this acceptance, committed on purpose; the raw per-run JSON in `var/` is
machine-local and gitignored, and an acceptance that lives only there cannot be
re-checked after the machine changes.

| prior entries | old append (median) | fixed append (median) |
|---|---|---|
| 0 | 3.63 ms | 1.99 ms |
| 1 000 | 23.84 ms | 2.09 ms |
| 10 000 | 150.71 ms | 1.96 ms |
| 50 000 | 749.42 ms | 2.33 ms |

Absolute milliseconds do not reproduce — not between machines, and barely between
runs on one machine. The old path measured 2 617 ms at 50k entries when this
benchmark was first run, and 749 ms / 928 ms now. What does reproduce, and what the
acceptance is stated in terms of, is the slope and the fractions:

- fixed path, growth from an empty log to a 50k log: ×1.17 and ×1.03 on the two
  recorded runs, against the gate of ×3.0. Four earlier runs on the pre-review writer
  (`8c90ba9`, before the witness self-hash and the append-time cross-check existed)
  gave ×0.91, ×0.95, ×1.0, ×1.23 — the review added per-append work and the slope
  stayed where it was;
- old path over the same distance: ×206 and ×189 here, ×210…×354 there — linear in
  `n`, as advertised by the code that re-parsed the file per append;
- projected provenance share of a full 86 400-episode confirmatory run: 1.4% and
  1.2% here (1.1–2.1% across the earlier four) — against the gate of 15%. On the old
  path the same run projects to 25–40 hours of logging alone: the linear coefficient
  measured on synthetic ~100-byte payloads gives ~27 h, the live P4.2.v1 log with
  ~700-byte episode payloads gives ~42 h.

The honest trade: an append now costs a couple of milliseconds even on a short log,
because every write pays one extra `fsync`, an atomic witness replace, and — since the
review — a re-hash of the witnessed tail line. The old code could be cheaper per
append while the log was small (a direct probe on an empty file once measured 0.7 ms)
— which is exactly the regime where nobody noticed anything was wrong.

`scripts/bench_provenance_append.py` pins the pre-fix revision it compares against
(`OLD_REV`), and refuses to run if asked to load a revision that already contains the
fixed writer — otherwise, once the fix is committed, the benchmark would silently
measure the new path against itself and report a win with no competitor. Every full
run appends one entry to `runs_detail` in the summary and recomputes the aggregates
over all recorded runs taking worst values (max growth, max share), not averages: an
acceptance that averages away a single bad run is not an acceptance. A `--quick` run
is refused entry — the gate is defined at 50k entries, and a half-populated summary
would be a different benchmark protocol wearing the same name.

P4.2.v1 was aborted for this defect before completion: 305 of 1080 cells, 24 421
episodes written, chain intact, no CSV, no sidecar, frozen gate never invoked, no partial
data interpreted. The abort needed correcting after the fact: the first stop (19:29) killed
the shell wrapper while the worker process kept running for another 1.5 hours, so the
archive holds both the 19:31 snapshot and the real final state (21:07), with
`post_abort_correction.md` recording which number came from which. Seeds 60–119 were
exercised by that run and are not reusable in the replacement protocol.

Operational rule taken from it: stopping a background run means checking the worker, not
the wrapper — `ps aux | grep run_p4_2` plus a size-sampling check that the provenance file
stopped growing.

## Environment: simple_navigation

1D line world.

```python
class SimpleNavigation:
    def reset(self, seed=None) -> (observation, info)
    def step(self, action: str) -> (observation, reward, terminated, truncated, info)
```

Actions: `move_left`, `move_right`, `stay`.

Rules:
- Deterministic with fixed seed.
- Boundaries enforced; invalid action → `ValueError`.
- Target position → `terminated=True`.
- `max_steps` → `truncated=True`.
- Step cost: small negative per step.
- Boundary collision: penalty.
- `info` keeps the same keys (`position`, `prev_position`, `step`) after
  termination, so the loop never crashes on a post-terminal step.

## Provenance payload for a governed step

Every executed step records the three distinct things separately:
`proposal` (what the decoder asked for), `decision` (ALLOW/DENY/ESCALATE),
`enforced_action` (what the environment actually received), plus
`position_after` and `reward`. A record where these three are conflated cannot
be used to audit governance.

## Test Matrix

| Component | Tests |
|---|---|
| ActionDecoder | determinism, winner L/R, tie→stay, confidence [0,1], invalid shape→ValueError |
| GateKeeper | Decider+выбирает→ALLOW, Decider+якорит→DENY, forbidden verb→DENY, high-risk→ESCALATE, ALLOW passes proposal through for L/R/stay, decision invariants unconstructable, `from_proposal` copy + rejection |
| SimpleNavigation | reset deterministic, boundaries, target termination, max_steps truncation, invalid action→ValueError, stable info shape |
| Provenance | append/verify, on-disk payload tamper, bad entry hash, truncation seen by writer, two-process flock chain of 50 events; head witness: modified head / offset past EOF / true-hash-with-wrong-offset → append refuses and the log bytes are untouched; missing head → no silent adoption (only `recover_head`, which marks the chain and refuses a broken one); a lie confined to `entry_count` survives append but fails `verify_chain`; structurally, no whole-file scan on the append path |
| Closed Loop | left/right/stay proposal executed in env, DENY/ESCALATE→stay with movement blocked, proposal/decision/action recorded apart, forged ALLOW rejected |

## Acceptance Criteria

- All pytest pass.
- `ALLOW` executes the proposal, `DENY`/`ESCALATE` execute `stay`, and no test
  or script fabricates a decision the GateKeeper could not produce.
- Provenance is tamper-evident within the stated single-host threat model, with
  the limitation documented rather than papered over.
- Environment does not depend on GateKeeper; GateKeeper does not depend on LIF internals.
- README claims unchanged in scope (still `simulation_result`, no biological claims).
