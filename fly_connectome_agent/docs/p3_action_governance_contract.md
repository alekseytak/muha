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
  `log_byte_offset`); only bytes written past the witnessed offset are re-read,
  which is nothing unless another process appended in the meantime. The file is
  never parsed whole per append, and events are not retained in memory between
  appends (`get_events()` / `verify_chain()` opt into the full scan).
- Ordering inside `append()`: take `threading.Lock` + `fcntl.flock`, resync from
  the witness, write one canonical line, `fsync` the log, atomically replace the
  witness (`tmp` + `os.replace`), `fsync` the directory. A crash between the two
  durability points leaves the log *ahead* of its witness, and the next append
  reconciles that tail; the witness never runs ahead of the log.
- `append()` raises `ProvenanceIntegrityError` and writes nothing when the log is
  shorter than the witness claims, when bytes past the witnessed offset do not
  continue the chain, or when the tail ends mid-line.
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
  valid chain is indistinguishable from a truncated one. Rewriting the log and
  recomputing a matching witness is out of scope, and always was.
- No blockchain claims.

### Why the writer was rewritten (P4.2.v1 abort)

The previous `append()` re-read and re-parsed the whole JSONL on every call to find
the chain tail: O(n) per append, therefore O(N²) per run. Measured median cost of a
single append, old versus fixed implementation (`.venv/bin/python
scripts/bench_provenance_append.py`, JSON lands in `var/provenance_append_benchmark.json`):

| prior entries | old append (median) | fixed append (median) |
|---|---|---|
| 0 | 4.19 ms | 2.09 ms |
| 1 000 | 30.40 ms | 2.15 ms |
| 10 000 | 200.54 ms | 2.25 ms |
| 50 000 | 1 192.45 ms | 1.90 ms |

The table is one run; it reproduces. Three consecutive runs gave a growth of the fixed
path of ×0.91, ×0.95 and ×1.0 between an empty log and a 50k log — flat, within noise —
while the old path over the same distance grew ×285, ×224 and ×210. Absolute milliseconds
wander with disk and payload size; the only stable fact is the slope, and the old slope was
linear in `n` while the new one is not a slope at all.

The honest trade: an append now costs ~2 ms even on a short log, because every write pays
one extra `fsync` plus an atomic witness replace. The old code could be cheaper per append
while the log was small (a direct probe on an empty file once measured 0.7 ms) — which is
exactly the regime where nobody noticed anything was wrong.

Acceptance gates, both satisfied by the shipped artifact: median at 50k entries ≤ 3× the
empty-log median (measured ×0.91, gate ×3.0), and provenance share of a full
86 400-episode confirmatory run < 15% of runtime (measured 165 s against ≈ 14 570 s of
simulation, i.e. 1.1%; across the three runs 1.1–1.2%). The same run on the old path
projects to 25–40 hours of provenance alone — the linear coefficient measured on a
synthetic 100-byte payload gives ~27 h, the live P4.2.v1 log with ~700-byte episode
payloads gives ~42 h.

`scripts/bench_provenance_append.py` pins the pre-fix revision it compares against
(`OLD_REV`), and refuses to run if asked to load a revision that already contains the
fixed writer — otherwise, once the fix is committed, the benchmark would silently measure
the new path against itself and report a win with no competitor.

P4.2.v1 was aborted for this defect before completion (240/1080 cells, no CSV, no
sidecar, frozen gate never invoked). Its artifacts live apart from any future result,
under `var/aborted/p4_2_v1_infrastructure_abort/`. Seeds 60–119 were exercised by that
run and are not reusable in the replacement protocol.

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
| Provenance | append/verify, on-disk payload tamper, bad entry hash, truncation seen by writer, two-process flock chain of 50 events |
| Closed Loop | left/right/stay proposal executed in env, DENY/ESCALATE→stay with movement blocked, proposal/decision/action recorded apart, forged ALLOW rejected |

## Acceptance Criteria

- All pytest pass.
- `ALLOW` executes the proposal, `DENY`/`ESCALATE` execute `stay`, and no test
  or script fabricates a decision the GateKeeper could not produce.
- Provenance is tamper-evident within the stated single-host threat model, with
  the limitation documented rather than papered over.
- Environment does not depend on GateKeeper; GateKeeper does not depend on LIF internals.
- README claims unchanged in scope (still `simulation_result`, no biological claims).
