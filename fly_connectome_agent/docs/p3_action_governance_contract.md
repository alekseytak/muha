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
- `verify_chain()` re-reads the file from disk (never trusts the memory cache)
  and checks payload hash, entry hash and chain linkage.
- Appends are serialized by `threading.Lock` **and** `fcntl.flock` (POSIX), so
  two processes writing the same file cannot fork the chain; the chain tail is
  read under the lock and the line is `fsync`-ed before release.
- Also fails when the log shrank below what this instance wrote, or when a
  cached event no longer matches its stored bytes.
- Threat model is *tamper-evident, single-host*: no external anchoring/Registry.
  A fresh reader cannot detect a truncation that is itself a valid shorter
  chain — it needs a writer that witnessed the longer log. Full-file rewrite by
  someone who can recompute hashes is out of scope.
- No blockchain claims.

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
