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
```

Immutable, frozen, JSON-serializable.

### PolicyDecision

```python
@dataclass(frozen=True)
class PolicyDecision:
    status: Literal["ALLOW", "DENY", "ESCALATE"]
    enforced_action: str  # move_left | move_right | stay
    reason: str
```

GateKeeper rules:
- `Decider` role: verbs `выбирает` → ALLOW.
- `Decider` role: L5 verbs (e.g. `якорит`) → DENY.
- Forbidden verbs → DENY.
- High-risk context → ESCALATE.
- DENY and ESCALATE → enforced_action = `stay`.
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
- `verify_chain()` checks all hashes.
- Single-process atomic append via `threading.Lock` (MVP).
- No blockchain/Registry claims.
- Tamper detection: on-disk modification detected on new `ProvenanceLog` instance.

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

## Test Matrix

| Component | Tests |
|---|---|
| ActionDecoder | determinism, winner L/R, tie→stay, confidence [0,1], invalid shape→ValueError |
| GateKeeper | Decider+выбирает→ALLOW, Decider+якорит→DENY, forbidden verb→DENY, high-risk→ESCALATE, DENY/ESCALATE→stay, no SNN mutation |
| SimpleNavigation | reset deterministic, boundaries, target termination, max_steps truncation, invalid action→ValueError |
| Provenance | 3 events→verify True, tampered payload→verify False, bad previous_hash→reject, deterministic hash |
| Closed Loop | ALLOW path, DENY path→stay |

## Acceptance Criteria

- All pytest pass.
- No changes in `science/`.
- No exploratory imports.
- Environment does not depend on GateKeeper.
- GateKeeper does not depend on LIF internals.
- Provenance passes tamper test.
- Closed-loop covers ALLOW and DENY.
- README claims unchanged.
