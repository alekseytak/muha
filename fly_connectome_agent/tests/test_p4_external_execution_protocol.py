"""P4.2d — external execution protocol tests.

Proves the three infrastructure-only guards:
- preflight refuses a dirty tree / wrong git SHA / wrong manifest digest / a
  non-empty output path (and accepts a fully green environment);
- the environment fingerprint carries every mandatory field;
- the external-bundle collector refuses when a required artifact (CSV, sidecar,
  provenance head-witness) is missing, hashes every required artifact, and can
  never be marked COMPLETE without all required hashes.

No simulation, no run, no v3 manifest, no seeds. All fixtures are synthetic and
live under pytest's tmp_path.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import p4_environment_fingerprint as fpmod  # noqa: E402
import p4_external_preflight as pf  # noqa: E402
import p4_collect_external_bundle as co  # noqa: E402

FROZEN_V2 = "sha256:5f5cae42a2f0373933ead1c61307eac97f6a864f25b0927dbda4643e94d75123"
GOOD_SHA = "a" * 40


def _fingerprint():
    return fpmod.collect_fingerprint(
        {"protocol_id": "p4.2.oracle-baseline.v2"}, FROZEN_V2,
        execution_mode="external-full-run", git_sha=GOOD_SHA, hostname="runner-01")


def _passing_inputs():
    """A fully green preflight snapshot; each test perturbs exactly one field."""
    return {
        "git_sha_current": GOOD_SHA,
        "git_sha_expected": GOOD_SHA,
        "working_tree_porcelain": [],
        "requirements_problems": [],
        "tests_passed": True,
        "manifest_digest": FROZEN_V2,
        "frozen_protocol_digest": FROZEN_V2,
        "protocol_id": "p4.2.oracle-baseline.v2",
        "fingerprint": _fingerprint(),
        "occupied_paths": [],
        "stray_files": [],
        "provenance_jsonl_exists": False,
        "provenance_head_exists": False,
    }


def _make_bundle(tmp_path: pathlib.Path, *, drop: str | None = None) -> pathlib.Path:
    """Write every required external-bundle artifact with distinct bytes."""
    base = tmp_path / "external_bundle"
    base.mkdir(parents=True, exist_ok=True)
    for role, name in co.REQUIRED_ARTIFACTS.items():
        if role == drop:
            continue
        (base / name).write_text(f"{role} content\n", encoding="utf-8")
    return base


# ── preflight refusals ──────────────────────────────────────────────────────

def test_preflight_accepts_fully_green_environment():
    ok, failures = pf.evaluate_preflight(_passing_inputs())
    assert ok, failures
    assert failures == []


def test_preflight_refuses_dirty_tree():
    inputs = _passing_inputs()
    inputs["working_tree_porcelain"] = ["M scripts/run_p4_2_oracle.py"]
    ok, failures = pf.evaluate_preflight(inputs)
    assert not ok
    assert len(failures) == 1 and failures[0].startswith("clean_tree")


def test_preflight_refuses_wrong_git_sha():
    inputs = _passing_inputs()
    inputs["git_sha_current"] = "b" * 40
    ok, failures = pf.evaluate_preflight(inputs)
    assert not ok
    assert len(failures) == 1 and failures[0].startswith("git_sha")


def test_preflight_refuses_wrong_manifest_digest():
    inputs = _passing_inputs()
    inputs["manifest_digest"] = "sha256:" + "0" * 64
    ok, failures = pf.evaluate_preflight(inputs)
    assert not ok
    assert len(failures) == 1 and failures[0].startswith("frozen_digest")


def test_preflight_refuses_non_empty_output_path():
    inputs = _passing_inputs()
    inputs["occupied_paths"] = ["run_csv"]
    ok, failures = pf.evaluate_preflight(inputs)
    assert not ok
    assert len(failures) == 1 and failures[0].startswith("empty_outputs")


def test_preflight_refuses_existing_provenance_journal():
    # No-unsafe-resume guard: a leftover journal means the run is not fresh.
    inputs = _passing_inputs()
    inputs["provenance_jsonl_exists"] = True
    ok, failures = pf.evaluate_preflight(inputs)
    assert not ok
    assert len(failures) == 1 and failures[0].startswith("fresh_journal")


# ── environment fingerprint ───────────────────────────────────────────────────

def test_fingerprint_contains_all_mandatory_fields():
    fp = _fingerprint()
    assert fpmod.missing_fields(fp) == []
    for field in fpmod.REQUIRED_FIELDS:
        assert field in fp and fp[field] not in (None, "", {})
    # git SHA / protocol / digest / execution mode round-trip exactly.
    assert fp["git_sha"] == GOOD_SHA
    assert fp["protocol_id"] == "p4.2.oracle-baseline.v2"
    assert fp["manifest_digest"] == FROZEN_V2
    assert fp["execution_mode"] == "external-full-run"
    # host is anonymized: machine class present, raw hostname never leaks.
    assert fp["host"]["machine_class"]
    assert "runner-01" not in json.dumps(fp)


# ── external-bundle collector ────────────────────────────────────────────────

def test_collector_refuses_missing_provenance_head(tmp_path):
    base = _make_bundle(tmp_path, drop="provenance_head_witness")
    with pytest.raises(co.Refused) as exc:
        co.build_bundle_record(base)
    assert "provenance_head_witness" in str(exc.value)


def test_collector_refuses_missing_sidecar(tmp_path):
    base = _make_bundle(tmp_path, drop="csv_sidecar")
    with pytest.raises(co.Refused) as exc:
        co.build_bundle_record(base)
    assert "csv_sidecar" in str(exc.value)


def test_collector_refuses_missing_csv(tmp_path):
    base = _make_bundle(tmp_path, drop="run_csv")
    with pytest.raises(co.Refused) as exc:
        co.build_bundle_record(base)
    assert "run_csv" in str(exc.value)


def test_collector_hashes_every_required_artifact(tmp_path):
    base = _make_bundle(tmp_path)
    record = co.build_bundle_record(base)
    hashed_roles = {e["role"] for e in record["files"]}
    assert hashed_roles == set(co.HASHED_ROLES)
    for entry in record["files"]:
        assert entry["sha256"].startswith("sha256:") and len(entry["sha256"]) == 71
        assert isinstance(entry["bytes"], int) and entry["bytes"] > 0
    # the hashes manifest never hashes itself.
    assert record["self_excluded"] == co.REQUIRED_ARTIFACTS["file_hashes_manifest"]
    assert "file_hashes_manifest" not in hashed_roles


def test_bundle_cannot_be_marked_complete_without_all_hashes(tmp_path):
    # A genuinely full bundle is COMPLETE...
    full = co.build_bundle_record(_make_bundle(tmp_path))
    assert full["complete"] is True and co.is_complete(full)

    # ...but a record missing any required hash is refused, never COMPLETE.
    partial = {
        "files": [dict(e) for e in full["files"] if e["role"] != "run_log"],
        "self_excluded": full["self_excluded"],
    }
    assert co.is_complete(partial) is False

    base = _make_bundle(tmp_path / "second", drop="run_log")
    with pytest.raises(co.Refused):
        co.build_bundle_record(base)
