"""P4.2c merged-gate adapter tests.

The scientific gate may only render a verdict on a COMPLETE merged experiment.
This proves the adapter (check_p4_2_oracle_gate.build_merged_sidecar /
.verify_merged_sidecar) accepts a genuine merged bundle whose whole evidence
chain re-hashes, and refuses each way that chain can be forged:

  1. an individual (partial) shard CSV;
  2. a merged CSV with no merged sidecar;
  3. a missing merged result manifest;
  4. a merged result manifest whose SHA does not match the sidecar anchor;
  5. coverage_verified != true;
  6. merge_status != COMPLETE;
  7. source_shard_count that disagrees with the manifest;
  8. a source shard whose on-disk bytes no longer match its recorded hash;
  +  merged CSV hash mismatch, and merged manifest bytes tampered.

No simulation runs; everything is synthetic fixtures under tmp_path.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import merge_p4_shards as merge_mod  # noqa: E402
import check_p4_2_oracle_gate as gate  # noqa: E402
import p4_2_protocol as proto  # noqa: E402

ARMS = ["rstdp_mixed", "no_plasticity", "m_zero",
        "weight_shuffled_frozen", "direction_shuffled_frozen", "oracle_reflex"]
STREAMS = ["left_target", "right_target", "mixed"]
DIGEST = "sha256:" + "f" * 64
GIT_SHA = "a" * 40
ENV_FP = {"python": "3.13.0", "numpy": "2.0.0"}


def _file_hash(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return f"sha256:{h.hexdigest()}"


def _seed_shard(sid: str, seeds: list[int]) -> dict:
    return {
        "shard_schema_version": "1.0.0",
        "protocol_id": "p4.2.oracle-baseline.v3",
        "manifest_digest": DIGEST,
        "git_sha": GIT_SHA,
        "shard_id": sid,
        "shard_axis": "seeds",
        "seed_subset": seeds,
        "arm_subset": list(ARMS),
        "task_streams": list(STREAMS),
        "episodes_per_seed": 80,
        "partial": True,
        "non_confirmatory": False,
        "csv_sha256": "sha256:" + "0" * 64,
        "provenance_jsonl_sha256": "sha256:" + "0" * 64,
        "provenance_head_sha256": "sha256:" + "0" * 64,
        "environment_fingerprint": ENV_FP,
        "started_at_utc": "2026-10-04T12:00:00Z",
        "finished_at_utc": "2026-10-04T12:15:00Z",
        "cells_in_shard": len(ARMS) * len(seeds) * len(STREAMS),
    }


def _write_shard(tmp_path, shard):
    sid = shard["shard_id"]
    d = tmp_path / sid
    d.mkdir(parents=True, exist_ok=True)
    csvf = d / f"{sid}.csv"
    rows = [{"arm_id": a, "seed": str(sd), "task": st}
            for a in shard["arm_subset"] for sd in shard["seed_subset"] for st in shard["task_streams"]]
    with open(csvf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm_id", "seed", "task"])
        w.writeheader()
        w.writerows(rows)
    provf = d / f"{sid}.prov.jsonl"
    provf.write_text('{"kind":"start"}\n')
    headf = d / f"{sid}.prov.jsonl.head.json"
    headf.write_text('{"last_hash":"x"}')
    shard["csv_sha256"] = _file_hash(csvf)
    shard["provenance_jsonl_sha256"] = _file_hash(provf)
    shard["provenance_head_sha256"] = _file_hash(headf)
    sp = d / f"{sid}.shard_manifest.json"
    sp.write_text(json.dumps(shard))
    return shard, sp


def _bundle(tmp_path):
    """Build a genuine merged bundle (2 seed shards) with a real sidecar.

    Returns dict of the artifacts under test. merged CSV bytes exist BEFORE the
    manifest is built, so every recorded hash matches the disk as the gate reads it.
    """
    parent = {"protocol_id": "p4.2.oracle-baseline.v3", "_digest": DIGEST,
              "arms": [{"arm_id": a} for a in ARMS], "task_streams": STREAMS,
              "episodes_per_seed": 80}
    shards = [_seed_shard("shard-a", list(range(180, 190))),
              _seed_shard("shard-b", list(range(190, 200)))]
    written = [_write_shard(tmp_path, s) for s in shards]
    shard_paths = [sp for _, sp in written]

    out_dir = tmp_path / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = out_dir / "p4_2_merged.csv"
    rows = []
    for s, _ in written:
        with open(out_dir.parent / s["shard_id"] / f"{s['shard_id']}.csv", newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    with open(merged_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm_id", "seed", "task"])
        w.writeheader()
        w.writerows(rows)

    manifest = merge_mod.build_merged_manifest(parent, [s for s, _ in written],
                                               merged_csv, len(rows), shard_paths)
    manifest_path = out_dir / "p4_2_merged_result_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")

    sidecar = gate.build_merged_sidecar(manifest_path)
    sidecar_path = gate.sidecar_path_for(merged_csv)
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return {
        "merged_csv": merged_csv, "sidecar": sidecar, "sidecar_path": sidecar_path,
        "manifest": manifest, "manifest_path": manifest_path,
        "shards": [s for s, _ in written], "shard_paths": shard_paths, "rows": len(rows),
    }


# --- Positive: a genuine merged bundle passes the adapter ----------------------

def test_genuine_merged_bundle_passes(tmp_path):
    b = _bundle(tmp_path)
    report = gate.verify_merged_sidecar(b["sidecar"], str(b["merged_csv"]))
    assert report["merged"] is True
    assert report["merge_axis"] == "seeds"
    assert report["source_shard_count"] == 2
    assert b["rows"] == 6 * 20 * 3  # all cells present


def test_non_merged_sidecar_is_left_alone(tmp_path):
    """A single-process (non-sharded) run sidecar has no merge fields; the adapter
    returns {'merged': False} so the existing full-run path keeps working."""
    sidecar = {"manifest_digest": DIGEST, "partial": False, "non_confirmatory": False}
    assert gate.verify_merged_sidecar(sidecar, "whatever.csv") == {"merged": False}


# --- (1) individual partial shard + (2) missing merged sidecar -----------------

def test_individual_partial_shard_fails_gate(tmp_path):
    digest = proto.FROZEN_PROTOCOL_DIGEST
    csv_file = tmp_path / "shard-a.csv"
    csv_file.write_text("arm,seed,task,success_rate\n")
    sidecar = gate.sidecar_path_for(csv_file)
    sidecar.write_text(json.dumps({"manifest_digest": digest, "partial": True,
                                   "non_confirmatory": False}))
    with pytest.raises(gate.Refused, match="partial"):
        gate.check_sidecars([str(csv_file)], [str(sidecar)], digest)


def test_merged_csv_without_sidecar_fails_gate(tmp_path):
    digest = proto.FROZEN_PROTOCOL_DIGEST
    csv_file = tmp_path / "p4_2_merged.csv"
    csv_file.write_text("arm,seed,task,success_rate\n")
    # sidecar list passed WITHOUT the merged csv's expected sidecar
    with pytest.raises(gate.Refused, match="sidecar"):
        gate.check_sidecars([str(csv_file)], [], digest)


# --- (3) missing merged result manifest ----------------------------------------

def test_missing_merged_manifest_fails(tmp_path):
    b = _bundle(tmp_path)
    b["manifest_path"].unlink()
    with pytest.raises(gate.Refused, match="merged result manifest"):
        gate.verify_merged_sidecar(b["sidecar"], str(b["merged_csv"]))


# --- (4) forged merged sidecar: manifest SHA anchor does not match -------------

def test_forged_merged_sidecar_fails(tmp_path):
    b = _bundle(tmp_path)
    forged = dict(b["sidecar"])
    forged["merged_result_manifest_sha256"] = "sha256:" + "1" * 64
    with pytest.raises(gate.Refused, match="SHA mismatch"):
        gate.verify_merged_sidecar(forged, str(b["merged_csv"]))


# --- (5) coverage_verified != true ---------------------------------------------

def test_coverage_not_verified_fails(tmp_path):
    b = _bundle(tmp_path)
    forged = dict(b["sidecar"])
    forged["coverage_verified"] = False
    with pytest.raises(gate.Refused, match="coverage_verified"):
        gate.verify_merged_sidecar(forged, str(b["merged_csv"]))


# --- (6) merge_status != COMPLETE ------------------------------------------------

def test_incomplete_merge_status_fails(tmp_path):
    b = _bundle(tmp_path)
    forged = dict(b["sidecar"])
    forged["merge_status"] = "PARTIAL"
    with pytest.raises(gate.Refused, match="merge_status"):
        gate.verify_merged_sidecar(forged, str(b["merged_csv"]))


# --- (7) source_shard_count mismatch --------------------------------------------

def test_source_shard_count_mismatch_fails(tmp_path):
    b = _bundle(tmp_path)
    forged = dict(b["sidecar"])
    forged["source_shard_count"] = 99
    with pytest.raises(gate.Refused, match="source_shard_count"):
        gate.verify_merged_sidecar(forged, str(b["merged_csv"]))


# --- (8) source shard hash mismatch (shard file swapped after merge) ------------

def test_source_shard_hash_mismatch_fails(tmp_path):
    b = _bundle(tmp_path)
    first_shard_csv = b["shard_paths"][0].parent / f"{b['shards'][0]['shard_id']}.csv"
    first_shard_csv.write_text("arm_id,seed,task\nrstdp_mixed,180,mixed\n")  # tampered bytes
    with pytest.raises(gate.Refused, match="source shard hash mismatch"):
        gate.verify_merged_sidecar(b["sidecar"], str(b["merged_csv"]))


# --- (15) merged CSV hash mismatch ----------------------------------------------

def test_merged_csv_hash_mismatch_fails(tmp_path):
    b = _bundle(tmp_path)
    with open(b["merged_csv"], "a") as f:
        f.write("rstdp_mixed,999,mixed\n")  # extra row changes the bytes
    with pytest.raises(gate.Refused, match="merged CSV hash mismatch"):
        gate.verify_merged_sidecar(b["sidecar"], str(b["merged_csv"]))


# --- (16) merged manifest bytes tampered (re-hash no longer matches sidecar) ----

def test_merged_manifest_hash_mismatch_fails(tmp_path):
    b = _bundle(tmp_path)
    tampered = dict(b["manifest"])
    tampered["shard_count"] = 42  # edit on disk, sidecar anchor stays put
    b["manifest_path"].write_text(json.dumps(tampered, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    with pytest.raises(gate.Refused, match="SHA mismatch"):
        gate.verify_merged_sidecar(b["sidecar"], str(b["merged_csv"]))
