"""P4.2b merge and coverage tests.

Proves that:
- two valid arm-shards merge correctly;
- each specific malformation causes refusal;
- individual shard CSV cannot pass scientific gate.

Tests work with synthetic fixtures (tmp CSV + shard manifests) and do NOT
run any simulation.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import pathlib
import sys

import pytest

REPO = Path_file = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import merge_p4_shards as merge_mod  # noqa: E402
import check_p4_shard_coverage as cov_mod  # noqa: E402
import p4_2_protocol as proto  # noqa: E402


ARMS = ["rstdp_mixed", "no_plasticity", "m_zero",
        "weight_shuffled_frozen", "direction_shuffled_frozen", "oracle_reflex"]
SEEDS = list(range(180, 240))
STREAMS = ["left_target", "right_target", "mixed"]
DIGEST = "sha256:" + "f" * 64
GIT_SHA = "a" * 40
ENV_FP = {"python": "3.13.0", "numpy": "2.0.0"}


def _make_shard(shard_id: str, arm_subset: list[str], seeds: list[int] | None = None,
                streams: list[str] | None = None, **overrides) -> dict:
    s = seeds or SEEDS
    st = streams or STREAMS
    return {
        "shard_schema_version": "1.0.0",
        "protocol_id": "p4.2.oracle-baseline.v3",
        "manifest_digest": DIGEST,
        "git_sha": GIT_SHA,
        "shard_id": shard_id,
        "shard_axis": "arms",
        "seed_subset": s,
        "arm_subset": arm_subset,
        "task_streams": st,
        "episodes_per_seed": 80,
        "partial": True,
        "non_confirmatory": False,
        "csv_sha256": "sha256:" + "b" * 64,
        "provenance_jsonl_sha256": "sha256:" + "c" * 64,
        "provenance_head_sha256": "sha256:" + "d" * 64,
        "environment_fingerprint": ENV_FP,
        "started_at_utc": "2026-10-04T12:00:00Z",
        "finished_at_utc": "2026-10-04T12:15:00Z",
        "cells_in_shard": len(arm_subset) * len(s) * len(st),
        **overrides,
    }


@pytest.fixture()
def parent_manifest() -> dict:
    """Minimal parent manifest for coverage checks."""
    return {
        "protocol_id": "p4.2.oracle-baseline.v3",
        "_digest": DIGEST,
        "arms": [{"arm_id": a} for a in ARMS],
        "task_streams": STREAMS,
        "episodes_per_seed": 80,
        "seed_sets": {"confirmatory": {"range": [180, 239], "count": 60}},
    }


def _patch_parent_seeds(monkeypatch, parent):
    """confirmatory_seeds() reads from manifest; patch for synthetic parent."""
    monkeypatch.setattr(proto, "confirmatory_seeds", lambda m: SEEDS)


# --- Coverage: merge success (5 tests) ----------------------------------------

def test_two_valid_arm_shards_cover_all_arms(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    s1 = _make_shard("shard-a", ARMS[:3])
    s2 = _make_shard("shard-b", ARMS[3:])
    problems = cov_mod.check_coverage([s1, s2], parent_manifest)
    assert problems == [], problems


def test_merged_cells_equal_full_experiment(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    s1 = _make_shard("shard-a", ARMS[:3])
    s2 = _make_shard("shard-b", ARMS[3:])
    total = sum(len(s["arm_subset"]) * len(s["seed_subset"]) * len(s["task_streams"])
                for s in [s1, s2])
    assert total == 6 * 60 * 3  # 1080


def test_merge_csvs_concatenate_without_overlap(tmp_path):
    """Merge produces byte-for-byte shard rows."""
    rows_a = [{"arm_id": "rstdp_mixed", "seed": "180", "task": "mixed", "val": "1"}]
    rows_b = [{"arm_id": "oracle_reflex", "seed": "180", "task": "mixed", "val": "2"}]
    csv_a = tmp_path / "shard-a.csv"
    csv_b = tmp_path / "shard-b.csv"
    for path, rows in [(csv_a, rows_a), (csv_b, rows_b)]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["arm_id", "seed", "task", "val"])
            w.writeheader()
            w.writerows(rows)

    shards = [
        _make_shard("shard-a", ARMS[:3], csv_sha256=_file_hash(csv_a)),
        _make_shard("shard-b", ARMS[3:], csv_sha256=_file_hash(csv_b)),
    ]
    shard_paths = [tmp_path / "shard-a.shard_manifest.json",
                   tmp_path / "shard-b.shard_manifest.json"]
    for sp, s in zip(shard_paths, shards):
        sp.write_text(json.dumps(s))

    merged = merge_mod.merge_csvs(shards, shard_paths)
    assert len(merged) == 2
    assert merged[0]["arm_id"] == "rstdp_mixed"
    assert merged[1]["arm_id"] == "oracle_reflex"


def test_merged_manifest_has_correct_constants(parent_manifest, monkeypatch, tmp_path):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    merged_csv = tmp_path / "merged.csv"
    merged_csv.write_text("arm_id,seed,task\n")
    s1 = _make_shard("shard-a", ARMS[:3])
    s2 = _make_shard("shard-b", ARMS[3:])
    m = merge_mod.build_merged_manifest(parent_manifest, [s1, s2], merged_csv, 1080)
    assert m["partial"] is False
    assert m["coverage_verified"] is True
    assert m["merge_status"] == "COMPLETE"
    assert m["shard_count"] == 2
    assert m["total_cells"] == 1080
    assert m["expected_cells"] == 1080


def test_burned_seed_overlap_is_refused(parent_manifest, monkeypatch):
    """Shard using P4.2.v1/v2 seeds must be refused by merge."""
    _patch_parent_seeds(monkeypatch, parent_manifest)
    burned_seeds = list(range(120, 180))  # v2 seeds
    s1 = _make_shard("shard-x", ARMS[:3], seeds=burned_seeds)
    s2 = _make_shard("shard-y", ARMS[3:], seeds=burned_seeds)
    with pytest.raises(merge_mod.MergeRefused, match="burned"):
        merge_mod.check_no_burned_overlap([s1, s2])


# --- Coverage: merge rejection (14 tests) -------------------------------------

def test_overlapping_arm_subsets_fail(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    s1 = _make_shard("shard-a", ARMS[:4])  # overlaps on arm index 3
    s2 = _make_shard("shard-b", ARMS[3:])
    problems = cov_mod.check_coverage([s1, s2], parent_manifest)
    assert any("overlap" in p for p in problems), problems


def test_missing_arm_fails(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    s1 = _make_shard("shard-a", ARMS[:3])
    s2 = _make_shard("shard-b", ARMS[3:5])  # oracle_reflex missing
    problems = cov_mod.check_coverage([s1, s2], parent_manifest)
    assert any("отсутствуют" in p for p in problems), problems


def test_different_manifest_digest_fails():
    s1 = _make_shard("shard-a", ARMS[:3])
    s2 = _make_shard("shard-b", ARMS[3:])
    s2["manifest_digest"] = "sha256:" + "e" * 64
    parent = {"protocol_id": "p4.2.oracle-baseline.v3", "_digest": s1["manifest_digest"],
              "arms": [{"arm_id": a} for a in ARMS], "task_streams": STREAMS,
              "episodes_per_seed": 80,
              "seed_sets": {"confirmatory": {"range": [180, 239], "count": 60}}}
    with pytest.raises(merge_mod.MergeRefused, match="manifest_digest"):
        merge_mod.validate_shard_identity([s1, s2], parent)


def test_different_git_sha_fails():
    s1 = _make_shard("shard-a", ARMS[:3])
    s2 = _make_shard("shard-b", ARMS[3:])
    s2["git_sha"] = "b" * 40
    parent = {"protocol_id": "p4.2.oracle-baseline.v3", "_digest": DIGEST,
              "arms": [{"arm_id": a} for a in ARMS]}
    with pytest.raises(merge_mod.MergeRefused, match="git_sha"):
        merge_mod.validate_shard_identity([s1, s2], parent)


def test_different_environment_fingerprint_fails():
    s1 = _make_shard("shard-a", ARMS[:3])
    s2 = _make_shard("shard-b", ARMS[3:])
    s2["environment_fingerprint"] = {"python": "3.12.0", "numpy": "2.0.0"}
    parent = {"protocol_id": "p4.2.oracle-baseline.v3", "_digest": DIGEST,
              "arms": [{"arm_id": a} for a in ARMS]}
    with pytest.raises(merge_mod.MergeRefused, match="environment_fingerprint"):
        merge_mod.validate_shard_identity([s1, s2], parent)


def test_duplicate_csv_cell_fails(tmp_path):
    csv_file = tmp_path / "shard-a.csv"
    rows = [{"arm_id": "rstdp_mixed", "seed": "180", "task": "mixed"}]
    with open(csv_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm_id", "seed", "task"])
        w.writeheader()
        w.writerows(rows + rows)  # duplicate

    s1 = _make_shard("shard-a", ARMS[:3])
    shard_path = tmp_path / "shard-a.shard_manifest.json"
    shard_path.write_text(json.dumps(s1))
    with pytest.raises(merge_mod.MergeRefused, match="дубликат"):
        merge_mod.merge_csvs([s1], [shard_path])


def test_missing_seed_in_shard_fails(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    partial_seeds = SEEDS[:50]  # only 50 of 60
    s1 = _make_shard("shard-a", ARMS[:3], seeds=partial_seeds)
    s2 = _make_shard("shard-b", ARMS[3:], seeds=partial_seeds)
    problems = cov_mod.check_coverage([s1, s2], parent_manifest)
    assert any("seeds missing" in p for p in problems), problems


def test_missing_stream_fails(parent_manifest, monkeypatch):
    _patch_parent_seeds(monkeypatch, parent_manifest)
    s1 = _make_shard("shard-a", ARMS[:3], streams=["left_target", "right_target"])
    s2 = _make_shard("shard-b", ARMS[3:])
    problems = cov_mod.check_coverage([s1, s2], parent_manifest)
    assert any("task_streams" in p for p in problems), problems


def test_bad_csv_sha_fails(tmp_path):
    s1 = _make_shard("shard-a", ARMS[:3], csv_sha256="sha256:" + "0" * 64)
    csv_file = tmp_path / "shard-a.csv"
    csv_file.write_text("arm_id,seed,task\n")
    head_file = tmp_path / "shard-a.prov.jsonl.head.json"
    head_file.write_text("{}")
    prov_file = tmp_path / "shard-a.prov.jsonl"
    prov_file.write_text("{}\n")
    sp = tmp_path / "shard-a.shard_manifest.json"
    sp.write_text(json.dumps(s1))
    with pytest.raises(merge_mod.MergeRefused, match="CSV hash"):
        merge_mod.validate_shard_hashes([s1], [sp])


def test_bad_provenance_jsonl_sha_fails(tmp_path):
    s1 = _make_shard("shard-a", ARMS[:3], provenance_jsonl_sha256="sha256:" + "0" * 64)
    csv_file = tmp_path / "shard-a.csv"
    csv_file.write_text("arm_id,seed,task\n")
    s1["csv_sha256"] = _file_hash(csv_file)
    prov_file = tmp_path / "shard-a.prov.jsonl"
    prov_file.write_text("{}\n")
    head_file = tmp_path / "shard-a.prov.jsonl.head.json"
    head_file.write_text("{}")
    s1["provenance_head_sha256"] = _file_hash(head_file)
    sp = tmp_path / "shard-a.shard_manifest.json"
    sp.write_text(json.dumps(s1))
    with pytest.raises(merge_mod.MergeRefused, match="provenance JSONL hash"):
        merge_mod.validate_shard_hashes([s1], [sp])


def test_bad_head_witness_sha_fails(tmp_path):
    s1 = _make_shard("shard-a", ARMS[:3], provenance_head_sha256="sha256:" + "0" * 64)
    csv_file = tmp_path / "shard-a.csv"
    csv_file.write_text("arm_id,seed,task\n")
    s1["csv_sha256"] = _file_hash(csv_file)
    prov_file = tmp_path / "shard-a.prov.jsonl"
    prov_file.write_text("{}\n")
    s1["provenance_jsonl_sha256"] = _file_hash(prov_file)
    head_file = tmp_path / "shard-a.prov.jsonl.head.json"
    head_file.write_text("{}")
    sp = tmp_path / "shard-a.shard_manifest.json"
    sp.write_text(json.dumps(s1))
    with pytest.raises(merge_mod.MergeRefused, match="provenance head hash"):
        merge_mod.validate_shard_hashes([s1], [sp])


def test_non_confirmatory_shard_identity_fails():
    s1 = _make_shard("shard-a", ARMS[:3], non_confirmatory=True)
    s2 = _make_shard("shard-b", ARMS[3:])
    parent = {"protocol_id": "p4.2.oracle-baseline.v3", "_digest": DIGEST,
              "arms": [{"arm_id": a} for a in ARMS]}
    with pytest.raises(merge_mod.MergeRefused, match="non_confirmatory"):
        merge_mod.validate_shard_identity([s1, s2], parent)


def test_failed_provenance_verification_fails(tmp_path):
    """(#12) Цепь provenance обязана сходиться сама по себе, даже если хеши
    в манифесте честные: манифест ничего не обещал о сломанной цепи, но журнал
    противоречит собственному witness.

    Сначала собираем валидный журнал писателем, потом портим payload одной строки,
    не пересчитывая entry_hash — verify_chain обязан вернуть False, merge — отказаться.
    """
    from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog

    prov = tmp_path / "shard-a.prov.jsonl"
    log = ProvenanceLog(str(prov))
    log.append({"kind": "start"})
    log.append({"kind": "cell"})

    s1 = _make_shard("shard-a", ARMS[:3])
    sp = tmp_path / "shard-a.shard_manifest.json"
    sp.write_text(json.dumps(s1))
    # валидная цепь не вызывает отказа
    assert merge_mod.verify_shard_provenance([s1], [sp]) is None

    # ломаем цепь: payload первой записи меняется, её хеши остаются старыми
    lines = prov.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["payload"] = {"kind": "tampered"}
    lines[0] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    prov.write_text("\n".join(lines) + "\n")
    with pytest.raises(merge_mod.MergeRefused, match="chain verification failed"):
        merge_mod.verify_shard_provenance([s1], [sp])


def test_aborted_protocol_result_fails():
    """(#13/4) Shard от закрытого aborted-протокола (P4.2.v1) не merge-ится:
    статус живёт в реестре кода, а не в манифесте shard."""
    s1 = _make_shard("shard-a", ARMS[:3], protocol_id="p4.2.oracle-baseline.v1")
    s2 = _make_shard("shard-b", ARMS[3:])
    with pytest.raises(merge_mod.MergeRefused, match="aborted"):
        merge_mod.check_not_aborted([s1, s2])


def test_individual_shard_cannot_pass_scientific_gate(tmp_path):
    """(#14) Научный гейт отвергает sidecar отдельного shard (partial=true) и
    принимает только merged результат. Проверка идёт настоящим кодом гейта
    check_p4_2_oracle_gate.check_sidecars, а не репликой."""
    import check_p4_2_oracle_gate as gate

    digest = proto.FROZEN_PROTOCOL_DIGEST
    csv_file = tmp_path / "shard-a.csv"
    csv_file.write_text("arm,seed,stream,success_rate\n")
    sidecar = gate.sidecar_path_for(csv_file)
    sidecar.write_text(json.dumps({
        "manifest_digest": digest, "partial": True, "non_confirmatory": False}))
    # отдельный shard: partial=true -> гейт отказывает
    with pytest.raises(gate.Refused, match="partial"):
        gate.check_sidecars([str(csv_file)], [str(sidecar)], digest)
    # тот же CSV после merge (partial=false) проходит эту проверку sidecar'ов
    sidecar.write_text(json.dumps({
        "manifest_digest": digest, "partial": False, "non_confirmatory": False}))
    sides = gate.check_sidecars([str(csv_file)], [str(sidecar)], digest)
    assert len(sides) == 1


# --- Helper ---

def _file_hash(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return f"sha256:{h.hexdigest()}"
