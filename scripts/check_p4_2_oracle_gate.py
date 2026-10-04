#!/usr/bin/env python
"""P4.2 — гейт сравнения R-STDP против oracle по замороженному протоколу.

Порядок проверки не случаен: сначала «имеем ли мы право вообще считать», потом
статистика. Прогон, покрывающий 59 из 60 объявленных seed, не «почти полный», а
неприменимый: нехватка одного seed меняет и power, и смысл paired-теста.

Разделение исходов тоже принципиально:
  exit 2 — считать отказались (протокол изменён, матрица неполная, инвариант
           контроль нарушен). Это не «проверка не прошла», это «проверки не было»;
  exit 1 — посчитано честно, required-гипотезы не прошли;
  exit 0 — посчитано честно, required-гипотезы прошли.

Oracle сравнивается, но гейтом не является: его роль — показать величину эффекта
относительно фиксированной проводки, которая уже знает оба ответа. Проигрыш ему
заносится в отчёт как результат, а не как повод донастроить learning rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

import p4_2_protocol as proto  # noqa: E402
import run_p4_validation as p4  # noqa: E402

REFUSE = 2
MIXED_STREAM = "mixed"   # code name of the stream the primary metric is defined on


class Refused(Exception):
    """Право считать отсутствует: прогон не соответствует замороженному протоколу."""


def sidecar_path_for(csv_path: pathlib.Path) -> pathlib.Path:
    """Runner пишет sidecar рядом с CSV: <csv>.meta.json.

    Связь по имени файла, а не по порядку в аргументах: переставленные шарды не
    должны иметь возможности проверить sidecar не того прогона.
    """
    return csv_path.with_suffix(csv_path.suffix + ".meta.json")


def check_sidecars(csv_paths: list[str], sidecar_paths: list[str], digest: str) -> list[dict]:
    """Каждый shard обязан иметь свой sidecar, с замороженным digest и partial=false.

    Первым делом проверяется якорь: если гейт вызван по манифесту, которого нет
    в замороженной pre-registration, сравнивать sidecar с его digest бессмысленно —
    подделка совпадёт сама с собой.
    """
    if digest != proto.FROZEN_PROTOCOL_DIGEST:
        raise Refused(
            f"гейт идёт не по замороженному протоколу: digest {digest} против "
            f"{proto.FROZEN_PROTOCOL_DIGEST}. Проверять прогон по правленному протоколу "
            "нельзя, даже если sidecar с ним согласован.")
    want = {sidecar_path_for(pathlib.Path(c)) for c in csv_paths}
    got = {pathlib.Path(x) for x in sidecar_paths}
    if want - got:
        raise Refused(
            "не хватает sidecar для shard: "
            f"{sorted(str(w) for w in want - got)} — без него нельзя доказать, что "
            "эти строки сделаны по текущему протоколу, а не по правленному после запуска")
    if got - want:
        raise Refused(f"sidecar без своего CSV: {sorted(str(g) for g in got - want)}")
    sides: list[tuple[str, dict]] = []
    for path in sorted(got):
        if not path.exists():
            # Названный sidecar мог пройти сверку имён (она сравнивает множества), а
            # потом исчезнуть: раньше это было FileNotFoundError с трейсбеком и rc=1,
            # то есть «проверки не было» выглядело как «прогон не прошёл».
            raise Refused(
                f"названный sidecar отсутствует на диске: {path} — проверка целостности "
                "не выполнена. Это отказ (rc=2), а не научный результат:"
                " отсутствие файла ничего не утверждает о прогоне")
        side = json.loads(path.read_text(encoding="utf-8"))
        if side.get("manifest_digest") != digest:
            raise Refused(f"прогон сделан по другому протоколу: sidecar "
                          f"{side.get('manifest_digest')} против текущего {digest}")
        if side.get("partial"):
            raise Refused(f"{path.name} помечен partial (подрезка seed/arms) — "
                          "полный гейт по нему не считается")
        if side.get("non_confirmatory"):
            raise Refused(f"{path.name} помечен non_confirmatory (прогон по не-замороженному "
                          "манифесту) — confirmatory-вердикт по нему не выносится")
        sides.append((path.name, side))
    return sides


# --- Merged-gate adapter (P4.2c) ----------------------------------------------
#
# Научный вердикт имеет право считаться только по полному эксперименту. При
# seed-axis/arms-axis шардинге полный эксперимент — это merged-CSV, к которому
# приложен merged-манифест, а тот по ссылкам опирается на исходные shard-файлы.
# Adapter разворачивает эту цепочку доверия в одну перепроверяемую связку:
#   sidecar(merged) -> merged_result_manifest.json -> merged CSV -> каждый
#   shard-манифест -> CSV/provenance/head каждого shard. Любой разрыв — отказ
#   (rc=2, «проверки не было»), а не «прогон не прошёл».

MERGED_MANIFEST_NAME = "p4_2_merged_result_manifest.json"
MERGED_SIDECAR_FIELDS = ("merged_result_manifest_sha256", "merge_status",
                         "source_shard_count")


def sha256_file(path) -> str:
    """SHA-256 файла на диске в форме «sha256:<hex>» — тот же формат, что в манифестах."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def build_merged_sidecar(merged_manifest_path) -> dict:
    """Собирает обычный final-sidecar для merged-CSV по merged-манифесту.

    Sidecar — это то, что гейт читает первым (рядом с CSV как <csv>.meta.json),
    поэтому в него кладём якоря, которые гейт затем пересчитывает: digest
    протокола, git_sha, путь к merged-манифесту и его хеш, хеш merged-CSV и
    количество shard'ов. partial обязан быть ложным — это и есть отличие
    «полного» результата от отдельного shard.
    """
    mm = pathlib.Path(merged_manifest_path)
    manifest = json.loads(mm.read_text(encoding="utf-8"))
    merged_csv = pathlib.Path(manifest["merged_csv_path"])
    if not merged_csv.exists():
        raise Refused(f"merged-манифест ссылается на CSV, которого нет: {merged_csv}")
    if manifest.get("partial") is not False:
        raise Refused("merged-манифест помечен partial — sidecar полного прогона не строится")
    if manifest.get("coverage_verified") is not True:
        raise Refused("merged-манифест без coverage_verified=true не имеет права на sidecar")
    if manifest.get("merge_status") != "COMPLETE":
        raise Refused("merged-манифест со статусом != COMPLETE не становится вердиктом")
    return {
        "protocol_id": manifest["protocol_id"],
        "manifest_digest": manifest["manifest_digest"],
        "git_sha": manifest["git_sha"],
        "partial": False,
        "non_confirmatory": False,
        "coverage_verified": True,
        "merge_status": manifest["merge_status"],
        "merge_axis": manifest.get("merge_axis"),
        "merged_result_manifest": str(mm),
        "merged_result_manifest_sha256": sha256_file(mm),
        "merged_csv_sha256": sha256_file(merged_csv),
        "source_shard_count": int(manifest["shard_count"]),
        "shard_ids": list(manifest["shard_ids"]),
    }


def verify_merged_sidecar(sidecar: dict, merged_csv_path) -> dict:
    """Строгая сверка merged-результата. Срабатывает только если sidecar несёт
    merge-поля; для одиночного (не-шардованного) прогона возвращает {'merged': False}.

    Проверка пересобирает evidence с диска, поэтому «merge» нельзя заявить,
    подделав один sidecar или один манифест: нужны согласованные merged-манифест +
    merged-CSV + исходные shard-файлы, и каждый записанный digest обязан
    совпасть с пересчитанным.
    """
    if not all(k in sidecar for k in MERGED_SIDECAR_FIELDS):
        return {"merged": False}

    # 1. Константы самого sidecar: partial/non_confirmatory/coverage/status.
    if sidecar.get("partial"):
        raise Refused("merged sidecar помечен partial — отдельный shard вердиктом не считается")
    if sidecar.get("non_confirmatory"):
        raise Refused("merged sidecar помечен non_confirmatory — confirmatory-вердикт не выносится")
    if sidecar.get("coverage_verified") is not True:
        raise Refused("merged sidecar: coverage_verified != true — полное покрытие не заявлено")
    if sidecar.get("merge_status") != "COMPLETE":
        raise Refused(f"merged sidecar: merge_status {sidecar.get('merge_status')!r} != COMPLETE")

    # 2. Merged-манифест: найти, перехешировать, сверить с якорем sidecar.
    mm_path = sidecar.get("merged_result_manifest")
    if mm_path:
        mm_path = pathlib.Path(mm_path)
    else:
        mm_path = pathlib.Path(merged_csv_path).with_name(MERGED_MANIFEST_NAME)
    if not mm_path.exists():
        raise Refused(f"merged result manifest отсутствует: {mm_path} — merged-вердикту нечем подтверждаться")
    mm_sha = sha256_file(mm_path)
    if mm_sha != sidecar.get("merged_result_manifest_sha256"):
        raise Refused(
            f"merged result manifest SHA mismatch: записан {sidecar.get('merged_result_manifest_sha256')}"
            f"… , на диске {mm_sha}")

    manifest = json.loads(mm_path.read_text(encoding="utf-8"))
    if manifest.get("merge_status") != "COMPLETE":
        raise Refused("merged result manifest: merge_status != COMPLETE")
    if manifest.get("coverage_verified") is not True:
        raise Refused("merged result manifest: coverage_verified != true")
    if manifest.get("partial") is not False:
        raise Refused("merged result manifest: partial != false")
    if manifest.get("manifest_digest") != sidecar.get("manifest_digest"):
        raise Refused("merged result manifest: manifest_digest разошёлся со sidecar")
    if manifest.get("git_sha") != sidecar.get("git_sha"):
        raise Refused("merged result manifest: git_sha разошёлся со sidecar")

    # 3. source_shard_count обязан сойтись трижды: sidecar ↔ shard_count ↔ факт-список.
    src = manifest.get("source_shards", [])
    if int(sidecar["source_shard_count"]) != int(manifest.get("shard_count", -1)) \
            or int(manifest.get("shard_count", -1)) != len(src):
        raise Refused(
            f"source_shard_count mismatch: sidecar={sidecar['source_shard_count']}, "
            f"manifest.shard_count={manifest.get('shard_count')}, source_shards={len(src)}")

    # 4. Merged CSV: то, что гейт реально читает, должно совпасть со sidecar И с манифестом.
    csv_sha = sha256_file(merged_csv_path)
    if csv_sha != sidecar.get("merged_csv_sha256"):
        raise Refused("merged CSV hash mismatch со sidecar")
    if csv_sha != manifest.get("merged_csv_sha256"):
        raise Refused("merged CSV hash mismatch с merged-манифестом")

    # 5. Каждое исходное shard-evidence пересчитывается с диска. Это тот случай,
    # когда shard-файл подменили уже после merge: манифест цел, sidecar цел, а
    # запись про chunk больше не подтверждается его собственными байтами.
    for entry in src:
        smp = entry.get("shard_manifest_path")
        if not smp or not pathlib.Path(smp).exists():
            raise Refused(f"source shard {entry.get('shard_id')}: shard manifest не найден ({smp})")
        d = pathlib.Path(smp).parent
        sid = entry["shard_id"]
        for fname, key, label in [(f"{sid}.csv", "csv_sha256", "shard CSV"),
                                  (f"{sid}.prov.jsonl", "provenance_jsonl_sha256", "shard provenance"),
                                  (f"{sid}.prov.jsonl.head.json", "provenance_head_sha256", "shard head")]:
            fp = d / fname
            if not fp.exists():
                raise Refused(f"source shard {sid}: файл не найден: {fname}")
            if sha256_file(fp) != entry.get(key):
                raise Refused(f"source shard hash mismatch: {sid} {label}")
        # subset/identities shard-манифеста должны совпасть с тем, что записал merge
        sm = json.loads(pathlib.Path(smp).read_text(encoding="utf-8"))
        if sorted(sm["seed_subset"]) != sorted(entry["seed_subset"]) \
                or sorted(sm["arm_subset"]) != sorted(entry["arm_subset"]) \
                or sm["manifest_digest"] != entry["manifest_digest"]:
            raise Refused(f"source shard {sid}: identity/subset разошлись с merged-манифестом")

    return {
        "merged": True,
        "merge_axis": manifest.get("merge_axis"),
        "source_shard_count": len(src),
        "merged_csv": str(merged_csv_path),
    }


def load_rows(paths: list[str], codes: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        src = pathlib.Path(path)
        if not src.exists():
            # Отказ стартовать, а не падение стека: «проверки не было» обязано
            # читаться как exit 2, иначе трейсбек неотличим от «не прошло».
            raise Refused(f"{path}: файла прогона нет на диске")
        raw = p4.read_csv(src)
        if not raw:
            raise Refused(f"{path}: пустой CSV")
        rows.extend(raw)
    known = set(codes.values())
    unknown = sorted({r["arm"] for r in rows} - known)
    if unknown:
        raise Refused(f"в CSV есть arms вне протокола: {unknown}")
    return rows


def coverage_gate(rows: list[dict], manifest: dict, codes: dict[str, str], streams: list[str]) -> None:
    declared = set(proto.confirmatory_seeds(manifest))
    seen = {int(r["seed"]) for r in rows}
    if seen - declared:
        raise Refused(f"в прогоне есть seed вне множества {sorted(declared)[:3]}…: "
                      f"{sorted(seen - declared)[:8]}")
    missing = declared - seen
    if missing:
        raise Refused(f"прогон неполный: не хватает {len(missing)} seed из {len(declared)} "
                      f"(например {sorted(missing)[:8]}). Гейт на укороченном множестве — это "
                      f"не тот тест, который был объявлен.")

    arm_ids = {a["arm_id"] for a in manifest["arms"]}
    want = {(codes[arm_id], stream, seed) for arm_id in arm_ids for stream in streams for seed in declared}
    got = {(r["arm"], r["task"], int(r["seed"])) for r in rows}
    if want - got:
        raise Refused(f"не хватает {len(want - got)} ячеек (arm x stream x seed), например "
                      f"{sorted(want - got)[:4]}")
    if got - want:
        raise Refused(f"лишние ячейки вне протокола: {sorted(got - want)[:4]}")
    if len(rows) != len(want):
        raise Refused(f"дубликаты ячеек: строк {len(rows)}, ожидалось {len(want)}")


def disjointness_gate(manifest: dict, prior_csvs: list[str]) -> list[str]:
    """Отчёт о несечении seed: объявленные множества + фактические seed из чужих CSV."""
    declared = set(proto.confirmatory_seeds(manifest))
    notes: list[str] = []
    for other in manifest["seed_sets"]["disjoint_from"]:
        seeds = set(proto.expand_seed_range(other, other["id"]))
        overlap = sorted(declared & seeds)
        notes.append(f"{other['id']}: n={len(seeds)}, пересечение с confirmatory = {len(overlap)}")
        if overlap:
            raise Refused(f"confirmatory пересекается с {other['id']}: {overlap[:8]}")
    for path in prior_csvs:
        p = pathlib.Path(path)
        if not p.exists():
            notes.append(f"{p.name}: файла нет на диске — сверка только по объявленным диапазонам")
            continue
        prior = {int(r["seed"]) for r in p4.read_csv(p)}
        overlap = sorted(declared & prior)
        notes.append(f"{p.name}: seed там {len(prior)}, пересечение = {len(overlap)}")
        if overlap:
            raise Refused(f"{p.name} уже использовал эти seed: {overlap[:8]}")
    return notes


def vector_by_seed(rows: list[dict], code_arm: str, stream: str, column: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for r in rows:
        if r["arm"] == code_arm and r["task"] == stream:
            out[int(r["seed"])] = float(r[column])
    return out


def sign_comparison(a: dict[int, float], b: dict[int, float]) -> dict:
    seeds = sorted(set(a) & set(b))
    xs = [a[s] for s in seeds]
    ys = [b[s] for s in seeds]
    wins, losses, p = p4.paired_sign_test(xs, ys)
    ties = len(seeds) - wins - losses
    return {
        "n_seeds_paired": len(seeds),
        "wins": wins, "losses": losses, "ties": ties,
        "n_informative": wins + losses,
        "raw_p": p,
        "mean_treatment": float(np.mean(xs)) if xs else float("nan"),
        "mean_comparator": float(np.mean(ys)) if ys else float("nan"),
        "mean_difference": float(np.mean([x - y for x, y in zip(xs, ys)])) if xs else float("nan"),
    }


def holm(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        adj[idx] = running
    return adj


def integrity_gates(rows: list[dict], manifest: dict, codes: dict[str, str]) -> list[str]:
    """То, без чего сравнение не значит ничего: контроль действительно заморожен,
    лечение действительно двигает веса, награда действительно не течёт в M=0."""
    notes: list[str] = []

    viol = int(sum(r.get("governance_violations", 0) or 0 for r in rows))
    budget = manifest["provenance"]["violation_budget"]
    if viol > budget:
        raise Refused(f"governance violations = {viol} при бюджете {budget}")
    notes.append(f"governance violations: {viol} (бюджет {budget})")

    for arm in manifest["arms"]:
        if arm["learner"] != "none":
            continue
        dw = [abs(float(r["mean_abs_dw"])) for r in rows if r["arm"] == arm["code_arm"]]
        worst = max(dw) if dw else float("nan")
        if worst > 1e-12:
            raise Refused(f"{arm['arm_id']} объявлен замороженным, но |Δw| доходит до {worst:.3e}: "
                          f"контроль не заморожен, сравнение с ним меряет не то")
        notes.append(f"{arm['arm_id']}: Δw ≡ 0 (max |Δw|={worst:.1e}) — контроль действительно фиксированный")

    treatment = [a for a in manifest["arms"] if a["role"] == "treatment"][0]
    dw_t = [float(r["mean_abs_dw"]) for r in rows if r["arm"] == treatment["code_arm"]]
    if not dw_t or max(dw_t) <= 1e-12:
        raise Refused("у treatment-arms веса не сдвинулись ни на одном seed: сравнивать "
                      "обучение с замороженными не имеет смысла")
    notes.append(f"{treatment['arm_id']}: |Δw| > 0 на {sum(1 for d in dw_t if d > 1e-12)}/{len(dw_t)} ячеек")

    zero = next((a["code_arm"] for a in manifest["arms"] if a["arm_id"] == "m_zero"), None)
    if zero and "no_plasticity" in codes:
        frozen = vector_by_seed(rows, codes["no_plasticity"], MIXED_STREAM, "success_mirror_min")
        mz = vector_by_seed(rows, zero, MIXED_STREAM, "success_mirror_min")
        diff = [abs(frozen[s] - mz[s]) for s in set(frozen) & set(mz)]
        if diff and max(diff) > 1e-12:
            raise Refused(f"m_zero != no_plasticity (max расхождение {max(diff):.4f}): M≡0 ведёт себя "
                          f"не как «нет обучения», а как отдельный режим — H2 теряет смысл")
        notes.append("m_zero ≡ no_plasticity поведенчески (инвариант reward-слепоты соблюдён)")
    return notes


def oracle_symmetry_report(rows: list[dict], manifest: dict, codes: dict[str, str], stream: str) -> dict:
    """Oracle обязан быть зеркально симметричным — иначе это right_bias в костюме."""
    oracle = [a for a in manifest["arms"] if a["role"] == "upper_bound_characterization"][0]
    per_seed = [r for r in rows if r["arm"] == oracle["code_arm"] and r["task"] == stream]
    gaps = [abs(float(r["success_mirror[right]"]) - float(r["success_mirror[left]"])) for r in per_seed]
    return {
        "arm": oracle["arm_id"],
        "n_seeds": len(per_seed),
        "mean_abs_mirror_gap": float(np.mean(gaps)) if gaps else float("nan"),
        "max_abs_mirror_gap": float(np.max(gaps)) if gaps else float("nan"),
        "min_half_mean": float(np.mean([float(r["success_mirror_min"]) for r in per_seed])) if per_seed else float("nan"),
        "both_halves_ge_0_8": sum(
            1 for r in per_seed
            if float(r["success_mirror[right]"]) >= 0.8 and float(r["success_mirror[left]"]) >= 0.8
        ),
    }


def bundle_gate(manifest: dict, *, csv_paths: list[str], per_seed_out: str | None,
                json_out: str | None) -> list[str]:
    """Вердикт обязан лечь в объявленный result_bundle.

    Манифест v2 перечисляет девять артефактов, и три из них (stdout гейта, verdict
    JSON, per-seed исходы) создаёт только этот скрипт. Если --json-out и
    --per-seed-out не переданы или ведут наружу bundle-папки, прогон технически
    посчитан, а собирать результат не из чего: список из девяти файлов превращается
    в пожелание. Полноту bundle проверяет collect_p4_2_bundle.py --check, здесь —
    только куда пишутся артефакты гейта.
    """
    bundle = manifest.get("result_bundle")
    if not bundle:
        return []                      # протоколы до v2 bundle не объявляли
    bdir = (REPO / bundle["directory"]).resolve()
    by_role = {a["role"]: a["filename"] for a in bundle["artifacts"]}
    for role, flag in (("gate_verdict_json", "--json-out"), ("per_seed_outcomes", "--per-seed-out")):
        if not (json_out if flag == "--json-out" else per_seed_out):
            raise Refused(
                f"result_bundle объявляет артефакт {role!r} ({by_role.get(role)}), но {flag} не "
                "передан: вердикта в bundle нет, и девять заявленных файлов не соберутся")
    named = {"--csv": list(csv_paths), "--json-out": [json_out], "--per-seed-out": [per_seed_out]}
    for flag, paths in named.items():
        for raw in paths:
            path = pathlib.Path(raw).resolve()
            if bdir not in path.parents:
                raise Refused(
                    f"{flag} {raw} вне объявленного result_bundle.directory "
                    f"{bundle['directory']}: артефакт снаружи списка манифеста не попадёт в "
                    "bundle и не будет захеширован")
    return [f"артефакты гейта пишутся в {bundle['directory']} "
            f"(verdict={by_role['gate_verdict_json']}, per_seed={by_role['per_seed_outcomes']})"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(proto.DEFAULT_MANIFEST))
    ap.add_argument("--csv", nargs="+", required=True, help="один или несколько shard-CSV (объединяются)")
    ap.add_argument("--sidecar", nargs="+", required=True,
                    help="sidecar каждого shard-CSV (<csv>.meta.json); без него "
                         "проверка протокола не считается сделанной")
    ap.add_argument("--prior-csv", nargs="*", default=[], help="CSV более ранних прогонов — сверка seed")
    ap.add_argument("--per-seed-out", help="куда выгрузить raw per-seed исходы")
    ap.add_argument("--json-out", help="куда выгрузить машиночитаемый вердикт")
    a = ap.parse_args()

    manifest_path = pathlib.Path(a.manifest)
    try:
        manifest = proto.load(manifest_path)
    except proto.ProtocolError as exc:
        print(f"\nГЕЙТ НЕ СЧИТАЕТСЯ: манифест не проходит замороженный протокол: {exc}")
        return REFUSE
    digest_file = proto.digest_of_file(manifest_path)
    codes = proto.code_arms(manifest)
    streams = proto.task_streams(manifest)
    col = proto.PRIMARY_METRIC_TO_COLUMN[manifest["primary_metric"]]
    alpha = manifest["statistical_test"]["alpha"]

    print(proto.describe(manifest))
    print(f"\n  digest файла: {digest_file}")
    print(f"  замороженный протокол: {proto.ACTIVE_PROTOCOL_ID} {proto.FROZEN_PROTOCOL_DIGEST}")
    # Раньше здесь стояло сравнение digest_file с manifest["_digest"] — тавтология:
    # обе величины считаются из одних байт, и подмена манифеста через --manifest
    # проходила бы мимо. Якорь обязан жить вне проверяемого файла.
    try:
        proto.require_frozen(digest_file, where="гейт")
    except proto.ProtocolError as exc:
        print(f"\nГЕЙТ НЕ СЧИТАЕТСЯ: {exc}")
        return REFUSE

    try:
        for name, side in check_sidecars(a.csv, a.sidecar, digest_file):
            print(f"  sidecar {name}: digest совпадает, partial={side.get('partial')}, "
                  f"git={side.get('git', {}).get('rev')} dirty={side.get('git', {}).get('dirty')}")
        # Порядок именно такой: отсутствующий названный sidecar — отказ про сам
        # контракт проверки, он должен дойти до пользователя раньше, чем
        # замечание о том, куда разложен bundle. Оба — rc=2, и ни один артефакт
        # до этих строк не пишется, так что перестановка ничего не ослабляет.
        for note in bundle_gate(manifest, csv_paths=a.csv, per_seed_out=a.per_seed_out,
                                json_out=a.json_out):
            print("  " + note)
        # Merged-evidence: если sidecar несёт merge-поля, полный вердикт обязан
        # опираться на перепроверяемую цепочку sidecar → merged-манифест → merged
        # CSV → исходные shard-файлы. Отдельный shard (partial) уже отвергнут
        # check_sidecars выше; здесь ловится подделка уже «склеенного» результата.
        for csvp in a.csv:
            sc_path = sidecar_path_for(pathlib.Path(csvp))
            sc = json.loads(sc_path.read_text(encoding="utf-8"))
            report = verify_merged_sidecar(sc, csvp)
            if report.get("merged"):
                print(f"  merged evidence: axis={report['merge_axis']}, "
                      f"source_shards={report['source_shard_count']}, CSV={report['merged_csv']}")
        rows = load_rows(a.csv, codes)
        print(f"\nзагружено строк: {len(rows)} из {len(a.csv)} файла(ов)")
        coverage_gate(rows, manifest, codes, streams)
        for note in disjointness_gate(manifest, a.prior_csv or []):
            print("  " + note)
        for note in integrity_gates(rows, manifest, codes):
            print("  " + note)
    except Refused as exc:
        print(f"\nГЕЙТ НЕ СЧИТАЕТСЯ: {exc}")
        return REFUSE

    mixed = MIXED_STREAM if MIXED_STREAM in streams else streams[-1]
    print(f"\n=== Первичная метрика {manifest['primary_metric']} (колонка '{col}'), stream '{mixed}' ===")

    per_seed_rows: list[dict] = []
    results: list[dict] = []
    by_id = {h["id"]: h for h in manifest["hypotheses"]}
    family = manifest["statistical_test"]["family"]

    for hid in sorted(by_id, key=lambda x: int(x[1:])):
        h = by_id[hid]
        t = vector_by_seed(rows, codes[h["treatment"]], mixed, col)
        c = vector_by_seed(rows, codes[h["comparator"]], mixed, col)
        stat = sign_comparison(t, c)
        stat.update({
            "hypothesis": hid, "kind": h["kind"],
            "treatment": h["treatment"], "comparator": h["comparator"],
            "direction": h["direction"], "in_family": hid in family,
            "statement": h["statement"],
        })
        results.append(stat)
        for seed in sorted(set(t) & set(c)):
            per_seed_rows.append({
                "hypothesis": hid, "seed": seed, "stream": mixed,
                "treatment_arm": h["treatment"], "comparator_arm": h["comparator"],
                "treatment_metric": t[seed], "comparator_metric": c[seed],
                "difference": t[seed] - c[seed],
                "outcome": "win" if t[seed] > c[seed] else ("loss" if c[seed] > t[seed] else "tie"),
            })

    fam_p = [r["raw_p"] for r in results if r["in_family"]]
    adj = holm(fam_p)
    k = 0
    for r in results:
        if r["in_family"]:
            r["adjusted_p"] = adj[k]
            k += 1
        else:
            r["adjusted_p"] = None

    print(f"{'H':4s} {'kind':15s} {'сравнение':42s} {'W/L/T':>9s} {'raw p':>8s} {'Holm p':>8s} {'мин-пол.':>18s} итог")
    required_ok = True
    for r in results:
        comparison = f"{r['treatment']} > {r['comparator']}" if r["direction"] == "greater" \
            else f"{r['treatment']} vs {r['comparator']}"
        spread = f"{r['mean_treatment']:.3f} vs {r['mean_comparator']:.3f}"
        adj_txt = f"{r['adjusted_p']:.4f}" if r["adjusted_p"] is not None else "  n/a"
        if r["kind"] == "characterization":
            verdict = "CHARACTERIZATION (порога нет)"
        else:
            better = r["direction"] != "greater" or r["wins"] > r["losses"]
            ok = better and (r["adjusted_p"] is not None and r["adjusted_p"] < alpha)
            verdict = "ПРОЙДЕНА" if ok else ("НЕ ПРОЙДЕНА" if r["kind"] == "required" else "не пройдена (desirable)")
            if r["kind"] == "required":
                required_ok = required_ok and ok
        print(f"{r['hypothesis']:4s} {r['kind']:15s} {comparison:42s} "
              f"{r['wins']:2d}/{r['losses']:2d}/{r['ties']:2d} {r['raw_p']:8.4f} {adj_txt:>8s} {spread:>18s} {verdict}")

    print(f"\n  ties исключены из знаменателя (политика '{manifest['statistical_test']['tie_handling']}'); "
          f"n_informative указан как W+L")
    for r in results:
        if r["ties"]:
            print(f"    {r['hypothesis']}: ties = {r['ties']} из {r['n_seeds_paired']} пар")

    oracle_rep = oracle_symmetry_report(rows, manifest, codes, mixed)
    print(f"\n=== Oracle как верхняя граница (симметрия) ===")
    print(f"  {oracle_rep['arm']}: n={oracle_rep['n_seeds']}, |правое − левое| средн "
          f"{oracle_rep['mean_abs_mirror_gap']:.3f} (max {oracle_rep['max_abs_mirror_gap']:.3f}), "
          f"min-half {oracle_rep['min_half_mean']:.3f}, обе половины ≥0.8: "
          f"{oracle_rep['both_halves_ge_0_8']}/{oracle_rep['n_seeds']}")
    if oracle_rep["max_abs_mirror_gap"] > 0.35:
        print("  ПОДОЗРИТЕЛЬНО: у заявленного симметричного oracle большой разброс между "
              "половинами — проверить, не right_bias ли это")

    print("\n=== Вторичные метрики (mixed stream) ===")
    print(f"{'arm':26s} {'succ':>6s} {'шаги':>6s} {'награда':>9s} {'|dw|':>7s} {'bound':>6s} "
          f"{'|Л−П|':>7s} {'обе≥0.8':>9s}")
    for arm in manifest["arms"]:
        rws = p4.sub(rows, arm["code_arm"], mixed)
        if not rws:
            continue
        gaps = [abs(float(r["success_mirror[right]"]) - float(r["success_mirror[left]"])) for r in rws]
        both8 = sum(1 for r in rws if float(r["success_mirror[right]"]) >= 0.8
                    and float(r["success_mirror[left]"]) >= 0.8)
        print(f"{arm['arm_id']:26s} {p4.fmt(p4.mean_of(rws, 'success_rate'), 2):>6s} "
              f"{p4.fmt(p4.mean_of(rws, 'steps_to_target'), 1):>6s} "
              f"{p4.fmt(p4.mean_of(rws, 'cumulative_reward'), 1):>9s} "
              f"{p4.fmt(p4.mean_of(rws, 'mean_abs_dw'), 4):>7s} "
              f"{p4.fmt(p4.mean_of(rws, 'fraction_weights_at_bound'), 2):>6s} "
              f"{statistics.fmean(gaps):7.3f} {both8:4d}/{len(rws):<4d}")

    decision = classify(manifest, results, oracle_rep)
    print(f"\n=== Решение по заранее объявленной таблице ===\n  {decision['key']}: {decision['text']}")

    if a.per_seed_out:
        path = pathlib.Path(a.per_seed_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
            w.writeheader()
            w.writerows(per_seed_rows)
        print(f"raw per-seed исходы → {path} ({len(per_seed_rows)} строк)")
    if a.json_out:
        payload = {
            "protocol_id": manifest["protocol_id"],
            "manifest_digest": digest_file,
            "supersedes_protocol_id": manifest.get("supersedes_protocol_id"),
            "frozen_protocol_digest": proto.FROZEN_PROTOCOL_DIGEST,
            "primary_metric": manifest["primary_metric"],
            "task_stream_scored": mixed,
            "alpha": alpha,
            "comparisons": results,
            "oracle_symmetry": oracle_rep,
            "decision": decision,
            "required_gate_passed": required_ok,
            "unit_of_analysis": manifest["statistical_test"]["unit_of_analysis"],
        }
        out_json = pathlib.Path(a.json_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"вердикт → {a.json_out}")

    print("\nвердикт гейта:", "REQUIRED-ГИПОТЕЗЫ ПРОЙДЕНЫ" if required_ok else "REQUIRED-ГИПОТЕЗЫ НЕ ПРОЙДЕНЫ")
    return 0 if required_ok else 1


def classify(manifest: dict, results: list[dict], oracle_rep: dict) -> dict:
    """Маппинг исхода в заранее объявленное решение — без «дообъяснения» post factum."""
    required = [r for r in results if r["kind"] == "required"]
    desirable = [r for r in results if r["kind"] == "desirable"]
    oracle = next((r for r in results if r["kind"] == "characterization"), None)

    def passed(r: dict) -> bool:
        adj = r.get("adjusted_p")
        return r["wins"] > r["losses"] and adj is not None and adj < manifest["statistical_test"]["alpha"]

    rule = manifest.get("decision_rule_after_p42", {})
    all_required = bool(required) and all(passed(r) for r in required)
    all_desirable = bool(desirable) and all(passed(r) for r in desirable)
    below_oracle = bool(oracle is not None and oracle["mean_treatment"] < oracle["mean_comparator"])
    # «Заморозка стабильнее обучения» — это не пропущенная desirable-проверка, а
    # проигранная со знаком минус: comparator выше treatment по среднему.
    frozen_above = any(not passed(r) and r["mean_comparator"] > r["mean_treatment"] for r in desirable)

    if not all_required:
        key = "fails_h1_or_h2"
    elif frozen_above and below_oracle:
        key = "oracle_and_frozen_stably_above_rstdp"
    elif below_oracle:
        key = "beats_weak_but_below_oracle"
    elif all_desirable:
        key = "beats_all_and_comparable_to_oracle"
    else:
        key = "beats_weak_but_below_oracle"
    return {"key": key, "text": rule.get(key, "(ключ не объявлен в протоколе)"),
            "inputs": {"required_passed": all_required, "desirable_passed": all_desirable,
                       "frozen_control_above_treatment": frozen_above,
                       "treatment_below_oracle": below_oracle}}


if __name__ == "__main__":
    raise SystemExit(main())
