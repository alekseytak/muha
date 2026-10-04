#!/usr/bin/env python3
"""Предзапусковой аудит P4.2: десять критериев, каждый проверяется кодом.

Зачем: раздел «что проверено» в протоколе не должен быть списком обещаний. Этот
скрипт берёт замороженный манифест, живые модули и содержимое тестов и отвечает на
каждый критерий запуском, а не абзацем. Никаких confirmatory-эпизодов: там, где
нужно что-то прогнать, берутся seed'ы из fixture_probe (объявлены вне всех гейтов).

Критерии 9 и 10 появились вместе с P4.2.v2: преемственность протоколов и полнота
bundle — это ровно то, что в v1 осталось на словах и что нельзя проверить постфактум.

Запуск: .venv/bin/python scripts/audit_p4_2_prereqs.py
Код возврата: 0 — все критерии подтверждены; 1 — хотя бы один провален.
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import sys
import tempfile
from types import SimpleNamespace

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

import p4_2_protocol as proto  # noqa: E402
import check_p4_2_oracle_gate as gate  # noqa: E402
import collect_p4_2_bundle as bundle  # noqa: E402
import run_p4_2_oracle as runner  # noqa: E402
from fly_connectome_agent.src.engineering.governance.gatekeeper import (  # noqa: E402
    GateKeeper, PolicyDecision, VerbAct,
)
from fly_connectome_agent.src.engineering.harness import symmetric_toy as toy  # noqa: E402
from fly_connectome_agent.src.engineering.harness.symmetric_toy import (  # noqa: E402
    ORACLE_ARM, MIRROR_PERM, ToySettings, init_weights, make_modulator, run_arm,
)
from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog  # noqa: E402

TESTS = REPO / "fly_connectome_agent" / "tests"
FAIL = "ПРОВАЛ"


class Fail(Exception):
    """Критерий не подтверждён."""


def ok(detail: str) -> str:
    return detail


def va(role="Decider", verb="выбирает", context=None, proposed_action="move_left"):
    return VerbAct(event_id="e", subject_id="s", role=role, verb=verb, object_ref="o",
                   domain="simulation", context=context or {}, trace_id="t",
                   proposal_ref="p", proposed_action=proposed_action)


# --- 1. ALLOW сохраняет предложенное действие, и тест это ловит ----------------

def c1_allow_keeps_proposal(manifest, settings, tmp) -> str:
    gk = GateKeeper()
    for action in ("move_left", "move_right", "stay"):
        d = gk.evaluate(va(proposed_action=action))
        if (d.status, d.enforced_action) != ("ALLOW", action):
            raise Fail(f"ALLOW для {action} вернул {d.status}/{d.enforced_action}")
    prop = SimpleNamespace(action_type="move_left", action_id="prop-1")
    v = VerbAct.from_proposal(prop, event_id="e", subject_id="s", role="Decider",
                              verb="выбирает", domain="simulation", trace_id="t")
    if v.proposed_action != "move_left":
        raise Fail("from_proposal не скопировал действие декодера")
    # Тест обязан быть асимметричным: с move_right подмена не видна.
    src = (TESTS / "test_gatekeeper.py").read_text(encoding="utf-8") + \
          (TESTS / "test_p3_closed_loop.py").read_text(encoding="utf-8")
    if not re.search(r'status.*ALLOW.*\n.*enforced_action.*==.*"move_left"', src) \
            and 'enforced_action == "move_left"' not in src:
        raise Fail("в тестах нет ALLOW-проверки с move_left — подмена осталась бы невидимой")
    return ok("ALLOW → proposed (лево/право/stay), from_proposal, асимметричный тест найден")


# --- 2. DENY/ESCALATE навязывают только stay ----------------------------------

def c2_deny_escalate_stay(manifest, settings, tmp) -> str:
    gk = GateKeeper()
    cases = [
        ("DENY", va(role="Hacker", verb="взломать")),
        ("DENY", va(verb="удалить")),
        ("DENY", va(verb="якорит")),                       # L5-глагол не у Registrar
        ("ESCALATE", va(context={"risk_level": "high"})),
        ("ESCALATE", va(context={"danger": "critical"})),
    ]
    for want, act in cases:
        d = gk.evaluate(act)
        if (d.status, d.enforced_action) != (want, "stay"):
            raise Fail(f"{want}: получил {d.status}/{d.enforced_action}")
    for forged in (
        dict(status="ALLOW", proposed_action="move_left", enforced_action="stay", reason="x"),
        dict(status="DENY", proposed_action="move_left", enforced_action="move_left", reason="x"),
        dict(status="ESCALATE", proposed_action="move_right", enforced_action="move_right", reason="x"),
    ):
        try:
            PolicyDecision(**forged)
        except (ValueError, TypeError):
            continue
        raise Fail(f"подделка конструируется: {forged}")
    return ok("5 путей DENY/ESCALATE дают stay; ALLOW без proposals / DENY не-stay не конструируются")


# --- 3. on-disk tamper ловится свежим читателем -------------------------------

def c3_on_disk_tamper(manifest, settings, tmp) -> str:
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "chain.jsonl"
        log = ProvenanceLog(path=str(path))
        log.append({"event": "a"})
        log.append({"event": "b"})
        if ProvenanceLog(path=str(path)).verify_chain() is not True:
            raise Fail("чистый файл не проходит проверку — тест ничего не доказывает")
        lines = path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[0])
        entry["payload"]["event"] = "HACKED"
        path.write_text(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
                        + lines[1] + "\n", encoding="utf-8")
        if ProvenanceLog(path=str(path)).verify_chain() is not False:
            raise Fail("подменённый payload на диске не пойман свежим читателем")
        # Порча entry_hash — тоже через файл, не через кэш писавшего процесса.
        entry = json.loads(lines[1])
        entry["entry_hash"] = "0" * 64
        path.write_text(lines[0] + "\n" + json.dumps(entry, sort_keys=True,
                        separators=(",", ":")) + "\n", encoding="utf-8")
        if ProvenanceLog(path=str(path)).verify_chain() is not False:
            raise Fail("битый entry_hash на диске не пойман")
    return ok("tamper payload и битый hash ловятся новым ProvenanceLog(path), а не кэшем")


# --- 4. confirmatory seed'ы ---------------------------------------------------

def _seeds_on_disk() -> tuple[set[int], list[str]]:
    """Какие seed'ы реально встречаются в артефактах var/ на этой машине.

    Скан нужен затем, чтобы несечение проверялось по журналам и CSV, а не по
    объявленным в манифесте диапазонам: v2 обязан быть disjoint от всего, что
    когда-либо касалось этой инфраструктуры, включая прерванный v1. Направление
    ошибки выбрано консервативное: лишний seed, найденный в безобидном bench-файле,
    только запретит брать этот seed, а пропущенный — позволил бы переиспользовать
    сожжённый.
    """
    found: set[int] = set()
    sources: list[str] = []
    var = REPO / "var"
    for path in sorted(var.rglob("*.csv")):
        seeds: set[int] = set()
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    raw = (row.get("seed") or "").strip()
                    if raw.lstrip("-").isdigit():
                        seeds.add(int(raw))
        except (UnicodeDecodeError, csv.Error):
            continue
        if seeds:
            found |= seeds
            sources.append(f"{path.relative_to(var)}:{len(seeds)}")
    for path in sorted(var.rglob("*.jsonl")):
        seeds = set()
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if '"seed"' not in line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    payload = event.get("payload", {}) if isinstance(event, dict) else {}
                    for value in (payload.get("seed"), *(payload.get("seeds") or [])):
                        if isinstance(value, int):
                            seeds.add(value)
        except (UnicodeDecodeError, OSError):
            continue
        if seeds:
            found |= seeds
            sources.append(f"{path.relative_to(var)}:{len(seeds)}")
    return found, sources


def c4_seeds(manifest, settings, tmp) -> str:
    spec = manifest["seed_sets"]["confirmatory"]
    conf = proto.confirmatory_seeds(manifest)
    lo, hi = spec["range"]
    if conf != list(range(lo, hi + 1)):
        raise Fail(f"confirmatory = {conf[:3]}…{conf[-3:]} (n={len(conf)}) — это не сплошной "
                   f"диапазон {lo}–{hi}")
    if spec["count"] != len(conf):
        raise Fail(f"count={spec['count']} против {len(conf)} seed'ов в диапазоне")
    probe = set(proto.probe_seeds(manifest))
    declared = set(conf) | probe
    used_evidence = 0
    for other in manifest["seed_sets"]["disjoint_from"]:
        seeds = set(proto.expand_seed_range(other, other["id"]))
        if declared & seeds:
            raise Fail(f"{other['id']} пересекается с confirmatory/probe: {sorted(declared & seeds)[:5]}")
        ev = other.get("evidence")
        if not ev:
            continue
        used_evidence += 1
        rec = json.loads((REPO / ev).read_text(encoding="utf-8"))
        values = set(rec["burned_seed_set"]["values"])
        if values != seeds:
            raise Fail(f"{other['id']}: в манифесте {len(seeds)} seed'ов, в записи {ev} — "
                       f"{len(values)}; симметричная разница {sorted(values ^ seeds)[:5]}")
        if rec.get("coverage", {}).get("evidence_complete") is not True:
            raise Fail(f"{ev}: помечена как неполная — несечение по ней не подтверждается")
        if rec.get("protocol_id") == manifest["protocol_id"]:
            raise Fail(f"{ev}: доказательство из того же протокола")
    on_disk, sources = _seeds_on_disk()
    clash = sorted(declared & on_disk)
    if clash:
        raise Fail(f"seed'ы {clash[:8]} уже встречаются в артефактах var/ ({len(sources)} файлов) — "
                   "объявленное несечение расходится с диском")
    return ok(f"n={len(conf)} = {lo}–{hi}; probe {sorted(probe)}; disjoint_from "
              f"{len(manifest['seed_sets']['disjoint_from'])} множеств (с доказательством: "
              f"{used_evidence}); на диске seed'ов {len(on_disk)} из {len(sources)} файлов, "
              f"пересечений с {lo}–{hi} нет")


# --- 5. единица статистики ----------------------------------------------------

def c5_unit_is_seed(manifest, settings, tmp) -> str:
    if manifest["statistical_test"]["unit_of_analysis"] != "seed":
        raise Fail(f"unit_of_analysis = {manifest['statistical_test']['unit_of_analysis']}")
    schema = json.loads(proto.SCHEMA_PATH.read_text(encoding="utf-8"))
    const = schema["properties"]["statistical_test"]["properties"]["unit_of_analysis"].get("const")
    if const != "seed":
        raise Fail(f"в схеме это не заперто: const={const!r}")
    return ok("манифест: seed; схема: const 'seed' (episode/provenance_entry невалидны)")


# --- 6. oracle ----------------------------------------------------------------

def c6_oracle(manifest, settings, tmp) -> str:
    decl = next(a for a in manifest["arms"] if a["arm_id"] == "oracle_reflex")
    spec = toy.ARM_SPECS[ORACLE_ARM]
    for key, want in (("learner", "none"), ("reward_coupled", False)):
        if decl[key] != want or spec[key] != want:
            raise Fail(f"oracle {key}: манифест {decl[key]!r}, код {spec[key]!r}, ждали {want!r}")
    if decl["role"] != "upper_bound_characterization" or decl["gate"] != "none":
        raise Fail(f"oracle объявлен гейтом: role={decl['role']}, gate={decl['gate']}")
    if ORACLE_ARM in toy.ARMS:
        raise Fail("oracle попал в матрицу P4.1 — воспроизведение P4.1 изменилось")
    w = init_weights("oracle", np.random.default_rng(0), settings)
    if not np.array_equal(w, w[MIRROR_PERM]):
        raise Fail("вектор весов oracle не инвариантен отражению лево↔право")
    m = make_modulator(reward_coupled=False, settings=settings)
    got = [float(m.compute(task_reward=r)) for r in (10.0, -1.0, 0.5)]
    if any(v != 0.0 for v in got):
        raise Fail(f"reward-blind нарушен: M={got} при наградах 10 / -1 / 0.5")
    seed = proto.probe_seeds(manifest)[0]
    res = run_arm(ORACLE_ARM, "mixed", seed, episodes=5, settings=settings)
    moved = float(np.max(np.abs(res.dw_per_edge))) if len(res.dw_per_edge) else 0.0
    if moved > 1e-15 or not np.allclose(res.weights_final, res.weights_init, atol=1e-15):
        raise Fail(f"oracle двинулся за 5 прогонных эпизодов: max|dw|={moved}")
    return ok("non-plastic, reward-blind (M≡0 при r=10), зеркально симметричен, gate=none, "
              "Δw≡0 на живом прогоне, вне матрицы P4.1")


# --- 7. у раннера нет способа расширить или перенастроить прогон --------------

ALLOWED_CLI = {"manifest", "out", "provenance", "seeds", "arms", "probe", "describe", "verbose",
               "experimental_manifest"}


def c7_runner_cli(manifest, settings, tmp) -> str:
    src = pathlib.Path(runner.__file__).read_text(encoding="utf-8")
    flags = set(re.findall(r'add_argument\("--([a-z-]+)"', src))
    if flags != {f.replace("_", "-") for f in ALLOWED_CLI}:
        raise Fail(f"набор флагов изменился: {sorted(flags)}")
    forbidden = re.findall(r'add_argument\("--(eta|episode|window|sigma|alpha|gate|threshold|'
                           r'weight|strong|weak|regime)[a-z-]*"', src, re.I)
    if forbidden:
        raise Fail(f"CLI даёт менять параметры прогона: {forbidden}")
    declared = proto.confirmatory_seeds(manifest)
    outside = declared[0] - 1            # seed ровно за границей объявленного множества
    try:
        runner.check_subset([outside], declared, "seeds")
    except SystemExit:
        pass
    else:
        raise Fail(f"check_subset пропустил seed {outside} вне замороженного множества")
    runner.check_subset(declared[: len(declared) // 2], declared, "seeds")   # подрезка разрешена
    # Якорь pre-registration: он в коде, а не в проверяемом файле, иначе
    # --manifest указывает на правленый протокол и само сравнение digest'ов
    # становится тавтологией.
    if proto.digest_of_file(proto.DEFAULT_MANIFEST) != proto.FROZEN_PROTOCOL_DIGEST:
        raise Fail("манифест в репозитории разошёлся с замороженным digest в коде")
    try:
        runner.freeze_guard({"_digest": "sha256:" + "0" * 64}, False, str(runner.DEFAULT_OUT))
    except runner.Refused:
        pass
    else:
        raise Fail("runner принимает не-замороженный манифест без флага")
    if runner.freeze_guard({"_digest": proto.FROZEN_PROTOCOL_DIGEST}, False,
                           str(runner.DEFAULT_OUT)) is not False:
        raise Fail("замороженный манифест помечен как experimental")
    try:
        gate.check_sidecars([], [], "sha256:" + "0" * 64)
    except gate.Refused:
        pass
    else:
        raise Fail("гейт принимает не-замороженный digest")
    return ok(f"флагов ровно {len(flags)}: {sorted(flags)}; расширение seed/arm — SystemExit; "
              "эпизоды/веса/пороги из CLI не задаются; digest заморожен в коде, runner и "
              "gate отказывают не-замороженному")


# --- 8. partial-прогон не принимается ----------------------------------------

def c8_partial_refused(manifest, settings, tmp) -> str:
    digest = manifest["_digest"]
    if digest != proto.FROZEN_PROTOCOL_DIGEST:
        raise Fail("манифест расходится с якорем в коде — проверять нечего")
    csv_path = tmp / "shard.csv"
    csv_path.write_text("arm,task,seed\n", encoding="utf-8")
    side = tmp / (csv_path.name + ".meta.json")

    def write(partial):
        side.write_text(json.dumps({"manifest_digest": digest, "partial": partial}),
                        encoding="utf-8")

    def refused(label, **kw):
        try:
            gate.check_sidecars(**kw)
        except gate.Refused:
            return
        raise Fail(f"{label} — гейт не отказал")

    write(True)
    refused("partial=True принят гейтом",
            csv_paths=[str(csv_path)], sidecar_paths=[str(side)], digest=digest)
    write(False)
    refused("прогон без sidecar считается валидным",
            csv_paths=[str(csv_path)], sidecar_paths=[], digest=digest)
    # Прогон, сделанный по другой (например, отредактированной после запуска)
    # версии протокола, обязан ловиться по digest, а не совпадать с ним "сам с собой".
    digest_now = proto.load(pathlib.Path(proto.DEFAULT_MANIFEST))["_digest"]
    write(False)
    side.write_text(json.dumps({"manifest_digest": "sha256:" + "0" * 64, "partial": False}),
                    encoding="utf-8")
    refused("stale sidecar принят гейтом",
            csv_paths=[str(csv_path)], sidecar_paths=[str(side)], digest=digest_now)
    side.write_text(json.dumps({"manifest_digest": digest_now, "partial": False}),
                    encoding="utf-8")
    refused("sidecar без своего CSV принят гейтом",
            csv_paths=[], sidecar_paths=[str(side)], digest=digest_now)
    if not gate.check_sidecars([str(csv_path)], [str(side)], digest_now):
        raise Fail("полный прогон не вернулся пару (имя, sidecar)")
    return ok("partial / нет sidecar / чужой digest / лишний sidecar → Refused; полный прогон проходит")


# --- 9. реестр протоколов: закрытый v1 не считается, v2 не запущен ------------

def _must_refuse(label, fn, *, exc, needle=""):
    try:
        fn()
    except exc as got:
        if needle and needle not in str(got):
            raise Fail(f"{label}: отказ есть, но звучит не про то — {str(got).splitlines()[0][:120]}")
        return
    raise Fail(f"{label} — отказа не случилось")


def c9_lineage_registry(manifest, settings, tmp) -> str:
    if proto.ACTIVE_PROTOCOL_ID != manifest["protocol_id"]:
        raise Fail(f"в коде активен {proto.ACTIVE_PROTOCOL_ID}, а загружен {manifest['protocol_id']}")
    v1 = proto.frozen_entry("p4.2.oracle-baseline.v1")
    v2 = proto.frozen_entry("p4.2.oracle-baseline.v2")
    if v1["status"] != manifest["supersedes_status"]:
        raise Fail(f"статус v1 в реестре {v1['status']!r}, манифест заявляет "
                   f"{manifest['supersedes_status']!r}")
    if proto.digest_of_file(proto.V1_MANIFEST) != v1["digest"]:
        raise Fail("файл v1 разошёлся со своим якорем: закрытый протокол перестал быть "
                   "проверяемым — правка после остановки")
    if proto.digest_of_file(proto.DEFAULT_MANIFEST) != v2["digest"]:
        raise Fail("файл v2 разошёлся с якорем в реестре")
    _must_refuse("подсчёт прогона по закрытому v1", lambda: proto.require_frozen(v1["digest"], where="аудит"),
                 exc=proto.ProtocolError, needle="закрытому протоколу не считают")
    if proto.require_frozen(v2["digest"], where="аудит", protocol_id="p4.2.oracle-baseline.v2") != v2["digest"]:
        raise Fail("активный протокол не проходит собственную сверку")
    allowed, flag = proto.run_authorization(manifest["protocol_id"])
    if allowed:
        raise Fail(f"confirmatory-прогон {manifest['protocol_id']} уже авторизован ({flag}) — "
                   "а должен быть разрешён только отдельным review-актом")
    _must_refuse("запуск confirmatory без авторизации",
                 lambda: runner.authorize_run(manifest), exc=runner.Refused, needle="не авторизован")
    for path in (runner.V1_PROV, runner.ABORTED_ROOT / "p4_2_v1_infrastructure_abort" / "x.jsonl"):
        _must_refuse(f"provenance-путь {path.name} из архива v1",
                     lambda p=path: runner.provenance_guard(manifest, False, str(p)),
                     exc=runner.Refused, needle="прерванному P4.2.v1")
    # схема обязана требовать lineage от v2 и не требовать его от v1
    from jsonschema import Draft202012Validator
    schema = json.loads(proto.SCHEMA_PATH.read_text(encoding="utf-8"))
    stripped = json.loads(proto.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    stripped.pop("supersedes_protocol_id")
    stripped.pop("supersedes_status")
    errors = [e.message for e in Draft202012Validator(schema).iter_errors(stripped)
              if "supersedes" in e.message]
    if len(errors) != 2:
        raise Fail(f"без supersedes-полей схема выдала {errors} — ждали оба поля")
    v1_doc = json.loads(proto.V1_MANIFEST.read_text(encoding="utf-8"))
    v1_errors = [e.message for e in Draft202012Validator(schema).iter_errors(v1_doc)
                 if "supersedes" in e.message]
    if v1_errors:
        raise Fail(f"v1 стал невалиден после появления нового правила: {v1_errors}")
    return ok(f"v1 заякорен и закрыт ({v1['status']}), digest v1 совпадает с файлом; по v1 — "
              f"«по закрытому протоколу не считают»; v2 активен, run={flag!r}; журнал v1 "
              "для раннера запрещён; схема требует lineage от v2 и не трогает v1")


# --- 10. bundle: что лежит в результате и чем это хешится ---------------------

def c10_bundle_contract(manifest, settings, tmp) -> str:
    bundle_spec = manifest.get("result_bundle")
    if not bundle_spec:
        raise Fail("в активном протоколе нет result_bundle")
    roles = [a["role"] for a in bundle_spec["artifacts"]]
    if sorted(roles) != sorted(proto.REQUIRED_BUNDLE_ROLES):
        raise Fail(f"роли bundle разошлись с реестром кода: манифест {sorted(roles)}, "
                   f"код {sorted(proto.REQUIRED_BUNDLE_ROLES)}")
    missing_files = []
    for art in bundle_spec["artifacts"]:
        head = str(art["producer"]).split()[0]
        if "/" in head and not (REPO / head).exists():
            missing_files.append(f"{art['role']} → {head}")
    if missing_files:
        raise Fail("producer указывает на несуществующий файл: " + "; ".join(missing_files))
    paths = bundle.declared_paths(manifest)
    if paths["provenance_head_witness"].name != paths["provenance_jsonl"].name + ".head.json":
        raise Fail("witness назван не по суффиксу писателя — в bundle лежал бы файл, "
                   "которого прогон не создавал")
    if runner.DEFAULT_OUT != paths["run_csv"] or runner.DEFAULT_PROV != paths["provenance_jsonl"]:
        raise Fail(f"дефолты раннера разошлись с манифестом: out={runner.DEFAULT_OUT}, "
                   f"prov={runner.DEFAULT_PROV}")
    _must_refuse("гейт пишет вердикт мимо bundle",
                 lambda: gate.bundle_gate(manifest, csv_paths=[str(runner.DEFAULT_OUT)],
                                          per_seed_out=None, json_out=None),
                 exc=gate.Refused, needle="не передан")
    report = bundle.check_empty(manifest)
    if not report["clean"]:
        raise Fail("пути v2 уже заняты: "
                   + "; ".join(f"{i['path']} ({i['bytes']} байт)" for i in report["occupied"])
                   + ((" ; лишнее: " + ", ".join(report["stray_files"])) if report["stray_files"] else ""))
    return ok(f"{len(roles)} ролей = реестр кода; producers существуют; witness по суффиксу "
              f"{'.head.json'}; дефолты раннера совпадают с манифестом; "
              f"{report['directory']} пуст и все {report['declared']} путей свободны")


CRITERIA = (
    ("1. ALLOW сохраняет proposed action (move_left, не move_right)", c1_allow_keeps_proposal),
    ("2. DENY/ESCALATE навязывают только stay", c2_deny_escalate_stay),
    ("3. on-disk tamper ловится свежим читателем", c3_on_disk_tamper),
    ("4. confirmatory = объявленный диапазон, disjoint со всем на диске", c4_seeds),
    ("5. unit_of_analysis = seed (и в манифесте, и в схеме)", c5_unit_is_seed),
    ("6. oracle симметричен, non-plastic, reward-blind, gate=none", c6_oracle),
    ("7. у CLI раннера нет способа расширить/перенастроить прогон", c7_runner_cli),
    ("8. partial-прогон writes partial=true и гейтом не принимается", c8_partial_refused),
    ("9. lineage: v1 заякорен и закрыт, v2 активен, запуск не авторизован", c9_lineage_registry),
    ("10. bundle: девять ролей из кода, producers живые, пути v2 пусты", c10_bundle_contract),
)


def main() -> int:
    manifest = proto.load(pathlib.Path(proto.DEFAULT_MANIFEST))
    settings = proto.build_settings(manifest)
    tmp = pathlib.Path(tempfile.mkdtemp())
    print(f"protocol {manifest['protocol_id']}  digest {manifest['_digest']}")
    print(f"seeds {manifest['seed_sets']['confirmatory']['range']}  "
          f"episodes {manifest['episodes_per_seed']}  eta {settings.eta}\n")
    bad = 0
    for title, fn in CRITERIA:
        try:
            detail = fn(manifest, settings, tmp)
        except Fail as exc:
            bad += 1
            print(f"  {FAIL:7s} {title}\n          {exc}")
            continue
        print(f"  ок      {title}\n          {detail}")
    print(f"\n{'все ' + str(len(CRITERIA)) + ' критериев подтверждены запуском' if not bad else f'провалено: {bad}'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
