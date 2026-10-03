#!/usr/bin/env python3
"""Предзапусковой аудит P4.2: восемь критериев, каждый проверяется кодом.

Зачем: раздел «что проверено» в протоколе не должен быть списком обещаний. Этот
скрипт берёт замороженный манифест, живые модули и содержимое тестов и отвечает на
каждый критерий запуском, а не абзацем. Никаких confirmatory-эпизодов: там, где
нужно что-то прогнать, берутся seed'ы из fixture_probe (объявлены вне всех гейтов).

Запуск: .venv/bin/python scripts/audit_p4_2_prereqs.py
Код возврата: 0 — все критерии подтверждены; 1 — хотя бы один провален.
"""
from __future__ import annotations

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

def c4_seeds(manifest, settings, tmp) -> str:
    conf = proto.confirmatory_seeds(manifest)
    if conf != list(range(60, 120)):
        raise Fail(f"confirmatory = {conf[:3]}…{conf[-3:]} (n={len(conf)}), ждали ровно 60–119")
    used = set(conf)
    for other in manifest["seed_sets"]["disjoint_from"]:
        seeds = set(proto.expand_seed_range(other, other["id"]))
        if used & seeds:
            raise Fail(f"{other['id']} пересекается с confirmatory: {sorted(used & seeds)[:5]}")
        if seeds & set(proto.probe_seeds(manifest)):
            raise Fail(f"{other['id']} пересекается с fixture_probe")
    if set(range(0, 60)) & used:
        raise Fail("0–59 (seed'ы P4.1) попали в confirmatory")
    return ok(f"n={len(conf)} ровно 60–119; disjoint_from={len(manifest['seed_sets']['disjoint_from'])} "
              f"множеств, пересечений нет; 0–59 запрещены")


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

ALLOWED_CLI = {"manifest", "out", "provenance", "seeds", "arms", "probe", "describe", "verbose"}


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
    try:
        runner.check_subset([60, 120], declared, "seeds")
    except SystemExit:
        pass
    else:
        raise Fail("check_subset пропустил seed 120 вне замороженного множества")
    runner.check_subset([60, 89], declared, "seeds")   # подрезка разрешена
    return ok(f"флагов ровно {len(flags)}: {sorted(flags)}; расширение seed/arm — SystemExit; "
              "эпизоды/веса/пороги из CLI не задаются")


# --- 8. partial-прогон не принимается ----------------------------------------

def c8_partial_refused(manifest, settings, tmp) -> str:
    digest = manifest["_digest"]
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


CRITERIA = (
    ("1. ALLOW сохраняет proposed action (move_left, не move_right)", c1_allow_keeps_proposal),
    ("2. DENY/ESCALATE навязывают только stay", c2_deny_escalate_stay),
    ("3. on-disk tamper ловится свежим читателем", c3_on_disk_tamper),
    ("4. confirmatory = ровно 60–119, disjoint от 0–59", c4_seeds),
    ("5. unit_of_analysis = seed (и в манифесте, и в схеме)", c5_unit_is_seed),
    ("6. oracle симметричен, non-plastic, reward-blind, gate=none", c6_oracle),
    ("7. у CLI раннера нет способа расширить/перенастроить прогон", c7_runner_cli),
    ("8. partial-прогон writes partial=true и гейтом не принимается", c8_partial_refused),
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
    print(f"\n{'все восемь критериев подтверждены запуском' if not bad else f'провалено: {bad}'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
