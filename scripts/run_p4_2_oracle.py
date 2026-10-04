#!/usr/bin/env python
"""P4.2 — прогон сравнения R-STDP против oracle-рефлекса.

Запускается ТОЛЬКО из манифеста: ни episodes, ни eta, ни набор задач, ни метрику
сюда передать нельзя. Единственные CLI-ограничители — `--seeds` и `--arms`, и оба
умеют только РЕЗАТЬ объявленное множество пополам для шардов; прогон с подрезанным
множеством помечается в sidecar как partial, и гейт такой CSV брать не будет.
Это и есть разница между «повторить прогон частями» и «пересобрать эксперимент под
ответ»: здесь второе не выражается словами.

Единица наблюдения — seed (не эпизод и не строка provenance), поэтому прогон
обязан покрыть все 60 seed x все streams: дырку в матрице гейт считает не
«недостающим наблюдением», а нарушением протокола.

Запуск (полный, один процесс):
    .venv/bin/python scripts/run_p4_2_oracle.py \
        --out var/p4_2_v2/run.csv --provenance var/p4_2_v2/provenance.jsonl
По шардам (пример — нарезка объявленных seed'ов):
    ... --seeds 120-149 --out var/p4_2_v2/shards/run_a.csv
    ... --seeds 150-179 --out var/p4_2_v2/shards/run_b.csv
Проверка инструмента без прогона:
    ... --describe      (план и digest, ни одного эпизода)
    ... --probe         (seed'ы 900-903, вне любых гейтов, 8 эпизодов)

Что здесь сторожится отдельно от протокола (P4.2.v1 этого не имел и потому встал):
  - authorize_run: confirmatory-прогон идёт только если review поставил в реестре
    run=authorized. Замороженный файл не может выдать разрешение сам себе;
  - fresh provenance: журнал v2 обязан начинаться с пустого пути. Журнал v1 —
    архив: он не читается, не восстанавливается через recover_head и не продолжается;
  - bundle placement: артефакты прогона лежат в объявленном result_bundle-каталоге.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

import p4_2_protocol as proto  # noqa: E402
import run_p4_validation as p4  # noqa: E402

DEFAULT_OUT = REPO / "var" / "p4_2_v2" / "run.csv"
DEFAULT_PROV = REPO / "var" / "p4_2_v2" / "provenance.jsonl"
# Архив прерванного v1. Путь к нему запрещён раннеру явно: «не продолжать v1»
# должно быть отказом кода, а не памяткой о том, какой флаг не надо жать.
V1_PROV = REPO / "var" / "p4_2_oracle.prov.jsonl"
ABORTED_ROOT = REPO / "var" / "aborted"
PROBE_EPISODES = 8
REFUSE = 2          # «прогона по этому протоколу не будет», тот же смысл, что у гейта


class Refused(Exception):
    """Право считать отсутствует: прогон не по замороженному pre-registration."""


def git_state() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:  # нет git / не репозиторий — пишем пусто, не падаем
            return ""

    return {"rev": run("rev-parse", "--short", "HEAD"), "dirty": "yes" if run("status", "--porcelain") else "no"}


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def check_subset(chosen: list[int], allowed: list[int], what: str) -> None:
    extra = sorted(set(chosen) - set(allowed))
    if extra:
        raise SystemExit(
            f"--{what} содержит значения вне замороженного протокола: {extra[:8]}. "
            f"Добавить seed/arm в уже объявленный прогон — это не шардинг, а новый эксперимент."
        )


def summarise(rows: list[dict], manifest: dict, arm_ids: list[str], streams: list[str],
              codes: dict[str, str]) -> None:
    col = proto.PRIMARY_METRIC_TO_COLUMN[manifest["primary_metric"]]
    print("\n=== P4.2 сводка (среднее по seed; единица наблюдения — seed) ===")
    header = f"{'arm':26s} {'stream':7s} {'n':>3s} {'succ':>6s} {'мин-половина':>12s} {'награда':>9s} {'|dw|':>7s} {'bound':>6s}"
    print(header)
    print("-" * len(header))
    for arm_id in arm_ids:
        code = codes[arm_id]
        for stream in streams:
            rws = p4.sub(rows, code, stream)
            if not rws:
                continue
            print(
                f"{arm_id:26s} {stream:7s} {len(rws):3d} "
                f"{p4.fmt(p4.mean_of(rws, 'success_rate'), 2):>6s} "
                f"{p4.fmt(p4.mean_of(rws, col), 3):>12s} "
                f"{p4.fmt(p4.mean_of(rws, 'cumulative_reward'), 1):>9s} "
                f"{p4.fmt(p4.mean_of(rws, 'mean_abs_dw'), 4):>7s} "
                f"{p4.fmt(p4.mean_of(rws, 'fraction_weights_at_bound'), 2):>6s}"
            )
    viol = int(sum(r.get("governance_violations", 0) or 0 for r in rows))
    print(f"\n  governance violations за прогон: {viol} "
          f"(бюджет протокола: {manifest['provenance']['violation_budget']})")


def freeze_guard(manifest: dict, experimental: bool, out_path: str) -> bool:
    """True если прогон явно объявлен experimental (не замороженный протокол).

    Правила, без которых pre-registration ничего не стоит:
    - не-замороженный манифест без --experimental-manifest — отказ;
    - --experimental-manifest на замороженном манифесте — отказ: флаг не должен
      становиться «ритуальной» кнопкой, иначе по нему нельзя судить о прогоне;
    - experimental не имеет права писать в confirmatory путь: такой CSV потом
      неотличим от настоящего прогона глазами.
    """
    digest = manifest["_digest"]
    frozen = digest == proto.FROZEN_PROTOCOL_DIGEST
    if frozen:
        if experimental:
            raise Refused("--experimental-manifest не нужен: digest совпадает с замороженным "
                          "протоколом. Флаг без правки только размывает границу.")
        return False
    if not experimental:
        raise Refused(
            f"digest манифеста {digest} не равен замороженному протоколу "
            f"{proto.ACTIVE_PROTOCOL_ID} ({proto.FROZEN_PROTOCOL_DIGEST}). Правка протокола "
            "после заморозки — это уже не P4.2; для отладочного прогона нужны "
            "--experimental-manifest и другой --out.")
    if pathlib.Path(out_path).resolve() == pathlib.Path(DEFAULT_OUT).resolve():
        raise Refused(f"experimental-прогон не может писать в confirmatory путь {DEFAULT_OUT}")
    return True


def bundle_dir(manifest: dict) -> pathlib.Path | None:
    """Каталог объявленного result_bundle (None для протоколов до v2)."""
    b = manifest.get("result_bundle")
    return (REPO / b["directory"]).resolve() if b else None


def authorize_run(manifest: dict) -> None:
    """Confirmatory-прогон разрешён только если review поставил run=authorized.

    Разрешение живёт в коде (p4_2_protocol.FROZEN_PROTOCOLS), а не в манифесте:
    замороженный план не может выдать себе право на запуск, иначе «запрещено до
    review» было бы строкой в том же файле, который это запрещает. Акцент на
    confirmatory: experimental-прогон (уже помечен non_confirmatory и гейтом не
    принимается) остаётся доступен — инфраструктуру надо чем-то проверять.
    """
    allowed, flag = proto.run_authorization(manifest["protocol_id"])
    if not allowed:
        raise Refused(
            f"запуск по {manifest['protocol_id']} не авторизован (run={flag!r} в реестре кода). "
            "Confirmatory-прогон начинается только после review этого коммита: право даёт "
            "отдельный акт — строка run=\"authorized\" в p4_2_protocol.FROZEN_PROTOCOLS, а не "
            "правка манифеста. --describe и --probe работают и без авторизации.")


def provenance_guard(manifest: dict, experimental: bool, prov_path: str) -> None:
    """Журнал v2 обязан начинаться с чистого листа, и v1 к нему непричастен.

    Правила зависят от того, чей это прогон:
    - путь внутри архива v1 (или ровно v1-журнал) запрещён всем: продолжение
      stopped-прогона дало бы цепь, у которой начало из одного протокола, а
      продолжение из другого;
    - непустой файл запрещён confirmatory: это уже начатый прогон, а продолжение
      чужой цепи возможно только через recover_head(), который для v2 запрещён
      контрактом. experimental остаётся исключением намеренно: на нём проверяется
      сам контракт adoption (test_runner_refuses_a_provenance_log_it_cannot_trust),
      и раннер обязан доходить до писателя, а не вставать раньше него;
    - писать в bundle-каталог запрещён experimental: отладочная запись легла бы
      рядом с подтверждённым протоколом.
    """
    path = pathlib.Path(prov_path).resolve()
    if path == V1_PROV.resolve() or ABORTED_ROOT in path.parents:
        raise Refused(
            f"provenance-путь {path} относится к прерванному P4.2.v1 (архив "
            f"{ABORTED_ROOT.relative_to(REPO)}). Его нельзя ни продолжить, ни "
            "восстановить через recover_head: v2 обязан писать новый пустой журнал, "
            "иначе цепь начнётся событиями закрытого протокола.")
    if not experimental and path.exists() and path.stat().st_size > 0:
        raise Refused(
            f"provenance-файл {path} уже непустой ({path.stat().st_size} байт). "
            "Confirmatory-прогон обязан создавать свежий пустой путь: продолжение "
            "чужого журнала требует recover_head(), а для v2 это запрещено контрактом — "
            "удалите артефакты незавершённой попытки через collect_p4_2_bundle.py "
            "--check-empty (он показывает, что мешает) и запустите заново.")
    if experimental:
        bdir = bundle_dir(manifest)
        if bdir is not None and bdir in path.parents:
            raise Refused(f"experimental-прогон не может писать в bundle-каталог {bdir.relative_to(REPO)}: "
                          "отладочная запись легла бы рядом с подтверждённым протоколом")


def bundle_guard(manifest: dict, experimental: bool, out_path: str) -> None:
    """Артефакты confirmatory-прогона обязаны попасть в объявленный bundle.

    Иначе «result_bundle из девяти артефактов» остаётся описанием папки, которую
    никто не создавал: CSV в var/ отдельно, witness где-то отдельно.
    """
    bdir = bundle_dir(manifest)
    if bdir is None:
        return
    path = pathlib.Path(out_path).resolve()
    if experimental:
        if bdir in path.parents:
            raise Refused(f"experimental-прогон не может писать в bundle-каталог "
                          f"{bdir.relative_to(REPO)}")
        return
    if bdir not in path.parents:
        raise Refused(f"--out {path} вне объявленного result_bundle.directory "
                      f"{bdir.relative_to(REPO)}: вердикт собирается по списку артефактов из "
                      "манифеста, и файл снаружи этого списка в bundle не попадёт")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(proto.DEFAULT_MANIFEST))
    ap.add_argument("--experimental-manifest", action="store_true",
                    help="разрешить прогон по НЕ замороженному манифесту: в sidecar и "
                         "provenance пишется non_confirmatory=true, а в confirmatory путь "
                         "(var/p4_2_oracle.csv) писать запрещено — гейт такой CSV не примет")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--provenance", default=str(DEFAULT_PROV))
    ap.add_argument("--seeds", help="подрезка объявленного множества для шарда (напр. 60-89)")
    ap.add_argument("--arms", help="подрезка объявленных arms для шарда (имена из манифеста)")
    ap.add_argument("--probe", action="store_true",
                    help=f"проверка инструмента на seed'ах вне гейтов, {PROBE_EPISODES} эпизодов; CSV не пишется")
    ap.add_argument("--describe", action="store_true", help="план и digest без единого эпизода")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    try:
        manifest = proto.load(pathlib.Path(a.manifest))
    except proto.ProtocolError as exc:
        # «проверки не было» = exit 2, а не трейсбек с exit 1.
        print(f"ЗАПУСК ОТКЛОНЁН: манифест не проходит замороженный протокол: {exc}",
              file=sys.stderr)
        return REFUSE
    try:
        experimental = freeze_guard(manifest, a.experimental_manifest, a.out)
    except Refused as exc:
        print(f"ЗАПУСК ОТКЛОНЁН: {exc}", file=sys.stderr)
        return REFUSE

    codes = proto.code_arms(manifest)

    print(proto.describe(manifest))
    if experimental:
        print("\nNON-CONFIRMATORY: манифест не равен замороженному протоколу "
              f"({proto.FROZEN_PROTOCOL_DIGEST}); прогон помечен non_confirmatory=true "
              "и не может быть принят гейтом.")

    if a.describe:
        print("\n--describe: ни одного эпизода не симулировано.")
        return 0

    if not a.probe:
        # Отказы до первого эпизода. v1 встал на той же ошибки: проверка целостности
        # журнала случилась внутри append, а не до прогона, и час процессорного
        # времени ушёл на то, чтобы обнаружить, что считать нечего.
        try:
            # Авторизация касается только confirmatory: experimental-прогон помечен
            # non_confirmatory и гейтом не принимается, поэтому он не может присвоить
            # себе чужой результат — но проверять на нём писателя и запись надо.
            if not experimental:
                authorize_run(manifest)
            provenance_guard(manifest, experimental, a.provenance)
            bundle_guard(manifest, experimental, a.out)
        except Refused as exc:
            print(f"ЗАПУСК ОТКЛОНЁН: {exc}", file=sys.stderr)
            return REFUSE

    settings = proto.build_settings(manifest)
    all_seeds = proto.confirmatory_seeds(manifest)
    streams = proto.task_streams(manifest)
    arm_ids = [arm["arm_id"] for arm in manifest["arms"]]

    episodes = manifest["episodes_per_seed"]
    if a.probe:
        seeds = proto.probe_seeds(manifest)
        if not seeds:
            raise SystemExit("в манифесте нет seed_sets.fixture_probe — пробному прогону негде жить")
        episodes = PROBE_EPISODES
        print(f"\nFIXTURE PROBE: seeds {seeds}, {episodes} эпизодов. "
              f"Этот прогон не входит ни в один гейт и CSV не пишет.")
    else:
        seeds = parse_seeds(a.seeds) if a.seeds else all_seeds
        check_subset(seeds, all_seeds, "seeds")

    chosen_arms = arm_ids
    if a.arms:
        chosen_arms = [s.strip() for s in a.arms.split(",") if s.strip()]
        unknown = [s for s in chosen_arms if s not in arm_ids]
        if unknown:
            raise SystemExit(f"--arms: неизвестные arm_id {unknown}; протокол знает {arm_ids}")
        check_subset(chosen_arms, arm_ids, "arms")

    if a.probe:
        prov = None
    else:
        from fly_connectome_agent.src.engineering.logging.provenance_log import (
            ProvenanceLog, ProvenanceIntegrityError)
        prov = ProvenanceLog(str(a.provenance))
        try:
            prov.append({
                "kind": "p4_2_protocol",
                "protocol_id": manifest["protocol_id"],
                "manifest_digest": manifest["_digest"],
                "frozen_protocol_digest": proto.FROZEN_PROTOCOL_DIGEST,
                "supersedes_protocol_id": manifest.get("supersedes_protocol_id"),
                "result_bundle_directory": (manifest.get("result_bundle") or {}).get("directory"),
                "non_confirmatory": experimental,
                "seeds": [seeds[0], seeds[-1]] if seeds else [],
                "n_seeds": len(seeds),
                "episodes_per_seed": manifest["episodes_per_seed"],
                "arms": chosen_arms,
                "task_streams": manifest["task_streams"],
                "primary_metric": manifest["primary_metric"],
                "operating_point": manifest["operating_point"]["values"],
                "git": git_state(),
            })
        except ProvenanceIntegrityError as exc:
            # Отказ до первого же эпизода, а не через десять часов: прожигать
            # confirmatory-бюджет на журнале, который писатель продолжать не
            # собирается, нельзя. Тот же смысл кода возврата, что у остальных
            # отказов запуска и у гейта: 2 = «прогона не было», а не «не прошло».
            print(f"ЗАПУСК ОТКЛОНЁН: provenance-файл {a.provenance} нельзя "
                  f"продолжить: {exc}\nДля нового прогона укажите чистый путь, "
                  "для осознанного восстановления — вызовите "
                  "recover_head(reason=...) и перезапустите.", file=sys.stderr)
            return REFUSE

    code_arms = [codes[arm_id] for arm_id in chosen_arms]
    t0 = time.time()
    rows = p4.collect(code_arms, streams, seeds, episodes, settings, prov, a.verbose)
    elapsed = time.time() - t0

    if a.probe:
        summarise(rows, manifest, chosen_arms, streams, codes)
        print("\nprobe: результат не записан и не гейтится. Полный прогон: без --probe.")
        return 0

    out = pathlib.Path(a.out)
    p4.write_csv(rows, out)

    partial = set(seeds) != set(all_seeds) or set(chosen_arms) != set(arm_ids)
    sidecar = out.with_suffix(out.suffix + ".meta.json")
    sidecar.write_text(json.dumps({
        "protocol_id": manifest["protocol_id"],
        "manifest_digest": manifest["_digest"],
        "frozen_protocol_digest": proto.FROZEN_PROTOCOL_DIGEST,
        "supersedes_protocol_id": manifest.get("supersedes_protocol_id"),
        "result_bundle_directory": (manifest.get("result_bundle") or {}).get("directory"),
        "non_confirmatory": experimental,
        "manifest_path": str(pathlib.Path(a.manifest)),
        "partial": partial,
        "seeds_run": len(seeds),
        "seeds_declared": len(all_seeds),
        "arms_run": chosen_arms,
        "arms_declared": arm_ids,
        "task_streams_run": manifest["task_streams"],
        "episodes_per_seed": manifest["episodes_per_seed"],
        "operating_point_overrides": manifest["operating_point"].get("overrides", {}),
        "cells": len(rows),
        "episodes_total": len(rows) * manifest["episodes_per_seed"],
        "governance_violations": int(sum(r.get("governance_violations", 0) or 0 for r in rows)),
        "wall_clock_seconds": round(elapsed, 1),
        "git": git_state(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy_available": p4._HAVE_SCIPY,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"sidecar → {sidecar.name} (digest {manifest['_digest'][:19]}…, partial={partial}, "
          f"non_confirmatory={experimental})")

    if prov is not None:
        intact = prov.verify_chain()
        print(f"provenance chain: {'ok' if intact else 'НАРУШЕНА'}, записей: {prov.count}")

    summarise(rows, manifest, chosen_arms, streams, codes)
    print("\nВердикт считается отдельно: scripts/check_p4_2_oracle_gate.py "
          "(полный набор seed обязателен).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
