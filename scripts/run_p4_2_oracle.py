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
        --out var/p4_2_oracle.csv --provenance var/p4_2_oracle.prov.jsonl
По шардам (пример — по 3 arms):
    ... --arms rstdp_mixed,no_plasticity,m_zero --out var/p4_2_oracle_a.csv
    ... --arms weight_shuffled_frozen,direction_shuffled_frozen,oracle_reflex \
        --out var/p4_2_oracle_b.csv
Проверка инструмента без прогона:
    ... --describe      (план и digest, ни одного эпизода)
    ... --probe         (seed'ы 900-903, вне любых гейтов, 8 эпизодов)
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

DEFAULT_OUT = REPO / "var" / "p4_2_oracle.csv"
DEFAULT_PROV = REPO / "var" / "p4_2_oracle.prov.jsonl"
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
            f"digest манифеста {digest} не равен замороженному "
            f"{proto.FROZEN_PROTOCOL_DIGEST}. Правка протокола после заморозки — это уже "
            "не P4.2; для отладочного прогона нужны --experimental-manifest и другой --out.")
    if pathlib.Path(out_path).resolve() == pathlib.Path(DEFAULT_OUT).resolve():
        raise Refused(f"experimental-прогон не может писать в confirmatory путь {DEFAULT_OUT}")
    return True


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
        from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog
        prov = ProvenanceLog(str(a.provenance))
        prov.append({
            "kind": "p4_2_protocol",
            "protocol_id": manifest["protocol_id"],
            "manifest_digest": manifest["_digest"],
            "frozen_protocol_digest": proto.FROZEN_PROTOCOL_DIGEST,
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
