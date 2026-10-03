#!/usr/bin/env python3
"""Приёмка писателя провенанса: сколько стоит один append при разной длине журнала.

Зачем этот скрипт вообще существует: P4.2.v1 был прерван не из-за науки, а из-за
того, что append() перечитывал и парсил весь JSONL ради хвоста цепи. Стоимость
одной записи росла линейно с длиной журнала, суммарное время прогона — квадратично,
и 95% часов уходило на записью журнала.

Что меряется:
  уровни 0 / 1k / 10k / 50k уже записанных событий;
  для каждого уровня — медиана и p95 одного append нового пути и старого пути
  (старая реализация берётся из git-ревизии до исправления, на копии журнала;
  ревизия зашита в OLD_REV — иначе после коммита с правкой скрипт мерил бы
  новый путь дважды и отчитывался победой без соревнования);
  приёмочные границы: медиана на 50k не дороже трёх медиан пустого журнала,
  а спроецированные накладные расходы provenance на полный confirmatory-прогон
  меньше 15% времени симуляции.

Запуск:  .venv/bin/python scripts/bench_provenance_append.py
Результат: var/provenance_append_benchmark.json (+ таблица в stdout) — сырой
лог замера, он в .gitignore; и versioned summary
fly_connectome_agent/docs/benchmarks/provenance_writer_p4_2a.json — он, наоборот,
обязан жить в git: без него performance-приёмка переживает смену машины только
в пересказе. Каждый запуск дописывает в summary одну строку runs_detail и
пересчитывает агрегаты по всем накопленным запускам; --no-summary подавляет
это, --summary PATH кладёт в другое место.

Журналы пишутся в var/_bench/, старый путь проверяется на копиях и не остаётся;
ничего в прогонах и в репозитории этот скрипт не меняет.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import shutil
import statistics
import subprocess
import sys
import time
import types

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fly_connectome_agent.src.engineering.logging.provenance_log import (  # noqa: E402
    ProvenanceLog,
)

# Последняя ревизия, где append() всё ещё перечитывал журнал целиком. Зашита
# намеренно: HEAD со временем уезжает вперёд, а «старый путь» обязан оставаться
# именно тем кодом, из-за которого P4.2.v1 был прерван.
OLD_REV = "c54e8be63896826fe265ef928cd940bdc7689a26"
LEVELS = (0, 1_000, 10_000, 50_000)
SAMPLES_NEW = 200          # сколько append'ов мерить на новом пути
SAMPLES_OLD = 5            # старый путь дорог: 5 замеров на 50k ≈ 10 секунд
B_DIR = REPO / "var" / "_bench"
WRITER_SRC = (REPO / "fly_connectome_agent" / "src" / "engineering" / "logging" /
              "provenance_log.py")
ARCHIVE = REPO / "var" / "aborted" / "p4_2_v1_infrastructure_abort"
RUN_BUDGET_EPISODES = 86_400   # 6 arms x 60 seeds x 3 streams x 80 episodes (v1 = v2 по бюджету)

# Приёмочные границы — часть контракта, а не настройка запуска: менять их можно
# только вместе с протоколом, и summary обязан их печатать, чтобы читатель видел,
# с чем сравнивалось число.
GATE_NEW_GROWTH = 3.0
GATE_RUNTIME_SHARE = 0.15
BENCH_PROTOCOL_VERSION = "1.0"
SUMMARY_PATH = (REPO / "fly_connectome_agent" / "docs" / "benchmarks" /
                "provenance_writer_p4_2a.json")


def load_old_writer(rev: str = OLD_REV) -> types.ModuleType:
    """Дорефакторная реализация писателя из зашитой ревизии."""
    src = subprocess.run(
        ["git", "show", f"{rev}:fly_connectome_agent/src/engineering/logging/provenance_log.py"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout
    if "_resync_locked" in src or "HEAD_SUFFIX" in src:
        raise SystemExit(
            f"ревизия {rev} уже содержит исправленный писатель — сравнивать не с чем; "
            "укажите --old-rev на коммит до правки (для v1 это c54e8be)")
    if "_read_disk_events_unlocked" not in src:
        raise SystemExit(
            f"ревизия {rev} не похожа на дорефакторный писатель: не найден hot path "
            "_read_disk_events_unlocked — замер старого пути был бы ложью")
    mod = types.ModuleType("provenance_log_old")
    # регистрация в sys.modules обязательна: @dataclass(frozen=True) внутри
    # старый модуль резолвит свой cls.__module__ через sys.modules
    sys.modules["provenance_log_old"] = mod
    exec(compile(src, "provenance_log_old.py", "exec"), mod.__dict__)
    return mod


def build_log(path: pathlib.Path, entries: int) -> None:
    """Завести журнал длиной `entries` событий — быстрым новым путём."""
    for extra in (path, path.with_name(path.name + ".head.json")):
        if extra.exists():
            extra.unlink()
    log = ProvenanceLog(str(path))
    for i in range(entries):
        log.append({"i": i, "pad": "x" * 24})
    assert log.count == entries, f"журнал не добрал до {entries}: {log.count}"


def time_appends(writer: types.ModuleType | None, path: pathlib.Path, samples: int,
                 tag_start: int) -> dict:
    cls = writer.ProvenanceLog if writer is not None else ProvenanceLog
    log = cls(str(path))
    timings = []
    for i in range(samples):
        t0 = time.perf_counter()
        log.append({"bench": tag_start + i})
        timings.append(time.perf_counter() - t0)
    ordered = sorted(timings)
    return {
        "samples": samples,
        "median_ms": round(ordered[len(ordered) // 2] * 1000, 4),
        "p95_ms": round(ordered[int(len(ordered) * 0.95) - 1] * 1000, 4),
        "min_ms": round(ordered[0] * 1000, 4),
    }


def sim_only_seconds_per_episode() -> tuple[float, str]:
    """Стоимость одной симуляции без provenance — из архива прерванного прогона.

    Берётся самый первый интервал лога (20 ячеек = 1600 эпизодов), когда журнал был
    короток, и из него вычитается измеренная стоимость append старого пути на
    середине этого диапазона. Оценка грубая, но она и не должна быть точной: важна
    доля накладных расходов, а она меняется на порядки.
    """
    fallback = (0.167, "паспортная оценка из интервала 0–20 ячеек, если архива нет")
    table_path = ARCHIVE / "append_latency_table.json"
    if not table_path.exists():
        return fallback
    table = json.loads(table_path.read_text(encoding="utf-8"))
    rows = table.get("темп_прогона_из_лога") or []
    if not rows:
        return fallback
    first = rows[0]
    episodes = first["cells_done"] * 80
    per_episode = first["elapsed_s"] / episodes
    # старый append линеен по длине журнала: 0.679 с при 16 622 записях
    old_at = table["прямое_измерение_append_на_копии"]["медиана_16622_с"] / 16_622
    middle = per_episode - old_at * episodes / 2
    if middle <= 0:
        return fallback
    return round(middle, 4), (
        f"интервал {first['cells_done']} ячеек за {first['elapsed_s']}с ⇒ {per_episode:.4f} с/эпизод, "
        f"минус аппроксимация старого append ({old_at * 1000:.4f} мс/запись × {episodes / 2} записей в среднем)")


def git_state() -> dict:
    """Что именно меряется: HEAD и совпадает ли измеряемый писатель с коммитом.

    Глобальное «рабочее дерево грязное» здесь бесполезно: summary, в которое этот
    скрипт сам же и пишет, делает дерево грязным на любом запуске. Значим ровно
    один файл — тот, чья скорость измеряется.
    """
    rel = str(WRITER_SRC.relative_to(REPO))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    same = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel],
                          cwd=REPO).returncode == 0
    return {"head": head, "writer_matches_committed_source": same}


def aggregate(detail: list[dict]) -> dict:
    """Агрегаты summary по всем записанным запускам.

    Худшие из измеренных значений — максимум роста и максимум доли, а не средние:
    среднее прячет единственный плохой прогон, а приёмка тем и отличается от
    отчёта, что проваливается на одном плохом числе.
    """
    growths = [r["new_growth_ratio"] for r in detail]
    old_growths = [r["old_growth_ratio"] for r in detail]
    shares = [r["provenance_runtime_fraction"] for r in detail]
    return {
        "benchmark_protocol_version": BENCH_PROTOCOL_VERSION,
        "what_this_proves": (
            "append() в журнале провенанса стоит O(1) по длине журнала: медиана "
            "одной записи на 50 000 записей не дороже медианы на пустом журнале "
            "более чем втрое, а спроецированные накладные расходы провенанса на "
            "полный confirmatory-прогон меньше 15% времени симуляции"),
        "old_revision": OLD_REV,
        "fixed_revision": detail[-1]["writer_revision"]["head"],
        "writer_revisions": sorted({r["writer_revision"]["head"] for r in detail}),
        "runs": len(detail),
        "prior_entry_counts": list(LEVELS),
        "new_growth_ratio_max": max(growths),
        "old_growth_ratio_range": [min(old_growths), max(old_growths)],
        "acceptance_max_new_growth_ratio": GATE_NEW_GROWTH,
        "provenance_runtime_fraction": max(shares),
        "acceptance_max_fraction": GATE_RUNTIME_SHARE,
        "result": "pass" if all(r["result"] == "pass" for r in detail) else "fail",
    }


def record_run(summary_path: pathlib.Path, run: dict) -> dict | None:
    """Добавить один замер в versioned summary и пересчитать агрегаты по всем."""
    if run["prior_entry_counts"] != list(LEVELS):
        print("summary не обновлён: прогон покрыл не все уровни приёмки (--quick), "
              f"а граница меряется на {LEVELS[-1]} записях")
        return None
    detail: list[dict] = []
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        version = existing.get("benchmark_protocol_version")
        if version != BENCH_PROTOCOL_VERSION:
            raise SystemExit(
                f"в {summary_path} записан протокол бенчмарка {version!r}, а не "
                f"{BENCH_PROTOCOL_VERSION!r}: смешивать замеры разных протоколов "
                "в одном агрегате нельзя — заведите новый файл")
        detail = existing.get("runs_detail") or []
    run = {"run": len(detail) + 1, **run}
    detail.append(run)
    summary = aggregate(detail)
    summary["raw_local_artifact"] = (
        "var/provenance_append_benchmark.json — сырые медианы/p95 этого запуска, "
        "в .gitignore; воспроизводится этим скриптом")
    summary["how_to_refresh"] = (
        ".venv/bin/python scripts/bench_provenance_append.py "
        "(дописывает runs_detail и пересчитывает агрегаты)")
    summary["runs_detail"] = detail
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="только уровни 0 и 1k (для отладки)")
    ap.add_argument("--old-rev", default=OLD_REV,
                    help="ревизия с дорефакторным писателем (по умолчанию зашита)")
    ap.add_argument("--summary", default=SUMMARY_PATH, type=pathlib.Path,
                    help="versioned summary, куда дописывается этот замер")
    ap.add_argument("--no-summary", action="store_true",
                    help="только var-артефакт, summary не трогать")
    a = ap.parse_args()
    levels = LEVELS[:2] if a.quick else LEVELS

    old = load_old_writer(a.old_rev)
    if B_DIR.exists():
        shutil.rmtree(B_DIR)
    B_DIR.mkdir(parents=True)

    rows = []
    for level in levels:
        log_path = B_DIR / f"log_{level}.jsonl"
        print(f"уровень {level}: заведение журнала…", flush=True)
        build_log(log_path, level)

        new = time_appends(None, log_path, SAMPLES_NEW, level * 1000)

        scratch = B_DIR / f"log_{level}_old.jsonl"
        shutil.copy2(log_path, scratch)
        head = scratch.with_name(scratch.name + ".head.json")
        if head.exists():
            head.unlink()          # старый путь не знает про witness — не мешаем ему
        old_row = time_appends(old, scratch, SAMPLES_OLD, level * 1000 + 7)
        scratch.unlink()
        for extra in scratch.parent.glob(f"log_{level}_old.jsonl*"):
            extra.unlink()

        ratio = round(old_row["median_ms"] / new["median_ms"], 1)
        rows.append({"prior_entries": level, "new": new, "old": old_row,
                     "old_over_new_median": ratio})
        print(f"  новый {new['median_ms']:.3f} мс (p95 {new['p95_ms']:.3f}), "
              f"старый {old_row['median_ms']:.3f} мс, разница ×{ratio}", flush=True)

    empty = rows[0]["new"]["median_ms"]
    top = rows[-1]["new"]["median_ms"]
    growth = round(top / empty, 2)
    sim_s, sim_note = sim_only_seconds_per_episode()
    overhead_total = top / 1000 * RUN_BUDGET_EPISODES
    sim_total = sim_s * RUN_BUDGET_EPISODES
    share = overhead_total / (overhead_total + sim_total)

    verdict = {
        "уровни": [r["prior_entries"] for r in rows],
        "медиана_пустого_мс": empty,
        f"медиана_{rows[-1]['prior_entries']}_мс": top,
        "рост_медианы_в_раза": growth,
        "граница_роста": GATE_NEW_GROWTH,
        "рост_в_порядке": growth <= GATE_NEW_GROWTH,
        "оценка_симуляции_с_на_эпизод": sim_s,
        "откуда_оценка": sim_note,
        "проект_провенанс_секунд_на_полный_прогон": round(overhead_total),
        "проект_симуляция_секунд_на_полный_прогон": round(sim_total),
        "доля_провенанса": round(share, 4),
        "граница_доли": GATE_RUNTIME_SHARE,
        "доля_в_порядке": share <= GATE_RUNTIME_SHARE,
        "старый_путь_в_архиве": "0.679 с на append при 16 622 записях (замер на копии живого журнала)",
    }

    state = git_state()
    out = {
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": platform.platform(),
        "python": platform.python_version(),
        "commit_under_test": state["head"],
        "writer_matches_committed_source": state["writer_matches_committed_source"],
        "samples_new_path": SAMPLES_NEW,
        "samples_old_path": SAMPLES_OLD,
        "levels": rows,
        "acceptance": verdict,
    }
    (REPO / "var" / "provenance_append_benchmark.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    both_ok = verdict["рост_в_порядке"] and verdict["доля_в_порядке"]
    run = {
        "measured_utc": out["measured_utc"],
        "machine": out["machine"],
        "python": out["python"],
        "writer_revision": state,
        "old_revision": a.old_rev,
        "prior_entry_counts": [r["prior_entries"] for r in rows],
        "new_median_ms_by_level": [r["new"]["median_ms"] for r in rows],
        "old_median_ms_by_level": [r["old"]["median_ms"] for r in rows],
        "new_growth_ratio": growth,
        "old_growth_ratio": round(rows[-1]["old"]["median_ms"] / rows[0]["old"]["median_ms"], 2),
        "provenance_runtime_fraction": round(share, 4),
        "sim_only_seconds_per_episode": sim_s,
        "samples_new_path": SAMPLES_NEW,
        "samples_old_path": SAMPLES_OLD,
        "result": "pass" if both_ok else "fail",
    }

    print(f"\n{'записей до':>11} | {'новый, мс':>10} | {'старый, мс':>11} | "
          f"{'во сколько раз':>14}")
    for r in rows:
        print(f"{r['prior_entries']:>11} | {r['new']['median_ms']:>10.3f} | "
              f"{r['old']['median_ms']:>11.3f} | {r['old_over_new_median']:>10}")
    print(f"\nприёмка: рост медианы ×{growth} (граница {GATE_NEW_GROWTH}) — "
          f"{'ПРОЙДЕНО' if verdict['рост_в_порядке'] else 'ПРОВАЛЕНО'}; "
          f"доля provenance в полном прогоне {share * 100:.1f}% (граница {GATE_RUNTIME_SHARE:.0%}) — "
          f"{'ПРОЙДЕНО' if verdict['доля_в_порядке'] else 'ПРОВАЛЕНО'}")
    shutil.rmtree(B_DIR)  # журналы-песочница восстановимы запуском, не оставляем 25 МБ
    if a.no_summary:
        print("summary не тронут (--no-summary)")
    else:
        summary = record_run(a.summary, run)
        if summary is not None:
            print(f"summary: {a.summary.relative_to(REPO)} — запусков {summary['runs']}, "
                  f"результат {summary['result']}, худший рост ×{summary['new_growth_ratio_max']}, "
                  f"худшая доля {summary['provenance_runtime_fraction']:.1%}")
    return 0 if both_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
