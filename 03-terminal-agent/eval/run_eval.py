#!/usr/bin/env python3
"""Eval: насколько точно модель превращает фразу в VerbAct.

Что измеряем: только РАЗБОР (глагол + домен + объект). Действия не выполняются —
гейт, песочница и реестр в этом прогоне не участвуют.

Воспроизводимость фиксируется в отчёте:
  * sha256 файла датасета (dataset_digest);
  * провайдер, модель, base_url;
  * temperature, max_tokens, seed (если провайдер его принимает);
  * точное время старта и общее число запросов.

Запуск:
  python3 eval/run_eval.py                                  # дефолт из config.yaml
  python3 eval/run_eval.py --provider literouter \\
      --base-url https://api.literouter.com/v1 \\
      --model deepseek-v3.2:free --key-env LITEROUTER_API_KEY
  python3 eval/run_eval.py --limit 5 --no-save              # быстрая проверка
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.agent import Agent                      # noqa: E402
from core.llm_client import LLMClient, LLMUnavailable  # noqa: E402

CASES_PATH = ROOT / "eval" / "cases.json"          # обучающий набор
HOLDOUT_PATH = ROOT / "eval" / "cases_holdout.json"  # контрольный набор
RESULTS_DIR = ROOT / "eval" / "results"


def dataset_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "cases" not in data:
        raise SystemExit(f"датасет повреждён: {path}")
    ids = [case["id"] for case in data["cases"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("в датасете повторяются id кейсов")
    return data


def build_client(args: argparse.Namespace, fallback_config: dict) -> LLMClient:
    llm_cfg = (fallback_config.get("llm") or {})
    provider = args.provider or llm_cfg.get("provider", "ollama")
    base_url = args.base_url or llm_cfg.get("base_url")
    model = args.model or llm_cfg.get("model")
    key_env = args.key_env or llm_cfg.get("api_key_env")
    credentials = args.credentials or llm_cfg.get("credentials_file")
    return LLMClient(
        base_url=base_url,
        model=model,
        provider=provider,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        api_key_env=key_env,
        credentials_file=credentials,
        seed=args.seed,
    )


def run_one(agent: Agent, client: LLMClient, case: Dict[str, Any],
            system_prompt: str) -> Dict[str, Any]:
    """Один кейс: запрос к модели + разбор тем же парсером, что и в бою."""
    started = time.time()
    record: Dict[str, Any] = {
        "id": case["id"],
        "prompt": case["prompt"],
        "expected_verbs": case["verbs"],
        "expected_domains": case["domains"],
    }
    try:
        raw = client.chat(case["prompt"], system_prompt=system_prompt)
    except LLMUnavailable as exc:
        record.update(
            latency_s=round(time.time() - started, 2),
            error=f"{type(exc).__name__}: {exc}",
            verb=None, domain=None, verb_ok=False, domain_ok=False, ok=False,
        )
        return record

    latency = round(time.time() - started, 2)
    verb_act = agent._parse_verb_act(raw, case["prompt"])
    verb = verb_act.verb.value
    domain = verb_act.domain or "ui"
    record.update(
        raw=raw[:400],
        verb=verb,
        domain=domain,
        object=verb_act.object[:200],
        parse_reason=verb_act.parse_reason,
        latency_s=latency,
        fallback_to_think=(verb == "think" and "think" not in case["verbs"]),
        verb_ok=verb in case["verbs"],
        domain_ok=domain in case["domains"],
        ok=(verb in case["verbs"]) and (domain in case["domains"]),
    )
    return record


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    answered = [r for r in rows if not r.get("error")]
    ok = [r for r in rows if r.get("ok")]
    verb_ok = [r for r in rows if r.get("verb_ok")]
    domain_ok = [r for r in rows if r.get("domain_ok")]
    latencies = [r["latency_s"] for r in answered if r.get("latency_s")]

    per_verb: Dict[str, Dict[str, int]] = {}
    for row in answered:
        expected = row["expected_verbs"][0]
        bucket = per_verb.setdefault(expected, {"cases": 0, "ok": 0})
        bucket["cases"] += 1
        bucket["ok"] += 1 if row["ok"] else 0

    return {
        "cases": total,
        "answered": len(answered),
        "errors": total - len(answered),
        "verb_accuracy": round(len(verb_ok) / total, 3) if total else 0.0,
        "domain_accuracy": round(len(domain_ok) / total, 3) if total else 0.0,
        "joint_accuracy": round(len(ok) / total, 3) if total else 0.0,
        "think_fallbacks": sum(1 for r in answered if r.get("fallback_to_think")),
        "latency_mean_s": round(statistics.mean(latencies), 2) if latencies else None,
        "latency_median_s": round(statistics.median(latencies), 2) if latencies else None,
        "per_expected_verb": per_verb,
    }


def render_markdown(report: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    meta = report["meta"]
    summary = report["summary"]
    lines = [
        f"# Eval verb-parser — {meta['started_utc']}",
        "",
        "| параметр | значение |",
        "|---|---|",
        f"| провайдер | {meta['provider']} |",
        f"| модель | {meta['model']} |",
        f"| base_url | {meta['base_url']} |",
        f"| temperature / max_tokens / seed | {meta['temperature']} / {meta['max_tokens']} / {meta['seed']} |",
        f"| системный промпт | {'компактный' if meta.get('compact') else 'полный'} ({meta['system_prompt_chars']} симв.) |",
        f"| набор | {meta['dataset']} |",
        f"| dataset_digest | `{meta['dataset_digest'][:16]}…` |",
        f"| кейсов | {summary['cases']} |",
        "",
        "| метрика | значение |",
        "|---|---|",
        f"| точность глагола | {summary['verb_accuracy']:.1%} |",
        f"| точность домена | {summary['domain_accuracy']:.1%} |",
        f"| глагол И домен | {summary['joint_accuracy']:.1%} |",
        f"| отказов в THINK | {summary['think_fallbacks']} |",
        f"| ошибок провайдера | {summary['errors']} |",
        f"| задержка, медиана / среднее | {summary['latency_median_s']} / {summary['latency_mean_s']} с |",
        "",
        "| ожидаемый глагол | кейсов | верно | доля |",
        "|---|---|---|---|",
    ]
    for verb, bucket in sorted(summary["per_expected_verb"].items()):
        share = bucket["ok"] / bucket["cases"] if bucket["cases"] else 0
        lines.append(f"| {verb} | {bucket['cases']} | {bucket['ok']} | {share:.0%} |")

    failures = [r for r in rows if not r.get("ok")]
    if failures:
        lines += ["", "## Провалы", "", "| кейс | фраза | ожидалось | получено |", "|---|---|---|---|"]
        for row in failures:
            got = f"{row.get('verb')}/{row.get('domain')}"
            want = f"{'/'.join(row['expected_verbs'])}/{'/'.join(row['expected_domains'])}"
            phrase = row["prompt"].replace("|", "\\|")
            lines.append(f"| {row['id']} | {phrase} | {want} | {got} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval разбора глаголов")
    parser.add_argument("--provider")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--key-env", dest="key_env")
    parser.add_argument("--credentials")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128, dest="max_tokens")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--compact", action="store_true",
                        help="использовать компактный промпт шага-продолжения")
    parser.add_argument("--holdout", action="store_true",
                        help="прогнать контрольный набор (не участвовал в правке промпта)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cases_path = HOLDOUT_PATH if args.holdout else CASES_PATH
    dataset = load_cases(cases_path)
    cases = dataset["cases"]
    if args.limit:
        cases = cases[: args.limit]

    agent = Agent(config_path=str(ROOT / "config.yaml"))
    system_prompt = (
        agent.compact_planning_prompt() if args.compact
        else agent._build_system_prompt()          # ровно тот промпт, что в бою
    )
    client = build_client(args, agent.config)

    if not client.api_key:
        raise SystemExit(
            f"нет ключа для {client.provider}: задайте {client.api_key_env} "
            f"или укажите --credentials"
        )

    print(f"провайдер: {client.provider} | модель: {client.model} | кейсов: {len(cases)}")
    digest = dataset_digest(cases_path)
    print(f"набор: {cases_path.name} | dataset_digest: {digest[:16]}…\n", flush=True)

    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        row = run_one(agent, client, case, system_prompt)
        rows.append(row)
        mark = "ok  " if row.get("ok") else "FAIL"
        got = row.get("verb") or row.get("error", "")[:40]
        print(f"[{index:2d}/{len(cases)}] {mark} {case['id']:18s} -> {got}/{row.get('domain')}"
              f"  ({row.get('latency_s')}s)", flush=True)

    summary = summarize(rows)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = {
        "meta": {
            "dataset": str(cases_path.relative_to(ROOT)),
            "dataset_digest": digest,
            "provider": client.provider,
            "model": client.model,
            "base_url": client.base_url,
            "temperature": client.temperature,
            "max_tokens": client.max_tokens,
            "seed": client.seed,
            "started_utc": started,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "system_prompt_chars": len(system_prompt),
            "compact": bool(args.compact),
            "system_prompt_source": "core/agent.py::_build_system_prompt",
            "parser": "core/agent.py::_parse_verb_act",
        },
        "summary": summary,
        "rows": rows,
    }

    print()
    print(render_markdown(report, rows))

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_model = client.model.replace("/", "_").replace(":", "-")
        tag = "holdout-" if args.holdout else ""
        out = RESULTS_DIR / f"{tag}{started.replace(':', '')}_{safe_model}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nотчёт сохранён: {out}")

    return 0 if summary["joint_accuracy"] >= 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())