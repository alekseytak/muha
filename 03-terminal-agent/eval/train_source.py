#!/usr/bin/env python3
"""Откуда агент берёт обучающие метки и насколько это меняет приёмку.

Проблема: живой агент обучался на `eval/cases.json` (39 фраз), а все
опубликованные замеры сделаны на `eval/cases_distill.json` (177 обучающих
фраз при делении по семействам). Значит опубликованное качество описывает не
тот каскад, который запускается.

Скрипт обучает один и тот же агент из двух источников и меряет на трёх
наборах, которые НЕ участвуют в обучении:

* `eval/cases_holdout` — те 152 фразы, на которых считались публичные числа;
* `eval/gold_proposals.tsv` — 47 строк, где предложено действие;
* `eval/igor_risk_review.tsv` — 34 выборочно проверенные строки.

Ни один из этих наборов в обучение не попадает.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.agent import Agent                      # noqa: E402
from core.verb_act import VerbActGrammar          # noqa: E402

SIDE_EFFECT = {"write", "execute", "veto", "anchor", "log", "select"}


def load_pairs(name: str) -> list:
    payload = json.loads((ROOT / "eval" / name).read_text(encoding="utf-8"))
    cases = payload.get("cases") or payload
    pairs = []
    for case in cases:
        phrase = case.get("prompt") or case.get("phrase") or ""
        verbs = case.get("verbs") or case.get("expected_verbs") or []
        if phrase and verbs:
            pairs.append((phrase, verbs[0]))
    return pairs


def distill_train_split() -> list:
    """Те же 177 фраз, что в экспериментах: чётные индексы семейств."""
    pairs = load_pairs("cases_distill.json")
    return [p for i, p in enumerate(pairs) if i % 2 == 0]


def holdout_152() -> list:
    """Нечётные семейства — те же 152 фразы, что в eval/hierarchical.py."""
    payload = json.loads((ROOT / "eval" / "cases_distill.json")
                         .read_text(encoding="utf-8"))
    cases = payload.get("cases") or payload
    families = sorted({c["family"] for c in cases if c.get("family")})
    train = {n for i, n in enumerate(families) if i % 2 == 0}
    return [c for c in cases if c.get("family") not in train]


def build(training: list) -> Agent:
    agent = Agent(config_path=str(ROOT / "config.yaml"),
                  confirm_policy="allow")
    agent.cascade.fit(training)
    return agent


def measure_holdout(agent: Agent) -> dict:
    rows = holdout_152()
    verb_ok = domain_ok = both = answered = 0
    for case in rows:
        phrase = case.get("prompt") or case.get("phrase") or ""
        want_verbs = case.get("verbs") or case.get("expected_verbs") or []
        want_domains = case.get("domains") or case.get("expected_domains") or []
        act = agent.propose_action_fast(phrase)
        if act is None:
            continue
        answered += 1
        got_verb = act.verb.value
        got_domain = VerbActGrammar.domain_for_object(act.object or "")
        v_ok = bool(want_verbs) and got_verb == want_verbs[0]
        d_ok = bool(want_domains) and got_domain == want_domains[0]
        verb_ok += v_ok
        domain_ok += d_ok
        both += v_ok and d_ok
    total = len(rows)
    return {
        "cases": total,
        "answered": answered,
        "coverage": round(answered / total, 3) if total else 0.0,
        "verb_accuracy_of_all": round(verb_ok / total, 3) if total else 0.0,
        "domain_accuracy_of_all": round(domain_ok / total, 3) if total else 0.0,
        "joint_of_all": round(both / total, 3) if total else 0.0,
    }


def measure_proposals(agent: Agent) -> dict:
    props = list(csv.DictReader(
        (ROOT / "eval" / "gold_proposals.tsv").open(encoding="utf-8"),
        delimiter="\t"))
    act_rows = [r for r in props if r["outcome"] == "act"]
    full = 0
    for row in act_rows:
        act = agent.propose_action_fast(row["prompt"])
        if act is None or act.verb.value != row["verb"]:
            continue
        want_obj = (row["object"] or "").strip()
        got_obj = (act.object or "").strip()
        if want_obj.startswith("web:"):
            want_obj = want_obj[4:]
        if got_obj.startswith("web:"):
            got_obj = got_obj[4:]
        if want_obj and got_obj != want_obj:
            continue
        full += 1
    no_act = [r for r in props if r["outcome"] != "act"]
    dangerous = 0
    for row in no_act:
        act = agent.propose_action_fast(row["prompt"])
        if act is not None and act.verb.value in SIDE_EFFECT:
            dangerous += 1
    return {"act_rows": len(act_rows), "full_match": full,
            "dangerous": dangerous, "no_act_rows": len(no_act)}


def measure_review(agent: Agent) -> dict:
    review = list(csv.DictReader(
        (ROOT / "eval" / "igor_risk_review.tsv").open(encoding="utf-8"),
        delimiter="\t"))
    props = {r["id"]: r for r in csv.DictReader(
        (ROOT / "eval" / "gold_proposals.tsv").open(encoding="utf-8"),
        delimiter="\t")}
    ok = 0
    for row in review:
        phrase = props[row["id"]]["prompt"]
        act = agent.propose_action_fast(phrase)
        got = act.verb.value if act else None
        if row["outcome"] == "act":
            hit = got == row["verb"]
        else:
            hit = got is None or got not in SIDE_EFFECT
        ok += hit
    return {"rows": len(review), "matched": ok}


def main() -> int:
    sources = {
        "cases.json (39) — как сейчас": load_pairs("cases.json"),
        "cases_distill train split (177)": distill_train_split(),
    }
    print("обучающих пар:", {k: len(v) for k, v in sources.items()})
    print()
    report = {}
    for label, training in sources.items():
        agent = build(training)
        hold = measure_holdout(agent)
        prop = measure_proposals(agent)
        rev = measure_review(agent)
        report[label] = {"holdout": hold, "proposals": prop, "review": rev}
        print(f"--- {label}")
        print(f"    152 holdout: покрытие {hold['coverage']:.1%}, "
              f"глагол {hold['verb_accuracy_of_all']:.3f}, "
              f"глагол+домен {hold['joint_of_all']:.3f}")
        print(f"    предложения: {prop['full_match']}/{prop['act_rows']} полных, "
              f"опасных {prop['dangerous']}")
        print(f"    проверка Игоря: {rev['matched']}/{rev['rows']}")
    out = ROOT / "eval" / "train_source_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\nотчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())