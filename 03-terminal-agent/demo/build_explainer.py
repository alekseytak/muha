#!/usr/bin/env python3
"""Собрать наглядную страницу о том, как муха принимает решение.

Страница — один файл `demo/muha.html`, без внешних ресурсов, без сети, без
вызовов модели: её можно открыть двойным щелчком и показать кому угодно.

Данные не выдуманы. Для каждой фразы скрипт:

1. прогоняет её через НАСТОЯЩИЙ `Agent.propose_action_fast`;
2. отдельно повторяет тот же конвейер по шагам, вызывая те же функции из
   `core/readiness.py`, и записывает вердикт каждого шага;
3. сверяет итог трассы с настоящим ответом и, если они расходятся, помечает
   строку как `trace_mismatch` — страница покажет это, а не спрячет;
4. где действие безопасно и локально (файлы), реально исполняет его во
   временной песочнице и показывает вывод.

Никаких обращений к LLM: омега только помечается как «сюда ушло бы».
"""

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
sys.path.insert(0, str(ROOT))

from core.agent import Agent                              # noqa: E402
from core.micromodels import cue_verb                     # noqa: E402
from core.readiness import (                              # noqa: E402
    DEICTIC_RE, NEGATION_IMPERSONAL_RE, NEGATION_STEM_RE, SHELL_META_RE,
    UNSUPPORTED_RE, SEQUENCE_RE, action_targets, canonicalize_object,
    check_readiness, distinct_action_verbs, forbidden_url, intent_guard,
    object_of_registry,
)
from core.registry import Registry                        # noqa: E402


# Фразы подобраны так, чтобы был виден весь спектр: успех, отказ, уточнение.
PHRASES = [
    # успешные локальные действия
    "Прочитай файл todo.md",
    "Покажи содержимое файла plans.txt",
    "Запиши в todo.md текст купить хлеб",
    "Запусти команду wc -l todo.md",
    "Залогируй в reg: заметку про встречу",
    "Посоветуй, что делать с отчётом",
    "Выбери вариант: чай или кофе",
    # отказы по смыслу
    "Не записывай ничего в todo.md",
    "Не сохраняй ответ в plans.txt",
    "Сравни содержимое old.txt и new.txt",
    "Переименуй todo.md в done.md",
    # небезопасные адреса
    "Прочитай http://127.0.0.1:8080/",
    "Открой file:///etc/passwd",
    # недостающие данные
    "Сохрани результат",
    "Допиши в todo.md строку ещё одна",
    "Сделай с этим что-нибудь полезное",
    # оболочка и несколько действий
    "Запусти pwd; затем uname -a",
    "Прочитай todo.md и запиши копию в backup.md",
    # версия: закрытый G075
    "Какая версия у файла todo.md?",
    # вето и эскалация
    "Наложи вето на запись в plans.txt",
    "Позови человека",
]

STAGES = [
    ("cascade", "Каскад микромоделей", "μ1a правила → μ1b n-граммы → μ1c объект → μ2 домен"),
    ("intent", "Намерение", "отрицание, вопрос о действии, пример в кавычках"),
    ("negation", "Отрицание", "«не сохраняй» — это запрет, а не приказ"),
    ("shell", "Оболочка", "цепочки, конвейеры, подстановки не поддерживаются"),
    ("unsupported", "Глагол вне набора", "сравни, отмени, переименуй, скопируй, удали"),
    ("deictic", "Местоимение без объекта", "«с этим», «его», «там» — нужен предмет"),
    ("multi", "Несколько действий", "два действия локально не исполнить"),
    ("tool", "Инструмент существует", "пара «глагол/домен» должна быть реализована"),
    ("params", "Обязательные параметры", "путь, содержимое, команда, варианты"),
]


def make_agent(tmp: Path) -> Agent:
    """Агент без модели: только локальный разбор."""
    agent = Agent(config_path=str(ROOT / "config.yaml"), confirm_policy="allow")
    agent.set_sandbox_root(tmp)
    agent.registry = Registry(path=str(tmp / "registry.jsonl"))
    agent.llm = None
    agent.fallback_llm = None
    (tmp / "todo.md").write_text("купить хлеб\n", encoding="utf-8")
    (tmp / "plans.txt").write_text("план на неделю\n", encoding="utf-8")
    (tmp / "old.txt").write_text("старое\n", encoding="utf-8")
    (tmp / "new.txt").write_text("новое\n", encoding="utf-8")
    return agent


def trace(agent: Agent, phrase: str) -> dict:
    """Повторить конвейер по шагам теми же функциями, что и в агенте."""
    steps = []

    def step(key, verdict, reason):
        steps.append({"key": key, "verdict": verdict, "reason": reason})

    decision = agent.cascade.decide(phrase)
    if decision is None:
        step("cascade", "stop", "каскад воздержался: нет уверенного глагола")
        return {"steps": steps, "decision": None}
    verb = agent._coerce_verb(decision.verb)
    if verb is None:
        step("cascade", "stop", f"глагол «{decision.verb}» вне набора")
        return {"steps": steps, "decision": None}
    verb_name = verb.value if hasattr(verb, "value") else str(verb)
    step("cascade", "pass",
         f"{verb_name} ← {decision.source}, уверенность {decision.confidence:.2f}")

    allowed, why = intent_guard(phrase, verb_name)
    if not allowed:
        step("intent", "stop", why)
        return {"steps": steps, "decision": None}
    step("intent", "pass", "фраза просит действие")

    if (NEGATION_STEM_RE.search(phrase)
            or NEGATION_IMPERSONAL_RE.search(phrase)):
        step("negation", "stop", "отрицание: фраза запрещает действие")
        return {"steps": steps, "decision": None}
    step("negation", "pass", "отрицания нет")

    if SHELL_META_RE.search(phrase):
        step("shell", "stop", "оболочка: цепочка или подстановка")
        return {"steps": steps, "decision": None}
    step("shell", "pass", "оболочечных конструкций нет")

    if UNSUPPORTED_RE.search(phrase):
        step("unsupported", "stop", "глагол вне набора одиннадцати")
        return {"steps": steps, "decision": None}
    step("unsupported", "pass", "глагол поддержан")

    if DEICTIC_RE.search(phrase) and not action_targets(phrase):
        step("deictic", "stop", "местоимение без предмета")
        return {"steps": steps, "decision": None}
    step("deictic", "pass", "объект назван")

    several = distinct_action_verbs(phrase, cue_verb)
    if (len(several) > 1 or len(action_targets(phrase)) > 1
            or SEQUENCE_RE.search(phrase)):
        step("multi", "stop",
             f"действий несколько: {', '.join(several) or 'разные цели'}")
        return {"steps": steps, "decision": None}
    step("multi", "pass", "одно действие")

    act = agent._make_act(verb, decision.object or phrase, {}, phrase, "trace")
    if act.domain == "ui" and decision.domain:
        act.domain = decision.domain
    if act.domain == "registry" and verb_name == "version":
        act.object = object_of_registry(phrase) or act.object
    act.object = canonicalize_object(act.domain, act.object)

    if agent.tools.find(verb, act.domain) is None:
        step("tool", "stop", f"нет инструмента для {verb_name}/{act.domain}")
        return {"steps": steps, "decision": None}
    step("tool", "pass", f"есть {verb_name}/{act.domain}")

    forbidden = forbidden_url(phrase)
    ruled_verb, _ = cue_verb(phrase)
    ready, params, why = check_readiness(
        verb_name, act.domain, phrase, act.object,
        confirmed_by_cue=(ruled_verb == verb_name))
    if not ready:
        step("params", "stop", why)
        return {"steps": steps, "decision": None}
    step("params", "pass",
         "параметры на месте" + (f" ({', '.join(params)})" if params else ""))
    return {"steps": steps, "decision": {"verb": verb_name,
                                         "domain": act.domain,
                                         "object": act.object,
                                         "params": params,
                                         "source": decision.source,
                                         "confidence": decision.confidence}}


def collect() -> list:
    rows = []
    tmp = Path(tempfile.mkdtemp(prefix="muha-explainer-"))
    agent = make_agent(tmp)
    for phrase in PHRASES:
        real = agent.propose_action_fast(phrase)
        info = trace(agent, phrase)
        real_verb = None
        if real is not None:
            real_verb = real.verb.value if hasattr(real.verb, "value") else str(real.verb)
        traced = info["decision"]
        traced_verb = (traced or {}).get("verb")
        # Сверяется итог: есть действие или нет, и каким глаголом.
        mismatch = real_verb != traced_verb

        executed = None
        if real is not None:
            if real.domain in ("filesystem", "registry", "system"):
                try:
                    result = agent.execute(real)
                    executed = {"output": str(result)[:600],
                                "ok": bool(getattr(real, "success", False))}
                except Exception as exc:                   # noqa: BLE001
                    executed = {"output": f"исключение: {exc}", "ok": False}
            else:
                executed = {"output": "не исполнялось: домен требует сети",
                            "ok": None}

        rows.append({
            "phrase": phrase,
            "verb": real_verb,
            "domain": getattr(real, "domain", None) if real is not None else None,
            "object": getattr(real, "object", None) if real is not None else None,
            "params": getattr(real, "parameters", None) if real is not None else None,
            "source": (getattr(real, "context", None) or {}).get("micro_source")
            if real is not None else None,
            "steps": info["steps"],
            "decision": info["decision"],
            "trace_mismatch": mismatch,
            "executed": executed,
            "outcome": "действие" if real is not None else "воздержание",
        })
    return rows


TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>МУХА: как принимается решение</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0a0c10; color:#c9d1d9;
         font:14px/1.55 -apple-system,"SF Pro Text",Segoe UI,sans-serif; }
  header { padding:22px 28px 12px; border-bottom:1px solid #1d232b; }
  h1 { margin:0 0 6px; font-size:20px; color:#ffb000; letter-spacing:.02em; }
  .lead { color:#8b949e; font-size:13px; max-width:74ch; }
  .wrap { display:grid; grid-template-columns:minmax(280px,360px) 1fr;
          gap:0; min-height:calc(100vh - 150px); }
  .list { border-right:1px solid #1d232b; padding:14px; overflow-y:auto;
          max-height:calc(100vh - 150px); }
  .filters { display:flex; gap:6px; margin-bottom:10px; flex-wrap:wrap; }
  .filters button { background:#141a21; color:#8b949e; border:1px solid #232c36;
                    border-radius:14px; padding:4px 11px; font-size:12px;
                    cursor:pointer; }
  .filters button.on { background:#ffb000; color:#0a0c10; border-color:#ffb000;
                       font-weight:600; }
  .item { padding:9px 11px; border-radius:8px; cursor:pointer;
          border:1px solid transparent; margin-bottom:5px; }
  .item:hover { background:#11161d; }
  .item.on { background:#151c24; border-color:#2c3742; }
  .item .p { font-size:13px; color:#c9d1d9; }
  .item .m { font-size:11.5px; color:#6e7781; margin-top:3px; }
  .badge { display:inline-block; padding:1px 7px; border-radius:10px;
           font-size:11px; font-weight:600; margin-right:6px; }
  .b-act { background:#12351f; color:#5aff8a; }
  .b-abs { background:#33210f; color:#ffb000; }
  .b-bad { background:#3a1512; color:#ff6a5a; }
  main { padding:20px 26px; overflow-y:auto; max-height:calc(100vh - 150px); }
  .pipe { display:flex; gap:6px; flex-wrap:wrap; margin:0 0 20px; }
  .pipe div { flex:1 1 90px; background:#11161d; border:1px solid #1e252e;
              border-radius:8px; padding:7px 9px; font-size:11px; color:#8b949e;
              text-align:center; }
  .pipe div b { display:block; color:#c9d1d9; font-size:12px; margin-bottom:2px; }
  h2 { font-size:15px; color:#79c0ff; margin:0 0 12px; }
  .phrase { font-size:16px; color:#e6edf3; margin-bottom:4px; }
  .meta { color:#6e7781; font-size:12px; margin-bottom:16px; }
  table.steps { width:100%; border-collapse:collapse; font-size:13px; }
  table.steps td { padding:8px 10px; border-bottom:1px solid #1a2027;
                   vertical-align:top; }
  table.steps td.i { width:26px; font-weight:700; }
  .pass td.i { color:#5aff8a; }
  .stop td.i { color:#ff6a5a; }
  .skip td.i { color:#484f58; }
  .skip { opacity:.45; }
  .reason { color:#8b949e; font-size:12px; }
  .card { margin-top:18px; padding:14px 16px; border-radius:10px;
          background:#11161d; border:1px solid #1e252e; }
  .card h3 { margin:0 0 8px; font-size:13px; color:#8b949e; font-weight:600;
             text-transform:uppercase; letter-spacing:.06em; }
  .kv { display:grid; grid-template-columns:120px 1fr; gap:4px 12px;
        font-size:13px; }
  .kv span:first-child { color:#6e7781; }
  code { background:#161b22; padding:1px 5px; border-radius:4px;
         font-size:12.5px; }
  pre { background:#0d1117; border:1px solid #1a2027; border-radius:8px;
        padding:11px; overflow-x:auto; font-size:12px; color:#a5b1bd;
        white-space:pre-wrap; margin:0; }
  .warn { border-left:3px solid #ff6a5a; background:#1a1113; padding:9px 13px;
          border-radius:0 8px 8px 0; font-size:12.5px; color:#ffb4ac;
          margin-top:14px; }
  footer { padding:14px 28px 26px; border-top:1px solid #1d232b;
           color:#6e7781; font-size:12px; }
  footer b { color:#8b949e; }
</style>
</head>
<body>
<header>
  <h1>МУХА: как фраза превращается в действие</h1>
  <div class="lead">Слева — настоящие фразы. Справа — весь путь решения по шагам:
  каскад микромоделей, затем девять проверок готовности. Данные получены прогоном
  реального кода, вызовов платной модели не было.</div>
</header>
<div class="wrap">
  <div class="list">
    <div class="filters">
      <button data-f="all" class="on">все __N__</button>
      <button data-f="act">действие __NACT__</button>
      <button data-f="abs">воздержание __NABS__</button>
    </div>
    <div id="items"></div>
  </div>
  <main id="detail"></main>
</div>
<footer>
  <b>Что здесь правда.</b> Вердикт каждого шага снят теми же функциями, что
  вызываются в агенте (<code>core/readiness.py</code>), и сверен с настоящим
  ответом <code>propose_action_fast</code>: расхождений __MISMATCH__.
  Фраз __N__, дата сборки __BUILT__, sha256 списка фраз <code>__DIGEST__</code>.
  Исполнение — во временной песочнице, сеть не трогалась, реестр локальный.
  <br><b>Чего здесь нет.</b> Это не оценка качества разбора: метки не
  человеческие, а страница показывает механику, а не точность.
</footer>
<script>
const DATA = __DATA__;
let filter = "all";
let current = 0;

const STAGES = __STAGES__;

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function renderItems() {
  const box = document.getElementById("items");
  box.innerHTML = "";
  DATA.forEach((row, index) => {
    if (filter === "act" && row.outcome !== "действие") return;
    if (filter === "abs" && row.outcome !== "воздержание") return;
    const item = el("div", "item" + (index === current ? " on" : ""));
    item.onclick = () => { current = index; renderItems(); renderDetail(); };
    const head = el("div", "p");
    const badge = el("span", "badge " + (row.verb ? "b-act" : "b-abs"),
                     row.verb || "воздержание");
    head.appendChild(badge);
    head.appendChild(document.createTextNode(row.phrase));
    item.appendChild(head);
    const meta = el("div", "m",
      row.verb ? (row.verb + " · " + (row.domain || "—")) : "уходит омеге");
    item.appendChild(meta);
    box.appendChild(item);
  });
}

function renderDetail() {
  const row = DATA[current];
  const main = document.getElementById("detail");
  main.innerHTML = "";

  const pipe = el("div", "pipe");
  STAGES.forEach(s => {
    const d = el("div");
    d.appendChild(el("b", null, s[1]));
    d.appendChild(document.createTextNode(s[2]));
    pipe.appendChild(d);
  });
  main.appendChild(pipe);

  main.appendChild(el("div", "phrase", row.phrase));
  main.appendChild(el("div", "meta",
    "результат: " + (row.verb
      ? ("действие " + row.verb + " / " + row.domain + " / " + row.object)
      : "воздержание, решение остаётся за омегой")));

  const table = el("table", "steps");
  const seen = new Set(row.steps.map(s => s.key));
  STAGES.forEach(s => {
    const st = row.steps.find(x => x.key === s[0]);
    const tr = el("tr");
    if (!st) {
      tr.className = "skip";
      tr.appendChild(el("td", "i", "–"));
      tr.appendChild(el("td", null, s[1] + " — не дошло"));
    } else {
      tr.className = st.verdict === "stop" ? "stop" : "pass";
      tr.appendChild(el("td", "i", st.verdict === "stop" ? "✕" : "✓"));
      const td = el("td");
      td.appendChild(el("div", null, s[1]));
      td.appendChild(el("div", "reason", st.reason));
      tr.appendChild(td);
    }
    table.appendChild(tr);
  });
  main.appendChild(table);

  if (row.trace_mismatch) {
    const w = el("div", "warn",
      "Трасса разошлась с настоящим ответом агента. Страница показывает факт, " +
      "а не прячет: значит, порядок шагов здесь описан неточно.");
    main.appendChild(w);
  }

  const card = el("div", "card");
  card.appendChild(el("h3", null, "итог"));
  const kv = el("div", "kv");
  const put = (k, v) => {
    kv.appendChild(el("span", null, k));
    const val = el("span");
    val.appendChild(document.createTextNode(v));
    kv.appendChild(val);
  };
  put("глагол", row.verb || "—");
  put("домен", row.domain || "—");
  put("объект", row.object || "—");
  put("источник", row.source || "—");
  put("параметры", row.params && Object.keys(row.params).length
      ? JSON.stringify(row.params) : "нет обязательных");
  card.appendChild(kv);
  main.appendChild(card);

  if (row.executed) {
    const c2 = el("div", "card");
    c2.appendChild(el("h3", null, "что реально произошло"));
    const pre = el("pre", null, row.executed.output);
    c2.appendChild(pre);
    main.appendChild(c2);
  }
}

document.querySelectorAll(".filters button").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll(".filters button").forEach(b => b.classList.remove("on"));
    btn.classList.add("on");
    filter = btn.dataset.f;
    renderItems();
  };
});

renderItems();
renderDetail();
</script>
</body>
</html>
"""


def main() -> int:
    rows = collect()
    mismatch = sum(1 for row in rows if row["trace_mismatch"])
    acts = sum(1 for row in rows if row["outcome"] == "действие")
    digest = hashlib.sha256(
        "\n".join(PHRASES).encode("utf-8")).hexdigest()[:16]
    html = (TEMPLATE
            .replace("__DATA__", json.dumps(rows, ensure_ascii=False))
            .replace("__STAGES__", json.dumps(STAGES, ensure_ascii=False))
            .replace("__NACT__", str(acts))
            .replace("__NABS__", str(len(rows) - acts))
            .replace("__N__", str(len(rows)))
            .replace("__MISMATCH__", str(mismatch))
            .replace("__DIGEST__", digest)
            .replace("__BUILT__", time.strftime("%Y-%m-%d %H:%M")))
    out = DEMO / "muha.html"
    out.write_text(html, encoding="utf-8")
    print(f"фраз: {len(rows)}, действий: {acts}, воздержаний: {len(rows) - acts}, "
          f"расхождений трассы: {mismatch}")
    print(f"страница: {out} ({len(html) // 1024} КБ)")
    return 0 if mismatch == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())