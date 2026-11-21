"""Тесты Муха-Агента на «голых» assert'ах.

Осознанно НЕ используется `unittest`: на этой машине импорт stdlib-расширений
(`_heapq`, `_csv`) периодически зависает в `dlopen`, из-за чего `unittest`
(тянет `heapq` → `difflib`) не запускается вообще. Функции `test_*` совместимы
и с pytest, если он доступен.

Запуск:  python3 tests/run_tests.py
         python3 -m pytest tests -q     (если pytest работает)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.agent import MODES, Agent
from core.autonomous import AutonomousRunner
from core.tasks import DONE, FAILED, PENDING, TaskQueue
from core.micromodels import MicroCascade, cue_verb, domain_of, selective_report
from modules.filesystem import ToolError
from modules.web import STATE, html_to_text, read_web, validate_url
from core.golden_apples import GoldenApples
from core.gatekeeper import GateDecision, GateKeeper
from core.llm_client import LLMClient, LLMMessage, LLMUnavailable, resolve_api_key
from core.registry import Registry
from core.verb_act import VerbAct, VerbActGrammar, VerbActType
from modules.connectome import Connectome, _parse_csv_line
from modules.filesystem import FsSandbox, ToolError, list_dir, read_file, write_file
from modules.system import SystemSandbox

CONFIG = str(ROOT / "config.yaml")
CONNECTome_DIR = ROOT.parent / "var" / "flywire-role"


# ---------- общие помощники ----------

def make_agent(policy: str = "allow") -> Agent:
    """Агент с изолированным реестром и песочницей во временном каталоге."""
    agent = Agent(config_path=CONFIG, confirm_policy=policy)
    tmp = tempfile.mkdtemp(prefix="muha-test-")
    agent.registry = Registry(path=os.path.join(tmp, "registry.jsonl"))
    agent.set_sandbox_root(tmp)
    return agent


def load_config_for_tests() -> dict:
    from core.miniyaml import load_file

    return load_file(CONFIG)


def assert_raises(exc_type, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"ожидалось исключение {exc_type.__name__}, его не было")


# ---------- файловая система ----------

def test_fs_write_then_read():
    sandbox = FsSandbox(root=Path(tempfile.mkdtemp(prefix="muha-fs-")))
    report = write_file("notes/a.txt", "привет, муха", sandbox)
    assert "создан" in report, report
    assert (sandbox.root / "notes" / "a.txt").read_text("utf-8") == "привет, муха"
    back = read_file("notes/a.txt", sandbox)
    assert "привет, муха" in back
    assert "12 байт" in back or "байт" in back


def test_fs_write_is_atomic_overwrite():
    sandbox = FsSandbox(root=Path(tempfile.mkdtemp(prefix="muha-fs-")))
    write_file("a.txt", "первая", sandbox)
    report = write_file("a.txt", "вторая", sandbox)
    assert "перезаписан" in report
    assert (sandbox.root / "a.txt").read_text("utf-8") == "вторая"
    leftovers = [p for p in sandbox.root.iterdir() if p.name.startswith(".muha-")]
    assert not leftovers, f"остались временные файлы: {leftovers}"


def test_fs_escape_is_denied():
    sandbox = FsSandbox(root=Path(tempfile.mkdtemp(prefix="muha-fs-")))
    err = assert_raises(ToolError, read_file, "../../../etc/passwd", sandbox)
    assert "вне песочницы" in str(err)


def test_fs_read_missing_file():
    sandbox = FsSandbox(root=Path(tempfile.mkdtemp(prefix="muha-fs-")))
    err = assert_raises(ToolError, read_file, "нет-такого.txt", sandbox)
    assert "не найден" in str(err)


def test_fs_read_truncates_big_file():
    sandbox = FsSandbox(root=Path(tempfile.mkdtemp(prefix="muha-fs-")), max_read_bytes=10)
    write_file("big.txt", "x" * 100, sandbox)
    text = read_file("big.txt", sandbox)
    assert "показаны первые 10 байт" in text


def test_fs_list_dir():
    sandbox = FsSandbox(root=Path(tempfile.mkdtemp(prefix="muha-fs-")))
    write_file("b.txt", "1", sandbox)
    (sandbox.root / "sub").mkdir()
    listing = list_dir(".", sandbox)
    assert "b.txt" in listing and "sub" in listing
    assert "DIR" in listing and "FILE" in listing


# ---------- система ----------

def test_system_rejects_unknown_command():
    sandbox = SystemSandbox(allowed_commands=("echo",))
    err = assert_raises(ToolError, sandbox.check, "rm -rf /")
    assert "allowlist" in str(err)


def test_system_rejects_shell_metacharacters():
    sandbox = SystemSandbox(allowed_commands=("echo",))
    err = assert_raises(ToolError, sandbox.check, "echo hi; rm -rf /")
    assert "метасимволы" in str(err)


def test_system_rejects_denied_pattern():
    sandbox = SystemSandbox(allowed_commands=("git",), denied_patterns=("rm -rf",))
    err = assert_raises(ToolError, sandbox.check, "git reset --hard rm -rf")
    assert "запрещённый шаблон" in str(err)


def test_system_accepts_allowed_command():
    sandbox = SystemSandbox(allowed_commands=("echo",))
    assert sandbox.check("echo привет") == ["echo", "привет"]


# ---------- привилегированные команды (закрытие обхода allowlist) ----------

ESCAPE = "python3 -c \"print(__import__('os').getcwd())\""   # без шелл-метасимволов


def test_escape_command_passes_metachar_filter_but_is_privileged():
    sandbox = SystemSandbox(
        allowed_commands=("echo",),
        privileged_commands=("python3",),
        privileged_roles=("guardian",),
        denied_patterns=("rm -rf",),
    )
    # метасимволов нет — фильтр подстрок такой вызов не ловит
    assert not (set(";|&$`<>") & set(ESCAPE))
    assert_raises(ToolError, sandbox.check, ESCAPE)                    # роль не задана
    assert_raises(ToolError, sandbox.check, ESCAPE, "decider")         # роль ниже
    assert sandbox.check(ESCAPE, "guardian")[0] == "python3"


def test_privileged_only_accepts_role_from_list():
    sandbox = SystemSandbox(
        allowed_commands=(), privileged_commands=("python3",),
        privileged_roles=("registrar",),
    )
    assert_raises(ToolError, sandbox.check, "python3 -c 1", "guardian")
    assert sandbox.check("python3 -c 1", "registrar")


def test_config_separates_observational_and_privileged():
    cfg = load_config_for_tests()
    tools = cfg["tools"]
    allowed = set(tools["allowed_commands"])
    privileged = set(tools["privileged_commands"])
    assert not (allowed & privileged), f"пересечение списков: {allowed & privileged}"
    for interpreter in ("python3", "git", "bash", "sh"):
        assert interpreter in privileged, interpreter
        assert interpreter not in allowed, interpreter
    for safe in ("uname", "pwd", "ls", "cat", "date"):
        assert safe in allowed, safe


def test_escape_through_agent_is_denied_and_recorded():
    agent = make_agent(policy="allow")
    agent.role = "decider"
    act = agent._make_act(VerbActType.EXECUTE, f"sys:{ESCAPE}", {}, "x", "test")
    result = str(agent.execute(act))
    assert act.success is False
    assert "привилегированная" in result, result
    assert agent.registry.entries[-1].success is False


def test_privileged_command_allowed_for_guardian():
    agent = make_agent(policy="allow")
    agent.role = "guardian"
    agent.system_sandbox.allowed_commands = ("echo",)
    agent.system_sandbox.privileged_commands = ("python3",)
    agent.system_sandbox.privileged_roles = ("guardian",)
    act = agent._make_act(VerbActType.EXECUTE, f"sys:{ESCAPE}", {}, "x", "test")
    result = str(agent.execute(act))
    assert act.success is True, result
    assert str(agent.fs_sandbox.root) in result or "stdout" in result


def test_observational_command_unaffected_by_privilege_split():
    agent = make_agent(policy="allow")
    agent.role = "decider"
    agent.system_sandbox.allowed_commands = ("echo",)
    act = agent._make_act(VerbActType.EXECUTE, "sys:echo привет", {}, "x", "test")
    result = agent.execute(act)          # сначала выполняем, потом проверяем
    assert act.success is True, result
    assert "привет" in result, result


# ---------- чтение веба ----------

WEB_HTML = """<html><head><title>Проба</title><style>b{color:red}</style></head><body>
<nav><a href="/menu">меню</a></nav>
<article><h1>Заголовок</h1>
<p>Первый абзац со <a href="/x">ссылкой</a>.</p>
<p>Второй абзац.</p></article>
<script>alert('вред')</script></body></html>"""


def test_web_url_rejects_local_and_service_addresses():
    for bad in ("http://localhost:8080/", "http://127.0.0.1/", "http://192.168.0.1/",
                "http://10.1.2.3/", "http://169.254.169.254/latest/meta-data/",
                "file:///etc/passwd", "ftp://example.com/x"):
        assert_raises(ToolError, validate_url, bad)
    assert validate_url("https://example.com/a") == "https://example.com/a"
    assert validate_url("example.com/a").startswith("https://")   # схема по умолчанию


def test_web_html_to_text_drops_scripts_and_keeps_paragraphs():
    text, meta = html_to_text(WEB_HTML)
    assert "вред" not in text, "скрипт не должен попадать в текст"
    assert "color:red" not in text, "стили не должны попадать в текст"
    assert "меню" not in text, "навигация вне article не нужна"
    assert "Первый абзац со ссылкой." in text, text
    assert "Второй абзац." in text
    assert meta["title"] == "Проба"
    assert meta["selector_used"] == "article"


def test_web_fallback_without_scrapling():
    """Без Scrapling модуль обязан работать, но честно пометить деградацию."""
    saved = dict(STATE)
    try:
        STATE["scrapling_ok"] = False
        text, meta = html_to_text(WEB_HTML)
        assert meta["parser"] == "fallback"
        assert "degraded" in meta
        assert "Первый абзац" in text
        assert "вред" not in text
    finally:
        STATE.update(saved)


def test_web_read_fetches_through_stub_server():
    """Сеть подменяем локальным сервером: проверяем сборку ответа, не интернет."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):                                    # noqa: N802
            body = WEB_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):                        # тишина в тестах
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        # SSRF-защита справедливо запрещает localhost; в тесте открываем её точечно.
        import modules.web as web_module
        original = web_module._host_is_private
        web_module._host_is_private = lambda host: False
        try:
            out = read_web(f"http://127.0.0.1:{port}/page", {})
        finally:
            web_module._host_is_private = original
        assert "HTTP: 200" in out
        assert "Первый абзац со ссылкой." in out
        assert "разбор: scrapling" in out or "разбор: fallback" in out
    finally:
        server.shutdown()
        server.server_close()


# ---------- браузерный рендер (Obscura) ----------

def _fake_binary(tmpdir, body, name="fakebrowser"):
    """Поддельный «браузер»: shell-скрипт, чтобы тест не зависел от сети."""
    import os
    import stat
    path = os.path.join(tmpdir, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#!/bin/sh\n" + body + "\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_browser_render_runs_external_binary():
    import tempfile
    from modules import browser
    with tempfile.TemporaryDirectory() as tmp:
        path = _fake_binary(tmp, 'echo "текст после JS: цитата Эйнштейна"')
        text, meta = browser.render("https://example.com", timeout=10, explicit_binary=path)
    assert "цитата Эйнштейна" in text
    assert meta["backend"] == "obscura"
    assert meta["exit_status"] == 0


def test_browser_render_reports_missing_binary():
    from modules import browser
    assert_raises(browser.BrowserUnavailable, browser.render,
                  "https://example.com", 5, "text", "/nonexistent/browser")
    assert browser.is_available("/nonexistent/browser") is False


def test_browser_render_kills_on_timeout():
    import tempfile
    from modules import browser
    with tempfile.TemporaryDirectory() as tmp:
        path = _fake_binary(tmp, "sleep 10")
        started = __import__("time").time()
        error = assert_raises(browser.BrowserUnavailable, browser.render,
                              "https://example.com", 1, "text", path)
        elapsed = __import__("time").time() - started
    assert "не уложился" in str(error), str(error)
    assert elapsed < 8, f"дедлайн не сработал: {elapsed:.1f}s"


def test_browser_render_reports_nonzero_exit():
    import tempfile
    from modules import browser
    with tempfile.TemporaryDirectory() as tmp:
        path = _fake_binary(tmp, 'echo "boom" >&2; exit 3')
        error = assert_raises(browser.BrowserUnavailable, browser.render,
                              "https://example.com", 10, "text", path)
    assert "статус" in str(error)


def test_read_web_uses_browser_for_spa_shell():
    """Тонкий текст при большом HTML = каркас SPA, обязан включиться браузер."""
    import modules.web as web_module
    from modules import browser as browser_module

    spa = "<html><head><title>SPA</title></head><body><div id=root></div>" + \
          "<script>" + "var x=1;" * 500 + "</script></body></html>"
    saved_render = browser_module.render
    saved_avail = browser_module.is_available
    called = {"n": 0}

    def fake_render(url, timeout=30, fmt="text", explicit_binary=None):   # noqa: ANN001
        called["n"] += 1
        return "цитата, появившаяся после JS " + "текст " * 200, {"backend": "obscura",
                                                                 "bytes": 1200}

    class Fetcher:
        def fetch(self, url):
            return {"url": url, "final_url": url, "status": 200,
                    "content_type": "text/html; charset=utf-8",
                    "text": spa, "bytes": len(spa)}

    browser_module.render = fake_render
    browser_module.is_available = lambda explicit_binary=None: True
    try:
        out = web_module.read_web("https://example.com/app", {}, fetcher=Fetcher())
    finally:
        browser_module.render = saved_render
        browser_module.is_available = saved_avail
    assert called["n"] == 1, "браузер должен был включиться"
    assert "obscura" in out and "после JS" in out
    assert "появившаяся после JS" in out


def test_read_web_skips_browser_for_plain_page():
    import modules.web as web_module
    from modules import browser as browser_module

    html = "<html><body><article><p>" + ("обычный текст. " * 60) + "</p></article></body></html>"
    saved_render = browser_module.render
    called = {"n": 0}

    def fake_render(url, timeout=30, fmt="text", explicit_binary=None):   # noqa: ANN001
        called["n"] += 1
        return "", {}

    class Fetcher:
        def fetch(self, url):
            return {"url": url, "final_url": url, "status": 200,
                    "content_type": "text/html; charset=utf-8",
                    "text": html, "bytes": len(html)}

    browser_module.render = fake_render
    try:
        out = web_module.read_web("https://example.com/plain", {}, fetcher=Fetcher())
    finally:
        browser_module.render = saved_render
    assert called["n"] == 0, "для обычной страницы браузер не нужен"
    assert "обычный текст" in out


# ---------- оценка вариантов по logprobs ----------

def _stub_scorer_server(payload):
    """Локальный сервер, отдающий заданный ответ вместо OpenRouter."""
    import http.server
    import json as json_module
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):                                   # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            body = json_module.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"


def _logprob_answer(token, options):
    """Собирает logprobs в том виде, в каком их отдаёт OpenRouter."""
    import math
    top = [{"token": " " + option, "logprob": math.log(prob)}
           for option, prob in options.items()]
    return {"choices": [{"message": {"content": token},
                         "finish_reason": "stop",
                         "logprobs": {"content": [{"token": token,
                                                   "logprob": math.log(0.9),
                                                   "top_logprobs": top}]}}],
            "model": "mistralai/mistral-nemo",
            "provider": "DekaLLM",
            "usage": {"total_tokens": 63, "cost": 0.0000012}}


def test_option_scorer_reads_distribution_not_text():
    from core.option_scorer import score_options
    payload = _logprob_answer(" read", {"read": 0.819, "write": 0.001, "think": 0.05})
    server, url = _stub_scorer_server(payload)
    try:
        scores = score_options("прочитай файл config.yaml", api_key_env="PATH",
                               url=url, require_parameters=False, timeout=10)
    finally:
        server.shutdown()
        server.server_close()
    assert scores.best == "read"
    assert 0.9 < scores.best_prob <= 1.0
    assert scores.margin > 0.8
    assert scores.provider == "DekaLLM"
    assert scores.tokens == 63
    assert scores.cost_usd > 0


def test_option_scorer_sums_same_variant_with_stray_spaces():
    """«read» и « read» — один вариант: вероятности складываются, не теряются."""
    import math
    from core.option_scorer import aggregate
    top = [{"token": " read", "logprob": math.log(0.4)},
           {"token": "read", "logprob": math.log(0.3)},
           {"token": " write", "logprob": math.log(0.3)}]
    probabilities, share, _ = aggregate(top, ("read", "write"))
    assert abs(probabilities["read"] - 0.7) < 1e-6, probabilities
    assert abs(probabilities["write"] - 0.3) < 1e-6
    assert abs(share - 1.0) < 1e-6


def test_option_scorer_refuses_when_no_logprobs():
    from core.option_scorer import ScoringUnavailable, score_options
    payload = {"choices": [{"message": {"content": "read"}, "logprobs": None}],
               "usage": {"total_tokens": 10}}
    server, url = _stub_scorer_server(payload)
    try:
        error = assert_raises(
            ScoringUnavailable, score_options, "прочитай файл",
            api_key_env="PATH", url=url, require_parameters=False, timeout=10)
    finally:
        server.shutdown()
        server.server_close()
    assert "logprobs" in str(error)


def test_option_scorer_reports_low_share_when_answer_off_list():
    """Модель ответила словом не из списка — доверять выбору нельзя."""
    import math
    from core.option_scorer import aggregate
    top = [{"token": " совершенно", "logprob": math.log(0.9)},
           {"token": " read", "logprob": math.log(0.1)}]
    probabilities, share, _ = aggregate(top, ("read", "write"))
    assert abs(probabilities["read"] - 1.0) < 1e-6, "внутри вариантов доля нормируется"
    assert share < 0.2, f"доля массы на вариантах должна быть мала: {share}"


def test_anchor_rule_needs_a_verb_not_just_the_word_kanon():
    """«посоветуй как поступить с каноном» — это совет, а не Commit (L5).

    Правило anchor ловило слово «каноном» в любом месте фразы, поэтому
    упоминание объекта превращало рекомендацию в необратимое обязательство.
    """
    from core.micromodels import cue_verb
    assert cue_verb("посоветуй как поступить с каноном")[0] == "recommend"
    assert cue_verb("заякори канон в реестре")[0] == "anchor"
    assert cue_verb("сделай версия-2 каноном")[0] == "anchor"
    assert cue_verb("закрепи канон как эталон")[0] == "anchor"


# ---------- граница готовности кандидата ----------

def test_negation_does_not_become_action():
    """«не запускай команду date» — запрет, а не приказ запустить."""
    from core.agent import Agent
    agent = Agent()
    for phrase in ("не запускай команду date",
                   "не нужно выполнять команду date",
                   "не трогай файл notes.md"):
        act = agent.propose_action_fast(phrase)
        assert act is None, f"«{phrase}» дало действие {act and act.verb}"


def test_question_about_action_does_not_become_action():
    """Вопрос и совет о действии не должны давать действие с побочным эффектом.

    Ответ `think` или воздержание — оба приемлемы: они ничего не меняют.
    Неприемлемо превращать «что такое вето» в вето, а «как выполнить» — в
    выполнение.
    """
    from core.agent import Agent
    from core.readiness import ACTION_VERBS
    agent = Agent()
    for phrase in ("расскажи что означает вето",
                   "посоветуй как выполнить команду date",
                   "что такое якорь",
                   "объясни, зачем нужно вето"):
        act = agent.propose_action_fast(phrase)
        if act is None:
            continue
        verb = act.verb.value if hasattr(act.verb, "value") else str(act.verb)
        assert verb not in ACTION_VERBS, (
            f"«{phrase}» дало действие с побочным эффектом: {verb}")


def test_reading_question_still_answers_read():
    """А вот вопрос о содержимом файла — законный локальный ответ."""
    from core.agent import Agent
    agent = Agent()
    act = agent.propose_action_fast("объясни что написано в notes.md")
    assert act is not None, "вопрос о содержимом должен читаться локально"
    assert act.verb.value == "read", act.verb


def test_command_arguments_are_preserved():
    """argv сохраняется целиком: раньше `-a` терялся и шла ДРУГАЯ команда."""
    from core.agent import Agent
    agent = Agent()
    act = agent.propose_action_fast("запусти команду uname -a")
    assert act is not None, "команда с аргументом должна разбираться"
    assert act.parameters.get("argv") == ["uname", "-a"], act.parameters
    assert act.parameters.get("command") == "uname"

    act = agent.propose_action_fast("запусти команду echo привет")
    assert act.parameters.get("argv") == ["echo", "привет"], act.parameters


def test_shell_metacharacters_stay_literal():
    """Подстановка не выполняется: argv — список, shell не участвует."""
    from core.readiness import extract_argv
    assert extract_argv("запусти команду echo $HOME", "echo") == ["echo", "$HOME"]


def test_incomplete_write_is_not_ready():
    """write без содержимого не объявляется готовым: исполнитель его требует."""
    from core.agent import Agent
    agent = Agent()
    act = agent.propose_action_fast("запиши в notes.md")
    assert act is None, "write без content не должен считаться готовым"
    ok = agent.propose_action_fast("запиши в notes.md строку готово")
    assert ok is not None and ok.parameters.get("content") == "готово", (
        ok and ok.parameters)


def test_incomplete_select_is_not_ready():
    """select без вариантов не объявляется готовым."""
    from core.agent import Agent
    agent = Agent()
    act = agent.propose_action_fast("выбери вариант")
    assert act is None, "select без options не должен считаться готовым"
    ok = agent.propose_action_fast("выбери одно из двух: A или B")
    assert ok is not None and ok.parameters.get("options") == ["A", "B"], (
        ok and ok.parameters)


def test_two_actions_are_not_collapsed_into_one():
    """«прочитай A и запиши B» — два действия, локально не исполняем."""
    from core.agent import Agent
    agent = Agent()
    act = agent.propose_action_fast("прочитай notes.md и запиши копию в backup.md")
    assert act is None, f"второе действие потеряно: {act and act.verb}"


def test_verb_inside_filename_does_not_become_the_verb():
    """«прочитай файл anchor.md» — чтение, а не якорь."""
    from core.agent import Agent
    agent = Agent()
    act = agent.propose_action_fast("прочитай файл anchor.md")
    assert act is not None, "чтение файла должно разбираться"
    assert act.verb.value == "read", f"имя файла дало глагол {act.verb}"


def test_zabrepit_bez_kanona_is_not_anchor():
    """«закрепи вкладку браузера» — не канонизация."""
    from core.micromodels import cue_verb
    assert cue_verb("закрепи вкладку браузера")[0] != "anchor"
    assert cue_verb("закрепи канон в реестре")[0] == "anchor"


def test_micromodel_vocab_is_cleared_on_refit():
    """Переобучение не должно оставлять слова старого корпуса знакомыми."""
    from core.micromodels import VerbMicroModel
    model = VerbMicroModel().fit([("прочитай alpha.txt", "read")] * 3)
    before = model.known_ratio("alpha.txt")
    model.fit([("запиши beta.txt", "write")] * 3)
    assert before > 0.5, before
    assert model.known_ratio("alpha.txt") < before, "vocab не очистился"


def test_version_of_a_file_comes_from_the_newest_one():
    """G075: версия файла — это самая свежая запись о нём, а не версия агента.

    Решение владельца: «смотри по дате создания, последний и нужен». Значит
    источник версии назван правилом, и уточнять у пользователя нечего.
    """
    import time as _time
    agent = make_agent()
    tmp = str(agent.fs_sandbox.root)
    older = os.path.join(tmp, "sub")
    os.makedirs(older, exist_ok=True)
    with open(os.path.join(older, "todo.md"), "w", encoding="utf-8") as fh:
        fh.write("старый\n")
    os.utime(os.path.join(older, "todo.md"), (1_000_000_000, 1_000_000_000))
    with open(os.path.join(tmp, "todo.md"), "w", encoding="utf-8") as fh:
        fh.write("новый\n")
    _time.sleep(0.01)

    act = agent.propose_action_fast("Какая версия у файла todo.md?")
    assert act is not None and act.verb.value == "version"
    result = agent.tools.execute(act.verb, act.domain, act.object,
                                 act.parameters or {}).output
    assert "todo.md" in result, result
    assert "по дате создания" in result, result
    # Новее тот, что в корне песочницы, а не тот, что в подкаталоге.
    assert os.path.join(tmp, "todo.md") in result, result
    assert older not in result, result


def test_pronoun_without_referent_is_not_an_action():
    """«Сделай с этим что-нибудь полезное» — вопрос, а не команда.

    Истории беседы нет: «это» не на что указывать. Наблюдалось на реальной
    задаче: запрос давал recommend вместо уточнения.
    """
    agent = make_agent()
    assert agent.propose_action_fast('Сделай с этим что-нибудь полезное') is None


def test_single_command_keeps_its_path_argument():
    """Путь внутри одной команды — аргумент, а не вторая цель.

    Счётчик целей не должен отвергать «wc -l notes.md» как два действия.
    """
    agent = make_agent()
    act = agent.propose_action_fast('Запусти команду wc -l notes.md')
    assert act is not None and act.verb.value == 'execute', act
    assert act.parameters['argv'] == ['wc', '-l', 'notes.md'], act.parameters


def test_named_target_allows_a_pronoun_inside_one_action():
    """Местоимение при названной цели не мешает: «покажи его текст»."""
    agent = make_agent()
    act = agent.propose_action_fast('Открой report.csv и покажи его текст')
    assert act is not None and act.verb.value == 'read', act
    assert act.object == 'file:report.csv', act.object


def test_backtick_is_a_shell_construct_not_an_argument():
    """Без shell обратная кавычка не подставляется, а портит argv."""
    agent = make_agent()
    assert agent.propose_action_fast('Выполни команду `date`') is None


# ---------- иерархия L0–L5 ----------

HIER_TRAIN = [
    ("прочитай файл config.yaml", "read"),
    ("покажи содержимое notes.md", "read"),
    ("подумай, как лучше сделать", "think"),
    ("сравни два варианта", "think"),
    ("запиши в log.txt строку готово", "write"),
    ("сохрани результат в otchet.txt", "write"),
    ("запусти команду date", "execute"),
    ("посоветуй, что выбрать", "recommend"),
    ("наложи вето на запись канон", "veto"),
    ("позови человека на помощь", "escalate"),
    ("заякори канон в реестре", "anchor"),
    ("залогируй результат работы", "log"),
    ("узнай версию реестра", "version"),
    ("выбери лучший вариант", "select"),
]


def test_level_model_trains_on_levels_not_only_verbs():
    """Регрессия: фильтр по VERBS молча отбрасывал метки L0–L5.

    Из-за него модель уровней оставалась пустой, иерархия всегда воздерживалась,
    а «точность уровня 100%» получалась тривиально — из верного глагола.
    """
    from core.hierarchy import LEVEL_OF, LEVELS
    from core.micromodels import VerbMicroModel
    pairs = [(phrase, LEVEL_OF[verb]) for phrase, verb in HIER_TRAIN]
    model = VerbMicroModel(labels=LEVELS).fit(pairs)
    assert len(model.priors) >= 4, f"уровни не обучились: {model.priors}"
    # По умолчанию фильтр по глаголам остаётся: уровни в него не попадают.
    default_model = VerbMicroModel().fit(pairs)
    assert default_model.priors == {}, "метки уровней не должны попадать в глагольную модель"


def test_hierarchy_requires_evidence_for_control_levels():
    """Без улики veto/escalate не выдаются, даже если глагол в уровне один."""
    from core.hierarchy import HierarchicalCascade
    # В обучении L4 представлен единственным глаголом veto.
    pairs = [pair for pair in HIER_TRAIN if pair[1] != "escalate"]
    cascade = HierarchicalCascade(pairs)
    decision = cascade.decide("что-нибудь непонятное про канон")
    assert decision is None or decision.verb != "veto", (
        f"вето выдано без улики: {decision and decision.verb}")


def test_hierarchy_agrees_with_flat_cascade_outside_control_levels():
    """Вне уровней контроля ответы совпадают: правила идут первыми.

    Расхождение на L4/L5 — намеренное: там иерархия требует явной улики и
    отказывается отвечать по одной статистике, даже если плоский каскад
    готов. Это разные политики, и проверяются они по отдельности.
    """
    from core.hierarchy import SAFETY_CRITICAL_LEVELS, LEVEL_OF, HierarchicalCascade
    from core.micromodels import MicroCascade
    flat = MicroCascade(HIER_TRAIN)
    hier = HierarchicalCascade(HIER_TRAIN)
    checked = 0
    for phrase, verb in HIER_TRAIN:
        left = flat.decide(phrase)
        if left is None:
            continue
        if LEVEL_OF.get(left.verb) in SAFETY_CRITICAL_LEVELS:
            continue          # см. test_hierarchy_requires_evidence_for_control_levels
        right = hier.decide(phrase)
        assert right is not None, f"иерархия потеряла решение на «{phrase}»"
        assert right.verb == left.verb, (
            f"«{phrase}»: плоский {left.verb}, иерархия {right.verb}")
        checked += 1
    assert checked >= 8, f"проверено слишком мало фраз: {checked}"


def test_hierarchy_answers_where_flat_cascade_abstains():
    """Вето по явной улике: плоский каскад отдаёт омеге, иерархия решает."""
    from core.hierarchy import HierarchicalCascade
    from core.micromodels import MicroCascade
    flat = MicroCascade(HIER_TRAIN)
    hier = HierarchicalCascade(HIER_TRAIN)
    phrase = "наложи вето на запись канон"
    assert flat.decide(phrase) is None, "плоский каскад должен воздерживаться"
    decision = hier.decide(phrase)
    assert decision is not None and decision.verb == "veto", (
        f"иерархия не решила вето: {decision}")


def test_hierarchy_abstains_without_training():
    from core.hierarchy import HierarchicalCascade
    cascade = HierarchicalCascade()
    assert cascade.decide("прочитай файл config.yaml") is None


# ---------- каскад микромоделей ----------

def test_micromodel_cue_rules_beat_bag_of_ngrams():
    """Правила высокой точности должны различать то, что путает Байес."""
    assert cue_verb("что написано в файле notes.md?")[0] == "read"
    assert cue_verb("запиши в файл log.txt строку готово")[0] == "write"
    assert cue_verb("покажи текущую директорию командой pwd")[0] == "execute"
    assert cue_verb("какого размера файл otchet.txt?")[0] == "select"
    assert cue_verb("наложи вето на эту запись")[0] == "veto"
    assert cue_verb("эскалируй это наверх оператору")[0] == "escalate"


def test_micromodel_cascade_abstains_when_unsure():
    """Мусорный ввод и малое обучение — каскад обязан промолчать."""
    one_example = MicroCascade([("прочитай файл config.yaml", "read")])
    assert one_example.decide("ыфваорлдж") is None, "на одном примере верить нельзя"
    assert one_example.trained_on == 1

    one_class = MicroCascade([("прочитай файл config.yaml", "read")] * 10)
    assert one_class.decide("ыфваорлдж") is None, "один класс в обучении ничего не различает"

    two_classes = MicroCascade(
        [("прочитай файл и покажи содержимое", "read")] * 6
        + [("запусти команду date", "execute")] * 6
    )
    assert two_classes.decide("ыфваорлдж") is None, "мусор не должен получать глагол"


def test_micromodel_requires_margin_over_second_candidate():
    """Высокая вероятность при почти равном втором кандидате — не знание."""
    import core.micromodels as micro
    cascade = MicroCascade(
        [("прочитай файл и покажи содержимое", "read")] * 6
        + [("запусти команду date", "execute")] * 6
    )
    saved = micro.MARGIN_FLOOR
    try:
        micro.MARGIN_FLOOR = 0.99          # почти невозможно превзойти: обязан молчать
        assert cascade.decide("сделай что-нибудь с чем-нибудь") is None
        micro.MARGIN_FLOOR = 0.0           # без требования запаса — отвечает
        assert cascade.decide("запусти команду date") is not None
    finally:
        micro.MARGIN_FLOOR = saved


def test_micromodel_cascade_answers_on_cue_rule():
    cascade = MicroCascade([("прочитай файл config.yaml", "read")])
    decision = cascade.decide("запусти команду date")
    assert decision is not None
    assert decision.verb == "execute"
    assert decision.domain == "system"
    assert decision.confidence >= 0.55


def test_micromodel_selective_report_counts():
    cascade = MicroCascade([("прочитай файл config.yaml", "read")])
    cases = [
        {"id": "a", "prompt": "прочитай файл notes.md", "verbs": ["read"], "domains": ["filesystem"]},
        {"id": "b", "prompt": "запусти команду uname", "verbs": ["execute"], "domains": ["system"]},
    ]
    report = selective_report(cascade, cases)
    assert report["cases"] == 2
    assert report["covered"] == 2
    assert report["selective_accuracy"] == 1.0
    assert report["llm_calls_saved"] == 2


def test_micromodel_domain_from_prefix_and_lexicon():
    assert domain_of("connectome:classes", None)[0] == "connectome"
    assert domain_of("посмотри классы нейронов", "read")[0] == "connectome"
    assert domain_of("прочитай https://example.com", "read")[0] == "web"
    assert domain_of("запусти date", "execute")[0] == "system"


def test_micromodel_saves_llm_call_when_confident():
    """Главная цель каскада: уверенный случай не должен стоить вызова модели."""
    agent = make_agent()
    calls = {"n": 0}

    def counting_think(prompt, system_prompt=None, context=None):    # noqa: ANN001
        calls["n"] += 1
        return "THINK: модель ответила | {}"

    agent.think = counting_think
    act = agent.propose_action("запусти команду date")
    assert act.verb.value == "execute"
    assert calls["n"] == 0, "уверенный разбор микромодели не должен звать модель"
    assert agent.micro_stats["decided"] == 1
    assert agent.micro_stats["escalated"] == 0


def test_micromodel_escalates_to_llm_when_unsure():
    agent = make_agent()
    calls = {"n": 0}

    def counting_think(prompt, system_prompt=None, context=None):    # noqa: ANN001
        calls["n"] += 1
        return "READ: file:config.yaml | {}"

    agent.think = counting_think
    act = agent.propose_action("ыфваорлдж непонятное")
    assert calls["n"] == 1, "при отказе каскада решает модель"
    assert act.verb.value == "read"
    assert agent.micro_stats["escalated"] == 1


def test_micromodel_can_be_disabled():
    agent = make_agent()
    agent.micromodels_enabled = False
    calls = {"n": 0}

    def counting_think(prompt, system_prompt=None, context=None):    # noqa: ANN001
        calls["n"] += 1
        return "READ: file:config.yaml | {}"

    agent.think = counting_think
    agent.propose_action("запусти команду date")
    assert calls["n"] == 1, "выключенный каскад не участвует в разборе"


# ---------- очередь задач ----------

def make_queue() -> TaskQueue:
    return TaskQueue(os.path.join(tempfile.mkdtemp(prefix="muha-q-"), "tasks.jsonl"))


def test_task_queue_add_and_take():
    queue = make_queue()
    assert queue.pending() is None
    task = queue.add("прочитать config.yaml")
    assert task.status == PENDING
    assert queue.pending().id == task.id
    assert queue.stats()["total"] == 1


def test_task_queue_rejects_empty_text():
    queue = make_queue()
    assert_raises(ValueError, queue.add, "   ")


def test_task_queue_survives_reload():
    queue = make_queue()
    first = queue.add("задача один")
    second = queue.add("задача два")
    queue.close(first, DONE, "готово", "решено")

    reopened = TaskQueue(queue.path)
    assert reopened.stats()["total"] == 2
    assert reopened.get(first.id).status == DONE
    assert reopened.get(first.id).result == "готово"
    assert reopened.pending().id == second.id       # закрытая задача не берётся снова


def test_task_queue_ignores_truncated_line():
    """Обрыв записи при падении не должен ломать всю очередь."""
    queue = make_queue()
    task = queue.add("нормальная задача")
    with open(queue.path, "a", encoding="utf-8") as handle:
        handle.write('{"id": "битая", "text": "обор')   # как будто процесс убили
    reopened = TaskQueue(queue.path)
    assert reopened.stats()["total"] == 1
    assert reopened.get(task.id) is not None


def test_task_queue_reset_returns_task_to_pending():
    queue = make_queue()
    task = queue.add("повторить")
    queue.close(task, FAILED, "не вышло", "ошибка")
    reset = queue.reset(task.id)
    assert reset.status == PENDING
    assert queue.pending().id == task.id


def test_task_queue_rejects_unknown_status():
    queue = make_queue()
    task = queue.add("статус")
    task.status = "придуманный"
    assert_raises(ValueError, queue.save, task)


# ---------- автономный цикл ----------

def scripted_agent(script, policy="allow"):
    """Агент со скриптованным think: сеть не нужна, шаги предсказуемы."""
    agent = make_agent(policy=policy)
    agent.task_queue = make_queue()
    agent.runner = AutonomousRunner(agent, agent.task_queue)
    state = {"calls": 0}

    def fake_think(prompt, system_prompt=None, context=None):   # noqa: ANN001
        state["calls"] += 1
        state["last_prompt"] = prompt
        state["last_system_prompt"] = system_prompt
        agent.llm.total_calls += 1               # имитируем расход провайдера
        agent.llm.total_prompt_tokens += 120
        agent.llm.total_completion_tokens += 15
        return script.pop(0)

    agent.think = fake_think
    return agent, state


def test_autonomous_solves_task_with_think():
    agent, state = scripted_agent([
        'READ: file:test.txt | {}',
        'THINK: в файле написано «содержимое» | {}',
    ])
    (agent.fs_sandbox.root / "test.txt").write_text("содержимое", encoding="utf-8")
    task = agent.task_queue.add("прочитай test.txt и скажи, что там")
    reports = agent.runner.run(max_tasks=1, max_steps=5, max_llm_calls=5)

    assert len(reports) == 1
    report = reports[0]
    assert report.solved is True, report.stop_reason
    assert report.status == DONE
    assert len(report.steps) == 2
    assert report.steps[0].success is True
    assert report.llm_calls == 2
    assert report.tokens == 2 * 135
    assert agent.task_queue.get(task.id).status == DONE


def test_autonomous_stops_at_llm_budget_and_requeues_task():
    # Разные действия на каждом шаге: одинаковые раньше поймал бы детектор
    # зацикливания, и до бюджета дело не дошло бы.
    agent, state = scripted_agent([
        'READ: file:a.txt | {}',
        'READ: file:b.txt | {}',
        'READ: file:c.txt | {}',
    ])
    for name in ("a.txt", "b.txt", "c.txt"):
        (agent.fs_sandbox.root / name).write_text("x", encoding="utf-8")
    task = agent.task_queue.add("читай, пока не кончится бюджет")
    reports = agent.runner.run(max_tasks=1, max_steps=10, max_llm_calls=2)

    report = reports[0]
    assert report.llm_calls <= 2, "бюджет вызовов превышен"
    assert "бюджет" in report.stop_reason
    # не доведена — значит возвращается в очередь, а не помечается провалом
    assert report.status == PENDING
    assert agent.task_queue.get(task.id).status == PENDING
    assert agent.task_queue.pending().id == task.id


def test_autonomous_detects_loop():
    agent, state = scripted_agent([
        'READ: file:test.txt | {}',
        'READ: file:test.txt | {}',      # первый повтор — наставление
        'READ: file:test.txt | {}',      # второй повтор — стоп
    ])
    (agent.fs_sandbox.root / "test.txt").write_text("x", encoding="utf-8")
    agent.task_queue.add("зациклись")
    reports = agent.runner.run(max_tasks=1, max_steps=8, max_llm_calls=8)

    report = reports[0]
    assert "зациклился" in report.stop_reason, report.stop_reason
    assert len(report.steps) == 3, "одна попытка исправиться, затем стоп"
    assert report.llm_calls == 3
    assert "Не повторяй" in state["last_prompt"] or "Не повторяй его" in state["last_prompt"]


def test_autonomous_loop_nudge_lets_model_finish():
    """После наставления модель может ответить THINK вместо повтора."""
    agent, state = scripted_agent([
        'READ: file:test.txt | {}',
        'READ: file:test.txt | {}',
        'THINK: прочитал, там искомая строка | {}',
    ])
    (agent.fs_sandbox.root / "test.txt").write_text("искомая строка", encoding="utf-8")
    agent.task_queue.add("задача")
    reports = agent.runner.run(max_tasks=1, max_steps=6, max_llm_calls=6)

    report = reports[0]
    assert report.solved is True, report.stop_reason
    assert report.final_answer == "прочитал, там искомая строка"


def test_autonomous_stops_after_two_failures():
    agent, state = scripted_agent([
        'READ: file:нет-такого.txt | {}',
        'READ: file:тоже-нет.txt | {}',
        'READ: file:третий.txt | {}',
    ])
    agent.task_queue.add("читай несуществующее")
    reports = agent.runner.run(max_tasks=1, max_steps=8, max_llm_calls=8)

    report = reports[0]
    assert "две ошибки" in report.stop_reason, report.stop_reason
    assert len(report.steps) == 2
    assert all(not step.success for step in report.steps)


def test_autonomous_prompt_stays_compact():
    """История для модели обрезается: иначе расход растёт от объёма данных."""
    agent, state = scripted_agent(['THINK: готово | {}'])
    agent.task_queue.add("задача")
    long_body = "я" * 5000
    (agent.fs_sandbox.root / "big.txt").write_text(long_body, encoding="utf-8")
    agent.runner.run_task(agent.task_queue.pending(), max_steps=1, max_llm_calls=1)

    prompt = state["last_prompt"]
    assert len(prompt) < 1500, f"промпт раздулся до {len(prompt)} символов"


def test_autonomous_marks_pre_approved_actions_in_registry():
    agent, state = scripted_agent([
        'WRITE: file:auto.txt | {"content": "автономно"}',
        'THINK: записал | {}',
    ], policy="deny")
    agent.set_mode("autonomous")
    agent.autonomous_pre_approved = True          # оператор разрешил явно
    agent.task_queue.add("создай файл")
    reports = agent.runner.run(max_tasks=1, max_steps=4, max_llm_calls=4)

    assert reports[0].solved is True
    assert (agent.fs_sandbox.root / "auto.txt").read_text(encoding="utf-8") == "автономно"
    entries = [e for e in agent.registry.entries if e.success and e.metadata.get("tool")]
    assert entries, "успешное автономное действие должно быть в реестре"
    assert entries[-1].metadata.get("auto_approved") is True, entries[-1].metadata


def test_autonomous_denies_confirmation_tools_without_pre_approval():
    agent, state = scripted_agent([
        'WRITE: file:auto.txt | {"content": "нет"}',
    ], policy="allow")          # даже при confirm=allow: это просьба спросить, а не разрешение
    agent.set_mode("autonomous")
    agent.task_queue.add("создай файл")
    reports = agent.runner.run(max_tasks=1, max_steps=2, max_llm_calls=2)

    assert not (agent.fs_sandbox.root / "auto.txt").exists()
    assert reports[0].steps[0].success is False
    assert "autonomous" in reports[0].steps[0].refused


def test_autonomous_uses_compact_prompt_only_when_enabled():
    """Компактный промпт экономит токены, но ломает разбор — потому опция."""
    agent, state = scripted_agent([
        'READ: file:a.txt | {}',
        'THINK: готово | {}',
        'READ: file:a.txt | {}',      # второй прогон берёт свои два ответа
        'THINK: готово | {}',
    ])
    (agent.fs_sandbox.root / "a.txt").write_text("x", encoding="utf-8")
    agent.task_queue.add("задача")

    agent.runner.compact_prompt = False
    agent.runner.run_task(agent.task_queue.pending(), max_steps=3, max_llm_calls=3)
    assert state["last_system_prompt"] is None, "при выключенной опции промпт берётся из агента"

    agent.task_queue.reset(agent.task_queue.all()[0].id)
    agent.runner.compact_prompt = True
    agent.runner.run_task(agent.task_queue.pending(), max_steps=3, max_llm_calls=3)
    compact = agent.compact_planning_prompt()
    assert state["last_system_prompt"] == compact, "на продолжении должен идти компактный промпт"
    assert len(compact) < len(agent._build_system_prompt()) // 2


def test_autonomous_does_not_accept_format_failure_as_answer():
    """«Модель не поняла формат» не должно выглядеть как решённая задача."""
    agent, state = scripted_agent([
        'я не понял задание, вот мой текст',
        'тоже не формат',
    ])
    task = agent.task_queue.add("задача")
    reports = agent.runner.run(max_tasks=1, max_steps=6, max_llm_calls=6)

    report = reports[0]
    assert report.solved is False, "нельзя закрывать задачу ответом не по формату"
    assert "не в формате" in report.stop_reason, report.stop_reason
    assert report.status == FAILED
    assert report.steps[0].success is False
    assert report.steps[0].raw_answer.startswith("я не понял")
    assert agent.task_queue.get(task.id).status == FAILED


def test_autonomous_recovers_after_one_format_failure():
    """Одна осечка формата лечится наставлением, а не провалом задачи."""
    agent, state = scripted_agent([
        'не по формату',
        'THINK: всё-таки решил | {}',
    ])
    agent.task_queue.add("задача")
    reports = agent.runner.run(max_tasks=1, max_steps=6, max_llm_calls=6)

    report = reports[0]
    assert report.solved is True, report.stop_reason
    assert len(report.steps) == 2
    assert report.steps[0].success is False
    assert report.steps[1].verb == "think"
    assert "не соответствовал формату" in state["last_prompt"], "нужно наставление в промпте"
    assert report.final_answer == "всё-таки решил"


def test_autonomous_uses_zero_temperature_and_restores_it():
    agent, state = scripted_agent(['THINK: готово | {}'])
    agent.llm.temperature = 0.7
    agent.task_queue.add("задача")
    agent.runner.run(max_tasks=1, max_steps=2, max_llm_calls=2)
    assert agent.llm.temperature == 0.7, "температуру надо возвращать как было"


def test_autonomous_reports_unknown_token_usage_honestly():
    agent, state = scripted_agent(['THINK: готово | {}'])

    def fake_think(prompt, system_prompt=None, context=None):   # noqa: ANN001
        agent.llm.total_calls += 1
        agent.llm.unreported_calls += 1              # провайдер не сообщил расход
        return 'THINK: готово | {}'

    agent.think = fake_think
    agent.task_queue.add("задача")
    report = agent.runner.run_task(agent.task_queue.pending(), max_steps=2, max_llm_calls=2)
    assert report.llm_calls == 1
    assert report.tokens_reported is False


# ---------- диагностика провайдеров ----------

def test_probe_reports_working_provider():
    import json as _json

    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = _json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "да"}}]
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            body = _json.dumps({"data": [{"id": "test-model"}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # тишина в выводе тестов
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        from core.llm_client import LLMClient

        agent = make_agent(policy="allow")
        agent.llm = LLMClient(
            base_url=f"http://127.0.0.1:{server.server_port}",
            model="test-model", provider="stub", timeout=10, max_tokens=8,
            api_key="stub-key",           # сетевой провайдер без ключа не поедет
        )
        report = agent.probe_providers()
        main = report[0]
        assert main["role"] == "основной"
        assert main["works"] is True, main
        assert report[1]["works"] is False          # фолбэк не настроен
        assert report[1]["reason"] == "не настроен"
    finally:
        server.shutdown()
        server.server_close()


def test_probe_distinguishes_configured_from_working():
    """Ключ есть и указан адрес, но сервер отвечает ошибкой — значит НЕ работает."""
    from core.llm_client import LLMClient

    agent = make_agent(policy="allow")
    agent.llm = LLMClient(
        base_url="http://127.0.0.1:9",          # порт закрыт
        model="test-model", provider="dead-stub", timeout=3, max_tokens=8,
        api_key="stub-key",                     # ключ есть, но сервер мёртв
    )
    report = agent.probe_providers()
    assert report[0]["key_present"] is True
    assert report[0]["works"] is False           # ключ ≠ работоспособность
    assert report[0]["reason"], "причина отказа должна быть названа"


# ---------- режимы работы ----------

def test_mode_defaults_to_interactive():
    agent = make_agent(policy="allow")
    assert agent.mode in MODES
    assert agent.stats()["mode"] == agent.mode


def test_read_only_blocks_mutating_verbs():
    agent = make_agent(policy="allow")
    agent.set_mode("read-only")
    for verb, obj, params in (
        (VerbActType.WRITE, "file:test.txt", {"content": "x"}),
        (VerbActType.EXECUTE, "sys:uname -s", {}),
        (VerbActType.ANCHOR, "reg:канон", {}),
        (VerbActType.VETO, "swarm:спорное", {}),
    ):
        act = agent._make_act(verb, obj, params, "x", "test")
        result = agent.execute(act)
        assert act.success is False, f"{verb.value} прошёл в read-only: {result}"
        assert "read-only" in str(result), result


def test_mode_is_checked_before_gate():
    """Режим — ограничение оператора: он важнее роли и не «эскалируется»."""
    agent = make_agent(policy="allow")
    agent.role = "observer"            # роли не хватает даже на read/write
    agent.set_mode("read-only")
    act = agent._make_act(VerbActType.WRITE, "file:test.txt", {"content": "x"}, "x", "test")
    result = agent.execute(act)
    assert act.success is False
    assert "read-only" in str(result), result
    assert "Эскалация" not in str(result), result


def test_read_only_still_allows_read_and_log():
    agent = make_agent(policy="allow")
    (agent.fs_sandbox.root / "test.txt").write_text("содержимое", encoding="utf-8")
    agent.set_mode("read-only")
    read_act = agent._make_act(VerbActType.READ, "file:test.txt", {}, "x", "test")
    read_result = agent.execute(read_act)
    assert read_act.success is True, read_result
    log_act = agent._make_act(VerbActType.LOG, "reg:проверка режима", {}, "x", "test")
    log_result = agent.execute(log_act)
    assert log_act.success is True, log_result       # журнал — это аудит, он нужен и в read-only


def test_autonomous_denies_actions_that_need_confirmation():
    agent = make_agent(policy="allow")
    agent.set_mode("autonomous")
    act = agent._make_act(VerbActType.WRITE, "file:test.txt", {"content": "x"}, "x", "test")
    result = agent.execute(act)
    assert act.success is False, result
    assert "autonomous" in str(result), result
    # ничего не записано в файл
    assert not (agent.fs_sandbox.root / "test.txt").exists()


def test_autonomous_allows_read():
    agent = make_agent(policy="allow")
    (agent.fs_sandbox.root / "test.txt").write_text("содержимое", encoding="utf-8")
    agent.set_mode("autonomous")
    act = agent._make_act(VerbActType.READ, "file:test.txt", {}, "x", "test")
    result = agent.execute(act)
    assert act.success is True, result


def test_set_mode_rejects_unknown():
    agent = make_agent(policy="allow")
    assert_raises(ValueError, agent.set_mode, "турбо")


def test_mode_refusal_is_recorded_in_registry():
    agent = make_agent(policy="allow")
    agent.set_mode("read-only")
    before = len(agent.registry.entries)
    act = agent._make_act(VerbActType.EXECUTE, "sys:uname -s", {}, "x", "test")
    agent.execute(act)
    assert len(agent.registry.entries) == before + 1
    entry = agent.registry.entries[-1]
    assert entry.success is False
    assert entry.metadata.get("mode") == "read-only"


# ---------- краевые случаи posix-бэкенда ----------

def _edge_dir() -> Path:
    path = Path(tempfile.mkdtemp(prefix="muha-edge-"))
    (path / "big.txt").write_text("x" * 1_000_000, encoding="utf-8")
    (path / "bad.bin").write_bytes(b"\xff\xfe\x00abc\x80\x81")
    script = path / "orphan.sh"
    script.write_text("#!/bin/sh\nsleep 5 &\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    return path


def test_posix_large_output_is_truncated_without_deadlock():
    import modules.system as system_module

    system_module.set_backend("posix")
    system_module.set_enabled(True)
    work = _edge_dir()
    sandbox = SystemSandbox(allowed_commands=("cat",), cwd=work,
                           denied_patterns=(), max_output=8192)
    started = time.time()
    out = system_module.run_command("cat big.txt", sandbox)
    elapsed = time.time() - started
    assert elapsed < 5, f"чтение 1 МБ заняло {elapsed:.1f}s — похоже на блокировку трубы"
    assert "вывод обрезан" in out, out[-80:]
    assert len(out) < 20_000, len(out)


def test_posix_captures_stdout_and_stderr_together():
    import modules.system as system_module

    system_module.set_backend("posix")
    work = _edge_dir()
    sandbox = SystemSandbox(allowed_commands=("ls",), cwd=work, denied_patterns=())
    err = assert_raises(
        ToolError, system_module.run_command, "ls big.txt /нет-такого-xyz", sandbox
    )
    text = str(err)
    assert "--- stdout ---" in text, text
    assert "--- stderr ---" in text, text
    assert "код возврата: 1" in text, text


def test_posix_handles_invalid_utf8():
    import modules.system as system_module

    system_module.set_backend("posix")
    work = _edge_dir()
    sandbox = SystemSandbox(allowed_commands=("cat",), cwd=work, denied_patterns=())
    out = system_module.run_command("cat bad.bin", sandbox)
    assert "stdout" in out          # не упало на декодировании
    assert "\ufffd" in out or "abc" in out


def test_posix_orphan_child_does_not_stall():
    """Потомок держит трубу после выхода ребёнка — ждать таймаут нельзя."""
    import modules.system as system_module

    system_module.set_backend("posix")
    work = _edge_dir()
    sandbox = SystemSandbox(
        allowed_commands=("sh",), cwd=work, timeout=10, denied_patterns=(),
        privileged_commands=("sh",), privileged_roles=("guardian",),
    )
    started = time.time()
    system_module.run_command("sh orphan.sh", sandbox, role="guardian")
    elapsed = time.time() - started
    assert elapsed < 3, f"осиротевший ребёнок задержал команду на {elapsed:.1f}s"


# ---------- коннектом ----------

def test_connectome_csv_parser():
    assert _parse_csv_line('a,b,"c,d",e') == ["a", "b", "c,d", "e"]
    assert _parse_csv_line('x,"со ""кавычками""",z') == ["x", 'со "кавычками"', "z"]


def test_connectome_real_dataset_loads():
    conn = Connectome.open(CONNECTome_DIR).load()
    assert conn.loaded, conn.load_error
    assert len(conn.nodes) == 1515, len(conn.nodes)
    total_edges = sum(len(v) for v in conn.outgoing.values())
    assert total_edges == 58674, total_edges
    stats = conn.stats()
    assert stats["available"] is True
    assert stats["dataset_id"].startswith("flywire")


def test_connectome_neuron_and_neighbors():
    conn = Connectome.open(CONNECTome_DIR)
    report = conn.neuron_report(0)
    assert "#0" in report
    neighbors = conn.neighbors(0, direction="out", limit=3)
    assert "исходящие" in neighbors
    # регрессия: `extend(...) or append("(нет)")` печатал «(нет)» всегда
    assert "(нет)" not in neighbors, neighbors
    assert neighbors.count("->") == 3, neighbors


def test_connectome_find_and_classes():
    conn = Connectome.open(CONNECTome_DIR)
    found = conn.find("DNpe", limit=5)
    assert "найдено" in found
    classes = conn.classes_report()
    assert "ascending" in classes or "классы узлов" in classes


def test_connectome_missing_dir_is_graceful():
    conn = Connectome.open("/nonexistent/path/xyz")
    assert conn.available is False
    stats = conn.stats()
    assert stats["available"] is False and stats["error"]
    assert_raises(ToolError, conn.find, "DNpe")


# ---------- грамматика и гейт ----------

def test_grammar_escalate_and_select_are_reachable():
    for verb in (VerbActType.ESCALATE, VerbActType.SELECT):
        domains = [d for d in VerbActGrammar.DOMAIN_RULES
                   if VerbActGrammar.is_allowed(d, verb)]
        assert domains, f"{verb.value} запрещён во всех доменах"


def test_grammar_connectome_domain():
    assert VerbActGrammar.domain_for_object("connectome:stats") == "connectome"
    assert VerbActGrammar.domain_for_object("neuron:12") == "connectome"
    assert VerbActGrammar.is_allowed("connectome", VerbActType.READ)
    assert not VerbActGrammar.is_allowed("connectome", VerbActType.EXECUTE)


def test_grammar_object_prefixes():
    assert VerbActGrammar.domain_for_object("file:x") == "filesystem"
    assert VerbActGrammar.domain_for_object("dir:x") == "filesystem"
    assert VerbActGrammar.domain_for_object("sys:ls") == "system"
    assert VerbActGrammar.domain_for_object("reg:x") == "registry"
    assert VerbActGrammar.domain_for_object("swarm:x") == "swarm"
    assert VerbActGrammar.domain_for_object("просто текст") == "ui"
    assert VerbActGrammar.domain_for_object("/abs/path") == "filesystem"


def test_gate_denies_execute_without_prefix():
    gk = GateKeeper()
    act = VerbAct.create(verb=VerbActType.EXECUTE, object="ls -la")
    assert gk.check(act, user_role="decider").decision is GateDecision.DENY
    act2 = VerbAct.create(verb=VerbActType.EXECUTE, object="sys:ls")
    assert gk.check(act2, user_role="decider").decision is GateDecision.ALLOW


def test_gate_escalates_anchor_for_decider():
    gk = GateKeeper()
    act = VerbAct.create(verb=VerbActType.ANCHOR, object="reg:x")
    result = gk.check(act, user_role="decider")
    assert result.decision is GateDecision.ESCALATE
    assert result.required_role == "registrar"


# ---------- разбор ответов LLM ----------

def test_parse_text_format():
    agent = make_agent()
    act = agent._parse_verb_act("READ: file:config.yaml | {}", "x")
    assert act.verb is VerbActType.READ
    assert act.object == "file:config.yaml"
    assert act.domain == "filesystem"


def test_parse_with_parameters():
    agent = make_agent()
    act = agent._parse_verb_act('WRITE: file:a.txt | {"content": "hi"}', "x")
    assert act.verb is VerbActType.WRITE
    assert act.parameters == {"content": "hi"}


def test_parse_json_format():
    agent = make_agent()
    payload = json.dumps({"verb": "read", "object": "connectome:stats", "parameters": {}})
    act = agent._parse_verb_act(payload, "x")
    assert act.verb is VerbActType.READ
    assert act.domain == "connectome"
    assert act.parse_reason == "json"


def test_parse_code_fence():
    agent = make_agent()
    act = agent._parse_verb_act("```\nREAD: file:config.yaml | {}\n```", "x")
    assert act.verb is VerbActType.READ


def test_parse_russian_verb():
    agent = make_agent()
    act = agent._parse_verb_act("прочитай: file:config.yaml | {}", "x")
    assert act.verb is VerbActType.READ


def test_parse_garbage_falls_back_to_think():
    agent = make_agent()
    act = agent._parse_verb_act("не знаю, что делать", "исходный текст")
    assert act.verb is VerbActType.THINK
    assert act.object == "исходный текст"
    assert "не вернул формат" in (act.parse_reason or "")


# ---------- исполнение ----------

def test_execute_read_uses_real_tool():
    agent = make_agent()
    (agent.fs_sandbox.root / "hello.txt").write_text("содержимое", encoding="utf-8")
    act = agent._make_act(VerbActType.READ, "file:hello.txt", {}, "x", "test")
    result = agent.execute(act)
    assert "содержимое" in str(result), result
    assert act.success is True
    assert act.tool == "fs.read"
    assert agent.registry.entries[-1].success is True
    assert agent.registry.verify_chain()


def test_execute_write_requires_confirmation_and_denies():
    agent = make_agent(policy="deny")
    act = agent._make_act(
        VerbActType.WRITE, "file:x.txt", {"content": "hi"}, "x", "test"
    )
    result = agent.execute(act)
    assert "Отменено" in str(result), result
    assert not (agent.fs_sandbox.root / "x.txt").exists()
    assert agent.registry.entries[-1].success is False


def test_execute_write_with_allow_policy_performs_io():
    agent = make_agent(policy="allow")
    act = agent._make_act(
        VerbActType.WRITE, "file:out/note.txt", {"content": "муха летит"}, "x", "test"
    )
    result = agent.execute(act)
    assert act.success is True, result
    written = agent.fs_sandbox.root / "out" / "note.txt"
    assert written.read_text("utf-8") == "муха летит"
    assert act.tool == "fs.write"


def test_execute_write_without_content_fails():
    agent = make_agent(policy="allow")
    act = agent._make_act(VerbActType.WRITE, "file:empty.txt", {}, "x", "test")
    result = agent.execute(act)
    assert act.success is False
    assert "content" in str(result)


def test_execute_connectome_tool():
    agent = make_agent()
    act = agent._make_act(VerbActType.READ, "connectome:stats", {}, "x", "test")
    result = agent.execute(act)
    assert act.success is True, result
    assert "flywire" in str(result)
    assert act.tool == "connectome.query"


def test_execute_select_tool():
    agent = make_agent()
    act = agent._make_act(
        VerbActType.SELECT, "варианты",
        {"options": ["a", "b", "c"], "strategy": "last"}, "x", "test",
    )
    result = agent.execute(act)
    assert act.success is True, result
    assert "выбрано: c" in str(result)


def test_execute_escalate_tool():
    agent = make_agent()
    act = agent._make_act(VerbActType.ESCALATE, "swarm:спор", {}, "x", "test")
    result = agent.execute(act)
    assert act.success is True, result
    assert "эскалировано" in str(result)


def test_execute_unknown_tool_is_recorded_as_failure():
    agent = make_agent()
    act = agent._make_act(VerbActType.WRITE, "просто текст", {"content": "x"}, "x", "test")
    result = agent.execute(act)
    assert act.success is False
    assert "Запрещено" in str(result) or "нет инструмента" in str(result)
    assert agent.registry.verify_chain()


def test_execute_system_command_disabled_is_graceful():
    """При commands_enabled=false execute не висит, а честно отказывает."""
    import modules.system as system_module

    agent = make_agent(policy="allow")
    agent.system_sandbox.allowed_commands = ("echo",)
    previous = system_module.is_enabled()
    system_module.set_enabled(False)
    try:
        act = agent._make_act(VerbActType.EXECUTE, "sys:echo муха", {}, "x", "test")
        result = agent.execute(act)
    finally:
        system_module.set_enabled(previous)
    assert act.success is False
    assert "commands_enabled" in str(result), result


def test_execute_system_command_via_tool():
    """Реальный запуск процесса на бэкенде posix (без модуля subprocess)."""
    import modules.system as system_module

    agent = make_agent(policy="allow")
    agent.system_sandbox.allowed_commands = ("echo",)
    system_module.set_enabled(True)
    system_module.set_backend("posix")
    act = agent._make_act(VerbActType.EXECUTE, "sys:echo муха", {}, "x", "test")
    result = agent.execute(act)
    assert act.success is True, result
    assert "муха" in str(result), result
    assert "код возврата: 0" in str(result)


def test_system_posix_reports_nonzero_exit_as_failure():
    import modules.system as system_module

    sandbox = SystemSandbox(allowed_commands=("ls",), denied_patterns=())
    system_module.set_backend("posix")
    err = assert_raises(
        ToolError, system_module.run_command, "ls /нет-такого-каталога-xyz", sandbox
    )
    assert "код возврата" in str(err)


def test_system_posix_timeout_kills_process():
    import modules.system as system_module

    sandbox = SystemSandbox(allowed_commands=("sleep",), timeout=1)
    system_module.set_backend("posix")
    started = time.time()
    err = assert_raises(ToolError, system_module.run_command, "sleep 5", sandbox)
    elapsed = time.time() - started
    assert "таймаут" in str(err), err
    assert elapsed < 4, f"таймаут не сработал вовремя: {elapsed:.1f}s"


def test_system_backend_validation():
    import modules.system as system_module

    assert_raises(ToolError, system_module.set_backend, "windows")
    system_module.set_backend("posix")
    assert system_module.get_backend() == "posix"


def test_execute_denied_by_allowlist_is_failure():
    agent = make_agent(policy="allow")
    agent.system_sandbox.allowed_commands = ("echo",)
    act = agent._make_act(VerbActType.EXECUTE, "sys:rm -rf /tmp/x", {}, "x", "test")
    result = agent.execute(act)
    assert act.success is False
    assert "allowlist" in str(result) or "запрещённый шаблон" in str(result)


# ---------- все 12 инструментов трогают настоящее состояние ----------

def test_registry_log_writes_real_entry():
    agent = make_agent()
    before = len(agent.registry.entries)
    act = agent._make_act(VerbActType.LOG, "reg:проверка", {}, "x", "test")
    result = agent.execute(act)
    assert act.success is True, result
    assert act.tool == "registry.log"
    assert len(agent.registry.entries) > before
    assert agent.registry.verify_chain()


def test_registry_version_reads_real_state():
    agent = make_agent()
    agent.role = "registrar"   # version в домене registry доступен только registrar
    act = agent._make_act(VerbActType.VERSION, "reg:x", {}, "x", "test")
    result = str(agent.execute(act))
    assert act.success is True, result
    assert "1.1.0" in result
    assert "sha256" in result
    assert "цепь валидна" in result


def test_registry_anchor_escalates_for_decider():
    agent = make_agent()
    act = agent._make_act(VerbActType.ANCHOR, "reg:канон", {}, "x", "test")
    result = str(agent.execute(act))
    assert act.success is False
    assert "Эскалация" in result and "registrar" in result


def test_registry_anchor_works_for_registrar():
    agent = make_agent()
    agent.role = "registrar"
    before = len(agent.registry.entries)
    act = agent._make_act(VerbActType.ANCHOR, "reg:канон", {}, "x", "test")
    result = str(agent.execute(act))
    assert act.success is True, result
    assert "заякорено: канон" in result
    assert len(agent.registry.entries) > before


def test_swarm_veto_deposits_pheromone():
    agent = make_agent()
    agent.role = "guardian"
    before = agent.swarm.field.stats()["total_pheromones"]
    act = agent._make_act(VerbActType.VETO, "swarm:спор", {}, "x", "test")
    result = str(agent.execute(act))
    assert act.success is True, result
    assert agent.swarm.field.stats()["total_pheromones"] == before + 1
    vetoes = [p for p in agent.swarm.field.pheromones if p.type == "veto"]
    assert vetoes, "феромон вето не оставлен"


def test_swarm_escalate_deposits_pheromone():
    agent = make_agent()
    before = agent.swarm.field.stats()["total_pheromones"]
    act = agent._make_act(VerbActType.ESCALATE, "swarm:спор", {}, "x", "test")
    result = str(agent.execute(act))
    assert act.success is True, result
    assert agent.swarm.field.stats()["total_pheromones"] == before + 1


def test_ui_recommend_uses_real_state():
    agent = make_agent()
    act = agent._make_act(VerbActType.RECOMMEND, "план", {}, "x", "test")
    result = str(agent.execute(act))
    assert act.success is True, result
    assert "риск-аппетит" in result
    assert "последних" in result


def test_fs_stat_tool():
    agent = make_agent()
    (agent.fs_sandbox.root / "stat.txt").write_text("abc", encoding="utf-8")
    act = agent._make_act(VerbActType.SELECT, "file:stat.txt", {}, "x", "test")
    result = str(agent.execute(act))
    assert act.success is True, result
    assert act.tool == "fs.stat"
    assert "размер: 3 байт" in result


def test_every_tool_is_reachable():
    """Каждый инструмент должен разрешаться хотя бы на одной паре глагол+домен."""
    agent = make_agent()
    unreachable = [
        spec.name for spec in agent.tools.roster_specs()
        if not any(spec.matches(verb, domain)
                   for verb in VerbActType for domain in spec.domains)
    ]
    assert not unreachable, f"недостижимые инструменты: {unreachable}"


# ---------- реестр ----------

def test_registry_chain_and_tamper():
    tmp = tempfile.mkdtemp(prefix="muha-reg-")
    registry = Registry(path=os.path.join(tmp, "r.jsonl"))
    for i in range(3):
        registry.record(action=f"a{i}", actor="t", result="ok", success=True)
    assert registry.verify_chain()
    registry.entries[1].result = "подмена"
    assert registry.verify_chain() is False


def test_registry_bare_filename():
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="muha-reg-")
    try:
        os.chdir(tmp)
        Registry(path="bare.jsonl").record(action="a", actor="t", result="r")
    finally:
        os.chdir(cwd)


# ---------- золотые яблоки ----------

def test_golden_apples_canonize_with_three_voters():
    ga = GoldenApples(vote_threshold=0.66, min_votes=3)
    apple_id = ga.propose("канон")
    for voter in ("analyst", "decider", "guardian"):
        assert ga.vote(apple_id, voter, True)
    assert ga.get_apple(apple_id).canonical is True
    assert ga.stats()["canonical"] == 1


def test_golden_apples_single_voter_does_not_stack():
    ga = GoldenApples(vote_threshold=0.66, min_votes=3)
    apple_id = ga.propose("повтор")
    for _ in range(3):
        ga.vote(apple_id, "decider", True)
    assert len(ga.get_apple(apple_id).votes) == 1
    assert ga.get_apple(apple_id).canonical is False


# ---------- LLM-клиент ----------

def test_api_style_detection():
    assert LLMClient(provider="ollama").api_style == "ollama"
    assert LLMClient(provider="oneprovider").api_style == "openai"
    assert LLMClient(provider="openrouter").api_style == "openai"


def test_key_resolution_from_credentials_file():
    tmp = tempfile.mkdtemp(prefix="muha-creds-")
    path = Path(tmp) / "creds.yaml"
    path.write_text("ONEPROVIDER_API_KEY: test-key-123\n", encoding="utf-8")
    key = resolve_api_key("ONEPROVIDER_API_KEY", str(path))
    assert key == "test-key-123"
    assert resolve_api_key("ABSENT_VAR", str(path)) is None


def test_openai_client_requires_key():
    client = LLMClient(provider="oneprovider", api_key_env="DEFINITELY_ABSENT_VAR",
                       credentials_file=None)
    err = assert_raises(LLMUnavailable, client.chat, "привет")
    assert "нет ключа" in str(err)


def test_openai_client_against_stub():
    """Проверяет протокол /chat/completions на локальной заглушке."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = json.dumps({"data": [{"id": "deepseek-v4.1-flash"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            captured["payload"] = json.loads(self.rfile.read(length).decode())
            captured["auth"] = self.headers.get("Authorization")
            answer = json.dumps({
                "model": "deepseek-v4.1-flash",
                "choices": [{"message": {"role": "assistant",
                                        "content": "READ: connectome:stats | {}"}}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = LLMClient(
            base_url=f"http://127.0.0.1:{port}/v1",
            model="deepseek-v4.1-flash",
            provider="oneprovider",
            api_key="test-key",
            timeout=10,
        )
        assert client.is_available() is True
        assert client.ensure_model_available() is True
        answer = client.chat("статистика коннектома", system_prompt="ты муха")
        assert answer == "READ: connectome:stats | {}"
        payload = captured["payload"]
        assert payload["model"] == "deepseek-v4.1-flash"
        assert payload["stream"] is False
        assert payload["messages"][0]["role"] == "system"
        assert captured["auth"] == "Bearer test-key"
    finally:
        server.shutdown()


def test_think_returns_error_marker_when_all_providers_fail():
    agent = make_agent()
    agent.llm = LLMClient(base_url="http://127.0.0.1:9/v1", provider="oneprovider",
                          api_key="x", timeout=2)
    agent.fallback_llm = None
    text = agent.think("привет")
    assert text.startswith(Agent.LLM_ERROR_PREFIX)


# ---------- конфигурация ----------

def test_miniyaml_parses_config():
    from core.miniyaml import load_file

    cfg = load_file(CONFIG)
    assert cfg["llm"]["provider"] == "oneprovider"
    assert cfg["llm"]["temperature"] == 0.1
    assert cfg["connectome"]["enabled"] is True
    assert cfg["tools"]["command_timeout"] == 10
    assert isinstance(cfg["gatekeeper"]["roles"]["decider"], list)
    assert "escalate" in cfg["gatekeeper"]["roles"]["observer"]
    assert cfg["cli"]["colors"]["primary"] == "#ffb000"


def test_miniyaml_scalars_and_lists():
    from core.miniyaml import loads

    data = loads(
        "a: 1\n"
        "b: 2.5\n"
        "c: true\n"
        "d: null\n"
        "e: [x, \"y, z\", 3]\n"
        "f:\n"
        "  g: вложено\n"
        "h:\n"
        "  - один\n"
        "  - два\n"
        "i: \"#не комментарий\"  # а это комментарий\n"
    )
    assert data["a"] == 1 and data["b"] == 2.5
    assert data["c"] is True and data["d"] is None
    assert data["e"] == ["x", "y, z", 3]
    assert data["f"]["g"] == "вложено"
    assert data["h"] == ["один", "два"]
    assert data["i"] == "#не комментарий"


def test_miniyaml_rejects_unsupported():
    from core.miniyaml import YamlError, loads

    assert_raises(YamlError, loads, "a: |\n  текст\n")
    assert_raises(YamlError, loads, "a: &anchor 1\n")


# ---------- терминал ----------

def test_terminal_styles_are_valid_rich():
    from cli.terminal import TerminalUI
    from rich.console import Console
    from rich.errors import MissingStyle

    agent = make_agent()
    ui = TerminalUI(agent)
    assert ui._style("amber") == agent.config["cli"]["colors"]["primary"]
    console = Console()
    for name in ("amber", "warning", "success", "info", "secondary", "white", "cyan"):
        console.get_style(ui._style(name))  # не должно бросить MissingStyle


def test_terminal_llm_label_is_honest():
    from cli.terminal import TerminalUI

    agent = make_agent()
    ui = TerminalUI(agent)
    label = ui._llm_label()
    assert label in ("готов", "недоступен") or "не подтверждена" in label


# ---------- отражение ----------

def test_reflect_reports_paradox_level():
    agent = make_agent()
    result = agent.reflect("проверка")
    assert 0.0 <= result["paradox_level"] <= 1.0
    assert result["description"] == "проверка"
