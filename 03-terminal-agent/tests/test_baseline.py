"""Сквозная приёмка узкого baseline: файл, команда, чтение веба."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from test_agent import make_agent
from core.verb_act import VerbActGrammar


def test_file_write_read_exact_bytes():
    agent = make_agent()
    write = agent.propose_action_fast('запиши в result.txt строку готово-42')
    assert write is not None
    agent.execute(write)
    assert write.success, write.result
    assert (agent.fs_sandbox.root / 'result.txt').read_bytes() == 'готово-42'.encode()
    read = agent.propose_action_fast('прочитай файл result.txt')
    assert read is not None and read.verb.value == 'read'
    assert read.domain == VerbActGrammar.domain_for_object(read.object)
    assert 'готово-42' in agent.execute(read)
    assert read.success and agent.registry.entries[-1].success


def test_command_arguments_reach_real_process():
    agent = make_agent()
    act = agent.propose_action_fast('запусти команду echo BASELINE_ARG_42')
    assert act is not None
    result = agent.execute(act)
    assert act.success, result
    assert '--- stdout ---\nBASELINE_ARG_42' in result, result


def test_negation_does_not_reach_confirmation_or_process():
    agent = make_agent()
    confirmations = []
    agent._ask_confirmation = lambda *args: confirmations.append(args) or True
    agent.think = lambda *a, **kw: 'EXECUTE: sys:echo SHOULD_NOT_RUN | {}'
    act = agent.propose_action('не запускай команду echo SHOULD_NOT_RUN')
    result = agent.execute(act)
    assert not act.success, result
    assert not confirmations
    assert not agent.action_history
    assert agent.registry.entries[-1].success is False


def test_web_read_through_real_http_and_tool():
    import http.server
    import threading
    import modules.web as web

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'<html><article><p>BASELINE_WEB_42</p></article></html>'
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original = web._host_is_private
    try:
        # Исключение только для принадлежащего тесту loopback-сервера.
        web._host_is_private = lambda host: False if host == '127.0.0.1' else original(host)
        agent = make_agent()
        act = agent.propose_action_fast(f'прочитай http://127.0.0.1:{server.server_port}/page')
        assert act is not None and act.domain == 'web'
        assert act.domain == VerbActGrammar.domain_for_object(act.object)
        result = agent.execute(act)
        assert act.success and 'BASELINE_WEB_42' in result, result
        assert agent.registry.entries[-1].success
    finally:
        web._host_is_private = original
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_incomplete_parameters_abstain_and_plain_read_stays_local():
    agent = make_agent()
    assert agent.propose_action_fast('запиши в notes.md') is None
    assert agent.propose_action_fast('выбери вариант') is None
    agent.think = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('лишний LLM'))
    act = agent.propose_action('прочитай файл notes.md')
    assert act.verb.value == 'read'
    assert act.domain == VerbActGrammar.domain_for_object(act.object)


if __name__ == '__main__':
    failures = []
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(name, 'PASS')
            except AssertionError as exc:
                failures.append(name)
                print(name, 'FAIL:', exc)
    raise SystemExit(bool(failures))
