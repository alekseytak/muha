"""Tests for ProvenanceLog."""
from __future__ import annotations
import os, tempfile, json, time, pytest
from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog, _canonical_json, _sha256


@pytest.fixture
def log():
    tmp = tempfile.mktemp(suffix=".jsonl")
    p = ProvenanceLog(path=tmp)
    yield p
    for path in (tmp, tmp + ".head.json", tmp + ".head.json.tmp"):
        if os.path.exists(path): os.remove(path)


class TestAppendVerify:
    def test_three_events_verified(self, log):
        log.append({"event": "a"}); log.append({"event": "b"}); log.append({"event": "c"})
        assert log.count == 3; assert log.verify_chain() is True
    def test_single_verified(self, log):
        log.append({"event": "only"})
        assert log.verify_chain() is True


class TestTamperDetection:
    def test_tampered_payload_in_memory_fails(self, log):
        log.append({"event": "a"}); log.append({"event": "b"})
        events = log.get_events(); events[0].payload["event"] = "tampered"
        assert log.verify_chain() is False

    def test_corrupted_entry_hash_fails(self, log):
        log.append({"event": "a"})
        log.append({"event": "b"})
        assert log.verify_chain() is True
        entries = log.get_events()
        with open(log.path, "w") as f:
            for i, e in enumerate(entries):
                entry = e.to_log_entry()
                if i == 0:
                    entry["payload_hash"] = "0" * 64
                f.write(__import__("json").dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        new_log = ProvenanceLog(path=log.path)
        assert new_log.verify_chain() is False


class TestCanonicalHash:
    def test_deterministic_same_content(self):
        d1 = {"b": 1, "a": 2}; d2 = {"a": 2, "b": 1}
        assert _sha256(_canonical_json(d1)) == _sha256(_canonical_json(d2))
    def test_different_content_different_hash(self):
        assert _sha256(_canonical_json({"k": "v1"})) != _sha256(_canonical_json({"k": "v2"}))
    def test_compact_no_spaces(self):
        assert " " not in _canonical_json({"key": "value"})


class TestChainIntegrity:
    def test_hash_depends_on_previous(self, log):
        e1 = log.append({"s": 1}); e2 = log.append({"s": 2})
        assert e1.entry_hash == e2.previous_hash; assert e2.entry_hash != e1.entry_hash
    def test_empty_chain_verified(self, log):
        assert log.verify_chain() is True; assert log.count == 0


class TestOnDiskTamperDetection:
    def test_tampered_file_fails_new_log(self, log):
        log.append({"event": "a"})
        log.append({"event": "b"})
        path = log.path
        with open(path, "r") as f:
            lines = f.readlines()
        entry = json.loads(lines[0])
        entry["payload"]["event"] = "HACKED"
        with open(path, "w") as f:
            f.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
            f.write(lines[1])
        new_log = ProvenanceLog(path=path)
        assert new_log.verify_chain() is False

    def test_missing_entry_fails_writer_log(self, log):
        """A writer that already recorded 2 events must notice the file shrinking."""
        log.append({"event": "a"})
        log.append({"event": "b"})
        path = log.path
        with open(path, "r") as f:
            lines = f.readlines()
        with open(path, "w") as f:
            f.write(lines[0])
        assert log.verify_chain() is False

    def test_truncated_log_is_detected_against_the_witness(self, log):
        """Свидетель в файле закрывает дыру, которую не закрывал свежий читатель.

        Раньше укороченный, но внутренне валидный цепной хвост выглядил для
        newcomer'а нормально. Теперь рядом лежит head witness с числом записей,
        смещением и последним хэшем, и расхождение видно даже процессу, который
        ничего не писал.
        """
        log.append({"event": "a"})
        log.append({"event": "b"})
        path = log.path
        with open(path, "r") as f:
            lines = f.readlines()
        with open(path, "w") as f:
            f.write(lines[0])
        assert ProvenanceLog(path=path).verify_chain() is False

    def test_without_the_witness_a_fresh_reader_still_cannot_see_truncation(self, log):
        """Честное ограничение: witness — это тоже файл на том же диске.

        Если вместе с журналом удалён и witness, свежий читатель снова остаётся
        один на один с короткой, но валидной цепью: отличить её от украденной
        нельзя. Внешнего анкоринга в P3 нет и здесь.
        """
        log.append({"event": "a"})
        log.append({"event": "b"})
        path = log.path
        with open(path, "r") as f:
            lines = f.readlines()
        with open(path, "w") as f:
            f.write(lines[0])
        os.remove(log.head_path)
        assert ProvenanceLog(path=path).verify_chain() is True

    def test_rewritten_witness_is_detected(self, log):
        """Порча самого head witness ловится: цифры обязаны сходиться с журналом."""
        log.append({"event": "a"})
        log.append({"event": "b"})
        head = json.loads(open(log.head_path, encoding="utf-8").read())
        head["entry_count"] = 5
        with open(log.head_path, "w", encoding="utf-8") as f:
            json.dump(head, f, sort_keys=True)
        assert ProvenanceLog(path=log.path).verify_chain() is False
        head["entry_count"] = 2
        head["log_byte_offset"] = head["log_byte_offset"] + 1
        with open(log.head_path, "w", encoding="utf-8") as f:
            json.dump(head, f, sort_keys=True)
        assert ProvenanceLog(path=log.path).verify_chain() is False

    def test_forged_log_with_recomputed_witness_is_out_of_scope(self, log):
        """Честная граница tamper-EVIDENT: перестроить журнал и witness целиком
        вместе — детекту не под силу; это и есть причина, по которой журнал не
        назван неизменяемым реестром."""
        for event in ("a", "b", "c"):
            log.append({"event": event})
        path = log.path
        fresh = ProvenanceLog(path=path)
        fresh.clear()
        rewritten = ProvenanceLog(path=path)
        rewritten.append({"event": "a"})
        rewritten.append({"event": "B YANKED"})
        assert rewritten.verify_chain() is True  # внутренне согласованная цепь
        assert fresh.count == 2

    def test_unreadable_witness_fails_verification_instead_of_crashing(self, log):
        """Нечитаемый head witness — провал проверки, а не трейсбек в аудите."""
        log.append({"event": "a"})
        with open(log.head_path, "w", encoding="utf-8") as f:
            f.write("{не json")
        assert ProvenanceLog(path=log.path).verify_chain() is False

    def test_recovery_reconciles_log_that_ran_ahead_of_the_witness(self, log, tmp_path):
        """Падение между fsync журнала и заменой witness.

        Строка оказалась в журнале, а witness — нет. Восстановление обязано
        довести witness до правды и продолжить цепь от фактического хвоста, а не
        от сброшенного значения и не «с нуля».
        """
        log.append({"event": "a"})
        e2 = log.append({"event": "b"})
        line = _canonical_json(e2.to_log_entry()) + "\n"
        # моделируем crash: witness откатывается на одну запись назад
        head = json.loads(open(log.head_path, encoding="utf-8").read())
        head["entry_count"] = 1
        head["last_hash"] = e2.previous_hash
        head["log_byte_offset"] = len(open(log.path, encoding="utf-8").read().splitlines()[0]) + 1
        with open(log.head_path, "w", encoding="utf-8") as f:
            json.dump(head, f, sort_keys=True)
        reopened = ProvenanceLog(path=log.path)
        assert reopened.count == 2                      # хвост дочитан
        e3 = reopened.append({"event": "c"})
        assert e3.previous_hash == e2.entry_hash         # цепь продолжена, не сброшена
        assert reopened.verify_chain() is True
        assert reopened.count == 3

    def test_append_does_not_reparse_the_whole_log(self):
        """Старый путь O(n) на append обязан отсутствовать в коде, а не только в замерах."""
        import inspect

        append_src = inspect.getsource(ProvenanceLog.append)
        resync_src = inspect.getsource(ProvenanceLog._resync_locked)
        assert "_read_disk_events" not in append_src + resync_src
        # единственный допустимый полный проход — legacy-ветка без witness
        assert "_events_from(self._read_all(f))" in resync_src
        assert 'f.seek(head["log_byte_offset"])' in resync_src

    def test_append_cost_stays_bounded_as_the_log_grows(self, tmp_path):
        """10 000 append'ов: стоимость одной записи не должна расти с длиной журнала.

        Это ровно тот дефект, из-за которого P4.2.v1 был прерван: append перечитывал
        весь JSONL, суммарно получалось O(N^2), и 95% времени прогона уходило на
        запись журнала. Граница acceptance: медиана последних 2 000 не более чем втрое
        больше медианы первых 2 000.
        """
        import statistics

        path = str(tmp_path / "grow.jsonl")
        p = ProvenanceLog(path=path)
        timings = []
        for i in range(10_000):
            t = time.perf_counter()
            p.append({"i": i})
            timings.append(time.perf_counter() - t)
        first = statistics.median(timings[:2000])
        last = statistics.median(timings[-2000:])
        assert p.count == 10_000
        assert p.verify_chain() is True
        assert last <= first * 3, f"append дорожает с длиной: {first * 1e3:.2f} мс -> {last * 1e3:.2f} мс"

    def test_two_processes_append_without_forking_the_chain(self, log):
        """Inter-process flock: concurrent appenders converge to one chain."""
        import subprocess, sys, textwrap

        child = textwrap.dedent(
            """
            import sys
            sys.path.insert(0, {root!r})
            from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog
            log = ProvenanceLog(path=sys.argv[1])
            for i in range(25):
                log.append({{"pid": __import__('os').getpid(), "n": i}})
            """
        ).format(root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        path = log.path
        procs = [subprocess.Popen([sys.executable, "-c", child, path]) for _ in range(2)]
        for p in procs:
            assert p.wait() == 0
        # The parent never wrote to this file; both children appended to it.
        witness = ProvenanceLog(path=path)
        assert witness.count == 50
        assert witness.verify_chain() is True
