"""Tests for ProvenanceLog."""
from __future__ import annotations
import os, tempfile, json, time, pytest
from fly_connectome_agent.src.engineering.logging.provenance_log import (
    ProvenanceLog, ProvenanceIntegrityError, _canonical_json, _head_digest, _sha256,
)


def _forge_head(head_path, **changes):
    """Переписать witness с новым набором полей И честным digest.

    Без пересчёта digest подделка ловилась бы само-хэшем witness'а — это отдельный
    (лёгкий) случай. Здесь же проверяется то, что должно ловиться независимо: witness,
    внутренне согласованный, но не согласованный с журналом.
    """
    head = json.loads(open(head_path, encoding="utf-8").read())
    head.update(changes)
    head["head_digest"] = _head_digest(head)
    with open(head_path, "w", encoding="utf-8") as f:
        json.dump(head, f, sort_keys=True, separators=(",", ":"))
    return head


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
        от сброшенного значения и не «с нуля». Откатанный witness остаётся
        внутренне валидным (digest пересчитан) — ловить его должен не само-хэш,
        а сверка с журналом.
        """
        log.append({"event": "a"})
        e2 = log.append({"event": "b"})
        # моделируем crash: witness откатывается на одну запись назад
        data = open(log.path, "rb").read()
        _forge_head(log.head_path, entry_count=1, last_hash=e2.previous_hash,
                    log_byte_offset=data.index(b"\n") + 1)
        reopened = ProvenanceLog(path=log.path)
        assert reopened.count == 2                      # хвост дочитан
        e3 = reopened.append({"event": "c"})
        assert e3.previous_hash == e2.entry_hash         # цепь продолжена, не сброшена
        assert reopened.verify_chain() is True
        assert reopened.count == 3

    def test_append_does_not_reparse_the_whole_log(self):
        """Старый путь O(n) на append обязан отсутствовать в коде, а не только в замерах."""
        import inspect

        hot_src = inspect.getsource(ProvenanceLog.append) + inspect.getsource(ProvenanceLog._resync_locked)
        for forbidden in ("_read_disk_events", "_read_all", "self.recover_head("):
            assert forbidden not in hot_src, f"горячий путь снова зовёт {forbidden}"
        assert 'f.seek(offset)' in hot_src or 'log_byte_offset' in hot_src
        # единственный полный проход в писателе — явный adoption и read-only счётчик
        assert "self._tail_events(f.read())" in inspect.getsource(ProvenanceLog.recover_head)
        assert "_read_all" in inspect.getsource(ProvenanceLog.count.fget)

    # --- witness как единый артефакт с журналом -------------------------------

    def test_modified_head_hash_refuses_append_even_with_a_valid_digest(self, log):
        """(а) подправленный last_hash → append отказывает, а не продолжает воздух.

        Witness с пересчитанным digest выглядит самодостаточным, поэтому его поля
        обязаны сверяться с фактическим хвостом журнала перед каждой записью.
        """
        log.append({"event": "a"})
        log.append({"event": "b"})
        _forge_head(log.head_path, last_hash="0" * 64)
        before = open(log.path, "rb").read()
        with pytest.raises(ProvenanceIntegrityError, match="не сходится с фактическим хвостом"):
            ProvenanceLog(path=log.path).append({"event": "c"})
        assert open(log.path, "rb").read() == before      # отказ ничего не дописал
        assert ProvenanceLog(path=log.path).verify_chain() is False

    def test_head_pointing_past_eof_refuses_append(self, log):
        """(б) offset за концом файла → append отказывает."""
        log.append({"event": "a"})
        log.append({"event": "b"})
        _forge_head(log.head_path, log_byte_offset=os.path.getsize(log.path) + 4096)
        with pytest.raises(ProvenanceIntegrityError, match="за конец файла"):
            ProvenanceLog(path=log.path).append({"event": "c"})
        assert ProvenanceLog(path=log.path).verify_chain() is False

    def test_head_with_true_hash_but_wrong_offset_refuses_append(self, log):
        """(в) хэш настоящий, смещение нет: на указанном месте лежит другая запись."""
        log.append({"event": "a"})
        log.append({"event": "b"})
        data = open(log.path, "rb").read()
        boundary_of_first = data.index(b"\n") + 1       # конец первой записи
        _forge_head(log.head_path, log_byte_offset=boundary_of_first)
        with pytest.raises(ProvenanceIntegrityError, match="не сходится с фактическим хвостом"):
            ProvenanceLog(path=log.path).append({"event": "c"})

    def test_missing_head_never_appends_silently(self, log):
        """(г) witness удалён: append отказывает, adoption обязан быть явным.

        Раньше на этом месте писатель молча перечитывал весь журнал — ровно тот
        путь, который стоил P4.2.v1 его abort, и который к тому же позволял
        подсунуть цепь, отличную от лежащей на диске.
        """
        log.append({"event": "a"})
        log.append({"event": "b"})
        os.remove(log.head_path)
        fresh = ProvenanceLog(path=log.path)
        assert fresh.count == 2                         # посмотреть можно
        before = open(log.path, "rb").read()
        with pytest.raises(ProvenanceIntegrityError, match="recover_head"):
            fresh.append({"event": "c"})                 # дописать молча — нельзя
        assert open(log.path, "rb").read() == before

    def test_recover_head_adopts_the_log_and_marks_the_adoption_in_the_chain(self, log):
        """Явный adoption: один полный проход, witness заводится, событие-маркер в цепи."""
        log.append({"event": "a"})
        log.append({"event": "b"})
        os.remove(log.head_path)
        fresh = ProvenanceLog(path=log.path)
        head = fresh.recover_head(reason="потерян при переносе каталога")
        events = fresh.get_events()
        marker = events[-1].payload["provenance_head_recovered"]
        assert marker["adopted_entries"] == 2 and "переносе" in marker["reason"]
        # вернувшийся witness уже включает маркер: два принятых события плюс он сам
        assert head["entry_count"] == 3 and head["last_hash"] == events[-1].entry_hash
        assert fresh.count == 3                          # маркер — часть журнала
        assert fresh.verify_chain() is True
        e4 = fresh.append({"event": "c"})                # дальше цепь идёт как обычно
        assert e4.previous_hash == events[-1].entry_hash
        assert fresh.verify_chain() is True

    def test_recover_head_refuses_a_log_whose_chain_does_not_close(self, log):
        """Adoption не обязан принимать мусор: разорванную цепь чинят не этим путём."""
        log.append({"event": "a"})
        log.append({"event": "b"})
        os.remove(log.head_path)
        lines = open(log.path, encoding="utf-8").read().splitlines(True)
        forged = json.loads(lines[1])
        forged["previous_hash"] = "f" * 64
        lines[1] = json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n"
        with open(log.path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        with pytest.raises(ProvenanceIntegrityError, match="разорвана"):
            ProvenanceLog(path=log.path).recover_head()
        assert not os.path.exists(log.head_path)          # refuse = не заводить witness

    def test_lie_in_entry_count_alone_survives_append_but_not_verification(self, log):
        """Честная граница: счётчик witness'а на цепь не влияет, и потому ловится аудитом.

        Смещение и хэш сверяются с журналом на каждом append, entry_count — нет: он
        нужен для отчёта, а не для продолжения цепи. Поэтому враньё в нём сходит
        писателю, но не проходит verify_chain().
        """
        log.append({"event": "a"})
        log.append({"event": "b"})
        _forge_head(log.head_path, entry_count=99)
        ProvenanceLog(path=log.path).append({"event": "c"})   # цепь продолжается верно
        assert ProvenanceLog(path=log.path).verify_chain() is False

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
