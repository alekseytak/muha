"""Tests for ProvenanceLog."""
from __future__ import annotations
import os, tempfile, json, pytest
from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog, _canonical_json, _sha256


@pytest.fixture
def log():
    tmp = tempfile.mktemp(suffix=".jsonl")
    p = ProvenanceLog(path=tmp)
    yield p
    if os.path.exists(tmp): os.remove(tmp)


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

    def test_truncation_needs_a_witness_not_a_fresh_reader(self, log):
        """Honest limit: a fresh reader has no anchor, so a shorter valid chain
        looks fine to it. Detection requires a writer that saw the longer log
        (previous test) or external anchoring, which P3.1 does not provide."""
        log.append({"event": "a"})
        log.append({"event": "b"})
        path = log.path
        with open(path, "r") as f:
            lines = f.readlines()
        with open(path, "w") as f:
            f.write(lines[0])
        assert ProvenanceLog(path=path).verify_chain() is True

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
