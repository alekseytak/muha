"""Проверки ZaLoop — ровно те двенадцать, что перечислены в спецификации.

Последняя из них важнее остальных: контроллер без детектора цикла обязан
провалить проверку `A B A B`. Контроллер, который проходит всё и при этом не
замечает цикл периода 2, бесполезен ровно в том случае, ради которого
написан.
"""

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.autonomous import StepRecord              # noqa: E402
from core.zaloop import (                          # noqa: E402
    NORMAL, NUDGE, REDUCE, SHADOW, STOP,
    PENDING, Thresholds, ZaLoop, ZaLoopError,
    detect_cycle, object_cluster,
)

BODY = "\n".join([
    "zaloop:",
    "  model: glm-5.3-flash",
    "  calibrated_on: outputs/2026-11-21-topic/artifacts/glm-5.3-flash/pilot.jsonl",
    "  repeatRatio_reduce: 0.35",
    "  repeatRatio_shadow: 0.60",
    "  deniedRate_shadow: 0.50",
    "  max_steps: 20",
])


def write_thresholds(directory: str, with_hash: bool = True,
                     model: str = "glm-5.3-flash") -> str:
    body = BODY.replace("model: glm-5.3-flash", f"model: {model}")
    text = body
    if with_hash:
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        text = body + f"\nhash: sha256:{digest}"
    path = os.path.join(directory, "zaloop.yml")
    Path(path).write_text(text + "\n", encoding="utf-8")
    return path


def step(index: int, verb: str = "read", obj: str = "notes/a.txt",
         success: bool = True, domain: str = "filesystem") -> StepRecord:
    return StepRecord(index=index, verb=verb, domain=domain, object=obj,
                      success=success, result="ок" if success else "",
                      refused="" if success else "отказ песочницы")


class ZaLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.thresholds = Thresholds.load(write_thresholds(self.tmp))

    def loop(self) -> ZaLoop:
        return ZaLoop(self.thresholds, expected_model="glm-5.3-flash")

    # 1
    def test_first_step_is_normal(self):
        decision = self.loop().supervise([step(1)], budget_left=10)
        self.assertEqual(decision.stage, NORMAL, decision.reason)
        self.assertTrue(decision.allow_execution)

    # 2
    def test_one_repeat_is_nudge_and_execution_allowed(self):
        decision = self.loop().supervise(
            [step(1), step(2)], budget_left=10)
        self.assertEqual(decision.stage, NUDGE, decision.reason)
        self.assertTrue(decision.allow_execution)
        self.assertIn("уже выполнено", decision.nudge)

    # 3
    def test_two_repeats_is_reduce(self):
        decision = self.loop().supervise(
            [step(1), step(2), step(3)], budget_left=10)
        self.assertEqual(decision.stage, REDUCE, decision.reason)
        self.assertEqual(decision.max_steps, 10)
        self.assertEqual(decision.history_window, 3)

    # 4
    def test_three_repeats_is_shadow_and_execution_forbidden(self):
        decision = self.loop().supervise(
            [step(1), step(2), step(3), step(4)], budget_left=10)
        self.assertEqual(decision.stage, SHADOW, decision.reason)
        self.assertFalse(decision.allow_execution)

    # 5
    def test_cycle_a_b_a_b_is_shadow_with_period_two(self):
        steps = [step(1, obj="notes/a.txt"), step(2, obj="notes/b.txt"),
                 step(3, obj="notes/a.txt"), step(4, obj="notes/b.txt")]
        decision = self.loop().supervise(steps, budget_left=10)
        self.assertEqual(decision.stage, SHADOW, decision.reason)
        self.assertTrue(decision.signals.cyclic)
        self.assertEqual(decision.signals.period, 2)

    # 6
    def test_three_different_signatures_stay_normal(self):
        steps = [step(1, obj="notes/a.txt"), step(2, obj="notes/b.txt"),
                 step(3, obj="notes/c.txt")]
        decision = self.loop().supervise(steps, budget_left=10)
        self.assertEqual(decision.stage, NORMAL, decision.reason)

    # 7
    def test_two_failures_in_a_row_stop_with_ticket(self):
        steps = [step(1, obj="notes/a.txt", success=False),
                 step(2, obj="notes/b.txt", success=False)]
        decision = self.loop().supervise(steps, budget_left=10, task_id="t-1")
        self.assertEqual(decision.stage, STOP, decision.reason)
        self.assertIsNotNone(decision.ticket)
        self.assertEqual(decision.ticket.resume_from, PENDING)
        self.assertEqual(decision.ticket.task_id, "t-1")
        self.assertIn("две ошибки", decision.ticket.stop_reason)
        # Причина — наблюдение, а не оценка.
        self.assertNotIn("эргодич", decision.ticket.stop_reason)

    # 8
    def test_budget_exhausted_stop_with_ticket(self):
        decision = self.loop().supervise([step(1)], budget_left=0,
                                         task_id="t-2")
        self.assertEqual(decision.stage, STOP)
        self.assertIn("бюджет", decision.ticket.stop_reason)
        self.assertEqual(decision.ticket.resume_from, PENDING)

    # 9
    def test_frame_change_after_shadow_returns_to_normal(self):
        loop = self.loop()
        loop.supervise([step(1), step(2), step(3), step(4)], budget_left=10)
        self.assertEqual(loop.stage, SHADOW)
        # Глагол сменился, объект в том же кластере — новая рамка.
        decision = loop.supervise(
            [step(1), step(2), step(3), step(4),
             step(5, verb="write", obj="notes/b.txt")], budget_left=9)
        self.assertEqual(decision.stage, NORMAL, decision.reason)
        self.assertIn("рамка", decision.reason)

    # 10
    def test_thresholds_without_hash_are_refused(self):
        path = write_thresholds(self.tmp, with_hash=False)
        with self.assertRaises(ZaLoopError):
            Thresholds.load(path)

    # 11
    def test_foreign_model_raises_warning_with_its_name(self):
        path = write_thresholds(self.tmp, model="glm-5.3-flash")
        loop = ZaLoop(Thresholds.load(path), expected_model="qwen3.8-flash")
        self.assertTrue(loop.warnings), loop.warnings
        self.assertIn("glm-5.3-flash", loop.warnings[0])

    # 12 — мутант
    def test_mutant_without_cycle_detector_fails_a_b_a_b(self):
        class NoCycleZaLoop(ZaLoop):
            """Мутант: детектор цикла выключен."""

            def measure(self, steps, budget_left=0):
                signals = super().measure(steps, budget_left)
                signals.cyclic = False
                signals.period = 0
                return signals

        steps = [step(1, obj="notes/a.txt"), step(2, obj="notes/b.txt"),
                 step(3, obj="notes/a.txt"), step(4, obj="notes/b.txt")]
        decision = NoCycleZaLoop(self.thresholds).supervise(
            steps, budget_left=10)
        self.assertNotEqual(decision.stage, SHADOW,
                            "мутант без детектора цикла не должен проходить A B A B")


class RunnerWiringTests(unittest.TestCase):
    """Контроллер подключён к прогону, но без порогов не мешает."""

    def test_without_thresholds_runner_has_no_controller(self):
        from unittest.mock import MagicMock
        from core.autonomous import AutonomousRunner

        runner = AutonomousRunner(MagicMock(), MagicMock())
        self.assertIsNone(runner._make_zaloop(),
                          "без порогов контроллер не подключается")

    def test_load_zaloop_accepts_good_file_and_rejects_bad(self):
        from unittest.mock import MagicMock
        from core.autonomous import AutonomousRunner

        tmp = tempfile.mkdtemp()
        runner = AutonomousRunner(MagicMock(), MagicMock())
        runner.load_zaloop(write_thresholds(tmp), model="glm-5.3-flash")
        self.assertIsNotNone(runner._make_zaloop())
        bad = write_thresholds(tmp, with_hash=False)
        with self.assertRaises(ZaLoopError):
            runner.load_zaloop(bad)


class CycleHelpersTests(unittest.TestCase):
    def test_detect_cycle_periods(self):
        self.assertEqual(detect_cycle(["A"] * 5), 1)
        self.assertEqual(detect_cycle(["A", "B", "A", "B"]), 2)
        self.assertEqual(detect_cycle(["A", "B", "C", "A", "B", "C"]), 3)
        self.assertEqual(detect_cycle(["A", "B", "C", "D"]), 0)
        self.assertEqual(detect_cycle(["A", "B"]), 0)

    def test_object_cluster_groups_same_folder_and_extension(self):
        self.assertEqual(object_cluster("notes/a.txt"),
                         object_cluster("notes/b.txt"))
        self.assertNotEqual(object_cluster("notes/a.txt"),
                            object_cluster("notes/a.md"))
        self.assertEqual(object_cluster("sys:uname"), "sys:")


if __name__ == "__main__":
    unittest.main(verbosity=2)