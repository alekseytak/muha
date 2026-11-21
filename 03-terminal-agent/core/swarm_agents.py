"""Рой мух: несколько независимых разборщиков вместо одного.

Зачем это существует. До сих пор «рой» в проекте был мёртвым кодом:
`Swarm.register_agent` и `sense_swarm` не вызывались нигде, а решения
принимал единственный `Agent`. Здесь рой становится настоящим: несколько
мух с РАЗНЫМИ профилями независимо разбирают одну фразу, голосуют и либо
сходятся, либо честно передают решение омеге.

Три вещи, которые делают это роем, а не украшением:

1. Мухи различаются поведением, а не именем. У каждой свой порог
   уверенности, свой риск-аппетит и свои предпочтения. Осторожная молчит
   там, где быстрая говорит; хранитель не любит побочных действий.
2. Голосование влияет на результат. Побеждает глагол с большинством;
   если согласия нет — решение уходит омеге, а не выдумывается.
3. Разногласие видно. В решении остаются все голоса с причинами, поэтому
   «рой сказал» можно проверить, а не принять на веру.

Цена. Мухи пользуются локальным каскадом микромоделей, то есть голосование
бесплатно. Модель зовётся только при расколе — это и есть экономия.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .swarm import PheromoneField

# Глаголы с побочным эффектом: осторожные мухи их не предлагают без улики.
SIDE_EFFECT = ("write", "execute", "veto", "anchor", "log")


@dataclass(frozen=True)
class FlyProfile:
    """Характер одной мухи.

    Профиль — это не косметика: порог и риск-аппетит меняют, скажет ли муха
    слово. Две мухи с одинаковым профилем были бы одной мухой, поэтому
    профили подобраны так, чтобы расходиться на разных типах фраз.
    """

    name: str
    confidence_floor: float
    risk_appetite: float
    prefers: Tuple[str, ...] = ()
    note: str = ""
    # Движок, которым муха смотрит на фразу. Это и есть настоящее различие:
    # муха-правила и муха-статистика видят одну фразу по-разному, а не
    # просто по-разному её оценивают.
    engine: str = "полный"
    # Представление фразы: муха может смотреть на сырой текст или на текст
    # с замаскированными именами файлов. Это второе измерение различия:
    # движок отвечает «чем смотреть», представление — «на что смотреть».
    representation: str = "сырая"


FLIES: Tuple[FlyProfile, ...] = (
    FlyProfile(
        "правила", 0.50, 0.55, engine="правила",
        note="верит только правилам-уликам, статистику не слушает",
    ),
    FlyProfile(
        "статистика", 0.45, 0.55, engine="статистика",
        note="верит только n-граммам, правила игнорирует",
    ),
    FlyProfile(
        "полный", 0.50, 0.70, ("select", "recommend", "read"),
        engine="полный",
        note="полный каскад: правила, затем статистика",
    ),
    FlyProfile(
        "хранитель", 0.60, 0.10, ("veto", "anchor", "log"),
        engine="полный",
        note="сторож: побочные действия только при явной улике",
    ),
    FlyProfile(
        "структурная", 0.50, 0.55, engine="полный",
        representation="маска",
        note="смотрит на фразу без имён файлов: только структура",
    ),
)


@dataclass
class Vote:
    """Голос одной мухи."""

    fly: str
    verb: Optional[str]
    confidence: float
    reason: str

    @property
    def spoke(self) -> bool:
        return self.verb is not None


@dataclass
class SwarmDecision:
    """Итог голосования: что решил рой и насколько он сошёлся."""

    verb: Optional[str]
    domain: Optional[str]
    object: Optional[str]
    votes: List[Vote] = field(default_factory=list)
    agreement: float = 0.0
    needs_omega: bool = True
    reason: str = ""

    @property
    def spoken(self) -> int:
        return sum(1 for vote in self.votes if vote.spoke)

    def summary(self) -> str:
        """Короткая строка для журнала и демо."""
        spoke = self.spoken
        if self.verb is None:
            return f"рой: молчит ({self.reason})"
        return (f"рой: {self.verb} при согласии "
                f"{self.agreement:.0%} ({spoke} из {len(self.votes)} мух)")


class FlySwarm:
    """Рой, который голосует по одной фразе.

    Поле феромонов используется по назначению: победивший глагол оставляет
    след, поэтому следующая фраза того же типа видит, что рой здесь уже
    решал. Это память, а не принятие решения — решение принимают голоса.
    """

    def __init__(self, cascades, flies: Tuple[FlyProfile, ...] = FLIES,
                 quorum: float = 0.5, decay_rate: float = 0.95) -> None:
        """`cascades` — словарь движков: имя → каскад.

        Принимается и один каскад: тогда все мухи смотрят им, и рой выродится
        в одну муху. Это разрешено для тестов, но в замерах так не делается.
        """
        if not isinstance(cascades, dict):
            cascades = {"полный": cascades}
        self.cascades = cascades
        self.cascade = cascades.get("полный") or next(iter(cascades.values()))
        self.flies = flies
        self.quorum = quorum
        self.field = PheromoneField(decay_rate=decay_rate)
        self.history: List[SwarmDecision] = []

    # --- настроение -----------------------------------------------------
    def fear_level(self) -> float:
        """Насколько рой напуган, по последним решениям.

        Источник — не украшение: страх поднимает осторожность. Если рой
        недавно расходился, осторожные мухи становятся строже.
        """
        recent = self.history[-10:]
        if not recent:
            return 0.0
        splits = sum(1 for decision in recent if decision.needs_omega)
        return min(1.0, splits / len(recent))

    def risk_appetite(self, profile: FlyProfile) -> float:
        """Риск-аппетит мухи с поправкой на страх роя.

        Как в stage8: страх снижает готовность рисковать. Здесь это −0.4
        при полном страхе, ровно как в отчёте по гормонам.
        """
        return max(0.0, profile.risk_appetite - 0.4 * self.fear_level())

    # --- голосование ----------------------------------------------------
    def vote(self, phrase: str) -> SwarmDecision:
        """Собрать голоса и решить."""
        votes: List[Vote] = []

        for profile in self.flies:
            engine = self.cascades.get(profile.engine) or self.cascade
            seen = represent(phrase, profile.representation)
            decision = engine.decide(seen)
            if decision is None:
                votes.append(Vote(
                    profile.name, None, 0.0,
                    f"движок «{profile.engine}» на представлении "
                    f"«{profile.representation}» не дал глагола"))
                continue
            verb = decision.verb
            confidence = float(decision.confidence)
            # Предпочтение — небольшой сдвиг, а не подмена решения:
            # 0.03 не переворачивает уверенный голос каскада.
            if verb in profile.prefers:
                confidence += 0.03
            if confidence < profile.confidence_floor:
                votes.append(Vote(
                    profile.name, None, round(confidence, 3),
                    f"уверенность {confidence:.2f} ниже порога "
                    f"{profile.confidence_floor:.2f}"))
                continue
            if verb in SIDE_EFFECT and self.risk_appetite(profile) < 0.2:
                votes.append(Vote(
                    profile.name, None, round(confidence, 3),
                    f"{verb}: побочное действие без готовности рисковать"))
                continue
            votes.append(Vote(profile.name, verb, round(confidence, 3),
                              f"{verb} ← {decision.source}"))

        counts: Dict[str, int] = {}
        by_engine: Dict[str, set] = {}
        for item, profile in zip(votes, self.flies):
            if item.verb:
                counts[item.verb] = counts.get(item.verb, 0) + 1
                # Ключ — пара «движок/представление»: две мухи, смотрящие
                # одним движком на разные представления, могут разойтись, и
                # это расхождение тоже должно блокировать угадывание.
                key = f"{profile.engine}/{profile.representation}"
                by_engine.setdefault(key, set()).add(item.verb)

        # Условие ответа — согласие ДВИЖКОВ, а не большинство мух.
        # Причина измерена: там, где движки расходятся, одиночный путь
        # ошибается заметно чаще (точность 0.963 против 1.000 у согласных).
        # Значит расхождение движков — признак неоднозначности, и угадывать
        # по большинству здесь нечего: решение уходит омеге.
        engine_verbs = set()
        for verbs in by_engine.values():
            engine_verbs |= verbs
        if len(engine_verbs) > 1:
            detail = "; ".join(f"{name}: {', '.join(sorted(verbs))}"
                               for name, verbs in sorted(by_engine.items()))
            result = SwarmDecision(
                None, None, None, votes, 0.0, True,
                f"движки разошлись ({detail})")
            self._remember(phrase, result)
            return result

        if not counts:
            result = SwarmDecision(
                None, None, None, votes, 0.0, True,
                "ни одна муха не взялась")
            self._remember(phrase, result)
            return result

        spoken = sum(counts.values())
        best_verb, best_count = max(counts.items(),
                                    key=lambda pair: (pair[1], -_order(pair[0])))
        agreement = best_count / spoken
        tied = sum(1 for verb, count in counts.items() if count == best_count) > 1

        if tied or agreement < self.quorum:
            result = SwarmDecision(
                None, None, None, votes, agreement, True,
                f"раскол: {', '.join(f'{v}×{c}' for v, c in counts.items())}")
            self._remember(phrase, result)
            return result

        winner = self.cascades.get("полный") or self.cascade
        context = winner.decide(phrase)
        result = SwarmDecision(
            best_verb,
            context.domain if context else None,
            context.object if context else None,
            votes, agreement, False, f"согласие {agreement:.0%}",
        )
        self._remember(phrase, result)
        return result

    def _remember(self, phrase: str, decision: SwarmDecision) -> None:
        """Оставить след и запомнить решение."""
        self.history.append(decision)
        if decision.verb:
            self.field.deposit(
                position=(0.0, 0.0),
                strength=decision.agreement,
                agent_id="рой",
                pheromone_type=f"verb:{decision.verb}",
            )
        self.field.evaporate()

    # --- наблюдение -----------------------------------------------------
    def stats(self) -> Dict[str, object]:
        total = len(self.history)
        if not total:
            return {"решений": 0, "мух": len(self.flies)}
        split = sum(1 for decision in self.history if decision.needs_omega)
        return {
            "решений": total,
            "мух": len(self.flies),
            "расколов": split,
            "доля расколов": round(split / total, 3),
            "страх": round(self.fear_level(), 3),
            "феромон": self.field.stats(),
        }


def represent(phrase: str, kind: str) -> str:
    """Представление фразы, которое видит муха.

    «маска» убирает имена файлов и адреса: муха смотрит на структуру запроса
    и не цепляется за слово в имени файла. «сырая» — как сказано.
    """
    if kind == "маска":
        try:
            from .readiness import mask_objects
            return mask_objects(phrase)
        except Exception:                       # noqa: BLE001
            return phrase
    return phrase


def _order(verb: str) -> int:
    """Устойчивый порядок при равном числе голосов.

    Без него `max` выбирал бы произвольно, и один и тот же прогон мог давать
    разные ответы. Порядок — алфавитный, он не привилегирует ни один глагол.
    """
    return sum(ord(ch) for ch in verb)