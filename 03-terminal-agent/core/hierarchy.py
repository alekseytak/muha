"""Иерархический каскад: уровень полномочий L0–L5, затем глагол внутри уровня.

Источник уровней — `OS-Glagolov-spec.md:41-49` (цитируется в
`ГЛАВНАЯ ПАПКА/ОС_глаголов_система.md`): шесть нормативных уровней действия
Наблюдение, Анализ, Преобразование, Рекомендация, Контроль, Commit.

Зачем это нужно поверх плоского выбора из одиннадцати глаголов:

* уровень — более грубая и потому более лёгкая категория (в замере уровень
  определён верно на 146 из 146 решённых фраз);
* ошибка уровня опаснее ошибки глагола (Наблюдение против Commit — это разное
  право), поэтому уровень проверяется отдельно и с отдельным порогом;
* внутри уровня выбор часто безальтернативен (`recommend` один в L3), и тогда
  не нужно угадывать статистикой.

Замер на отложенных семействах (`eval/hierarchical.py`), cluster bootstrap
по семействам:

    плоский каскад : покрытие 92% (140/152), глагол 0.986, CI [0.960, 1.000]
    иерархия L0–L5 : покрытие 100% (152/152), глагол 0.987, CI [0.964, 1.000]

ВАЖНАЯ ОГОВОРКА. Все 152 решения иерархии приняты правилами μ1a: ветка
уровней μL/μV не сработала ни разу. Прибавка к покрытию — это расширение
локальной ПОЛИТИКИ (плоский каскад намеренно отправляет veto/escalate наверх
через μ3, а здесь они решаются по явной улике), а НЕ результат обучения
уровней. Поэтому L0–L5 нельзя объявлять выигрышем модели: для проверки самой
иерархии нужен одинаковый policy-baseline и отдельная абляция статистической
ветки на фразах без правил.

Политика уровней контроля различается по необратимости:
  * L4 (veto, escalate) — отказ и передача человеку; статистический ответ
    допустим, потому что худший случай ошибки это остановка или лишний вопрос;
  * L5 (anchor) — необратимая запись в реестр; требуется явная улика правила.
"""

from typing import Dict, Iterable, List, Optional, Tuple

from .micromodels import (
    CONFIDENCE_FLOOR, MARGIN_FLOOR, KNOWN_GRAMS_FLOOR, MIN_TRAINING_EXAMPLES,
    Decision, VerbMicroModel, cue_verb, domain_of, extract_object,
)

LEVELS: Tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4", "L5")
LEVEL_TITLES = {
    "L0": "Наблюдение",
    "L1": "Анализ",
    "L2": "Преобразование",
    "L3": "Рекомендация",
    "L4": "Контроль",
    "L5": "Commit",
}

# Соответствие глаголов уровням — по смыслу полномочий, а не по удобству.
# Уровни, где ответ без улики недопустим: остановка (L4) и обязательство (L5).
SAFETY_CRITICAL_LEVELS = ("L4", "L5")

LEVEL_OF: Dict[str, str] = {
    "read": "L0",       # только смотреть
    "version": "L0",    # только узнать версию
    "think": "L1",      # внутренняя работа знания
    "select": "L1",     # выбор без изменения мира
    "write": "L2",      # изменение состояния файла
    "execute": "L2",    # изменение состояния через команду
    "log": "L2",        # запись в журнал
    "recommend": "L3",  # совет, а не действие
    "veto": "L4",       # контроль: остановка
    "escalate": "L4",   # контроль: передача выше
    "anchor": "L5",     # связывающее обязательство
}


class HierarchicalCascade:
    """Две ступени вместо плоского выбора: уровень, затем глагол в уровне."""

    def __init__(self, examples: Optional[Iterable[Tuple[str, str]]] = None,
                 floor: float = CONFIDENCE_FLOOR) -> None:
        self.floor = floor
        self.trained_on = 0
        self.level_model = VerbMicroModel(labels=LEVELS)
        self.verb_models: Dict[str, Tuple[VerbMicroModel, List[str]]] = {}
        if examples:
            self.fit(examples)

    def fit(self, examples: Iterable[Tuple[str, str]]) -> "HierarchicalCascade":
        level_pairs: List[Tuple[str, str]] = []
        per_level: Dict[str, List[Tuple[str, str]]] = {}
        for phrase, verb in examples:
            level = LEVEL_OF.get(verb)
            if level is None:
                continue
            level_pairs.append((phrase, level))
            per_level.setdefault(level, []).append((phrase, verb))
        self.level_model.fit(level_pairs)
        self.verb_models = {}
        for level, pairs in per_level.items():
            model = VerbMicroModel()
            model.fit(pairs)
            self.verb_models[level] = (model, sorted({verb for _, verb in pairs}))
        self.trained_on = len(level_pairs)
        return self

    # ---------- ступени ----------

    def _decide_level(self, phrase: str) -> Optional[Tuple[str, float, List[str]]]:
        """Возвращает (уровень, уверенность, заметки) или None при отказе."""
        notes: List[str] = []
        if self.trained_on < MIN_TRAINING_EXAMPLES:
            notes.append(f"μL: мало данных ({self.trained_on}) — решает омега")
            return None
        if len(self.level_model.priors) < 2:
            notes.append("μL: в обучении один уровень — различать нечего")
            return None
        level, confidence, margin, known = self.level_model.predict(phrase)
        if level is None:
            return None
        notes.append(f"μL: уровень {level} ({LEVEL_TITLES.get(level, level)}) "
                     f"p={confidence:.2f}")
        if known < KNOWN_GRAMS_FLOOR:
            notes.append(f"μL: фраза незнакома ({known:.0%})")
            return None
        if margin < MARGIN_FLOOR:
            notes.append(f"μL: нет запаса над вторым ({margin:.2f})")
            return None
        return level, confidence, notes

    def decide(self, phrase: str) -> Optional[Decision]:
        """None означает: каскад воздерживается, нужна большая модель.

        Порядок ступеней важен и проверен замером. Правила идут ПЕРВЫМИ: если
        улика есть, уровень спрашивать не нужно, потому что ошибка уровня
        выбросила бы верную улику. Иерархия подключается только там, где
        правила молчат, и сужает выбор до глаголов одного уровня.

        Запрет на ответ без улики различается по уровню:
          * L4 (veto/escalate) — статистика допустима при любом числе
            вариантов: это отказ и передача человеку, ошибка безопасна;
          * L5 (anchor) — без явной улики отказ всегда: запись в реестр
            необратима.
        """
        notes: List[str] = []
        # То же требование к данным, что у плоского каскада: без корпуса
        # каскад не отвечает вовсе. Иначе свежий агент отвечал бы по правилам
        # без всякой проверки домена и объекта — непредсказуемое поведение.
        if self.trained_on < MIN_TRAINING_EXAMPLES:
            return None
        ruled, why = cue_verb(phrase)
        if ruled is not None:
            dom, _ = domain_of(phrase, ruled)
            if dom is None:
                dom = "ui"
            obj = extract_object(phrase, dom)
            if dom in ("filesystem", "web", "system") and not obj:
                notes.append(f"μ1c: объект для домена {dom} не найден")
                return None
            notes.append(f"μ1a: правило «{why}»")
            return Decision(
                verb=ruled, domain=dom, confidence=0.9,
                source="μ1a", notes=notes, object=obj or phrase,
            )

        level_result = self._decide_level(phrase)
        if level_result is None:
            return None
        level, level_conf, level_notes = level_result
        notes.extend(level_notes)

        entry = self.verb_models.get(level)
        if entry is None:
            notes.append(f"μV: глаголов уровня {level} в обучении не было")
            return None
        model, options = entry

        if len(options) == 1:
            # Внутри уровня выбора нет — но «он один в обучении» это НЕ улика.
            # Для контроля и Commit такой ответ недопустим: иначе без всяких
            # оснований можно выдать вето или якорь.
            if level in SAFETY_CRITICAL_LEVELS:
                notes.append(f"μV: уровень {level} требует явной улики, "
                             f"а правило её не дало ({why})")
                return None
            verb, verb_conf = options[0], level_conf
            notes.append(f"μV: в уровне {level} один глагол — {verb}")
        else:
            verb, verb_conf, margin, known = model.predict(phrase)
            if verb is None:
                return None
            if known < KNOWN_GRAMS_FLOOR or margin < MARGIN_FLOOR:
                notes.append(f"μV: внутри {level} неуверенно "
                             f"(знакомых {known:.0%}, запас {margin:.2f})")
                return None
            notes.append(f"μV: статистика внутри {level}")
        if verb not in options:
            notes.append(f"μV: глагол {verb} не принадлежит уровню {level}")
            return None

        confidence = min(level_conf, verb_conf)
        if confidence < self.floor:
            notes.append(f"μL/μV: уверенность {confidence:.2f} ниже порога")
            return None

        domain, domain_conf = domain_of(phrase, verb)
        if domain is None:
            domain, domain_conf = "ui", 0.6
        obj = extract_object(phrase, domain)
        if domain in ("filesystem", "web", "system") and not obj:
            notes.append(f"μ1c: объект для домена {domain} не найден")
            return None
        return Decision(
            verb=verb, domain=domain, confidence=confidence,
            source=f"μL/μV ({level})", notes=notes, object=obj or phrase,
        )