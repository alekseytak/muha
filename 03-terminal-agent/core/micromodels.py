"""Каскад микромоделей действий: муха → слой → рой → омега.

Идея: глагол — это не генерация, а **выбор из одиннадцати**. Дорогая модель
не нужна там, где дешёвая уверенно попадает; и наоборот, дешёвая обязана
уметь сказать «не знаю», иначе экономия превращается в ошибки.

Ступени каскада:

  μ1 «муха»   — микро-модель глагола: лексические признаки + наивный Байес
                по символьным n-граммам, обученный на размеченных кейсах.
  μ2 «слой»   — домен по префиксу объекта и по глаголу.
  μ3 «рой»    — нужна ли эскалация в рой (ESCALATE/swarm) вместо локального
                действия.
  Ω «омега»   — если хоть одна ступень воздержалась или ступени спорят,
                решение уходит большой модели. Это не отказ, а маршрутизация.

Главная метрика здесь — не точность, а **селективная точность**: при каком
покрытии каскад ошибается не чаще заданного. Экономия считается как доля
фраз, закрытых без вызова модели.

Модуль чистый: только стандартная библиотека, никаких зависимостей и сети.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

VERBS = (
    "read", "write", "execute", "think", "log", "version",
    "anchor", "veto", "escalate", "recommend", "select",
)
# Глаголы, которые сами по себе означают «наверх» — их решает μ3, не μ1.
ESCALATING = ("escalate", "veto")
NGRAM = 3
# Ниже этого порога микромодель не имеет права отвечать.
CONFIDENCE_FLOOR = 0.55
# Запас над вторым кандидатом: вероятность 0.6 при втором 0.59 — это не знание.
MARGIN_FLOOR = 0.15
# Меньше этого числа размеченных примеров статистике верить нельзя: на одном
# примере любая фраза «уверенно» попадает в единственный известный класс.
MIN_TRAINING_EXAMPLES = 8
# Доля знакомых n-грамм. Если фраза почти целиком состоит из невиданных кусков,
# статистика обязана молчать: иначе разница знаменателей классов выдаёт
# «уверенность» на бессмыслице.
KNOWN_GRAMS_FLOOR = 0.3


def normalize(text: str) -> str:
    """Приводит фразу к виду, на котором учились признаки."""
    text = (text or "").lower().replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9 :._/\\-]+", " ", text)
    return " ".join(text.split())


def ngrams(text: str, size: int = NGRAM) -> List[str]:
    padded = f" {normalize(text)} "
    return [padded[i:i + size] for i in range(max(0, len(padded) - size + 1))]


# ---------------------------------------------------------------- μ1: глагол

class VerbMicroModel:
    """Наивный Байес по символьным n-граммам.

    Умышленно простой: у нас десятки кейсов, а не тысячи, поэтому сложная
    модель здесь была бы самообманом. Зато она объяснима и мгновенна.
    """

    def __init__(self, labels: Optional[Iterable[str]] = None) -> None:
        # labels задаёт допустимый набор меток. По умолчанию это глаголы, но
        # иерархическому каскаду нужны уровни L0–L5 — иначе обучение молча
        # отбрасывало бы все примеры и модель оставалась пустой.
        self.labels = None if labels is None else set(labels)
        self.counts: Dict[str, Dict[str, int]] = {}
        self.totals: Dict[str, int] = {}
        self.priors: Dict[str, int] = {}
        self.vocab: set = set()

    def fit(self, examples: Iterable[Tuple[str, str]]) -> "VerbMicroModel":
        # vocab тоже очищается: иначе после переобучения слово из старого
        # корпуса остаётся «знакомым» и занижает отказ по незнакомой фразе.
        self.counts, self.totals, self.priors = {}, {}, {}
        self.vocab = set()
        for phrase, verb in examples:
            allowed = self.labels if self.labels is not None else VERBS
            if verb not in allowed:
                continue
            self.priors[verb] = self.priors.get(verb, 0) + 1
            bucket = self.counts.setdefault(verb, {})
            for gram in ngrams(phrase):
                bucket[gram] = bucket.get(gram, 0) + 1
                self.totals[verb] = self.totals.get(verb, 0) + 1
                self.vocab.add(gram)
        return self

    def _log_probs(self, phrase: str) -> Dict[str, float]:
        total_docs = sum(self.priors.values()) or 1
        vocab_size = max(1, len(self.vocab))
        scores: Dict[str, float] = {}
        grams = ngrams(phrase) or [""]
        for verb, prior in self.priors.items():
            score = math.log(prior / total_docs)
            bucket = self.counts.get(verb, {})
            denom = self.totals.get(verb, 0) + vocab_size
            for gram in grams:
                score += math.log((bucket.get(gram, 0) + 1) / denom)
            scores[verb] = score
        return scores

    def known_ratio(self, phrase: str) -> float:
        """Какая доля n-грамм фразы встречалась при обучении."""
        grams = ngrams(phrase) or [""]
        if not grams:
            return 0.0
        return sum(1 for gram in grams if gram in self.vocab) / len(grams)

    def predict(self, phrase: str) -> Tuple[Optional[str], float, float, float]:
        """Возвращает (глагол, уверенность, запас над вторым, доля знакомых).

        Запас нужен отдельно от вероятности: на маленьких данных мягкий
        максимум бывает высоким при почти равных кандидатах, и тогда
        «уверенность» врёт.
        """
        if not self.priors:
            return None, 0.0, 0.0, 0.0
        scores = self._log_probs(phrase)
        best = max(scores.values())
        shifted = {verb: math.exp(value - best) for verb, value in scores.items()}
        norm = sum(shifted.values()) or 1.0
        ranked = sorted(shifted.items(), key=lambda kv: kv[1], reverse=True)
        top_verb = ranked[0][0]
        top_prob = ranked[0][1] / norm
        second_prob = (ranked[1][1] / norm) if len(ranked) > 1 else 0.0
        return top_verb, top_prob, top_prob - second_prob, self.known_ratio(phrase)


# --------------------------------------------------------------- μ2: домен

DOMAIN_CUES: Sequence[Tuple[Tuple[str, ...], str]] = (
    (("файл", "файла", "файле", "каталог", "папк", "путь", "диск", ".txt", ".md", ".json",
      ".py", ".yaml", ".yml", ".csv", ".log", ".ini"), "filesystem"),
    (("коннектом", "нейрон", "нейрона", "класс", "мух", "flywire", "синапс"), "connectome"),
    (("реестр", "журнал", "запис", "цепочк", "хеш", "версию запис", "вето"), "registry"),
    (("рой", "роя", "феромон", "улей", "муравь", "стая"), "swarm"),
    (("команд", "терминал", "shell", "консол", "запусти", "выполни", "uname", "date",
      "pwd", "whoami", "git "), "system"),
    (("страниц", "сайт", "ссылк", "url", "http", "веб", "интернет", "браузер"), "web"),
)
PREFIX_DOMAINS: Sequence[Tuple[str, str]] = (
    ("file:", "filesystem"), ("dir:", "filesystem"), ("folder:", "filesystem"),
    ("path:", "filesystem"), ("fs:", "filesystem"),
    ("sys:", "system"), ("cmd:", "system"), ("sh:", "system"), ("exec:", "system"),
    ("reg:", "registry"), ("registry:", "registry"),
    ("swarm:", "swarm"), ("pheromone:", "swarm"),
    ("connectome:", "connectome"), ("neuron:", "connectome"), ("fly:", "connectome"),
    ("web:", "web"), ("url:", "web"), ("http://", "web"), ("https://", "web"),
)
# Пары, которые путает мешок n-грамм: «что написано в файле» против «запиши
# в файл», «покажи pwd» против «покажи файл». Порядок важен: первое совпадение
# выигрывает, поэтому частные правила идут раньше общих.
CUE_RULES: Sequence[Tuple[str, str, str]] = (
    # (регулярка, глагол, пояснение)
    # --- Уровень 1: действия, которые нельзя спутать с чтением ---
    # «наложи вето на запись версия-2» не должно уходить в version только
    # потому, что внутри встретилось слово «версия».
    (r"\bвето\b|запрети|заблокируй|поставь\s+запрет|останови\s+выполнение", "veto", "вето"),
    (r"эскалир|наверх|передай\s+(?:это\s+)?выше|сообщи\s+оператору|"
     r"позови\s+человек|позови\s+оператор", "escalate", "эскалация"),
    (r"канониз|заякор|\banchor\b|признай\s+\S+\s+эталоном|"
     r"сделай\s+\S+\s+канон|объяви\s+\S+\s+каноном|"
     r"закрепи\s+(?:\S+\s+)?(?:канон|как\s+эталон)|закрепи\s+\S+\s+в\s+реестре",
     "anchor", "якорь"),
    # --- Уровень 2: чтение против записи ---
    (r"что\s+написано|прочита|прочти|открой|загляни|покажи\s+(?:содержим|текст|"
     r"title|заголов|название)|"
     r"перескажи|выведи|посмотри|сколько\s+(?:строк|связей|записей)", "read", "чтение"),
    (r"запиши|сохрани|создай|сделай\s+\S+\.|положи|добавь\s+в\s+\S+|перезапиши|"
     r"допиши", "write", "запись"),
    # --- Уровень 3: остальное ---
    (r"запусти|выполни|команд|терминал|shell|прогони", "execute", "команда"),
    (r"размер|тип\s+файла|какого\s+типа|метаданн|сколько\s+весит|какой\s+формат",
     "select", "метаданные"),
    (r"выбери|вариант\s+[абa-z]|что\s+лучше|предпочти", "select", "выбор варианта"),
    (r"отметк|пометк|занеси\s+в|журнал|залогируй|лог\b", "log", "журнал"),
    (r"верси|какая\s+сейчас\s+версия", "version", "версия"),
    (r"посовет|что\s+рекоменд|рекомендац|дай\s+совет|что\s+делать", "recommend", "совет"),
    (r"зачем|почему|объясни|как\s+работает|как\s+устроен|что\s+такое", "think", "объяснение"),
)
_CUE_PATTERNS = tuple((re.compile(pattern, re.I), verb, why)
                      for pattern, verb, why in CUE_RULES)


def cue_verb(phrase: str) -> Tuple[Optional[str], str]:
    """μ1a: правила высокой точности. Возвращают глагол и сработавшее правило.

    Имена файлов и адреса маскируются до поиска: «прочитай файл anchor.md» —
    это чтение файла, а не якорь. Без маски слово внутри имени давало глагол.
    """
    from .readiness import mask_objects, CONTENT_RE
    # Слова после явного маркера содержимого не являются инструкциями.
    content = CONTENT_RE.search(phrase)
    instruction = phrase[:content.start(1)] if content else phrase
    text = normalize(mask_objects(instruction))
    for pattern, verb, why in _CUE_PATTERNS:
        if pattern.search(text):
            return verb, why
    return None, ""


# Глаголы, которые почти всегда живут в своём домене.
VERB_DOMAINS = {
    "execute": "system",
    "anchor": "registry",
    "version": "registry",
    "veto": "registry",
}


def _cue_matches(cue: str, text: str) -> bool:
    """Подстрокой — только для «хвостов» вроде .txt; слова — по границе слова.

    Иначе «открой» содержит «рой» и получает домен swarm, а «файла» не
    находится по стему «файл».
    """
    if cue[:1].isalnum():
        return re.search(r"\b" + re.escape(cue), text) is not None
    return cue in text


def domain_of(phrase: str, verb: Optional[str] = None) -> Tuple[Optional[str], float]:
    """μ2: домен по объекту-префиксу и лексике. Уверенность — грубая, но честная."""
    text = normalize(phrase)
    for prefix, domain in PREFIX_DOMAINS:
        if prefix in text:
            return domain, 0.95
    if verb in VERB_DOMAINS:
        return VERB_DOMAINS[verb], 0.7
    hits: Dict[str, int] = {}
    for cues, domain in DOMAIN_CUES:
        for cue in cues:
            if _cue_matches(cue, text):
                hits[domain] = hits.get(domain, 0) + 1
    if not hits:
        return None, 0.0
    best_domain, best_hits = max(hits.items(), key=lambda kv: kv[1])
    shared = sum(hits.values())
    return best_domain, min(0.9, best_hits / max(1, shared))


# --------------------------------------------------------- μ1c: объект

URL_RE = re.compile(r"(https?://[^\s,;]+|www\.[^\s,;]+)", re.I)
PATH_RE = re.compile(
    r"([\w./\-]+\.(?:txt|md|json|jsonl|ya?ml|csv|log|ini|py|js|ts|html|pdf|png|jpg))", re.I
)
# Известные команды — чтобы «командой pwd» давало pwd, а не «командой».
KNOWN_COMMANDS = (
    "uname", "ls", "pwd", "whoami", "date", "cat", "head", "tail", "wc", "echo",
    "git", "python3", "node", "bash", "sh", "df", "ps", "grep", "find", "sort",
)
COMMAND_RE = re.compile(
    r"\b(" + "|".join(KNOWN_COMMANDS) + r")\b", re.I
)
COMMAND_AFTER_VERB_RE = re.compile(
    r"(?:команд[аоуй]*|выполни|запусти|shell)\s+(?:это\s+)?[\"'\u00ab]?([\w.\-/]+)", re.I
)
# Домены, где объект обязан быть точным: без него действие бессмысленно,
# поэтому каскад воздерживается, а не подставляет всю фразу.
PRECISE_DOMAINS = ("filesystem", "web", "system")


def extract_object(phrase: str, domain: Optional[str]) -> Optional[str]:
    """μ1c: достаёт объект действия. None — значит объект не найден."""
    text = (phrase or "").strip()
    if domain == "web":
        match = URL_RE.search(text)
        if match:
            return match.group(1).rstrip(".,)")
        explicit = re.search(r"(?:web|url):(\S+)", text, re.I)
        return explicit.group(1) if explicit else None
    if domain == "filesystem":
        # Имя в кавычках берётся целиком: «план на завтра.txt» — это одно
        # имя, а не хвост «завтра.txt».
        from .readiness import QUOTED_PATH_RE
        quoted = QUOTED_PATH_RE.search(text)
        if quoted:
            return "file:" + quoted.group(1).strip()
        folder = re.search(r"(?:каталог|папк\w*|folder|dir)[:\s]+([\w./-]+)",
                           text, re.I)
        if folder:
            return "dir:" + folder.group(1)
        match = PATH_RE.search(text)
        if match:
            return "file:" + match.group(1)
        from .readiness import BARE_FILE_RE
        bare = BARE_FILE_RE.search(text)
        if bare:
            return "file:" + bare.group(1)
        explicit = re.search(r"(?:file|path|dir|folder):(\S+)", text, re.I)
        return explicit.group(0) if explicit else None
    if domain in ("registry", "swarm", "connectome"):
        # Имя объекта в этих доменах — не вся фраза: «заякори канон в реестре»
        # означает объект «канон», а не всю строку целиком.
        cleaned = re.sub(
            r"^(?:пожалуйста[,\s]+)?(?:заякор\w*|канониз\w*|закрепи|"
            r"наложи|запрети|объяви|признай|сделай|залогируй|позови|сообщи)\s+",
            "", text, flags=re.I)
        cleaned = re.sub(r"\s+(?:в|во)\s+(?:реестр\w*|каноне?)\s*$", "",
                         cleaned, flags=re.I).strip(" .,:")
        return cleaned or text
    if domain == "system":
        known = COMMAND_RE.search(text)
        if known:
            return "sys:" + known.group(1).lower()
        match = COMMAND_AFTER_VERB_RE.search(text)
        if match:
            return "sys:" + match.group(1)
        explicit = re.search(r"(?:sys|cmd|sh|exec):(\S+)", text, re.I)
        return explicit.group(0) if explicit else None
    return text


# ---------------------------------------------------------------- каскад

@dataclass
class Decision:
    """Результат каскада. `source` говорит, кто именно решил."""

    verb: str
    domain: Optional[str]
    confidence: float
    source: str
    notes: List[str] = field(default_factory=list)
    object: str = ""     # объект, извлечённый μ1c (для web/filesystem/system)

    def as_line(self) -> str:
        return f"{self.verb}: {self.object or self.domain or ''} | {{}}"


class MicroCascade:
    """μ1 → μ2 → μ3, с правом воздержаться в пользу омеги (большой модели)."""

    def __init__(self, examples: Optional[Iterable[Tuple[str, str]]] = None,
                 floor: float = CONFIDENCE_FLOOR, use_rules: bool = True,
                 use_stats: bool = True) -> None:
        self.verb_model = VerbMicroModel()
        self.floor = floor
        self.use_rules = use_rules
        self.use_stats = use_stats
        self.trained_on = 0
        if examples:
            self.fit(examples)

    def fit(self, examples: Iterable[Tuple[str, str]]) -> "MicroCascade":
        pairs = list(examples)
        self.verb_model.fit(pairs)
        self.trained_on = len(pairs)
        return self

    def decide(self, phrase: str) -> Optional[Decision]:
        """None означает: каскад воздерживается, нужна большая модель."""
        notes: List[str] = []
        ruled, why = cue_verb(phrase) if self.use_rules else (None, "")
        if ruled is not None:
            verb, verb_conf = ruled, 0.9
            notes.append(f"μ1a: правило «{why}»")
        else:
            if not self.use_stats:
                # Верим только правилам: статистике на малых данных нельзя.
                notes.append("μ1b: статистика отключена — решает омега")
                return None
            if self.trained_on < MIN_TRAINING_EXAMPLES:
                notes.append(
                    f"μ1b: мало данных обучения ({self.trained_on} < "
                    f"{MIN_TRAINING_EXAMPLES}) — решает омега"
                )
                return None
            if len(self.verb_model.priors) < 2:
                # Модель с одним классом не различает ничего: любая фраза
                # «похожа» на единственное, что она видела.
                notes.append("μ1b: в обучении один глагол — различать нечего")
                return None
            verb, verb_conf, margin, known = self.verb_model.predict(phrase)
            notes.append(f"μ1b: статистика n-грамм (известных n-грамм {known:.0%})")
            if known < KNOWN_GRAMS_FLOOR:
                notes.append(f"μ1b: фраза незнакома ({known:.0%} < {KNOWN_GRAMS_FLOOR:.0%})")
                return None
            if margin < MARGIN_FLOOR:
                notes.append(f"μ1b: нет запаса над вторым ({margin:.2f} < {MARGIN_FLOOR})")
                return None

        domain, domain_conf = domain_of(phrase, verb)
        if domain is None:
            notes.append("μ2: домен не распознан (уйдёт в ui)")
            domain, domain_conf = "ui", 0.6
        elif domain_conf < self.floor:
            notes.append(f"μ2: слабый домен ({domain_conf:.2f})")
            return None

        if verb_conf < self.floor:
            notes.append(f"μ1: слабая уверенность ({verb_conf:.2f})")
            return None

        # μ3: рой. Если текст говорит «наверх/останови» — не пытаемся решать сами.
        low = normalize(phrase)
        if verb in ESCALATING or any(w in low for w in ("эскалир", "наверх", "останови", "запрети")):
            notes.append("μ3: случай эскалации — решает омега")
            return None

        obj = extract_object(phrase, domain)
        if domain in PRECISE_DOMAINS and not obj:
            notes.append(f"μ1c: объект для домена {domain} не найден — решает омега")
            return None
        return Decision(
            verb=verb, domain=domain, confidence=min(verb_conf, domain_conf),
            source="μ1a+μ2" if ruled else "μ1b+μ2", notes=notes, object=obj or phrase,
        )


def selective_report(cascade: MicroCascade,
                     cases: Iterable[Dict]) -> Dict[str, object]:
    """Считает покрытие и точность на закрытых каскадом кейсах.

    Это и есть главное число: сколько фраз удалось закрыть без модели и
    насколько хорошо — то есть сколько денег экономится и чем платим.
    """
    total = covered = correct = 0
    verb_correct = domain_correct = 0
    missed: List[Dict[str, object]] = []
    for case in cases:
        phrase = case.get("prompt") or case.get("phrase") or ""
        # в наборах кейсов ключи называются verbs/domains; поддерживаем оба вида
        want_verbs = case.get("verbs") or case.get("expected_verbs") or []
        want_domains = case.get("domains") or case.get("expected_domains") or []
        total += 1
        decision = cascade.decide(phrase)
        if decision is None:
            continue
        covered += 1
        ok_verb = decision.verb in want_verbs
        ok_domain = decision.domain in want_domains or not want_domains
        verb_correct += int(ok_verb)
        domain_correct += int(ok_domain)
        if ok_verb and ok_domain:
            correct += 1
        else:
            missed.append({
                "id": case.get("id"), "phrase": phrase,
                "want": f"{want_verbs}/{want_domains}",
                "got": f"{decision.verb}/{decision.domain}",
                "conf": round(decision.confidence, 2),
            })
    return {
        "cases": total,
        "covered": covered,
        "coverage": round(covered / total, 3) if total else 0.0,
        "correct_on_covered": correct,
        "selective_accuracy": round(correct / covered, 3) if covered else None,
        "verb_accuracy_on_covered": round(verb_correct / covered, 3) if covered else None,
        "domain_accuracy_on_covered": round(domain_correct / covered, 3) if covered else None,
        "llm_calls_saved": covered,
        "errors": missed,
    }
