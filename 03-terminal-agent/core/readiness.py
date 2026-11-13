"""Граница готовности кандидата: можно ли исполнять локально разобранное действие.

Разбор силён там, где фраза — это команда, и слаб там, где фраза лишь
*упоминает* действие. Раньше эти случаи не различались, и локальный роутер
отвечал действием на вопрос о действии:

    «не запускай команду date»        -> execute  (отрицание как приказ)
    «расскажи что означает вето»      -> veto     (вопрос как действие)
    «посоветуй как выполнить date»    -> execute  (совет как команда)
    «прочитай файл anchor.md»         -> anchor   (глагол внутри имени файла)

Этот модуль вводит один консервативный ответ на такие фразы: **воздержание**.
Воздержание не ошибка и не отказ — это передача решения большой модели.
Принцип: локально исполняется только то, что разобрано до конца; всё
недосказанное уходит наверх, а не исполняется частично.

Три проверки, каждая появилась из конкретного сбоя:

1. `intent_guard` — намерение: отрицание, вопрос/объяснение о действии,
   цитата, несколько действий.
2. `mask_objects` — имя файла или адрес не должен давать глагол
   («anchor.md» не делает фразу якорем).
3. `check_readiness` — обязательные параметры инструмента: `write` без
   содержимого и `select` без вариантов не готовы к исполнению.
"""

import re
import shlex
from typing import Dict, List, Optional, Sequence, Tuple

# --- 1. Намерение -------------------------------------------------------

# Отрицание перед действием: «не запускай», «не надо», «не трогай».
NEGATION_RE = re.compile(
    r"\bне\s+(?:надо\s+|нужно\s+|стоит\s+|будем\s+)?"
    r"(?:запуска|выполня|выполни|дела|делай|пиши|запиши|создава|удаля|"
    r"трога|открыва|меня|ставь|пробуй|смей)",
    re.I,
)
NEGATION_IMPERSONAL_RE = re.compile(
    r"\b(?:не\s+нужно|не\s+надо|нельзя|запрещено|не\s+стоит|"
    r"ничего\s+не|без\s+(?:моего\s+)?разрешени)\b", re.I)

# Основы глаголов в отрицании. Нужны потому, что глагол может быть выбран
# статистикой, а не правилом, — тогда «не сохраняй» негде было заметить.
NEGATION_STEM_RE = re.compile(
    r"\bне\s+(?:\S+\s+){0,2}?"
    r"(?:сохраня|записыва|пиши|запиши|создава|запуска|выполня|выполни|"
    r"удаля|трога|открыва|открой|меня|ставь|дела|делай|чита|прочти|"
    r"логир|канониз|закрепля|закрепи|якор|эскалир|запреща|печата|"
    r"напечата|показыва|загру)", re.I)

# Разделитель продолжения: «pwd, затем uname -a» — два действия.
SEQUENCE_RE = re.compile(r"\b(?:затем|потом|после\s+этого)\b", re.I)

# Вопрос или объяснение О действии: фраза говорит о действии, а не просит его.
META_RE = re.compile(
    r"\b(?:что\s+(?:означает|значит|такое)|расскажи|объясни|поясни|"
    r"в\s+чём\s+(?:смысл|разница)|чем\s+отличается|"
    r"как\s+(?:работает|устроен|выполнить|запустить|сделать|записать|"
    r"прочитать|открыть)|зачем\s+нуж|почему\s+нуж|посоветуй|подскажи|"
    r"можно\s+ли|стоит\s+ли|а\s+если\s+я|хочу\s+понять)\b",
    re.I,
)

# Действия, на которые вопрос о действии нельзя отвечать действием.
# `read` и `think` здесь нет намеренно: «объясни, что написано в файле» —
# это законный запрос чтения, и локальный ответ здесь уместен.
ACTION_VERBS = frozenset(
    {"write", "execute", "veto", "anchor", "log", "select", "escalate"})

# А вот в проверке «сколько действий во фразе» чтение участвует: иначе
# «прочитай A и запиши B» считалось бы одним действием и теряло второе.
MULTI_ACTION_VERBS = ACTION_VERBS | {"read"}

# Цитата: «слово "вето"», «фраза «запиши»».
QUOTE_RE = re.compile(r"[\"«'“]([^\"»'”]{1,80})[\"»'”]")

# --- 2. Маскирование объектов ------------------------------------------

# Путь или имя файла с расширением: config.yaml, notes.md, /tmp/x.txt
PATH_TOKEN_RE = re.compile(r"(?:\S*/)?[\w\-.]+\.(?:[A-Za-z]{1,5})\b")
# Имя в кавычках может содержать пробелы: «план на завтра.txt».
QUOTED_PATH_RE = re.compile(
    r"[\"«']([^\"»']{1,80}?\.[A-Za-z]{1,5})[\"»']")
# Файл без расширения после слова «файл»: «файл без расширения LICENSE».
BARE_FILE_RE = re.compile(
    r"файл\w*\s+(?:без\s+расширени\w*\s+)?([A-Za-z][\w-]{2,})", re.I)
URL_TOKEN_RE = re.compile(r"https?://\S+", re.I)
MASK = "«объект»"


def mask_objects(phrase: str) -> str:
    """Убирает из фразы имена файлов и адреса перед поиском глагола.

    «прочитай файл anchor.md» не должен становиться `anchor` только потому,
    что в имени файла встретилось слово anchor. Маска сохраняет длину
    смысловой части и не даёт правилам срабатывать на именах.
    """
    text = URL_TOKEN_RE.sub(MASK, phrase or "")
    text = QUOTED_PATH_RE.sub(MASK, text)
    text = PATH_TOKEN_RE.sub(MASK, text)
    text = BARE_FILE_RE.sub(lambda m: "файл " + MASK, text)
    return text


# Оболочки нет, поэтому цепочки и подстановки не поддерживаются вовсе:
# «echo first; echo second» с буквальным argv напечатало бы «first; echo second».
# Обратная кавычка тоже подстановка: без shell она прошла бы мусорным
# аргументом, и команда выполнилась бы не та, что просили.
SHELL_META_RE = re.compile(r"[;|]|&&|\$\(|`")

# Глаголы, которых нет в наборе из одиннадцати. Локальный роутер их не решает.
UNSUPPORTED_RE = re.compile(
    r"^\s*(?:сравни|сравнить|отмени|отменить|переименуй|переместить|"
    r"скопируй|удали|удалить)\b", re.I)

# Местоимение без названного объекта: истории беседы нет, «это» не на что
# указывать. «Сделай с этим что-нибудь полезное» — это вопрос, а не действие.
DEICTIC_RE = re.compile(
    r"\b(?:с\s+этим|это|этот|эта|эти|его|ее|её|их|туда|оттуда|там|то\s+же)\b",
    re.I)

# --- 3. Обязательные параметры -----------------------------------------

CONTENT_RE = re.compile(
    r"(?:строку|строкой|текст|текстом|содержимым|содержимое|записью|"
    r"значение)\s+"
    r"[\"«']?([^\"»']+?)[\"»']?\s*$",
    re.I)
OPTIONS_RE = re.compile(
    r"(?:вариант[а-я]*|одно\s+из\s+(?:двух|трёх|четырёх|\d+)|из\s+двух|"
    r"одно\s+из)\s*[:\-]?\s*(.+)$", re.I)
OR_SPLIT_RE = re.compile(r"\s+(?:или|либо)\s+|,\s*", re.I)


EMPTY_FILE_RE = re.compile(r"пуст\w+\s+файл", re.I)


def extract_content(phrase: str) -> Optional[str]:
    """Содержимое для write: «запиши в notes.md строку готово» -> «готово»."""
    text = (phrase or "").strip()
    if EMPTY_FILE_RE.search(text):
        return ""          # пустой файл — это законное пустое содержимое
    quoted = QUOTE_RE.search(text)
    if quoted:
        inside = quoted.group(1).strip()
        # Кавычки с именем файла — это объект, а не содержимое. Проверка
        # должна принимать пробелы внутри имени («мой план.txt»).
        is_name = bool(QUOTED_PATH_RE.search(quoted.group(0))) or \
            bool(PATH_TOKEN_RE.fullmatch(inside))
        if inside and not is_name:
            return inside
    # Содержимое не может быть именем файла: «в файл «мой план.txt» запиши
    # строку готово» — имя относится к объекту, а содержимое «готово».
    match = CONTENT_RE.search(QUOTED_PATH_RE.sub(" ", text))
    if match:
        value = match.group(1).strip()
        if value and not PATH_TOKEN_RE.fullmatch(value):
            return value
    return None


def extract_options(phrase: str) -> List[str]:
    """Варианты для select: «выбери A или B» -> ["A", "B"]."""
    text = (phrase or "").strip()
    match = OPTIONS_RE.search(text)
    candidate = match.group(1) if match else text
    parts = [part.strip(" .;:") for part in OR_SPLIT_RE.split(candidate)]
    # Служебные слова в начале первого варианта — часть вопроса, а не вариант:
    # «выбери между быстрым или дешёвым» -> ["быстрым", "дешёвым"].
    if parts:
        parts[0] = re.sub(
            r"^(?:(?:выбери|выбрать|между|что|из|лучше|вариант[а-я]*)\s+)+",
            "", parts[0], flags=re.I)
    parts = [part for part in parts if part and not PATH_TOKEN_RE.fullmatch(part)]
    return parts if len(parts) >= 2 else []


APPEND_RE = re.compile(r"\b(?:допиши|дописать|добавь\s+в|в\s+конец)\b", re.I)


def extract_append(phrase: str) -> bool:
    """Просит ли пользователь дописать, а не перезаписать."""
    return bool(APPEND_RE.search(phrase or ""))


def extract_argv(phrase: str, command: str) -> List[str]:
    """Полный argv команды, без shell.

    «запусти команду uname -a» -> ["uname", "-a"]. Раньше сохранялось только
    имя команды, и аргумент `-a` молча терялся, то есть исполнялась ДРУГАЯ
    команда. Здесь хвост после имени команды разбирается целиком; shell не
    участвует, поэтому подстановки и пайпы остаются обычным текстом.
    """
    text = (phrase or "").strip()
    if not command:
        return []
    match = re.search(r"\b" + re.escape(command) + r"\b(.*)$", text, re.I)
    tail = match.group(1).strip() if match else ""
    tail = tail.strip(" .;,")
    # «date и покажи вывод» — одно действие, «и покажи» не аргумент.
    # Режем только перед словом-продолжением, чтобы не съесть настоящий
    # аргумент вроде «echo 1 и 2».
    tail = re.split(
        r"(?:^|[\s,]+)(?:и|затем|потом|после\s+этого)\s+"
        r"(?=(?:покажи|показыва|выведи|выводи|скажи|напечатай|запусти|"
        r"выполни|прочита|прочти|открой))",
        tail)[0].strip(" .;,")
    if not tail:
        return [command]
    try:
        args = shlex.split(tail)
    except ValueError:
        args = tail.split()
    return [command] + args


COMMAND_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")


def check_readiness(verb: str, domain: str, phrase: str,
                    obj: Optional[str],
                    confirmed_by_cue: bool = True
                    ) -> Tuple[bool, Dict[str, object], str]:
    """Готов ли кандидат к исполнению. Возвращает (готов, параметры, причина).

    Причина непустая всегда, когда ответ «не готов»: она попадает в журнал,
    чтобы воздержание было видно, а не выглядело как сбой.
    """
    params: Dict[str, object] = {}
    # Статистика не имеет права придумывать действие с побочным эффектом:
    # «сравни содержимое old.txt и new.txt» без улики давало write.
    if verb in ("write", "execute", "anchor") and not confirmed_by_cue:
        return (False, params,
                f"{verb}: глагол угадан статистикой без явной улики — "
                f"побочное действие требует правила")
    if verb == "execute" and domain == "system":
        command = str(obj or "").split(":", 1)[-1].strip()
        if not command:
            return False, params, "execute: команда не найдена"
        if not COMMAND_NAME_RE.match(command):
            return (False, params,
                    f"execute: «{command}» не похоже на имя команды — "
                    f"команда не названа")
        params["command"] = command
        params["argv"] = extract_argv(phrase, command)
        return True, params, ""
    if verb == "write" and domain == "filesystem":
        path = str(obj or "")
        if path.startswith("file:"):
            path = path[5:]
        looks_like_path = bool(PATH_TOKEN_RE.search(path))
        if not looks_like_path:
            return (False, params,
                    "write: путь не назван — записывать некуда")
        content = extract_content(phrase)
        if content is None:
            return (False, params,
                    "write: не найдено содержимое (parameters.content) — "
                    "исполнитель его требует")
        if extract_append(phrase):
            # Дописывание нельзя молча подменять перезаписью. Файловый
            # инструмент режима append не имеет, поэтому отправляем наверх.
            return (False, params,
                    "write: требуется дописывание, а инструмент умеет только "
                    "перезапись — подменять режим без спроса нельзя")
        params["content"] = content
        return True, params, ""
    if verb == "select" and domain in ("ui", "swarm", "connectome"):
        options = extract_options(phrase)
        if not options:
            return (False, params,
                    "select: не найдены варианты (parameters.options) — "
                    "исполнитель их требует")
        params["options"] = options
        return True, params, ""
    if verb == "read" and domain in ("filesystem", "web"):
        if not obj:
            return False, params, f"read: объект домена {domain} не найден"
        return True, params, ""
    return True, params, ""


def intent_guard(phrase: str, verb: str) -> Tuple[bool, str]:
    """Можно ли отвечать действием. (разрешено, причина отказа)."""
    text = phrase or ""
    masked = mask_objects(text)

    if (NEGATION_RE.search(masked) or NEGATION_IMPERSONAL_RE.search(masked)
            or NEGATION_STEM_RE.search(masked)):
        return False, "отрицание: фраза запрещает действие, а не просит его"

    if META_RE.search(masked) and verb in ACTION_VERBS:
        return False, f"вопрос или объяснение о действии «{verb}», а не приказ"

    quoted = QUOTE_RE.search(text)
    if quoted:
        inside = quoted.group(1).strip()
        # В кавычках может быть что угодно: имя файла, содержимое записи или
        # упоминание действия. Считать примером можно только последнее,
        # иначе «запиши текст «Привет, мир!»» отвергается как цитата команды.
        from .micromodels import cue_verb as _cue
        quoted_verb, _ = _cue(inside)
        if quoted_verb in ACTION_VERBS and not PATH_TOKEN_RE.fullmatch(inside):
            return False, "действие упомянуто в кавычках как пример"

    return True, ""


def distinct_action_verbs(phrase: str, cue_lookup) -> List[str]:
    """Сколько разных действий требует фраза.

    «прочитай notes.md и запиши копию в backup.md» — это два действия, и
    локально исполнять только первое значит молча потерять второе.
    """
    masked = mask_objects(phrase)
    # Содержимое записи — это текст, а не команды: во фразе «запиши текст
    # прочитай и запусти» глаголы внутри содержимого не являются действиями.
    content = extract_content(phrase)
    if content:
        masked = masked.replace(content, " ")
    found: List[str] = []
    # Делим по союзам и разбираем каждую часть отдельно.
    for chunk in re.split(
            r"\s+и\s+|\s*;\s*|,?\s+(?:затем|потом|после\s+этого)\s+",
            masked):
        verb, _ = cue_lookup(chunk)
        if verb in MULTI_ACTION_VERBS and verb not in found:
            found.append(verb)
    return found


# --- 4. Канонизация объекта --------------------------------------------

# Префикс, которым домен обозначается в объекте.
DOMAIN_PREFIX = {
    "filesystem": "file:",
    "system": "sys:",
    "registry": "reg:",
    "connectome": "connectome:",
    "web": "web:",
    "swarm": "pheromone:",
}
KNOWN_PREFIXES = ("file:", "path:", "dir:", "folder:", "fs:", "sys:", "cmd:",
                  "sh:", "exec:", "reg:", "registry:", "swarm:", "pheromone:",
                  "connectome:", "neuron:", "fly:", "fw:", "web:", "url:",
                  "http://", "https://")


def object_of_registry(phrase: str) -> Optional[str]:
    """Объект реестра: сначала путь из фразы, потом очищенный остаток.

    «Какая версия у файла report.txt?» — объект `report.txt`, а не вся фраза.
    """
    match = PATH_TOKEN_RE.search(phrase or "")
    if match:
        return match.group(0)
    return None


def canonicalize_object(domain: str, obj: Optional[str]) -> Optional[str]:
    """Приводит объект к виду, из которого домен выводится однозначно.

    Иначе слои расходятся: каскад говорит `registry`, а грамматика по тому же
    объекту выводит `ui`, и проверка прав идёт не по той паре. Канонизация
    должна происходить ДО проверки прав, а не заменяться доверием к модели.
    """
    if not obj:
        return obj
    text = str(obj)
    lowered = text.lower()
    if any(lowered.startswith(prefix) for prefix in KNOWN_PREFIXES):
        return text
    prefix = DOMAIN_PREFIX.get(domain)
    if not prefix:
        return text          # ui и прочее без префикса — домен и так ясен
    return prefix + text


COMMAND_AFTER_MARKER_RE = re.compile(
    r"(?:запусти|выполни|прогони)\s+(?:команду\s+)?([A-Za-z][\w.-]*)", re.I)


def action_targets(phrase: str) -> List[str]:
    """Различные цели действия: адреса, пути, команды.

    Считать глаголы недостаточно: «выполни pwd, затем uname -a» — две команды
    с ОДНИМ глаголом execute. Но и пути считать всегда нельзя: в
    «wc -l notes.md» путь — аргумент единственной команды, а не вторая цель.
    """
    text = phrase or ""
    targets = set()
    for match in URL_TOKEN_RE.finditer(text):
        targets.add(match.group(0))
    masked = URL_TOKEN_RE.sub(" ", text)
    masked = QUOTED_PATH_RE.sub(lambda m: " " + m.group(1) + " ", masked)
    paths = {m.group(0).lower() for m in PATH_TOKEN_RE.finditer(masked)}
    commands = [m.group(1).lower()
                for m in COMMAND_AFTER_MARKER_RE.finditer(masked)]
    targets.update(commands)
    if not commands:
        targets.update(paths)
    elif len(commands) > 1:
        targets.update(paths)
    return sorted(targets)

