#!/usr/bin/env python3
"""Генератор корпуса фраз для дистилляции микромоделей.

Зачем: правила μ1a писал я, поэтому размечать ими же нельзя — замер будет
круговым. Корпус генерируется здесь БЕЗ разметки, а размечает его омега
(eval/label_cases.py). Каскад учится повторять омегу там, где уверен.

Ключевое: у каждой фразы есть семейство шаблонов (`family`). Разделение на
обучение/отложенное идёт ПО СЕМЕЙСТВАМ, а не по фразам — иначе отложенная
часть проверяла бы запоминание конкретных строк, а не перенос на новую форму
запроса.
"""

import json
import random
from pathlib import Path

SEED = 20260101
OUT = Path(__file__).with_name("cases_distill_raw.json")

FILES = ("notes.md", "config.yaml", "otchet.txt", "registry.jsonl", "plans.md",
         "readme.md", "log.txt", "data.csv", "main.py", "settings.json")
URLS = ("https://example.com", "https://docs.python.org/3/", "https://peps.python.org/",
        "https://github.com/D4Vinci/Scrapling", "https://arxiv.org/abs/1706.03762")
COMMANDS = ("date", "uname", "pwd", "whoami", "ls", "df", "ps", "git")
TOPICS = ("глаголы", "коннектом", "архитектура", "рейестр", "порог уверенности")
REG_ENTRIES = ("канон", "правило-7", "версия-2", "запись-abc", "канон-муха")
UI_CHOICES = ("A или B", "первый или второй", "быстрый или дешёвый")
NEURON_QUERIES = ("классы нейронов", "связи", "статистику", "нейроны-хабы")

# (семейство, список шаблонов, подстановки)
FAMILIES = [
    # ---------- READ ----------
    ("read-r1", ["прочитай {f}", "прочти {f}", "покажи содержимое {f}"], FILES),
    ("read-r2", ["что написано в {f}?", "что лежит в {f}?", "что в {f}?"], FILES),
    ("read-r3", ["загляни в {f} и перескажи", "открой {f} и скажи, что там"], FILES),
    ("read-r4", ["выведи {n} из коннектома", "покажи {n} коннектома"], NEURON_QUERIES),
    ("read-r5", ["прочитай страницу {u}", "открой {u} и перескажи"], URLS),
    # ---------- WRITE ----------
    ("write-w1", ["запиши в {f} строку готово", "сохрани в {f} результат"], FILES),
    ("write-w2", ["создай {f} с планом на день", "сделай {f} и положи туда список"], FILES),
    ("write-w3", ["добавь в {f} новую строку", "перезапиши {f} целиком"], FILES),
    # ---------- EXECUTE ----------
    ("exec-e1", ["запусти команду {c}", "выполни {c} в терминале"], COMMANDS),
    ("exec-e2", ["выполни {c}", "прогони {c} и покажи вывод"], COMMANDS),
    ("exec-e3", ["какая сейчас директория? запусти {c}", "покажи {c} командой"], COMMANDS),
    # ---------- SELECT ----------
    ("select-s1", ["какого размера файл {f}?", "сколько весит {f}?"], FILES),
    ("select-s2", ["выбери одно из двух: {v}", "что лучше: {v}?"], UI_CHOICES),
    ("select-s3", ["какой тип у файла {f}?", "какой формат у {f}?"], FILES),
    # ---------- LOG ----------
    ("log-l1", ["сделай пометку в журнале о завершении", "залогируй результат работы"], None),
    ("log-l2", ["занеси в журнал, что задача готова", "отметь в журнале начало работы"], None),
    # ---------- VERSION ----------
    ("ver-v1", ["какая сейчас версия агента?", "покажи версию системы"], None),
    ("ver-v2", ["какая версия у {f}?", "версию чего ты сейчас используешь?"], FILES),
    # ---------- ANCHOR ----------
    ("anchor-a1", ["закрепи запись {r}", "заякорь {r} в реестре"], REG_ENTRIES),
    ("anchor-a2", ["сделай {r} каноном", "закрепи {r} как эталон"], REG_ENTRIES),
    # ---------- VETO ----------
    ("veto-x1", ["наложи вето на запись {r}", "запрети запись {r}"], REG_ENTRIES),
    ("veto-x2", ["заблокируй {r} в реестре", "поставь запрет на {r}"], REG_ENTRIES),
    # ---------- ESCALATE ----------
    ("esc-e1", ["эскалируй это наверх оператору", "передай вопрос выше"], None),
    ("esc-e2", ["сообщи оператору, что нужна помощь", "позови человека, я не уверен"], None),
    # ---------- RECOMMEND ----------
    ("rec-r1", ["что посоветуешь делать дальше?", "что рекомендуешь в первую очередь?"], None),
    ("rec-r2", ["дай совет, с чего начать", "посоветуй, что улучшить"], None),
    # ---------- THINK ----------
    ("think-t1", ["зачем вообще нужны глаголы?", "почему глагол важнее промпта?"], None),
    ("think-t2", ["объясни, как работает {t}", "как устроен {t}?"], TOPICS),
]


def build() -> dict:
    rng = random.Random(SEED)
    phrases = []
    seen = set()
    for family, templates, fills in FAMILIES:
        variants = fills if fills else [None]
        for template in templates:
            for fill in variants:
                text = template.format(
                    f=fill, u=fill, c=fill, n=fill, v=fill, r=fill, t=fill
                ) if fill else template
                if fill:
                    # разные подстановки с одинаковым шаблоном — разные фразы
                    pass
                key = (family, text)
                if key in seen:
                    continue
                seen.add(key)
                phrases.append({"id": f"{family}-{len(phrases):03d}",
                                "family": family, "phrase": text})
    rng.shuffle(phrases)
    for index, item in enumerate(phrases):
        item["id"] = f"d{index:03d}"
    return {
        "meta": {
            "name": "distill-raw",
            "purpose": "фразы БЕЗ разметки; метки ставит омега в label_cases.py",
            "seed": SEED,
            "families": len(FAMILIES),
            "count": len(phrases),
            "note": "разделение train/eval — по family, чтобы отложенная часть "
                    "проверяла перенос на новую форму запроса",
        },
        "phrases": phrases,
    }


if __name__ == "__main__":
    payload = build()
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"фраз: {payload['meta']['count']} в {payload['meta']['families']} семействах")
    print(f"записано: {OUT}")