"""Гейт схемы журнала решений: другие агенты читают журнал только если колонки на месте.

Журнал var/decisions.tsv лежит в git (остальной var/ игнорируется): это единственное
место, где записаны принятые решения и замеренные числа, а не пересказ. Три строки
уже легли пятью полями вместо шести — человек этого не заметит, а следующий агент
перестанет понимать, где «почему», а где «чем проверено».
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_journal import HEADER, validate_lines  # noqa: E402


def test_pravilnaya_stroka_prohodit():
    row = "\t".join(["2026-10-30T14:29:48Z", "wave5", "reshenie", "pochemu", "chem", "itog"])
    assert validate_lines([HEADER, row]) == []


def test_stroka_s_pyatyu_polyami_lovitsya():
    row = "2026-10-30T14:29:48Z\twave5\treshenie\tpochemu\tchem"
    errors = validate_lines([HEADER, row])
    assert len(errors) == 1 and "5" in errors[0], errors


def test_bez_zagolovka_padaet():
    assert validate_lines(["ts\tphase\tdecision\twhy\tevidence"])


def test_pustoe_pole_ne_prohodit():
    row = "2026-10-30T14:29:48Z\twave5\t\tpochemu\tchem\titog"
    assert validate_lines([HEADER, row])


def test_crivoy_taymstamp_lovitsya():
    row = "30.10.2026\twave5\treshenie\tpochemu\tchem\titog"
    assert validate_lines([HEADER, row])


def test_data_only_stamp_dopustim_dlya_starovyh_strok():
    row = "2026-09-25\twave2\treshenie\tpochemu\tchem\titog"
    assert validate_lines([HEADER, row]) == []


def test_fakticheskiy_zhurnal_v_sheme():
    path = Path(__file__).resolve().parents[2] / "var" / "decisions.tsv"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert validate_lines(lines) == [], validate_lines(lines)[:3]
