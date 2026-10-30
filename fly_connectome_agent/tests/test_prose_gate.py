"""Гейт письменности должен уметь падать: ловит он дефекты и не кричит ли на легальное.

Проверка, которая не может упасть, бесполезна — поэтому правило из
scripts/check_prose.py покрыто поведенческими тестами, а не только прогоном по
репозиторию. Примеры дефектов собраны из кодовых точек: если написать их
литералом, этот файл сам попадёт под гейт.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_prose import offending_words  # noqa: E402

LATIN_E = chr(0x65)      # латинская e
LATIN_R = chr(0x52)      # латинская R, похожа на кириллическую Р
IDEOGRAM = chr(0x90BB)   # иероглиф «сосед»


def test_losit_latinskuyu_glasnuyu_na_kontse_kirillitsy():
    assert offending_words("коннектом" + LATIN_E + " реально не читается")


def test_losit_latinskuyu_bukvu_vmesto_kirillocheskoy():
    assert offending_words(LATIN_R + "оль super_class ne proveryali")


def test_losit_ieroglif_v_proze():
    assert offending_words("zatem ih " + IDEOGRAM + " okruzhenie")


def test_razdelitel_spasaet_legalnyy_termin():
    assert not offending_words("toy-сеть и numpy-массив легальны")
    assert not offending_words("seed'ов повторений, control-условия")
    assert not offending_words("super_class lezhit v Node.class_id")


def test_stroka_koda_bez_kirillitsy_ne_defekt():
    assert not offending_words("    def io_channels(graph, *, inputs) -> IOChannels:")


def test_escape_posledovatelnost_ne_proza():
    assert not offending_words('print("stroka' + chr(92) + 'n")')
