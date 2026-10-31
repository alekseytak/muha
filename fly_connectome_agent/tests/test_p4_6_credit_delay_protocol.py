"""Предрегистрация P4.6 (время доставки кредита) проверена кодом.

Три вещи, которые нельзя держать на доверии к тексту манифеста:

1. **Заморозка.** Digest манифеста заперт литералом. Правка любой цифры после
   заморозки краснит тест, а не тихо меняет протокол задним числом.
2. **Отсутствие тихого конфауна.** v4 потеряла вывод именно потому, что вместе с
   привязкой изменилась амплитуда кредита. Здесь задержка тоже бьёт по двум
   признакам сразу (угасание следа и его загрязнение спайками интервала
   задержки), поэтому манифест обязан объявлять внутренний контроль равной
   амплитуды, а рабочая точка — совпадать с p4.5 во всём, кроме правила третьей
   компоненты.
3. **Grid'ы.** Имя клетки (`{arm}-s{seed}.tsv`) задержку не кодирует, поэтому
   уровни D живут в отдельных запусках и отдельных каталогах. Раннер обязан
   отказывать на undeclared-задержке, на чужом наборе рук и на чужом каталоге, а
   манифесты без `grids` (все предыдущие волны) — верифицироваться по-старому.

Ни один тест не запускает прогон: проверяется целостность предрегистрации и
поведение отказа.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import run_p4_loop as runner  # noqa: E402

from fly_connectome_agent.src.engineering.harness.snn_corridor import (  # noqa: E402
    DELAY_ARMS,
    LoopSettings,
)

MANIFEST = REPO / "fly_connectome_agent/manifests/p4_6_credit_delay.json"
MANIFEST_V5 = REPO / "fly_connectome_agent/manifests/p4_5_neuron_credit_amplitude.json"

# Якорь заморозки: digest файла на диске обязан совпасть с этим значением.
FROZEN_SHA256 = "d098378ed9966030f660bac9697284c08aac2cf947cf582ff84f9cee4d1ee241"

# Литералы, а не вычитывание из манифеста: сверять манифест с самим собой —
# тавтология, правка обоих полей прошла бы незамеченной.
CONFIRMATORY = [701, 711, 721]
FORBIDDEN_SEEDS = {7, 182, 501, 511, 521, 601, 611, 621}
WORKING_POINT = dict(episodes=500, window_steps=100, i_drive=20.0, gain=6.0,
                     sigma=3.0, eta=0.03, pool_contrast=0.0, neuron_credit=0.0,
                     coherence_gain=0.0, max_episode_steps=12, width=7, height=7,
                     start=(1, 1), target=(5, 5))
GRID_OUT = {0: "var/p4_6_d0", 20: "var/p4_6_d20", 80: "var/p4_6_d80"}
GRID_ARMS = {0: ["plastic", "no_plasticity", "oracle"],
             20: ["delay_live", "delay_frozen"],
             80: ["delay_live", "delay_frozen"]}
GRID_ID = {0: "d0", 20: "d20", 80: "d80"}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _settings(delay: int) -> dict:
    return {**WORKING_POINT, "credit_delay_steps": delay}


def _refuses(arms: list[str], seeds: list[int], settings: dict, out: str,
             needle: str, path: Path = MANIFEST) -> None:
    with pytest.raises(SystemExit) as exc:
        runner.verify_protocol(str(path), arms, seeds, settings, out)
    text = str(exc.value)
    assert needle in text, text


# --- 1. заморозка ----------------------------------------------------------

def test_digest_manifesta_zamorozhen():
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == FROZEN_SHA256


def test_protocol_id_i_preregistraciya(manifest):
    assert manifest["protocol_id"] == "p4.6.spike-verb.credit-delay.v1"
    assert manifest["pre_registered"] is True
    assert manifest["supersedes"] is None
    assert manifest["claims_allowed"] == ["simulation_result", "engineering_performance"]


# --- 2. сетка задержек -----------------------------------------------------

def test_setka_zaderzhek_obyavlena_tremya_urovniami(manifest):
    grids = manifest["grids"]
    assert sorted(g["credit_delay_steps"] for g in grids) == [0, 20, 80]
    assert [g["grid_id"] for g in grids] == ["d0", "d20", "d80"]
    assert {g["out"] for g in grids} == set(GRID_OUT.values())
    for g in grids:
        assert g["arms"] == GRID_ARMS[g["credit_delay_steps"]]
        assert g["role"].strip()


def test_uroven_zaderzhek_ne_vyishodit_za_okno(manifest):
    """D <= window_steps — не удобство, а условие того, что очередь не больше
    одного элемента и доставка попадает внутрь следующего окна."""
    window = manifest["operating_point"]["values"]["window_steps"]
    for g in manifest["grids"]:
        assert 0 <= g["credit_delay_steps"] <= window


def test_ruki_setki_obyavleny_v_verhnem_spiske(manifest):
    top = {a["arm_id"] for a in manifest["arms"]}
    for g in manifest["grids"]:
        assert set(g["arms"]) <= top, g["grid_id"]
    assert top == set(GRID_ARMS[0]) | set(DELAY_ARMS)


def test_delay_reki_v_kode_i_v_manifeste_odni_i_te_zhe(manifest):
    declared = {a["arm_id"]: a["delay_mode"] for a in manifest["arms"]}
    assert {arm for arm, mode in declared.items() if mode != "none"} == set(DELAY_ARMS)
    assert {DELAY_ARMS[arm]: arm for arm in DELAY_ARMS} == {
        mode: arm for arm, mode in declared.items() if mode != "none"}


# --- 3. руки и предсказание амплитуды --------------------------------------

def test_treatament_kontrol_i_referens_nazvany_i_ne_putany(manifest):
    treat, ctrl, ref = (manifest["treatment_arm"], manifest["amplitude_control_arm"],
                        manifest["reference_arm"])
    assert (treat, ctrl, ref) == ("delay_live", "delay_frozen", "plastic")
    assert manifest["primary_delay_steps"] == 80
    assert manifest["secondary_delay_steps"] == 20
    roles = {a["arm_id"]: a["role"] for a in manifest["arms"]}
    assert roles[treat] == "treatment_v6"
    assert roles[ctrl] == "amplitude_control"
    assert roles[ref] == "reference_immediate_v1"


def test_osnovnoy_uroven_raven_vremeni_ugasaniya(manifest):
    """D = tau выбрано для того, чтобы предсказание было округляемым числом,
    а полоса H4 ловила ошибку механизма, а не совпадение случайностей."""
    values = manifest["operating_point"]["values"]
    assert manifest["primary_delay_steps"] == int(values["tau_eligibility"])
    h4 = next(h for h in manifest["hypotheses"] if h["id"] == "H4")
    assert h4["kind"] == "mechanism"
    assert h4["relative_band"] == 0.15
    assert h4["expected_from_formula"] == round(math.exp(-1.0), 4) == 0.3679


def test_pravilo_kredita_skalyar_v1(manifest):
    """Линия формы кредита закрыта в v2-v5; если бы v6 принёс ещё и пульной или
    пер-нейронный режим, сравнивать было бы не время, а две починки сразу."""
    values = manifest["operating_point"]["values"]
    assert values["pool_contrast"] == 0.0 and values["neuron_credit"] == 0.0
    assert manifest["modulator"]["pool_rule"]["kappa"] == 0.0
    assert manifest["modulator"]["neuron_rule"]["lambda"] == 0.0
    assert "neuron" not in json.dumps([g["arms"] for g in manifest["grids"]])
    assert "v1" in manifest["frozen_inputs"]["timing_rule"]["rule"]


def test_kontrol_ravnoy_amplitudy_obyavlen_do_chisel(manifest):
    """Без frozen задержка меняет амплитуду и адрес одним движением — ровно та
    ошибка, из-за которой вывод v4 пришлось перепроверять в v5."""
    timing = manifest["frozen_inputs"]["timing_rule"]
    assert timing["arm_delay_live"] and timing["arm_delay_frozen"]
    reason = manifest["operating_point"]["derivation"]["why_frozen_is_the_control"]
    assert "амплитудно-равного" in reason and "v4" in reason
    assert "decay**D" in timing["arm_delay_frozen"]


# --- 4. рабочая точка не сдвинута -----------------------------------------

def test_rabochaya_tochka_nasleduetsya_ot_p4_5(manifest):
    """Физика обязана остаться ровно v5: v6 меняет только время доставки.
    Единственное разрешённое отличие — выключенная третья компонента."""
    v5 = json.loads(MANIFEST_V5.read_text(encoding="utf-8"))
    mine, theirs = manifest["operating_point"]["values"], v5["operating_point"]["values"]
    assert set(mine) == set(theirs)
    diff = {k for k in mine if mine[k] != theirs[k]}
    assert diff == {"neuron_credit"}, diff
    assert mine["neuron_credit"] == 0.0


def test_zaderzhka_ne_vhodit_v_operating_point(manifest):
    """Уровни задержки — часть сетки запусков, а не рабочей точки: иначе одна
    клетка не смогла бы сказать, из какого она grid'а."""
    assert "credit_delay_steps" not in manifest["operating_point"]["values"]
    assert all("credit_delay_steps" in g for g in manifest["grids"])


# --- 5. seed'ы -------------------------------------------------------------

def test_seedy_podtverzhdeniya_sviezhi(manifest):
    chosen = manifest["seed_sets"]["confirmatory"]
    assert chosen["values"] == CONFIRMATORY and chosen["count"] == 3
    assert not set(chosen["values"]) & FORBIDDEN_SEEDS


def test_seedy_ne_peresekayutsya_ni_s_kem(manifest):
    chosen = set(manifest["seed_sets"]["confirmatory"]["values"])
    burned: set[int] = set()
    for entry in manifest["seed_sets"]["burned"]:
        assert entry["id"] != manifest["seed_sets"]["confirmatory"]["id"]
        if "values" in entry:
            assert not chosen & set(entry["values"]), entry["id"]
            burned |= set(entry["values"])
        elif "range" in entry:
            lo, hi = entry["range"]
            assert not chosen & set(range(lo, hi + 1)), entry["id"]
            burned |= set(range(lo, hi + 1))
    assert burned >= FORBIDDEN_SEEDS, "в реестре сожжённых потеряны seed'ы 7/182/50x/60x"
    ids = [e["id"] for e in manifest["seed_sets"]["burned"]]
    assert len(ids) == len(set(ids))
    assert "p4_6_pilot" in ids and "p4_6_smoke" in ids


# --- 6. таблица исходов, гипотезы, гейты ----------------------------------

def test_tablica_ishodov_pokryvaet_chetyire_sochetaniya(manifest):
    rows = manifest["outcome_table"]
    combos = {(r["A"], r["B"]) for r in rows}
    assert combos == {(True, True), (True, False), (False, True), (False, False)}
    assert all(r["reading"].strip() for r in rows)
    signs = manifest["outcome_table_signs"]
    assert set(signs) == {"A", "B"}
    assert "delay_live" in signs["A"] and "delay_frozen" in signs["B"]


def test_gipotezy_sravnivayut_para_mi_na_odnom_seedzie(manifest):
    """A и B — парные сравнения на одном seed'е: прирост delay_live против
    немедленного v1 и против замороженного контроля при том же D."""
    by_id = {h["id"]: h for h in manifest["hypotheses"]}
    assert [h["kind"] for h in manifest["hypotheses"]] == \
        ["required", "required", "required", "mechanism", "characterization"]
    assert by_id["H2"]["endpoint"] == "delta_plastic_minus_delta_delay_live_d80"
    assert by_id["H3"]["endpoint"] == "delta_delay_frozen_d80_minus_delta_delay_live_d80"
    assert by_id["H1"]["direction"] == "greater"
    assert by_id["H5"]["kind"] == "characterization"


def test_geyty_desyat_i_kazhdyy_s_pravilom_i_chteniem(manifest):
    guards = manifest["guards"]
    assert [g["id"] for g in guards] == [f"G{i}" for i in range(1, 11)]
    assert all(g["rule"].strip() and g["on_fail"].strip() for g in guards)
    names = {g["id"]: g["name"] for g in guards}
    assert names["G9"] == "mechanism_fired" and names["G10"] == "grid_integrity"
    assert "deliveries" in guards[8]["rule"] and "pending_at_end" in guards[8]["rule"]
    assert "d0" in guards[9]["rule"]


def test_mehanika_gasit_kredit_rovno_na_exp_minus_D_over_tau(manifest):
    """Манифест обещает равенство, исполняемое кодом, а не метафору: frozen
    гасит слепок собственной константой обучателя."""
    timing = manifest["frozen_inputs"]["timing_rule"]
    assert "decay = exp(-dt/tau_eligibility)" in timing["arm_delay_frozen"]
    assert "не пересчитанная" in timing["arm_delay_frozen"]
    assert "бит-в-бит" in timing["identity"] and "unit-тест" in timing["identity"]


def test_uroven_pyat_shagov_isklyuchen_yavno(manifest):
    """Пятишаговая задержка из постановки задачи (множитель 0.9394) не вошла в
    сетку: при n=3 это под уровнем различимости. Отказ задокументирован, а не
    забыт."""
    derivation = manifest["operating_point"]["derivation"]
    assert "0.9394" in derivation["why_levels_20_and_80"]
    assert any("0.9394" in lim for lim in manifest["known_limitations"])
    assert 5 not in [g["credit_delay_steps"] for g in manifest["grids"]]


def test_statistika_opisatelna_pri_treh_seedah(manifest):
    st = manifest["statistics"]
    assert st["unit_of_observation"] == "seed" and st["n_seeds"] == 3
    assert st["claim"] == "descriptive"
    assert "0.125" in st["note"]


# --- 7. поведение отказа раннера ------------------------------------------

def test_zaderzhka_ne_iz_setki_otsivaetsya(manifest):
    for delay in (5, 81, 100):
        _refuses(GRID_ARMS[80], CONFIRMATORY, _settings(delay), GRID_OUT[80],
                 "не соответствует ни один grid")
    with pytest.raises(ValueError):
        LoopSettings(**{**WORKING_POINT, "credit_delay_steps": -1}, seed=701)
    assert [g["credit_delay_steps"] for g in manifest["grids"]] == [0, 20, 80]


def test_ruki_grida_otsivayutsya(manifest):
    _refuses(GRID_ARMS[80] + ["plastic"], CONFIRMATORY, _settings(80), GRID_OUT[80],
             "руки")
    _refuses(["delay_live"], CONFIRMATORY, _settings(80), GRID_OUT[80], "руки")


def test_katalog_vyvoda_otsivaetsya(manifest):
    """Смешать grid'ы в одном каталоге — значит потерять, какая клетка с какой
    задержкой: имя клетки задержку не кодирует."""
    _refuses(GRID_ARMS[80], CONFIRMATORY, _settings(80), "var/p4_6_d0",
             "каталог вывода")
    _refuses(GRID_ARMS[80], CONFIRMATORY, _settings(80), "var/p4_6_mixed",
             "каталог вывода")


def test_chuzhaya_rabochaya_tochka_i_seedy_otsivayutsya(manifest):
    _refuses(GRID_ARMS[80], [701, 711, 722], _settings(80), GRID_OUT[80], "seed'ы")
    moved = {**_settings(80), "eta": 0.03 * math.exp(-1.0)}
    _refuses(GRID_ARMS[80], CONFIRMATORY, moved, GRID_OUT[80], "eta")
    shifted = {**_settings(80), "window_steps": 80}
    _refuses(GRID_ARMS[80], CONFIRMATORY, shifted, GRID_OUT[80], "window_steps")
    third = {**_settings(80), "neuron_credit": 0.5}
    _refuses(GRID_ARMS[80], CONFIRMATORY, third, GRID_OUT[80], "neuron_credit")


def test_kazhdyy_grid_veriyaetsya_svoim_zapuskom(manifest):
    for delay, arms in GRID_ARMS.items():
        info = runner.verify_protocol(str(MANIFEST), arms, CONFIRMATORY,
                                      _settings(delay), GRID_OUT[delay])
        assert info["protocol_id"] == manifest["protocol_id"]
        assert info["sha256"] == FROZEN_SHA256
        assert info["grid_id"] == GRID_ID[delay]
        assert info["credit_delay_steps"] == delay


def test_manifest_bez_setok_verificiruetsya_po_staromu(manifest):
    """Обратная совместимость: все предыдущие волны (p4.2-p4.5) без поля `grids`
    обязаны верифицироваться как раньше, иначе правка ради v6 сломала бы
    воспроизведение старых прогонов."""
    v5 = json.loads(MANIFEST_V5.read_text(encoding="utf-8"))
    assert "grids" not in v5
    arms = [a["arm_id"] for a in v5["arms"]]
    seeds = v5["seed_sets"]["confirmatory"]["values"]
    settings = {**WORKING_POINT,
                "neuron_credit": v5["operating_point"]["values"]["neuron_credit"]}
    info = runner.verify_protocol(str(MANIFEST_V5), arms, seeds, settings,
                                  "var/p4_5_confirm")
    assert "grid_id" not in info and "credit_delay_steps" not in info
    assert info["protocol_id"] == v5["protocol_id"]
    assert sorted(seeds) != sorted(manifest["seed_sets"]["confirmatory"]["values"])


def test_bez_kataloga_grid_versiya_prohodima(manifest):
    """out_dir — необязательный аргумент: сам grid при этом всё равно выбирается
    по задержке, а проверка каталога просто не выполняется."""
    info = runner.verify_protocol(str(MANIFEST), GRID_ARMS[20], CONFIRMATORY,
                                  _settings(20))
    assert info["grid_id"] == "d20"


def test_granicy_nastroyki_zaderzhki():
    """Границы `credit_delay_steps` принадлежат настройке, а не руке: plastic с
    задержкой отвергается на уровне CorridorAgent (см. test_credit_delay.py)."""
    settings = LoopSettings(**_settings(10), seed=701)
    assert settings.credit_delay_steps == 10
    with pytest.raises(ValueError):
        LoopSettings(window_steps=20, credit_delay_steps=21)
    LoopSettings(window_steps=20, credit_delay_steps=20)
