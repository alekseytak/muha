"""P4.2 — oracle baseline: свойства проводки и свойства гейта.

Две половины файла проверяют разные вещи, и обе нужны до запуска прогона.

Первая — сам oracle. Утверждения «fixed, symmetric, non-plastic, reward-blind»
не выглядят правдой сами по себе: right_bias тоже «рефлекс, который знает
ответ», но знает он его только с одной стороны. Поэтому симметрия проверяется
перестановкой слотов (вектор должен совпасть целиком, а не только четыре
сенсорных ребра), а рядом стоит opposite-test: right_bias под той же
перестановкой обязан рассыпаться. Проверка, которая не может упасть, бесполезна.

Вторая — замороженный протокол и гейт. Смысл pre-registration не в тексте, а в
том, что правка после факта падает: silent сдвиг eta, объявление oracle
гейтом, недостающий seed в матрице, «замороженный» контроль с nonzero Δw. Всё
это здесь ловится на синтетических данных, без единого смоделированного эпизода.

Поведенческие тесты идут на seed'ах 900-901 (manifest seed_sets.fixture_probe) —
вне confirmatory 120-179 и вне обоих множеств P4.1, так что ни один тест не
засчитывается как наблюдение.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import p4_2_protocol as proto  # noqa: E402
import check_p4_2_oracle_gate as gate  # noqa: E402
import run_p4_2_oracle as runner  # noqa: E402

from fly_connectome_agent.src.engineering.harness import symmetric_toy as toy  # noqa: E402
from fly_connectome_agent.src.engineering.harness.symmetric_toy import (  # noqa: E402
    MIRROR_PERM,
    SLOT_CONTRA_L,
    SLOT_CONTRA_R,
    SLOT_DIRECT_L,
    SLOT_DIRECT_R,
    ToySettings,
    init_weights,
    make_modulator,
    run_arm,
)

# Перестановка слотов при зеркальном отражении лево↔право выводится из графа в
# harness (MIRROR_PERM); ниже она закреплена литералом, чтобы вывод не мог
# "самосогласоваться". Если вектор весов инвариантен этой перестановке, у
# проводки нет встроенного предпочтения направления.
EXPECTED_MIRROR_PERM = [2, 3, 0, 1, 5, 4, 7, 6, 9, 8]

# Якорь pre-registration: digest замороженного манифеста, зашитый в код.
FROZEN = proto.FROZEN_PROTOCOL_DIGEST

PROBE_SEEDS = [900, 901]  # вне всех гейтов

# Seed для отладочных прогонов внутри тестов: вне всех объявленных множеств.
DEBUG_SEED = 500


def _fast(**overrides) -> ToySettings:
    base = dict(window_steps=10, max_steps=8, track_length=9, cell_distance=3,
                edge_margin=1, episodes=2)
    base.update(overrides)
    return ToySettings(**base)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return proto.load()


# --- 1. проводка oracle ---------------------------------------------------

def test_mirror_perm_is_the_expected_involution():
    """Вывод перестановки обязан совпадать с задокументированным литералом.

    Без этой привязки проверка симметрии стала бы тавтологией: перестановку
    можно было бы подстроить так, чтобы любой вектор оказывался "симметричным".
    """
    assert MIRROR_PERM == EXPECTED_MIRROR_PERM
    assert sorted(MIRROR_PERM) == list(range(len(MIRROR_PERM)))
    assert [MIRROR_PERM[i] for i in MIRROR_PERM] == list(range(len(MIRROR_PERM)))


def test_oracle_direct_paths_strong_and_contra_weak():
    w = init_weights("oracle", np.random.default_rng(0))
    s = ToySettings()
    assert w[SLOT_DIRECT_L] == w[SLOT_DIRECT_R] == pytest.approx(s.strong)
    assert w[SLOT_CONTRA_L] == w[SLOT_CONTRA_R] == pytest.approx(s.weak)
    assert s.strong > s.weak


def test_oracle_vector_is_mirror_invariant_wholesale():
    w = init_weights("oracle", np.random.default_rng(0))
    assert np.array_equal(w, w[MIRROR_PERM]), \
        f"oracle несимметричен: {list(w)} против {list(w[MIRROR_PERM])}"


def test_right_bias_is_not_mirror_invariant():
    """Opposite-test: тот же механизм проверки обязан ловить однобокий рефлекс."""
    w = init_weights("right_bias", np.random.default_rng(0))
    assert not np.array_equal(w, w[MIRROR_PERM])


def test_symmetric_regime_is_mirror_invariant_too():
    """Базовая проводка лечения тоже должна быть инвариантна — иначе H1 сравнивает
    симметричное обучение с асимметричным и вывод не принадлежит эксперименту."""
    w = init_weights("symmetric", np.random.default_rng(0))
    assert np.array_equal(w, w[MIRROR_PERM])


def test_unknown_regime_still_rejected():
    with pytest.raises(ValueError):
        init_weights("oracle_but_bias", np.random.default_rng(0))


# --- 2. fixed / non-plastic / reward-blind -------------------------------

def test_oracle_arm_spec_declares_frozen_and_reward_blind():
    spec = toy.ARM_SPECS[toy.ORACLE_ARM]
    assert spec["learner"] == "none", "oracle обязан быть не обучаемым"
    assert spec["init"] == "oracle"
    assert spec["reward_coupled"] is False, "oracle не должен видеть награду"


def test_oracle_not_added_to_p4_1_matrix():
    """P4.1 заморожен: если oracle попадёт в ARMS, задокументированная команда
    P4.1 начнёт печатать другую матрицу, и «повторить прошлый результат» станет
    неправдой."""
    assert toy.ORACLE_ARM not in toy.ARMS
    assert toy.ORACLE_ARM in toy.P4_2_ARMS
    assert set(toy.P4_2_ARMS) <= set(toy.ARMS) | {toy.ORACLE_ARM}
    # пластичные shuffled-варианты в P4.2 не нужны: они отвечают «может ли
    # обучение исправить старт», а P4.2 отвечает «чем обучение лучше фиксированного».
    assert "weight_shuffled" not in toy.P4_2_ARMS and "direction_shuffled" not in toy.P4_2_ARMS


def test_modulator_without_reward_coupling_returns_zero():
    m = make_modulator(reward_coupled=False, settings=ToySettings())
    assert m.compute(task_reward=10.0) == pytest.approx(0.0)
    assert m.compute(task_reward=-1.0) == pytest.approx(0.0)


def test_oracle_run_moves_no_weights():
    r = run_arm(toy.ORACLE_ARM, "right", PROBE_SEEDS[0], episodes=2, settings=_fast())
    assert np.allclose(r.dw_per_edge, 0.0, atol=1e-15)
    assert np.allclose(r.weights_final, r.weights_init, atol=1e-15)
    assert max(abs(float(e.mean_abs_dw)) for e in r.episodes) == pytest.approx(0.0)


# --- 3. oracle не пустой и не односторонний ------------------------------

def test_oracle_solves_both_mirrors_at_declared_operating_point(manifest):
    """Инструмент обязан работать до того, как по нему примут решение:
    замороженный control с нулевым успехом сравнивать не с чем."""
    settings = proto.build_settings(manifest)
    per_mirror = {}
    for task in ("right", "left"):
        rates = [run_arm(toy.ORACLE_ARM, task, seed, episodes=5, settings=settings).success_rate
                 for seed in PROBE_SEEDS]
        per_mirror[task] = rates
        assert all(x > 0.0 for x in rates), f"oracle не решает зеркало {task}: {rates}"
    gap = abs(np.mean(per_mirror["right"]) - np.mean(per_mirror["left"]))
    assert gap <= 0.3, f"oracle односторонний: right={per_mirror['right']} left={per_mirror['left']}"


def test_right_bias_is_one_sided_on_the_same_seeds(manifest):
    """Поведенческий контраст: right_bias решает правое зеркало и проваливает
    левое. Если бы он начал решать оба, «симметричный oracle» перестал бы быть
    различимой вещью и этот тест обязан был бы покраснеть первым."""
    settings = proto.build_settings(manifest)
    per_task = {}
    for task in ("right", "left"):
        agent = toy.ToyAgent(learner_kind="none", init_regime="right_bias", seed=PROBE_SEEDS[0],
                             settings=settings)
        per_task[task] = float(np.mean([agent.run_episode(task, ep).reached for ep in range(3)]))
    assert per_task["right"] > per_task["left"], f"right_bias перестал быть односторонним: {per_task}"


# --- 4. манифест: shape и cross-field ------------------------------------

def test_manifest_loads(manifest):
    assert manifest["protocol_id"] == "p4.2.oracle-baseline.v2"
    assert manifest["pre_registered"] is True
    assert "structural_effect" not in manifest["claims_allowed"]
    assert manifest["_digest"].startswith("sha256:")


def test_manifest_seed_sets_are_disjoint(manifest):
    conf = set(proto.confirmatory_seeds(manifest))
    probe = set(proto.probe_seeds(manifest))
    assert conf == set(range(120, 180))
    assert probe == set(range(900, 904))
    for other in manifest["seed_sets"]["disjoint_from"]:
        assert not conf & set(proto.expand_seed_range(other, other["id"]))
        assert not probe & set(proto.expand_seed_range(other, other["id"]))
    assert not conf & probe


def test_manifest_declares_six_arms_and_three_streams(manifest):
    assert len(manifest["arms"]) == 6
    assert manifest["task_streams"] == ["left_target", "right_target", "mixed"]
    roles = {a["role"] for a in manifest["arms"]}
    assert "upper_bound_characterization" in roles
    assert [a["gate"] for a in manifest["arms"] if a["role"] == "upper_bound_characterization"] == ["none"]


def test_digest_changes_on_one_number(manifest):
    """Единственное изменение в протоколе обязано менять digest — иначе
    «digest совпадает» ничего не гарантирует."""
    base = proto.digest_of_file(proto.DEFAULT_MANIFEST)
    tampered = copy.deepcopy(manifest)
    tampered.pop("_digest")
    tampered["operating_point"]["values"]["eta"] = 1.6
    tampered["operating_point"]["overrides"]["eta"] = 1.6
    assert proto.canonical_digest(tampered) != base


def test_silent_eta_shift_is_rejected(manifest):
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    bad["operating_point"]["values"]["eta"] = 1.6          # override не тронут
    with pytest.raises(proto.ProtocolError, match="eta"):
        proto.validate(bad)


def test_eta_override_must_be_declared_in_values(manifest):
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    bad["operating_point"]["overrides"]["eta"] = 1.4
    with pytest.raises(proto.ProtocolError, match="overrides.eta"):
        proto.validate(bad)


def test_declared_override_equal_to_default_is_rejected(manifest):
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    bad["operating_point"]["overrides"]["sigma"] = bad["operating_point"]["values"]["sigma"]
    with pytest.raises(proto.ProtocolError, match="равно дефолту"):
        proto.validate(bad)


def test_oracle_declared_plastic_is_rejected(manifest):
    """Манифест не может объявить oracle не Обучаемым, если код для него течёт
    plasticity: сверка с ARM_SPECS падает, а не предупреждает."""
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    for arm in bad["arms"]:
        if arm["arm_id"] == "oracle_reflex":
            arm["learner"] = "stdp"
    with pytest.raises(proto.ProtocolError, match="oracle_reflex"):
        proto.validate(bad)


def test_oracle_as_gate_is_rejected(manifest):
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    for arm in bad["arms"]:
        if arm["arm_id"] == "oracle_reflex":
            arm["gate"] = "required"
    with pytest.raises(proto.ProtocolError, match="gate=none"):
        proto.validate(bad)


def test_removing_the_oracle_arm_breaks_the_protocol(manifest):
    """P4.2 без upper bound не отвечает на свой вопрос — протокол это запрещает."""
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    bad["arms"] = [a for a in bad["arms"] if a["arm_id"] != "oracle_reflex"]
    bad["hypotheses"] = [h for h in bad["hypotheses"] if h["kind"] != "characterization"]
    with pytest.raises(proto.ProtocolError, match="oracle"):
        proto.validate(bad)


def test_wrong_seed_count_is_rejected(manifest):
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    bad["seed_sets"]["confirmatory"]["count"] = 30
    with pytest.raises(proto.ProtocolError, match="count"):
        proto.validate(bad)


def test_confirmatory_overlap_with_p4_1_is_rejected(manifest):
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    bad["seed_sets"]["confirmatory"]["range"] = [50, 109]
    bad["seed_sets"]["confirmatory"]["count"] = 60
    with pytest.raises(proto.ProtocolError, match="пересекаются"):
        proto.validate(bad)


def test_probe_seeds_inside_confirmatory_is_rejected(manifest):
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    bad["seed_sets"]["fixture_probe"]["range"] = [60, 63]
    with pytest.raises(proto.ProtocolError, match="fixture_probe"):
        proto.validate(bad)


def test_required_hypothesis_outside_family_is_rejected(manifest):
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    bad["statistical_test"]["family"] = ["H1", "H3", "H4"]     # H2 пропущена
    with pytest.raises(proto.ProtocolError, match="H2"):
        proto.validate(bad)


def test_characterization_inside_family_is_rejected(manifest):
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    bad["statistical_test"]["family"] = ["H1", "H2", "H3", "H4", "H5"]
    with pytest.raises(proto.ProtocolError, match="H5"):
        proto.validate(bad)


def test_episodes_must_match_across_the_manifest(manifest):
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    bad["episodes_per_seed"] = 40
    with pytest.raises(proto.ProtocolError, match="episodes_per_seed"):
        proto.validate(bad)


def test_run_budget_is_recounted(manifest):
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    bad["run_budget"]["cells"] = 480                            # посчитано для 8 arms
    with pytest.raises(proto.ProtocolError, match="run_budget"):
        proto.validate(bad)


def test_schema_rejects_missing_mixed_stream(manifest):
    """Shape-уровень: без mixed-потока первичная метрика неопределима, и это
    должно падать ещё до cross-field проверок."""
    from jsonschema import Draft202012Validator

    schema = json.loads(proto.SCHEMA_PATH.read_text(encoding="utf-8"))
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    bad["task_streams"] = ["left_target", "right_target"]
    assert list(Draft202012Validator(schema).iter_errors(bad))


def test_schema_rejects_nonzero_violation_budget(manifest):
    from jsonschema import Draft202012Validator

    schema = json.loads(proto.SCHEMA_PATH.read_text(encoding="utf-8"))
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    bad["provenance"]["violation_budget"] = 3
    assert list(Draft202012Validator(schema).iter_errors(bad))


def test_schema_rejects_episode_as_unit_of_analysis(manifest):
    """Единица статистики заперта в seed. «episode» или «provenance_entry»
    превратили бы 4800 эпизодов в 4800 репликаций — схема обязана это отклонить."""
    from jsonschema import Draft202012Validator

    schema = json.loads(proto.SCHEMA_PATH.read_text(encoding="utf-8"))
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    bad["statistical_test"]["unit_of_analysis"] = "episode"
    assert list(Draft202012Validator(schema).iter_errors(bad))


def test_schema_rejects_post_hoc_protocol(manifest):
    """pre_registered=false — это протокол, написанный после запуска."""
    from jsonschema import Draft202012Validator

    schema = json.loads(proto.SCHEMA_PATH.read_text(encoding="utf-8"))
    bad = copy.deepcopy(manifest)
    bad.pop("_digest")
    bad["pre_registered"] = False
    assert list(Draft202012Validator(schema).iter_errors(bad))


def _write_pair(tmp_path, digest, partial=False, arm="plastic"):
    """Один shard прогона как он лежит на диске: CSV + sidecar с этим именем."""
    csv_path = tmp_path / f"p4_2_{arm}.csv"
    csv_path.write_text("arm,task,seed\n", encoding="utf-8")
    side = tmp_path / (csv_path.name + ".meta.json")
    side.write_text(json.dumps({"manifest_digest": digest, "partial": partial}),
                    encoding="utf-8")
    return csv_path, side


def test_sidecar_bound_by_name_not_by_order(tmp_path):
    digest = FROZEN
    a = _write_pair(tmp_path, digest, arm="a")
    b = _write_pair(tmp_path, digest, arm="b")
    csvs = [str(a[0]), str(b[0])]
    sides = [str(b[1]), str(a[1])]          # намеренно в другом порядке
    got = gate.check_sidecars(csvs, sides, digest)
    assert sorted(name for name, _ in got) == sorted(p.name for p in (a[1], b[1]))


def test_missing_sidecar_is_refused(tmp_path):
    # Прогон без sidecar нельзя отличить от прогона по правленному протоколу.
    digest = FROZEN
    a, _ = _write_pair(tmp_path, digest, arm="a")
    b, _ = _write_pair(tmp_path, digest, arm="b")
    with pytest.raises(gate.Refused, match="не хватает sidecar"):
        gate.check_sidecars([str(a), str(b)], [str(a.with_suffix(a.suffix + ".meta.json"))], digest)


def test_named_but_absent_sidecar_is_refused(tmp_path):
    """Отсутствующий sidecar — отказ, а не трейсбек.

    Сверка имён сравнивает множества, поэтому имя, которое просто не существует на
    диске, проходит её незаметно. Дальше шёл json.loads(read_text()) на отсутствующем
    пути: FileNotFoundError, rc=1 и трейсбек — то есть «проверки не было» выводилось
    как «прогон не прошёл». На этот путь наткнулся бы разбор прерванного P4.2.v1:
    ни CSV, ни sidecar'а у него на диске нет.
    """
    csv_path = tmp_path / "p4_2_a.csv"
    csv_path.write_text("arm,task,seed\n", encoding="utf-8")
    absent = tmp_path / "p4_2_a.csv.meta.json"
    assert not absent.exists()
    with pytest.raises(gate.Refused, match="отсутствует на диске"):
        gate.check_sidecars([str(csv_path)], [str(absent)], FROZEN)


def test_gate_cli_exits_2_for_absent_named_sidecar(tmp_path):
    """Тот же случай на уровне CLI: коды возврата — часть контракта."""
    csv_path = tmp_path / "p4_2_a.csv"
    csv_path.write_text("arm,task,seed\n", encoding="utf-8")
    rc, out = _run_cli("check_p4_2_oracle_gate.py", "--manifest", proto.DEFAULT_MANIFEST,
                       "--csv", csv_path, "--sidecar", tmp_path / "p4_2_a.csv.meta.json")
    assert rc == gate.REFUSE == 2, out[-500:]
    assert "НЕ СЧИТАЕТСЯ" in out, out[-500:]
    assert "отсутствует на диске" in out, out[-500:]
    assert "Traceback" not in out, out[-500:]


def test_partial_sidecar_is_refused(tmp_path):
    digest = FROZEN
    csv_path, side = _write_pair(tmp_path, digest, partial=True)
    with pytest.raises(gate.Refused, match="partial"):
        gate.check_sidecars([str(csv_path)], [str(side)], digest)


def test_stale_digest_sidecar_is_refused(tmp_path):
    """Прогон по старому протоколу: гейт заморожен, sidecar нет.

    Замороженный digest здесь — «текущий», а в sidecar лежит другой: ветку
    «по другому протоколу» надо отличать от ветки «гейт идёт не по протоколу».
    """
    csv_path, side = _write_pair(tmp_path, "sha256:" + "b" * 64)
    with pytest.raises(gate.Refused, match="другому протоколу"):
        gate.check_sidecars([str(csv_path)], [str(side)], FROZEN)


def test_orphan_sidecar_is_refused(tmp_path):
    digest = FROZEN
    csv_path, side = _write_pair(tmp_path, digest)
    other = tmp_path / "p4_2_gone.csv.meta.json"
    other.write_text(json.dumps({"manifest_digest": digest, "partial": False}), encoding="utf-8")
    with pytest.raises(gate.Refused, match="без своего CSV"):
        gate.check_sidecars([str(csv_path)], [str(side), str(other)], digest)


# --- 4b. якорь pre-registration: digest заморожен в коде, а не в файле ----
#
# До P4.2a гейт сверял digest файла прогона с digest'ом манифеста, который сам
# же из этого файла и загрузил. Две величины из одних байт совпадают всегда,
# поэтому подмена манифеста (`--manifest /tmp/edited.json`) проходила и runner,
# и gate, не оставив следа: sidecar был согласован с подделкой. Якорь обязан
# жить вне проверяемого файла — тогда ловится любая правка, включая ту, которая
# остаётся валидной по схеме.

def _edited_manifest(tmp_path, mutate, name="edited.json"):
    """Копия замороженного манифеста с одной точечной правкой.

    Правки подбираются так, чтобы схема и cross-field проверки их НЕ отклоняли:
    иначе тест доказывал бы формат, а не силу якоря.
    """
    m = copy.deepcopy(proto.load())
    m.pop("_digest")
    mutate(m)
    p = tmp_path / name
    p.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    assert proto.digest_of_file(p) != FROZEN
    return p


def _shift_eta(m):
    m["operating_point"]["values"]["eta"] = 1.6
    m["operating_point"]["overrides"]["eta"] = 1.6


def _widen_seeds(m):
    """Confirmatory-множество расширено вдвое; run_budget пересчитан, чтобы
    правка осталась валидной — ловить её должен только якорь.

    Расширение идёт вправо от объявленного диапазона: новая половина обязана
    остаться disjoint со всеми сожжёнными множествами, иначе правку ловил бы
    cross-field, а не якорь, и тест доказывал бы не то.
    """
    m["seed_sets"]["confirmatory"]["range"] = [120, 239]
    m["seed_sets"]["confirmatory"]["count"] = 120
    rb = m.get("run_budget")
    if rb:
        cells = len(m["arms"]) * 120 * len(m["task_streams"])
        rb["cells"] = cells
        rb["episodes"] = cells * m["episodes_per_seed"]


def _reword_hypothesis(m):
    m["hypotheses"][0]["statement"] += " (переформулировано после заморозки)"


EDITS = [_shift_eta, _widen_seeds, _reword_hypothesis]
EDIT_IDS = ["eta=1.6", "seeds-120-239", "текст-гипотезы"]


def _run_cli(script, *args):
    """Вызов через CLI, потому что коды возврата — часть контракта: 2 значит
    «проверки не было», 1 — «не прошло». Прямой вызов функций этого не ловит."""
    res = subprocess.run([sys.executable, str(REPO / "scripts" / script), *map(str, args)],
                         capture_output=True, text=True, timeout=120)
    return res.returncode, res.stdout + res.stderr


def test_frozen_digest_is_pinned_in_code_not_derived_from_the_file():
    """Якорь сверяется с литералом, а не с самим собой через реестр.

    Литерал ниже пережил бы случайную правку в FROZEN_PROTOCOLS, потому что
    выводится из байтов заморозки, а не из кода. v1-якорь закреплён вторым
    литералом: смена версии не должна иметь возможности незаметно стереть
    историю прерванного протокола.
    """
    assert proto.digest_of_file(proto.DEFAULT_MANIFEST) == FROZEN
    assert FROZEN == ("sha256:5f5cae42a2f0373933ead1c61307eac97f6a864f25b0927"
                      "dbda4643e94d75123")
    assert proto.FROZEN_PROTOCOLS["p4.2.oracle-baseline.v1"]["digest"] == (
        "sha256:6e343c298c5367ad1cc713db8a8d4e1f981467432a65361"
        "20286305476de6f62")


def test_require_frozen_accepts_the_pin_and_refuses_everything_else():
    assert proto.require_frozen(FROZEN, where="тест") == FROZEN
    for bogus in ("sha256:" + "f" * 64, "", FROZEN[:-4] + "aaaa"):
        with pytest.raises(proto.ProtocolError, match="не совпадает с замороженным"):
            proto.require_frozen(bogus, where="тест")


@pytest.mark.parametrize("mutate", EDITS, ids=EDIT_IDS)
def test_the_edit_survives_the_schema_so_digest_is_the_only_defence(mutate, tmp_path):
    """Без этого теста три отказа ниже бесполезны: если бы правку ломала сама
    схема, отказ ничего не доказывал бы про якорь."""
    m = proto.load(_edited_manifest(tmp_path, mutate))
    assert m["_digest"] != FROZEN


@pytest.mark.parametrize("mutate", EDITS, ids=EDIT_IDS)
def test_gate_refuses_edited_manifest_even_when_sidecar_agrees(mutate, tmp_path):
    """Ровно та атака, которую принимал старый гейт: подменённый манифест и
    sidecar, согласованный с подменой."""
    edited = proto.digest_of_file(_edited_manifest(tmp_path, mutate))
    a, b = (_write_pair(tmp_path, edited, arm=x) for x in ("a", "b"))
    with pytest.raises(gate.Refused, match="не по замороженному протоколу"):
        gate.check_sidecars([str(a[0]), str(b[0])], [str(a[1]), str(b[1])], edited)


def test_gate_still_accepts_the_frozen_pair(tmp_path):
    """Позитивный контроль: якорь не должен превращаться в «отказывать всегда»."""
    a, b = (_write_pair(tmp_path, FROZEN, arm=x) for x in ("a", "b"))
    got = gate.check_sidecars([str(a[0]), str(b[0])], [str(a[1]), str(b[1])], FROZEN)
    assert sorted(name for name, _ in got) == sorted(x[1].name for x in (a, b))


@pytest.mark.parametrize("mutate", EDITS, ids=EDIT_IDS)
def test_gate_cli_exits_2_for_edited_manifest(mutate, tmp_path):
    p = _edited_manifest(tmp_path, mutate)
    missing = tmp_path / "nope.csv"
    rc, out = _run_cli("check_p4_2_oracle_gate.py", "--manifest", p, "--csv", missing,
                       "--sidecar", str(missing) + ".meta.json")
    assert rc == gate.REFUSE, out[-500:]
    assert "НЕ СЧИТАЕТСЯ" in out, out[-500:]
def test_original_manifest_is_accepted_by_runner_cli(tmp_path):
    p = tmp_path / "frozen_copy.json"
    p.write_bytes(proto.DEFAULT_MANIFEST.read_bytes())
    assert proto.digest_of_file(p) == FROZEN
    rc, out = _run_cli("run_p4_2_oracle.py", "--manifest", p, "--describe")
    assert rc == 0, out[-500:]
    assert "NON-CONFIRMATORY" not in out, out[-500:]


def test_non_confirmatory_sidecar_cannot_pass_the_gate(tmp_path):
    a = _write_pair(tmp_path, FROZEN, arm="a")
    b = _write_pair(tmp_path, FROZEN, arm="b")
    b[1].write_text(json.dumps({"manifest_digest": FROZEN, "partial": False,
                                "non_confirmatory": True}), encoding="utf-8")
    with pytest.raises(gate.Refused, match="non_confirmatory"):
        gate.check_sidecars([str(a[0]), str(b[0])], [str(a[1]), str(b[1])], FROZEN)


# --- 4b. runner: якорь проверяется до первого смоделированного эпизода -----

def test_freeze_guard_accepts_frozen_without_the_flag():
    assert runner.freeze_guard({"_digest": FROZEN}, False, str(runner.DEFAULT_OUT)) is False


def test_freeze_guard_refuses_unfrozen_without_the_flag():
    with pytest.raises(runner.Refused, match="не равен замороженному"):
        runner.freeze_guard({"_digest": "sha256:" + "1" * 64}, False, str(runner.DEFAULT_OUT))


def test_freeze_guard_refuses_the_flag_on_frozen_manifest():
    """Флаг не должен становиться ритуальной кнопкой: по нему нельзя судить о
    прогоне, если его жмут «на всякий случай»."""
    with pytest.raises(runner.Refused, match="не нужен"):
        runner.freeze_guard({"_digest": FROZEN}, True, "var/debug.csv")


def test_freeze_guard_refuses_experimental_in_the_confirmatory_path():
    with pytest.raises(runner.Refused, match="confirmatory путь"):
        runner.freeze_guard({"_digest": "sha256:" + "1" * 64}, True, str(runner.DEFAULT_OUT))


def test_freeze_guard_allows_experimental_into_a_separate_out():
    assert runner.freeze_guard({"_digest": "sha256:" + "1" * 64}, True, "var/expl.csv") is True


@pytest.mark.parametrize("mutate", EDITS, ids=EDIT_IDS)
def test_runner_cli_exits_2_for_edited_manifest(mutate, tmp_path):
    rc, out = _run_cli("run_p4_2_oracle.py", "--manifest", _edited_manifest(tmp_path, mutate),
                       "--describe")
    assert rc == runner.REFUSE, out[-500:]
    assert "ЗАПУСК ОТКЛОНЁН" in out, out[-500:]


def test_runner_cli_experimental_writes_elsewhere_and_announces_it(tmp_path):
    p = _edited_manifest(tmp_path, _shift_eta, name="eta.json")
    out_csv = tmp_path / "expl.csv"
    rc, out = _run_cli("run_p4_2_oracle.py", "--manifest", p, "--experimental-manifest",
                       "--describe", "--out", out_csv)
    assert rc == 0, out[-500:]
    assert "NON-CONFIRMATORY" in out, out[-500:]

def test_the_old_self_comparison_could_not_catch_the_swap(tmp_path):
    """Противотест: ровно эта подмена удовлетворяла OLD-проверке.

    Старый гейт сравнивал digest файла с `_digest` манифеста, загруженного из
    этого же файла, — совпадение гарантировано по построению. Если это перестанет
    быть правдой, изменилась канонизация digest, и вся секция 4b проверяет уже
    другое.
    """
    p = _edited_manifest(tmp_path, _shift_eta)
    loaded = proto.load(p)
    assert proto.digest_of_file(p) == loaded["_digest"]
    with pytest.raises(proto.ProtocolError, match="не совпадает с замороженным"):
        proto.require_frozen(proto.digest_of_file(p), where="тест")


def test_confirmatory_sidecar_would_carry_the_frozen_digest(manifest):
    """Цепочка, которую пишет runner: sidecar.manifest_digest = manifest["_digest"].

    Для замороженного файла это ровно якорь, значит confirmatory-sidecar обязан
    нести FROZEN_PROTOCOL_DIGEST. Сам запись файла покрывает experimental-тест
    ниже: она гоняет настоящую запись файла, а не её пересказ.
    """
    assert manifest["_digest"] == proto.digest_of_file(proto.DEFAULT_MANIFEST) == FROZEN


def _one_episode_experimental_manifest(tmp_path, name="expl.json"):
    """Манифест-малютка: 1 seed × 1 episode. Настоящий прогон, который успевает
    закончиться внутри теста, — на нём можно проверять и успех раннера, и его
    отказ, что на замороженном бюджете в 86 400 эпизодов недоступно.

    Seed'ом отладки взят 500: он вне confirmatory-множества 120–179, вне
    fixture-проб 900–903 и вне всех сожжённых диапазонов — иначе отладочный
    прогон сам начал бы засчитываться как наблюдение, и cross-field отклонял
    бы манифест до того, как до него доедет раннер.
    """
    m = copy.deepcopy(proto.load())
    m.pop("_digest")
    _widen_seeds(m)
    m["seed_sets"]["confirmatory"]["range"] = [DEBUG_SEED, DEBUG_SEED]
    m["seed_sets"]["confirmatory"]["count"] = 1
    m["episodes_per_seed"] = 1
    # episodes живёт в двух местах сразу: episodes_per_seed обязан совпасть с
    # operating_point.values.episodes, а отклонение от дефолта кода должно быть
    # объявлено в overrides с обоснованием. Иначе отказ раннера был бы про формат,
    # а не про якорь.
    op = m["operating_point"]
    op["values"]["episodes"] = 1
    op["overrides"]["episodes"] = 1
    op["override_rationale"]["episodes"] = "1 эпизод: проверка записи, не прогон"
    m["run_budget"]["cells"] = len(m["arms"]) * 1 * len(m["task_streams"])
    m["run_budget"]["episodes"] = m["run_budget"]["cells"] * 1
    p = tmp_path / name
    p.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def test_experimental_run_writes_sidecar_the_gate_refuses(tmp_path):
    """Настоящая запись sidecar: один эпизод, non_confirmatory попадает в файл прогона."""
    p = _one_episode_experimental_manifest(tmp_path)
    out_csv = tmp_path / "expl.csv"
    rc, log = _run_cli("run_p4_2_oracle.py", "--manifest", p, "--experimental-manifest",
                       "--seeds", str(DEBUG_SEED), "--arms", "oracle_reflex", "--out", out_csv,
                       "--provenance", tmp_path / "expl.prov.jsonl")
    assert rc == 0, log[-800:]
    side = json.loads((tmp_path / "expl.csv.meta.json").read_text(encoding="utf-8"))
    assert side["non_confirmatory"] is True
    assert side["frozen_protocol_digest"] == FROZEN
    assert side["manifest_digest"] != FROZEN
    with pytest.raises(gate.Refused, match="не по замороженному протоколу"):
        gate.check_sidecars([str(out_csv)], [str(tmp_path / "expl.csv.meta.json")],
                            side["manifest_digest"])
    # Порядок проверок: сначала якорь, потом содержимое sidecar. Если гейт вызван
    # по замороженному протоколу, а в sidecar лежит другой digest, срабатывает
    # именно эта ветка — метка non_confirmatory приходит следующей (её собственный
    # случай покрыт test_non_confirmatory_sidecar_cannot_pass_the_gate).
    with pytest.raises(gate.Refused, match="другому протоколу"):
        gate.check_sidecars([str(out_csv)], [str(tmp_path / "expl.csv.meta.json")], FROZEN)


def test_runner_refuses_a_provenance_log_it_cannot_trust(tmp_path):
    """Журнал без witness — отказ до первого эпизода, и продолжение только со следом.

    Для confirmatory-прогона это разница между «потратили час на отказ» и
    «потратили 40 часов на отказ». Раннер обязан встать на той же проверке,
    на которой встанет писатель, и пустить прогон дальше только после явного
    recover_head(), которое записывает adoption в саму цепь.
    """
    from fly_connectome_agent.src.engineering.logging.provenance_log import ProvenanceLog
    m = _one_episode_experimental_manifest(tmp_path)
    prov_path = tmp_path / "expl.prov.jsonl"
    seed_log = ProvenanceLog(str(prov_path))
    seed_log.append({"kind": "осталось от другого прогона"})
    seed_log.append({"kind": "осталось от другого прогона"})
    Path(seed_log.head_path).unlink()      # журнал без witness — состояние abort-архива
    before = prov_path.read_bytes()
    args = ("run_p4_2_oracle.py", "--manifest", m, "--experimental-manifest",
            "--seeds", str(DEBUG_SEED), "--arms", "oracle_reflex",
            "--out", tmp_path / "expl.csv", "--provenance", prov_path)

    rc, out = _run_cli(*args)
    assert rc == runner.REFUSE, out[-800:]      # «прогона не было», а не «не прошло»
    assert "recover_head" in out, out[-800:]
    assert prov_path.read_bytes() == before, "отказ не должен дописывать ни байта"

    head = seed_log.recover_head(reason="тест: adoption после потери witness")
    lines = [json.loads(line) for line in prov_path.read_text(encoding="utf-8").splitlines()]
    assert lines[2]["payload"]["provenance_head_recovered"]["adopted_entries"] == 2
    assert head["entry_count"] == 3            # два принятых + маркер уже в цепи
    assert ProvenanceLog(str(prov_path)).verify_chain() is True

    rc, out = _run_cli(*args)
    assert rc == 0, out[-800:]
    after = [json.loads(line) for line in prov_path.read_text(encoding="utf-8").splitlines()]
    assert after[:3] == lines[:3], "прогон не должен переписывать ни старые записи, ни маркер"
    assert len(after) > 3, "прогон ничего не дописал — тест пустой"
    assert after[3]["previous_hash"] == lines[2]["entry_hash"], \
        "прогон продолжается за маркером, а не в обход него"
    assert ProvenanceLog(str(prov_path)).verify_chain() is True


# --- 5. арифметика гейта --------------------------------------------------

def _row(arm: str, seed: int, min_half: float, dw: float = 0.0, stream: str = "mixed",
         viol: int = 0, right: float | None = None, left: float | None = None) -> dict:
    return {
        "arm": arm, "task": stream, "seed": seed,
        "success_mirror_min": min_half,
        "success_mirror[right]": min_half if right is None else right,
        "success_mirror[left]": min_half if left is None else left,
        "success_rate": min_half, "cumulative_reward": 10.0 * min_half,
        "steps_to_target": 8.0, "mean_abs_dw": dw,
        "fraction_weights_at_bound": 0.0, "governance_violations": viol,
    }


def _full_matrix(rows_manifest: dict, value_of) -> list[dict]:
    seeds = proto.confirmatory_seeds(rows_manifest)
    codes = proto.code_arms(rows_manifest)
    streams = proto.task_streams(rows_manifest)
    return [_row(codes[a["arm_id"]], s, value_of(a, s), dw=0.01 if a["role"] == "treatment" else 0.0,
                 stream=st)
            for a in rows_manifest["arms"] for st in streams for s in seeds]


def test_holm_matches_known_values():
    # m=3: шаг-down, монотонно. 0.01 -> 3*0.01=0.03; 0.03 -> 2*0.03=0.06;
    # 0.04 -> 1*0.04=0.04, но Holm не даёт уменьшаться -> 0.06.
    assert gate.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert gate.holm([0.5, 0.5]) == pytest.approx([1.0, 1.0])


def test_sign_comparison_counts_and_discloses_ties():
    a = {i: float(i % 3) for i in range(9)}
    b = {i: 1.0 for i in range(9)}
    stat = gate.sign_comparison(a, b)
    assert stat["n_seeds_paired"] == 9
    assert stat["ties"] == 3                      # i%3 == 1 при i = 1,4,7
    assert stat["wins"] + stat["losses"] + stat["ties"] == 9
    assert stat["n_informative"] == stat["wins"] + stat["losses"]


def test_coverage_gate_accepts_full_matrix(manifest):
    rows = _full_matrix(manifest, lambda a, s: 0.5)
    gate.coverage_gate(rows, manifest, proto.code_arms(manifest), proto.task_streams(manifest))


def test_missing_run_csv_is_a_refusal_not_a_traceback(manifest, tmp_path):
    # «Проверки не было» обязано отличаться от «не прошло»: голый FileNotFoundError
    # даёт код 1, тот же, что у непройденного гейта. Отказ = Refused = exit 2.
    with pytest.raises(gate.Refused, match="нет на диске"):
        gate.load_rows([str(tmp_path / "nope.csv")], proto.code_arms(manifest))


def test_coverage_gate_refuses_missing_seed(manifest):
    codes = proto.code_arms(manifest)
    rows = _full_matrix(manifest, lambda a, s: 0.5)
    seeds = set(proto.confirmatory_seeds(manifest))
    doomed = max(seeds)
    rows = [r for r in rows if r["seed"] != doomed]
    with pytest.raises(gate.Refused, match="неполный"):
        gate.coverage_gate(rows, manifest, codes, proto.task_streams(manifest))


def test_coverage_gate_refuses_foreign_seed(manifest):
    """Чужой seed — ровно на единицу меньше множества: 60 и 900 сюда принести
    уже нельзя, они объявлены сожжёнными, и отказ был бы про другой причине."""
    codes = proto.code_arms(manifest)
    rows = _full_matrix(manifest, lambda a, s: 0.5)
    foreign = min(proto.confirmatory_seeds(manifest)) - 1
    rows.append(_row(codes["rstdp_mixed"], foreign, 0.5))
    with pytest.raises(gate.Refused, match="вне множества"):
        gate.coverage_gate(rows, manifest, codes, proto.task_streams(manifest))


def test_integrity_gate_refuses_unfrozen_frozen_arm(manifest):
    codes = proto.code_arms(manifest)
    rows = _full_matrix(manifest, lambda a, s: 0.5)
    for r in rows:
        if r["arm"] == codes["no_plasticity"]:
            r["mean_abs_dw"] = 0.7          # «замороженный» контроль потёк
    with pytest.raises(gate.Refused, match="no_plasticity"):
        gate.integrity_gates(rows, manifest, codes)


def test_integrity_gate_refuses_governance_violation(manifest):
    codes = proto.code_arms(manifest)
    rows = _full_matrix(manifest, lambda a, s: 0.5)
    rows[0]["governance_violations"] = 1
    with pytest.raises(gate.Refused, match="governance"):
        gate.integrity_gates(rows, manifest, codes)


def test_integrity_gate_refuses_a_treatment_that_did_not_move(manifest):
    """Если веса лечения не сдвинулись, сравнивать «обучение» не с чем:
    прогон обязан быть отвергнут, а не вердиктнут."""
    codes = proto.code_arms(manifest)
    rows = _full_matrix(manifest, lambda a, s: 0.5)
    for r in rows:
        if r["arm"] == codes["rstdp_mixed"]:
            r["mean_abs_dw"] = 0.0
    with pytest.raises(gate.Refused, match="не сдвинулись"):
        gate.integrity_gates(rows, manifest, codes)


def test_m_zero_matching_no_plasticity_is_an_invariant_not_a_coincidence(manifest):
    codes = proto.code_arms(manifest)
    rows = _full_matrix(manifest, lambda a, s: 0.5)
    for r in rows:
        if r["arm"] == codes["m_zero"]:
            r["success_mirror_min"] = 0.9
    with pytest.raises(gate.Refused, match="m_zero"):
        gate.integrity_gates(rows, manifest, codes)


def _comparisons(manifest, rows) -> list[dict]:
    """Тот же путь, что и в main(): sign test по первичной метрике + Holm по family."""
    codes = proto.code_arms(manifest)
    family = manifest["statistical_test"]["family"]
    results = []
    for h in manifest["hypotheses"]:
        t = gate.vector_by_seed(rows, codes[h["treatment"]], "mixed", "success_mirror_min")
        c = gate.vector_by_seed(rows, codes[h["comparator"]], "mixed", "success_mirror_min")
        stat = gate.sign_comparison(t, c)
        stat.update({"hypothesis": h["id"], "kind": h["kind"], "in_family": h["id"] in family,
                     "treatment": h["treatment"], "comparator": h["comparator"],
                     "direction": h["direction"]})
        results.append(stat)
    adj = gate.holm([r["raw_p"] for r in results if r["in_family"]])
    k = 0
    for r in results:
        if r["in_family"]:
            r["adjusted_p"] = adj[k]
            k += 1
    return results


def test_classify_losing_to_oracle_is_not_a_failure(manifest):
    """Проигрыш oracle — публикуемый результат, а не провал гейта: required
    гипотезы пройдены, решение = «идти в P5 как в исследование механизма»."""
    rows = _full_matrix(manifest, lambda a, s: 0.9 if a["arm_id"] == "oracle_reflex" else
                        (0.7 if a["role"] == "treatment" else 0.2))
    decision = gate.classify(manifest, _comparisons(manifest, rows), {})
    assert decision["key"] == "beats_weak_but_below_oracle"
    assert "P5" in decision["text"]
    assert decision["inputs"]["frozen_control_above_treatment"] is False


def test_classify_beats_everything_including_oracle(manifest):
    rows = _full_matrix(manifest, lambda a, s: 0.8 if a["role"] == "treatment" else
                        (0.7 if a["arm_id"] == "oracle_reflex" else 0.2))
    decision = gate.classify(manifest, _comparisons(manifest, rows), {})
    assert decision["key"] == "beats_all_and_comparable_to_oracle"


def test_classify_frozen_control_above_treatment_is_the_worst_row(manifest):
    """Четвёртая строка таблицы обязана достигаться: если случайно-асимметричная
    заморозка стабильнее обучения и oracle выше — это диагноз credit assignment
    (P4.3), а не «идём к коннектому»."""
    rows = _full_matrix(manifest, lambda a, s: {
        "oracle_reflex": 0.9, "rstdp_mixed": 0.3, "weight_shuffled_frozen": 0.6,
        "direction_shuffled_frozen": 0.2, "no_plasticity": 0.1, "m_zero": 0.1,
    }[a["arm_id"]])
    decision = gate.classify(manifest, _comparisons(manifest, rows), {})
    assert decision["inputs"]["frozen_control_above_treatment"] is True
    assert decision["key"] == "oracle_and_frozen_stably_above_rstdp"
    assert "P4.3" in decision["text"]


def test_classify_failed_required_gate_forbids_connectome(manifest):
    codes = proto.code_arms(manifest)
    rows = _full_matrix(manifest, lambda a, s: 0.3 if a["role"] == "treatment" else 0.6)
    decision = gate.classify(manifest, _comparisons(manifest, rows), {})
    assert decision["key"] == "fails_h1_or_h2"
