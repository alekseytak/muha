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
вне confirmatory 60-119 и вне обоих множеств P4.1, так что ни один тест не
засчитывается как наблюдение.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import p4_2_protocol as proto  # noqa: E402
import check_p4_2_oracle_gate as gate  # noqa: E402

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

PROBE_SEEDS = [900, 901]  # вне всех гейтов


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
    assert manifest["protocol_id"] == "p4.2.oracle-baseline.v1"
    assert manifest["pre_registered"] is True
    assert "structural_effect" not in manifest["claims_allowed"]
    assert manifest["_digest"].startswith("sha256:")


def test_manifest_seed_sets_are_disjoint(manifest):
    conf = set(proto.confirmatory_seeds(manifest))
    probe = set(proto.probe_seeds(manifest))
    assert conf == set(range(60, 120))
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
    digest = "sha256:" + "a" * 64
    a = _write_pair(tmp_path, digest, arm="a")
    b = _write_pair(tmp_path, digest, arm="b")
    csvs = [str(a[0]), str(b[0])]
    sides = [str(b[1]), str(a[1])]          # намеренно в другом порядке
    got = gate.check_sidecars(csvs, sides, digest)
    assert sorted(name for name, _ in got) == sorted(p.name for p in (a[1], b[1]))


def test_missing_sidecar_is_refused(tmp_path):
    # Прогон без sidecar нельзя отличить от прогона по правленному протоколу.
    digest = "sha256:" + "a" * 64
    a, _ = _write_pair(tmp_path, digest, arm="a")
    b, _ = _write_pair(tmp_path, digest, arm="b")
    with pytest.raises(gate.Refused, match="не хватает sidecar"):
        gate.check_sidecars([str(a), str(b)], [str(a.with_suffix(a.suffix + ".meta.json"))], digest)


def test_partial_sidecar_is_refused(tmp_path):
    digest = "sha256:" + "a" * 64
    csv_path, side = _write_pair(tmp_path, digest, partial=True)
    with pytest.raises(gate.Refused, match="partial"):
        gate.check_sidecars([str(csv_path)], [str(side)], digest)


def test_stale_digest_sidecar_is_refused(tmp_path):
    csv_path, side = _write_pair(tmp_path, "sha256:" + "a" * 64)
    with pytest.raises(gate.Refused, match="другому протоколу"):
        gate.check_sidecars([str(csv_path)], [str(side)], "sha256:" + "b" * 64)


def test_orphan_sidecar_is_refused(tmp_path):
    digest = "sha256:" + "a" * 64
    csv_path, side = _write_pair(tmp_path, digest)
    other = tmp_path / "p4_2_gone.csv.meta.json"
    other.write_text(json.dumps({"manifest_digest": digest, "partial": False}), encoding="utf-8")
    with pytest.raises(gate.Refused, match="без своего CSV"):
        gate.check_sidecars([str(csv_path)], [str(side), str(other)], digest)


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
    codes = proto.code_arms(manifest)
    rows = _full_matrix(manifest, lambda a, s: 0.5)
    rows.append(_row(codes["rstdp_mixed"], 120, 0.5))
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
