"""Тесты кредита на уровне нейрона (P4.4, руки `neuron` и `neuron_shuffled`).

Правило: третья компонента M перестаёт быть одной на пул и становится одной на
постсинаптический нейрон. Вес нейрона — его доля в массе eligibility окна:
`E_i = сумма |elig| по пластичным рёбрам в i`. Фактор строится по рангу, а не по
сырой массе: сырая масса тяжелохвостая, ранг ограничен и не требует подбора
порога.

    s_i = 1 − λ + 2λ·rank_i/(m − 1),   rank по (E_i, id) по возрастанию

Набор активных нейронов (`E_i > 0`) получает среднее ровно 1, поэтому средний
фактор равен 1, а не срезается, как в v3. `neuron_shuffled` — нулевая рука:
тот же набор факторов переставлен между теми же нейронами детерминированным
ГПСЧ; разница `neuron` − `neuron_shuffled` показывает, важна ли привязка к
участию, а не просто неоднородность кредита.

Среднее 1 по нейронам не равно сохранению суммы кредита: фактор умножается на
массу, поэтому `neuron` доставляет больше `|dw|`, чем v1 (в v4 это и оказалось
конфаундом). Руки `neuron_renorm` и `neuron_renorm_shuffled` делят факторы на
`rho = Σ s_i·E_i / Σ E_i`, то есть держат доставленный кредит равным v1 шаг в
шаг; различие остаётся только в распределении.

Проверяется:

- λ = 0 даёт бит-в-бит веса v1 (иначе сравнение смешивается с посторонней
  правкой);
- среднее фактора по активным нейронам равно 1, границы [1−λ, 1+λ] соблюдены;
- фактор монотонен по массе eligibility и не трогает нейроны с нулевой массой;
- перестановка сохраняет набор факторов и детерминирована по (seed, эпизод, шаг);
- dw рёбер равен `eta·M·s_post·elig` — фактор применяется к посту, а не к ребру;
- перенормировка держит `Σ |M|·|elig|` равным v1, а без неё сумма растёт;
- руки кредита и руки контраста не пересекаются.
"""
import numpy as np
import pytest

from fly_connectome_agent.src.engineering.harness.snn_corridor import (
    ARMS,
    NEURON_ARMS,
    Channel,
    CorridorAgent,
    LoopSettings,
)

NEURONS = 9
IN_LEFT = (0, 1)
IN_RIGHT = (2, 3)
OUT_LEFT = (4, 5)
OUT_RIGHT = (6, 7)


def make_agent(arm: str, **overrides) -> CorridorAgent:
    from fly_connectome_agent.tests.test_pool_modulator import make_graph

    settings = LoopSettings(
        episodes=1, window_steps=20, i_drive=60.0, gain=6.0, sigma=4.0,
        eta=0.02, max_episode_steps=40, width=5, height=5,
        start=(1, 1), target=(3, 3), seed=11, **overrides,
    )
    channel = Channel(inputs_left=IN_LEFT, inputs_right=IN_RIGHT,
                      outputs_left=OUT_LEFT, outputs_right=OUT_RIGHT)
    return CorridorAgent(graph=make_graph(), channel=channel,
                         settings=settings, arm=arm)


def mass_pattern(agent: CorridorAgent) -> np.ndarray:
    """Ровно по одному активному нейрону с разной массой и один нулевой.

    Пост-нейроны 4, 5, 6 получают массу 4, 8, 12 (elig 1, 2, 3), нейроны 7 и 8
    остаются с нулём. Так ранг известен заранее: 4 < 5 < 6, а 7 и 8 не в игре.
    """
    post = agent.learner.edge_index[:, 1]
    elig = np.ones(agent.learner.n_plastic)
    elig[post == 5] = 2.0
    elig[post == 6] = 3.0
    elig[post == 7] = 0.0
    elig[post == 8] = 0.0
    agent.learner.eligibility = elig
    return post


def test_rezhim_kredita_obyavlen_dlya_kazhdoy_ruki():
    assert NEURON_ARMS == {
        "neuron": "rank",
        "neuron_shuffled": "shuffled",
        "neuron_renorm": "rank_renorm",
        "neuron_renorm_shuffled": "shuffled_renorm",
    }
    for arm, mode in NEURON_ARMS.items():
        assert arm in ARMS
        assert make_agent(arm, neuron_credit=0.5).neuron_credit_mode == mode
    assert make_agent("plastic").neuron_credit_mode == "none"


def test_lamda_nol_daet_bityo_v_bit_vesa_v1():
    v1 = make_agent("plastic")
    v1.run(episodes=1)

    for arm in NEURON_ARMS:
        agent = make_agent(arm, neuron_credit=0.0)
        agent.run(episodes=1)
        assert np.array_equal(v1.learner.get_weights(), agent.learner.get_weights()), arm


def test_faktor_ravnomeren_po_aktivnym_neyronam():
    agent = make_agent("neuron", neuron_credit=0.5)
    post = mass_pattern(agent)

    M = agent._neuron_modulator(4.0, episode=0, step=0)

    assert M is not None
    by_neuron = {n: float(M[post == n][0]) for n in (4, 5, 6, 7, 8)}
    assert by_neuron[4] == pytest.approx(4.0 * 0.5)
    assert by_neuron[5] == pytest.approx(4.0 * 1.0)
    assert by_neuron[6] == pytest.approx(4.0 * 1.5)
    assert by_neuron[7] == pytest.approx(4.0)
    assert by_neuron[8] == pytest.approx(4.0)

    factors = np.array([by_neuron[n] / 4.0 for n in (4, 5, 6)])
    assert factors.mean() == pytest.approx(1.0, abs=1e-12)
    assert factors.min() == pytest.approx(1.0 - 0.5)
    assert factors.max() == pytest.approx(1.0 + 0.5)


def test_faktor_monotonen_po_masse_i_ne_menyaet_znak():
    agent = make_agent("neuron", neuron_credit=0.5)
    post = mass_pattern(agent)

    for value in (3.0, -2.0):
        M = agent._neuron_modulator(value, episode=3, step=7)
        assert np.all(np.sign(M) * np.sign(value) >= 0)
        assert abs(M[post == 4][0]) < abs(M[post == 5][0]) < abs(M[post == 6][0])


def test_perestanovka_sohranyaet_nabor_faktorov_i_determinirovana():
    ranked = make_agent("neuron", neuron_credit=0.5)
    shuffled = make_agent("neuron_shuffled", neuron_credit=0.5)
    post_ranked = mass_pattern(ranked)
    post_shuffled = mass_pattern(shuffled)

    assert np.allclose(post_ranked, post_shuffled)
    M1 = shuffled._neuron_modulator(4.0, episode=2, step=5)
    M2 = shuffled._neuron_modulator(4.0, episode=2, step=5)
    assert np.array_equal(M1, M2)

    ranked_factors = np.sort([ranked._neuron_modulator(4.0, 0, 0)[post_ranked == n][0]
                              for n in (4, 5, 6)])
    seen_difference = False
    for episode in range(6):
        M = shuffled._neuron_modulator(4.0, episode=episode, step=0)
        factors = np.sort([M[post_shuffled == n][0] for n in (4, 5, 6)])
        assert np.allclose(factors, ranked_factors)
        if not np.allclose([M[post_shuffled == n][0] for n in (4, 5, 6)],
                           [ranked._neuron_modulator(4.0, 0, 0)[post_ranked == n][0]
                            for n in (4, 5, 6)]):
            seen_difference = True
    assert seen_difference, "перестановка ни разу не отличалась от ранговой — нулевая рука пуста"


def test_dw_raven_eta_umnozhit_M_na_faktor_na_eligibility():
    agent = make_agent("neuron", neuron_credit=0.5)
    mass_pattern(agent)
    elig = agent.learner.eligibility.copy()
    # Веса уводим с границы: иначе sign-constraints срежут положительный dw и
    # проверка формулы превратится в проверку клипа.
    agent.learner.set_weights(np.full(agent.learner.n_plastic, 0.5))
    weights_before = agent.learner.get_weights().copy()

    M = agent._neuron_modulator(2.0, episode=1, step=1)
    dw = agent.learner.apply_modulator(M)

    assert np.allclose(dw, agent.settings.eta * M * elig)
    assert np.allclose(agent.learner.get_weights() - weights_before, dw)


def test_perenormirovka_derzhit_kredit_ravnym_v1():
    for arm in ("neuron_renorm", "neuron_renorm_shuffled"):
        agent = make_agent(arm, neuron_credit=0.5)
        mass_pattern(agent)
        elig = np.abs(agent.learner.eligibility)

        M = agent._neuron_modulator(2.0, episode=1, step=1)

        assert M is not None
        assert float(np.sum(np.abs(M) * elig)) == pytest.approx(
            2.0 * float(np.sum(elig)), rel=1e-12)


def test_bez_perenormirovki_kredit_usilen():
    """Пин конфаунда v4: среднее 1 по нейронам всё равно даёт больше |dw|.

    На шаблоне массы (4, 8, 12) факторы (0.5, 1, 1.5) дают
    0.5·4 + 1·8 + 1.5·12 = 28 против 24 у v1.
    """
    agent = make_agent("neuron", neuron_credit=0.5)
    mass_pattern(agent)
    elig = np.abs(agent.learner.eligibility)

    M = agent._neuron_modulator(2.0, episode=1, step=1)

    assert float(np.sum(np.abs(M) * elig)) == pytest.approx(2.0 * 28.0)


def test_perenormirovka_sohranyaet_poryadok_i_nol_massu():
    agent = make_agent("neuron_renorm", neuron_credit=0.5)
    post = mass_pattern(agent)

    M = agent._neuron_modulator(4.0, episode=0, step=0)

    by_neuron = {n: float(M[post == n][0]) for n in (4, 5, 6, 7, 8)}
    assert by_neuron[4] < by_neuron[5] < by_neuron[6]
    assert by_neuron[7] == pytest.approx(4.0)
    assert by_neuron[8] == pytest.approx(4.0)
    # После деления на rho нижняя граница уходит ниже 1−λ — это заявлено, не баг.
    assert by_neuron[4] / 4.0 < 0.5


def test_ruki_kredita_i_kontrasta_ne_peresekayutsya():
    for arm in NEURON_ARMS:
        agent = make_agent(arm, neuron_credit=0.5, pool_contrast=0.5)
        assert agent._edge_modulator(4.0, "move_left") is None
    for arm in ("pool", "pool_pos", "pool_neg"):
        agent = make_agent(arm, neuron_credit=0.5, pool_contrast=0.5)
        assert agent._neuron_modulator(4.0, 0, 0) is None


def test_lamda_ogranichena_diapazonom():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            LoopSettings(neuron_credit=bad)


def test_vyrozhdennyy_sluchay_odnogo_aktivnogo_neyrona():
    agent = make_agent("neuron", neuron_credit=0.5)
    post = agent.learner.edge_index[:, 1]
    elig = np.zeros(agent.learner.n_plastic)
    elig[post == 4] = 1.0
    agent.learner.eligibility = elig

    assert agent._neuron_modulator(4.0, episode=0, step=0) is None
