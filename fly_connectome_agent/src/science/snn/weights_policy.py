"""Политика подписанных весов: структурный граф -> безопасные для LIF веса.

Мотивация (волна 2, реальные данные FlyWire): у ConnectomeGraph при
weight=None веса откатываются на сырые synapse_counts (models.py) — все
положительные, вплоть до 2405 на одно ребро. Напрямую в SparseLIF такие
веса дают гарантированный runaway и теряют знак E/I.

Честный контракт:
- ЗНАК — из данных: edge_class ("excitatory"/"inhibitory"), проставленный
  ингестом по колонке Excitatory аннотаций пресинаптического нейрона.
- МАГНИТУДА — модельное допущение: min(synapse_count, count_saturation)
  * w_syn_per_count. Обе константы — инженерные, НЕ свойства измерений;
  согласованы с диапазонами пластичности (STDPParams: exc [0, w_exc_max],
  inh [-w_inh_max, 0]), чтобы policy-веса могли служить стартом для STDP.
- Ребро без неизвестного/отсутствующего edge_class — fail-fast, как и
  неизвестный нейротрансмиттер в plasticity_sparse.

Уровень утверждений: engineering assumption, не биологический факт.
"""
from __future__ import annotations

import numpy as np

from ..graph.models import ConnectomeGraph

_EXC_CLASS = "excitatory"
_INH_CLASS = "inhibitory"

# Инженерные константы по умолчанию (модельные допущения, не данные):
# 20 синапсов * 0.05 = 1.0 — верхняя планка совпадает с w_exc_max/w_inh_max
# из STDPParams, т.е. policy никогда не выдаёт веса за границы пластичности.
DEFAULT_W_SYN_PER_COUNT = 0.05
DEFAULT_COUNT_SATURATION = 20.0


def signed_structural_weights(
    graph: ConnectomeGraph,
    *,
    w_syn_per_count: float = DEFAULT_W_SYN_PER_COUNT,
    count_saturation: float = DEFAULT_COUNT_SATURATION,
    w_exc_max: float = 1.0,
    w_inh_max: float = 1.0,
) -> np.ndarray:
    """Вернуть подписанные веса (E,) для SNN из структурного графа.

    знак(edge) = +1 если edge_class == "excitatory", -1 если "inhibitory";
    |w| = min(synapse_count, count_saturation) * w_syn_per_count,
    затем дополнительно clipped до w_exc_max / w_inh_max.

    Чистая функция: graph не мутируется. Неизвестный или отсутствующий
    edge_class -> ValueError (fail-fast, как unknown NT в пластичности).
    """
    if w_syn_per_count < 0:
        raise ValueError("w_syn_per_count must be non-negative")
    if count_saturation < 0:
        raise ValueError("count_saturation must be non-negative")
    if w_exc_max < 0 or w_inh_max < 0:
        raise ValueError("weight bounds must be non-negative")

    edge_classes = graph.edge_classes
    magnitude = np.minimum(graph.synapse_counts, count_saturation) * w_syn_per_count

    normalized = np.array(
        [(str(cls).strip().lower() if cls is not None else "") for cls in edge_classes],
        dtype=object,
    )
    exc_mask = normalized == _EXC_CLASS
    inh_mask = normalized == _INH_CLASS
    unknown_mask = ~(exc_mask | inh_mask)
    if np.any(unknown_mask):
        bad = np.nonzero(unknown_mask)[0]
        shown = ", ".join(
            f"edge {int(i)}={edge_classes[i]!r}" for i in bad[:5]
        )
        raise ValueError(
            f"неизвестный edge_class: ребра не могут получить знак ({shown}); "
            f"ожиданы 'excitatory'/'inhibitory'"
        )

    sign = np.where(exc_mask, 1.0, -1.0)
    weights = sign * magnitude
    weights[exc_mask] = np.clip(weights[exc_mask], 0.0, w_exc_max)
    weights[inh_mask] = np.clip(weights[inh_mask], -w_inh_max, 0.0)
    return weights


def graph_with_signed_weights(
    graph: ConnectomeGraph,
    *,
    w_syn_per_count: float = DEFAULT_W_SYN_PER_COUNT,
    count_saturation: float = DEFAULT_COUNT_SATURATION,
    w_exc_max: float = 1.0,
    w_inh_max: float = 1.0,
) -> ConnectomeGraph:
    """Тот же граф, но weights = подписанные веса политики.

    Исходный graph не мутируется; edges/nodes/manifest сохраняются.
    """
    weights = signed_structural_weights(
        graph,
        w_syn_per_count=w_syn_per_count,
        count_saturation=count_saturation,
        w_exc_max=w_exc_max,
        w_inh_max=w_inh_max,
    )
    return graph.copy_with_edges(graph.edges, weights=weights)
