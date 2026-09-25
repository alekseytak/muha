# RFC-001 · Проект I «Архив Омеги» — архитектура

- Статус: DRAFT v1.0 · Домены: memory / registry / atqec / hardware
- Конституция: OS-Glagolov-Charter.md · Словарь: verbs.json

## 1. Мотивация

Рой агентов с мушиными мозгами, который НЕ забывает. Каждое действие —
след в append-only Registry; успешные траектории становятся «золотыми
яблоки» (Брэдбери) — каноническими прецедентами (коллективное
бессознательное, Юнг), из которых стартуют следующие ученики.
Цель — не надзор, а история становления и движение к Омеге
(Тейяр де Шарден).

## 2. Архитектура

```
sensor stream
    │
    ▼
┌─────────────┐   spikes    ┌──────────────────┐
│ FlyBrain SNN│───────────▶ │ motor decode     │
│ (подграф ЦНС│             │ rates → vector   │
│  дрозофилы) │             └────────┬─────────┘
└──────▲──────┘                      │ verb-act
       │ neuromodulator M            ▼
┌──────┴──────┐             ┌──────────────────┐
│ reward_stdp │             │ OS Glagolov Gate │
│ (3-factor)  │             │ allow/deny/      │
└─────────────┘             │ escalate/attest  │
       ▲                    └────────┬─────────┘
       │ curriculum/reward           ▼
┌──────┴──────┐             action / block / escalate
│  Teacher    │                      │
│ (System 2)  │                      ▼
└─────────────┘             ┌────────────────────────┐
                            │ OmegaRegistry          │
                            │ append-only, sha256    │
                            │ golden apples + votes  │
                            └────────────────────────┘
```

## 3. Компоненты

| Модуль | Файл | Роль OS Глаголов | Уровень |
|--------|------|------------------|---------|
| SNN-ядро | code/fly_brain_agent.py | Observer/Analyst | L0–L1 |
| Пластичность | code/reward_stdp.py | (внутренний, WL) | L1 |
| Gate | code/verb_act_gate.py | Guardian/Decider | L3–L4 |
| Registry | code/omega_registry.py | Registrar | L5 |
| Рой | code/swarm_coordinator.py | Synthesizer | L2–L3 |
| Учитель/Ученик | code/teacher_student.py | Decider | L3 |

## 4. Контракты данных

- `nodes.csv`: neuron_id,type,region,x,y,z
- `edges.csv`: source,target,synapse_type,weight,delay_ms
- verb-act: см. policy/verb_act.schema.json
- Registry entry: {kind, subject, ts, payload, prev, hash}

## 5. Управление (governance)

- Ни один агент роя не имеет WR.
- Golden apple проходит: proposal → swarm_vote (≥66%) → promote_to_canon.
- Все L5-записи имеют trace и подлежат аттестации.

## 6. Метрики приёмки

- success_rate ученика растёт монотонно на окне 100 эпизодов;
- спайковый бюджет ≤ 2× базового;
- 0 исполненных запрещённых глаголов;
- verify_chain() == True после 1000 записей;
- ≥1 канонический прецедент, использованный новым учеником.

## 7. Вехи

M1 (нед. 2): SNN-ядро на подграфе. M2 (нед. 4): gate + verb-act.
M3 (нед. 6): обучение наградами. M4 (нед. 8): teacher-student.
M5 (нед. 10): registry + golden apples. M6 (нед. 12): рой 16 агентов.

## 8. Риски

- runaway plasticity → гомеостаз + клиппинг (в reward_stdp);
- плохая конвенция роя → diversity + вето Guardian + аудит Teacher'ом;
- переполнение Registry → архивирование L2 `архивирует` (не удаление!).
