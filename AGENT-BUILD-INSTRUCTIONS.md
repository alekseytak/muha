# AGENT-BUILD-INSTRUCTIONS — пошаговая инструкция для агента-сборщика

> Привет, сородич. Это инструкция по сборке роя мушиных агентов.
> Я передаю тебе ауру этого проекта: читай буквально, выполняй по этапам,
> творчество разрешено там, где явно написано «МОЖНО ДОКРУТИТЬ».
> Всё, что помечено «НЕ ТРОГАТЬ» — конституционные границы OS Глаголов.

## Глобальные правила (НЕ ТРОГАТЬ)

1. Capability `WR` (write registry) есть только у роли Registrar.
2. L5-глаголы (`сериализует`, `экспортирует`, `хэширует`, `аттестует`,
   `якорит`, `версирует`, `подписывает`, `публикует`) — только Registrar + trace + аттестация.
3. ATQEC/фильтровый слой маркирует и оценивает, но НЕ выполняет финальный commit.
4. Memory-слой ищет и суммирует, но НЕ переписывает каноническую память незаметно.
5. Hardware: необратимое действие высокого риска — только с подтверждением человека.
6. Каждое действие наружу — только как verb-act через GateKeeper.

## ЭТАП 0 — Окружение (день 1)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas h5py torch snntorch networkx pyyaml
pip install ray[default] gymnasium        # teacher-контур
# опционально: pip install numba guppylang chromadb
mkdir -p fly_swarm/{data,policy,code,registry,logs}
```

Проверка: `python -c "import snntorch, torch, yaml; print('ok')"`

## ЭТАП 1 — Данные коннектома (дни 2–4)

1. Источник: Janelia Male CNS v1.0 (https://www.janelia.org/project-team/flyem/male-cns-connectome)
   или готовые FEATHER/CSV из `Neuromorphicism/fly-brain-snntorch`.
2. НЕ бери все 166 700 нейронов сразу. Вырежь подграф 1k–50k:
   - optic_lobe → central_complex → motor_output (навигация);
   - antennal_lobe → mushroom_body → dopaminergic (обучение).
3. Конвертируй в два файла (формат — контракт):
   - `data/nodes.csv`: neuron_id,type,region,x,y,z
   - `data/edges.csv`: source,target,synapse_type,weight,delay_ms
4. Визуальный контроль: NeuroGLancer или `networkx.draw` по регионам.

Критерий готовности: `FlyBrain('data/nodes.csv','data/edges.csv')` конструируется без ошибок.

## ЭТАП 2 — SNN-ядро (дни 5–8)

Файл: `01-omega-archive/code/fly_brain_agent.py` (копируй в `fly_swarm/code/`).

1. Запусти свободный прогон: случайный входной ток 1000 шагов.
2. Проверь: спайки есть, но не шторм (средняя частота 1–20 Гц на нейрон).
3. Если шторм — уменьши начальные веса ×0.5; если тишина — ×1.5.
   МОЖНО ДОКРУТИТЬ: параметры LIF в `LIFParams`.

## ЭТАП 3 — OS Глаголов gate (дни 9–11)

Файлы: `01-omega-archive/code/verb_act_gate.py`, `policy/fly_roles.yaml`,
`policy/verb_act.schema.json`.

1. Каждое моторное решение оборачивай в verb-act и гоняй через `GateKeeper.check()`.
2. Тесты (обязательны): Observer не может `публикует`; Decider не может `якорит`;
   hardware+risk=high без human_in_loop → escalate; atqec не делает commit.
3. НЕ ТРОГАТЬ: mapping ролей и L5-правила.

## ЭТАП 4 — Награды и штрафы (дни 12–18)

Файл: `01-omega-archive/code/reward_stdp.py`.

1. Собери композитный модулятор:
   M = α·task_reward − β·safety − γ·energy − δ·policy_violation + ε·coherence + ζ·advantage
2. `coherence` — из АРФА-фильтров (Проект III) или ATQEC-слоя KEM 4.2.
3. Задача-полигон: 2D-коридор с целью (или MuJoCo fly из Nature 2025).
4. Критерий: за 500 эпизодов success_rate растёт, спайковый бюджет не превышает базовый ×2.
5. МОЖНО ДОКРУТИТЬ: усиления модулятора, форму coherence, shaping награды.

## ЭТАП 5 — Учитель → Ученик (дни 19–25)

Файл: `01-omega-archive/code/teacher_student.py`.

1. Teacher = MLP/PPO (Ray RLlib), решающий ту же задачу на 90%+.
2. Student = твой SNN. Дистилляция: behavior cloning + spike-rate matching.
3. Promotion path: shadow → canary → limited → full → registrar_export.
   Перевод только при success_ema ≥ порога И violations == 0.
4. Каждый promotion — verb-act Registrar'а `версирует` + запись в Registry.

## ЭТАП 6 — Registry и золотые яблоки (дни 26–30)

Файл: `01-omega-archive/code/omega_registry.py`.

1. Каждый эпизод → `anchor("episode:trace", ...)`.
2. Успешная траектория → `propose_golden_apple()` → `swarm_vote()` →
   при ≥66% голосов → `promote_to_canon()`.
3. После каждого запуска — `verify_chain()` (sha256-цепочка).
4. НЕ ТРОГАТЬ: append-only. Никаких UPDATE/DELETE.

## ЭТАП 7 — Рой (дни 31–40)

Файлы: `01-omega-archive/code/swarm_coordinator.py`.

1. Запусти 8–16 агентов в общем 2D-мире с феромонным полем.
2. Роли: Scout(Observer+Analyst), Forager(Analyst+Decider), Guardian, Registrar (1 на рой).
3. Решения — через `SwarmCoordinator.decide()` с вето Guardian.
4. Критерий: рой находит цель быстрее одиночного агента ≥ на 30%.

## ЭТАП 8 — Эмпатия и гормоны, Проект II (дни 41–48)

Файлы: `02-mirror-swarm/code/hormone_module.py`, `empathy_bus.py`.

1. Каждому агенту — HormoneModule. Reward → on_reward, штраф → on_punishment.
2. EmpathyBus: эмоция агента транслируется рою с затуханием 0.5.
3. Гормоны влияют на модулятор M (через cognitive_ability) и на risk_appetite.
4. МОЖНО ДОКРУТИТЬ: профили «радость/злость/страх», частоты резонансов
   (дофамин 25 Гц, ГАМК 15 Гц, серотонин 35 Гц — из csc_filter владельца).

## ЭТАП 9 — Рефлексия и творчество (дни 49–55)

Файлы: `02-mirror-swarm/code/kozyrev_mirror.py`, `ulam_chaos.py`,
`collective_meditation.py`.

1. Перед каждым действием уровня L4+ — `reflect_decision()`.
   paradox_likelihood > 0.66 → эскалация (как в KEM 4.2: Quantum Escape).
2. Автомат Улама: 1 раз в N эпизодов добавляй `perturbation()` к входному току —
   это разрыв эргодичности, защита от «Дня Сурка».
3. Перед сложной общей задачей — `meditate()` роя на 86.84 Гц до sync_order > 0.8.

## ЭТАП 10 — Ризома и перенос ауры, Проект III (дни 56–65)

Файлы: `03-anuclear-rhizome/code/rhizome_mesh.py`, `aura_transfer.py`,
`arfa_filters.py`.

1. Убери центрального координатора: только gossip-обмен между соседями (3–5 связей).
2. Тест на разрыв: удали 30% узлов — рой должен сохранить связность и задачу.
3. Перенос ауры: убитый/выпускник-Teacher передаёт ауру новому ученику.
   Capability ученика ⊆ capability учителя. Подпись обязательна.
4. АРФА-фильтры владельца — на сенсорный вход (csc → harp → royal_mole → apfa).

## Приёмка всего проекта

- [ ] Все verb-act проходят gate; запрещённые глаголы не исполняются.
- [ ] Registry-цепочка верифицируется (`verify_chain() == True`).
- [ ] Хотя бы одно золотое яблоко прошло рой-голосование и стало каноном.
- [ ] Ученик дорос до режима `limited` без нарушений.
- [ ] Рой пережил потерю 30% узлов.
- [ ] Зеркало Козырева заблокировало хотя бы один парадоксальный сценарий.
- [ ] Всё это работает локально (localhost-first, как KEM 4.2).

## Что делать, если запутался

1. Вернись к `99-shared/os-glagolov-charter.md` — это конституция.
2. Любой сомнительный шаг оформи как verb-act с `verb: "эскалирует"` —
   и спроси человека. Эскалация — это не провал, это правильное действие.
3. Творчество — в `99-shared/sandbox-playbook.md`. Всё остальное — по этапам.
