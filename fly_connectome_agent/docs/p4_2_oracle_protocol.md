# P4.2 — Oracle baseline: pre-registered protocol

**Статус: ЗАМОРОЖЕН И НЕ ЗАПУЩЕН.** Ни одного confirmatory эпизода не симулировано на момент
написания этого документа. Единственный прогон, который уже сделан, — `fixture probe` на seed'ах
900–903 (раздел «Fixture probe» ниже); он объявлен вне всех гейтов и не входит ни в одну статистику.

**digest манифеста:** `sha256:6e343c298c53…` — полное значение печатает
`.venv/bin/python scripts/p4_2_protocol.py`, сверяется гейтом. Правка любого числа в
`manifests/p4_2_oracle_confirmatory.json` меняет digest, а гейт откажет старому прогону
(`sidecar.manifest_digest != digest файла`). Изменить протокол после запуска можно только так:
новый `protocol_id` (`…v2`), новый манифест, новое множество seed — и в отчёте это будет второй
эксперимент, а не «уточнённый первый».

---

## Вопрос

> Даёт ли reward-modulated STDP преимущество над заранее правильно заданной, но не обучающейся
> проводкой на тех же задачах и при том же бюджете эпизодов?

Это **не** вопрос «может ли сеть учиться» — на него ответил P4.1 (`docs/p4_validation.md`).
P4.2 отвечает на вопрос о величине эффекта относительно фиксированного контроллера, который
уже знает оба ответа.

## Arms

| arm_id | code_arm | пластичность | старт | награда в M | роль | гейт |
|---|---|---|---|---|---|---|
| `rstdp_mixed` | `plastic` | да | симметричный слабый | да | treatment | required/desirable |
| `no_plasticity` | `no_plasticity` | нет | тот же симметричный | да | topology control | required |
| `m_zero` | `m_zero` | eligibility, M≡0 | тот же симметричный | нет | reward control | required |
| `weight_shuffled_frozen` | `weight_shuffled_frozen` | нет | случайная перестановка | да | structural control | desirable |
| `direction_shuffled_frozen` | `direction_shuffled_frozen` | нет | анти-рефлекс | да | direction control | desirable |
| `oracle_reflex` | `oracle_reflex` | нет | **оба прямых пути сильные** | нет | upper bound | **none** |

`oracle_reflex` = `S_L→C_L` и `S_R→C_R` по `strong=70`, оба контралатеральных по `weak=12`,
остальные рёбра как у всех. Это симметричная версия режима `right_bias`.

Три свойства oracle обеспечены не текстом, а кодом и тестами
(`fly_connectome_agent/tests/test_oracle_baseline.py`):

| свойство | как держится |
|---|---|
| symmetric | вектор весов инвариантен перестановке слотов лево↔право `S_L→C_L↔S_R→C_R, S_L→C_R↔S_R→C_L, S_L→INH↔S_R→INH, INH→C_L↔INH→C_R, C_L→C_R↔C_R→C_L`. Тот же тест на `right_bias` **обязан** падать — иначе проверка ничего не значит. |
| fixed | `learner="none"` → `NoPlasticityBaseline`, и гейт отвергает прогон, если у любого объявленного замороженным arm max `|Δw| > 1e-12`. |
| reward-blind | `reward_coupled=False` → профиль `no_modulator`, `M ≡ 0` даже при `task_reward=10`. Манифест не может объявить oracle непластичным, если код для него течёт пластичность: cross-field проверка сравнивает декларацию с `ARM_SPECS`. |

## Гипотезы и статистика

| id | сравнение | метрика | kind | в поправке |
|---|---|---|---|---|
| H1 | `rstdp_mixed` > `no_plasticity` | mixed min-half | required | да |
| H2 | `rstdp_mixed` > `m_zero` | mixed min-half | required | да |
| H3 | `rstdp_mixed` > `weight_shuffled_frozen` | mixed min-half | desirable | да |
| H4 | `rstdp_mixed` > `direction_shuffled_frozen` | mixed min-half | desirable | да |
| H5 | `rstdp_mixed` vs `oracle_reflex` | mixed min-half | **characterization** | **нет** |

- **Единица наблюдения — seed.** Прогон даёт 60 наблюдений на сравнение, а не 4800 эпизодов
  и не 24 000 записей provenance. Ни эпизод, ни строка лога не увеличивают n.
- **Первичная метрика** — `success_mirror_min`: для одного seed минимум из доли успешных
  эпизодов на правом и на левом зеркале mixed-потока. На одиночной задаче случайно
  асимметричной заморозке достаточно угадать одно из двух направлений; на mixed с обоими
  половинами угадать нельзя, и только там критерий «лучше всех baseline» фальсифицируем.
- **Тест** — парный exact sign test (binomtest, двусторонний), пары по одинаковому seed,
  одинаковому потоку и одинаковому бюджету.
- **Ties** исключаются из знаменателя и **обязательно объявляются**: гейт печатает `W/L/T`
  и отдельно список гипотез с ненулевым числом ties. `n_informative = W + L`.
- **Поправка** — Holm по семейству из 4 проверок (H1–H4). Число проверок и состав семьи
  зафиксированы до запуска. H5 в семействе не участвует: у сравнения с oracle нет порога
  прохода, поэтому корректировать нечего.
- **Проход required-гипотезы** — `wins > losses` **И** `Holm-скорректированная p < 0.05`.
  Большинство без значимости получает отдельный ярлык «НЕ ПРОЙДЕНА», а не «почти прошла».
- **Отчёт** — на каждую проверку: wins, losses, ties, n_informative, raw p, adjusted p,
  разность средних, и выгрузка raw per-seed исходов (`--per-seed-out`).

## Seed-множества

| множество | seeds | n | использование |
|---|---|---|---|
| `p4_2_confirmatory` | 60–119 | 60 | все гейты |
| `p4_2_fixture_probe` | 900–903 | 4 | только проверка инструмента, вне статистики |
| `p4_1_exploratory` | 0–19 | 20 | запрещены (уже использованы) |
| `p4_1_confirmatory` | 20–59 | 40 | запрещены (уже использованы) |

Несечение проверяется дважды: cross-field валидатором по объявленным диапазонам и гейтом по
фактическим seed'ам в чужих CSV (`--prior-csv var/p4_mixed80_a.csv var/p4_conf60_a.csv …`).

## Операционная точка

Берётся из дефолтов `ToySettings`, плюс ровно два заранее объявленных override:

| ключ | значение | источник |
|---|---|---|
| `eta` | 1.5 | override (P4.1 confirmatory считался при 1.5) |
| `episodes` | 80 | override (бюджет, на котором mixed-гейт P4.1 стал разрешимым) |
| `window_steps` | 60 | дефолт кода |
| `sigma` | 3.5 | дефолт кода |
| `sym_weight` / `strong` / `weak` | 45 / 70 / 12 | дефолт кода |
| геометрия | `track_length=17, cell_distance=6, edge_margin=2, max_steps=25` | дефолт кода |
| награда | `step_cost=0, target_reward=10, boundary_penalty=-1` | дефолт кода |
| `tau_eligibility` / `command_inh_weight` | 80 / −40 | дефолт кода |
| `plastic_slots` | `[0,1,2,3]` | дефолт кода |

Третьего способа сдвинуть параметр нет: значение, не равное дефолту и не перечисленное в
`overrides`, — это отказ валидации.

## Fixture probe (seed'ы 900–903, 10 эпизодов; в статистику не входит)

Прогон `.venv/bin/python var/p4_2_fixture_probe.py`:

| arm | right | left | mixed min-half |
|---|---|---|---|
| `oracle_reflex` | 1.00 | 1.00 | **1.00** |
| `plastic` (R-STDP) | 0.525 | 0.300 | 0.100 |
| `no_plasticity` | 0.150 | 0.200 | 0.050 |
| `right_bias` (контроль однобокости) | 1.00 | 0.10 | — |

Что это уже говорит до запуска — и что поэтому объявляется заранее:

1. **Oracle на потолке.** Фиксированная симметричная проводка решает оба зеркала почти
   идеально. Ожидаемый исход H5 — `rstdp_mixed` заметно **ниже** oracle. По заранее
   объявленной таблице это строка «beats_weak_but_below_oracle», а не провал:
   R-STDP учится из контингентной награды, но на тривиальной симметричной задаче не
   превосходит hand-specified политику.
2. **Инструмент не сломан:** oracle решает (min-half > 0), `right_bias` под тем же тестом
   рассыпается на левом зеркале — значит, проверка различает симметричный и однобокий рефлекс.
3. **10 эпизодов ≠ 80.** У `plastic` на коротком бюджете min-half 0.10, на 80 эпизодах P4.1
   было 0.51–0.53. Probe ничего не предсказывает для гейта и не используется как аргумент.

## Отказ считать (exit 2), а не «не прошло» (exit 1)

Гейт **не выносит** научного вердикта, если:

- гейт вызван по манифесту, чей digest не равен `FROZEN_PROTOCOL_DIGEST` из `scripts/p4_2_protocol.py` — якорь живёт в коде, а не в проверяемом файле (см. «Якорь pre-registration» ниже);
- sidecar прогона помечен `non_confirmatory` (прогон делался по не-замороженному манифесту через `--experimental-manifest`);
- sidecar помечен `partial` (подрезка seed или arms);
- в матрице не хватает хотя бы одной ячейки `arm × stream × seed` или есть лишняя;
- в прогоне есть seed вне 60–119 или пересекающийся с множествами P4.1;
- `governance violations > 0` (бюджет 0);
- объявленный замороженным arm дал `|Δw| > 0`;
- у treatment-arms веса не сдвинулись ни на одном seed (сравнивать «обучение» не с чем);
- `m_zero` поведенчески ≠ `no_plasticity` (тогда H2 меряет не «нет награды», а отдельный режим);
- файла прогона нет на диске (это `Refused`, а не `FileNotFoundError`: у трейсбека
  код 1, неотличимый от «не прошло»);
- у какого-то shard-CSV нет его sidecar — `--sidecar` обязателен и связывается с CSV
  **по имени** (`<csv>.meta.json`), а не по порядку в командной строке: прогон без
  sidecar неотличим от прогона по манифесту, изменённому задним числом;
- sidecar лишний (на пропущенный шард) или помечен `partial`.

Отдельные коды возврата нужны затем, чтобы «проверки не было» нельзя было прочитать как
«проверка не прошла».

## Якорь pre-registration: digest заморожен в коде (P4.2a)

До P4.2a гейт сверял digest файла манифеста с `_digest` того же манифеста, который
сам из этого файла и загрузил. Две величины, посчитанные от одних байт, совпадают
всегда, поэтому подмена работала:

```bash
.venv/bin/python scripts/run_p4_2_oracle.py --manifest /tmp/edited_protocol.json ...
.venv/bin/python scripts/check_p4_2_oracle_gate.py --manifest /tmp/edited_protocol.json ...
```

Runner записывал в sidecar digest подменённого манифеста, гейт пересчитывал digest
того же подменённого манифеста — и принимал прогон. Заморозка, которую можно
переопределить аргументом командной строки, не заморозка.

Сейчас якорь — константа `FROZEN_PROTOCOL_DIGEST` в `scripts/p4_2_protocol.py`, и
она обязана совпасть с digest'ом того единственного манифеста, по которому идёт
confirmatory-прогон:

- `scripts/check_p4_2_oracle_gate.py` refuse'ит до чтения любых CSV (`require_frozen`),
  а `check_sidecars` дополнительно требует `manifest_digest ==` этого же якоря;
- `scripts/run_p4_2_oracle.py` refuse'ит до первого смоделированного эпизода
  (`freeze_guard`), а в sidecar и provenance пишет `manifest_digest`,
  `frozen_protocol_digest` и `non_confirmatory`;
- ловится не только правка чисел: правка **формулировки гипотезы**, не портящая
  схему, тоже меняет digest и тоже отклоняется (`test_..._текст-гипотезы`).

`--manifest` сохранён, но не как лазейка: другой манифест принимается только вместе
с `--experimental-manifest`, который (а) запрещает писать в confirmatory путь
`var/p4_2_oracle.csv` и (б) помечает прогон `non_confirmatory=true` — такой CSV гейт
не примет никогда. На замороженном манифесте этот флаг тоже отказывает: кнопка,
которую жмут «на всякий случай», перестаёт значить что-либо.

Digest: `sha256:6e343c298c5367ad1cc713db8a8d4e1f981467432a6536120286305476de6f62`
(canonical JSON; raw SHA-256 файла `1bfb03db…` — другой, тоже легитимный якорь, но
сравнивается в коде именно canonical).

## Решение после P4.2 (объявлено до запуска)

| исход | решение |
|---|---|
| required пройдены, desirable пройдены, `≈ oracle` | P5 (загрузка реального коннектома) с осторожным оптимизмом |
| required пройдены, но ниже oracle | P5 допускается как исследование механизма обучения, **не** как доказательство превосходства |
| не пройдена H1 или H2 | к реальному коннектому не идти; чинить toy-задачу и learning rule |
| oracle **и** случайно-асимметричная заморозка стабильнее | сначала P4.3 (усложнить credit assignment), масштабирование запрещено |

Маппинг исхода в строку делает функция `classify()` в `scripts/check_p4_2_oracle_gate.py`, а
не человек: таблица в манифесте и код — один и тот же порядок.

## Что перепроверено по сырым данным до запуска P4.2

Три пункта ревью, закрытые не текстом отчёта, а пересчётом:

1. **P3.1 по Git.** На HEAD `3b65f99` (закоммичено) `gatekeeper.py` уже содержит
   `ALLOW → enforced_action=verb_act.proposed_action`, а `DENY`/`ESCALATE` —
   `SAFE_FALLBACK = "stay"`; `move_right` в файле встречается ровно один раз, в
   множестве `_VALID_ACTIONS`. Подмена была в **тесте** (`test_allow_passes_proposed_move_right`:
   предложенное и навязанное оба `move_right`, проверка не различала их). Сейчас
   тесты используют асимметричные предложения (`move_left`) и подделку решений
   (`PolicyDecision(status="ALLOW", …, enforced_action="stay")` обязан не
   конструироваться), плюс on-disk tamper provenance (`TestOnDiskTamperDetection`:
   правка байтов в файле → новый читатель `verify_chain() is False`). **Но:** весь
   Правки P3.1 входят в тот же commit, что и эта pre-registration, поэтому
   закрывать P3 надо по объекту в Git, а не по рабочему дереву:
   `git show $(git log -1 --format=%H -- fly_connectome_agent/src/engineering/governance/gatekeeper.py)`
   — и сравнивать строки там, а не в файле, который кто-то мог править после запуска
   тестов.
2. **Знаковый тест P4.1 пересчитан из сырых строк** (`scripts/recheck_p4_stats.py`,
   двусторонний exact `binomtest`, ties отдельно): критерий 1 — task A **19W/1L/0T**
   (n=20, p = 4.005e-05), task B **18W/2L/0T** (p = 4.025e-04); критерий 3 на mixed
   min-half по fresh seeds 20–59 — `no_plasticity` 37/39 (T=1, p = 2.841e-09),
   `m_zero` 37/39 (T=1), `weight_shuffled_frozen` 31/38 (T=2, p = 1.162e-04),
   `direction_shuffled_frozen` 40/40 (T=0, p = 1.819e-12). Заявленное в
   `docs/p4_validation.md` сходится; в отчёт внесены точные значения и число ties.
3. **Единица анализа — seed.** 60 seeds × 80 episodes × условия = 24 000 строк
   provenance, но n от этого не меняется. В P4.2 `unit_of_analysis` заперт в схеме
   как `const: "seed"`, и это закрыто падающими проверками, а не словом:
   `test_schema_rejects_episode_as_unit_of_analysis` (схема обязана отклонить
   `episode`) и `test_schema_rejects_post_hoc_protocol` (схема обязана отклонить
   `pre_registered: false`).

## Предзапусковой аудит: восемь критериев, каждый проверен кодом

`scripts/audit_p4_2_prereqs.py` берёт замороженный манифест, живые модули и текст
тестов и отвечает на восемь пунктов запуском, а не абзацем. Скрипт не трогает
confirmatory seed'ы: там, где нужно что-то прогнать, берутся seed'ы из
`fixture_probe` (объявлены вне всех гейтов) и 5 эпизодов.

```bash
.venv/bin/python scripts/audit_p4_2_prereqs.py   # 0 — все восемь подтверждены, 1 — провал
```

1. ALLOW сохраняет proposed action, и тест ловит подмену (проверка идёт на `move_left`).
2. DENY/ESCALATE навязывают только `stay`; подделка `PolicyDecision` не конструируется.
3. on-disk tamper ловится **свежим** `ProvenanceLog`, а не кэшем писавшего процесса.
4. confirmatory — ровно 60–119, disjoint с 0–59 и с `fixture_probe`.
5. `unit_of_analysis = seed` в манифесте и заперт `const` в схеме.
6. oracle симметричен (вектор инвариантен `MIRROR_PERM`), non-plastic (Δw ≡ 0 на живом
   прогоне), reward-blind (M ≡ 0 при r = 10), `gate: none`, вне матрицы P4.1.
7. У CLI раннера нет флагов для расширения seed-множества или изменения arms,
   эпизодов, порогов гейта и параметров обучения; `--seeds`/`--arms` умеют только
   подрезать объявленное (`check_subset` → SystemExit).
8. partial-прогон пишет `partial=true` и отвергается гейтом; отвергаются также
   отсутствие sidecar, чужой `manifest_digest` и sidecar без своего CSV.

## Что P4.2 не проверяет

- Не проверяет, что R-STDP вообще способен превзойти фиксированную политику на задаче, где
  ответ нельзя задать руками. Для этого нужен P4.3.
- Не добавляет новых нейронов, новых правил, «физики» и не трогает eligibility/модулятор.
- Не даёт права на biological claim: результат — `simulation_result` на 5-нейронной toy-сети.
- Не подключает real connectome CSV: это P5, и он наступает только после вердикта P4.2.

## Как запустить (пока не запущено)

```bash
# план и digest, ни одного эпизода
.venv/bin/python scripts/run_p4_2_oracle.py --describe

# полный прогон, один процесс: 6 arms x 60 seeds x 3 streams = 1080 ячеек, 86 400 эпизодов
.venv/bin/python scripts/run_p4_2_oracle.py \
    --out var/p4_2_oracle.csv --provenance var/p4_2_oracle.prov.jsonl

# или по шардам (подрезка помечается partial, гейт берёт только объединение полного набора)
.venv/bin/python scripts/run_p4_2_oracle.py --arms rstdp_mixed,no_plasticity,m_zero \
    --out var/p4_2_oracle_a.csv --provenance var/p4_2_oracle_a.prov.jsonl
.venv/bin/python scripts/run_p4_2_oracle.py \
    --arms weight_shuffled_frozen,direction_shuffled_frozen,oracle_reflex \
    --out var/p4_2_oracle_b.csv --provenance var/p4_2_oracle_b.prov.jsonl

# вердикт (sidecar каждого шарда обязателен; --manifest указывать не нужно,
#  он и так обязан совпасть с FROZEN_PROTOCOL_DIGEST)
.venv/bin/python scripts/check_p4_2_oracle_gate.py \
    --csv var/p4_2_oracle_a.csv var/p4_2_oracle_b.csv \
    --sidecar var/p4_2_oracle_a.csv.meta.json var/p4_2_oracle_b.csv.meta.json \
    --prior-csv var/p4_mixed80_a.csv var/p4_conf60_a.csv var/p4_conf60_b.csv \
    --per-seed-out var/p4_2_per_seed.csv --json-out var/p4_2_verdict.json
```

Ориентир по времени: измеренная скорость ≈ 15 эпизодов/с на свободной машине ⇒ ≈ 1.6 часа;
под загрузкой системы прогон P4.1 медленно падал до 0.27 эп/с, поэтому шардинг по arms разрешён
и предусмотрен. `var/` в `.gitignore`, поэтому CSV и sidecar — локальные артефакты; все числа
воспроизводимы командами выше.
