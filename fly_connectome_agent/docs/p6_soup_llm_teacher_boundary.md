# P6 — Soup LLM Teacher Boundary

## Purpose

Soup может использоваться только как внешний advisory LLM layer,
который читает завершённые experiment result bundles и предлагает
объяснения / следующие эксперименты.

Soup — YAML-oriented toolkit для fine-tuning/post-training LLM
([github.com/MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup)).
Это НЕ SNN simulator и НЕ нейроморфный framework. SNN и LLM — разные
обучающие системы:

```
Fly SNN:   sparse graph + LIF + edge-sparse R-STDP
           learning from spikes, eligibility traces, reward modulator

Soup LLM:  gradient-based language-model fine-tuning / post-training
           learning from text / structured data
```

Их нельзя смешивать в одном learning loop.

## Architecture

```
FlyConnectomeAgent result bundle
  → immutable export
  → Soup / LLM teacher analysis
  → TeacherAnalysisProposal
  → Human review
  → new versioned experiment manifest
  → optional future SNN experiment
```

Поток строго направлен: от замороженного артефакта к тексту-предложению,
и никогда обратно в рантайм мухи.

## Hard boundaries

Soup **cannot**:

- execute actions;
- call `environment.step`;
- mutate SNN weights;
- mutate connectome graph;
- mutate R-STDP state;
- call GateKeeper;
- write provenance;
- create or modify confirmed manifests;
- approve or reject scientific hypotheses;
- issue KEMDecision;
- interact with Registry.

Каждый пункт — не «нежелательно», а физически недоступен: Soup работает в
изолированном окружении без write-доступа к проекту.

## Allowed inputs

Only immutable, completed artifacts:

- experiment manifest;
- connectome manifest;
- gate verdict JSON;
- per-seed result CSV;
- provenance hash manifest;
- environment fingerprint;
- approved documentation excerpts.

Ни один живой артефакт (provenance log в процессе записи, незавершённый
CSV) не подаётся на вход LLM.

## Allowed outputs

Only `TeacherAnalysisProposal` (см. `schemas/teacher_analysis_proposal.schema.json`):

- result summary;
- anomalies;
- limitations;
- uncertainty statements;
- suggested next experiments;
- requested missing evidence.

## Output status

Все LLM-выходы обязаны нести:

| флаг | значение |
|---|---|
| `advisory_only` | `true` |
| `policy_eligible` | `false` |
| `scientific_claim_eligible` | `false` |
| `requires_human_review` | `true` |
| `may_modify_snn` | `false` |
| `may_execute_action` | `false` |
| `may_write_provenance` | `false` |
| `may_modify_manifest` | `false` |

Схема закрепляет каждый из них как `const`; любое иное значение — ошибка
валидации. Это не «договорённость», а формальное ограничение формата.

## Human review rule

A TeacherAnalysisProposal cannot change any experiment.

Only a human-approved proposal can create a new:

- `protocol_id`;
- manifest;
- seed range;
- experiment plan.

LLM не имеет привилегий, которые есть у review: ни права на запуск, ни
права на изменение замороженного протокола.

## Isolation

Soup must run in a separate environment:

- separate venv or Docker;
- pinned Soup commit/version;
- no shared writable state with FlyConnectomeAgent;
- read-only access to input result bundle;
- write only to a separate `teacher_outputs/` directory.

Ни один файл проекта не открывается Soup на запись. Каталог вывода —
вне дерева репозитория или в gitignored-пути.

## Forbidden field names

Схема отвергает любые попытки передать исполняемую семантику:

```
action, enforced_action, weight_update, synapse_update,
plasticity_update, manifest_override, seed_override,
gate_override, provenance_write, registry_write,
execute, shell_command
```

Эти имена — не «зарезервированы на будущее», а гарантированно
отсутствуют в allowed properties. `additionalProperties: false` делает
их введение ошибкой валидации без отдельного `not`-списка.

## Non-goals

Soup does not:

- simulate a fly brain;
- validate biological claims;
- train the SNN;
- replace R-STDP;
- replace governance;
- act as an autonomous researcher.

## Precondition for any Soup activity

Никаких действий с Soup до тех пор, пока:

1. P4.2.v2 прошёл independent review;
2. P4.2.v2 run завершён;
3. frozen gate дал verdict;
4. result bundle сохранён и захеширован;
5. P5.0 ещё не начат либо имеет отдельный approved plan;
6. пользователь явно одобрил P6 pilot (`P6 Soup pilot approved`).

До выполнения шести условий единственный допустимый артефакт —
документация и схема в этом коммите.

## Status

```
P6 DOCUMENTATION/SCHEMA PREPARED
SOUP NOT INSTALLED
SOUP NOT AUTHORIZED TO RUN
```
