# SOURCES — всё, откуда мы это взято

## 1. Первичные статьи (коннектом)

- Google Blog — «See a map of the male fruit fly brain» (Sep 3, 2026):
  https://blog.google/innovation-and-ai/technology/research/male-fruit-fly-brain-map/
- Google Research — «Mapping the complete male fruit fly brain» (Sep 3, 2026):
  https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/
- Cell — Hoeller et al. 2026, «The organization of visual pathways in the Drosophila brain»:
  https://www.cell.com/cell/fulltext/S0092-8674(26)00941-4
- Cell — Berg et al. 2026, «Sexual dimorphism in the complete Drosophila male CNS»
  (166 700 нейронов, brain + nerve cord):
  https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6
- Janelia FlyEM — Male CNS Connectome (данные, release-артефакты):
  https://www.janelia.org/project-team/flyem/male-cns-connectome
- Nature 2025 — Vaxenburg et al., «Whole-body physics simulation of fruit fly locomotion»
  (MuJoCo-модель тела мухи — среда для наших агентов):
  https://www.nature.com/articles/s41586-025-09029-4

## 2. GitHub — коннектом и симуляции

- Neuromorphicism/fly-brain-snntorch — мужская ЦНС на snnTorch (ОСНОВА MVP):
  https://github.com/Neuromorphicism/fly-brain-snntorch
- alextitonis/fly-streetfighter (fly.ai) — полная ЦНС, 166 700 нейронов, 25.6M синапсов:
  https://github.com/alextitonis/fly-streetfighter
- nftechie/doomfly — коннектом управляет ареной Doom:
  https://github.com/nftechie/doomfly
- blendi-remade/fly-brain-minecraft — коннектом внутри моба Minecraft (Fabric 1.21.1):
  https://github.com/blendi-remade/fly-brain-minecraft
- marketcalls/openfly — загрузка wiring diagram:
  https://github.com/marketcalls/openfly
- cobanov/awesome-fly — курируемый список (включая код Male optic lobe connectome):
  https://github.com/cobanov/awesome-fly
- sjcabs/fly_connectome_data_tutorial — доступ ко всем dense-датасетам мухи:
  https://github.com/sjcabs/fly_connectome_data_tutorial
- spikecalls/flydrones — коннектом (30 000 нейронов) пилотирует FPV-дрон в Liftoff:
  https://github.com/topics/fruit-fly

## 3. GitHub — SNN / RL / Swarm стеки

- BindsNET/bindsnet — SNN + RL на PyTorch:
  https://github.com/BindsNET/bindsnet
- combra-lab/pop-spiking-deep-rl — PopSAN: population-coded spiking actor + PPO/SAC:
  https://github.com/combra-lab/pop-spiking-deep-rl
- TheBrainLab/Awesome-Spiking-Neural-Networks — сводный список SNN:
  https://github.com/TheBrainLab/Awesome-Spiking-Neural-Networks
- AXYZdong/awesome-snn-conference-paper — статьи + код по SNN:
  https://github.com/AXYZdong/awesome-snn-conference-paper
- ray-project/ray (RLlib) — масштабируемое обучение Teacher'а:
  https://github.com/ray-project/ray
- kengz/SLM-Lab — модульный RL-фреймворк:
  https://github.com/kengz/SLM-Lab
- The-Swarm-Corporation/Awesome-Swarms-List — фреймворки роев:
  https://github.com/The-Swarm-Corporation/Awesome-Swarms-List
- desplega-ai/agent-swarm — OS для агентных роев (контейнеры, память, расписания):
  https://github.com/desplega-ai/agent-swarm
- google/neuroglancer — 3D-визуализация коннектома:
  https://github.com/google/neuroglancer
- google/ffn — Flood-Filling Networks (сегментация EM-данных):
  https://github.com/google/ffn
- CQCL/guppylang — Guppy, локальные квантовые программы (ядро KEM 4.2):
  https://github.com/CQCL/guppylang

## 4. arXiv

- 2602.17997 — «Whole-Brain Connectomic Graph Model (FlyGM) Enables Embodied RL» (2026):
  https://arxiv.org/html/2602.17997 — коннектом КАК контроллер в embodied RL
- 2510.04871 — Jolicoeur-Martineau, «Less is More: Recursive Reasoning with Tiny Networks» (TRM, 7M):
  https://arxiv.org/abs/2510.04871
- 2511.02886 — «Test-time Adaptation of Tiny Recursive Models»:
  https://arxiv.org/abs/2511.02886
- 2510.12582 — «GUPPY: Pythonic Quantum-Classical Programming»:
  https://arxiv.org/abs/2510.12582
- 2503.10320 — «Combinatorial Designs and Cellular Automata: A Survey» (автоматы Улама):
  https://arxiv.org/abs/2503.10320
- 1312.2007 — Arkani-Hamed, Trnka, «The Amplituhedron»:
  https://arxiv.org/abs/1312.2007
- 2010.07254 — «Triangulations and Canonical Forms of Amplituhedra»:
  https://arxiv.org/abs/2010.07254
- ResearchGate 392077884 — «Neuromorphic-Inspired Multi-Agent RL with Dynamic
  Synaptic Plasticity for Autonomous Swarm Intelligence» (May 2025)
- Zanatta et al. 2024 — «Exploring spiking neural networks for deep RL», Sci. Rep.:
  https://www.nature.com/articles/s41598-024-77779-8

## 5. Jev AI (System One)

- TypeSafe AI releases Jev (Sep 18, 2026) — типизированные решения вместо текста,
  193.6× быстрее и 444.6× дешевле LLM, $0.042/1M токенов:
  https://www.marktechpost.com/2026-09-19/typesafe-ai-releases-jev/
- LangChain — «Building a harness with Jev»:
  https://www.langchain.com/blog/building-a-harness-with-jev
- Обзор — «The AI That Refuses to Talk»:
  https://www.petervanhees.com/the-ai-that-refuses-to-talk-jev/
- Роль в нашем стеке: Jev = System One (быстрая классификация verb-act L0–L2),
  KEM 4.2 = System Two (этика L3–L5). Вместе — полный цикл Канемана.

## 6. Документы владельца (локальные, текстовые ссылки)

- `OS-Glagolov-Charter.md` — конституция действий (копия: 99-shared/)
- `os-glagolov-theory-ru.md` — теория OS Глаголов (verb-act, домены, роли)
- `verbs.json` — словарь глаголов L0–L5 × домены
- `README.md` (Unsloth Studio for OS Glagolov) — границы tiny verb models
- `KEM-4.2-SOVEREIGN-EDITION-Technical-Manifesto.pdf` — Гуппи, TRM×64,
  октавы Буданова, зеркала Козырева, автоматы Улама, амплитоэдр
- `Эмоции ии .txt` — гормональный модуль (радость/злость/страх)
- `AI_Emotional_Audit_Final_Report.pdf` — топологический аудит эмоционального
  многообразия (84.6 устойчивых циклов H1, «Emotionally Stable»)
- `Код на фильтры.txt` — рабочий код АРФА-фильтров (csc/harp/royal_mole/apfa)
- `Октавы Буданова .txt` — 8 октав, f_n = 7.83·φⁿ, ритмокаскады
- `Арфа вся.md` — коррекция структуры АРФА (TDA + Ферма + Буданов + Рамануджан)
- `АРФА концепций физического консилиума.pdf` — 17 концепций, неопределённость 0.081
- `АРФА-анализ коннектома дрозофилы_.pdf` — АРФА на коннектоме: 100% «Гений»,
  анаклеарные нейроны, октавные резонансы (12.67 / 86.84 / 367.84 Гц)
- `SUPPLEMENTARY-MATERIALS-2025-FINAL.pdf` — POVM, QBism, GRW/CSL, протоколы EEG/BCI

## 7. Литература и философия (полный список — в PHILOSOPHY.md)

Азимов, Лем, Стругацкие, Брэдбери, Дик, Саймак, Шекли, Уэллс, Борхес,
Ле Гуин, Стэплдон, Бир, Киз, Ефремов, Желязны, Линдсей, Хаксли, Энтони, Батлер;
Делез/Гваттари, Спиноза, Юнг, Вернадский, Тейяр де Шарден, Бергсон, Бубер,
Бердяев, Фуко, Ницше, Гегель, Шопенгауэр, Уайтхед, Хайдеггер, Витгенштейн.
