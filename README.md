# AMI / SF Duel

Агент для дуэлей Shadow Fiend 1v1 в Dota 2. Проект включает сбор обучающих
примеров в настоящем движке Dota, обучение по действиям учителя, self-play
и разрабатываемый симулятор, который выполняет матчи и PPO на GPU.

В репозитории сохранены исходники, конфигурации, тесты, сторонние зависимости
с лицензиями и необходимые таблицы окружения. Для обучения с нуля нужно
собрать новые матчи учителя; обученные веса и датасеты сюда не входят.
Сохранение кода позволяет повторить процесс обучения, но не восстановить
конкретные потерянные веса или побитово повторить старый запуск.

## Устройство проекта

Есть два контура выполнения:

- **Настоящая Dota:** выделенные серверы Windows → protobuf состояния мира →
  Go-учитель или Python policy server → Lua-исполнитель Bot API. Записи матчей
  проходят аудит, затем используются для behavior cloning и recurrent PPO.
- **GPU:** таблицы окружения → Rust/Burn/CubeCL → пакет симулируемых матчей →
  наблюдения → политика с LSTM → recurrent PPO. Модель можно экспортировать
  в PyTorch и оценить на настоящих серверах через тот же исполнитель.

### Python: модель, данные и обучение

| Модуль | Назначение |
| --- | --- |
| [`src/sf_duel/spec.py`](src/sf_duel/spec.py) | Контракт наблюдений, действий, предметов и целевых объектов. |
| [`observations.py`](src/sf_duel/observations.py) | Подготовка входов, нормализация, маски доступности и дополнение батчей. |
| [`model.py`](src/sf_duel/model.py) | PyTorch policy/value: энкодеры объектов, карта, LSTM, условные головы действий и выбор цели. |
| [`rules.py`](src/sf_duel/rules.py), [`teacher.py`](src/sf_duel/teacher.py) | Правила дуэли и Python-логика учителя. Нативный Go-учитель находится отдельно. |
| [`native_data.py`](src/sf_duel/native_data.py) | Чтение нативных protobuf-записей, восстановление входов сети и обучающих целей. |
| [`bc_data.py`](src/sf_duel/bc_data.py), [`bc_metrics.py`](src/sf_duel/bc_metrics.py) | Кэш последовательностей для imitation learning и метрики. |
| [`demonstrations.py`](src/sf_duel/demonstrations.py), [`telemetry.py`](src/sf_duel/telemetry.py) | Формат демонстраций и телеметрии Lua-контура. |
| [`native_policy_server.py`](src/sf_duel/native_policy_server.py) | Батчированный инференс, отдельная LSTM-память агентов, routing политик и запись rollout. |
| [`server.py`](src/sf_duel/server.py) | Сервис первоначального Lua-контура. |
| [`live/`](src/sf_duel/live) | Запуск и остановка Dota, консоль, установка скриптов, изоляция воркеров и RPC. |
| [`selfplay/`](src/sf_duel/selfplay) | Конфигурация, reward shaping, подготовка rollout, контроль качества и recurrent PPO. |
| [`protos/`](src/sf_duel/protos) | Готовые Python bindings protobuf. |

### Настоящие серверы и учителя

| Каталог / скрипт | Назначение |
| --- | --- |
| [`bots/`](bots) | Собственный Lua-код: действия Bot API, наблюдения, исполнение нативных команд, адаптер и монитор учителя. |
| [`tools/nativecollect/`](tools/nativecollect) | Go-учитель, получение world state, protobuf recorder, подтверждения исполнения и reward frames. `duel.proto` задаёт формат записи. |
| [`scripts/run_native_teacher.py`](scripts/run_native_teacher.py) | Один матч в изолированном сервере; может маршрутизировать команды из policy server. |
| [`scripts/collect_native_parallel.py`](scripts/collect_native_parallel.py) | Очередь матчей на нескольких серверах, повторы, аудит, train/validation split и resume. |
| [`scripts/audit_native_teacher.py`](scripts/audit_native_teacher.py), [`check_native_inputs.py`](scripts/check_native_inputs.py) | Аудит записи, целей обучения и полноты входов модели. |
| [`scripts/prepare_bc_cache.py`](scripts/prepare_bc_cache.py), [`train_bc.py`](scripts/train_bc.py) | Подготовка кэша и обучение по действиям учителя. |
| [`scripts/selfplay_cycle.py`](scripts/selfplay_cycle.py), [`train_selfplay.py`](scripts/train_selfplay.py) | Сбор и обучение одного цикла self-play. |
| [`scripts/evaluate_native_policy.py`](scripts/evaluate_native_policy.py) | Оценка политики против Go-учителя с чередованием сторон. |
| [`scripts/prepare_teacher.py`](scripts/prepare_teacher.py) | Сборка Lua-учителя из зафиксированного Open Hyper AI. |
| [`scripts/run_teacher_review.py`](scripts/run_teacher_review.py), [`collect_parallel.py`](scripts/collect_parallel.py), [`collect_teacher.py`](scripts/collect_teacher.py) | Первоначальный сбор через Lua/Open Hyper AI и запись реплеев. |
| [`tools/replaycheck/`](tools/replaycheck) | Проверка реплеев настоящей Dota через Manta. |
| [`configs/selfplay.json`](configs/selfplay.json) | Параметры PPO, сбора и пула соперников для реальных серверов. |

Остальные скрипты в [`scripts/`](scripts) проверяют механику, задержки,
качество поведения, перенос критика и продолжение обучения.
`run_mia_*.py`, `run_bc_continuation.py`, `run_pilot_followthrough.py` и
`gpu/scripts/pilot_run.py` содержат пути к конкретным локальным экспериментам.
Их исходники сохранены, но запуск после клонирования требует подготовки
входов и проверки параметров. Основные точки входа приведены ниже.

### GPU-симулятор

| Модуль в `gpu/src/` | Назначение |
| --- | --- |
| [`main.rs`](gpu/src/main.rs), [`runtime.rs`](gpu/src/runtime.rs) | CLI, выбор GPU, запуск CubeCL и проверка совместимости с Burn. |
| [`contract.rs`](gpu/src/contract.rs), [`model.rs`](gpu/src/model.rs), [`cell.rs`](gpu/src/cell.rs) | Контракт модели, policy/value и LSTM. |
| [`world.rs`](gpu/src/world.rs) | Таблицы окружения, состояние матчей и размещение буферов на GPU. |
| [`physics.rs`](gpu/src/physics.rs), [`navigation.rs`](gpu/src/navigation.rs) | Движение, столкновения, атаки, способности, экономика, крипы и генерация навигации. |
| [`observe.rs`](gpu/src/observe.rs), [`sampling.rs`](gpu/src/sampling.rs) | Наблюдения с учётом видимости, маски действий и sampling. |
| [`experience.rs`](gpu/src/experience.rs), [`learner.rs`](gpu/src/learner.rs), [`gradients.rs`](gpu/src/gradients.rs) | Последовательности rollout, GAE, recurrent PPO и градиенты. |
| [`selfplay.rs`](gpu/src/selfplay.rs), [`evaluation.rs`](gpu/src/evaluation.rs) | Замороженные соперники и оценка без обновления весов. |
| [`artifacts.rs`](gpu/src/artifacts.rs), [`trace.rs`](gpu/src/trace.rs) | Checkpoint, resume, ограниченная очередь записи и трассы матчей. |
| [`bridge.rs`](gpu/src/bridge.rs) | Импорт/экспорт весов и сопоставление вычислений с PyTorch. |
| [`fidelity.rs`](gpu/src/fidelity.rs), [`lane_fidelity.rs`](gpu/src/lane_fidelity.rs), [`aggro_fidelity.rs`](gpu/src/aggro_fidelity.rs), [`requiem_fidelity.rs`](gpu/src/requiem_fidelity.rs) | Сравнение механики симулятора с эталонами Dota. |
| [`checks.rs`](gpu/src/checks.rs), [`gpu_tests.rs`](gpu/src/gpu_tests.rs), [`startup_audit.rs`](gpu/src/startup_audit.rs), [`departure.rs`](gpu/src/departure.rs), [`policy_trace.rs`](gpu/src/policy_trace.rs), [`lr_diagnostic.rs`](gpu/src/lr_diagnostic.rs) | Регрессии, benchmark, проверка начала матча, истории политики и обновлений PPO. |

[`gpu/scripts/`](gpu/scripts) содержит калибровку, извлечение правил и геометрии,
сравнения с Dota, экспорт в PyTorch и HTML-просмотрщик трасс. В
[`gpu/bots/`](gpu/bots) лежат Lua-сценарии для измерений в настоящем движке.
[`tests/`](tests) содержит Python/Lua-проверки и фикстуры.
[`diagnostics/`](diagnostics) содержит отобранные инструменты проверки нативного
транспорта и окружения; старые проверки, требовавшие отдельный проект `dotabots`,
не входят в этот репозиторий.

## Правила дуэли и контракт

- Solo Mid, `GetGameMode() == 21`, два Shadow Fiend.
- Первая смерть героя или потеря центральной T1 завершает матч.
- На 600-й секунде победитель определяется по last hits; равенство даёт ничью.
- Bottle, Raindrops, Soul Ring, руны, нейтральные предметы и нейтральные крипы запрещены.
- Фиксированная прокачка койлов, ауры, ульта и правого таланта 10; Frenzy не используется.
- Начальный Wraith Band; далее Boots и Bracer. Разрешены TP, Mango, Flask, Tango, Clarity.
- Фиксированная прокачка и курьер относятся к инфраструктуре исполнения.

Политика получает собственное состояние, видимые объекты, публичные сведения
о противнике, карту, предыдущую команду и маски известности. Приватные параметры
скрытого противника во вход политики не попадают. Выходы включают команду,
координаты, конкретную цель, предмет, покупку и value. HOLD зарезервирован
в контракте, но отключён маской установленного Bot API.

Нативный сбор по умолчанию использует решение каждые четыре тика и offset
четыре тика. Self-play задаёт offset шесть тиков и режим обработки опоздавшей
текущей команды. Симулятор использует 30 тиков/с, решение каждые четыре тика
и offset шесть тиков с учётом передачи команды движку. При сравнении политик
нужно согласовывать эти параметры, способ выбора действий, FP16-округление
и расписание сброса LSTM. Они влияют на поведение даже при одинаковых весах.

## Окружение и установка

Основное окружение запуска реальных серверов — Windows, Steam Dota 2 и NTFS.
Код ожидает игру в `C:/Program Files (x86)/Steam/steamapps/common/dota 2 beta`:
см. `DEFAULT_GAME` в [`live/game.py`](src/sf_duel/live/game.py).
Воркеры используют hardlink и junction к установленной игре и создаются заново
через [`live/install.py`](src/sf_duel/live/install.py). Их каталоги в Git не нужны.

Часть WSL-команд привязана к `C:/AMI`, дистрибутиву `Ubuntu` и venv
`~/dota-compat-check`. В Rust-командах `export` и `replay` Linux Python имеет
путь `/home/zelt/dota-compat-check/bin/python`. При другом расположении нужно
адаптировать эти пути. Для восстановления с минимумом настроек используйте
`C:/AMI` и запускайте команды из корня проекта.

Потребуются:

- Python **3.12+**, NumPy, PyTorch и protobuf. Сгенерированные bindings требуют
  protobuf **7.35.1+**. `pyproject.toml` объявляет NumPy и опциональный PyTorch;
  protobuf нужно установить отдельно.
- Go **1.26+** для нативного сборщика и проверки реплеев.
- Rust **1.97.0** через rustup; версия задана в `gpu/rust-toolchain.toml`.
  Для Windows также нужны C++ build tools и Windows SDK.
- GPU с Vulkan для стандартной Rust-сборки. Для HIP предусмотрена feature `hip`.
  Обучение Python BC/PPO использует `torch.cuda` и требует совместимую GPU-сборку
  PyTorch; в AMD/WSL это ROCm. CPU-версия подходит для Python unit tests и
  отдельного режима инференса, но не для этих обучающих скриптов.

Windows PowerShell:

```powershell
cd C:\AMI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[model]" "protobuf>=7.35.1"
```

Для Python-обучения в WSL создайте `~/dota-compat-check`, установите подходящий
PyTorch/ROCm и остальные зависимости. [`scripts/wsl_python.sh`](scripts/wsl_python.sh)
задаёт переменные ROCm, путь к исходникам и запускает Python этого venv.
[`diagnostics/wsl-requirements.txt`](diagnostics/wsl-requirements.txt) сохраняет
снимок использовавшихся пакетов; ROCm wheel требует соответствующего источника
пакетов и драйвера. Обычная установка CPU PyTorch его не заменяет.

Go-сборки из исходников; готовые `.exe` не хранятся в Git:

```powershell
Push-Location tools/nativecollect
go build -o nativecollect.exe .
Pop-Location
Push-Location tools/replaycheck
go build -o replaycheck.exe .
Pop-Location
```

Bindings Python и Go уже включены. Их регенерация для обычной сборки не нужна.
`tools/generate_worldstate.py` обновляет схему из сети и изменяет исходники:
запускайте его только при намеренном обновлении интерфейса.

## Обучение через настоящую Dota

### 1. Собрать и проверить матчи учителя

Нативный Go-учитель не требует обученных весов или сборки Open Hyper AI.
Скрипт сам готовит Lua-исполнитель, создаёт воркеры и фиксирует источники записи.
Каталог нового запуска должен отсутствовать.

```powershell
python scripts/collect_native_parallel.py --run C:/AMI/runs/teacher-new --matches 4 --workers 2 --validation-matches 1 --timescale 4 --until 600
```

Для продолжения используйте ту же команду с `--resume`. Значения числа матчей,
validation, ускорения и длительности должны совпадать с исходным запуском.
`--workers` поддерживает слоты 0–19. Игровые файлы для hardlink должны находиться
на совместимом NTFS-томе. Windows может запросить доступ сервера через firewall;
предусмотрен [`scripts/allow_worker_firewall.ps1`](scripts/allow_worker_firewall.ps1).

Сбор формирует `dataset.json` с разделением по целым матчам. Два SF одного матча
не должны попадать в разные части датасета. Неуспешные попытки сохраняются отдельно;
только прошедшие аудит записи допускаются в набор. При отсутствии подтверждения
действия наблюдение может сохраняться, а цель получает маску обучения.

Один матч и отдельные аудиты:

```powershell
python scripts/run_native_teacher.py --run C:/AMI/runs/native-one --worker 0 --until 600 --wall-timeout 600
python scripts/audit_native_teacher.py C:/AMI/runs/native-one
python scripts/check_native_inputs.py C:/AMI/runs/native-one
```

Для Lua/Open Hyper AI-контура `scripts/prepare_teacher.py` создаёт `teacher_bots/`
из `external/openhyperai/bots` и собственных адаптеров. Этот генерируемый runtime
тоже не требуется переносить в Git.

### 2. Подготовить кэш и обучить policy

```powershell
python scripts/prepare_bc_cache.py C:/AMI/runs/teacher-new --output C:/AMI/runs/teacher-cache
wsl -d Ubuntu -- bash /mnt/c/AMI/scripts/wsl_python.sh /mnt/c/AMI/scripts/train_bc.py /mnt/c/AMI/runs/teacher-cache --output /mnt/c/AMI/runs/bc-new --epochs 5 --minutes 60
```

`train_bc.py` создаёт модель с нуля; `--resume` позволяет явно выбрать сохранённый
checkpoint. Имена каталогов произвольные, каждый новый training output должен
быть отдельным. Это небольшой пример запуска, а не гарантия достаточного объёма
данных или качества агента.

### 3. Self-play и оценка

После BC передайте новый checkpoint явно. Встроенные пути к старым весам
в свежем клоне отсутствуют.

```powershell
# Проверить конфигурацию и наличие checkpoint, без сбора и обучения.
python scripts/selfplay_cycle.py --checkpoint C:/AMI/runs/bc-new/best.pt --run C:/AMI/runs/selfplay-new
# Запустить один цикл: сбор, подготовка rollout, PPO и оценка.
python scripts/selfplay_cycle.py --checkpoint C:/AMI/runs/bc-new/best.pt --run C:/AMI/runs/selfplay-new --execute
python scripts/evaluate_native_policy.py --checkpoint C:/AMI/runs/bc-new/best.pt --run C:/AMI/runs/evaluation-new --matches 10 --workers 2
```

Для одного сбора без PPO у `selfplay_cycle.py` есть `--collect-only`.
Состав замороженных соперников задаётся `--opponent`, параметры — через
`--config`. Reward shaping, маски выборок, проверка log probability и PPO
находятся в `src/sf_duel/selfplay/`.

## GPU: подготовка, запуск и перенос в Dota

### Таблицы окружения

Симулятору нужны измеренные правила и геометрия. В Git сохранены необходимые
входы по существующим путям, поскольку код и тесты ссылаются на них:

| Данные | Назначение |
| --- | --- |
| [`runs/gpu/dota-calibration-003/`](runs/gpu/dota-calibration-003) | Базовый `engine.json`, исходные правила, калибровка, геометрия наблюдений, движения и FOW. Это стандартный путь CLI и GPU-тестов. |
| [`runs/gpu/record-*.json`](runs/gpu) | Отобранные измеренные параметры механики и свидетельства калибровки; используются генераторами правил. |
| [`runs/gpu/dota-authored-fow-001/`](runs/gpu/dota-authored-fow-001), [`dota-native-grid-001/`](runs/gpu/dota-native-grid-001) | Данные FOW и проходимости для пересборки окружения. |
| [`locked-environment/`](runs/gpu/overnight-8h-20261102-001/pilot-core-mia2-25m-003/locked-environment) | Дополнительный снимок входных таблиц с атласом статической видимости и профилем снаряда SF. Для использования выберите его `engine.json` в конфигурации и пересоберите навигацию. |
| [`tests/fixtures/`](tests/fixtures) и три эталона в `runs/gpu/` | Измерения округления урона, предыдущих действий, границ койлов, отмены атаки и агро, на которые прямо ссылаются регрессионные тесты. |

Три эталона: `lane-conformance-001/windup-cancellation-001/native.json`,
`dota-server-aggro-014-melee-clean/server-scenarios.json` и
`overnight-8h-20261102-001/raze-hull16-qualified.native.json`.
Это тестовые входы; полные журналы и истории обучения не сохраняются.
Файлы `.f32` здесь содержат карту, а `.u32` — атлас видимости; весов модели
среди них нет.

Навигационный атлас размером 128 МиБ исключён и строится из геометрии.
`engine.json` хранит абсолютный путь `calibration`: после клонирования в другое
место сначала обновите его для используемого окружения. Пример для стандартного:

```powershell
python -c "import json,pathlib; p=pathlib.Path('runs/gpu/dota-calibration-003/engine.json'); d=json.loads(p.read_text()); d['calibration']=str(p.parent.resolve()); p.write_text(json.dumps(d,indent=2))"
cargo build --release --locked --manifest-path gpu/Cargo.toml
.\gpu\target\release\sf-duel-gpu.exe build-navigation runs/gpu/dota-calibration-003/engine.json
```

`build-navigation` обновляет ссылку на атлас в выбранном `engine.json`.
Дополнительный снимок окружения подготавливается той же командой с его путём.
При новой версии Dota калибровку следует повторить: старые таблицы фиксируют
конкретную сборку игры. Начальные команды — `gpu/scripts/calibrate.py --run ...`,
`rules.py <calibration>` и `engine.py <calibration>`. Сборка `engine.py` также
читает измерения `record-*.json`, FOW и геометрию из указанных выше путей;
одного запуска калибровки недостаточно для проверки всей механики.

### Запуск с нуля

`gpu/config.json` не задаёт исходные веса и позволяет создать случайную модель.
Конфигурации `gpu/config-100.json` и похожие варианты содержат ссылки на локальный
архив весов. Для нового эксперимента их нужно заменить на свои пути либо убрать
`initial_checkpoint` и `opponents`.

```powershell
.\gpu\target\release\sf-duel-gpu.exe qualify
# Один экспериментальный цикл PPO в новом каталоге.
.\gpu\target\release\sf-duel-gpu.exe train gpu/config.json runs/gpu/new-training --cycles 1 --draft
# Продолжение использует тот же config и каталог.
.\gpu\target\release\sf-duel-gpu.exe train gpu/config.json runs/gpu/new-training --cycles 1 --resume --draft
# Запросить завершение с сохранением.
.\gpu\target\release\sf-duel-gpu.exe stop runs/gpu/new-training
```

Симулятор находится в разработке: `behavior_validated` у сохранённых таблиц
равен `false`. Флаг `--draft` разрешает эксперимент с таким окружением.
Скорость симуляции и уменьшение PPO loss сами по себе не подтверждают перенос
поведения в Dota. Для этого нужны сравнения механики, независимые оценки
с чередованием сторон и полные матчи в настоящем движке.

Стандартный backend — Vulkan на дискретном GPU 0. HIP-сборка использует:

```powershell
cargo build --release --locked --manifest-path gpu/Cargo.toml --no-default-features --features hip
```

Для неё требуется отдельное совместимое ROCm/HIP-окружение.

### Экспорт и оценка

Ниже `CHECKPOINT_DIR` — каталог нового GPU-checkpoint с `model.bpk`,
`BASELINE.pt` — самостоятельно подготовленная политика для сравнения.

```powershell
.\gpu\target\release\sf-duel-gpu.exe export CHECKPOINT_DIR runs/gpu/agent.pt
python gpu/scripts/evaluate_dota.py --checkpoint runs/gpu/agent.pt --before BASELINE.pt --run runs/gpu/dota-evaluation-new --matches 20 --workers 2 --inference-device cpu --inference-host windows
```

По умолчанию оценщик использует WSL; `--inference-host windows` с CPU запускает
Python-инференс непосредственно в Windows. `gpu/scripts/weights.py` делает
обратный мост PyTorch → Rust. Команда `replay` строит HTML из сохранённых трасс;
её CLI и остальные команды доступны через `sf-duel-gpu --help`.

## Проверки

Из корня проекта, в окружении с установленными зависимостями:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH="$PWD/src;$PWD/tests"
python -m unittest discover -s tests -v
Push-Location tools/nativecollect
go test ./...
Pop-Location
Push-Location tools/replaycheck
go test ./...
Pop-Location
```

После сборки навигационного атласа можно запускать GPU-регрессии:

```powershell
cargo test --release --locked --manifest-path gpu/Cargo.toml
```

GPU-тесты требуют доступного GPU и подготовленных таблиц окружения; некоторые
сравнивают вычисления с сохранёнными нативными фикстурами. Новые проверки против
движка выполняются через `gpu/scripts/lane_conformance.py`, `scenarios.py`,
`server_scenarios.py`, `evaluate_dota.py` и Rust-команды `replay-lane`,
`validate-combat`, `validate-movement`, `validate-aggro`, `validate-requiem`.

## Состав Git и сторонний код

Веса `.pt/.bpk`, optimizer state, датасеты BC/PPO, журналы, игровые реплеи,
бинарники, каталоги `target*`, установленные пакеты, игровые файлы, рабочие
воркеры и генерируемый `teacher_bots/` исключены из исходного коммита.
Из `runs/` включены только перечисленные таблицы и регрессионные фикстуры.

Этот снимок подготовлен с изменением только корневого README. Правила исключения
на исходном компьютере записаны в локальный `.git/info/exclude`; они не переносятся
при клонировании. Перед последующими коммитами настройте исключения в своём клоне
и проверяйте `git diff --cached --stat`, особенно после сборки или обучения.

- [`external/openhyperai/`](external/openhyperai) — снимок
  [Open Hyper AI](https://github.com/forest0xia/dota2bot-OpenHyperAI), commit
  `cb814c6c8dc51ed08045d6efd9f4a48147992711`, MIT; метаданные в `UPSTREAM.json`.
- [`external/manta/`](external/manta) — Manta v1.5.2, MIT, с локальной поправкой
  обработки номера сборки сервера; см. `LOCAL_PATCH.txt`. Используется через
  `replace` в `tools/replaycheck/go.mod`.
- [`external/worldstate/`](external/worldstate) — зафиксированная схема
  SteamTracking/Protobufs и её адаптация без изменения wire format;
  происхождение и хеш записаны в `UPSTREAM.json`.

Лицензии сторонних компонентов сохранены в их каталогах. Отдельная лицензия
для собственного кода проекта пока не задана.
