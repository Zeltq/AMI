"""Render the bounded Russian review report from measured JSON artifacts."""
import json
from pathlib import Path

root=Path('reports/P07')
read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
c=read('comparison.json');v=read('verification.json');legacy=read('legacy-bytes.json');smoke=read('collect-smoke.json')
rows=[]
for r in c['simulator']:
    t=r['at_600'];rows.append(f"| {r['environment']} | {r['side']} | {int(t['lh'])}/{t['matches']} | {int(t['deny'])}/{t['matches']} | {int(r['deaths'])}/200 | {int(r['final_lh'])}/200; {int(r['final_deny'])}/200 |")
for r in c['dota']:rows.append(f"| Dota P03 | {r['side']} | {r['lh']}/20 | {r['deny']}/20 | 0/20 | {r['lh']}/20; {r['deny']}/20 |")
r0=c['simulator'][0]['at_600'];r1=c['simulator'][2]['at_600'];old=r0['deny']/r0['matches'];new=r1['deny']/r1['matches']
durations='; '.join(f"{name}: {v[name]['minutes']:.3f}" for name in ('baseline-004','regression-004','evaluation-005','collect-v4','rust-full'))
text=f'''# P07 — ggd-v4 и окружение 005

## Результаты

**Завершено.** Без обучения моделей. Dota build 6970; правила промаха дополнительно подтверждены пользователем, дальнейшие Dota-тесты им разрешено пропустить. Детали и исходные малые измерения доступны в [P07/](P07/).

### A. Видимый и скрытый вражеский SF: Bot API

| Величина | Доступна / проверено | Как получена и ограничение |
|---|---|---|
| GetAnimActivity / GetAnimCycle | по 1922/1922 видимых отсчёта | 8/8 кастов: три койла имеют activity=1500, ульта=1515; cycle койла продолжает idle. Три койла по одной активности неразличимы до эффекта. |
| Текущий скилл во время cast point | 204/204 cast-отсчёта, 8/8 кастов | GetCurrentActiveAbility → GetName плюс IsCastingAbility; различимы все 4 скилла. |
| GetAbilityByName / GetAbilityInSlot | 7688/7688; 11532/11532 handles | Видимый враг; доступны handles. |
| Кулдаун врага | 0/7688 информативных значений | GetCooldownTimeRemaining возвращает **-1**. В 2217/2217 пар с ненулевым собственным CD у врага -1. |
| Уровни врага | 7688/7688 совпадений с собственным наблюдателем | GetLevel у ability. |
| Способности при потере видимости | 0/5112 успешных GetAbilityByName; 0/7668 GetAbilityInSlot | Сохранённый handle недействителен; сведения не используются. |
| Стаки / время дебаффа на себе | по 841/841 отсчёту с модификатором | GetModifierByName, GetModifierStackCount, GetModifierRemainingDuration. |
| Стаки / время дебаффа на видимом враге | по 841/841 отсчёту с модификатором | Те же getters; отсутствие модификатора означает 0, ошибка getter — unknown. |
| Мана / максимум маны видимого врага | по 1922/1922 | GetMana / GetMaxMana. В тумане — по 0/1278 успешных чтений. |

[Числители, знаменатели, ошибки и события API](P07/api.json), [CSV](P07/api-events.csv), [точные Lua-источники](P07/probes/spells-006/).

### B. ggd-v4 и паритет

Поверх честного ggd-v3 добавлены one-hot каста/«нет», причинная фаза с known, часы с **наблюдаемого начала** каждого каста, уровни и дебаффы обоих героев. Часы учитывают и отменённый каст, не являются оценкой CD. Oracle выключен. Фаза неизвестна при первом наблюдении уже начавшегося каста; невидимая строка полностью скрыта. [Схема, шкалы, запуск native](P07/contract.md).

| Проверка | Совпало / проверено; ошибка |
|---|---|
| Производственный Lua → Python, включая туман | 130496/130496 значений, max Δ=0 |
| Записанные события Dota → производственный GPU observer | 102464/102464 полей в допуске 1e-4; max Δ=1,389e-5; 0/51232 ошибок known |
| Реальные касты Dota / GPU-физики | 4/4 скилла; 98/102 сопоставленных cast-отсчёта; max Δ=1,388e-5 |
| Те же поля после FP16 | 2152/2156 побитово; 2156/2156 в допуске 5e-4; max Δ=0,00012207 |
| Граница окончания каста | 4/4 отличаются на 1 тик (33,3 мс); это разрешённый временной допуск |
| Дебафф после реального койла | 55/55 отсчётов; max Δstacks=0, max Δremaining=7,749e-6 с |
| Старые входы ggd-v2/v3: 2 матча, первые 1024 решения на актёра | {legacy['equal_blocks']}/{legacy['total_blocks']} блоков, {legacy['equal_bytes']}/{legacy['equal_bytes']} байт идентичны старому бинарнику |
| collect-teacher v10, ggd-v4/005 | 2/2 матча; 39/39 проверенных блоков, 17834/17834 валидных решений; 97 команд скиллов, 209 кадров видимого каста врага |
| Oracle в этом сборе | 7028736/7028736 нулевых значений/known |

[Паритет](P07/actual-cast-parity.json), [GPU](P07/gpu-parity.json), [байты v2/v3](P07/legacy-bytes.json), [сбор](P07/collect-smoke.json). Начало каста/первый эффект выравниваются; полная идентичность игровых траекторий не подразумевается. В replay-паритете сравниваются 1601/2039 кадров с согласованным уровнем 7; остальные прогревают причинную память.

### C. Промах и 004 / 005 / Dota

| Обычные атаки Dota | Та же высота: промахи/атаки | Снизу вверх: промахи/атаки |
|---|---:|---:|
| SF | 0/240 | 60/240 |
| Мили-крип | 0/240 | 0/240 |
| Рейндж-крип | 0/240 | 59/240 |
| Башня как атакующий | 0/240 | 61/240 |
| SF: цель сверху при выпуске, на одной высоте при попадании | 0/240 | — |
| SF: цель на одной высоте при выпуске, сверху при попадании | — | 60/240 |

Итого для трёх дальних источников сверху цель: 180/720 промахов. Счётчики попаданий атрибутированы по entity атакующего; BAT сокращён до 0,8 с. [Метод, длительности, хеши](P07/dota-miss.json), [выпуски](P07/releases.csv), [события урона](P07/damage-events.csv). Правило 005: независимый seed-RNG draw с p=0,25 при попадании обычного дальнего снаряда, если категория высоты источника больше категории цели (`Z=512−128·level`). Применяется и к измеренным атакам башен; ветка способностей и ближних атак не меняется.

По **200/200** матчей teacher-v9 vs teacher-v9, seed=20261103, горизонт 600 с/штатный конец. LH/deny на 10:00 — доступные снимки 600±1 тик, после раннего конца цензурированы. Смерти и итоговые LH/deny — все 200 матчей. Dota — исходные 20 матчей [P03](P03.md).

| Среда | Сторона | LH 10:00 / N | Deny 10:00 / N | Смерти / матчи | Итог LH/матчи; deny/матчи |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Регрессия 004: **200/200** per_match записей полностью совпали, включая все минутные снимки; хеши равны. [Регрессия](P07/regression.json), [сравнение](P07/comparison.json), [все 400 матчей компактно](P07/games.json). Проверки: Python **193/193**, Rust **97/97**, Go `go test ./...` exit 0. [Команды и результаты](P07/verification.json).

## Выводы

- **Подтверждено:** API скрывает CD врага (-1), но предоставляет имя активного каста, уровни, ману и дебафф. v4 передаёт измеренные публичные сведения; oracle равен нулю.
- **Подтверждено:** в этой серии промах **не сократил** разрыв deny Radiant. 004: {r0['deny']:.0f}/{r0['matches']}={old:.3f}; 005: {r1['deny']:.0f}/{r1['matches']}={new:.3f}; Dota: 650/20=32,500. Разрыв вырос с {32.5-old:.3f} до {32.5-new:.3f} deny на матч.
- **Подтверждено:** по концам всех матчей LH обеих сторон снизились 16723/400 → 16020/400; deny 10556/400 → 10573/400; смерти 6/400 → 8/400. Учителя не изменены.
- **Гипотеза:** оставшийся разрыв требует других причин, например таймингов/позиционирования. Один seed не доказывает значимость небольшого изменения deny. Исторические 24,23 в P03 относятся к 003 и другой серии; здесь честный контроль 004 даёт {old:.2f}.

## Что сделано

- `bots/agent.lua`, native supplement/protobuf, `observations.py`, `enemy_observations.py`, модели и экспорт: отдельный ggd-v4. `gpu/src/enemy_observe.rs`: производственный observer, причинная память, replay/реальные cast-проверки. Подключение `main.rs`/`imitation.rs` минимальное.
- `physics.rs`, `world.rs`: флаг `uphill_miss` по умолчанию false; отдельная 005, копии активов с теми же SHA-256. [Engine diff](P07/environment.json). Код v9/v10 и все 4 запрещённых файла P08 совпадают с origin/main. [Хеши](P07/provenance.json).
- `gpu/scripts/p07_*.py`, `tests/test_p07_*.py`, GPU-тесты, малые JSON/CSV и точные Lua-источники зондов. Обучение не выполнялось.

## Команды, которыми получены результаты

Рабочая директория `C:/AMI-wt/P07`. Шесть дословных Dota-команд — [calibration-commands.json](P07/calibration-commands.json); все команды Rust/сборов/оценок — [verification-plan.json](P07/verification-plan.json), [verification.json](P07/verification.json). Замороженные версии сценариев приложены рядом с метриками.

```powershell
C:/Python314/python.exe gpu/scripts/p07_environment.py
C:/Python314/python.exe gpu/scripts/p07_finish.py
C:/Python314/python.exe gpu/scripts/p07_compat.py
C:/Python314/python.exe gpu/scripts/p07_summary.py C:/AMI/runs/p07/spells-006/dota.log reports/P07/api.json
C:/Python314/python.exe gpu/scripts/p07_parity.py C:/AMI/runs/p07/spells-006/dota.log C:/AMI/runs/p07/spell-fixture.json reports/P07/lua-python-parity.json
C:/Python314/python.exe gpu/scripts/p07_results.py dota
C:/Python314/python.exe gpu/scripts/p07_results.py parity
C:/Python314/python.exe gpu/scripts/p07_results.py evaluation
C:/Python314/python.exe gpu/scripts/p07_audit.py
$env:PYTHONPATH='src;tests;C:/AMI/tools/lua_test_runtime'; C:/Python314/python.exe -m unittest discover -s tests -v
```

В `C:/AMI-wt/P07/tools/nativecollect`: `go test ./...`. Длительности длинных этапов, мин (wall управляющего процесса): {durations}. Длительности Dota — в calibration-commands; GPU-драйвер опрашивал завершение каждые 10 с.

## Что не получилось или не проверено

- Первые зонды исправлены после неверного unit-list constant, неподходящего timer host и устаревших handles. Принудительный PerformAttack на склоне не принят как измерение промаха. Длинные ранние сцены портились посторонними атаками/концом матча; приняты изолированные прогоны с атрибуцией entity_hurt.
- `[P07] fix: retry transient Windows probe status replacement`: замену JSON блокировал читатель; добавлен ограниченный retry, тест. `[P07] fix: keep probes alive when shared heartbeat index is busy`: transient read-lock общего индекса больше не прерывает измерение; тест. Финальный build выполнен с exit 0.
- Не проверены PRD/корреляция промахов, перемещение/переиспользование слота источника в полёте и атаки по башне как цели. Модель использует независимые draws 25%; источник берётся из текущего состояния на impact.
- Полный игровой запуск обученной v4-модели в Dota, обучение и DAgger v4 не выполнялись. Native v4 требует нового supplement и настройки профиля; без него кадр отклоняется. Граница окончания каста различается на один тик; допуски приведены выше.

## Пути к локальным артефактам

- `C:/AMI/runs/p07/dota-calibration-005/engine.json`, `baseline-004/`, `regression-004/`, `evaluation-005/`: окружения и полные отчёты 200 матчей.
- `C:/AMI/runs/p07/collect-v4/`: 2 матча, 39 GGDDemo2-блоков; `compat-ggd-v2-*`, `compat-ggd-v3-*`: записи для сравнения байтов.
- `C:/AMI/runs/p07/spells-006/`, `sf-final/`, `melee-final/`, `ranged-001/`, `tower-001/`, `impact-003/`: сырые Dota-логи, protobuf и runtime-сценарии; `spell-fixture.json`.
- `C:/AMI/runs/p07/target/release/sf-duel-gpu.exe`, `baseline.exe`, `*.status.json`, `*.log`; общий статус — `C:/AMI/runs/pilot-status.json`, P07/P07_jobs. Веса/датасеты/бинарники/сырые логи не коммитились.

DONE P07
'''
assert len(text.splitlines())<=150,len(text.splitlines())
Path('reports/P07.md').write_text(text,encoding='utf-8')
print('report lines:',len(text.splitlines()))
