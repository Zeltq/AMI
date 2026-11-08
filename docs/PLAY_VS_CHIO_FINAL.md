# Игра против P18 continue-final

По итоговому сообщению чата «Train CHiO with lastorder-fb-v2» от08.11.2026 выбрана continue-final.
Это блок362 продолжения: +557.9006 минуты обучения после исходного Self-Play360m.
Источник: `C:/AMI/runs/p18/saved/continuation-best-final/weights`.
CPU-копия: `C:/AMI/runs/p17/models/p18-continue-final.pt`.
Проверено совпадение49/49 тензоров при экспорте; SHA256 CPU-файла:
`c1456bd28d32b235d3fa89298cb9d9ff80696580ac14d09fd94cec95fbf64951`.

## Текущее состояние

Совместимый исполнитель p17-v2 уже установлен для человека Radiant/бота Dire.
Проверены19/19 статических файлов установленного комплекта. Переустановка сейчас не нужна.
Файлы приказов могут меняться в игре; они не сравниваются с исходными пустыми файлами.
Новая подготовленная копия хранится в `C:/AMI/runs/p17/human-p18-final/prepared-20261108-182349-579387`.
Она не устанавливалась. Реальная существующая установка и backup относятся к `human-p18-360m`.
Выбор загружаемых весов задаёт `--model` policy server; этот аргумент заменяет прежнюю модель360m.
При подготовке не запускались GPU, обучение, симуляторные оценки или графическая Dota.

## Запуск

В параметрах запуска Dota в Steam должны быть `-console -condebug`.
Перед созданием лобби откройте PowerShell и выполните:

```powershell
cd C:/AMI-wt/P17
$model = 'C:/AMI/runs/p17/models/p18-continue-final.pt'
$dota = 'C:/Program Files (x86)/Steam/steamapps/common/dota 2 beta/game/dota'
$session = 'C:/AMI/runs/p17/human-p18-final/session-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
python scripts/play_vs_chio.py --model $model --team 3 --seed 20261103 --log "$dota/console.log" --bots "$dota/scripts/vscripts/bots" --output $session
```

Оставьте окно открытым. Одновременно должен работать только один policy server.
Модель работает на CPU/BelowNormal, sampling, один поток; смена console.MATCH_ID.log отслеживается автоматически.

Создайте закрытое лобби: **Local Host**, читы включены, боты **Local Dev Script**.
Выберите **Shadow Fiend за Radiant**; противник SF за Dire управляется моделью.
После появления героев выполните в игровой консоли:

```text
sv_cheats 1
script_reload_code bots/chio_play_setup
```

Сначала бот проходит от базы к миду по общему скрипту, затем управление принимает модель.
В новой session/status.json должны быть model=p18-continue-final.pt, executor_version=p17-v2,
action_selection=sampling, растущие requests/accepted и пустой errors.
В предыдущем человеческом матче канал наблюдений и подтверждения исполнения работали;
поведение и сила именно continue-final в Dota ещё не проверены.

## После игры

Остановите policy через Ctrl+C и закройте Dota.
Чтобы вернуть ботов, которые были до установки CHiO360m, используйте существующий backup:

```powershell
cd C:/AMI-wt/P17
python scripts/install_chio_bot.py --restore --artifacts C:/AMI/runs/p17/human-p18-360m
```

Если позже понадобится вновь установить CHiO из подготовленных исходников, при закрытой Dota:

```powershell
cd C:/AMI-wt/P17
python scripts/install_chio_bot.py --install --human-team 2 --artifacts C:/AMI/runs/p17/human-p18-final --model C:/AMI/runs/p17/models/p18-continue-final.pt --source C:/AMI/runs/p18/saved/continuation-best-final/weights
```

Только после этой новой установки для restore используйте `--artifacts C:/AMI/runs/p17/human-p18-final`.
Не устанавливайте повторно поверх незавершённой установки: сначала выполните соответствующий restore.
Подробности подготовки и проверки: `reports/P17/human-continue-final.json`.
