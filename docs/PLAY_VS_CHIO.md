# Игра против CHiO (P17)

Исполнитель работает на CPU. Графическую Dota запускает только пользователь.
На выделенном сервере проверены13/13 действий, задержка7тиков и отмена каста.
**Local Host с человеком ещё не проверен:** графический клиент в P17 не запускался.
Доступ к серверному Lua подтверждён в обычном dedicated gamemode21; если listen-server
не разрешит `script_reload_code`, игра с честным MOVE28 пока невозможна.
Не подменяйте MOVE28 на Action_MoveDirectly: это другое движение.

## Подготовка и установка

Рабочая папка — `C:/AMI-wt/P17`. Нужен Windows Python с torch и numpy.
Установщик уже подготовлен, но в пользовательскую папку ботов **не выполнялся**.
Без аргументов он только создаёт комплект в `C:/AMI/runs/p17/human/prepared-*`:

```powershell
cd C:/AMI-wt/P17
python scripts/install_chio_bot.py --human-team 2
```

Перед установкой закройте Dota. Человек будет Radiant, бот — Dire:

```powershell
python scripts/install_chio_bot.py --install --human-team 2
```

Установка делает полную резервную копию текущих файлов, проверяет SHA256,
сохраняет `C:/AMI/runs/p17/human/installation.json` до замены содержимого.
Сам каталог или junction `game/dota/scripts/vscripts/bots` сохраняется.
Для человека Dire используйте `--human-team 3`, а у policy server ниже `--team 2`.
Повторная установка требует предварительного восстановления.

## Policy server и лобби

Проверьте `C:/AMI/runs/p17/models/selected.json`: поле `model` содержит путь
последних скопированных и проверенных весов. P15-smoke проверяет только транспорт
и может оставаться на базе; его не следует считать обученной игровой моделью P16.
Burn bundle можно передать напрямую либо экспортировать в CPU checkpoint:

```powershell
python scripts/export_chio_cpu.py --bundle C:/AMI/runs/p17/models/ИМЯ/weights --output C:/AMI/runs/p17/models/ИМЯ.pt
```

1. В параметрах запуска Steam для Dota добавьте `-console -condebug`.
   Эти параметры пользователь меняет сам; P17 их не устанавливал.
2. Запустите CPU policy server в PowerShell, указав **существующий** путь весов:

```powershell
cd C:/AMI-wt/P17
python scripts/play_vs_chio.py --model "C:/AMI/runs/p17/models/ИМЯ.pt" --team 3 --log "C:/Program Files (x86)/Steam/steamapps/common/dota 2 beta/game/dota/console.log" --bots "C:/Program Files (x86)/Steam/steamapps/common/dota 2 beta/game/dota/scripts/vscripts/bots" --output "C:/AMI/runs/p17/human/session-001"
```

   Для новой сессии используйте новое имя output. Процесс использует BelowNormal,
   один CPU-поток PyTorch, GPU не используется. Старые строки console.log пропускаются.
3. Запустите Dota, создайте закрытое лобби с сервером **Local Host**,
   включите читы и выберите **Local Dev Script** для ботов. Выберите Shadow Fiend
   за Radiant. Дождитесь появления своего героя и противоположного SF.
4. В игровой консоли выполните две команды:

```text
sv_cheats 1
script_reload_code bots/chio_play_setup
```

   Скрипт проверит состав, уберёт лишних ботов, выдаст Wraith Band обоим SF и
   загрузит мост28 с телеметрией. Проверяйте `SF_P17_BRIDGE_READY` и `SF_PLAY_EVENTS_READY`
   в console.log, рост requests/accepted в `session-001/status.json`.
   При `Cheat command ... ignored` мост не загрузился. Если команды недоступны в
   Local Host даже при разрешённых читах, этот вариант игры ещё невозможен:
   подтверждена только выделенная серверная среда, автоматического обхода из Bot API нет.

Ручных этапов после подготовки: установка, запуск policy server, создание лобби,
две консольные команды. Автоматического запуска серверного Lua из Bot API нет.
Policy server читает только наблюдения Bot API; серверные события нужны отчёту,
они не поступают в модель. Решение каждые6тиков; исполнение на observed+7.
Модель может проигрывать: совпадение действий и транспорта не гарантирует перенос силы из симулятора.

## Завершение и восстановление

Остановите policy server через Ctrl+C, закройте Dota. Восстановление одной командой:

```powershell
cd C:/AMI-wt/P17
python scripts/install_chio_bot.py --restore
```

Проверяется исходная резервная копия и восстановленные байты. Содержимое после игры
также сохраняется отдельной копией перед восстановлением. Настройки запуска Steam
`-console -condebug` при необходимости удаляет пользователь.

Выделенные ночные проверки используют один и тот же уже разрешённый EXE
`C:/AMI/runs/p17/bridge-001/runtime/game/bin/win64/dota2.exe` с bind127.0.0.1.
Установка для человека не создаёт новый EXE и не меняет firewall.
