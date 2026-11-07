# Игра против P18 Self-Play 360m

Выбран сохранённый checkpoint000436:360.4325 минуты обучения на lastorder-fb-v2.
Источник: `C:/AMI/runs/p18/saved/self-play-360m/weights`.
CPU-копия: `C:/AMI/runs/p17/models/p18-self-play-360m.pt`;49/49 тензоров проверены побитово.
Исполнитель p17-v2 совместим с контрактом этих весов. Перенос силы этой модели в Dota ещё не проверен.
Автоматическое переключение console.log→console.MATCH_ID.log уже исправлено в policy server.

1. Закройте Dota. В PowerShell выполните установку с отдельной резервной копией текущих ботов:

```powershell
cd C:/AMI-wt/P17
$model = 'C:/AMI/runs/p17/models/p18-self-play-360m.pt'
$source = 'C:/AMI/runs/p18/saved/self-play-360m/weights'
$artifacts = 'C:/AMI/runs/p17/human-p18-360m'
python scripts/install_chio_bot.py --install --human-team 2 --artifacts $artifacts --model $model --source $source
```

Если команда завершилась ошибкой, не продолжайте запуск. Повторная установка в тот же artifacts требует restore.
Модель для игры задаётся аргументом `--model` policy server; policy-config.json в комплекте хранит те же пути.

2. В параметрах запуска Dota в Steam должны быть `-console -condebug`.
В том же PowerShell запустите CPU policy и оставьте окно открытым:

```powershell
$dota = 'C:/Program Files (x86)/Steam/steamapps/common/dota 2 beta/game/dota'
$session = 'C:/AMI/runs/p17/human-p18-360m/session-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
python scripts/play_vs_chio.py --model $model --team 3 --seed 20261103 --log "$dota/console.log" --bots "$dota/scripts/vscripts/bots" --output $session
```

Вычисления модели на CPU/BelowNormal, sampling, один поток. Графический клиент запускает пользователь.
Для каждого нового запуска policy используйте новую папку session. Одновременно нужен только один policy server.

3. Создайте закрытое лобби: **Local Host**, читы включены, боты **Local Dev Script**.
Выберите **Shadow Fiend за Radiant**. Противоположным SF за Dire управляет модель.
После появления героев выполните в игровой консоли:

```text
sv_cheats 1
script_reload_code bots/chio_play_setup
```

Скрипт подготавливает дуэль и мост. До первого входа в зону мида бот следует общей скриптовой навигации.
Затем решения принимает модель; путь назад на базу не включает скрипт повторно.

4. Если бот стоит, в последней session/status.json проверяйте:
`model` оканчивается на p18-self-play-360m.pt, `executor_version` равен p17-v2,
`requests` и `accepted` растут, `errors` пуст. Поле `log` показывает текущий журнал.
Нулевые requests означают отсутствие наблюдений; принятые STOP могут быть решениями модели.
В прошлой человеческой сессии Local Host уже передавал наблюдения и принимал команды;
полный матч новой модели с человеком пока не проверен.

5. После игры: Ctrl+C в окне policy, затем закройте Dota. Восстановление прежних файлов:

```powershell
cd C:/AMI-wt/P17
python scripts/install_chio_bot.py --restore --artifacts C:/AMI/runs/p17/human-p18-360m
```

Подготовка выполнена без установки. Проверки: export roundtrip49/49; install/restore и ротация журнала5/5 тестов.
