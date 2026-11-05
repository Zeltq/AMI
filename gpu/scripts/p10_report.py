"""Write P10 evaluation and completed human-match results, at most 150 lines."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/P10';RUN=Path('C:/AMI/runs/p10')


def number(value):
    return f'{value:.2f}'.rstrip('0').rstrip('.')


def main():
    sim=json.loads((OUT/'simulator.json').read_text());dota=json.loads((OUT/'dota.json').read_text())
    assert len(sim['games'])==64 and len(dota['games'])==8
    assert json.loads((RUN/'native-eight/status.json').read_text())['stage']=='complete'
    assert json.loads((RUN/'installation.json').read_text())['installed']
    fixes=[line for line in (ROOT/'reports/P10.md').read_text(encoding='utf-8').splitlines() if line.startswith('- `[P10] fix:')]
    tests=json.loads((OUT/'verification.json').read_text())['python']['tests']
    human=json.loads((OUT/'human-match.json').read_text()) if (OUT/'human-match.json').exists() else None
    lines=['# P10 — ZOg 000470 в настоящей Dota','', '## Результаты','',
        '8/8 матчей Dota против Lua teacher-v9, по 4 за сторону; 64/64 в симуляторе 005 против v9, по 32. Seed 20261105, sampling. Числа относятся к модели; LH/deny суммируются до первой смерти/конца. Матч человека: 0/1, точка ожидания пользователя.','',
        '| Среда / сторона | Победы / N | Ничьи / N | Поражения / N | LH / N | Deny / N | Смерти / N | Убийства / N | Время, сумма с / N |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    datasets=[('Dota',dota),('005',sim)]
    for name,data in datasets:
        for team,s in data['summary'].items():
            n=s['matches'];label=name+' '+('Radiant' if team=='2' else 'Dire')
            values=[s['wins'],s['draws'],s['losses'],*[s[k]['numerator'] for k in ('lh','deny','deaths','kills')],s['time']['sum']]
            lines.append('| '+label+' | '+' | '.join(f'{number(v)}/{n}' for v in values)+' |')
    lines+=['','| Среда / сторона | Койлы / матчи | Попадания по целям / койлы | По герою / койлы | Команды атаки героя / решения | Отклонено / решения |','|---|---:|---:|---:|---:|---:|']
    for name,data in datasets:
        for team,s in data['summary'].items():
            val=lambda k:s[k]['numerator'];casts=val('raze_casts');decisions=val('decisions');n=s['matches']
            rejected=f"{val('rejected')}/{number(decisions)}" if s['rejected']['measured'] else f'нет измерения; 0/{n} охват'
            lines.append(f"| {name} {'Radiant' if team=='2' else 'Dire'} | {number(casts)}/{n} | {number(val('raze_hit_events'))}/{number(casts)} | {number(val('raze_hero_hits'))}/{number(casts)} | {number(val('hero_attack_commands'))}/{number(decisions)} | {rejected} |")
    lines+=['','Попадание — событие урона по одной цели; один койл может дать несколько попаданий. Решения Dota цензурированы на границе конца; незавершённые команды не считаются пропущенными.','',
            '| Среда / сторона | Причины конца / матчи | Выход на мид: сумма игровых с / достигшие | Первая собственная смерть: сумма с / наблюдавшиеся |','|---|---|---:|---:|']
    for name,data in datasets:
        for team,s in data['summary'].items():
            reasons=', '.join(f'{k} {v}/{s["matches"]}' for k,v in s['reasons'].items())
            timing=lambda k:f"{number(s[k]['sum'])}/{s[k]['observed']}" if s[k]['observed'] else f"нет; 0/{s['matches']}"
            lines.append(f"| {name} {'Radiant' if team=='2' else 'Dire'} | {reasons} | {timing('mid_game_time')} | {timing('first_death')} |")
    lines.append('')
    for team,s in dota['summary'].items():
        rejected=s['rejected']['numerator']
        lines.append(f"- Отказы {'Radiant' if team=='2' else 'Dire'}: "+', '.join(f'{k} {v}/{rejected}' for k,v in s['rejected_reasons'].items())+'.')
    transport={k:sum(g['transport'][k] for g in dota['games']) for k in ('checked_before_terminal','acknowledged','late_executions','unacknowledged')}
    n=transport['checked_before_terminal']
    lines+=['',f"ACK {transport['acknowledged']}/{n}; опоздания {transport['late_executions']}/{n}; без ACK {transport['unacknowledged']}/{n}. Мид: радиус <2000, |x−y|<1200, только живые; Dota измеряется каждые 0,1 с. Полные обе стороны, причины, таймлайны/сводки: [Dota](P10/dota.json), [005](P10/simulator.json).",'',
        '| Проверка | Совпало / проверено |','|---|---:|',
        '| CPU-экспорт против checkpoint | 59/59 тензоров побитово |',
        '| Native ggd-v4, статы героя после 0:00 | 20/20 FP16 |',
        '| Статические поля башен | 52/52 известных; 4/56 неизвестны |',
        '| Lua → Python, публичные поля и known | 130368/130368; max Δ=0 |',
        '| Те же события → GPU observer | 119424/119424 в 1e-4; max Δ=1,395e-5 |',
        '| Реальные касты Dota / симулятора | 4/4 скилла; 2156/2156 FP16; граница 4/4 отличается на 1 тик |',
        '| Дебафф койла после эффекта | 55/55 кадров; Δstacks=0, Δвремя≤7,75e-6 с |',
        '| Восстановление копии текущих ботов | 29/29 файлов; 32/32 элемента; 2/2 варианта (каталог и junction) |',
        f'| Тесты | Python {tests}/{tests}; Go 1/1 пакет с тестами |',
        '| Принудительная смерть в отдельном зонде | 1/1 смерть, 1/1 убийца; 0/2 недостигнутые отметки выданы за измерение |','',
        '## Наблюдения','',
        '- **Подтверждено:** контракт CPU: ggd-v4, FP16→FP32, previous из выданной команды, LSTM reset-128/смерть, sampling; honest=true, oracle=false. [Происхождение](P10/provenance.json).',
        '- **Подтверждено:** исходный dump у фонтана совпал по 14/20 статам; после выравнивания фазы первого снимка после 0:00 — 20/20. Сохранены [оба измерения](P10/static-unmatched-phase.json) и [паритет фаз](P10/static-parity.json).',
        '- **Подтверждено:** при первом появлении врага старый список handles от команд не содержал SF; отдельный публичный supplement устранил этот сбой. Если Lua ещё не видит врага, поля v4 неизвестны.',
        '- **Подтверждено:** пользователь отменил продолжение после респавна: первая смерть любого участника заканчивает дуэль; после неё пользователь выходит. Native executor после смерти не расширялся.',
        '- **Подтверждено:** первый пользовательский запуск выявил сбой setup. В отдельном зонде 10/10 имён ботов пусты, старый setup выдаёт Unexpected bot. Исправленный setup удалил 8/8 лишних ботов, сохранил 2 SF и 0 Lina; человек в этом dedicated-зонде имитирован только проверкой IsFakeClient. [Проверка](P10/setup-fix.json).',
        '- **Подтверждено:** в Dota 0/8 побед, 2/8 смерти, 0/8 убийств; в 005 — 21/64 побед, 4/64 смерти, 15/64 убийств. Смерти Dota: башня на 103,07 с и мили-крип на 393,63 с; это не first blood героя.',
        '- **Гипотеза:** часть target_unavailable вызвана исчезновением цели за семь тиков задержки; смерть цели и уход в туман отдельно не разделены.','',
        '## Что сделано','',
        '- `scripts/install_play_bot.py`, `scripts/play_vs_model.py`: полные копии, проверяемое восстановление содержимого с сохранением каталога/junction, CPU-сервис, одна дуэль, итог и телеметрия.',
        '- `bots/play_model_events.lua`, native supplement/adapter, evaluator: профиль v4, публичные события, штатные sockets и `-condebug`. Физика, учителя и веса не изменены.',
        '- `docs/PLAY_VS_MODEL.md`: параметры Steam, лобби Local Host/Local Dev Script, команды, F9, завершение и восстановление. RCON не требуется; рабочая консольная команда — `script_reload_code`.',
        '- Скрипты установлены: человек Radiant, модель Dire; [копия и хеши](P10/installation.json). После проверок Dota/collector/policy server: 0/0/0 процессов; [проверка](P10/cleanup.json). Heartbeat приостановлен.',
        '- `gpu/scripts/p10_*.py`, `reports/P10/*.json`: воспроизводимые измерения и малые результаты; каждый JSON <1 МБ.',*fixes,'',
        '## Команды, которыми получены результаты','',
        'Из `C:/AMI-wt/P10`; дословные длинные команды и все попытки: [commands.json](P10/commands.json).','```powershell',
        'python gpu/scripts/export_policy.py C:/AMI/runs/p09/P09c/continue90/rl-continue-001/checkpoint-000470/weights C:/AMI/runs/p10/ZOg-000470.pt',
        'python gpu/scripts/p07_parity.py C:/AMI/runs/p10/spell-probe/dota.log C:/AMI/runs/p10/spell-fixture.json reports/P10/lua-python-parity.json',
        'python gpu/scripts/p10_cast_parity.py',
        'python gpu/scripts/p10_static_parity.py C:/AMI/runs/p10/native-smoke4 C:/AMI/runs/p10/sim-static reports/P10/static-parity.json',
        'python gpu/scripts/p10_results.py --simulator C:/AMI/runs/p10/sim-v9/report.json --output reports/P10/simulator.json',
        'python gpu/scripts/p10_results.py --dota C:/AMI/runs/p10/native-eight --output reports/P10/dota.json',
        "$env:PYTHONPATH='src;scripts;tests;C:/AMI/tools/lua_test_runtime'; python -m unittest discover -s tests -v",
        'python gpu/scripts/p10_audit.py',
        'python gpu/scripts/p10_report.py','```',
        'Из `tools/nativecollect`: `go build -o C:/AMI/runs/p10/nativecollect.exe .`; `go test ./...`. Установка: `python scripts/install_play_bot.py --human-team 2`; восстановление: `python scripts/install_play_bot.py --restore`.']
    jobs=[json.loads(p.read_text()) for p in RUN.glob('native-eight*-job.json')]
    sim_job=json.loads((RUN/'sim-v9-job.json').read_text())
    lines += [f"Долгие процессы: Dota с продолжениями {sum(j['minutes'] for j in jobs):.3f} мин; 64 матча 005 {sim_job['minutes']:.3f} мин.",'',
        '## Что не получилось или не проверено','',
        '- Матч человека против работающей политики ещё не проверен (0/1): первый запуск остановился на setup. Графический клиент и настройки Steam агент не запускал/не менял. После исправления ожидается повтор пользователя.',
        '- Начальные неудачные smoke и прерванные подготовительные попытки сохранены. Четыре принятых матча не повторялись; серия продолжена до восьми. Разовый сбой чтения gameinfo.gi не воспроизведён, его причина не установлена.',
        '- CPU sampling перезапускался при resume; потоки RNG CPU/GPU не идентичны. Восемь матчей не устанавливают общую силу модели. Полный динамический вектор и траектории не объявляются идентичными.',
        '- Симулятор 005 штатно завершил 1/64 матч по T1; его правила не изменены. Счётчиков отклонённых команд симуляторный отчёт не содержит (0/64 охват).',
        '- Папка Steam при проверке оказалась обычным каталогом, а не указанным junction; тип и содержимое до установки сохранены. Тестовый junction также восстановлен без изменения ссылки.','',
        '## Пути к локальным артефактам','',
        '- `C:/AMI/runs/p10/ZOg-000470.pt`, `nativecollect.exe`; исходный checkpoint: `C:/AMI/runs/p09/P09c/continue90/rl-continue-001/checkpoint-000470`.',
        '- `C:/AMI/runs/p10/native-eight/match-00…07`: логи, protobuf, match.json; неудачные попытки — `attempts/`, прежние сервисы — `service-before-resume-*`.',
        '- `C:/AMI/runs/p10/{sim-v9,sim-static,spell-probe,cast-parity,terminal-probe,native-smoke*}`; статусы — `*-job.json` и `C:/AMI/runs/pilot-status.json`.',
        '- `C:/AMI/runs/p10/installation.json`, `bots-backup-<время>`, `restore-test/`, `restore-junction-test/`; будущие матчи человека — `human/<время>/`.', '', 'DONE P10']
    if human:
        assert Path(human['session']).name=='20261105-032741-652789','Behavior annotations describe the audited first human match'
        summary=human['summary'];model=summary['model_team'];winner=human['native_outcome']['winner'];t=human['transport']
        lines[4]=lines[4].replace('Матч человека: 0/1, точка ожидания пользователя.','Матч человека: 1/1 завершён, отдельно от автоматических серий.')
        result_lines=['Матч человека против ZOg: **first blood — Radiant (человек), 82,23 с (1:22,23)**. Native-конец 82,27 с, разница один тик.','',
            '| Участник / сторона | Победы / N | LH / N | Deny / N | Убийства / N | Смерти / N | Койлы / N | Койлы по герою: попадания / касты | Обычные попадания по герою / N |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for team in (2,3):
            s=summary['sides'][str(team)];name='ZOg' if team==model else 'Человек'
            values=[int(winner==team),s['last']['lh'],s['last']['deny'],s['kills'],s['deaths'],s['raze_casts']]
            result_lines.append(f"| {name} / {'Radiant' if team==2 else 'Dire'} | "+' | '.join(f'{v}/1' for v in values)+f" | {s['raze_hero_hits']}/{s['raze_casts']} | {s['hero_attack_hits']}/1 |")
        result_lines+=['',f"LH/deny на 5:00 и 10:00: у обоих отсутствуют, охват каждой отметки 0/1 — матч закончился раньше. Модель: ACK {t['acknowledged']}/{t['checked']}, отклонено {t['rejected']}/{t['checked']} (target_unavailable 3/4, purchase_failed 1/4), опоздания 0/{t['checked']}, пропуски 0/{t['checked']}; 1/1160 команда на границе смерти исключена. Все {human['service']['requests']}/1160 запросов CPU учтены, ошибок 0/1160. Команды Go для стороны человека не исполнялись и из анализа исключены.",
            'Проверка: пересчитанный итог равен сохранённому; LH/deny обеих сторон совпали с native; веса имеют прежний SHA-256. [Итог и события](P10/human-match.json), [все команды модели](P10/human-decisions.csv).','']
        at=lines.index('## Наблюдения');lines[at:at]=result_lines
        observation_lines=[
            '- **Подтверждено, человек–ZOg:** бот вышел на мид в −46,87 с. Из 28/1159 команд ATTACK: 14/28 на героя, 9/28 на вражеских крипов, 5/28 на deny. По герою нанесены 2 обычных попадания (60,60 и 80,50 с), суммарно 74 урона; отношение попаданий к командам не является точностью отдельных атак.',
            '- **Подтверждено, койлы:** четыре каста ZOg в 18,67 / 28,53 / 30,13 / 39,47 с; два дальних задели 1 и 3 крипов, два средних не дали событий урона. По человеку 0/4. После последнего каста мана 16–72 при стоимости койла 75; новых кастов не было.',
            '- **Подтверждено, добивания:** счёт ZOg изменился на deny 1 в 62,03 с и LH 1 в 65,83 с; к концу 1/1 LH и 1/1 deny. У человека 6/1 LH и 3/1 deny. Время изменения счёта измерялось с шагом около 0,1 с.',
            '- **Подтверждено, отход:** при HP<25% было 98/1159 решений: MOVE 85/98, атака героя 7/98, deny 1/98, STOP 3/98, CONTINUE 2/98. В 63/85 MOVE цель ближе к своей T1. Между 69,03 и 81,97 с расстояние до T1 уменьшилось 1134→444, HP 131/626→9/626; смерть наступила в 82,23 с.',
            '- **Подтверждено:** направление движения и выданные команды измерены; намерение бота и причинный вклад смены приказов, маршрута и реакции на урон не установлены. Один матч не определяет поведение против любого человека.','']
        at=lines.index('## Что сделано');lines[at:at]=observation_lines
        at=lines.index('python gpu/scripts/p10_report.py')
        lines.insert(at,'python gpu/scripts/p10_human_results.py C:/AMI/runs/p10/human/20261105-032741-652789 > C:/AMI/runs/p10/human-final-analysis.log')
        for i,line in enumerate(lines):
            if line.startswith('- Матч человека против работающей политики ещё'):
                lines[i]='- Человеческий путь проверен в 1/1 завершённом матче; 5:00/10:00 не достигнуты. Для этой отчётной правки выполнены проверки сырых данных и генератора; Python/Go-наборы выше относятся к ранее выполненным проверкам кода, повторно не запускались.'
            elif line.startswith('- Скрипты установлены:'):
                lines[i]='- Скрипты установлены: человек Radiant, модель Dire; [копия и хеши](P10/installation.json). Процессы завершённой сессии остановились; [проверка](P10/human-cleanup.json). Новый запуск пользователя и его клиент не прерывались. Heartbeat завершённой сессии приостановлен.'
            elif line.startswith('- `C:/AMI/runs/p10/installation.json`'):
                lines[i]='- `C:/AMI/runs/p10/installation.json`, `bots-backup-<время>`, тесты восстановления; матч человека — `C:/AMI/runs/p10/human/20261105-032741-652789/` (dota.log, native-decisions.pb.gz, summary.json, duel-result.json).'
    assert len(lines)<=150,len(lines)
    (ROOT/'reports/P10.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(f'{len(lines)} lines')


if __name__=='__main__':main()
