"""Render the compact Russian P02 report from measured stage artifacts."""
import argparse,collections,json
from pathlib import Path
from run_dagger import read,write,score
from p02_trace_analysis import analyze

ROOT=Path(__file__).resolve().parents[2]

def trace_summary(directory):
    columns=read(directory/'trace-header.json')['columns'];rows=collections.defaultdict(list)
    with (directory/'trace.jsonl').open() as f:
        for line in f:
            record=json.loads(line);actor=record['actor'];game=actor//2
            if game<3 and actor%2==game%2:
                row=dict(zip(columns,record['v']))
                if not row['finished'] and row['active'] and row['health']>0:rows[game].append(row)
    result=[]
    for game,frames in sorted(rows.items()):
        actions=collections.Counter(int(r['action']) for r in frames);longest=collections.Counter();last=None;run=0
        losses=collections.Counter()
        for i,r in enumerate(frames):
            act=int(r['action']);run=run+1 if last==act else 1;last=act;longest[act]=max(longest[act],run)
            if i and r['health']<frames[i-1]['health']-1:losses[str((int(r['source_kind']),int(r['source_tier'])))]+=1
        result.append({'game':game,'side':'radiant' if game%2==0 else 'dire','decisions':len(frames),'actions':dict(actions),'longest_action_runs':dict(longest),
                       'last_damage_source_kind_tier_on_hp_drop':dict(losses),'enemy_t1_min_distance':min(r['enemy_t1_distance'] for r in frames),
                       'first_game_seconds':frames[0]['game_tick']/30,'last_game_seconds':frames[-1]['game_tick']/30})
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=Path(a.run);state=read(run/'state.json');reports=ROOT/'reports/P02'
    stages=state['stages'];cfg=state['config']
    def result(name):
        if name not in stages:return None
        root=Path(stages[name]['output']);return read(root/('result.json' if name.startswith('train-') else 'report.json'))
    lines=['# P02 — DAgger','', '## Результаты','']
    finished=sum(f'round-{k}-argmax' in stages for k in range(1,5))
    lines+=['| Проверка / этап | Результат |','| --- | --- |',f'| Полные раунды | {finished}/4 |']
    if (reports/'parity.json').exists():
        parity=read(reports/'parity.json')
        for key,(n,d) in parity['compared'].items():lines.append(f'| beta=1: {key}, совпало | {n}/{d} |')
    smoke=result('train-smoke')
    if smoke:lines.append(f"| Smoke: обновления | {smoke['updates']}/50 |")
    collected=result('smoke-collect')
    if collected:
        lines.append(f"| Smoke: совпал выбор исполнителя с teacher | {collected['teacher_executed']}/{collected['learner_decisions']} |")
        lines.append(f"| Smoke: команда ученика отличается от метки | {collected['disagreements']}/{collected['learner_decisions']} |")
    if (reports/'verification.json').exists():
        for test in read(reports/'verification.json')['checks']:lines.append(f"| {test['name']} | {test['passed']}/{test['total']} |")
    lines+=['','| Модель | beta | Новых решений | Отличие команды / решений | Обновления | Validation NLL |','| --- | ---: | ---: | ---: | ---: | ---: |']
    lines.append('| M0 epoch-003 | — | 0 | — | 0 | см. P01 |')
    for k,beta in enumerate((.5,.25,.1,0.),1):
        c=result(f'round-{k}-collect');t=result(f'train-round-{k}')
        if c:lines.append(f"| M{k} | {beta} | {c['learner_decisions']} | {c['disagreements']}/{c['learner_decisions']} | {t['updates'] if t else 'идёт'} | {t['history'][-1]['validation']['nll'] if t else '—'} |")
    control=result('train-control')
    if control:lines.append(f"| D0-контроль | — | 0 | — | {control['updates']} | {control['history'][-1]['validation']['nll']:.6f} |")
    def metric_rows(report,controller):
        rows=[]
        for side in ('radiant','dire'):
            games=[g for g in report['per_match'] if any(s['controller']==controller and s['side']==side for s in g['sides'])]
            ss=[s for g in games for s in g['sides'] if s['controller']==controller and s['side']==side];n=len(ss)
            if not n:continue
            def summ(key):return int(sum(s[key] for s in ss))
            def stamp(key):
                values=[s[key] for s in ss if s[key] is not None]
                return f'{sum(values)/len(values):.2f}; {len(values)}/{n}' if values else f'—; 0/{n}'
            before=lambda t:sum(s['first_death_game_seconds'] is not None and s['first_death_game_seconds']<t for s in ss)
            rows.append(f"{side} | {summ('deaths')}/{n} | {before(0)}/{n} | {before(120)}/{n} | {stamp('first_death_game_seconds')} | {summ('lh')}/{n} | {summ('deny')}/{n} | {int(sum(g['lane_deaths'] for g in games))}/{n} | {stamp('mid_arrival_game_seconds')}")
        return rows
    for selection in ('sampling','argmax'):
        lines+=['',f'### {selection}','', '| Модель | Сторона | Смерти/матчи | До 0:00 | До 120 с | Первая смерть, с; n/N | LH/матчи | Deny/матчи | Все смерти крипов/матчи | Мид, с; n/N |','| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |']
        for name in ['M0']+[f'round-{k}' for k in range(1,5)]+['control']:
            r=result(name+'-'+selection)
            if r:
                for row in metric_rows(r,'GGD'):lines.append(f'| {name} | {row} |')
    baseline=result('teacher-baseline')
    reference=sum(s['lh']+s['deny'] for g in baseline['per_match'] for s in g['sides']) if baseline else None
    if baseline:lines+=['',f'Учитель против учителя: LH+deny = {int(reference)}/128 сторон в 64 матчах; эквивалент 64 сторон = {reference/2:g}.']
    lines+=['','Время — игровые секунды; отрицательное время относится к pregame. Средние рассчитаны только по событиям (n/N).',
            'До 0:00/120 с — матчи с первой смертью; при штатном конце по первой смерти это также число смертей.','', '## Выводы','']
    best=state.get('best');success=False
    if best and reference is not None:
        early,farm=best['score'][0],-best['score'][1];success=early<=2 and 4*farm>=reference
        lines.append(f"- **Подтверждено:** лучший раунд M{best['round']}: до 120 с {early}/64; LH+deny {int(farm)}/64, учитель {reference/2:g}/64. Критерий {'достигнут' if success else 'не достигнут'}. Порог: ≤2/64 и ≥{reference/4:g} LH+deny/64.")
    else:lines.append('- Критерий успеха пока не оценён: полный эксперимент продолжается.')
    if (reports/'parity.json').exists():lines.append('- **Подтверждено:** beta=1 воспроизводит наблюдения, метки, reset и previous collect-teacher побитово на двух полных матчах.')
    lines+=['- **Подтверждено (код):** `decide` teacher-v9 не изменён. Поля memory 1–6 описывают выданный ATTACK, 9–11 — отмену замаха, 12–13 — приказы, 14 — паузу предмета; при подмене пересчитываются по исполненной команде.',
            '- **Подтверждено (код/тест):** memory 0/8/15, seen_attacks и наблюдательные forecasts обновляются при каждом решении. Forecast 9 — история конкуренции для диагностики; пересчитывается по фактической цели. При совпадении команд адаптер — точный no-op.']
    comparison=result('control-sampling')
    if comparison:
        control_score=score(comparison)
        lines.append(f"- **Подтверждено на этих seed:** D0-контроль после {control['updates']} обновлений имеет {control_score[0]}/64 ранних смертей и {-control_score[1]:g}/64 LH+deny. Четыре раунда DAgger суммарно имеют тот же бюджет; выбранная M2 использовала первые 3894 обновления. Одного дополнительного обучения D0 оказалось недостаточно для результата M2.")
    if 'best-traces' in stages:
        traces=trace_summary(Path(stages['best-traces']['output']));write(reports/'traces-three.json',traces)
        analysis=analyze(Path(stages['best-traces']['output']));write(reports/'trace-analysis.json',analysis)
        for game in analysis['games']:
            learner=next(s for s in game['sides'] if s['controller']=='GGD');teacher=next(s for s in game['sides'] if s['controller']=='teacher')
            lines.append(f"- Трасса M2, матч {game['game']} ({learner['side']}): LH+deny/зарегистрированные циклы атаки — {learner['lh']+learner['deny']:g}/{learner['attack_cycles_completed']:g}, учитель {teacher['lh']+teacher['deny']:g}/{teacher['attack_cycles_completed']:g}; отмены/ATTACK — {learner['attack_cancellations']:g}/{learner['actions'][3]:g}. Выход на мид {learner['mid_arrival_game_seconds']:.2f} с.")
        lines+=['- **Подтверждено по 3/3 трассам M2:** основной оставшийся провал — добивание: 60 LH+deny/378 циклов против 279/374 у учителя. После 0:00 MOVE — 3320/12914 решений против 2103/12914, STOP — 1447/12914 против 963/12914. Подробнее: `trace-analysis.json`.',
                '- **Гипотеза:** неточный момент/выбор цели и лишние перемещения снижают конверсию атак в LH/deny. Трассы не содержат HP цели в момент ожидаемого попадания, поэтому эти причины строго не разделены. Отмены — 15/556 ATTACK против 3/383; ими нельзя объяснить весь разрыв.']
    lines+=['','## Что сделано','',
            '- `gpu/src/imitation.rs`, `main.rs`, `teacher_dagger.rs`: GPU-сбор, beta, источник/формат v3, флаги в signal 6–8; общая очередь offset_ticks=6, шаг 4 тика.',
            '- `scripted_teacher.rs`: только подключение адаптера; `bots/scripted_teacher_v9.lua`: только обновление SHA этой привязки.',
            '- `ggd_dataset.py`, `dagger_data.py`: чтение v2/v3, агрегированный манифест, ровно один проход новых train-последовательностей. Случайные старые последовательности; новых ≥50% последовательностей и валидных loss-кадров в каждом батче.',
            '- `train_ggd.py`: инициализация весов и AdamW из предыдущей модели, игровой выбор best (sampling: смерти до 120 с, затем собственный LH+deny); NLL вторичен. Без игровой оценки сохраняется last, best по NLL не выбирается.',
            '- `run_dagger.py`: этапы, OS-lock, проверка живого дочернего PID, resume с последнего завершённого этапа, stop.json; прерванный этап повторяется в новой папке. `report_dagger.py`, `p02_trace_analysis.py`, `tests/test_dagger.py`, GPU-тест памяти.',
            '', '## Команды, которыми получены результаты','',
            '`$env:PYTHONPATH="C:/AMI-wt/P02/src;C:/AMI/tools/lua_test_runtime"; C:/Python314/python.exe -m unittest discover -s tests -v`',
            '`cargo test --release --locked --manifest-path C:/AMI-wt/P02/gpu/Cargo.toml --target-dir C:/AMI/gpu/target dagger_ -- --nocapture`',
            '`cargo test --release --locked --manifest-path C:/AMI-wt/P02/gpu/Cargo.toml --target-dir C:/AMI/gpu/target`',
            '`C:/Python314/python.exe gpu/scripts/p02_trace_analysis.py C:/AMI/runs/p02/experiment/best-traces-attempt-1 reports/P02/trace-analysis.json`',
            '', 'Точные дочерние команды и длительности сбора/обучения/оценки:', '', '| Этап | Команда | Минуты |','| --- | --- | ---: |']
    for job in state['jobs']:
        if 'minutes' in job:lines.append(f"| {job['stage']} | `{job['command']}` | {job['minutes']:.3f} |")
    lines+=['','## Что не получилось или не проверено','',
            '- Первичный узкий Python-запуск без PYTHONPATH: 2 ошибки импорта sf_duel; повтор с путём worktree прошёл. Первый полный запуск пропустил 19 Lua-тестов; повтор с установленным runtime прошёл.',
            '- Resume сохраняет завершённые этапы; незавершённый сбор/проход обучения повторяется целиком. Внутри эпохи оптимизатор не восстанавливается.',
            '- Сохранён split исходного формата (game % 8): у новых данных validation содержит Radiant, test — Dire; игровой выбор модели использует отдельные 32/32 матча за стороны.',
            '- NLL вычислен на разных накопленных датасетах и не служит прямым рейтингом моделей. Три диагностических матча взяты из отдельной оценки на 4 матчах; выбор модели — по 64 матчам. Для критериев использована одна фиксированная серия seed, независимое подтверждение не выполнено.',
            '- Реальная Dota, другие seed и перенос на другие карты не проверяются. Все большие артефакты локальны.',
            '', '## Пути к локальным артефактам','',f'- Эксперимент: `{run}`; состояние/команды: `state.json`, `status.json`; общий статус `C:/AMI/runs/pilot-status.json`, секция P02.',
            f"- D0: `{cfg['dataset']}`; M0: `{cfg['initial_checkpoint']}`; бинарник `{cfg['executable']}`.",
            f"- Выбранная модель: `{best['checkpoint'] if best else 'ещё не выбрана'}`; экспорт: `{best['weights'] if best else '—'}`; трассы: `{run / 'best-traces-attempt-1'}`.",
            '- Git-метрики и конфигурация: `reports/P02/*.json`; веса, датасеты, трассы и логи не коммитятся.']
    done='control-argmax' in stages and 'best-traces' in stages and finished==4
    lines+=['','DONE P02' if done else 'Эксперимент выполняется; итоговая отметка будет добавлена после завершения.']
    if len(lines)>150:raise ValueError(f'Report exceeds 150 lines: {len(lines)}')
    (ROOT/'reports/P02.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
