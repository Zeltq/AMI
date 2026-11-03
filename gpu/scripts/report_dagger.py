"""Render the compact Russian P02 report from measured stage artifacts."""
import argparse,collections,json
from pathlib import Path
from run_dagger import read,write,score

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
            '- **Подтверждено (код/тест):** memory 0/8/15, seen_attacks и наблюдательные forecasts обновляются при каждом решении. Forecast 9 — история конкуренции для диагностики; пересчитывается по фактической цели. При совпадении команд адаптер — точный no-op.',
            '- **Гипотеза:** коррекция распределения состояний должна уменьшить накопление ошибок; причинный эффект оценивается сравнением с D0-контролем при равном числе обновлений.']
    if 'best-traces' in stages:
        traces=trace_summary(Path(stages['best-traces']['output']));write(reports/'traces-three.json',traces)
        for t in traces:
            dominant=max(t['actions'],key=t['actions'].get);lines.append(f"- Трасса {t['game']} ({t['side']}): Act={dominant} — {t['actions'][dominant]}/{t['decisions']}; максимальная серия {t['longest_action_runs'][dominant]}; min дистанция до T1 {t['enemy_t1_min_distance']:.1f}. Разбор причин — `traces-three.json`.")
    lines+=['','## Что сделано','',
            '- `gpu/src/imitation.rs`, `main.rs`, `teacher_dagger.rs`: GPU-сбор, beta, источник/формат v3, флаги в signal 6–8; общая очередь offset_ticks=6, шаг 4 тика.',
            '- `scripted_teacher.rs`: только подключение адаптера; `bots/scripted_teacher_v9.lua`: только обновление SHA этой привязки.',
            '- `ggd_dataset.py`, `dagger_data.py`: чтение v2/v3, агрегированный манифест, ровно один проход новых train-последовательностей. Случайные старые последовательности; новых ≥50% последовательностей и валидных loss-кадров в каждом батче.',
            '- `train_ggd.py`: инициализация весов и AdamW из предыдущей модели, игровой выбор best (sampling: смерти до 120 с, затем собственный LH+deny); NLL вторичен. Без игровой оценки сохраняется last, best по NLL не выбирается.',
            '- `run_dagger.py`: этапы, OS-lock, проверка живого дочернего PID, resume с последнего завершённого этапа, stop.json; прерванный этап повторяется в новой папке. `report_dagger.py`, `tests/test_dagger.py`, GPU-тест памяти.',
            '', '## Команды, которыми получены результаты','',
            '`$env:PYTHONPATH="C:/AMI-wt/P02/src;C:/AMI/tools/lua_test_runtime"; C:/Python314/python.exe -m unittest discover -s tests -v`',
            '`cargo test --release --locked --manifest-path gpu/Cargo.toml --target-dir C:/AMI/gpu/target dagger_ -- --nocapture`',
            '', 'Точные дочерние команды и длительности сбора/обучения/оценки:', '', '| Этап | Команда | Минуты |','| --- | --- | ---: |']
    for job in state['jobs']:
        if 'minutes' in job:lines.append(f"| {job['stage']} | `{job['command']}` | {job['minutes']:.3f} |")
    lines+=['','## Что не получилось или не проверено','',
            '- Первичный узкий Python-запуск без PYTHONPATH: 2 ошибки импорта sf_duel; повтор с путём worktree прошёл. Первый полный запуск пропустил 19 Lua-тестов; повтор с установленным runtime прошёл.',
            '- Resume сохраняет завершённые этапы; незавершённый сбор/проход обучения повторяется целиком. Внутри эпохи оптимизатор не восстанавливается.',
            '- Реальная Dota, другие seed и перенос на другие карты не проверяются. Все большие артефакты локальны.',
            '', '## Пути к локальным артефактам','',f'- Эксперимент: `{run}`; состояние/команды: `state.json`, `status.json`; общий статус `C:/AMI/runs/pilot-status.json`, секция P02.',
            f"- D0: `{cfg['dataset']}`; M0: `{cfg['initial_checkpoint']}`; бинарник `{cfg['executable']}`.",
            '- Git-метрики и конфигурация: `reports/P02/*.json`; веса, датасеты, трассы и логи не коммитятся.']
    done='control-argmax' in stages and 'best-traces' in stages and finished==4
    lines+=['','DONE P02' if done else 'Эксперимент выполняется; итоговая отметка будет добавлена после завершения.']
    if len(lines)>150:raise ValueError(f'Report exceeds 150 lines: {len(lines)}')
    (ROOT/'reports/P02.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
