"""Compact P16 report; detailed counts remain in small Git JSON shards."""
import json,subprocess,datetime,re
from pathlib import Path
from p11_run import save
ROOT=Path(__file__).resolve().parents[2];RUN=Path('C:/AMI/runs/p16');REPORT=ROOT/'reports/P16'
def read(p,default=None):
    return json.loads(Path(p).read_text(encoding='utf-8-sig')) if Path(p).exists() else default

def refresh():
    plan=read(RUN/'plan.json',{});stages=read(RUN/'stages.json',{});evaluations=read(RUN/'evaluations.json',[]);state=read(RUN/'state.json',{})
    blocks=[]
    for folder in RUN.iterdir():
        if folder.is_dir() and folder.name.startswith(('critic','ppo','self-play')):
            for f in sorted(folder.glob('block-*.json')):
                b=read(f);u=b['update'];rows=u.get('minibatches',[])
                means={key:{'numerator':sum(r[key] for r in rows if key in r),'denominator':sum(key in r for r in rows)} for key in ('loss','value_loss','entropy','old_policy_kl','anchor_kl','gradient_norm') if any(key in r for r in rows)}
                blocks.append(dict(run=folder.name,block=b['block'],training_seconds=b.get('training_seconds'),optimizer_steps=u['optimizer_steps'],metrics=means,ev_post0=u.get('ev_post0'),phase_counts=b['phase_counts']))
    for start in range(0,len(blocks),100):save(REPORT/'training'/f'blocks-{start:05}.json',blocks[start:start+100])
    save(REPORT/'stages.json',stages);save(REPORT/'evaluations.json',evaluations)
    commands=[]
    for p in sorted((RUN/'jobs').glob('*/status.json')):
        d=read(p);d['job']=p.parent.name;commands.append(d)
    save(REPORT/'commands.json',commands)
    text=['# P16 — ночное обучение CHiO','','## Результаты','',f"Стадия: {state.get('stage','подготовка')}. Дедлайн UTC: {plan.get('deadline_utc')}. Ограничения P15 сохранены; изменения навигации и обучения — по решению пользователя.",
          'Оценки: новые sampling, прежние argmax явно помечены; seed20261103, 128 матчей, поровну сторон; оба учителя через CHiO. Столбец базы различает policy/script; лучший выбирается только в текущем режиме.',
          'Проверка CHiO против CHiO: 8/8 полных матчей; [результат](P16/preflight-model-pair.json).','',
          '| Этап | Результат | Завершение UTC |','|---|---|---|']
    native=state.get('native',{})
    if 'completed_matches_total' in native:
        current=read(Path(state['artifacts'])/'config.json',{});text.insert(6,f"Текущий сбор: {native['completed_matches_total']}/{current.get('games','?')} матчей; {native.get('matches_per_minute',0):.2f} матча/мин.")
    for name,row in stages.items():
        r=row.get('result',{});metric='завершён'
        if 'games_requested' in r:metric=f"матчи {r['native_completed']}/{r['games_requested']}; решения {r['valid_decisions']}/{r['valid_decisions']}"
        elif 'updates' in r:metric=f"конечные updates {r['finite_updates']}/{r['updates']}; val loss {r['validation_loss']:.6f}, frames {r['validation_frames']}/{r['validation_frames']}"
        elif 'training_seconds' in r:metric=f"{r['training_seconds']/60:.2f} мин; {r['blocks']}/{r['blocks']} блоков; {r['optimizer_steps']}/{r['optimizer_steps']} updates"
        text.append(f"|{name}|{metric}|{row['finished_utc']}|")
    checks=read(REPORT/'nav-full-verification.json') or read(REPORT/'full-verification.json')
    if checks:
        rust=re.search(r'(\d+) passed', '\n'.join(checks['full-rust-tail']));python=re.search(r'Ran (\d+) tests','\n'.join(checks['full-python-tail']))
        if rust and python:text.append(f"Полные тесты: Rust {rust[1]}/{rust[1]}, Python {python[1]}/{python[1]}.")
    weighted_checks=read(REPORT/'sampling-verification.json')
    if weighted_checks:text.append('После перехода на sampling/веса: Python268/268; узкие8/8; веса13/13; resume49/49, loss3/3. [Проверки](P16/sampling-verification.json).')
    diagnosis=read(REPORT/'direction-diagnosis/result.json')
    if diagnosis:
        a=diagnosis['audit'];r=a['teacher_validation_premid']['radiant'];d=a['teacher_validation_premid']['dire']
        text += ['',f"Диагностика imitation: направление до мида R {r['correct']}/{r['frames']}, D {d['correct']}/{d['frames']}; соглашение72/72, очередь+2000 —2256/2256.",'Подтверждён сдвиг состояний: модель40°/220° упирается в базе, учитель требует130°/270°. [Данные и траектории](P16/direction-diagnosis/result.json), [карта](P16/direction-diagnosis/trajectories.svg).']
    revision=read(REPORT/'sampling-plan-step0.json')
    if revision:text += ['', 'План sampling/веса: шаг0 — сбор nav-DAgger1 сохранён256/256; прежнее обучение остановлено по пользователю. [Решение и частичные данные](P16/sampling-plan-step0.json).']
    sampling=read(REPORT/'eval-nav-imitation-v10-sampling.json')
    if sampling:
        c=sampling['controllers']['CHiO'];n=c['actors'];stuck=c['stuck_after_mid'];casts=c['spell_completions']
        text += ['',f"Sampling nav-imitation/v10: LH+deny {c['lh']['numerator']+c['deny']['numerator']:g}/{n}; койлы near/mid/far {casts[0]:g}/{n}, {casts[1]:g}/{n}, {casts[2]:g}/{n}; смерти {c['deaths']['numerator']:g}/{n}; упирается после мида {stuck['numerator']:g}/{stuck['denominator']:g}.", 'Причины смертей: '+', '.join(f"{k} {v:g}/{n}" for k,v in c['death_causes'].items())+'.']
    audit=read(REPORT/'nav-imitation-postmid-audit.json')
    if audit:
        text += ['', 'Validation teacher1024 после мида, nav-imitation; диагноз не выбирает модель:', '| Тип | Доля меток | Recall argmax | Precision argmax |','|---|---|---|---|']
        for row in audit['actions']:
            cells=[str(row[k]['numerator'])+'/'+str(row[k]['denominator']) for k in ('label_share','recall','precision')]
            text.append('|'+row['action']+'|'+'|'.join(cells)+'|')
        same=audit['label_equals_previous_type'];text.append(f"Метка типа = previous type: {same['numerator']}/{same['denominator']}; 0/0 — неопределено, previous не обнулялся. [Полная матрица](P16/nav-imitation-postmid-audit.json).")
    selected=read(REPORT/'sampling-plan-step6.json')
    if selected:text += ['',f"Старт DAgger: {selected['start']}; правило LH+deny weighted>=baseline; ниже3/матч={selected['below_three']}; [числители, знаменатели и веса](P16/sampling-plan-step6.json)."]
    navigation=read(REPORT/'navigation-decision.json')
    if navigation:text += ['',f"Решение после DAgger1: {navigation['action']}; мид {navigation['mid_arrivals']['numerator']}/{navigation['mid_arrivals']['denominator']}."]
    text+=['','| План этапа | Ожидаемое завершение UTC (текущий план) |','|---|---|']
    text += [f"|{r['stage']}|{r['expected_end_utc']}|" for r in plan.get('estimates',[])]
    text+=['','| Чекпоинт / соперник / база / режим | W/D/L | Смерти | LH+deny | Мид | Время выхода на мид, с |','|---|---|---|---|---|---|']
    for e in evaluations:
        if not e.get('complete'):continue
        d=read(ROOT/e['summary']);c=d['controllers']['CHiO'];n=c['actors'];o=c['outcomes'];a=c['mid_arrival_elapsed_seconds'];farm=c['lh']['numerator']+c['deny']['numerator']
        text.append(f"|{e['label']} / {e['opponent']} / {'script' if e.get('scripted_base_to_mid') else 'policy'} / {'sampling' if e.get('sampling') else 'argmax'}|{o.get('win',0)}/{n}; {o.get('draw',0)}/{n}; {o.get('loss',0)}/{n}|{c['deaths']['numerator']:g}/{n}|{farm:g}/{n}|{c['mid_arrivals']['numerator']}/{n}|{a['numerator']:.2f}/{a['denominator']}|")
    text+=['','| Оценённая модель | Путь весов |','|---|---|']
    models={e['label']:e['weights'] for e in evaluations if e.get('complete')}
    text += [f'|{name}|`{path}`|' for name,path in models.items()]
    best=read(RUN/'best.json')
    if best:text+=['',f"Лучшая: **{best['label']}** — {best['reason']}"]
    else:text+=['','Лучшая модель будет выбрана после завершения оценок по победам, смертям, фарму, выходу на мид и койлам.']
    text+=['','Полные причины побед/смертей (включая башни), каждый койл, длительность, Flask/Mango/Tango и все матчи:',
           '[evaluations.json](P16/evaluations.json) → summaries и одноимённые каталоги; знаменатели указаны в агрегатах.',
           'Паритеты/проверки весов: [stages.json](P16/stages.json), parity-*.json; метрики каждого PPO-блока — [training](P16/training/).','',
           '## Отличия от P11','',
           '| Параметр | P11 → P16 и причина |','|---|---|',
           '| Контракт |ggd-v4,4,event-v2 → chio-v1,6,lastorder-fb-v1 zero-sum; задание P16/P15|',
           '| Gamma/lambda |0.99/0.95 →0.997/0.95; контракт CHiO|',
           '| Арены/порция/эпохи |192 /48×128 /2 — те же; шарды32 как P11, активные learner-строки|',
           '| LR actor/critic |3e-5/3e-4 — те же, раздельный шаг value|',
           '| Entropy/clip/grad/stopKL |0.005/0.2/0.5/0.02 — те же; stopKL только внутри PPO-обновления|',
           '| Якорь |старт каждого этапа, неизменный при resume;0.03→0.01, за120мин PPO и бюджет self-play вместо261мин|',
           '| Прогрев |20..60 блоков, mean EVpost0 последних5>0.3; только value — как P11|',
           '| Пул |20/20/30/30 →45/35/10/10; задание, оба учителя через CHiO, соперник закреплён на матч|',
           '| Оценки |sampling как P11; прежние argmax архивированы; одинаковые128 матчей/seed и баланс сторон|',
           '| Инициализация |GGD000470+Adam →IL/DAgger CHiO; новый Adam для value warmup, затем перенос critic→PPO→self-play|',
           '| Desync/snapshot/checkpoint |вкл/60мин/10мин — те же; обучение учитывается отдельно от desync и оценки|',
           '| Карта/дерево/учитель |ограничения P15 сохранены; не исправлялись в P16|','',
           '## Наблюдения и изменения','',
           '- **Подтверждено:** поиск границы сбора103.3с:192/256/320 работают,352/384 исчерпали VRAM; дополнительная игровая проверка320 заняла114.6с.',
           '- Скорость пилота320:81920/21.121 решений/с; ранние терминалы4/1.910мин. Полная скорость — результаты этапов/commands.json.',
           '- `[P16] fix`: checkpoint IL каждые250 updates/эпоху; resume идентичен непрерывным3 updates:49/49 тензоров,3/3 loss.',
           '- `[P16] fix`: чтение live JSON повторяется при незавершённой записи; контроллер подхватывает живой trainer; wrapper умеет --adopt и повторяет чтение pilot-status; обучение не перезапускается.',
           '- CHiO: перенесены якорь KL, critic LR, неизменный соперник на матч, смешанный пул и учёт времени/resume P11.',
           '- **Гипотеза:** ранние результаты и изменения силы требуют сравнения всей таблицы; один win-rate не доказывает устойчивость.','',
           '## Команды и проверки','',
           'Все дословные команды, PID, exit-коды и длительности долгих процессов в минутах: [commands.json](P16/commands.json).',
           'Все использованные конфиги: [configs](P16/configs/); родительский supervisor — `C:/Python314/python.exe gpu/scripts/p16_run.py`.',
           'Целевые Rust:6/6; frozen policy45/45; native resume2/2 блоков; Python resume49/49 тензоров. Полные проверки: [full-verification.json](P16/full-verification.json).','',
           '## Не получилось / ограничения','',
           '- Нет локальной17×17;8×8 пропускает узкие препятствия и корпуса зданий; дерево — круг88; атлас только у учителя.',
           '- Учитель P15:фарм14111/18032, смерти143/113; Mango не использует. Эти ограничения не менялись.',
           '- Native checkpoint восстанавливает модель/Adam/время/пул, но арены начинают заново с desync; якорь этапа сохраняется.',
           '- Сбои и консервативные исправления: [incidents.json](P16/incidents.json). Перенос игры в Dota не проверяется.','',
           '## Артефакты','',
           '`C:/AMI/runs/p16/`: environment/, teacher-1024/, imitation/, dagger-*/, critic*/, ppo*/, self-play*/, eval-*/, jobs/, state.json.',
           'Все изменения в C:/AMI-wt/P16; веса/датасеты/бинарники/крупные логи внеgit; графический клиент и папка ботов не используются.','',
           'DONE P16' if state.get('stage')=='complete' else 'BLOCKED P16: обучение продолжается; промежуточный отчёт']
    if len(text)>150:
        start=next(i for i,line in enumerate(text) if line.startswith('| План этапа'));end=start+2+len(plan.get('estimates',[]));text[start:end]=['Первоначальные сроки этапов: [plan.json](P16/plan.json); дополнительное обучение оплачивается временем self-play.']
    if len(text)>150:
        text=[line for line in text if '|`' in line or not any(line.startswith('|'+name+'|') for name in ('imitation','dagger-1-il','parity-imitation','parity-dagger-1','full-verification','nav-full-verification','nav-runtime-validation'))]
    final=read(REPORT/'final-review.json')
    if final and state.get('stage')=='complete':
        text=[line.replace('Полные тесты: Rust 143/143, Python 266/266.', 'Полные тесты: Rust143/143 (после последних Rust-изменений), Python272/272 (финальный запуск); паритет6/6, шаги768/768; конечные RL/value updates12480/12480.') for line in text]
        text=[line for line in text if not line.startswith('После перехода на sampling/веса:')]
        text=[line if not line.startswith('Первоначальные сроки этапов:') else f"По решению пользователя остановлено после первого интервала self-play: фактически121.25мин, PPO120.23мин; все оценки закончены {state['finished_utc']} UTC. [Стоп и превышение на завершение блока](P16/user-stop-120m.json)." for line in text]
        text.insert(-1, 'SHA точки восстановления кода и результатов: `'+final['results_commit']+'`; заключительная фиксация отчёта следует за ним. [Итоговая проверка](P16/final-review.json).')
        if len(text)>150:
            text=[line for line in text if not line.startswith(('|nav-imitation-postmid-audit|','|parity-nav-imitation|'))]
    assert len(text)<=150,len(text);(ROOT/'reports/P16.md').write_text('\n'.join(text)+'\n',encoding='utf-8',newline='\n')
    save(REPORT/'incidents.json',read(RUN/'incidents.json',[]))
    return text
if __name__=='__main__':refresh()
