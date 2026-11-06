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
    text=['# P16 — ночное обучение CHiO','','## Результаты','',f"Стадия: {state.get('stage','подготовка')}. Дедлайн UTC: {plan.get('deadline_utc')}. Контракт и ограничения P15 сохранены.",
          'Оценки: argmax, seed20261103, 128 матчей, поровну сторон; оба учителя через CHiO.',
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
    checks=read(REPORT/'full-verification.json')
    if checks:
        rust=re.search(r'(\d+) passed', '\n'.join(checks['full-rust-tail']));python=re.search(r'Ran (\d+) tests','\n'.join(checks['full-python-tail']))
        if rust and python:text.append(f"Полные тесты: Rust {rust[1]}/{rust[1]}, Python {python[1]}/{python[1]}.")
    text+=['','| План этапа | Ожидаемое завершение UTC (первоначальная оценка) |','|---|---|']
    text += [f"|{r['stage']}|{r['expected_end_utc']}|" for r in plan.get('estimates',[])]
    text+=['','| Чекпоинт / соперник | W/D/L | Смерти | LH+deny | Мид | Время выхода на мид, с |','|---|---|---|---|---|---|']
    for e in evaluations:
        if not e.get('complete'):continue
        d=read(ROOT/e['summary']);c=d['controllers']['CHiO'];n=c['actors'];o=c['outcomes'];a=c['mid_arrival_elapsed_seconds'];farm=c['lh']['numerator']+c['deny']['numerator']
        text.append(f"|{e['label']} / {e['opponent']}|{o.get('win',0)}/{n}; {o.get('draw',0)}/{n}; {o.get('loss',0)}/{n}|{c['deaths']['numerator']:g}/{n}|{farm:g}/{n}|{c['mid_arrivals']['numerator']}/{n}|{a['numerator']:.2f}/{a['denominator']}|")
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
           '| Оценки |sampling →argmax; задание; одинаковые128 матчей/seed и баланс сторон|',
           '| Инициализация |GGD000470+Adam →IL/DAgger CHiO; новый Adam для value warmup, затем перенос critic→PPO→self-play|',
           '| Desync/snapshot/checkpoint |вкл/60мин/10мин — те же; обучение учитывается отдельно от desync и оценки|',
           '| Карта/дерево/учитель |ограничения P15 сохранены; не исправлялись в P16|','',
           '## Наблюдения и изменения','',
           '- **Подтверждено:** поиск границы сбора103.3с:192/256/320 работают,352/384 исчерпали VRAM; дополнительная игровая проверка320 заняла114.6с.',
           '- Скорость пилота320:81920/21.121 решений/с; ранние терминалы4/1.910мин. Полная скорость — результаты этапов/commands.json.',
           '- `[P16] fix`: checkpoint IL каждые250 updates/эпоху; resume идентичен непрерывным3 updates:49/49 тензоров,3/3 loss.',
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
    assert len(text)<=150,len(text);(ROOT/'reports/P16.md').write_text('\n'.join(text)+'\n',encoding='utf-8',newline='\n')
    save(REPORT/'incidents.json',read(RUN/'incidents.json',[]))
    return text
if __name__=='__main__':refresh()
