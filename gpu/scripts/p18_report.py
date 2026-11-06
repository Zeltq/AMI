"""P18 compact Russian report; full ratios and match shards accompany every evaluation."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path('C:/AMI/runs/p18');REPORT=ROOT/'reports/P18'
def read(p,default=None):
    return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else default
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def row(label,opponent,d):
    c=d['controllers']['CHiO'];n=c['actors'];sec=d['duration_game_seconds']['numerator'];mins=sec/60
    ratio=lambda field:f"{c[field]['numerator']:g}/{mins:.4f}"
    casts='; '.join(f"{c['raze_hero_hit_rates'][k]['numerator']:g}/{c['raze_hero_hit_rates'][k]['denominator']:g}" for k in ['near','mid','far'])
    outcomes='; '.join(f"{c['outcomes'].get(k,0)}/{n}" for k in ['win','draw','loss'])
    return f"|{label} / {opponent}|{outcomes}|{c['deaths']['numerator']:g}/{n}|{ratio('lh')}|{ratio('deny')}|{ratio('hero_attacks_released')}|{casts}|{sec:.3f}/{n}|"
def refresh():
    REPORT.mkdir(exist_ok=True);state=read(RUN/'state.json',{});plan=read(RUN/'plan.json',{});stages=read(RUN/'stages.json',{});entries=read(RUN/'evaluations.json',[])
    for name in ['state','plan','stages','evaluations','incidents']:
        value=read(RUN/f'{name}.json');
        if value is not None:save(REPORT/f'{name}.json',value)
    training=REPORT/'training';training.mkdir(exist_ok=True)
    for folder in RUN.iterdir():
        if folder.is_dir() and folder.name.startswith(('critic','ppo','self-play')):
            target=training/folder.name;target.mkdir(exist_ok=True)
            for block in folder.glob('block-*.json'):
                destination=target/block.name
                if not destination.exists():destination.write_bytes(block.read_bytes())
    commands={p.parent.name:read(p) for p in (RUN/'jobs').glob('*/status.json')};save(REPORT/'commands.json',commands)
    lines=['# P18 — CHiO, lastorder-fb-v2','',f"Стадия: {state.get('stage','подготовка')}. Начало критика UTC: {plan.get('start_critic_utc','ещё не начат')}. Дедлайн UTC: {plan.get('deadline_utc','начало критика + 12 часов')}.",'','Награда: kills +0.7, deaths −2.2, outcome 0; прочие компоненты v1 сохранены.','Навигация: C:/AMI/runs/p16/environment-script-navigation/engine.json; до передачи управления решения исключены из обучения.','Sampling во всех запусках; оценки 128 матчей, seed 20261103, swap. Первая previous — политика после прогрева критика.','Оценки идут в отдельной последовательной очереди параллельно обучению (P16 parallel probe: 2/2 процессов). training_seconds учитывает только блоки обучения. Self-play останавливается по UTC-дедлайну на границе блока; финальные оценки завершаются после него.','','| Этап | Время обучения, с / мин | Блоки | Updates |','|---|---|---|---|']
    for name,stage in stages.items():
        r=stage['result'];lines.append(f"|{name}|{r['training_seconds']:.3f}/60|{r['blocks']}/{r['blocks']}|{r['optimizer_steps']}/{r['optimizer_steps']}|")
    freeze=read(REPORT/'critic-freeze.json')
    if freeze:lines+=['',f"Неизменность политики: {freeze['equal']}/{freeze['total']} тензоров по SHA256."]
    lines+=['','Проверки: [verification.json](P18/verification.json). Сбои и восстановления: [incidents.json](P18/incidents.json).','Все команды и PID: [commands.json](P18/commands.json); конфиги: [configs](P18/configs).','','Каждая частота: счётчик / суммарные игровые минуты; pregame не вычитается. Койлы: попадания / касты near; mid; far.','','| Модель / соперник | W; D; L | Смерти | LH/мин | deny/мин | Атаки героя/мин | Койлы | Длительность, с/матчи |','|---|---|---|---|---|---|---|---|---|']
    for label in ['sampling-dagger-2','ppo-120m','self-play-120m']:
        for opponent in ['v10','v9']:
            p=ROOT/f'reports/P16/eval-{label}-{opponent}-sampling.json'
            if p.exists():lines.append(row('P16 '+label,opponent,read(p)))
    for e in entries:lines.append(row('P18 '+e['label'],e['opponent'],read(ROOT/e['summary'])))
    lines+=['','Причины исходов и смертей, касты/попадания, все матчи: [evaluations.json](P18/evaluations.json) → summary и одноимённые каталоги. Все счётчики причин делятся на actors соответствующей группы.','','| Модель | Веса |','|---|---|']
    for label in dict.fromkeys(e['label'] for e in entries):lines.append('|'+label+'|`'+next(e['weights'] for e in entries if e['label']==label)+'`|')
    if state.get('stage')=='complete':
        candidates=[]
        for label in dict.fromkeys(e['label'] for e in entries):
            es=[e for e in entries if e['label']==label];cs=[read(ROOT/e['summary']) for e in es];teachers=[d for e,d in zip(es,cs) if e['opponent'] in ['v10','v9']]
            if len(teachers)!=2:continue
            n=sum(d['controllers']['CHiO']['actors'] for d in cs);wins=sum(d['controllers']['CHiO']['outcomes'].get('win',0) for d in cs);deaths=sum(d['controllers']['CHiO']['deaths']['numerator'] for d in cs);minutes=sum(d['duration_game_seconds']['numerator']/60 for d in teachers)
            farm=sum(d['controllers']['CHiO'][k]['numerator'] for d in teachers for k in ['lh','deny']);hits=sum(v['numerator'] for d in cs for v in d['controllers']['CHiO']['raze_hero_hit_rates'].values());casts=sum(v['denominator'] for d in cs for v in d['controllers']['CHiO']['raze_hero_hit_rates'].values())
            candidates.append(dict(label=label,weights=es[0]['weights'],ratios=dict(wins=[wins,n],deaths=[deaths,n],farm=[farm,minutes],hits=[hits,casts]),metrics=[wins/n,-deaths/n,farm/minutes,hits/max(1,casts)]))
        for c in candidates:c['score']=sum(sum(o['metrics'][i]<v for o in candidates) for i,v in enumerate(c['metrics']))
        best=max(candidates,key=lambda c:c['score']);save(REPORT/'best.json',dict(best=best,candidates=candidates));lines+=['',f"Лучшая по сумме рангов побед, выживания, фарма/мин и точности койлов без порога: **{best['label']}**. Все соперники учитываются в победах/смертях/койлах; фарм сравнивается против v10/v9. [Все кандидаты и дроби](P18/best.json)."]
    lines+=['','Ограничения: арены при resume запускаются заново; веса, Adam, якорь, пул и время восстанавливаются. Перенос в Dota не проверяется.']
    lines+=['DONE P18' if state.get('stage')=='complete' else 'BLOCKED P18: выполнение продолжается; итог ещё не готов']
    assert len(lines)<=150,len(lines)
    (ROOT/'reports/P18.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':refresh()
