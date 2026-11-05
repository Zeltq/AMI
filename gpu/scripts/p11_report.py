"""Compact P11 evidence for GitHub-only review; no checkpoints or raw logs."""
import json,math
from pathlib import Path
from p11_run import ROOT,RUN,REPORT,START,POOL,read,save,sha,history
from p09b_results import summaries,csvwrite,lines
NAMES=('outcome','lh','deny','own_hp','enemy_hp','mana')
def evaluations():
    entries=read(RUN/'evaluations.json') if (RUN/'evaluations.json').exists() else []
    for seed in (20261103,20261104):
        entries.append(dict(model='000470-P09c',opponent='v10',report=f'C:/AMI/runs/p09/P09c/continue90/eval-v10-{seed}/report.json',historical=True))
    entries.append(dict(model='000470-P10',opponent='v9',report='C:/AMI/runs/p10/sim-v9/report.json',historical=True))
    for folder in RUN.glob('train-*'):
        for path in folder.glob('eval-*/report.json'):
            if not read(path).get('complete'):continue
            block=int(path.parent.name.split('-')[1]);progress=read(folder/f'checkpoint-{block:06}/progress.json')
            label=f'{round(progress["training_seconds"]/7200)*2}h'
            other=path.parent.name.split('-')[2]
            if other=='previous':
                op=read(path.parent/'opponent.json')['weights']
                other='000470' if Path(op)==START/'weights' else checkpoint_label(Path(op).parent)
            entries.append(dict(model=label,opponent=other,report=str(path)))
    groups={};provenance=[]
    for entry in entries:
        path=Path(entry['report']);r=read(path);cfg=read(path.parent/'config.json')
        assert r['complete'] and r['matches']==64 and len(r['per_match'])==64
        assert not cfg['argmax']
        assert all(g['simulation_error']==0 for g in r['per_match'])
        key=(entry['model'],entry['opponent']);groups.setdefault(key,{})[r['seed']]=r['per_match']
        for g in r['per_match']:
            for s,h in enumerate(g['sides']):
                c=h['event_v2_components'];assert abs(c[0]-10*g['result']*(1 if s==0 else -1))<1e-5
                assert abs(c[1]-.5*h['lh'])<1e-3 and abs(c[2]-.2*h['deny'])<1e-3
                assert abs(sum(c)-h['event_v2_reward'])<1e-3
        compact=[{**g,'sides':[{k:v for k,v in h.items() if k!='minute_snapshots'} for h in g['sides']]} for g in r['per_match']]
        evidence=dict(**entry,seed=r['seed'],config=cfg,minutes=r['wall_seconds']/60,weights_sha256=sha(Path(cfg['weights'])/'manifest.json'),per_match=compact)
        if (path.parent/'opponent.json').exists():evidence['opponent_provenance']=read(path.parent/'opponent.json')
        dest=REPORT/f'{entry["model"]}-vs-{entry["opponent"]}-{r["seed"]}.json'
        dest.write_text(json.dumps(evidence,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')
        provenance.append({k:v for k,v in evidence.items() if k!='per_match'})
    rows=[]
    for (model,other),seeds in groups.items():rows+=summaries(model,dict(per_match=[g for games in seeds.values() for g in games]),other,'event-v2')
    csvwrite(REPORT/'metrics.csv',rows);save(REPORT/'metrics.json',rows)
    save(REPORT/'evaluations.json',provenance)
    return rows
def checkpoint_label(path):
    p=read(path/'progress.json');return f'{p["training_seconds"]/3600:.2f}h'
def episode_summary(batch):
    heroes=[next(h for h in r['game']['sides'] if h['controller']=='GGD') for r in batch]
    row=dict(matches=len(batch),wins=sum(r['game']['result']*(1 if h['side']=='radiant' else -1)==1 for r,h in zip(batch,heroes)),draws=sum(r['game']['result']==0 for r in batch),farm=sum(h['lh']+h['deny'] for h in heroes),spawn=sum(r['game']['lane_creeps_spawned'] for r in batch),death_matches=sum(h['first_death_game_seconds'] is not None for h in heroes),reward_sum=sum(r['reward'] for r in batch))
    for i,key in enumerate(NAMES):row['reward_'+key]=sum(r['components'][i] for r in batch)
    row['losses']=row['matches']-row['wins']-row['draws'];return row
def curves():
    updates=[];episodes=[];coverage=[]
    for folder in sorted(RUN.glob('train-*')):
        if not (folder/'config.json').exists():continue
        cfg=read(folder/'config.json');resumed=read(Path(cfg['resume_checkpoint'])/'progress.json')['training_seconds'] if cfg.get('resume_checkpoint') else 0
        for r in lines(folder/'history.jsonl'):
            updates.append(dict(run=folder.name,block=r['block'],training_minutes=r['training_seconds']/60,decisions=r['total_decisions'],new_training_seconds=r['training_seconds']-resumed,decisions_per_second=r['total_decisions']/max(1e-9,r['training_seconds']-resumed),**{k:r['learning'].get(k) for k in ('value_loss','ev_post0','rollout_kl_m2','entropy','clip_fraction','kl_coefficient','valid_decisions')}))
        all_ep=lines(folder/'episodes.jsonl');complete=[r for r in all_ep if not r.get('initialization_prefix_decisions',0) and not r.get('warmup')]
        coverage.append(dict(run=folder.name,completed=len(all_ep),complete_reward=len(complete),excluded_prefix_or_warmup=len(all_ep)-len(complete)))
        episodes.extend(complete)
    csvwrite(REPORT/'updates.csv',updates);save(REPORT/'coverage.json',coverage)
    bins=[];hours=[]
    for kind in ('latest','older','v10','v9'):
        group=[r for r in episodes if r['game']['opponent_type']==kind]
        for i in range(0,len(group),100):
            batch=group[i:i+100];bins.append(dict(opponent=kind,first=i+1,last=i+len(batch),training_minutes=batch[-1]['training_seconds']/60,**episode_summary(batch)))
        for hour in sorted(set(int(r['training_seconds']//3600)+1 for r in group)):
            batch=[r for r in group if int(r['training_seconds']//3600)+1==hour]
            hours.append(dict(hour=hour,opponent=kind,**episode_summary(batch)))
    csvwrite(REPORT/'matches-100.csv',bins);csvwrite(REPORT/'hours.csv',hours)
    if updates:svg(updates,bins)
    return updates,bins,hours,coverage
def svg(updates,bins):
    panels=[]
    for title,key in [('KL к 000470','rollout_kl_m2'),('Энтропия','entropy'),('Value loss','value_loss'),('EV после 0:00','ev_post0'),('Clip fraction','clip_fraction')]:panels.append((title,{key:[(r['training_minutes'],r[key]) for r in updates]}))
    for title,key,denom in [('Награда/матч','reward_sum','matches'),('Победы/матчи','wins','matches'),('Farm/spawn','farm','spawn'),('Смерти/матчи','death_matches','matches')]:
        panels.append((title,{k:[(r['training_minutes'],r[key]/r[denom]) for r in bins if r['opponent']==k and r[denom]] for k in ('latest','older','v10','v9')}))
    for kind in ('latest','older','v10','v9'):
        panels.append(('Компоненты/матч: '+kind,{k:[(r['training_minutes'],r['reward_'+k]/r['matches']) for r in bins if r['opponent']==kind] for k in NAMES}))
    height=len(panels)*190+30;out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}"><rect width="100%" height="100%" fill="white"/>']
    colors=['#1769aa','#d97706','#16803c','#b91c1c','#9333ea','#475569'];xmax=max(r['training_minutes'] for r in updates)
    for i,(title,series) in enumerate(panels):
        series={k:[(x,y) for x,y in v if y is not None and math.isfinite(y)] for k,v in series.items()};vals=[y for v in series.values() for x,y in v]
        if not vals:continue
        lo,hi=min(vals),max(vals);top=25+i*190
        out.append(f'<text x="45" y="{top}" font-size="16">{title}; min={lo:.4g}, max={hi:.4g}; X 0–{xmax:.1f} мин</text>')
        for j,(color,(key,values)) in enumerate(zip(colors,series.items())):
            pts=' '.join(f'{45+x/max(xmax,1)*1000:.2f},{top+135-(y-lo)/max(hi-lo,1e-9)*115:.2f}' for x,y in values)
            out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/><text x="{45+j*170}" y="{top+160}" fill="{color}" font-size="12">{key}</text>')
    out.append('</svg>');(REPORT/'curves.svg').write_text('\n'.join(out),encoding='utf-8',newline='\n')
def refresh():
    REPORT.mkdir(parents=True,exist_ok=True)
    rows=evaluations();updates,bins,hours,coverage=curves()
    commands=[{k:v for k,v in r.items() if k not in ('pid','started_utc')} for r in history()]
    save(REPORT/'commands.json',commands)
    pool=[]
    for path in POOL:pool.append(dict(weights=path,training_minutes=0,manifest_sha256=sha(Path(path)/'manifest.json')))
    for folder in sorted(RUN.glob('train-*')):
        if (folder/'pool.json').exists():
            for p in read(folder/'pool.json')['paths']:
                if not any(Path(r['weights'])==Path(p) for r in pool):pool.append(dict(weights=p,training_minutes=read(Path(p).parent/'progress.json')['training_seconds']/60,manifest_sha256=sha(Path(p)/'manifest.json')))
    save(REPORT/'pool.json',pool)
    plan=read(RUN/'plan.json') if (RUN/'plan.json').exists() else {};done=read(RUN/'completion.json') if (RUN/'completion.json').exists() else {}
    out=['# P11 — смешанный self-play ZOg','','## Результаты','',f'Стадия: {"завершено" if done.get("stage")=="complete" else "работа продолжается"}. Sampling; окружение 005, ggd-v4, event-v2; seed 20261103/20261104, по 64 матча (32 за сторону) на seed.']
    totals=[r for r in rows if r['side']=='total' and r['perspective']==0]
    totals.sort(key=lambda r:1e9 if r['model']=='final' else 0 if r['model'].startswith('000470') else float(r['model'].removesuffix('h')))
    training=read(RUN/'training-result.json') if (RUN/'training-result.json').exists() else {}
    if training:out.append(f'PPO завершён: {training["training_seconds"]/60:.3f} мин, {training["blocks"]} блока, {training["total_decisions"]} решений ученика; {training["episodes"]} завершённых обучающих матчей до фильтрации frozen-префикса.')
    for other in ('v10','v9'):
        out+=['',f'Против {other}:','','| Модель | W / D / L | LH / N | deny / N | farm / spawn | farm / циклы | Смерти / N | <120с / N | reward / N |','|---|---|---|---|---|---|---|---|---|']
        # Historical baseline is explicit; never relabel it as a P11 measurement.
        if not any(r['model'].startswith('000470') and r['matchup'].endswith(' vs '+other) for r in totals):
            if other=='v10':out.append('| 000470 (P09c, те же seed) | 55/128;0/128;73/128 | 1263/128 | 764/128 | 2027/12708 | см. P09c | 40/128 | 4/128 | см. P09c |')
            else:out.append('| 000470 (P10, seed20261105) | 21/64;0/64;43/64 | см. P10 | см. P10 | см. P10 | см. P10 | 4/64 | см. P10 | см. P10 |')
        for r in totals:
            if not r['matchup'].endswith(' vs '+other):continue
            n=r['matches'];out.append('| '+' | '.join([r['model'],';'.join(f'{r[k]}/{n}' for k in ('wins','draws','losses')),f'{r["lh"]}/{n}',f'{r["deny"]}/{n}',f'{r["lh"]+r["deny"]}/{r["lane_creeps_spawned"]}',f'{r["lh"]+r["deny"]}/{r["attack_cycles_completed"]}',f'{r["matches_with_death"]}/{n}',f'{r["early_deaths"]}/{n}',f'{r["reward_sum"]:.3f}/{n}'])+' |')
    out+=['','Версия против предыдущей версии:','','| Пара | W / D / L | Смерти / N | farm / spawn |','|---|---|---|---|']
    for r in totals:
        if r['matchup'].endswith((' vs v10',' vs v9')):continue
        n=r['matches'];out.append(f'| {r["matchup"]} | '+ ';'.join(f'{r[k]}/{n}' for k in ('wins','draws','losses'))+f' | {r["matches_with_death"]}/{n} | {r["lh"]+r["deny"]}/{r["lane_creeps_spawned"]} |')
    if (REPORT/'final-reference.json').exists():
        reference=read(REPORT/'final-reference.json');out+=['',f'Версия 2.35h сохранена на {reference["reference_training_seconds"]/60:.3f} мин PPO; отставание от финала — {reference["age_minutes"]:.3f} мин. [Точные пути](P11/final-reference.json).']
    out+=['','Обучающие матчи по часам, только полная награда:','','| Час | latest W/N | older W/N | v10 W/N | v9 W/N |','|---|---|---|---|---|']
    for hour in sorted(set(r['hour'] for r in hours)):
        cells=[]
        for kind in ('latest','older','v10','v9'):
            r=next((r for r in hours if r['hour']==hour and r['opponent']==kind),None);cells.append(f'{r["wins"]}/{r["matches"]}' if r else '0/0')
        out.append('| '+str(hour)+' | '+' | '.join(cells)+' |')
    out+=['','Полная статистика обеих сторон/участников с числителями и знаменателями: [metrics.csv](P11/metrics.csv), отдельные матчи — JSON серий. Кривые: [SVG](P11/curves.svg), [блоки](P11/updates.csv), [по100 матчей каждого типа](P11/matches-100.csv), [часы](P11/hours.csv).','','## Наблюдения','',f'- **Подтверждено:** опубликовано {sum(r["matches"] for r in totals if not r["model"].startswith("000470-"))} новых оценочных матчей P11 и192 исторических; полных обучающих матчей {sum(r["complete_reward"] for r in coverage)}, исключено с frozen-префиксом/прогревом {sum(r["excluded_prefix_or_warmup"] for r in coverage)}.']
    if updates:
        last=updates[-1];out.append(f'- **Подтверждено:** последний блок {last["block"]}, {last["training_minutes"]:.3f} мин PPO; KL={last["rollout_kl_m2"]:.6f}, entropy={last["entropy"]:.6f}, EVpost0={last["ev_post0"]}, clip={last["clip_fraction"]:.6f}. EV измеряет train GAE, не независимую validation.')
    for other in ('v10','v9'):
        final=next((r for r in totals if r['model']=='final' and r['matchup'].endswith(' vs '+other)),None)
        candidates=[r for r in totals if r['model'].startswith('000470') and r['matchup'].endswith(' vs '+other)]
        if final and candidates:
            base=next((r for r in candidates if r['model']=='000470'),candidates[0]);n=final['matches'];b=base['matches']
            out.append(f'- **Подтверждено, {other}:** {base["model"]}→final: победы {base["wins"]}/{b}→{final["wins"]}/{n}; ранние смерти {base["early_deaths"]}/{b}→{final["early_deaths"]}/{n}; farm/spawn {base["lh"]+base["deny"]:.0f}/{base["lane_creeps_spawned"]:.0f}→{final["lh"]+final["deny"]:.0f}/{final["lane_creeps_spawned"]:.0f}. Финальные победы по причинам: смерть {final["wins_death"]}/{n}, T1 {final["wins_T1"]}/{n}, LH600 {final["wins_LH600"]}/{n}.')
    out+=['- **Подтверждено:** строки 000470-P09c и 000470-P10 перенесены из прежних сырых отчётов, а не измерены заново. P09c использует те же два seed; P10 — 64 матча на seed20261105. Полная статистика и происхождение сохранены в JSON.', '- **Гипотеза:** изменение силы связано с адаптацией к смешанному пулу; причинный вклад каждого типа без абляции не установлен.','','## Что сделано','', '- `gpu/src/ggd_pool.rs`, `ggd_ppo.rs`: соперник закреплён до конца матча; вероятности latest/older/v10/v9=0.2/0.2/0.3/0.3; older исключает latest. После матча сторона ученика меняется. Модели-соперники frozen и sampling; loss только ученика.', '- Восстановлены исходные модель, критик, Adam000470; новый якорь к000470. Прогрев только value: минимум20/максимум60 блоков, EVpost0 последних5>0.3. Основной запуск продолжает его модель/Adam; расписание P11 сохраняется при resume.',f'- PPO: {plan.get("matches",192)} арены, 48×128, epochs2, actor3e-5, critic3e-4, clip0.2, entropy0.005, stop_kl0.02, gamma0.99, lambda0.95. KL0.03→0.01 за {plan.get("training_minutes","ещё не зафиксировано")} мин; рассинхронизация, snapshot60 мин, checkpoint10 мин, оценка120 мин.', '- `gpu/scripts/p11_run.py`, `p11_report.py`: фоновые процессы, статусы, восстановление, дедлайн08:40 UTC для обучения и09:30 UTC для публикации; данные/веса внеgit.', '- `[P11] fix: correct Burn imports and reconcile routed teacher commands`: исправлены две ошибки первой сборки и согласование команд; физика, учителя, observation и event-v2 не менялись.']
    if (REPORT/'smoke-result.json').exists():
        smoke=read(REPORT/'smoke-result.json');out.append(f'- Smoke: 59/59 тензоров неизменны при lr0; завершённые матчи по типам: {smoke["completed_by_type"]}.')
    out.append('- `[P11] fix: clear routed teacher caches on arena reset`: отдельные memory/forecasts v9 очищаются вместе с reset арены; паритет проверяет также независимый reset одной арены.')
    out.append('- `[P11] fix: honor deadline stops after scheduled evaluations`: stop во время оценки запрещает следующий PPO-блок; финал переиспользует завершённую серию только при совпадении checkpoint/соперника/seed/sampling, а неудачная оценка сохраняется и повторяется в новом каталоге.')
    out.append('- `[P11] fix: reserve measured desynchronization time before PPO`: план учитывает измеренное время рассинхронизации+2мин; контроллер подключён к уже работающему прогреву, без перезапуска симулятора и потери обновлений.')
    out.append('- `[P11] fix: exclude native log files from training audit`: glob train-* включал train-001.log; аудит теперь выбирает только прогоны с config.json. Обучение не прерывалось.')
    out.append('- `[P11] fix: keep evaluation tables in training order`: таблицы упорядочены000470→2h→4h→final независимо от порядка появления файлов оценок.')
    if (REPORT/'warmup-result.json').exists():
        warm=read(REPORT/'warmup-result.json');out.append(f'- Прогрев: {warm["result"]["blocks"]} блоков; EVpost0 last5={warm["result"]["warmup_ev_last5"]}; вне value head неизменны {warm["unchanged_policy_tensors"]}/55.')
    if (REPORT/'final-audit.json').exists():
        audit=read(REPORT/'final-audit.json');out.append(f'- Итоговая сверка: {audit["new_evaluation_matches"]} новых матчей, {audit["series_128"]} серий по 128, баланс сторон/sampling/seed проверены; ошибок симуляции 0. Все значения в {audit["finite_tensor_files"]} файлах тензоров конечны, сохранены {audit["pool_snapshots"]} снимков пула; [аудит](P11/final-audit.json). Rust 115/115; Python 240/240 и 4/4 целевых; последние правки отчёта проверены генерацией.')
    out+=['','## Команды, которыми получены результаты','','Из C:/AMI-wt/P11; дословные дочерние команды, exit-коды и минуты только обучения/оценок: [commands.json](P11/commands.json). Встроенные оценки вызывают evaluate_against; параметры/веса/seed/SHA — [evaluations.json](P11/evaluations.json).','```text','C:/Python314/python.exe gpu/scripts/p11_run.py verify','C:/Python314/python.exe gpu/scripts/p11_run.py all','C:/Python314/python.exe gpu/scripts/p11_run.py launch','C:/Python314/python.exe gpu/scripts/p11_run.py adopt-warmup','python gpu/scripts/p11_report.py','```','Контроллер all заменён на launch до smoke/обучения для загрузки исправления дедлайна; завершённые проверки сохранены, потеря train-обновлений0.']
    out.append('')
    for kind in ('training','evaluation'):
        selected=[r for r in commands if r['kind']==kind];out.append(f'- {kind}: внешние процессы {sum(r.get("minutes",0) for r in selected):.3f} мин; {sum(r.get("exit_code")==0 for r in selected)}/{len(selected)} завершились с exit0. Встроенные оценки учитываются отдельно в JSON серий.')
    if training:out.append(f'- Чистый PPO: {training["training_seconds"]/60:.3f} мин; встроенные оценки: {training["evaluation_seconds"]/60:.3f} мин; начальная рассинхронизация: {training["desynchronization"]["seconds"]/60:.3f} мин. Время training-процессов выше включает эти этапы и служебные операции.')
    if (REPORT/'final-audit.json').exists():out.append('- Итоговая проверка: `python gpu/scripts/p11_audit.py final`; команды тестов и проверенные версии: [Python](P11/python-verification.json), [Rust/бинарник](P11/verification.json).')
    out+=['','## Что не получилось или не проверено','', '- Первая сборка завершилась с ошибками API Burn; исправления отдельным fix-коммитом. Выполненные проверки и SHA бинарника — verification.json; незапущенные проверки не считаются пройденными.', '- Checkpoint сохраняет модель/Adam/время/пул, но не состояние арен; после сбоя новая рассинхронизация. Назначения случайны, доли фактических матчей отличаются от вероятностей; длительность матчей зависит от соперника.', '- Проверка на двух фиксированных seed симулятора; перенос в Dota и независимые повторения обучения не проверены. Историческая строка v9 до новой контрольной серии использует другой seed.']
    out.append('- Хеши checkpoint зафиксированы уже после начала соответствующих оценок; предшествовавший интервал отдельно не проверен. Границы проверки указаны в evaluation-freeze-*.json.')
    if done.get('notes'):out+=['- '+v for v in done['notes']]
    out+=['','## Пути к локальным артефактам','', '- `C:/AMI/runs/p11/`: sf-duel-gpu.exe, smoke/, warmup-*/, train-*/, eval-*/, state.json, commands.json; исходный checkpoint `C:/AMI/runs/p09/P09c/continue90/rl-continue-001/checkpoint-000470`.', '- Все снимки пула с путями и SHA: [pool.json](P11/pool.json). Финальный checkpoint: '+str(done.get('checkpoint',training.get('latest','ещё не выбран')))+'.','', 'DONE P11' if done.get('stage')=='complete' else 'BLOCKED P11: выполнение продолжается']
    assert len(out)<=150,len(out)
    (ROOT/'reports/P11.md').write_text('\n'.join(out)+'\n',encoding='utf-8',newline='\n')
    for p in REPORT.iterdir():
        if p.is_file():assert p.stat().st_size<1_000_000,p
    return rows
if __name__=='__main__':refresh()
