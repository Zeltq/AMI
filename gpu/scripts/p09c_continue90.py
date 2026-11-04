"""User-requested PPO continuation with an amendable limit, then three evaluations."""
import json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
import p09c_run as runner
from p09c_run import read,save,sha,pid_alive
from p09b_results import summaries,csvwrite,lines
BASE=Path('C:/AMI/runs/p09/P09c')
START=BASE/'rl-main-001/checkpoint-000314'
PILOT=BASE/'rl-pilot1-001/checkpoint-000126/weights'
BINARY=BASE/'sf-duel-gpu.exe'
RUN=BASE/'continue90';REPORT=runner.ROOT/'reports/P09c/continue90'

def make_config(base,progress,resume,training_minutes=90):
    assert training_minutes>0
    cfg=dict(base)
    # Preserve the completed KL schedule: it remains clamped to 0.05.
    # Use the stored f64 exactly; decimal round-trip differences must not
    # accidentally request another schedule transition.
    for key in ('schedule_minutes','kl_decay_start_minutes','kl_start','kl_end'):cfg[key]=progress[key]
    cfg.update(minutes=progress['training_seconds']/60+training_minutes,resume_checkpoint=str(resume),initialize_weights=None,evaluate=False,skip_warmup=True,critic_only=False,begin_decay_on_resume=False,stop_at_minutes=0,max_blocks=0)
    return cfg

def setup():
    runner.RUN=RUN;runner.REPORT=REPORT
    # A fresh ledger records this explicit new grant separately from completed
    # P09c. The request determines the native target; 300 is only an emergency cap.
    runner.publish=publish
    RUN.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    os.environ['SF_DUEL_DATA_ROOT']='C:/AMI'

def compact_report(label,path):
    r=read(path);cfg=read(path.parent/'config.json')
    data=dict(model=f"RL+{read(REPORT/'request.json')['training_minutes']:g}",opponent=label,seed=r['seed'],source=str(path),config=cfg,evaluation_minutes=r['wall_seconds']/60,source_sha256=sha(path),weights_manifest_sha256=sha(Path(cfg['weights'])/'manifest.json'))
    if (path.parent/'opponent.json').exists():data['opponent_provenance']=read(path.parent/'opponent.json')
    data['per_match']=[{**g,'sides':[{k:v for k,v in h.items() if k!='minute_snapshots'} for h in g['sides']]} for g in r['per_match']]
    (REPORT/f'vs-{label}-{r["seed"]}.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    return r

def refresh():
    import p09c_results as plotting
    plotting.RUN=RUN;plotting.REPORT=REPORT
    updates,bins=plotting.curves()
    request=read(REPORT/'request.json');requested=request['training_minutes'];model=f'RL+{requested:g}';groups={};rows=[]
    for entry in read(RUN/'evaluations.json') if (RUN/'evaluations.json').exists() else []:
        path=Path(entry['report']);r=compact_report(entry['opponent'],path)
        groups.setdefault(entry['opponent'],[]).extend(r['per_match'])
    for label,games in groups.items():rows+=summaries(model,dict(per_match=games),label,'event-v2')
    save(REPORT/'metrics.json',rows);csvwrite(REPORT/'metrics.csv',rows)
    trained=max([r['training_minutes']-request['start_training_minutes'] for r in updates]+[0.])
    stage=read(RUN/'state.json').get('stage','preparing') if (RUN/'state.json').exists() else 'preparing'
    text=[f'# P09c: продолжение +{requested:g} минут по запросу пользователя','', '## Результаты','',f'Дополнительное обучение: {trained:.3f}/{requested:g} минут PPO. Завершённых контрольных серий: {sum(len(g)==128 for g in groups.values())}/3.','', '| Соперник | Победы / ничьи / поражения | LH за матч | Deny за матч | Смерть / матчи | До120с / матчи |','|---|---|---|---|---|---|']
    for label in ('v10','pilot','current'):
        found=[r for r in rows if r['matchup']==model+' vs '+label and r['side']=='total' and r['perspective']==0]
        if not found:continue
        r=found[0];n=r['matches'];text.append(f'| {label} | {r["wins"]}/{n}; {r["draws"]}/{n}; {r["losses"]}/{n} | {r["lh"]/n:.2f} | {r["deny"]/n:.2f} | {r["matches_with_death"]}/{n} | {r["early_deaths"]}/{n} |')
    text+=['', 'Полная статистика обеих моделей по сторонам: [metrics.csv](metrics.csv); отдельные матчи и provenance — `vs-OPPONENT-SEED.json`. Кривые: [SVG](main-curves.svg), [PPO](rl-updates.csv), [по100 матчей](rl-100-matches.csv).','', '## Наблюдения','', '- **Подтверждено:** этот этап запущен по прямому запросу пользователя 05.11.2026: «Даю тебе еще 1,5 часа на обучение. После этого проведи 128 матчей против учителя и по 128 матчей против модели после пилота и против текущей модели».','- **Подтверждено:** `current` зафиксирован до продолжения: checkpoint000314 исходного финала; `pilot` — checkpoint000126 после40мин пилота. Это отдельные неизменные соперники.','', '## Что сделано','', '- Восстановлены модель и Adam из текущего финала. Event-v2, ggd-v4, окружение005, teacher-v10; 192 арены, minibatch48×128,2 PPO-эпохи, actor lr3e-5, critic lr3e-4, clip0.2, entropy0.005, stop_kl0.02, gamma0.99, GAE lambda0.95.','- KL-якорь остаётся к исходной ZOg-d1, коэффициент0.05 постоянный. Завершённое расписание сохранено без сброса. Дополнительного прогрева критика нет. Checkpoint каждые30мин; после восстановления арены рассинхронизируются.',f'- {requested:g}мин считаются по native-времени PPO сверх сохранённых100.370835мин; подготовка арен и оценки отдельно. Оценки только после обучения:2seed×64 на соперника, sampling,64 за каждую сторону.','', '## Команды, которыми получены результаты','', '`C:/Python314/python.exe gpu/scripts/p09c_continue90.py` из C:/AMI-wt/P09c. Дословные дочерние команды и минуты процессов обучения/оценок: [commands.json](commands.json).','',f'Новый этап имеет отдельный учёт: {runner.used():.3f} GPU-мин. Предыдущие292.772521мин не сброшены и сохранены в [request.json](request.json); действующая авторизация —{requested:g}мин обучения плюс384 оценочных матча.','', '## Что не получилось или не проверено','', '- Текущая стадия: '+stage+'. До завершения трёх серий сравнительные выводы не делаются.','- Фиксированные seed20261103/20261104; перенос в настоящую Dota не проверяется. Состояние арен не сериализуется. Raw throughput старого бинарника после resume неверен; CSV делит новые решения на новое время.','', '## Пути к локальным артефактам','',f'- Новый этап: {RUN}; checkpoint и history/episodes в `rl-continue-NNN`; оценки в `eval-OPPONENT-SEED`.','- Старт: C:/AMI/runs/p09/P09c/rl-main-001/checkpoint-000314. Пилот: C:/AMI/runs/p09/P09c/rl-pilot1-001/checkpoint-000126.','', 'Работа продолжается: дополнительный этап по запросу пользователя.']
    if request.get('amendments'):
        at=text.index('## Наблюдения')+2
        text.insert(at, f'- **Подтверждено:** позднее пользователь сократил этот этап до {requested:g} минут суммарно: «'+request['amendments'][-1]['user_request']+'». Три оценки сохранены; [изменение лимита](limit-change.json).')
        text.insert(at+1, '- Исходный native-конфиг с90мин сохранён как provenance запуска. Новый лимит применяется штатным stop.json на границе PPO-блока с сохранением checkpoint/Adam; готовый результат передаётся контроллеру оценок. Каталог continue90 сохраняет историческое имя.')
    assert len(text)<=150;(REPORT/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8',newline='\n')
    save(REPORT/'budget.json',dict(additional_native_training_target_minutes=requested,previous_gpu_minutes=request['previous_gpu_minutes'],additional_gpu_process_minutes=runner.used(),cumulative_gpu_process_minutes=request['previous_gpu_minutes']+runner.used(),controller_emergency_cap_minutes=runner.LIMIT,evaluations_requested=384))

def publish(message):
    refresh()
    for p in REPORT.rglob('*'):
        if p.is_file():assert p.stat().st_size<1_000_000,p
    subprocess.run(['git','add','reports/P09c/continue90'],cwd=runner.ROOT,check=True)
    names=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=runner.ROOT,text=True).splitlines()
    assert all(n.startswith('reports/P09c/continue90/') for n in names),'Unexpected staged files'
    subprocess.run(['git','diff','--cached','--stat'],cwd=runner.ROOT,check=True)
    if names:subprocess.run(['git','commit','-m','[P09c] '+message],cwd=runner.ROOT,check=True)
    subprocess.run(['git','-c','core.sshCommand=C:/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=15','push','-u','origin','agent/P09c-zog-rl2'],cwd=runner.ROOT,check=True,timeout=120)

def train():
    progress=read(START/'progress.json');base=read(BASE/'rl-main-001/config.json');cfg=make_config(base,progress,START,read(REPORT/'request.json')['training_minutes'])
    target=cfg['minutes'];completed=RUN/'training-result.json'
    if completed.exists():
        result=read(completed);tolerance=30 if result.get('limit_amended') else 0
        assert result['training_seconds']>=target*60-tolerance
        if result.get('limit_amended'):assert abs(result['training_seconds']-target*60)<=30
        return result
    while True:
        candidates=[]
        for path in RUN.glob('rl-continue-*/checkpoint-*/progress.json'):
            if all((path.parent/f).exists() for f in ('model.bpk','optimizer.bpk','weights/manifest.json')):candidates.append((read(path)['training_seconds'],path.parent))
        if candidates:
            elapsed,path=max(candidates);cfg['resume_checkpoint']=str(path)
            if elapsed>=target*60:
                result=dict(latest=str(path),training_seconds=elapsed,recovered_after_checkpoint=True);save(completed,result);return result
        attempt=1+sum(r['stage'].startswith('rl-continue-') for r in runner.history());name=f'rl-continue-{attempt:03}';out=RUN/name
        save(REPORT/(name+'.json'),cfg)
        try:runner.run(name,'training',[BINARY,'train-ggd',REPORT/(name+'.json'),out],out/'result.json',True)
        except Exception:
            available=[p for p in RUN.glob('rl-continue-*/checkpoint-*/progress.json') if all((p.parent/f).exists() for f in ('model.bpk','optimizer.bpk','weights/manifest.json'))]
            if not available or runner.used()>=runner.LIMIT:raise
            continue
        result=read(out/'result.json');assert result['training_seconds']>=target*60,'Native training ended before the active requested limit'
        save(completed,result);publish('publish requested continuation training');return result

def evaluate(result):
    final=Path(result['latest'])/'weights';registry=read(RUN/'evaluations.json') if (RUN/'evaluations.json').exists() else []
    for label,opponent in [('v10',None),('pilot',PILOT),('current',START/'weights')]:
        games=[]
        for seed in (20261103,20261104):
            out=RUN/f'eval-{label}-{seed}';cmd=[BINARY,'evaluate-ggd',runner.ENGINE,final,out,'--mode','teacher','--matches',64,'--seconds',600,'--seed',seed,'--sampling','--teacher-version','v10']
            if opponent:cmd+=['--opponent-weights',opponent]
            runner.run(out.name,'evaluation',cmd,out/'report.json',True)
            r=read(out/'report.json');assert r['complete'] and r['matches']==64;games+=r['per_match']
            ecfg=read(out/'config.json');assert Path(ecfg['weights'])==final and not ecfg['argmax'] and r['seed']==seed
            assert sum(g['sides'][g['game']%2]['side']=='radiant' for g in r['per_match'])==32
            if opponent:assert read(out/'opponent.json')['manifest_sha256']==sha(opponent/'manifest.json')
            for g in r['per_match']:
                assert g['simulation_error']==0
                for side,h in enumerate(g['sides']):
                    c=h['event_v2_components'];assert c[0]==10*g['result']*(1 if side==0 else -1)
                    assert abs(c[1]-.5*h['lh'])<1e-3 and abs(c[2]-.2*h['deny'])<1e-3 and abs(sum(c)-h['event_v2_reward'])<1e-3
            entry=dict(opponent=label,report=str(out/'report.json'))
            if entry not in registry:registry.append(entry);save(RUN/'evaluations.json',registry)
        assert len(games)==128;publish('publish continuation evaluation vs '+label)

def main():
    setup();lock=runner.exclusive();assert sha(BINARY)==read(runner.ROOT/'reports/P09c/verified.json')['exe_sha256']
    source=read(runner.ROOT/'reports/P09c/final-checkpoint-audit.json')
    assert sha(START/'model.bpk')==source['model_sha256'] and sha(START/'optimizer.bpk')==source['optimizer_sha256']
    request=REPORT/'request.json'
    if not request.exists():
        base=read(BASE/'pilot-final/completion.json');progress=read(START/'progress.json')
        save(request,dict(user_explicitly_requested_continuation=True,user_request='Все сохраняется, продолжаем точно также обучать модель. Даю тебе еще 1,5 часа на обучение. После этого проведи 128 матчей против учителя и по 128 матчей против модели после пилота и против текущей модели. В отчете явно укажи, что я попросил тебя продолжить.',recorded_utc=datetime.now(timezone.utc).isoformat(),training_minutes=90,start_training_minutes=progress['training_seconds']/60,previous_gpu_minutes=base['gpu_minutes'],source_checkpoint=str(START),model_sha256=sha(START/'model.bpk'),optimizer_sha256=sha(START/'optimizer.bpk'),pilot_manifest_sha256=sha(PILOT/'manifest.json'),current_manifest_sha256=sha(START/'weights/manifest.json'),anchor_manifest_sha256=progress['anchor_sha256'],new_warmup=False,from_scratch=False,semantics='Resume model+Adam, same event-v2 and hyperparameters, completed KL schedule stays at .05; evaluations separately after +90 native training minutes.'))
    try:
        runner.status(dict(stage='authorized continuation; preparing publication',pid=None,background_work_remaining=True))
        publish('record active user-requested continuation limit')
        result=train();evaluate(result)
        runner.status(dict(stage='continue90 complete; final audit/report/chat summary pending',pid=None,background_work_remaining=True,checkpoint=result['latest'],report=str(REPORT/'report.md'),gpu_minutes=runner.used()))
    except Exception as exc:runner.status(dict(stage='failed',error=str(exc),background_work_remaining=True));raise

if __name__=='__main__':main()
