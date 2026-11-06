"""Autonomous P16 stage supervisor; no policy-contract changes or metric-based RL stopping."""
import concurrent.futures,ctypes,datetime,json,msvcrt,os,shutil,subprocess,threading,time
from pathlib import Path
from p16_report import refresh
from p15_eval_summary import summarize
ROOT=Path(__file__).resolve().parents[2];RUN=Path('C:/AMI/runs/p16');REPORT=ROOT/'reports/P16'
EXE=RUN/'target/release/sf-duel-gpu.exe';BASE=RUN/'p15-base.exe';PY='C:/Python314/python.exe'
LOCK=threading.RLock();POOL=concurrent.futures.ThreadPoolExecutor(max_workers=1);FUTURES=[]

def read(p,default=None):
    p=Path(p)
    if not p.exists():return default
    # Live trainers overwrite status.json; a reader can see a truncated write.
    for attempt in range(20):
        try:return json.loads(p.read_text(encoding='utf-8-sig'))
        except (json.JSONDecodeError,PermissionError):
            if attempt==19:raise
            time.sleep(.05)
def save(p,v):
    with LOCK:
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(f'.{os.getpid()}.pending')
        temp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
        for attempt in range(20):
            try:temp.replace(p);return
            except PermissionError:
                if attempt==19:raise
                time.sleep(.05)
def current_engine():return read(RUN/'navigation-mode.json',{}).get('engine',str(RUN/'environment/engine.json'))
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def alive(pid):
    if not pid:return False
    kernel=ctypes.windll.kernel32;kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:return False
    code=ctypes.c_ulong();kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(code));kernel.CloseHandle(ctypes.c_void_p(handle));return code.value==259

def state(stage,**fields):
    with LOCK:
        value=dict(stage=stage,controller_pid=os.getpid(),updated_utc=utc(),background_work_remaining=stage!='complete',**fields);save(RUN/'state.json',value)
        p=Path('C:/AMI/runs/pilot-status.json');shared=read(p,{});shared['P16']=value;save(p,shared)
def incident(stage,reason,**fields):
    with LOCK:
        rows=read(RUN/'incidents.json',[]);rows.append(dict(utc=utc(),stage=stage,reason=reason,**fields));save(RUN/'incidents.json',rows)
def publish(message):
    with LOCK:
        refresh()
        for p in REPORT.rglob('*'):
            if p.is_file():assert p.stat().st_size<1000000,(p,p.stat().st_size)
        subprocess.run(['git','add','reports/P16.md','reports/P16'],cwd=ROOT,check=True)
        staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
        assert all(x=='reports/P16.md' or x.startswith('reports/P16/') for x in staged),staged
        subprocess.run(['git','diff','--cached','--stat'],cwd=ROOT,check=True)
        if staged:subprocess.run(['git','commit','-m','[P16] '+message],cwd=ROOT,check=True)
        for attempt in range(4):
            result=subprocess.run(['git','-c','core.sshCommand=C:/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=15','push','-u','origin','agent/P16-chio-night'],cwd=ROOT,timeout=120)
            if result.returncode==0:break
            time.sleep(10)
        else:raise RuntimeError('Git push failed four times; local results are committed')
def config(name,value):
    p=REPORT/'configs'/f'{name}.json';save(p,value);return p

def launch(name,cmd):
    job=RUN/'jobs'/name
    if (job/'status.json').exists():return
    subprocess.Popen([PY,'gpu/scripts/p16_job.py',name,'--',*map(str,cmd)],cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
    until=time.monotonic()+30
    while not (job/'status.json').exists():
        if time.monotonic()>until:raise RuntimeError('Job wrapper did not start: '+name)
        time.sleep(.2)
def wait(name,artifact=None,on_tick=None,foreground=True):
    job=RUN/'jobs'/name
    while True:
        d=read(job/'status.json',{});native={}
        if artifact:
            native=read(Path(artifact).parent/'status.json',{})
            if native.get('stage')=='collecting-chio':
                native['completed_matches_total']=native['cohort_offset']+native['finished']
                native['matches_per_minute']=native['completed_matches_total']/max(native['wall_seconds']/60,1e-9)
            if native.get('block',0)>0:
                b=read(Path(artifact).parent/f"block-{native['block']:06}.json",{});update=b.get('update',{});rows=update.get('minibatches',[])
                native['metrics']={key:sum(r[key] for r in rows if key in r)/sum(key in r for r in rows) for key in ('loss','value_loss','entropy','old_policy_kl','anchor_kl','gradient_norm') if any(key in r for r in rows)}
                native['metrics']['ev_post0']=update.get('ev_post0');native['phase_counts']=b.get('phase_counts')
        if d.get('stage') in ('complete','failed'):
            if d.get('exit_code')!=0:raise RuntimeError(f"{name}: exit={d.get('exit_code')}; {job/'stderr.log'}")
            return read(artifact) if artifact else d
        if d and not alive(d.get('pid')) and not alive(d.get('child_pid')):raise RuntimeError('Lost background process: '+name)
        if foreground:state(name,pid=d.get('child_pid'),command=d.get('command'),native=native,artifacts=str(Path(artifact).parent) if artifact else str(job))
        if on_tick:on_tick(native)
        time.sleep(10)
def run_job(name,cmd,artifact=None,on_tick=None,foreground=True):
    launch(name,cmd);return wait(name,artifact,on_tick,foreground)
def done(name,result,folder):
    with LOCK:
        stages=read(RUN/'stages.json',{});stages[name]=dict(result=result,folder=str(folder),finished_utc=utc());save(RUN/'stages.json',stages)
    publish('complete '+name)
def cached(name):return read(RUN/'stages.json',{}).get(name)
def linux(p):
    text=str(p).replace('\\','/');return '/mnt/'+text[0].lower()+text[2:] if len(text)>1 and text[1]==':' else text
def torch_command(script,*args):return ['wsl','-d','Ubuntu','--','bash',linux(ROOT/'scripts/wsl_python.sh'),linux(ROOT/'gpu/scripts'/script),*map(linux,args)]
def supervised(name,cfg):
    if cached(name):return cached(name)
    folder=RUN/name;cp=config(name,cfg)
    for attempt in range(1,5):
        job=name if attempt==1 else f'{name}-resume-{attempt}'
        args=torch_command('train_chio.py',cp,folder)
        if attempt>1 or (folder/'checkpoint.pt').exists():args+=['--resume',linux(folder/'checkpoint.pt')]
        try:r=run_job(job,args,folder/'result.json');break
        except RuntimeError as error:
            incident(name,str(error),checkpoint=str(folder/'checkpoint.pt'))
            if not (folder/'checkpoint.pt').exists():raise
    else:raise RuntimeError('Repeated supervised failure: '+name)
    assert r['passed'] and r['finite_updates']==r['updates'];done(name,r,folder);return cached(name)
def collect(name,cfg,dagger=False,adopt=False):
    if cached(name):
        data=cached(name);launch(name+'-audit',[PY,'gpu/scripts/p15_dataset_audit.py',data['folder'],REPORT/(name+'-audit.json')]);return data
    for attempt in range(1,5):
        folder=RUN/(name if attempt==1 else f'{name}-attempt-{attempt}');job=folder.name;cp=config(job,cfg)
        try:r=run_job(job,[BASE if adopt else EXE,'collect-dagger' if dagger else 'collect-teacher',cp,folder],folder/'report.json');break
        except RuntimeError as error:
            stderr=(RUN/'jobs'/job/'stderr.log').read_text(encoding='utf-8',errors='replace')
            if 'memory' not in stderr.lower():raise
            collection=cfg['collection'] if dagger else cfg;old=collection['parallel_matches'];collection['parallel_matches']=max(32,old-32)
            incident(name,str(error),partial_data=str(folder),old_parallel=old,new_parallel=collection['parallel_matches'],restart_reason='No native arena checkpoint; preserve failed cohort data and repeat full requested count with the same seed.')
    else:raise RuntimeError('Repeated collection failure')
    collection=cfg['collection'] if dagger else cfg
    assert r['native_completed']==collection['games'] and not r['partial'];done(name,r,folder)
    launch(name+'-audit',[PY,'gpu/scripts/p15_dataset_audit.py',folder,REPORT/(name+'-audit.json')]);return cached(name)
def parity(name,dataset,training):
    if cached(name):return
    folder=RUN/name
    run_job(name+'-fixture',torch_command('p16_parity_fixture.py',dataset,Path(training)/'checkpoint.pt',folder),folder/'provenance.json')
    r=run_job(name,[EXE,'check-chio-parity',folder/'weights',folder/'fixture',folder/'result.json'],folder/'result.json')
    assert r['passed'] and r['steps']==128
    save(REPORT/(name+'.json'),r);done(name,r,folder)
def evaluation(label,weights,opponent,previous=None,foreground=True):
    name='eval-'+label+'-'+opponent+'-sampling'
    with LOCK:
        existing=[e for e in read(RUN/'evaluations.json',[]) if e['label']==label and e['opponent']==opponent and e.get('complete') and e.get('sampling',False)]
    if existing:return existing[0]
    cfg=dict(engine=current_engine(),matches=128,seconds=600,seed=20261103,left='v10',right='v9' if opponent=='v9' else 'v10',swap=True,weights=str(weights),sampling=True,zero_sum=True)
    if previous:cfg['opponent_weights']=str(previous)
    for attempt in range(1,4):
        job=name if attempt==1 else name+f'-retry-{attempt}';folder=RUN/job;cp=config(job,cfg)
        try:r=run_job(job,[EXE,'evaluate-chio',cp,folder],folder/'report.json',foreground=foreground);break
        except RuntimeError as error:incident(name,str(error),attempt=attempt)
    else:
        incident(name,'Evaluation retries exhausted; ongoing training remains active')
        raise RuntimeError('Evaluation failed: '+name)
    assert r['complete'] and len(r['per_match'])==128
    summary=REPORT/(name+'.json');d=summarize(folder/'report.json',summary)
    d['duration_elapsed_seconds']={'numerator':sum(g['game_seconds']+read(Path(cfg['engine']))['pre_game_seconds'] for g in r['per_match']),'denominator':128}
    for c in d['controllers'].values():
        c['counter_denominator']=c['actors'];c['raze_denominator_note']='Hits on enemy hero / all completed casts of this spell; no inferred model intent.'
    save(summary,d);row=dict(sampling=True,scripted_base_to_mid=bool(r.get('scripted_base_to_mid',False)),label=label,opponent=opponent,weights=str(weights),opponent_weights=str(previous) if previous else None,complete=True,summary=str(summary.relative_to(ROOT)).replace('\\','/'),folder=str(folder),finished_utc=utc(),seconds=r['wall_seconds'])
    with LOCK:
        rows=read(RUN/'evaluations.json',[]);rows.append(row);save(RUN/'evaluations.json',rows)
    publish('evaluate '+label+' against '+opponent);return row

def evaluate_group(label,weights,previous=None,foreground=True):
    with LOCK:
        existing={e['opponent'] for e in read(RUN/'evaluations.json',[]) if e['label']==label and e.get('complete') and e.get('sampling',False)}
    needed={'v10','v9'}|({'previous'} if previous else set())
    if needed<=existing:return
    evaluation(label,weights,'v10',foreground=foreground);evaluation(label,weights,'v9',foreground=foreground)
    if previous:evaluation(label,weights,'previous',previous,foreground)
    with LOCK:save(RUN/'previous-evaluated.json',dict(label=label,weights=str(weights)))
def queue_evaluation(label,weights,parallel):
    previous=read(RUN/'previous-evaluated.json',{}).get('weights')
    if parallel:FUTURES.append(POOL.submit(evaluate_group,label,weights,previous,False))
    else:evaluate_group(label,weights,previous)
def full_verification(prefix=''):
    if cached(prefix+'full-verification'):return
    (RUN/'tmp').mkdir(exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(RUN/'tmp');os.environ['SF_DUEL_TEST_ARTIFACTS']=str(RUN/'test-artifacts');os.environ['SF_DUEL_DATA_ROOT']='C:/AMI';os.environ['PYTHONPATH']=';'.join([str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests'),str(ROOT/'gpu/scripts'),'C:/AMI/tools/lua_test_runtime']);os.environ['PYTHONDONTWRITEBYTECODE']='1'
    r=run_job(prefix+'full-rust',['cargo','test','--release','--locked','--manifest-path','gpu/Cargo.toml','--target-dir',RUN/'target','--','--test-threads=1'])
    p=run_job(prefix+'full-python',[PY,'-m','unittest','discover','-s','tests','-v'])
    run_job(prefix+'release-final',['cargo','build','--release','--locked','--manifest-path','gpu/Cargo.toml','--target-dir',RUN/'target'])
    evidence={'rust':r,'python':p}
    for name in ['full-rust','full-python']:
        evidence[name+'-tail']=(RUN/'jobs'/(prefix+name)/('stdout.log' if name=='full-rust' else 'stderr.log')).read_text(encoding='utf-8',errors='replace').splitlines()[-7:]
    save(REPORT/(prefix+'full-verification.json'),evidence);done(prefix+'full-verification',evidence,RUN/'jobs')
def dagger_capacity(weights):
    existing=read(RUN/'dagger-capacity.json')
    if existing:return existing['parallel_matches']
    rows=[];chosen=128;started=time.monotonic()
    for n in [192,256,288,320]:
        name=f'capacity-dagger-{n}';cfg=read(REPORT/'configs/dagger-1-collect.json');cfg['collection'].update(games=n,parallel_matches=n,max_blocks=1);cfg['weights']=str(weights);cp=config(name,cfg);folder=RUN/name
        try:r=run_job(name,[EXE,'collect-dagger',cp,folder],folder/'report.json');rows.append(dict(matches=n,passed=True,result=r));chosen=n
        except RuntimeError as error:rows.append(dict(matches=n,passed=False,error=str(error)));break
        if time.monotonic()-started>150:break
    result=dict(parallel_matches=chosen,rows=rows,wall_seconds=time.monotonic()-started);save(RUN/'dagger-capacity.json',result);save(REPORT/'dagger-capacity.json',result);publish('measure DAgger collection capacity');return chosen

def parallel_probe(weights):
    known=read(RUN/'parallel-evaluation.json')
    if known:return known['allowed']
    cfg=read(REPORT/'configs/ppo-v10.json');cfg.update(engine=current_engine(),initial_weights=str(weights),desynchronize=False,max_blocks=6 if read(RUN/'navigation-mode.json',{}).get('active') else 1,minutes=10,checkpoint_minutes=0);cp=config('parallel-ppo-smoke',cfg)
    e=dict(engine=current_engine(),matches=128,seconds=60 if read(RUN/'navigation-mode.json',{}).get('active') else 1,seed=20261103,weights=str(weights),opponent_weights=str(weights),sampling=True,zero_sum=True);ep=config('parallel-eval-smoke',e)
    launch('parallel-ppo-smoke',[EXE,'train-ggd',cp,RUN/'parallel-ppo-smoke']);launch('parallel-eval-smoke',[EXE,'evaluate-chio',ep,RUN/'parallel-eval-smoke'])
    results={}
    for name in ['parallel-ppo-smoke','parallel-eval-smoke']:
        try:wait(name);results[name]=True
        except RuntimeError as error:results[name]=False;incident(name,str(error))
    result=dict(allowed=all(results.values()),results=results);save(RUN/'parallel-evaluation.json',result);save(REPORT/'parallel-evaluation.json',result);publish('qualify concurrent RL evaluation');return result['allowed']
def native_training(name,cfg,parallel=False):
    if cached(name):return cached(name)
    evaluated=[int(e['label'].split('-')[-1][:-1])*60 for e in read(RUN/'evaluations.json',[]) if e['label'].startswith('self-play-') and e['label'].endswith('m') and e.get('complete')]
    next_eval=max(evaluated,default=0)+120*60;pending_label=None
    for attempt in range(1,10):
        folder=RUN/(name if attempt==1 else name+f'-attempt-{attempt}');job=folder.name;cp=config(job,cfg)
        def tick(native):
            nonlocal next_eval,pending_label
            seconds=native.get('training_seconds',0)
            cutoff=read(RUN/'plan.json',{}).get('self_play_stop_utc')
            if name=='self-play' and cutoff and datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(cutoff):
                save(folder/'stop.json',{'reason':'wall-clock budget; reserve final evaluations before 11-hour deadline'})
            # The final checkpoint is evaluated by the caller; native loop forces checkpoints at 120-minute boundaries.
            if name=='self-play' and seconds>=next_eval and seconds<cfg['minutes']*60-30:
                latest=read(folder/'latest.json');label=f'self-play-{int(next_eval/60)}m'
                if latest:
                    if parallel:queue_evaluation(label,Path(latest['path'])/'weights',True)
                    else:pending_label=label;save(folder/'stop.json',{'reason':'scheduled evaluation; resume same stage anchor and clock afterward'})
                    next_eval+=120*60
            # Evaluation failures are repaired separately; never abandon a live training process.
        try:
            r=run_job(job,[EXE,'train-ggd',cp,folder],folder/'result.json',on_tick=tick)
            if pending_label:
                queue_evaluation(pending_label,folder/'final-weights',False);pending_label=None
                if r['training_seconds']<cfg['minutes']*60:cfg['resume_checkpoint']=r['checkpoint'];continue
            break
        except RuntimeError as error:
            latest=read(folder/'latest.json')
            incident(name,str(error),checkpoint=latest)
            if not latest:raise
            cfg['resume_checkpoint']=latest['path']
    else:raise RuntimeError('Native training retry/segment limit: '+name)
    done(name,r,folder);return cached(name)

def choose_best(eligible_labels=None):
    entries=read(RUN/'evaluations.json',[]);labels=sorted({e['label'] for e in entries});candidates=[]
    for label in labels:
        if eligible_labels is not None and label not in eligible_labels:continue
        selected=[e for e in entries if bool(e.get('scripted_base_to_mid',False))==bool(read(RUN/'navigation-mode.json',{}).get('active')) and e.get('sampling',False) and e['label']==label and e['opponent'] in ('v10','v9') and e.get('complete')]
        if len(selected)!=2:continue
        cs=[read(ROOT/e['summary'])['controllers']['CHiO'] for e in selected];n=sum(c['actors'] for c in cs)
        wins=sum(c['outcomes'].get('win',0) for c in cs);draws=sum(c['outcomes'].get('draw',0) for c in cs);deaths=sum(c['deaths']['numerator'] for c in cs)
        farm=sum(c['lh']['numerator']+c['deny']['numerator'] for c in cs);mid=sum(c['mid_arrivals']['numerator'] for c in cs)
        hits=sum(v['numerator'] for c in cs for v in c['raze_hero_hit_rates'].values());casts=sum(v['denominator'] for c in cs for v in c['raze_hero_hit_rates'].values())
        tower=sum(sum(v for k,v in c['death_causes'].items() if 'tower' in k.lower()) for c in cs);arrival=sum(c['mid_arrival_elapsed_seconds']['numerator'] for c in cs)
        metrics=[(wins+.5*draws)/n,1-deaths/n,farm/n,mid/n,hits/max(casts,1),1-tower/n,-arrival/mid if mid else -1000000]
        candidates.append(dict(label=label,weights=selected[0]['weights'],metrics=metrics,wins=wins,draws=draws,deaths=deaths,farm=farm,mid=mid,hits=hits,casts=casts,tower_deaths=tower,actors=n))
    weights=[3,2,2,2,1,1,1]
    for c in candidates:c['rank_score']=sum(w*sum(other['metrics'][i]<c['metrics'][i] for other in candidates)/max(1,len(candidates)-1) for i,w in enumerate(weights))/sum(weights)
    winner=max(candidates,key=lambda c:c['rank_score']);n=winner['actors']
    reason=(f"предварительный сбалансированный выбор по рангам побед, выживаемости, фарма, выхода/времени на мид, койлов и смертей от башни (веса3/2/2/2/1/1/1; без порога). "
            f"Против обоих учителей суммарно: W {winner['wins']}/{n}, смерти {winner['deaths']:g}/{n}, LH+deny {winner['farm']:g}/{n}, мид {winner['mid']}/{n}, койлы {winner['hits']:g}/{winner['casts']:g}. Это эвристика выбора, не доказательство переноса в Dota.")
    result=dict(label=winner['label'],weights=winner['weights'],reason=reason,candidates=candidates);save(RUN/'best.json',result);save(REPORT/'best.json',result);return result

def navigation_gate():
    """User revision: diagnose first; decide only after completed DAgger-1 evaluation."""
    summary=read(REPORT/'eval-dagger-1-v10.json');arrival=summary['controllers']['CHiO']['mid_arrivals']
    while True:
        review=read(RUN/'direction-review.json',{})
        if review.get('classification')=='distribution-shift':
            decision=dict(mid_arrivals=arrival,required_majority=65,action='continue-policy' if arrival['numerator']>arrival['denominator']/2 else 'scripted-navigation-restart',reason='User conditional instruction after DAgger 1',utc=utc())
            save(RUN/'navigation-decision.json',decision);save(REPORT/'navigation-decision.json',decision);publish('decide navigation after DAgger 1')
            if decision['action']=='continue-policy':return
            while not (RUN/'navigation-recovery-ready.json').exists():
                state('navigation-recovery-required',decision=decision);time.sleep(10)
            raise RuntimeError('Navigation recovery is ready; hand off to the revised pipeline before further training')
        state('direction-diagnosis-pending',mid_arrivals=arrival,review=review);time.sleep(10)

def finish_rl(previous,il,older_weights):
    cfg=read(REPORT/'configs/critic.json');cfg['engine']=current_engine();cfg['initial_weights']=str(Path(previous['folder'])/'weights');critic=native_training('critic',cfg)
    old=read(Path(previous['folder'])/'weights/manifest.json');new=read(Path(critic['folder'])/'final-weights/manifest.json');keys=[k for k in old['tensors'] if not k.startswith('value_')]
    equal=sum(old['tensors'][k]['sha256']==new['tensors'][k]['sha256'] for k in keys);assert equal==len(keys)==45
    save(REPORT/'critic-freeze.json',dict(equal=equal,total=len(keys)));publish('verify frozen critic policy')
    parallel=parallel_probe(Path(critic['folder'])/'final-weights')
    cfg=read(REPORT/'configs/ppo-v10.json');cfg.update(engine=current_engine(),initial_weights=str(Path(critic['folder'])/'final-weights'),bootstrap_checkpoint=critic['result']['checkpoint'])
    ppo=native_training('ppo',cfg,parallel);queue_evaluation('ppo-120m',Path(ppo['folder'])/'final-weights',parallel)
    plan=read(RUN/'plan.json');deadline=datetime.datetime.fromisoformat(plan['deadline_utc']);remaining=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()/60
    evaluations=read(RUN/'evaluations.json',[]);worst=max([e['seconds']/60 for e in evaluations]+[10]);reserve=3*worst+20
    plan['self_play_stop_utc']=(deadline-datetime.timedelta(minutes=3*worst+5)).isoformat();save(RUN/'plan.json',plan);save(REPORT/'plan.json',plan)
    cfg=read(REPORT/'configs/self-play.json');cfg.update(engine=current_engine(),initial_weights=str(Path(ppo['folder'])/'final-weights'),bootstrap_checkpoint=ppo['result']['checkpoint'],minutes=max(1,int(remaining-reserve)),opponent_pool=older_weights+[str(Path(ppo['folder'])/'final-weights')]);cfg['schedule_minutes']=cfg['minutes']
    # Persist the stage duration once. Restarts preserve this schedule and its anchor.
    frozen=RUN/'self-play-stage-config.json'
    if frozen.exists():cfg=read(frozen)
    else:save(frozen,cfg)
    sp=native_training('self-play',cfg,parallel)
    for future in FUTURES:future.result()
    evaluate_group('self-play-final',Path(sp['folder'])/'final-weights',read(RUN/'previous-evaluated.json')['weights'])
    choose_best()
    state('complete',deadline_utc=plan['deadline_utc'],finished_utc=utc());publish('complete all training and evaluations')

def navigation_validation():
    if cached('nav-runtime-validation'):return
    cfg=dict(engine=current_engine(),games=8,parallel_matches=8,seconds=600,seed=20261103,max_blocks=3,observation_profile='chio-v1',teacher_version='v10',zero_sum=True,telemetry=False)
    cp=config('nav-collect-smoke',cfg);folder=RUN/'nav-collect-smoke';r=run_job('nav-collect-smoke',[EXE,'collect-teacher',cp,folder],folder/'report.json')
    manifest=read(folder/'dataset.json');assert manifest['scripted_base_to_mid'] and r['partial']
    audit=run_job('nav-collect-smoke-audit',[PY,'gpu/scripts/p15_dataset_audit.py',folder,REPORT/'nav-collect-smoke-audit.json'],REPORT/'nav-collect-smoke-audit.json');assert audit['passed']
    cfg=read(REPORT/'configs/ppo-v10.json');cfg.update(engine=current_engine(),initial_weights=str(RUN/'dagger-1-il/weights'),matches=2,max_blocks=1,batch_sequences=1,epochs=1,desynchronize=False,minutes=10,checkpoint_minutes=0)
    cp=config('nav-ppo-mask-smoke',cfg);folder=RUN/'nav-ppo-mask-smoke';ppo=run_job('nav-ppo-mask-smoke',[EXE,'train-ggd',cp,folder],folder/'result.json')
    assert ppo['optimizer_steps']>0 and ppo['phase_counts']==[2.,0.,0.,0.],ppo
    result=dict(collect=r,audit=audit,ppo=ppo,expected_trainable_learner_decisions={'numerator':2,'denominator':256},reason='Only Radiant learner decisions126/127 are after scripted handoff in first128-step block; Dire handoff is128.')
    save(REPORT/'nav-runtime-validation.json',result);done('nav-runtime-validation',result,RUN)

def navigation_pipeline():
    """Reuses teacher1024 with exact masks; old experiments remain immutable."""
    full_verification('nav-');navigation_validation()
    teacher=cached('teacher-1024');cfg=read(REPORT/'configs/imitation.json');cfg.update(engine=current_engine(),exclude_premid=True,dataset=teacher['folder']);cfg.pop('initialize',None);cfg.pop('resume',None)
    il=supervised('nav-imitation',cfg);parity('parity-nav-imitation',teacher['folder'],il['folder']);evaluate_group('nav-imitation',Path(il['folder'])/'weights')
    previous=il;older=[str(Path(il['folder'])/'weights')]
    for i in [1,2]:
        cfg=read(REPORT/f'configs/dagger-{i}-collect.json');cfg['collection'].update(engine=current_engine(),parallel_matches=256);cfg['weights']=str(Path(previous['folder'])/'weights')
        data=collect(f'nav-dagger-{i}-256',cfg,dagger=True)
        cfg=read(REPORT/f'configs/dagger-{i}-train.json');cfg.update(engine=current_engine(),exclude_premid=True,dataset=data['folder'],replay_datasets=[teacher['folder']],initialize=str(Path(previous['folder'])/'checkpoint.pt'))
        previous=supervised(f'nav-dagger-{i}-il',cfg);wait(f'nav-dagger-{i}-256-audit',REPORT/f'nav-dagger-{i}-256-audit.json')
        parity(f'parity-nav-dagger-{i}',data['folder'],previous['folder']);evaluate_group(f'nav-dagger-{i}',Path(previous['folder'])/'weights');older.append(str(Path(previous['folder'])/'weights'))
    best=choose_best({'nav-imitation','nav-dagger-1','nav-dagger-2'});folder=str(Path(best['weights']).parent);candidate=next(row for row in read(RUN/'stages.json').values() if row['folder']==folder)
    save(REPORT/'pre-rl-selection.json',dict(best=best['label'],weights=best['weights'],reason=best['reason'],continue_rl_even_if_navigation_or_farm_remains_weak=True));publish('select best scripted-phase model and proceed to RL')
    finish_rl(candidate,il,older)

def main():
    guard=(RUN/'controller.lock').open('a+b');guard.seek(0);guard.write(b'1');guard.flush();guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
    try:
        if read(RUN/'sampling-plan.json',{}).get('active'):
            import p16_sampling_pipeline,sys
            return p16_sampling_pipeline.run(sys.modules[__name__])
        if read(RUN/'navigation-mode.json',{}).get('active'):return navigation_pipeline()
        teacher=collect('teacher-1024',read(REPORT/'configs/collect-1024.json'),adopt=True)
        full_verification()
        cfg=read(REPORT/'configs/imitation.json');cfg['dataset']=teacher['folder'];il=supervised('imitation',cfg)
        wait('teacher-1024-audit',REPORT/'teacher-1024-audit.json');parity('parity-imitation',teacher['folder'],il['folder']);evaluate_group('imitation',Path(il['folder'])/'weights')
        previous=il;capacity=dagger_capacity(Path(il['folder'])/'weights')
        for i in [1,2]:
            cfg=read(REPORT/f'configs/dagger-{i}-collect.json');cfg['weights']=str(Path(previous['folder'])/'weights');cfg['collection']['parallel_matches']=min(256,capacity)
            data=collect(f'dagger-{i}-256',cfg,dagger=True)
            cfg=read(REPORT/f'configs/dagger-{i}-train.json');cfg.update(dataset=data['folder'],replay_datasets=[teacher['folder']],initialize=str(Path(previous['folder'])/'checkpoint.pt'))
            previous=supervised(f'dagger-{i}-il',cfg);wait(f'dagger-{i}-256-audit',REPORT/f'dagger-{i}-256-audit.json')
            parity(f'parity-dagger-{i}',data['folder'],previous['folder']);evaluate_group(f'dagger-{i}',Path(previous['folder'])/'weights')
            if i==1:navigation_gate()
        finish_rl(previous,il,[str(Path(il['folder'])/'weights'),str(Path(read(RUN/'stages.json')['dagger-1-il']['folder'])/'weights'),str(Path(previous['folder'])/'weights')])
    except Exception as error:
        incident('supervisor',repr(error));state('needs-repair',error=repr(error));publish('record interruption for autonomous repair');raise
if __name__=='__main__':main()
