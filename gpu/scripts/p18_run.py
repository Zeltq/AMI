"""Autonomous P18 stage supervisor; no policy-contract changes or metric-based RL stopping."""
import concurrent.futures,ctypes,datetime,json,msvcrt,os,shutil,subprocess,threading,time
from pathlib import Path
from p18_report import refresh
from p15_eval_summary import summarize
ROOT=Path(__file__).resolve().parents[2];RUN=Path('C:/AMI/runs/p18');REPORT=ROOT/'reports/P18'
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
def current_engine():return 'C:/AMI/runs/p16/environment-script-navigation/engine.json'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def alive(pid):
    if not pid:return False
    kernel=ctypes.windll.kernel32;kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:return False
    code=ctypes.c_ulong();kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(code));kernel.CloseHandle(ctypes.c_void_p(handle));return code.value==259

def state(stage,**fields):
    with LOCK:
        value=dict(stage=stage,controller_pid=os.getpid(),updated_utc=utc(),background_work_remaining=stage not in ('complete','stopped-by-user'),**fields);save(RUN/'state.json',value)
        p=Path('C:/AMI/runs/pilot-status.json');shared=read(p,{});shared['P18']=value;save(p,shared)
def incident(stage,reason,**fields):
    with LOCK:
        rows=read(RUN/'incidents.json',[]);rows.append(dict(utc=utc(),stage=stage,reason=reason,**fields));save(RUN/'incidents.json',rows)
def publish(message):
    with LOCK:
        refresh()
        for p in REPORT.rglob('*'):
            if p.is_file():assert p.stat().st_size<1000000,(p,p.stat().st_size)
        subprocess.run(['git','add','reports/P18.md','reports/P18'],cwd=ROOT,check=True)
        staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
        assert all(x=='reports/P18.md' or x.startswith('reports/P18/') for x in staged),staged
        subprocess.run(['git','diff','--cached','--stat'],cwd=ROOT,check=True)
        if staged:subprocess.run(['git','commit','-m','[P18] '+message],cwd=ROOT,check=True)
        for attempt in range(4):
            result=subprocess.run(['git','-c','core.sshCommand=C:/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=15','push','-u','origin','agent/P18-chio-reward-v2'],cwd=ROOT,timeout=120)
            if result.returncode==0:break
            time.sleep(10)
        else:raise RuntimeError('Git push failed four times; local results are committed')
def config(name,value):
    p=REPORT/'configs'/f'{name}.json';save(p,value);return p

def launch(name,cmd):
    job=RUN/'jobs'/name
    if (job/'status.json').exists():return
    subprocess.Popen([PY,'gpu/scripts/p18_job.py',name,'--',*map(str,cmd)],cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
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
def evaluation(label,weights,opponent,previous=None,foreground=True):
    name='eval-'+label+'-'+opponent+'-sampling'
    with LOCK:
        existing=[e for e in read(RUN/'evaluations.json',[]) if e['label']==label and e['opponent']==opponent and e.get('complete') and e.get('sampling',False)]
    if existing:return existing[0]
    cfg=read(REPORT/'configs'/('eval-ppo-120m-previous-sampling.json' if previous else 'eval-ppo-120m-v10-sampling.json'));cfg.update(weights=str(weights),right='v9' if opponent=='v9' else 'v10');cfg.pop('opponent_weights',None)
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
    evaluation(label,weights,'v10',foreground=foreground);evaluation(label,weights,'v9',foreground=foreground)
    if previous:evaluation(label,weights,'previous',previous,foreground)
    for opponent,path in REFERENCES.items():evaluation(label,weights,opponent,path,foreground)
    with LOCK:save(RUN/'previous-evaluated.json',dict(label=label,weights=str(weights)))

def queue_evaluation(label,weights,parallel):
    def work():
        previous=read(RUN/'previous-evaluated.json',{}).get('weights')
        return evaluate_group(label,weights,previous,not parallel)
    if parallel:
        queued=read(RUN/'evaluation-queue.json',{})
        if label not in queued:queued[label]=str(weights);save(RUN/'evaluation-queue.json',queued)
        if label not in QUEUED:FUTURES.append(POOL.submit(work));QUEUED.add(label)
    else:work()
QUEUED=set()

def native_training(name,cfg,parallel=False):
    if cached(name):return cached(name)
    cfg=dict(cfg)
    for attempt in range(1,100):
        job=name if attempt==1 else name+f'-attempt-{attempt}';folder=RUN/job
        cp=config(job,cfg)
        def tick(native):
            seconds=native.get('training_seconds',0)
            if name!='self-play':return
            plan=read(RUN/'plan.json');deadline=datetime.datetime.fromisoformat(plan['deadline_utc'])
            if datetime.datetime.now(datetime.timezone.utc)>=deadline:
                save(folder/'stop.json',dict(reason='P18 twelve hour deadline'));return
            completed=set(read(RUN/'evaluation-queue.json',{})) if parallel else {e['label'] for e in read(RUN/'evaluations.json',[]) if e['opponent']=='p16-self-play' and e.get('complete')}
            boundary=int(seconds//7200)*120
            if boundary and f'self-play-{boundary}m' not in completed:
                if parallel:
                    latest=read(folder/'latest.json')
                    if latest and read(Path(latest['path'])/'state.json',{}).get('training_seconds',0)>=boundary*60:queue_evaluation(f'self-play-{boundary}m',Path(latest['path'])/'weights',True)
                else:save(folder/'stop.json',dict(reason='scheduled evaluation',label=f'self-play-{boundary}m'))
        try:r=run_job(job,[EXE,'train-ggd',cp,folder],folder/'result.json',on_tick=tick)
        except RuntimeError as error:
            latest=read(folder/'latest.json');incident(name,str(error),checkpoint=latest)
            if latest:cfg['resume_checkpoint']=latest['path']
            elif attempt>1 and cfg.get('resume_checkpoint'):pass
            continue
        stop=read(folder/'stop.json',{})
        if stop.get('label'):
            queue_evaluation(stop['label'],folder/'final-weights',False)
        deadline=datetime.datetime.fromisoformat(read(RUN/'plan.json')['deadline_utc'])
        if name=='self-play' and stop.get('label') and r['training_seconds']<cfg['minutes']*60 and datetime.datetime.now(datetime.timezone.utc)<deadline:
            cfg['resume_checkpoint']=r['checkpoint'];continue
        done(name,r,folder);return cached(name)
    raise RuntimeError('Native retry limit')

REFERENCES={'p16-dagger':'C:/AMI/runs/p16/sampling-dagger-2-il/weights','p16-ppo':'C:/AMI/runs/p16/ppo/final-weights','p16-self-play':'C:/AMI/runs/p16/self-play/final-weights'}
def main():
    if (RUN/'user-stop.json').exists():
        print('P18 stopped by user; explicit authorization required to resume.');return
    guard=(RUN/'controller.lock').open('a+b');guard.seek(0);guard.write(b'1');guard.flush();guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
    try:
        plan=read(RUN/'plan.json')
        if not plan:
            now=datetime.datetime.now(datetime.timezone.utc);plan=dict(start_critic_utc=now.isoformat(),deadline_utc=(now+datetime.timedelta(hours=12)).isoformat());plan['self_play_stop_utc']=plan['deadline_utc'];save(RUN/'plan.json',plan)
        for label,weights in read(RUN/'evaluation-queue.json',{}).items():queue_evaluation(label,Path(weights),True)
        state('critic-start',deadline_utc=plan['deadline_utc']);publish('record twelve hour deadline and start critic')
        critic=native_training('critic',read(REPORT/'configs/critic.json'))
        old=read(Path(REFERENCES['p16-dagger'])/'manifest.json');new=read(Path(critic['folder'])/'final-weights/manifest.json');keys=[k for k in old['tensors'] if not k.startswith('value_')]
        equal=sum(old['tensors'][k]['sha256']==new['tensors'][k]['sha256'] for k in keys);assert equal==len(keys)==45
        save(REPORT/'critic-freeze.json',dict(equal=equal,total=len(keys)));publish('verify frozen policy')
        if not read(RUN/'previous-evaluated.json'):save(RUN/'previous-evaluated.json',dict(label='critic-start-policy',weights=str(Path(critic['folder'])/'final-weights')))
        cfg=read(REPORT/'configs/ppo.json');cfg.update(initial_weights=str(Path(critic['folder'])/'final-weights'),bootstrap_checkpoint=critic['result']['checkpoint'])
        ppo=native_training('ppo',cfg);queue_evaluation('ppo-120m',Path(ppo['folder'])/'final-weights',True)
        frozen=RUN/'self-play-stage-config.json'
        if frozen.exists():cfg=read(frozen)
        else:
            minutes=max(1,(datetime.datetime.fromisoformat(plan['deadline_utc'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()/60)
            latest=RUN/'self-play-initial-latest/weights'
            if not latest.exists():shutil.copytree(Path(ppo['folder'])/'final-weights',latest)
            cfg=read(REPORT/'configs/self-play.json');cfg.update(initial_weights=str(Path(ppo['folder'])/'final-weights'),bootstrap_checkpoint=ppo['result']['checkpoint'],minutes=minutes,schedule_minutes=minutes,opponent_pool=list(REFERENCES.values())+[str(Path(ppo['folder'])/'final-weights'),str(latest)]);save(frozen,cfg)
        sp=native_training('self-play',cfg,True)
        for future in FUTURES:future.result()
        evaluate_group('self-play-final',Path(sp['folder'])/'final-weights',read(RUN/'previous-evaluated.json')['weights'])
        state('complete',deadline_utc=plan['deadline_utc'],finished_utc=utc());publish('complete training and all evaluations')
    except Exception as error:
        incident('supervisor',repr(error));state('needs-repair',error=repr(error));publish('record interruption for autonomous repair');raise
if __name__=='__main__':main()
