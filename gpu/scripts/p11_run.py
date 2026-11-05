"""P11 process supervisor. Artifacts stay outside git; evidence is published separately."""
import argparse,hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=Path('C:/AMI/runs/p11');REPORT=ROOT/'reports/P11'
EXE=RUN/'sf-duel-gpu.exe';TARGET=RUN/'target'
ENGINE='C:/AMI/runs/p07/dota-calibration-005/engine.json'
START=Path('C:/AMI/runs/p09/P09c/continue90/rl-continue-001/checkpoint-000470')
POOL=['C:/AMI/runs/p09/P09c/rl-pilot1-001/checkpoint-000126/weights','C:/AMI/runs/p09/P09c/rl-main-001/checkpoint-000314/weights',str(START/'weights')]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');temp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(**row):
    v=dict(controller_pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),worktree=str(ROOT),artifacts=str(RUN),**row)
    save(RUN/'state.json',v)
    p=Path('C:/AMI/runs/pilot-status.json')
    for _ in range(5):
        try:
            shared=read(p);shared['P11']=v;save(p,shared);break
        except (OSError,ValueError):time.sleep(.2)
def history():return read(RUN/'commands.json') if (RUN/'commands.json').exists() else []
def run(name,kind,cmd,artifact=None,on_tick=None):
    cmd=list(map(str,cmd));log=RUN/(name+'.log');started=time.monotonic()
    with log.open('w',encoding='utf-8') as f:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        row=dict(stage=name,kind=kind,command_literal=subprocess.list2cmdline(cmd),pid=child.pid,artifact=str(artifact) if artifact else None,log=str(log),started_utc=datetime.now(timezone.utc).isoformat(),exit_code=None)
        hist=history();hist.append(row);save(RUN/'commands.json',hist)
        while child.poll() is None:
            native={}
            if artifact and (Path(artifact).parent/'status.json').exists():
                try:native=read(Path(artifact).parent/'status.json')
                except (OSError,ValueError):pass
            status(stage=name,pid=child.pid,command=row['command_literal'],log=str(log),native=native,background_work_remaining=True)
            if on_tick:on_tick(child,native)
            time.sleep(10)
        row['exit_code']=child.returncode
        if kind in ('training','evaluation'):row['minutes']=(time.monotonic()-started)/60
        hist[-1]=row;save(RUN/'commands.json',hist)
        status(stage=name+' '+('complete' if child.returncode==0 else 'failed'),pid=None,exit_code=child.returncode,background_work_remaining=True)
        if child.returncode:raise RuntimeError(f'{name}: exit {child.returncode}; {log}')
    return read(artifact) if artifact else None
def publish(message):
    from p11_report import refresh
    refresh()
    subprocess.run(['git','add','reports/P11.md','reports/P11'],cwd=ROOT,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    assert all(p=='reports/P11.md' or p.startswith('reports/P11/') for p in staged),staged
    subprocess.run(['git','diff','--cached','--stat'],cwd=ROOT,check=True)
    if staged:subprocess.run(['git','commit','-m','[P11] '+message],cwd=ROOT,check=True)
    subprocess.run(['git','-c','core.sshCommand=C:/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=15','push','-u','origin','agent/P11-selfplay'],cwd=ROOT,check=True,timeout=120)
def base_config():
    return dict(engine=ENGINE,initial_weights=str(START/'weights'),bootstrap_checkpoint=str(START),matches=192,seed=20261105,minutes=10,max_blocks=0,batch_sequences=48,epochs=2,learning_rate=3e-5,critic_learning_rate=3e-4,warmup_blocks=60,warmup_min_blocks=20,entropy=.005,kl_start=.03,kl_end=.01,stop_kl=.02,checkpoint_minutes=10,evaluate=False,evaluation_minutes=120,skip_warmup=True,critic_only=False,desynchronize=True,verify_phase_histogram=True,teacher_version='v10',reward_mode='event-v2',mana_weight=.5,separate_critic_lr=True,resume_checkpoint=None,schedule_minutes=240,stop_at_minutes=0,kl_decay_start_minutes=0,begin_decay_on_resume=False,opponent_pool=POOL,snapshot_minutes=60)
def verify(prefix=''):
    cmd=['cargo','test','--release','--locked','--manifest-path',ROOT/'gpu/Cargo.toml','--target-dir',TARGET]
    run(prefix+'rust-targeted','verification',cmd+['p11_','--','--nocapture','--test-threads=1'])
    run(prefix+'rust-full','verification',cmd+['--','--test-threads=1'])
    run(prefix+'build','build',['cargo','build','--release','--locked','--manifest-path',ROOT/'gpu/Cargo.toml','--target-dir',TARGET])
    import shutil
    shutil.copy2(TARGET/'release/sf-duel-gpu.exe',EXE)
    save(REPORT/'verification.json',dict(executable_sha256=sha(EXE),commands=[r for r in history() if r['kind'] in ('verification','build')]))
def smoke():
    cfg=base_config();cfg.update(learning_rate=0,critic_learning_rate=0,checkpoint_minutes=10)
    save(REPORT/'smoke.json',cfg)
    result=run('smoke','training',[EXE,'train-ggd',REPORT/'smoke.json',RUN/'smoke'],RUN/'smoke/result.json')
    weights=Path(result['latest'])/'weights'
    comparisons={p.name:sha(p)==sha(weights/p.name) for p in (START/'weights').glob('*.f32')}
    assert len(comparisons)==59 and all(comparisons.values()),comparisons
    from p09b_results import lines
    episodes=lines(RUN/'smoke/episodes.jsonl');types={k:sum(r['game']['opponent_type']==k for r in episodes) for k in ('latest','older','v10','v9')}
    assert all(types.values()),types
    assert all(r['game']['simulation_error']==0 for r in episodes)
    save(REPORT/'smoke-result.json',dict(result=result,unchanged_tensors=comparisons,completed_by_type=types,pool=read(RUN/'smoke/pool.json')))
    publish('publish 10 minute zero learning rate smoke')

def checkpoints(prefix='train-'):
    return sorted([(read(p)['training_seconds'],p.parent) for p in RUN.glob(prefix+'*/checkpoint-*/progress.json') if all((p.parent/f).exists() for f in ('model.bpk','optimizer.bpk','weights/manifest.json'))],key=lambda x:x[0])

def training_budget(remaining):
    return max(t for t in range(1,int(remaining)) if t+36*int(t//120)+5<=remaining)

def warmup():
    completed=RUN/'warmup-result.json'
    if completed.exists():return read(completed)
    cfg=base_config();cfg.update(skip_warmup=False,critic_only=True,minutes=120,kl_start=.03,kl_end=.03,schedule_minutes=120,checkpoint_minutes=3)
    while True:
        available=checkpoints('warmup-')
        if available:cfg.update(resume_checkpoint=str(available[-1][1]),bootstrap_checkpoint=None)
        attempt=1+len(list(RUN.glob('warmup-*/config.json')));name=f'warmup-{attempt:03}';folder=RUN/name
        save(REPORT/(name+'.json'),cfg)
        try:result=run(name,'training',[EXE,'train-ggd',REPORT/(name+'.json'),folder],folder/'result.json');break
        except RuntimeError:
            publish('fix: retain failed warmup evidence and resume checkpoint')
            if datetime.now(timezone.utc)>=datetime(2026,11,5,8,0,tzinfo=timezone.utc):raise
    weights=Path(result['latest'])/'weights'
    manifest=read(START/'weights/manifest.json')
    policy=[START/'weights'/v['file'] for k,v in manifest['tensors'].items() if not k.startswith(('value_first','value_second'))]
    unchanged=sum(sha(p)==sha(weights/p.name) for p in policy)
    assert len(policy)==55 and unchanged==55,(len(policy),unchanged)
    save(completed,result);save(REPORT/'warmup-result.json',dict(result=result,unchanged_policy_tensors=unchanged))
    publish('publish mixed pool critic warmup');return result

def train(warm):
    plan_path=RUN/'plan.json';cutoff=datetime(2026,11,5,8,38,tzinfo=timezone.utc)
    if plan_path.exists():plan=read(plan_path)
    else:
        remaining=(cutoff-datetime.now(timezone.utc)).total_seconds()/60
        # Reserve 36 minutes for every intervening three-opponent evaluation.
        target=training_budget(remaining)
        plan=dict(training_minutes=target,matches=192,training_stop_utc=cutoff.isoformat(),deadline_utc='2026-11-05T09:30:00+00:00',evaluation_reserve_minutes=36,reason='Fixed clock budget before PPO; no metric-based early stopping')
        save(plan_path,plan);save(REPORT/'plan.json',plan)
    cfg=base_config();cfg.update(bootstrap_checkpoint=warm['latest'],minutes=plan['training_minutes'],schedule_minutes=plan['training_minutes'],evaluate=True)
    state_path=RUN/'publications.json';publications=read(state_path) if state_path.exists() else dict(reports=[],last_minutes=0)
    def tick(child,native):
        now=datetime.now(timezone.utc)
        if now>=cutoff:save(folder/'stop.json',dict(reason='Hard deadline reserve',utc=now.isoformat()))
        reports=[str(p) for p in RUN.glob('train-*/eval-*/report.json')]
        elapsed=native.get('latest',{}).get('training_seconds',native.get('training_seconds',0))/60
        new=set(reports)-set(publications['reports'])
        if new or elapsed-publications['last_minutes']>=60:
            try:
                publish('publish scheduled evaluation' if new else 'publish hourly training curves')
                publications['reports']=reports;publications['last_minutes']=elapsed;save(state_path,publications)
            except Exception as exc:
                save(RUN/'publication-error.json',dict(error=str(exc),utc=now.isoformat()))
    while True:
        available=checkpoints()
        if available:
            elapsed,path=available[-1]
            if elapsed>=cfg['minutes']*60 or datetime.now(timezone.utc)>=cutoff:
                return dict(latest=str(path),training_seconds=elapsed,recovered=True)
            cfg.update(resume_checkpoint=str(path),bootstrap_checkpoint=None)
        attempt=1+len(list(RUN.glob('train-*/config.json')));name=f'train-{attempt:03}';folder=RUN/name
        save(REPORT/(name+'.json'),cfg)
        try:
            result=run(name,'training',[EXE,'train-ggd',REPORT/(name+'.json'),folder],folder/'result.json',tick)
            save(RUN/'training-result.json',result);publish('publish completed PPO and curves');return result
        except RuntimeError:
            publish('fix: retain failed PPO evidence and resume checkpoint')
            if datetime.now(timezone.utc)>=cutoff and not checkpoints():raise

def evaluate(model,label,other,opponent=None):
    registry_path=RUN/'evaluations.json';registry=read(registry_path) if registry_path.exists() else []
    for seed in (20261103,20261104):
        name=f'eval-{label}-vs-{other}-{seed}';folder=RUN/name
        # A scheduled series can coincide with the final checkpoint. Reuse only
        # identical weights, opponent, sampling and seed, and preserve provenance.
        for candidate in sorted(RUN.glob('train-*/eval-*/report.json')):
            r=read(candidate);cfg=read(candidate.parent/'config.json')
            op=read(candidate.parent/'opponent.json')['weights'] if (candidate.parent/'opponent.json').exists() else None
            same_op=(Path(op)==Path(opponent)/'weights') if op and opponent else (op is None and opponent is None and cfg['teacher_version']==('v9' if other=='v9' else 'v10'))
            if r.get('complete') and r['seed']==seed and r['matches']==64 and not cfg['argmax'] and Path(cfg['weights'])==Path(model)/'weights' and same_op:
                folder=candidate.parent;break
        if not (folder/'report.json').exists():
            cmd=[EXE,'evaluate-ggd',ENGINE,Path(model)/'weights',folder,'--mode','teacher','--matches',64,'--seconds',600,'--seed',seed,'--sampling','--teacher-version','v9' if other=='v9' else 'v10']
            if opponent:cmd+=['--opponent-weights',Path(opponent)/'weights']
            while True:
                if (folder/'config.json').exists():
                    folder=RUN/(name+f'-attempt-{len(list(RUN.glob(name+"*"))):03}');cmd[4]=folder
                try:run(folder.name,'evaluation',cmd,folder/'report.json');break
                except RuntimeError:
                    publish('fix: retain failed evaluation and retry a fresh directory')
                    if datetime.now(timezone.utc)>=datetime(2026,11,5,9,25,tzinfo=timezone.utc):raise
        entry=dict(model=label,opponent=other,report=str(folder/'report.json'))
        if entry not in registry:registry.append(entry);save(registry_path,registry)
    publish('publish '+label+' evaluation vs '+other)

def overnight():
    assert (REPORT/'smoke-result.json').exists(),'Verified smoke required'
    warm=warmup();result=train(warm);final=Path(result['latest'])
    desired=result['training_seconds']-7200
    earlier=min(checkpoints(),key=lambda x:abs(x[0]-desired)) if desired>0 else (0,START)
    save(REPORT/'final-reference.json',dict(final=str(final),reference=str(earlier[1]),final_training_seconds=result['training_seconds'],reference_training_seconds=earlier[0],age_minutes=(result['training_seconds']-earlier[0])/60))
    for other in ('v10','v9'):evaluate(final,'final',other)
    reference_label='000470' if earlier[1]==START else f'{earlier[0]/3600:.2f}h'
    evaluate(final,'final',reference_label,earlier[1])
    if earlier[1]!=START:evaluate(final,'final','000470',START)
    notes=[]
    if datetime.now(timezone.utc)<datetime(2026,11,5,9,12,tzinfo=timezone.utc):evaluate(START,'000470','v9')
    else:notes.append('Новая контрольная серия000470–v9 пропущена: остаток до дедлайна меньше18мин.')
    from p11_audit import finish
    finish()
    save(RUN/'completion.json',dict(stage='complete',checkpoint=str(final),training_seconds=result['training_seconds'],notes=notes))
    publish('publish final evaluation and report')
def main():
    RUN.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (RUN/'controller.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        mode=sys.argv[1]
        try:
            if mode=='verify':verify()
            elif mode=='smoke':smoke()
            elif mode=='overnight':overnight()
            elif mode=='all':
                verify('final-');smoke();overnight()
            else:raise ValueError(mode)
            status(stage=mode+' complete',pid=None,background_work_remaining=False)
        except Exception as e:status(stage=mode+' failed',pid=None,error=str(e),background_work_remaining=True);raise
if __name__=='__main__':main()
