"""Ten-hour P18 continuation: new anchor, preserved Adam, 45/45/10/0 pool."""
import datetime,hashlib,msvcrt,shutil
from pathlib import Path
import p18_run as r
SOURCE=r.RUN/'saved/self-play-360m'
NAME='self-play-continuation'
REFERENCES={'p16-ppo':'C:/AMI/runs/p16/ppo/final-weights','p16-self-play':'C:/AMI/runs/p16/self-play/final-weights','p18-ppo':str(r.RUN/'ppo/final-weights'),'p18-360m':str(SOURCE/'weights')}
QUEUE=r.RUN/'continuation-evaluation-queue.json';PREVIOUS=r.RUN/'continuation-previous.json'
FUTURES=[];QUEUED=set()
def prepare():
    evidence=r.read(r.RUN/'selected-checkpoint.json');verified=0
    for item in evidence['files']:
        assert hashlib.sha256((SOURCE/item['file']).read_bytes()).hexdigest()==item['sha256'];verified+=1
    pool=[]
    def add(path):
        path=str(Path(path).resolve());assert (Path(path)/'manifest.json').is_file(),path
        if path not in pool:pool.append(path)
    for p in sorted((r.RUN/'self-play').glob('checkpoint-*/state.json')):
        for path in r.read(p).get('pool',{}).get('paths',[]):add(path)
    for path in [r.RUN/'ppo/final-weights',r.RUN/'self-play/final-weights',SOURCE/'weights']:add(path)
    latest=r.RUN/'continuation-initial-latest/weights'
    if not latest.exists():shutil.copytree(SOURCE/'weights',latest)
    add(latest)
    cfg=r.read(SOURCE/'config.json');cfg.update(initial_weights=str(SOURCE/'weights'),bootstrap_checkpoint=str(SOURCE),resume_checkpoint=None,minutes=600,schedule_minutes=600,kl_start=.03,kl_end=.01,pool_weights=[45,45,10,0],opponent_pool=pool)
    r.config(NAME,cfg)
    info=dict(source_checkpoint=str(SOURCE),source_weights=str(SOURCE/'weights'),source_manifest_sha256=hashlib.sha256((SOURCE/'weights/manifest.json').read_bytes()).hexdigest(),model_sha256=hashlib.sha256((SOURCE/'model.bpk').read_bytes()).hexdigest(),source_training_seconds=r.read(SOURCE/'state.json')['training_seconds'],verified_files=verified,pool_paths=pool,duration_minutes=600,user_reported_dota_progress=True)
    r.save(r.RUN/'continuation-source.json',info);r.save(r.REPORT/'continuation-source.json',info);return cfg

def historical():
    # Complete the common four fixed references for every previously evaluated P18 model.
    rows=r.read(r.RUN/'evaluations.json',[])
    for label in ['ppo-120m','self-play-120m','self-play-240m','self-play-360m']:
        weights=next(e['weights'] for e in rows if e['label']==label)
        for opponent,path in REFERENCES.items():r.evaluation(label,weights,opponent,path,False)

def queue(label,weights):
    with r.LOCK:
        ledger=r.read(QUEUE,{})
        if label not in ledger:ledger[label]=dict(weights=str(weights));r.save(QUEUE,ledger)
    if label in QUEUED:return
    QUEUED.add(label)
    def work():
        with r.LOCK:
            ledger=r.read(QUEUE);item=ledger[label]
            if 'previous' not in item:item['previous']=r.read(PREVIOUS)['weights'];r.save(QUEUE,ledger)
        r.evaluate_group(label,Path(item['weights']),item['previous'],False)
        r.save(PREVIOUS,dict(label=label,weights=item['weights']))
    FUTURES.append(r.POOL.submit(work))

def train(cfg):
    if r.cached(NAME):return r.cached(NAME)
    cfg=dict(cfg);deadline=datetime.datetime.fromisoformat(r.read(r.RUN/'continuation-plan.json')['deadline_utc'])
    for attempt in range(1,100):
        job=NAME if attempt==1 else f'{NAME}-attempt-{attempt}';folder=r.RUN/job;cp=r.config(job,cfg)
        def tick(native):
            if datetime.datetime.now(datetime.timezone.utc)>=deadline:
                r.save(folder/'stop.json',dict(reason='P18 continuation ten-hour UTC deadline'));return
            boundary=int(native.get('training_seconds',0)//7200)*120
            if boundary:
                label=f'continue-{boundary}m';latest=r.read(folder/'latest.json')
                if label not in r.read(QUEUE,{}) and latest and r.read(Path(latest['path'])/'state.json',{}).get('training_seconds',0)>=boundary*60:queue(label,Path(latest['path'])/'weights')
        try:result=r.run_job(job,[r.EXE,'train-ggd',cp,folder],folder/'result.json',on_tick=tick)
        except RuntimeError as error:
            latest=r.read(folder/'latest.json');r.incident(NAME,str(error),checkpoint=latest)
            if latest:cfg['resume_checkpoint']=latest['path']
            continue
        r.done(NAME,result,folder);return r.cached(NAME)
    raise RuntimeError('Continuation retry limit')

def main():
    if (r.RUN/'continuation-stop.json').exists():raise RuntimeError('Continuation explicitly stopped')
    guard=(r.RUN/'controller.lock').open('a+b');guard.seek(0);guard.write(b'1');guard.flush();guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
    r.REFERENCES=REFERENCES
    try:
        frozen=r.REPORT/f'configs/{NAME}.json';cfg=r.read(frozen) if frozen.exists() else prepare()
        plan=r.read(r.RUN/'continuation-plan.json')
        if not plan:
            now=datetime.datetime.now(datetime.timezone.utc);plan=dict(start_utc=now.isoformat(),deadline_utc=(now+datetime.timedelta(hours=10)).isoformat(),duration_minutes=600,interpretation='UTC wall deadline; native training time excludes preparation; evaluations run concurrently',new_anchor=True,pool_weights=[45,45,10,0]);r.save(r.RUN/'continuation-plan.json',plan)
        if not PREVIOUS.exists():r.save(PREVIOUS,dict(label='self-play-360m',weights=str(SOURCE/'weights')))
        r.state(NAME,deadline_utc=plan['deadline_utc'],source=str(SOURCE));r.publish('start ten-hour continuation with new anchor and 45-45-10-0 pool')
        FUTURES.append(r.POOL.submit(historical))
        for label,item in r.read(QUEUE,{}).items():queue(label,item['weights'])
        result=train(cfg)
        for future in FUTURES:future.result()
        r.evaluate_group('continue-final',Path(result['folder'])/'final-weights',r.read(PREVIOUS)['weights'])
        r.state('complete',deadline_utc=plan['deadline_utc'],finished_utc=r.utc());r.publish('complete ten-hour continuation and all seven-opponent evaluations')
    except Exception as error:
        r.incident('continuation-supervisor',repr(error));r.state('needs-repair',error=repr(error));r.publish('record continuation interruption for repair');raise
if __name__=='__main__':main()
