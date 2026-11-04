"""Revised P08 stages with sampled WDDM memory, shared budget and immediate git reports."""
import argparse, json, os, re, subprocess, time
from pathlib import Path
import p08_run as common
ROOT, REPO = common.ROOT, common.REPO
REPORT = REPO/'reports/P08/revised'
EXE = ROOT/'revised-gpu.exe'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
def budget():
    base=read(REPORT/'base-budget.json')
    records=[read(p) for p in REPORT.glob('stage-*.json')]
    used=base['legacy_gpu_process_wall_seconds']+base['legacy_gpu_test_seconds']+base.get('revised_gpu_test_seconds',0)+sum(r['wall_seconds'] for r in records)
    return dict(limit_seconds=21600,used_seconds=used,remaining_seconds=21600-used,reserved_test_seconds=300)
def publish(stage):
    if (ROOT/stage/'history.jsonl').exists():
        import p08_results
        p08_results.OUT=REPORT;p08_results.curves(stage)
    save(REPORT/'budget.json',budget())
    for p in REPORT.iterdir():
        if p.is_file() and p.stat().st_size>=1_000_000:raise RuntimeError(f'oversized report {p}')
    subprocess.run(['git','add','reports/P08/revised'],cwd=REPO,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).splitlines()
    if not all(p.startswith('reports/P08/revised/') for p in staged):raise RuntimeError('unrelated staged changes; publish stopped')
    subprocess.run(['git','diff','--cached','--stat'],cwd=REPO,check=True)
    if staged:
        subprocess.run(['git','commit','-m',f'[P08] experiment: record {stage}'],cwd=REPO,check=True)
        subprocess.run(['git','push','-u','origin','agent/P08-ppo-pilot'],cwd=REPO,check=True)

def collect_evaluations(stage):
    """Export completed in-process hourly evaluations; their wall time belongs to the parent stage."""
    import p08_results
    added=[]
    for path in sorted((ROOT/stage).glob('eval-*/report.json')):
        name=f'{stage}-{path.parent.name}'
        destination=REPORT/f'{name}.json'
        if destination.exists():continue
        report=read(path)
        if not report.get('complete'):continue
        payload=(json.dumps(report,separators=(',',':'))+'\n').encode('utf-8')
        if len(payload)>=1_000_000:raise RuntimeError(f'oversized evaluation report {path}')
        (REPORT/f'{name}-matches.json').write_bytes(payload)
        save(destination,dict(parent_stage=stage,source=str(path),evaluation=p08_results.evaluation(path),budget_note='Included in parent process wall time; do not add again'))
        added.append(name)
    return added
def run_stage(name,cmd,limit,allow_failure=False):
    assert re.fullmatch(r'rev-[a-z0-9-]+',name)
    if (REPORT/f'stage-{name}.json').exists():raise RuntimeError('stage already recorded; use a fresh attempt name')
    remaining=budget()['remaining_seconds']-300
    if limit>remaining:raise RuntimeError(f'budget guard: requested wall limit {limit}, available {remaining}')
    root=ROOT/name;root.mkdir(exist_ok=True)
    started=time.time();timed_out=False
    with (ROOT/f'{name}.stdout.log').open('w')as out,(ROOT/f'{name}.stderr.log').open('w')as err:
        child=subprocess.Popen(cmd,cwd=REPO,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        monitor=subprocess.Popen(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(REPO/'gpu/scripts/p08_memory.ps1'),'-TargetPid',str(child.pid),'-OutputPath',str(root/'gpu-memory.jsonl')],cwd=REPO,stdout=subprocess.DEVNULL,stderr=(ROOT/f'{name}-memory.err').open('w'),creationflags=subprocess.CREATE_NO_WINDOW)
        state=dict(stage=name,controller_pid=os.getpid(),child_pid=child.pid,argv=cmd,command=subprocess.list2cmdline(cmd),artifacts=str(root),started_unix=started,wall_limit_seconds=limit)
        while child.poll()is None:
            state['elapsed_seconds']=time.time()-started
            if (root/'status.json').exists():
                try:state['progress']=read(root/'status.json')
                except (OSError,json.JSONDecodeError):pass
            save(ROOT/'revised-controller.json',state);common.shared(state)
            for evaluation in collect_evaluations(name):publish(evaluation)
            if state['elapsed_seconds']>limit:
                timed_out=True;child.terminate();child.wait(timeout=30);break
            time.sleep(2)
        elapsed=time.time()-started
        try:monitor.wait(timeout=10)
        except subprocess.TimeoutExpired:monitor.terminate();monitor.wait()
    samples=[]
    if (root/'gpu-memory.jsonl').exists():
        for line in (root/'gpu-memory.jsonl').read_text(encoding='utf-8-sig').splitlines():
            try:samples.append(json.loads(line))
            except json.JSONDecodeError:pass
    history=[json.loads(x)for x in (root/'history.jsonl').read_text(encoding='utf-8').splitlines()]if(root/'history.jsonl').exists()else[]
    result=read(root/'result.json')if(root/'result.json').exists()else None
    peak={k:max((float(x.get(k)or 0) for x in samples),default=0)for k in ['dedicated_bytes','shared_bytes','committed_bytes','adapter_dedicated_bytes']}
    record=dict(stage=name,argv=cmd,command=state['command'],returncode=child.returncode,timed_out=timed_out,wall_seconds=elapsed,memory_samples=len(samples),sampled_peak_memory=peak,memory_measurement='WDDM process/adapter counters sampled every 2 seconds plus query time; sampled peaks',result=result,history=history)
    if history:
        decisions=sum(h['learning']['valid_decisions']for h in history)
        record.update(learner_decisions=decisions,collection_decisions_per_second=decisions/sum(h['collection_seconds']for h in history),learner_decisions_per_second=decisions/sum(h['collection_seconds']+h['optimization_seconds']for h in history))
    if(root/'provenance.json').exists():record['provenance']=read(root/'provenance.json')
    if(root/'config.json').exists():record['config']=read(root/'config.json')
    if len(history)>100:
        files=[]
        for start in range(0,len(history),100):
            filename=f'{name}-history-{start//100:03}.json';save(REPORT/filename,history[start:start+100]);files.append(filename)
        record['history_files']=files;record['history']=[]
    if(root/'report.json').exists():
        import p08_results
        record['evaluation']=p08_results.evaluation(root/'report.json')
        (REPORT/f'{name}-matches.json').write_bytes((json.dumps(read(root/'report.json'),separators=(',',':'))+'\n').encode())
    save(REPORT/f'stage-{name}.json',record)
    state.update(stage='complete'if child.returncode==0 else'failed',returncode=child.returncode,timed_out=timed_out)
    save(ROOT/'revised-controller.json',state);common.shared(state)
    collect_evaluations(name);publish(name)
    if child.returncode and not allow_failure:raise SystemExit(child.returncode)
    return record

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    sub.add_parser('capacity')
    train=sub.add_parser('train');train.add_argument('name');train.add_argument('config',type=Path);train.add_argument('--wall-limit',type=float,required=True)
    ev=sub.add_parser('evaluate');ev.add_argument('name');ev.add_argument('weights');ev.add_argument('--matches',type=int,default=32);ev.add_argument('--selection',choices=['sampling','argmax'],default='sampling');ev.add_argument('--wall-limit',type=float,default=900)
    a=p.parse_args();import msvcrt
    with(ROOT/'controller.lock').open('a+b')as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        if a.mode=='capacity':
            for n in [64,128,192]:
                name=f'rev-cap{n}';cfg=REPORT/f'cap{n}.json'
                run_stage(name,[str(EXE),'train-ggd',str(cfg),str(ROOT/name)],300,allow_failure=True)
        elif a.mode=='train':
            cfg=read(a.config)
            if cfg['matches']==16 or cfg['learning_rate']==3e-6:raise RuntimeError('superseded configuration forbidden')
            run_stage(a.name,[str(EXE),'train-ggd',str(a.config.resolve()),str(ROOT/a.name)],a.wall_limit)
        else:
            engine=read(REPORT/'cap64.json')['engine']
            run_stage(a.name,[str(EXE),'evaluate-ggd',engine,a.weights,str(ROOT/a.name),'--mode','teacher','--matches',str(a.matches),'--seconds','600','--seed','20261103','--'+a.selection,'--trace'],a.wall_limit)
if __name__=='__main__':main()
