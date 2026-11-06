"""Wait for frozen P16 checkpoints, copy them, then run serial CPU Dota series."""
import argparse
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from play_vs_chio import below_normal,save
from p17_matches import match
from p17_summary import summarize
from p17_p16_guard import check as guard,alive
from sf_duel.chio_cpu import load_bundle

RUN=Path('C:/AMI/runs/p17');P16=Path('C:/AMI/runs/p16');P16ROOT=Path('C:/AMI-wt/P16')


def read(path,default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def fingerprint(weights):
    m=read(weights/'manifest.json')
    return hashlib.sha256(json.dumps({k:(v['shape'],v.get('sha256')) for k,v in m['tensors'].items()},sort_keys=True).encode()).hexdigest()


def candidates():
    stages=read(P16/'stages.json',{});result=[]
    # DAgger 2 is mandatory even if a newer RL checkpoint already exists.
    if 'dagger-2-il' in stages and 'parity-dagger-2' in stages:
        path=Path(stages['dagger-2-il']['folder'])/'weights'
        result.append(('dagger-2',path))
    for folder in [*P16.glob('ppo'),*P16.glob('ppo-attempt-*'),*P16.glob('self-play'),*P16.glob('self-play-attempt-*')]:
        latest=read(folder/'latest.json',{})
        if latest.get('block',0)>0:
            weights=Path(latest['path'])/'weights'
            if (weights/'manifest.json').exists():result.append((folder.name+'-'+Path(latest['path']).name,weights))
        if (folder/'result.json').exists() and (folder/'final-weights/manifest.json').exists():result.append((folder.name+'-final',folder/'final-weights'))
    for _,weights in result:
        if not weights.resolve().is_relative_to(P16.resolve()):raise ValueError('P16 checkpoint path escapes runs/p16')
    return result


def simulator_reports(weights):
    result=[];sha=fingerprint(weights)
    for e in read(P16/'evaluations.json',[]):
        if not e.get('complete'):continue
        source=Path(e['weights'])
        if not (source/'manifest.json').exists() or fingerprint(source)!=sha:continue
        path=P16ROOT/e['summary']
        if path.exists():result.append(dict(evaluation=e,summary=read(path)))
    return result


def publish(message):
    paths=['reports/P17.md','reports/P17']
    subprocess.run(['git','add',*paths],cwd=ROOT,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    if any(p!='reports/P17.md' and not p.startswith('reports/P17/') for p in staged):raise RuntimeError('Unrelated staged files; refusing automatic commit')
    if not staged:return
    subprocess.run(['git','diff','--cached','--stat'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','[P17] '+message],cwd=ROOT,check=True)
    subprocess.run(['git','-c','core.sshCommand=C:/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=15','push'],cwd=ROOT,check=True,timeout=90)


def refresh_report(series,done=False):
    path=ROOT/'reports/P17.md';text=path.read_text(encoding='utf-8')
    begin='<!-- P17_SERIES_BEGIN -->';end='<!-- P17_SERIES_END -->'
    lines=[begin,'| Модель P16 | Матчи | W/D/L | LH / deny / смерти | Мид | Отчёты симулятора |','|---|---:|---|---|---:|---:|']
    for name,item in series.items():
        games=item.get('games',[]);n=len(games)
        wins=sum(g['winner']==g['model_team'] for g in games);draws=sum(g['winner']==0 for g in games)
        sides=[g['summary']['sides'][str(g['model_team'])] for g in games]
        lh=sum(s['lh'] for s in sides);deny=sum(s['deny'] for s in sides);deaths=sum(s['deaths'] for s in sides);mid=sum(s['mid_time'] is not None for s in sides)
        lines.append(f"|{name}|{n}/8|{wins}/{n};{draws}/{n};{n-wins-draws}/{n}|{lh}/{n};{deny}/{n};{deaths}/{n}|{mid}/{n}|{len(item.get('simulator',[]))}/2|")
    if len(lines)>18:lines=lines[:2]+lines[-16:] # Full history remains in JSON.
    lines+=[end]
    replacement='\n'.join(lines)
    if begin in text:text=text[:text.index(begin)]+replacement+text[text.index(end)+len(end):]
    else:text=text.replace('## Приоритет P16',replacement+'\n\n## Приоритет P16')
    dagger=series.get('dagger-2',{});count=len(dagger.get('games',[]))
    text=text.replace('P16 DAgger2:0/8 матчей, веса ещё ожидаются; RL:0/8, ещё не появились.',f'P16: подробные результаты и события в [series.json](P17/series.json); DAgger2 {count}/8 матчей.')
    if done:
        text='\n'.join(text.splitlines()[:-1])+"\nBLOCKED P17: серверная часть завершена; Local Host с человеком не проверен, полный replay наблюдений недоступен без GPU\n"
    if len(text.splitlines())>150:raise RuntimeError('Report exceeds150 lines')
    path.write_text(text,encoding='utf-8');save(ROOT/'reports/P17/series.json',series)


def run():
    below_normal();series=read(RUN/'series.json',{});state_path=RUN/'supervisor-status.json'
    def state(stage,**kw):
        row=dict(stage=stage,pid=os.getpid(),command=subprocess.list2cmdline([sys.executable,*sys.argv]),
                 cwd=str(ROOT),artifacts=[str(RUN/'series.json'),str(state_path)],background_work_remaining=stage!='complete',**kw)
        save(state_path,row)
        if stage!='matches':save(ROOT/'runs/pilot-status.json',{'P17':row})
    with (RUN/'supervisor.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        while True:
            try:guard()
            except RuntimeError as e:
                state('paused-for-p16',reason=str(e));time.sleep(30);continue
            available=candidates();pending=[]
            for name,weights in available:
                sha=fingerprint(weights)
                checked=[series.get('dagger-2',{})] if name=='dagger-2' else series.values()
                if any(v.get('fingerprint')==sha and len(v.get('games',[]))>=8 for v in checked):continue
                pending.append((name,weights,sha))
            if pending:
                chosen=next((r for r in pending if r[0]=='dagger-2'),None) or max(pending,key=lambda r:(r[1]/'manifest.json').stat().st_mtime)
                name,weights,sha=chosen
                item=series.setdefault(name,dict(source=str(weights),fingerprint=sha,games=[]))
                frozen=RUN/'models'/name/'weights'
                if not frozen.exists():shutil.copytree(weights,frozen)
                load_bundle(frozen) # Verify every tensor before any Dota launch.
                if fingerprint(frozen)!=sha:raise RuntimeError('Checkpoint changed while copying')
                if name=='dagger-2':
                    reference=P16/'parity-dagger-2';parity_path=frozen.parent/'cpu-parity.json'
                    if fingerprint(reference/'roundtrip-weights')!=sha:raise RuntimeError('DAgger2 Burn parity export differs from selected weights')
                    if not parity_path.exists():
                        subprocess.run([sys.executable,str(ROOT/'gpu/scripts/p17_cpu_parity.py'),'--reference',str(reference),'--output',str(parity_path)],check=True)
                    item['cpu_parity']=read(parity_path)
                checkpoint=RUN/'models'/(name+'.pt')
                if not checkpoint.exists():subprocess.run([sys.executable,str(ROOT/'scripts/export_chio_cpu.py'),'--bundle',str(frozen),'--output',str(checkpoint)],check=True)
                while len(item['games'])<8:
                    try:guard()
                    except RuntimeError as e:
                        state('paused-for-p16',reason=str(e));time.sleep(30);continue
                    i=len(item['games']);state('matches',model=name,completed=i,requested=8,weights=str(checkpoint))
                    with (RUN/'runtime.lock').open('a+b') as runtime_lock:
                        runtime_lock.seek(0);msvcrt.locking(runtime_lock.fileno(),msvcrt.LK_NBLCK,1)
                        existing=list((RUN/'matches'/name).glob(f'match-{i:03d}-attempt-*')) if (RUN/'matches'/name).exists() else []
                        run_path=RUN/'matches'/name/f'match-{i:03d}-attempt-{len(existing)+1}'
                        try:r=match(run_path,checkpoint,2+i%2,600,RUN/'bridge-001/runtime')
                        except Exception as e:
                            item.setdefault('errors',[]).append(dict(run=str(run_path),error=str(e)));save(RUN/'series.json',series)
                            if len(existing)>=2:raise
                            state('retry',error=str(e));time.sleep(30);continue
                    summary=summarize(run_path);r.pop('acks',None);r.update(summary=summary,run=str(run_path))
                    item['games'].append(r);item['simulator']=simulator_reports(weights)
                    save(RUN/'series.json',series);refresh_report(series)
                item['simulator']=simulator_reports(weights);save(RUN/'series.json',series);refresh_report(series)
                save(RUN/'models/selected.json',dict(model=str(checkpoint),source=str(weights),fingerprint=sha,pipeline_only=False,matches=8))
                publish('evaluate '+name+' in8 dedicated Dota matches')
                continue
            p16=read(P16/'state.json',{});p16stage=p16.get('stage')
            # Final summaries may arrive after Dota series, so attach them when ready.
            for item in series.values():item['simulator']=simulator_reports(Path(item['source']))
            if p16stage=='complete' and len(series.get('dagger-2',{}).get('games',[]))>=8:
                save(RUN/'series.json',series);refresh_report(series,done=True);publish('finish P16 checkpoint comparisons')
                state('complete',models=len(series));return
            state('waiting-for-p16-checkpoint',p16_stage=p16stage,p16_pid=p16.get('pid'),p16_alive=alive(p16.get('pid')),
                  completed_models=sum(len(s.get('games',[]))>=8 for s in series.values()))
            time.sleep(30)


if __name__=='__main__':
    try:run()
    except BaseException as error:
        save(RUN/'supervisor-status.json',dict(stage='failed',pid=os.getpid(),error=repr(error),background_work_remaining=False));raise
