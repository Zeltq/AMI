"""Wait for frozen P16 checkpoints, copy them, then run serial CPU Dota series."""
import argparse
import hashlib
import json
import msvcrt
import os
import re
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
from sf_duel.chio_version import EXECUTOR_VERSION

RUN=Path('C:/AMI/runs/p17');P16=Path('C:/AMI/runs/p16');P16ROOT=Path('C:/AMI-wt/P16')


def read(path,default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def fingerprint(weights):
    m=read(weights/'manifest.json')
    return hashlib.sha256(json.dumps({k:(v['shape'],v.get('sha256')) for k,v in m['tensors'].items()},sort_keys=True).encode()).hexdigest()


def dagger_name():
    # User explicitly retired the original imitation/DAgger chain for P17 matches.
    return 'sampling-dagger-2' if read(P16/'sampling-plan.json',{}).get('active') else 'nav-dagger-2'


def candidates():
    stages=read(P16/'stages.json',{});result=[]
    # DAgger 2 is mandatory even if a newer RL checkpoint already exists.
    dagger=dagger_name()
    if dagger+'-il' in stages and 'parity-'+dagger in stages:
        path=Path(stages[dagger+'-il']['folder'])/'weights'
        result.append((dagger,path))
    for folder in [*P16.glob('ppo'),*P16.glob('ppo-attempt-*'),*P16.glob('self-play'),*P16.glob('self-play-attempt-*')]:
        config=read(folder/'config.json',{})
        engine=config.get('engine')
        if not engine or not read(Path(engine),{}).get('chio_script_navigation'):continue
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
        if not e.get('complete') or not e.get('sampling',False):continue
        source=Path(e['weights'])
        if not (source/'manifest.json').exists() or fingerprint(source)!=sha:continue
        path=P16ROOT/e['summary']
        if path.exists():result.append(dict(evaluation=e,summary=read(path),
            directly_comparable=bool(e.get('scripted_base_to_mid',False)),
            action_selection='sampling',dota_base_navigation='script',simulator_base_navigation='script' if e.get('scripted_base_to_mid') else 'policy'))
    return result


def base_name(name):
    return name.removesuffix('-'+EXECUTOR_VERSION)


def pending_candidates(series):
    available=candidates();pending=[];seen=set()
    for name,item in series.items():
        if item.get('executor_version','p17-v1')!=EXECUTOR_VERSION or len(item.get('games',[]))>=8:continue
        weights=Path(item['source'])
        if not weights.resolve().is_relative_to(P16.resolve()):raise ValueError('Resume source escapes runs/p16')
        if fingerprint(weights)!=item['fingerprint']:raise RuntimeError('Resume source checkpoint changed')
        available.append((name,weights))
    for name,weights in available:
        sha=fingerprint(weights)
        # Previously completed v1 series remain historical results; never rerun them.
        if any(v.get('fingerprint')==sha and len(v.get('games',[]))>=8 for v in series.values()):continue
        name=base_name(name)+'-'+EXECUTOR_VERSION
        if name not in seen:pending.append((name,weights,sha));seen.add(name)
    return pending


def dagger_count(series):
    return max((len(v.get('games',[])) for k,v in series.items() if base_name(k)==dagger_name()),default=0)


def choose_candidate(pending,series):
    mandatory=next((r for r in pending if base_name(r[0])==dagger_name()),None)
    unfinished=next((r for r in pending if r[0] in series and len(series[r[0]].get('games',[]))<8),None)
    return mandatory or unfinished or max(pending,key=lambda r:(r[1]/'manifest.json').stat().st_mtime)


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
    lines=[begin,'| Модель P16 | Матчи | W/D/L | LH / deny / смерти | Мид | Оценки v9/v10 |','|---|---:|---|---|---:|---:|']
    for name,item in series.items():
        games=item.get('games',[]);n=len(games)
        wins=sum(g['winner']==g['model_team'] for g in games);draws=sum(g['winner']==0 for g in games)
        sides=[g['summary']['sides'][str(g['model_team'])] for g in games]
        lh=sum(s['lh'] for s in sides);deny=sum(s['deny'] for s in sides);deaths=sum(s['deaths'] for s in sides);mid=sum(s['mid_time'] is not None for s in sides)
        opponents={r.get('evaluation',{}).get('opponent') for r in item.get('simulator',[]) if r.get('directly_comparable')}
        coverage=len(opponents & {'v9','v10'})
        label=name if name.endswith('-'+EXECUTOR_VERSION) else name+' [v1]'
        if item.get('retired_reason'):label+=' (partial)'
        lines.append(f"|{label}|{n}/8|{wins}/{n};{draws}/{n};{n-wins-draws}/{n}|{lh}/{n};{deny}/{n};{deaths}/{n}|{mid}/{n}|{coverage}/2|")
    if len(lines)>18:lines=lines[:3]+lines[-15:] # Full history remains in JSON.
    lines+=[end]
    replacement='\n'.join(lines)
    if begin in text:text=text[:text.index(begin)]+replacement+text[text.index(end)+len(end):]
    else:text=text.replace('## Приоритет P16',replacement+'\n\n## Приоритет P16')
    name=dagger_name();count=dagger_count(series)
    text=re.sub(r'^P16(?: DAgger2:|: подробные).*$',f'P16: подробные результаты и события в [series.json](P17/series.json); {name} {count}/8 матчей.',text,flags=re.MULTILINE)
    if done:
        text='\n'.join(text.splitlines()[:-1])+"\nBLOCKED P17: серверная часть завершена; Local Host с человеком не проверен, полный replay наблюдений недоступен без GPU\n"
    overflow=len(text.splitlines())-150
    if overflow>0:
        rows=lines[3:-1]
        if overflow>=len(rows):raise RuntimeError('Report exceeds150 lines without removable history rows')
        replacement='\n'.join(lines[:3]+rows[overflow:]+[end])
        text=text[:text.index(begin)]+replacement+text[text.index(end)+len(end):]
    path.write_text(text,encoding='utf-8');save(ROOT/'reports/P17/series.json',series)


def run():
    below_normal();series=read(RUN/'series.json',{});state_path=RUN/'supervisor-status.json'
    for item in series.values():
        item.setdefault('executor_version','p17-v1')
        if item['executor_version']!=EXECUTOR_VERSION and len(item.get('games',[]))<8:
            item['retired_reason']='User requested executor v2; incomplete v1 series retained without replay.'
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
            pending=pending_candidates(series)
            if pending:
                chosen=choose_candidate(pending,series)
                name,weights,sha=chosen
                item=series.setdefault(name,dict(source=str(weights),fingerprint=sha,games=[],action_selection='sampling',dota_base_navigation='script',executor_version=EXECUTOR_VERSION,
                    p16_navigation_mode=read(P16/'navigation-mode.json',{})))
                save(RUN/'series.json',series);refresh_report(series)
                frozen=RUN/'models'/name/'weights'
                if not frozen.exists():shutil.copytree(weights,frozen)
                load_bundle(frozen) # Verify every tensor before any Dota launch.
                if fingerprint(frozen)!=sha:raise RuntimeError('Checkpoint changed while copying')
                if base_name(name)==dagger_name():
                    reference=P16/('parity-'+base_name(name));parity_path=frozen.parent/'cpu-parity.json'
                    if fingerprint(reference/'roundtrip-weights')!=sha:raise RuntimeError('DAgger2 Burn parity export differs from selected weights')
                    if not parity_path.exists():
                        subprocess.run([sys.executable,str(ROOT/'gpu/scripts/p17_cpu_parity.py'),'--reference',str(reference),'--output',str(parity_path)],check=True)
                    item['cpu_parity']=read(parity_path)
                checkpoint=RUN/'models'/(name+'.pt')
                if not checkpoint.exists():subprocess.run([sys.executable,str(ROOT/'scripts/export_chio_cpu.py'),'--bundle',str(frozen),'--output',str(checkpoint)],check=True)
                while len(item['games'])<8:
                    try:guard(force=True)
                    except RuntimeError as e:
                        state('paused-for-p16',reason=str(e));time.sleep(30);continue
                    i=len(item['games']);state('matches',model=name,completed=i,requested=8,weights=str(checkpoint))
                    with (RUN/'runtime.lock').open('a+b') as runtime_lock:
                        runtime_lock.seek(0);msvcrt.locking(runtime_lock.fileno(),msvcrt.LK_NBLCK,1)
                        existing=list((RUN/'matches'/name).glob(f'match-{i:03d}-attempt-*')) if (RUN/'matches'/name).exists() else []
                        run_path=RUN/'matches'/name/f'match-{i:03d}-attempt-{len(existing)+1}'
                        try:r=match(run_path,checkpoint,2+i%2,600,RUN/'bridge-001/runtime',seed=20261103+i)
                        except Exception as e:
                            item.setdefault('errors',[]).append(dict(run=str(run_path),error=str(e)));save(RUN/'series.json',series)
                            if len(existing)>=2:raise
                            state('retry',error=str(e));time.sleep(30);continue
                    summary=summarize(run_path);r.pop('acks',None);r.update(summary=summary,run=str(run_path))
                    item['games'].append(r);item['simulator']=simulator_reports(weights)
                    save(RUN/'series.json',series);refresh_report(series)
                item['simulator']=simulator_reports(weights);save(RUN/'series.json',series);refresh_report(series)
                save(RUN/'models/selected.json',dict(model=str(checkpoint),source=str(weights),fingerprint=sha,pipeline_only=False,matches=8,executor_version=EXECUTOR_VERSION))
                publish('evaluate '+name+' in8 dedicated Dota matches')
                continue
            p16=read(P16/'state.json',{});p16stage=p16.get('stage')
            # Final summaries may arrive after Dota series, so attach them when ready.
            for item in series.values():item['simulator']=simulator_reports(Path(item['source']))
            if p16stage=='complete' and dagger_count(series)>=8:
                save(RUN/'series.json',series);refresh_report(series,done=True);publish('finish P16 checkpoint comparisons')
                state('complete',models=len(series));return
            state('waiting-for-p16-checkpoint',p16_stage=p16stage,p16_pid=p16.get('pid'),p16_alive=alive(p16.get('pid')),
                  completed_models=sum(len(s.get('games',[]))>=8 for s in series.values()))
            time.sleep(30)


if __name__=='__main__':
    try:run()
    except BaseException as error:
        save(RUN/'supervisor-status.json',dict(stage='failed',pid=os.getpid(),error=repr(error),background_work_remaining=False));raise
