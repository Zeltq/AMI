"""CPU-only CHiO policy server: Bot API log frames -> timed Lua commands."""
import argparse
import ctypes
import gzip
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.chio_cpu import load_bundle,Session
from sf_duel.chio_observations import Observer,command
from sf_duel.live.bridge import literal
from sf_duel.live.atomic import replace_text


def below_normal():
    if os.name=='nt':
        k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
        k.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
        if not k.SetPriorityClass(k.GetCurrentProcess(),0x4000):raise ctypes.WinError(ctypes.get_last_error())
    else:os.nice(5)


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def serve(model_path,log,bots,output,team,stop,ready,audit=False):
    model,_=load_bundle(model_path);session=Session(model);observer=Observer()
    output.mkdir(parents=True,exist_ok=True)
    stats=dict(pid=os.getpid(),device='cpu',model=str(model_path),requests=0,errors=[],resets=0,acks=0,accepted=0)
    offset=0;tail='';parts={};queue=[];previous=None;last_deaths=0;last_tick=None;last_save=0;audit_case=None
    save(output/'status.json',stats);ready.write_text(str(os.getpid()))
    with gzip.open(output/'observations.jsonl.gz','wt',encoding='utf-8') as trace, (output/'decisions.jsonl').open('w') as decisions:
        while not stop.exists():
            if not log.exists():time.sleep(.02);continue
            with log.open(errors='replace') as f:f.seek(offset);data=tail+f.read();offset=f.tell()
            lines=data.split('\n');tail=lines.pop()
            for line in lines:
                if audit and 'SF_P17_AUDIT_CASE|' in line:audit_case=json.loads(line.split('SF_P17_AUDIT_CASE|',1)[1])
                if 'SF_CHIO_ACK|' in line:
                    ack=json.loads(line.split('SF_CHIO_ACK|',1)[1])
                    if ack['team']==team:stats['acks']+=1;stats['accepted']+=int(ack['accepted'])
                if 'SF_CHIO_OBS|' not in line:continue
                fields=line.split('SF_CHIO_OBS|',1)[1].split('|',4)
                side,tick,part,count=map(int,fields[:4])
                if side!=team:continue
                key=(side,tick);chunks=parts.setdefault(key,{})
                chunks[part]=fields[4]
                if len(chunks)!=count:continue
                body=''.join(chunks[i] for i in range(1,count+1));del parts[key]
                if last_tick is not None and tick<=last_tick:continue
                started=time.perf_counter()
                try:
                    s=json.loads(body)
                    death=s['deaths']!=last_deaths
                    if death:observer=Observer();previous=None
                    last_deaths=s['deaths']
                    b=observer.encode(s,previous,s.get('pending_cast'))
                    action,value,reset=session.decide(b,tick,s['match'],death)
                    if audit:
                        kind=audit_case['case'] if audit_case else 1
                        if kind==13:kind=9 if tick-audit_case['start_tick']<12 else 1
                        if not b['action_mask'][0,kind]:kind=1
                        action=dict(action=kind,direction=0,enemy_target=0,ally_target=0)
                        for name in ('enemy_target','ally_target'):
                            valid=b[name+'_mask'][0].nonzero()
                            if len(valid):action[name]=int(valid[0,0])
                    d=command(s,action,stats['requests']+1);queue.append(d);queue=queue[-16:]
                    retries=replace_text(bots/f'chio_commands_{team}.lua','return '+literal(queue)+'\n')
                    stats['write_retries']=stats.get('write_retries',0)+retries
                    previous=action;last_tick=tick;stats['requests']+=1;stats['resets']+=int(reset)
                    ms=(time.perf_counter()-started)*1000
                    stats['last_tick']=tick;stats['last_ms']=ms;stats['max_ms']=max(stats.get('max_ms',0),ms)
                    trace.write(json.dumps(s,separators=(',',':'))+'\n')
                    decisions.write(json.dumps(dict(**d,value=value,inference_ms=ms,reset=reset))+'\n');decisions.flush()
                except Exception as e:
                    stats['errors'].append(dict(tick=tick,error=repr(e)));save(output/'status.json',stats);raise
            if time.monotonic()-last_save>2:save(output/'status.json',stats);last_save=time.monotonic()
            time.sleep(.005)
    stats['stage']='complete';save(output/'status.json',stats)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True);p.add_argument('--bots',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--team',type=int,choices=(2,3),default=2)
    p.add_argument('--audit',action='store_true')
    a=p.parse_args();below_normal();a.output.mkdir(parents=True,exist_ok=False)
    serve(a.model,a.log,a.bots,a.output,a.team,a.output/'stop',a.output/'ready',a.audit)


if __name__=='__main__':main()
