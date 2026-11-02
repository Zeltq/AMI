"""Run a bounded live check/teacher match; never launches an optimizer."""
import argparse
import json
import hashlib
import zipfile
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import http.client
import uuid
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.game import Game
from sf_duel.live.bridge import LogPackets,publish
from sf_duel.live import rpc
from sf_duel.rules import Referee

def get_stats():
    return rpc.request('GET','/stats')

def request_action(snapshot):
    start=time.monotonic();response=rpc.request('POST','/act',snapshot)
    response['rpc_wall_seconds']=time.monotonic()-start
    return response

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['probe','policy','teacher'],required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--game-seconds',type=float,default=20)
    parser.add_argument('--timescale',type=float,default=1)
    parser.add_argument('--hz',type=float,default=15)
    parser.add_argument('--game-mode',type=int,default=21)
    parser.add_argument('--game-port',type=int,default=27155)
    parser.add_argument('--wall-timeout',type=float,default=240)
    args=parser.parse_args()
    if args.mode=='teacher':parser.error('Teacher training-data collection is paused pending user review; use run_teacher_review.py')
    run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=True)
    if (run/'snapshots.jsonl.gz').exists(): raise RuntimeError('Run directory already contains observations')
    try:
        with urllib.request.urlopen('http://127.0.0.1:12260/stats',timeout=1):
            raise RuntimeError('Port 12260 is already occupied; refusing to stop another service')
    except (OSError,http.client.HTTPException):pass
    linux_run='/mnt/'+run.drive[0].lower()+'/'+run.as_posix()[3:]
    service_log=(run/'service.log').open('wb')
    service_id=uuid.uuid4().hex
    process=subprocess.Popen(['wsl','-d','Ubuntu','--','bash','/mnt/c/AMI/scripts/wsl_python.sh',
        '-m','sf_duel.server','--mode',args.mode,'--run',linux_run,'--service-id',service_id],
        stdout=service_log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    from sf_duel.teacher import Teacher
    sources={}
    for folder,pattern in (('src','*.py'),('bots','*.lua'),('scripts','*.py')):
        for source in sorted((ROOT/folder).rglob(pattern)):
            if source.name=='settings.lua' or source.name.startswith('response_'):continue
            sources[source.relative_to(ROOT).as_posix()]=hashlib.sha256(source.read_bytes()).hexdigest()
    (run/'source-manifest.json').write_text(json.dumps({'sources':sources,'teacher_version':Teacher.VERSION,
        'arguments':vars(args)},indent=2),encoding='utf-8')
    with zipfile.ZipFile(run/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sources:archive.write(ROOT/name,name)
    stats={};outcome=None;pool=None
    try:
        deadline=time.monotonic()+60
        while True:
            if process.poll() is not None: raise RuntimeError(f'Service ended; inspect {run/"service.log"}')
            try:
                stats=get_stats()
                if stats.get('service_id')!=service_id:raise RuntimeError('Unexpected service owns the endpoint')
                break
            except OSError:
                if time.monotonic()>deadline: raise TimeoutError('GPU service did not start')
                time.sleep(.2)
        with Game(run,uuid.uuid4().hex,args.timescale,args.hz,game_mode=args.game_mode,game_port=args.game_port) as game:
            packets=LogPackets(game.log);futures={}
            pool=ThreadPoolExecutor(max_workers=2)
            referee=Referee();clocks={}
            start=time.monotonic()
            next_stats=0
            while time.monotonic()-start<args.wall_timeout:
                for snapshot in packets.read():
                    clocks[snapshot['team']]=snapshot['time']
                    pid=snapshot['player']
                    if pid in futures: raise RuntimeError('More than one outstanding observation for a hero')
                    futures[pid]=pool.submit(request_action,snapshot)
                if packets.errors:raise RuntimeError(f'Game rule implementation failed: {packets.errors[-1]}')
                while packets.events:
                    event=packets.events.pop(0);referee.add(event)
                    with (run/'terminal-events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
                native_result=referee.result()
                if native_result is not None and referee.ready(clocks):
                    outcome=native_result;break
                for pid,future in list(futures.items()):
                    if future.done():
                        publish(pid,future.result(),ROOT/'bots');del futures[pid]
                if time.monotonic()>=next_stats:
                    stats=get_stats();next_stats=time.monotonic()+.1
                if stats['errors']: raise RuntimeError(f'Inference error: {stats["errors"][-1]}')
                latest=stats.get('latest',{})
                if '2' in latest and '3' in latest:
                    rad,dire=latest['2'],latest['3']
                    now=min(rad['time'],dire['time'])
                    if args.game_seconds<600 and now>=args.game_seconds and stats['requests']>=4 and not futures:
                        outcome={'reason':'diagnostic_cut','winner':None}
                        break
                time.sleep(.002)
            else: raise TimeoutError('No complete game observations before wall timeout')
    finally:
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        try:stats=get_stats()
        except OSError:pass
        try:
            req=urllib.request.Request('http://127.0.0.1:12260/shutdown',data=b'',method='POST',
                headers={'X-SF-Service':service_id})
            urllib.request.urlopen(req,timeout=5).close()
            process.wait(timeout=15)
        except (OSError,subprocess.TimeoutExpired):
            process.terminate();process.wait(timeout=10)
        service_log.close()
        report={'mode':args.mode,'outcome':outcome,'stats':stats,'timescale':args.timescale,'requested_hz':args.hz}
        (run/'run-result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({'run':str(run),'outcome':outcome,'requests':stats.get('requests'),
                          'api_errors':stats.get('api_errors'),'errors':stats.get('errors')},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
