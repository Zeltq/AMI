"""Collect native-game Lua teacher demonstrations. No model/trainer is imported."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prepare_teacher import prepare
from sf_duel.live.game import Game
from sf_duel.live.game import DEFAULT_GAME
from sf_duel.live.bridge import LogPackets
from sf_duel.demonstrations import TeacherWriter
from sf_duel.rules import Referee


def collect(run,bots,args,index,content_root=None,game_port=27155,stop_event=None):
    if run.exists():raise RuntimeError(f'Use a new directory: {run}')
    run.mkdir(parents=True)
    shutil.copy2(bots/'ADAPTER.json',run/'source-manifest.json')
    manifest=json.loads((bots/'ADAPTER.json').read_text())
    for name,digest in manifest['hashes'].items():
        if hashlib.sha256((bots/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Teacher source changed: {name}')
    go_mode=getattr(args,'recorder','python')=='go'
    writer=None if go_mode else TeacherWriter(run/'frames.jsonl.gz',args.hz)
    recorder=None;recorder_log=None;go_result={}
    ref=Referee();latest={};clocks={};outcome=None;errors=[];stats=None
    recording=(f'replays/{args.replay_prefix}_{index:03d}'
               if args.replay_prefix and (not args.record_first or index==1) else None)
    started=time.monotonic()
    try:
        with Game(run,uuid.uuid4().hex,args.timescale,args.hz,bot_root=bots,
                  recording=recording,collect_teacher=True,content_root=content_root,game_port=game_port,
                  game=Path(content_root).parents[1] if content_root else DEFAULT_GAME,
                  engine_threads=getattr(args,'engine_threads',None)) as game:
            game.command('dota_bot_set_difficulty 3')
            if go_mode:
                recorder_log=(run/'go.log').open('wb')
                recorder=subprocess.Popen([getattr(args,'recorder_exe',str(ROOT/'tools/nativecollect/nativecollect.exe')),
                    '--run',str(run),'--log',str(game.log),'--schema',str(bots/'telemetry_schema.json'),
                    '--hz',str(args.hz),'--until',str(args.until),'--timeout',str(args.wall_timeout)],
                    stdout=recorder_log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            game.command('echo SF_CAPTURE_READY')
            packets=LogPackets(game.log,start_marker='SF_CAPTURE_READY')
            next_stop_check=0
            while time.monotonic()-started<args.wall_timeout:
                if stop_event is not None and time.monotonic()>=next_stop_check:
                    next_stop_check=time.monotonic()+1
                    if stop_event.is_set():raise RuntimeError('Pool cancelled after another worker failed')
                if go_mode:
                    if recorder.poll() is not None:
                        go_result=json.loads((run/'go-result.json').read_text())
                        if recorder.returncode or go_result['errors']:raise RuntimeError(go_result['errors'])
                        outcome=go_result['outcome'];break
                    if game.process.poll() is not None:raise RuntimeError('Dota exited before Go recorded the terminal result')
                    time.sleep(.02);continue
                snapshots=packets.read()
                if packets.errors:raise RuntimeError(packets.errors)
                for s in snapshots:
                    try:writer.append(s)
                    except Exception:
                        (run/'rejected-frame.json').write_text(json.dumps(s,indent=2))
                        raise
                    latest[s['team']]=s;clocks[s['team']]=s['time']
                for event in packets.events:ref.add(event)
                packets.events.clear()
                if ref.ready(clocks):outcome=ref.result();break
                if args.until<600 and len(latest)==2 and min(s['time'] for s in latest.values())>=args.until:
                    outcome={'reason':'diagnostic_cut','time':args.until};break
                if game.process.poll() is not None:raise RuntimeError('Dota exited before terminal result')
                time.sleep(.01)
            else:raise TimeoutError('Teacher collection timed out')
            if not go_mode and packets.parts:raise RuntimeError('Incomplete telemetry at match end')
    except BaseException as exc:
        errors.append(str(exc));raise
    finally:
        if recorder is not None and recorder.poll() is None:
            recorder.terminate();recorder.wait(timeout=5)
        if recorder_log:recorder_log.close()
        stats=go_result.get('data',{}) if go_mode else writer.close(outcome['time'] if outcome else None)
        result={'outcome':outcome,'errors':errors,'data':stats,'decision_hz':args.hz,
                'recording':recording,'training_started':False,'spec_version':1,'format_version':2 if go_mode else 1,
                'eligible':not errors and outcome is not None and outcome['reason']!='diagnostic_cut',
                'eligible_label_rule':'observation.time < outcome.time - 0.001; both perspectives belong to one match',
                'wall_seconds':time.monotonic()-started,
                'sides':go_result.get('sides',{}) if go_mode else {t:{k:s['self'][k] for k in ('last_hits','denies','level','health')} for t,s in latest.items()}}
        (run/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'run':str(run),**result},ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--matches',type=int,default=1);parser.add_argument('--hz',type=int,default=15)
    parser.add_argument('--timescale',type=float,default=50);parser.add_argument('--until',type=float,default=600)
    parser.add_argument('--wall-timeout',type=float,default=540);parser.add_argument('--replay-prefix')
    parser.add_argument('--record-first',action='store_true')
    parser.add_argument('--recorder',choices=('python','go'),default='python')
    args=parser.parse_args()
    if args.hz<=0 or 30%args.hz or args.matches<1:parser.error('hz must divide 30, matches must be positive')
    base=Path(args.run).resolve()
    if base.exists():parser.error('Use a new run directory')
    base.mkdir(parents=True);bots=prepare()
    with zipfile.ZipFile(base/'teacher-source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        manifest=json.loads((bots/'ADAPTER.json').read_text())
        for name in sorted(manifest['hashes']):archive.write(bots/name,'teacher_bots/'+name)
        for name in ('ADAPTER.json','LICENSE'):archive.write(bots/name,'teacher_bots/'+name)
        for folder in ('src/sf_duel','scripts'):
            for path in sorted((ROOT/folder).rglob('*.py')):archive.write(path,path.relative_to(ROOT).as_posix())
    results=[]
    for index in range(1,args.matches+1):results.append(collect(base/f'match-{index:03d}',bots,args,index))
    (base/'dataset.json').write_text(json.dumps({'matches':results,'split_unit':'whole match, both sides',
        'eligible_matches':sum(r['eligible'] for r in results),'training_started':False,
        'source_sha256':hashlib.sha256((base/'teacher-source.zip').read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':main()
