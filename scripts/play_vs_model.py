"""Serve ZOg on CPU for a user-launched Steam Local Host duel; never launch Dota."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from selfplay_cycle import Config, policy_service
from install_play_bot import ARTIFACTS, backup_contents
from sf_duel.live.game import DEFAULT_GAME

LAUNCH_OPTIONS='-console -condebug -insecure -botworldstatetosocket_radiant 12250 -botworldstatetosocket_dire 12251 -botworldstatetosocket_frames 1 -botworldstatesocket_threaded'


def events(path):
    with Path(path).open(encoding='utf-8',errors='replace') as stream:
        for line in stream:
            if 'SF_PLAY_EVENT|' in line:
                try:
                    yield json.loads(line.split('SF_PLAY_EVENT|',1)[1])
                except json.JSONDecodeError:
                    continue


def summarize(log,model_team,end=600):
    rows=list(events(log))
    all_rows=rows
    deaths=[r for r in rows if r['kind']=='death' and r['time']<=end+1/30+.001]
    first=min(deaths,key=lambda r:r['time']) if deaths else None
    if first:end=min(end,first['time'])
    rows=[r for r in rows if r['time']<=end+1/30+.001]
    sides={}
    for team in (2,3):
        samples=[r for r in rows if r['kind']=='hero' and r['team']==team]
        score_samples=[r for r in all_rows if r['kind']=='hero' and r['team']==team and r['time']<=end+(.2 if first else 1/30+.001)]
        scoreboard_kills=score_samples[-1].get('kills',0) if score_samples else 0
        direct_kills=sum(r['kind']=='death' and r.get('killer_team')==team and r.get('team')!=team and r.get('killer')=='npc_dota_hero_nevermore' for r in rows)
        spells=Counter(r.get('ability','') for r in rows if r['kind']=='cast' and r.get('team')==team)
        hurts=[r for r in rows if r['kind']=='hurt' and r.get('team')==team]
        hits=[r for r in hurts if r.get('ability','').startswith('nevermore_shadowraze')]
        mid=next((r['time'] for r in samples if r['alive'] and r['x']**2+r['y']**2<2000**2 and abs(r['x']-r['y'])<1200),None)
        marks={}
        for mark in (300,600):
            sample=min(samples,key=lambda r:abs(r['time']-mark)) if samples else None
            marks[str(mark)]={k:sample[k] for k in ('time','lh','deny')} if end>=mark-1/30 and sample and abs(sample['time']-mark)<=.15 else None
        sides[str(team)]=dict(last=samples[-1] if samples else None,at_seconds=marks,
            deaths=sum(r['kind']=='death' and r.get('team')==team for r in rows),
            kills=max(scoreboard_kills,direct_kills),casts=dict(spells),
            raze_casts=sum(v for k,v in spells.items() if k.startswith('nevermore_shadowraze')),
            raze_hit_events=len(hits),raze_hero_hits=sum(r['target']=='npc_dota_hero_nevermore' for r in hits),
            hero_attack_hits=sum(r['target']=='npc_dota_hero_nevermore' and not r.get('ability') for r in hurts),mid_game_time=mid,
            telemetry_samples=len(samples))
    credited=next((team for team in (2,3) if sides[str(team)]['kills']>0),None)
    blood=dict(first,credited_team=credited) if first and credited is not None else None
    return dict(model_team=model_team,first_death=first,first_blood=blood,end=end,sides=sides,
                hit_definition='entity_hurt events, one event per damaged target; hero hits separately',
                missing_marks='null means duel ended early or no sample within 0.15 seconds')


def print_summary(result):
    names={2:'Radiant',3:'Dire'}
    death=result['first_death']
    if death:
        loser=death['team'];winner=5-loser
        print(f"Первая смерть: {names[loser]}, {death['time']:.2f} с. Победитель дуэли: {names[winner]}.",flush=True)
        blood=result['first_blood']
        print(f"First blood: {names[blood['credited_team']]}, {blood['time']:.2f} с." if blood else 'First blood герою не засчитан.',flush=True)
    else:
        print('Первая смерть не зарегистрирована; итог и полнота записи — в summary.json.',flush=True)
    for team in (2,3):
        side=result['sides'][str(team)]
        for mark in (300,600):
            row=side['at_seconds'][str(mark)]
            value=f"LH {row['lh']}, deny {row['deny']}" if row else 'нет измерения (дуэль закончилась раньше либо отсутствует снимок)'
            print(f"{names[team]}, {mark//60}:00: {value}",flush=True)
        print(f"{names[team]}: смертей {side['deaths']}, койлов {side['raze_casts']}, попаданий по герою {side['raze_hero_hits']}.",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ARTIFACTS/'ZOg-000470.pt')
    p.add_argument('--collector',type=Path,default=ARTIFACTS/'nativecollect.exe')
    p.add_argument('--artifacts',type=Path,default=ARTIFACTS)
    p.add_argument('--console-log',type=Path,default=DEFAULT_GAME/'game/dota/console.log')
    p.add_argument('--seed',type=int,default=20261105)
    p.add_argument('--summarize',type=Path,help='Existing session directory; starts no processes')
    a=p.parse_args()
    if a.summarize:
        cfg=json.loads((a.summarize/'session.json').read_text())
        result=summarize(a.summarize/'dota.log',cfg['model_team'])
        (a.summarize/'summary.json').write_text(json.dumps(result,indent=2))
        print_summary(result);return
    import msvcrt
    with Path('C:/AMI/runs/p05/evaluation.lock').open('a+b') as lock:
        lock.write(b'1');lock.flush();lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        play(a)


def play(a):
    installed=json.loads((a.artifacts/'installation.json').read_text())
    if not installed['installed']:raise RuntimeError('Install the bot scripts first')
    bots=Path(installed['path'])
    if str(bots.resolve())!=installed['resolved']:raise RuntimeError('Bots path changed')
    session=a.artifacts/'human'/datetime.now().strftime('%Y%m%d-%H%M%S-%f');session.mkdir(parents=True)
    backup,_=backup_contents(bots,a.artifacts)
    print(f'Before command writes: backup {backup}',flush=True)
    print('Restore original bots: python scripts/install_play_bot.py --restore',flush=True)
    for team in (2,3):(bots/f'native_commands_{team}.lua').write_text('return {}\n')
    config=dict(model_team=installed['model_team'],human_team=installed['human_team'],checkpoint=str(a.checkpoint),
                seed=a.seed,launch_options=LAUNCH_OPTIONS,backup=str(backup),started_utc=datetime.now(timezone.utc).isoformat())
    (session/'session.json').write_text(json.dumps(config,indent=2))
    log=session/'dota.log';log.touch()
    position=a.console_log.stat().st_size if a.console_log.exists() else 0
    collector=None
    try:
        with policy_service(a.checkpoint,session,Config(workers=1,seed=a.seed),[],record=False,
                            inference_device='cpu',selection='sampling',inference_host='windows'):
            print('CPU policy ready. Launch Dota yourself from Steam.',flush=True)
            print('Steam launch options: '+LAUNCH_OPTIONS,flush=True)
            print('Lobby: 1v1 Solo Mid, Local Host, Local Dev Script for both teams, fill empty slots.',flush=True)
            print('Pick Shadow Fiend on '+('Radiant' if config['human_team']==2 else 'Dire')+'.',flush=True)
            print('After spawn, console: sv_cheats 1; script_reload_code bots/play_model_setup',flush=True)
            print('F9 pauses; first death ends the duel. Ctrl+C stops this server. Session: '+str(session),flush=True)
            deadline=time.monotonic()+1800;ready=False;pending=''
            with log.open('ab',buffering=0) as output, (session/'collector.log').open('wb') as clog:
                while time.monotonic()<deadline:
                    if a.console_log.exists():
                        if a.console_log.stat().st_size<position:position=0
                        with a.console_log.open('rb') as source:
                            source.seek(position);data=source.read();position=source.tell()
                        output.write(data);pending+=data.decode('utf-8',errors='replace')
                        if not ready and 'SF_PLAY_DUEL_READY|'+str(config['human_team']) in pending:
                            ready=True;(session/'ready').touch()
                            command=[str(a.collector),'--duel','--run',str(session),'--bots',str(bots),'--log',str(log),
                                     '--radiant','12250','--dire','12251','--period','4','--offset','7','--until','600',
                                     '--timeout','3600','--policy-address','127.0.0.1:12360','--policy-team',str(config['model_team'])]
                            collector=subprocess.Popen(command,stdout=clog,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                            (session/'collector-command.json').write_text(json.dumps(dict(pid=collector.pid,command=command),indent=2))
                            print('Duel connected. Unpause with F9.',flush=True);deadline=time.monotonic()+3600
                        pending=pending[-4096:]
                    if collector and collector.poll() is not None:
                        if collector.returncode:print('Collector reported an error; inspect collector.log.',flush=True)
                        # The scoreboard terminal can precede the server event log
                        # by one tick. Drain it before computing the death summary.
                        time.sleep(.15)
                        if a.console_log.exists():
                            with a.console_log.open('rb') as source:
                                source.seek(position);output.write(source.read())
                        break
                    time.sleep(.01)
                else:raise TimeoutError('No setup within 30 minutes, or duel exceeded 60 wall minutes')
    except KeyboardInterrupt:
        print('Stopped by user.',flush=True)
    finally:
        if collector and collector.poll() is None:
            collector.terminate();collector.wait(timeout=10)
        result=summarize(log,config['model_team'])
        (session/'summary.json').write_text(json.dumps(result,indent=2))
        print_summary(result)
        print('Session: '+str(session),flush=True)


if __name__=='__main__':
    main()
