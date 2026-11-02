"""Play a diagnostic teacher duel, optionally record ONE human review demo."""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.game import Game
from sf_duel.rules import Referee
from prepare_teacher import prepare

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--until',type=float,default=600);parser.add_argument('--timescale',type=float,default=2)
    parser.add_argument('--record');parser.add_argument('--wall-timeout',type=float,default=540)
    args=parser.parse_args();run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=True)
    if (run/'dota.pid').exists():raise RuntimeError('Use a new run directory')
    bots=prepare();ref=Referee();latest={};clocks={};offset=0;buffer=b'';errors=[];outcome=None
    with Game(run,uuid.uuid4().hex,args.timescale,15,bot_root=bots,recording=args.record) as game:
        game.command('dota_bot_set_difficulty 3')
        game.command('help host_timescale')
        game.command('host_timescale')
        start=time.monotonic()
        while time.monotonic()-start<args.wall_timeout:
            with game.log.open('rb') as f:
                f.seek(offset);buffer+=f.read();offset=f.tell()
            lines=buffer.split(b'\n');buffer=lines.pop()
            for line in lines:
                for prefix in (b'SF_STATUS|',b'SF_END|',b'SF_RULE_ERROR|'):
                    i=line.find(prefix)
                    if i<0:continue
                    data=json.loads(line[i+len(prefix):].rstrip(b'\r'))
                    data['received_wall_seconds']=time.monotonic()-start
                    if prefix==b'SF_STATUS|':
                        latest[data['team']]=data;clocks[data['team']]=data['time']
                        with (run/'diagnostics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(data)+'\n')
                        if data['api_errors']:errors.append(data['api_errors'])
                    elif prefix==b'SF_END|':ref.add(data)
                    else:errors.append(data)
            if errors:raise RuntimeError(f'Teacher rule/API error: {errors[-1]}')
            if ref.ready(clocks):outcome=ref.result();break
            if len(latest)==2 and min(s['time'] for s in latest.values())>=args.until and args.until<600:
                outcome={'reason':'diagnostic_cut','time':args.until};break
            if game.process.poll() is not None:raise RuntimeError('Dedicated server exited; inspect dota.log')
            time.sleep(.02)
        else:raise TimeoutError('Teacher did not finish')
    result={'outcome':outcome,'latest':latest,'errors':errors,'recording':args.record,
        'teacher':'Open Hyper AI SF duel adapter','training_started':False,'demonstrations_collected':False}
    (run/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'run':str(run),'outcome':outcome,'sides':{t:{k:s['hero'].get(k) for k in
        ('last_hits','denies','level','health')} for t,s in latest.items()}},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
