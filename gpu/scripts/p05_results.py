"""Small, reviewable transfer metrics from P01 and P05 raw artifacts."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import records
from sf_duel.spec import Act


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values,denominator):
    present=[v for v in values if v is not None]
    return dict(sum=sum(present),count=len(present),denominator=denominator,
                mean=statistics.mean(present) if present else None,
                stdev=statistics.stdev(present) if len(present)>1 else None,
                min=min(present) if present else None,max=max(present) if present else None)


def terminal_deaths(events,end,reason):
    # Native scoreboard can end the game one tick before the next Lua Think
    # reports the same death. Allow that callback delay only for a death end;
    # never include a death after the requested timeout horizon.
    tolerance=1/30+.001 if reason=='death' else .001
    return [{**v,'time':min(v['time'],end)} for v in events if v['time']<=end+tolerance]


def summarize(rows):
    result={}
    for team in (2,3):
        side=[r for r in rows if r['team']==team];n=len(side)
        commands=[sum(r['actions'][i] for r in side) for i in range(11)]
        result[str(team)]=dict(matches=n,deaths=sum(r['deaths'] for r in side),
            matches_with_death=sum(r['deaths']>0 for r in side),
            pregame_deaths=sum(r['first_death'] is not None and r['first_death']<0 for r in side),
            lh=sum(r['lh'] for r in side),deny=sum(r['deny'] for r in side),
            actions=commands,decisions=sum(commands),
            action_fractions=[dict(numerator=v,denominator=sum(commands)) for v in commands])
        for key in ('first_death','mid_game_time','mid_elapsed','t1_mean','game_seconds'):
            result[str(team)][key]=distribution([r.get(key) for r in side],n)
        result[str(team)]['t1_samples']=sum(r.get('t1_samples',0) for r in side)
    return result


def simulator(path):
    source=json.loads(path.read_text());rows=[]
    if not source['complete'] or source['matches']!=32:raise ValueError('Expected complete P01 cohort')
    for game in source['per_match']:
        s=next(s for s in game['sides'] if s['controller']=='GGD')
        rows.append(dict(index=game['game'],team=2 if s['side']=='radiant' else 3,
                         lh=s['lh'],deny=s['deny'],deaths=s['deaths'],first_death=s['first_death_game_seconds'],
                         mid_game_time=s['mid_arrival_game_seconds'],mid_elapsed=s['mid_arrival_elapsed_seconds'],
                         actions=s['actions'],game_seconds=game['game_seconds'],t1_mean=None,t1_samples=0))
    return dict(source=str(path),sha256=digest(path),games=rows,summary=summarize(rows),
                t1_limitation='P01 sampling report has no positions or T1 accumulator; no trace exists. No surrogate distance is substituted.')


def dota_match(match):
    run=Path(match['run']);team=match['model_team'];samples=[];deaths=[];mid=None
    for line in (run/'dota.log').read_text(errors='replace').splitlines():
        for marker in ('HERO','MID','DEATH'):
            prefix='SF_P05_'+marker+'|'
            if not line.startswith(prefix):continue
            value=json.loads(line[len(prefix):])
            if value['team']!=team:continue
            if marker=='HERO':samples.append(value)
            elif marker=='DEATH':deaths.append(value)
            else:mid=value['time']
    if not samples:raise ValueError('No hero telemetry')
    counts=Counter();positions=[];sequence=0;first=None;end=match['result']['time']
    for r in records(run/'native-decisions.pb.gz'):
        if not r.HasField('decision') or r.decision.team!=team:continue
        d=r.decision
        if d.sequence!=sequence+1:raise ValueError('Non-contiguous policy sequence')
        sequence=d.sequence
        if first is None:first=r.observation.dota_time
        if r.observation.dota_time>=end:continue
        counts[d.action]+=1
        own=next(u for u in r.observation.units if u.name=='npc_dota_hero_nevermore' and u.team_id==team)
        name='npc_dota_'+('goodguys' if team==2 else 'badguys')+'_tower1_mid'
        tower=next((u for u in r.observation.units if u.name==name),None)
        if own.is_alive and tower:positions.append(math.hypot(own.location.x-tower.location.x,own.location.y-tower.location.y))
    samples=[v for v in samples if v['time']<=end+.001]
    deaths=terminal_deaths(deaths,end,match['result']['reason'])
    if mid is not None and mid>end:mid=None
    s=match['result']['sides'][str(team)]
    return dict(index=match['index'],team=team,lh=s['last_hits'],deny=s['denies'],
                deaths=max([v['deaths'] for v in samples]+[v['deaths'] for v in deaths]),
                first_death=deaths[0]['time'] if deaths else None,
                mid_game_time=mid,mid_elapsed=mid-first if mid is not None else None,
                first_decision_time=first,first_telemetry_time=samples[0]['time'],
                actions=[counts[i] for i in range(11)],game_seconds=end,
                t1_mean=statistics.mean(positions) if positions else None,t1_samples=len(positions),
                acknowledged=s['acknowledged'],unacknowledged=s['unacknowledged'],
                late=s['late_executions'],rejected=s['rejected'],
                wall_minutes=match['wall_minutes'],terminal=match['result']['reason'],
                source=str(run),source_manifest=json.loads((run/'source-manifest.json').read_text()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--simulator',type=Path,required=True)
    p.add_argument('--dota',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    matches=json.loads((a.dota/'matches.json').read_text());rows=[dota_match(m) for m in matches]
    result=dict(simulator=simulator(a.simulator),dota=dict(games=rows,summary=summarize(rows),
                status=json.loads((a.dota/'status.json').read_text()),config=json.loads((a.dota/'config.json').read_text())),
                action_names=[x.name for x in Act],
                method='Own policy outcomes at normal terminal or 600 game seconds, censored after end; first death via Lua per tick. Mid threshold matches ggd_metrics.rs. Dota elapsed mid measured from first policy decision. T1: per-match mean over live policy observations (4 ticks), then unweighted mean over matches.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:result[key]['summary'] for key in ('simulator','dota')},ensure_ascii=False))


if __name__=='__main__':main()
