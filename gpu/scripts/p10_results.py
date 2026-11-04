"""Small reproducible P10 transfer tables; preserve counts and missing values."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from play_vs_model import summarize
from sf_duel.native_data import records
from evaluate_dota_teacher import pending_at_terminal


def aggregate(games):
    result={}
    for team in (2,3):
        rows=[g for g in games if g['team']==team];n=len(rows)
        counters=('lh','deny','deaths','kills','raze_casts','raze_hit_events','raze_hero_hits','hero_attack_commands','rejected','decisions')
        data=dict(matches=n,wins=sum(g['outcome']=='win' for g in rows),losses=sum(g['outcome']=='loss' for g in rows),draws=sum(g['outcome']=='draw' for g in rows),
                  reasons=dict(Counter(g['reason'] for g in rows)),rejected_reasons=dict(sum((Counter(g.get('rejected_reasons',{})) for g in rows),Counter())))
        for key in counters:
            valid=[g[key] for g in rows if g.get(key) is not None]
            data[key]=dict(numerator=sum(valid) if valid else None,denominator=n,measured=len(valid))
        for key in ('time','first_death','mid_game_time'):
            valid=[g[key] for g in rows if g.get(key) is not None]
            data[key]=dict(sum=sum(valid),observed=len(valid),matches=n)
        result[str(team)]=data
    return result


def simulator(path):
    source=json.loads(path.read_text());games=[]
    for g in source['per_match']:
        s=next(s for s in g['sides'] if s['controller']=='GGD');team=2 if s['side']=='radiant' else 3
        result=g['result']*(1 if team==2 else -1)
        games.append(dict(index=g['game'],team=team,time=g['game_seconds'],outcome='win' if result>0 else 'loss' if result<0 else 'draw',reason=g['outcome_reason'],
            lh=s['lh'],deny=s['deny'],deaths=s['deaths'],kills=s['hero_kills'],first_death=s['first_death_game_seconds'],mid_game_time=s['mid_arrival_game_seconds'],
            raze_casts=sum(s['spell_completions'][:3]),raze_hit_events=s['raze_creep_hits']+s['raze_hero_hits'],raze_hero_hits=s['raze_hero_hits'],
            hero_attack_commands=s['attack_hero_commands'],rejected=None,decisions=sum(s['actions']),both_sides=g['sides']))
    return dict(games=games,summary=aggregate(games),complete=source['complete'],source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                limitation='Simulator evaluation does not expose rejected command counts; null is not zero.')


def dota(cohort):
    games=[]
    for m in json.loads((cohort/'matches.json').read_text()):
        team=m['model_team'];run=Path(m['run']);end=m['result']['time'];telemetry=summarize(run/'dota.log',team,end)
        s=telemetry['sides'][str(team)];out=m['result']['outcome'];winner=out.get('winner') if out else None
        counts=Counter();rejects=Counter();decisions={};hero_attacks=0
        for r in records(run/'native-decisions.pb.gz'):
            if r.HasField('decision') and r.decision.team==team:
                d=r.decision
                if pending_at_terminal(d,end):continue
                decisions[d.sequence]=d;counts[d.action]+=1
                if d.action==3 and any(u.handle==d.target and u.name=='npc_dota_hero_nevermore' and u.team_id!=team for u in r.observation.units):hero_attacks+=1
            elif r.HasField('execution') and r.execution.team==team and r.execution.sequence in decisions and not r.execution.accepted:
                rejects[r.execution.reason]+=1
        own=m['result']['sides'][str(team)]
        games.append(dict(index=m['index'],team=team,time=end,outcome='draw' if winner is None else 'win' if winner==team else 'loss',reason=m['result']['reason'],
            lh=own['last_hits'],deny=own['denies'],deaths=s['deaths'],kills=s['kills'],first_death=telemetry['first_blood']['time'] if s['deaths'] and telemetry['first_blood'] else None,
            mid_game_time=s['mid_game_time'],raze_casts=s['raze_casts'],raze_hit_events=s['raze_hit_events'],raze_hero_hits=s['raze_hero_hits'],
            hero_attack_commands=hero_attacks,rejected=sum(rejects.values()),rejected_reasons=dict(rejects),decisions=len(decisions),actions=dict(counts),
            transport=own,telemetry=telemetry,source=str(run)))
    return dict(games=games,summary=aggregate(games),source=str(cohort))


def main():
    p=argparse.ArgumentParser();p.add_argument('--simulator',type=Path);p.add_argument('--dota',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=simulator(a.simulator) if a.simulator else dota(a.dota)
    a.output.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    if a.output.stat().st_size>=1_000_000:raise ValueError('Report exceeds 1 MB')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
