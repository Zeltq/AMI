"""Audit the completed human duel from native records and server events."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from play_vs_model import events,summarize
from sf_duel.native_data import records,lane_creep
from sf_duel.spec import Act
from evaluate_dota_teacher import pending_at_terminal


def main():
    p=argparse.ArgumentParser();p.add_argument('session',type=Path);a=p.parse_args()
    out=ROOT/'reports/P10';session=a.session
    config=json.loads((session/'session.json').read_text());team=config['model_team']
    raw=json.loads((session/'duel-result.json').read_text());end=raw['time']
    summary=summarize(session/'dota.log',team,end)
    saved=json.loads((session/'summary.json').read_text())
    assert summary==saved,'Recomputed human summary differs from saved session'
    stats=json.loads((session/'service/stats.json').read_text())
    assert not stats['errors'] and raw['outcome']['winner']==5-summary['first_death']['team']
    assert abs(raw['outcome']['time']-summary['first_death']['time'])<=1/30+.001
    decisions=[];acks={};total=0;pending=0;raze_costs=set()
    for r in records(session/'native-decisions.pb.gz'):
        if r.HasField('model_supplement') and r.model_supplement.team==team:
            raze_costs.update(s.mana_cost for s in r.model_supplement.skills if s.name.startswith('nevermore_shadowraze'))
        if r.HasField('execution') and r.execution.team==team:
            x=r.execution;acks[x.sequence]=dict(tick=x.tick,accepted=x.accepted,reason=x.reason)
        if not r.HasField('decision') or r.decision.team!=team:continue
        d=r.decision;total+=1
        if pending_at_terminal(d,end):pending+=1;continue
        h=next(u for u in r.observation.units if u.name=='npc_dota_hero_nevermore' and u.team_id==team)
        enemy=next((u for u in r.observation.units if u.name=='npc_dota_hero_nevermore' and u.team_id!=team and u.is_alive),None)
        target=next((u for u in r.observation.units if u.handle==d.target),None)
        kind='none'
        if d.action==3:
            kind='hero' if target and target.name=='npc_dota_hero_nevermore' and target.team_id!=team else 'deny' if target and lane_creep(target) and target.team_id==team else 'enemy_creep' if target and lane_creep(target) else 'other'
        tower=next(u for u in r.observation.units if u.name==('npc_dota_badguys_tower1_mid' if team==3 else 'npc_dota_goodguys_tower1_mid'))
        row=dict(sequence=d.sequence,time=d.observed_dota_time,execute_tick=d.execute_tick,action=Act(d.action).name,target_kind=kind,
                 health=h.health,health_max=h.health_max,mana=h.mana,x=h.location.x,y=h.location.y,
                 own_t1_distance=math.hypot(h.location.x-tower.location.x,h.location.y-tower.location.y),
                 enemy_distance=math.hypot(h.location.x-enemy.location.x,h.location.y-enemy.location.y) if enemy else None,
                 move_goal_t1_distance=math.hypot(d.x-tower.location.x,d.y-tower.location.y) if d.action in (1,2) else None)
        decisions.append(row)
    for row in decisions:
        ack=acks.get(row['sequence']);assert ack is not None and ack['tick']==row['execute_tick'],row
        row['accepted']=ack['accepted'];row['reject_reason']=ack['reason']
    assert total==stats['requests']==raw['sides'][str(team)]['decisions']
    for side in ('2','3'):
        assert raw['sides'][side]['last_hits']==summary['sides'][side]['last']['lh']
        assert raw['sides'][side]['denies']==summary['sides'][side]['last']['deny']
    data=list(events(session/'dota.log'));terminal=summary['first_death']['time']
    data=[r for r in data if r['time']<=terminal+.001]
    farm_changes=[]
    for side in (2,3):
        previous=None
        for row in data:
            if row['kind']!='hero' or row['team']!=side:continue
            farm=(row['lh'],row['deny'])
            if previous is not None and farm!=previous:farm_changes.append({k:row[k] for k in ('time','team','lh','deny')})
            previous=farm
    casts=[]
    for r in data:
        if r['kind']!='cast' or r.get('team')!=team or not r.get('ability','').startswith('nevermore_shadowraze'):continue
        hits=[x for x in data if x['kind']=='hurt' and x['team']==team and x.get('ability')==r['ability'] and abs(x['time']-r['time'])<.001]
        casts.append(dict(time=r['time'],ability=r['ability'],targets=len(hits),hero_targets=sum(x['target']=='npc_dota_hero_nevermore' for x in hits)))
    selected=[min(decisions,key=lambda r:abs(r['time']-t)) for t in (0,18.7,28.5,30.1,39.5,60,61,65,66.4,68.4,70,75,80,82)]
    action_counts=Counter(r['action'] for r in decisions)
    target_counts=Counter(r['target_kind'] for r in decisions if r['action']=='ATTACK')
    rejects=Counter(r['reject_reason'] for r in decisions if not r['accepted'])
    low=[r for r in decisions if r['health']/r['health_max']<.25]
    result=dict(session=str(session),summary=summary,native_outcome=raw['outcome'],transport=dict(total_decisions=total,
        checked=len(decisions),terminal_censored=pending,acknowledged=len(decisions),rejected=sum(rejects.values()),reasons=dict(rejects),
        late=0,missing=0,scope='Model team only. Go teacher decisions on the human side are unexecuted diagnostics.'),
        actions=dict(action_counts),attack_targets=dict(target_counts),casts=casts,farm_changes=farm_changes,
        selected_decisions=selected,low_health=dict(threshold_fraction=.25,decisions=len(low),actions=dict(Counter(r['action'] for r in low)),
        attack_targets=dict(Counter(r['target_kind'] for r in low if r['action']=='ATTACK')),
        move_goals_closer_to_own_t1=sum(r['move_goal_t1_distance']<r['own_t1_distance'] for r in low if r['action']=='MOVE'),
        first=low[0] if low else None,last=low[-1] if low else None),
        observed_raze_mana_costs=sorted(raze_costs),
        mana_after_last_raze=dict(min=min(r['mana'] for r in decisions if r['time']>casts[-1]['time']),max=max(r['mana'] for r in decisions if r['time']>casts[-1]['time'])),
        hero_damage_events=[r for r in data if r['kind']=='hurt' and r['target']=='npc_dota_hero_nevermore'],
        service=stats,verification=dict(saved_summary_equal=True,all_model_decisions_accounted=True,all_due_commands_acknowledged_on_time=True,final_farm_matches_native=True),
        limitations=['One human match; intent of a movement command is not observable.', 'Last periodic hero sample can precede the death event; event determines death and final HP=0.', 'Human-side native teacher commands were never executed and are excluded.'],
        source_sha256={name:hashlib.sha256((session/name).read_bytes()).hexdigest() for name in ('dota.log','native-decisions.pb.gz','summary.json','duel-result.json','session.json')})
    result['checkpoint_sha256']=hashlib.sha256(Path(config['checkpoint']).read_bytes()).hexdigest()
    assert result['checkpoint_sha256']==json.loads((out/'provenance.json').read_text())['export_sha256']
    (out/'human-match.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    with (out/'human-decisions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(decisions[0]));writer.writeheader();writer.writerows(decisions)
    assert all((out/name).stat().st_size<1_000_000 for name in ('human-match.json','human-decisions.csv'))
    print(json.dumps({k:result[k] for k in ('transport','actions','attack_targets','casts','farm_changes','selected_decisions','low_health','mana_after_last_raze')},indent=2))


if __name__=='__main__':main()
