"""Audit all P12 human sessions and the P10 baseline using the same definitions."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from play_vs_model import events,summarize
from sf_duel.native_data import records,lane_creep
from sf_duel.spec import Act
from evaluate_dota_teacher import pending_at_terminal

OUT=ROOT/'reports/P12'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(session,label):
    config=json.loads((session/'session.json').read_text())
    raw=json.loads((session/'duel-result.json').read_text());team=config['model_team'];end=raw['time']
    summary=summarize(session/'dota.log',team,end)
    assert summary==json.loads((session/'summary.json').read_text()),'Saved summary mismatch'
    stats=json.loads((session/'service/stats.json').read_text())
    assert not stats['errors'] and raw.get('outcome') is not None
    terminal=summary['end']
    data=[r for r in events(session/'dota.log') if r['time']<=terminal+.001]
    decisions=[];acks={};total=0;raze_costs=set()
    for r in records(session/'native-decisions.pb.gz'):
        if r.HasField('model_supplement') and r.model_supplement.team==team:
            raze_costs.update(s.mana_cost for s in r.model_supplement.skills if s.name.startswith('nevermore_shadowraze'))
        if r.HasField('execution') and r.execution.team==team:
            e=r.execution;acks[e.sequence]=dict(tick=e.tick,accepted=e.accepted,reason=e.reason)
        if not r.HasField('decision') or r.decision.team!=team:continue
        d=r.decision;total+=1
        if pending_at_terminal(d,end):continue
        h=next(u for u in r.observation.units if u.name=='npc_dota_hero_nevermore' and u.team_id==team)
        own=next(u for u in r.observation.units if u.name.endswith('tower1_mid') and u.team_id==team)
        enemy=next(u for u in r.observation.units if u.name.endswith('tower1_mid') and u.team_id!=team)
        target=next((u for u in r.observation.units if u.handle==d.target),None)
        kind='none'
        if d.action==Act.ATTACK:
            kind='hero' if target and target.name=='npc_dota_hero_nevermore' and target.team_id!=team else 'deny' if target and lane_creep(target) and target.team_id==team else 'enemy_creep' if target and lane_creep(target) else 'other'
        distance=lambda tower:math.hypot(h.location.x-tower.location.x,h.location.y-tower.location.y)
        decisions.append(dict(sequence=d.sequence,time=d.observed_dota_time,execute_tick=d.execute_tick,
            action=Act(d.action).name,target_kind=kind,health=h.health,health_max=h.health_max,mana=h.mana,
            x=h.location.x,y=h.location.y,alive=h.is_alive,own_t1_distance=distance(own),enemy_t1_distance=distance(enemy),
            move_goal_t1_distance=math.hypot(d.x-own.location.x,d.y-own.location.y) if d.action in (Act.MOVE,Act.MOVE_DIRECT) else None))
    for row in decisions:
        ack=acks.get(row['sequence']);assert ack is not None and ack['tick']==row['execute_tick']
        row.update(accepted=ack['accepted'],reject_reason=ack['reason'])
    assert total==stats['requests']==raw['sides'][str(team)]['decisions']
    for side in ('2','3'):
        assert raw['sides'][side]['last_hits']==summary['sides'][side]['last']['lh']
        assert raw['sides'][side]['denies']==summary['sides'][side]['last']['deny']
    damages=[r for r in data if r['kind']=='hurt' and r['target']=='npc_dota_hero_nevermore']
    deaths=[]
    for row in data:
        if row['kind']!='death':continue
        hits=[h for h in damages if h['target_team']==row['team'] and h['team']==row.get('killer_team') and abs(h['time']-row['time'])<1/30+.001]
        killer=row.get('killer','');ability=hits[-1]['ability'] if hits and killer=='npc_dota_hero_nevermore' else None
        cause='tower' if 'tower' in killer else 'creep' if 'creep' in killer else ('hero_spell' if ability else 'hero_attack') if killer=='npc_dota_hero_nevermore' and hits else 'hero_unspecified' if killer=='npc_dota_hero_nevermore' else killer
        deaths.append(dict(row,cause=cause,ability=ability))
    low=[d for d in decisions if d['health_max']>0 and d['health']/d['health_max']<.25]
    moves=[d for d in low if d['action']=='MOVE']
    retreat=[d for d in moves if d['move_goal_t1_distance']<d['own_t1_distance']]
    low_segments=[];segment=[]
    for d in decisions:
        if d['health_max']>0 and d['health']/d['health_max']<.25:segment.append(d)
        elif segment:low_segments.append(segment);segment=[]
    if segment:low_segments.append(segment)
    live=[d for d in decisions if d['time']>=0 and d['alive']]
    tower_zones={}
    for radius in (700,900):
        inside=[d['enemy_t1_distance']<=radius for d in live]
        entries=[d['time'] for i,d in enumerate(live) if inside[i] and (i==0 or not inside[i-1])]
        tower_zones[str(radius)]=dict(samples=sum(inside),checked=len(live),entries=len(entries),entry_seconds=entries)
    casts=[]
    for r in data:
        if r['kind']!='cast' or r.get('team')!=team or not r.get('ability','').startswith('nevermore_shadowraze'):continue
        hits=[x for x in data if x['kind']=='hurt' and x['team']==team and x.get('ability')==r['ability'] and abs(x['time']-r['time'])<.001]
        casts.append(dict(time=r['time'],ability=r['ability'],hit_events=len(hits),hero_hits=sum(x['target']=='npc_dota_hero_nevermore' for x in hits)))
    rejects=Counter(d['reject_reason'] for d in decisions if not d['accepted'])
    later_mana=[d['mana'] for d in decisions if casts and d['time']>casts[-1]['time']]
    result=dict(label=label,session=str(session),config=config,summary=summary,native_outcome=raw['outcome'],deaths=deaths,
        transport=dict(total=total,checked=len(decisions),terminal_censored=total-len(decisions),acknowledged=len(decisions),
                       rejected=sum(rejects.values()),reasons=dict(rejects),late=0,missing=0),
        actions=dict(Counter(d['action'] for d in decisions)),attack_targets=dict(Counter(d['target_kind'] for d in decisions if d['action']=='ATTACK')),
        casts=casts,hero_damage_events=damages,observed_raze_mana_costs=sorted(raze_costs),
        mana_after_last_raze=dict(min=min(later_mana),max=max(later_mana)) if later_mana else None,
        hero_damage_by_team={str(t):dict(damage=sum(d['damage'] for d in damages if d['team']==t),
            events=sum(d['team']==t for d in damages),attack_damage=sum(d['damage'] for d in damages if d['team']==t and not d['ability'])) for t in (2,3)},
        low_health=dict(threshold_fraction=.25,decisions=len(low),actions=dict(Counter(d['action'] for d in low)),
            move_commands=len(moves),move_goals_closer_to_own_t1=len(retreat),accepted_retreat_commands=sum(d['accepted'] for d in retreat),
            segments=[dict(start=s[0]['time'],end=s[-1]['time'],own_t1_start=s[0]['own_t1_distance'],own_t1_end=s[-1]['own_t1_distance'],
                           hp_start=s[0]['health'],hp_end=s[-1]['health']) for s in low_segments]),
        enemy_t1=dict(min_distance=min((d['enemy_t1_distance'] for d in live),default=None),zones=tower_zones),
        verification=dict(saved_summary_equal=True,final_farm_matches_native=True,model_requests_accounted=True),
        service=stats,checkpoint_sha256=sha(Path(config['checkpoint'])),
        source_sha256={name:sha(session/name) for name in ('session.json','summary.json','duel-result.json','dota.log','native-decisions.pb.gz')},
        definitions=dict(retreat='HP<25%, MOVE goal closer to own T1; intention not observable; observed segment distances reported separately.',
            tower_entry='Alive post-0:00 decision samples at 4/30s: entering center-distance <=700; <=900 sensitivity also reported. Geometric proximity, not proof of tower aggro.',
            death='Killer from entity_killed; ability matched to same-team hero damage within one tick when available.',
            scope='Human-side Go teacher commands are never executed and are excluded. Events after terminal death excluded.'))
    if config.get('checkpoint_sha256'):assert config['checkpoint_sha256']==result['checkpoint_sha256']
    with (OUT/f'{label}-decisions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(decisions[0]),lineterminator='\n');writer.writeheader();writer.writerows(decisions)
    (OUT/f'{label}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    for suffix in ('.json','-decisions.csv'):assert (OUT/f'{label}{suffix}').stat().st_size<1_000_000
    return result


def main():
    root=Path('C:/AMI/runs/p12/human');rows=[]
    for session in sorted(root.iterdir()):
        if not session.is_dir():continue
        required=('session.json','summary.json','duel-result.json','native-decisions.pb.gz','dota.log','service/stats.json')
        missing=[name for name in required if not (session/name).exists()]
        if missing:rows.append(dict(session=str(session),complete=False,missing=missing));continue
        label='human-'+session.name;result=audit(session,label)
        rows.append(dict(session=str(session),complete=True,metrics=f'{label}.json',first_blood=result['summary']['first_blood']))
    baseline=audit(Path('C:/AMI/runs/p10/human/20261105-032741-652789'),'human-p10-baseline')
    previous=json.loads((ROOT/'reports/P10/human-match.json').read_text(encoding='utf-8'))
    assert baseline['summary']==previous['summary']
    assert baseline['attack_targets']==previous['attack_targets']
    assert baseline['hero_damage_events']==previous['hero_damage_events']
    assert baseline['low_health']['move_goals_closer_to_own_t1']==previous['low_health']['move_goals_closer_to_own_t1']
    index=dict(sessions=rows,completed=sum(r['complete'] for r in rows),total=len(rows),baseline='human-p10-baseline.json',
               baseline_matches_published_p10=True)
    (OUT/'human-index.json').write_text(json.dumps(index,indent=2)+'\n',newline='\n')
    print(json.dumps(index,indent=2))


if __name__=='__main__':main()
