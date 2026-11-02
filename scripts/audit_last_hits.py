"""Offline analysis of saved games and reward signals; does not run or train bots."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import numpy as np
from sf_duel.native_data import records,lane_creep
from sf_duel.selfplay.rewards import gae
from sf_duel.spec import Act,UNIT_FIELDS


def summarize_data(root):
    meta=json.loads((root/'dataset.json').read_text());actions=Counter();attacks=Counter();lh_events=0;positive_lh=0;advantages_lh=[];rewards_lh=[];all_adv=[];known=Counter();frames=0;hp_ratios=[];distances=[];attack_points=[];idle_low=Counter()
    fields=list(UNIT_FIELDS);idx={k:fields.index(k) for k in fields};width=len(fields)
    for entry in meta['episodes']:
        f=root/entry['path']
        a={k:np.load(f/(k+'.npy'),mmap_mode='r') for k in ['labels','time','loss_valid','reward_state','rewards','terminals','old_value','heroes','creeps','creep_valid','creep_ally','execution_known','execution_accepted']}
        valid=(a['time']>=0)&a['loss_valid'];frames+=int(valid.sum());actions.update(map(int,a['labels'][valid,0]));which=np.flatnonzero(valid&(a['labels'][:,0]==Act.ATTACK));pc=a['creeps'].shape[1]
        own=a['heroes'][:,0]
        for key in ['health','attack_damage','attack_range','attack_point','attack_interval','attack_cycle','projectile_speed','is_attacking']:
            known[key]+=int((own[valid,width+idx[key]]>0).sum())
        for i in which:
            target=int(a['labels'][i,3]);kind='hero' if target==1 else 'creep' if 2<=target<2+pc else 'tower_or_other';attacks[kind]+=1
            if a['execution_known'][i]:attacks['ack_known']+=1;attacks['ack_accepted']+=bool(a['execution_accepted'][i])
            if kind!='creep':continue
            c=a['creeps'][i,target-2];attacks['ally' if a['creep_ally'][i,target-2] else 'enemy']+=1
            if c[width+idx['health']] and own[i,width+idx['attack_damage']]:
                damage=float(own[i,idx['attack_damage']])*UNIT_FIELDS['attack_damage'];hp=float(c[idx['health']])*UNIT_FIELDS['health']
                if damage>0:hp_ratios.append(hp/damage)
            distance=math.hypot(float(c[idx['x']]-own[i,idx['x']]),float(c[idx['y']]-own[i,idx['y']]))*8192;distances.append(distance)
            attack_points.append(float(own[i,idx['attack_point']])*UNIT_FIELDS['attack_point'])
            attacks['outside_observed_range']+=distance>float(own[i,idx['attack_range']])*UNIT_FIELDS['attack_range']
        lh=np.diff(a['reward_state'][:,0],prepend=a['reward_state'][0,0]);event=np.flatnonzero((lh>0)&valid);event=event[event>0]-1
        lh_events+=len(event);rewards_lh.extend(map(float,a['rewards'][event]));positive_lh+=int((a['rewards'][event]>0).sum())
        # Exact for cycles with no critic-head replacement/warmup; report scope.
        adv,_=gae(a['rewards'],a['old_value'],a['terminals'],gamma=1,lam=.95)
        advantages_lh.extend(map(float,adv[event]));all_adv.extend(map(float,adv[valid]))
    def stats(v):return {'count':len(v),'mean':float(np.mean(v)) if v else None,'median':float(np.median(v)) if v else None,'p90':float(np.quantile(v,.9)) if v else None}
    return {'frames':frames,'action_counts':{Act(k).name:v for k,v in actions.items()},'attacks':dict(attacks),'target_hp_over_raw_damage':{**stats(hp_ratios),'fraction_above_2':float(np.mean(np.array(hp_ratios)>2)),'fraction_below_1':float(np.mean(np.array(hp_ratios)<=1))},'attack_distance':stats(distances),'attack_point':stats(attack_points),'own_feature_known_fraction':{k:v/frames for k,v in known.items()},'LH_events':lh_events,'LH_transition_positive_reward':positive_lh,'LH_transition_rewards':stats(rewards_lh),'LH_transition_advantages_using_recorded_values':{**stats(advantages_lh),'fraction_positive':float(np.mean(np.array(advantages_lh)>0))},'all_advantages_using_recorded_values':{**stats(all_adv),'std':float(np.std(all_adv))},'recorded_value_warning':'GAE matches actual PPO only when critic was not replaced or warmed before this cycle; otherwise diagnostic of collection-time critic only.'}


def inspect_game(match):
    decisions={2:[],3:[]};acks={};shots={2:[],3:[]};seen={2:{},3:{}}
    for r in records(Path(match['run'])/'native-decisions.pb.gz'):
        if r.HasField('execution'):
            e=r.execution;acks[e.team,e.sequence]={'tick':e.tick,'accepted':e.accepted,'reason':e.reason}
        if not r.HasField('decision'):continue
        d=r.decision;s=r.observation;t=d.team;hero=next((u for u in s.units if u.name=='npc_dota_hero_nevermore' and u.team_id==t),None)
        if hero is None:continue
        target=next((u for u in s.units if u.handle==d.target),None);is_creep=target is not None and lane_creep(target)
        decisions[t].append({'seq':d.sequence,'tick':d.observed_tick,'time':d.observed_dota_time,'action':d.action,'target':d.target,'creep':is_creep,'target_hp':target.health if target is not None else None,'damage':hero.attack_damage,'distance':math.hypot(target.location.x-hero.location.x,target.location.y-hero.location.y) if target is not None else None,'range':hero.attack_range,'attack_point':hero.attack_anim_point,'last_hits':hero.last_hits,'denies':hero.denies,'attack_target':hero.attack_target_handle,'action_type':hero.action_type})
        for u in s.units:
            if not lane_creep(u):continue
            for shot in u.incoming_tracking_projectiles:
                if not shot.is_attack or shot.caster_handle!=hero.handle:continue
                key=(shot.handle,u.handle)
                if s.dota_time-seen[t].get(key,-1e9)>2:shots[t].append({'time':s.dota_time,'target':u.handle,'hp':u.health})
                seen[t][key]=s.dota_time
    rows=[];traces=[]
    for t in (2,3):
        commands=[]
        for d in decisions[t]:
            ack=acks.get((t,d['seq']))
            if ack and (ack['accepted'] or ack['reason']=='purchase_failed') and d['action']!=Act.CONTINUE:
                commands.append({**d,'execution_time':d['time']+(ack['tick']-d['tick'])/30})
        creep_attacks=[(i,d) for i,d in enumerate(commands) if d['action']==Act.ATTACK and d['creep'] and d['time']>=0]
        quick=Counter();potential_cancel=0;next_delays=[];within_point=Counter()
        for i,d in creep_attacks:
            if i+1>=len(commands):continue
            nxt=commands[i+1];dt=nxt['execution_time']-d['execution_time']
            next_delays.append(dt)
            if dt<d['attack_point']-.001:within_point[Act(nxt['action']).name]+=1
            if dt<=.3:
                quick[Act(nxt['action']).name]+=1
                if nxt['action'] in (Act.MOVE,Act.MOVE_DIRECT,Act.STOP,Act.HOLD) and not any(x['target']==d['target'] and d['execution_time']-.134<=x['time']<=nxt['execution_time']+.134 for x in shots[t]):
                    potential_cancel+=1
                    if len(traces)<4:traces.append({'team':t,'attack':d,'next':nxt,'seconds_to_next':dt})
        rows.append({'team':t,'creep_attack_commands':len(creep_attacks),'own_creep_projectiles_seen':sum(x['time']>=0 for x in shots[t]),'next_noncontinue_within_300ms':dict(quick),'next_noncontinue_before_observed_attack_point':dict(within_point),'median_seconds_to_next_command':float(np.median(next_delays)) if next_delays else None,'move_stop_within300ms_no_observed_shot':potential_cancel,'attack_rejections':dict(Counter(acks[t,d['seq']]['reason'] for d in decisions[t] if d['action']==Act.ATTACK and (t,d['seq']) in acks and not acks[t,d['seq']]['accepted']))})
    return {'run':match['run'],'duration':match['time'],'actors':rows,'potential_cancel_examples':traces}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();r=Path(a.run);out=r/'last-hit-audit';out.mkdir(exist_ok=True)
    datasets=[('MIA0',r/'pilot-dataset-all-onpolicy'),('after_cycle1',r/'MIA1/cycle-2/dataset'),('after_cycle2',r/'MIA2/cycle-1/dataset')]
    result={}
    for label,root in datasets:
        result[label]=summarize_data(root);(out/'dataset-summary.json').write_text(json.dumps(result,indent=2));print('dataset '+label,flush=True)
    groups=[('MIA0',[Path(p['path']) for p in json.loads((r/'pilot-data-reference.json').read_text())['parts']]),('after_cycle1',sorted((r/'MIA1/cycle-2').glob('collection*'))),('after_cycle2',sorted((r/'MIA2/cycle-1').glob('collection*'))),('after_cycle3',sorted((r/'MIA2/cycle-2').glob('collection*')))]
    raw={}
    for label,folders in groups:
        games=[]
        for f in folders:
            if not f.is_dir():continue
            p=f/'collection.json';p=p if p.exists() else f/'progress.json'
            if not p.exists():continue
            m=json.loads(p.read_text());games.extend(g for g in m.get('matches',m.get('outcomes',[])) if set(g['routing'].values())=={'current'})
        indices=np.linspace(0,len(games)-1,min(6,len(games)),dtype=int)
        raw[label]=[]
        for i in indices:
            raw[label].append(inspect_game(games[i]));(out/'raw-sample.json').write_text(json.dumps(raw,indent=2));print(f'raw {label} {i}',flush=True)
    (out/'complete.json').write_text(json.dumps({'dataset_groups':list(result),'sample_games':{k:len(v) for k,v in raw.items()},'production_modified':False},indent=2))


if __name__=='__main__':main()
