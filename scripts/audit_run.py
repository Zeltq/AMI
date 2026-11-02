"""Audit recorded observations/decisions and report, without training a model."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.observations import encode
from sf_duel.spec import Act,ENEMY_PRIVATE,ITEMS

def read(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)

def audit(run):
    run=Path(run)
    counts=Counter();actions=Counter();executions=Counter();previous={};last_times={};last_positions={}
    rates=defaultdict(list);ack={};decisions={};failures=[];known=Counter();total=Counter()
    firsts={};lasts={};frenzy_levels=set();spell_levels=set();max_creeps=0;max_trees=0
    observations={};game_modes=set();purchases=Counter();usable=Counter()
    safe_items=set(ITEMS)|{'item_circlet','item_slippers','item_recipe_wraith_band','item_gauntlets','item_recipe_bracer'}
    for s in read(run/'snapshots.jsonl.gz'):
        team=s['team'];counts['snapshots']+=1
        if s.get('self',{}).get('health',0)<=0:continue
        try:e=encode(s)
        except Exception as exc:failures.append(f'encode team={team} seq={s["seq"]}: {exc}');continue
        if team in last_times:
            if s['time']<=last_times[team]:failures.append('Non-increasing game time')
            rates[team].append(s['time']-last_times[team])
        last_times[team]=s['time'];firsts.setdefault(team,s);lasts[team]=s
        refs=[{k:r.get(k) for k in ('id','type','attackable','tp')} for r in e.refs]
        observations[(team,s['seq'])]={'legal':s['legal'],'refs':refs,
            'creeps':len(s['creeps']),'trees':len(s['trees'])}
        game_modes.add(s.get('probe',{}).get('game_mode'))
        if s.get('enemy'):counts['visible_enemy_snapshots']+=1
        else:
            counts['hidden_enemy_snapshots']+=1
            if e.arrays['heroes'][1].any():failures.append('Hidden enemy leaked into network input')
        ids=[c['id'] for c in s['creeps']]
        if len(set(ids))!=len(ids):failures.append('Duplicate creep identities')
        max_creeps=max(max_creeps,len(ids));max_trees=max(max_trees,len(s['trees']))
        for slot in s.get('probe',{}).get('skill_slots',[]):
            if slot.get('name')=='nevermore_frenzy':frenzy_levels.add(slot.get('level',0))
        spell_levels.add(tuple(v.get('level',0) for v in s['skills']))
        for item in s.get('probe',{}).get('inventory',[]):
            if item.get('name') not in safe_items:failures.append('Forbidden item: '+str(item))
        execution=s.get('execution')
        if isinstance(execution,dict) and 'seq' in execution:
            key=(team,execution['seq'])
            if key not in ack:
                ack[key]=execution;executions['accepted' if execution.get('accepted') else execution.get('reason','rejected')]+=1
                if execution.get('purchase'):purchases[str(execution['purchase'])]+=1
        for name,row in s['self'].items():
            if isinstance(row,(int,float,bool)):known[name]+=1
        counts['projectile_rows']+=len(s.get('projectiles',[]))
    if any(level>0 for level in frenzy_levels):failures.append('Frenzy was leveled')
    for reason,count in executions.items():
        if reason not in ('accepted','late_response','spell_unavailable','item_unavailable','target_no_longer_visible_or_alive'):
            failures.append(f'Unexpected executor failure ({count}): {reason}')
    for d in read(run/'decisions.jsonl.gz'):
        counts['decisions']+=1;a=d['action'];command=d['command'];key=(d['team'],d['seq'])
        decisions[key]=d;actions[str(a['action'])]+=1
        info=observations.get(key)
        if info is None:
            failures.append('Decision without matching observation');continue
        legal=info['legal'];primary=a['action'];buy=a['buy'];item=a['item']
        if not 0<=primary<len(legal['actions']) or not legal['actions'][primary]:failures.append('Illegal primary action')
        if not 0<=buy<len(legal['buys']) or not legal['buys'][buy]:failures.append('Illegal purchase choice')
        if primary==Act.ITEM and (not 0<=item<len(legal['items']) or not legal['items'][item]):failures.append('Illegal item choice')
        if primary in (Act.MOVE,Act.MOVE_DIRECT) or (primary==Act.ITEM and item==3):
            if not all(0<=a[k]<=128 for k in ('x','y')):failures.append('Coordinate outside policy grid')
        target_index=a['target'];refs=info['refs'];n,t=info['creeps'],info['trees']
        if d['controller']=='policy':
            layout=d.get('target_layout') or {'creeps':max(16,1<<max(0,n-1).bit_length()),
                'trees':max(32,1<<max(0,t-1).bit_length())}
            nc,nt=layout['creeps'],layout['trees']
            if 2<=target_index<2+nc:target_index=target_index if target_index<2+n else -1
            elif 2+nc<=target_index<6+nc:target_index=target_index-nc+n
            elif 6+nc<=target_index<6+nc+nt:
                j=target_index-6-nc;target_index=6+n+j if j<t else -1
            elif target_index==6+nc+nt:target_index=len(refs)-1
        target=refs[target_index] if 0<=target_index<len(refs) else {}
        if primary==Act.ATTACK and not target.get('attackable'):failures.append('Illegal attack target')
        if primary==Act.ITEM and item==3 and not target.get('tp'):failures.append('Illegal teleport destination')
        if primary==Act.ITEM and item==6 and target.get('type')!='tree':failures.append('Illegal tango target')
        if primary==Act.ATTACK or (primary==Act.ITEM and item in (3,6)):
            if command.get('target_id')!=target.get('id'):failures.append('Policy target changed during command encoding')
        confirmation=ack.get(key)
        if confirmation and confirmation.get('accepted'):usable[d['controller']]+=1
        if confirmation and a['action']==int(Act.ATTACK) and confirmation.get('accepted'):
            if confirmation.get('target_id')!=command.get('target_id'):failures.append('Executor substituted an attack target')
    sides={}
    for team,last in lasts.items():
        first=firsts[team];deltas=rates[team]
        span=last['time']-first['time']
        sides[team]={'first_time':first['time'],'last_time':last['time'],
            'effective_hz':len(deltas)/span if span>0 else 0.,
            'maximum_step_gap':max(deltas) if deltas else None,
            'distance_from_start':math.hypot(last['self']['x']-first['self']['x'],last['self']['y']-first['self']['y']),
            'last_hits':last['self'].get('last_hits'),'denies':last['self'].get('denies'),
            'level':last['self'].get('level')}
    if game_modes!={21}:failures.append('Not confirmed native Solo Mid mode: '+str(game_modes))
    report={'passed':not failures and counts['snapshots']>0,'counts':dict(counts),
        'actions':dict(actions),'executions':dict(executions),'sides':sides,
        'confirmed_game_modes':sorted(game_modes,key=str),'acknowledged_decisions':dict(usable),
        'purchase_results':dict(purchases),
        'spell_level_combinations':sorted(spell_levels),'frenzy_levels':sorted(frenzy_levels),
        'max_creeps':max_creeps,'max_trees':max_trees,'available_self_fields':sorted(known),
        'failures':sorted(set(failures))}
    (run/'audit.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args()
    print(json.dumps(audit(args.run),indent=2))
