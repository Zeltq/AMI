"""Summarize measured P03 duels; censor checkpoints after a terminal event."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'gpu/scripts'))
from ggd_dataset import read_block,layout
from sf_duel.spec import UNIT_FIELDS

POINTS=(60,120,180,300,600)


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')


def stats(values,denominator):
    values=[float(v)for v in values if v is not None]
    return dict(n=len(values),denominator=denominator,sum=sum(values),mean=statistics.mean(values)if values else None,
                std=statistics.stdev(values)if len(values)>1 else(0 if values else None),minimum=min(values)if values else None,maximum=max(values)if values else None)


def checkpoints(samples,end):
    return {str(t):next((dict(time=s['time'],lh=s['lh'],deny=s['deny'],health=s['health'],mana=s['mana'])for s in samples if t-1e-3<=s['time']<=end+1e-3),None)if end+1e-3>=t else None for t in POINTS}


def dota_game(run,index,seed):
    events={};ends={};acks=[]
    for line in (run/'dota.log').read_text(errors='replace').splitlines():
        if line.startswith('SF_P03_V9|'):
            _,team,kind,payload=line.split('|',3)
            if kind=='SF_V9_END':ends[int(team)]=json.loads(payload)
            elif kind=='SF_V9_ACK':acks.append(json.loads(payload))
        elif line.startswith(('SF_P03_HERO|','SF_P03_MID|','SF_P03_HERO_DEATH|','SF_P03_LH|','SF_P03_ATTACK_RELEASE|','SF_P03_DEATH|')):
            kind,payload=line.split('|',1);events.setdefault(kind,[]).append(json.loads(payload))
    deaths=events.get('SF_P03_DEATH',[])
    terminals=[d['time']for d in deaths if d['name']=='npc_dota_hero_nevermore'or'tower1_mid'in d['name']]
    end=min([600,*terminals,*[r['time']for r in ends.values()]])
    lane=[d for d in deaths if 0<=d['time']<=end+1e-3 and(d['name'].startswith('npc_dota_creep_')or'siege'in d['name'])]
    sides=[]
    for team in (2,3):
        samples=sorted([s for s in events['SF_P03_HERO']if s['team']==team and s['time']<=end+1e-3],key=lambda s:s['time'])
        final=samples[-1];terminal=ends.get(team,{});lh=terminal.get('lh',final['lh']);deny=terminal.get('deny',final['deny'])
        releases=[s for s in events.get('SF_P03_ATTACK_RELEASE',[])if s['team']==team and 0<=s['time']<=end+1e-3]
        positions=[s['t1_distance']for s in events.get('SF_P03_LH',[])if s['team']==team and s['time']<=end+1e-3 and s.get('t1_distance')is not None for _ in range(s['count'])]
        hero_deaths=[d['time']for d in deaths if d['name']=='npc_dota_hero_nevermore'and d['team']==team and d['time']<=end+1e-3]
        mid=next((s['time']for s in events.get('SF_P03_MID',[])if s['team']==team),None)
        sides.append(dict(side='radiant'if team==2 else'dire',mid_time=mid,mid_elapsed=mid-samples[0]['time']if mid is not None else None,first_observation_time=samples[0]['time'],
                          deaths=len(hero_deaths),first_death_time=min(hero_deaths)if hero_deaths else None,lh=lh,deny=deny,
                          attack_creep=sum(s['target'].startswith('npc_dota_creep_')or'siege'in s['target']for s in releases),
                          attack_hero=sum(s['target']=='npc_dota_hero_nevermore'for s in releases),
                          attack_unknown=sum(s['target']=='unknown'for s in releases),attack_total=len(releases),
                          points=checkpoints(samples,end),lh_distances=positions,
                          credited_events=sum(d['killer']=='npc_dota_hero_nevermore'and d['killer_team']==team for d in lane)))
    return dict(index=index,seed=seed,game_seconds=end,lane_deaths=len(lane),credited=sum(s['lh']+s['deny']for s in sides),sides=sides,
                ack_count=len(acks),late_acks=sum(a['late_ticks']>0 for a in acks),max_late_ticks=max((a['late_ticks']for a in acks),default=0),run=str(run))


def simulator_games(root):
    manifest=json.loads((root/'dataset.json').read_text());source=json.loads((root/'game-metrics.json').read_text());games=[]
    for g in source:
        sides=[]
        for s in g['sides']:
            if s['attack_hero_commands']or s['attack_structure_commands']or s['actions'][5]:
                raise RuntimeError('Non-creep attack/HOLD observed; target breakdown needs direct timer telemetry')
            sides.append(dict(side=s['side'],mid_time=s['mid_arrival_game_seconds'],mid_elapsed=s['mid_arrival_elapsed_seconds'],deaths=s['deaths'],first_death_time=s['first_death_game_seconds'],
                              lh=s['lh'],deny=s['deny'],attack_creep=s['attack_cycles_completed'],attack_hero=0,attack_unknown=0,attack_total=s['attack_cycles_completed'],
                              attack_creep_commands=s['attack_enemy_creep_commands']+s['deny_commands'],points={str(t):None for t in POINTS},lh_distances=[]))
        games.append(dict(index=g['game'],seed=manifest['config']['seed'],game_seconds=g['game_seconds'],lane_deaths=g['lane_deaths'],credited=g['credited'],sides=sides))
    by_id={g['index']:g for g in games};groups,_=layout();uf=list(UNIT_FIELDS);columns={}
    for name in ('heroes','towers'):columns[name]=groups[name][0]
    seen={};last_lh={}
    pregame=json.loads((root/'environment/engine.json').read_text())['pre_game_seconds']
    for entry in manifest['files']:
        header,obs,signals=read_block(root/entry['file']);inverse={v:i for i,v in enumerate(header['columns'])}
        def column(group,row,name):return inverse[columns[group]+row*96+uf.index(name)]
        for a,actor in enumerate(header['actors']):
            g=by_id[actor//2];side=g['sides'][actor%2];base=seen.get(actor,0)
            for step in np.flatnonzero(signals[a,:,11]>.5):
                row=obs[a,step];time=-pregame+(base+int(step))*4/30
                lh=round(float(row[column('heroes',0,'last_hits')])*150);deny=round(float(row[column('heroes',0,'denies')])*50)
                for target in POINTS:
                    if side['points'][str(target)]is None and target-1e-4<=time<=g['game_seconds']+1e-3:
                        side['points'][str(target)]=dict(time=time,lh=lh,deny=deny,health=float(row[column('heroes',0,'health')])*2000,mana=float(row[column('heroes',0,'mana')])*1000)
                increment=lh-last_lh.get(actor,0)
                if increment>0:
                    tower=0 if actor%2==0 else 2
                    dx=(float(row[column('heroes',0,'x')])-float(row[column('towers',tower,'x')]))*8192
                    dy=(float(row[column('heroes',0,'y')])-float(row[column('towers',tower,'y')]))*8192
                    side['lh_distances'].extend([math.hypot(dx,dy)]*increment)
                last_lh[actor]=lh
            seen[actor]=base+header['horizon']
    # Final 600s state is sampled by the telemetry kernel after the last physics
    # step; that actor is already ineligible for the next demonstration frame.
    for g,original in zip(games,source):
        if g['game_seconds']>=600:
            for side,s in zip(g['sides'],original['sides']):
                if side['points']['600']is None:side['points']['600']=dict(time=g['game_seconds'],lh=s['lh'],deny=s['deny'],health=s['health'],mana=s['mana'])
    return games


def summary(games):
    result=dict(matches=len(games),lane_deaths=stats([g['lane_deaths']for g in games],len(games)),
                credited=dict(numerator=sum(g['credited']for g in games),denominator=sum(g['lane_deaths']for g in games)),sides={})
    for side_index,name in enumerate(('radiant','dire')):
        rows=[g['sides'][side_index]for g in games];n=len(rows)
        side={key:stats([r.get(key)for r in rows],n)for key in ('mid_time','mid_elapsed','first_death_time','lh','deny','attack_creep','attack_hero','attack_unknown','attack_total')}
        side['deaths']=dict(numerator=sum(r['deaths']for r in rows),denominator=n,per_match=stats([r['deaths']for r in rows],n))
        side['credited']=dict(numerator=sum(r['lh']+r['deny']for r in rows),denominator=sum(g['lane_deaths']for g in games),per_match=stats([(r['lh']+r['deny'])/g['lane_deaths']for r,g in zip(rows,games)if g['lane_deaths']],n))
        side['points']={str(t):{key:stats([r['points'][str(t)][key]for r in rows if r['points'][str(t)]is not None],n)for key in ('lh','deny','health','mana')}for t in POINTS}
        distances=[d for r in rows for d in r['lh_distances']];side['lh_distance']=stats(distances,sum(r['lh']for r in rows))
        result['sides'][name]=side
    return result


def ranking(dota,simulator):
    best={}
    for name in ('radiant','dire'):
        a=dota['sides'][name];b=simulator['sides'][name]
        candidates=[(key,None,a[key],b[key])for key in ('mid_elapsed','attack_creep','lh_distance')]
        for key,times in [('lh',(120,300,600)),('deny',(120,300,600)),('health',(60,180,300,600)),('mana',(60,180,300,600))]:
            candidates.extend((key,t,a['points'][str(t)][key],b['points'][str(t)][key])for t in times)
        for family,t,x,y in candidates:
            if x['n']<10 or y['n']<10 or not y['mean']:continue
            gap=abs(x['mean']-y['mean'])/abs(y['mean'])
            if family not in best or gap>best[family]['relative_gap']:
                best[family]=dict(family=family,side=name,checkpoint=t,dota_mean=x['mean'],simulator_mean=y['mean'],dota_n=x['n'],simulator_n=y['n'],relative_gap=gap)
    return sorted(best.values(),key=lambda r:r['relative_gap'],reverse=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dota',type=Path,required=True);parser.add_argument('--simulator',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    dota=[]
    for match in json.loads((args.dota/'matches.json').read_text()):
        if match['exit_code']==0:dota.append(dota_game(Path(match['run']),match['index'],match['seed']))
    save(args.output/'dota-games.json',dota)
    simulator=[]
    for seed in (20261103,20261104):
        root=args.simulator/f'seed-{seed}'
        if not(root/'dataset.json').exists():continue
        games=simulator_games(root);simulator.extend(games);save(args.output/f'simulator-games-{seed}.json',games)
    result=dict(dota=summary(dota),simulator=summary(simulator),scope=dict(time_origin='Dota game seconds',std='sample standard deviation',checkpoints='first observation at/after checkpoint, censored at terminal; GPU 600s uses terminal telemetry',gpu_attack_target_scope='attack cycles from ATTACK_READY increments; all target commands are creep and HOLD is absent, so no hero/structure acquisition',lh_position_precision='Dota callback tick; GPU first decision observing LH, 4/30s and FP16 coordinates'))
    result['ranking']=ranking(result['dota'],result['simulator'])
    result['scope']['ranking']='Largest absolute relative mean gap / abs(simulator mean), one maximum per metric family over sides and requested checkpoints; >=10 measured cases per side; game-clock arrival replaced by elapsed arrival to remove known pregame offset'
    save(args.output/'comparison.json',result)
    print(json.dumps(dict(dota_matches=len(dota),simulator_matches=len(simulator),dota_lane_deaths=result['dota']['lane_deaths']['sum'],simulator_lane_deaths=result['simulator']['lane_deaths']['sum'])))


if __name__=='__main__':main()
