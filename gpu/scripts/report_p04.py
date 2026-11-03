"""Export small, reviewable P04 metrics and exact regression comparison."""
import argparse,hashlib,json
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2)
    assert len(text.encode())<1_000_000,p
    p.write_text(text,encoding='utf-8',newline='\n')

def summarize(report):
    rows={}
    keys=['lh','deny','deaths','raze_hero_hits','raze_creep_last_hits','hero_attacks_released','hero_attack_orders_executed','hero_orders_causing_creep_aggro','damage_from_hero','damage_from_creeps','damage_from_tower']
    for game in report['per_match']:
        for i,s in enumerate(game['sides']):
            key=(s['controller'],s['side'])
            r=rows.setdefault(key,dict(controller=key[0],side=key[1],matches=0,wins=0,draws=0,losses=0,incomplete=0,first_death_sum_seconds=0.,first_death_count=0,casts=0,**{k:0 for k in keys}))
            r['matches']+=1
            result=game['result']
            outcome='incomplete' if result is None else 'draws' if result==0 else 'wins' if result==[1,-1][i] else 'losses'
            r[outcome]+=1
            for k in keys:r[k]+=s[k]
            r['casts']+=sum(s['spell_completions'][:3])
            if s['first_death_game_seconds'] is not None:
                r['first_death_sum_seconds']+=s['first_death_game_seconds'];r['first_death_count']+=1
    return list(rows.values())

def main():
    p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    base=read(a.runs/'baseline/report.json');after=read(a.runs/'regression/report.json')
    ignore={'wall_seconds','aggregate_realtime_factor','total_wall_seconds_including_setup_and_trace_export','environment_fingerprint'}
    comparable=lambda d:{k:v for k,v in d.items() if k not in ignore}
    equal=comparable(base)==comparable(after)
    engines=[read(a.runs/name/'environment/engine.json') for name in ['baseline','regression']]
    for e in engines:e.pop('calibration')
    assets={p.name:hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256((a.runs/'regression/environment'/p.name).read_bytes()).hexdigest() for p in (a.runs/'baseline/environment').iterdir() if p.name!='engine.json'}
    equal=equal and engines[0]==engines[1] and all(assets.values())
    write(a.output/'regression.json',{'passed':equal,'matches_equal':sum(x==y for x,y in zip(base['per_match'],after['per_match'])),'matches':base['matches'],'stable_top_level_fields_equal':sum(base[k]==after[k] for k in comparable(base)),'stable_top_level_fields':len(comparable(base)),'excluded_fields':sorted(ignore),'fingerprint_note':'Contains output-specific calibration path and kernel source hash; event instrumentation changes the latter. All engine values except calibration path and every binary asset are compared separately.','engine_values_equal':engines[0]==engines[1],'binary_assets_equal':assets,'baseline_sides':base['sides'],'after_sides':after['sides']})
    write(a.output/'baseline.json',base)
    summaries={}
    for name in ['v9-v9','v10-v9','v10-v10','v10-v9-revision','v10-v10-revision','v10-v9-revision2','v10-v10-revision2']:
        path=a.runs/name/'report.json'
        if not path.exists():continue
        report=read(path);summary=summarize(report);summaries[name]=summary
        write(a.output/(name+'-summary.json'),{'complete':report['complete'],'matches':report['matches'],'seed':report['seed'],'minutes':report['wall_seconds']/60,'sides':summary})
        keep={'controller','side','lh','deny','deaths','first_death_game_seconds','spell_completions','raze_hero_hits','raze_creep_last_hits','hero_attacks_released','hero_attack_orders_executed','hero_orders_causing_creep_aggro','damage_from_hero','damage_from_creeps','damage_from_tower'}
        for game in report['per_match']:
            game['sides']=[{k:v for k,v in s.items() if k in keep} for s in game['sides']]
        write(a.output/(name+'-matches.json'),report)
    suffix=next(s for s in ['-revision2','-revision',''] if 'v10-v10'+s in summaries)
    mixed=[s for s in summaries['v10-v9'+suffix] if s['controller']=='v10']
    selfplay=summaries['v10-v10'+suffix];reference=summaries['v9-v9']
    total=lambda rows,key:sum(r[key] for r in rows)
    wins=total(mixed,'wins');n=total(mixed,'matches');lh=total(selfplay,'lh');ref_lh=total(reference,'lh')
    parity=read(a.runs/'parity.json') if (a.runs/'parity.json').exists() else None
    criteria={'wins':{'numerator':wins,'denominator':n,'threshold_strict':0.6,'passed':wins>n*.6},
              'farm':{'numerator':lh,'denominator':ref_lh,'v10_actor_matches':total(selfplay,'matches'),'v9_actor_matches':total(reference,'matches'),'threshold':.85,'passed':lh>=ref_lh*.85},
              'casts':{'numerator':total(mixed,'casts'),'denominator':n,'threshold':3,'passed':total(mixed,'casts')>=3*n},
              'hero_attacks':{'numerator':total(mixed,'hero_attacks_released'),'denominator':n,'threshold':5,'passed':total(mixed,'hero_attacks_released')>=5*n},
              'regression':{'passed':equal},'parity':{'tested':parity is not None,'passed':parity['passed'] if parity else False}}
    for key,metric,threshold in [('casts','casts',3),('hero_attacks','hero_attacks_released',5)]:
        criteria[key]['self_play_numerator']=total(selfplay,metric)
        criteria[key]['self_play_denominator']=total(selfplay,'matches')
        criteria[key]['passed'] &= total(selfplay,metric)>=threshold*total(selfplay,'matches')
    write(a.output/'criteria.json',{'criteria':criteria,'all_passed':all(c['passed'] for c in criteria.values()),'version_suffix':suffix})
    if parity:write(a.output/'parity.json',parity)
    if (a.runs/'commands.json').exists():write(a.output/'commands.json',read(a.runs/'commands.json'))
    print(json.dumps(criteria,indent=2))

if __name__=='__main__':main()
