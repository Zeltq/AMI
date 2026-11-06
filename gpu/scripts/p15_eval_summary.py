"""Publish every P15 match as small reviewable JSON shards and aggregate ratios."""
import argparse,collections,json,statistics
from pathlib import Path

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(value,indent=2)+'\n';assert len(text.encode())<1000000;path.write_text(text,encoding='utf-8',newline='\n')
def summarize(source,output):
    report=json.loads(source.read_text());games=report['per_match'];rows=[s for g in games for s in g['sides']];groups={}
    controllers=sorted({s['controller'] for s in rows})
    for group,controller,side in [(c,c,None) for c in controllers]+[(c+':'+side,c,side) for c in controllers for side in ('radiant','dire')]:
        selected=[(g,s) for g in games for s in g['sides'] if s['controller']==controller and (side is None or s['side']==side)];n=len(selected);out={'actors':n}
        for field in ('lh','deny','deaths','hero_attacks_released','attack_hero_commands','raze_hero_hits','cancelled_casts','gold_received','gold_spent'):
            out[field]={'numerator':sum(s.get(field,0) for _,s in selected),'denominator':n}
        farm=sum(s['lh']+s['deny'] for _,s in selected);spawned=sum(g['lane_creeps_spawned'] for g in {g['game']:g for g,_ in selected}.values())
        out['lh_deny_per_spawned_creep']={'numerator':farm,'denominator':spawned}
        arrivals=[s['mid_arrival_elapsed_seconds'] for _,s in selected if s['mid_arrival_elapsed_seconds'] is not None]
        out['mid_arrivals']={'numerator':len(arrivals),'denominator':n};out['mid_arrival_elapsed_seconds']={'numerator':sum(arrivals),'denominator':len(arrivals)}
        for field in ('stuck_fraction','stuck_before_mid','stuck_after_mid'):
            out[field]={key:sum(s[field][key] for _,s in selected) for key in ('numerator','denominator')}
        for field in ('purchases','consumable_uses','reward_components'):
            values=[s[field] if isinstance(s[field],dict) else {str(i):v for i,v in enumerate(s[field])} for _,s in selected]
            keys={k for value in values for k in value};out[field]={k:sum(value.get(k,0) for value in values) for k in sorted(keys)}
        out['spell_completions']=[sum(s['spell_completions'][i] for _,s in selected) for i in range(4)]
        out['raze_hero_hit_rates']={name:{'numerator':sum(s['raze_hero_hits_by_spell'][i] for _,s in selected),'denominator':out['spell_completions'][i]} for i,name in enumerate(('near','mid','far'))}
        out['raze_denominator_note']='Completed v10 razes: both harass and farm branches require a predicted hero hit; one enemy hero per duel.'
        outcomes=collections.Counter();reasons=collections.Counter();deaths=collections.Counter()
        for g,s in selected:
            result=g['result']*(1 if s['side']=='radiant' else -1);name='win' if result>0 else 'loss' if result<0 else 'draw';outcomes[name]+=1;reasons[name+':'+g['outcome_reason']]+=1
            for cause in s['death_causes']:deaths[cause['cause']]+=cause['count']
        out['outcomes']=dict(outcomes);out['outcome_reasons']=dict(reasons);out['death_causes']=dict(deaths);groups[group]=out
    result={'source':str(source),'complete':report['complete'],'matches':{'numerator':len(games),'denominator':report['matches']},'seed':report['seed'],
            'duration_game_seconds':{'numerator':sum(g['game_seconds'] for g in games),'denominator':len(games)},'wall_minutes':report['wall_seconds']/60,'controllers':groups,'spawn_denominator':'each distinct match counted once per group'}
    save(output,result)
    for start in range(0,len(games),10):save(output.with_suffix('')/f'matches-{start:03}.json',games[start:start+10])
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();r=summarize(a.source,a.output);print(json.dumps({k:v for k,v in r.items() if k!='controllers'}))
