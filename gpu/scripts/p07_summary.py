"""Reduce P07 Dota probes to small reviewable measured tables."""
import argparse
from collections import Counter
import json
from pathlib import Path

def records(path, marker):
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        if marker in line:
            yield json.loads(line.split(marker, 1)[1])

def summarize(path):
    ticks = list(records(path, 'SF_P07_API|'))
    casts = list(records(path, 'SF_P07_CAST|'))
    groups = {}
    for group in ('own', 'enemy_visible', 'enemy_hidden'):
        rows = [u for t in ticks for u in t['units'] if
                ('own' if u['own'] else 'enemy_visible' if u['visible'] else 'enemy_hidden') == group]
        fields = {}
        for field in ('GetAnimActivity','GetAnimCycle','GetMana','GetMaxMana','IsCastingAbility','modifier','stacks','remaining','active'):
            fields[field] = dict(ok=sum(r.get(field,{}).get('ok',False) for r in rows),
                                 values=sum('value' in r.get(field,{}) for r in rows), n=len(rows))
        for method in ('abilities', 'slots'):
            rs = [a for r in rows for a in r[method]]
            fields[method] = dict(handles=sum(a['handle_present'] for a in rs),
                                 cooldown=sum(a['cooldown'].get('ok',False) for a in rs),
                                 positive_cooldown=sum(a['cooldown'].get('value',0)>0 for a in rs),
                                 level=sum(a['level'].get('ok',False) for a in rs), n=len(rs))
            fields[method]['cooldown_values']=dict(Counter(str(a['cooldown'].get('value')) for a in rs if not a['cooldown'].get('value',0)>0))
        groups[group]=fields
    events=[]
    for c in casts:
        samples=[(t,u) for t in ticks if c['game_time']<=t['game_time']<c['game_time']+2
                 for u in t['units'] if not u['own'] and u['visible'] and u['team']==c['team']]
        events.append(dict(**c, samples=len(samples), activities=dict(Counter(str(u['GetAnimActivity'].get('value')) for _,u in samples)),
                           active_names=dict(Counter(str(u['active'].get('value')) for _,u in samples)),
                           casting_samples=sum(u['IsCastingAbility'].get('value',False) for _,u in samples)))
    own={(round(t['game_time']*30),u['team']):u for t in ticks for u in t['units'] if u['own']}
    paired=Counter();errors=Counter()
    for t in ticks:
        for u in t['units']:
            if u['own'] or not u['visible']:continue
            other=own.get((round(t['game_time']*30),u['team']))
            if not other:continue
            for i,ability in enumerate(u['abilities']):
                a=other['abilities'][i]
                paired['ability_pairs']+=1
                paired['equal_levels']+=ability['level'].get('value')==a['level'].get('value')
                if a['cooldown'].get('value',0)>0:
                    paired['own_positive_cooldown']+=1
                    paired['enemy_unavailable_sentinel_while_own_positive']+=ability['cooldown'].get('value')==-1
            for name in ('GetMana','GetMaxMana','stacks','remaining'):
                if 'value' in u.get(name,{}) and 'value' in other.get(name,{}):
                    paired[name+'_pairs']+=1
                    errors[name]=max(errors[name],abs(u[name]['value']-other[name]['value']))
    return dict(source=str(path), ticks=len(ticks), groups=groups, casts=events,paired=dict(paired),paired_max_abs_error=dict(errors))

def misses(path):
    rows=list(records(path,'SF_P07_MISS|'));groups={}
    for r in rows:
        key=r['kind']+'/'+r['mode'];g=groups.setdefault(key,dict(misses=0,attacks=0,negative_damage=0,z=[]))
        g['misses']+=r['damage']==0;g['negative_damage']+=r['damage']<0;g['attacks']+=1
        z=[r['source_z'],r['target_release_z'],r['target_impact_z']]
        if z not in g['z']:g['z'].append(z)
    return dict(source=str(path),groups=groups,complete='SF_P07_MISS_DONE' in Path(path).read_text(errors='replace'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('log');p.add_argument('output');p.add_argument('--miss',action='store_true');a=p.parse_args()
    result=(misses if a.miss else summarize)(a.log)
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
