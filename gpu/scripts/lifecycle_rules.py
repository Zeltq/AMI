"""Fit level1 lifecycle constants on native001; check separate native captures."""
import json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def main():
    train=ROOT/'runs/gpu/dota-lifecycle-001/lifecycle.json'
    validation=[ROOT/'runs/gpu/dota-lifecycle-002-dire/lifecycle.json',ROOT/'runs/gpu/dota-lifecycle-004-tower-first/lifecycle.json']
    captured=json.loads(train.read_text());assert captured['complete']
    positions={str(e['team']):{k:e['after'][k] for k in ('x','y','z')} for e in captured['events'] if e['transition']=='respawn'}
    checks=[]
    for path in [train,*validation]:
        d=json.loads(path.read_text());assert d['complete'];deaths={}
        for e in d['events']:
            before,after=e['before'],e['after'];team=str(e['team'])
            if e['transition']=='death':
                deaths[team]=after['time']
                worth=before.get('net_worth',before['gold']+505)
                assert before['gold']-after['gold']==math.floor(worth/40)
                assert after['souls']==math.ceil(before['souls']*.7)
            else:
                delay=after['time']-deaths[team]
                error=max(abs(after[k]-positions[team][k]) for k in ('x','y','z'))
                assert 12<=delay<=12.1 and error<.01
                assert after['health']==after['max_health'] and after['mana']==after['max_mana']
                checks.append({'source':str(path.relative_to(ROOT)),'team':int(team),'delay_seconds':delay,'position_max_error':error,'split':'train' if path==train else 'validation'})
    result={'training_eligible':False,'train':str(train.relative_to(ROOT)),'validation':[str(p.relative_to(ROOT)) for p in validation],
        'respawns':positions,'respawn_seconds':12,'checks':checks,'passed':True,
        'death_gold':'floor(networth/40), capped by unreliable gold; native examples15 and23',
        'hero_kill_souls':4,'death_soul_retention':.7,
        'gold_scope':'Empirical level1 first hero326, first tower202, later tower133. General bounty formula and later hero rewards unverified.',
        'limitations':['Level1 only. Higher-level deaths end the duel before respawn.','Native sampled respawn includes1-2Think ticks of delay; GPU uses12seconds.','Two completed validation captures are distinct from training capture; engine RNG was uncontrolled.']}
    out=ROOT/'runs/gpu/record-lifecycle-rules.json';out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
