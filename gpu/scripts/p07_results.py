"""Small reviewable P07 measurements and reproducible source provenance."""
import argparse,csv,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.enemy_observations import FIELDS
from sf_duel.observations import features
from p07_summary import records

RUN=Path('C:/AMI/runs/p07');OUT=ROOT/'reports/P07'
def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def dota():
    accepted=['sf-final','melee-final','ranged-001','tower-001','impact-003']
    groups=[];releases=[];hurts=[];sources=[]
    for name in accepted:
        root=RUN/name;result=json.loads((root/'result.json').read_text(encoding='utf-8'))
        assert result['complete'],name
        rows=list(records(root/'dota.log','SF_P07_NATURAL|'));assert len(rows)==2,name
        for r in rows:
            assert r['attacks']==240 and 0<=r['misses']<=240,r
            groups.append(dict(run=name,**r))
        releases.extend(dict(run=name,**r) for r in records(root/'dota.log','SF_P07_RELEASE|'))
        hurts.extend(dict(run=name,case=r['case'],time=r['time'],attributed=r['attributed'],**r['event']) for r in records(root/'dota.log','SF_P07_HURT|'))
        frozen=OUT/'probes'/name;frozen.mkdir(parents=True,exist_ok=True)
        for file in ('p03_events.lua','p03_controller.lua'):
            shutil.copy2(root/'bots'/file,frozen/file)
        sources.append(dict(run=name,minutes=json.loads((root/'status.json').read_text(encoding='utf-8'))['elapsed_seconds']/60,
                            script_sha256=sha(frozen/'p03_events.lua'),log_sha256=sha(root/'dota.log')))
    for name,rows in [('releases.csv',releases),('damage-events.csv',hurts)]:
        with (OUT/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));writer.writeheader();writer.writerows(rows)
    pooled=[g for g in groups if g['mode']=='uphill' and g['kind'] in ('sf','ranged','tower')]
    save('dota-miss.json',dict(groups=groups,sources=sources,pooled_ranged_uphill=dict(misses=sum(g['misses'] for g in pooled),attacks=sum(g['attacks'] for g in pooled),probability=.25),
         rule='25% ordinary ranged attacks; target height at impact; melee and abilities excluded',
         user_confirmation='User confirmed the rule and waived further Dota miss testing during P07.',
         caveats=['Attack interval shortened to 0.8 BAT; fixture sources invulnerable in accepted runs.',
                  'Damage events are attributed by attacker entity; unrelated damage is excluded.',
                  'Tower is the attacker in tower-001. Attacks against a tower were not successfully measured.']))
    api=[]
    for t in records(RUN/'spells-006/dota.log','SF_P07_API|'):
        for u in t['units']:
            if u['own']:continue
            r=dict(time=t['game_time'],observer=t['team'],team=u['team'],visible=u['visible'])
            for field in ('GetAnimActivity','GetAnimCycle','GetMana','GetMaxMana','IsCastingAbility','stacks','remaining','active'):
                r[field]=u.get(field,{}).get('value','unknown')
            for i,a in enumerate(u['abilities'],1):
                for key in ('cooldown','level'):r[f'{key}_{i}']=a[key].get('value','unknown')
            api.append(r)
    with (OUT/'api-events.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(api[0]));writer.writeheader();writer.writerows(api)
    frozen=OUT/'probes'/'spells-006';frozen.mkdir(parents=True,exist_ok=True)
    for name in ('p03_events.lua','p03_controller.lua'):shutil.copy2(RUN/'spells-006/bots'/name,frozen/name)

def actual_parity():
    gpu=json.loads((OUT/'gpu-parity.json').read_text(encoding='utf-8'));ticks=list(records(RUN/'spells-006/dota.log','SF_P07_API|'))
    columns=[f'{hero}_{known}{field}' for hero in ('self','enemy') for known in ('','known_') for field in FIELDS]
    traces=OUT/'simulator-cast-traces.csv'
    if 'actual_casts' in gpu:
        cases=gpu.pop('actual_casts')
        with traces.open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['spell','tick',*columns])
            for case in cases:
                for r in case['samples']:writer.writerow([case['spell'],r['tick'],*r['values']])
        gpu['actual_casts_file']=traces.name;save('gpu-parity.json',gpu)
    else:
        cases=[dict(spell=i,samples=[]) for i in range(1,5)]
        with traces.open(encoding='utf-8') as f:
            for row in csv.DictReader(f):cases[int(row['spell'])-1]['samples'].append(dict(tick=int(row['tick']),values=[float(row[c]) for c in columns]))
    dota=[]
    for t in ticks:
        if t['team']!=3:continue
        own=next((u for u in t['units'] if u['own']),{})
        enemy=next((u for u in t['units'] if not u['own']),{})
        dota.append((t['game_time'],features(own.get('v4') or {},FIELDS),features(enemy.get('v4') or {},FIELDS)))
    groups=[];phase_max=0.;known_mismatches=0
    for case in cases:
        spell=case['spell'];d=[r[2] for r in dota if r[2][spell]==1]
        s=[np.asarray(r['values'][32:]) for r in case['samples'] if r['values'][32+spell]==1]
        n=min(len(d),len(s));cols=[*range(6),5+spell,*range(10,14)]
        value_error=max(float(np.abs(x[cols]-y[cols]).max()) for x,y in zip(d,s))
        known_error=sum(int(np.count_nonzero(x[np.array(cols)+16]!=y[np.array(cols)+16])) for x,y in zip(d,s))
        phase_max=max(phase_max,value_error);known_mismatches+=known_error
        groups.append(dict(spell=spell,dota_cast_frames=len(d),simulator_cast_frames=len(s),compared_frames=n,
                           fields=n*len(cols)*2,max_abs_error=value_error,known_mismatches=known_error,
                           duration_difference_seconds=abs(len(d)-len(s))/30))
    # First Radiant raze hits a previously unstacked Dire; align the first effect frame.
    start=next(c['game_time'] for c in records(RUN/'spells-006/dota.log','SF_P07_CAST|') if c['team']==2 and c['spell']==1)
    d=[r[1] for r in dota if start<r[0]<start+3 and r[1][14]>0]
    s=[np.asarray(r['values'][:32]) for r in cases[0]['samples'] if r['values'][14]>0]
    n=min(len(d),len(s));assert n>0
    stack_error=max(abs(x[14]-y[14])*16 for x,y in zip(d,s));timer_error=max(abs(x[15]-y[15])*10 for x,y in zip(d,s))
    result=dict(groups=groups,max_abs_cast_field_error=phase_max,known_mismatches=known_mismatches,
                debuff=dict(compared_frames=n,max_stack_error=stack_error,max_remaining_seconds_error=timer_error),
                time_tolerance_seconds=1/30+.0001,value_tolerance=.0001,
                passed=phase_max<.0001 and known_mismatches==0 and stack_error==0 and timer_error<1/30+.0001 and all(abs(g['dota_cast_frames']-g['simulator_cast_frames'])<=1 for g in groups),
                scope='Actual casts in Dota and production physics; align observed onset and first debuff effect, allow one simulation tick at the cast boundary')
    save('actual-cast-parity.json',result);print(json.dumps(result));assert result['passed'],result

def evaluation():
    baseline=json.loads((RUN/'baseline-004/report.json').read_text())
    old=json.loads((RUN/'regression-004/report.json').read_text());new=json.loads((RUN/'evaluation-005/report.json').read_text())
    canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':')).encode()
    eq=sum(canonical(x)==canonical(y) for x,y in zip(baseline['per_match'],old['per_match']))
    assert eq==200 and len(old['per_match'])==200
    save('regression.json',dict(equal_matches=eq,matches=200,baseline_sha256=hashlib.sha256(canonical(baseline['per_match'])).hexdigest(),
         current_sha256=hashlib.sha256(canonical(old['per_match'])).hexdigest(),scope='All serialized per_match fields, including every minute snapshot; executable timing/provenance excluded'))
    table=[];games=[]
    for name,data in [('004',old),('005',new)]:
        assert data['complete'] and len(data['per_match'])==200
        for side in (0,1):
            rows=[g['sides'][side] for g in data['per_match']]
            ten=[next((m for m in r['minute_snapshots'] if abs(m['game_seconds']-600)<.1),None) for r in rows];ten=[r for r in ten if r]
            table.append(dict(environment=name,side=['Radiant','Dire'][side],matches=200,deaths=sum(r['deaths'] for r in rows),
                 final_lh=sum(r['lh'] for r in rows),final_deny=sum(r['deny'] for r in rows),at_600=dict(matches=len(ten),lh=sum(r['lh'] for r in ten),deny=sum(r['deny'] for r in ten))))
        for g in data['per_match']:
            games.append(dict(environment=name,game=g['game'],seconds=g['game_seconds'],result=g['result'],sides=[{k:s[k] for k in ('side','lh','deny','deaths')} for s in g['sides']]))
    save('comparison.json',dict(simulator=table,dota=[dict(side='Radiant',matches=20,lh=710,deny=650,deaths=0),dict(side='Dire',matches=20,lh=694,deny=629,deaths=0)],dota_source='reports/P03.md',seed=20261103))
    save('games.json',games);print(json.dumps(table))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['dota','parity','evaluation']);a=p.parse_args()
    {'dota':dota,'parity':actual_parity,'evaluation':evaluation}[a.stage]()
