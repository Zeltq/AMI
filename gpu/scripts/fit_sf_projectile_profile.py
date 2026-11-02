"""Measured SF attachment curves for GPU launch geometry, diagnostic only.

Uses actual attachment transforms; never rewinds first-visible projectiles.
This fit is a candidate until other speeds/turns and prior regressions pass.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from lane_conformance import admission, atomic
from lane_projectile_audit import components, rows


def curve(points):
    grouped=defaultdict(list)
    for phase,forward,lateral,height in points:
        if .25<=phase<=.70:grouped[round(phase,3)].append([phase,forward,lateral,height])
    result=[];spreads=[]
    for key,values in sorted(grouped.items()):
        middle=[statistics.median(v[i]for v in values)for i in range(4)]
        spread=max(math.dist(v[1:],middle[1:])for v in values)
        if spread>16:raise ValueError('Attachment pose not determined by sequence/cycle within16units')
        result.append(middle);spreads.append(spread)
    if len(result)<2:raise ValueError('Insufficient animation-phase coverage')
    return result,max(spreads)


def collect_samples(source):
    if not source.get('capture_passed')or not source.get('complete'):raise ValueError('Complete admitted native capture required')
    points=[[],[]];rates=[];evidence=[]
    for case in source['cases']:
        record=next(r for r in source['records']if r['id']==case['id'])
        if not admission(case,record)['passed']:raise ValueError('Native admission failed')
        actor='radiant'if case['actor_team']==2 else'dire';old=None;old_release=0
        origin=round(record['frames'][0]['game_time']*30)
        tracking={}
        for s in source['tracking_projectile_samples']:
            tick=s['tick']-origin
            if tick<0 or tick>case['duration_ticks']:continue
            for target in s['targets']:
                for p in target['shots']:
                    if p.get('caster_team')==case['actor_team']and p.get('caster_name')=='npc_dota_hero_nevermore':
                        tracking.setdefault(tick,[]).append(p)
        for frame in record['frames']:
            u=rows(frame)[actor];animation=u.get('animation',{});sequence=animation.get('sequence');phase=animation.get('cycle')
            if sequence in('attack01_anim','attack02_anim')and isinstance(phase,(float,int)):
                i=0 if sequence=='attack01_anim'else 1;attachment=u['attachments'][f'attach_attack{i+1}']
                relative=components(attachment,u);points[i].append([phase,relative['forward'],relative['lateral'],relative['height']])
                if old and old['animation']['sequence']==sequence:
                    delta=phase-old['animation']['cycle']
                    if .001<delta<.1:rates.append(delta*30*u['interval']/1.6)
            if u.get('last_attack')and u['last_attack']!=old_release:
                shot=tracking.get(frame['tick'],[])
                if len(shot)!=1:raise ValueError('Creation frame lacks one unambiguous tracking sample')
                candidates=[(math.dist([v[k]for k in('x','y','z')],[shot[0][k]for k in('x','y','z')]),name)for name,v in u['attachments'].items()if name.startswith('attach_attack')]
                distance,name=min(candidates)
                if distance>.01:raise ValueError('First observed shot does not match a measured attachment')
                evidence.append(dict(case=case['id'],tick=frame['tick'],sequence=sequence,cycle=phase,attachment=name,position_error=distance))
                old_release=u['last_attack']
            old=u
    return points,rates,evidence


def fit(sources):
    if isinstance(sources,dict):sources=[sources]
    points=[[],[]];rates=[];evidence=[]
    for source in sources:
        p,r,e=collect_samples(source);points[0].extend(p[0]);points[1].extend(p[1]);rates.extend(r);evidence.extend(e)
    if not rates:raise ValueError('No measured animation playback rates')
    fitted=[curve(p)for p in points];rate=statistics.median(rates)
    return dict(cycles_per_base_second=rate,defer_initial_movement=False,
                curves=[[[p[0],p[1],p[2]]for p in c[0]]for c in fitted]),dict(
                measured_creation_matches=evidence,cycles_per_base_second=rate,
                attachment_curves_xyz=[c[0]for c in fitted],max_same_phase_spread=[c[1]for c in fitted],
                rate_range=[min(rates),max(rates)],
                limits=['Calibration candidate; only declared source speeds and poses covered.',
                        'GPU tracking remains2D; measured Z preserved in this report.',
                        'Existing alternating hand choice unchanged, native may repeat a hand.',
                        'First-step delay is a separate ablation, disabled in this candidate.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('native',type=Path,nargs='+');p.add_argument('--engine',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--report',required=True,type=Path)
    a=p.parse_args();raw=[p.read_bytes()for p in a.native];profile,report=fit([json.loads(r)for r in raw]);engine=json.loads(a.engine.read_text())
    engine['sf_projectile_animation']=profile;engine['behavior_validated']=False
    hashes=[hashlib.sha256(r).hexdigest()for r in raw]
    engine['sf_projectile_profile_source']=dict(native=[str(p)for p in a.native],sha256=hashes,calibration_only=True)
    report.update(source_native_sha256=hashes,engine_sha256=hashlib.sha256(a.engine.read_bytes()).hexdigest(),training_eligible=False)
    atomic(a.output,engine);atomic(a.report,report)
    print(json.dumps(dict(engine=str(a.output),creation_matches=len(report['measured_creation_matches']),curve_lengths=[len(c)for c in profile['curves']],rate=profile['cycles_per_base_second'],max_spread=report['max_same_phase_spread'])))


if __name__=='__main__':main()
