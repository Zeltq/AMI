"""Compare saved native/GPU tracking shots without launching either engine.

Positions are actual first-visible samples. They are not claimed to be muzzle
attachments. Raw signed coordinates and hand-independent components are both
reported; neither a mirrored hand nor a best sampling phase can certify a pass.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from lane_conformance import admission, atomic


def rows(frame):
    return {u['id']:u for u in frame['units']}


def releases(record, actor, native):
    result = []; old = 0.
    for frame in record['frames']:
        u = rows(frame)[actor]
        value = u.get('last_attack', 0.) if native else u['attack_hand']
        if value and value != old:
            result.append(dict(tick=frame['tick'], time=value if native else None))
        old = value
    return result


def components(shot, source):
    angle = math.radians(source['facing'])
    dx, dy = shot['x']-source['x'], shot['y']-source['y']
    return dict(forward=dx*math.cos(angle)+dy*math.sin(angle),
                lateral=-dx*math.sin(angle)+dy*math.cos(angle),
                height=None if shot.get('z') is None else shot['z']-source['z'])


def match_native_target(target, frame):
    candidates = [u for u in frame['units'] if u.get('name') == target['name']
                  and u['team'] == target['team']
                  and math.hypot(u['x']-target['x'], u['y']-target['y']) <= 16.]
    if len(candidates) != 1:
        raise ValueError('Missing or ambiguous native projectile target')
    return candidates[0]['id']


def impact(record, target, start, end):
    hits = []
    for before, after in zip(record['frames'], record['frames'][1:]):
        if start <= after['tick'] < end:
            delta = rows(before)[target]['health'] - rows(after)[target]['health']
            if delta > 1.2: hits.append(dict(tick=after['tick'], hp_drop=delta))
    if len(hits) > 1: raise ValueError('Multiple impacts within one shot interval')
    return hits[0] if hits else None


def velocities(samples):
    result = []
    for first, second in zip(samples, samples[1:]):
        if second['tick'] != first['tick']+1: continue
        delta = [(second[k]-first[k])*30 for k in ('x','y')]
        if first.get('z') is not None and second.get('z') is not None:
            delta.append((second['z']-first['z'])*30)
        result.append(dict(tick=first['tick'], xy_speed=math.hypot(*delta[:2]),
                           speed_3d=math.sqrt(sum(d*d for d in delta)) if len(delta)==3 else None))
    return result


def timing_audit(native, gpu):
    """Stationary single-target timing, including STOP/restart and direct events.

    This certifies only observed events. Pending final projectiles and full-state
    differences are reported separately and cannot be inferred from this result.
    """
    if not native.get('complete') or not native.get('capture_passed'):
        raise ValueError('Complete admitted native capture required')
    results=[]
    for case in native['cases']:
        nr=next(r for r in native['records'] if r['id']==case['id'])
        gr=next(r for r in gpu['records'] if r['id']==case['id'])
        if not admission(case,nr)['passed']:raise ValueError('Native admission failed')
        actor='radiant' if case['actor_team']==2 else 'dire'
        if (case['family']!='attack' or len(case['units'])!=3 or not case['orders']
                or any(o['actor']!=actor or o['action']not in(3,4)
                       or (o['action']==3 and o.get('target')!='victim') for o in case['orders'])):
            raise ValueError('Timing audit requires stationary ATTACK/STOP on one victim')
        if gr.get('orders')!=case['orders']:raise ValueError('GPU command log differs from manifest')
        if [f['tick']for f in nr['frames']]!=[f['tick']for f in gr['frames']]:
            raise ValueError('Different captured ticks')
        events=[e for e in nr.get('attack_events',[])if e['actor']==actor]
        nstart=[e['tick']for e in events if e['event']=='start'] if events else None
        nrelease=[e['tick']for e in events if e['event']=='release'] if events else [e['tick']for e in releases(nr,actor,True)]
        grelease=[e['tick']for e in releases(gr,actor,False)]
        gstart=[];winding=False
        for frame in gr['frames']:
            active=rows(frame)[actor].get('windup',0.)>0.
            if active and not winding:gstart.append(frame['tick'])
            winding=active
        def impacts(record):
            return [b['tick']for a,b in zip(record['frames'],record['frames'][1:])
                    if rows(a)['victim']['health']-rows(b)['victim']['health']>1.2]
        nhit,ghit=impacts(nr),impacts(gr)
        tolerance=case.get('comparison',{}).get('event_ticks',1)
        def compare(expected,actual):
            delta=[b-a for a,b in zip(expected,actual)]
            return dict(native=expected,gpu=actual,delta_ticks=delta,count_equal=len(expected)==len(actual),
                        passed=len(expected)==len(actual)and all(abs(d)<=tolerance for d in delta))
        release=compare(nrelease,grelease);hit=compare(nhit,ghit)
        start=compare(nstart,gstart)if nstart is not None else None
        adequate=bool(nrelease and nhit)
        results.append(dict(id=case['id'],release=release,impact=hit,start=start,
            release_source='direct callbacks'if events else'polled last_attack',
            pending_or_unobserved_impacts=dict(native=len(nrelease)-len(nhit),gpu=len(grelease)-len(ghit)),
            event_tolerance_ticks=tolerance,adequate=adequate,
            timing_passed=adequate and release['passed']and hit['passed']and(start is None or start['passed'])))
    return dict(cases=results,timing_passed=bool(results)and all(c['timing_passed']for c in results),
                training_updates=0,training_eligible=False,
                scope='Observed start/release/HP-loss timings only; full-state conformance and pending impacts are not certified.')


def audit(native, gpu):
    if not native.get('complete') or not native.get('capture_passed'):
        raise ValueError('Complete admitted native capture required')
    nrecords = {r['id']:r for r in native['records']}
    grecords = {r['id']:r for r in gpu['records']}
    results = []
    for case in native['cases']:
        nr, gr = nrecords[case['id']], grecords[case['id']]
        if not admission(case, nr)['passed']: raise ValueError('Native admission failed')
        if gr.get('orders')!=case['orders']: raise ValueError('GPU command log differs from manifest')
        if case['family'] != 'attack' or len(case['units']) != 3:
            raise ValueError('Audit requires single attacker and one victim, no competing attacks')
        actor = 'radiant' if case['actor_team']==2 else 'dire'
        if not case['orders'] or any(o['actor']!=actor or o['action']!=3 or o['target']!='victim' for o in case['orders']):
            raise ValueError('Only stationary attacks on one victim supported')
        nf, gf = {f['tick']:f for f in nr['frames']}, {f['tick']:f for f in gr['frames']}
        if sorted(nf) != sorted(gf): raise ValueError('Different captured ticks')
        if any('projectiles' not in f for f in gf.values()): raise ValueError('Missing GPU shot capture')
        origin = round(nr['frames'][0]['game_time']*30)
        by_tick = {}
        for sample in native['tracking_projectile_samples']:
            tick = sample['tick']-origin
            if tick not in nf: continue
            for target in sample['targets']:
                for shot in target['shots']:
                    if shot.get('caster_team')!=case['actor_team'] or shot.get('caster_name')!='npc_dota_hero_nevermore': continue
                    if not target['available']: raise ValueError('Tracking unavailable')
                    name = match_native_target(target, nf[tick])
                    if name != 'victim': continue
                    if tick in by_tick: raise ValueError('Ambiguous duplicate tracking shot')
                    by_tick[tick] = dict(shot, tick=tick)
        nrel, grel = releases(nr, actor, True), releases(gr, actor, False)
        if len(nrel) != len(grel): raise ValueError('Different number of releases; explicit sequence diagnosis required')
        shots = []
        for index, (ne, ge) in enumerate(zip(nrel, grel)):
            nend = nrel[index+1]['tick'] if index+1<len(nrel) else max(nf)+1
            gend = grel[index+1]['tick'] if index+1<len(grel) else max(gf)+1
            ns = [by_tick[t] for t in sorted(by_tick) if ne['tick']<=t<nend]
            gs = []
            for tick in range(ge['tick'], gend):
                found = [p for p in gf[tick]['projectiles'] if p['source']==actor and p['target']=='victim']
                if len(found)>1: raise ValueError('Ambiguous GPU tracking shot')
                if found: gs.append(dict(found[0],tick=tick))
            if not ns or not gs: raise ValueError('Released shot not observed in both captures')
            npos, gpos = components(ns[0], rows(nf[ns[0]['tick']])[actor]), components(gs[0], rows(gf[gs[0]['tick']])[actor])
            np, gp = impact(nr,'victim',ne['tick'],nend), impact(gr,'victim',ge['tick'],gend)
            shots.append(dict(index=index, native_release=ne, gpu_release=ge,
                native_first_visible=ns[0],gpu_first_visible=gs[0],
                native_relative_first_visible=npos,gpu_relative_first_visible=gpos,
                native_measured_velocities=velocities(ns),gpu_measured_velocities=velocities(gs),
                native_impact=np,gpu_impact=gp,impact_delta_ticks=None if not np or not gp else gp['tick']-np['tick'],
                right_censored=np is None or gp is None,
                first_visible_forward_difference=gpos['forward']-npos['forward'],
                first_visible_absolute_lateral_difference=abs(gpos['lateral'])-abs(npos['lateral'])))
        results.append(dict(id=case['id'],team=case['actor_team'],shots=shots))
    return dict(cases=results,training_updates=0,training_eligible=False,
                scope='Measured stationary shot positions and timings; no phase shifts or hand mirroring applied to raw data.',
                limitations=['First visible position is not the creation attachment.',
                             'Native is3D; GPU trackingZ is explicitly unavailable.',
                             'Short final shots may be right-censored.',
                             'Hand-dependent signed differences alone do not identify a physics error.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--native',required=True,type=Path);p.add_argument('--gpu',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--timing-only',action='store_true')
    args=p.parse_args();raw=args.native.read_bytes();gpu=json.loads(args.gpu.read_text())
    sha=hashlib.sha256(raw).hexdigest()
    if gpu['native_sha256']!=sha: raise ValueError('GPU trace is from another native reference')
    report=(timing_audit if args.timing_only else audit)(json.loads(raw),gpu);report.update(native_sha256=sha,gpu_sha256=hashlib.sha256(args.gpu.read_bytes()).hexdigest())
    atomic(args.output,report)
    if args.timing_only:
        print(json.dumps(dict(cases=len(report['cases']),timing_passed=report['timing_passed'],
                              failed=[c['id']for c in report['cases']if not c['timing_passed']])))
        return
    print(json.dumps(dict(cases=len(report['cases']),shots=sum(len(c['shots'])for c in report['cases']),
        impacts=[dict(case=c['id'],shot=s['index'],native=s['native_impact'],gpu=s['gpu_impact'])for c in report['cases']for s in c['shots']])))


if __name__=='__main__':main()
