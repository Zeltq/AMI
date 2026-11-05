"""Summarize P15 capture/physics traces without discarding failed cases."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def summarize(capture, simulation=None):
    frames, orders = defaultdict(list), defaultdict(list)
    for f in capture['frames']:
        frames[f['case']].append(f)
    for o in capture['orders']:
        orders[o['case']].append(o)
    rows = []
    for case in capture['cases']:
        fs, os = frames[case['case']], orders[case['case']]
        positions = [(f['case_tick'], f['units'][0]['x'], f['units'][0]['y']) for f in fs]
        steps = [(b[0], math.hypot(b[1]-a[1], b[2]-a[2])*30)
                 for a, b in zip(positions, positions[1:]) if b[0] == a[0]+1 and b[0] >= 38]
        run, longest = 0, 0
        for _, speed in steps:
            run = run+1 if speed < 152.5 else 0
            longest = max(longest, run)
        expected = capture['config']['cases'][case['case']-1]
        initial = next((f['units'][0] for f in fs if f['case_tick'] == 36), None)
        facing_error = abs((initial['facing']-expected['facing']+180) % 360-180) if initial else None
        rows.append(dict(case=case['case'], name=case['name'], repeat=case['repeat_index'],
                         frames=len(fs), expected_frames=180, orders=len(os), expected_orders=21,
                         exact_ticks=[f['case_tick'] for f in fs] == list(range(180)),
                         cadence_ok=all(b['tick']-a['tick'] == 6 for a,b in zip(os,os[1:])),
                         latency_ok=all(o['tick']-o['observation_tick'] == 7 for o in os),
                         initial_facing_error=facing_error,
                         first=positions[36][1:], last=positions[173][1:],
                         mean_speed=sum(v for _,v in steps)/len(steps),
                         longest_slow_ticks=longest, stopped_over_one_second=longest>30))
    quality = dict(cases=len(rows), frames=len(capture['frames']), orders=len(capture['orders']),
                   structurally_valid_cases=sum(r['exact_ticks'] and r['orders']==21 and r['cadence_ok'] and
                                   r['latency_ok'] and r['initial_facing_error'] is not None and
                                   r['initial_facing_error']<1 for r in rows),
                   latency_ticks=dict(Counter(o['tick']-o['observation_tick'] for o in capture['orders'])))
    by_name = defaultdict(list)
    for r in rows:
        by_name[r['name']].append(r)
    totals = {name: dict(cases=len(rs), stopped=sum(r['stopped_over_one_second'] for r in rs),
                        mean_speed=sum(r['mean_speed'] for r in rs)/len(rs)) for name,rs in by_name.items()}
    if simulation:
        for name in totals:
            cases = [c for c in simulation['cases'] if c['name']==name]
            samples = [s for c in cases for s in c['samples']]
            totals[name].update(sim_cases=len(cases), paired_samples=len(samples),
                               max_error=max(s['error'] for s in samples),
                               mean_error=sum(s['error'] for s in samples)/len(samples))
    return dict(quality=quality, totals=totals, cases=rows, config=capture['config'],
                elapsed_minutes=capture['elapsed_minutes'], source_sha256=capture['source_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--simulation', type=Path)
    args=parser.parse_args()
    result=summarize(json.loads(args.capture.read_text()), json.loads(args.simulation.read_text()) if args.simulation else None)
    result['capture_sha256']=hashlib.sha256(args.capture.read_bytes()).hexdigest()
    result['capture']=str(args.capture)
    if args.simulation:
        result['simulation']=str(args.simulation)
        result['simulation_sha256']=hashlib.sha256(args.simulation.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result['quality']))


if __name__=='__main__':
    main()
