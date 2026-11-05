"""Check same-skill continuation and STOP against matched native controls."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path


def summarize(c):
    frames,orders=defaultdict(list),defaultdict(list)
    for f in c['frames']:frames[f['case']].append(f)
    for o in c['orders']:orders[o['case']].append(o)
    rows=[]
    for case in c['cases']:
        fs=frames[case['case']];os=orders[case['case']]
        definition=c['config']['cases'][case['case']-1]
        fire=[f['case_tick'] for f in fs if f['cooldown']>0]
        phase=[f['case_tick'] for f in fs if f['in_phase']]
        rows.append(dict(case=case['case'],spell=definition['spell'],mode=definition['mode'],repeat=definition['repeat'],
                         first_fire=fire[0] if fire else None,first_phase=phase[0] if phase else None,last_phase=phase[-1] if phase else None,
                         samples=len(fs),orders=len(os),latency_ok=all(o['tick']-o['observation_tick']==7 for o in os),
                         complete_ticks=[f['case_tick'] for f in fs]==list(range(180))))
    controls={(r['spell'],r['repeat']):r for r in rows if r['mode']=='once'}
    repeats=[r for r in rows if r['mode']=='repeat'];cancels=[r for r in rows if r['mode']=='cancel']
    parity=sum(r['first_fire'] is not None and r['first_fire']==controls[(r['spell'],r['repeat'])]['first_fire'] for r in repeats)
    cancellation=sum(r['first_fire'] is None and r['first_phase'] is not None and r['last_phase']<=44 for r in cancels)
    quality=sum(r['complete_ticks'] and r['latency_ok'] for r in rows)
    return dict(cases=rows,repeat_matches_control=dict(numerator=parity,denominator=len(repeats)),
                stop_cancels=dict(numerator=cancellation,denominator=len(cancels)),
                complete_capture=dict(numerator=quality,denominator=len(rows)),
                passed=bool(repeats and cancels) and parity==len(repeats) and cancellation==len(cancels) and quality==len(rows),
                elapsed_minutes=c['elapsed_minutes'],source_sha256=c['source_sha256'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    path=a.run/'capture.json';c=json.loads(path.read_text());result=summarize(c)
    result['capture_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    traces=a.output.parent/(a.output.stem+'-traces');traces.mkdir(exist_ok=True)
    for spell in range(1,5):
        for mode in ('once','repeat','cancel'):
            ids={r['case'] for r in result['cases'] if r['spell']==spell and r['mode']==mode}
            file=traces/f'spell{spell}-{mode}.csv'
            with file.open('w',newline='') as stream:
                writer=csv.writer(stream,lineterminator='\n');writer.writerow(['case','tick','case_tick','mana','cooldown','in_phase'])
                for f in c['frames']:
                    if f['case'] in ids:writer.writerow([f['case'],f['tick'],f['case_tick'],round(f['mana'],4),round(f['cooldown'],5),int(f['in_phase'])])
            assert file.stat().st_size<1_000_000
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','source_sha256')}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
