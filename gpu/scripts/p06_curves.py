"""Turn measured Bot API attack segments into a production GPU curve fixture."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('summary',type=Path);p.add_argument('fixture',type=Path);p.add_argument('report',type=Path)
    a=p.parse_args();fixture=[];coverage={};bins=defaultdict(list)
    for file in a.summary.glob('*-attacks.csv'):
        segments=defaultdict(list)
        with file.open() as f:
            for r in csv.DictReader(f):segments[(r['team'],r['id'],r['start'])].append(r)
        kept=0;total=sum(map(len,segments.values()))
        for rows in segments.values():
            # A first sighting halfway through a swing cannot locate its start.
            if float(rows[0]['cycle'])>1e-4:continue
            for r in rows:
                name=r['name'];k=0 if r['kind'].startswith('sf_') else 1 if r['kind']=='tower' else 3 if r['kind']=='ranged' else 2
                # Observer team is not the unit's team; derive nonhero team
                # from its canonical engine name. SF rate is side-independent.
                team=1 if 'badguys' in name else 0
                group=name+('/'+r['kind'] if k==0 else '')
                f=dict(kind=k,team=team,interval=float(r['interval']),elapsed=float(r['elapsed']),cycle=float(r['cycle']),group=group)
                fixture.append(f);kept+=1
                tick=round(f['elapsed']*30);bins[(group,tick)].append(f['cycle'])
        coverage[file.stem]=dict(used=kept,total=total,segments=len(segments),criterion='first observed attack cycle <= 0.0001')
    a.fixture.write_text(json.dumps(fixture,separators=(',',':')),encoding='utf-8')
    a.report.mkdir(parents=True,exist_ok=True)
    (a.report/'curve-coverage.json').write_text(json.dumps(coverage,indent=2))
    with (a.report/'dota-curves.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['group','elapsed_seconds','samples','cycle_min','cycle_mean','cycle_max'])
        for (group,tick),v in sorted(bins.items()):writer.writerow([group,tick/30,len(v),min(v),sum(v)/len(v),max(v)])
    print(json.dumps(dict(fixture=str(a.fixture),samples=len(fixture),coverage=coverage),indent=2))


if __name__=='__main__':main()
