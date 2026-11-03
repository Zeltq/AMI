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
                base=1.6 if k==0 else .9 if k==1 else 1
                rate=1/1.1 if k==0 else 1.035952 if k==1 else 15/14 if k==2 and team==1 else 30/29 if k==3 and team==1 else 1
                simulated=min(1,max(0,f['elapsed']-1/30)*rate*base/f['interval'])
                tick=round(f['elapsed']*30);bins[(group,tick)].append((f['cycle'],simulated))
        coverage[file.stem]=dict(used=kept,total=total,segments=len(segments),criterion='first observed attack cycle <= 0.0001')
    a.fixture.write_text(json.dumps(fixture,separators=(',',':')),encoding='utf-8')
    a.report.mkdir(parents=True,exist_ok=True)
    (a.report/'curve-coverage.json').write_text(json.dumps(coverage,indent=2))
    with (a.report/'dota-curves.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['group','elapsed_seconds','samples','cycle_min','cycle_mean','cycle_max','simulator_mean'])
        for (group,tick),pairs in sorted(bins.items()):
            v=[x[0] for x in pairs];sim=[x[1] for x in pairs]
            writer.writerow([group,tick/30,len(v),min(v),sum(v)/len(v),max(v),sum(sim)/len(sim)])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for ax,kind in zip(axes.flat,('melee','ranged','tower','nevermore')):
        groups=sorted({g for g,t in bins if kind in g and 'flagbearer' not in g and 'sf_enemy' not in g})
        for group in groups:
            points=sorted((t,pairs) for (g,t),pairs in bins.items() if g==group)
            label='Dire' if 'badguys' in group else 'SF' if kind=='nevermore' else 'Radiant'
            line=ax.plot([t/30 for t,v in points],[sum(x[0] for x in v)/len(v) for t,v in points],label='Dota '+label)[0]
            ax.plot([t/30 for t,v in points],[sum(x[1] for x in v)/len(v) for t,v in points],'--',color=line.get_color(),label='Simulator '+label)
        ax.set_title(kind);ax.set_xlim(0,1.2);ax.set_ylim(0,1.03);ax.grid(alpha=.2);ax.legend(fontsize=7)
        ax.set_xlabel('Seconds since observed attack start');ax.set_ylabel('Attack cycle')
    fig.suptitle('Mean curves; all complete observed starts, including one-tick phase uncertainty')
    fig.tight_layout();fig.savefig(a.report/'attack-curves.svg');plt.close(fig)
    print(json.dumps(dict(fixture=str(a.fixture),samples=len(fixture),coverage=coverage),indent=2))


if __name__=='__main__':main()
