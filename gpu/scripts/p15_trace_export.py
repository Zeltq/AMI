"""Export reviewable per-scenario CSV, each strictly below one megabyte."""
import argparse
import csv
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--simulation',default='simulator.json')
    a=p.parse_args()
    c=json.loads((a.run/'capture.json').read_text())
    simulation=a.run/a.simulation
    s=json.loads(simulation.read_text()) if simulation.exists() else dict(cases=[])
    sim={(case['case'],f['tick']):f for case in s['cases'] for f in case['samples']}
    names={v['case']:v['name'] for v in c['cases']}
    a.output.mkdir(parents=True,exist_ok=True)
    for name in sorted(set(names.values())):
        path=a.output/(name+'.csv')
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream,lineterminator='\n')
            writer.writerow(['case','tick','case_tick','native_x','native_y','native_z','native_facing','sim_x','sim_y','error_xy','sim_z','error_z','colliders_xy'])
            for f in c['frames']:
                if names[f['case']]!=name:continue
                native=f['units'][0];model=sim.get((f['case'],f['tick']))
                values=[f['case'],f['tick'],f['case_tick'],*[round(native[k],4) for k in ('x','y','z','facing')]]
                values+=([round(v,4) for v in model['sim']]+[round(model['error'],4)]) if model else ['','','']
                values += [round(model[k],4) if model and k in model else '' for k in ('sim_z','error_z')]
                values.append('|'.join(f"{u['kind']}:{u['x']:.3f}:{u['y']:.3f}" for u in f['units'][1:] if abs(u['x']-native['x'])<2500))
                writer.writerow(values)
        if path.stat().st_size>=1_000_000:
            raise ValueError(f'{path} exceeds artifact size limit')
    for name in ('bot_generic.lua','p15_server.lua'):
        shutil.copy2(a.run/'bots'/name,a.output/name)


if __name__=='__main__':
    main()
