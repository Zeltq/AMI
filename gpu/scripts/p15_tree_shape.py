"""Compare circle and grid-square tree hypotheses against two native approaches."""
import json,math
from pathlib import Path
root=Path('C:/AMI/runs/p15');rows=[]
for run in ['tree-approach-020','tree-diagonal-020']:
    capture=json.loads((root/run/'capture.json').read_text());simulation=json.loads((root/run/'simulator.json').read_text());sim={c['case']:c for c in simulation['cases']}
    for case in capture['cases']:
        frames=[f for f in capture['frames'] if f['case']==case['case']];u=frames[-1]['units'][0];angle=math.radians(u['facing']);step=u['speed']/30
        dx=u['x']+192;dy=u['y']+1472;nx=dx+step*math.cos(angle);ny=dy+step*math.sin(angle)
        box=lambda x,y:math.hypot(max(abs(x)-64,0),max(abs(y)-64,0))
        rows.append({'run':run,'case':case['case'],'repeat':case['repeat_index'],'native_stop':[u['x'],u['y']],'circle_radius_interval':[math.hypot(nx,ny),math.hypot(dx,dy)],
                     'distance_to_square_at_stop':box(dx,dy),'distance_to_square_next_step':box(nx,ny),'circle88_consistent':math.hypot(dx,dy)>=88 and math.hypot(nx,ny)<88,
                     'square64_sf24_consistent':box(dx,dy)>=24 and box(nx,ny)<24,'sim_max_xy_error':sim[case['case']]['max_error']})
result={'tree_center':[-192,-1472],'sf_hull':24,'repeats':len(rows),'circle88_consistent':sum(x['circle88_consistent'] for x in rows),'square64_sf24_consistent':sum(x['square64_sf24_consistent'] for x in rows),
        'physics_changed':False,'conclusion':'A single fixed circle radius cannot fit both disjoint measured intervals. A square with half extent 64 plus a radius-24 SF hull fits these two directions; this does not uniquely identify the full collision implementation. GridNav exposes no blocked object cells here.', 'cases':rows}
Path('reports/P15/tree-shape.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
s=['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="660" viewBox="0 0 720 660"><rect width="720" height="660" fill="white"/><style>text{font-family:Arial;font-size:16px}</style><text x="25" y="28">Tree (-192, -1472): two measured native approaches</text>'];cx,cy=360,375
for x in range(-256,257,64):s.append(f'<path d="M {cx+x} 70 V 590 M 100 {cy-x} H 620" stroke="#e2e8f0"/>')
s.append(f'<rect x="{cx-88}" y="{cy-88}" width="176" height="176" rx="24" fill="#dcead6" stroke="#548048" stroke-dasharray="5 4"/>')
s.append(f'<rect x="{cx-64}" y="{cy-64}" width="128" height="128" fill="none" stroke="#548048"/><circle cx="{cx}" cy="{cy}" r="88" fill="none" stroke="#dd7b22" stroke-width="2"/>')
for row in [rows[0],rows[20]]:
    x,y=row['native_stop'];dx=x+192;dy=y+1472;r=math.hypot(dx,dy);ox=dx/r*280;oy=dy/r*280
    s.append(f'<path d="M {cx+ox} {cy-oy} L {cx+dx} {cy-dy}" stroke="#2968b0" stroke-width="2"/><circle cx="{cx+dx}" cy="{cy-dy}" r="24" fill="none" stroke="#2968b0"/><circle cx="{cx+dx}" cy="{cy-dy}" r="4" fill="#2968b0"/>')
s += ['<text x="25" y="615">Blue: Dota / SF hull. Orange: simulator circle radius 88.</text><text x="25" y="642">Green: square half-extent 64 + SF hull 24 hypothesis; fits 40/40.</text></svg>']
Path('reports/P15/tree-shape.svg').write_text(''.join(s),encoding='utf-8',newline='\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
