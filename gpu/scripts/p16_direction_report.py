"""Publish direction audit, exact native command checks and three trajectory maps."""
import csv,json,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RUN=Path('C:/AMI/runs/p16');OUT=ROOT/'reports/P16/direction-diagnosis';OUT.mkdir(exist_ok=True)
audit=json.loads((RUN/'direction-audit-clock/result.json').read_text());rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((RUN/'direction-trajectories/direction-diagnostics.csv').open())]
engine=json.loads((RUN/'environment/engine.json').read_text());grid=np.fromfile(RUN/'environment'/engine['movement_geometry'],dtype='<f4').reshape(512,512,2)
scenes=[];errors=[];moves=0
for match in range(3):
    data=[r for r in rows if r['match']==match];slow=0;first=None
    for i,r in enumerate(data):
        distance=math.hypot(r['after_x']-r['x'],r['after_y']-r['y']);slow=slow+1 if r['order'] in (1,2) and distance<.5*r['move_speed']*.2 else 0
        if slow>=6 and first is None:first=r
        if r['action']==0:
            angle=math.radians(r['direction']*10);x=max(-8192,min(8192,r['x']+2000*math.cos(angle)));y=max(-8192,min(8192,r['y']+2000*math.sin(angle)))
            errors.append(math.hypot(r['queued_x']-x,r['queued_y']-y));assert r['queued_action']==2;moves+=1
    assert first is not None
    for suffix,selected in [('first60',[r for r in data if r['elapsed']<=60]),('game0-60',[r for r in data if r['game_seconds']>=0])]:
        with (OUT/f'trajectory-{match}-{suffix}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(selected)
    scenes.append(dict(match=match,side='radiant' if match%2==0 else 'dire',first_stuck=first,last_position=[data[-1]['x'],data[-1]['y']],frames=len(data)))
assert max(errors)<.003
result=dict(classification='distribution-shift',audit=audit,trajectories=scenes,queue_point_checks={'numerator':moves,'denominator':moves,'max_error':max(errors),'tolerance':.003},direction_unit_cases={'numerator':72,'denominator':72},
            evidence='No axis mirroring/bin-order/+2000 executor error found. Teacher-state best offset is zero; stuck-state angle errors are varied. Native Radiant predicts 40 degrees versus teacher 130; Dire predicts 220 versus teacher 270. Student points nearly toward mid while the teacher routes around base obstacles.',
            pre_mid_definition='Exact recorded decision index*0.2 < native mid_arrival_elapsed_seconds; only teacher MOVE labels. First idle/STOP label is excluded. Stuck DAgger rows use the documented observation proxy; native trajectories use actual order and displacement.')
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="560" viewBox="0 0 1260 560"><rect width="1260" height="560" fill="white"/><style>text{font-family:Arial,sans-serif;font-size:13px}.title{font-size:17px;font-weight:bold}</style>']
for i,scene in enumerate(scenes):
    stop=scene['first_stuck'];x,y=stop['x'],stop['y'];ox=i*420+20;oy=90;size=380;radius=500;scale=size/(2*radius)
    def point(px,py):return ox+size/2+(px-x)*scale,oy+size/2-(py-y)*scale
    svg += [f'<text x="{ox}" y="24" class="title">{scene["side"]}, match {i}</text>',f'<text x="{ox}" y="46">Stop ({x:.2f}, {y:.2f}), t={stop["elapsed"]:.1f}s</text>',f'<text x="{ox}" y="67">Model {stop["direction"]*10:.0f}°, teacher {stop["teacher_direction"]*10:.0f}°</text>',f'<clipPath id="p{i}"><rect x="{ox}" y="{oy}" width="{size}" height="{size}"/></clipPath>',f'<g clip-path="url(#p{i})"><rect x="{ox}" y="{oy}" width="{size}" height="{size}" fill="#f1f5f9"/>']
    for gy in range(max(0,int((y-radius+8192)/32)),min(512,int((y+radius+8192)/32)+1)):
        for gx in range(max(0,int((x-radius+8192)/32)),min(512,int((x+radius+8192)/32)+1)):
            if grid[gy,gx,1]<.5:
                px,py=point(-8192+gx*32,-8192+(gy+1)*32);svg.append(f'<rect x="{px}" y="{py}" width="{32*scale}" height="{32*scale}" fill="#64748b"/>')
    for tree in engine['trees']:
        if abs(tree['x']-x)<radius+90 and abs(tree['y']-y)<radius+90:
            px,py=point(tree['x'],tree['y']);svg.append(f'<circle cx="{px}" cy="{py}" r="{88*scale}" fill="#86b879" fill-opacity=".5"/>')
    for j,v in enumerate(engine['prototypes']):
        if (2<=j<6 or j>=9) and abs(v[0]-x)<radius+170 and abs(v[1]-y)<radius+170:
            px,py=point(v[0],v[1]);svg.append(f'<circle cx="{px}" cy="{py}" r="{168*scale}" fill="#e9b36b" fill-opacity=".6"/>')
    trajectory=[r for r in rows if r['match']==i and r['elapsed']<=60];points=' '.join(f'{a:.2f},{b:.2f}' for a,b in [point(r['x'],r['y']) for r in trajectory]);svg.append(f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="3"/>')
    px,py=point(x,y);svg.append(f'<circle cx="{px}" cy="{py}" r="5" fill="#111827"/>')
    for key,color in [('direction','#dc2626'),('teacher_direction','#7c3aed')]:
        angle=math.radians(stop[key]*10);tx,ty=point(x+250*math.cos(angle),y+250*math.sin(angle));svg.append(f'<path d="M {px} {py} L {tx} {ty}" stroke="{color}" stroke-width="4"/>')
    svg += ['</g>',f'<rect x="{ox}" y="{oy}" width="{size}" height="{size}" fill="none" stroke="#334155"/>']
svg+=['<text x="20" y="505">Blue: first 60s trajectory. Red: student direction. Purple: teacher direction. x right, y up.</text><text x="20" y="530">Gray: blocked terrain32. Green: trees circle88. Orange: building hull+SF. Matches0/2 repeat the same Radiant blockage.</text></svg>'];text=''.join(svg);ET.fromstring(text);(OUT/'trajectories.svg').write_text(text,encoding='utf-8',newline='\n')
for p in OUT.iterdir():assert p.stat().st_size<1000000
print(json.dumps({k:v for k,v in result.items() if k!='audit'}))
