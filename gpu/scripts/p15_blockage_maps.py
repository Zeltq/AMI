"""Vector maps of the three observed CHiO translator blockage locations."""
import json,math,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
engine=json.loads(Path('C:/AMI/runs/p15/dota-calibration-005-chio/engine.json').read_text());root=Path(engine['calibration']);grid=np.fromfile(root/engine['movement_geometry'],dtype='<f4').reshape(512,512,2)
scenes=[('Radiant: здание базы',-5420,-5356,50),('Dire: здание базы',4844,4912,220),('Dire: группа деревьев',2782,2768,None)]
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="530" viewBox="0 0 1260 530"><rect width="1260" height="530" fill="white"/><style>text{font-family:Arial,sans-serif;font-size:14px}.title{font-size:17px;font-weight:bold}</style>']
for i,(title,x,y,angle) in enumerate(scenes):
 ox=i*420+20;oy=70;scale=.5;size=380;radius=size/scale/2
 def point(px,py):return ox+size/2+(px-x)*scale,oy+size/2-(py-y)*scale
 svg += [f'<text x="{ox}" y="26" class="title">{title}</text>',f'<text x="{ox}" y="49">Герой ≈ ({x}, {y}), окно {int(radius*2)}×{int(radius*2)}</text>',f'<clipPath id="p{i}"><rect x="{ox}" y="{oy}" width="{size}" height="{size}"/></clipPath>',f'<g clip-path="url(#p{i})"><rect x="{ox}" y="{oy}" width="{size}" height="{size}" fill="#f1f5f9"/>']
 for gy in range(max(0,int((y-radius+8192)/32)),min(512,int((y+radius+8192)/32)+1)):
  for gx in range(max(0,int((x-radius+8192)/32)),min(512,int((x+radius+8192)/32)+1)):
   if grid[gy,gx,1]<.5:
    px,py=point(-8192+gx*32,-8192+(gy+1)*32);svg.append(f'<rect x="{px}" y="{py}" width="16" height="16" fill="#475569"/>')
 for tree in engine['trees']:
  if abs(tree['x']-x)<radius+100 and abs(tree['y']-y)<radius+100:
   px,py=point(tree['x'],tree['y']);svg += [f'<circle cx="{px}" cy="{py}" r="32" fill="#86b879" fill-opacity=".7"/>',f'<circle cx="{px}" cy="{py}" r="44" fill="none" stroke="#416539" stroke-dasharray="3 3"/>']
 for j,v in enumerate(engine['prototypes']):
  if (2<=j<6 or j>=9) and abs(v[0]-x)<radius+170 and abs(v[1]-y)<radius+170:
   px,py=point(v[0],v[1]);svg += [f'<circle cx="{px}" cy="{py}" r="72" fill="#e9b36b" fill-opacity=".7"/>',f'<circle cx="{px}" cy="{py}" r="84" fill="none" stroke="#a56116" stroke-dasharray="3 3"/>']
 px,py=point(x,y);svg.append(f'<circle cx="{px}" cy="{py}" r="12" fill="#cf3047"/>')
 if angle is not None:
  tx,ty=point(x+180*math.cos(math.radians(angle)),y+180*math.sin(math.radians(angle)));svg.append(f'<path d="M {px} {py} L {tx} {ty}" stroke="#cf3047" stroke-width="3"/>')
 svg += ['</g>',f'<rect x="{ox}" y="{oy}" width="{size}" height="{size}" fill="none" stroke="#334155"/>',f'<text x="{ox}" y="475">Ось x →, ось y ↑; прямое движение показано красным.</text>']
svg += ['<text x="20" y="510">Серое: непроходимая сетка 32. Зелёное: деревья. Охра: здания. Пунктир: радиус препятствия + корпус героя 24.</text></svg>']
text=''.join(svg);ET.fromstring(text);assert len(text.encode())<1000000;Path('reports/P15/blockage-maps.svg').write_text(text,encoding='utf-8',newline='\n')
