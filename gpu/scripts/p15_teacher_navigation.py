"""Build a teacher-only CHiO atlas with the existing navigation builder."""
import argparse,hashlib,json,math,shutil,subprocess,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('engine',type=Path);p.add_argument('executable',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
engine=json.loads(a.engine.read_text());assert engine['chio_direct'];base=Path(engine['calibration']);original={k:hashlib.sha256((base/engine[k]).read_bytes()).hexdigest() for k in ('geometry','movement_geometry','navigation','fow_blockers','static_vision') if engine.get(k)}
a.output.mkdir(parents=True,exist_ok=False);physical=np.fromfile(base/engine['movement_geometry'],dtype='<f4').reshape(512,512,2);derived=physical.copy();walk=physical[:,:,1]>.5
# Conservative cell coverage: a 32-unit cell has a 22.63-unit half diagonal.
# One-cell square erosion keeps the 24-unit SF hull off blocked cell boundaries.
padded=np.pad(walk,1,constant_values=False);clear=np.ones_like(walk)
for dy in range(3):
 for dx in range(3):clear &= padded[dy:dy+512,dx:dx+512]
margin=math.sqrt(2)*16
obstacles=[(t['x'],t['y'],88+margin) for t in engine['trees']]
obstacles += [(v[0],v[1],168+margin) for i,v in enumerate(engine['prototypes']) if 2<=i<6 or i>=9]
for x,y,radius in obstacles:
 lo=np.maximum(np.floor((np.array([x,y])-radius+8192)/32).astype(int),0);hi=np.minimum(np.floor((np.array([x,y])+radius+8192)/32).astype(int),511)
 if np.any(lo>hi):continue
 xx=-8192+(np.arange(lo[0],hi[0]+1)+.5)*32;yy=-8192+(np.arange(lo[1],hi[1]+1)+.5)*32
 clear[lo[1]:hi[1]+1,lo[0]:hi[0]+1] &= (xx[None,:]-x)**2+(yy[:,None]-y)**2>=radius**2
derived[:,:,1]=clear;derived.astype('<f4').tofile(a.output/'teacher-clearance.f32')
config={**engine,'calibration':a.output.as_posix(),'movement_geometry':'teacher-clearance.f32'}
(a.output/'engine.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8',newline='\n')
started=time.monotonic();subprocess.run([str(a.executable),'build-navigation',str(a.output/'engine.json')],check=True)
source=a.output/'movement-navigation64.u32';target=base/'chio-teacher-navigation64.u32';shutil.copy2(source,target)
updated=json.loads(a.engine.read_text());updated['chio_teacher_navigation']=target.name
a.engine.write_text(json.dumps(updated,indent=2)+'\n',encoding='utf-8',newline='\n')
for k,sha in original.items():assert hashlib.sha256((base/engine[k]).read_bytes()).hexdigest()==sha
report={'file':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size,'teacher_only':True,'unchanged_original_asset_sha256':original,
 'tree_radius':88,'hero_hull':24,'building_clearance':168,'cell_coverage_margin':margin,'clear_cells':int(clear.sum()),'total_cells':int(clear.size),'source':'existing navigation::build, 64-unit nodes / 256-unit goal bins',
 'clearance_geometry_sha256':hashlib.sha256((a.output/'teacher-clearance.f32').read_bytes()).hexdigest(),'wall_minutes':(time.monotonic()-started)/60}
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(report))
