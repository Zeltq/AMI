"""Compare the unchanged 8x8 model terrain samples with the direct collision grid."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('C:/AMI/runs/p15');engine=json.loads((ROOT/'dota-calibration-005-chio/engine.json').read_text());base=Path(engine['calibration'])
geometry=np.fromfile(base/engine['geometry'],dtype='<f4').reshape(512,512,2)
movement=np.fromfile(base/engine['movement_geometry'],dtype='<f4').reshape(512,512,2)
def cell(x,y):return np.clip(np.floor((np.array([y,x])+8192)/32).astype(int),0,511)
points=[('radiant_base_exit',-5420,-5356),('dire_base_exit',6876,6632),('river_ramp',-1200,-1008)]
rows=[]
for name,x,y in points:
    samples=[];missed=0;blocked=0;disagree=0
    for iy in range(8):
        for ix in range(8):
            wx=x+(ix-3.5)*256;wy=y+(iy-3.5)*256;cy,cx=cell(wx,wy);original=geometry[cy,cx];actual=movement[cy,cx]
            samples.append(dict(ix=ix,iy=iy,x=wx,y=wy,height=float(original[0]),observable_passability=float(original[1]),collision_passability=float(actual[1])))
            disagree+=int(original[1]!=actual[1])
            for dy in np.arange(-112,128,32):
                for dx in np.arange(-112,128,32):
                    sy,sx=cell(wx+dx,wy+dy);bad=movement[sy,sx,1]<.5;blocked+=int(bad);missed+=int(bad and original[1]>.5)
    trees=[(i,t) for i,t in enumerate(engine['trees']) if (t['x']-x)**2+(t['y']-y)**2<1600**2]
    rows.append(dict(name=name,hero=[x,y],terrain_samples=samples,unrepresented_blocked_fine_cells=missed,blocked_fine_cells=blocked,
                     sampled_geometry_disagreements=disagree,nearby_tree_candidates=len(trees),tree_list_capacity=256,
                     caveat='Passability is additionally hidden outside sight; candidate trees additionally require visibility. Counts are an optimistic information bound.'))
result=dict(map_shape=[4,8,8],spacing=256,nominal_extent=2048,outer_sample_span=1792,collision_grid_spacing=32,
            channels=['height_level/5 (public)','passability if visible','visibility','reserved zero'],trees_in_terrain=False,
            tree_features='Separate up-to-256 visible living trees within 1600, in static table order, dx/dy/z/distance with known flags.',
            rows=rows,conclusion='8x8 point samples cannot resolve 32-unit cliff edges or an 88-unit tree clearance. The model also gets tree coordinates, but no local swept-hull clearance channel.',
            proposal='For a separately approved profile: 32x32 at 32 units (1024 span), obstacle distance or passability plus trees, with a coarser context map. Retrain CNN; do not alter chio-v1 silently.')
Path('reports/P15/terrain-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({r['name']:{k:r[k] for k in ('unrepresented_blocked_fine_cells','blocked_fine_cells','sampled_geometry_disagreements','nearby_tree_candidates')} for r in rows}))
