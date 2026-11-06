"""Audit the native GridNav export against observed blockage coordinates."""
import csv,hashlib,json,math,shutil
from pathlib import Path
import numpy as np
root=Path('C:/AMI/runs/p15');run=root/'gridnav-start-001';bundle=np.load(run/'live-grid.npz');meta=json.loads(str(bundle['metadata']));g=bundle['grid32'];out=Path('reports/P15/live-grid');out.mkdir(parents=True,exist_ok=True)
locations=[('radiant_building',-5420,-5356,50),('dire_building',4844,4912,220),('dire_trees',2782,2768,220),('native_tree_stop',-192,-1375.0017089844,270),('tree_center',-192,-1472,0),('building_center',4944,4776,0)]
def at(x,y):
    ix=int(np.clip(math.floor((x+8192)/32),0,511));iy=int(np.clip(math.floor((y+8192)/32),0,511));v=g[iy,ix]
    return {'traversable':bool(v[0]),'blocked':bool(v[1]),'height':float(v[2])}
results=[]
for name,x,y,angle in locations:
    hull=[at(x+24*math.cos(i*math.pi/4),y+24*math.sin(i*math.pi/4)) for i in range(8)]
    nx=x+305/30*math.cos(math.radians(angle));ny=y+305/30*math.sin(math.radians(angle))
    next_hull=[at(nx+24*math.cos(i*math.pi/4),ny+24*math.sin(i*math.pi/4)) for i in range(8)]
    results.append({'name':name,'position':[x,y],'center':at(x,y),'blocked_hull_samples':sum(v['blocked'] or not v['traversable'] for v in hull),'hull_samples':8,
                    'next_step_blocked_hull_samples':sum(v['blocked'] or not v['traversable'] for v in next_hull)})
    with (out/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['x','y','traversable','blocked','ground_height'])
        for dy in range(-8,9):
            for dx in range(-8,9):
                px=(math.floor(x/32)+dx+.5)*32;py=(math.floor(y/32)+dy+.5)*32;v=at(px,py);w.writerow([px,py,int(v['traversable']),int(v['blocked']),v['height']])
with (out/'trees.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['id','x','y','z','alive'],lineterminator='\n');w.writeheader();w.writerows(meta['trees'])
(out/'buildings.json').write_text(json.dumps(meta['buildings'],indent=2)+'\n',encoding='utf-8',newline='\n')
report=json.loads((run/'report.json').read_text());report.update(locations=results,grid_native_spacing=64,subcells_equal_to_64_grid=bool(np.array_equal(bundle['grid64'][:,:,:2].repeat(2,0).repeat(2,1),g[:,:,:2])),
    matched_blockage_locations={'numerator':sum(r['blocked_hull_samples']>0 for r in results[:3]),'denominator':3},
    conclusion='The requested GridNav queries do not expose the tree/building collision layer in this stock dedicated mode. They do not establish whether the actual tree collision shape is circular or grid-aligned. Keep the separately approved clearance atlas; do not replace it with this incomplete obstacle mask.',
    local_host_verified=False,physics_changed_for_this_audit=False)
Path('reports/P15/live-grid.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
env=root/'dota-calibration-005-chio';shutil.copy2(run/'live-grid.npz',env/'chio-live-grid.npz');engine=json.loads((env/'engine.json').read_text());engine['chio_live_grid']='chio-live-grid.npz';(env/'engine.json').write_text(json.dumps(engine,indent=2)+'\n',encoding='utf-8',newline='\n')
assert hashlib.sha256((env/'chio-live-grid.npz').read_bytes()).hexdigest()==report['sha256']
print(json.dumps({'matched':report['matched_blockage_locations'],'grid_spacing':64,'local_host_verified':False,'sha256':report['sha256']}))
