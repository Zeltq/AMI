"""Bake authored Dota FOW blocker cells and team revealers from an offline entity dump."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np

def entity_rows(source):
 text=Path(source).read_text(encoding='utf-8-sig')
 for block in re.split(r'^====\d+====\s*$',text,flags=re.M):
  yield {k:v.strip()for k,v in re.findall(r'^([\w.]+)\s+(.+)$',block,flags=re.M)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--entities',default='runs/gpu/dota-map-entities-001/maps/dota/entities/default_ents.vents');p.add_argument('--output',required=True);a=p.parse_args()
 source=Path(a.entities);nodes=[];revealers=[]
 for row in entity_rows(source):
  kind=row.get('classname','').strip('"')
  if kind not in ('ent_fow_blocker_node','ent_fow_revealer'):continue
  xyz=json.loads(row['origin'])
  if kind=='ent_fow_blocker_node':nodes.append({'x':xyz[0],'y':xyz[1],'z':xyz[2],'name':row.get('targetname','').strip('"')})
  else:
   radius=float(row['visionrange'].strip('"'));team=int(row['teamnumber'].strip('"'))-2
   if team not in(0,1):raise ValueError('Unexpected map revealer team')
   revealers.append({'x':xyz[0],'y':xyz[1],'team':team,'day':radius,'night':radius,'unobstructed':True,'name':kind})
 if not nodes or not revealers:raise RuntimeError('Missing authored FOW entities')
 grid=np.zeros((512,512),np.float32);outside=0
 for n in nodes:
  # Node coordinates are centres of64-unit fog cells; a few nodes have small
  # manual offsets. Raster the corresponding cell, not a fitted circle.
  x=int(np.floor((n['x']+8192)/64))*2;y=int(np.floor((n['y']+8192)/64))*2
  if x<0 or x>=512 or y<0 or y>=512:outside+=1;continue
  grid[y:y+2,x:x+2]=1
 root=Path(a.output);root.mkdir(parents=True,exist_ok=False);grid.tofile(root/'fow_blockers.f32')
 report={'entities_source':str(source),'entities_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'blocker_nodes':nodes,'revealers':revealers,'grid':{'shape':[512,512],'origin':[-8192,-8192],'cell_size':32,'node_cell_size':64,'file':'fow_blockers.f32','blocked_cells':int(grid.sum()),'nodes_outside_current_geometry':outside},'scope':'Public authored map data. Blocker raster and unobstructed-revealer semantics require native visibility validation; outer-map geometry expansion remains pending.'}
 (root/'fow.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'blocker_nodes':len(nodes),'revealers':len(revealers),'grid':report['grid']}))
if __name__=='__main__':main()
