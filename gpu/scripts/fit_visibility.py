"""Offline LOS hypothesis checks on stationary native snapshots; never a CPU training simulator."""
import argparse,json,sys,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import records

def cells(xy):return np.clip(np.floor((xy+8192)/32).astype(int),0,511)

def tree_grid(trees):
 nearest=np.full((512,512),-1,np.int32);distance=np.full((512,512),np.inf)
 for i,t in enumerate(trees):
  low=cells(t-128);high=cells(t+128);y,x=np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1]
  d=(-8192+(x+.5)*32-t[0])**2+(-8192+(y+.5)*32-t[1])**2;region=np.s_[low[1]:high[1]+1,low[0]:high[0]+1];mask=d<distance[region];distance[region][mask]=d[mask];nearest[region][mask]=i
 return nearest

def predict(sample,geometry,trees,nearest,radius,target_grid,source_grid,ignore_target,steps,terrain_clearance):
 origins=sample['sources'][:,:2].copy();targets=sample['targets'].copy()
 if source_grid:origins=(np.floor((origins+8192)/source_grid)+.5)*source_grid-8192
 if target_grid:targets=(np.floor((targets+8192)/target_grid)+.5)*target_grid-8192
 delta=targets[None,:,:]-origins[:,None,:];length2=(delta**2).sum(-1);in_range=length2<=sample['sources'][:,2,None]**2
 f=np.arange(1,steps)/steps;rays=origins[:,None,None,:]+delta[:,:,None,:]*f[None,None,:,None];idx=cells(rays);source_cells=cells(origins)
 height=geometry[source_cells[:,1],source_cells[:,0],0];uphill=(geometry[idx[:,:,:,1],idx[:,:,:,0],0]<height[:,None,None]).any(-1)
 if sample.get('fow')is not None:uphill|=(sample['fow'][idx[:,:,:,1],idx[:,:,:,0]]>.5).any(-1)
 if terrain_clearance:
  opaque=(geometry[idx[:,:,:,1],idx[:,:,:,0],1]<.5)&(length2[:,:,None]*f[None,None,:]**2>terrain_clearance**2)
  uphill|=opaque.any(-1)
 tree_ids=nearest[idx[:,:,:,1],idx[:,:,:,0]];tree_xy=trees[np.maximum(0,tree_ids)];offset=rays-tree_xy
 along=((tree_xy-origins[:,None,None,:])*delta[:,:,None,:]).sum(-1)
 blocked=(tree_ids>=0)&((offset**2).sum(-1)<radius**2)&(along>0)&(along<length2[:,:,None])
 if ignore_target:blocked&=tree_ids!=sample['tree_ids'][None,:,None]
 seen=(in_range&~uphill&~blocked.any(-1)).any(0)
 for source in sample.get('revealers',[]):seen|=((targets-np.array([source['x'],source['y']]))**2).sum(-1)<=source['day']**2
 return seen

def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',default='runs/gpu/dota-vision-003/vision.json');p.add_argument('--engine',default='runs/gpu/dota-calibration-003/engine.json');p.add_argument('--output',required=True);p.add_argument('--terrain-block-test',action='store_true');p.add_argument('--authored-fow');a=p.parse_args()
 engine=json.loads(Path(a.engine).read_text());geometry=np.fromfile(Path(engine['calibration'])/engine['geometry'],np.float32).reshape(512,512,2);trees=np.array([[t['x'],t['y']]for t in engine['trees']]);nearest=tree_grid(trees);tree_lookup={tuple(p):i for i,p in enumerate(trees)}
 initial={};source=Path('runs/teacher-pilot-192/match-0001-attempt-09/native-decisions.pb.gz')
 for row in records(source):
  if row.HasField('decision')and row.decision.team not in initial:initial[row.decision.team]=row
  if len(initial)==2:break
 fow=None;revealers=[]
 if a.authored_fow:
  root=Path(a.authored_fow);fow=np.fromfile(root/'fow_blockers.f32',np.float32).reshape(512,512);revealers=json.loads((root/'fow.json').read_text())['revealers']
 frames=json.loads(Path(a.capture).read_text())['frames'];samples=[]
 for side in (2,3):
  rows=[r for r in frames if r['team']==side]
  for time in (-80,1,301):
   r=min(rows,key=lambda x:abs(x['time']-time));day=r['vision_api']['GetCurrentVisionRange']==1800
   sources=np.array([[u.location.x,u.location.y,u.vision_range_daytime if day else u.vision_range_nighttime]for u in initial[side].observation.units if u.team_id==side and u.is_alive and u.vision_range_daytime])
   targets=[[r['x']+(x-3.5)*256,r['y']+(y-3.5)*256]for y in range(8)for x in range(8)]+[[t['x'],t['y']]for t in r['trees']]
   sample={'team':side,'time':r['time'],'sources':sources,'targets':np.array(targets),'expected':np.array([v['visible']for v in r['grid']]+[v['visible']for v in r['trees']]),'tree_ids':np.array([-1]*64+[tree_lookup.get((t['x'],t['y']),-1)for t in r['trees']]),'fow':fow,'revealers':[v for v in revealers if v['team']==side-2]};samples.append(sample)
 results=[]
 for radius,target_grid,source_grid,ignore_target,steps,terrain_clearance in itertools.product((32,64)if a.terrain_block_test else(16,24,32,48,64),(0,32,64),(0,)if a.terrain_block_test else(0,64),(False,True),(32,)if a.terrain_block_test else(32,64),(0,96,192,384)if a.terrain_block_test else(0,)):
  errors=[]
  for sample in samples:
   mismatch=predict(sample,geometry,trees,nearest,radius,target_grid,source_grid,ignore_target,steps,terrain_clearance)!=sample['expected'];errors.append({'team':sample['team'],'time':sample['time'],'grid_errors':int(mismatch[:64].sum()),'tree_errors':int(mismatch[64:].sum())})
  results.append({'radius':radius,'target_grid':target_grid,'source_grid':source_grid,'ignore_target_tree':ignore_target,'ray_steps':steps,'terrain_clearance':terrain_clearance,'errors':sum(v['grid_errors']+v['tree_errors']for v in errors),'samples':errors})
 results.sort(key=lambda r:r['errors']);baseline=next(r for r in results if (r['radius'],r['target_grid'],r['source_grid'],r['ignore_target_tree'],r['ray_steps'],r['terrain_clearance'])==(64,0,0,True,32,0))
 report={'capture':a.capture,'native_source_record':str(source),'authored_fow':a.authored_fow,'baseline':baseline,'best_candidates':results[:12],'scope':'Stationary fountain positions only, using all native allied vision sources. This is a hypothesis-selection diagnostic; independently moved heroes and held-out recordings must validate a change before calling LOS correct.'};Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'baseline':baseline,'best':results[0]},indent=2))
if __name__=='__main__':main()
