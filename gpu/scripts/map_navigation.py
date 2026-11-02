"""Decode installed GNV navigation, preserving Bot API terrain as a separate ABI.

Bit0 is the walkable flag. Validate this interpretation on disjoint recorded
train/validation matches and native movement before using it for physics.
"""
import argparse,collections,hashlib,json,struct
from pathlib import Path
import numpy as np
from vpk import Vpk

def main():
 p=argparse.ArgumentParser();p.add_argument('--map',default='C:/Program Files (x86)/Steam/steamapps/common/dota 2 beta/game/dota/maps/dota.vpk');p.add_argument('--calibration',default='runs/gpu/dota-calibration-003');p.add_argument('--cache',default='runs/teacher-pilot-192-cache');p.add_argument('--movement',default='runs/gpu/dota-movement-003-passability/scenarios.json');p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.output);root.mkdir(parents=True,exist_ok=True);raw=Vpk(a.map).read('maps/dota.gnv')
 magic,cell,cx,cy,width,height,ox,oy=struct.unpack('<Ifffiiii',raw[:32])
 if magic!=0xfadebead or cell!=64 or (cx,cy)!=(32,32)or len(raw)!=32+width*height:raise ValueError('Unsupported GNV header/layout')
 flags=np.frombuffer(raw[32:],np.uint8).reshape(height,width)
 def query(pos):
  xy=np.floor(pos/cell).astype(int)-[ox,oy]
  if ((xy<0)|(xy>=[width,height])).any():raise ValueError('Position outside installed navigation bounds')
  return flags[xy[:,1],xy[:,0]]
 grid=np.fromfile(Path(a.calibration)/'geometry.f32','<f4').reshape(512,512,2);report={}
 cache=Path(a.cache);matches=json.loads((cache/'dataset.json').read_text())['matches']
 for split,count in(('train',4),('validation',2)):
  group=[m for m in matches if m['split']==split];hist=collections.Counter();old_pass=0;samples=0;sources=[]
  for index in np.linspace(0,len(group)-1,count,dtype=int):
   m=group[index]
   for team in(2,3):
    file=cache/f"match-{m['index']:04d}-team-{team}.npz";sources.append(str(file))
    with np.load(file)as x:
     for key,mask in(('heroes','hero_valid'),('creeps','creep_valid')):
      pos=x[key][:,:,:2].astype(np.float32)[x[mask].astype(bool)]*8192;v=query(pos);hist.update(map(int,v));samples+=len(pos)
      ij=np.floor((pos+8192)/32).astype(int).clip(0,511);old_pass+=int((grid[ij[:,1],ij[:,0],1]>.5).sum())
  walkable=sum(n for flag,n in hist.items()if flag&1);report[split]={'sources':sources,'samples':samples,'flags':dict(hist),'gnv_walkable_fraction':walkable/samples,'api_passable_fraction':old_pass/samples}
  if walkable!=samples:raise ValueError(f'Unexpected blocked native positions in {split}: {hist}')
 movement=json.loads(Path(a.movement).read_text());frames=[f for c in movement['scenarios']for f in c['frames']if f['team']==2];pos=np.array([[f['x'],f['y']]for f in frames]);v=query(pos)
 report['movement']={'source':a.movement,'samples':len(frames),'gnv_walkable':int(((v&1)>0).sum()),'native_api_false':sum(not f['passable']for f in frames)}
 if not movement['capture_passed']or not np.all(v&1):raise ValueError('Movement capture does not support the GNV walkability interpretation')
 xx,yy=np.meshgrid(-8192+(np.arange(512)+.5)*32,-8192+(np.arange(512)+.5)*32);pos=np.stack([xx.ravel(),yy.ravel()],1)
 physical=grid.copy();physical[:,:,1]=((query(pos)&1)>0).reshape(512,512);physical.astype('<f4').tofile(root/'movement-geometry.f32');(root/'dota.gnv').write_bytes(raw)
 result={'source_map':a.map,'map_sha256':hashlib.sha256(Path(a.map).read_bytes()).hexdigest(),'gnv_sha256':hashlib.sha256(raw).hexdigest(),'header':{'cell':cell,'center_offsets':[cx,cy],'shape':[height,width],'origin_cell':[ox,oy],'world_bounds':[ox*cell,oy*cell,(ox+width)*cell,(oy+height)*cell]},'walkable_bit':0,'flags':dict(collections.Counter(map(int,flags.ravel()))),'verification':report,'physics_file':'movement-geometry.f32','observation_geometry_unchanged':True,'scope':'Central512x512 physics map; full GNV bounds preserved but outer simulation not yet expanded. Static navigation does not replace live tree/building/unit collision.'}
 (root/'navigation-rules.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items()if k!='verification'}))
if __name__=='__main__':main()
