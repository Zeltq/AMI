"""Fit the height-level/physical-Z relation on train recordings; check held-out ones."""
import argparse,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import records

def samples(match,grid):
 source=Path(match['run'])/'native-decisions.pb.gz';values=[]
 for row in records(source):
  if not row.HasField('decision')or row.decision.sequence%4:continue
  team=row.decision.team
  unit=next((u for u in row.observation.units if u.name=='npc_dota_hero_nevermore'and u.team_id==team and u.is_alive),None)
  if unit is None:continue
  p=unit.location;x=int(np.clip((p.x+8192)//32,0,511));y=int(np.clip((p.y+8192)//32,0,511));values.append((float(grid[y,x,0]),p.z))
 if not values:raise RuntimeError(f'No live hero positions in {source}')
 return np.asarray(values,dtype=np.float64),str(source)

def error(values,offset,slope):
 delta=np.abs(offset+slope*values[:,0]-values[:,1])
 return dict(samples=len(values),mean=float(delta.mean()),median=float(np.median(delta)),p95=float(np.quantile(delta,.95)),maximum=float(delta.max()),within_4_units=float(np.mean(delta<=4)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/teacher-pilot-192-cache');p.add_argument('--calibration',default='runs/gpu/dota-calibration-003');p.add_argument('--output',required=True);a=p.parse_args()
 matches=json.loads((Path(a.cache)/'dataset.json').read_text())['matches'];grid=np.fromfile(Path(a.calibration)/'geometry.f32','<f4').reshape(512,512,2)
 groups={split:sorted((m for m in matches if m['split']==split),key=lambda m:m['index'])for split in ('train','validation')}
 values={};sources={}
 for split,count in (('train',4),('validation',2)):
  group=groups[split];indices=np.linspace(0,len(group)-1,count,dtype=int);parts=[samples(group[i],grid)for i in indices];values[split]=np.concatenate([v for v,_ in parts]);sources[split]=[s for _,s in parts]
  print(split,len(values[split]),flush=True)
 hist=defaultdict(Counter)
 for level,z in values['train']:hist[int(level)][round(z/8)*8]+=1
 modes={k:counts.most_common(1)[0][0]for k,counts in hist.items()};levels=np.array(sorted(modes));z=np.array([modes[k]for k in levels]);slope,offset=np.polyfit(levels,z,1)
 if not(np.isclose(slope,-128)and np.isclose(offset,512)):raise RuntimeError(f'Unexpected measured height relation: {modes}')
 report={'sources':sources,'calibration':a.calibration,'fitted_on':'train','plane_z_by_height_level':modes,'physical_z_offset':512.,'physical_z_slope':-128.,
  'errors':{s:{'previous_z_equals_level_times128':error(v,0,128),'measured_flat_plane_mapping':error(v,512,-128)}for s,v in values.items()},
  'limitation':'Discrete height levels do not reproduce continuous ramp height; these remaining errors are reported. Native terrain CNN inputs retain the original categorical height levels.'}
 Path(a.output).write_text(json.dumps(report,indent=2));print(a.output)

if __name__=='__main__':main()
