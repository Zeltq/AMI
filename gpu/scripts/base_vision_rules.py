"""Capture immutable base-building vision from training; check held-out geometry/radii."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import records

def sources(match):
 source=Path(match['run'])/'native-decisions.pb.gz';teams={}
 for row in records(source):
  if not row.HasField('decision')or row.decision.team in teams:continue
  team=row.decision.team;result=[]
  for u in row.observation.units:
   fixed=('rax_'in u.name or u.name.endswith('_fort')or u.name.endswith('_fillers')or u.name=='ent_dota_halloffame')
   if u.team_id==team and u.is_alive and fixed and u.vision_range_daytime:
    result.append({'name':u.name,'x':round(u.location.x,1),'y':round(u.location.y,1),'team':team-2,'day':u.vision_range_daytime,'night':u.vision_range_nighttime})
  teams[team]=sorted(result,key=lambda u:(u['name'],u['x'],u['y']))
  if len(teams)==2:break
 if len(teams)!=2:raise RuntimeError(f'No both-side base sources in{source}')
 return teams[2]+teams[3],str(source)

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/teacher-pilot-192-cache');p.add_argument('--output',required=True);a=p.parse_args()
 matches=json.loads((Path(a.cache)/'dataset.json').read_text())['matches'];train=next(m for m in sorted(matches,key=lambda m:m['index'])if m['split']=='train');values,source=sources(train)
 checks=[]
 for m in [x for x in sorted(matches,key=lambda m:m['index'])if x['split']=='validation'][:2]:
  actual,path=sources(m);checks.append({'source':path,'matches_train':actual==values,'sources':len(actual)})
 if not values or len(values)>64:raise RuntimeError('Invalid base source count')
 report={'source':source,'split':'train','sources':values,'validation':checks,'scope':'Public stationary sources outside the allowed attack-target ABI. Matches terminate on the mid T1 death, before lane creeps can destroy these base buildings. Cosmetic hall-of-fame source is taken from actual native metadata.'}
 Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps({'sources':len(values),'validation':checks}))
 if not all(v['matches_train']for v in checks):raise RuntimeError('Base vision differs on held-out records')
if __name__=='__main__':main()
