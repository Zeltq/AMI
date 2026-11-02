"""Extract public structure positions from a training recording, never validation."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import records
from sf_duel.spec import TOWERS
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/teacher-pilot-192-cache');p.add_argument('--output',required=True);a=p.parse_args()
 dataset=json.loads((Path(a.cache)/'dataset.json').read_text())
 for match in sorted(dataset['matches'],key=lambda m:m['index']):
  if match['split']!='train':continue
  path=str(match['run']).replace('\\','/')
  if sys.platform!='win32' and len(path)>1 and path[1]==':':path='/mnt/'+path[0].lower()+path[2:]
  source=Path(path)/'native-decisions.pb.gz'
  for row in records(source):
   if not row.HasField('observation'):continue
   fountains=[{'team':u.team_id,'x':u.location.x,'y':u.location.y,'z':u.location.z} for u in row.observation.units if u.name=='dota_fountain']
   if {v['team']for v in fountains}!={2,3}:raise RuntimeError('Missing public fountain positions')
   towers=[{'name':u.name,'team':u.team_id,'x':u.location.x,'y':u.location.y,'z':u.location.z} for u in row.observation.units if 'tower' in u.name and u.name not in TOWERS]
   # Fixed ordering is also the simulator's prerequisite chain: bot 1/2/3,
   # top 1/2/3, mid 3, and the two tier-four towers, for each side.
   ordered=[]
   for team,prefix in ((2,'goodguys'),(3,'badguys')):
    for lane in ('bot','top'):
     for tier in (1,2,3):ordered.extend(t for t in towers if t['team']==team and t['name']==f'npc_dota_{prefix}_tower{tier}_{lane}')
    ordered.extend(t for t in towers if t['team']==team and t['name']==f'npc_dota_{prefix}_tower3_mid')
    ordered.extend(sorted((t for t in towers if t['team']==team and t['name']==f'npc_dota_{prefix}_tower4'),key=lambda t:(t['x'],t['y'])))
   if len(ordered)!=18:raise RuntimeError(f'Expected 18 background towers, got {len(ordered)}')
   report={'source':str(source),'split':'train','public_geometry_only':True,'fountains':sorted(fountains,key=lambda v:v['team']),'background_towers':ordered}
   Path(a.output).write_text(json.dumps(report,indent=2));print(a.output);return
  raise RuntimeError(f'No observation in {source}')
 raise RuntimeError('No training match available')
if __name__=='__main__':main()
