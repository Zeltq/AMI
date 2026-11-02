"""Check the native vision clock/radii separately on training and held-out records."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import records

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/teacher-pilot-192-cache');p.add_argument('--output',required=True);a=p.parse_args()
 matches=json.loads((Path(a.cache)/'dataset.json').read_text())['matches'];groups={}
 for split in ('train','validation'):
  match=next(m for m in sorted(matches,key=lambda m:m['index'])if m['split']==split and m.get('time',0)>=310);source=Path(match['run'])/'native-decisions.pb.gz';samples={};units={}
  for row in records(source):
   if not row.HasField('decision'):continue
   s=row.observation;t=s.dota_time
   if 0<=t<1 or 299<=t<301:continue
   bucket='pregame'if t<0 else 'first_half'if t<300 else 'second_half'
   if bucket in samples:continue
   samples[bucket]={'dota_time':t,'time_of_day':s.time_of_day,'team':s.team_id}
   for u in s.units:
    if u.team_id==s.team_id and u.is_alive and (u.name=='npc_dota_hero_nevermore'or'tower'in u.name or u.name=='dota_fountain'):
     units[u.name]={'day':u.vision_range_daytime,'night':u.vision_range_nighttime}
   if len(samples)==3:break
  if not samples:raise RuntimeError(f'No vision samples in{source}')
  errors=[abs(v['time_of_day']-(.25+max(0.,v['dota_time'])/600))for v in samples.values()]
  passed=len(samples)==3 and max(errors)<.0002
  groups[split]={'source':str(source),'samples':samples,'unit_radii':units,'fraction_clock_check_passed':passed,'maximum_fraction_error':max(errors)}
 report={'splits':groups,'scope':'Supplied vision radii and continuous clock values. Fractions near thresholds alone do not establish IsDaytime(), particularly in pregame. Direct API capture required. Not a line-of-sight certification.','passed':all(x['fraction_clock_check_passed']for x in groups.values())}
 Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 if not report['passed']:raise RuntimeError('Native clock disagrees with assumed phase or recording lacks coverage')
if __name__=='__main__':main()
