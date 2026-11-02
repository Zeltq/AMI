"""Fit the two SF projectile attachment positions from diagnostic Dota captures."""
import argparse,json,math,statistics
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('references',nargs='+');p.add_argument('--output',required=True);a=p.parse_args();samples=[]
 for path in a.references:
  captured=json.loads(Path(path).read_text())
  if not captured['capture_passed']:raise ValueError('Unvalidated capture')
  for c in captured['scenarios']:
   tick=c['first_projectile_tick']
   if tick is None:continue
   source=next(f for f in c['frames'] if f['tick']==tick and f['team']==2)
   target=next(f for f in c['frames'] if f['tick']==tick and f['team']==3);shot=target['projectiles'][0]
   dx=target['x']-shot['x'];dy=target['y']-shot['y'];length=math.hypot(dx,dy)
   x=shot['x']-40*dx/length-source['x'];y=shot['y']-40*dy/length-source['y'];angle=math.radians(source['facing'])
   samples.append({'source':path,'scenario':c['name'],'forward':x*math.cos(angle)+y*math.sin(angle),'lateral':-x*math.sin(angle)+y*math.cos(angle)})
 report={'training_eligible':False,'split':'calibration','muzzle_forward':statistics.mean(s['forward'] for s in samples),
  'muzzle_side':statistics.mean(abs(s['lateral']) for s in samples),'samples':samples,
  'approximation':'Rewind one 30Hz projectile tick from the first observed tracking location. Alternate mirrored attachment; exact animation phase is not modeled.'}
 Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='samples'}))
if __name__=='__main__':main()
