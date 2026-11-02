"""Held-out moved-hero check of the fixed LOS rule selected on stationary captures."""
import argparse,json
from pathlib import Path
import numpy as np
from fit_visibility import tree_grid,predict

def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--engine',default='runs/gpu/dota-calibration-003/engine.json');p.add_argument('--authored-fow',default='runs/gpu/dota-authored-fow-001');p.add_argument('--output',required=True);a=p.parse_args()
 engine=json.loads(Path(a.engine).read_text());root=Path(engine['calibration']);geometry=np.fromfile(root/engine['geometry'],np.float32).reshape(512,512,2);trees=np.array([[v['x'],v['y']]for v in engine['trees']]);nearest=tree_grid(trees);lookup={tuple(v):i for i,v in enumerate(trees)};fowroot=Path(a.authored_fow);fow=np.fromfile(fowroot/'fow_blockers.f32',np.float32).reshape(512,512);revealers=json.loads((fowroot/'fow.json').read_text())['revealers']
 capture=json.loads(Path(a.capture).read_text());assert capture['complete']and capture['moved_heroes'];rows=[];totals={name:{'grid_points':0,'grid_errors':0,'tree_points':0,'tree_errors':0,'false_visible':0,'false_hidden':0}for name in('legacy_los','authored_los')}
 for r in capture['frames']:
  if r.get('source_errors')or not r.get('sources'):raise RuntimeError('Incomplete native source API capture')
  targets=[[r['x']+(x-3.5)*256,r['y']+(y-3.5)*256]for y in range(8)for x in range(8)]+[[t['x'],t['y']]for t in r['trees']]
  sample={'sources':np.array([[s['x'],s['y'],s['radius']]for s in r['sources']]),'targets':np.array(targets),'tree_ids':np.array([-1]*64+[lookup.get((t['x'],t['y']),-1)for t in r['trees']]),'expected':np.array([v['visible']for v in r['grid']]+[v['visible']for v in r['trees']])}
  result={'team':r['team'],'time':r['time'],'x':r['x'],'y':r['y'],'sources':len(r['sources'])}
  for name in totals:
   authored=name=='authored_los';sample['fow']=fow if authored else None;sample['revealers']=[v for v in revealers if v['team']==r['team']-2]if authored else[]
   got=predict(sample,geometry,trees,nearest,64,64 if authored else 0,0,not authored,64 if authored else 32,0);expected=sample['expected'];delta=got!=expected
   metrics={'grid_points':64,'grid_errors':int(delta[:64].sum()),'tree_points':len(delta)-64,'tree_errors':int(delta[64:].sum()),'false_visible':int((got&~expected).sum()),'false_hidden':int((~got&expected).sum())};result[name]=metrics
   for k,v in metrics.items():totals[name][k]+=v
  rows.append(result)
 report={'capture':a.capture,'fit_capture':'runs/gpu/dota-vision-003/vision.json','selection_on_this_capture':False,'fixed_rule':{'target_cell':64,'tree_radius':64,'ray_samples':64,'authored_fow':a.authored_fow},'frames':len(rows),'totals':totals,'rows':rows,'scope':'Offline environment comparison on actually measured allied source positions/radii. No parameter fitting, policy inference, PPO, or match simulation. Samples are temporally correlated; this is not full-map LOS certification.'};Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'frames':len(rows),'totals':totals},indent=2))
if __name__=='__main__':main()
