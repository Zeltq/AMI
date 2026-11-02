"""Check HP/mana attribute rounding on separate train/validation match samples."""
import argparse,json,collections
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/teacher-pilot-192-cache');p.add_argument('--engine',default='runs/gpu/dota-calibration-003/engine.json');p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.cache);matches=json.loads((root/'dataset.json').read_text())['matches'];e=json.loads(Path(a.engine).read_text());base=e['prototypes'][0];hero=e['hero'];item=e['items'][0];report={}
 for split,count in(('train',4),('validation',2)):
  group=[m for m in matches if m['split']==split];errors=collections.defaultdict(list);sources=[]
  for k in np.linspace(0,len(group)-1,count,dtype=int):
   for team in(2,3):
    file=root/f"match-{group[k]['index']:04d}-team-{team}.npz";sources.append(str(file))
    with np.load(file)as d:
     u=d['heroes'][:,0,:48].astype(np.float64);it=np.rint(d['items'][:,:,0].astype(np.float64)*10);u=u[(it[:,0]==1)&(it[:,2]==0)]
     level=np.rint(u[:,23]*15);strength=item[9]+hero[3]*(level-1);intellect=item[11]+hero[5]*(level-1)
     for name,actual,pred in[
       ('hp_continuous',u[:,8]*2000,base[8]+strength*22),('hp_floor_attribute',u[:,8]*2000,base[8]+(np.floor(hero[0]+strength+1e-5)-hero[0])*22),
       ('mana_continuous',u[:,11]*1000,base[11]+intellect*12),('mana_floor_attribute',u[:,11]*1000,base[11]+(np.floor(hero[2]+intellect+1e-5)-hero[2])*12)]:errors[name].extend(abs(actual-pred))
  report[split]={'sources':sources,'errors':{k:{'samples':len(v),'mean':float(np.mean(v)),'p95':float(np.quantile(v,.95)),'max':float(np.max(v))}for k,v in errors.items()}}
  if report[split]['errors']['hp_floor_attribute']['max']>.5 or report[split]['errors']['mana_floor_attribute']['max']>.25:raise ValueError(f'Native attribute rounding differs in {split}')
 result={'chosen_rule':'HP and mana use floored total strength/intelligence; keep continuous attributes for attack speed,magic resist and regeneration','selection':'Rule chosen on train; validation checks the fixed rule','verification':report,'scope':'Own heroes with one equipped Wraith Band and no equipped Bracer, cached FP16 observations. Sub-unit residuals are cache rounding; does not certify all item/stat/armor rules.'}
 Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({s:r['errors']for s,r in report.items()}))
if __name__=='__main__':main()
