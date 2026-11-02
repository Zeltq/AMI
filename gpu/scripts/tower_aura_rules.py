"""Check fixed tower aura rules on Radiant calibration and held-out Dire journey."""
import argparse,json,math
from pathlib import Path

def verify(path):
 x=json.loads(Path(path).read_text());assert x['complete'];ends=[-math.inf,-math.inf];pred=[];actual=[];frames=[]
 for f in x['frames']:
  t=f['time']
  for tower in f['towers']:
   if tower['health']>0 and tower['distance']<=900+24:
    rank=0 if 'tower1_'in tower['name']else 1;ends[rank]=t+.5
  bonus=5 if ends[1]>t+1e-5 else 3 if ends[0]>t+1e-5 else 0
  if t<=-83:continue
  assert f['level']==1 and f['max_health']==582,'Journey must preserve level1 and one Wraith Band'
  frames.append(f);pred.append((6.75+bonus,2.35+(3 if bonus==5 else 1 if bonus==3 else 0)));actual.append((f['armor'],f['regen']))
 def transitions(values):
  out=[];last=None
  for f,value in zip(frames,values):
   value=tuple(round(v,2)for v in value)
   if value!=last:out.append({'tick':f['tick'],'time':f['time'],'armor':value[0],'regen':value[1]});last=value
  return out
 p=transitions(pred);n=transitions(actual);passed=len(p)==len(n)and all(abs(a['tick']-b['tick'])<=2 and a['armor']==b['armor']and a['regen']==b['regen']for a,b in zip(p,n))
 mismatches=sum(any(abs(a-b)>.01 for a,b in zip(pv,nv))for pv,nv in zip(pred,actual))
 return {'source':path,'samples':len(frames),'predicted_events':p,'native_events':n,'boundary_mismatch_frames':mismatches,'passed':passed}

def main():
 p=argparse.ArgumentParser();p.add_argument('--train',default='runs/gpu/dota-tower-aura-001/aura.json');p.add_argument('--validation',default='runs/gpu/dota-tower-aura-002-dire/aura.json');p.add_argument('--output',required=True);a=p.parse_args()
 result={'radius':900,'target_hull':24,'linger_seconds':.5,'tier1':{'armor':3,'regen':1},'tier2_plus':{'armor':5,'regen':3},'stacking':'strongest active tier; separate linger timers','selection':'Fixed rules from Radiant journey; Dire journey held out','train':verify(a.train),'validation':verify(a.validation),'tolerance_ticks':2,'scope':'Level1 heroes along two native paths. No destruction,creep-target or aura-break test yet.'};result['passed']=result['train']['passed']and result['validation']['passed']
 Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
 if not result['passed']:raise RuntimeError('Native tower aura rules differ')
if __name__=='__main__':main()
