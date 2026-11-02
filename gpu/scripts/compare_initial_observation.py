"""Compare an explicit GPU initial-input dump with the first native training frames."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import NativeAdapter,records
from sf_duel.observations import encode
from sf_duel.spec import UNIT_FIELDS,SKILL_FIELDS,ITEM_FIELDS,GLOBAL_FIELDS

def main():
 p=argparse.ArgumentParser();p.add_argument('--dump',required=True);p.add_argument('--output',required=True);p.add_argument('--cache',default='runs/teacher-pilot-192-cache');p.add_argument('--native-run');p.add_argument('--decision-index',type=int,default=0);p.add_argument('--checkpoint');a=p.parse_args()
 if a.decision_index<0:p.error('decision-index must be nonnegative')
 if a.native_run:source=Path(a.native_run)/'native-decisions.pb.gz'
 else:
  dataset=json.loads((Path(a.cache)/'dataset.json').read_text());match=next(m for m in sorted(dataset['matches'],key=lambda m:m['index'])if m['split']=='train');source=Path(match['run'])/'native-decisions.pb.gz'
 first={};models={};counts={2:0,3:0}
 for row in records(source):
  if row.HasField('model_supplement'):models[(row.model_supplement.team,row.model_supplement.tick)]=row.model_supplement
  if row.HasField('decision'):
   team=row.decision.team
   if counts[team]==a.decision_index:first[team]=row
   counts[team]+=1
  if len(first)==2 and all((t,r.decision.observed_tick)in models for t,r in first.items()):break
 if len(first)!=2:raise RuntimeError('Missing initial native frames')
 # This order is the public packed layout in gpu/src/observe.rs.
 layout={'heroes':(2,96),'creeps':(256,96),'towers':(4,96),'trees':(256,8),'projectiles':(512,12),
  'skills':(4,12),'items':(8,16),'enemy_items':(8,16),'terrain':(4,8,8),'global_features':(6,),'previous':(30,),
  'hero_valid':(2,),'creep_valid':(256,),'creep_ally':(256,),'tree_valid':(256,),'projectile_valid':(512,),
  'action_mask':(11,),'buy_mask':(7,),'item_mask':(8,),'attack_mask':(519,),'tree_mask':(519,),'tp_mask':(519,),'self_mask':(519,)}
 width=sum(int(np.prod(s))for s in layout.values());assert width==37324,width
 raw=np.fromfile(a.dump,np.float32).reshape(2,width);offset=0;actual={}
 for key,shape in layout.items():
  size=int(np.prod(shape));actual[key]=raw[:,offset:offset+size].reshape(2,*shape);offset+=size
 model=None
 if a.checkpoint:
  import torch
  from sf_duel.model import Policy
  torch.set_num_threads(2);model=Policy().eval();model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['state_dict'])
 sides={}
 for side,team in enumerate((2,3)):
  row=first[team];snapshot=NativeAdapter().snapshot(row,models[(team,row.decision.observed_tick)]);native=encode(snapshot).arrays;groups={};expected_groups={}
  for key,shape in layout.items():
   expected=np.zeros(shape,np.float32);v=native[key]
   if key in ('attack_mask','tree_mask','tp_mask','self_mask'):
    n,t=len(native['creeps']),len(native['trees']);expected[:2]=v[:2];expected[2:2+n]=v[2:2+n];expected[258:262]=v[2+n:6+n];expected[262:262+t]=v[6+n:6+n+t];expected[-1]=v[-1]
   elif expected.shape==v.shape:expected[:]=v
   else:expected[:len(v)]=v
   # Both live paths round observations to FP16 before policy inference.
   expected=expected.astype(np.float16).astype(np.float32);got=actual[key][side].astype(np.float16).astype(np.float32);delta=np.abs(got-expected);expected_groups[key]=expected
   indices=np.argwhere(delta>0.002);examples=[]
   fields={'heroes':UNIT_FIELDS,'creeps':UNIT_FIELDS,'towers':UNIT_FIELDS,'skills':SKILL_FIELDS,'items':ITEM_FIELDS,'enemy_items':ITEM_FIELDS,'global_features':GLOBAL_FIELDS}.get(key)
   for index in sorted(indices.tolist(),key=lambda i:float(delta[tuple(i)]),reverse=True)[:24]:
    entry={'index':index,'native':float(expected[tuple(index)]),'gpu':float(got[tuple(index)])}
    if fields:
     f=index[-1];names=list(fields);entry['field']=names[f%len(names)];entry['known']=f>=len(names)
    examples.append(entry)
   groups[key]={'above_0_002':len(indices),'max_absolute_error':float(delta.max()),'examples':examples}
  native_trees={tuple(np.round(r[:2]*2048,1))for r in native['trees']};gpu_trees={tuple(np.round(r[:2]*2048,1))for r,valid in zip(actual['trees'][side],actual['tree_valid'][side])if valid}
  side_report={'native_time':snapshot['time'],'native_counts':{k:len(native[k])for k in ('creeps','trees','projectiles')},'tree_position_sets':{'common':len(native_trees&gpu_trees),'native_only':len(native_trees-gpu_trees),'gpu_only':len(gpu_trees-native_trees)},'groups':groups}
  if model is not None:
   def probabilities(arrays):
    b={k:torch.from_numpy(v.copy()).unsqueeze(0).bool()if k.endswith('_mask')or k.endswith('_valid')or k=='creep_ally'else torch.from_numpy(v.copy()).unsqueeze(0).float()for k,v in arrays.items()}
    with torch.no_grad():
     h,_,_,_=model(b);logits=model.action_head(h).masked_fill(~b['action_mask'],torch.finfo(torch.float32).min);return logits.softmax(-1)[0].tolist()
   gpu={k:actual[k][side].astype(np.float16).astype(np.float32)for k in layout};side_report['action_probabilities_zero_memory']={'native':probabilities(expected_groups),'gpu':probabilities(gpu)}
   side_report['action_probabilities_zero_memory']['gpu_with_one_native_group']={k:probabilities(dict(gpu,**{k:expected_groups[k]}))for k in ('heroes','towers','trees','tree_valid','terrain','action_mask')}
  sides[str(team)]=side_report
 report={'source':str(source),'split':'diagnostic_recording'if a.native_run else'train','dump':a.dump,'decision_index':a.decision_index,'sides':sides,'scope':'Initial-input diagnostic, not same-state dynamics parity. Native/GPU time and tree ordering may differ; variable rows need identity alignment before interpreting element errors. The optional probabilities use zero memory, not the recorded recurrent history.'}
 Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps({s:{k:v['above_0_002']for k,v in d['groups'].items()if v['above_0_002']}for s,d in sides.items()},indent=2))
if __name__=='__main__':main()
