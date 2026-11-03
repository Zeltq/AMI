"""Compare an explicit GPU initial-input dump with the first native training frames."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.native_data import NativeAdapter,records
from sf_duel.observations import encode
from sf_duel.spec import UNIT_FIELDS,SKILL_FIELDS,ITEM_FIELDS,GLOBAL_FIELDS,COMBAT_UNIT_FIELDS,COMBAT_PROJECTILE_FIELDS,TREE_FIELDS,PROJECTILE_FIELDS

def observation_layout(profile='legacy'):
 from ggd_dataset import layout
 groups,width=layout()
 if profile=='legacy':groups={k:v for k,v in groups.items()if not k.endswith('_combat')};width=37324
 return {k:v[1]for k,v in groups.items()},width

def field_summary(delta,fields,actual=None,expected=None):
 if fields is None:return []
 result=[]
 for i,(name,scale)in enumerate(fields.items()):
  values=delta[...,i];known=delta[...,i+len(fields)]
  if actual is not None:values=np.where((actual[...,i+len(fields)]>.5)&(expected[...,i+len(fields)]>.5),values,0)
  if np.any(values>.01)or np.any(known>.01):
   entry=dict(field=name,scale=scale,above_one_percent=int(np.count_nonzero(values>.01)),elements=int(values.size),
              max_normalized_error=float(values.max(initial=0)),max_field_error=float(values.max(initial=0)*scale),known_mismatches=int(np.count_nonzero(known>.01)))
   if actual is not None and np.any(values>.01):
    index=np.unravel_index(int(np.argmax(values)),values.shape);at=(*index,i)
    entry['example']=dict(row=[int(v)for v in index],native_normalized=float(expected[at]),gpu_normalized=float(actual[at]),native=float(expected[at]*scale),gpu=float(actual[at]*scale))
   result.append(entry)
 return result

def group_fields(key):
 return {'heroes':UNIT_FIELDS,'creeps':UNIT_FIELDS,'towers':UNIT_FIELDS,'trees':TREE_FIELDS,'projectiles':PROJECTILE_FIELDS,
         'skills':SKILL_FIELDS,'items':ITEM_FIELDS,'enemy_items':ITEM_FIELDS,'global_features':GLOBAL_FIELDS,
         'hero_combat':COMBAT_UNIT_FIELDS,'creep_combat':COMBAT_UNIT_FIELDS,'tower_combat':COMBAT_UNIT_FIELDS,'projectile_combat':COMBAT_PROJECTILE_FIELDS}.get(key)

def padded_arrays(arrays,layout):
 out={}
 for key,shape in layout.items():
  value=arrays[key];padded=np.zeros(shape,np.float32)
  if key in ('attack_mask','tree_mask','tp_mask','self_mask'):
   n,t=len(arrays['creeps']),len(arrays['trees']);padded[:2]=value[:2];padded[2:2+n]=value[2:2+n];padded[258:262]=value[2+n:6+n];padded[262:262+t]=value[6+n:6+n+t];padded[-1]=value[-1]
  elif padded.shape==value.shape:padded[:]=value
  else:padded[:len(value)]=value
  out[key]=padded.astype(np.float16).astype(np.float32)
 return out

def align_rows(native,gpu):
 """Match variable visible rows by kind/team and nearest world position."""
 result={}
 for key,valid,extras,base in [('creeps','creep_valid',['creep_combat','creep_ally'],2),('trees','tree_valid',[],262),('projectiles','projectile_valid',['projectile_combat'],None)]:
  ni=np.flatnonzero(native[valid]);gi=np.flatnonzero(gpu[valid]);candidates=[]
  def position(arrays,i):
   p=arrays[key][i,:2].astype(float)
   return p*8192 if key=='creeps' else p*2048+arrays['heroes'][0,:2]*8192
  for i in ni:
   for j in gi:
    if key=='creeps' and (native['creep_ally'][i]!=gpu['creep_ally'][j] or not np.array_equal(native[key][i,45:48],gpu[key][j,45:48])):continue
    distance=float(np.linalg.norm(position(native,i)-position(gpu,j)))
    if distance<=(4 if key=='trees' else 2048):candidates.append((distance,int(i),int(j)))
  used_n=set();used_g=set();pairs=[]
  for distance,i,j in sorted(candidates):
   if i not in used_n and j not in used_g:used_n.add(i);used_g.add(j);pairs.append((i,j))
  for group in [key,valid,*extras]:
   old=native[group];value=np.zeros_like(old)
   for i,j in pairs:value[j]=old[i]
   native[group]=value
  if base is not None:
   for mask in ('attack_mask','tree_mask','tp_mask','self_mask'):
    old=native[mask][base:base+len(native[key])].copy();native[mask][base:base+len(native[key])]=0
    for i,j in pairs:native[mask][base+j]=old[i]
  result[key]=dict(matched=len(pairs),native_count=len(ni),gpu_count=len(gi),native_unmatched=len(ni)-len(pairs),gpu_unmatched=len(gi)-len(pairs))
 return result

def trajectory_comparison(args):
 from ggd_dataset import read_block
 from sf_duel.combat_observations import CombatMemory
 layout,width=observation_layout('ggd-v2');native={2:{},3:{}};memory={2:CombatMemory(),3:CombatMemory()}
 for line in Path(args.lua_log).read_text(errors='replace').splitlines():
  if not line.startswith('SF_P03_OBSERVATION|'):continue
  row=json.loads(line.split('|',1)[1]);snapshot=row['snapshot'];team=snapshot['team'];index=0 if row['initial']else row['creep_decision']
  native[team][index]=(snapshot,encode(snapshot,row.get('previous'),combat_inputs=True,combat_memory=memory[team]).arrays)
 root=Path(args.gpu_dataset);manifest=json.loads((root/'dataset.json').read_text())if(root/'dataset.json').exists()else{'files':[{'file':p.name}for p in sorted(root.glob('*.ggd.gz'))]};gpu={2:{},3:{}};post={2:0,3:0}
 for entry in manifest['files']:
  header,compact,signals=read_block(root/entry['file'])
  for actor_index,actor in enumerate(header['actors']):
   if actor not in (0,1):continue
   team=actor+2
   for step in range(header['horizon']):
    if signals[actor_index,step,11]<.5:continue
    flat=np.zeros(width,np.float32);flat[header['columns']]=compact[actor_index,step];groups={};offset=0
    for key,shape in layout.items():size=int(np.prod(shape));groups[key]=flat[offset:offset+size].reshape(shape);offset+=size
    if 0 not in gpu[team]:gpu[team][0]=groups
    if groups['creep_valid'].any() and post[team]<10:post[team]+=1;gpu[team][post[team]]=groups
  if all(len(gpu[t])==11 for t in (2,3)):break
 pairs=[];fields={}
 for team in (2,3):
  if set(native[team])!=set(range(11))or set(gpu[team])!=set(range(11)):raise RuntimeError(f'Missing initial/ten creep decisions for team {team}')
  for index in range(11):
   snapshot,arrays=native[team][index];expected=padded_arrays(arrays,layout);actual=gpu[team][index];alignment=align_rows(expected,actual);groups={}
   for key in layout:
    delta=np.abs(actual[key]-expected[key]);entries=field_summary(delta,group_fields(key),actual[key],expected[key]);different=int(np.count_nonzero(delta>.01))
    if different:groups[key]=dict(above_one_percent=different,elements=int(delta.size),max_normalized_error=float(delta.max()),fields=entries)
    for entry in entries:
     name=key+'.'+entry['field'];summary=fields.setdefault(name,dict(group=key,field=entry['field'],scale=entry['scale'],pairs_different=0,pairs=22,above_one_percent=0,packed_value_slots=22*int(np.prod(layout[key][:-1])),max_normalized_error=0,known_mismatches=0))
     if entry.get('example')and entry['max_normalized_error']>=summary['max_normalized_error']:summary['example']={**entry['example'],'team':team,'decision':index}
     summary['pairs_different']+=1;summary['above_one_percent']+=entry['above_one_percent'];summary['known_mismatches']+=entry['known_mismatches'];summary['max_normalized_error']=max(summary['max_normalized_error'],entry['max_normalized_error'])
   pairs.append(dict(team=team,decision=index,native_time=snapshot['time'],gpu_time=float(actual['global_features'][0]*600),alignment=alignment,groups=groups))
 report=dict(profile='ggd-v2',width=width,pairs_compared=len(pairs),lua_log=args.lua_log,gpu_dataset=str(root),threshold_normalized=.01,
             fields=sorted(fields.values(),key=lambda r:r['max_normalized_error'],reverse=True),pairs=pairs,
             scope='Independent teacher trajectories: initial and first ten visible-creep decisions, each side. Variable entities matched by kind/team and nearest world position; trees within 4 units. Known-mask differences reported separately. Not a same-state dynamics test.')
 Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(pairs=len(pairs),fields=len(fields),output=args.output)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--dump');p.add_argument('--output',required=True);p.add_argument('--cache',default='runs/teacher-pilot-192-cache');p.add_argument('--native-run');p.add_argument('--decision-index',type=int,default=0);p.add_argument('--checkpoint');p.add_argument('--profile',choices=['legacy','ggd-v2'],default='legacy');p.add_argument('--lua-log');p.add_argument('--gpu-dataset');a=p.parse_args()
 if a.lua_log or a.gpu_dataset:
  if not(a.lua_log and a.gpu_dataset):p.error('--lua-log and --gpu-dataset are required together')
  trajectory_comparison(a);return
 if not a.dump:p.error('--dump is required for native recording comparisons')
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
 layout,width=observation_layout(a.profile)
 raw=np.fromfile(a.dump,np.float32).reshape(2,width);offset=0;actual={}
 for key,shape in layout.items():
  size=int(np.prod(shape));actual[key]=raw[:,offset:offset+size].reshape(2,*shape);offset+=size
 model=None
 if a.checkpoint:
  import torch
  from sf_duel.model import Policy
  torch.set_num_threads(2);model=Policy(combat_inputs=a.profile=='ggd-v2').eval();model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['state_dict'])
 sides={}
 for side,team in enumerate((2,3)):
  row=first[team];snapshot=NativeAdapter().snapshot(row,models[(team,row.decision.observed_tick)]);native=encode(snapshot,combat_inputs=a.profile=='ggd-v2').arrays;groups={};expected_groups={}
  for key,shape in layout.items():
   expected=np.zeros(shape,np.float32);v=native[key]
   if key in ('attack_mask','tree_mask','tp_mask','self_mask'):
    n,t=len(native['creeps']),len(native['trees']);expected[:2]=v[:2];expected[2:2+n]=v[2:2+n];expected[258:262]=v[2+n:6+n];expected[262:262+t]=v[6+n:6+n+t];expected[-1]=v[-1]
   elif expected.shape==v.shape:expected[:]=v
   else:expected[:len(v)]=v
   # Both live paths round observations to FP16 before policy inference.
   expected=expected.astype(np.float16).astype(np.float32);got=actual[key][side].astype(np.float16).astype(np.float32);delta=np.abs(got-expected);expected_groups[key]=expected
   indices=np.argwhere(delta>0.002);examples=[]
   fields=group_fields(key)
   for index in sorted(indices.tolist(),key=lambda i:float(delta[tuple(i)]),reverse=True)[:24]:
    entry={'index':index,'native':float(expected[tuple(index)]),'gpu':float(got[tuple(index)])}
    if fields:
     f=index[-1];names=list(fields);entry['field']=names[f%len(names)];entry['known']=f>=len(names)
    examples.append(entry)
   groups[key]={'above_0_002':len(indices),'above_0_01':int(np.count_nonzero(delta>.01)),'fields_above_one_percent':field_summary(delta,fields,got,expected),'max_absolute_error':float(delta.max()),'examples':examples}
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
