"""Installed flagbearer rules checked on disjoint records and native spawn capture."""
import argparse,json,collections,hashlib,math
from pathlib import Path
import numpy as np
from rules import parse

def aura_check(frames,team):
 ends={};samples=0;mismatches=0
 for f in frames:
  if f['team']!=team:continue
  creeps=[c for c in f['creeps']if c['team']==team];flags=[c for c in creeps if 'flagbearer'in c['name']]
  for c in creeps:
   hull=8 if 'ranged'in c['name']else 16
   if any(math.hypot(c['x']-a['x'],c['y']-a['y'])<=700+hull for a in flags):ends[c['id']]=f['time']+.5
   expected=(2 if 'ranged'in c['name']else .5)+(3 if ends.get(c['id'],-math.inf)>f['time']+1e-5 else 0)
   if f['time']>=120:samples+=1;mismatches+=int(abs(expected-c['regen'])>.01)
 return {'team':team,'samples':samples,'boundary_mismatch_frames':mismatches,'fraction':mismatches/samples,'sampling_hz':10,'scope':'Two side views of the same diagnostic game, correlated samples; fixed700+hull radius and.5s linger.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/teacher-pilot-192-cache');p.add_argument('--calibration',default='runs/gpu/dota-calibration-003');p.add_argument('--capture',default='runs/gpu/dota-aggro-001/aggro.json');p.add_argument('--output',required=True);a=p.parse_args()
 rules=Path(a.calibration)/'rules';units=next(iter(parse((rules/'npc_units.txt').read_text(encoding='utf-8-sig')).values()));flag=units['npc_dota_creep_goodguys_flagbearer'];abilities=parse((rules/'npc_abilities.txt').read_text(encoding='utf-8-sig'))['DOTAAbilities'];values=abilities['flagbearer_creep_aura_effect']['AbilityValues']
 root=Path(a.cache);matches=json.loads((root/'dataset.json').read_text())['matches'];report={}
 for split,count in(('train',8),('validation',4)):
  group=[m for m in matches if m['split']==split];mr=[collections.Counter(),collections.Counter()];gold=collections.defaultdict(collections.Counter);sources=[]
  for k in np.linspace(0,len(group)-1,count,dtype=int):
   for team in(2,3):
    file=root/f"match-{group[k]['index']:04d}-team-{team}.npz";sources.append(str(file))
    with np.load(file)as d:
     c=d['creeps'].astype(float);mask=d['creep_valid'].astype(bool)&(c[:,:,45:48].sum(2)<.5);time=d['global_features'][:,0].astype(float)*600
     for b in(0,1):mr[b].update(map(float,np.round(c[:,:,22][mask&((time>=120+330*b)&(time<450+150*b))[:,None]],4)))
     u=d['heroes'][:,0,:48].astype(float);level=np.rint(u[:,23]*15);xp=u[:,25]*2000;lh=u[:,26]*150;g=u[:,24]*2000;low=(mask&(c[:,:,7]*2000<150)).any(1)
     for t in range(1,len(u)):
      if not low[t-1]or level[t]!=level[t-1]:continue
      dx=xp[t-1]-xp[t];dl=round(lh[t]-lh[t-1]);dg=round(g[t]-g[t-1])
      if abs(dx-60)<1.5 and 0<=dg<100:gold['60xp_lh'+str(dl)][dg]+=1
  before=mr[0].most_common(1)[0][0];after=mr[1].most_common(1)[0][0]
  assert abs(before-.4)<.0003 and abs(after-.424)<.0003,'Unexpected flagbearer magic resistance'
  kills=gold['60xp_lh1'];assert sum(kills.values())>=10 and all(77<=v<=91 for v in kills),'Flagbearer kill gold does not support doubled normal bounty plus10'
  report[split]={'sources':sources,'magic_resist_before450':dict(mr[0]),'magic_resist_after450':dict(mr[1]),'gold_deltas_60xp':{k:dict(v)for k,v in gold.items()}}
 capture=json.loads(Path(a.capture).read_text());frames=capture['frames'];first=min(f['time']for f in frames if any('flagbearer'in c['name']for c in f['creeps']))
 assert first==120,'Unexpected initial flagbearer spawn'
 regen=collections.Counter(round(c['regen'],3)for f in frames for c in f['creeps']if 'flagbearer'in c['name'])
 assert set(regen)=={3.5},'Unexpected flagbearer self aura regen'
 aura=[aura_check(frames,team)for team in(2,3)]
 assert all(v['fraction']<.001 for v in aura),'Flagbearer aura boundaries differ'
 result={'spawn_first_seconds':120,'spawn_period_seconds':60,'replace_melee_count':1,'base_magic_resist':float(flag['MagicalResistance'])/100,'magic_resist_growth':'At spawn:1-(1-base)*0.96^min(15,floor(time/450)); old surviving creeps keep old stats','aura_radius':float(values['radius']),'aura_regen':float(values['bonus_health_regen']),'bonus_xp':float(values['bonus_xp']),'xp':float(flag['BountyXP'])+float(values['bonus_xp']),'gold':'Hero last hit:2*normal melee bounty+10. No extra gold observed on60XP without last hit. Whole integer bounty rolls. Later bounty growth still needs direct checks.','native_capture':a.capture,'native_flag_regen_histogram':dict(regen),'source_sha256':{name:hashlib.sha256((rules/name).read_bytes()).hexdigest()for name in('npc_units.txt','npc_abilities.txt')},'verification':report,'limitation':'Cache gold deltas include passive gold and FP16 rounding; two-hero mode does not establish multi-hero gold sharing. Aura radius/linger boundary and death-removal checks remain.'}
 result['aura_linger_seconds']=.5;result['native_aura_boundary_check']=aura
 Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items()if k!='verification'}))
if __name__=='__main__':main()
