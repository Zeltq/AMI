"""Independent audit and per-step supervision mask for the native experiment."""
import argparse,collections,gzip,hashlib,json,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.protos.duel_pb2 import Record

def records(path):
 with gzip.open(path,'rb') as f:
  while header:=f.read(4):
   if len(header)!=4:raise ValueError('truncated record header')
   n=struct.unpack('<I',header)[0]
   if not 0<n<32*1024*1024:raise ValueError('invalid record length')
   data=f.read(n)
   if len(data)!=n:raise ValueError('truncated protobuf record')
   yield Record.FromString(data)

def audit(run):
 run=Path(run);result=json.loads((run/'duel-result.json').read_text());path=run/'native-decisions.pb.gz'
 decisions={};executions={};last={};counts=collections.Counter();action_counts=collections.Counter()
 for row in records(path):
  if row.HasField('decision'):
   d=row.decision;s=row.observation;key=(d.team,d.sequence)
   assert key not in decisions
   assert d.team==s.team_id and d.observed_tick==round((s.game_time if result.get('clock')=='game_time' else s.dota_time)*30)
   assert d.execute_tick-d.observed_tick==result['offset_ticks']
   assert d.observed_tick%result['period_ticks']==0
   if row.HasField('supplement'):
    assert row.supplement.team==d.team and row.supplement.tick<=d.observed_tick,'Future auxiliary observation'
   if d.team in last:
    previous=last[d.team];assert previous.sequence+1==d.sequence and previous.observed_tick+result['period_ticks']==d.observed_tick
   last[d.team]=d
   heroes=[u for u in s.units if u.name=='npc_dota_hero_nevermore' and u.team_id==d.team]
   assert len(heroes)==1 and heroes[0].is_alive
   if d.action==3:
    targets=[u for u in s.units if u.handle==d.target]
    assert len(targets)==1 and targets[0].is_alive and targets[0].team_id in (2,3)
    assert targets[0].name=='npc_dota_hero_nevermore' or targets[0].name.startswith('npc_dota_creep_') or targets[0].name.endswith('_siege')
   if d.action==10 and d.item=='item_tango':
    assert any(t.id==d.target for t in row.supplement.trees),'Tango target absent from observation'
   decisions[key]=d;action_counts[str(d.action)]+=1
  if row.HasField('execution'):
   e=row.execution;key=(e.team,e.sequence)
   assert key in decisions and key not in executions
   assert e.tick==decisions[key].execute_tick
   assert e.actual_target==decisions[key].target
   executions[key]=e
 with gzip.open(run/'supervision-index.jsonl.gz','wt',encoding='utf-8') as output:
  for key,d in sorted(decisions.items()):
   e=executions.get(key);execution_time=(d.observed_dota_time+result['offset_ticks']/30) if result.get('clock')=='game_time' else d.execute_tick/30
   before_end=execution_time<result['time']-.001
   valid=bool(before_end and e and e.accepted)
   reason='accepted' if valid else 'episode_ended' if not before_end else 'no_execution_ack' if e is None else e.reason
   counts[reason]+=1
   output.write(json.dumps({'team':d.team,'sequence':d.sequence,'observed_tick':d.observed_tick,'execute_tick':d.execute_tick,'action_loss_mask':valid,'keep_recurrent_observation':True,'reason':reason})+'\n')
 report={'passed':True,'decisions':len(decisions),'executions':len(executions),'action_counts':dict(action_counts),'supervision':dict(counts),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_observation_adapter_validated':False,'teacher_strength_validated':False,'training_started':False}
 (run/'native-audit.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();print(json.dumps(audit(a.run)))
