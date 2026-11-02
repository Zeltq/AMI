"""Two bounded frozen-policy native prefixes; never report a match win rate."""
import argparse,hashlib,json,os,sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'gpu/scripts')]
from selfplay_cycle import Config,collect,atomic
from sf_duel.selfplay.quality import audit_match
from behavior_report import game_behavior,summarize

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--before',required=True);p.add_argument('--selection',choices=('argmax','sampling'),default='argmax');p.add_argument('--workers',type=int,choices=(1,2,4),default=2);p.add_argument('--seconds',type=int,default=90);p.add_argument('--reset-origin',type=float);a=p.parse_args();os.chdir(ROOT)
 if not 1<=a.seconds<=90:p.error('Departure prefixes must be1..90game seconds')
 run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=False);current=Path(a.checkpoint).resolve();before=Path(a.before).resolve()
 identity={str(path):hashlib.sha256(path.read_bytes()).hexdigest()for path in (current,before)}
 import torch
 from sf_duel.native_policy_server import policy_contract
 torch.set_num_threads(2)
 for path in (current,before):
  if policy_contract(torch.load(path,map_location='cpu',weights_only=False))!=(128,'f16'):raise ValueError('Both native policies must match theGPU128/f16 contract')
 cfg=replace(Config(),workers=a.workers,timescale=1.,seed=20261221,enforce_quality=False).validate()
 atomic(run/'question.json',{'question':'Does theGPU argmax departure/AFK behavior reproduce in realDota under identical frozen weights and memory contracts?','selection_both_actors':a.selection,'memory_both_actors':{'reset':128,'storage':'f16','reset_origin_seconds':a.reset_origin},'matches':a.workers,'map_sides_in_each_match':[2,3],'until_game_seconds':a.seconds,'wall_timeout_each':240 if a.seconds<=40 else 300,'attempts_each':1,'training_updates':0,'weights':identity,'pid':os.getpid()})
 try:
  games=collect(current,run/'collection',cfg,[str(before)],matches=a.workers,until=a.seconds,versus_previous=True,launch_in_batches=True,inference_device='cpu',selection=a.selection,wall_timeout=240 if a.seconds<=40 else 300,attempts=1,reset_origin=a.reset_origin)
  checks=[audit_match(game,prefix_seconds=a.seconds)for game in games]
  if not all(check['passed']for check in checks):raise RuntimeError('Prefix transport audit failed; no behavioral acceptance')
  behaviors=[game_behavior(game)for game in games]
  if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha for path,sha in identity.items()):raise RuntimeError('Frozen weights changed')
  result={'complete':True,'selection':a.selection,'memory_reset_origin_seconds':a.reset_origin,'training_updates':0,'prefix_game_seconds':a.seconds,'scope':'Bounded diagnostic prefixes, not full match evaluation; no strength estimate. Strict original transport limits retained.','games':behaviors,'behavior':summarize(behaviors),'transport':checks}
  atomic(run/'departure.json',result);print(json.dumps({'complete':True,'report':str(run/'departure.json')}),flush=True)
 except BaseException as error:atomic(run/'failed.json',{'complete':False,'error':repr(error),'training_updates':0});raise
if __name__=='__main__':main()
