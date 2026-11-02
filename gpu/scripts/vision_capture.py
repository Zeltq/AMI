"""Capture stationary native SF sight across the pregame/day/night boundaries."""
import argparse,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.game import Game

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--worker',type=int,default=19);p.add_argument('--stop-time',type=float,default=302);p.add_argument('--move',action='store_true');p.add_argument('--static-only',action='store_true');p.add_argument('--static-atlas',action='store_true');a=p.parse_args()
 if a.static_only and a.move:raise ValueError('Static vision capture requires stationary heroes')
 if a.static_atlas and not a.static_only:raise ValueError('Atlas requires measured static-only vision')
 run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=False);bots=run/'bots';bots.mkdir()
 for source,name in [(ROOT/'bots/hero_selection.lua','hero_selection.lua'),(ROOT/'bots/agent.lua','agent.lua'),(ROOT/'gpu/bots/vision.lua','bot_generic.lua')]:shutil.copy2(source,bots/name)
 (bots/'vision_route.lua').write_text('return {move='+('true'if a.move else'false')+',static_atlas='+('true'if a.static_atlas else'false')+'}\n')
 (bots/'item_purchase_generic.lua').write_text('function ItemPurchaseThink() end\n')
 (bots/'ability_item_usage_generic.lua').write_text('function AbilityUsageThink() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end\n')
 if a.static_only:shutil.copy2(ROOT/'gpu/bots/static_vision.lua',bots/'gpu_server.lua')
 content=ROOT/f'workers/{a.worker:02d}/game/dota';frames=[];clocks=[];static=[];atlas=[];offset=0;tail='';complete=False
 with Game(run,'gpu-vision',4,30,bot_root=bots,content_root=content,game=content.parents[1],game_port=27900+a.worker,engine_threads=2)as game:
  if a.static_only:
   game.command('dota_creeps_no_spawning 1')
   game.command('script_reload_code bots/gpu_server')
  deadline=time.monotonic()+240
  while time.monotonic()<deadline:
   if game.process.poll()is not None:raise RuntimeError('Dota exited during vision capture')
   with game.log.open(errors='replace')as f:f.seek(offset);text=tail+f.read();offset=f.tell()
   lines=text.split('\n');tail=lines.pop()
   for line in lines:
    if 'SF_GPU_ERROR|'in line:raise RuntimeError(line)
    for tag,target in [('VISION',frames),('CLOCK',clocks),('STATIC',static),('ATLAS',atlas)]:
     marker=f'SF_GPU_{tag}|'
     if marker in line:target.append(json.loads(line.split(marker,1)[1]))
   latest={side:max((f['time']for f in frames if f['team']==side),default=-1000)for side in (2,3)}
   status={'complete':False,'pid':game.pid,'latest_time_by_team':latest,'clock_transitions':clocks,'frames':len(frames),'training_eligible':False}
   temp=run/'progress.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(run/'progress.json')
   if min(latest.values())>a.stop_time:complete=True;break
   time.sleep(.1)
  if not complete:raise TimeoutError('No both-side vision coverage through night transition')
 if a.static_only:
  if not static or any(u['day']!=0 or u['night']!=0 for item in static for u in item['units']):raise ValueError('Native static-vision intervention was not confirmed')
  for side in (2,3):
   settled=[f for f in frames if f['team']==side and f.get('game_time',f['tick']/30)>static[-1]['game_time']+1]
   if not settled or any(f['vision_api'].get('GetCurrentVisionRange')!=0 for f in settled):raise ValueError('Bot API hero vision is not zero after intervention')
 if a.static_atlas:
  keys=[(r['team'],r['phase'],r['row'])for r in atlas]
  expected={(team,phase,row)for team in(2,3)for phase in('pregame','day','night')for row in range(128)}
  if len(keys)!=len(set(keys))or set(keys)!=expected:raise ValueError('Static atlas has incomplete/duplicate rows')
 result={'complete':True,'training_eligible':False,'diagnostic':True,'timescale':4,'moved_heroes':a.move,'static_only':a.static_only,'static_interventions':static,'frames':frames,'atlas_rows':atlas,'scope':'Empirical visibility and available bot vision APIs; optional static-only intervention zeroes heroes/courier vision and disables creep spawning.'}
 (run/'vision.json').write_text(json.dumps(result,indent=2));print(json.dumps({'frames':len(frames)}),flush=True)
if __name__=='__main__':main()
