"""Native pregame death, respawn, kill rewards and soul-loss diagnostic."""
import argparse,json,shutil,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.game import Game

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--worker',type=int,default=19);p.add_argument('--killer-team',type=int,choices=[2,3],default=2);p.add_argument('--repeat-kill',action='store_true');p.add_argument('--tower-first',action='store_true');p.add_argument('--raw',action='store_true');a=p.parse_args()
 if a.repeat_kill and a.tower_first:p.error('choose one lifecycle scenario')
 run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=False);bots=run/'bots';bots.mkdir()
 for source,name in[(ROOT/'bots/hero_selection.lua','hero_selection.lua'),(ROOT/'bots/agent.lua','agent.lua'),(ROOT/'gpu/bots/lifecycle.lua','bot_generic.lua')]:shutil.copy2(source,bots/name)
 (bots/'lifecycle_config.lua').write_text('return {killer='+str(a.killer_team)+',repeat_kill='+str(a.repeat_kill).lower()+',tower_first='+str(a.tower_first).lower()+'}\n')
 (bots/'item_purchase_generic.lua').write_text('function ItemPurchaseThink() end\n')
 (bots/'ability_item_usage_generic.lua').write_text('function AbilityUsageThink() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end\n')
 content=ROOT/f'workers/{a.worker:02d}/game/dota';frames=[];offset=0;tail='';complete=False;ended=False;collector=None
 ports=(29000+a.worker*2,29001+a.worker*2)if a.raw else None
 with Game(run,'gpu-lifecycle',4,30,bot_root=bots,content_root=content,game=content.parents[1],game_port=27900+a.worker,engine_threads=2,worldstate_ports=ports,worldstate_frames=1)as game:
  if a.raw:
   with (run/'collector.stdout.log').open('w')as out,(run/'collector.stderr.log').open('w')as err:
    collector=subprocess.Popen([str(ROOT/'tools/nativecollect/nativecollect.exe'),'--run',str(run/'raw'),'--radiant',str(ports[0]),'--dire',str(ports[1])],stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
  for query in('find dota_gold','find dota_hero_respawn','find dota_hero_xp','find bounty'):game.command(query)
  deadline=time.monotonic()+120
  while time.monotonic()<deadline:
   ended=game.process.poll()is not None
   with game.log.open(errors='replace')as f:f.seek(offset);text=tail+f.read();offset=f.tell()
   lines=text.split('\n');tail=lines.pop()
   for line in lines:
    if 'SF_GPU_ERROR|'in line:raise RuntimeError(line)
    if 'SF_GPU_LIFE|'in line:frames.append(json.loads(line.split('SF_GPU_LIFE|',1)[1]))
   if frames:
    latest={side:next((f for f in reversed(frames)if f['team']==side),None)for side in(2,3)};temp=run/'progress.tmp';temp.write_text(json.dumps({'frames':len(frames),'latest':latest,'pid':game.pid},indent=2),encoding='utf-8');temp.replace(run/'progress.json')
    if any(f['done']for f in frames):complete=True;break
   if ended:
    if a.raw and a.repeat_kill:break
    raise RuntimeError('Dota exited during lifecycle capture')
   time.sleep(.1)
  if not complete and not ended:raise TimeoutError('Lifecycle scenario did not reach both deaths and respawns')
 if collector is not None:
  try:collector.wait(timeout=8)
  except subprocess.TimeoutExpired:collector.terminate();collector.wait(timeout=5)
 events=[]
 for team in(2,3):
  own=[f for f in frames if f['team']==team];previous=own[0]
  for f in own[1:]:
   if f['alive']!=previous['alive']:events.append({'team':team,'transition':'respawn'if f['alive']else'death','before':previous,'after':f})
   previous=f
 (run/'lifecycle.json').write_text(json.dumps({'complete':complete,'native_game_ended':ended,'raw_capture':a.raw,'collector_exit_code':collector.returncode if collector is not None else None,'training_eligible':False,'timescale':4,'killer_team':a.killer_team,'repeat_kill':a.repeat_kill,'tower_first':a.tower_first,'frames':frames,'events':events,'scope':'Level1 pregame lifecycle and kill-reward diagnostic; no policy inference; random engine attack rolls; does not establish higher-level bounty formulas.'},indent=2),encoding='utf-8');print(json.dumps({'frames':len(frames),'complete':complete,'events':events}),flush=True)
if __name__=='__main__':main()
