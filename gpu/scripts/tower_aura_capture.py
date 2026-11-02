"""Capture armor/regeneration changes along a native journey past mid towers."""
import argparse,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.game import Game

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--worker',type=int,default=19);p.add_argument('--team',type=int,choices=[2,3],default=2);a=p.parse_args()
 run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=False);bots=run/'bots';bots.mkdir()
 for source,name in[(ROOT/'bots/hero_selection.lua','hero_selection.lua'),(ROOT/'bots/agent.lua','agent.lua'),(ROOT/'gpu/bots/tower_aura.lua','bot_generic.lua')]:shutil.copy2(source,bots/name)
 (bots/'aura_config.lua').write_text('return {team='+str(a.team)+'}\n')
 (bots/'item_purchase_generic.lua').write_text('function ItemPurchaseThink() end\n')
 (bots/'ability_item_usage_generic.lua').write_text('function AbilityUsageThink() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end\n')
 content=ROOT/f'workers/{a.worker:02d}/game/dota';frames=[];offset=0;tail='';complete=False
 with Game(run,'gpu-tower-aura',4,30,bot_root=bots,content_root=content,game=content.parents[1],game_port=27900+a.worker,engine_threads=2)as game:
  deadline=time.monotonic()+120
  while time.monotonic()<deadline:
   if game.process.poll()is not None:raise RuntimeError('Dota exited during tower aura capture')
   with game.log.open(errors='replace')as f:f.seek(offset);text=tail+f.read();offset=f.tell()
   lines=text.split('\n');tail=lines.pop()
   for line in lines:
    if 'SF_GPU_ERROR|'in line:raise RuntimeError(line)
    if 'SF_GPU_AURA|'in line:frames.append(json.loads(line.split('SF_GPU_AURA|',1)[1]))
   if frames:
    temp=run/'progress.tmp';temp.write_text(json.dumps({'frames':len(frames),'latest':frames[-1],'pid':game.pid},indent=2),encoding='utf-8');temp.replace(run/'progress.json')
    if frames[-1]['arrived']and frames[-1]['time']> -83:complete=True;break
   time.sleep(.1)
  if not complete:raise TimeoutError('Native hero did not finish tower-aura route')
 (run/'aura.json').write_text(json.dumps({'complete':True,'training_eligible':False,'timescale':4,'frames':frames,'scope':'Native armor,regen,modifiers and distances to all allied towers during one journey; does not alone certify aura stacking or destruction.'},indent=2),encoding='utf-8');print(json.dumps({'frames':len(frames),'last_time':frames[-1]['time']}),flush=True)
if __name__=='__main__':main()
