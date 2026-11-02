"""Capture exact XP thresholds and unitemized stats at diagnostic spawn levels.

Separate from all training and held-out match recordings. One isolated Dota process.
"""
import argparse,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.game import Game

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--worker',type=int,default=18)
 p.add_argument('--max-level',type=int,default=15);a=p.parse_args()
 root=Path(a.run).resolve();root.mkdir(parents=True,exist_ok=False);bots=root/'bots';bots.mkdir()
 shutil.copy2(ROOT/'bots/hero_selection.lua',bots/'hero_selection.lua')
 shutil.copy2(ROOT/'gpu/bots/levels.lua',bots/'bot_generic.lua')
 (bots/'item_purchase_generic.lua').write_text('function ItemPurchaseThink() end\n')
 (bots/'ability_item_usage_generic.lua').write_text('function AbilityUsageThink() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end\n')
 content=ROOT/f'workers/{a.worker:02d}/game/dota';rows=[]
 with Game(root/'capture','gpu-levels',4,15,bot_root=bots,content_root=content,game=content.parents[1],
            game_port=27900+a.worker,engine_threads=2) as game:
  game.command('find dota_bot')
  for level in range(1,a.max_level+1):
   if level>1:game.command('dota_bot_give_level 1')
   deadline=time.monotonic()+30
   while time.monotonic()<deadline:
    text=game.log.read_text(errors='replace');found={}
    if 'SF_GPU_ERROR|' in text:raise RuntimeError(text.split('SF_GPU_ERROR|')[-1].splitlines()[0])
    for line in text.splitlines():
     if 'SF_GPU_LEVEL|' in line:
      row=json.loads(line.split('SF_GPU_LEVEL|',1)[1])
      if row['level']==level:found[row['team']]=row
    if len(found)==2:break
    time.sleep(.05)
   else:raise TimeoutError(f'No level stats: {level}')
   if any(v['level']!=level for v in found.values()):raise RuntimeError(f'Diagnostic spawn level not applied: {found}')
   if len({v['xp_needed'] for v in found.values()})!=1:raise RuntimeError('Side XP mismatch')
   rows.append(found[2]);print(json.dumps(rows[-1]),flush=True)
   (root/'levels.json').write_text(json.dumps({'training_eligible':False,'diagnostic_cheats':True,'levels':rows,
    'complete':level==a.max_level},indent=2))
if __name__=='__main__':main()
