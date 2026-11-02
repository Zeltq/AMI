"""Diagnostic Tango cut: measure passability before and after one real tree."""
import argparse,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.game import Game
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--worker',type=int,default=19);a=p.parse_args()
 run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=False);bots=run/'bots';bots.mkdir()
 shutil.copy2(ROOT/'bots/hero_selection.lua',bots/'hero_selection.lua');shutil.copy2(ROOT/'gpu/bots/tree_cut.lua',bots/'bot_generic.lua')
 (bots/'item_purchase_generic.lua').write_text('function ItemPurchaseThink() end\n')
 (bots/'ability_item_usage_generic.lua').write_text('function AbilityUsageThink() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end\n')
 content=ROOT/f'workers/{a.worker:02d}/game/dota';found={}
 with Game(run,'gpu-tree-cut',4,30,bot_root=bots,content_root=content,game=content.parents[1],game_port=27900+a.worker,engine_threads=2) as game:
  deadline=time.monotonic()+50
  while time.monotonic()<deadline:
   text=game.log.read_text(errors='replace')
   if 'SF_GPU_ERROR|' in text:raise RuntimeError(text.split('SF_GPU_ERROR|')[-1].splitlines()[0])
   for line in text.splitlines():
    for tag in ('BEFORE','AFTER'):
     if 'SF_GPU_TREE_'+tag+'|' in line:found[tag.lower()]=json.loads(line.split('SF_GPU_TREE_'+tag+'|',1)[1])
   if 'after' in found:break
   time.sleep(.05)
  else:raise TimeoutError('Tango cut did not complete')
 found['training_eligible']=False
 found['passability_changed']=sum(a['passable']!=b['passable'] for a,b in zip(found['before']['grid'],found['after']['grid']))
 (run/'tree-cut.json').write_text(json.dumps(found,indent=2));print(json.dumps({'changed_cells':found['passability_changed'],'removed':not found['after']['tree_still_nearby']}),flush=True)
if __name__=='__main__':main()
