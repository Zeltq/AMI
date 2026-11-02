"""Check whether this dedicated Dota runtime exposes server-side scenario control."""
import argparse,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.game import Game

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--worker',type=int,default=19);p.add_argument('--help-filter');a=p.parse_args()
    run=Path(a.run).resolve();run.mkdir(parents=True,exist_ok=False);bots=run/'bots';bots.mkdir()
    shutil.copy2(ROOT/'bots/hero_selection.lua',bots/'hero_selection.lua')
    (bots/'bot_generic.lua').write_text('local once=false\nfunction Think() if once then return end once=true print("SF_GPU_BOT_VM|Entities="..type(Entities).." CreateUnitByName="..type(CreateUnitByName).." ExecuteOrderFromTable="..type(ExecuteOrderFromTable)) end\n')
    (bots/'gpu_probe.lua').write_text('''print("SF_GPU_VSCRIPT_READY")
if Entities then print("SF_GPU_VSCRIPT_ENTITY",Entities:First():GetClassname())end
local ok,err=pcall(function()
 for _,x in ipairs({0,80,880,1280,1330,1400})do
  local gx=GridNav:WorldToGridPosX(x);local gy=GridNav:WorldToGridPosY(-816)
  print("SF_GPU_VSCRIPT_GRID|"..x.."|"..gx.."|"..GridNav:GridPosToWorldCenterX(gx).."|"..gy.."|"..GridNav:GridPosToWorldCenterY(gy))
 end
 local json=require('game/dkjson');local nearby={}
 local function call(e,name)local f=e[name];if type(f)=='function'then local yes,v=pcall(f,e);if yes then return v end end end
 for _,e in ipairs(Entities:FindAllInSphere(Vector(1330,-816,0),450))do
  local p=e:GetAbsOrigin();nearby[#nearby+1]={class=e:GetClassname(),name=e:GetName(),x=p.x,y=p.y,z=p.z,hull=call(e,'GetHullRadius'),solid=call(e,'GetSolid'),collision_group=call(e,'GetCollisionGroup')}
 end
 print('SF_GPU_VSCRIPT_OBSTACLES|'..json.encode(nearby))
end)
if not ok then print("SF_GPU_VSCRIPT_GRID_ERROR|"..tostring(err))end
''')
    (bots/'item_purchase_generic.lua').write_text('function ItemPurchaseThink() end\n')
    (bots/'ability_item_usage_generic.lua').write_text('function AbilityUsageThink() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end\n')
    content=ROOT/f'workers/{a.worker:02d}/game/dota'
    with Game(run,'gpu-vscript-probe',4,30,bot_root=bots,content_root=content,game=content.parents[1],game_port=27900+a.worker,engine_threads=2,developer_mode=bool(a.help_filter))as game:
        for query in ['find script','script print("SF_GPU_VSCRIPT_READY")','script_reload_code bots/gpu_probe','script_reload_code bots/gpu_probe.lua','script_help']:game.command(query)
        if a.help_filter:
            game.command('developer 1');game.command('script_help '+a.help_filter)
        time.sleep(2)
        log=game.log.read_text(errors='replace');markers=[s for s in log.splitlines() if s.startswith(('SF_GPU_VSCRIPT_','SF_GPU_BOT_VM|'))]
    report={'training_eligible':False,'available':any('SF_GPU_VSCRIPT_READY'in s for s in markers),'markers':markers,'pid':game.pid}
    (run/'probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
