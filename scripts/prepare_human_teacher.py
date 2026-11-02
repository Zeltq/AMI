"""Prepare an isolated graphical Dota client for a human versus teacher-v9 duel."""
import argparse,hashlib,json,shutil,subprocess,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sf_duel.live.install import isolated_content
from sf_duel.live.game import DEFAULT_GAME
from sf_duel.live.bridge import literal

def prepare(output,human_team=2,game=DEFAULT_GAME,lobby=False):
    output=Path(output).resolve()
    if human_team not in (2,3):raise ValueError('Human team must be 2 or 3')
    if output.exists():raise FileExistsError('Choose a fresh prepared match directory')
    source=output/'bots';source.mkdir(parents=True)
    for name in ('agent.lua','scripted_teacher_v9.lua','scripted_teacher_v9_live.lua','teacher_v9_duel_setup.lua'):shutil.copy2(ROOT/'bots'/name,source/name)
    engine=json.loads((ROOT/'runs/gpu/dota-calibration-003/engine.json').read_text())
    teacher_team=5-human_team
    settings=dict(endpoint='',interval=4/30,match=uuid.uuid4().hex,controller_type='teacher-v9',human_team=human_team,teacher_team=teacher_team,
        teacher_combat=engine['combat'],teacher_prototypes={0:engine['prototypes'][0][16],1:engine['prototypes'][2][16],2:engine['prototypes'][6][16],3:engine['prototypes'][7][16],4:engine['prototypes'][8][16]},muzzle_forward=engine['hero'][24],muzzle_side=engine['hero'][25])
    if lobby:
        settings['native_lobby']=True
    (source/'settings.lua').write_text('return '+literal(settings)+'\n')
    (source/'human_ready.lua').write_text('return '+('true' if lobby else 'false')+'\n')
    callback='''local controller=require("bots/scripted_teacher_v9_live")
local failed=false
function Think()
 if failed then return end
 local ok,why=pcall(controller.Think)
 if not ok then failed=true;print("SF_V9_ERROR|"..tostring(why)) end
end
'''
    (source/'bot_generic.lua').write_text(callback)
    (source/'item_purchase_generic.lua').write_text(callback.replace('function Think()','function ItemPurchaseThink()'))
    (source/'ability_item_usage_generic.lua').write_text('function AbilityUsageThink() end\nfunction AbilityLevelUpThink() end\nfunction CourierUsageThink() end\nfunction ItemUsageThink() end\nfunction BuybackUsageThink() end\n')
    (source/'team_desires.lua').write_text('-- Full control is supplied by bot_generic.lua.\n')
    (source/'hero_selection.lua').write_text('''local settings=require("bots/settings")
local selected={}
function GetBotNames()
 local names={};for i=1,5 do names[i]="SF_DUEL_"..GetTeam().."_"..i end;return names
end
function Think()
 for _,team in ipairs({GetTeam()}) do
  local first=true
  for _,pid in ipairs(GetTeamPlayers(team)) do
   if IsPlayerBot(pid) then
    local sf=(settings.teacher_team==nil or team==settings.teacher_team) and first;first=false
    if not selected[pid] then
     local hero=sf and "npc_dota_hero_nevermore" or "npc_dota_hero_lina"
     SelectHero(pid,hero);selected[pid]=true;print("SF_V9_PICK|"..team.."|"..pid.."|"..hero)
    end
   end
  end
 end
end
function UpdateLaneAssignments()
 local lanes={};for _,pid in ipairs(GetTeamPlayers(GetTeam())) do lanes[pid]=LANE_MID end;return lanes
end
''')
    if lobby:
        (output/'cfg').mkdir();(output/'cfg/ami_teacher_1v1.cfg').write_text("sv_cheats 1\nscript require('bots/teacher_v9_duel_setup').run("+str(human_team)+")\n")
        config=dict(policy='scripted teacher v9',bots=str(source),game=str(Path(game).resolve()),human_team=human_team,teacher_team=teacher_team,lobby_script='Local Dev Script',server='Local Host',fill_empty_slots=True,bot_count_after_setup=1,duel_setup='exec ami_teacher_1v1',gpu_teacher_sha256=hashlib.sha256((ROOT/'gpu/src/scripted_teacher.rs').read_bytes()).hexdigest())
        (output/'lobby.json').write_text(json.dumps(config,indent=2)+'\n')
        (output/'source-manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.lua')},indent=2)+'\n')
        for name,option in [('Install.cmd',''),('Restore.cmd',' --restore')]:
            (output/name).write_text('@echo off\n"'+sys.executable+'" "'+str(ROOT/'scripts/install_teacher_lobby.py')+'" "'+str(output)+'"'+option+'\nif errorlevel 1 pause\n')
        (output/'Open-Dota.cmd').write_text('@echo off\n"'+sys.executable+'" "'+str(ROOT/'scripts/install_teacher_lobby.py')+'" "'+str(output)+'" --launch\nif errorlevel 1 pause\n')
        return output
    content,runtime=isolated_content(output/'runtime',source,game)
    link=content/'scripts/vscripts/bots'
    def quote(s):return "'"+str(s).replace("'","''")+"'"
    subprocess.run(['powershell','-NoProfile','-Command',f'New-Item -ItemType Junction -Path {quote(link)} -Target {quote(runtime)} | Out-Null'],check=True,capture_output=True)
    # A console log must be private, even if Steam's installation has an old one.
    console=content/'console.log'
    if console.exists():console.unlink()
    password=uuid.uuid4().hex
    cfg='\n'.join(['sv_lan 1','sv_cheats 1',f'rcon_password {password}','fps_max 60','dota_bot_practice_script 0','dota_bot_practice_start 1',f'dota_bot_practice_team {human_team}','dota_bot_practice_gamemode 21','dota_bot_set_difficulty 3','map start gamemode 21',''])
    (content/'cfg/ami_teacher_v9.cfg').write_text(cfg)
    config=dict(human_team=human_team,teacher_team=teacher_team,executable=str(output/'runtime/game/bin/win64/dota2.exe'),content=str(content),bots=str(runtime),console_log=str(console),port=27295,rcon_password=password,period_ticks=4,offset_ticks=6,effective_execution_ticks=7,game_seconds=600,policy='scripted teacher v9',gpu_teacher_sha256=hashlib.sha256((ROOT/'gpu/src/scripted_teacher.rs').read_bytes()).hexdigest(),live_tested=False)
    (output/'match.json').write_text(json.dumps(config,indent=2)+'\n')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime.glob('*.lua')}
    (output/'source-manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (output/'Play.ps1').write_text('$ErrorActionPreference="Stop"\n& "'+sys.executable+'" "'+str(ROOT/'scripts/play_human_teacher.py')+'" "'+str(output)+'"\nif ($LASTEXITCODE -ne 0) { throw "Teacher match setup failed; inspect the session logs." }\n')
    (output/'Play.cmd').write_text('@echo off\n"'+sys.executable+'" "'+str(ROOT/'scripts/play_human_teacher.py')+'" "'+str(output)+'"\nif errorlevel 1 pause\n')
    (output/'Play-manual.cmd').write_text('@echo off\n"'+sys.executable+'" "'+str(ROOT/'scripts/play_human_teacher.py')+'" "'+str(output)+'" --manual\nif errorlevel 1 pause\n')
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--human-team',type=int,default=2);p.add_argument('--lobby',action='store_true');a=p.parse_args();print(prepare(a.output,a.human_team,lobby=a.lobby))
