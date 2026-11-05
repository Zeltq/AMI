"""Exercise lobby setup on an isolated dedicated server with all bot slots."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from evaluate_dota_teacher import prepare_bots
from sf_duel.live.install import isolated_content
from sf_duel.live.game import Game


class FullLobby(Game):
    def command(self,text):
        if text.startswith('kickid ') or text.startswith('ent_remove_all '):return
        return super().command(text)


def main():
    run=Path(sys.argv[1]);run.mkdir()
    source=run/'bots'
    engine=json.loads(Path('C:/AMI/runs/p07/dota-calibration-005/engine.json').read_text())
    prepare_bots(source,0,engine,20261105,'ggd-v4')
    shutil.copy2(ROOT/'bots/teacher_v9_duel_setup.lua',source/'teacher_v9_duel_setup.lua')
    (source/'setup_probe.lua').write_text('''local json=require('game/dkjson')
for pid=0,23 do
 if PlayerResource:IsValidPlayerID(pid) then
  local h=PlayerResource:GetSelectedHeroEntity(pid)
  print('SF_SETUP_PLAYER|'..json.encode({pid=pid,name=PlayerResource:GetPlayerName(pid),fake=PlayerResource:IsFakeClient(pid),hero=h and h:GetUnitName(),team=h and h:GetTeamNumber()}))
 end
end
local ok,why=pcall(function()require('bots/teacher_v9_duel_setup').run(2)end)
print('SF_SETUP_PROBE|'..tostring(ok)..'|'..tostring(why))
''')
    if '--fixed' in sys.argv:
        shutil.copy2(ROOT/'bots/play_model_lobby.lua',source/'play_model_lobby.lua')
        (source/'setup_probe.lua').write_text('''local actual=PlayerResource
-- Dedicated servers have no human. Only classify SF player 0 as human for this
-- diagnostic; all entity queries, ResetPlayer and DisconnectClient remain real.
PlayerResource=setmetatable({IsFakeClient=function(_,pid) return pid~=0 and actual:IsFakeClient(pid) end},
 {__index=function(_,key) return function(_,...) return actual[key](actual,...) end end})
local ok,result=pcall(function() return require('bots/play_model_lobby').run(2) end)
PlayerResource=actual
print('SF_SETUP_PROBE|'..tostring(ok)..'|'..(ok and tostring(result.removed) or tostring(result)))
if ok then
 local host=Entities:FindAllByClassname('npc_dota_hero_nevermore')[1]
 host:SetContextThink('check_cleanup',function()
  print('SF_SETUP_HEROES|'..#Entities:FindAllByClassname('npc_dota_hero_nevermore')..'|'..#Entities:FindAllByClassname('npc_dota_hero_lina'))
  return nil
 end,.5)
end
''')
    content,_=isolated_content(run/'runtime',source)
    with FullLobby(run,'P10-setup',1,7.5,bot_root=source,content_root=content,game=content.parents[1],game_port=27305,engine_threads=2) as game:
        game.command('script_reload_code bots/setup_probe')
        game.wait_log('SF_SETUP_PROBE|',timeout=15)
        if '--fixed' in sys.argv:game.wait_log('SF_SETUP_HEROES|',timeout=5)
    for line in (run/'dota.log').read_text(errors='replace').splitlines():
        if line.startswith(('SF_SETUP_PLAYER|','SF_SETUP_PROBE|','SF_SETUP_HEROES|','SF_PLAY_COMPOSITION|')):print(line)
    if '--fixed' in sys.argv:
        log=(run/'dota.log').read_text(errors='replace')
        assert 'SF_SETUP_PROBE|true|8' in log and 'SF_SETUP_HEROES|2|0' in log


if __name__=='__main__':main()
