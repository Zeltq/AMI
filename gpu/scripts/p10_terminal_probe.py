"""Dedicated telemetry/cleanup diagnostic; not a model evaluation match."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from evaluate_dota_teacher import prepare_bots
from sf_duel.live.install import isolated_content
from sf_duel.live.game import Game
from play_vs_model import summarize


def main():
    run=Path('C:/AMI/runs/p10/terminal-probe');run.mkdir()
    engine=json.loads(Path('C:/AMI/runs/p07/dota-calibration-005/engine.json').read_text())
    source=run/'bots';prepare_bots(source,0,engine,20261105,'ggd-v4')
    shutil.copy2(ROOT/'bots/teacher_v9_duel_setup.lua',source/'teacher_v9_duel_setup.lua')
    (source/'terminal_probe.lua').write_text('''require('bots/play_model_events').start()
local heroes={}
for _,h in pairs(Entities:FindAllByClassname('npc_dota_hero_nevermore')) do heroes[h:GetTeamNumber()]=h end
assert(heroes[2] and heroes[3])
local ok,why=pcall(function()require('bots/teacher_v9_duel_setup').run(2)end)
assert(not ok,'Dedicated bots must not be mistaken for a human')
print('SF_PLAY_SETUP_REJECTED|'..tostring(why))
heroes[2]:SetContextThink('terminal_probe',function()
 heroes[3]:Kill(nil,heroes[2])
 print('SF_PLAY_FORCED_DEATH')
 return nil
end,2)
''')
    content,_=isolated_content(run/'runtime',source)
    with Game(run,'P10-terminal',1,7.5,bot_root=source,content_root=content,game=content.parents[1],game_port=27305,engine_threads=2) as game:
        game.command('script_reload_code bots/terminal_probe')
        game.wait_log('SF_PLAY_FORCED_DEATH',timeout=15)
        game.wait_log('SF_PLAY_SETUP_REJECTED',timeout=5)
    result=summarize(run/'dota.log',3)
    assert result['first_blood']['team']==3 and result['first_blood']['killer_team']==2,result
    assert result['sides']['3']['deaths']==1 and result['sides']['2']['kills']==1,result
    assert result['sides']['3']['at_seconds']['300'] is None
    result['scope']='Forced server diagnostic, no policy: death attribution and human-only setup rejection'
    (ROOT/'reports/P10/terminal-probe.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
