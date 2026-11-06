from pathlib import Path
from lupa.lua51 import LuaRuntime

ROOT=Path(__file__).resolve().parents[1]


def test_same_attack_rearms_only_finished_in_range_order():
    lua=LuaRuntime();lua.execute('''
tick=7;attacks=0;commands={};active=3;anim=10;target_present=true;target_x=400;alive=true
GAME_STATE_PRE_GAME=1;GAME_STATE_GAME_IN_PROGRESS=2;BOT_ACTION_TYPE_ATTACK=3;ACTIVITY_ATTACK=10;ACTIVITY_ATTACK2=11
function GetGameState()return 1 end
function GameTime()return tick/30 end
function GetTeam()return 2 end
function print()end
target={IsNull=function()return false end,CanBeSeen=function()return true end,IsAlive=function()return alive end,
 GetTeam=function()return 3 end,GetLocation=function()return {x=target_x,y=0}end,GetBoundingRadius=function()return 24 end}
bot={GetUnitName=function()return 'npc_dota_hero_nevermore'end,IsAlive=function()return true end,
 IsCastingAbility=function()return false end,IsUsingAbility=function()return false end,
 GetAttackTarget=function()return target_present and target or nil end,GetCurrentActionType=function()return active end,
 GetAnimActivity=function()return anim end,GetLocation=function()return {x=0,y=0}end,
 GetAttackRange=function()return 525 end,GetBoundingRadius=function()return 24 end,
 Action_AttackUnit=function()attacks=attacks+1;active=3;target_present=true;anim=10 end}
function GetBot()return bot end
package.preload['bots/agent']=function()return {resolve=function()return target end}end
package.preload['bots/chio_economy']=function()return {}end
package.preload['bots/settings']=function()return {}end
package.preload['game/dkjson']=function()return {encode=function()return '{}'end}end
package.preload['bots/chio_commands_2']=function()return commands end
''')
    executor=lua.execute((ROOT/'bots/chio_executor.lua').read_text())
    def issue(n):
        lua.execute(f'tick={7+6*(n-1)};commands={{{{sequence={n},observed_tick={6*(n-1)},execute_tick={7+6*(n-1)},action=4,target=1}}}}')
        executor.Think()
    issue(1);issue(2);assert lua.globals().attacks==1  # no windup reset
    lua.execute('active=0;target_present=false;anim=10');issue(3);assert lua.globals().attacks==1
    lua.execute('anim=0;target_x=600');issue(4);assert lua.globals().attacks==1  # out of range
    lua.execute('target_x=400');issue(5);assert lua.globals().attacks==2
    issue(6);assert lua.globals().attacks==2  # rearmed order remains uninterrupted
    lua.execute('active=0;target_present=false;anim=0;alive=false');issue(7);assert lua.globals().attacks==2
