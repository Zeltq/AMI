-- Run the real executor with a tiny Bot API fixture.
package.preload['bots/agent']=function() return {infrastructure=function() end} end
package.preload['bots/native_model_observation']=function() return {emit=function() end} end
package.preload['bots/settings']=function() return {interval=4/30} end
package.preload['game/dkjson']=function() return {encode=function() return '{}' end} end
GAME_STATE_PRE_GAME=4;GAME_STATE_GAME_IN_PROGRESS=5
local tick=100
local orders={}
local bot={GetUnitName=function() return 'npc_dota_hero_nevermore' end,GetNearbyTrees=function() return {} end,
 Action_MoveToLocation=function(self,p) orders[#orders+1]=p.x end}
GetBot=function() return bot end
GetTeam=function() return 2 end
GetGameState=function() return 4 end
GameTime=function() return tick/30 end
DotaTime=function() return -80 end
GetItemCost=function() return 50 end
GetItemStockCount=function() return 1 end
Vector=function(x,y,z) return {x=x,y=y,z=z} end
GetBotByHandle=function() return nil end
local commands={allow_late=true}
package.preload['bots/native_commands_2']=function() return commands end
local executor=dofile('bots/native_executor.lua')
local function move(seq,due)
 return {sequence=seq,observed_tick=due-6,execute_tick=due,action=1,x=seq,y=0}
end
commands[106]=move(1,106)
tick=105;executor.Think();assert(#orders==0,'must not execute early')
tick=108;executor.Think();assert(#orders==1 and orders[1]==1,'late command must execute')
tick=109;executor.Think();assert(#orders==1,'must not repeat command')
commands[110]=move(2,110);commands[114]=move(3,114)
tick=115;executor.Think();assert(#orders==2 and orders[2]==3,'newest due command must supersede older one')
commands[118]=move(2,118)
tick=119;executor.Think();assert(#orders==2,'old sequence must not overwrite newer one')
commands[126]=move(4,126)
tick=125;executor.Think();assert(#orders==2,'future command must wait')
tick=126;executor.Think();assert(#orders==3 and orders[3]==4,'on-time command must execute')
commands.allow_late=false;commands[130]=move(5,130)
tick=132;executor.Think();assert(#orders==3,'strict mode must retain original behavior')
print('native executor clock, late, superseding and duplicate tests passed')
-- Handles can resolve to an engine object without the full unit API after the
-- observed target disappears. Reject it, never crash or select another target.
local attacked=0
bot.Action_AttackUnit=function() attacked=attacked+1 end
TEAM_NEUTRAL=4
commands.allow_late=true
local function attack(seq,due)
 return {sequence=seq,observed_tick=due-6,execute_tick=due,action=3,target=123}
end
GetBotByHandle=function() return {IsNull=function() return false end} end
commands[140]=attack(6,140);tick=140;executor.Think()
assert(attacked==0,'incomplete unit API must reject target without crashing')
local target={IsNull=function() return false end,CanBeSeen=function() return true end,IsAlive=function() return true end,GetTeam=function() return 3 end}
GetBotByHandle=function() return target end
commands[144]=attack(7,144);tick=144;executor.Think()
assert(attacked==1,'valid observed target must still be attacked')
target.CanBeSeen=function() return false end
commands[148]=attack(8,148);tick=148;executor.Think()
assert(attacked==1,'invisible target must not be attacked')
print('native executor stale handle and visibility tests passed')
