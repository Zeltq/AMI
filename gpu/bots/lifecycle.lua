-- Diagnostic only: SF kills passive SF, then dies to the Dire mid tower.
local json=require('game/dkjson')
local agent=require('bots/agent')
local config=require('bots/lifecycle_config')
local st={skill_index=1,permanent=0,seq=0}
local last_tick=-1
local first_kill_time=nil
local function souls(bot)
 for i=0,bot:NumModifiers()-1 do if bot:GetModifierName(i)=='modifier_nevermore_necromastery'then return bot:GetModifierStackCount(i)end end
 return 0
end
function Think()
 local ok,err=pcall(function()
  local bot=GetBot()
  if not bot or bot:GetUnitName()~='npc_dota_hero_nevermore'then return end
  local phase=GetGameState()
  if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
  local tick=math.floor(GameTime()*30+.5)
  if tick==last_tick then return end
  last_tick=tick;agent.infrastructure(bot,st)
  local team=GetTeam();local dead=GetHeroDeaths(bot:GetPlayerID());local enemy_pid=GetTeamPlayers(GetOpposingTeam())[1];local enemy_deaths=GetHeroDeaths(enemy_pid)
  if team==config.killer and enemy_deaths>0 and not first_kill_time then first_kill_time=DotaTime()end
  local repeat_kill=config.repeat_kill==true
  local done=team==config.killer and ((repeat_kill and enemy_deaths>=2)or(not repeat_kill and dead>0 and bot:IsAlive()))
  if bot:IsAlive()then
   if config.tower_first and team==config.killer and dead==0 then bot:Action_MoveToLocation(team==TEAM_RADIANT and Vector(524,600,0)or Vector(-1544,-1356,0))
   elseif done then bot:Action_ClearActions(true)
   elseif team~=config.killer and dead>0 and not repeat_kill then bot:Action_ClearActions(true)
   elseif team==config.killer and first_kill_time and not repeat_kill then
    if done then bot:Action_ClearActions(true)
    elseif DotaTime()>=first_kill_time+15 then bot:Action_MoveToLocation(team==TEAM_RADIANT and Vector(524,600,0)or Vector(-1544,-1356,0))
    else bot:Action_ClearActions(true)end
   else
    local goal=team==TEAM_RADIANT and Vector(80,-816,0)or Vector(464,-816,0)
    if GetUnitToLocationDistance(bot,goal)>48 then bot:Action_MoveToLocation(goal)
    elseif team==config.killer then
     local target=nil
     for _,u in ipairs(GetUnitList(UNIT_LIST_ENEMY_HEROES))do if u:GetUnitName()=='npc_dota_hero_nevermore'and u:CanBeSeen()and u:IsAlive()then target=u;break end end
     if target then bot:Action_AttackUnit(target,true)else bot:Action_ClearActions(true)end
    else bot:Action_ClearActions(true)end
   end
  end
  local p=bot:GetLocation();local row={tick=tick,time=DotaTime(),team=team,alive=bot:IsAlive(),deaths=dead,enemy_deaths=enemy_deaths,x=p.x,y=p.y,z=p.z,health=bot:GetHealth(),max_health=bot:GetMaxHealth(),mana=bot:GetMana(),max_mana=bot:GetMaxMana(),gold=bot:GetGold(),level=bot:GetLevel(),xp_needed=bot:GetXPNeededToLevel(),souls=souls(bot),facing=bot:GetFacing(),done=done}
  if type(bot.GetRespawnTime)=='function'then local success,value=pcall(bot.GetRespawnTime,bot);if success then row.respawn_time=value end end
  if type(bot.GetNetWorth)=='function'then local success,value=pcall(bot.GetNetWorth,bot);if success then row.net_worth=value end end
  if type(GetHeroNetWorth)=='function'then local success,value=pcall(GetHeroNetWorth,bot:GetPlayerID());if success then row.hero_net_worth=value end end
  print('SF_GPU_LIFE|'..json.encode(row))
 end)
 if not ok then print('SF_GPU_ERROR|'..tostring(err))end
end
