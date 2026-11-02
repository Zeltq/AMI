-- Diagnostic journey from the Radiant fountain through both mid tower auras.
local json=require('game/dkjson')
local agent=require('bots/agent')
local config=require('bots/aura_config')
local st={skill_index=1,permanent=0,seq=0}
local last_tick=-1
function Think()
 local ok,err=pcall(function()
  local bot=GetBot()
  if not bot or bot:GetUnitName()~='npc_dota_hero_nevermore'then return end
  local phase=GetGameState()
  if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
  local tick=math.floor(GameTime()*30+.5)
  if tick==last_tick then return end
  last_tick=tick;agent.infrastructure(bot,st)
  if GetTeam()~=config.team then bot:Action_ClearActions(true);return end
  local goal=config.team==TEAM_RADIANT and Vector(-400,-300,0)or Vector(-100,200,0);local arrived=GetUnitToLocationDistance(bot,goal)<32
  if arrived then bot:Action_ClearActions(true)else bot:Action_MoveToLocation(goal)end
  local p=bot:GetLocation();local row={tick=tick,time=DotaTime(),team=GetTeam(),x=p.x,y=p.y,z=p.z,arrived=arrived,level=bot:GetLevel(),health=bot:GetHealth(),max_health=bot:GetMaxHealth(),armor=bot:GetArmor(),regen=bot:GetHealthRegen(),mana_regen=bot:GetManaRegen(),modifiers={},towers={}}
  for i=0,bot:NumModifiers()-1 do row.modifiers[#row.modifiers+1]=bot:GetModifierName(i)end
  for i=0,10 do local tower=GetTower(GetTeam(),i)
   if tower then local q=tower:GetLocation();row.towers[#row.towers+1]={name=tower:GetUnitName(),x=q.x,y=q.y,health=tower:GetHealth(),distance=GetUnitToUnitDistance(bot,tower)}end
  end
  print('SF_GPU_AURA|'..json.encode(row))
 end)
 if not ok then print('SF_GPU_ERROR|'..tostring(err))end
end
