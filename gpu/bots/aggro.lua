-- Diagnostic only: first-wave targets and short hero-attack aggro pulses.
local json=require('game/dkjson')
local agent=require('bots/agent')
local config=require('bots/aggro_config')
local st={skill_index=1,permanent=0,seq=0}
local last_tick=-1
local ids={}
local next_id=0
local pulses=config.pulses or{20,20.5,21,23.1,26.2}
local pulse=1
local function unit(u)
 if not ids[u]then next_id=next_id+1;ids[u]=next_id end
 local p=u:GetLocation()
 return{id=ids[u],name=u:GetUnitName(),team=u:GetTeam(),x=p.x,y=p.y,z=p.z,health=u:GetHealth(),max_health=u:GetMaxHealth(),facing=u:GetFacing(),range=u:GetAttackRange(),speed=u:GetCurrentMovementSpeed(),regen=u:GetHealthRegen(),magic_resist=u:GetMagicResist()}
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
  local t=DotaTime();local goal=GetTeam()==TEAM_RADIANT and Vector(-600,-400,0)or(config.close and Vector(-320,-320,0)or Vector(500,400,0))
  if t< -5 and GetUnitToLocationDistance(bot,goal)>16 then bot:Action_MoveToLocation(goal)else bot:Action_ClearActions(true)end
  if config.pull and GetTeam()==TEAM_RADIANT and t>=20.4 and t<21.4 then bot:Action_MoveDirectly(goal+Vector(-305,0,0))end
  if GetTeam()==TEAM_RADIANT and pulse<=#pulses and t>=pulses[pulse]then
   local enemy=nil
   for _,u in pairs(GetUnitList(UNIT_LIST_ENEMY_HEROES))do if u:CanBeSeen()then enemy=u;break end end
   if enemy then bot:Action_AttackUnit(enemy,true)end
   print('SF_GPU_AGGRO_ORDER|'..json.encode({tick=tick,time=t,accepted=enemy~=nil,pulse=pulse}))
   pulse=pulse+1
  end
  if t< -5 or tick%3~=0 then return end
  local p=bot:GetLocation();local row={tick=tick,time=t,team=GetTeam(),hero=unit(bot),creeps={},passable=IsLocationPassable(p),passable_zero=IsLocationPassable(Vector(p.x,p.y,0))}
  for _,u in pairs(GetUnitList(UNIT_LIST_ALLIED_CREEPS))do
   if u:IsAlive()and(string.find(u:GetUnitName(),'npc_dota_creep_')or string.find(u:GetUnitName(),'siege'))then
    local c=unit(u);local target=u:GetAttackTarget()
    if target then c.target=unit(target)end
    c.modifiers={};for i=0,u:NumModifiers()-1 do c.modifiers[#c.modifiers+1]=u:GetModifierName(i)end
    row.creeps[#row.creeps+1]=c
   end
  end
  print('SF_GPU_AGGRO|'..json.encode(row))
 end)
 if not ok then print('SF_GPU_ERROR|'..tostring(err))end
end
