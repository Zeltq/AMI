local M={}
local timing_probe=(function()
-- Bot-VM getters only. This observer never changes orders or world state.
local json=require('game/dkjson')
local methods={'GetAnimActivity','GetAnimCycle','GetLastAttackTime','GetAttackPoint','GetSecondsPerAttack','GetAttackTarget','GetIncomingTrackingProjectiles','GetAttackProjectileSpeed','GetAttackDamage','GetAttackRange','GetTurnRate','GetBoundingRadius','GetCurrentMovementSpeed','GetFacing','GetHealth','GetMaxHealth','GetHealthRegen','GetArmor'}
local ids,next_id={},0
local function id(u)
 if u==nil then return 0 end
 if not ids[u] then next_id=next_id+1;ids[u]=next_id end
 return ids[u]
end
local function read(u,m)
 if type(u[m])~='function' then return {ok=false,error='unavailable'} end
 local ok,v=pcall(u[m],u)
 if not ok then return {ok=false,error=tostring(v)} end
 if m=='GetAttackTarget' then v=id(v)
 elseif m=='GetIncomingTrackingProjectiles' then
  local shots={}
  for _,s in pairs(v or {}) do
   local p=s.location
   shots[#shots+1]={caster=id(s.caster),x=p and p.x,y=p and p.y,z=p and p.z,velocity=s.velocity,is_attack=s.is_attack}
  end
  v=shots
 end
 return {ok=true,value=v}
end
local M={}
function M.tick(bot)
 local rows={}
 for _,u in pairs(GetUnitList(UNIT_LIST_ALL)) do
  if u and not u:IsNull() and u:CanBeSeen() then
   local name=u:GetUnitName()
   if name=='npc_dota_hero_nevermore' or string.find(name,'creep_',1,true) or (string.find(name,'tower',1,true) and string.find(name,'mid',1,true)) then
    local p=u:GetLocation()
    -- Observe lane units near the two SFs; all visible towers are inexpensive.
    local b=bot:GetLocation()
    if name=='npc_dota_hero_nevermore' or string.find(name,'tower',1,true) or (p.x-b.x)^2+(p.y-b.y)^2<1800^2 then
     local r={id=id(u),name=name,team=u:GetTeam(),own=u==bot,x=p.x,y=p.y,hp=u:GetHealth()}
     for _,m in ipairs(methods) do r[m]=read(u,m) end
     rows[#rows+1]=r
    end
   end
  end
 end
 print('SF_P06_TICK|'..json.encode({team=bot:GetTeam(),time=DotaTime(),game_time=GameTime(),state=GetGameState(),units=rows}))
end
return M

end)()
local settings=require("bots/settings")
for k,v in pairs(require("bots/p03_settings")) do settings[k]=v end
local json=require("game/dkjson")
local original_print=print
function print(message,...)
 if type(message)=="string" and string.find(message,"SF_V9_",1,true)==1 then
  original_print("SF_P03_V9|"..GetTeam().."|"..message)
 else original_print(message,...) end
end
settings.observation_profile="ggd-v3"
local agent=require("bots/agent")
local original_snapshot=agent.snapshot
local first=true
local first_snapshot_time=nil
local creep_decisions=0
local previous={action=0,x=64,y=64,item=0,buy=0}
local core=require("bots/scripted_teacher_v9")
local decide=core.decide
core.decide=function(scene,memory)
 local action,diagnostics=decide(scene,memory)
 previous={action=action.action,x=64,y=64,item=action.item or 0,buy=action.buy or 0}
 if action.action==1 then previous.x=64+(action.x-scene.own.x)/16;previous.y=64+(action.y-scene.own.y)/16 end
 for _,u in ipairs(scene.units) do
  if u.id==action.target_id then previous.target_offset={(u.x-scene.own.x)/2048,(u.y-scene.own.y)/2048};break end
 end
 return action,diagnostics
end
agent.snapshot=function(bot,st)
 local s=original_snapshot(bot,st)
 s.p03_action_type=bot:GetCurrentActionType()
 s.p03_entity_index_available=type(bot.GetEntityIndex)=="function"
 if first_snapshot_time==nil then first_snapshot_time=s.time end
 if settings.capture_start_phases and (s.time-first_snapshot_time<=2 or math.abs(s.time)<=2) then
  original_print("SF_P05_PHASE|"..json.encode({snapshot=s,previous=previous}))
 end
 if first or (#s.creeps>0 and creep_decisions<10) then
  if #s.creeps>0 then creep_decisions=creep_decisions+1 end
  original_print("SF_P03_OBSERVATION|"..json.encode({initial=first,creep_decision=creep_decisions,snapshot=s,previous=previous}))
  first=false
 end
 return s
end
local controller=require("bots/scripted_teacher_v9_live")
local failed=false
local last_tick=nil
local last_second=nil
local last_attack=nil
local last_lh=0
local arrived=false
local dead=false
function M.Think()
 if failed then return end
 local ok,why=pcall(function()
  local bot=GetBot()
  if not bot or bot:GetUnitName()~="npc_dota_hero_nevermore" then return end
  local tick=math.floor(GameTime()*30+.5)
  if tick==last_tick then return end
  last_tick=tick
  timing_probe.tick(bot)
  controller.Think()
  local position=bot:GetLocation()
  if not arrived and bot:IsAlive() and position.x^2+position.y^2<2000^2 and math.abs(position.x-position.y)<1200 then
   arrived=true;original_print("SF_P03_MID|"..json.encode({team=bot:GetTeam(),time=DotaTime()}))
  end
  if not dead and GetHeroDeaths(bot:GetPlayerID())>0 then
   dead=true;original_print("SF_P03_HERO_DEATH|"..json.encode({team=bot:GetTeam(),time=DotaTime()}))
  end
  local second=math.floor(DotaTime())
  if second~=last_second then
   last_second=second
   local p=bot:GetLocation()
   original_print("SF_P03_HERO|"..json.encode({team=bot:GetTeam(),time=DotaTime(),health=bot:GetHealth(),mana=bot:GetMana(),lh=bot:GetLastHits(),deny=bot:GetDenies(),deaths=GetHeroDeaths(bot:GetPlayerID()),x=p.x,y=p.y}))
  end
  local attack=bot:GetLastAttackTime()
  if last_attack and attack>last_attack then
   local target=bot:GetAttackTarget()
   original_print("SF_P03_ATTACK_RELEASE|"..json.encode({team=bot:GetTeam(),time=DotaTime(),last_attack_time=attack,target=target and target:GetUnitName() or "unknown"}))
  end
  last_attack=attack
  if bot:GetLastHits()>last_lh then
   local p=bot:GetLocation();local tower=GetTower(bot:GetTeam(),TOWER_MID_1);local q=tower and tower:GetLocation()
   original_print("SF_P03_LH|"..json.encode({team=bot:GetTeam(),time=DotaTime(),count=bot:GetLastHits()-last_lh,x=p.x,y=p.y,t1_distance=q and math.sqrt((p.x-q.x)^2+(p.y-q.y)^2) or nil}))
  end
  last_lh=bot:GetLastHits()
 end)
 if not ok then failed=true;print("SF_V9_ERROR|"..tostring(why)) end
end

return M
