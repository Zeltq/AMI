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
