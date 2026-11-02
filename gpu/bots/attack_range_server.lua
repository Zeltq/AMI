-- Stock attack reach/animation diagnostic, no damage/stat overrides.
local ok,err=pcall(function()
local json=require('game/dkjson');local sides={}
for _,h in ipairs(Entities:FindAllByClassname('npc_dota_hero_nevermore'))do sides[h:GetTeamNumber()]=h end
local r,d=sides[2],sides[3];assert(r and d)
for _,h in ipairs({r,d})do if not h:FindItemInInventory('item_wraith_band')then h:AddItemByName('item_wraith_band')end end
local tower=Entities:FindByName(nil,'dota_badguys_tower1_mid');assert(tower)
local function stop(h)ExecuteOrderFromTable({UnitIndex=h:entindex(),OrderType=DOTA_UNIT_ORDER_STOP})end
local cases={}
for _,dist in ipairs({540,550,565,573,580})do cases[#cases+1]={kind='hero',distance=dist}end
for _,dist in ipairs({650,670,690,700,720})do cases[#cases+1]={kind='tower',distance=dist}end
local which=0;local target,start,ordered;local waiting=false
local function setup()
 which=which+1;if which>#cases then return false end
 local c=cases[which];stop(r);stop(d)
 if c.kind=='tower'and GameRules:GetDOTATime(false,true)<.2 then which=which-1;waiting=true;return true end
 waiting=false
 if c.kind=='hero'then
  FindClearSpaceForUnit(r,Vector(80,-816,0),true);FindClearSpaceForUnit(d,Vector(80+c.distance,-816,0),true);target=d
 else
  local p=tower:GetAbsOrigin();FindClearSpaceForUnit(r,Vector(p.x,p.y-c.distance,p.z),true);FindClearSpaceForUnit(d,Vector(80,-1516,0),true);target=tower
 end
 r:SetForwardVector(Vector(1,0,0));if c.kind=='tower'then r:SetForwardVector(Vector(0,1,0))end
 for _,h in ipairs({r,d})do h:SetHealth(h:GetMaxHealth());h:SetMana(h:GetMaxMana());h:RemoveModifierByName('modifier_fountain_aura_buff')end
 start=GameRules:GetGameTime();ordered=false
 print('SF_GPU_SERVER_CASE|'..json.encode({case=which,kind=c.kind,distance=c.distance,target=target:entindex()}));return true
end
local function row(u)
 local p=u:GetAbsOrigin();local t=u:GetAttackTarget();local f=u:GetForwardVector()
 return{entity=u:entindex(),name=u:GetUnitName(),team=u:GetTeamNumber(),x=p.x,y=p.y,z=p.z,health=u:GetHealth(),range=u:Script_GetAttackRange(),hull=u:GetHullRadius(),invulnerable=u:IsInvulnerable(),visible_to_r=r:CanEntityBeSeenByMyTeam(u),aggro_target=u:GetAggroTarget()and u:GetAggroTarget():entindex()or nil,
 attack_target=t and t:entindex()or nil,cycle=u:GetCycle(),activity=u:GetSequence(),last_attack=u:GetLastAttackTime(),facing=math.atan2(f.y,f.x)*180/math.pi}
end
assert(setup())
r:SetContextThink('sf_gpu_attack_range',function()
 local success,result=pcall(function()
  if waiting then stop(r);stop(d);if GameRules:GetDOTATime(false,true)>=.2 then setup()end;return .03 end
  local elapsed=GameRules:GetGameTime()-start;stop(d)
  if elapsed>=.3 and not ordered then ExecuteOrderFromTable({UnitIndex=r:entindex(),OrderType=DOTA_UNIT_ORDER_ATTACK_TARGET,TargetIndex=target:entindex()});ordered=true end
  print('SF_GPU_SERVER_AGGRO|'..json.encode({case=which,time=GameRules:GetDOTATime(false,true),game_time=GameRules:GetGameTime(),elapsed=elapsed,units={row(r),row(d),row(tower)}}))
  if elapsed>=1.5 then if not setup()then print('SF_GPU_SERVER_DONE|{}');return nil end end
  return .03
 end)
 if not success then print('SF_GPU_SERVER_ERROR|'..tostring(result));return nil end
 return result
end,0)
end)
if not ok then print('SF_GPU_SERVER_ERROR|'..tostring(err))end
