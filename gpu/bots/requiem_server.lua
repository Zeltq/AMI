-- Controlled stock Requiem, diagnostic only. Only level, souls and positions set.
local ok,err=pcall(function()
local json=require('game/dkjson');local config=require('bots/gpu_server_config');local sides={}
for _,h in ipairs(Entities:FindAllByClassname('npc_dota_hero_nevermore'))do sides[h:GetTeamNumber()]=h end
local team=config.actor_team or 2
local r,d=sides[team],sides[5-team];assert(r and d,'two SF required')
-- Isolate the spell. Keep tower bodies/terrain but prevent unrelated hits
-- after a feared hero enters range, which contaminated heldout006 case6.
for _,tower in ipairs(Entities:FindAllByClassname('npc_dota_tower'))do tower:SetAttackCapability(DOTA_UNIT_CAP_NO_ATTACK)end
for _,h in ipairs({r,d})do
 if not h:FindItemInInventory('item_wraith_band')then h:AddItemByName('item_wraith_band')end
 while h:GetLevel()<6 do h:HeroLevelUp(false)end
end
local ability=r:FindAbilityByName('nevermore_requiem');assert(ability);ability:SetLevel(1)
local necro=r:FindModifierByName('modifier_nevermore_necromastery');assert(necro,'soul modifier missing')
local function stop(h)ExecuteOrderFromTable({UnitIndex=h:entindex(),OrderType=DOTA_UNIT_ORDER_STOP})end
local function call(u,name,...)
 local f=u[name];if type(f)~='function'then return nil end
 local success,result=pcall(f,u,...);if success then return result end
end
local function row(h)
 local p=h:GetAbsOrigin();local mods={}
 for i=0,h:GetModifierCount()-1 do
  local name=h:GetModifierNameByIndex(i)
  if string.find(name,'requiem')or string.find(name,'fear')or string.find(name,'necromastery')then
   local m=h:FindModifierByName(name);mods[#mods+1]={name=name,remaining=m:GetRemainingTime(),stacks=m:GetStackCount()}
  end
 end
 local cursor=call(h,'GetCursorPosition');local goal=call(h,'GetInitialGoalPosition')
 return{entity=h:entindex(),team=h:GetTeamNumber(),x=p.x,y=p.y,z=p.z,health=h:GetHealth(),max_health=h:GetMaxHealth(),mana=h:GetMana(),level=h:GetLevel(),magic_resist=call(h,'Script_GetMagicalArmorValue',nil),modifiers=mods,casting=call(h,'IsCastingAbility'),channeling=call(h,'IsChanneling'),speed=call(h,'GetIdealSpeed'),cursor=cursor and{x=cursor.x,y=cursor.y}or nil,initial_goal=goal and{x=goal.x,y=goal.y}or nil}
end
local cases={{distance=150,souls=2},{distance=400,souls=2},{distance=800,souls=2},{distance=400,souls=8},{distance=400,souls=8,angle=22.5},{distance=1200,souls=2},{distance=1400,souls=2},{distance=1500,souls=2}}
if config.validation_cases then cases={{distance=1100,souls=2},{distance=1300,souls=2},{distance=1350,souls=2},{distance=600,souls=8,angle=-22.5},{distance=600,souls=8,angle=45},{distance=600,souls=8,angle=90}}end
local which=0;local start,ordered,before_probe,after_probe,first_fear
local function probe(phase,elapsed)
 local before=d:GetHealth();ApplyDamage({victim=d,attacker=r,damage=100,damage_type=DAMAGE_TYPE_MAGICAL,ability=ability})
 print('SF_GPU_SERVER_ORDER|'..json.encode({case=which,elapsed=elapsed,action='magic_probe',phase=phase,before=before,after=d:GetHealth(),modifiers=row(d).modifiers}))
end
local function setup()
 which=which+1;if which>#cases then return false end
 local c=cases[which];stop(r);stop(d)
 for _,h in ipairs({r,d})do
  h:RemoveModifierByName('modifier_nevermore_requiem_fear');h:RemoveModifierByName('modifier_nevermore_requiem_slow');h:RemoveModifierByName('modifier_fountain_aura_buff')
  h:SetHealth(h:GetMaxHealth());h:SetMana(h:GetMaxMana())
 end
 FindClearSpaceForUnit(r,Vector(80,-816,0),true)
 local a=(c.angle or 0)*math.pi/180
 FindClearSpaceForUnit(d,Vector(80+c.distance*math.cos(a),-816+c.distance*math.sin(a),0),true)
 r:SetForwardVector(Vector(1,0,0));d:SetForwardVector(Vector(-1,0,0))
 necro:SetStackCount(c.souls);ability:EndCooldown()
 for _,team in ipairs({2,3})do AddFOWViewer(team,Vector(80,-816,0),2500,30,false)end
 start=GameRules:GetGameTime();ordered=false;before_probe=false;after_probe=false;first_fear=nil
 print('SF_GPU_SERVER_CASE|'..json.encode({case=which,distance=c.distance,souls=c.souls,angle=c.angle or 0,time=GameRules:GetDOTATime(false,true)}));return true
end
assert(setup())
r:SetContextThink('sf_gpu_requiem',function()
 local success,result=pcall(function()
  local elapsed=GameRules:GetGameTime()-start
  -- A player/bot cannot issue orders during fear. Server orders may bypass
  -- that restriction; never inject Stop into the forced movement being tested.
  if not d:HasModifier('modifier_nevermore_requiem_fear')then stop(d)end
  if config.probe_magic and elapsed>=.15 and not before_probe then probe('before',elapsed);before_probe=true;d:SetHealth(d:GetMaxHealth())end
  if elapsed>=.3 and not ordered then
   ExecuteOrderFromTable({UnitIndex=r:entindex(),OrderType=DOTA_UNIT_ORDER_CAST_NO_TARGET,AbilityIndex=ability:entindex()});ordered=true
   print('SF_GPU_SERVER_ORDER|'..json.encode({case=which,elapsed=elapsed,time=GameRules:GetDOTATime(false,true),action='requiem'}))
  end
  if d:HasModifier('modifier_nevermore_requiem_fear')then
   if not first_fear then first_fear=elapsed end
   if config.probe_magic and elapsed>=first_fear+.1 and not after_probe then probe('during',elapsed);after_probe=true end
  end
  print('SF_GPU_SERVER_AGGRO|'..json.encode({case=which,time=GameRules:GetDOTATime(false,true),game_time=GameRules:GetGameTime(),elapsed=elapsed,units={row(r),row(d)}}))
  assert(r:IsAlive()and d:IsAlive(),'fixture unexpectedly killed a hero')
  if elapsed>=7 then if not setup()then print('SF_GPU_SERVER_DONE|{}');return nil end end
  return .03
 end)
 if not success then print('SF_GPU_SERVER_ERROR|'..tostring(result));return nil end
 return result
end,0)
end)
if not ok then print('SF_GPU_SERVER_ERROR|'..tostring(err))end
