-- Stock-engine transitions. All writes are explicit diagnostic setup/orders.
local ok,err=pcall(function()
local json=require('game/dkjson')
local cases=require('bots/lane_cases')
for _,case in ipairs(cases)do if case.capture_attack_events or case.capture_damage then
 LinkLuaModifier('modifier_sf_lane_attack_events','bots/lane_attack_events.lua',LUA_MODIFIER_MOTION_NONE)
 break
end end
local heroes={}
for _,h in ipairs(Entities:FindAllByClassname('npc_dota_hero_nevermore'))do heroes[h:GetTeamNumber()]=h end
assert(heroes[2]and heroes[3],'two SF heroes required')
local entities,identity={},{}
local fixed_damage={}
local which=0;local current=nil;local origin=nil;local last_tick=nil;local command=1
local pending=nil;local settle_until=nil;local natural_registered={};local next_creep=0
local function call(u,name,...)
 local fn=u[name];if type(fn)~='function'then return nil end
 local yes,value=pcall(fn,u,...);if yes then return value end
 return nil
end
local function stop(u)ExecuteOrderFromTable({UnitIndex=u:entindex(),OrderType=DOTA_UNIT_ORDER_STOP})end
local function target(u,name)
 local value=call(u,name);if value and not value:IsNull()then return identity[value:entindex()]end
 return nil
end
local function row(id,u)
 local p=u:GetAbsOrigin();local v=u:GetForwardVector();local name=u:GetUnitName()
 local kind=string.find(name,'hero_')and'hero'or string.find(name,'ranged')and'ranged'or string.find(name,'siege')and'siege'or'melee'
 local result={id=id,entity=u:entindex(),kind=kind,team=u:GetTeamNumber(),name=name,
  x=p.x,y=p.y,z=p.z,facing=math.atan2(v.y,v.x)*180/math.pi,alive=u:IsAlive(),
  health=u:GetHealth(),max_health=u:GetMaxHealth(),mana=u:GetMana(),max_mana=u:GetMaxMana(),
  health_regen=call(u,'GetHealthRegen'),mana_regen=call(u,'GetManaRegen'),
  damage=call(u,'GetAverageTrueAttackDamage',nil),attack_point=call(u,'GetAttackAnimationPoint'),
  interval=1/u:GetAttacksPerSecond(false),range=call(u,'Script_GetAttackRange'),
  projectile_speed=call(u,'GetProjectileSpeed'),armor=call(u,'GetPhysicalArmorValue',false),
  magic_resist=call(u,'Script_GetMagicalArmorValue',false,nil),move_speed=call(u,'GetIdealSpeed'),
  level=call(u,'GetLevel'),attack_target=target(u,'GetAttackTarget'),aggro_target=target(u,'GetAggroTarget'),
  out_of_game=call(u,'IsOutOfGame'),class=u:GetClassname(),
  moving=call(u,'IsMoving'),idle=call(u,'IsIdle'),no_draw=call(u,'IsNoDraw'),effects=call(u,'GetEffects'),
  hidden_from_gui=call(u,'IsHiddenFromGUI'),visible_to_r=call(heroes[2],'CanEntityBeSeenByMyTeam',u),visible_to_d=call(heroes[3],'CanEntityBeSeenByMyTeam',u),
  last_attack=call(u,'GetLastAttackTime'),raze_stacks=u:GetModifierStackCount('modifier_nevermore_shadowraze_debuff',nil)}
 if current.capture_attachments then
  result.animation={sequence=call(u,'GetSequence'),cycle=call(u,'GetCycle')}
  result.model_name=call(u,'GetModelName');result.attachments={};result.attachment_ids={}
  for _,attachment in ipairs({'attach_attack1','attach_attack2','attach_hitloc'})do
   local index=call(u,'ScriptLookupAttachment',attachment);result.attachment_ids[attachment]=index or -1
   if index and index>0 then
    local location=call(u,'GetAttachmentOrigin',index)
    if location then result.attachments[attachment]={x=location.x,y=location.y,z=location.z}end
   end
  end
 end
 result.damage_control=fixed_damage[id]
 if current.capture_lane_path and kind~='hero'then
  result.padded_collision_radius=call(u,'GetPaddedCollisionRadius')
  result.remaining_path_length=call(u,'GetRemainingPathLength')
  result.must_reach_each_goal=call(u,'GetMustReachEachGoalEntity')
  local goal=call(u,'GetInitialGoalEntity')
  if goal and not goal:IsNull()then
   local q=goal:GetAbsOrigin();result.lane_goal={name=goal:GetName(),x=q.x,y=q.y,z=q.z}
  end
  local q=call(u,'GetInitialGoalPosition')
  if q then result.lane_goal_position={x=q.x,y=q.y,z=q.z}end
 end
 if current.capture_raze_geometry then result.hull_radius=call(u,'GetHullRadius')end
 if kind=='hero'then
  local player=u:GetPlayerOwnerID();result.last_hits=PlayerResource:GetLastHits(player);result.denies=PlayerResource:GetDenies(player)
  result.abilities={}
  for _,name in ipairs({'nevermore_shadowraze1','nevermore_shadowraze2','nevermore_shadowraze3','nevermore_requiem','nevermore_dark_lord','nevermore_necromastery'})do
   local a=u:FindAbilityByName(name);if a then result.abilities[name]=a:GetLevel()end
  end
  if current.capture_raze_geometry then
   result.raze_geometry={}
   for j=1,3 do
    local a=u:FindAbilityByName('nevermore_shadowraze'..j)
    result.raze_geometry[j]={distance=a:GetSpecialValueFor('shadowraze_range'),radius=a:GetSpecialValueFor('shadowraze_radius')}
   end
  end
 end
 return result
end
local function register_natural()
 local pending_units={}
 for _,class in ipairs({'npc_dota_creep_lane','npc_dota_creep_siege'})do
  for _,u in ipairs(Entities:FindAllByClassname(class))do
   local goal=call(u,'GetInitialGoalEntity');local name=goal and goal:GetName()or''
   if string.find(name,'mid')then
    local p=u:GetAbsOrigin()
    if string.match(name,'_1$')or call(u,'IsOutOfGame')==true then
     pending_units[#pending_units+1]={entity=u:entindex(),name=u:GetUnitName(),goal=name,x=p.x,y=p.y,out_of_game=call(u,'IsOutOfGame')}
    else
     local token=u:entindex()..':'..tostring(call(u,'GetCreationTime'))
     if not natural_registered[token]then
      next_creep=next_creep+1;local id='natural-'..next_creep
      natural_registered[token]=true;entities[id]=u;identity[u:entindex()]=id
     end
    end
   end
  end
 end
 return pending_units
end
local function emit(tick)
 local pending_units={};if current.mode=='natural'then pending_units=register_natural()end
 local units={}
 for id,u in pairs(entities)do if not u:IsNull()then units[#units+1]=row(id,u)end end
 table.sort(units,function(a,b)return a.id<b.id end)
 print('SF_LANE_FRAME|'..json.encode({case=current.id,tick=tick,dota_time=GameRules:GetDOTATime(false,true),game_time=GameRules:GetGameTime(),units=units,pending_pool=pending_units,timescale=current.capture_lane_path and Convars:GetFloat('host_timescale')or nil}))
end
local function setup()
 if current and current.mode=='controlled'then
  for id,u in pairs(entities)do if id~='radiant'and id~='dire'and not u:IsNull()then UTIL_Remove(u)end end
 end
 which=which+1;if which>#cases then print('SF_LANE_DONE|{}');return false end
 current=cases[which];entities={};identity={};fixed_damage={};command=1;origin=nil;last_tick=nil;_G.SF_LANE_ATTACK_CONTEXT=nil
 if current.mode=='controlled'then
  for _,class in ipairs({'npc_dota_creep_lane','npc_dota_creep_siege'})do
   for _,u in ipairs(Entities:FindAllByClassname(class))do UTIL_Remove(u)end
  end
 end
 for _,spec in ipairs(current.units)do
  if spec.kind=='hero'then
   local h=heroes[spec.team]
   if current.mode=='controlled'then
    local old=h
    h=PlayerResource:ReplaceHeroWith(h:GetPlayerOwnerID(),'npc_dota_hero_nevermore',0,0)
    assert(h,'fresh controlled hero required');heroes[spec.team]=h
    if old and not old:IsNull()and old:entindex()~=h:entindex()then UTIL_Remove(old)end
   end
   assert(h:IsAlive(),'living hero required')
   stop(h);FindClearSpaceForUnit(h,Vector(spec.x,spec.y,0),true)
   h:SetForwardVector(Vector(math.cos(math.rad(spec.facing)),math.sin(math.rad(spec.facing)),0))
   h:SetAcquisitionRange(0);h:SetIdleAcquire(false)
   if not h:FindItemInInventory('item_wraith_band')then h:AddItemByName('item_wraith_band')end
   while h:GetLevel()<(spec.level or 1)do h:HeroLevelUp(false)end
   for _,modifier in ipairs({'modifier_nevermore_shadowraze_debuff','modifier_fountain_aura_buff','modifier_nevermore_requiem_fear'})do h:RemoveModifierByName(modifier)end
   for j=1,3 do local a=h:FindAbilityByName('nevermore_shadowraze'..j);a:SetLevel(math.min(4,1+math.floor((h:GetLevel()-1)/2)));a:EndCooldown()end
   -- Match bots/agent.lua infrastructure build, including its passive ranks.
   local level=h:GetLevel();local presence=level>=9 and 4 or level>=8 and 3 or level>=4 and 2 or level>=2 and 1 or 0
   h:FindAbilityByName('nevermore_dark_lord'):SetLevel(presence)
   local ultimate=h:FindAbilityByName('nevermore_requiem');ultimate:SetLevel(level>=6 and 1 or 0);ultimate:EndCooldown()
   h:SetHealth(h:GetMaxHealth());h:SetMana(h:GetMaxMana())
   entities[spec.id]=h;identity[h:entindex()]=spec.id
  end
 end
 -- Reused heroes settle before fresh NPC creation. No old cooldown/projectile
 -- state is imported silently into the GPU fixture.
 settle_until=GameRules:GetGameTime()+2.5;pending=true
 return true
end
local function activate()
 if current.mode=='natural'and GameRules:GetDOTATime(false,true)<-1 then return false end
 for _,spec in ipairs(current.units)do
  if spec.kind~='hero'then
   local name=spec.kind=='siege'and(spec.team==2 and'npc_dota_goodguys_siege'or'npc_dota_badguys_siege')or('npc_dota_creep_'..(spec.team==2 and'goodguys_'or'badguys_')..spec.kind)
   local u=CreateUnitByName(name,Vector(spec.x,spec.y,0),true,nil,nil,spec.team)
   u:SetForwardVector(Vector(math.cos(math.rad(spec.facing)),math.sin(math.rad(spec.facing)),0))
   if spec.health then u:SetHealth(spec.health)end
   if spec.health_fraction then u:SetHealth(math.floor(u:GetMaxHealth()*spec.health_fraction))end
   entities[spec.id]=u;identity[u:entindex()]=spec.id
  end
 end
 if current.damage_mode=='fixed_base_roll'then
  for id,u in pairs(entities)do
   local before=u:GetAverageTrueAttackDamage(nil)
   local value=(u:GetBaseDamageMin()+u:GetBaseDamageMax())/2
   u:SetBaseDamageMin(value);u:SetBaseDamageMax(value)
   -- The getters include primary-attribute damage; the setters write the raw
   -- base component. Measure and remove that added component explicitly.
   local extra=u:GetAverageTrueAttackDamage(nil)-before
   u:SetBaseDamageMin(value-extra);u:SetBaseDamageMax(value-extra)
   local after=u:GetAverageTrueAttackDamage(nil)
   assert(math.abs(after-before)<.001,'fixed roll changed mean damage for '..id)
   assert(math.abs(u:GetBaseDamageMin()-u:GetBaseDamageMax())<.001,'fixed roll retained base variance')
   fixed_damage[id]={mean_before=before,mean_after=after,raw_fixed_base=value-extra,
                     reported_min=u:GetBaseDamageMin(),reported_max=u:GetBaseDamageMax()}
  end
 end
 origin=math.floor(GameRules:GetGameTime()*30+.5);last_tick=origin;pending=false
 if current.capture_attack_events or current.capture_damage then
  _G.SF_LANE_ATTACK_CONTEXT={id=current.id,origin=origin,identity=identity,capture_damage=current.capture_damage}
  for _,h in pairs(current.capture_damage and entities or heroes)do
   assert(h:AddNewModifier(h,nil,'modifier_sf_lane_attack_events',{}),'native event listener unavailable')
  end
 end
 emit(0);return true
end
assert(setup())
local coordinator=Entities:FindByName(nil,'dota_badguys_tower1_mid')
assert(coordinator,'stable map thinker required')
coordinator:SetContextThink('sf_lane_conformance',function()
 local yes,result=pcall(function()
  if pending then if GameRules:GetGameTime()>=settle_until then activate()end;return .03 end
  local tick=math.floor(GameRules:GetGameTime()*30+.5)
  if tick==last_tick then return .03 end
  last_tick=tick;local elapsed=tick-origin
  emit(elapsed)
  if elapsed>=current.duration_ticks then if not setup()then return nil end;return .03 end
  while command<=#current.orders and current.orders[command].tick<=elapsed do
   local c=current.orders[command];local actor=entities[c.actor];local victim=c.target and entities[c.target]or nil
   assert(actor and actor:IsAlive(),'missing/dead command actor')
   local order={UnitIndex=actor:entindex(),Queue=false}
   if c.action==3 then assert(victim and not victim:IsNull(),'missing target');order.OrderType=DOTA_UNIT_ORDER_ATTACK_TARGET;order.TargetIndex=victim:entindex()
   elseif c.action==4 then order.OrderType=DOTA_UNIT_ORDER_STOP
   elseif c.action==1 then order.OrderType=DOTA_UNIT_ORDER_MOVE_TO_POSITION;order.Position=Vector(c.x,c.y,0)
   elseif c.action>=6 and c.action<=8 then order.OrderType=DOTA_UNIT_ORDER_CAST_NO_TARGET;order.AbilityIndex=actor:FindAbilityByName('nevermore_shadowraze'..(c.action-5)):entindex()
   else error('unsupported declared action')end
   ExecuteOrderFromTable(order)
   local executed={case=current.id,tick=elapsed,actor=c.actor,action=c.action,target=c.target,x=c.x,y=c.y}
   print('SF_LANE_ORDER|'..json.encode(executed));command=command+1
  end
  return .03
 end)
 if not yes then print('SF_LANE_ERROR|'..tostring(result));return nil end
 return result
end,0)
end)
if not ok then print('SF_LANE_ERROR|'..tostring(err))end
