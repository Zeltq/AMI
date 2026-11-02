-- Controlled stock-engine aggro, diagnostic only; no policy/training data.
local ok,err=pcall(function()
local json=require('game/dkjson')
local config=require('bots/gpu_server_config')
local function call(u,name,...)
 local f=u[name];if type(f)~='function'then return nil end
 local success,result=pcall(f,u,...);if success then return result end
 return nil
end
local sides={}
for _,h in ipairs(Entities:FindAllByClassname('npc_dota_hero_nevermore'))do sides[h:GetTeamNumber()]=h end
assert(sides[2]and sides[3],'two SF heroes required')
local r,d=sides[2],sides[3]
for _,h in ipairs({r,d})do
 if not h:FindItemInInventory('item_wraith_band')then h:AddItemByName('item_wraith_band')end
 h:SetGold(0,true);h:SetGold(95,false)
 local raze=h:FindAbilityByName('nevermore_shadowraze3');if raze then raze:SetLevel(1)end
end
local function order(u,kind,target,position)
 ExecuteOrderFromTable({UnitIndex=u:entindex(),OrderType=kind,TargetIndex=target and target:entindex()or nil,Position=position})
end
local function stop(u)order(u,DOTA_UNIT_ORDER_STOP)end
local function row(u)
 local p=u:GetAbsOrigin();local a=call(u,'GetAttackTarget');local g=call(u,'GetAggroTarget');local forward=u:GetForwardVector()
 return{entity=u:entindex(),name=u:GetUnitName(),class=u:GetClassname(),lane=call(u,'IsLaneCreep'),team=u:GetTeamNumber(),x=p.x,y=p.y,z=p.z,health=u:GetHealth(),max_health=u:GetMaxHealth(),
  facing=math.atan2(forward.y,forward.x)*180/math.pi,attack_target=a and a:entindex()or nil,aggro_target=g and g:entindex()or nil,
  acquisition=call(u,'GetAcquisitionRange'),range=call(u,'Script_GetAttackRange'),attack_point=call(u,'GetAttackAnimationPoint'),attacks_per_second=call(u,'GetAttacksPerSecond',false),
  cycle=call(u,'GetCycle'),activity=call(u,'GetSequence'),last_attack=call(u,'GetLastAttackTime'),sees_r=call(u,'CanEntityBeSeenByMyTeam',r),visible_to_r=call(r,'CanEntityBeSeenByMyTeam',u)}
end
local cases={}
if config.suite=='priority'then
 cases={{radius=100,bait_offset=200,pulses={},duration=3},{radius=100,bait_offset=80,pulses={},duration=3},{radius=200,bait_offset=80,pulses={},duration=3},{radius=400,bait_offset=200,pulses={},duration=3},{radius=100,bait_offset=400,pulses={},duration=3},{radius=200,bait_offset=250,pulses={},duration=3}}
elseif config.suite=='pull'then
 cases={{radius=400,pulses={1},pull=true},{radius=400,pulses={1,3.4},pull=true},{radius=400,pulses={1,4.2},pull=true}}
else
 local radii=config.kind=='melee'and{100,400,475,490,500,510}or(config.kind=='ranged'and{450,475,490,510,590,610}or{400,450,475,490,510,790,810})
 for _,radius in ipairs(radii)do cases[#cases+1]={radius=radius,pulses={1}}end
end
if config.natural_wave then cases={{radius=100,bait_offset=200,pulses={},duration=3},{radius=100,bait_offset=400,pulses={},duration=3},{radius=200,bait_offset=80,pulses={},duration=3}}end
local which=0;local creep,bait,start;local pulse,last_pulse,moving,moved,stopped_move;local waiting=false
local function setup()
 if creep then UTIL_Remove(creep);UTIL_Remove(bait);creep=nil;bait=nil end
 if which>=#cases then return false end
 if config.natural_wave then
  if GameRules:GetDOTATime(false,true)<.1 then waiting=true;return true end
  local natural_class=config.kind=='siege'and'npc_dota_creep_siege'or'npc_dota_creep_lane'
  local natural_name=config.kind=='siege'and'npc_dota_badguys_siege'or('npc_dota_creep_badguys_'..config.kind)
  for _,u in ipairs(Entities:FindAllByClassname(natural_class))do
   local goal=call(u,'GetInitialGoalEntity');local name=goal and goal:GetName()or''
   if u:IsAlive()and u:GetTeamNumber()==3 and u:GetUnitName()==natural_name and string.find(name,'mid')and not string.match(name,'_1$')then creep=u;break end
  end
  if not creep then waiting=true;return true end
  -- One genuine wave NPC, two heroes and the diagnostic bait only. This
  -- avoids later real waves contaminating the ranged/siege priority cases.
  for _,class in ipairs({'npc_dota_creep_lane','npc_dota_creep_siege'})do
   for _,u in ipairs(Entities:FindAllByClassname(class))do if u:entindex()~=creep:entindex()then UTIL_Remove(u)end end
  end
 end
 waiting=false
 which=which+1;if which>#cases then return false end
 local c=cases[which]
 FindClearSpaceForUnit(r,Vector(80,-816,0),true);FindClearSpaceForUnit(d,(config.suite=='pull'or config.suite=='priority')and Vector(80,-1516,0)or Vector(464,-1000,0),true)
 if config.suite=='pull'then AddFOWViewer(2,d:GetAbsOrigin(),128,30,false)end
 if config.suite=='priority'then for _,team in ipairs({2,3})do AddFOWViewer(team,Vector(80,-816,0),2500,30,false)end end
 r:SetForwardVector(Vector(1,0,0));d:SetForwardVector(Vector(-1,0,0));stop(r);stop(d)
 for _,h in ipairs({r,d})do h:SetHealth(h:GetMaxHealth());h:SetMana(h:GetMaxMana())end
 local name=config.kind=='siege'and'npc_dota_badguys_siege'or('npc_dota_creep_badguys_'..config.kind)
 if config.natural_wave then FindClearSpaceForUnit(creep,Vector(80+c.radius,-816,0),true)
 else creep=CreateUnitByName(name,Vector(80+c.radius,-816,0),true,nil,nil,3)end
 bait=CreateUnitByName('npc_dota_creep_goodguys_melee',Vector(80+c.radius+(c.bait_offset or 80),-816,0),true,nil,nil,2)
 creep:SetForwardVector(Vector(1,0,0));bait:SetForwardVector(Vector(-1,0,0))
 start=GameRules:GetGameTime();pulse=1;last_pulse=-100;moving=false;moved=false;stopped_move=false
 local goal=call(creep,'GetInitialGoalEntity')
 print('SF_GPU_SERVER_CASE|'..json.encode({case=which,kind=config.kind,radius=c.radius,bait_offset=c.bait_offset,pulses=c.pulses,pull=c.pull or false,time=GameRules:GetDOTATime(false,true),actor_mode=config.actor_mode,controlled_target_vision=config.suite=='pull',controlled_all_vision=config.suite=='priority',duration=c.duration or 7,natural_wave=config.natural_wave or false,initial_goal=goal and goal:GetName()or nil}))
 return true
end
assert(setup())
r:SetContextThink('sf_gpu_aggro',function()
 local success,result=pcall(function()
  if waiting then if not setup()then print('SF_GPU_SERVER_DONE|{}');return nil end;return .03 end
  local elapsed=GameRules:GetGameTime()-start;local c=cases[which]
  stop(d)
  if config.actor_mode~='bot'then
  if not moving and elapsed>last_pulse+.034 then stop(r)end
  if c.pull and elapsed>=1.2 and elapsed<3.35 and elapsed>last_pulse+.034 then
   order(r,DOTA_UNIT_ORDER_MOVE_TO_POSITION,nil,Vector(-1200,-816,0));moving=true;moved=true
  elseif c.pull and elapsed>=3.35 and not stopped_move then stop(r);moving=false;stopped_move=true end
  if pulse<=#c.pulses and elapsed>=c.pulses[pulse]then
   order(r,DOTA_UNIT_ORDER_ATTACK_TARGET,d);last_pulse=elapsed
   print('SF_GPU_SERVER_ORDER|'..json.encode({case=which,time=GameRules:GetDOTATime(false,true),game_time=GameRules:GetGameTime(),elapsed=elapsed,action='attack',actor=r:entindex(),target=d:entindex(),creep=row(creep),hero=row(r)}));pulse=pulse+1
  end
  end
  print('SF_GPU_SERVER_AGGRO|'..json.encode({case=which,time=GameRules:GetDOTATime(false,true),game_time=GameRules:GetGameTime(),elapsed=elapsed,units={row(r),row(d),row(creep),row(bait)}}))
  if elapsed>=(c.duration or 7)then
   if not setup()then print('SF_GPU_SERVER_DONE|'..json.encode({cases=#cases}));return nil end
  end
  return .03
 end)
 if not success then print('SF_GPU_SERVER_ERROR|'..tostring(result));return nil end
 return result
end,0)
print('SF_GPU_SERVER_READY')
end)
if not ok then print('SF_GPU_SERVER_ERROR|'..tostring(err))end
