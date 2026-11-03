-- Natural attack orders. Count releases and HP-loss events separately.
local ok,err=pcall(function()
 local json=require('game/dkjson');local heroes={}
 for _,h in ipairs(Entities:FindAllByClassname('npc_dota_hero_nevermore')) do heroes[h:GetTeamNumber()]=h end
 local host=heroes[2];assert(host and heroes[3])
 local function stop(u) ExecuteOrderFromTable({UnitIndex=u:entindex(),OrderType=DOTA_UNIT_ORDER_STOP}) end
 local lows,highs={},{}
 for x=-800,800,32 do for y=-1000,600,32 do
  local p=GetGroundPosition(Vector(x,y,0),host)
  if GridNav:IsTraversable(p) and not GridNav:IsBlocked(p) then
   if math.abs(p.z)<.1 then lows[#lows+1]=p elseif math.abs(p.z-128)<.1 then highs[#highs+1]=p end
  end
 end end
 local distance=1e9;local low,high
 for _,p in ipairs(lows) do for _,q in ipairs(highs) do
  local d=(p.x-q.x)^2+(p.y-q.y)^2
  if d<distance then low=p;high=q;distance=d end
 end end
 assert(low and high and distance<250^2,'no close planar height boundary')
 local same;distance=1e9
 for _,p in ipairs(lows) do local d=(p.x-low.x)^2+(p.y-low.y)^2;if d>=96^2 and d<distance then same=p;distance=d end end
 print('SF_P07_TERRAIN|'..json.encode({low={low.x,low.y,low.z},high={high.x,high.y,high.z},same={same.x,same.y,same.z}}))
 local tower,enemy_tower
 for _,u in ipairs(Entities:FindAllByClassname('npc_dota_tower')) do
  if not tower and u:GetTeamNumber()==2 then tower=u end
  if u:GetUnitName()=='npc_dota_badguys_tower1_mid' then enemy_tower=u end
  u:SetAttackCapability(DOTA_UNIT_CAP_NO_ATTACK)
  if not (_G.SF_P07_TOWER_TARGET and u:GetUnitName()=='npc_dota_badguys_tower1_mid') then u:AddNewModifier(u,nil,'modifier_invulnerable',{}) end
 end
 local melee=CreateUnitByName('npc_dota_creep_goodguys_melee',low,false,nil,nil,2)
 local ranged=CreateUnitByName('npc_dota_creep_goodguys_ranged',low,false,nil,nil,2)
 local target=_G.SF_P07_TOWER_TARGET and enemy_tower or CreateUnitByName('npc_dota_creep_badguys_melee',high,false,nil,nil,3)
 assert(target,'missing target')
 print('SF_P07_FIXTURE|'..json.encode({creep_class=melee:GetClassname(),target=target:GetUnitName(),invulnerable=target:IsInvulnerable(),attack_immune=target:IsAttackImmune()}))
 target:SetAttackCapability(DOTA_UNIT_CAP_NO_ATTACK);target:SetBaseHealthRegen(0);target:SetMoveCapability(DOTA_UNIT_CAP_MOVE_NONE)
 local participants={host,heroes[3],melee,ranged,tower,target};local retained={}
 for _,u in ipairs(participants) do retained[u:entindex()]=true;stop(u);u:SetAttackCapability(DOTA_UNIT_CAP_NO_ATTACK);u:SetAbsOrigin(Vector(-3500,-3500,256));if u~=target then u:AddNewModifier(u,nil,'modifier_invulnerable',{}) end end
 ListenToGameEvent('entity_killed',function(e) if retained[e.entindex_killed] then print('SF_P07_FIXTURE_DEATH|'..json.encode(e)) end end,nil)
 local cases={}
 for _,pair in ipairs({{'sf',host},{'melee',melee},{'ranged',ranged},{'tower',tower}}) do
  for _,mode in ipairs({'same','uphill'}) do cases[#cases+1]={kind=pair[1],source=pair[2],mode=mode} end
 end
 if _G.SF_P07_NONHERO then table.remove(cases,1);table.remove(cases,1) end
 if _G.SF_P07_TOWER_TARGET then cases={{kind='sf_to_tower',source=host,mode='same'},{kind='sf_to_tower',source=host,mode='uphill'}} end
 if _G.SF_P07_KIND then local selected={};for _,c in ipairs(cases) do if c.kind==_G.SF_P07_KIND then selected[#selected+1]=c end end;cases=selected end
 if _G.SF_P07_IMPACT then
  same=low
  for _,p in ipairs(lows) do local d=(p.x-high.x)^2+(p.y-high.y)^2
   if d>450^2 and d<500^2 and (p.x-same.x)^2+(p.y-same.y)^2>350^2 then low=p;break end
  end
  assert((low.x-high.x)^2+(low.y-high.y)^2>450^2,'impact fixture requires flight time')
  cases={{kind='sf',source=host,mode='up_to_same'},{kind='sf',source=host,mode='same_to_up'}}
 end
 local which=0;local last,attacks,hits,finish,before,case_start,next_attack
 ListenToGameEvent('entity_hurt',function(e)
  local c=cases[which]
  if c and e.entindex_killed==target:entindex() then
   local attributed=e.entindex_attacker==c.source:entindex()
   if attributed then hits=hits+1 end
   print('SF_P07_HURT|'..json.encode({case=which,time=GameRules:GetGameTime(),attributed=attributed,event=e}))
  end
 end,nil)
 local function setup()
  which=which+1;local c=cases[which];if not c then return false end
  for _,u in ipairs(participants) do stop(u);u:SetAttackCapability(DOTA_UNIT_CAP_NO_ATTACK);u:SetAbsOrigin(Vector(-3500,-3500,256)) end
  c.source:SetAbsOrigin(low);target:SetAbsOrigin((c.mode=='uphill' or c.mode=='up_to_same') and high or same)
  AddFOWViewer(2,low,1500,2,false);AddFOWViewer(3,low,1500,2,false)
  c.source:SetMoveCapability(DOTA_UNIT_CAP_MOVE_NONE)
  c.source:SetBaseAttackTime(.8)
  c.source:SetAttackCapability(c.kind=='melee' and DOTA_UNIT_CAP_MELEE_ATTACK or DOTA_UNIT_CAP_RANGED_ATTACK)
  target:SetHealth(target:GetMaxHealth());before=target:GetHealth();last=c.source:GetLastAttackTime();attacks=0;hits=0;finish=nil;next_attack=nil;case_start=GameRules:GetGameTime()
  ExecuteOrderFromTable({UnitIndex=c.source:entindex(),OrderType=DOTA_UNIT_ORDER_ATTACK_TARGET,TargetIndex=target:entindex(),Queue=false})
  return true
 end
 assert(setup())
 host:SetContextThink('p07_natural_miss',function()
  local success,result=pcall(function()
   local c=cases[which];local now=GameRules:GetGameTime()
   for _,u in ipairs(Entities:FindAllByClassname(melee:GetClassname())) do if not retained[u:entindex()] then UTIL_Remove(u) end end
   AddFOWViewer(2,low,1500,.2,false);AddFOWViewer(3,low,1500,.2,false)
   if attacks==0 and now-case_start<.3 then ExecuteOrderFromTable({UnitIndex=c.source:entindex(),OrderType=DOTA_UNIT_ORDER_ATTACK_TARGET,TargetIndex=target:entindex(),Queue=false}) end
   if attacks==0 and now-case_start>2 and now-case_start<2.04 then
    local p=c.source:GetAbsOrigin();local q=target:GetAbsOrigin();local t=c.source:GetAttackTarget()
    print('SF_P07_DEBUG|'..json.encode({hp=target:GetHealth(),source={p.x,p.y,p.z},target={q.x,q.y,q.z},range=c.source:Script_GetAttackRange(),attack_target=t and t:entindex(),last=c.source:GetLastAttackTime()}))
   end
   local hp=target:GetHealth()
   if hp<before then print('SF_P07_HIT|'..json.encode({case=which,time=now,damage=before-hp})) end
   target:SetHealth(target:GetMaxHealth());before=target:GetHealth()
   local release=c.source:GetLastAttackTime()
   if release~=last then
    last=release;attacks=attacks+1
    print('SF_P07_RELEASE|'..json.encode({case=which,kind=c.kind,mode=c.mode,attack=attacks,time=now,source_z=c.source:GetAbsOrigin().z,target_z=target:GetAbsOrigin().z}))
    if _G.SF_P07_IMPACT then
     stop(c.source);target:SetAbsOrigin(c.mode=='up_to_same' and same or high);next_attack=now+.8
    end
    if attacks>=240 then stop(c.source);finish=now end
   end
   if next_attack and now>=next_attack and not finish then
    next_attack=nil;target:SetAbsOrigin(c.mode=='up_to_same' and high or same)
    ExecuteOrderFromTable({UnitIndex=c.source:entindex(),OrderType=DOTA_UNIT_ORDER_ATTACK_TARGET,TargetIndex=target:entindex(),Queue=false})
   end
   if finish and now-finish>1 then
    print('SF_P07_NATURAL|'..json.encode({case=which,kind=c.kind,mode=c.mode,attacks=attacks,hits=hits,misses=attacks-hits}))
    if not setup() then print('SF_V9_END|P07 natural miss complete');return nil end
   end
   assert(now-case_start<600,'case exceeded 600 seconds')
   return .03
  end)
  if not success then print('SF_P03_PROBE_ERROR|'..tostring(result));return nil end
  return result
 end,0)
end)
if not ok then print('SF_P03_PROBE_ERROR|'..tostring(err)) end
