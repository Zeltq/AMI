-- Stock fountain aura radius/linger; no model inference or stat overrides.
local ok,err=pcall(function()
local json=require('game/dkjson');local heroes,fountains={},{}
for _,h in ipairs(Entities:FindAllByClassname('npc_dota_hero_nevermore'))do heroes[h:GetTeamNumber()]=h end
for _,f in ipairs(Entities:FindAllByClassname('ent_dota_fountain'))do fountains[f:GetTeamNumber()]=f end
assert(heroes[2]and heroes[3]and fountains[2]and fountains[3],'SF/fountain entities missing')
local function stop(h)ExecuteOrderFromTable({UnitIndex=h:entindex(),OrderType=DOTA_UNIT_ORDER_STOP})end
local function call(h,name)local fn=h[name];if type(fn)=='function'then local yes,v=pcall(fn,h);if yes then return v end end end
local cases={}
for _,distance in ipairs({1100,1200,1224,1240,1250,1264,1280,1300})do cases[#cases+1]={distance=distance,duration=4}end
cases[#cases+1]={distance=2000,linger=true,duration=5}
local which=0;local start,departed
local function place(h,team,distance)
 local p=fountains[team]:GetAbsOrigin();local sign=team==2 and 1 or-1
 FindClearSpaceForUnit(h,Vector(p.x+sign*distance/math.sqrt(2),p.y+sign*distance/math.sqrt(2),p.z),true);stop(h)
end
local function row(h,team)
 local p=h:GetAbsOrigin();local f=fountains[team]:GetAbsOrigin();local m=h:FindModifierByName('modifier_fountain_aura_buff')
 return{team=team,x=p.x,y=p.y,z=p.z,source_x=f.x,source_y=f.y,source_z=f.z,distance=math.sqrt((p.x-f.x)^2+(p.y-f.y)^2),hull=h:GetHullRadius(),health=h:GetHealth(),max_health=h:GetMaxHealth(),mana=h:GetMana(),max_mana=h:GetMaxMana(),health_regen=call(h,'GetHealthRegen'),mana_regen=call(h,'GetManaRegen'),buff=m~=nil,remaining=m and m:GetRemainingTime()or 0}
end
local function setup()
 which=which+1;if which>#cases then return false end
 local c=cases[which]
 for team,h in pairs(heroes)do h:RemoveModifierByName('modifier_fountain_aura_buff');place(h,team,c.linger and 500 or c.distance)end
 start=GameRules:GetGameTime();departed=false
 print('SF_GPU_SERVER_CASE|'..json.encode({case=which,time=GameRules:GetDOTATime(false,true),distance=c.distance,linger=c.linger or false,duration=c.duration}));return true
end
assert(setup())
heroes[2]:SetContextThink('sf_gpu_fountain',function()
 local success,result=pcall(function()
  local elapsed=GameRules:GetGameTime()-start;local c=cases[which]
  if c.linger and elapsed>=.5 and not departed then
   for team,h in pairs(heroes)do place(h,team,c.distance)end;departed=true
   print('SF_GPU_SERVER_ORDER|'..json.encode({case=which,elapsed=elapsed,action='leave_fountain'}))
  end
  for _,h in pairs(heroes)do stop(h)end
  print('SF_GPU_SERVER_AGGRO|'..json.encode({case=which,elapsed=elapsed,time=GameRules:GetDOTATime(false,true),units={row(heroes[2],2),row(heroes[3],3)}}))
  if elapsed>=c.duration then if not setup()then print('SF_GPU_SERVER_DONE|{}');return nil end end
  return .03
 end)
 if not success then print('SF_GPU_SERVER_ERROR|'..tostring(result));return nil end;return result
end,0)
end)
if not ok then print('SF_GPU_SERVER_ERROR|'..tostring(err))end
