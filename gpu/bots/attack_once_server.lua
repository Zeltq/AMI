-- Actual BotAPI once/reissue semantics. Server only sets stock fixtures.
local ok,err=pcall(function()
local json=require('game/dkjson');local config=require('bots/gpu_server_config');local sides={}
for _,h in ipairs(Entities:FindAllByClassname('npc_dota_hero_nevermore'))do sides[h:GetTeamNumber()]=h end
local r,d=sides[2],sides[3];assert(r and d)
for _,h in ipairs({r,d})do if not h:FindItemInInventory('item_wraith_band')then h:AddItemByName('item_wraith_band')end end
local cases={{name='once_then_continue',pulses={.3}},{name='three_separate_attacks',pulses={.3,1.5,2.7}},{name='same_target_reissued_during_windup',pulses={.3,.5,.7}}}
local which=0;local start
local function stop(h)ExecuteOrderFromTable({UnitIndex=h:entindex(),OrderType=DOTA_UNIT_ORDER_STOP})end
local function row(h)
 local p=h:GetAbsOrigin();local t=h:GetAttackTarget();local g=h:GetAggroTarget();local v=h:GetForwardVector()
 return{entity=h:entindex(),team=h:GetTeamNumber(),x=p.x,y=p.y,z=p.z,health=h:GetHealth(),max_health=h:GetMaxHealth(),facing=math.atan2(v.y,v.x)*180/math.pi,last_attack=h:GetLastAttackTime(),attack_target=t and t:entindex()or nil,aggro_target=g and g:entindex()or nil,activity=h:GetSequence(),cycle=h:GetCycle()}
end
local function setup()
 which=which+1;if which>#cases then return false end
 stop(r);stop(d);FindClearSpaceForUnit(r,Vector(80,-816,0),true);FindClearSpaceForUnit(d,Vector(480,-816,0),true)
 r:SetForwardVector(Vector(1,0,0));d:SetForwardVector(Vector(-1,0,0))
 for _,h in ipairs({r,d})do h:SetHealth(h:GetMaxHealth());h:SetMana(h:GetMaxMana());h:RemoveModifierByName('modifier_fountain_aura_buff')end
 start=GameRules:GetGameTime();local c=cases[which]
 print('SF_GPU_SERVER_CASE|'..json.encode({case=which,name=c.name,time=GameRules:GetDOTATime(false,true),pulses=c.pulses,hold_attack=true,duration=4,actor_mode='bot'}));return true
end
assert(setup())
r:SetContextThink('sf_gpu_once',function()
 local success,result=pcall(function()
  stop(d);local elapsed=GameRules:GetGameTime()-start
  print('SF_GPU_SERVER_AGGRO|'..json.encode({case=which,elapsed=elapsed,time=GameRules:GetDOTATime(false,true),game_time=GameRules:GetGameTime(),units={row(r),row(d)}}))
  if elapsed>=4 then if not setup()then print('SF_GPU_SERVER_DONE|{}');return nil end end
  return .03
 end)
 if not success then print('SF_GPU_SERVER_ERROR|'..tostring(result));return nil end;return result
end,0)
end)
if not ok then print('SF_GPU_SERVER_ERROR|'..tostring(err))end
