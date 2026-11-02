-- Synchronous native kill transactions, before mode21's second-kill shutdown.
local ok,err=pcall(function()
local json=require('game/dkjson');local config=require('bots/gpu_server_config')
local sides={};for _,h in ipairs(Entities:FindAllByClassname('npc_dota_hero_nevermore'))do sides[h:GetTeamNumber()]=h end
local r,d=sides[2],sides[3];assert(r and d)
local function stop(h)ExecuteOrderFromTable({UnitIndex=h:entindex(),OrderType=DOTA_UNIT_ORDER_STOP})end
local function place()
 FindClearSpaceForUnit(r,Vector(80,-816,0),true);FindClearSpaceForUnit(d,Vector(464,-816,0),true)
 for _,h in ipairs({r,d})do stop(h);h:SetHealth(h:GetMaxHealth());h:SetMana(h:GetMaxMana())end
end
for _,h in ipairs({r,d})do
 if not h:FindItemInInventory('item_wraith_band')then h:AddItemByName('item_wraith_band')end
 h:SetGold(0,true);h:SetGold(95,false)
end
d:SetGold(config.victim_gold,false);place()
local function row(h)
 local id=h:GetPlayerOwnerID();local p=h:GetAbsOrigin()
 local modifiers={};for i=0,h:GetModifierCount()-1 do modifiers[#modifiers+1]=h:GetModifierNameByIndex(i)end
 return{team=h:GetTeamNumber(),alive=h:IsAlive(),health=h:GetHealth(),mana=h:GetMana(),x=p.x,y=p.y,z=p.z,
  gold=PlayerResource:GetGold(id),reliable=PlayerResource:GetReliableGold(id),unreliable=PlayerResource:GetUnreliableGold(id),net_worth=PlayerResource:GetNetWorth(id),
  xp=h:GetCurrentXP(),souls=h:GetModifierStackCount('modifier_nevermore_necromastery',h),deaths=PlayerResource:GetDeaths(id),level=h:GetLevel(),invulnerable=h:IsInvulnerable(),modifiers=modifiers}
end
local ticks=0
local function capture(label)
 print('SF_GPU_SERVER_AGGRO|'..json.encode({case=ticks<60 and 1 or 2,event=label,time=GameRules:GetDOTATime(false,true),units={row(r),row(d)}}))
end
local function kill(source)
 capture('before_kill')
 -- Bypass spawn protection only in this economy transaction diagnostic.
 -- Natural respawn protection is a separate fidelity check.
 ApplyDamage({victim=d,attacker=source,damage=10000,damage_type=DAMAGE_TYPE_PURE,damage_flags=DOTA_DAMAGE_FLAG_BYPASSES_INVULNERABILITY})
 capture('after_kill')
 assert(not d:IsAlive(),'controlled kill blocked; do not accept a no-op as a second kill')
end
local tower=Entities:FindByName(nil,'dota_goodguys_tower1_mid')
if config.first_source=='tower'then assert(tower,'Radiant mid T1 missing')end
r:SetContextThink('sf_gpu_economy',function()
 local success,result=pcall(function()
  ticks=ticks+1;stop(r);stop(d)
  if ticks==30 then kill(config.first_source=='tower'and tower or r)
  elseif ticks==60 then d:RespawnHero(false,false);place();capture('manual_respawn')
  elseif ticks==90 then kill(r);print('SF_GPU_SERVER_DONE|'..json.encode({kills=2}));return nil end
  return .03
 end)
 if not success then print('SF_GPU_SERVER_ERROR|'..tostring(result));return nil end
 return result
end,0)
print('SF_GPU_SERVER_READY')
end)
if not ok then print('SF_GPU_SERVER_ERROR|'..tostring(err))end
