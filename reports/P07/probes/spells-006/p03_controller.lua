local M={}
local json=require('game/dkjson')
local settings=require('bots/settings');settings.observation_profile='ggd-v4'
local agent=require('bots/agent')
local spells={'nevermore_shadowraze1','nevermore_shadowraze2','nevermore_shadowraze3','nevermore_requiem'}
local seen={}
local function read(u,m,...)
 if not u then return {ok=false,error='nil handle'} end
 if type(u[m])~='function' then return {ok=false,error='unavailable'} end
 local ok,v=pcall(u[m],u,...)
 if not ok then return {ok=false,error=tostring(v)} end
 return {ok=true,value=v}
end
local last_tick,last_second=nil,nil
function M.Think()
 local bot=GetBot();if not bot or bot:GetUnitName()~='npc_dota_hero_nevermore' then return end
 local tick=math.floor(GameTime()*30+.5);if tick==last_tick then return end;last_tick=tick
 local ok,err=pcall(function()
  for _,u in pairs(GetUnitList(UNIT_LIST_ALL)) do
    if u:GetUnitName()=='npc_dota_hero_nevermore' then seen[u:GetPlayerID()]={unit=u,team=u:GetTeam(),own=u==bot} end
  end
  local rows={}
  for id,entry in pairs(seen) do
   local u=entry.unit;local visibility=read(u,'CanBeSeen')
   local r={id=id,own=entry.own,visible=visibility.value==true,visibility=visibility,team=entry.team,abilities={},slots={}}
   for _,m in ipairs({'GetAnimActivity','GetAnimCycle','GetMana','GetMaxMana','IsCastingAbility','IsUsingAbility'}) do r[m]=read(u,m) end
   local active=read(u,'GetCurrentActiveAbility');r.active=active.ok and active.value and read(active.value,'GetName') or {ok=active.ok,error=active.error}
   for i,name in ipairs(spells) do
    local a=read(u,'GetAbilityByName',name)
    r.abilities[i]={handle_ok=a.ok,handle_present=a.value~=nil,error=a.error,cooldown=read(a.value,'GetCooldownTimeRemaining'),level=read(a.value,'GetLevel')}
   end
   for i=0,5 do
    local a=read(u,'GetAbilityInSlot',i)
    r.slots[#r.slots+1]={slot=i,handle_ok=a.ok,handle_present=a.value~=nil,error=a.error,name=read(a.value,'GetName'),cooldown=read(a.value,'GetCooldownTimeRemaining'),level=read(a.value,'GetLevel')}
   end
   local mod=read(u,'GetModifierByName','modifier_nevermore_shadowraze_debuff');r.modifier=mod
   if mod.ok and type(mod.value)=='number' and mod.value>=0 then
    r.stacks=read(u,'GetModifierStackCount',mod.value);r.remaining=read(u,'GetModifierRemainingDuration',mod.value)
   end
   if r.visible then r.v4=agent.public_spells(u,bot) end
   rows[#rows+1]=r
  end
  print('SF_P07_API|'..json.encode({team=bot:GetTeam(),time=DotaTime(),game_time=GameTime(),units=rows}))
  local second=math.floor(DotaTime());if second~=last_second then last_second=second;print('SF_P03_HERO|'..json.encode({team=bot:GetTeam(),time=DotaTime()})) end
 end)
 if not ok then print('SF_P03_PROBE_ERROR|'..tostring(err)) end
end

return M
