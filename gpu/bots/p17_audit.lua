-- Privileged fixtures and measurements for action audit only, not model matches.
local ok,err=pcall(function()
 local json=require('game/dkjson');local heroes={}
 for _,h in pairs(Entities:FindAllByClassname('npc_dota_hero_nevermore')) do heroes[h:GetTeamNumber()]=h end
 local own=assert(heroes[2]);local enemy=assert(heroes[3]);local last=-1;local previous=-1
 local start=math.ceil(GameRules:GetGameTime()*30)+60
 local spells={'nevermore_shadowraze1','nevermore_shadowraze2','nevermore_shadowraze3','nevermore_requiem'}
 own:SetContextThink('p17_audit',function()
  local success,result=pcall(function()
   local tick=math.floor(GameRules:GetGameTime()*30+.5);if tick==last then return end;last=tick
   local elapsed=tick-start;if elapsed<0 then return end
   local index=math.floor(elapsed/180);local ct=elapsed%180
   if index>13 then print('SF_P17_AUDIT_DONE|true');return 'done' end
   if index~=previous then
    previous=index
    for _,u in pairs(Entities:FindAllByClassname('npc_dota_creep_lane')) do UTIL_Remove(u) end
    for team,h in pairs(heroes) do
     h:Stop();h:SetAbsOrigin(GetGroundPosition(Vector(team==2 and 80 or 780,-816,0),h));h:SetForwardVector(Vector(1,0,0))
     while h:GetLevel()<7 do h:HeroLevelUp(false) end
     for i,name in ipairs(spells) do local a=h:FindAbilityByName(name);a:SetLevel(i==4 and 1 or 4);a:EndCooldown() end
     h:RemoveModifierByName('modifier_flask_healing');h:RemoveModifierByName('modifier_tango_heal')
     h:SetMana(h:GetMaxMana());h:SetHealth(h:GetMaxHealth())
    end
    own:AddNewModifier(own,nil,'modifier_invulnerable',{})
    if index>=10 and index<=12 then
     own:RemoveModifierByName('modifier_invulnerable')
     enemy:SetAbsOrigin(GetGroundPosition(Vector(5000,5000,0),enemy))
    end
    for slot=0,16 do local item=own:GetItemInSlot(slot);if item then own:RemoveItem(item);UTIL_Remove(item) end end
    for _,name in ipairs({'item_wraith_band','item_flask','item_enchanted_mango','item_tango'}) do own:AddItemByName(name) end
    if index==10 or index==12 then own:SetHealth(own:GetMaxHealth()-300) end
    if index==11 then own:SetMana(own:GetMaxMana()-200) end
    if index==2 or index==3 then
     local team=index==2 and 3 or 2
     local u=CreateUnitByName(team==2 and 'npc_dota_creep_goodguys_melee' or 'npc_dota_creep_badguys_melee',Vector(280,-816,0),true,nil,nil,team)
     u:SetHealth(u:GetMaxHealth()*.45);u:SetIdleAcquire(false);u:SetAcquisitionRange(0);u:Hold()
    elseif index==5 then
     local tower=Entities:FindByName(nil,'dota_badguys_tower1_mid')
     if not tower then for _,u in pairs(Entities:FindAllByClassname('npc_dota_tower')) do if u:GetUnitName()=='npc_dota_badguys_tower1_mid' then tower=u;break end end end
     assert(tower,'enemy mid tower');local p=tower:GetAbsOrigin();own:SetAbsOrigin(GetGroundPosition(p+Vector(-400,0,0),own))
    elseif index>=6 and index<=8 then enemy:SetAbsOrigin(GetGroundPosition(Vector(80+({200,450,700})[index-5],-816,0),enemy))
    elseif index==12 then own:SetAbsOrigin(GetGroundPosition(Vector(-192,-1192,0),own)) end
    print('SF_P17_AUDIT_CASE|'..json.encode({case=index,start_tick=tick}))
   end
   local a=own:FindAbilityByName('nevermore_requiem');local p=own:GetAbsOrigin()
   print('SF_P17_AUDIT_FRAME|'..json.encode({case=index,case_tick=ct,tick=tick,x=p.x,y=p.y,mana=own:GetMana(),
    requiem_phase=a:IsInAbilityPhase(),requiem_cooldown=a:GetCooldownTimeRemaining()}))
  end)
  if not success then print('SF_P17_ERROR|'..tostring(result));return nil end
  if result=='done' then return nil end
  return 1/30
 end,0)
end)
if not ok then print('SF_P17_ERROR|'..tostring(err)) end
