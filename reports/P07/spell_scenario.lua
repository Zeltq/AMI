-- Privileged fixture control and ground truth. Never used as policy input.
local success,failure=pcall(function()
if not _G.SF_P07_SCENARIO then
 _G.SF_P07_SCENARIO=true
 local json=require('game/dkjson')
 local spells={'nevermore_shadowraze1','nevermore_shadowraze2','nevermore_shadowraze3','nevermore_requiem'}
 local start,heroes,step=nil,{},-1
 local host=Entities:FindAllByClassname('npc_dota_hero_nevermore')[1];assert(host,'missing SF')
 host:SetContextThink('p07_spells',function()
  local ok,err=pcall(function()
   if not start then
    for _,h in pairs(Entities:FindAllByClassname('npc_dota_hero_nevermore')) do heroes[h:GetTeamNumber()]=h end
    if not heroes[2] or not heroes[3] then return end
    for team,h in pairs(heroes) do
     h:SetAbsOrigin(GetGroundPosition(Vector(team==2 and 80 or 380,-816,0),h));h:SetForwardVector(Vector(team==2 and 1 or -1,0,0))
     for _,name in ipairs(spells) do local a=h:FindAbilityByName(name);if a then a:SetLevel(name=='nevermore_requiem' and 1 or 4) end end
     while h:GetLevel()<7 do h:HeroLevelUp(false) end
     h:SetBaseHealthRegen(1000);h:SetMana(h:GetMaxMana())
    end
    start=GameRules:GetGameTime()
   end
   local t=GameRules:GetGameTime()-start;local next_step=math.floor(t/4)
   if t<32 then for _,team in ipairs({2,3}) do AddFOWViewer(team,Vector(380,-816,0),1500,.1,false) end end
   if next_step~=step then
    step=next_step
    local index=step%4+1;local team=math.floor(step/4)%2==0 and 3 or 2;local h=heroes[team]
    if step<8 then
     local distance=({200,450,700,300})[index]
     heroes[2]:SetAbsOrigin(GetGroundPosition(Vector(80,-816,0),heroes[2]));heroes[3]:SetAbsOrigin(GetGroundPosition(Vector(80+distance,-816,0),heroes[3]))
     h:SetMana(h:GetMaxMana());local a=h:FindAbilityByName(spells[index]);a:EndCooldown()
     ExecuteOrderFromTable({UnitIndex=h:entindex(),OrderType=DOTA_UNIT_ORDER_CAST_NO_TARGET,AbilityIndex=a:entindex(),Queue=false})
     print('SF_P07_CAST|'..json.encode({time=GameRules:GetDOTATime(false,true),game_time=GameRules:GetGameTime(),spell=index,team=team}))
    elseif step==8 then heroes[3]:SetAbsOrigin(GetGroundPosition(Vector(5000,5000,0),heroes[3]));print('SF_P07_FOG') end
   end
  end)
  if not ok then print('SF_P03_PROBE_ERROR|'..tostring(err));return nil end
  return 1/30
 end,0)
end
end)
if not success then print('SF_P03_PROBE_ERROR|'..tostring(failure)) end
