-- Optional diagnostic, after the ordinary first minute of the fixed smoke match.
-- Created creeps and teleportation must never be included in a baseline match.
local json=require('game/dkjson')
local heroes={}
for _,h in pairs(Entities:FindAllByClassname('npc_dota_hero_nevermore')) do heroes[h:GetTeamNumber()]=h end
local function state(h)
 local p=h:GetPlayerOwnerID()
 return {gold=PlayerResource:GetGold(p),xp=h:GetCurrentXP(),lh=PlayerResource:GetLastHits(p),deny=PlayerResource:GetDenies(p)}
end
heroes[2]:SetContextThink('p19_siege_probe',function()
 if GameRules:GetDOTATime(false,true)<60 then return .1 end
 local ok,err=pcall(function()
  for team,h in pairs(heroes) do FindClearSpaceForUnit(h,Vector(-3000+(team-2)*200,-3000,128),true) end
  for _,team in ipairs({2,3}) do
   local h=heroes[team];local other=heroes[5-team]
   for _,denied in ipairs({false,true}) do
    local creep_team=denied and team or 5-team
    local name=creep_team==2 and 'npc_dota_goodguys_siege' or 'npc_dota_badguys_siege'
    local u=CreateUnitByName(name,h:GetAbsOrigin()+Vector(50,0,0),true,nil,nil,creep_team)
    u:SetHealth(1)
    local before=state(h);local other_before=state(other)
    h:PerformAttack(u,true,true,true,false,false,false,true)
    local after=state(h);local other_after=state(other)
    print('SF_P19|'..json.encode({kind='controlled_siege',team=team,denied=denied,
     time=GameRules:GetDOTATime(false,true),before=before,after=after,other_before=other_before,
     other_after=other_after,creep_dead=not u:IsAlive(),diagnostic=true}))
   end
  end
 end)
 if not ok then print('SF_P19_PROBE_ERROR|'..tostring(err)) end
 return nil
end,.1)
