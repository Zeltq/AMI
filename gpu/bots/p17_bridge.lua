-- Server-side command transport only. Never exports privileged state to policy.
local json=require('game/dkjson')
local last={}
print('SF_P17_BRIDGE_READY|'..json.encode({tick=math.floor(GameRules:GetGameTime()*30+.5),order=28}))
local heroes={}
for _,h in pairs(Entities:FindAllByClassname('npc_dota_hero_nevermore')) do heroes[h:GetTeamNumber()]=h end
local clock_unit=assert(heroes[2] or heroes[3],'No SF for bridge clock')
clock_unit:SetContextThink('p17_bridge',function()
 local success,failure=pcall(function()
 local tick=math.floor(GameRules:GetGameTime()*30+.5)
 for team=2,3 do
  local module='bots/chio_commands_'..team
  package.loaded[module]=nil
  local ok,commands=pcall(require,module)
  if ok then
   for _,d in ipairs(commands) do
    if d.action==0 and d.sequence>(last[team] or 0) and d.execute_tick<=tick then
     last[team]=d.sequence
     local unit=heroes[team]
     local accepted=unit and not unit:IsNull() and unit:IsAlive() and unit:GetTeamNumber()==team
     local reason=''
     if d.navigation_ready==false then accepted=false;reason='navigation' end
     -- A late command is dropped: do not silently change the contract clock.
     if tick~=d.execute_tick then accepted=false;reason='late' end
     if d.execute_tick~=d.observed_tick+7 then accepted=false;reason='clock' end
     if accepted then
      if d.action==0 then
       ExecuteOrderFromTable({UnitIndex=unit:entindex(),OrderType=28,Position=Vector(d.x,d.y,0),Queue=false})
      elseif d.action==1 then unit:Stop()
      else accepted=false;reason='unsupported' end
     end
     print('SF_CHIO_ACK|'..json.encode({team=team,sequence=d.sequence,observed_tick=d.observed_tick,
       execute_tick=d.execute_tick,tick=tick,accepted=accepted and true or false,reason=reason,action=d.action}))
    end
   end
  end
 end
 end)
 if not success then print('SF_P17_ERROR|'..tostring(failure));return nil end
 return 1/30
end,0)
