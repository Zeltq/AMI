-- Privileged diagnostic only. Never sent to either policy.
local json=require('game/dkjson')
ListenToGameEvent('npc_spawned',function(e)
 local ok,why=pcall(function()
  local u=EntIndexToHScript(e.entindex)
  if not u or u:IsNull() then return end
  local name=u:GetUnitName();local t=GameRules:GetDOTATime(false,false)
  if t>2 or string.find(name,'npc_dota_creep_',1,true)~=1 then return end
  local p=u:GetAbsOrigin()
  print('SF_P05_SPAWN|'..json.encode({time=t,name=name,team=u:GetTeamNumber(),entity=e.entindex,x=p.x,y=p.y,z=p.z}))
 end)
 if not ok then print('SF_P03_PROBE_ERROR|'..tostring(why)) end
end,nil)
print('SF_P05_SPAWN_READY')
