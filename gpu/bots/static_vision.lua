-- Diagnostic intervention only: expose building-provided vision separately.
local json=require('game/dkjson')
local ok,err=pcall(function()
 local changed={}
 for _,class in ipairs({'npc_dota_hero_nevermore','npc_dota_courier'})do
  for _,u in ipairs(Entities:FindAllByClassname(class))do
   u:SetDayTimeVisionRange(0);u:SetNightTimeVisionRange(0)
   changed[#changed+1]={name=u:GetUnitName(),team=u:GetTeamNumber(),day=u:GetDayTimeVisionRange(),night=u:GetNightTimeVisionRange()}
  end
 end
 assert(#changed>=2,'both SF vision ranges must be disabled')
 print('SF_GPU_STATIC|'..json.encode({units=changed,time=GameRules:GetDOTATime(false,true),game_time=GameRules:GetGameTime()}))
end)
if not ok then print('SF_GPU_ERROR|'..tostring(err))end
