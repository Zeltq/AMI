-- Read-only server clock probe; installed before pregame, never policy input.
if not _G.SF_P06_CLOCK_INSTALLED then
 _G.SF_P06_CLOCK_INSTALLED=true
 local json=require('game/dkjson')
 local function sample()
  print('SF_P06_STATE|'..json.encode({state=GameRules:State_Get(),game_time=GameRules:GetGameTime(),dota_time=GameRules:GetDOTATime(true,false)}))
 end
 ListenToGameEvent('game_rules_state_change',sample,nil)
 sample()
end
