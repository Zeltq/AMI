-- Server bridge only. Starting items are ordinary shop purchases, including for bots.
require('bots/p17_bridge')
require('bots/play_model_events').start()
local json=require('game/dkjson')
ListenToGameEvent('entity_killed',function(e)
 local u=e.entindex_killed and EntIndexToHScript(e.entindex_killed)
 if not u or u:IsNull() then return end
 local name=u:GetUnitName()
 if name=='npc_dota_goodguys_tower1_mid' or name=='npc_dota_badguys_tower1_mid' then
  print('SF_CHIO_TERMINAL|'..json.encode({reason='tower',team=u:GetTeamNumber(),time=GameRules:GetDOTATime(false,true),tick=math.floor(GameRules:GetGameTime()*30+.5)}))
 end
end,nil)
