-- Read-only server event telemetry, separate from the bot policy VM.
local M={}
function M.start(seed)
 local json=require('game/dkjson')
 local scope='hero command only'
 if type(RandomSeed)=='function' then RandomSeed(seed);scope='engine RandomSeed and hero command' end
 print('SF_P03_SEED|'..json.encode({seed=seed,scope=scope,random_seed_api=type(RandomSeed)}))
 ListenToGameEvent('entity_killed',function(e)
  local ok,why=pcall(function()
   local u=EntIndexToHScript(e.entindex_killed)
   if not u or u:IsNull() then return end
   local name=u:GetUnitName()
   if string.find(name,'npc_dota_creep_',1,true)~=1 and not string.find(name,'siege',1,true) and name~='npc_dota_hero_nevermore' and not string.find(name,'tower',1,true) then return end
   local killer=e.entindex_attacker and EntIndexToHScript(e.entindex_attacker)
   print('SF_P03_DEATH|'..json.encode({time=GameRules:GetDOTATime(false,false),name=name,team=u:GetTeamNumber(),entity=e.entindex_killed,killer=killer and not killer:IsNull() and killer:GetUnitName() or 'unknown',killer_team=killer and not killer:IsNull() and killer:GetTeamNumber() or -1}))
  end)
  if not ok then print('SF_P03_PROBE_ERROR|'..tostring(why)) end
 end,nil)
 print('SF_P03_EVENTS_READY')
end
local ok,why=pcall(function()M.start(require('bots/p03_settings').p03_seed)end)
if not ok then print('SF_P03_PROBE_ERROR|'..tostring(why)) end
return M
