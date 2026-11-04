-- Server telemetry only. These events never enter the model observation.
local M={}
function M.start()
 if _G.SF_PLAY_EVENTS then return end
 _G.SF_PLAY_EVENTS=true
 local json=require('game/dkjson')
 local function entity(index)
  local u=index and EntIndexToHScript(index)
  if u and not u:IsNull() then return u end
 end
 local function emit(kind,row)
  row.kind=kind;row.time=GameRules:GetDOTATime(false,true)
  print('SF_PLAY_EVENT|'..json.encode(row))
 end
 ListenToGameEvent('dota_player_used_ability',function(e)
  local pid=e.PlayerID or e.playerid
  emit('cast',{player=pid,team=pid and PlayerResource:GetTeam(pid),ability=e.abilityname})
 end,nil)
 ListenToGameEvent('entity_hurt',function(e)
  local a=entity(e.entindex_attacker);local v=entity(e.entindex_killed)
  if not a or not v or a:GetUnitName()~='npc_dota_hero_nevermore' then return end
  local inf=entity(e.entindex_inflictor)
  emit('hurt',{team=a:GetTeamNumber(),target_team=v:GetTeamNumber(),target=v:GetUnitName(),
   ability=inf and inf:GetName() or '',damage=e.damage,attacker=e.entindex_attacker,victim=e.entindex_killed})
 end,nil)
 ListenToGameEvent('entity_killed',function(e)
  local v=entity(e.entindex_killed);local a=entity(e.entindex_attacker)
  if v and v:GetUnitName()=='npc_dota_hero_nevermore' then
   emit('death',{team=v:GetTeamNumber(),killer_team=a and a:GetTeamNumber(),killer=a and a:GetUnitName()})
  end
 end,nil)
 local host=Entities:FindAllByClassname('npc_dota_hero_nevermore')[1]
 assert(host,'SF must be spawned before telemetry starts')
 host:SetContextThink('sf_play_telemetry',function()
  for _,h in pairs(Entities:FindAllByClassname('npc_dota_hero_nevermore')) do
   local pid=h:GetPlayerOwnerID();local p=h:GetAbsOrigin()
   emit('hero',{team=h:GetTeamNumber(),player=pid,lh=PlayerResource:GetLastHits(pid),deny=PlayerResource:GetDenies(pid),
    deaths=PlayerResource:GetDeaths(pid),kills=PlayerResource:GetKills(pid),health=h:GetHealth(),alive=h:IsAlive(),x=p.x,y=p.y})
  end
  return .1
 end,0)
 print('SF_PLAY_EVENTS_READY')
end
return M
