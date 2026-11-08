-- Read-only server instrumentation. Native mode 21 has no game-mode entity.
local stage='initial'
local loaded,load_error=pcall(function()
if _G.P19_ECONOMY_AUDIT then return end
_G.P19_ECONOMY_AUDIT=true
local json=require('game/dkjson')
local function emit(kind,row)
 row.kind=kind;row.time=GameRules:GetDOTATime(false,true)
 row.tick=math.floor(GameRules:GetGameTime()*30+.5)
 print('SF_P19|'..json.encode(row))
end
local function safe(o,k,...)
 if o and type(o[k])=='function' then local ok,v=pcall(o[k],o,...);if ok then return v end end
end
local function hero(h)
 local id=h:GetPlayerOwnerID();local inv={}
 for slot=0,20 do local it=h:GetItemInSlot(slot);if it then inv[#inv+1]={slot=slot,name=it:GetName(),charges=it:GetCurrentCharges()} end end
 return {team=h:GetTeamNumber(),player=id,entity=h:entindex(),gold=PlayerResource:GetGold(id),
 reliable=PlayerResource:GetReliableGold(id),unreliable=PlayerResource:GetUnreliableGold(id),
 xp=h:GetCurrentXP(),lh=PlayerResource:GetLastHits(id),deny=PlayerResource:GetDenies(id),
 level=h:GetLevel(),alive=h:IsAlive(),deaths=PlayerResource:GetDeaths(id),inventory=inv,
 bot_difficulty=safe(PlayerResource,'GetBotDifficulty',id)}
end
local mode=GameRules:GetGameModeEntity()
if mode then
 mode:SetModifyGoldFilter(function(_,e) emit('gold',e);return true end,{})
 mode:SetModifyExperienceFilter(function(_,e) emit('xp',e);return true end,{})
end
stage='events'
ListenToGameEvent('entity_killed',function(e)
 local u=e.entindex_killed and EntIndexToHScript(e.entindex_killed)
 local a=e.entindex_attacker and EntIndexToHScript(e.entindex_attacker)
 if not u or u:IsNull() then return end
 local heroes={};for _,h in pairs(Entities:FindAllByClassname('npc_dota_hero_nevermore')) do heroes[#heroes+1]=hero(h) end
 emit('kill',{victim=u:GetUnitName(),entity=u:entindex(),team=u:GetTeamNumber(),heroes=heroes,
 attacker=a and a:GetUnitName(),attacker_entity=a and a:entindex(),attacker_team=a and a:GetTeamNumber(),
 bounty_min=safe(u,'GetMinimumGoldBounty'),bounty_max=safe(u,'GetMaximumGoldBounty'),
 xp_bounty=safe(u,'GetDeathXP'),flagbearer=safe(u,'HasModifier','modifier_creep_flagbearer'),
 position={x=u:GetAbsOrigin().x,y=u:GetAbsOrigin().y}})
end,nil)
ListenToGameEvent('dota_item_purchased',function(e) emit('purchase_event',e) end,nil)
local previous={};local bucket=nil
stage='think'
local thinker=mode or Entities:FindByClassname(nil,'npc_dota_hero_nevermore')
assert(thinker,'SF thinker unavailable')
thinker:SetContextThink('p19_economy',function()
 local ok,err=pcall(function()
  local t=GameRules:GetDOTATime(false,true);local current=math.max(-9,math.floor((t+.001)/10))
  local sample=current~=bucket and t>=-90.04
  for _,h in pairs(Entities:FindAllByClassname('npc_dota_hero_nevermore')) do
   local r=hero(h);local key=r.team;local signature=json.encode(r)
   if previous[key]~=signature then emit('state',r);previous[key]=signature end
   if sample then r.target_time=current*10;emit('sample',r) end
  end
  if sample then bucket=current end
 end)
 if not ok then emit('error',{message=tostring(err)});return nil end
 return 1/30
end,0)
local cvars={}
stage='cvars'
for _,name in ipairs({'dota_bot_difficulty','dota_bot_radiant_difficulty','dota_bot_dire_difficulty',
 'dota_bot_level','dota_gamemode','host_timescale','sv_cheats','dota_tutorial_skip_pregame',
 'dota_1v1_skip_strategy','dota_starting_gold','dota_gold_per_tick','dota_gold_tick_time'}) do
 cvars[name]=safe(Convars,'GetStr',name)
end
emit('ready',{cvars=cvars,filters_available=mode~=nil,mode=safe(GameRules,'GetGameMode'),gold_per_tick=safe(GameRules,'GetGoldPerTick'),
 gold_tick_time=safe(GameRules,'GetGoldTickTime')})
end)
if not loaded then print('SF_P19_LOAD_ERROR|'..stage..'|'..tostring(load_error)) end
