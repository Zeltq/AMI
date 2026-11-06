local agent=require('bots/agent')
local economy=require('bots/chio_economy')
local json=require('game/dkjson')
local settings=require('bots/settings')
settings.chio=true;settings.observation_profile='ggd-v4'
local spells={'nevermore_shadowraze1','nevermore_shadowraze2','nevermore_shadowraze3','nevermore_requiem'}
local items={'item_flask','item_enchanted_mango','item_tango'}
local st={skill_index=1,permanent=0,seq=0,start_bought=true};local last=-1;local sequence=0;local commands={};local casting=nil
local M={}
local function execute(bot,d)
 if d.action==0 then return nil,'server_bridge' end
 if d.action==1 then bot:Action_ClearActions(true);casting=nil;st.attack=nil;return true,'stop' end
 if d.action>=2 and d.action<=5 then
  local u=agent.resolve(d.target)
  if not u or u:IsNull() or not u:CanBeSeen() or not u:IsAlive() then return false,'target_unavailable' end
  if d.action==3 and (u:GetTeam()~=GetTeam() or u:GetHealth()>u:GetMaxHealth()*.5) then return false,'deny_unavailable' end
  if d.action~=3 and u:GetTeam()==GetTeam() then return false,'wrong_team' end
  if st.attack==u then return true,'same_attack' end
  bot:Action_AttackUnit(u,true);st.attack=u;return true,'attack'
 end
 st.attack=nil
 if d.action>=6 and d.action<=9 then
  if d.continuation then return true,'cast_continuation' end
  if casting==d.action and (bot:IsCastingAbility() or bot:IsUsingAbility()) then return true,'same_cast' end
  local a=bot:GetAbilityByName(spells[d.action-5])
  if not a or not a:IsFullyCastable() then return false,'spell_unavailable' end
  bot:Action_UseAbility(a);casting=d.action;return true,'cast'
 end
 local name=items[d.action-9];if not name then return false,'unknown_action' end
 local slot=bot:FindItemSlot(name);local a=slot>=0 and slot<=5 and bot:GetItemInSlot(slot) or nil
 if not a or not a:IsFullyCastable() then return false,'item_unavailable' end
 if d.action==10 and bot:HasModifier('modifier_flask_healing') then return false,'already_healing' end
 if d.action==12 then
  local nearest=nil;local distance=math.huge;local p=bot:GetLocation()
  for _,tree in ipairs(bot:GetNearbyTrees(1600)) do
   local q=GetTreeLocation(tree);local ds=(p.x-q.x)^2+(p.y-q.y)^2
   if IsLocationVisible(q) and (ds<distance or (ds==distance and tree<(nearest or math.huge))) then nearest=tree;distance=ds end
  end
  if nearest~=d.target then return false,'tree_changed' end
  bot:Action_UseAbilityOnTree(a,nearest)
 else bot:Action_UseAbilityOnEntity(a,bot) end
 return true,'item'
end
function M.Think()
 local bot=GetBot();if not bot or bot:GetUnitName()~='npc_dota_hero_nevermore' then return end
 local phase=GetGameState();if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
 local tick=math.floor(GameTime()*30+.5);if tick==last then return end;last=tick
 if tick%6==0 and not settings.audit then economy.Think(bot,st) end
 if not bot:IsAlive() then casting=nil;st.attack=nil;return end
 if casting and not bot:IsCastingAbility() and not bot:IsUsingAbility() then casting=nil end
 local module='bots/chio_commands_'..GetTeam();package.loaded[module]=nil
 local ok,result=pcall(require,module);if ok then commands=result end
 for _,d in ipairs(commands) do
  if d.sequence>sequence and d.execute_tick<=tick then
   sequence=d.sequence
   if d.action~=0 then
    local accepted,why=false,'late'
    if d.execute_tick==tick and d.execute_tick==d.observed_tick+7 then accepted,why=execute(bot,d) end
    print('SF_CHIO_ACK|'..json.encode({team=GetTeam(),sequence=d.sequence,tick=tick,observed_tick=d.observed_tick,
     execute_tick=d.execute_tick,action=d.action,accepted=accepted,reason=why}))
   else st.attack=nil end
  end
 end
 if tick%6==0 then
  local pending=casting
  for _,d in ipairs(commands) do if d.execute_tick>tick and d.action>=6 and d.action<=9 and not d.continuation then pending=d.action end end
  st.seq=st.seq+1
  local snapshot=agent.snapshot(bot,st);snapshot.tick=tick;snapshot.pending_cast=pending
  snapshot.deaths=GetHeroDeaths(bot:GetPlayerID());snapshot.player=bot:GetPlayerID()
  local body=json.encode(snapshot);local count=math.ceil(#body/1800)
  for part=1,count do print('SF_CHIO_OBS|'..GetTeam()..'|'..tick..'|'..part..'|'..count..'|'..body:sub((part-1)*1800+1,part*1800)) end
 end
end
return M
