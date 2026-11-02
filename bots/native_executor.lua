-- Native-state decisions are produced by Go. No farming or aiming policy here.
local agent=require("bots/agent")
local model_observation=require("bots/native_model_observation")
local json=require("game/dkjson")
local next_aux=-10000
local settings=require("bots/settings")
local st={skill_index=1,permanent=0,seq=0}
local last_tick=nil
local last_sequence=0
local spells={"nevermore_shadowraze1","nevermore_shadowraze2","nevermore_shadowraze3","nevermore_requiem"}
local allowed={item_boots=true,item_bracer=true,item_flask=true,item_enchanted_mango=true,item_tango=true,item_clarity=true,item_tpscroll=true}
local ended=false
local cached_commands={}
local function ack(d,tick,accepted,reason)
 print("SF_NATIVE_ACK|"..GetTeam().."|"..d.sequence.."|"..tick.."|"..(accepted and 1 or 0).."|"..(d.target or 0).."|"..(reason or ""))
end
local function execute(bot,d)
 local target=d.action==3 and d.target and GetBotByHandle(d.target) or nil
 if d.action==0 then
 elseif d.action==1 then bot:Action_MoveToLocation(Vector(d.x,d.y,0))
 elseif d.action==2 then bot:Action_MoveDirectly(Vector(d.x,d.y,0))
 elseif d.action==4 then bot:Action_ClearActions(true)
 elseif d.action==5 then bot:Action_HoldPosition()
 elseif d.action==3 then
  if (type(target)~="table" and type(target)~="userdata") or type(target.IsNull)~="function" or type(target.CanBeSeen)~="function" or type(target.IsAlive)~="function" or type(target.GetTeam)~="function" or target:IsNull() or not target:CanBeSeen() or not target:IsAlive() then return false,"target_unavailable" end
  if target:GetTeam()==TEAM_NEUTRAL then error("neutral target") end
  bot:Action_AttackUnit(target,true)
 elseif d.action>=6 and d.action<=9 then
  local a=bot:GetAbilityByName(spells[d.action-5])
  if not a or not a:IsFullyCastable() then return false,"spell_unavailable" end
  bot:Action_UseAbility(a)
 elseif d.action==10 then
  if not allowed[d.item] then error("forbidden item") end
  local slot=bot:FindItemSlot(d.item);local a=slot>=0 and (slot<=5 or (d.item=="item_tpscroll" and slot==15)) and bot:GetItemInSlot(slot) or nil
  if not a or not a:IsFullyCastable() then return false,"item_unavailable" end
  if d.item=="item_flask" or d.item=="item_enchanted_mango" or d.item=="item_clarity" then bot:Action_UseAbilityOnEntity(a,bot)
  elseif d.item=="item_tango" then
   local found=false
   for _,tree in ipairs(bot:GetNearbyTrees(1600)) do if tree==d.target then found=true;break end end
   if not found then return false,"tree_unavailable" end
   bot:Action_UseAbilityOnTree(a,d.target)
  elseif d.item=="item_tpscroll" then bot:Action_UseAbilityOnLocation(a,Vector(d.x,d.y,0))
  else return false,"unsupported_item_target" end
 else error("unsupported native action") end
 if d.buy and d.buy~="" then
  if not allowed[d.buy] then error("forbidden purchase") end
  local _,ok=agent.order_item(bot,d.buy)
  if not ok then return false,"purchase_failed" end
 end
 return true,""
end
local M={}
function M.Think()
 local bot=GetBot();if not bot or bot:GetUnitName()~="npc_dota_hero_nevermore" then return end
 local phase=GetGameState();if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
 local tick=math.floor(GameTime()*30+.5);if tick==last_tick then return end;last_tick=tick
 agent.infrastructure(bot,st)
 if DotaTime()>=next_aux then
  next_aux=DotaTime()+.5
  local info={team=GetTeam(),tick=tick,shop={},trees={}}
  for name,_ in pairs(allowed) do info.shop[#info.shop+1]={name=name,price=GetItemCost(name),stock=GetItemStockCount(name)} end
  for _,tree in ipairs(bot:GetNearbyTrees(1600)) do local p=GetTreeLocation(tree);if IsLocationVisible(p) then info.trees[#info.trees+1]={id=tree,x=p.x,y=p.y,z=p.z} end end
  print("SF_NATIVE_AUX|"..json.encode(info))
 end
 if DotaTime()>=0 and not ended then
  for _,pid in ipairs(GetTeamPlayers(GetOpposingTeam())) do
   if GetSelectedHeroName(pid)=="npc_dota_hero_nevermore" and GetHeroDeaths(pid)>0 then
    ended=true;print("SF_NATIVE_END|"..GetOpposingTeam().."|"..DotaTime().."|death|0")
   end
  end
  local tower=GetTower(GetTeam(),TOWER_MID_1)
  local reason=not bot:IsAlive() and "death" or ((not tower or not tower:IsAlive()) and "tower" or (DotaTime()>=600 and "timeout" or nil))
  if reason then ended=true;print("SF_NATIVE_END|"..GetTeam().."|"..DotaTime().."|"..reason.."|"..bot:GetLastHits()) end
 end
 if ended then return end
 local module="bots/native_commands_"..GetTeam();package.loaded[module]=nil
 local ok,commands=pcall(require,module)
 if ok then cached_commands=commands else commands=cached_commands end
 if tick%math.floor(settings.interval*30+.5)==0 then model_observation.emit(bot,tick,commands) end
 local d=commands[tick]
 if commands.allow_late then
  -- Execute the newest due command once. An older late command must never
  -- override a newer order; future commands keep their scheduled delay.
  d=nil
  for key,candidate in pairs(commands) do
   if type(key)=="number" and candidate.execute_tick<=tick and candidate.sequence>last_sequence then
    if not d or candidate.sequence>d.sequence then d=candidate end
   end
  end
 end
 if d and d.sequence>last_sequence then
  last_sequence=d.sequence
  if d.execute_tick>tick or (not commands.allow_late and d.execute_tick~=tick) or d.observed_tick>=tick then error("invalid command clock") end
  local accepted,why=execute(bot,d);ack(d,tick,accepted,why)
 end
end
return M
