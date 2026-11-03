-- Recording-only inputs. Never consulted by the Go teacher.
local json=require("game/dkjson")
local agent=require("bots/agent")
local settings=require("bots/settings")
local M={}
local spells={"nevermore_shadowraze1","nevermore_shadowraze2","nevermore_shadowraze3","nevermore_requiem"}
local items={"item_wraith_band","item_boots","item_bracer","item_tpscroll","item_enchanted_mango","item_flask","item_tango","item_clarity"}
local function ability(a,name)
 local r={name=name}
 if a then r.level=a:GetLevel();r.mana_cost=a:GetManaCost();r.cast_point=a:GetCastPoint();r.channel_time=a:GetChannelTime() end
 return r
end
local function extra(u,handle,own)
 local a=u:GetAnimActivity();local cast=false;local channel=false
 for i=1,6 do cast=cast or a==_G["ACTIVITY_CAST_ABILITY_"..i];channel=channel or a==_G["ACTIVITY_CHANNEL_ABILITY_"..i] end
 local attacking=a==ACTIVITY_ATTACK or a==ACTIVITY_ATTACK2
 local using=cast or channel
 if own then attacking=u:GetCurrentActionType()==BOT_ACTION_TYPE_ATTACK;cast=u:IsCastingAbility();channel=u:IsChanneling();using=u:IsUsingAbility() end
 local result={handle=handle,attack_interval=u:GetSecondsPerAttack(),attacking=attacking,casting=cast,channeling=channel,using_ability=using}
 if settings.observation_profile=="ggd-v4" and u:GetUnitName()=="npc_dota_hero_nevermore" then result.v4_json=json.encode(agent.public_spells(u,GetBot()) or {}) end
 return result
end
function M.emit(bot,tick,commands)
 local p=bot:GetLocation();local info={team=GetTeam(),tick=tick,x=p.x,y=p.y,can_hold=type(bot.Action_HoldPosition)=="function",terrain={},skills={},items={},units={extra(bot,0,true)},towers={},skill_slots={}}
 local grid={{},{},{}}
 for y=1,8 do for x=1,8 do
  local loc=Vector(p.x+(x-4.5)*256,p.y+(y-4.5)*256,p.z);local visible=IsLocationVisible(loc)
  grid[1][#grid[1]+1]=GetHeightLevel(loc)/5
  grid[2][#grid[2]+1]=visible and IsLocationPassable(loc) and 1 or 0
  grid[3][#grid[3]+1]=visible and 1 or 0
 end end
 for channel=1,3 do for _,v in ipairs(grid[channel]) do info.terrain[#info.terrain+1]=v end end
 for _,name in ipairs(spells) do info.skills[#info.skills+1]=ability(bot:GetAbilityByName(name),name) end
 for _,name in ipairs(items) do local slot=bot:FindItemSlot(name);info.items[#info.items+1]=ability(slot>=0 and bot:GetItemInSlot(slot) or nil,name) end
 for _,handle in ipairs(commands.units or {}) do
  local u=GetBotByHandle(handle)
  if (type(u)=="table" or type(u)=="userdata") and not u:IsNull() and u~=bot and type(u.CanBeSeen)=="function" and type(u.GetSecondsPerAttack)=="function" and u:CanBeSeen() and u:IsAlive() then info.units[#info.units+1]=extra(u,handle,false) end
 end
 for _,team in ipairs({TEAM_RADIANT,TEAM_DIRE}) do for _,which in ipairs({TOWER_MID_1,TOWER_MID_2}) do
  local u=GetTower(team,which);if u then local q=u:GetLocation();info.towers[#info.towers+1]={id=#info.towers+1,x=q.x,y=q.y,z=q.z} else info.towers[#info.towers+1]={id=#info.towers+1} end
 end end
 local home=GetShopLocation(GetTeam(),SHOP_HOME);info.base={id=0,x=home.x,y=home.y,z=home.z}
 local boots=agent.owned(bot,"item_boots")>0;local bracer=agent.owned(bot,"item_bracer")>0
 info.next_permanent=bracer and 2 or (boots and 1 or 0)
 info.next_cost=info.next_permanent<2 and agent.missing_cost(bot,boots and "item_bracer" or "item_boots") or 0
 info.start_bought=agent.owned(bot,"item_wraith_band")>0
 for slot=0,23 do local a=bot:GetAbilityInSlot(slot);if a then info.skill_slots[#info.skill_slots+1]={name=a:GetName(),level=a:GetLevel()} end end
 local body=json.encode(info);local count=math.ceil(#body/1800)
 for part=1,count do print("SF_MODEL|"..GetTeam().."|"..tick.."|"..part.."|"..count.."|"..string.sub(body,(part-1)*1800+1,part*1800)) end
end
return M
