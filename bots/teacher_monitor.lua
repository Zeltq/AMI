-- Rule adapter and sparse diagnostics only. No demonstration/action collection.
local agent=require("bots/agent")
local json=require("game/dkjson")
local settings=require("bots/settings")
local st={skill_index=1,permanent=0,seq=0}
local last_tick=nil
local next_status=-10000
local next_shop=-10000
local ended={}
local tick_calls=0
local max_tick_gap=0
local last_game_time=nil
local allowed={item_wraith_band=true,item_boots=true,item_bracer=true,item_tpscroll=true,
 item_enchanted_mango=true,item_flask=true,item_tango=true,item_clarity=true,
 item_circlet=true,item_slippers=true,item_gauntlets=true,item_recipe_bracer=true,item_recipe_wraith_band=true}
local M={}
local function violation(reason)
    print("SF_RULE_ERROR|"..json.encode({reason=reason,time=DotaTime(),team=GetTeam()}))
end
-- Enforce restrictions at the Lua action boundary, including upstream helpers.
for _,name in ipairs({"Action_AttackUnit","ActionPush_AttackUnit","ActionQueue_AttackUnit"}) do
    local native=CDOTA_Bot_Script[name]
    if type(native)=="function" then
        CDOTA_Bot_Script[name]=function(self,target,...)
            if target and target:GetTeam()==TEAM_NEUTRAL then violation("blocked_neutral_attack");return end
            return native(self,target,...)
        end
    end
end
for _,name in ipairs({"Action_PickUpRune","ActionPush_PickUpRune","ActionQueue_PickUpRune"}) do
    if type(CDOTA_Bot_Script[name])=="function" then
        CDOTA_Bot_Script[name]=function() violation("blocked_rune_pickup") end
    end
end
do
    local native=CDOTA_Bot_Script.ActionImmediate_PurchaseItem
    function CDOTA_Bot_Script:ActionImmediate_PurchaseItem(name)
        if not allowed[name] then violation("blocked_purchase:"..name);return PURCHASE_ITEM_DISALLOWED_ITEM end
        return native(self,name)
    end
end
for _,name in ipairs({"Action_UseAbilityOnEntity","ActionPush_UseAbilityOnEntity","ActionQueue_UseAbilityOnEntity"}) do
    local native=CDOTA_Bot_Script[name]
    if type(native)=="function" then
        CDOTA_Bot_Script[name]=function(self,ability,target,...)
            local item=ability:GetName()
            if (item=="item_flask" or item=="item_clarity" or item=="item_enchanted_mango") and target~=self then
                violation("blocked_ally_consumable");return
            end
            return native(self,ability,target,...)
        end
    end
end

local function shop(bot)
    if not st.start_bought then return end
    local hp=bot:GetHealth()/bot:GetMaxHealth()
    local mana_missing=bot:GetMaxMana()-bot:GetMana()
    local hp_missing=bot:GetMaxHealth()-bot:GetHealth()
    local function buy(name)
        if bot:GetGold()>=GetItemCost(name) and GetItemStockCount(name)~=0 then
            local _,success=agent.order_item(bot,name)
            return success
        end
        return false
    end
    if DotaTime()<0 and agent.owned(bot,"item_tango")==0 then return buy("item_tango") and 5 or 0 end
    if hp_missing>180 and agent.owned(bot,"item_flask")==0 and not bot:HasModifier("modifier_flask_healing") then
        if buy("item_flask") then return 4 end
    end
    if mana_missing>120 and agent.owned(bot,"item_enchanted_mango")==0 then
        if buy("item_enchanted_mango") then return 3 end
    end
    if hp_missing>100 and agent.owned(bot,"item_tango")==0 and not bot:HasModifier("modifier_tango_heal") then
        if buy("item_tango") then return 5 end
    end
    if mana_missing>180 and agent.owned(bot,"item_clarity")==0 then
        if buy("item_clarity") then return 6 end
    end
    local name=st.permanent==0 and "item_boots" or (st.permanent==1 and "item_bracer" or nil)
    if name and st.pending_permanent==nil then
        local reserve=hp<.65 and 110 or 0
        if bot:GetGold()>=agent.missing_cost(bot,name)+reserve then
            local _,success=agent.order_item(bot,name)
            if success then st.pending_permanent=name;return 1 end
        end
    end
end

function M.tick(skip_shop)
    local bot=GetBot()
    if bot==nil or bot:GetUnitName()~="npc_dota_hero_nevermore" then return end
    local phase=GetGameState()
    if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
    local tick=math.floor(DotaTime()*30+.5)
    if tick==last_tick then return end
    last_tick=tick
    tick_calls=tick_calls+1
    if last_game_time then max_tick_gap=math.max(max_tick_gap,DotaTime()-last_game_time) end
    last_game_time=DotaTime()
    agent.infrastructure(bot,st)
    local team=bot:GetTeam()
    local own_t1=GetTower(team,TOWER_MID_1)
    if own_t1 and own_t1:IsAlive() then st.tower_seen=true end
    if DotaTime()>=0 then
        local reason=nil
        if not bot:IsAlive() then reason="death"
        elseif st.tower_seen and (own_t1==nil or not own_t1:IsAlive()) then reason="tower"
        elseif DotaTime()>=600 then reason="timeout" end
        if reason and not ended[team] then
            ended[team]=true
            print("SF_END|"..json.encode({match=settings.match,team=team,time=DotaTime(),reason=reason,
                last_hits=bot:GetLastHits(),denies=bot:GetDenies(),health=bot:GetHealth()}))
        end
        -- The surviving bot can still report the first death if dead callbacks stop.
        for _,pid in ipairs(GetTeamPlayers(GetOpposingTeam())) do
            if GetSelectedHeroName(pid)=="npc_dota_hero_nevermore" and GetHeroDeaths(pid)>0 and not ended[5-team] then
                ended[5-team]=true
                print("SF_END|"..json.encode({match=settings.match,team=5-team,time=DotaTime(),reason="death"}))
            end
        end
    end
    if not skip_shop and bot:IsAlive() and DotaTime()>=next_shop then next_shop=DotaTime()+.5;shop(bot) end
    if DotaTime()<next_status then return end
    next_status=DotaTime()+.5;st.seq=st.seq+1
    local s=agent.snapshot(bot,st)
    M.current_snapshot=s;M.current_tick=tick
    local target=bot:GetAttackTarget()
    local target_name=target and target:GetUnitName() or nil
    local bad=nil
    if target_name and string.find(target_name,"npc_dota_neutral_")==1 then bad="neutral_target" end
    for _,row in ipairs(s.probe.inventory) do if not allowed[row.name] then bad="forbidden_item:"..row.name end end
    for _,row in ipairs(s.probe.skill_slots) do
        if row.name=="nevermore_frenzy" and row.level>0 then bad="frenzy_learned" end
        if row.name=="special_bonus_attributes" and row.level>0 then bad="attributes_learned" end
    end
    if bad then print("SF_RULE_ERROR|"..json.encode({reason=bad,team=team,time=DotaTime()})) end
    print("SF_STATUS|"..json.encode({time=DotaTime(),team=team,hero=s.self,inventory=s.probe.inventory,
        skills=s.probe.skill_slots,api_errors=s.errors,mode=bot:GetActiveMode(),target=target_name,
        courier=s.probe.courier_state,roster=s.probe.allied_heroes,
        schedule={calls=tick_calls,max_game_gap=max_tick_gap}}))
    tick_calls=0;max_tick_gap=0
end
function M.purchase(bot)
    if DotaTime()<next_shop then return 0 end
    next_shop=DotaTime()+.5
    return shop(bot) or 0
end
M.state=st
return M
