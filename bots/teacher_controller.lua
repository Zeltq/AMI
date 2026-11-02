-- Fully scripted OHA teacher. The sole clock and executor match the policy.
local agent=require("bots/agent")
local monitor=require("bots/teacher_monitor")
local settings=require("bots/settings")
local json=require("game/dkjson")
local telemetry=require("bots/telemetry")
local J=require(GetScriptDirectory().."/FunLib/jmz_func")
local lane=dofile(GetScriptDirectory().."/FunLib/override_generic/mode_laning_generic")
local hero=dofile(GetScriptDirectory().."/BotLib/hero_nevermore")
local items=require(GetScriptDirectory().."/ability_item_usage_generic")
local spell_names={nevermore_shadowraze1=6,nevermore_shadowraze2=7,nevermore_shadowraze3=8,nevermore_requiem=9}
local item_names={item_tpscroll=3,item_enchanted_mango=4,item_flask=5,item_tango=6,item_clarity=7}
local native={}
local collecting=false
local executing=false
local raw={}
local intent=nil
local sequence=0
local last_tick=nil
local attack_until=-10000
local current_mode=BOT_MODE_LANING
local issued=0
local receipts={}
local purchase_window=false
local function fail(reason) error("teacher controller: "..reason) end
local function serial(value)
    if type(value)=="number" or type(value)=="boolean" or type(value)=="string" then return value end
    if value==nil then return nil end
    if type(value.GetName)=="function" then return {name=value:GetName()} end
    if type(value.GetUnitName)=="function" then return {name=value:GetUnitName(),id=agent.identify(value)} end
    if value.x~=nil and value.y~=nil then return {x=value.x,y=value.y,z=value.z} end
    fail("cannot serialize action argument")
end
local function capture(self,name,...)
    if not collecting then fail("action outside decision: "..name) end
    local args={...};local row={method=name,arguments={}}
    for i,v in ipairs(args) do row.arguments[i]=serial(v) end
    raw[#raw+1]=row
    local kind=string.gsub(name,"^ActionQueue_", "Action_")
    kind=string.gsub(kind,"^ActionPush_", "Action_")
    if kind=="Action_ClearActions" then
        if args[1] then intent={action=4} end
    elseif kind=="Action_MoveToLocation" or kind=="Action_MoveDirectly" then
        intent={action=kind=="Action_MoveDirectly" and 2 or 1,location=args[1],source=kind}
    elseif kind=="Action_MoveToUnit" then
        intent={action=1,location=args[1]:GetLocation()}
    elseif kind=="Action_AttackUnit" then
        if args[1]:GetTeam()==TEAM_NEUTRAL then fail("neutral target") end
        intent={action=3,target_id=agent.identify(args[1]),target=args[1]}
    elseif kind=="Action_UseAbility" or kind=="Action_UseAbilityOnEntity" or kind=="Action_UseAbilityOnTree" or kind=="Action_UseAbilityOnLocation" then
        local ability=args[1];local name=ability:GetName()
        if spell_names[name] then
            if kind~="Action_UseAbility" then fail("unexpected targeted SF spell") end
            intent={action=spell_names[name]}
        elseif item_names[name] then
            intent={action=10,item=item_names[name]}
            if name=="item_tpscroll" then intent.location=args[2]
            elseif name=="item_tango" then intent.target_id=args[2]
            elseif kind~="Action_UseAbility" and args[2]~=self then
                fail("consumable target must be self: "..kind.." "..name)
            end
        else fail("unsupported ability: "..name) end
    else fail("unsupported order: "..name) end
end
-- All executable hero orders go through the shared executor below. Native mode
-- AI is disabled by bot_generic. Unexpected Lua orders fail instead of vanishing.
for name,fn in pairs(CDOTA_Bot_Script) do
    if type(fn)=="function" and (string.match(name,"^Action_") or string.match(name,"^ActionPush_") or string.match(name,"^ActionQueue_")) then
        native[name]=fn
    end
end
for name,fn in pairs(native) do
    CDOTA_Bot_Script[name]=function(self,...)
        if executing then issued=issued+1;return fn(self,...) end
        return capture(self,name,...)
    end
end
do
    local purchase=CDOTA_Bot_Script.ActionImmediate_PurchaseItem
    function CDOTA_Bot_Script:ActionImmediate_PurchaseItem(name)
        if collecting then fail("upstream purchase outside shared shop") end
        local result=purchase(self,name)
        if purchase_window then receipts[#receipts+1]={name=name,result=result} end
        return result
    end
end
for name,fn in pairs(CDOTA_Bot_Script) do
    if type(fn)=="function" and string.match(name,"^ActionImmediate_")
        and name~="ActionImmediate_PurchaseItem" and name~="ActionImmediate_LevelAbility"
        and name~="ActionImmediate_Courier" and name~="ActionImmediate_Chat" and name~="ActionImmediate_Ping" then
        CDOTA_Bot_Script[name]=function(self,...)
            fail("unsupported immediate order: "..name)
        end
    end
end
local original_mode=CDOTA_Bot_Script.GetActiveMode
function CDOTA_Bot_Script:GetActiveMode()
    if self==GetBot() then return current_mode end
    return original_mode(self)
end
local function grid(delta) return math.max(0,math.min(128,math.floor(delta/16+64.5))) end
local function command_for(s,choice)
    choice=choice or {action=0}
    local command={seq=s.seq,match=s.match,action=choice.action,buy=0,item=choice.item or 0,
        x=64,y=64,target_id=choice.target_id,observed_time=s.time,
        target_type=(choice.action==10 and choice.item==6) and "tree" or "unit"}
    if choice.action==1 or choice.action==2 then
        local dx,dy=choice.location.x-s.self.x,choice.location.y-s.self.y
        local scale=math.max(1,math.abs(dx)/1024,math.abs(dy)/1024)
        command.x=grid(dx/scale);command.y=grid(dy/scale)
    elseif choice.action==10 and choice.item==3 then
        local refs={s.base,s.towers[1],s.towers[2],s.towers[3],s.towers[4]}
        local best,dist=nil,math.huge
        for _,ref in ipairs(refs) do
            if ref.tp then
                local distance=(ref.x-choice.location.x)^2+(ref.y-choice.location.y)^2
                if distance<dist then best,dist=ref,distance end
            end
        end
        if not best or math.abs(choice.location.x-best.x)>1024 or math.abs(choice.location.y-best.y)>1024 then
            fail("TP destination outside policy action space")
        end
        command.target_id=best.id
        command.x=grid(choice.location.x-best.x);command.y=grid(choice.location.y-best.y)
    end
    return command
end
local function execute(s,command)
    local order={}
    for k,v in pairs(command) do order[k]=v end
    if command.action==1 or command.action==2 then
        order.x=s.self.x+(command.x-64)*16;order.y=s.self.y+(command.y-64)*16
    elseif command.action==10 and command.item==3 then
        local ref=s.base
        for _,tower in ipairs(s.towers) do if tower.id==command.target_id then ref=tower end end
        order.x=ref.x+(command.x-64)*16;order.y=ref.y+(command.y-64)*16
    end
    executing=true
    agent.execute(GetBot(),monitor.state,order)
    executing=false
    if command.action==3 then
        local bot=GetBot()
        attack_until=DotaTime()+bot:GetAttackPoint()/bot:GetAttackSpeed()+1/30
    end
end
local function emit(s)
    local body=json.encode(telemetry.pack(s));local count=math.ceil(#body/2000)
    for part=1,count do
        print("SF_OBS|"..s.player.."|"..s.seq.."|"..part.."|"..count.."|"..string.sub(body,(part-1)*2000+1,part*2000))
    end
end
local M={}
function M.Think()
    local bot=GetBot()
    if not bot or bot:GetUnitName()~="npc_dota_hero_nevermore" then return end
    local phase=GetGameState()
    if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
    local tick=math.floor(DotaTime()*30+.5)
    local period=math.floor(settings.interval*30+.5)
    if period<1 or math.abs(period-settings.interval*30)>.00001 then fail("frequency must divide 30 Hz") end
    if tick%period~=0 or tick==last_tick then return end
    last_tick=tick
    monitor.tick(true)
    if not bot:IsAlive() or DotaTime()>=600 then return end
    sequence=sequence+1
    local s=monitor.current_tick==tick and monitor.current_snapshot or agent.snapshot(bot,monitor.state)
    s.seq=sequence
    issued=0;receipts={}
    raw={};intent=nil;collecting=true
    local hp=bot:GetHealth()/bot:GetMaxHealth()
    current_mode=hp<.25 and BOT_MODE_RETREAT or BOT_MODE_LANING
    local harass=nil
    if hp>.5 and #bot:GetNearbyTowers(800,true)==0 then
        for _,enemy in ipairs(bot:GetNearbyHeroes(bot:GetAttackRange(),true,BOT_MODE_NONE)) do
            if enemy:CanBeSeen() and enemy:IsAlive() and not enemy:IsAttackImmune()
                and hp>=enemy:GetHealth()/enemy:GetMaxHealth()-.15 then
                harass=enemy;current_mode=BOT_MODE_ATTACK;break
            end
        end
    end
    -- Only the first selected combat/item action is executed at this epoch.
    if not s.self.fear and not bot:IsUsingAbility() and not bot:IsChanneling() then
        items.DuelItemUsageThink()
        if not intent then hero.SkillsComplement() end
        if not intent and DotaTime()>=attack_until then
            if current_mode==BOT_MODE_RETREAT then
                local fountain=GetAncient(GetTeam()):GetLocation()
                bot:Action_MoveToLocation(fountain)
            else
                lane.GetDesire();lane.Think()
                -- Farm/deny choices retain priority. Harass only replaces idle
                -- positioning, with a visible hero already in attack range.
                if harass and intent and intent.source=="Action_MoveToLocation" then
                    bot:Action_AttackUnit(harass,true)
                end
            end
        end
    end
    collecting=false
    local command=command_for(s,intent)
    execute(s,command)
    -- Purchase uses the observation's pre-purchase masks; components of a fixed
    -- item form one next_permanent decision. WB/skills/courier are infrastructure.
    purchase_window=true
    local buy=monitor.purchase(bot)
    purchase_window=false
    command.buy=buy
    s.demonstration={command=command,raw_actions=raw,execution=monitor.state.execution,
        decision_tick=tick,period_ticks=period,controller="oha-lua-1",issued_hero_orders=issued,
        purchase_receipts=receipts}
    if settings.collect_teacher then emit(s) end
end
return M
