-- This executor has no farming/aiming policy. Teacher decisions live in Python.
local json=require("game/dkjson")
local settings=require("bots/settings")
local spells={"nevermore_shadowraze1","nevermore_shadowraze2","nevermore_shadowraze3","nevermore_requiem"}
local item_names={"item_wraith_band","item_boots","item_bracer","item_tpscroll","item_enchanted_mango","item_flask","item_tango","item_clarity"}
local buys={"none","next_permanent","item_tpscroll","item_enchanted_mango","item_flask","item_tango","item_clarity"}
local components={item_wraith_band={"item_circlet","item_slippers","item_recipe_wraith_band"},
                  item_bracer={"item_circlet","item_gauntlets","item_recipe_bracer"}}
local tower_names={"npc_dota_goodguys_tower1_mid","npc_dota_goodguys_tower2_mid","npc_dota_badguys_tower1_mid","npc_dota_badguys_tower2_mid"}
local ids,registry,next_id={}, {},1
local state={}
local errors={}
local identities={}
local tree_positions={}
local fixed_methods={attack_range="GetAttackRange",attack_point="GetAttackPoint",projectile_speed="GetAttackProjectileSpeed"}
local cast_memory={}
local compare_getters=false
local function call(object,name,...)
    if object==nil then return nil end
    -- Unit and ability classes share names such as GetLevel. Only objects
    -- already identified as units may use a captured unit getter.
    local native=identities[object] and _G.SF_NATIVE_BOT_API and _G.SF_NATIVE_BOT_API[name]
    local method=native or object[name]
    if type(method)~="function" then errors[name]="unavailable";return nil end
    local value
    if native then
        -- The complete decision already has a protected boundary. A native
        -- getter failure rejects that frame/match there, without per-field
        -- protected-call setup for thousands of verified unit reads.
        value=method(object,...)
    else
        local ok,result=pcall(method,object,...)
        if not ok then errors[name]=tostring(result);return nil end
        value=result
    end
    if compare_getters and native and (type(value)=="number" or type(value)=="boolean" or type(value)=="string") then
        local wrapped=object[name]
        if method~=wrapped then
            local same,other=pcall(wrapped,object,...)
            if not same or other~=value then errors["native_getter_"..name]="native/OHA disagreement" end
        end
    end
    return value
end
local function global(name,...)
    local fn=_G[name]
    if type(fn)~="function" then errors[name]="unavailable";return nil end
    return fn(...)
end
local function id(unit)
    if unit==nil then return 0 end
    if ids[unit]==nil then ids[unit]=next_id;registry[next_id]=unit;next_id=next_id+1 end
    return ids[unit]
end
local function mod(unit,name)
    local index=call(unit,"GetModifierByName",name)
    if index~=nil and index>=0 then
        return call(unit,"GetModifierStackCount",index) or 0,call(unit,"GetModifierRemainingDuration",index)
    end
    return 0,0
end
local function unit_data(unit,bot,own)
    if unit==nil or call(unit,"IsNull")==true then return nil end
    local location=call(unit,"GetLocation")
    -- Unit visibility is authoritative. A revealed unit may be visible while
    -- its ground tile remains fogged (e.g. attack vision at a height boundary).
    local visible=own or (call(unit,"CanBeSeen")==true and location~=nil)
    if not visible then
        if settings.observation_profile=="ggd-v4" and cast_memory[unit] then cast_memory[unit].previous=nil;cast_memory[unit].start=nil end
        return nil
    end
    local p=location
    if p==nil then return nil end
    local fixed=identities[unit]
    if not fixed then
        fixed={name=call(unit,"GetUnitName"),team=call(unit,"GetTeam"),hero=call(unit,"IsHero")}
        for key,method in pairs(fixed_methods) do fixed[key]=call(unit,method) end
        if not fixed.hero then
            fixed.mana=call(unit,"GetMana");fixed.mana_max=call(unit,"GetMaxMana");fixed.mana_regen=call(unit,"GetManaRegen")
        end
        identities[unit]=fixed
    end
    if math.floor(DotaTime()*30+.5)%300==0 then
        for key,method in pairs(fixed_methods) do
            if fixed[key]~=call(unit,method) then errors["changed_fixed_"..key]="unexpected stat change" end
        end
        if not fixed.hero then
            for key,method in pairs({mana="GetMana",mana_max="GetMaxMana",mana_regen="GetManaRegen"}) do
                if fixed[key]~=call(unit,method) then errors["changed_creep_"..key]="unexpected stat change" end
            end
        end
    end
    local u={id=id(unit),name=fixed.name,team=fixed.team,visible=true,x=p.x,y=p.y,z=p.z,
        attack_range=fixed.attack_range,attack_point=fixed.attack_point,projectile_speed=fixed.projectile_speed}
    local methods={health="GetHealth",health_max="GetMaxHealth",mana="GetMana",mana_max="GetMaxMana",
        health_regen="GetHealthRegen",mana_regen="GetManaRegen",move_speed="GetCurrentMovementSpeed",
        facing="GetFacing",attack_damage="GetAttackDamage",attack_interval="GetSecondsPerAttack",
        armor="GetArmor",magic_resist="GetMagicResist",level="GetLevel"}
    for k,m in pairs(methods) do
        if not fixed.hero and (k=="mana" or k=="mana_max" or k=="mana_regen") then u[k]=fixed[k]
        else u[k]=call(unit,m) end
    end
    u.fear=call(unit,"HasModifier","modifier_nevermore_requiem_fear")
    u.flask=call(unit,"HasModifier","modifier_flask_healing")
    u.clarity=call(unit,"HasModifier","modifier_clarity_potion")
    u.tango=call(unit,"HasModifier","modifier_tango_heal")
    u.raze_stacks,u.raze_remaining=mod(unit,"modifier_nevermore_shadowraze_debuff")
    local activity=call(unit,"GetAnimActivity")
    u.attack_cycle=call(unit,"GetAnimCycle")
    if settings.observation_profile=="ggd-v3" or settings.observation_profile=="ggd-v4" then
        local last=call(unit,"GetLastAttackTime")
        local attacking=activity==ACTIVITY_ATTACK or activity==ACTIVITY_ATTACK2
        if activity~=nil then u.attack_animation=attacking end
        if type(last)=="number" and last>0 then u.since_attack=math.min(3,math.max(0,GameTime()-last)) end
        if type(last)=="number" and u.attack_interval and u.attack_point then
            local base=fixed.hero and 1.6 or (string.find(fixed.name or "","tower",1,true) and .9 or (string.find(fixed.name or "","siege",1,true) and 3 or 1))
            u.ready_in=last>0 and math.min(3,math.max(0,last+u.attack_interval-u.attack_point*u.attack_interval/base-GameTime())) or 0
        end
    end
    u.is_attacking=activity==ACTIVITY_ATTACK or activity==ACTIVITY_ATTACK2
    u.is_casting_ability=false;u.is_channeling=false
    for index=1,6 do
        if activity==_G["ACTIVITY_CAST_ABILITY_"..index] then u.is_casting_ability=true end
        if activity==_G["ACTIVITY_CHANNEL_ABILITY_"..index] then u.is_channeling=true end
    end
    u.is_using_ability=u.is_casting_ability or u.is_channeling
    if settings.observation_profile=="ggd-v4" and fixed.hero then
        local now=GameTime();local memory=cast_memory[unit] or {last={}}
        -- Sampling gaps can hide both cancellation and a new cast.
        if memory.time and now-memory.time>.2 then memory.previous=nil;memory.start=nil end
        local active=call(unit,"GetCurrentActiveAbility");local casting=call(unit,"IsCastingAbility")
        local spell=nil
        if casting==false then spell=0
        elseif casting==true and active then
            local name=call(active,"GetName")
            for i,n in ipairs(spells) do if name==n then spell=i end end
        end
        local v={};local debuff=call(unit,"GetModifierByName","modifier_nevermore_shadowraze_debuff")
        if debuff~=nil and debuff>=0 then
            v.raze_stacks=call(unit,"GetModifierStackCount",debuff);v.raze_remaining=call(unit,"GetModifierRemainingDuration",debuff)
        elseif debuff~=nil then v.raze_stacks=0;v.raze_remaining=0 end
        if spell~=nil then
            for i=0,4 do v["cast_"..i]=spell==i and 1 or 0 end
            if spell~=memory.previous then
                memory.start=spell>0 and memory.previous~=nil and now or nil
                if memory.start then memory.last[spell]=now end
            end
            if spell==0 then v.cast_phase=0
            elseif memory.start then v.cast_phase=math.min(1,math.max(0,now-memory.start)/(spell==4 and 1.67 or .55)) end
            memory.previous=spell
        else memory.previous=nil;memory.start=nil end
        for i,t in pairs(memory.last) do v["since_cast_"..i]=math.min(60,math.max(0,now-t)) end
        for i,name in ipairs(spells) do v["spell_level_"..i]=call(call(unit,"GetAbilityByName",name),"GetLevel") end
        memory.time=now;cast_memory[unit]=memory;u.v4=v
    end
    if unit==bot then
        u.is_using_ability=call(bot,"IsUsingAbility");u.is_casting_ability=call(bot,"IsCastingAbility")
        u.is_channeling=call(bot,"IsChanneling")
        u.gold=call(unit,"GetGold");u.xp_to_next=call(unit,"GetXPNeededToLevel")
        u.is_attacking=call(bot,"GetCurrentActionType")==BOT_ACTION_TYPE_ATTACK
        u.last_hits=call(unit,"GetLastHits");u.denies=call(unit,"GetDenies")
        u.souls=mod(unit,"modifier_nevermore_necromastery")
        local _,duration=mod(unit,"modifier_nevermore_requiem_fear");u.fear_remaining=duration
        local active=call(unit,"GetCurrentActiveAbility")
        if active~=nil then
            local n=call(active,"GetName")
            for i,name in ipairs(spells) do if n==name then u.active_spell=i end end
        else u.active_spell=0 end
    end
    local target=call(unit,"GetAttackTarget")
    u.attack_target_id=id(target) or 0
    u.attack_target_self=target==bot
    if target==nil then u.attack_target_enemy=false
    elseif call(target,"CanBeSeen") then
        u.attack_target_enemy=call(target,"GetUnitName")=="npc_dota_hero_nevermore" and call(target,"GetTeam")~=call(bot,"GetTeam")
    end
    u.type=fixed.hero and "hero" or "creep"
    return u
end
local function owned(bot,name)
    local count=0
    for slot=0,20 do
        local it=call(bot,"GetItemInSlot",slot)
        if it~=nil and call(it,"GetName")==name then count=count+math.max(1,call(it,"GetCurrentCharges") or 1) end
    end
    local courier=global("GetCourier",0)
    if courier~=nil then
        for slot=0,8 do
            local it=call(courier,"GetItemInSlot",slot)
            if it~=nil and call(it,"GetName")==name then count=count+math.max(1,call(it,"GetCurrentCharges") or 1) end
        end
    end
    return count
end
local function order_item(bot,name)
    -- Assembled items are requested through their legal shop components.
    local list=components[name] or {name}
    local results={};local success=true
    for _,component in ipairs(list) do
        if components[name]==nil or owned(bot,component)==0 then
            local result=call(bot,"ActionImmediate_PurchaseItem",component)
            results[#results+1]=result;success=success and result==PURCHASE_ITEM_SUCCESS
        end
    end
    return results,success
end
local function missing_cost(bot,name)
    if components[name]==nil then return global("GetItemCost",name) or 99999 end
    local cost=0
    for _,part in ipairs(components[name]) do
        if owned(bot,part)==0 then cost=cost+(global("GetItemCost",part) or 99999) end
    end
    return cost
end
local function infrastructure(bot,st)
    if st.pending_permanent~=nil and owned(bot,st.pending_permanent)>0 then
        st.permanent=(st.permanent or 0)+1;st.pending_permanent=nil
    end
    local level=call(bot,"GetLevel") or 1
    local build={spells[3],"nevermore_dark_lord",spells[3],"nevermore_dark_lord",spells[3],spells[4],spells[3],"nevermore_dark_lord","nevermore_dark_lord"}
    if (call(bot,"GetAbilityPoints") or 0)>0 then
        local name=build[st.skill_index or 1]
        if (st.skill_index or 1)==10 then
            name="special_bonus_unique_nevermore_7"
        elseif (st.skill_index or 1)>10 then name=nil end
        if name~=nil then
            local a=call(bot,"GetAbilityByName",name)
            if a~=nil and call(a,"CanAbilityBeUpgraded") then
                local level_before=call(a,"GetLevel") or 0
                call(bot,"ActionImmediate_LevelAbility",name)
                local level_after=call(a,"GetLevel") or 0
                if level_after>level_before then st.skill_index=(st.skill_index or 1)+1 end
            end
        end
    end
    if not st.start_bought then
        local cost=missing_cost(bot,"item_wraith_band")
        if owned(bot,"item_wraith_band")>0 then st.start_bought=true
        elseif (call(bot,"GetGold") or 0)>=cost and DotaTime()>=(st.next_start_attempt or -10000) then
            order_item(bot,"item_wraith_band");st.next_start_attempt=DotaTime()+.5
        end
    end
    if DotaTime()>=0 and DotaTime()>=(st.next_courier or 0) then
        st.next_courier=DotaTime()+4
        local courier=global("GetCourier",0)
        if courier~=nil then
            local loaded=false
            for slot=9,14 do if call(bot,"GetItemInSlot",slot)~=nil then loaded=true end end
            local status=global("GetCourierState",courier)
            if loaded and (status==COURIER_STATE_IDLE or status==COURIER_STATE_AT_BASE) then
                call(bot,"ActionImmediate_Courier",courier,COURIER_ACTION_TAKE_AND_TRANSFER_ITEMS)
            end
        end
    end
end
local function snapshot(bot,st)
    compare_getters=math.floor(DotaTime()*30+.5)%300==0
    local team=call(bot,"GetTeam")
    local own=unit_data(bot,bot,true)
    local p=call(bot,"GetLocation")
    local s={version=1,match=settings.match,team=team,player=call(bot,"GetPlayerID"),
        seq=st.seq,time=DotaTime(),side=team==TEAM_RADIANT and 1 or -1,self=own,
        creeps={},towers={},trees={},projectiles={},skills={},items={},enemy_items={},terrain={{},{},{},{}},
        errors=errors,execution=st.execution or {},next_permanent=st.permanent or 0}
    local enemy_unit=nil
    for _,enemy in ipairs(GetUnitList(UNIT_LIST_ENEMY_HEROES)) do
        if call(enemy,"GetUnitName")=="npc_dota_hero_nevermore" and call(enemy,"CanBeSeen") then
            s.enemy=unit_data(enemy,bot,false);if s.enemy~=nil then enemy_unit=enemy end;break
        end
    end
    local enemy_slots={};local enemy_names={}
    if enemy_unit~=nil then
        for slot=0,8 do
            local it=call(enemy_unit,"GetItemInSlot",slot)
            enemy_slots[slot]=it;enemy_names[slot]=it and call(it,"GetName") or nil
        end
    end
    for i,name in ipairs(item_names) do
        local row={}
        if enemy_unit~=nil and name~="item_tpscroll" then
            row.inventory=0;row.backpack=0
            for slot=0,8 do
                local it=enemy_slots[slot]
                if it~=nil and enemy_names[slot]==name then
                    local key=slot<=5 and "inventory" or "backpack"
                    row[key]=row[key]+math.max(1,call(it,"GetCurrentCharges") or 1)
                end
            end
        end
        s.enemy_items[i]=row
    end
    for _,kind in ipairs({UNIT_LIST_ALLIED_CREEPS,UNIT_LIST_ENEMY_CREEPS}) do
        for _,creep in ipairs(GetUnitList(kind)) do
            local name=call(creep,"GetUnitName") or ""
            local lane_creep=string.find(name,"npc_dota_creep_")==1 or name=="npc_dota_goodguys_siege" or name=="npc_dota_badguys_siege"
            if lane_creep and call(creep,"CanBeSeen") and call(creep,"IsAlive") then
                local u=unit_data(creep,bot,false)
                if u~=nil then
                    u.type="creep";u.creep_melee=string.find(name,"melee")~=nil
                    u.creep_ranged=string.find(name,"ranged")~=nil;u.creep_siege=string.find(name,"siege")~=nil
                    u.attackable=u.team~=team or (u.health or 0)<=(u.health_max or 0)*0.5
                    s.creeps[#s.creeps+1]=u
                end
            end
        end
    end
    table.sort(s.creeps,function(a,b) return a.id<b.id end)
    for i,name in ipairs(tower_names) do
        local tower_team=i<=2 and TEAM_RADIANT or TEAM_DIRE
        local which=(i==1 or i==3) and TOWER_MID_1 or TOWER_MID_2
        local tower=GetTower(tower_team,which)
        local pos=tower and call(tower,"GetLocation")
        local row=unit_data(tower,bot,tower_team==team) or {name=name,team=tower_team,visible=false}
        row.name=name;row.type="tower";row.id=id(tower)
        if tower_team==team and (tower==nil or call(tower,"IsAlive")==false) then row.health=0 end
        if pos~=nil then row.x=pos.x;row.y=pos.y;row.z=pos.z end
        row.tp=tower_team==team and tower~=nil and call(tower,"IsAlive")==true
        row.attackable=tower_team~=team and row.visible and (row.health or 0)>0
        s.towers[i]=row
    end
    -- TP home means the fountain/home shop, not the ancient. The latter is
    -- over 1024 units away on one axis and cannot represent a fountain TP.
    local base=global("GetShopLocation",team,SHOP_HOME)
    s.base={id=-1,type="base",tp=true,x=base.x,y=base.y,z=base.z,team=team}
    for _,tree in ipairs(call(bot,"GetNearbyTrees",1600) or {}) do
        local loc=tree_positions[tree]
        if not loc then loc=GetTreeLocation(tree);tree_positions[tree]=loc end
        if global("IsLocationVisible",loc) then
            s.trees[#s.trees+1]={id=tree,type="tree",x=loc.x,y=loc.y,z=loc.z}
        end
    end
    table.sort(s.trees,function(a,b) return a.id<b.id end)
    for channel=1,4 do for y=1,8 do s.terrain[channel][y]={} end end
    for y=1,8 do for x=1,8 do
        local loc=Vector(p.x+(x-4.5)*256,p.y+(y-4.5)*256,p.z)
        local visible=global("IsLocationVisible",loc)==true
        local height=global("GetHeightLevel",loc)
        local passable=nil
        if visible then passable=global("IsLocationPassable",loc) end
        s.terrain[1][y][x]=height and height/5 or 0
        s.terrain[2][y][x]=passable and 1 or 0
        s.terrain[3][y][x]=visible and 1 or 0
        s.terrain[4][y][x]=0
    end end
    for _,u in ipairs(s.creeps) do
        local x=math.floor((u.x-p.x)/256+4.5)+1;local y=math.floor((u.y-p.y)/256+4.5)+1
        if x>=1 and x<=8 and y>=1 and y<=8 then s.terrain[4][y][x]=s.terrain[4][y][x]+(u.team==team and 0.25 or -0.25) end
    end
    local projectile_targets={bot}
    if enemy_unit~=nil then projectile_targets[#projectile_targets+1]=enemy_unit end
    for _,u in ipairs(s.creeps) do projectile_targets[#projectile_targets+1]=registry[u.id] end
    for _,unit in ipairs(projectile_targets) do
        for _,projectile in ipairs(call(unit,"GetIncomingTrackingProjectiles") or {}) do
            local loc=projectile.location
            if loc~=nil and global("IsLocationVisible",loc) then
                local caster=projectile.caster
                local speed=projectile.velocity
                if speed==nil and projectile.is_attack and caster~=nil and call(caster,"CanBeSeen") then
                    speed=call(caster,"GetAttackProjectileSpeed")
                end
                s.projectiles[#s.projectiles+1]={x=loc.x,y=loc.y,speed=speed,
                    target_id=id(unit),caster_id=id(caster),
                    is_attack=projectile.is_attack,target_self=unit==bot,
                    caster_enemy=caster~=nil and call(caster,"GetTeam")~=team}
            end
        end
    end
    for i,name in ipairs(spells) do
        local a=call(bot,"GetAbilityByName",name)
        s.skills[i]={level=call(a,"GetLevel"),cooldown=call(a,"GetCooldownTimeRemaining"),
            mana_cost=call(a,"GetManaCost"),cast_point=call(a,"GetCastPoint"),
            castable=call(a,"IsFullyCastable"),channel_time=call(a,"GetChannelTime")}
    end
    local courier=global("GetCourier",0)
    local own_slots={};local own_names={};local courier_slots={};local courier_names={}
    for slot=0,20 do
        local it=call(bot,"GetItemInSlot",slot)
        own_slots[slot]=it;own_names[slot]=it and call(it,"GetName") or nil
    end
    if courier~=nil then
        for slot=0,8 do
            local it=call(courier,"GetItemInSlot",slot)
            courier_slots[slot]=it;courier_names[slot]=it and call(it,"GetName") or nil
        end
    end
    local any_item=false
    for i,name in ipairs(item_names) do
        local row={inventory=0,backpack=0,stash=0,courier=0,castable=false,slot=-1}
        for slot=0,20 do
            local it=own_slots[slot]
            if it~=nil and own_names[slot]==name then
                local count=math.max(1,call(it,"GetCurrentCharges") or 1)
                local key=slot<=5 and "inventory" or (slot<=8 and "backpack" or (slot<=14 and "stash" or "inventory"))
                row[key]=row[key]+count
                if slot<=5 or slot==15 or slot==16 then
                    row.slot=slot;row.cooldown=call(it,"GetCooldownTimeRemaining");row.mana_cost=call(it,"GetManaCost")
                    row.castable=call(it,"IsFullyCastable")==true and i>=4
                end
            end
        end
        if courier~=nil then
            for slot=0,8 do
                local it=courier_slots[slot]
                if it~=nil and courier_names[slot]==name then row.courier=row.courier+math.max(1,call(it,"GetCurrentCharges") or 1) end
            end
        end
        if i==7 and #s.trees==0 then row.castable=false end
        any_item=any_item or row.castable
        s.items[i]=row
    end
    local actions={true,true,true,false,true,true,false,false,false,false,any_item}
    actions[6]=type(bot.Action_HoldPosition)=="function"
    for _,u in ipairs(s.creeps) do if u.attackable then actions[4]=true end end
    if s.enemy~=nil and (s.enemy.health or 0)>0 then s.enemy.attackable=true;actions[4]=true end
    for _,u in ipairs(s.towers) do if u.attackable then actions[4]=true end end
    for i=1,4 do actions[6+i]=s.skills[i].castable==true and (s.skills[i].level or 0)>0 end
    if own.fear or own.health<=0 then for i=2,11 do actions[i]=false end end
    local item_mask={}
    for i,row in ipairs(s.items) do item_mask[i]=row.castable end
    local buy_mask={true,false,false,false,false,false,false}
    local gold=own.gold or 0
    local next_name=(st.permanent or 0)==0 and "item_boots" or ((st.permanent or 0)==1 and "item_bracer" or nil)
    if next_name~=nil and st.pending_permanent==nil then buy_mask[2]=gold>=missing_cost(bot,next_name) end
    for i=3,7 do
        local name=buys[i];local stock=global("GetItemStockCount",name)
        buy_mask[i]=gold>=(global("GetItemCost",name) or 99999) and (stock==nil or stock~=0)
    end
    if not st.start_bought then for i=2,7 do buy_mask[i]=false end end
    s.legal={actions=actions,items=item_mask,buys=buy_mask}
    s.probe={ability_points=call(bot,"GetAbilityPoints"),skill_index=st.skill_index,
        game_mode=GetGameMode(),raze_upgradeable=call(call(bot,"GetAbilityByName",spells[3]),"CanAbilityBeUpgraded")}
    if not st.cached_skill_slots or st.cached_skill_index~=st.skill_index or st.cached_skill_level~=own.level then
        st.cached_skill_slots={}
        for slot=0,23 do
            local a=call(bot,"GetAbilityInSlot",slot)
            st.cached_skill_slots[#st.cached_skill_slots+1]={slot=slot,name=call(a,"GetName"),level=call(a,"GetLevel")}
        end
        st.cached_skill_index=st.skill_index;st.cached_skill_level=own.level
    end
    s.probe.skill_slots=st.cached_skill_slots
    s.probe.courier_team=call(courier,"GetTeam")
    s.probe.courier_state=courier~=nil and global("GetCourierState",courier) or nil
    s.probe.allied_heroes={}
    for _,hero in ipairs(GetUnitList(UNIT_LIST_ALLIED_HEROES)) do
        s.probe.allied_heroes[#s.probe.allied_heroes+1]={name=call(hero,"GetUnitName"),player=call(hero,"GetPlayerID"),
            alive=call(hero,"IsAlive"),health=call(hero,"GetHealth"),null=call(hero,"IsNull")}
    end
    s.probe.inventory={}
    for slot=0,20 do
        local it=own_slots[slot]
        if it~=nil then s.probe.inventory[#s.probe.inventory+1]={slot=slot,name=own_names[slot]} end
    end
    compare_getters=false
    return s
end
local function execute(bot,st,response)
    local a=response.action or 0
    local ack={seq=response.seq,action=a,accepted=true}
    local target=response.target_id and registry[response.target_id] or nil
    local p=bot:GetLocation()
    if a==1 then bot:Action_MoveToLocation(Vector(response.x,response.y,p.z))
    elseif a==2 then bot:Action_MoveDirectly(Vector(response.x,response.y,p.z))
    elseif a==3 then
        if target~=nil and call(target,"IsAlive") and call(target,"CanBeSeen") then
            bot:Action_AttackUnit(target,true);ack.target_id=response.target_id;ack.target_name=call(target,"GetUnitName")
        else ack.accepted=false;ack.reason="target_no_longer_visible_or_alive" end
    elseif a==4 then bot:Action_ClearActions(true)
    elseif a==5 then
        if type(bot.Action_HoldPosition)=="function" then bot:Action_HoldPosition()
        else ack.accepted=false;ack.reason="hold_not_exposed_by_bot_api" end
    elseif a>=6 and a<=9 then
        local ability=bot:GetAbilityByName(spells[a-5])
        if ability~=nil and ability:IsFullyCastable() and ability:GetLevel()>0 then bot:Action_UseAbility(ability)
        else ack.accepted=false;ack.reason="spell_unavailable" end
    elseif a==10 then
        local name=item_names[(response.item or 0)+1]
        local item=nil
        for slot=0,20 do local it=bot:GetItemInSlot(slot);if it~=nil and it:GetName()==name and it:IsFullyCastable() then item=it;break end end
        if item==nil then ack.accepted=false;ack.reason="item_unavailable"
        elseif name=="item_tpscroll" then bot:Action_UseAbilityOnLocation(item,Vector(response.x,response.y,0))
        elseif name=="item_tango" then bot:Action_UseAbilityOnTree(item,response.target_id)
        elseif name=="item_enchanted_mango" or name=="item_flask" or name=="item_clarity" then bot:Action_UseAbilityOnEntity(item,bot)
        end
    end
    local buy=response.buy or 0
    if buy==1 and (st.permanent or 0)<2 and st.pending_permanent==nil then
        local name=(st.permanent or 0)==0 and "item_boots" or "item_bracer"
        local cost=missing_cost(bot,name)
        if bot:GetGold()>=cost then
            local success
            ack.purchase,success=order_item(bot,name)
            if not success then ack.purchase_rejected="engine_purchase_rejected" end
            if owned(bot,name)>0 then st.permanent=(st.permanent or 0)+1
            elseif success then st.pending_permanent=name end
        else ack.purchase_rejected="not_enough_gold" end
    elseif buy>=2 and buy<=6 then
        local name=buys[buy+1]
        if bot:GetGold()>=GetItemCost(name) then
            local success
            ack.purchase,success=order_item(bot,name)
            if not success then ack.purchase_rejected="engine_purchase_rejected" end
        else ack.purchase_rejected="not_enough_gold" end
    end
    ack.elapsed_game_seconds=response.observed_time and DotaTime()-response.observed_time or nil
    ack.rpc_wall_seconds=response.rpc_wall_seconds
    st.execution=ack
end
local function Think()
    local phase=GetGameState()
    if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
    local bot=GetBot()
    if bot==nil or bot:GetUnitName()~="npc_dota_hero_nevermore" then return end
    local pid=bot:GetPlayerID()
    local st=state[pid]
    if st==nil then st={seq=0,next_tick=math.floor(DotaTime()*30+.5)+6,skill_index=1,permanent=0};state[pid]=st end
    if st.ended then return end
    local tower=GetTower(bot:GetTeam(),TOWER_MID_1)
    local tower_alive=tower~=nil and call(tower,"IsAlive")==true
    if tower_alive then st.tower_seen=true end
    local reason=nil
    if DotaTime()>=0 then
        if bot:GetHealth()<=0 then reason="death"
        elseif st.tower_seen and not tower_alive then reason="tower"
        elseif DotaTime()>=600 then reason="timeout" end
    end
    if reason~=nil then
        st.ended=true
        print("SF_END|"..json.encode({match=settings.match,team=bot:GetTeam(),time=DotaTime(),reason=reason,
            last_hits=bot:GetLastHits(),denies=bot:GetDenies(),health=bot:GetHealth()}))
        return
    end
    infrastructure(bot,st)
    if st.rule_error~=nil then st.ended=true;return end
    if st.busy then
        local module="bots/response_"..settings.match.."_"..pid
        package.loaded[module]=nil
        local loaded,response=pcall(require,module)
        if loaded and type(response)=="table" and response.match==settings.match and response.seq==st.seq then
            st.busy=false
            if response.observed_time~=nil and DotaTime()-response.observed_time>0.5 then
                st.execution={seq=st.seq,accepted=false,reason="late_response"}
            else
                local success,why=pcall(execute,bot,st,response)
                if not success then st.execution={seq=st.seq,accepted=false,reason=tostring(why)};print("SF_EXECUTION_ERROR",pid,why) end
            end
        end
    end
    local tick=math.floor(DotaTime()*30+.5)
    if st.busy or tick<st.next_tick then return end
    local period=math.max(1,math.floor(settings.interval*30+.5))
    st.seq=st.seq+1;st.next_tick=(math.floor(tick/period)+1)*period
    local ok,data=pcall(snapshot,bot,st)
    if not ok then print("SF_SNAPSHOT_ERROR",pid,tostring(data));return end
    local body=json.encode(data)
    local count=math.ceil(#body/2000)
    for part=1,count do
        print("SF_OBS|"..pid.."|"..st.seq.."|"..part.."|"..count.."|"..string.sub(body,(part-1)*2000+1,part*2000))
    end
    st.busy=true
end

return {Think=Think, infrastructure=infrastructure, snapshot=snapshot, owned=owned, order_item=order_item,
    public_spells=function(unit,bot) local u=unit_data(unit,bot,unit==bot);return u and u.v4 end,
    missing_cost=missing_cost, identify=id, execute=execute, resolve=function(identity) return registry[identity] end}
