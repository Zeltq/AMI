-- Privileged fixture setup/truth only. No server fields are policy features.
local ok,err=pcall(function()
 local json=require('game/dkjson')
 local config=require('bots/p15_config')
 local heroes={}
 for _,h in pairs(Entities:FindAllByClassname('npc_dota_hero_nevermore')) do heroes[h:GetTeamNumber()]=h end
 assert(heroes[2] and heroes[3],'missing heroes')
 print('SF_P15_API|'..json.encode({move_direction=DOTA_UNIT_ORDER_MOVE_TO_DIRECTION,move_position=DOTA_UNIT_ORDER_MOVE_TO_POSITION,
   order_filter_available=GameRules:GetGameModeEntity()~=nil}))
 local start=math.ceil(GameRules:GetGameTime()*30)+150
 print('SF_P15_START|'..json.encode({start_tick=start}))
 local previous=-1;local last_tick=-1;local blockers={};local pending={}
 heroes[2]:SetContextThink('p15_direct',function()
  local success,failure=pcall(function()
   local tick=math.floor(GameRules:GetGameTime()*30+.5)
   if tick==last_tick then return end
   last_tick=tick
   local elapsed=tick-start
   if elapsed<0 then return end
   local index=math.floor(elapsed/config.case_ticks)+1
   local c=config.cases[index]
   if not c then print('SF_P15_DONE|true');return 'done' end
   local ct=elapsed%config.case_ticks
   if previous~=index then
    previous=index
    for _,u in pairs(Entities:FindAllByClassname('npc_dota_creep_lane')) do UTIL_Remove(u) end
    blockers={};pending={}
    for team,h in pairs(heroes) do
     h:Stop();h:SetAbsOrigin(GetGroundPosition(Vector(team==2 and c.x or (c.name=='enemy' and c.x+90 or 5000),team==2 and c.y or (c.name=='enemy' and c.y or 5000),0),h))
     local a=c.facing*math.pi/180;h:SetForwardVector(Vector(math.cos(a),math.sin(a),0))
     h:AddNewModifier(h,nil,'modifier_invulnerable',{});h:SetHealth(h:GetMaxHealth())
    end
    if c.name=='ally' or c.name=='wave' then
     local n=c.name=='ally' and 1 or 4
     for i=1,n do
      local team=c.name=='ally' and 2 or 3
      local u=CreateUnitByName(team==2 and 'npc_dota_creep_goodguys_melee' or 'npc_dota_creep_badguys_melee',Vector(c.x+90+(i-1)*70,c.y,0),true,nil,nil,team)
      u:SetIdleAcquire(false);u:SetAcquisitionRange(0);u:AddNewModifier(u,nil,'modifier_invulnerable',{})
      u:Hold()
      blockers[#blockers+1]=u
     end
    end
    print('SF_P15_CASE|'..json.encode({case=index,tick=tick,name=c.name,repeat_index=c['repeat']}))
   end
   if ct<=37 then
    local a=c.facing*math.pi/180
    heroes[2]:SetAngles(0,c.facing,0);heroes[2]:SetForwardVector(Vector(math.cos(a),math.sin(a),0))
   end
   if c.name=='wave' and ct==37 then
    for _,u in ipairs(blockers) do u:MoveToPosition(Vector(c.x-1000,c.y,0)) end
   end
   if config.server_direction then
    if ct>=30 and ct<156 and (ct-30)%6==0 then
     local p=heroes[2]:GetAbsOrigin();local a=c.direction*math.pi/18
     pending[#pending+1]={due=tick+7,x=p.x+config.distance*math.cos(a),y=p.y+config.distance*math.sin(a)}
    end
    while pending[1] and pending[1].due<=tick do
     local o=table.remove(pending,1)
     ExecuteOrderFromTable({UnitIndex=heroes[2]:entindex(),OrderType=DOTA_UNIT_ORDER_MOVE_TO_DIRECTION,
       Position=Vector(o.x,o.y,0),Queue=false})
     print('SF_P15_DIRECTION|'..json.encode({tick=tick,x=o.x,y=o.y,kind=DOTA_UNIT_ORDER_MOVE_TO_DIRECTION}))
    end
   end
   local units={}
   local function add(u,kind)
    local p=u:GetAbsOrigin();local a=u:GetForwardVector()
    units[#units+1]={id=u:entindex(),kind=kind,team=u:GetTeamNumber(),x=p.x,y=p.y,z=p.z,
       facing=math.atan2(a.y,a.x)*180/math.pi,hull=u:GetHullRadius(),speed=u:GetBaseMoveSpeed()}
   end
   add(heroes[2],0);add(heroes[3],0)
   for _,u in ipairs(blockers) do if not u:IsNull() then add(u,2) end end
   local p=heroes[2]:GetAbsOrigin()
   print('SF_P15_FRAME|'..json.encode({case=index,case_tick=ct,tick=tick,units=units,
     traversable=GridNav:IsTraversable(p),blocked=GridNav:IsBlocked(p)}))
  end)
  if not success then print('SF_P15_ERROR|'..tostring(failure));return nil end
  if failure=='done' then return nil end
  return 1/30
 end,0)
end)
if not ok then print('SF_P15_ERROR|'..tostring(err)) end
