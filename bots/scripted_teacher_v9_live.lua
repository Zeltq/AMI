-- Live Dota adapter for the v9 algorithm; the policy itself is in the pure core.
local core=require("bots/scripted_teacher_v9")
local agent=require("bots/agent")
local settings=require("bots/settings")
local navigation=settings.chio_script_navigation==true and require('bots/chio_navigation') or nil
local nav=navigation and navigation.new(true) or nil
local json=require("game/dkjson")
local M={}
local st={seq=0,skill_index=1,permanent=0,last_action=0}
local memory=core.new()
local pending={}
local last_tick=nil
local last_status=-10000
local gate=false
local announced=false
local ended=false
local dropped=0
local function call(u,name,default)
 if u and type(u[name])=="function" then local ok,value=pcall(u[name],u);if ok and value~=nil then return value end end
 return default
end
local function kind(r)
 if r.type=="hero" then return 0 end
 if r.type=="tower" or string.find(r.name or "","tower",1,true) then return 1 end
 if r.creep_siege or string.find(r.name or "","siege",1,true) then return 4 end
 if r.creep_ranged or string.find(r.name or "","ranged",1,true) then return 3 end
 return 2
end
local function unit(r,now)
 if not r or not r.id or not r.health or r.health<=0 or r.visible==false then return nil end
 local object=agent.resolve(r.id);local k=kind(r);local interval=r.attack_interval or call(object,"GetSecondsPerAttack",k==4 and 3 or 1)
 local base=k==0 and 1.6 or (k==1 and .9 or (k==4 and 3 or 1))
 local point=(r.attack_point or .5)*interval/base
 local last=call(object,"GetLastAttackTime",-10000)
 local ready=last+interval-point
 local activity=call(object,"GetAnimActivity",0);local cycle=r.attack_cycle or 1
 local attacking=activity==(_G.ACTIVITY_ATTACK or 1503) or activity==(_G.ACTIVITY_ATTACK2 or 1504)
 local windup=0
 if attacking and now-last>point*.75 then
  local threshold=k==4 and .292 or (r.attack_point or .5)
  if cycle<threshold then windup=now+(threshold-cycle)/math.max(.001,threshold)*point end
 end
 local target=r.attack_target_id or 0
 return {id=r.id,kind=k,team=r.team,x=r.x,y=r.y,hp=r.health,max_hp=r.health_max,
  damage=r.attack_damage or 0,armor=r.armor or 0,regen=r.health_regen or 0,
  range=r.attack_range or 0,speed=r.move_speed or 0,interval=interval,point=point,
  projectile_speed=r.projectile_speed or 0,facing=math.rad(r.facing or 0),
  ready=ready,windup=windup,windup_target=windup>0 and target or 0,target=target,
  order=st.last_action,flask=r.flask,tango=r.tango,attackable=r.attackable,
  turn_rate=call(object,"GetTurnRate",.9),phase_estimated=true}
end
local function build(s,now)
 local own=unit(s.self,now);local units={own};local enemy=unit(s.enemy,now)
 if enemy then units[#units+1]=enemy end
 for _,r in ipairs(s.creeps) do local u=unit(r,now);if u then units[#units+1]=u end end
 for _,r in ipairs(s.towers) do local u=unit(r,now);if u then units[#units+1]=u end end
 local shots={}
 for _,p in ipairs(s.projectiles) do
  if p.is_attack and p.target_id and p.speed and p.speed>0 then
   shots[#shots+1]={x=p.x,y=p.y,speed=p.speed,caster=p.caster_id,target=p.target_id}
  end
 end
 local count=function(i)local item=s.items[i+1];return (item.inventory or 0)+(item.backpack or 0)+(item.stash or 0)+(item.courier or 0) end
 local buys,items={},{}
 for i=0,6 do buys[i]=s.legal.buys[i+1]==true end
 for i=0,7 do items[i]=s.items[i+1].castable==true end
 local own_index=s.team==2 and 1 or 3;local enemy_index=s.team==2 and 3 or 1
 local function tower(r)
  if not r then return nil end
  return {x=r.x,y=r.y,alive=r.health==nil or r.health>0}
 end
 return {own=own,enemy=enemy,units=units,shots=shots,time=now,dota_time=s.time,
  execute_ticks=7,can_act=s.legal.actions[2] and not s.self.fear,
  can_buy=buys,can_item=items,flask_count=count(5),tango_count=count(6),
  own_tower=tower(s.towers[own_index]),enemy_tower=tower(s.towers[enemy_index]),trees=s.trees,
  combat=settings.teacher_combat,prototypes=settings.teacher_prototypes,
  muzzle_forward=settings.muzzle_forward,muzzle_side=settings.muzzle_side}
end
local function ready()
 if not settings.human_team then return true end
 if not announced then
  for _,pid in ipairs(GetTeamPlayers(settings.human_team)) do
   if not IsPlayerBot(pid) and GetSelectedHeroName(pid)=="npc_dota_hero_nevermore" then
    print("SF_V9_HUMAN_READY|"..settings.human_team.."|"..pid);announced=true;break
   end
  end
 end
 if settings.native_lobby and announced then return true end
 if not gate and announced then
  package.loaded["bots/human_ready"]=nil
  local ok,value=pcall(require,"bots/human_ready");gate=ok and value==true
 end
 return gate
end
function M.Think()
 local bot=GetBot()
 if not bot or bot:GetUnitName()~="npc_dota_hero_nevermore" then return end
 if settings.teacher_team and bot:GetTeam()~=settings.teacher_team then return end
 if type(IsPlayerBot)=="function" and not IsPlayerBot(bot:GetPlayerID()) then return end
 local phase=GetGameState()
 if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
 local tick=math.floor(GameTime()*30+.5)
 if last_tick==tick then return end
 if last_tick and tick<last_tick then st={seq=0,skill_index=1,permanent=0,last_action=0};memory=core.new();pending={};ended=false end
 last_tick=tick
 if not ready() or ended then return end
 if settings.chio then require('bots/chio_economy').Think(bot,st) else agent.infrastructure(bot,st) end
 if navigation and bot:IsAlive() then
  local policy_ready,handoff=navigation.update(nav,bot,tick)
  if handoff then memory=core.new();st.last_action=4;pending={} end
  if not policy_ready then
   if tick%6==0 then agent.snapshot(bot,st) end
   return
  end
 end
 if DotaTime()>=0 then
  local reason=nil
  if not bot:IsAlive() or GetHeroDeaths(bot:GetPlayerID())>0 then reason="teacher_death" end
  for _,pid in ipairs(GetTeamPlayers(GetOpposingTeam())) do
   if GetSelectedHeroName(pid)=="npc_dota_hero_nevermore" and GetHeroDeaths(pid)>0 then reason="opponent_death" end
  end
  local tower=GetTower(GetTeam(),TOWER_MID_1)
  if not tower or not tower:IsAlive() then reason="teacher_tower" end
  if DotaTime()>=600 then reason=reason or "time_limit" end
  if reason then
   ended=true;pending={};bot:Action_ClearActions(true)
   print("SF_V9_END|"..json.encode({time=DotaTime(),reason=reason,lh=bot:GetLastHits(),deny=bot:GetDenies(),source=core.source_sha256}))
   return
  end
 end
 local chosen=nil
 for due,command in pairs(pending) do
  if due<=tick then
   if not chosen or command.seq>chosen.seq then if chosen then dropped=dropped+1 end;chosen=command else dropped=dropped+1 end
   pending[due]=nil
  end
 end
 if chosen and bot:IsAlive() then
  assert(tick>=chosen.execute_tick and tick-chosen.observed_tick>=7,"early teacher command")
  agent.execute(bot,st,chosen)
  if chosen.action~=0 then st.last_action=chosen.action end
  print("SF_V9_ACK|"..json.encode({seq=chosen.seq,observed_tick=chosen.observed_tick,scheduled_tick=chosen.execute_tick,actual_tick=tick,late_ticks=tick-chosen.execute_tick,execution=st.execution}))
 end
 if tick%(settings.chio and 6 or 4)~=0 or not bot:IsAlive() then return end
 st.seq=st.seq+1
 local s=agent.snapshot(bot,st);local scene=build(s,GameTime())
 local action,diagnostics=core.decide(scene,memory)
 if settings.chio then action.buy=0 end
 action.seq=st.seq;action.observed_tick=tick;action.execute_tick=tick+7;action.observed_time=s.time
 pending[action.execute_tick]=action
 if action.action==3 then print("SF_V9_ATTACK|"..json.encode({time=s.time,target=action.target_id,forecast=diagnostics.forecasts[action.target_id],candidates=diagnostics.candidates,protected=diagnostics.protected})) end
 if GameTime()-last_status>=1 then
  last_status=GameTime()
  print("SF_V9_STATUS|"..json.encode({time=s.time,lh=bot:GetLastHits(),deny=bot:GetDenies(),health=bot:GetHealth(),x=scene.own.x,y=scene.own.y,action=action.action,candidates=diagnostics.candidates,protected=diagnostics.protected,dropped_late_commands=dropped,phase_source="observed last attack time and animation; estimated windup",source=core.source_sha256}))
 end
end
return M
