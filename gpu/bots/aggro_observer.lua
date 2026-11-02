-- BotAPI perspective paired with the server diagnostic; no actions/inference.
local json=require('game/dkjson')
local last=-1
local current_case=0;local pulse=1
local function unit(u,own)
 local p=u:GetLocation();local target=u:GetAttackTarget();local t=nil
 if target then local q=target:GetLocation();t={name=target:GetUnitName(),team=target:GetTeam(),x=q.x,y=q.y}end
 local activity=u:GetAnimActivity();local attacking=activity==ACTIVITY_ATTACK or activity==ACTIVITY_ATTACK2
 if own then attacking=u:GetCurrentActionType()==BOT_ACTION_TYPE_ATTACK end
 return{name=u:GetUnitName(),team=u:GetTeam(),x=p.x,y=p.y,z=p.z,health=u:GetHealth(),facing=u:GetFacing(),attack_target=t,cycle=u:GetAnimCycle(),attack_point=u:GetAttackPoint(),interval=u:GetSecondsPerAttack(),is_attacking=attacking,activity=activity,magic_resist=u:GetMagicResist(),fear=u:HasModifier('modifier_nevermore_requiem_fear')}
end
local function think()
 local bot=GetBot();if not bot or bot:GetUnitName()~='npc_dota_hero_nevermore'then return end
 local phase=GetGameState();if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
 local tick=math.floor(GameTime()*30+.5);if tick==last then return end;last=tick
 package.loaded['bots/gpu_actor_actions']=nil
 local loaded,c=pcall(require,'bots/gpu_actor_actions')
 if loaded and c.enabled and GetTeam()==TEAM_RADIANT then
  if c.case~=current_case then current_case=c.case or 0;pulse=1 end
  local elapsed=DotaTime()-(c.start or DotaTime())
  if c.pull and elapsed>=1.2 and elapsed<3.35 then bot:Action_MoveDirectly(Vector(-1200,-816,0))elseif not c.hold_attack or pulse==1 then bot:Action_ClearActions(true)end
  if c.pulses and pulse<=#c.pulses and elapsed>=c.pulses[pulse]then
   local enemy=nil
   for _,u in pairs(GetUnitList(UNIT_LIST_ENEMY_HEROES))do if u:GetUnitName()=='npc_dota_hero_nevermore'and u:IsAlive()and u:CanBeSeen()then enemy=u;break end end
   if enemy then bot:Action_AttackUnit(enemy,true)end
   print('SF_GPU_BOT_ORDER|'..json.encode({case=current_case,time=DotaTime(),tick=tick,elapsed=elapsed,accepted=enemy~=nil,pulse=pulse,action='attack'}));pulse=pulse+1
  end
 end
 local creeps={}
 for _,u in pairs(GetUnitList(UNIT_LIST_ALLIED_CREEPS))do if u:IsAlive()and(string.find(u:GetUnitName(),'npc_dota_creep_')or string.find(u:GetUnitName(),'siege'))then creeps[#creeps+1]=unit(u)end end
 print('SF_GPU_SERVER_OBSERVER|'..json.encode({time=DotaTime(),tick=tick,team=GetTeam(),hero=unit(bot,true),creeps=creeps}))
end
function Think()local ok,err=pcall(think);if not ok then print('SF_GPU_BOT_ERROR|'..tostring(err))end end
