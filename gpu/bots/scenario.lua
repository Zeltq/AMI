local json=require('game/dkjson')
local last_tick=-100000
local serial=0
local setup_serial=0
local setup_request=0
local setup_phase=0
local cached_commands={}
local infrastructure={skill_index=1,permanent=0,seq=0}
local agent=require('bots/agent')
function Think()
 local ok,err=pcall(function()
  local bot=GetBot()
  if not bot or bot:GetUnitName()~='npc_dota_hero_nevermore' then return end
  local phase=GetGameState()
  if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
  local tick=math.floor(GameTime()*30+.5)
  if tick==last_tick then return end
  last_tick=tick
  agent.infrastructure(bot,infrastructure)
  local enemy=nil
  for _,u in pairs(GetUnitList(UNIT_LIST_ENEMY_HEROES)) do
   if u:GetUnitName()=='npc_dota_hero_nevermore' and u:CanBeSeen() then enemy=u end
  end
  package.loaded['bots/gpu_commands']=nil
  local loaded,commands=pcall(require,'bots/gpu_commands')
  if loaded then cached_commands=commands else commands=cached_commands;print('SF_GPU_RETRY|'..tick) end
  if GetTeam()==TEAM_DIRE and (not commands.setup or setup_serial==commands.setup.serial) then bot:Action_ClearActions(true) end
  if commands.setup then
   local s=commands.setup
   if setup_serial~=s.serial then
    if setup_request~=s.serial then setup_request=s.serial;setup_phase=0 end
    local target=GetTeam()==TEAM_RADIANT and Vector(s.x0,s.y0,0) or Vector(s.x1,s.y1,0)
    local p=bot:GetLocation()
    if setup_phase==0 and GetUnitToLocationDistance(bot,target)<=48 then setup_phase=1 end
    if setup_phase==0 then bot:Action_MoveToLocation(target)
    else
     if GetTeam()==TEAM_RADIANT and enemy and not bot:IsFacingLocation(enemy:GetLocation(),3) then
      bot:Action_AttackUnit(enemy,true)
     else setup_serial=s.serial;bot:Action_ClearActions(true);print('SF_GPU_SETUP|'..json.encode({team=GetTeam(),serial=setup_serial,tick=tick})) end
    end
   end
   if setup_serial==s.serial then bot:Action_ClearActions(true) end
  elseif commands[1] and tick<commands[1].tick then bot:Action_ClearActions(true)
  end
  for _,c in ipairs(commands) do
   if c.team==GetTeam() and c.serial>serial and c.tick<=tick then
    serial=c.serial
    local accepted=true
    if c.action==3 and enemy then bot:Action_AttackUnit(enemy,true)
    elseif c.action==4 then bot:Action_ClearActions(true)
    elseif c.action>=6 and c.action<=8 then bot:Action_UseAbility(bot:GetAbilityByName('nevermore_shadowraze'..(c.action-5)))
    elseif c.action==1 then bot:Action_MoveToLocation(Vector(c.x,c.y,0))
    elseif c.action==2 then bot:Action_MoveDirectly(Vector(c.x,c.y,0)) else accepted=false end
    print('SF_GPU_ORDER|'..json.encode({team=GetTeam(),serial=serial,tick=tick,action=c.action,x=c.x,y=c.y,accepted=accepted}))
   end
  end
  local p=bot:GetLocation()
  local center=Vector(-8192+(math.floor((p.x+8192)/32)+.5)*32,-8192+(math.floor((p.y+8192)/32)+.5)*32,0)
  local frame={tick=tick,time=DotaTime(),team=GetTeam(),x=p.x,y=p.y,z=p.z,
   passable=IsLocationPassable(p),passable_zero=IsLocationPassable(Vector(p.x,p.y,0)),passable_cell_center=IsLocationPassable(center),height=GetHeightLevel(p),
   facing=bot:GetFacing(),move_speed=bot:GetCurrentMovementSpeed(),animation_cycle=bot:GetAnimCycle(),animation_activity=bot:GetAnimActivity(),health=bot:GetHealth(),mana=bot:GetMana(),attack_interval=bot:GetSecondsPerAttack(),
   attack_point=bot:GetAttackPoint(),damage=bot:GetAttackDamage(),armor=bot:GetArmor(),
   casting=bot:IsCastingAbility(),using=bot:IsUsingAbility(),action=bot:GetCurrentActionType(),projectiles={}}
  -- GetFacing may differ from the engine's facing predicate after turning.
  -- Independent directional probes distinguish the two in diagnostics only.
  frame.facing_sectors={}
  for i=0,63 do
   local angle=i*2*math.pi/64
   if bot:IsFacingLocation(Vector(p.x+math.cos(angle)*1000,p.y+math.sin(angle)*1000,p.z),3)then frame.facing_sectors[#frame.facing_sectors+1]=i*360/64 end
  end
  for _,shot in pairs(bot:GetIncomingTrackingProjectiles()) do
   local q=shot.location
   if q then frame.projectiles[#frame.projectiles+1]={x=q.x,y=q.y,speed=shot.velocity,is_attack=shot.is_attack} end
  end
  local a=bot:GetAbilityByName('nevermore_shadowraze3')
  frame.raze={level=a:GetLevel(),cooldown=a:GetCooldownTimeRemaining(),cast_point=a:GetCastPoint(),mana_cost=a:GetManaCost()}
  print('SF_GPU_FRAME|'..json.encode(frame))
 end)
 if not ok then print('SF_GPU_ERROR|'..tostring(err)) end
end
