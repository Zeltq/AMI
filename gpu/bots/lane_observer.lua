-- Public BotAPI state, authoritative active lane-creep enumeration.
local json=require('game/dkjson')
local cases=require('bots/lane_cases')
local last=-1;local ids={};local next_id=0
local function call(u,name,...)
 local f=u[name];if type(f)~='function'then return nil end
 local ok,v=pcall(f,u,...);if ok then return v end
end
local function kind(u)
 local n=u:GetUnitName()
 return string.find(n,'hero_')and'hero'or string.find(n,'ranged')and'ranged'or string.find(n,'siege')and'siege'or'melee'
end
local function ref(u)
 if not u or u:IsNull()then return nil end
 local p=u:GetLocation();if not p then return nil end
 return{name=u:GetUnitName(),kind=kind(u),team=u:GetTeam(),x=p.x,y=p.y}
end
local function row(u)
 local p=u:GetLocation();if not p then return nil end
 local k=kind(u);local id
 if k=='hero'then id=u:GetTeam()==2 and'radiant'or'dire'
 else
  if not ids[u]then next_id=next_id+1;ids[u]='bot-'..GetTeam()..'-'..next_id end
  id=ids[u]
 end
 local result={id=id,name=u:GetUnitName(),kind=k,team=u:GetTeam(),x=p.x,y=p.y,z=p.z,
  alive=u:IsAlive(),health=u:GetHealth(),max_health=u:GetMaxHealth(),mana=u:GetMana(),max_mana=u:GetMaxMana(),
  facing=u:GetFacing(),health_regen=u:GetHealthRegen(),mana_regen=u:GetManaRegen(),
  damage=u:GetAttackDamage(),interval=u:GetSecondsPerAttack(),attack_point=u:GetAttackPoint(),range=u:GetAttackRange(),
  projectile_speed=u:GetAttackProjectileSpeed(),armor=u:GetArmor(),magic_resist=u:GetMagicResist(),
  move_speed=u:GetCurrentMovementSpeed(),level=u:GetLevel(),attack_target_ref=ref(u:GetAttackTarget()),
  entity=call(u,'GetEntityIndex'),raze_stacks=0}
 local index=u:GetModifierByName('modifier_nevermore_shadowraze_debuff')
 if index and index>=0 then result.raze_stacks=u:GetModifierStackCount(index)end
 if k=='hero'then result.last_hits=u:GetLastHits();result.denies=u:GetDenies()end
 return result
end
function Think()
 local ok,err=pcall(function()
  local b=GetBot();if not b or b:GetUnitName()~='npc_dota_hero_nevermore'then return end
  local phase=GetGameState();if phase~=GAME_STATE_PRE_GAME and phase~=GAME_STATE_GAME_IN_PROGRESS then return end
  local tick=math.floor(GameTime()*30+.5);if tick==last then return end;last=tick
  local t=DotaTime();if t< -1.2 or t>61 then return end
  local units={row(b)}
  for _,u in pairs(GetUnitList(UNIT_LIST_ALLIED_CREEPS))do
   if u:IsAlive()and(string.find(u:GetUnitName(),'npc_dota_creep_')or string.find(u:GetUnitName(),'siege'))then units[#units+1]=row(u)end
  end
  print('SF_LANE_PUBLIC|'..json.encode({tick=tick,time=t,game_time=GameTime(),team=GetTeam(),units=units}))
 end)
 if not ok then print('SF_LANE_ERROR|'..tostring(err))end
end
