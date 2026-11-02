-- Read-only combat event listener. No properties, states, orders or return values
-- affecting an attack; only callbacks and measured transforms are exported.
local json=require('game/dkjson')
modifier_sf_lane_attack_events=class({})
function modifier_sf_lane_attack_events:IsHidden()return true end
function modifier_sf_lane_attack_events:IsPurgable()return false end
function modifier_sf_lane_attack_events:RemoveOnDeath()return true end
function modifier_sf_lane_attack_events:DeclareFunctions()
 return{MODIFIER_EVENT_ON_ATTACK_START,MODIFIER_EVENT_ON_ATTACK,MODIFIER_EVENT_ON_ATTACK_LANDED,MODIFIER_EVENT_ON_TAKEDAMAGE}
end
local function emit(self,name,event)
 local parent=self:GetParent();if event.attacker~=parent then return end
 local ctx=_G.SF_LANE_ATTACK_CONTEXT;if not ctx then return end
 local p=parent:GetAbsOrigin();local attachments={}
 for _,key in ipairs({'attach_attack1','attach_attack2'})do
  local index=parent:ScriptLookupAttachment(key)
  if index and index>0 then local v=parent:GetAttachmentOrigin(index);attachments[key]={x=v.x,y=v.y,z=v.z}end
 end
 print('SF_LANE_ATTACK_EVENT|'..json.encode({case=ctx.id,event=name,
  tick=math.floor(GameRules:GetGameTime()*30+.5)-ctx.origin,
  game_time=GameRules:GetGameTime(),dota_time=GameRules:GetDOTATime(false,true),
  actor=ctx.identity[parent:entindex()],target=event.target and ctx.identity[event.target:entindex()]or nil,
  attack_record=event.record,last_attack=parent:GetLastAttackTime(),
  animation={sequence=parent:GetSequence(),cycle=parent:GetCycle()},
  source={x=p.x,y=p.y,z=p.z},attachments=attachments}))
end
function modifier_sf_lane_attack_events:OnAttackStart(event)emit(self,'start',event)end
function modifier_sf_lane_attack_events:OnAttack(event)emit(self,'release',event)end
function modifier_sf_lane_attack_events:OnAttackLanded(event)emit(self,'landed',event)end
function modifier_sf_lane_attack_events:OnTakeDamage(event)
 local parent=self:GetParent();local ctx=_G.SF_LANE_ATTACK_CONTEXT
 if event.unit~=parent or not ctx or not ctx.capture_damage then return end
 print('SF_LANE_DAMAGE_EVENT|'..json.encode({case=ctx.id,
  tick=math.floor(GameRules:GetGameTime()*30+.5)-ctx.origin,
  actor=event.attacker and ctx.identity[event.attacker:entindex()]or nil,
  target=ctx.identity[parent:entindex()],damage=event.damage,original_damage=event.original_damage,
  damage_type=event.damage_type,health=parent:GetHealth(),regen=parent:GetHealthRegen()}))
end
