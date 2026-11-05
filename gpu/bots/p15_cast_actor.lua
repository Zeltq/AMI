-- Measure repeating the same cast and explicit STOP using the real Bot API.
local json=require('game/dkjson')
local config=require('bots/p15_config')
local spells={'nevermore_shadowraze1','nevermore_shadowraze2','nevermore_shadowraze3','nevermore_requiem'}
local clock={};local last=-1;local previous=-1;local pending={}
function Think()
 local ok,err=pcall(function()
  local bot=GetBot()
  if not bot or bot:GetUnitName()~='npc_dota_hero_nevermore' then return end
  local tick=math.floor(GameTime()*30+.5)
  if tick==last then return end;last=tick
  if not clock.start_tick then package.loaded['bots/p15_clock']=nil;clock=require('bots/p15_clock') end
  if not clock.start_tick then bot:Action_ClearActions(true);return end
  local elapsed=tick-clock.start_tick;local index=math.floor(elapsed/180)+1;local c=config.cases[index]
  if elapsed<0 or not c then return end
  local ct=elapsed%180
  if index~=previous then previous=index;pending={};bot:Action_ClearActions(true) end
  if GetTeam()==3 then bot:Action_ClearActions(true);return end
  if ct>=30 and ct<156 and (ct-30)%6==0 then
   if ct==30 or c.mode=='repeat' or (c.mode=='cancel' and ct==36) then
    pending[#pending+1]={due=tick+7,observation_tick=tick,stop=c.mode=='cancel' and ct==36}
   end
  end
  while pending[1] and pending[1].due<=tick do
   local o=table.remove(pending,1)
   if o.stop then bot:Action_ClearActions(true) else bot:Action_UseAbility(bot:GetAbilityByName(spells[c.spell])) end
   o.tick=tick;o.case=index;o.case_tick=ct;o.spell=c.spell
   print('SF_P15_ORDER|'..json.encode(o))
  end
  local ability=bot:GetAbilityByName(spells[c.spell]);local active=bot:GetCurrentActiveAbility()
  print('SF_P15_BOT|'..json.encode({case=index,case_tick=ct,tick=tick,mana=bot:GetMana(),
   cooldown=ability:GetCooldownTimeRemaining(),casting=bot:IsCastingAbility(),using=bot:IsUsingAbility(),
   active=active and active:GetName() or '',cast_point=ability:GetCastPoint()}))
 end)
 if not ok then print('SF_P15_ERROR|'..tostring(err)) end
end
