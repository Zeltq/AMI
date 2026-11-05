-- Privileged setup; completion is measured independently from Bot API frames.
local ok,err=pcall(function()
 local json=require('game/dkjson');local config=require('bots/p15_config')
 local names={'nevermore_shadowraze1','nevermore_shadowraze2','nevermore_shadowraze3','nevermore_requiem'}
 local heroes={};for _,h in pairs(Entities:FindAllByClassname('npc_dota_hero_nevermore')) do heroes[h:GetTeamNumber()]=h end
 assert(heroes[2] and heroes[3],'missing heroes')
 local start=math.ceil(GameRules:GetGameTime()*30)+150;local previous=-1;local last=-1
 print('SF_P15_START|'..json.encode({start_tick=start}))
 heroes[2]:SetContextThink('p15_casts',function()
  local success,result=pcall(function()
   local tick=math.floor(GameRules:GetGameTime()*30+.5);if tick==last then return end;last=tick
   local elapsed=tick-start;if elapsed<0 then return end
   local index=math.floor(elapsed/180)+1;local c=config.cases[index]
   if not c then print('SF_P15_DONE|true');return 'done' end
   local ct=elapsed%180
   if previous~=index then
    previous=index
    for team,h in pairs(heroes) do
     h:Stop();h:SetAbsOrigin(GetGroundPosition(Vector(team==2 and 80 or 5000,team==2 and -816 or 5000,0),h))
     h:AddNewModifier(h,nil,'modifier_invulnerable',{})
     while h:GetLevel()<7 do h:HeroLevelUp(false) end
     for i,name in ipairs(names) do local a=h:FindAbilityByName(name);a:SetLevel(i==4 and 1 or 4);a:EndCooldown() end
     h:SetMana(h:GetMaxMana());h:SetHealth(h:GetMaxHealth());h:SetForwardVector(Vector(1,0,0))
    end
    print('SF_P15_CASE|'..json.encode({case=index,tick=tick,name=c.name,repeat_index=c['repeat']}))
   end
   local h=heroes[2];local a=h:FindAbilityByName(names[c.spell]);local p=h:GetAbsOrigin()
   print('SF_P15_FRAME|'..json.encode({case=index,case_tick=ct,tick=tick,mana=h:GetMana(),cooldown=a:GetCooldownTimeRemaining(),
    in_phase=a:IsInAbilityPhase(),units={{x=p.x,y=p.y,z=p.z,kind=0,facing=0}}}))
  end)
  if not success then print('SF_P15_ERROR|'..tostring(result));return nil end
  if result=='done' then return nil end
  return 1/30
 end,0)
end)
if not ok then print('SF_P15_ERROR|'..tostring(err)) end
