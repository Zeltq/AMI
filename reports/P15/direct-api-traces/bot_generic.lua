-- Diagnostic Bot API executor. Orders are observed at t, executed at t+7.
local json=require('game/dkjson')
local config=require('bots/p15_config')
local last=-1
local clock={}
local pending={}
local last_case=-1
function Think()
 local ok,err=pcall(function()
  local bot=GetBot()
  if not bot or bot:GetUnitName()~='npc_dota_hero_nevermore' then return end
  local tick=math.floor(GameTime()*30+.5)
  if tick==last then return end
  last=tick
  if not clock.start_tick then package.loaded['bots/p15_clock']=nil;clock=require('bots/p15_clock') end
  if not clock.start_tick then bot:Action_ClearActions(true);return end
  local elapsed=tick-clock.start_tick
  local index=math.floor(elapsed/config.case_ticks)+1
  local c=config.cases[index]
  if elapsed<0 or not c then bot:Action_ClearActions(true);return end
  local ct=elapsed%config.case_ticks
  if index~=last_case then last_case=index;pending={};bot:Action_ClearActions(true) end
  if GetTeam()==3 then bot:Action_ClearActions(true);return end
  if ct>=config.first_decision and ct<156 and (ct-config.first_decision)%config.decision_ticks==0 then
   local p=bot:GetLocation();local angle=c.direction*math.pi/18
   pending[#pending+1]={due=tick+config.offset_ticks+1,observation_tick=tick,
     x=math.max(-8192,math.min(8192,p.x+2000*math.cos(angle))),
     y=math.max(-8192,math.min(8192,p.y+2000*math.sin(angle)))}
  end
  while pending[1] and pending[1].due<=tick do
   local order=table.remove(pending,1)
   bot:Action_MoveDirectly(Vector(order.x,order.y,0))
   order.tick=tick;order.case=index;order.case_tick=ct
   print('SF_P15_ORDER|'..json.encode(order))
  end
  local p=bot:GetLocation()
  print('SF_P15_BOT|'..json.encode({case=index,case_tick=ct,tick=tick,x=p.x,y=p.y,z=p.z,
    facing=bot:GetFacing(),speed=bot:GetCurrentMovementSpeed(),action=bot:GetCurrentActionType()}))
 end)
 if not ok then print('SF_P15_ERROR|'..tostring(err)) end
end
