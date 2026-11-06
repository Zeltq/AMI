local json=require('game/dkjson')
local last=-1;local seq=0;local reported=false
function Think()
 local bot=GetBot()
 if not bot or bot:GetUnitName()~='npc_dota_hero_nevermore' then return end
 local tick=math.floor(GameTime()*30+.5)
 if tick==last then return end;last=tick
 if not reported then
  reported=true
  local api={}
  for _,name in ipairs({'ExecuteOrderFromTable','SendToServerConsole','SendToConsole','Convars','GameRules','Entities','GetScriptDirectory'}) do api[name]=type(_G[name]) end
  print('SF_P17_BOT_API|'..json.encode({team=GetTeam(),api=api}))
 end
 if GetGameState()~=GAME_STATE_PRE_GAME and GetGameState()~=GAME_STATE_GAME_IN_PROGRESS then return end
 if GetTeam()~=2 then bot:Action_ClearActions(true);return end
 if tick%6==0 then
  seq=seq+1;local p=bot:GetLocation();local direction=(math.floor(seq/20)%4)*9
  local a=direction*math.pi/18
  print('SF_CHIO_REQUEST|'..json.encode({team=2,player=bot:GetPlayerID(),sequence=seq,observed_tick=tick,
   execute_tick=tick+7,action=0,direction=direction,x=math.max(-8192,math.min(8192,p.x+2000*math.cos(a))),
   y=math.max(-8192,math.min(8192,p.y+2000*math.sin(a)))}))
 end
 local p=bot:GetLocation()
 print('SF_P17_POSITION|'..json.encode({tick=tick,x=p.x,y=p.y,facing=bot:GetFacing()}))
end
